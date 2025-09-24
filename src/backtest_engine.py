""" the backtest engine """

import pandas as pd
import numpy as np
from datetime import time
from tqdm import tqdm
import yaml
import concurrent.futures

# Global variable to hold the backtester instance in each worker process
backtester_instance = None

def _init_worker(options_path, index_path, config_path):
    """Initializer for each worker process to create a single backtester instance."""
    global backtester_instance
    from backtest_engine import OptionsBacktester
    backtester_instance = OptionsBacktester(options_path, index_path, config_path)

def _run_strategy_batch_worker(params_batch):
    """
    Worker function for parallel execution of a batch of parameter sets.
    It now uses the global backtester_instance initialized for the worker.
    """
    if backtester_instance is None:
        raise RuntimeError("Worker process not initialized correctly.")
    
    results = []
    for params in params_batch:
        result = backtester_instance.run_single_strategy(params)
        result.update(params)
        results.append(result)
    return results

class OptionsBacktester:
    """
    OptionsBacktester is the core engine for brute-force backtesting of index option strategies.

    Features:
    - Loads options and index data from Parquet files.
    - Reads all parameter ranges and fees from a YAML config file. (configurable parameters/fees)
    - Supports full parameter grid-search.
    - Manages CE and PE legs independently, including all re-entry strategies.
    - Computes detailed performance metrics (ROI, max drawdown, reward:risk, expectancy, etc.) for each parameter combination.
    - Parallelizes & batches backtest runs.
    - Provides filtering for viable strategies.

    Usage:
        backtester = OptionsBacktester(options_data_path, index_data_path, config_path)
        results_df = backtester.run_optimization()
        filtered_df = backtester.filter_results(results_df)
    """
    def __init__(self, options_data_path, index_data_path, config_path="config.yaml"):
        self.index_data_path = index_data_path
        self.options_data_path = options_data_path
        self.config_path = config_path

        self.options_data = pd.read_parquet(options_data_path)
        self.index_data = pd.read_parquet(index_data_path)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # [timestamps to datetime && float32 instead of float64] for memory efficiency
        self.options_data['timestamp'] = pd.to_datetime(self.options_data['timestamp'])
        self.options_data['close'] = self.options_data['close'].astype(np.float32)
        self.index_data['timestamp'] = pd.to_datetime(self.index_data['timestamp'])
        self.index_data.set_index('timestamp', inplace=True)

        self.strike_interval = self.config.get('strike_interval', 50)

        # mapping close price to [timestamp, expiry, strike, and option_type] for O(1) lookups
        self._option_price_dict = {}
        for row in self.options_data.itertuples(index=False):
            key = (row.timestamp, row.expiry, row.strike, row.option_type)
            self._option_price_dict[key] = row.close

        self.options_data = None
        
    def get_atm_strike_from_ltp(self, timestamp):
        """
        function to determine ATM strike based on LTP (Last Traded Price) from options data.

        [args]
        timestamp: pd.Timestamp

        [returns]
        closest_strike: float
        """
        if self.index_data is None or self.index_data.empty:
            raise ValueError("Index data (underlying asset data) is required for ATM selection and index-based stop-loss.")
        
        try:
            index_price = self.index_data.loc[timestamp, 'index_close']
        except KeyError:
            return None
        
        # Find all available strikes at this timestamp
        available_strikes = set()
        for key in self._option_price_dict.keys():
            if key[0] == timestamp:
                available_strikes.add(key[2])
        
        if not available_strikes:
            return round(index_price / self.strike_interval) * self.strike_interval # generic fallback to the index data
        
        # Find the strike closest to current index price (LTP)
        closest_strike = min(available_strikes, key=lambda x: abs(x - index_price))
        
        return closest_strike
    
    def get_option_price(self, timestamp, expiry, strike, option_type):
        """
        Get option price for given timestamp, strike and type

        [args]
        timestamp: pd.Timestamp
        expiry: str
        strike: float
        option_type: str ("CE" or "PE")

        [returns]
        price: float or None
        """
        return self._option_price_dict.get((timestamp, expiry, strike, option_type), None)
    
    def run_single_strategy(self, params):
        """
        Run backtest for single parameter combination

        [args]
        params: dict of strategy parameters
        [returns]
        metrics: dict of performance metrics
        """
        entry_time = params.get('entry_time', '9:15')
        exit_time = params.get('exit_time', '15:30')
        strike_offset = params.get('strike_offset', 0)
        sl_type = params.get('sl_type')
        sl_value = params.get('sl_value')
        re_entry_type = params.get('re_entry_type')
        max_re_entries = params.get('max_re_entries', 0)
        option_type = params.get('option_type', 'CE')
        
        trades = []
        
        # Get unique dates
        dates = pd.to_datetime(self.index_data.index).normalize()
        unique_dates = sorted(dates.unique())
        
        dte = params.get('dte', 0)
        for date in unique_dates:
            daily_trades = self._run_daily_strategy(
                date, entry_time, exit_time, strike_offset, 
                sl_type, sl_value, re_entry_type, max_re_entries, option_type, dte
            )

            # Apply slippage and fees to each trade
            for trade in daily_trades:
                slippage = self.config.get('slippage_perc', 0)

                # Apply slippage based on position direction
                if trade.get('is_long', False):
                    # LONG: BUY to enter (pay more), SELL to exit (get less)
                    entry_price = trade['entry_price'] * (1 + slippage)
                    exit_price = trade['exit_price'] * (1 - slippage)
                else:
                    # SHORT: SELL to enter (get less), BUY to exit (pay more)
                    entry_price = trade['entry_price'] * (1 - slippage)
                    exit_price = trade['exit_price'] * (1 + slippage)

                brokerage = self.config.get('brokerage_per_trade', 0)

                if trade.get('is_long', False):  # LONG position
                    trade['pnl'] = exit_price - entry_price - brokerage
                else:  # SHORT position (default)
                    trade['pnl'] = entry_price - exit_price - brokerage
                trade['entry_price_slip'] = entry_price
                trade['exit_price_slip'] = exit_price
                trade['total_fees'] = brokerage
            trades.extend(daily_trades)

        return self._calculate_metrics(trades)
    
    def _run_daily_strategy(self, date, entry_time, exit_time, strike_offset, 
                           sl_type, sl_value, re_entry_type, max_re_entries, option_type, dte):
        """
        Run strategy for a single day with truly independent CE/PE leg management.
        
        [args]
        date: pd.Timestamp
        entry_time: time
        exit_time: time
        strike_offset: int (number of strikes away from ATM)
        sl_type: str
        sl_value: float
        re_entry_type: str
        max_re_entries: int
        option_type: str
        dte: int

        [returns]
        trades: list of trade dicts for the day
        """
        trades = []
        legs = ['CE', 'PE'] if option_type == 'BOTH' else [option_type]
        
        # Get day's data once
        day_start = pd.Timestamp.combine(date, time(9, 15))
        day_end = pd.Timestamp.combine(date, time(15, 30))
        day_index = self.index_data[day_start:day_end]
        if day_index.empty:
            return trades
            
        entry_timestamp = pd.Timestamp.combine(date, entry_time)
        exit_timestamp = pd.Timestamp.combine(date, exit_time)
        if entry_timestamp not in day_index.index:
            return trades
            
        try:
            index_price = day_index.loc[entry_timestamp, 'index_close']
        except KeyError:
            return trades
        
        # Find expiry for this DTE
        expiry_for_dte = self._find_expiry_for_dte(entry_timestamp, dte)
            
        # Get ATM strike at entry using LTP
        atm_strike = self.get_atm_strike_from_ltp(entry_timestamp)
        if atm_strike is None:
            return trades

        # Initialize independent leg states with original leg tracking
        leg_states = {}
        for leg in legs:
            # --- Leg-Specific Strike Calculation ---
            if leg == 'CE':
                ideal_target_strike = atm_strike + (strike_offset * self.strike_interval)
            else:  # PE
                ideal_target_strike = atm_strike - (strike_offset * self.strike_interval)

            # Find available strikes specifically for this leg
            available_strikes = {key[2] for key in self._option_price_dict.keys() 
                                 if key[0] == entry_timestamp and key[3] == leg}
            
            if not available_strikes:
                # If no strikes at exact time, search a 5-min window for this leg
                for key in self._option_price_dict.keys():
                    if abs((key[0] - entry_timestamp).total_seconds()) <= 300 and key[3] == leg:
                        available_strikes.add(key[2])

            final_target_strike = None
            if available_strikes:
                # Find the closest available strike to the ideal target
                final_target_strike = min(available_strikes, key=lambda x: abs(x - ideal_target_strike))
            
            entry_price = None
            if final_target_strike is not None:
                entry_price = self.get_option_price(entry_timestamp, expiry_for_dte, final_target_strike, leg)

            if entry_price is not None:
                leg_states[leg] = {
                    'position': {
                        'entry_time': entry_timestamp,
                        'strike': final_target_strike,
                        'entry_price': entry_price,
                        'option_type': leg,
                        'expiry': expiry_for_dte,
                        'dte': dte,
                        'entry_index_price': index_price,
                        'is_long': self.config.get('initial_position_long', False)
                    },
                    're_entry_count': 0,
                    'pending_reentry': None,
                    'original_leg': leg,
                    'original_is_long': self.config.get('initial_position_long', False)
                }
            else:
                leg_states[leg] = {
                    'position': None,
                    're_entry_count': 0,
                    'pending_reentry': None,
                    'original_leg': leg,
                    'original_is_long': self.config.get('initial_position_long', False)
                }
        
        # Process each timestamp independently for all legs
        for timestamp in day_index.index:
            if timestamp <= entry_timestamp or timestamp > exit_timestamp:
                continue
                
            current_index_price = day_index.loc[timestamp, 'index_close']
            
            # Process each leg independently
            for leg in legs:
                leg_state = leg_states[leg]
                
                # Handle pending re-entries first
                if leg_state.get('pending_reentry'):
                    new_position = self._check_reentry_condition(
                        leg_state['pending_reentry'], timestamp, day_index, expiry_for_dte
                    )
                    if new_position:
                        leg_state['position'] = new_position
                        leg_state['pending_reentry'] = None
                        leg_state['re_entry_count'] += 1
                
                # Process active position if one exists
                if leg_state['position'] is not None:
                    current_position = leg_state['position']
                    current_price = self.get_option_price(timestamp, expiry_for_dte, current_position['strike'], current_position['option_type'])
                    
                    if current_price is not None:
                        sl_hit = self._check_sl(current_position, current_price, sl_type, sl_value, current_index_price)
                        
                        if sl_hit:
                            # Record trade (PnL will be calculated with slippage later)
                            trade = {
                                'entry_time': current_position['entry_time'],
                                'exit_time': timestamp,
                                'strike': current_position['strike'],
                                'entry_price': current_position['entry_price'],
                                'exit_price': current_price,
                                'is_long': current_position.get('is_long', False),
                                'exit_reason': 'SL',
                                'option_type': current_position['option_type'],
                                'expiry': expiry_for_dte,
                                'dte': dte
                            }
                            trades.append(trade)
                            
                            # Handle re-entry with original leg context
                            if leg_state['re_entry_count'] < max_re_entries:
                                reentry_result = self._handle_reentry(
                                    current_position, timestamp, re_entry_type, 
                                    day_index, expiry_for_dte, leg_state['original_leg'], 
                                    leg_state['original_is_long'], dte
                                )
                                
                                # For RE-ASAP, re-entry is immediate
                                if re_entry_type in ['RE-ASAP', 'RE-ASAP_REVERSE']:
                                    leg_state['position'] = reentry_result
                                    if reentry_result:
                                        leg_state['re_entry_count'] += 1
                                else:
                                    # For other types, set a pending state
                                    leg_state['pending_reentry'] = reentry_result
                                    leg_state['position'] = None
                            else:
                                leg_state['position'] = None
        
        # Force exit all active positions at exit time
        for leg in legs:
            leg_state = leg_states[leg]
            if leg_state['position'] is not None:
                current_position = leg_state['position']
                exit_price = self.get_option_price(exit_timestamp, expiry_for_dte, current_position['strike'], current_position['option_type'])
                
                if exit_price is not None:
                    trade = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': exit_timestamp,
                        'strike': current_position['strike'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'is_long': current_position.get('is_long', False),
                        'exit_reason': 'TIME',
                        'option_type': current_position['option_type'],
                        'expiry': expiry_for_dte,
                        'dte': dte
                    }
                    trades.append(trade)
        
        return trades
    
    def _find_expiry_for_dte(self, timestamp, dte):
        """
        Find the expiry date that matches the desired DTE from the given timestamp.
        
        [args]
        timestamp: pd.Timestamp
        dte: int (days to expiry)

        [returns]
        expiry: str or None
        """
        expiries = {row[1] for row in self._option_price_dict.keys() if row[0] == timestamp}
        if not expiries:
            return None
        
        # If DTE is 0, return the first available expiry
        if dte == 0:
            return min(
            (pd.to_datetime(e) for e in expiries if pd.to_datetime(e).date() >= timestamp.date()),
            default=None
            )
            
        # find exact DTE match
        for expiry in expiries:
            try:
                expiry_dt = pd.to_datetime(expiry)
                dte_actual = (expiry_dt.date() - timestamp.date()).days
                if dte_actual == dte:
                    return expiry
            except Exception:
                continue
        
        # If no exact match is found, return the closest expiry
        best_expiry = None
        best_diff = float('inf')
        for expiry in expiries:
            try:
                expiry_dt = pd.to_datetime(expiry)
                dte_actual = (expiry_dt.date() - timestamp.date()).days
                if dte_actual < 0:  # skip expired contracts
                    continue
                diff = abs(dte_actual - dte)
                if diff < best_diff:
                    best_diff = diff
                    best_expiry = expiry
            except Exception:
                continue
        
        return best_expiry
    
    def _check_reentry_condition(self, pending_reentry, timestamp, day_index, expiry):
        """
        Check if pending re-entry condition is met on a tick-by-tick basis.

        [args]
        pending_reentry: dict with re-entry details
        timestamp: pd.Timestamp (current tick)
        day_index: pd.DataFrame
        expiry: str

        [returns]
        new_position: dict or None - a new position if conditions are met
        """
        re_entry_type = pending_reentry['type']
        is_reverse = re_entry_type.endswith("_REVERSE")
        base_type = re_entry_type.replace("_REVERSE", "")
        
        original_pos = pending_reentry['original_position']
        target_leg = pending_reentry['target_leg']
        is_long = pending_reentry['is_long']
        
        # Get current and previous index locations for momentum/dip checks
        current_time_loc = day_index.index.get_loc(timestamp)
        
        if base_type == 'RE-COST':
            if not is_reverse:
                # Re-enter the SAME trade if price returns to the original entry cost.
                current_price = self.get_option_price(timestamp, expiry, original_pos['strike'], target_leg)
                if current_price is not None and abs(current_price - original_pos['entry_price']) <= 0.5:
                    return self._create_new_position(timestamp, day_index, expiry, original_pos['strike'], current_price, target_leg, is_long, original_pos['dte'])
            else:
                # Buy the dip" or "Sell the rally": reversal
                if current_time_loc >= 1:
                    current_index_price = day_index.iloc[current_time_loc]['index_close']
                    prev_index_price = day_index.iloc[current_time_loc - 1]['index_close']
                    
                    # If new trade is a Long Call (bullish), wait for a dip (price falls).
                    if target_leg == 'CE' and current_index_price < prev_index_price:
                        return self._create_reentry_position_at_atm(timestamp, day_index, expiry, target_leg, is_long, original_pos['dte'])
                        
                    # If new trade is a Long Put (bearish), wait for a bounce (price rises).
                    elif target_leg == 'PE' and current_index_price > prev_index_price:
                        return self._create_reentry_position_at_atm(timestamp, day_index, expiry, target_leg, is_long, original_pos['dte'])

        elif base_type == 'RE-MOMENTUM':
            if current_time_loc >= 2:
                price1 = day_index.iloc[current_time_loc - 2]['index_close']
                price2 = day_index.iloc[current_time_loc - 1]['index_close']
                price3 = day_index.iloc[current_time_loc]['index_close'] # current price
                
                upward_momentum = price3 > price2 > price1
                downward_momentum = price3 < price2 < price1
                
                if not is_reverse:
                    # Re-enter on any momentum.
                    if upward_momentum or downward_momentum:
                        return self._create_reentry_position_at_atm(timestamp, day_index, expiry, target_leg, is_long, original_pos['dte'])
                else:
                    # Wait for momentum that CONFIRMS the new market view.
                    # If new trade is a Long Call (bullish), require upward momentum.
                    if target_leg == 'CE' and upward_momentum:
                        return self._create_reentry_position_at_atm(timestamp, day_index, expiry, target_leg, is_long, original_pos['dte'])
                    
                    # If new trade is a Long Put (bearish), require downward momentum.
                    elif target_leg == 'PE' and downward_momentum:
                        return self._create_reentry_position_at_atm(timestamp, day_index, expiry, target_leg, is_long, original_pos['dte'])
        
        return None
    
    def _handle_reentry(self, exited_position, exit_timestamp, re_entry_type, day_index, expiry, original_leg, original_is_long, dte):
        """
        Handle re-entry logic. For RE-ASAP, it returns a new position immediately.
        For other types, it returns a dictionary defining the pending re-entry condition.
        
        [args]
        exited_position: dict
        exit_timestamp: pd.Timestamp
        re_entry_type: str
        day_index: pd.DataFrame
        expiry: str
        original_leg: str ("CE" or "PE") - the leg that started this strategy
        original_is_long: bool - the original direction that started this strategy
        dte: int
        
        [returns]
        new_position (dict) or pending_reentry_details (dict) or None
        """
        is_reverse = re_entry_type.strip().endswith("_REVERSE")
        base_type = re_entry_type.replace("_REVERSE", "")
        
        target_leg = original_leg
        is_long = original_is_long

        if is_reverse:
            # Flip market opinion
            if original_leg == "CE" and not original_is_long:  
                # Short Call → Long Call
                target_leg, is_long = "CE", True
            elif original_leg == "PE" and not original_is_long:  
                # Short Put → Long Put
                target_leg, is_long = "PE", True
            elif original_leg == "CE" and original_is_long:  
                # Long Call → Short Call
                target_leg, is_long = "CE", False
            elif original_leg == "PE" and original_is_long:  
                # Long Put → Short Put
                target_leg, is_long = "PE", False
            
        if base_type == 'RE-ASAP':
            # Immediate re-entry at current ATM
            return self._create_reentry_position_at_atm(exit_timestamp, day_index, expiry, target_leg, is_long, dte)

        elif base_type in ['RE-COST', 'RE-MOMENTUM']:
            # Return a dictionary that defines the pending state
            return {
                'type': re_entry_type,
                'original_position': exited_position,
                'target_leg': target_leg,
                'is_long': is_long
            }

        return None
    
    def _create_new_position(self, timestamp, day_index, expiry, strike, entry_price, option_type, is_long, dte):
        """Helper to create a new position dictionary."""
        return {
            'entry_time': timestamp,
            'strike': strike,
            'entry_price': entry_price,
            'option_type': option_type,
            'expiry': expiry,
            'dte': dte,
            'entry_index_price': day_index.loc[timestamp, 'index_close'],
            'is_long': is_long
        }

    def _create_reentry_position_at_atm(self, timestamp, day_index, expiry, target_leg, is_long, dte):
        """Helper to create a new re-entry position at the current ATM strike."""
        new_atm_strike = self.get_atm_strike_from_ltp(timestamp)
        new_entry_price = self.get_option_price(timestamp, expiry, new_atm_strike, target_leg)
        if new_entry_price is not None:
            return self._create_new_position(timestamp, day_index, expiry, new_atm_strike, new_entry_price, target_leg, is_long, dte)
        return None

    def _check_sl(self, position, current_price, sl_type, sl_value, current_index_price=None):
        """
        Check if stop loss is hit for both SHORT and LONG positions

        [args]
        position: dict with keys - entry_price, option_type, entry_index_price, is_long
        current_price: float
        sl_type: str
        sl_value: float
        current_index_price: float or None

        [returns]
        bool: True if SL hit, else False
        """
        is_long = position.get('is_long', False)
        
        if sl_type == 'percent_premium':
            if is_long:
                loss_pct = (position['entry_price'] - current_price) / position['entry_price']
            else:
                loss_pct = (current_price - position['entry_price']) / position['entry_price']
            return loss_pct >= sl_value / 100
        
        elif sl_type == 'points':
            if is_long:
                loss = position['entry_price'] - current_price
            else:
                loss = current_price - position['entry_price']
            return loss >= sl_value
        
        elif sl_type == 'points_index':
            if current_index_price is None or position.get('entry_index_price') is None:
                return False
            entry_index_price = position['entry_index_price']
            
            if is_long:
                # LONG positions: opposite SL logic
                if position['option_type'] == 'CE':
                    index_move = entry_index_price - current_index_price  # Loss when index falls
                else:  # PE
                    index_move = current_index_price - entry_index_price  # Loss when index rises
            else:
                # SHORT positions: existing logic
                if position['option_type'] == 'CE':
                    index_move = current_index_price - entry_index_price
                else:  # PE
                    index_move = entry_index_price - current_index_price
            return index_move >= sl_value
            
        elif sl_type == 'percent_index':
            if current_index_price is None or position.get('entry_index_price') is None:
                return False
            entry_index_price = position['entry_index_price']
            
            if is_long:
                # LONG positions: opposite SL logic
                if position['option_type'] == 'CE':
                    index_move_pct = (entry_index_price - current_index_price) / entry_index_price
                else:  # PE
                    index_move_pct = (current_index_price - entry_index_price) / entry_index_price
            else:
                # SHORT positions: existing logic
                if position['option_type'] == 'CE':
                    index_move_pct = (current_index_price - entry_index_price) / entry_index_price
                else:  # PE
                    index_move_pct = (entry_index_price - current_index_price) / entry_index_price
            return index_move_pct >= sl_value / 100

        return False
    
    def _calculate_metrics(self, trades):
        """
        Calculate metrics
        
        [args]
        trades: list of trade dicts with pnl, entry_price, etc.
        
        [returns]
        metrics: dict with percentage-based metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'roi': 0,
                'reward_risk': 0,
                'expectancy': 0,
                'avg_profit_per_period': 0,
                'return_mdd_ratio': 0,
                'trades': trades
            }
        
        df = pd.DataFrame(trades)
        
        # 1. Win Rate (percentage of winning trades)
        win_rate = (df['pnl'] > 0).mean() * 100
        
        # 2. Expectancy per trade (average PnL as % of entry premium)
        df['pnl_pct'] = (pd.to_numeric(df['pnl'], errors='coerce') / pd.to_numeric(df['entry_price'], errors='coerce')) * 100
        expectancy = df['pnl_pct'].mean()
        
        # 3. Reward:Risk ratio (avg win % / avg loss %)
        wins = df[df['pnl'] > 0]['pnl_pct']
        losses = df[df['pnl'] < 0]['pnl_pct']
        avg_win_pct = wins.mean() if not wins.empty else 0
        avg_loss_pct = abs(losses.mean()) if not losses.empty else 0
        reward_risk = (avg_win_pct / avg_loss_pct) if avg_loss_pct > 0 else 0
        
        # 4. Cumulative return
        df['cumulative_pnl_pct'] = df['pnl_pct'].cumsum()
        
        # 5. Max Drawdown
        running_max = df['cumulative_pnl_pct'].expanding().max()
        drawdown = running_max - df['cumulative_pnl_pct']
        max_drawdown = drawdown.max()
        
        # 6. Total return (cumulative PnL %)
        total_return_pct = df['cumulative_pnl_pct'].iloc[-1]
        
        # 7. Annualized ROI (scale to 3-year equivalent for comparison)
        days_in_data = df['entry_time'].dt.date.nunique() if 'entry_time' in df.columns else 1
        
        # Scale to 3 years (assuming ~250 trading days per year)
        scaling_factor = (3 * 250) / days_in_data if days_in_data > 0 else 1
        roi_3yr_equivalent = total_return_pct * scaling_factor
        
        # 8. Average Profit per Period (daily average return %)
        avg_profit_per_period = total_return_pct / days_in_data if days_in_data > 0 else 0
        
        # 9. Return to MDD ratio
        return_mdd_ratio = roi_3yr_equivalent / max(max_drawdown, 0.01) if max_drawdown > 0 else 0
        
        # 10. Total PnL (sum of all trade PnLs)
        total_pnl = df['pnl'].sum()
        
        return {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'roi': roi_3yr_equivalent,  # Using 3-year equivalent ROI
            'reward_risk': reward_risk,
            'expectancy': expectancy,
            'avg_profit_per_period': avg_profit_per_period,
            'return_mdd_ratio': return_mdd_ratio,
            'trades': trades
        }
    
    def _discover_available_dtes(self):
        """
        Discover all available DTEs from the complete options data
        
        [returns]
        dte_list: list of unique DTE values found in data
        """
        dte_set = set()
        
        # Get all unique timestamps from options data
        timestamps = {key[0] for key in self._option_price_dict.keys()}
        
        for timestamp in timestamps:
            expiries = {key[1] for key in self._option_price_dict.keys() if key[0] == timestamp}
            
            for expiry in expiries:
                try:
                    expiry_dt = pd.to_datetime(expiry)
                    dte = (expiry_dt.date() - timestamp.date()).days
                    if dte >= 0:  # Only non-expired contracts
                        dte_set.add(dte)
                except Exception:
                    continue
        
        return sorted(list(dte_set))
    
    def run_optimization(self, max_workers=None, batch_size=10):
        """
        Run brute-force optimization across all parameter
        combinations in parallel using multiprocessing with batching.

        [args]
        max_workers: int or None - number of parallel processes to use
        batch_size: int - number of parameter combinations per batch

        [returns]
        results_df: pd.DataFrame with results for each parameter combination
        """
        entry_times = [time(int(t.split(":")[0]), int(t.split(":")[1])) for t in self.config['entry_times']]
        exit_times = [time(int(t.split(":")[0]), int(t.split(":")[1])) for t in self.config['exit_times']]

        strike_offsets = self.config['strike_offsets']
        option_types = self.config['option_types']
        max_re_entries_list = self.config['max_re_entries']
        re_entry_types = self.config['re_entry_types']
        sl_type_values = self.config['stop_loss']
        dte_list = self._discover_available_dtes()

        print(f"Discovered DTEs from data: {dte_list}")
        
        param_combinations = []
        for sl_type, sl_values in sl_type_values.items():
            for sl_value in sl_values:
                for entry_time in entry_times:
                    for exit_time in exit_times:
                        for strike_offset in strike_offsets:
                            for option_type in option_types:
                                for max_re_entries in max_re_entries_list:
                                    for re_entry_type in re_entry_types:
                                        for dte in dte_list:
                                            params = {
                                                'entry_time': entry_time,
                                                'exit_time': exit_time,
                                                'strike_offset': strike_offset,
                                                'sl_type': sl_type,
                                                'sl_value': sl_value,
                                                're_entry_type': re_entry_type,
                                                'max_re_entries': max_re_entries,
                                                'option_type': option_type,
                                                'dte': dte
                                            }
                                            param_combinations.append(params)

        # helper to chunk list
        def chunkify(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]

        results = []
        job_args = list(chunkify(param_combinations, batch_size))
        
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(self.options_data_path, self.index_data_path, self.config_path)
            ) as executor:
                for batch_num, batch_result in enumerate(tqdm(executor.map(_run_strategy_batch_worker, job_args),
                                                      total=len(job_args), desc="Running strategies")):
                    for run_num, r in enumerate(batch_result):
                        r['batch_id'] = batch_num
                        r['run_id'] = run_num
                        r['strategy_id'] = f"B{batch_num:04d}_R{run_num:04d}"
                        results.append(r)
        except Exception as e:
            print(f"Error during parallel execution: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        cols = ['strategy_id'] + [c for c in df.columns if c != 'strategy_id']
        return df[cols]

    def get_best_strategies(self, results_df):
        """
        Calculate a composite score for each strategy and find the best one per DTE.

        [args]
        results_df: pd.DataFrame with results for each parameter combination

        [returns]
        best_strategies_by_dte: dict mapping DTE to the best strategy's data
        """
        if results_df.empty:
            return {}
            
        # Get weights from config; fail gracefully if not defined
        weights = self.config.get('composite_score_weights', {})
        if not weights:
            print("Warning: 'composite_score_weights' not found in config.yaml. Using default weights.")
            weights = {
                'roi': 0.30,
                'return_mdd_ratio': 0.30,
                'win_rate': 0.20,
                'expectancy': 0.10,
                'avg_profit_per_period': 0.10
            }
        
        # Normalize metrics to be on a similar scale (0-100)
        for metric in weights.keys():
            if metric in results_df.columns:
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                if (max_val - min_val) > 0:
                    results_df[f'{metric}_norm'] = 100 * (results_df[metric] - min_val) / (max_val - min_val)
                else:
                    results_df[f'{metric}_norm'] = 50  # Assign a neutral score if all values are the same
        
        # Calculate composite score
        results_df['composite_score'] = 0
        for metric, weight in weights.items():
            if f'{metric}_norm' in results_df.columns:
                results_df['composite_score'] += results_df[f'{metric}_norm'] * weight
        
        # Find the best strategy for each DTE
        best_strategies_by_dte = {}
        for dte in results_df['dte'].unique():
            dte_df = results_df[results_df['dte'] == dte]
            best_strategy = dte_df.loc[dte_df['composite_score'].idxmax()]
            best_strategies_by_dte[dte] = best_strategy.to_dict()
            
        return best_strategies_by_dte

    def filter_results(self, results_df):
        """
        Filter backtest results based on criteria in config.yaml.

        [args]
        results_df: pd.DataFrame with results for each parameter combination

        [returns]
        filtered_df: pd.DataFrame with viable strategies
        """
        if results_df.empty:
            return pd.DataFrame()
            
        # Get filtration criteria from config; fail gracefully if not defined
        criteria = self.config.get('filtration_criteria', {})
        if not criteria:
            print("Warning: 'filtration_criteria' not found in config.yaml. Returning all strategies.")
            return results_df
        
        # Use criteria from config, with tolerant defaults if a key is missing
        min_roi = criteria.get('min_roi', float('-inf'))
        min_return_mdd = criteria.get('min_return_mdd', float('-inf'))
        min_reward_risk = criteria.get('min_reward_risk', float('-inf'))
        min_expectancy = criteria.get('min_expectancy', float('-inf'))
        min_win_rate = criteria.get('min_win_rate', -1)
        min_avg_profit_per_period = criteria.get('min_avg_profit_per_period', float('-inf'))
        max_drawdown = criteria.get('max_drawdown', float('inf'))
        
        # Apply filters
        filtered_df = results_df[
            (results_df['roi'] >= min_roi) &
            (results_df['return_mdd_ratio'] >= min_return_mdd) &
            (results_df['reward_risk'] >= min_reward_risk) &
            (results_df['expectancy'] >= min_expectancy) &
            (results_df['win_rate'] >= min_win_rate) &
            (results_df['avg_profit_per_period'] >= min_avg_profit_per_period) &
            (results_df['max_drawdown'] <= max_drawdown)
        ]
        
        return filtered_df