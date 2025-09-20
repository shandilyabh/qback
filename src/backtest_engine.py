""" the backtest engine """

import pandas as pd
import numpy as np
from datetime import time
from tqdm import tqdm
import yaml
import concurrent.futures

def _run_strategy_batch_worker(args):
    """
    Worker function for parallel execution of a batch of parameter sets.
    
    [args]
    args: (params_batch, options_data_path, index_data_path, config_path)
    
    [returns]
    list of metrics dicts for each parameter set in the batch
    """
    params_batch, options_data_path, index_data_path, config_path = args
    from backtest_engine import OptionsBacktester
    backtester = OptionsBacktester(index_data_path, options_data_path, config_path)
    results = []
    for params in params_batch:
        result = backtester.run_single_strategy(params)
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
    - Parallelizes backtest runs.
    - Provides filtering for viable strategies.

    Usage:
        backtester = OptionsBacktester(options_data_path, index_data_path, config_path)
        results_df = backtester.run_optimization()
        filtered_df = backtester.filter_results(results_df)
    """
    def __init__(self, options_data_path, index_data_path, config_path="../config.yaml"):
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
            return "No Index price was found for the given timestamp in index data"
        
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
                txn_fee = self.config.get('transaction_fee_per_lot', 0)
                total_fees = brokerage + txn_fee

                if trade.get('is_long', False):  # LONG position
                    trade['pnl'] = exit_price - entry_price - total_fees
                else:  # SHORT position (default)
                    trade['pnl'] = entry_price - exit_price - total_fees
                trade['entry_price_slip'] = entry_price
                trade['exit_price_slip'] = exit_price
                trade['total_fees'] = total_fees
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

        # for determining target strike based on offset  
        target_strike = atm_strike + (strike_offset * self.strike_interval)
        available_strikes = set(key[2] for key in self._option_price_dict.keys() if key[0] == entry_timestamp)
        if not available_strikes:
            ''' trying for nearby timestamps '''
            for key in self._option_price_dict.keys():
                if abs((key[0] - entry_timestamp).total_seconds()) <= 300:
                    available_strikes.add(key[2])
        if available_strikes:
            ''' find closest available strike to target '''
            target_strike = min(available_strikes, key=lambda x: abs(x - target_strike))
        
        # Initialize independent leg states with original leg tracking
        leg_states = {}
        for leg in legs:
            entry_price = self.get_option_price(entry_timestamp, expiry_for_dte, target_strike, leg)
            if entry_price is not None:
                leg_states[leg] = {
                    'position': {
                        'entry_time': entry_timestamp,
                        'strike': target_strike,
                        'entry_price': entry_price,
                        'option_type': leg,
                        'expiry': expiry_for_dte,
                        'dte': dte,
                        'entry_index_price': index_price,
                        'is_long': self.config.get('initial_position_long', False)
                    },
                    're_entry_count': 0,
                    'pending_reentry': None,
                    'original_leg': leg,  # Track original leg for _REVERSE logic
                    'original_is_long': self.config.get('initial_position_long', False)  # Track original direction
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
                
                # Handle pending re-entries
                if leg_state['pending_reentry'] is not None:
                    pending = leg_state.get('pending_reentry', None)
                    if self._check_reentry_condition(pending, expiry_for_dte, timestamp, day_index):
                        leg_state['position'] = pending['position']
                        leg_state['pending_reentry'] = None
                
                # Process active position
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
                                if reentry_result:
                                    if re_entry_type in ['RE-COST', 'RE-COST_REVERSE', 'RE-MOMENTUM', 'RE-MOMENTUM_REVERSE']:
                                        leg_state['pending_reentry'] = {
                                            'type': re_entry_type,
                                            'position': reentry_result,
                                            'original_position': current_position
                                        }
                                    else:
                                        leg_state['position'] = reentry_result
                                    leg_state['re_entry_count'] += 1
                                else:
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
        
        # Suppressed: DTE fallback warning
        # if best_expiry and best_diff > 0:
        #     print(f"Warning: Using fallback expiry with DTE {(pd.to_datetime(best_expiry).date() - timestamp.date()).days} instead of requested {dte}")
        
        return best_expiry
    
    def _check_reentry_condition(self, pending_reentry, expiry, timestamp, day_index):
        """
        Check if pending re-entry condition is met

        [args]
        pending_reentry: dict with keys - type, position, original_position
        expiry: str
        timestamp: pd.Timestamp
        day_index: pd.DataFrame

        [returns]
        bool: True if re-entry condition met, else False
        """
        base_type = pending_reentry['type'].replace("_REVERSE", "")
        
        if base_type == 'RE-COST':
            position = pending_reentry['position']
            original_position = pending_reentry['original_position']
            current_price = self.get_option_price(timestamp, expiry, position['strike'], position['option_type'])
            return current_price is not None and abs(current_price - original_position['entry_price']) <= 0.5
        
        return True  # RE-MOMENTUM and RE-ASAP are always ready
    
    def _handle_reentry(self, exited_position, exit_timestamp, re_entry_type, day_index, expiry, original_leg, original_is_long, dte):
        """
        Handle re-entry logic based on canonical re-entry types
        
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
        new_position: dict or None
        """
        is_reverse = re_entry_type.strip().endswith("_REVERSE")
        base_type = re_entry_type.replace("_REVERSE", "")
        
        if is_reverse:
            target_leg = 'PE' if original_leg == 'CE' else 'CE'
            is_long = not original_is_long
        else:
            target_leg = original_leg
            is_long = original_is_long

        if base_type == 'RE-ASAP':
            # Immediate re-entry at current ATM
            current_index_price = day_index.loc[exit_timestamp, 'index_close']
            new_atm_strike = self.get_atm_strike_from_ltp(exit_timestamp)
            new_entry_price = self.get_option_price(exit_timestamp, expiry, new_atm_strike, target_leg)
            if new_entry_price is not None:
                return {
                    'entry_time': exit_timestamp,
                    'strike': new_atm_strike,
                    'entry_price': new_entry_price,
                    'option_type': target_leg,
                    'expiry': expiry,
                    'dte': dte,
                    'entry_index_price': current_index_price,
                    'is_long': is_long
                }

        elif base_type == 'RE-COST':
            # Wait until original strike returns to original entry price
            return self._wait_for_cost_reentry(exited_position, exit_timestamp, day_index, expiry, target_leg, dte, is_long)

        elif base_type == 'RE-MOMENTUM':
            # Wait for momentum condition
            return self._wait_for_momentum_reentry(exited_position, exit_timestamp, day_index, expiry, target_leg, dte, is_long)

        return None
    
    def _wait_for_cost_reentry(self, exited_position, exit_timestamp, day_index, expiry, target_leg, dte, is_long):
        """
        Wait for original strike to return to (<= 0.5 pts) original entry price
        
        [args]
        exited_position: dict
        exit_timestamp: pd.Timestamp
        day_index: pd.DataFrame
        expiry: str
        target_leg: str ("CE" or "PE")
        dte: int
        is_long: bool

        [returns]
        new_position: dict or None
        """
        original_strike = exited_position['strike']
        original_entry_price = exited_position['entry_price']
        
        # Check remaining timestamps in the day
        remaining_timestamps = [ts for ts in day_index.index if ts > exit_timestamp]
        
        for timestamp in remaining_timestamps:
            current_price = self.get_option_price(timestamp, expiry, original_strike, target_leg)
            if current_price is not None and abs(current_price - original_entry_price) <= 0.5:  # Within 0.5 points
                current_index_price = day_index.loc[timestamp, 'index_close']
                return {
                    'entry_time': timestamp,
                    'strike': original_strike,
                    'entry_price': current_price,
                    'option_type': target_leg,
                    'expiry': expiry,
                    'dte': dte,
                    'entry_index_price': current_index_price,
                    'is_long': is_long
                }
        
        return None
    
    def _wait_for_momentum_reentry(self, exited_position, exit_timestamp, day_index, expiry, target_leg, dte, is_long):
        """
        Wait for momentum condition (simple momentum: 3 consecutive moves in same direction)
        
        [args]
        exited_position: dict
        exit_timestamp: pd.Timestamp
        day_index: pd.DataFrame
        expiry: str
        target_leg: str ("CE" or "PE")
        dte: int
        is_long: bool

        [returns]
        new_position: dict or None
        """
        remaining_timestamps = [ts for ts in day_index.index if ts > exit_timestamp]
        
        if len(remaining_timestamps) < 3:
            return None
        
        # Check for 3 consecutive moves in same direction
        for i in range(len(remaining_timestamps) - 2):
            ts1, ts2, ts3 = remaining_timestamps[i:i+3]
            
            price1 = day_index.loc[ts1, 'index_close']
            price2 = day_index.loc[ts2, 'index_close']
            price3 = day_index.loc[ts3, 'index_close']
            
            # Check for momentum (3 consecutive moves in same direction)
            if ((price2 > price1 and price3 > price2) or  # Upward momentum
                (price2 < price1 and price3 < price2)):    # Downward momentum
                
                current_index_price = day_index.loc[ts3, 'index_close']
                new_atm_strike = self.get_atm_strike_from_ltp(ts3)
                new_entry_price = self.get_option_price(ts3, expiry, new_atm_strike, target_leg)
                
                if new_entry_price is not None:
                    return {
                        'entry_time': ts3,
                        'strike': new_atm_strike,
                        'entry_price': new_entry_price,
                        'option_type': target_leg,
                        'expiry': expiry,
                        'dte': dte,
                        'entry_index_price': current_index_price,
                        'is_long': is_long
                    }
        
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
    
    def _calculate_realistic_capital(self, trades):
        """
        Calculate realistic capital based on assumed margin requirements
        
        [args]
        trades: list of trade dicts
        
        [returns]
        realistic_capital: float - actual capital required for this strategy
        """
        if not trades:
            return self.config.get('min_capital', 10000)  # Minimum capital
        
        max_margin_required = 0
        portfolio_capital = self.config.get('portfolio_capital', 500000)  # Default ₹5L
        
        # Calculate maximum concurrent margin requirement
        for trade in trades:
            if trade.get('is_long', False):
                # LONG positions: capital = premium paid
                capital_required = trade.get('entry_price_slip', trade['entry_price'])
            else:
                # SHORT positions: capital = SPAN + Exposure margin
                strike = trade.get('strike', 21500)
                lot_size = 50  # NIFTY lot size
                notional_value = strike * lot_size
                
                # Approximate margin: 15% of notional value
                margin_required = notional_value * 0.15
                capital_required = margin_required
            
            max_margin_required = max(max_margin_required, capital_required)
        
        # Ensure we don't exceed portfolio capital
        realistic_capital = min(max_margin_required, portfolio_capital)
        
        # Ensure minimum capital for meaningful ROI calculation
        min_capital = self.config.get('min_capital', 10000)
        return max(realistic_capital, min_capital)
    
    def _calculate_metrics(self, trades):
        """
        Calculates performance metrics

        [args]
        trades: list of dicts representing individual trades

        [returns]
        metrics: dict with the required performance metrics
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
        total_pnl = df['pnl'].sum()
        win_rate = (df['pnl'] > 0).mean() * 100
        avg_premium = df['entry_price_slip'].mean() if 'entry_price_slip' in df.columns else 1

        # maximum drawdown
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()

        # ROI calculation using realistic margin-based capital model
        realistic_capital = self._calculate_realistic_capital(trades)
        roi = (total_pnl / realistic_capital) * 100

        # Reward Risk ratio
        wins = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0
        reward_risk = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0

        # Expectancy per trade (mean pnl as % of entry)
        expectancy = (df['pnl'].mean() / avg_premium) * 100 if avg_premium else 0
        
        # Average profit per period (% of realistic capital per period)
        periods = df['entry_time'].dt.date.nunique() if 'entry_time' in df.columns else 0
        if periods > 0:
            avg_profit_per_period = (total_pnl / (periods * realistic_capital)) * 100
        else:
            avg_profit_per_period = 0

        # Max drawdown as % of realistic capital
        max_drawdown_perc = (max_drawdown / realistic_capital) * 100

        # Return to MDD (max drawdown ratio): return for each unit of downside risk
        return_mdd_ratio = roi / max(max_drawdown_perc, 1e-6) if max_drawdown_perc > 0 else 0

        return {
            'roi': roi,
            'total_pnl': total_pnl,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown_perc,
            'reward_risk': reward_risk,
            'expectancy': expectancy,
            'avg_profit_per_period': avg_profit_per_period,
            'return_mdd_ratio': return_mdd_ratio,
            'trades': trades
        }
    
    def run_optimization(self, max_workers=None, batch_size=20):
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
        dte_list = self.config.get('dte_list', [0])

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
        job_args = [(batch, self.index_data_path, self.options_data_path, self.config_path)
                    for batch in chunkify(param_combinations, batch_size)]
        
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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
    
    def filter_results(self, results_df):
        """
        Filter results based on time-adjusted criteria from config

        [args]
        results_df: pd.DataFrame - results from run_optimization()

        [returns]
        filtered_df: pd.DataFrame - filtered results
        """
        # Get filtration criteria from config
        criteria = self.config.get('filtration_criteria', {})
        
        # Time-adjusted criteria for 1-month backtest as default values:
        min_roi = criteria.get('min_roi', 3.0)
        min_return_mdd = criteria.get('min_return_mdd', 6.0)
        min_reward_risk = criteria.get('min_reward_risk', 1.2)
        min_expectancy = criteria.get('min_expectancy', 0.4)
        min_win_rate = criteria.get('min_win_rate', 62)
        min_avg_profit_per_period = criteria.get('min_avg_profit_per_period', 0.1)
        max_drawdown = criteria.get('max_drawdown', 2.0)
        
        filtered = results_df[
            (results_df['roi'] >= min_roi) &
            (results_df['return_mdd_ratio'] >= min_return_mdd) &
            (results_df['reward_risk'] >= min_reward_risk) &
            (results_df['expectancy'] >= min_expectancy) &
            (results_df['win_rate'] >= min_win_rate) &
            (results_df['avg_profit_per_period'] >= min_avg_profit_per_period) &
            (results_df['max_drawdown'] <= max_drawdown)
        ]
        return filtered.sort_values('roi', ascending=False)
