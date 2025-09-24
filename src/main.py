""" Main script to run options backtesting and optimization"""

import pandas as pd
from backtest_engine import OptionsBacktester

def main(config_path="config.yaml"):
    """
    Main execution function to run the backtest, filter results, and save outputs.
    """
    # --- 1. Initialize the Backtester ---
    # Paths are relative to the project root, where run_backtest.py is executed.
    options_data_path = 'Nifty-Options-Weekly.parquet'
    index_data_path = 'Nifty-Index-Monthly-Data.parquet'
    
    backtester = OptionsBacktester(
        options_data_path=options_data_path,
        index_data_path=index_data_path,
        config_path=config_path
    )

    print("Starting options backtesting optimization...")
    
    # --- 2. Run Optimization ---
    all_results_df = backtester.run_optimization()
    
    if all_results_df.empty:
        print("Optimization produced no results. Exiting.")
        return

    # --- 3. Save All Results ---
    all_results_df.to_csv('runs.csv', index=False)
    print(f"Completed {len(all_results_df)} strategy combinations. All results saved to runs.csv")
    
    # --- 4. Filter for Viable Strategies ---
    viable_strategies_df = backtester.filter_results(all_results_df)
    
    if viable_strategies_df.empty:
        print("No strategies met the filtration criteria.")
        return
        
    # --- 5. Save Viable Strategies ---
    viable_strategies_df.to_csv('viable_strategies.csv', index=False)
    print(f"Found {len(viable_strategies_df)} viable strategies. Saved to viable_strategies.csv")

    # --- 6. Get and Display Best Strategies from the VIABLE pool ---
    best_strategies_by_dte = backtester.get_best_strategies(viable_strategies_df)

    if not best_strategies_by_dte:
        print("Could not determine best strategies from the viable set.")
        return

    print("\n--- Top Performing Viable Strategies by DTE ---")
    print("=" * 50)
    
    # Find the single best strategy overall based on composite score
    overall_best_strategy = None
    max_score = -1

    for dte, strategy in best_strategies_by_dte.items():
        print(f"\nBest for DTE {dte}:")
        print(f"  - Strategy ID: {strategy['strategy_id']}")
        print(f"  - Composite Score: {strategy['composite_score']:.2f}/100")
        print(f"  - ROI (3-yr Equiv): {strategy['roi']:.2f}%")
        print(f"  - Win Rate: {strategy['win_rate']:.2f}%")
        print(f"  - Return/MDD Ratio: {strategy['return_mdd_ratio']:.2f}")
        print(f"  - Total Trades: {strategy['total_trades']}")
        
        if strategy['composite_score'] > max_score:
            max_score = strategy['composite_score']
            overall_best_strategy = strategy

    # --- 7. Save Trades for the Overall Best Strategy ---
    if overall_best_strategy:
        strategy_id = overall_best_strategy['strategy_id']
        trades_df = pd.DataFrame(overall_best_strategy['trades'])
        trades_filename = f'trades_{strategy_id}.csv'
        trades_df.to_csv(trades_filename, index=False)
        print("\n" + "=" * 50)
        print(f"Saved detailed trades for the best overall strategy ('{strategy_id}') to '{trades_filename}'")

if __name__ == "__main__":
    main()