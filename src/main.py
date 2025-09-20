""" Main script to run options backtesting and optimization"""

from backtest_engine import OptionsBacktester
import pandas as pd

def main(config_path="config.yaml"):
    # Initialize backtester (configurable parameters/fees)
    backtester = OptionsBacktester(
        'Nifty-Options-Weekly.parquet',
        'Nifty-Index-Monthly-Data.parquet',
        config_path=config_path
    )
    
    print("Starting options backtesting...")
    
    # Run optimization
    results = backtester.run_optimization()
    
    # Save all results
    results.to_csv('backtest_results.csv', index=False)
    print(f"Completed {len(results)} strategy combinations")
    
    # Filter viable strategies using AlgoTest criteria
    filtered_results = backtester.filter_results(results)

    if not filtered_results.empty:
        print(f"\nFound {len(filtered_results)} viable strategies:")
        print(filtered_results[['run_id', 'roi', 'win_rate', 'max_drawdown', 'total_trades']].head(10))
        
        # Save filtered results
        filtered_results.to_csv('filtered_strategies.csv', index=False)
        
        # Save detailed trades for best strategy
        best_strategy = filtered_results.iloc[0]
        trades_df = pd.DataFrame(best_strategy['trades'])
        trades_df.to_csv('trades_best_strategy.csv', index=False)
    else:
        print("No strategies met the AlgoTest-style filtering criteria.")
        print("You may want to ease thresholds further or check your data.")

if __name__ == "__main__":
    main()
