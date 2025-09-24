# qback - Options Backtesting Engine

A parallelized options backtesting engine designed for brute-force parameter optimization and strategy discovery. It tests various parameter combinations to identify profitable, robust trading strategies based on a configurable set of rules.

## Key Features

- **High-Speed, Parallelized Core**: Utilizes multiprocessing to run thousands of backtests per minute across all available CPU cores.
- **Comprehensive Parameter Grid-Search**: Tests all combinations of entry/exit times, strike offsets, stop-loss rules, and re-entry logic.
- **Advanced Re-entry Strategies**: Implements various re-entry mechanisms, including strategies that correctly reverse market opinion after a stop-loss.
- **Configurable Stop-Loss Types**: Supports multiple SL types, including percentage of premium, points-based, and index movement (points and percentage).
- **Composite Score Ranking**: Ranks viable strategies using a weighted composite score based on key performance metrics (ROI, Return/MDD, Win Rate, etc.).
- **Detailed Performance Metrics**: Calculates ROI, Max Drawdown, Return/MDD Ratio, Win Rate, Expectancy, and more for every strategy.
- **Data-Driven**: Uses the efficient Parquet file format for fast data loading and in-memory lookups.

## Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Your Strategy**:
    All parameters, fees, and ranking criteria are controlled in `src/config.yaml`. Edit this file to define:
    - Entry and exit times.
    - Stop-loss types and values.
    - Re-entry strategies and max attempts.
    - Slippage and brokerage costs.
    - Filtration criteria for viable strategies.
    - Weights for the composite score.

3.  **Run the Backtest**:
    Execute the backtest from the **src directory**:
    ```bash
    cd src
    python main.py
    ```

## Data Format

The engine requires data in the **Parquet** format for performance.

### Index Data (`.parquet`)
| Column        | Type      | Description                  |
|---------------|-----------|------------------------------|
| `timestamp`   | datetime  | The timestamp of the index price |
| `index_close` | float     | The closing price of the index |

### Options Data (`.parquet`)
| Column        | Type      | Description                  |
|---------------|-----------|------------------------------|
| `timestamp`   | datetime  | The timestamp of the option price |
| `expiry`      | datetime  | The expiry date of the contract |
| `strike`      | float     | The strike price of the option |
| `option_type` | string    | 'CE' for Call, 'PE' for Put  |
| `close`       | float     | The closing price of the premium |

> the aux/ directory has a program convert_to_parquet.py that helps convert csv(s) to parquet(s).

## Re-entry Strategies Explained

The engine tests a set of canonical re-entry strategies after a stop-loss is hit.

- **`RE-ASAP`**: Re-enters a new trade immediately at the current at-the-money (ATM) strike.
- **`RE-COST`**: Waits for the premium of the *original* option to return to its original entry price before re-entering.
- **`RE-MOMENTUM`**: Waits for a confirmation of momentum (3 consecutive moves) in the underlying index before re-entering at the new ATM.

### Reversal Logic (`_REVERSE`)

The `_REVERSE` strategies implement a **reversal of market opinion**. When a trade is stopped out, the engine assumes its initial market view was wrong and enters a new trade in the opposite direction.

- **`RE-MOMENTUM_REVERSE`**: If a bearish Short Call is stopped out by a rally, the engine waits for confirmation of **upward momentum** and then enters a **bullish Long Call**.
- **`RE-COST_REVERSE`**: If a bearish Short Call is stopped out by a rally, the engine waits for the first **pullback (dip)** in the index to enter a new **bullish Long Call** at a better price.

## Output Files

All output files are saved to the project root directory:

- **`runs.csv`**: Contains the detailed performance metrics for every single parameter combination tested.
- **`viable_strategies.csv`**: A filtered subset of `runs.csv` containing only the strategies that met the performance criteria defined in `config.yaml`.
- **`trades_<strategy_id>.csv`**: A detailed, trade-by-trade log for the single best-performing strategy identified from the viable set (ranked by composite score).

## Project Structure
```
.
├── aux/                      # Auxiliary scripts and notebooks
├── src/
│   ├── backtest_engine.py    # The core backtesting logic
│   ├── main.py               # Main execution script
│   ├── config.yaml           # ALL strategy parameters and fees
│   ├── Nifty-Index-Monthly-Data.parquet  # Sample index data
│   └── Nifty-Options-Weekly.parquet    # Sample options data
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Flowchart of the Backtest Engine

![Flowchart](workflow.jpg)