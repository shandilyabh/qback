import pandas as pd

# Convert index CSV to Parquet
index_csv = "synthetic_sensex_index_month.csv"
index_parquet = "synthetic_sensex_index_month.parquet"
df_index = pd.read_csv(index_csv)
df_index.to_parquet(index_parquet, index=False)
print(f"Converted {index_csv} --> {index_parquet}")

# Convert options CSV to Parquet
options_csv = "synthetic_sensex_options_month_weekly.csv"
options_parquet = "synthetic_sensex_options_month_weekly.parquet"
df_options = pd.read_csv(options_csv)
df_options.to_parquet(options_parquet, index=False)
print(f"Converted {options_csv} --> {options_parquet}")
