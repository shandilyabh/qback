import pandas as pd

# Convert index CSV to Parquet
index_csv = "Nifty-Index-Monthly-Data.csv"
index_parquet = "Nifty-Index-Monthly-Data.parquet"
df_index = pd.read_csv(index_csv)
df_index.to_parquet(index_parquet, index=False)
print(f"Converted {index_csv} --> {index_parquet}")

# Convert options CSV to Parquet
options_csv = "Nifty-Options-Weekly.csv"
options_parquet = "Nifty-Options-Weekly.parquet"
df_options = pd.read_csv(options_csv)
df_options.to_parquet(options_parquet, index=False)
print(f"Converted {options_csv} --> {options_parquet}")
