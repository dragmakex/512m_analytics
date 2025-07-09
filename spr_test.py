import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('defi_prime_rate.db')

# Get all tables in the database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in database:")
print(tables)

# Read the first table (assuming there's at least one)
if not tables.empty:
    table_name = tables.iloc[0]['name']
    print(f"\nReading table: {table_name}")
    
    # Import the table as a dataframe
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    print(f"\nDataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n--- HEAD ---")
    print(df.head())
    
    print("\n--- TAIL ---")
    print(df.tail())
    
    # Get the last day's data
    last_date = df['date'].max()
    last_day_data = df[df['date'] == last_date].iloc[0]
    
    print(f"\n--- BIGGEST TVL USD POOLS ON {last_date} ---")
    
    # Extract TVL columns and their values for the last day
    tvl_columns = [col for col in df.columns if col.startswith('tvlUsd_Pool_')]
    tvl_data = []
    
    for col in tvl_columns:
        pool_num = col.replace('tvlUsd_Pool_', '')
        tvl_value = last_day_data[col]
        apy_col = f'apy_Pool_{pool_num}'
        apy_value = last_day_data[apy_col] if apy_col in last_day_data else None
        
        if pd.notna(tvl_value) and tvl_value > 0:
            tvl_data.append({
                'Pool': f'Pool_{pool_num}',
                'TVL_USD': tvl_value,
                'APY': apy_value
            })
    
    # Sort by TVL and show top 10
    tvl_df = pd.DataFrame(tvl_data)
    if not tvl_df.empty:
        tvl_df = tvl_df.sort_values('TVL_USD', ascending=False)
        print(tvl_df.head(10))
    else:
        print("No TVL data found for the last day")
else:
    print("No tables found in the database")

# Close the connection
conn.close()
