import requests
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta

# DeFiLlama API endpoints
YIELDS_ENDPOINT = "https://yields.llama.fi/pools"
CHART_ENDPOINT = "https://yields.llama.fi/chart/"

def fetch_top_stablecoin_pools_by_tvl(limit=100):
    """
    Fetch the top stablecoin pools by TVL from DeFiLlama yields API
    
    Args:
        limit (int): Number of top pools to fetch
        
    Returns:
        list: List of pool dictionaries sorted by TVL
    """
    try:
        print(f"Fetching top {limit} stablecoin pools by TVL...")
        response = requests.get(YIELDS_ENDPOINT)
        
        if response.status_code == 200:
            data = response.json()
            
            # Filter for stablecoin pools only
            stablecoin_pools = []
            for pool in data['data']:
                if (pool.get('tvlUsd') is not None and pool['tvlUsd'] > 0 and 
                    pool.get('stablecoin') == True):
                    stablecoin_pools.append(pool)
            
            # Sort by TVL and get top pools
            stablecoin_pools.sort(key=lambda x: x['tvlUsd'], reverse=True)
            top_pools = stablecoin_pools[:limit]
            
            print(f"Successfully fetched {len(top_pools)} stablecoin pools with highest TVL")
            return top_pools
        else:
            print(f"Error fetching pools: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching pools: {e}")
        return []

def fetch_pool_chart_data(pool_id, pool_name, days=360):
    """
    Fetch historical chart data for a specific pool with rate limiting
    
    Args:
        pool_id (str): Pool ID from DeFiLlama
        pool_name (str): Pool name for logging
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: DataFrame with historical APY and TVL data, or None if failed
    """
    try:
        print(f"Fetching data for {pool_name}...")
        url = f"{CHART_ENDPOINT}{pool_id}"
        response = requests.get(url)
        
        # Handle rate limiting
        if response.status_code == 429:
            print(f"Rate limited for {pool_name}, waiting 2 seconds...")
            time.sleep(2)
            response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract data array
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                    except (ValueError, TypeError):
                        try:
                            df['date'] = pd.to_datetime(df['timestamp'])
                        except (ValueError, TypeError):
                            print(f"Could not parse timestamp format for pool {pool_name}")
                            return None
                    
                    df.set_index('date', inplace=True)
                    
                    # Make index timezone-naive
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Filter to last N days
                    cutoff_date = datetime.now() - timedelta(days=days)
                    df = df[df.index >= cutoff_date]
                    
                    print(f"Successfully fetched {len(df)} data points for {pool_name}")
                    return df
                else:
                    print(f"No timestamp found in data for pool {pool_name}")
                    return None
            else:
                print(f"Unexpected data format for pool {pool_name}")
                return None
        else:
            print(f"Error fetching chart data for pool {pool_name}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching chart data for pool {pool_name}: {e}")
        return None

def merge_and_save_pool_data(pool_data, db_filename="defi_prime_rate.db"):
    """
    Merge all pool data by date and save to SQLite database
    
    Args:
        pool_data (dict): Dictionary containing pool data
        db_filename (str): SQLite database filename
        
    Returns:
        pd.DataFrame: Merged and cleaned DataFrame, or None if failed
    """
    print("Merging pool data...")
    merged_df = None
    
    # Merge all pool dataframes
    for pool_id, pool_info in pool_data.items():
        df = pool_info['data']
        pool_name = pool_info['name']
        
        # Convert index to date only
        df = df.copy()
        df.index = pd.to_datetime(df.index).date
        df.index.name = 'date'
        
        # Select only numeric columns (apy and tvlUsd)
        numeric_cols = ['apy', 'tvlUsd']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) == 2:
            df_subset = df[available_cols].copy()
            
            # Aggregate by mean if multiple entries per day
            if len(df_subset) != len(df_subset.groupby(df_subset.index).size()):
                df_subset = df_subset.groupby(df_subset.index).mean()
            
            # Rename columns to avoid conflicts
            df_subset = df_subset.rename(columns={
                'apy': f'apy_{pool_name}',
                'tvlUsd': f'tvlUsd_{pool_name}'
            })
            
            # Merge with existing data
            if merged_df is None:
                merged_df = df_subset
            else:
                merged_df = pd.merge(merged_df, df_subset, left_index=True, right_index=True, how='outer')
    
    if merged_df is None or merged_df.empty:
        print("No data to merge. Exiting.")
        return None
    
    print(f"Successfully merged data for {len(pool_data)} pools")
    
    # Clean up data by removing NaN values
    print("Cleaning data by removing NaN values...")
    
    # Drop columns that are all NaN (pools with no data)
    initial_cols = len(merged_df.columns)
    merged_df = merged_df.dropna(axis=1, how='all')
    dropped_cols = initial_cols - len(merged_df.columns)
    if dropped_cols > 0:
        print(f"Dropped {dropped_cols} columns with all NaN values")
    
    # Drop rows that are all NaN (dates with no data)
    initial_rows = len(merged_df)
    merged_df = merged_df.dropna(axis=0, how='all')
    dropped_rows = initial_rows - len(merged_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with all NaN values")
    
    # Drop pools with incomplete data (either APY or TVL all NaN)
    apy_cols = [col for col in merged_df.columns if col.startswith('apy_')]
    tvl_cols = [col for col in merged_df.columns if col.startswith('tvlUsd_')]
    
    pools_to_drop = []
    for i, apy_col in enumerate(apy_cols):
        if i < len(tvl_cols):
            tvl_col = tvl_cols[i]
            if merged_df[apy_col].isna().all() or merged_df[tvl_col].isna().all():
                pools_to_drop.extend([apy_col, tvl_col])
    
    if pools_to_drop:
        merged_df = merged_df.drop(columns=pools_to_drop)
        print(f"Dropped {len(pools_to_drop)//2} pools with incomplete data")
    
    print(f"Final dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Calculate weighted average APY
    print("Calculating weighted average APY...")
    apy_cols = [col for col in merged_df.columns if col.startswith('apy_')]
    tvl_cols = [col for col in merged_df.columns if col.startswith('tvlUsd_')]
    
    merged_df['weighted_apy'] = 0
    total_tvl = merged_df[tvl_cols].sum(axis=1)
    
    for i in range(len(apy_cols)):
        apy_col = apy_cols[i]
        tvl_col = tvl_cols[i]
        merged_df['weighted_apy'] += merged_df[apy_col].fillna(0) * merged_df[tvl_col].fillna(0)
    
    merged_df['weighted_apy'] = merged_df['weighted_apy'] / total_tvl.replace(0, 1)
    
    # Calculate 14-day moving average
    merged_df['ma_apy_14d'] = merged_df['weighted_apy'].rolling(window=14, min_periods=1).mean()
    
    # Save to SQLite database
    print(f"Saving data to SQLite database: {db_filename}")
    conn = sqlite3.connect(db_filename)
    
    # Save main data
    merged_df.to_sql('pool_data', conn, if_exists='replace', index=True)
    
    # Save pool metadata (only for pools in final dataset)
    pool_metadata = []
    final_pool_names = set()
    
    # Extract pool names from column names
    for col in merged_df.columns:
        if col.startswith('apy_'):
            pool_name = col[4:]  # Remove 'apy_' prefix
            final_pool_names.add(pool_name)
    
    for pool_id, pool_info in pool_data.items():
        pool_name = pool_info['name']
        if pool_name in final_pool_names:
            pool_metadata.append({
                'pool_id': pool_id,
                'name': pool_name,
                'current_tvl': pool_info['current_tvl'],
                'current_apy': pool_info['current_apy'],
                'last_updated': datetime.now().isoformat()
            })
    
    metadata_df = pd.DataFrame(pool_metadata)
    metadata_df.to_sql('pool_metadata', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Data successfully saved to {db_filename}")
    print(f"Final dataset contains {len(final_pool_names)} pools with valid data")
    
    return merged_df

def load_data_from_db(db_filename="defi_prime_rate.db"):
    """
    Load data from SQLite database
    
    Args:
        db_filename (str): SQLite database filename
        
    Returns:
        tuple: (merged_df, metadata_df) or (None, None) if failed
    """
    try:
        print(f"Loading data from {db_filename}...")
        conn = sqlite3.connect(db_filename)
        
        # Check what tables exist
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {[table[0] for table in tables]}")
        
        # Load main data
        try:
            merged_df = pd.read_sql('SELECT * FROM pool_data', conn, index_col='index')
        except:
            try:
                merged_df = pd.read_sql('SELECT * FROM pool_data', conn)
                if len(merged_df.columns) > 0:
                    first_col = merged_df.columns[0]
                    if 'date' in first_col.lower() or 'time' in first_col.lower():
                        merged_df.set_index(first_col, inplace=True)
                    else:
                        merged_df.set_index(first_col, inplace=True)
            except Exception as e:
                print(f"Error reading pool_data table: {e}")
                cursor.execute("PRAGMA table_info(pool_data);")
                columns = cursor.fetchall()
                print("Pool_data table structure:")
                for col in columns:
                    print(f"  {col[1]} ({col[2]})")
                return None, None
        
        # Convert index to datetime if needed
        if not isinstance(merged_df.index, pd.DatetimeIndex):
            try:
                merged_df.index = pd.to_datetime(merged_df.index)
            except:
                print("Warning: Could not convert index to datetime")
        
        # Load metadata
        try:
            metadata_df = pd.read_sql('SELECT * FROM pool_metadata', conn)
        except Exception as e:
            print(f"Error reading pool_metadata table: {e}")
            metadata_df = pd.DataFrame()
        
        conn.close()
        
        print(f"Successfully loaded data for {len(metadata_df)} pools")
        return merged_df, metadata_df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None

def main():
    """
    Main function to fetch, process, and save DeFi Prime Rate data
    """
    # Fetch top 100 stablecoin pools by TVL
    print("\n=== Fetching top 100 stablecoin pools by TVL ===")
    top_pools = fetch_top_stablecoin_pools_by_tvl(100)
    
    if not top_pools:
        print("No pools fetched. Exiting.")
        return
    
    # Display top pools info
    print("\nTop 10 pools by TVL:")
    for i, pool in enumerate(top_pools[:10]):
        print(f"{i+1}. {pool.get('name', 'Unknown')} - TVL: ${pool['tvlUsd']:,.0f} - APY: {pool.get('apy', 0):.2f}%")
    
    # Fetch historical data for each pool
    print(f"\nFetching historical data for {len(top_pools)} pools...")
    pool_data = {}
    
    for i, pool in enumerate(top_pools):
        pool_id = pool['pool']
        pool_name = pool.get('name', f'Pool_{i}')
        
        df = fetch_pool_chart_data(pool_id, pool_name)
        if df is not None and not df.empty:
            pool_data[pool_id] = {
                'data': df,
                'name': pool_name,
                'current_tvl': pool['tvlUsd'],
                'current_apy': pool.get('apy', 0)
            }
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    if not pool_data:
        print("No pool data fetched successfully. Exiting.")
        return
    
    # Merge and save data to database
    merged_df = merge_and_save_pool_data(pool_data)
    
    if merged_df is None:
        print("Failed to merge and save data. Exiting.")
        return
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Database contains data for {len(pool_data)} pools")
    print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    print(f"Total data points: {len(merged_df)}")
    
    if 'weighted_apy' in merged_df.columns:
        current_apy = merged_df['weighted_apy'].iloc[-1]
        mean_apy = merged_df['weighted_apy'].mean()
        print(f"Current DeFi Prime Rate: {current_apy:.4f}%")
        print(f"Mean DeFi Prime Rate: {mean_apy:.4f}%")

if __name__ == "__main__":
    main()
