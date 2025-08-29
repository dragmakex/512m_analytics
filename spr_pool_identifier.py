#!/usr/bin/env python3
"""
Stablecoin Pool Data Extractor

This script connects to the defi_prime_rate.db database and extracts
the latest APY and TVL data for pools 0-15.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple


def connect_to_database(db_path: str = "defi_prime_rate.db") -> sqlite3.Connection:
    """
    Connect to the SQLite database.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        SQLite connection object
        
    Raises:
        FileNotFoundError: If database file doesn't exist
        sqlite3.Error: If connection fails
    """
    try:
        conn = sqlite3.connect(db_path)
        print(f"Successfully connected to database: {db_path}")
        return conn
    except FileNotFoundError:
        raise FileNotFoundError(f"Database file not found: {db_path}")
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to connect to database: {e}")


def get_latest_pool_data(conn: sqlite3.Connection, pool_range: Tuple[int, int] = (0, 15)) -> pd.DataFrame:
    """
    Get the latest APY and TVL data for the specified pool range.
    
    Args:
        conn: Database connection
        pool_range: Tuple of (start_pool, end_pool) inclusive
        
    Returns:
        DataFrame with latest pool data
    """
    start_pool, end_pool = pool_range
    
    # Build the query to get the latest data for pools in the specified range
    apy_columns = []
    tvl_columns = []
    
    for i in range(start_pool, end_pool + 1):
        apy_columns.append(f"apy_Pool_{i}")
        tvl_columns.append(f"tvlUsd_Pool_{i}")
    
    # Get the latest date
    latest_date_query = "SELECT MAX(date) FROM pool_data"
    latest_date = pd.read_sql_query(latest_date_query, conn).iloc[0, 0]
    
    print(f"Latest data date: {latest_date}")
    
    # Get data for the latest date
    columns = ["date"] + apy_columns + tvl_columns
    columns_str = ", ".join(columns)
    
    query = f"""
    SELECT {columns_str}
    FROM pool_data 
    WHERE date = ?
    """
    
    latest_data = pd.read_sql_query(query, conn, params=[latest_date])
    
    if latest_data.empty:
        print("No data found for the latest date")
        return pd.DataFrame()
    
    return latest_data


def format_pool_summary(data: pd.DataFrame, pool_range: Tuple[int, int] = (0, 15)) -> List[Dict]:
    """
    Format the pool data into a readable summary.
    
    Args:
        data: DataFrame with pool data
        pool_range: Tuple of (start_pool, end_pool) inclusive
        
    Returns:
        List of dictionaries with formatted pool information
    """
    if data.empty:
        return []
    
    start_pool, end_pool = pool_range
    pool_summary = []
    
    for i in range(start_pool, end_pool + 1):
        apy_col = f"apy_Pool_{i}"
        tvl_col = f"tvlUsd_Pool_{i}"
        
        if apy_col in data.columns and tvl_col in data.columns:
            apy_value = data[apy_col].iloc[0]
            tvl_value = data[tvl_col].iloc[0]
            
            # Handle NaN values
            if pd.isna(apy_value) or pd.isna(tvl_value):
                pool_summary.append({
                    'pool_id': f"Pool_{i}",
                    'apy': 'N/A',
                    'tvl_usd': 'N/A',
                    'tvl_formatted': 'N/A'
                })
            else:
                # Format TVL with appropriate units
                if tvl_value >= 1e9:
                    tvl_formatted = f"${tvl_value/1e9:.2f}B"
                elif tvl_value >= 1e6:
                    tvl_formatted = f"${tvl_value/1e6:.2f}M"
                elif tvl_value >= 1e3:
                    tvl_formatted = f"${tvl_value/1e3:.2f}K"
                else:
                    tvl_formatted = f"${tvl_value:.2f}"
                
                pool_summary.append({
                    'pool_id': f"Pool_{i}",
                    'apy': f"{apy_value:.4f}%",
                    'tvl_usd': tvl_value,
                    'tvl_formatted': tvl_formatted
                })
    
    return pool_summary


def print_pool_summary(pool_summary: List[Dict]) -> None:
    """
    Print a formatted summary of pool data.
    
    Args:
        pool_summary: List of pool data dictionaries
    """
    if not pool_summary:
        print("No pool data to display")
        return
    
    print("\n" + "="*80)
    print("STABLECOIN POOL DATA SUMMARY")
    print("="*80)
    print(f"{'Pool ID':<12} {'APY':<12} {'TVL (USD)':<20}")
    print("-"*80)
    
    for pool in pool_summary:
        print(f"{pool['pool_id']:<12} {pool['apy']:<12} {pool['tvl_formatted']:<20}")
    
    print("-"*80)
    
    # Calculate summary statistics
    valid_pools = [p for p in pool_summary if p['tvl_usd'] != 'N/A']
    if valid_pools:
        total_tvl = sum(p['tvl_usd'] for p in valid_pools)
        avg_apy = sum(float(p['apy'].rstrip('%')) for p in valid_pools) / len(valid_pools)
        
        if total_tvl >= 1e9:
            total_tvl_formatted = f"${total_tvl/1e9:.2f}B"
        elif total_tvl >= 1e6:
            total_tvl_formatted = f"${total_tvl/1e6:.2f}M"
        else:
            total_tvl_formatted = f"${total_tvl:,.0f}"
        
        print(f"Total TVL: {total_tvl_formatted}")
        print(f"Average APY: {avg_apy:.4f}%")
        print(f"Active Pools: {len(valid_pools)}/{len(pool_summary)}")


def main():
    """
    Main function to extract and display pool data.
    """
    try:
        # Connect to database
        conn = connect_to_database()
        
        # Get latest data for pools 0-15
        print("Extracting data for pools 0-15...")
        latest_data = get_latest_pool_data(conn, pool_range=(0, 15))
        
        if not latest_data.empty:
            # Format and display the data
            pool_summary = format_pool_summary(latest_data, pool_range=(0, 15))
            print_pool_summary(pool_summary)
        else:
            print("No data available for the specified pools")
        
        # Close connection
        conn.close()
        print("\nDatabase connection closed")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the defi_prime_rate.db file exists in the current directory")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
