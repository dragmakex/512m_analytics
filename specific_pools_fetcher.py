import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Load environment variables
load_dotenv()

# DeFiLlama API endpoints
CHART_ENDPOINT = "https://yields.llama.fi/chart/"

# Specific pool IDs to fetch with friendly names
POOL_IDS = [
    "aa70268e-4b52-42bf-a116-608b370f9501",
    "f981a304-bb6c-45b8-b0c5-fd2f515ad23a"
]

# Pool name mapping
POOL_NAMES = {
    "aa70268e-4b52-42bf-a116-608b370f9501": "USDC",
    "f981a304-bb6c-45b8-b0c5-fd2f515ad23a": "USDT"
}

# Theme palette for academic-style plotting
theme_palette = ['#f7f3ec', '#ede4da', '#b9a58f', '#574c40', '#36312a']
muted_blues = [
    '#2b3e50', '#3c5a77', '#4f7192', '#5f86a8', '#6f9bbd',
    '#86abc7', '#9bbad1', '#afc8da', '#c3d5e3', '#d7e2ec'
]

# Set academic-style plotting with serif fonts and beige background
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'figure.facecolor': theme_palette[0],
    'axes.facecolor': theme_palette[0],
    'savefig.facecolor': theme_palette[0]
})

def add_logo_overlay(ax, logo_path="512m_logo.png", alpha=0.05):
    """
    Add logo overlay to the center of the plot
    
    Args:
        ax: matplotlib axis object
        logo_path (str): path to logo image
        alpha (float): transparency level (0-1)
    """
    try:
        # Load the logo image
        logo_img = Image.open(logo_path)
        
        # Convert to numpy array and normalize
        logo_array = np.array(logo_img)
        
        # Get the center of the plot
        x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        
        # Calculate appropriate size for the logo (about 30% of plot width)
        plot_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        logo_width = plot_width * 0.3
        
        # Create offset image
        im = OffsetImage(logo_array, zoom=logo_width/logo_img.width, alpha=alpha)
        
        # Create annotation box at center
        ab = AnnotationBbox(im, (x_center, y_center), frameon=False)
        ax.add_artist(ab)
        
    except Exception as e:
        print(f"Warning: Could not add logo overlay: {e}")

def fetch_ethereum_price_data(start_date, end_date):
    """
    Fetch Ethereum price data from Polygon API
    
    Args:
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
        
    Returns:
        pd.DataFrame: DataFrame with Ethereum price data, or None if failed
    """
    try:
        api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            print("Warning: POLYGON_API_KEY not found in environment variables")
            return None
        
        # Convert dates to string format for Polygon API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Polygon API endpoint for Ethereum daily prices
        url = f"https://api.polygon.io/v2/aggs/ticker/X:ETHUSD/range/1/day/{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
        
        print(f"Fetching Ethereum price data from {start_str} to {end_str}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('results'):
                # Extract price data
                price_data = []
                for result in data['results']:
                    price_data.append({
                        'date': datetime.fromtimestamp(result['t'] / 1000),
                        'open': result['o'],
                        'high': result['h'],
                        'low': result['l'],
                        'close': result['c'],
                        'volume': result['v']
                    })
                
                df = pd.DataFrame(price_data)
                df.set_index('date', inplace=True)
                
                print(f"Successfully fetched {len(df)} days of Ethereum price data")
                return df
            else:
                print("No results found in Polygon API response")
                return None
        else:
            print(f"Error fetching Ethereum price data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching Ethereum price data: {e}")
        return None

def fetch_pool_chart_data(pool_id, days=700):
    """
    Fetch historical chart data for a specific pool with rate limiting
    
    Args:
        pool_id (str): Pool ID from DeFiLlama
        days (int): Number of days of historical data to fetch (max 1095 = 3 years)
        
    Returns:
        pd.DataFrame: DataFrame with historical APY and TVL data, or None if failed
    """
    try:
        print(f"Fetching data for pool {pool_id}...")
        url = f"{CHART_ENDPOINT}{pool_id}"
        response = requests.get(url)
        
        # Handle rate limiting
        if response.status_code == 429:
            print(f"Rate limited for pool {pool_id}, waiting 2 seconds...")
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
                            print(f"Could not parse timestamp format for pool {pool_id}")
                            return None
                    
                    df.set_index('date', inplace=True)
                    
                    # Make index timezone-naive
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Filter to last N days
                    cutoff_date = datetime.now() - timedelta(days=days)
                    df = df[df.index >= cutoff_date]
                    
                    print(f"Successfully fetched {len(df)} data points for pool {pool_id}")
                    return df
                else:
                    print(f"No timestamp found in data for pool {pool_id}")
                    return None
            else:
                print(f"Unexpected data format for pool {pool_id}")
                return None
        else:
            print(f"Error fetching chart data for pool {pool_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching chart data for pool {pool_id}: {e}")
        return None

def plot_pool_apy_trends(pool_data, price_data):
    """
    Create a figure with two subplots:
    1) 7-day moving average APY values
    2) Ethereum price 
    
    Args:
        pool_data (dict): Dictionary containing pool data
        price_data (pd.DataFrame): DataFrame containing Ethereum price data
    """
    if not pool_data:
        print("No pool data provided for plotting")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: 7-day moving average APY values for each pool
    for i, (pool_id, pool_info) in enumerate(pool_data.items()):
        df = pool_info['data']
        pool_name = POOL_NAMES.get(pool_id, f"Pool_{pool_id}")
        
        # Calculate 7-day moving average of APY
        if 'apy' in df.columns:
            ma_apy = df['apy'].rolling(window=7, min_periods=1).mean()
            
            # Use different colors from the muted_blues palette
            color = muted_blues[i % len(muted_blues)]
            
            ax1.plot(
                df.index,
                ma_apy,
                label=pool_name,
                linewidth=2,
                alpha=0.8,
                color=color
            )
    
    ax1.set_title('7-Day Moving Average APY on AAVE V3')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('APY (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add logo overlay to first subplot
    add_logo_overlay(ax1)
    
    # Plot 2: Ethereum price
    if price_data is not None and not price_data.empty and 'close' in price_data.columns:
        ax2.plot(
            price_data.index,
            price_data['close'],
            label='ETH Price',
            linewidth=2,
            color=theme_palette[3],  # 4th color in palette
            alpha=0.8
        )
        
        ax2.set_title('Ethereum Price (USD)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add logo overlay to second subplot
        add_logo_overlay(ax2)
    else:
        ax2.text(0.5, 0.5, 'Ethereum price data not available', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color=theme_palette[4])
        ax2.set_title('Ethereum Price')
    
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to fetch specific pool data and create plots
    """
    print("=== Specific Pool APY Analysis ===")
    print(f"Fetching data for {len(POOL_IDS)} specific pools...")
    
    # Fetch data for each pool
    pool_data = {}
    
    for pool_id in POOL_IDS:
        df = fetch_pool_chart_data(pool_id, days=700)  # 3 years of data
        if df is not None and not df.empty:
            pool_data[pool_id] = {
                'data': df,
                'id': pool_id
            }
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
    
    if not pool_data:
        print("No pool data fetched successfully. Exiting.")
        return
    
    print(f"\nSuccessfully fetched data for {len(pool_data)} pools")
    
    # Create plots
    print("\nCreating plots...")
    
    # Fetch Ethereum price data for the last 365 days
    start_date = datetime.now() - timedelta(days=700)
    end_date = datetime.now()
    price_data = fetch_ethereum_price_data(start_date, end_date)
    
    # Create plots
    plot_pool_apy_trends(pool_data, price_data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
