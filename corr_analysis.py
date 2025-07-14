# -----------------------------------------------------------------------------
# There's still issues with the calculation of the rolling correlation
# Will be fixed in the future -> check the raw data pulled from Polygon.io
# Don't worry about it for now
# -----------------------------------------------------------------------------

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set academic-style plotting
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
    'grid.alpha': 0.3
})

# Use a limited color palette
colors = ['#440154', '#31688e', '#35b779', '#90d743', '#fde725']

class PolygonDataFetcher:
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        
        self.base_url = "https://api.polygon.io"
        
    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetch stock data from Polygon.io"""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                df = df[['c']]  # Keep only close price
                df.columns = [ticker]
                return df
            else:
                print(f"No data returned for {ticker}. Response: {data}")
                return None
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def fetch_crypto_data(self, ticker, start_date, end_date):
        """Fetch crypto data from Polygon.io"""
        # For crypto, we need to use the crypto endpoint
        url = f"{self.base_url}/v2/aggs/ticker/X:{ticker}USD/range/1/day/{start_date}/{end_date}"
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                df = df[['c']]  # Keep only close price
                df.columns = [ticker]
                return df
            else:
                print(f"No data returned for {ticker}. Response: {data}")
                return None
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

def fetch_all_data():
    """Fetch data for all assets"""
    fetcher = PolygonDataFetcher()
    
    # Calculate date range (400 days from today)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Define assets
    assets = {
        'BTC': 'crypto',
        'ETH': 'crypto', 
        'SPY': 'stock',  # S&P 500 ETF
        'GLD': 'stock'   # Gold ETF
    }
    
    all_data = {}
    
    for ticker, asset_type in assets.items():
        print(f"Fetching {ticker} data...")
        
        if asset_type == 'crypto':
            df = fetcher.fetch_crypto_data(ticker, start_date, end_date)
        else:
            df = fetcher.fetch_stock_data(ticker, start_date, end_date)
        
        if df is not None:
            all_data[ticker] = df
            print(f"Successfully fetched {len(df)} data points for {ticker}")
        
        # Rate limiting
        time.sleep(0.1)
    
    return all_data

def calculate_rolling_correlation_heatmaps(data_dict):
    # Merge all dataframes
    merged_df = None
    for ticker, df in data_dict.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')
    merged_df = merged_df.sort_index().ffill().bfill()

    # Calculate returns
    returns_df = merged_df.pct_change().dropna()

    pairs = [
        ('BTC', 'SPY', 'Bitcoin vs S&P 500'),
        ('ETH', 'SPY', 'Ethereum vs S&P 500'),
        ('GLD', 'SPY', 'Gold vs S&P 500'),
        ('BTC', 'ETH', 'Bitcoin vs Ethereum')
    ]
    window_sizes = np.arange(14, 91)
    dates = returns_df.index

    heatmaps = {}
    for asset1, asset2, _ in pairs:
        heatmap = np.full((len(window_sizes), len(dates)), np.nan)
        for i, w in enumerate(window_sizes):
            # Use center alignment for rolling window
            corr = returns_df[asset1].rolling(window=w, center=True).corr(returns_df[asset2])
            heatmap[i, :] = corr.values
        heatmaps[(asset1, asset2)] = heatmap
    return heatmaps, window_sizes, dates, pairs


def plot_correlation_heatmaps(heatmaps, window_sizes, dates, pairs):
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    axes = axes.flatten()
    # Only plot the last 200 days
    if len(dates) > 200:
        plot_dates = dates[-200:]
        plot_idx = np.arange(len(dates))[-200:]
    else:
        plot_dates = dates
        plot_idx = np.arange(len(dates))
    for i, (asset1, asset2, title) in enumerate(pairs):
        ax = axes[i]
        heatmap = heatmaps[(asset1, asset2)][:, plot_idx]
        vmin = np.nanmin(heatmap)
        vmax = np.nanmax(heatmap)
        im = ax.imshow(
            heatmap,
            aspect='auto',
            origin='lower',
            extent=[mdates.date2num(plot_dates[0]), mdates.date2num(plot_dates[-1]), window_sizes[0], window_sizes[-1]],
            cmap='coolwarm',
            vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.set_ylabel('Window Size (days)')
        ax.set_xlabel('Date')
        ax.xaxis_date()
        fig.colorbar(im, ax=ax, orientation='vertical', label='Correlation')
    plt.tight_layout()
    plt.show()


def main():
    print("Starting correlation analysis...")
    data_dict = fetch_all_data()
    if not data_dict:
        print("No data fetched. Exiting.")
        return
    print(f"Successfully fetched data for {len(data_dict)} assets")
    print("Calculating rolling correlation heatmaps...")
    heatmaps, window_sizes, dates, pairs = calculate_rolling_correlation_heatmaps(data_dict)
    print("Creating correlation heatmaps...")
    plot_correlation_heatmaps(heatmaps, window_sizes, dates, pairs)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
