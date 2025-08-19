# -----------------------------------------------------------------------------
# There's still issues with the calculation of the rolling correlation
# Will be fixed in the future -> check the raw data pulled from Polygon.io
# Don't worry about it for now
# -----------------------------------------------------------------------------

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Theme palette
theme_palette = ['#f7f3ec', '#ede4da', '#b9a58f', '#574c40', '#36312a']
muted_blues = [
    '#2b3e50', '#3c5a77', '#4f7192', '#5f86a8', '#6f9bbd',
    '#86abc7', '#9bbad1', '#afc8da', '#c3d5e3', '#d7e2ec'
]

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
    'grid.alpha': 0.3,
    'figure.facecolor': theme_palette[0],
    'axes.facecolor': theme_palette[0],
    'savefig.facecolor': theme_palette[0]
})

def add_logo_overlay(ax, logo_path="512m_logo.png", alpha=0.1):
    """Add logo overlay to the center of the plot"""
    try:
        logo_img = Image.open(logo_path)
        logo_array = np.array(logo_img)
        x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        plot_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        logo_width = plot_width * 0.4
        im = OffsetImage(logo_array, zoom=logo_width/logo_img.width, alpha=alpha)
        ab = AnnotationBbox(im, (x_center, y_center), frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        print(f"Warning: Could not add logo overlay: {e}")

def fetch_polygon_data(symbol, start_date, end_date, api_key):
    """Fetch data from Polygon.io API"""
    base_url = "https://api.polygon.io"
    
    if symbol in ['BTC', 'ETH']:
        url = f"{base_url}/v2/aggs/ticker/X:{symbol}USD/range/1/day/{start_date}/{end_date}"
    else:
        url = f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    
    params = {
        'apiKey': api_key,
        'adjusted': 'true',
        'sort': 'asc'
    }
    
    try:
        print(f"Fetching data for {symbol}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('results'):
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df[['date', 'c']].rename(columns={'c': 'close'})
            df.set_index('date', inplace=True)
            
            print(f"Successfully fetched {len(df)} data points for {symbol}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            return df
        else:
            print(f"No data returned for {symbol}. Response: {data}")
            return None
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_rolling_correlation_detailed(returns_df, asset1, asset2, window=30):
    """
    Calculate rolling correlation with detailed debugging information
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        asset1 (str): First asset name
        asset2 (str): Second asset name
        window (int): Rolling window size
        
    Returns:
        pd.Series: Rolling correlation series
    """
    print(f"\n=== Detailed 30-day Correlation Analysis for {asset1} vs {asset2} ===")
    
    # Calculate rolling correlation
    corr_series = returns_df[asset1].rolling(window=window, center=False, min_periods=window//2).corr(returns_df[asset2])
    
    # Get valid correlations
    valid_corr = corr_series.dropna()
    
    if len(valid_corr) == 0:
        print(f"No valid correlations found for {window}-day window")
        return corr_series
    
    print(f"Total observations: {len(returns_df)}")
    print(f"Valid correlations: {len(valid_corr)}")
    print(f"Correlation statistics:")
    print(f"  Mean: {valid_corr.mean():.4f}")
    print(f"  Std: {valid_corr.std():.4f}")
    print(f"  Min: {valid_corr.min():.4f}")
    print(f"  Max: {valid_corr.max():.4f}")
    print(f"  Median: {valid_corr.median():.4f}")
    
    # Check for high correlations
    high_corr_90 = (valid_corr > 0.9).sum()
    high_corr_80 = (valid_corr > 0.8).sum()
    high_corr_70 = (valid_corr > 0.7).sum()
    
    print(f"Correlations > 90%: {high_corr_90} ({high_corr_90/len(valid_corr)*100:.1f}%)")
    print(f"Correlations > 80%: {high_corr_80} ({high_corr_80/len(valid_corr)*100:.1f}%)")
    print(f"Correlations > 70%: {high_corr_70} ({high_corr_70/len(valid_corr)*100:.1f}%)")
    
    # Show recent correlations
    recent_corr = valid_corr.tail(10)
    print(f"Recent 10 correlations: {recent_corr.values}")
    
    # Show highest correlations
    top_corr = valid_corr.nlargest(5)
    print(f"Top 5 correlations: {top_corr.values}")
    print(f"Top 5 correlation dates: {top_corr.index}")
    
    # Show lowest correlations
    bottom_corr = valid_corr.nsmallest(5)
    print(f"Bottom 5 correlations: {bottom_corr.values}")
    print(f"Bottom 5 correlation dates: {bottom_corr.index}")
    
    return corr_series

def plot_correlation_analysis(returns_df, corr_btc, corr_eth):
    """Create comprehensive correlation analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot 1: 30-day rolling correlations
    ax1 = axes[0]
    ax1.plot(corr_btc.index, corr_btc, label='BTC vs S&P 500', linewidth=2, color=muted_blues[0])
    ax1.plot(corr_eth.index, corr_eth, label='ETH vs S&P 500', linewidth=2, color=muted_blues[2])
    ax1.axhline(y=0, color=theme_palette[3], linestyle='--', alpha=0.7, linewidth=1)
    ax1.axhline(y=0.9, color=theme_palette[4], linestyle=':', alpha=0.7, linewidth=1.5, label='90% Correlation')
    ax1.axhline(y=0.8, color=theme_palette[2], linestyle=':', alpha=0.5, linewidth=1, label='80% Correlation')
    ax1.axhline(y=0.5, color=theme_palette[2], linestyle=':', alpha=0.5, linewidth=1, label='50% Correlation')
    
    ax1.set_title('30-Day Rolling Correlation: Cryptocurrencies vs S&P 500')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Correlation distribution
    ax2 = axes[1]
    valid_btc = corr_btc.dropna()
    valid_eth = corr_eth.dropna()
    
    ax2.hist(valid_btc, bins=30, alpha=0.7, label='BTC vs S&P 500', color=muted_blues[0], density=True)
    ax2.hist(valid_eth, bins=30, alpha=0.7, label='ETH vs S&P 500', color=muted_blues[2], density=True)
    ax2.axvline(x=0.9, color=theme_palette[4], linestyle=':', alpha=0.7, linewidth=1.5, label='90% Correlation')
    ax2.axvline(x=0.8, color=theme_palette[2], linestyle=':', alpha=0.5, linewidth=1, label='80% Correlation')
    
    ax2.set_title('Distribution of 30-Day Correlations')
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Returns scatter plot (recent data)
    ax3 = axes[2]
    recent_returns = returns_df.tail(100)  # Last 100 days
    
    ax3.scatter(recent_returns['SPY'], recent_returns['BTC'], alpha=0.6, color=muted_blues[0], s=20, label='BTC')
    ax3.scatter(recent_returns['SPY'], recent_returns['ETH'], alpha=0.6, color=muted_blues[2], s=20, label='ETH')
    
    # Add trend lines
    z_btc = np.polyfit(recent_returns['SPY'], recent_returns['BTC'], 1)
    z_eth = np.polyfit(recent_returns['SPY'], recent_returns['ETH'], 1)
    p_btc = np.poly1d(z_btc)
    p_eth = np.poly1d(z_eth)
    
    x_range = np.linspace(recent_returns['SPY'].min(), recent_returns['SPY'].max(), 100)
    ax3.plot(x_range, p_btc(x_range), color=muted_blues[0], linewidth=2, alpha=0.8)
    ax3.plot(x_range, p_eth(x_range), color=muted_blues[2], linewidth=2, alpha=0.8)
    
    ax3.set_title('Recent Returns Scatter Plot (Last 100 Days)')
    ax3.set_xlabel('S&P 500 Returns')
    ax3.set_ylabel('Crypto Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling correlation by different window sizes
    ax4 = axes[3]
    windows = [15, 30, 60, 90, 120]
    colors = [muted_blues[i] for i in range(len(windows))]
    
    for i, window in enumerate(windows):
        corr = returns_df['BTC'].rolling(window=window, center=False, min_periods=window//2).corr(returns_df['SPY'])
        ax4.plot(corr.index, corr, label=f'{window}-day', linewidth=1.5, color=colors[i], alpha=0.8)
    
    ax4.axhline(y=0.9, color=theme_palette[4], linestyle=':', alpha=0.7, linewidth=1.5, label='90% Correlation')
    ax4.set_title('BTC vs S&P 500: Rolling Correlation by Window Size')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Correlation Coefficient')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1, 1)
    
    # Format x-axis dates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add logo overlays
    for ax in axes:
        add_logo_overlay(ax)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to perform detailed correlation analysis"""
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set")
        return
    
    # Calculate date range (2 years from today)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Fetch data for each asset
    assets = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum', 
        'SPY': 'S&P 500 ETF'
    }
    
    data_frames = {}
    
    for symbol in assets.keys():
        df = fetch_polygon_data(symbol, start_date, end_date, api_key)
        if df is not None:
            data_frames[symbol] = df
        else:
            print(f"Failed to fetch data for {symbol}. Exiting.")
            return
    
    # Create combined dataset using crypto daily data
    btc_dates = data_frames['BTC'].index
    combined_df = pd.DataFrame(index=btc_dates)
    
    # Add crypto data
    combined_df['BTC'] = data_frames['BTC']['close']
    combined_df['ETH'] = data_frames['ETH']['close']
    
    # Add SPY data, forward-filling to get the most recent trading day price
    spy_series = data_frames['SPY']['close']
    combined_df['SPY'] = spy_series.reindex(btc_dates, method='ffill')
    
    # Drop any rows with NaN values
    combined_df = combined_df.dropna()
    
    # Check for weekend effect and create SPY-only dataset
    spy_dates = data_frames['SPY'].index
    spy_only_df = pd.DataFrame(index=spy_dates)
    spy_only_df['SPY'] = data_frames['SPY']['close']
    spy_only_df['BTC'] = data_frames['BTC']['close'].reindex(spy_dates, method='ffill')
    spy_only_df['ETH'] = data_frames['ETH']['close'].reindex(spy_dates, method='ffill')
    spy_only_df = spy_only_df.dropna()
    
    # Check weekend effect
    spy_returns = combined_df['SPY'].pct_change(fill_method=None)
    zero_spy_days = (spy_returns == 0).sum()
    total_days = len(spy_returns.dropna())
    print(f"\nWeekend effect check:")
    print(f"  Days with zero SPY returns: {zero_spy_days} out of {total_days} ({100*zero_spy_days/total_days:.1f}%)")
    
    spy_only_returns = spy_only_df.pct_change(fill_method=None).dropna()
    zero_spy_only_days = (spy_only_returns['SPY'] == 0).sum()
    print(f"  Days with zero SPY returns (SPY-only): {zero_spy_only_days} out of {len(spy_only_returns)} ({100*zero_spy_only_days/len(spy_only_returns):.1f}%)")
    
    print(f"\nCombined data shape: {combined_df.shape}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # Calculate returns
    returns_df = combined_df.pct_change(fill_method=None).dropna()
    print(f"Returns data shape: {returns_df.shape}")
    
    # Use SPY-only returns for correlation analysis (avoids weekend effect)
    print(f"\nUsing SPY-only returns for correlation analysis...")
    print(f"SPY-only returns shape: {spy_only_returns.shape}")
    
    # Calculate detailed 30-day correlations using SPY-only data
    corr_btc = calculate_rolling_correlation_detailed(spy_only_returns, 'BTC', 'SPY', window=30)
    corr_eth = calculate_rolling_correlation_detailed(spy_only_returns, 'ETH', 'SPY', window=30)
    
    # Create comprehensive plots
    print("\nCreating correlation analysis plots...")
    plot_correlation_analysis(spy_only_returns, corr_btc, corr_eth)
    
    # Additional analysis: Check if using different data sources would help
    print("\n=== Additional Analysis ===")
    print("Comparing different data alignment approaches:")
    
    # Approach 1: Current approach (crypto daily, SPY forward-filled)
    print("\nApproach 1: Crypto daily, SPY forward-filled")
    corr1_btc = returns_df['BTC'].rolling(window=30, center=False, min_periods=15).corr(returns_df['SPY'])
    print(f"BTC max correlation: {corr1_btc.max():.4f}")
    print(f"BTC correlations > 90%: {(corr1_btc > 0.9).sum()}")
    
    # Approach 2: Use only SPY trading days (recommended)
    print("\nApproach 2: SPY trading days only (recommended)")
    corr2_btc = spy_only_returns['BTC'].rolling(window=30, center=False, min_periods=15).corr(spy_only_returns['SPY'])
    print(f"BTC max correlation: {corr2_btc.max():.4f}")
    print(f"BTC correlations > 90%: {(corr2_btc > 0.9).sum()}")
    
    # Approach 3: Use log returns with SPY-only data
    print("\nApproach 3: Log returns (SPY-only)")
    log_returns_df = np.log(spy_only_df / spy_only_df.shift(1)).dropna()
    corr3_btc = log_returns_df['BTC'].rolling(window=30, center=False, min_periods=15).corr(log_returns_df['SPY'])
    print(f"BTC max correlation: {corr3_btc.max():.4f}")
    print(f"BTC correlations > 90%: {(corr3_btc > 0.9).sum()}")
    
    print("\nCorrelation analysis completed!")

if __name__ == "__main__":
    main()
