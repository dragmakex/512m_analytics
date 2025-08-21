import requests
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import time

# DeFiLlama API endpoints
CHART_ENDPOINT = "https://yields.llama.fi/chart/"

# Theme palette (from spr_plotter.py)
theme_palette = ['#f7f3ec', '#ede4da', '#b9a58f', '#574c40', '#36312a']

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

def rolling_beta(apy_series, market_series, window=30, min_periods=15):
    """
    Calculate rolling beta between two series using returns
    
    Args:
        apy_series: Series with APY data
        market_series: Series with market data
        window: Rolling window size
        min_periods: Minimum periods required for calculation
        
    Returns:
        pd.Series: Rolling beta values
    """
    beta_series = pd.Series(index=apy_series.index, dtype=float)
    
    for i in range(min_periods-1, len(apy_series)):
        start_idx = max(0, i - window + 1)
        apy_window = apy_series.iloc[start_idx:i+1]
        market_window = market_series.iloc[start_idx:i+1]
        
        if len(apy_window) >= min_periods:
            # Calculate returns (first difference)
            apy_returns = apy_window.diff().dropna()
            market_returns = market_window.diff().dropna()
            
            if len(apy_returns) > 1 and len(market_returns) > 1:
                # Align the series
                common_idx = apy_returns.index.intersection(market_returns.index)
                if len(common_idx) > 1:
                    apy_aligned = apy_returns.loc[common_idx]
                    market_aligned = market_returns.loc[common_idx]
                    
                    # Calculate beta
                    covariance = np.cov(apy_aligned, market_aligned)[0, 1]
                    market_variance = np.var(market_aligned)
                    
                    if market_variance > 0:
                        beta_series.iloc[i] = covariance / market_variance
    
    return beta_series

def fetch_pool_chart_data(pool_id, pool_name, days=360):
    """
    Fetch historical chart data for a specific pool from DeFiLlama
    
    Args:
        pool_id (str): Pool ID from DeFiLlama
        pool_name (str): Pool name for logging
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: DataFrame with historical APY data, or None if failed
    """
    try:
        print(f"Fetching data for pool {pool_id} ({pool_name})...")
        url = f"{CHART_ENDPOINT}{pool_id}"
        response = requests.get(url)
        
        # Handle rate limiting
        if response.status_code == 429:
            print(f"Rate limited, waiting 2 seconds...")
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
                    
                    # Keep only APY column
                    if 'apy' in df.columns:
                        df = df[['apy']].copy()
                        df.columns = ['pool_apy']
                        print(f"Successfully fetched {len(df)} data points for pool {pool_name}")
                        return df
                    else:
                        print(f"No APY column found for pool {pool_name}")
                        return None
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

def load_weighted_apy_from_db(db_filename="defi_prime_rate.db"):
    """
    Load weighted APY data from SQLite database
    
    Args:
        db_filename (str): SQLite database filename
        
    Returns:
        pd.DataFrame: DataFrame with weighted APY data, or None if failed
    """
    try:
        print(f"Loading weighted APY data from {db_filename}...")
        conn = sqlite3.connect(db_filename)
        
        # Load main data
        merged_df = pd.read_sql('SELECT date, weighted_apy FROM pool_data', conn)
        
        # Set date as index
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'])
            merged_df.set_index('date', inplace=True)
        
        conn.close()
        
        print(f"Successfully loaded weighted APY data")
        print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"Total data points: {len(merged_df)}")
        print(f"Date index type: {type(merged_df.index)}")
        print(f"Sample dates: {merged_df.index[:5].tolist()}")
        
        return merged_df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def prepare_analysis_data(pool_df, weighted_df):
    """
    Prepare and calculate all analysis data for plotting
    
    Args:
        pool_df (pd.DataFrame): DataFrame with pool APY data
        weighted_df (pd.DataFrame): DataFrame with weighted APY data
        
    Returns:
        dict: Dictionary containing all calculated data and statistics
    """
    print(f"Preparing analysis data...")
    
    # Ensure both dataframes have datetime index
    if not isinstance(pool_df.index, pd.DatetimeIndex):
        pool_df.index = pd.to_datetime(pool_df.index)
    if not isinstance(weighted_df.index, pd.DatetimeIndex):
        weighted_df.index = pd.to_datetime(weighted_df.index)
    
    # Normalize dates to remove time component for better matching
    pool_df_normalized = pool_df.copy()
    weighted_df_normalized = weighted_df.copy()
    
    pool_df_normalized.index = pool_df_normalized.index.normalize()
    weighted_df_normalized.index = weighted_df_normalized.index.normalize()
    
    # Merge the dataframes on date
    merged_df = pd.merge(pool_df_normalized, weighted_df_normalized, left_index=True, right_index=True, how='inner')
    
    if merged_df.empty:
        print("No overlapping data found between pool and weighted APY")
        return None
    
    print(f"Found {len(merged_df)} overlapping data points")
    
    # Calculate rolling correlation and beta (30-day window)
    merged_df['rolling_corr_30d'] = merged_df['pool_apy'].rolling(window=30, min_periods=15).corr(merged_df['weighted_apy'])
    merged_df['rolling_beta_30d'] = rolling_beta(merged_df['pool_apy'], merged_df['weighted_apy'], window=30, min_periods=15)
    
    # Calculate overall correlation
    correlation = merged_df['pool_apy'].corr(merged_df['weighted_apy'])
    
    # Prepare scatter plot data
    x = merged_df['weighted_apy'].values
    y = merged_df['pool_apy'].values
    
    # Remove NaN values for trend line calculation
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Calculate trend line and R-squared
    trend_data = {}
    if len(x_clean) > 1:
        # Calculate linear regression
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        
        # Calculate R-squared
        y_pred = p(x_clean)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        trend_data = {
            'coefficients': z,
            'polynomial': p,
            'r_squared': r_squared,
            'x_clean': x_clean,
            'y_clean': y_clean
        }
    
    # Calculate summary statistics
    stats = {
        'data_points': len(merged_df),
        'overall_correlation': correlation,
        'current_pool_apy': merged_df['pool_apy'].iloc[-1],
        'mean_pool_apy': merged_df['pool_apy'].mean(),
        'current_weighted_apy': merged_df['weighted_apy'].iloc[-1],
        'mean_weighted_apy': merged_df['weighted_apy'].mean(),
        'current_rolling_corr': merged_df['rolling_corr_30d'].iloc[-1] if 'rolling_corr_30d' in merged_df.columns else None,
        'mean_rolling_corr': merged_df['rolling_corr_30d'].mean() if 'rolling_corr_30d' in merged_df.columns else None,
        'current_rolling_beta': merged_df['rolling_beta_30d'].iloc[-1] if 'rolling_beta_30d' in merged_df.columns else None,
        'mean_rolling_beta': merged_df['rolling_beta_30d'].mean() if 'rolling_beta_30d' in merged_df.columns else None
    }
    
    return {
        'merged_df': merged_df,
        'trend_data': trend_data,
        'stats': stats
    }

def plot_pool_vs_weighted_apy(pool_df, weighted_df, pool_name):
    """
    Create a comprehensive analysis plot comparing pool APY vs weighted APY
    
    Args:
        pool_df (pd.DataFrame): DataFrame with pool APY data
        weighted_df (pd.DataFrame): DataFrame with weighted APY data
        pool_name (str): Name of the pool for display
    """
    
    # Prepare all analysis data
    analysis_data = prepare_analysis_data(pool_df, weighted_df)
    
    if analysis_data is None:
        print("Failed to prepare analysis data. Exiting.")
        return
    
    merged_df = analysis_data['merged_df']
    trend_data = analysis_data['trend_data']
    stats = analysis_data['stats']    
    
    # Create comprehensive multi-panel plot
    fig = plt.figure(figsize=(16, 8))
    
    # Create grid layout: 2 rows, 2 columns with better spacing
    gs = fig.add_gridspec(2, 2)
    
    # Panel 1: Time series comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        merged_df.index,
        merged_df['pool_apy'],
        label='yoUSD APY',
        alpha=0.8,
        linewidth=1.5,
        color=theme_palette[2]
    )
    ax1.plot(
        merged_df.index,
        merged_df['weighted_apy'],
        label='SPR APY',
        linewidth=2.5,
        color=theme_palette[3]
    )
    ax1.set_title('yoUSD APY vs Stablecoin Prime Rate Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('APY (%)')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 2: Rolling correlation
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        merged_df.index,
        merged_df['rolling_corr_30d'],
        linewidth=2,
        color=theme_palette[4]
    )
    ax2.set_title('30-Day Rolling Correlation')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Correlation')
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 3: Rolling beta
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(
        merged_df.index,
        merged_df['rolling_beta_30d'],
        linewidth=2,
        color=theme_palette[4]
    )
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('30-Day Rolling Beta')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Beta')
    ax3.legend()
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 4: Scatter plot with trend line (top right)
    ax4 = fig.add_subplot(gs[0, 1])
    
    # Scatter plot
    ax4.scatter(
        merged_df['weighted_apy'],
        merged_df['pool_apy'],
        alpha=0.6,
        s=20,
        color=theme_palette[2]
    )
    
    # Add trend line if available
    if trend_data:
        z = trend_data['coefficients']
        p = trend_data['polynomial']
        x_clean = trend_data['x_clean']
        
        # Plot trend line
        x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax4.plot(x_trend, p(x_trend), 
                color=theme_palette[3], 
                linewidth=2, 
                linestyle='--',
                label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
        

    
    ax4.set_xlabel('DeFi Prime Rate (%)')
    ax4.set_ylabel('yoUSD APY (%)')
    ax4.set_title('Daily Values Scatter Plot')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Set better axis ranges for scatter plot to fit the data
    x_min, x_max = merged_df['weighted_apy'].min(), merged_df['weighted_apy'].max()
    y_min, y_max = merged_df['pool_apy'].min(), merged_df['pool_apy'].max()
    
    # Add some padding to the ranges
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax4.set_xlim(x_min - x_padding, x_max + x_padding)
    ax4.set_ylim(y_min - y_padding, y_max + y_padding)
    

    
    # Add 512m logo overlay to each subplot
    try:
        from PIL import Image
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        
        # Load the logo image
        logo_img = Image.open("512m_logo.png")
        logo_array = np.array(logo_img)
        
        # List of all subplots
        subplots = [ax1, ax2, ax3, ax4]
        
        for ax in subplots:
            # Get the center of each subplot in data coordinates
            x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
            y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
            
            # Use a consistent logo size based on the figure size rather than subplot data range
            # This ensures all logos appear the same size across subplots
            fig_width_inches = 16  # from figsize
            logo_size_inches = 0.75  # consistent logo size in inches
            logo_zoom = logo_size_inches / (fig_width_inches / 2)  # divide by 2 for 2x2 grid
            
            # Create offset image with very low alpha
            im = OffsetImage(logo_array, zoom=logo_zoom, alpha=0.04)
            
            # Create annotation box at center of each subplot
            ab = AnnotationBbox(im, (x_center, y_center), frameon=False)
            ax.add_artist(ab)
        
    except Exception as e:
        print(f"Warning: Could not add logo overlay: {e}")
    
    plt.tight_layout()
    
    # Try to show the plot, and also save it
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
        print("Plot will be saved as file instead.")
    
    # Save the plot
    plot_filename = f"yousd_vs_defi_prime_rate_analysis.pdf"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor=theme_palette[0])
    print(f"Comprehensive analysis plot saved as: {plot_filename}")
    
    plt.close(fig)  # Close the main analysis figure
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Pool: {pool_name}")
    print(f"Data points: {stats['data_points']}")
    print(f"Overall correlation with DeFi Prime Rate: {stats['overall_correlation']:.3f}")
    
    # Rolling statistics
    if stats['current_rolling_corr'] is not None:
        print(f"Current 30-day correlation: {stats['current_rolling_corr']:.3f}")
        print(f"Mean 30-day correlation: {stats['mean_rolling_corr']:.3f}")
    
    if stats['current_rolling_beta'] is not None:
        print(f"Current 30-day beta: {stats['current_rolling_beta']:.3f}")
        print(f"Mean 30-day beta: {stats['mean_rolling_beta']:.3f}")
    
    print(f"yoUSD APY - Current: {stats['current_pool_apy']:.4f}%, Mean: {stats['mean_pool_apy']:.4f}%")
    print(f"DeFi Prime Rate - Current: {stats['current_weighted_apy']:.4f}%, Mean: {stats['mean_weighted_apy']:.4f}%")
    
    if trend_data:
        z = trend_data['coefficients']
        r_squared = trend_data['r_squared']
        print(f"Trend line: y = {z[0]:.3f}x + {z[1]:.3f}")
        print(f"R-squared: {r_squared:.3f}")

def main():
    """
    Main function to fetch pool data and create comparison plot
    """
    # Pool ID and name
    pool_id = "1994cc35-a2b9-434e-b197-df6742fb5d81"
    pool_name = "yoUSD"  # You can update this with the actual pool name if known
    
    print(f"=== yoUSD vs DeFi Prime Rate Analysis ===")
    print(f"Pool ID: {pool_id}")
    print(f"Pool Name: {pool_name}")
    
    # Fetch pool data from DeFiLlama
    print("\n=== Fetching yoUSD Data from DeFiLlama ===")
    pool_df = fetch_pool_chart_data(pool_id, pool_name, days=360)
    
    if pool_df is None or pool_df.empty:
        print("Failed to fetch yoUSD data. Exiting.")
        return
    
    # Load weighted APY data from database
    print("\n=== Loading DeFi Prime Rate Data from Database ===")
    weighted_df = load_weighted_apy_from_db("defi_prime_rate.db")
    
    if weighted_df is None or weighted_df.empty:
        print("Failed to load DeFi Prime Rate data. Exiting.")
        return
    
    # Create the comparison plot
    print("\n=== Creating Comprehensive Analysis ===")
    plot_pool_vs_weighted_apy(pool_df, weighted_df, pool_name)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
