import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Theme palette (from provided image)
# Order: background (1st), 2nd, 3rd, 4th, 5th
theme_palette = ['#f7f3ec', '#ede4da', '#b9a58f', '#574c40', '#36312a']
muted_blues = [
    '#2b3e50', '#3c5a77', '#4f7192', '#5f86a8', '#6f9bbd',
    '#86abc7', '#9bbad1', '#afc8da', '#c3d5e3', '#d7e2ec'
]

# Set academic-style plotting with serif fonts and beige background (1st palette color)
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
    # Background from palette (first color)
    'figure.facecolor': theme_palette[0],
    'axes.facecolor': theme_palette[0],
    'savefig.facecolor': theme_palette[0]
})

# Keep a general purpose color list for multi-series plots (fallback)
colors = theme_palette

def add_logo_overlay(ax, logo_path="512m_logo.png", alpha=0.1):
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
        
        # Calculate appropriate size for the logo (about 40% of plot width - half the previous size)
        plot_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        logo_width = plot_width * 0.4
        
        # Create offset image
        im = OffsetImage(logo_array, zoom=logo_width/logo_img.width, alpha=alpha)
        
        # Create annotation box at center
        ab = AnnotationBbox(im, (x_center, y_center), frameon=False)
        ax.add_artist(ab)
        
    except Exception as e:
        print(f"Warning: Could not add logo overlay: {e}")

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
        
        # Load main data
        merged_df = pd.read_sql('SELECT * FROM pool_data', conn)
        
        # Set date as index
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'])
            merged_df.set_index('date', inplace=True)
        
        # Load metadata
        metadata_df = pd.read_sql('SELECT * FROM pool_metadata', conn)
        
        conn.close()
        
        print(f"Successfully loaded data for {len(metadata_df)} pools")
        print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"Total data points: {len(merged_df)}")
        
        return merged_df, metadata_df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None

def plot_weighted_apy_trends(df):
    """
    Create a line plot showing daily and 14-day moving average weighted APY
    
    Args:
        df (pd.DataFrame): DataFrame with weighted_apy and ma_apy_14d columns
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot daily weighted APY
    ax.plot(
        df.index,
        df['weighted_apy'],
        label='Daily Weighted APY',
        alpha=0.6,
        linewidth=1.5,
        color=theme_palette[2]  # 3rd color in palette
    )
    
    # Plot 14-day moving average
    ax.plot(
        df.index,
        df['ma_apy_14d'],
        label='14-Day Moving Average',
        linewidth=2.5,
        color=theme_palette[3]  # 4th color in palette
    )
    
    # Customize the plot
    ax.set_title('Stablecoin Prime Rate: Daily vs 14-Day Moving Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('SPR APY (%)')
    ax.legend()
    ax.grid(True)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # No annotations - let the data speak for itself
    
    # Add logo overlay
    add_logo_overlay(ax)
    
    plt.tight_layout()
    plt.show()

def plot_pool_contributions(df, metadata_df):
    """
    Create a figure showing each pool's contribution to the weighted APY
    
    Args:
        df (pd.DataFrame): DataFrame with pool data
        metadata_df (pd.DataFrame): DataFrame with pool metadata
    """
    # Get the most recent data point
    latest_data = df.iloc[-1]
    
    # Extract APY and TVL columns
    apy_cols = [col for col in df.columns if col.startswith('apy_')]
    tvl_cols = [col for col in df.columns if col.startswith('tvlUsd_')]
    
    # Calculate contributions for the latest data point
    contributions = []
    pool_names = []
    
    for i, apy_col in enumerate(apy_cols):
        if i < len(tvl_cols):
            tvl_col = tvl_cols[i]
            apy_val = latest_data[apy_col]
            tvl_val = latest_data[tvl_col]
            
            if pd.notna(apy_val) and pd.notna(tvl_val) and tvl_val > 0:
                # Calculate contribution as (APY * TVL) / total_weighted_sum
                contribution = (apy_val * tvl_val) / (latest_data['weighted_apy'] * df[tvl_cols].sum(axis=1).iloc[-1])
                contributions.append(contribution * 100)  # Convert to percentage
                
                # Get pool name from metadata
                pool_num = apy_col.replace('apy_Pool_', '')
                pool_name = metadata_df[metadata_df['name'] == f'Pool_{pool_num}']['name'].iloc[0] if len(metadata_df[metadata_df['name'] == f'Pool_{pool_num}']) > 0 else f'Pool_{pool_num}'
                pool_names.append(pool_name)
    
    # Sort by contribution (descending)
    sorted_data = sorted(zip(contributions, pool_names), reverse=True)
    contributions, pool_names = zip(*sorted_data)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Top 15 pools by contribution
    top_n = min(15, len(contributions))
    # Use muted blue hues for bars
    bar_colors = [muted_blues[i % len(muted_blues)] for i in range(top_n)]
    
    bars1 = ax1.bar(range(top_n), contributions[:top_n], color=bar_colors, alpha=0.8)
    ax1.set_title(f'Top {top_n} Pools by Contribution to Weighted APY')
    ax1.set_xlabel('Pool')
    ax1.set_ylabel('Contribution (%)')
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels([name.replace('Pool_', 'P') for name in pool_names[:top_n]], 
                        rotation=45, ha='right')
    
    # No value labels - clean academic style
    
    # Plot 2: Cumulative contribution
    cumulative = np.cumsum(contributions)
    ax2.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        marker='o',
        linewidth=2,
        markersize=4,
        color=muted_blues[2]
    )
    ax2.set_title('Cumulative Contribution to Weighted APY')
    ax2.set_xlabel('Number of Pools')
    ax2.set_ylabel('Cumulative Contribution (%)')
    ax2.grid(True)
    
    # No annotations - clean academic style
    
    # Add logo overlay to both subplots
    add_logo_overlay(ax1)
    add_logo_overlay(ax2)
    
    plt.tight_layout()
    plt.show()

def plot_pool_contributions_over_time(df, metadata_df, top_n=7):
    """
    Create a single stacked 100% area chart showing the top 7 pools individually and the rest
    
    Args:
        df (pd.DataFrame): DataFrame with pool data
        metadata_df (pd.DataFrame): DataFrame with pool metadata
        top_n (int): Number of top pools to show individually (default 7)
    """
    # Extract APY and TVL columns
    apy_cols = [col for col in df.columns if col.startswith('apy_')]
    tvl_cols = [col for col in df.columns if col.startswith('tvlUsd_')]
    
    # Calculate contributions over time for each pool
    pool_contributions = {}
    pool_names = {}
    
    for i, apy_col in enumerate(apy_cols):
        if i < len(tvl_cols):
            tvl_col = tvl_cols[i]
            
            # Get pool number
            pool_num = apy_col.replace('apy_Pool_', '')
            pool_name = f'Pool_{pool_num}'
            pool_names[pool_num] = pool_name
            
            # Calculate contributions for each time point
            contributions = []
            for idx in range(len(df)):
                apy_val = df.iloc[idx][apy_col]
                tvl_val = df.iloc[idx][tvl_col]
                weighted_apy = df.iloc[idx]['weighted_apy']
                total_tvl = df.iloc[idx][tvl_cols].sum()
                
                if pd.notna(apy_val) and pd.notna(tvl_val) and tvl_val > 0 and total_tvl > 0:
                    contribution = (apy_val * tvl_val) / (weighted_apy * total_tvl) * 100
                else:
                    contribution = 0
                contributions.append(contribution)
            
            pool_contributions[pool_num] = contributions
    
    # Find top pools by contribution at the LATEST DATE (not average over entire period)
    latest_idx = len(df) - 1
    latest_contributions = {}
    for pool_num, contributions in pool_contributions.items():
        if latest_idx < len(contributions):
            latest_contributions[pool_num] = contributions[latest_idx]
    
    # Sort by latest contribution and get top N
    top_pools = sorted(latest_contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create the stacked area chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for stacked area chart
    stack_data = {}
    
    # Add top N pools individually (reverse order so top contributors appear at top of stack)
    for i, (pool_num, latest_contrib) in enumerate(reversed(top_pools)):
        contributions = pool_contributions[pool_num]
        
        # Use custom names for specific pools
        if pool_num == '0':
            display_name = 'Ethena sUSDe'
        elif pool_num == '1':
            display_name = 'Maple USDC'
        elif pool_num == '2':
            display_name = 'Sky sUSDS'
        elif pool_num == '3':
            display_name = 'AAVE USDT'
        elif pool_num == '4':
            display_name = 'Morpho Spark USDC'
        elif pool_num == '5':
            display_name = 'Sky DSR DAI'
        elif pool_num == '6':
            display_name = 'Usual USD0++'
        elif pool_num == '13':
            display_name = 'Fluid USDC'
        else:
            display_name = f'Pool_{pool_num}'
        
        stack_data[f'{display_name} ({latest_contrib:.1f}%)'] = contributions
    
    # Calculate "Other pools" contribution
    other_contributions = np.zeros(len(df))
    for idx in range(len(df)):
        total_contribution = 0
        for pool_num, _ in top_pools:
            contributions = pool_contributions[pool_num]
            if idx < len(contributions):
                total_contribution += contributions[idx]
        other_contributions[idx] = 100 - total_contribution
    
    stack_data['Other Pools'] = other_contributions
    
    # Create stacked area plot
    # Use blue hues for all named pools and specific brown for "Other Pools"
    stack_colors = []
    for i in range(len(stack_data) - 1):  # All except "Other Pools"
        # Use different blue hues for each named pool
        if i == 0:
            stack_colors.append(muted_blues[0])    # Darkest blue (#2b3e50)
        elif i == 1:
            stack_colors.append(muted_blues[1])    # Dark blue (#3c5a77)
        elif i == 2:
            stack_colors.append(muted_blues[2])    # Medium-dark blue (#4f7192)
        elif i == 3:
            stack_colors.append(muted_blues[3])    # Medium blue (#5f86a8)
        elif i == 4:
            stack_colors.append(muted_blues[4])    # Medium-light blue (#6f9bbd)
        elif i == 5:
            stack_colors.append(muted_blues[5])    # Light blue (#86abc7)
        elif i == 6:
            stack_colors.append(muted_blues[6])    # Lighter blue (#9bbad1)
        elif i == 7:
            stack_colors.append(muted_blues[7])    # Even lighter blue (#afc8da)
        else:
            stack_colors.append(muted_blues[i % len(muted_blues)])
    
    # Add specific brown color for "Other Pools"
    stack_colors.append(theme_palette[2])  # Medium brown (#b9a58f)
    
    ax.stackplot(
        df.index,
        stack_data.values(),
        labels=stack_data.keys(),
        colors=stack_colors,
        alpha=0.9
    )
    
    ax.set_title('Pool Contributions to Stablecoin Prime Rate Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution (%)')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
    ax.grid(True)
    ax.set_ylim(0, 100)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # No annotations - clean academic style
    
    # Add logo overlay
    add_logo_overlay(ax)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load data and create plots
    """
    # Load data from database
    df, metadata_df = load_data_from_db("defi_prime_rate.db")
    
    if df is None or metadata_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create the plots
    print("\nCreating plots...")
    
    # Plot 1: Weighted APY trends
    print("Creating weighted APY trends plot...")
    plot_weighted_apy_trends(df)
    
    # Plot 2: Pool contributions
    print("Creating pool contributions plot...")
    plot_pool_contributions(df, metadata_df)
    
    # Plot 3: Pool contributions over time
    print("Creating pool contributions over time plot...")
    plot_pool_contributions_over_time(df, metadata_df, top_n=7)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    current_apy = df['weighted_apy'].iloc[-1]
    mean_apy = df['weighted_apy'].mean()
    std_apy = df['weighted_apy'].std()
    min_apy = df['weighted_apy'].min()
    max_apy = df['weighted_apy'].max()
    
    print(f"Current DeFi Prime Rate: {current_apy:.4f}%")
    print(f"Mean DeFi Prime Rate: {mean_apy:.4f}%")
    print(f"Standard Deviation: {std_apy:.4f}%")
    print(f"Range: {min_apy:.4f}% - {max_apy:.4f}%")
    print(f"Total pools in dataset: {len(metadata_df)}")
    
    print("\nPlots created successfully!")

if __name__ == "__main__":
    main()
