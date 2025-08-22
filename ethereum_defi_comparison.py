#!/usr/bin/env python3
"""
Ethereum DeFi Analysis Script

This script uses the DeFi Llama API to pull Ethereum data including token price
and compares metrics from today vs November 2021, with visualizations.

Author: Generated for DeFi Analysis
Date: 2025
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import time
import json
from typing import Dict, List, Optional, Tuple

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('default')
sns.set_palette("husl")

class EthereumDeFiAnalyzer:
    """Class to analyze Ethereum DeFi metrics using DeFi Llama API"""
    
    def __init__(self):
        self.base_url = "https://api.llama.fi"
        self.coins_url = "https://coins.llama.fi"
        # Ethereum contract address for price data
        self.eth_address = "ethereum:0x0000000000000000000000000000000000000000"
        
        # Historical data for November 2021 (mid-month as reference)
        self.nov_2021_timestamp = int(datetime(2021, 11, 15).timestamp())
        self.current_timestamp = int(datetime.now().timestamp())
        
    def fetch_current_eth_price(self) -> Optional[float]:
        """Fetch current Ethereum price from DeFi Llama"""
        try:
            url = f"{self.coins_url}/prices/current/ethereum:0x0000000000000000000000000000000000000000"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'coins' in data and self.eth_address in data['coins']:
                return data['coins'][self.eth_address]['price']
            return None
        except Exception as e:
            print(f"Error fetching current ETH price: {e}")
            return None
    
    def fetch_historical_eth_price(self, timestamp: int) -> Optional[float]:
        """Fetch historical Ethereum price for a specific timestamp"""
        try:
            url = f"{self.coins_url}/prices/historical/{timestamp}/ethereum:0x0000000000000000000000000000000000000000"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'coins' in data and self.eth_address in data['coins']:
                return data['coins'][self.eth_address]['price']
            return None
        except Exception as e:
            print(f"Error fetching historical ETH price for timestamp {timestamp}: {e}")
            return None
    
    def fetch_protocol_data(self, protocol_slug: str = "ethereum") -> Optional[Dict]:
        """Fetch protocol data from DeFi Llama"""
        try:
            url = f"{self.base_url}/protocol/{protocol_slug}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching protocol data: {e}")
            return None
    
    def fetch_chains_data(self) -> Optional[List[Dict]]:
        """Fetch chains data to get Ethereum TVL"""
        try:
            url = f"{self.base_url}/chains"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching chains data: {e}")
            return None
    
    def get_ethereum_tvl_historical(self, timestamp: int) -> Optional[float]:
        """Get historical TVL for Ethereum at a specific timestamp"""
        try:
            # Convert timestamp to date string for historical TVL
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            url = f"{self.base_url}/v2/historicalChainTvl/Ethereum"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            # Find the closest date to our target
            target_date = datetime.fromtimestamp(timestamp).date()
            closest_tvl = None
            min_diff = float('inf')
            
            for entry in data:
                entry_date = datetime.fromtimestamp(entry['date']).date()
                diff = abs((entry_date - target_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_tvl = entry['tvl']
            
            return closest_tvl
        except Exception as e:
            print(f"Error fetching historical TVL: {e}")
            return None
    
    def get_current_ethereum_tvl(self) -> Optional[float]:
        """Get current Ethereum TVL"""
        try:
            chains_data = self.fetch_chains_data()
            if chains_data:
                for chain in chains_data:
                    if chain['name'].lower() == 'ethereum':
                        return chain['tvl']
            return None
        except Exception as e:
            print(f"Error getting current TVL: {e}")
            return None
    
    def collect_all_data(self) -> Dict:
        """Collect all required data for comparison"""
        print("Collecting Ethereum data...")
        
        # Current data
        print("Fetching current data...")
        current_price = self.fetch_current_eth_price()
        current_tvl = self.get_current_ethereum_tvl()
        
        # Historical data (November 2021)
        print("Fetching November 2021 data...")
        historical_price = self.fetch_historical_eth_price(self.nov_2021_timestamp)
        historical_tvl = self.get_ethereum_tvl_historical(self.nov_2021_timestamp)
        
        # Fallback historical price if API doesn't have it
        if historical_price is None:
            # Average ETH price in November 2021 was around $4,500
            historical_price = 4500.0
            print("Using fallback historical price: $4,500")
        
        # Fallback historical TVL if API doesn't have it
        if historical_tvl is None:
            # Ethereum TVL in November 2021 was around $180B
            historical_tvl = 180_000_000_000
            print("Using fallback historical TVL: $180B")
        
        return {
            'current': {
                'date': datetime.now(),
                'price': current_price,
                'tvl': current_tvl,
                'timestamp': self.current_timestamp
            },
            'november_2021': {
                'date': datetime.fromtimestamp(self.nov_2021_timestamp),
                'price': historical_price,
                'tvl': historical_tvl,
                'timestamp': self.nov_2021_timestamp
            }
        }
    
    def create_visualizations(self, data: Dict):
        """Create comparison visualizations"""
        # Use Agg backend for headless environment
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ethereum DeFi Metrics: November 2021 vs Today', fontsize=16, fontweight='bold')
        
        # Price comparison
        periods = ['November 2021', 'Today']
        prices = [data['november_2021']['price'], data['current']['price']]
        
        bars1 = ax1.bar(periods, prices, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        ax1.set_title('Ethereum Price Comparison', fontweight='bold')
        ax1.set_ylabel('Price (USD)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add value labels on bars
        for bar, price in zip(bars1, prices):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${price:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # TVL comparison
        tvls = [data['november_2021']['tvl'], data['current']['tvl']]
        tvls_billions = [tvl/1e9 for tvl in tvls]  # Convert to billions
        
        bars2 = ax2.bar(periods, tvls_billions, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        ax2.set_title('Ethereum TVL Comparison', fontweight='bold')
        ax2.set_ylabel('TVL (Billions USD)')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}B'))
        
        # Add value labels on bars
        for bar, tvl in zip(bars2, tvls_billions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${tvl:.0f}B', ha='center', va='bottom', fontweight='bold')
        
        # Price change analysis
        price_change = ((data['current']['price'] - data['november_2021']['price']) / 
                       data['november_2021']['price'] * 100)
        
        colors = ['green' if price_change > 0 else 'red']
        bars3 = ax3.bar(['Price Change'], [price_change], color=colors, alpha=0.8)
        ax3.set_title('Price Change Since November 2021', fontweight='bold')
        ax3.set_ylabel('Percentage Change (%)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add percentage label
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{price_change:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold')
        
        # TVL change analysis
        tvl_change = ((data['current']['tvl'] - data['november_2021']['tvl']) / 
                     data['november_2021']['tvl'] * 100)
        
        colors = ['green' if tvl_change > 0 else 'red']
        bars4 = ax4.bar(['TVL Change'], [tvl_change], color=colors, alpha=0.8)
        ax4.set_title('TVL Change Since November 2021', fontweight='bold')
        ax4.set_ylabel('Percentage Change (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add percentage label
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{tvl_change:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/workspace/ethereum_defi_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved chart: ethereum_defi_comparison.png")
        
        # Create a summary table
        self.create_summary_table(data, price_change, tvl_change)
    
    def create_summary_table(self, data: Dict, price_change: float, tvl_change: float):
        """Create a summary table of the comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [
            ['Metric', 'November 2021', 'Today', 'Change', 'Change (%)'],
            ['ETH Price', f"${data['november_2021']['price']:,.0f}", 
             f"${data['current']['price']:,.0f}",
             f"${data['current']['price'] - data['november_2021']['price']:,.0f}",
             f"{price_change:+.1f}%"],
            ['ETH TVL', f"${data['november_2021']['tvl']/1e9:.0f}B", 
             f"${data['current']['tvl']/1e9:.0f}B",
             f"${(data['current']['tvl'] - data['november_2021']['tvl'])/1e9:.0f}B",
             f"{tvl_change:+.1f}%"]
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, 3):
            for j in range(5):
                if j == 4:  # Change percentage column
                    color = '#e8f5e8' if float(table_data[i][j].replace('%', '').replace('+', '')) > 0 else '#ffe8e8'
                    table[(i, j)].set_facecolor(color)
        
        plt.title('Ethereum DeFi Metrics Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig('/workspace/ethereum_summary_table.png', dpi=300, bbox_inches='tight')
        print("Saved table: ethereum_summary_table.png")
    
    def print_summary(self, data: Dict):
        """Print a text summary of the analysis"""
        print("\n" + "="*60)
        print("ETHEREUM DEFI ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nDATA COMPARISON:")
        print(f"November 2021: {data['november_2021']['date'].strftime('%B %d, %Y')}")
        print(f"Today: {data['current']['date'].strftime('%B %d, %Y')}")
        
        print(f"\nETHEReum PRICE:")
        print(f"November 2021: ${data['november_2021']['price']:,.2f}")
        print(f"Today: ${data['current']['price']:,.2f}")
        price_change = ((data['current']['price'] - data['november_2021']['price']) / 
                       data['november_2021']['price'] * 100)
        print(f"Change: {price_change:+.1f}% (${data['current']['price'] - data['november_2021']['price']:+,.2f})")
        
        print(f"\nETHEREUM TVL (Total Value Locked):")
        print(f"November 2021: ${data['november_2021']['tvl']/1e9:.1f}B")
        print(f"Today: ${data['current']['tvl']/1e9:.1f}B")
        tvl_change = ((data['current']['tvl'] - data['november_2021']['tvl']) / 
                     data['november_2021']['tvl'] * 100)
        print(f"Change: {tvl_change:+.1f}% (${(data['current']['tvl'] - data['november_2021']['tvl'])/1e9:+.1f}B)")
        
        print(f"\nKEY INSIGHTS:")
        if price_change > 0:
            print(f"• Ethereum price has increased by {price_change:.1f}% since November 2021")
        else:
            print(f"• Ethereum price has decreased by {abs(price_change):.1f}% since November 2021")
            
        if tvl_change > 0:
            print(f"• Ethereum DeFi ecosystem has grown by {tvl_change:.1f}% in TVL")
        else:
            print(f"• Ethereum DeFi ecosystem has contracted by {abs(tvl_change):.1f}% in TVL")
        
        print("="*60)

def main():
    """Main function to run the analysis"""
    print("Ethereum DeFi Analysis Tool")
    print("Comparing metrics: November 2021 vs Today")
    print("-" * 50)
    
    analyzer = EthereumDeFiAnalyzer()
    
    try:
        # Collect all data
        data = analyzer.collect_all_data()
        
        # Verify we have the required data
        if not all([
            data['current']['price'], 
            data['current']['tvl'],
            data['november_2021']['price'],
            data['november_2021']['tvl']
        ]):
            print("Error: Could not collect all required data")
            return
        
        # Print summary
        analyzer.print_summary(data)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        analyzer.create_visualizations(data)
        
        print("\nAnalysis complete!")
        print("Generated files:")
        print("- ethereum_defi_comparison.png")
        print("- ethereum_summary_table.png")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        raise

if __name__ == "__main__":
    main()