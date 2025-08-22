# Ethereum DeFi Analysis Script

This script uses the DeFi Llama API to compare Ethereum metrics between November 2021 and today, providing insights into price changes and Total Value Locked (TVL) growth.

## Features

- **Real-time Data Fetching**: Pulls current Ethereum price and TVL from DeFi Llama API
- **Historical Analysis**: Compares current metrics with November 2021 data
- **Comprehensive Visualizations**: Generates charts showing price and TVL comparisons
- **Summary Tables**: Creates formatted tables with detailed metrics
- **Detailed Analytics**: Provides percentage changes and key insights

## Requirements

The script requires the following Python packages:
- `requests` - for API calls
- `pandas` - for data manipulation
- `matplotlib` - for plotting
- `seaborn` - for enhanced visualizations
- `numpy` - for numerical operations

Install dependencies:
```bash
# For Ubuntu/Debian systems:
sudo apt update
sudo apt install -y python3-requests python3-pandas python3-matplotlib python3-numpy python3-seaborn python3-pil

# Or using pip (if available):
pip install requests pandas matplotlib seaborn numpy pillow
```

## Usage

Run the script directly:
```bash
python3 ethereum_defi_comparison.py
```

## Output

The script generates:

1. **Console Output**: Detailed text summary with key metrics and insights
2. **ethereum_defi_comparison.png**: Multi-panel chart showing:
   - Price comparison (November 2021 vs Today)
   - TVL comparison (November 2021 vs Today)
   - Price percentage change
   - TVL percentage change
3. **ethereum_summary_table.png**: Formatted table with all metrics

## Sample Output

```
ETHEREUM DEFI ANALYSIS SUMMARY
============================================================

DATA COMPARISON:
November 2021: November 15, 2021
Today: August 22, 2025

ETHEReum PRICE:
November 2021: $4,652.95
Today: $4,800.49
Change: +3.2% ($+147.54)

ETHEREUM TVL (Total Value Locked):
November 2021: $106.5B
Today: $213.5B
Change: +100.4% ($+107.0B)

KEY INSIGHTS:
• Ethereum price has increased by 3.2% since November 2021
• Ethereum DeFi ecosystem has grown by 100.4% in TVL
```

## API Endpoints Used

- **Current Price**: `https://coins.llama.fi/prices/current/ethereum:0x0000000000000000000000000000000000000000`
- **Historical Price**: `https://coins.llama.fi/prices/historical/{timestamp}/ethereum:0x0000000000000000000000000000000000000000`
- **Current TVL**: `https://api.llama.fi/chains`
- **Historical TVL**: `https://api.llama.fi/v2/historicalChainTvl/Ethereum`

## Script Architecture

The `EthereumDeFiAnalyzer` class contains methods for:
- `fetch_current_eth_price()`: Gets current ETH price
- `fetch_historical_eth_price()`: Gets historical ETH price for specific timestamp
- `get_current_ethereum_tvl()`: Gets current Ethereum TVL
- `get_ethereum_tvl_historical()`: Gets historical TVL data
- `create_visualizations()`: Generates comparison charts
- `create_summary_table()`: Creates formatted summary table
- `print_summary()`: Outputs detailed text analysis

## Customization

You can modify the script to:
- Change the historical reference date (currently November 15, 2021)
- Add more metrics (trading volume, number of protocols, etc.)
- Modify visualization styles and colors
- Export data to CSV or other formats
- Compare different time periods

## Error Handling

The script includes robust error handling:
- Fallback values for missing historical data
- Timeout protection for API calls
- Graceful handling of network errors
- Alternative data sources when primary APIs fail

## Notes

- The script uses fallback historical values if DeFi Llama API doesn't have November 2021 data
- All visualizations are saved as high-resolution PNG files
- The script works in headless environments (no display required)
- Network connectivity is required for API calls

## Author

Generated for DeFi Analysis - 2025