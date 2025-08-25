# Cryptocurrency Portfolio Backtesting

This module provides a flexible framework for backtesting equal-weighted cryptocurrency portfolios using CoinGecko API data.

## Key Features

- **Equal Weight Rebalancing**: Automatically maintains equal weights across selected tokens
- **Quarterly Rebalancing**: Rebalances portfolio every quarter
- **Threshold-Based Rebalancing**: Triggers rebalancing when any token exceeds 40% or drops below 1% allocation  
- **Dynamic Token Addition**: Support for adding tokens during the backtest period
- **Comprehensive Analytics**: Performance metrics, visualizations, and detailed reporting

## Quick Start

### Basic Usage

```python
from portfolio_backtest import PortfolioBacktester

# Create backtester with $10,000 initial capital
portfolio = PortfolioBacktester(initial_capital=10000.0)

# Add tokens to portfolio
portfolio.add_token('bitcoin')
portfolio.add_token('ethereum') 
portfolio.add_token('cardano')
portfolio.add_token('solana')
portfolio.add_token('polkadot')

# Fetch data and run backtest
start_date = "2021-01-01"
end_date = "2024-01-01"

if portfolio.fetch_all_data(start_date, end_date):
    combined_data = portfolio.prepare_combined_data()
    results = portfolio.run_backtest(combined_data, start_date, end_date)
    
    # Display results
    portfolio.print_performance_summary(results)
    portfolio.plot_portfolio_performance(results)
```

### Adding Tokens During Backtest

```python
from datetime import datetime

# Add tokens at specific dates
portfolio.add_token('solana', datetime(2021, 6, 1))        # Add SOL in June 2021
portfolio.add_token('avalanche-2', datetime(2021, 9, 1))   # Add AVAX in Sept 2021
```

### Pre-built Portfolio Examples

```python
from portfolio_examples import example_defi_portfolio, example_large_cap_portfolio

# Use pre-configured DeFi portfolio
defi_portfolio = example_defi_portfolio()

# Use pre-configured large cap portfolio  
large_cap_portfolio = example_large_cap_portfolio()
```

## Running Examples

Execute the examples script for interactive demonstrations:

```bash
python portfolio_examples.py
```

This provides options to:
1. Run custom portfolio backtest
2. Compare multiple portfolio strategies
3. Test pre-built portfolio configurations

## Token IDs

Use CoinGecko token IDs (found on CoinGecko URLs). Common examples:

- `bitcoin` - Bitcoin (BTC)
- `ethereum` - Ethereum (ETH)
- `binancecoin` - Binance Coin (BNB)  
- `cardano` - Cardano (ADA)
- `solana` - Solana (SOL)
- `polkadot` - Polkadot (DOT)
- `chainlink` - Chainlink (LINK)
- `avalanche-2` - Avalanche (AVAX)
- `polygon` - Polygon (MATIC)
- `uniswap` - Uniswap (UNI)

## Rebalancing Logic

### Quarterly Rebalancing
- Automatically triggered at the end of each quarter
- Rebalances to equal weights among active tokens

### Threshold-Based Rebalancing  
- Triggered when any token allocation exceeds 40%
- Triggered when any active token drops below 1%  
- Helps maintain balanced portfolio and limits concentration risk

## Performance Metrics

The backtester calculates comprehensive performance statistics:

- **Total Return**: Overall portfolio return
- **Annualized Return**: Compound annual growth rate
- **Volatility**: Annualized standard deviation of daily returns
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline

## Visualizations

Generates four-panel performance dashboard:
1. Portfolio value over time with rebalancing markers
2. Cumulative returns progression  
3. Portfolio allocation over time (stacked area)
4. 30-day rolling volatility

## Rate Limits

CoinGecko free tier allows ~50 calls/minute. The script includes automatic rate limiting (1.2 second delays) to respect these limits.

## Dependencies

- pandas
- numpy
- matplotlib
- requests
- python-dotenv

Leverages existing project utilities from `utils.py` and `config.py` for consistent styling and helper functions.