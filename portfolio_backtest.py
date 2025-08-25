"""
Crypto Portfolio Backtesting Module

This module performs backtesting of an equal-weighted cryptocurrency portfolio
using CoinGecko API data with quarterly rebalancing and threshold-based rebalancing
when allocations exceed 40% or drop below 1%.
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dotenv import load_dotenv
import time

from config import setup_plotting_style, THEME_PALETTE, MUTED_BLUES
from utils import (
    safe_api_request, validate_dataframe, print_data_summary,
    normalize_datetime_index, add_logo_overlay
)

# Load environment variables
load_dotenv()

# CoinGecko API configuration
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
RATE_LIMIT_DELAY = 1.2  # CoinGecko free tier allows ~50 calls/min


class PortfolioBacktester:
    """
    Cryptocurrency portfolio backtesting engine with flexible rebalancing logic.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the portfolio backtester.
        
        Args:
            initial_capital: Starting portfolio value in USD
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.price_data = {}
        self.portfolio_history = []
        self.rebalance_dates = []
        self.tokens = []
        self.token_addition_dates = {}
        
    def add_token(self, token_id: str, start_date: datetime = None) -> None:
        """
        Add a token to the portfolio.
        
        Args:
            token_id: CoinGecko token ID (e.g., 'bitcoin', 'ethereum')
            start_date: Date when token should be added to portfolio
        """
        if token_id not in self.tokens:
            self.tokens.append(token_id)
            if start_date:
                self.token_addition_dates[token_id] = start_date
            print(f"Added {token_id} to portfolio")
    
    def fetch_token_data(self, token_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a token from CoinGecko API.
        
        Args:
            token_id: CoinGecko token ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with price data or None if failed
        """
        url = f"{COINGECKO_BASE_URL}/coins/{token_id}/market_chart/range"
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        params = {
            'vs_currency': 'usd',
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        try:
            print(f"Fetching data for {token_id}...")
            response = safe_api_request(url, max_retries=3)
            
            if not response or response.status_code != 200:
                print(f"Failed to fetch data for {token_id}")
                return None
            
            data = response.json()
            
            if 'prices' not in data or not data['prices']:
                print(f"No price data available for {token_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            df = df[['price']].rename(columns={'price': token_id})
            
            # Normalize to daily data (take last price per day)
            df = df.groupby(df.index.date).last()
            df.index = pd.to_datetime(df.index)
            
            print(f"Successfully fetched {len(df)} data points for {token_id}")
            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {token_id}: {e}")
            return None
    
    def fetch_all_data(self, start_date: str, end_date: str) -> bool:
        """
        Fetch historical data for all tokens in the portfolio.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n=== Fetching Portfolio Data ===")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Tokens: {self.tokens}")
        
        for token_id in self.tokens:
            token_data = self.fetch_token_data(token_id, start_date, end_date)
            if token_data is not None:
                self.price_data[token_id] = token_data
            else:
                print(f"Warning: Failed to fetch data for {token_id}")
        
        if not self.price_data:
            print("No price data fetched. Cannot proceed with backtest.")
            return False
        
        print(f"Successfully fetched data for {len(self.price_data)} tokens")
        return True
    
    def prepare_combined_data(self) -> Optional[pd.DataFrame]:
        """
        Combine all token price data into a single DataFrame.
        
        Returns:
            Combined DataFrame with all token prices
        """
        if not self.price_data:
            return None
        
        # Find common date range
        start_dates = [df.index.min() for df in self.price_data.values()]
        end_dates = [df.index.max() for df in self.price_data.values()]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        print(f"\n=== Preparing Combined Dataset ===")
        print(f"Common date range: {common_start} to {common_end}")
        
        # Create combined DataFrame
        combined_df = pd.DataFrame()
        
        for token_id, token_df in self.price_data.items():
            # Filter to common date range
            filtered_df = token_df[common_start:common_end]
            combined_df[token_id] = filtered_df.iloc[:, 0]  # First column is price
        
        # Forward fill missing values
        combined_df = combined_df.ffill().dropna()
        
        print_data_summary(combined_df, "Combined Portfolio Data")
        return combined_df
    
    def get_active_tokens(self, date: datetime) -> List[str]:
        """
        Get list of tokens that should be active in portfolio on given date.
        
        Args:
            date: Date to check
            
        Returns:
            List of active token IDs
        """
        active_tokens = []
        
        for token_id in self.tokens:
            # Check if token has an addition date
            if token_id in self.token_addition_dates:
                if date >= self.token_addition_dates[token_id]:
                    active_tokens.append(token_id)
            else:
                # Token is active from the beginning
                active_tokens.append(token_id)
        
        return active_tokens
    
    def calculate_equal_weights(self, active_tokens: List[str]) -> Dict[str, float]:
        """
        Calculate equal weights for active tokens.
        
        Args:
            active_tokens: List of active token IDs
            
        Returns:
            Dictionary mapping token IDs to target weights
        """
        if not active_tokens:
            return {}
        
        equal_weight = 1.0 / len(active_tokens)
        return {token: equal_weight for token in active_tokens}
    
    def needs_rebalancing(self, current_weights: Dict[str, float], 
                         target_weights: Dict[str, float]) -> bool:
        """
        Check if portfolio needs rebalancing based on threshold rules.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            True if rebalancing is needed, False otherwise
        """
        for token, weight in current_weights.items():
            if weight > 0.40:  # Over 40%
                return True
            if token in target_weights and weight < 0.01:  # Under 1% for active tokens
                return True
        
        return False
    
    def run_backtest(self, price_data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run the portfolio backtest with quarterly and threshold-based rebalancing.
        
        Args:
            price_data: Combined price data for all tokens
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            DataFrame with portfolio performance data
        """
        print(f"\n=== Running Portfolio Backtest ===")
        
        # Initialize portfolio
        dates = price_data.index
        portfolio_values = []
        portfolio_weights = []
        cash_balance = self.initial_capital
        token_holdings = {token: 0.0 for token in self.tokens}
        
        # Quarterly rebalance dates
        quarterly_dates = pd.date_range(
            start=dates[0], end=dates[-1], freq='Q'
        ).intersection(dates)
        
        last_rebalance_date = None
        
        for i, date in enumerate(dates):
            current_prices = price_data.loc[date]
            active_tokens = self.get_active_tokens(date)
            
            # Calculate current portfolio value and weights
            token_values = {}
            total_token_value = 0.0
            
            for token in active_tokens:
                if token in token_holdings and token in current_prices:
                    token_value = token_holdings[token] * current_prices[token]
                    token_values[token] = token_value
                    total_token_value += token_value
            
            total_portfolio_value = total_token_value + cash_balance
            
            # Calculate current weights
            current_weights = {}
            for token in active_tokens:
                if total_portfolio_value > 0:
                    current_weights[token] = token_values.get(token, 0) / total_portfolio_value
                else:
                    current_weights[token] = 0.0
            
            # Determine if rebalancing is needed
            target_weights = self.calculate_equal_weights(active_tokens)
            is_quarterly = date in quarterly_dates
            needs_threshold_rebalance = self.needs_rebalancing(current_weights, target_weights)
            
            should_rebalance = (
                (is_quarterly or needs_threshold_rebalance) and 
                (last_rebalance_date is None or date != last_rebalance_date)
            )
            
            # Rebalance if needed
            if should_rebalance and active_tokens:
                print(f"Rebalancing on {date.strftime('%Y-%m-%d')} "
                      f"({'Quarterly' if is_quarterly else 'Threshold'})")
                
                # Sell all current holdings
                for token in token_holdings:
                    if token_holdings[token] > 0 and token in current_prices:
                        cash_balance += token_holdings[token] * current_prices[token]
                        token_holdings[token] = 0.0
                
                # Buy according to target weights
                for token, target_weight in target_weights.items():
                    if token in current_prices and current_prices[token] > 0:
                        target_value = total_portfolio_value * target_weight
                        shares_to_buy = target_value / current_prices[token]
                        token_holdings[token] = shares_to_buy
                        cash_balance -= target_value
                
                self.rebalance_dates.append(date)
                last_rebalance_date = date
            
            # Store portfolio state
            portfolio_values.append(total_portfolio_value)
            portfolio_weights.append(current_weights.copy())
        
        # Create results DataFrame
        results_df = pd.DataFrame(index=dates)
        results_df['portfolio_value'] = portfolio_values
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        
        # Add individual token weights
        for token in self.tokens:
            weight_series = []
            for weights in portfolio_weights:
                weight_series.append(weights.get(token, 0.0))
            results_df[f'{token}_weight'] = weight_series
        
        print(f"Backtest completed. Final portfolio value: ${portfolio_values[-1]:,.2f}")
        print(f"Total return: {((portfolio_values[-1] / self.initial_capital) - 1) * 100:.2f}%")
        print(f"Number of rebalances: {len(self.rebalance_dates)}")
        
        return results_df
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            Dictionary of performance metrics
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {}
        
        total_return = (1 + returns_clean).prod() - 1
        annual_return = (1 + total_return) ** (365.25 / len(returns_clean)) - 1
        volatility = returns_clean.std() * np.sqrt(365.25)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns_clean).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def plot_portfolio_performance(self, results_df: pd.DataFrame) -> None:
        """
        Create comprehensive portfolio performance visualizations.
        
        Args:
            results_df: Results DataFrame from backtest
        """
        setup_plotting_style()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Portfolio value over time
        ax1.plot(results_df.index, results_df['portfolio_value'], 
                 linewidth=2, color=MUTED_BLUES[2])
        ax1.axhline(y=self.initial_capital, color=THEME_PALETTE[3], 
                   linestyle='--', alpha=0.7, label=f'Initial: ${self.initial_capital:,.0f}')
        
        # Mark rebalance dates
        for rebalance_date in self.rebalance_dates:
            if rebalance_date in results_df.index:
                portfolio_val = results_df.loc[rebalance_date, 'portfolio_value']
                ax1.axvline(x=rebalance_date, color=THEME_PALETTE[2], 
                           alpha=0.3, linestyle='-', linewidth=1)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        add_logo_overlay(ax1)
        
        # Cumulative returns
        cumulative_returns = (1 + results_df['returns']).cumprod() - 1
        ax2.plot(results_df.index, cumulative_returns * 100, 
                 linewidth=2, color=MUTED_BLUES[1])
        ax2.axhline(y=0, color=THEME_PALETTE[3], linestyle='--', alpha=0.7)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.grid(True, alpha=0.3)
        add_logo_overlay(ax2)
        
        # Portfolio allocation over time (stacked area)
        weight_cols = [col for col in results_df.columns if col.endswith('_weight')]
        if weight_cols:
            weight_data = results_df[weight_cols] * 100  # Convert to percentage
            weight_data.columns = [col.replace('_weight', '') for col in weight_data.columns]
            
            ax3.stackplot(results_df.index, *[weight_data[col] for col in weight_data.columns],
                         labels=weight_data.columns, alpha=0.7)
            ax3.set_title('Portfolio Allocation Over Time')
            ax3.set_ylabel('Allocation (%)')
            ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            ax3.grid(True, alpha=0.3)
            add_logo_overlay(ax3)
        
        # Rolling 30-day volatility
        rolling_vol = results_df['returns'].rolling(30).std() * np.sqrt(365.25) * 100
        ax4.plot(results_df.index, rolling_vol, linewidth=2, color=MUTED_BLUES[3])
        ax4.set_title('30-Day Rolling Volatility')
        ax4.set_ylabel('Annualized Volatility (%)')
        ax4.grid(True, alpha=0.3)
        add_logo_overlay(ax4)
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_summary(self, results_df: pd.DataFrame) -> None:
        """
        Print comprehensive performance summary.
        
        Args:
            results_df: Results DataFrame from backtest
        """
        metrics = self.calculate_performance_metrics(results_df['returns'])
        
        print(f"\n=== Portfolio Performance Summary ===")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${results_df['portfolio_value'].iloc[-1]:,.2f}")
        print(f"Total Return: {metrics.get('total_return', 0) * 100:.2f}%")
        print(f"Annualized Return: {metrics.get('annual_return', 0) * 100:.2f}%")
        print(f"Annualized Volatility: {metrics.get('volatility', 0) * 100:.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Number of Rebalances: {len(self.rebalance_dates)}")
        
        print(f"\n=== Token Allocation Summary (Final) ===")
        weight_cols = [col for col in results_df.columns if col.endswith('_weight')]
        final_weights = results_df[weight_cols].iloc[-1]
        for col in weight_cols:
            token_name = col.replace('_weight', '')
            weight = final_weights[col]
            print(f"{token_name}: {weight * 100:.1f}%")


def create_sample_portfolio() -> PortfolioBacktester:
    """
    Create a sample portfolio with major cryptocurrencies.
    
    Returns:
        Configured PortfolioBacktester instance
    """
    backtest = PortfolioBacktester(initial_capital=10000.0)
    
    # Add major cryptocurrencies
    backtest.add_token('bitcoin')
    backtest.add_token('ethereum')
    backtest.add_token('binancecoin')
    backtest.add_token('cardano')
    backtest.add_token('solana')
    backtest.add_token('polkadot')
    backtest.add_token('chainlink')
    
    # Example of adding tokens during backtest period
    # backtest.add_token('avalanche-2', datetime(2021, 9, 1))  # Add AVAX from Sept 2021
    
    return backtest


def main():
    """
    Main function to run the portfolio backtest.
    """
    print("=== Cryptocurrency Portfolio Backtesting ===")
    
    # Create sample portfolio
    portfolio = create_sample_portfolio()
    
    # Define backtest period
    start_date = "2021-01-01"
    end_date = "2024-01-01"
    
    try:
        # Fetch data
        if not portfolio.fetch_all_data(start_date, end_date):
            return
        
        # Prepare combined dataset
        combined_data = portfolio.prepare_combined_data()
        if combined_data is None:
            print("Failed to prepare combined dataset")
            return
        
        # Run backtest
        results = portfolio.run_backtest(combined_data, start_date, end_date)
        
        # Display results
        portfolio.print_performance_summary(results)
        portfolio.plot_portfolio_performance(results)
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        raise


if __name__ == "__main__":
    main()