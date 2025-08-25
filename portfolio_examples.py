"""
Portfolio Backtesting Examples

This module provides example configurations and usage patterns for the
cryptocurrency portfolio backtesting framework.
"""

from datetime import datetime
from portfolio_backtest import PortfolioBacktester


def example_large_cap_portfolio() -> PortfolioBacktester:
    """
    Large cap cryptocurrency portfolio example.
    
    Returns:
        Configured PortfolioBacktester for large cap tokens
    """
    portfolio = PortfolioBacktester(initial_capital=50000.0)
    
    # Add major cryptocurrencies from the start
    portfolio.add_token('bitcoin')
    portfolio.add_token('ethereum')
    portfolio.add_token('binancecoin')
    portfolio.add_token('cardano')
    portfolio.add_token('solana')
    
    return portfolio


def example_defi_portfolio() -> PortfolioBacktester:
    """
    DeFi-focused cryptocurrency portfolio example.
    
    Returns:
        Configured PortfolioBacktester for DeFi tokens
    """
    portfolio = PortfolioBacktester(initial_capital=25000.0)
    
    # Core DeFi tokens
    portfolio.add_token('uniswap')   # UNI
    portfolio.add_token('aave')      # AAVE
    portfolio.add_token('maker')     # MKR
    portfolio.add_token('chainlink') # LINK
    portfolio.add_token('curve-dao-token')  # CRV
    
    return portfolio


def example_dynamic_addition_portfolio() -> PortfolioBacktester:
    """
    Portfolio with tokens added dynamically during backtest period.
    
    Returns:
        Configured PortfolioBacktester with dynamic token additions
    """
    portfolio = PortfolioBacktester(initial_capital=30000.0)
    
    # Initial tokens (from start of backtest)
    portfolio.add_token('bitcoin')
    portfolio.add_token('ethereum')
    portfolio.add_token('cardano')
    
    # Add tokens at specific dates during backtest
    portfolio.add_token('solana', datetime(2021, 3, 1))        # SOL added March 2021
    portfolio.add_token('avalanche-2', datetime(2021, 9, 1))   # AVAX added September 2021
    portfolio.add_token('terra-luna', datetime(2021, 6, 1))    # LUNA added June 2021
    portfolio.add_token('polygon', datetime(2021, 5, 1))       # MATIC added May 2021
    
    return portfolio


def example_small_portfolio() -> PortfolioBacktester:
    """
    Small 5-token portfolio for testing.
    
    Returns:
        Configured PortfolioBacktester with 5 tokens
    """
    portfolio = PortfolioBacktester(initial_capital=10000.0)
    
    portfolio.add_token('bitcoin')
    portfolio.add_token('ethereum')
    portfolio.add_token('binancecoin')
    portfolio.add_token('cardano')
    portfolio.add_token('polkadot')
    
    return portfolio


def run_custom_backtest():
    """
    Example of running a custom backtest with specific parameters.
    """
    print("=== Custom Portfolio Backtest Example ===")
    
    # Create custom portfolio
    portfolio = PortfolioBacktester(initial_capital=15000.0)
    
    # Add tokens with some dynamic additions
    portfolio.add_token('bitcoin')
    portfolio.add_token('ethereum')
    portfolio.add_token('chainlink')
    portfolio.add_token('polkadot')
    portfolio.add_token('solana', datetime(2021, 6, 1))  # Add SOL mid-2021
    portfolio.add_token('avalanche-2', datetime(2021, 10, 1))  # Add AVAX late 2021
    
    # Custom date range
    start_date = "2021-01-01"
    end_date = "2023-12-31"
    
    try:
        # Fetch data
        print(f"Running backtest from {start_date} to {end_date}")
        if not portfolio.fetch_all_data(start_date, end_date):
            return
        
        # Prepare data
        combined_data = portfolio.prepare_combined_data()
        if combined_data is None:
            print("Failed to prepare combined dataset")
            return
        
        # Run backtest
        results = portfolio.run_backtest(combined_data, start_date, end_date)
        
        # Show results
        portfolio.print_performance_summary(results)
        portfolio.plot_portfolio_performance(results)
        
        print("\n=== Rebalance Dates ===")
        for i, date in enumerate(portfolio.rebalance_dates):
            print(f"{i+1:2d}. {date.strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"Error in custom backtest: {e}")


def compare_portfolios():
    """
    Example of comparing multiple portfolio strategies.
    """
    print("=== Portfolio Strategy Comparison ===")
    
    portfolios = {
        'Large Cap': example_large_cap_portfolio(),
        'DeFi Focus': example_defi_portfolio(), 
        'Small Portfolio': example_small_portfolio()
    }
    
    start_date = "2021-01-01"
    end_date = "2023-06-30"
    
    results_summary = {}
    
    for name, portfolio in portfolios.items():
        print(f"\n--- Running {name} Portfolio ---")
        
        try:
            if portfolio.fetch_all_data(start_date, end_date):
                combined_data = portfolio.prepare_combined_data()
                if combined_data is not None:
                    results = portfolio.run_backtest(combined_data, start_date, end_date)
                    metrics = portfolio.calculate_performance_metrics(results['returns'])
                    
                    results_summary[name] = {
                        'final_value': results['portfolio_value'].iloc[-1],
                        'total_return': metrics.get('total_return', 0),
                        'annual_return': metrics.get('annual_return', 0),
                        'volatility': metrics.get('volatility', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'rebalances': len(portfolio.rebalance_dates)
                    }
        except Exception as e:
            print(f"Error running {name} portfolio: {e}")
    
    # Print comparison table
    if results_summary:
        print(f"\n{'='*80}")
        print("PORTFOLIO COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Portfolio':<15} {'Final Value':<12} {'Total Ret':<10} {'Annual Ret':<11} {'Volatility':<11} {'Sharpe':<8} {'Max DD':<8} {'Rebal':<6}")
        print(f"{'-'*80}")
        
        for name, metrics in results_summary.items():
            print(f"{name:<15} "
                  f"${metrics['final_value']:>10,.0f} "
                  f"{metrics['total_return']*100:>8.1f}% "
                  f"{metrics['annual_return']*100:>9.1f}% "
                  f"{metrics['volatility']*100:>9.1f}% "
                  f"{metrics['sharpe_ratio']:>6.2f} "
                  f"{metrics['max_drawdown']*100:>6.1f}% "
                  f"{metrics['rebalances']:>4d}")


if __name__ == "__main__":
    # Run examples
    print("Choose an example to run:")
    print("1. Custom Portfolio Backtest")
    print("2. Portfolio Strategy Comparison")
    print("3. Large Cap Portfolio")
    print("4. DeFi Portfolio") 
    print("5. Dynamic Addition Portfolio")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        run_custom_backtest()
    elif choice == "2":
        compare_portfolios()
    elif choice == "3":
        portfolio = example_large_cap_portfolio()
        # Run with default parameters
        if portfolio.fetch_all_data("2021-01-01", "2024-01-01"):
            combined_data = portfolio.prepare_combined_data()
            if combined_data is not None:
                results = portfolio.run_backtest(combined_data, "2021-01-01", "2024-01-01")
                portfolio.print_performance_summary(results)
                portfolio.plot_portfolio_performance(results)
    elif choice == "4":
        portfolio = example_defi_portfolio()
        # Run with default parameters  
        if portfolio.fetch_all_data("2021-01-01", "2024-01-01"):
            combined_data = portfolio.prepare_combined_data()
            if combined_data is not None:
                results = portfolio.run_backtest(combined_data, "2021-01-01", "2024-01-01")
                portfolio.print_performance_summary(results)
                portfolio.plot_portfolio_performance(results)
    elif choice == "5":
        portfolio = example_dynamic_addition_portfolio()
        # Run with default parameters
        if portfolio.fetch_all_data("2021-01-01", "2024-01-01"):
            combined_data = portfolio.prepare_combined_data()
            if combined_data is not None:
                results = portfolio.run_backtest(combined_data, "2021-01-01", "2024-01-01")
                portfolio.print_performance_summary(results)
                portfolio.plot_portfolio_performance(results)
    else:
        print("Invalid choice. Running default custom backtest...")
        run_custom_backtest()