"""
Test script to verify CoinGecko API improvements.
"""

import time
from portfolio_backtest import PortfolioBacktester

def test_single_token():
    """Test fetching data for a single token with improved rate limiting."""
    print("=== Testing Single Token Fetch ===")
    
    portfolio = PortfolioBacktester(initial_capital=10000.0)
    portfolio.add_token('bitcoin')
    
    # Test with a shorter date range to reduce API calls
    start_date = "2024-01-01"
    end_date = "2024-01-31"  # Just one month
    
    success = portfolio.fetch_all_data(start_date, end_date)
    
    if success and portfolio.price_data:
        print("✅ Successfully fetched data!")
        for token, data in portfolio.price_data.items():
            print(f"  {token}: {len(data)} data points")
    else:
        print("❌ Failed to fetch data")

def test_multiple_tokens():
    """Test fetching data for multiple tokens with improved rate limiting."""
    print("\n=== Testing Multiple Tokens Fetch ===")
    
    portfolio = PortfolioBacktester(initial_capital=10000.0)
    
    # Add a few tokens
    portfolio.add_token('bitcoin')
    portfolio.add_token('ethereum')
    portfolio.add_token('binancecoin')
    
    # Test with a shorter date range
    start_date = "2024-01-01"
    end_date = "2024-01-15"  # Just two weeks
    
    success = portfolio.fetch_all_data(start_date, end_date)
    
    if success and portfolio.price_data:
        print("✅ Successfully fetched data for all tokens!")
        for token, data in portfolio.price_data.items():
            print(f"  {token}: {len(data)} data points")
    else:
        print("❌ Failed to fetch data for some tokens")
        print(f"Successfully fetched: {len(portfolio.price_data)}/{len(portfolio.tokens)} tokens")

if __name__ == "__main__":
    print("Testing CoinGecko API improvements...")
    
    # Check if API key is available
    import os
    api_key = os.getenv('COINGECKO_API_KEY')
    if api_key:
        print("✅ CoinGecko Pro API key detected - using Pro tier")
        print("This will provide much better rate limits and reliability.")
    else:
        print("⚠️  No CoinGecko API key found - using free tier")
        print("This will test the improved rate limiting and retry logic.")
    
    print()
    
    test_single_token()
    time.sleep(5)  # Wait between tests
    test_multiple_tokens()
    
    print("\n=== Test Complete ===")
    if api_key:
        print("✅ Pro API tests completed - should be much faster and more reliable!")
    else:
        print("✅ Free tier tests completed - improved rate limiting should work better.")
