"""
Test script to verify CoinGecko Pro API key is working.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_pro_api_key():
    """Test if the Pro API key is working."""
    print("=== Testing CoinGecko Pro API Key ===")
    
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        print("❌ No COINGECKO_API_KEY found in .env file")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...")
    
    # Test the Pro API endpoint
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    headers = {
        'x-cg-pro-api-key': api_key,
        'accept': 'application/json'
    }
    
    params = {
        'vs_currency': 'usd',
        'ids': 'bitcoin',
        'order': 'market_cap_desc',
        'per_page': 1,
        'page': 1,
        'sparkline': False
    }
    
    try:
        print("Making request to Pro API...")
        print(f"URL: {url}")
        print(f"Headers: {headers}")
        print(f"Params: {params}")
        
        response = requests.get(url, headers=headers, params=params)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Pro API request successful!")
            print(f"Response data: {data[:2] if isinstance(data, list) else data}")
            return True
        else:
            print(f"❌ Pro API request failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Pro API: {e}")
        return False

def test_free_api():
    """Test the free API for comparison."""
    print("\n=== Testing CoinGecko Free API ===")
    
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'ids': 'bitcoin',
        'order': 'market_cap_desc',
        'per_page': 1,
        'page': 1,
        'sparkline': False
    }
    
    try:
        print("Making request to free API...")
        response = requests.get(url, params=params)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Free API request successful!")
            print(f"Response data: {data[:2] if isinstance(data, list) else data}")
            return True
        else:
            print(f"❌ Free API request failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing free API: {e}")
        return False

if __name__ == "__main__":
    print("Testing CoinGecko API endpoints...")
    print()
    
    # Test Pro API
    pro_success = test_pro_api_key()
    
    # Test free API for comparison
    free_success = test_free_api()
    
    print("\n=== Test Summary ===")
    print(f"Pro API: {'✅ Working' if pro_success else '❌ Failed'}")
    print(f"Free API: {'✅ Working' if free_success else '❌ Failed'}")
    
    if pro_success:
        print("\n🎉 Your Pro API key is working correctly!")
        print("You should now be able to use the portfolio backtest without rate limiting.")
    else:
        print("\n⚠️  Pro API key test failed. Check your API key and try again.")
