import requests
import json
from datetime import datetime

def fetch_all_pools():
    """
    Fetch all pools from DeFiLlama yields API and find the specific pool
    
    Returns:
        list: List of all pools or None if failed
    """
    url = "https://yields.llama.fi/pools"
    
    try:
        print("Fetching all pools from DeFiLlama...")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            print(f"Error: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching all pools: {e}")
        return None

def find_pool_by_id(pools, target_pool_id):
    """
    Find a specific pool by its ID in the list of all pools
    
    Args:
        pools (list): List of all pools
        target_pool_id (str): The pool ID to find
        
    Returns:
        dict: Pool data if found, None otherwise
    """
    for pool in pools:
        if pool.get('pool') == target_pool_id:
            return pool
    return None

def fetch_pool_metadata(pool_id):
    """
    Fetch complete metadata for a specific pool from DeFiLlama yields API
    
    Args:
        pool_id (str): The pool ID to fetch metadata for
        
    Returns:
        dict: Complete pool metadata or None if failed
    """
    # First try to get all pools and find the specific one
    all_pools = fetch_all_pools()
    
    if all_pools:
        print(f"Found {len(all_pools)} total pools")
        target_pool = find_pool_by_id(all_pools, pool_id)
        
        if target_pool:
            print(f"Found target pool: {target_pool.get('name', 'Unknown')}")
            return {'data': target_pool}
        else:
            print(f"Pool ID {pool_id} not found in the pools list")
            return None
    else:
        print("Failed to fetch pools list")
        return None

def fetch_pool_chart_data(pool_id):
    """
    Fetch historical chart data for a specific pool
    
    Args:
        pool_id (str): The pool ID to fetch chart data for
        
    Returns:
        dict: Chart data or None if failed
    """
    url = f"https://yields.llama.fi/chart/{pool_id}"
    
    try:
        print(f"\nFetching chart data for pool ID: {pool_id}")
        print(f"URL: {url}")
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error fetching chart data: {e}")
        return None

def display_pool_metadata(metadata):
    """
    Display the pool metadata in a readable format
    
    Args:
        metadata (dict): Pool metadata from API
    """
    if not metadata:
        print("No metadata to display")
        return
    
    print("\n" + "="*80)
    print("POOL METADATA")
    print("="*80)
    
    # Basic pool information
    if 'data' in metadata:
        pool = metadata['data']
        
        print(f"Pool ID: {pool.get('pool', 'N/A')}")
        print(f"Name: {pool.get('name', 'N/A')}")
        print(f"Symbol: {pool.get('symbol', 'N/A')}")
        print(f"Project: {pool.get('project', 'N/A')}")
        print(f"Chain: {pool.get('chain', 'N/A')}")
        print(f"Category: {pool.get('category', 'N/A')}")
        
        print(f"\nToken Information:")
        print(f"  Token: {pool.get('token', 'N/A')}")
        print(f"  Token Address: {pool.get('tokenAddress', 'N/A')}")
        print(f"  Token Symbol: {pool.get('tokenSymbol', 'N/A')}")
        print(f"  Token Decimals: {pool.get('tokenDecimals', 'N/A')}")
        
        print(f"\nPool Characteristics:")
        print(f"  Stablecoin: {pool.get('stablecoin', 'N/A')}")
        print(f"  Lending Protocol: {pool.get('lendingProtocol', 'N/A')}")
        print(f"  Pool Meta: {pool.get('poolMeta', 'N/A')}")
        print(f"  Underlying Tokens: {pool.get('underlyingTokens', 'N/A')}")
        
        print(f"\nCurrent Metrics:")
        print(f"  TVL USD: ${pool.get('tvlUsd', 0):,.0f}")
        print(f"  APY: {pool.get('apy', 0) or 'N/A'}")
        print(f"  Base APY: {pool.get('apyBase', 'N/A')}")
        print(f"  Reward APY: {pool.get('apyReward', 'N/A')}")
        print(f"  APY Mean 30d: {pool.get('apyMean30d', 'N/A')}")
        print(f"  APY Pct 1M: {pool.get('apyPct1M', 'N/A')}")
        print(f"  APY Pct 30D: {pool.get('apyPct30D', 'N/A')}")
        
        print(f"\nRisk Metrics:")
        print(f"  Risk Score: {pool.get('riskScore', 'N/A')}")
        print(f"  Risk Level: {pool.get('riskLevel', 'N/A')}")
        
        print(f"\nAdditional Information:")
        print(f"  Pool URL: {pool.get('pool', 'N/A')}")
        print(f"  Metadata Updated: {pool.get('metadataUpdated', 'N/A')}")
        print(f"  Pool ID: {pool.get('pool', 'N/A')}")
        
        # Show all available fields
        print(f"\nAll Available Fields:")
        for key, value in pool.items():
            if key not in ['data']:  # Skip nested data
                print(f"  {key}: {value}")
    
    else:
        print("No 'data' field found in response")
        print("Raw response:")
        print(json.dumps(metadata, indent=2))

def display_chart_data(chart_data):
    """
    Display chart data summary
    
    Args:
        chart_data (dict): Chart data from API
    """
    if not chart_data:
        print("No chart data to display")
        return
    
    print("\n" + "="*80)
    print("CHART DATA SUMMARY")
    print("="*80)
    
    if 'data' in chart_data:
        data_points = chart_data['data']
        print(f"Total data points: {len(data_points)}")
        
        if data_points:
            # Show first and last data points
            print(f"\nFirst data point:")
            first = data_points[0]
            timestamp = first.get('timestamp', 0)
            if isinstance(timestamp, str):
                print(f"  Date: {timestamp}")
            else:
                print(f"  Date: {datetime.fromtimestamp(timestamp)}")
            print(f"  TVL: ${first.get('tvlUsd', 0):,.0f}")
            print(f"  APY: {first.get('apy', 0) or 'N/A'}")
            
            print(f"\nLast data point:")
            last = data_points[-1]
            timestamp = last.get('timestamp', 0)
            if isinstance(timestamp, str):
                print(f"  Date: {timestamp}")
            else:
                print(f"  Date: {datetime.fromtimestamp(timestamp)}")
            print(f"  TVL: ${last.get('tvlUsd', 0):,.0f}")
            print(f"  APY: {last.get('apy', 0) or 'N/A'}")
            
            # Show data structure
            print(f"\nData structure (first point):")
            for key, value in first.items():
                print(f"  {key}: {value}")
    
    else:
        print("No 'data' field found in chart response")
        print("Raw chart response:")
        print(json.dumps(chart_data, indent=2))

def save_metadata_to_file(metadata, chart_data, pool_id):
    """
    Save the metadata and chart data to JSON files
    
    Args:
        metadata (dict): Pool metadata
        chart_data (dict): Chart data
        pool_id (str): Pool ID for filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metadata
    metadata_filename = f"pool_{pool_id}_metadata_{timestamp}.json"
    try:
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved to: {metadata_filename}")
    except Exception as e:
        print(f"Error saving metadata: {e}")
    
    # Save chart data
    chart_filename = f"pool_{pool_id}_chart_{timestamp}.json"
    try:
        with open(chart_filename, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"Chart data saved to: {chart_filename}")
    except Exception as e:
        print(f"Error saving chart data: {e}")

def main():
    """
    Main function to fetch and display pool metadata
    """
    # The specific pool IDs you requested
    pool_ids = [
        "a87bbade-7728-43d9-bc23-2538812be3cc",
        "f981a304-bb6c-45b8-b0c5-fd2f515ad23a"
    ]
    
    print("DeFiLlama Pool Metadata Fetcher - Side by Side Comparison")
    print("="*70)
    
    all_metadata = {}
    all_chart_data = {}
    
    # Fetch metadata for all pools
    for pool_id in pool_ids:
        print(f"\n{'='*40}")
        print(f"FETCHING POOL: {pool_id}")
        print(f"{'='*40}")
        
        # Fetch pool metadata
        metadata = fetch_pool_metadata(pool_id)
        
        if metadata:
            all_metadata[pool_id] = metadata
            
            # Fetch chart data
            chart_data = fetch_pool_chart_data(pool_id)
            if chart_data:
                all_chart_data[pool_id] = chart_data
            else:
                print(f"Failed to fetch chart data for {pool_id}")
        else:
            print(f"Failed to fetch metadata for {pool_id}")
    
    # Display side by side comparison
    if len(all_metadata) > 1:
        print("\n" + "="*120)
        print("SIDE BY SIDE COMPARISON")
        print("="*120)
        
        # Get all unique fields from both pools
        all_fields = set()
        for metadata in all_metadata.values():
            if 'data' in metadata:
                all_fields.update(metadata['data'].keys())
        
        # Display comparison table
        print(f"{'Field':<25} {'Pool 1':<30} {'Pool 2':<30}")
        print("-" * 85)
        
        for field in sorted(all_fields):
            pool1_value = all_metadata[pool_ids[0]]['data'].get(field, 'N/A')
            pool2_value = all_metadata[pool_ids[1]]['data'].get(field, 'N/A')
            
            # Format values for display
            if isinstance(pool1_value, (int, float)) and field == 'tvlUsd':
                pool1_display = f"${pool1_value:,.0f}"
            elif isinstance(pool1_value, (int, float)) and 'apy' in field.lower():
                pool1_display = f"{pool1_value:.4f}%" if pool1_value else 'N/A'
            else:
                pool1_display = str(pool1_value)[:25] if pool1_value else 'N/A'
            
            if isinstance(pool2_value, (int, float)) and field == 'tvlUsd':
                pool2_display = f"${pool2_value:,.0f}"
            elif isinstance(pool2_value, (int, float)) and 'apy' in field.lower():
                pool2_display = f"{pool2_value:.4f}%" if pool2_value else 'N/A'
            else:
                pool2_display = str(pool2_value)[:25] if pool2_value else 'N/A'
            
            print(f"{field:<25} {pool1_display:<30} {pool2_display:<30}")
    
    # Save all data to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save combined metadata
    combined_filename = f"pool_comparison_metadata_{timestamp}.json"
    try:
        with open(combined_filename, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"\nCombined metadata saved to: {combined_filename}")
    except Exception as e:
        print(f"Error saving combined metadata: {e}")
    
    # Save combined chart data
    if all_chart_data:
        combined_chart_filename = f"pool_comparison_chart_{timestamp}.json"
        try:
            with open(combined_chart_filename, 'w') as f:
                json.dump(all_chart_data, f, indent=2)
            print(f"Combined chart data saved to: {combined_chart_filename}")
        except Exception as e:
            print(f"Error saving combined chart data: {e}")
    
    print("\n" + "="*120)
    print("Script completed")

if __name__ == "__main__":
    main()
