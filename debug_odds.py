import asyncio
import logging
from src.data.odds_api_client import get_odds_api_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_odds")

async def test_odds_fetching():
    client = get_odds_api_client()
    
    # Test 1: EPL with all markets
    print("\n--- Test 1: EPL with btts ---")
    try:
        data = await client.get_odds("soccer_epl", markets="h2h,totals,btts")
        if data: print(f"Success! Found {len(data)} matches.")
    except Exception as e:
        print(f"Failed: {e}")

    # Test 2: Serie A WITHOUT btts
    print("\n--- Test 2: Serie A WITHOUT btts ---")
    try:
        data = await client.get_odds("soccer_italy_serie_a", markets="h2h,totals")
        if data: print(f"Success! Found {len(data)} matches.")
    except Exception as e:
        print(f"Failed: {e}")

    # Test 3: Serie A WITH btts (to confirm failure)
    print("\n--- Test 3: Serie A WITH btts ---")
    try:
        data = await client.get_odds("soccer_italy_serie_a", markets="h2h,totals,btts")
        if data: print(f"Success! Found {len(data)} matches.")
    except Exception as e:
        print(f"Failed (Expected): {e}")

if __name__ == "__main__":
    asyncio.run(test_odds_fetching())
