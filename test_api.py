"""Test training with Serie A only."""
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.data.api_football_client import get_client

async def test_training():
    client = get_client()
    
    print("Testing API connection...")
    print("Fetching Serie A matches for 2024 season...")
    
    try:
        matches = await client.get_finished_matches(
            league_id=135,  # Serie A
            season=2024
        )
        
        print(f"✅ Got {len(matches)} finished matches!")
        
        if matches:
            print(f"\nSample match: {matches[0].home_team_name} vs {matches[0].away_team_name}")
            print(f"Score: {matches[0].home_goals}-{matches[0].away_goals}")
            print(f"Date: {matches[0].date}")
        
        return len(matches)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    result = asyncio.run(test_training())
    print(f"\nTotal matches: {result}")
