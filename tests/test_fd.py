"""Test football-data.org API."""
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.data.football_data_client import get_fd_client

async def test_fd():
    client = get_fd_client()
    
    print("Testing football-data.org API...")
    print("=" * 50)
    
    # Test Serie A 2024
    print("\nFetching Serie A 2024 matches...")
    matches = await client.get_finished_matches(league_id=135, season=2024)
    print(f"Got {len(matches)} matches!")
    
    if matches:
        print(f"\nSample: {matches[0].home_team_name} vs {matches[0].away_team_name}")
        print(f"Score: {matches[0].home_score}-{matches[0].away_score}")
    
    # Test Premier League 2024
    print("\nFetching Premier League 2024 matches...")
    matches_pl = await client.get_finished_matches(league_id=39, season=2024)
    print(f"Got {len(matches_pl)} matches!")
    
    return len(matches) + len(matches_pl)

if __name__ == "__main__":
    total = asyncio.run(test_fd())
    print(f"\n✅ Total matches: {total}")
