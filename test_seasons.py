import asyncio
import logging
from datetime import datetime
from src.config import get_settings
from src.data.football_data_client import get_fd_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_seasons_retrieval():
    settings = get_settings()
    fd_client = get_fd_client()
    
    # 1. Determine current season based on logic in train_models_background
    current_year = datetime.now().year
    current_month = datetime.now().month
    # Season starts in August (month 8)
    current_season = current_year if current_month >= 8 else current_year - 1
    
    # 2. Seasons to fetch (Current and 2 Previous)
    fd_seasons = [current_season, current_season - 1, current_season - 2]
    
    print(f"\n--- VERIFYING TRAINING DATA RETRIEVAL ---")
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Calculated Current Season: {current_season} ({current_season}-{current_season+1})")
    print(f"Target Seasons for Training: {fd_seasons}")
    
    # Check one league to see if data exists for these seasons
    target_league = settings.league_ids[0] # Usually Premier League (2021/39 for FD.org)
    
    for season in fd_seasons:
        try:
            print(f"\nChecking League {target_league} for Season {season}...")
            # We use a limit to just check connectivity and existence
            matches = await fd_client.get_finished_matches(league_id=target_league, season=season)
            print(f"✅ Success! Found {len(matches)} finished matches for season {season}")
            if matches:
                first = matches[0]
                last = matches[-1]
                print(f"   Sample Match (First): {first.utc_date} - {first.home_team_name} vs {first.away_team_name}")
                print(f"   Sample Match (Last):  {last.utc_date} - {last.home_team_name} vs {last.away_team_name}")
        except Exception as e:
            print(f"❌ Error fetching season {season}: {e}")

if __name__ == "__main__":
    asyncio.run(test_seasons_retrieval())
