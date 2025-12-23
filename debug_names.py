import asyncio
import logging
from src.data.odds_api_client import get_odds_api_client, normalize_team

# Configure logging
logging.basicConfig(level=logging.ERROR) # Only show errors, we will print output manually

async def check_names():
    client = get_odds_api_client()
    
    leagues_to_check = [
        ("soccer_spain_la_liga", "La Liga"),
        ("soccer_germany_bundesliga", "Bundesliga"),
        ("soccer_france_ligue_1", "Ligue 1")
    ]
    
    print("\n--- CHECKING TEAM NAMES ---")
    
    for key, name in leagues_to_check:
        print(f"\nFETCHING: {name} ({key})")
        try:
            # We removed btts from default, so this should work now
            matches = await client.get_odds(key, markets="h2h") 
            
            seen_teams = set()
            for m in matches:
                h = m['home_team']
                a = m['away_team']
                seen_teams.add(h)
                seen_teams.add(a)
            
            print(f"Found {len(seen_teams)} unique teams.")
            for t in sorted(seen_teams):
                norm = normalize_team(t)
                print(f"  RAW: '{t}'  ->  NORM: '{norm}'")
                
        except Exception as e:
            print(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    asyncio.run(check_names())
