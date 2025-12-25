import asyncio
import httpx
from src.config import get_settings

async def check_pl_2025():
    settings = get_settings()
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": settings.football_data_key}
    params = {"season": 2025, "status": "FINISHED"}
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            matches = data.get("matches", [])
            print(f"Found {len(matches)} finished matches for 2025.")
            if matches:
                print(f"Latest match: {matches[0]['utcDate']} - {matches[0]['homeTeam']['name']} vs {matches[0]['awayTeam']['name']}")
        else:
            print(f"Error: {resp.text}")

if __name__ == "__main__":
    asyncio.run(check_pl_2025())
