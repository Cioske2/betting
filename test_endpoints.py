import httpx
import json
import asyncio

async def test_upcoming_matches():
    url = "http://localhost:8000/api/upcoming-matches"
    params = {"league_id": 135, "days": 3}
    
    print(f"Testing endpoint: {url} with params {params}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=60.0)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("Response JSON:")
                print(json.dumps(data, indent=2))
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_upcoming_matches())
