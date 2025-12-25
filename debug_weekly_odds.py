import asyncio
import logging
from src.data.odds_api_client import get_odds_api_client, OddsApiClient, normalize_team

logging.basicConfig(level=logging.INFO)

async def run_check():
    client = get_odds_api_client()
    summary = {}

    for lid, sport_key in OddsApiClient.SPORT_MAP.items():
        print(f"\n--- League {lid} ({sport_key}) ---")
        try:
            data = await client.get_odds(sport_key, markets="h2h,totals")
            print(f"API returned {len(data)} matches for {sport_key}")

            all_odds = await client.get_all_league_odds([lid])
            # Count how many have each market
            total_norm = len(all_odds)
            one_x2 = sum(1 for v in all_odds.values() if v.get('1x2'))
            ou = sum(1 for v in all_odds.values() if v.get('ou_2.5') and v['ou_2.5'])
            btts = sum(1 for v in all_odds.values() if v.get('btts') and v['btts'])

            print(f"Normalized matches: {total_norm}; with 1x2: {one_x2}; ou_2.5: {ou}; btts: {btts}")

            # Show missing 1x2 sample
            missing = [k for k, v in all_odds.items() if not v.get('1x2')]
            if missing:
                print(f"Sample matches missing 1x2 (up to 5):")
                # Try to find corresponding raw match to inspect bookmakers
                raw_matches = { (m['home_team'].lower(), m['away_team'].lower()): m for m in data }
                for k in missing[:5]:
                    print(f" - {k}")
                    # attempt to split normalized key to raw names
                    parts = k.split('_vs_')
                    if len(parts) == 2:
                        h_raw = parts[0]
                        a_raw = parts[1]
                        # try to find in raw_matches by scanning
                        found = False
                        for rm in data:
                            if rm.get('home_team') and rm.get('away_team'):
                                if normalize_team(rm['home_team']) == h_raw and normalize_team(rm['away_team']) == a_raw:
                                    print(f"   Raw: {rm['home_team']} vs {rm['away_team']}; bookmakers={len(rm.get('bookmakers', []))}")
                                    for bm in rm.get('bookmakers', [])[:3]:
                                        markets = [mk['key'] for mk in bm.get('markets', [])]
                                        print(f"    - {bm.get('title')} markets: {markets}")
                                    found = True
                                    break
                        if not found:
                            print("   Raw match not found in API response (possible naming mismatch)")
            else:
                print("All normalized matches have 1x2")

            summary[sport_key] = {
                'api_count': len(data),
                'normalized_count': total_norm,
                '1x2': one_x2,
                'ou_2.5': ou,
                'btts': btts
            }

        except Exception as e:
            print(f"Error for {sport_key}: {e}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: api={v['api_count']} normalized={v['normalized_count']} 1x2={v['1x2']} ou_2.5={v['ou_2.5']} btts={v['btts']}")

if __name__ == '__main__':
    asyncio.run(run_check())
