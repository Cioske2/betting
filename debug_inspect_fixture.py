import asyncio
import logging
from src.api.endpoints import get_fd_client, get_odds_api_client, get_ensemble
from src.data.odds_api_client import normalize_team

logging.basicConfig(level=logging.INFO)

async def inspect_fixture(fixture_id=537957):
    fd = get_fd_client()
    odds = get_odds_api_client()
    ensemble = get_ensemble()

    # Fetch upcoming matches for EPL
    fixtures = await fd.get_upcoming_matches(league_id=39, days_ahead=14)
    target = None
    for f in fixtures:
        if f.match_id == fixture_id:
            target = f
            break

    if not target:
        print("Fixture not found in FD upcoming matches")
        return

    print(f"Found fixture: {target.home_team_name} vs {target.away_team_name} (id={target.match_id})")
    print(f"FD odds: home={target.home_win_odds} draw={target.draw_odds} away={target.away_win_odds}")

    # prediction
    prediction = ensemble.predict(
        home_team_id=target.home_team_id,
        away_team_id=target.away_team_id,
        home_team_name=target.home_team_name,
        away_team_name=target.away_team_name,
        league_id=39
    )
    print(f"Prediction probs: home={prediction.home_win_pct/100:.4f} draw={prediction.draw_pct/100:.4f} away={prediction.away_win_pct/100:.4f}")

    # Poisson
    home_l5 = []
    away_l5 = []
    poisson_pred = ensemble.poisson.predict(target.home_team_id, home_l5, target.away_team_id, away_l5)

    # defaults
    odds_1x2 = {"1": target.home_win_odds or 1.0, "X": target.draw_odds or 1.0, "2": target.away_win_odds or 1.0}
    margin = 0.93
    ou_odds = {"over": round((1.0 / max(0.01, poisson_pred.over_25_prob)) * margin, 2),
               "under": round((1.0 / max(0.01, poisson_pred.under_25_prob)) * margin, 2)}
    btts_odds = {"yes": round((1.0 / max(0.01, poisson_pred.btts_yes_prob)) * margin, 2),
                 "no": round((1.0 / max(0.01, poisson_pred.btts_no_prob)) * margin, 2)}

    # real odds
    all_odds = await odds.get_all_league_odds([39])
    h_norm = normalize_team(target.home_team_name)
    a_norm = normalize_team(target.away_team_name)
    key = f"{h_norm}_vs_{a_norm}"
    print(f"Normalized key: {key}")
    ro = all_odds.get(key)
    print(f"Real odds entry present: {bool(ro)}")
    if ro:
        print(ro)
        if ro.get('1x2'):
            odds_1x2 = ro['1x2']
            print('Overwritten with real 1x2:', odds_1x2)
        else:
            print('No 1x2 in real odds for this match')

    print('Final assembled odds_1x2:', odds_1x2)
    print('ou_odds:', ou_odds)
    print('btts_odds:', btts_odds)

if __name__ == '__main__':
    asyncio.run(inspect_fixture())