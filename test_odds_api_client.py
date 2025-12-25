import pytest
from src.data.odds_api_client import normalize_team, sanitize_markets


def test_normalize_team_variants():
    # Variants with and without 'FC' should normalize to same value
    assert normalize_team('Newcastle United FC') == normalize_team('Newcastle United') == 'newcastle'
    # Wolverhampton has alias 'WOLVES' for the full name
    assert normalize_team('Wolverhampton Wanderers FC') == normalize_team('Wolverhampton Wanderers') == 'wolves'


def test_sanitize_markets_removes_btts():
    assert sanitize_markets('h2h,totals,btts') == 'h2h,totals'
    assert sanitize_markets('btts') == 'h2h'  # fallback when nothing supported
    assert sanitize_markets('h2h') == 'h2h'
