import httpx
import logging
from typing import Dict, List, Optional, Any
from ..config import get_settings

logger = logging.getLogger(__name__)

def normalize_team(name: str) -> str:
    """Standardize team names for cross-API matching."""
    # 1. Basic cleanup
    name = name.upper().strip()
    
    # 2. Known aliases map (FD.org -> The Odds API / Generic simplifications)
    aliases = {
        "PARMA CALCIO 1913": "PARMA",
        "ACF FIORENTINA": "FIORENTINA",
        "US LECCE": "LECCE",
        "SS LAZIO": "LAZIO",
        "FC INTERNAZIONALE MILANO": "INTER MILAN",
        "INTERNAZIONALE": "INTER MILAN",
        "AC PISA 1909": "PISA",
        "US CREMONESE": "CREMONESE",
        "BOLOGNA FC 1909": "BOLOGNA",
        "US SASSUOLO CALCIO": "SASSUOLO",
        "ATALANTA BC": "ATALANTA",
        "SSC NAPOLI": "NAPOLI",
        "TORINO FC": "TORINO",
        "HELLAS VERONA FC": "HELLAS VERONA",
        "UDINESE CALCIO": "UDINESE",
        "CAGLIARI CALCIO": "CAGLIARI",
        "AC MONZA": "MONZA",
        "GENOA CFC": "GENOA",
        "COMO 1907": "COMO",
        "BRIGHTON & HOVE ALBION": "BRIGHTON",
        "BRIGHTON AND HOVE ALBION": "BRIGHTON",
        "WEST HAM UNITED FC": "WEST HAM UNITED",
        "TOTTENHAM HOTSPUR FC": "TOTTENHAM HOTSPUR",
        "NEWCASTLE UNITED FC": "NEWCASTLE UNITED",
        "LEICESTER CITY FC": "LEICESTER CITY",
        "WOLVERHAMPTON WANDERERS FC": "WOLVERHAMPTON WANDERERS",
        # Ligue 1
        "PARIS SAINT-GERMAIN FC": "PSG",
        "PARIS SG": "PSG",
        "AS MONACO FC": "MONACO",
        "OLYMPIQUE DE MARSEILLE": "MARSEILLE",
        "OLYMPIQUE LYONNAIS": "LYON",
        "LILLE OSC": "LILLE",
        # Bundesliga
        "BAYER 04 LEVERKUSEN": "BAYER LEVERKUSEN",
        "RB LEIPZIG": "RB LEIPZIG",
        "FC BAYERN MUNCHEN": "BAYERN MUNICH",
        "BORUSSIA DORTMUND": "DORTMUND",
        "VFB STUTTGART": "STUTTGART",
        "EINTRACHT FRANKFURT": "FRANKFURT",
        "VFL WOLFSBURG": "WOLFSBURG",
        "TSG 1899 HOFFENHEIM": "HOFFENHEIM",
        "1. FSV MAINZ 05": "MAINZ",
        "FC AUGSBURG": "AUGSBURG",
        # La Liga
        "CLUB ATHLETICO DE MADRID": "ATLETICO MADRID",
        "CA OSASUNA": "OSASUNA",
        "SEVILLA FC": "SEVILLA",
        "REAL BETIS BALOMPIE": "REAL BETIS",
        "REAL SOCIEDAD DE FUTBOL": "REAL SOCIEDAD",
        "VILLARREAL CF": "VILLARREAL",
        "ATHLETIC CLUB": "ATHLETIC BILBAO",
        "VALENCIA CF": "VALENCIA",
        "GETAFE CF": "GETAFE",
        "RCD MALLORCA": "MALLORCA"
    }
    
    if name in aliases:
        return aliases[name].lower()
        
    # 3. Dynamic cleanup if not an exact match alias
    # Remove common prefixes/suffixes
    remove_list = [
        " FC", " AFC", " CF", " SC", " AS", " SSC", " RC", " UD", " SD", " CD", " FK", " BK", 
        " BC", " AC", " US", " CALCIO", " 1913", " 1909", " 1907"
    ]
    
    for s in remove_list:
        name = name.replace(s, "")
        
    name = name.replace("&", "AND")
        
    return name.strip().lower()

class OddsApiClient:
    """
    Client for The Odds API (https://the-odds-api.com/)
    Provides real-time betting odds for various markets.
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    
    # Mapping our league IDs to The Odds API sport keys
    SPORT_MAP = {
        39: "soccer_epl",
        140: "soccer_spain_la_liga",
        135: "soccer_italy_serie_a",
        78: "soccer_germany_bundesliga",
        61: "soccer_france_ligue_one"
    }

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.the_odds_api_key
        if not self.api_key:
            # Fallback if config.py hasn't been updated yet
            import os
            self.api_key = os.getenv("THE_ODDS_API_KEY")
            
        self.client = httpx.AsyncClient(timeout=15.0)

    async def get_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h,totals") -> List[Dict]:
        """
        Fetch odds for a specific sport and markets.
        """
        if not self.api_key:
            logger.error("The Odds API key is missing")
            return []

        url = f"{self.BASE_URL}/{sport_key}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"The Odds API: Found {len(data)} matches for {sport_key}")
            return data
        except Exception as e:
            logger.error(f"Error fetching odds from The Odds API: {e}")
            if 'response' in locals():
                 logger.error(f"Response Body: {response.text}")
            return []

    async def get_all_league_odds(self, league_ids: List[int]) -> Dict[str, Any]:
        """
        Fetch and aggregate odds for multiple leagues.
        Returns a dict mapping team names (normalized) to their odds.
        """
        all_odds = {}
        for lid in league_ids:
            sport_key = self.SPORT_MAP.get(lid)
            if not sport_key:
                continue
            
            odds_data = await self.get_odds(sport_key)
            for match in odds_data:
                # Key match by normalized teams
                home_norm = normalize_team(match["home_team"])
                away_norm = normalize_team(match["away_team"])
                match_id = f"{home_norm}_vs_{away_norm}".lower()
                
                match_odds = {
                    "1x2": {},
                    "ou_2.5": {},
                    "btts": {}
                }
                
                # Extract odds from bookmakers
                if match.get("bookmakers"):
                    # Prioritize some bookmakers if available (e.g., Pinnacle, Bet365, Betfair)
                    # For now, we'll just use the first one but ensure it has the markets
                    for bm in match["bookmakers"]:
                        found_markets = [m["key"] for m in bm["markets"]]
                        
                        for market in bm["markets"]:
                            if market["key"] == "h2h" and not match_odds["1x2"]:
                                for outcome in market["outcomes"]:
                                    o_name = outcome["name"].lower()
                                    if o_name == match["home_team"].lower(): match_odds["1x2"]["1"] = outcome["price"]
                                    elif o_name == match["away_team"].lower(): match_odds["1x2"]["2"] = outcome["price"]
                                    elif "draw" in o_name or o_name == "x": match_odds["1x2"]["X"] = outcome["price"]
                            
                            elif market["key"] == "totals" and not match_odds["ou_2.5"]:
                                for outcome in market["outcomes"]:
                                    if outcome.get("point") == 2.5:
                                        if outcome["name"].lower() == "over": match_odds["ou_2.5"]["over"] = outcome["price"]
                                        else: match_odds["ou_2.5"]["under"] = outcome["price"]
                                        
                            elif market["key"] == "btts" and not match_odds["btts"]:
                                for outcome in market["outcomes"]:
                                    if outcome["name"].lower() == "yes": match_odds["btts"]["yes"] = outcome["price"]
                                    else: match_odds["btts"]["no"] = outcome["price"]
                        
                        # If we found at least H2H, we can stop searching bookies
                        if match_odds["1x2"]:
                            break
                
                all_odds[match_id] = match_odds
                logger.info(f"Loaded REAL odds for {match_id}")
                
        return all_odds

# Singleton
_odds_api_client = None

def get_odds_api_client():
    global _odds_api_client
    if _odds_api_client is None:
        _odds_api_client = OddsApiClient()
    return _odds_api_client
