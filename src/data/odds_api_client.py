import httpx
import logging
import unicodedata
from typing import Dict, List, Optional, Any
from ..config import get_settings

logger = logging.getLogger(__name__)

def remove_accents(input_str: str) -> str:
    """Remove accents from a string."""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_team(name: str) -> str:
    """Standardize team names for cross-API matching."""
    if not name:
        return ""

    # 1. Remove accents and convert to uppercase
    name = remove_accents(name).upper().strip()
    
    # 2. Known aliases map (FD.org -> Target Mapping)
    aliases = {
        # Serie A
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
        
        # Premier League (abbreviated common names)
        "BRIGHTON & HOVE ALBION": "BRIGHTON",
        "BRIGHTON AND HOVE ALBION": "BRIGHTON",
        "WEST HAM UNITED FC": "WEST HAM",
        "TOTTENHAM HOTSPUR FC": "TOTTENHAM",
        "NEWCASTLE UNITED FC": "NEWCASTLE",
        "NEWCASTLE UNITED": "NEWCASTLE",
        "LEICESTER CITY FC": "LEICESTER",
        "WOLVERHAMPTON WANDERERS FC": "WOLVES",
        "WOLVERHAMPTON WANDERERS": "WOLVES",
        "MANCHESTER UNITED FC": "MANCHESTER UNITED",
        "MANCHESTER CITY FC": "MANCHESTER CITY",
        "CHELSEA FC": "CHELSEA",
        "LIVERPOOL FC": "LIVERPOOL",
        "ARSENAL FC": "ARSENAL",
        "EVERTON FC": "EVERTON",
        "FULHAM FC": "FULHAM",
        "ASTON VILLA FC": "ASTON VILLA",
        "SUNDERLAND AFC": "SUNDERLAND",
        "BRENTFORD FC": "BRENTFORD",
        "AFC BOURNEMOUTH": "BOURNEMOUTH",
        "BURNLEY FC": "BURNLEY",
        "CRYSTAL PALACE FC": "CRYSTAL PALACE",
        "LEEDS UNITED FC": "LEEDS",
        "NOTTINGHAM FOREST FC": "NOTTINGHAM FOREST",
        
        # Ligue 1
        "PARIS SAINT-GERMAIN FC": "PSG",
        "PARIS SAINT GERMAIN": "PSG",
        "PARIS SG": "PSG",
        "AS MONACO FC": "MONACO",
        "AS MONACO": "MONACO",
        "OLYMPIQUE DE MARSEILLE": "MARSEILLE",
        "OLYMPIQUE LYONNAIS": "LYON",
        "LILLE OSC": "LILLE",
        "RACING CLUB DE LENS": "LENS",
        "RC LENS": "LENS",
        "OGC NICE": "NICE",
        "RC STRASBOURG ALSACE": "STRASBOURG",
        "STADE RENNAIS FC 1901": "RENNES",
        "FC NANTES": "NANTES",
        "LE HAVRE AC": "LE HAVRE",
        "ANGERS SCO": "ANGERS",
        "FC LORIENT": "LORIENT",
        "FC METZ": "METZ",
        "STADE BRESTOIS 29": "BREST",
        "AJ AUXERRE": "AUXERRE",
        "PARIS FC": "PARIS FC",
        
        # Bundesliga (abbreviated common names)
        "BAYER 04 LEVERKUSEN": "LEVERKUSEN",
        "RB LEIPZIG": "RB LEIPZIG",
        "FC BAYERN MUNCHEN": "BAYERN MUNICH",
        "BORUSSIA DORTMUND": "DORTMUND",
        "VFB STUTTGART": "STUTTGART",
        "EINTRACHT FRANKFURT": "FRANKFURT",
        "VFL WOLFSBURG": "WOLFSBURG",
        "TSG 1899 HOFFENHEIM": "HOFFENHEIM",
        "1. FSV MAINZ 05": "MAINZ",
        "FC AUGSBURG": "AUGSBURG",
        "BORUSSIA MONCHENGLADBACH": "BORUSSIA MOENCHENGLADBACH",
        "BORUSSIA M'GLADBACH": "BORUSSIA MOENCHENGLADBACH",
        "FC KOLN": "KOELN",
        "1. FC KOELN": "KOELN",
        "FC SCHALKE 04": "SCHALKE 04",
        "UNION BERLIN": "UNION BERLIN",
        "HERTHA BSC": "HERTHA",
        "FC HEIDENHEIM 1846": "HEIDENHEIM",
        
        # La Liga
        "CLUB ATHLETICO DE MADRID": "ATLETICO MADRID",
        "CLUB ATLETICO DE MADRID": "ATLETICO MADRID",
        "ATLETICO MADRID": "ATLETICO MADRID",
        "ATHLETICO MADRID": "ATLETICO MADRID",
        "CA OSASUNA": "OSASUNA",
        "SEVILLA FC": "SEVILLA",
        "REAL BETIS BALOMPIE": "REAL BETIS",
        "REAL SOCIEDAD DE FUTBOL": "REAL SOCIEDAD",
        "VILLARREAL CF": "VILLARREAL",
        "ATHLETIC CLUB": "ATHLETIC BILBAO",
        "ATHLETIC BILBAO": "ATHLETIC BILBAO",
        "VALENCIA CF": "VALENCIA",
        "GETAFE CF": "GETAFE",
        "RCD MALLORCA": "MALLORCA",
        "RAYO VALLECANO DE MADRID": "RAYO VALLECANO",
        "RC CELTA DE VIGO": "CELTA VIGO",
        "CELTA VIGO": "CELTA VIGO",
        "ELCHE CF": "ELCHE",
        "RCD ESPANYOL DE BARCELONA": "ESPANYOL",
        "FC BARCELONA": "BARCELONA",
        "LEVANTE UD": "LEVANTE",
        "REAL MADRID CF": "REAL MADRID",
        "DEPORTIVO ALAVES": "ALAVES",
        "REAL OVIEDO": "OVIEDO",
        "OVIEDO": "OVIEDO",
        "GIRONA FC": "GIRONA"
    }
    
    if name in aliases:
        return aliases[name].lower()
        
    # 3. Dynamic cleanup if not an exact match alias
    # Remove common prefixes/suffixes
    # After cleanup we re-check aliases (so variants with/without 'FC' are covered)

    remove_list = [
        " FC", " AFC", " CF", " SC", " AS ", " SSC", " RC ", " UD", " SD", " CD", " FK", " BK", 
        " BC", " AC ", " US ", " CALCIO", " 1913", " 1909", " 1907", " 1901"
    ]
    
    # Standardize some prefixes that might be at the start
    if name.startswith("RC "): name = name[3:]
    if name.startswith("AS "): name = name[3:]
    if name.startswith("FC "): name = name[3:]
    if name.startswith("UD "): name = name[3:]
    if name.startswith("SD "): name = name[3:]
    if name.startswith("CD "): name = name[3:]

    for s in remove_list:
        name = name.replace(s, "")
        
    name = name.replace("&", "AND")
    
    # Re-check alias mapping after cleanup (catches forms with/without suffixes)
    if name in aliases:
        return aliases[name].lower()
        
    return name.strip().lower()


def sanitize_markets(markets: str) -> str:
    """Normalize requested markets to supported ones.

    Removes unsupported markets and falls back to 'h2h' if
    nothing remains supported. This keeps the default behavior safe
    when calling the external API.
    
    Supported: h2h (1X2), totals (O/U), doublechance (1X, X2, 12)
    """
    supported = ["h2h", "totals", "doublechance"]
    parts = [m.strip().lower() for m in markets.split(",") if m.strip()]
    filtered = [m for m in parts if m in supported]
    if not filtered:
        return "h2h"
    return ",".join(filtered)

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

    async def get_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h,totals,doublechance") -> List[Dict]:
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
                    "btts": {},
                    "double_chance": {}  # 1X, X2, 12
                }
                
                # Extract odds from bookmakers
                if match.get("bookmakers"):
                    # Prioritize some bookmakers if available (e.g., Pinnacle, Bet365, Betfair)
                    # For now, we'll just use the first one but ensure it has the markets
                    for bm in match["bookmakers"]:
                        found_markets = [m["key"] for m in bm["markets"]]
                        
                        for market in bm["markets"]:
                            if market["key"] == "h2h" and not match_odds["1x2"]:
                                # Compare normalized team names to handle different abbreviations/aliases
                                home_norm_local = normalize_team(match["home_team"])
                                away_norm_local = normalize_team(match["away_team"])
                                for outcome in market["outcomes"]:
                                    o_norm = normalize_team(outcome["name"])
                                    # direct normalized match
                                    if o_norm == home_norm_local:
                                        match_odds["1x2"]["1"] = outcome["price"]
                                    elif o_norm == away_norm_local:
                                        match_odds["1x2"]["2"] = outcome["price"]
                                    elif "draw" in outcome["name"].lower() or outcome["name"].lower() == "x":
                                        match_odds["1x2"]["X"] = outcome["price"]
                                # If we still failed to map both teams, try loose name matching
                                if match_odds["1x2"] == {}:
                                    for outcome in market["outcomes"]:
                                        o_name = outcome["name"].lower()
                                        if match["home_team"].lower() in o_name:
                                            match_odds["1x2"]["1"] = outcome["price"]
                                        elif match["away_team"].lower() in o_name:
                                            match_odds["1x2"]["2"] = outcome["price"]
                                        elif "draw" in o_name or o_name == "x":
                                            match_odds["1x2"]["X"] = outcome["price"]
                                # If still empty, log for debugging
                                if match_odds["1x2"] == {}:
                                    logger.debug(f"Could not map h2h outcomes to teams for match {match.get('id')} ({match.get('home_team')} vs {match.get('away_team')}). Outcomes: {[o['name'] for o in market.get('outcomes', [])]}")
                            
                            elif market["key"] == "totals" and not match_odds["ou_2.5"]:
                                for outcome in market["outcomes"]:
                                    if outcome.get("point") == 2.5:
                                        if outcome["name"].lower() == "over": match_odds["ou_2.5"]["over"] = outcome["price"]
                                        else: match_odds["ou_2.5"]["under"] = outcome["price"]
                                        
                            elif market["key"] == "btts" and not match_odds["btts"]:
                                for outcome in market["outcomes"]:
                                    if outcome["name"].lower() == "yes": match_odds["btts"]["yes"] = outcome["price"]
                                    else: match_odds["btts"]["no"] = outcome["price"]
                                    
                            elif market["key"] == "doublechance" and not match_odds["double_chance"]:
                                # Double Chance: 1X (Home or Draw), X2 (Draw or Away), 12 (Home or Away)
                                for outcome in market["outcomes"]:
                                    name = outcome["name"].upper().replace(" ", "")
                                    # The Odds API uses patterns like "Home or Draw", "Draw or Away", etc.
                                    if "HOME" in name and "DRAW" in name:
                                        match_odds["double_chance"]["1X"] = outcome["price"]
                                    elif "AWAY" in name and "DRAW" in name:
                                        match_odds["double_chance"]["X2"] = outcome["price"]
                                    elif ("HOME" in name and "AWAY" in name) or name == "12":
                                        match_odds["double_chance"]["12"] = outcome["price"]
                        
                                # If we found at least H2H, we can stop searching bookies
                        if match_odds["1x2"]:
                            break
                
                # No external API-Football fallback: use only The Odds API markets
                # (API-Football free plan does not guarantee current season data)


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
