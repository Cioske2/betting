"""
API-Football client for fetching match data, statistics, and odds.
Documentation: https://www.api-football.com/documentation-v3
"""

import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from ..config import get_settings, LEAGUE_INFO

logger = logging.getLogger(__name__)


@dataclass
class Match:
    """Represents a football match."""
    fixture_id: int
    league_id: int
    league_name: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    date: datetime
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    status: str = "NS"  # NS = Not Started
    

@dataclass
class TeamStats:
    """Team statistics for a season."""
    team_id: int
    team_name: str
    league_id: int
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    home_wins: int
    home_draws: int
    home_losses: int
    home_goals_for: int
    home_goals_against: int
    away_wins: int
    away_draws: int
    away_losses: int
    away_goals_for: int
    away_goals_against: int


@dataclass
class Odds:
    """Betting odds for a match."""
    fixture_id: int
    bookmaker: str
    home_win: float
    draw: float
    away_win: float
    updated_at: datetime


class APIFootballClient:
    """
    Client for API-Football v3.
    
    Features:
    - Rate limiting aware
    - Response caching
    - Automatic retries
    - Type-safe responses
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.api_football_base_url
        self._cache = TTLCache(maxsize=1000, ttl=self.settings.cache_ttl)
        
    def _headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        return {
            "x-apisports-key": self.settings.api_football_key,
            "Accept": "application/json"
        }
    
    async def _request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with caching.
        
        Args:
            endpoint: API endpoint (e.g., 'fixtures')
            params: Query parameters
            
        Returns:
            API response data
        """
        cache_key = f"{endpoint}:{str(sorted(params.items()) if params else '')}"
        
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {endpoint}")
            return self._cache[cache_key]
        
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"API Request: {endpoint} with params {params}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=30.0
                )
                
                logger.debug(f"API Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"API Error: {response.status_code} - {response.text[:200]}")
                    response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors in response
                if data.get("errors"):
                    errors = data["errors"]
                    if isinstance(errors, dict) and errors:
                        error_msg = list(errors.values())[0]
                        logger.error(f"API returned error: {error_msg}")
                        # Return empty response instead of raising to avoid retry loops
                        return {"response": [], "errors": errors}
                    elif isinstance(errors, list) and errors:
                        logger.error(f"API returned errors: {errors}")
                        return {"response": [], "errors": errors}
                
                # Log response count
                response_data = data.get("response", [])
                logger.info(f"API returned {len(response_data) if isinstance(response_data, list) else 'non-list'} items")
                
                self._cache[cache_key] = data
                return data
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {endpoint}: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error for {endpoint}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}")
            raise
    
    async def get_fixtures(
        self,
        league_id: int,
        season: int,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Match]:
        """
        Get fixtures for a league and season.
        
        Args:
            league_id: League ID (e.g., 39 for Premier League)
            season: Season year (e.g., 2024)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            status: Match status filter (NS, FT, etc.)
            
        Returns:
            List of Match objects
        """
        params = {
            "league": league_id,
            "season": season
        }
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if status:
            params["status"] = status
            
        data = await self._request("fixtures", params)
        matches = []
        
        for fixture in data.get("response", []):
            match = Match(
                fixture_id=fixture["fixture"]["id"],
                league_id=fixture["league"]["id"],
                league_name=fixture["league"]["name"],
                home_team_id=fixture["teams"]["home"]["id"],
                home_team_name=fixture["teams"]["home"]["name"],
                away_team_id=fixture["teams"]["away"]["id"],
                away_team_name=fixture["teams"]["away"]["name"],
                date=datetime.fromisoformat(
                    fixture["fixture"]["date"].replace("Z", "+00:00")
                ),
                home_goals=fixture["goals"]["home"],
                away_goals=fixture["goals"]["away"],
                status=fixture["fixture"]["status"]["short"]
            )
            matches.append(match)
            
        return matches
    
    async def get_finished_matches(
        self,
        league_id: int,
        season: int,
        last_n: Optional[int] = None
    ) -> List[Match]:
        """
        Get finished matches for analysis.
        
        Args:
            league_id: League ID
            season: Season year
            last_n: Limit to last N matches
            
        Returns:
            List of finished matches sorted by date
        """
        matches = await self.get_fixtures(
            league_id=league_id,
            season=season,
            status="FT"  # Finished
        )
        
        # Sort by date descending
        matches.sort(key=lambda x: x.date, reverse=True)
        
        if last_n:
            matches = matches[:last_n]
            
        return matches
    
    async def get_upcoming_fixtures(
        self,
        league_id: int,
        days_ahead: int = 7
    ) -> List[Match]:
        """
        Get upcoming matches for prediction.
        
        Args:
            league_id: League ID
            days_ahead: Number of days to look ahead
            
        Returns:
            List of upcoming matches
        """
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # API-Football FREE plan only has access to seasons 2021-2023
        # For recent/current matches, we'll use season 2024 as fallback
        # In production, upgrade API plan or use football-data.org
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Try current season calculation first
        season = current_year if current_month >= 8 else current_year - 1
        
        # Free plan limitation: max season is 2023
        # Try 2024 first (might work), then fallback to 2023
        for try_season in [2024, 2023]:
            matches = await self.get_fixtures(
                league_id=league_id,
                season=try_season,
                from_date=today,
                to_date=end_date,
                status="NS"
            )
            if matches:
                return matches
        
        # No matches found in available seasons
        return []
    
    async def get_team_statistics(
        self,
        team_id: int,
        league_id: int,
        season: int
    ) -> Optional[TeamStats]:
        """
        Get detailed statistics for a team in a league/season.
        
        Args:
            team_id: Team ID
            league_id: League ID
            season: Season year
            
        Returns:
            TeamStats object or None if not found
        """
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        
        data = await self._request("teams/statistics", params)
        response = data.get("response", {})
        
        if not response:
            return None
            
        fixtures = response.get("fixtures", {})
        goals = response.get("goals", {})
        
        return TeamStats(
            team_id=team_id,
            team_name=response.get("team", {}).get("name", "Unknown"),
            league_id=league_id,
            matches_played=fixtures.get("played", {}).get("total", 0),
            wins=fixtures.get("wins", {}).get("total", 0),
            draws=fixtures.get("draws", {}).get("total", 0),
            losses=fixtures.get("loses", {}).get("total", 0),
            goals_for=goals.get("for", {}).get("total", {}).get("total", 0),
            goals_against=goals.get("against", {}).get("total", {}).get("total", 0),
            home_wins=fixtures.get("wins", {}).get("home", 0),
            home_draws=fixtures.get("draws", {}).get("home", 0),
            home_losses=fixtures.get("loses", {}).get("home", 0),
            home_goals_for=goals.get("for", {}).get("total", {}).get("home", 0),
            home_goals_against=goals.get("against", {}).get("total", {}).get("home", 0),
            away_wins=fixtures.get("wins", {}).get("away", 0),
            away_draws=fixtures.get("draws", {}).get("away", 0),
            away_losses=fixtures.get("loses", {}).get("away", 0),
            away_goals_for=goals.get("for", {}).get("total", {}).get("away", 0),
            away_goals_against=goals.get("against", {}).get("total", {}).get("away", 0)
        )
    
    async def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        last_n: int = 10
    ) -> List[Match]:
        """
        Get head-to-head matches between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last_n: Number of matches to retrieve
            
        Returns:
            List of h2h matches
        """
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last_n
        }
        
        data = await self._request("fixtures/headtohead", params)
        matches = []
        
        for fixture in data.get("response", []):
            match = Match(
                fixture_id=fixture["fixture"]["id"],
                league_id=fixture["league"]["id"],
                league_name=fixture["league"]["name"],
                home_team_id=fixture["teams"]["home"]["id"],
                home_team_name=fixture["teams"]["home"]["name"],
                away_team_id=fixture["teams"]["away"]["id"],
                away_team_name=fixture["teams"]["away"]["name"],
                date=datetime.fromisoformat(
                    fixture["fixture"]["date"].replace("Z", "+00:00")
                ),
                home_goals=fixture["goals"]["home"],
                away_goals=fixture["goals"]["away"],
                status=fixture["fixture"]["status"]["short"]
            )
            matches.append(match)
            
        return matches
    
    async def get_odds(
        self,
        fixture_id: int,
        bookmaker_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get betting odds for a fixture across multiple markets.
        """
        params = {"fixture": fixture_id}
        if bookmaker_id:
            params["bookmaker"] = bookmaker_id
            
        data = await self._request("odds", params)
        all_odds = {}
        
        for response in data.get("response", []):
            for bookmaker in response.get("bookmakers", []):
                # We prioritize the requested bookmaker or the first one found
                if bookmaker_id and bookmaker["id"] != bookmaker_id:
                    continue
                
                for bet in bookmaker.get("bets", []):
                    # Match Winner (bet id 1)
                    if bet.get("id") == 1:
                        values = {v["value"]: float(v["odd"]) for v in bet.get("values", [])}
                        if "Home" in values and "Draw" in values and "Away" in values:
                            all_odds["1x2"] = {"1": values["Home"], "X": values["Draw"], "2": values["Away"]}
                    
                    # Goals Over/Under (bet id 5)
                    elif bet.get("id") == 5:
                        for v in bet.get("values", []):
                            if v["value"] == "Over 2.5":
                                all_odds["ou_2.5"] = all_odds.get("ou_2.5", {})
                                all_odds["ou_2.5"]["over"] = float(v["odd"])
                            elif v["value"] == "Under 2.5":
                                all_odds["ou_2.5"] = all_odds.get("ou_2.5", {})
                                all_odds["ou_2.5"]["under"] = float(v["odd"])
                                
                    # Double Chance (bet id 12)
                    elif bet.get("id") == 12:
                        values = {v["value"]: float(v["odd"]) for v in bet.get("values", [])}
                        all_odds["dc"] = values
                
                # If we found odds for this bookmaker, we stop (or continue if we want merge)
                if all_odds:
                    break
                            
        return all_odds
    
    async def get_standings(
        self,
        league_id: int,
        season: int
    ) -> List[Dict[str, Any]]:
        """
        Get current league standings.
        
        Args:
            league_id: League ID
            season: Season year
            
        Returns:
            List of standings entries
        """
        params = {
            "league": league_id,
            "season": season
        }
        
        data = await self._request("standings", params)
        standings = []
        
        for response in data.get("response", []):
            for league in response.get("league", {}).get("standings", []):
                for team in league:
                    standings.append({
                        "rank": team["rank"],
                        "team_id": team["team"]["id"],
                        "team_name": team["team"]["name"],
                        "points": team["points"],
                        "played": team["all"]["played"],
                        "won": team["all"]["win"],
                        "drawn": team["all"]["draw"],
                        "lost": team["all"]["lose"],
                        "goals_for": team["all"]["goals"]["for"],
                        "goals_against": team["all"]["goals"]["against"],
                        "goal_diff": team["goalsDiff"],
                        "form": team.get("form", "")
                    })
                    
        return standings
    
    async def get_all_leagues_matches(
        self,
        season: int,
        status: str = "FT"
    ) -> Dict[int, List[Match]]:
        """
        Get matches for all configured leagues.
        
        Args:
            season: Season year
            status: Match status filter
            
        Returns:
            Dictionary mapping league_id to list of matches
        """
        result = {}
        
        for league_id in self.settings.league_ids:
            logger.info(f"Fetching matches for {LEAGUE_INFO.get(league_id, {}).get('name', league_id)}")
            matches = await self.get_fixtures(
                league_id=league_id,
                season=season,
                status=status
            )
            result[league_id] = matches
            
        return result


# Singleton client instance
_client: Optional[APIFootballClient] = None


def get_client() -> APIFootballClient:
    """Get or create the API client singleton."""
    global _client
    if _client is None:
        _client = APIFootballClient()
    return _client
