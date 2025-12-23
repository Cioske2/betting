"""
Football-Data.org API Client.

Free API for top European leagues including current season.
Documentation: https://www.football-data.org/documentation/quickstart
"""

import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from cachetools import TTLCache
import logging

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FDMatch:
    """Match from Football-Data.org."""
    match_id: int
    competition_id: int
    competition_name: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    utc_date: datetime
    status: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    # Betting odds (if available)
    home_win_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_win_odds: Optional[float] = None


# Mapping from football-data.org competition codes to our league IDs
COMPETITION_MAPPING = {
    "PL": 39,    # Premier League
    "PD": 140,   # La Liga (Primera Division)
    "SA": 135,   # Serie A
    "BL1": 78,   # Bundesliga
    "FL1": 61,   # Ligue 1
}

# Reverse mapping
LEAGUE_TO_COMPETITION = {v: k for k, v in COMPETITION_MAPPING.items()}


class FootballDataClient:
    """
    Client for football-data.org API.
    
    Free tier includes:
    - Top 5 European leagues
    - Current and past seasons
    - 10 requests per minute
    """
    
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self):
        self.settings = get_settings()
        self._cache = TTLCache(maxsize=500, ttl=self.settings.cache_ttl)
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        return {
            "X-Auth-Token": self.settings.football_data_key,
            "Accept": "application/json"
        }
    
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an API request."""
        cache_key = f"{endpoint}:{str(sorted(params.items()) if params else '')}"
        
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {endpoint}")
            return self._cache[cache_key]
        
        url = f"{self.BASE_URL}/{endpoint}"
        logger.info(f"FD API Request: {endpoint} with params {params}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code == 429:
                    logger.warning("Rate limited by football-data.org")
                    return {"matches": [], "error": "rate_limited"}
                
                if response.status_code != 200:
                    logger.error(f"FD API Error: {response.status_code} - {response.text[:200]}")
                    return {"matches": [], "error": response.text}
                
                data = response.json()
                self._cache[cache_key] = data
                
                return data
                
        except Exception as e:
            logger.error(f"FD API request failed: {e}")
            return {"matches": [], "error": str(e)}
    
    async def get_matches(
        self,
        competition_code: str,
        season: Optional[int] = None,
        status: Optional[str] = None,
        matchday: Optional[int] = None
    ) -> List[FDMatch]:
        """
        Get matches for a competition.
        
        Args:
            competition_code: Competition code (PL, SA, PD, BL1, FL1)
            season: Season year (e.g., 2024 for 2024-2025)
            status: SCHEDULED, LIVE, IN_PLAY, PAUSED, FINISHED, etc.
            matchday: Specific matchday
            
        Returns:
            List of FDMatch objects
        """
        params = {}
        if season:
            params["season"] = season
        if status:
            params["status"] = status
        if matchday:
            params["matchday"] = matchday
        
        data = await self._request(f"competitions/{competition_code}/matches", params)
        matches = []
        
        for m in data.get("matches", []):
            try:
                match = FDMatch(
                    match_id=m["id"],
                    competition_id=COMPETITION_MAPPING.get(competition_code, 0),
                    competition_name=m.get("competition", {}).get("name", "Unknown"),
                    home_team_id=m["homeTeam"]["id"],
                    home_team_name=m["homeTeam"]["name"],
                    away_team_id=m["awayTeam"]["id"],
                    away_team_name=m["awayTeam"]["name"],
                    utc_date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
                    status=m["status"],
                    home_score=m.get("score", {}).get("fullTime", {}).get("home"),
                    away_score=m.get("score", {}).get("fullTime", {}).get("away"),
                    home_win_odds=m.get("odds", {}).get("homeWin"),
                    draw_odds=m.get("odds", {}).get("draw"),
                    away_win_odds=m.get("odds", {}).get("awayWin")
                )
                matches.append(match)
            except Exception as e:
                logger.warning(f"Failed to parse match: {e}")
                continue
        
        logger.info(f"FD API returned {len(matches)} matches for {competition_code}")
        return matches
    
    async def get_finished_matches(
        self,
        league_id: int,
        season: int
    ) -> List[FDMatch]:
        """
        Get finished matches for a league.
        
        Args:
            league_id: Our internal league ID (39, 140, 135, 78, 61)
            season: Season year
            
        Returns:
            List of finished matches
        """
        competition_code = LEAGUE_TO_COMPETITION.get(league_id)
        if not competition_code:
            logger.warning(f"Unknown league ID: {league_id}")
            return []
        
        matches = await self.get_matches(
            competition_code=competition_code,
            season=season,
            status="FINISHED"
        )
        
        return sorted(matches, key=lambda x: x.utc_date, reverse=True)
    
    async def get_upcoming_matches(
        self,
        league_id: int,
        days_ahead: int = 7,
        limit: int = 50
    ) -> List[FDMatch]:
        """
        Get upcoming scheduled matches for the next N days.
        
        Args:
            league_id: Our internal league ID
            days_ahead: Number of days to look ahead
            limit: Max matches to return
            
        Returns:
            List of scheduled matches
        """
        competition_code = LEAGUE_TO_COMPETITION.get(league_id)
        if not competition_code:
            return []
        
        from datetime import date, timedelta
        today = date.today()
        future = today + timedelta(days=days_ahead)
        
        params = {
            "dateFrom": today.isoformat(),
            "dateTo": future.isoformat(),
            "status": "SCHEDULED"
        }
        
        data = await self._request(f"competitions/{competition_code}/matches", params)
        matches = []
        
        for m in data.get("matches", []):
            try:
                match = FDMatch(
                    match_id=m["id"],
                    competition_id=league_id,
                    competition_name=m.get("competition", {}).get("name", "Unknown"),
                    home_team_id=m["homeTeam"]["id"],
                    home_team_name=m["homeTeam"]["name"],
                    away_team_id=m["awayTeam"]["id"],
                    away_team_name=m["awayTeam"]["name"],
                    utc_date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
                    status=m["status"],
                    home_score=m.get("score", {}).get("fullTime", {}).get("home"),
                    away_score=m.get("score", {}).get("fullTime", {}).get("away")
                )
                matches.append(match)
            except Exception as e:
                logger.warning(f"Failed to parse upcoming match: {e}")
                continue
        
        # Sort by date ascending (nearest first)
        matches.sort(key=lambda x: x.utc_date)
        return matches[:limit]
    
    async def get_standings(
        self,
        league_id: int,
        season: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current league standings.
        
        Args:
            league_id: Our internal league ID
            season: Optional season year
            
        Returns:
            List of standings entries
        """
        competition_code = LEAGUE_TO_COMPETITION.get(league_id)
        if not competition_code:
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        data = await self._request(f"competitions/{competition_code}/standings", params)
        standings = []
        
        for standing in data.get("standings", []):
            if standing.get("type") == "TOTAL":
                for team in standing.get("table", []):
                    standings.append({
                        "rank": team["position"],
                        "team_id": team["team"]["id"],
                        "team_name": team["team"]["name"],
                        "points": team["points"],
                        "played": team["playedGames"],
                        "won": team["won"],
                        "drawn": team["draw"],
                        "lost": team["lost"],
                        "goals_for": team["goalsFor"],
                        "goals_against": team["goalsAgainst"],
                        "goal_diff": team["goalDifference"],
                        "form": team.get("form", "")
                    })
        
        return standings


# Singleton instance
_fd_client: Optional[FootballDataClient] = None


def get_fd_client() -> FootballDataClient:
    """Get or create the Football-Data client singleton."""
    global _fd_client
    if _fd_client is None:
        _fd_client = FootballDataClient()
    return _fd_client
