"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # API-Football
    api_football_key: str = Field(
        default="",
        description="API key for api-football.com"
    )
    api_football_base_url: str = "https://v3.football.api-sports.io"
    
    # Football-Data.org (Free for current season)
    football_data_key: str = Field(
        default="",
        description="API key for football-data.org (free, current season)"
    )
    
    # League IDs for API-Football
    # 39=Premier League, 140=La Liga, 135=Serie A, 78=Bundesliga, 61=Ligue 1
    leagues: str = Field(
        default="39,140,135,78,61",
        description="Comma-separated league IDs"
    )
    
    # Model weights for ensemble
    poisson_weight: float = Field(default=0.4, ge=0, le=1)
    xgboost_weight: float = Field(default=0.6, ge=0, le=1)
    
    # Value bet settings
    min_edge: float = Field(
        default=0.05,
        description="Minimum edge (5%) to consider a value bet"
    )
    kelly_fraction: float = Field(
        default=0.25,
        description="Fraction of Kelly criterion for stake sizing"
    )
    
    # Feature engineering
    form_matches: int = Field(
        default=5,
        description="Number of recent matches for form calculation"
    )
    h2h_matches: int = Field(
        default=10,
        description="Number of head-to-head matches to consider"
    )
    
    # Cache
    cache_ttl: int = Field(
        default=3600,
        description="Cache time-to-live in seconds"
    )
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    @property
    def league_ids(self) -> List[int]:
        """Parse league IDs from comma-separated string."""
        return [int(x.strip()) for x in self.leagues.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# League metadata for reference
LEAGUE_INFO = {
    39: {"name": "Premier League", "country": "England"},
    140: {"name": "La Liga", "country": "Spain"},
    135: {"name": "Serie A", "country": "Italy"},
    78: {"name": "Bundesliga", "country": "Germany"},
    61: {"name": "Ligue 1", "country": "France"},
}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
