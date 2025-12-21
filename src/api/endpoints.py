"""
FastAPI endpoints for Football Betting Prediction API.

Provides REST endpoints for:
- Match predictions (1X2 probabilities)
- Value bet analysis
- Team statistics and form
- Head-to-head analysis
- Model training and status
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..config import get_settings, LEAGUE_INFO
from ..data.api_football_client import get_client, APIFootballClient, Match
from ..data.football_data_client import get_fd_client, FDMatch
from ..features.feature_engineering import FeatureEngineer, MatchFeatures
from ..models.ensemble import EnsemblePredictor
from ..betting.value_bet import ValueBetAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances (will be initialized on startup)
_ensemble: Optional[EnsemblePredictor] = None
_feature_engineer: Optional[FeatureEngineer] = None
_value_analyzer: Optional[ValueBetAnalyzer] = None
_training_in_progress: bool = False

# Model save paths
import os
import pickle
from pathlib import Path

MODEL_DIR = Path("models")
POISSON_PATH = MODEL_DIR / "poisson_model.pkl"
XGBOOST_PATH = MODEL_DIR / "xgboost_model.pkl"
FEATURE_ENG_PATH = MODEL_DIR / "feature_engineer.pkl"


def save_models(ensemble: EnsemblePredictor, feature_eng: FeatureEngineer) -> None:
    """Save trained models to disk."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Save Poisson model
    with open(POISSON_PATH, "wb") as f:
        pickle.dump(ensemble.poisson, f)
    
    # Save XGBoost model  
    with open(XGBOOST_PATH, "wb") as f:
        pickle.dump(ensemble.xgboost, f)
    
    # Save feature engineer
    with open(FEATURE_ENG_PATH, "wb") as f:
        pickle.dump(feature_eng, f)
    
    logger.info(f"Models saved to {MODEL_DIR}")


def load_models(ensemble: EnsemblePredictor) -> bool:
    """Load models from disk if available. Returns True if loaded."""
    if not POISSON_PATH.exists() or not XGBOOST_PATH.exists():
        logger.info("No saved models found")
        return False
    
    try:
        with open(POISSON_PATH, "rb") as f:
            ensemble.poisson = pickle.load(f)
        
        with open(XGBOOST_PATH, "rb") as f:
            ensemble.xgboost = pickle.load(f)
        
        logger.info("Models loaded from disk")
        return True
    except Exception as e:
        logger.warning(f"Failed to load models: {e}")
        return False


def get_ensemble() -> EnsemblePredictor:
    """Get or create ensemble predictor, loading saved models if available."""
    global _ensemble
    if _ensemble is None:
        settings = get_settings()
        _ensemble = EnsemblePredictor(
            poisson_weight=settings.poisson_weight,
            xgboost_weight=settings.xgboost_weight
        )
        # Try to load saved models
        load_models(_ensemble)
    return _ensemble


def get_feature_engineer() -> FeatureEngineer:
    """Get or create feature engineer, loading from disk if available."""
    global _feature_engineer
    if _feature_engineer is None:
        settings = get_settings()
        
        # Try to load saved feature engineer
        if FEATURE_ENG_PATH.exists():
            try:
                with open(FEATURE_ENG_PATH, "rb") as f:
                    _feature_engineer = pickle.load(f)
                logger.info("Feature engineer loaded from disk")
            except Exception as e:
                logger.warning(f"Failed to load feature engineer: {e}")
                _feature_engineer = FeatureEngineer(form_matches=settings.form_matches)
        else:
            _feature_engineer = FeatureEngineer(form_matches=settings.form_matches)
    return _feature_engineer


def get_value_analyzer() -> ValueBetAnalyzer:
    """Get or create value bet analyzer."""
    global _value_analyzer
    if _value_analyzer is None:
        settings = get_settings()
        _value_analyzer = ValueBetAnalyzer(
            min_edge=settings.min_edge,
            kelly_fraction=settings.kelly_fraction
        )
    return _value_analyzer


# ============== Request/Response Models ==============

class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    home_team_id: int = Field(..., description="Home team ID from API-Football")
    away_team_id: int = Field(..., description="Away team ID from API-Football")
    home_team_name: str = Field(..., description="Home team name")
    away_team_name: str = Field(..., description="Away team name")
    league_id: int = Field(default=39, description="League ID from API-Football")


class ValueBetRequest(BaseModel):
    """Request model for value bet analysis."""
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    league_id: int = 39
    odds: Dict[str, float] = Field(
        ...,
        description="Bookmaker odds: {'1': 2.50, 'X': 3.20, '2': 2.80}",
        examples=[{"1": 2.50, "X": 3.20, "2": 2.80}]
    )


class TrainRequest(BaseModel):
    """Request model for training endpoint."""
    league_ids: Optional[List[int]] = Field(
        default=None,
        description="League IDs to train on (defaults to all configured)"
    )
    season: int = Field(
        default=2024,
        description="Season year to train on"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    match: Dict[str, Any]
    probabilities: Dict[str, float]
    model_details: Dict[str, Any]
    expected_goals: Dict[str, float]
    prediction: str
    confidence: str
    timestamp: str


class ValueBetResponse(BaseModel):
    """Response model for value bet analysis."""
    match: str
    value_bets: List[Dict]
    all_outcomes: List[Dict]
    best_value: Optional[Dict]
    recommendation: str
    total_edge_percent: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models: Dict[str, Any]
    api_configured: bool
    timestamp: str


# ============== Endpoints ==============

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health and model status.
    
    Returns model readiness and API configuration status.
    """
    global _training_in_progress
    settings = get_settings()
    ensemble = get_ensemble()
    
    return {
        "status": "healthy",
        "models": ensemble.get_model_status(),
        "api_configured": bool(settings.api_football_key),
        "training_in_progress": _training_in_progress,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/leagues", tags=["Data"])
async def list_leagues():
    """
    List available leagues with their IDs.
    
    Returns configured leagues and their metadata.
    """
    settings = get_settings()
    return {
        "configured_leagues": [
            {
                "id": lid,
                "name": LEAGUE_INFO.get(lid, {}).get("name", "Unknown"),
                "country": LEAGUE_INFO.get(lid, {}).get("country", "Unknown")
            }
            for lid in settings.league_ids
        ],
        "all_known_leagues": LEAGUE_INFO
    }


@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_match(request: PredictRequest):
    """
    Predict 1X2 probabilities for a match.
    
    Uses ensemble of Poisson and XGBoost models.
    Returns probabilities for Home Win, Draw, and Away Win.
    """
    try:
        global _training_in_progress
        ensemble = get_ensemble()
        
        if not ensemble.poisson.is_fitted:
            if _training_in_progress:
                raise HTTPException(
                    status_code=503,
                    detail="Models are currently training in background. Please wait 1-2 minutes and try again."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Models not trained yet. Please train models first using /api/train-current"
                )
        
        prediction = ensemble.predict(
            home_team_id=request.home_team_id,
            away_team_id=request.away_team_id,
            home_team_name=request.home_team_name,
            away_team_name=request.away_team_name,
            league_id=request.league_id
        )
        
        result = prediction.to_dict()
        return PredictionResponse(
            match=result["match"],
            probabilities=result["probabilities"],
            model_details=result["model_details"],
            expected_goals=result["expected_goals"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            timestamp=result["timestamp"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for {request.home_team_name} vs {request.away_team_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/predict/{home_team_id}/{away_team_id}", tags=["Predictions"])
async def predict_by_ids(
    home_team_id: int,
    away_team_id: int,
    league_id: int = Query(default=39, description="League ID")
):
    """
    Quick prediction by team IDs only.
    
    Team names will show as IDs if not provided.
    """
    ensemble = get_ensemble()
    
    prediction = ensemble.predict(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=f"Team {home_team_id}",
        away_team_name=f"Team {away_team_id}",
        league_id=league_id
    )
    
    return prediction.to_dict()


@router.post("/value-bet", response_model=ValueBetResponse, tags=["Betting"])
async def analyze_value_bet(request: ValueBetRequest):
    """
    Analyze betting value for a match.
    
    Compares model predictions with bookmaker odds.
    Identifies value bets and calculates optimal stakes.
    """
    ensemble = get_ensemble()
    analyzer = get_value_analyzer()
    
    # Get model prediction
    prediction = ensemble.predict(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        home_team_name=request.home_team_name,
        away_team_name=request.away_team_name,
        league_id=request.league_id
    )
    
    # Convert probabilities to 0-1 scale
    model_probs = {
        "1": prediction.home_win_pct / 100,
        "X": prediction.draw_pct / 100,
        "2": prediction.away_win_pct / 100
    }
    
    # Analyze value
    analysis = analyzer.analyze_match(
        model_probs=model_probs,
        bookmaker_odds=request.odds,
        home_team=request.home_team_name,
        away_team=request.away_team_name
    )
    
    result = analysis.to_dict()
    return ValueBetResponse(
        match=result["match"],
        value_bets=result["value_bets"],
        all_outcomes=result["all_outcomes"],
        best_value=result["best_value"],
        recommendation=result["recommendation"],
        total_edge_percent=result["total_edge_percent"]
    )


@router.get("/fixtures/upcoming", tags=["Data"])
async def get_upcoming_fixtures(
    league_id: int = Query(default=39, description="League ID"),
    days: int = Query(default=7, ge=1, le=30, description="Days ahead")
):
    """
    Get upcoming fixtures for a league.
    
    Returns matches scheduled in the next N days.
    """
    client = get_client()
    
    try:
        fixtures = await client.get_upcoming_fixtures(
            league_id=league_id,
            days_ahead=days
        )
        
        return {
            "league_id": league_id,
            "league_name": LEAGUE_INFO.get(league_id, {}).get("name", "Unknown"),
            "fixtures": [
                {
                    "fixture_id": f.fixture_id,
                    "home_team": f.home_team_name,
                    "home_team_id": f.home_team_id,
                    "away_team": f.away_team_name,
                    "away_team_id": f.away_team_id,
                    "date": f.date.isoformat(),
                    "status": f.status
                }
                for f in fixtures
            ],
            "count": len(fixtures)
        }
    except Exception as e:
        logger.error(f"Failed to fetch fixtures: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/team/{team_id}/stats", tags=["Data"])
async def get_team_stats(
    team_id: int,
    league_id: int = Query(default=39, description="League ID"),
    season: int = Query(default=2024, description="Season year")
):
    """
    Get detailed statistics for a team.
    
    Returns season statistics including goals, wins, form etc.
    """
    client = get_client()
    
    try:
        stats = await client.get_team_statistics(
            team_id=team_id,
            league_id=league_id,
            season=season
        )
        
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"No statistics found for team {team_id}"
            )
        
        return {
            "team_id": stats.team_id,
            "team_name": stats.team_name,
            "league_id": league_id,
            "season": season,
            "matches_played": stats.matches_played,
            "record": {
                "wins": stats.wins,
                "draws": stats.draws,
                "losses": stats.losses
            },
            "goals": {
                "scored": stats.goals_for,
                "conceded": stats.goals_against,
                "difference": stats.goals_for - stats.goals_against
            },
            "home": {
                "wins": stats.home_wins,
                "draws": stats.home_draws,
                "losses": stats.home_losses,
                "goals_for": stats.home_goals_for,
                "goals_against": stats.home_goals_against
            },
            "away": {
                "wins": stats.away_wins,
                "draws": stats.away_draws,
                "losses": stats.away_losses,
                "goals_for": stats.away_goals_for,
                "goals_against": stats.away_goals_against
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch team stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/h2h/{team1_id}/{team2_id}", tags=["Data"])
async def get_head_to_head(
    team1_id: int,
    team2_id: int,
    last_n: int = Query(default=10, ge=1, le=50, description="Number of matches")
):
    """
    Get head-to-head history between two teams.
    
    Returns recent matches with results.
    """
    client = get_client()
    
    try:
        matches = await client.get_head_to_head(
            team1_id=team1_id,
            team2_id=team2_id,
            last_n=last_n
        )
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        
        match_list = []
        for m in matches:
            if m.home_goals is not None and m.away_goals is not None:
                if m.home_team_id == team1_id:
                    if m.home_goals > m.away_goals:
                        team1_wins += 1
                    elif m.home_goals < m.away_goals:
                        team2_wins += 1
                    else:
                        draws += 1
                else:
                    if m.away_goals > m.home_goals:
                        team1_wins += 1
                    elif m.away_goals < m.home_goals:
                        team2_wins += 1
                    else:
                        draws += 1
            
            match_list.append({
                "fixture_id": m.fixture_id,
                "date": m.date.isoformat(),
                "home_team": m.home_team_name,
                "away_team": m.away_team_name,
                "score": f"{m.home_goals}-{m.away_goals}" if m.home_goals is not None else "N/A",
                "league": m.league_name
            })
        
        return {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "summary": {
                "team1_wins": team1_wins,
                "draws": draws,
                "team2_wins": team2_wins,
                "total_matches": len(matches)
            },
            "matches": match_list
        }
    except Exception as e:
        logger.error(f"Failed to fetch h2h: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/standings/{league_id}", tags=["Data"])
async def get_standings(
    league_id: int,
    season: int = Query(default=2024, description="Season year")
):
    """
    Get current league standings.
    
    Returns full table with points, goals, and form.
    """
    client = get_client()
    
    try:
        standings = await client.get_standings(
            league_id=league_id,
            season=season
        )
        
        return {
            "league_id": league_id,
            "league_name": LEAGUE_INFO.get(league_id, {}).get("name", "Unknown"),
            "season": season,
            "standings": standings
        }
    except Exception as e:
        logger.error(f"Failed to fetch standings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", tags=["Model"])
async def train_model(request: TrainRequest):
    """
    Train/retrain the prediction models.
    
    Fetches historical data and fits both Poisson and XGBoost models.
    This may take several minutes.
    """
    settings = get_settings()
    client = get_client()
    ensemble = get_ensemble()
    feature_eng = get_feature_engineer()
    
    league_ids = request.league_ids or settings.league_ids
    
    try:
        # Fetch historical matches
        all_matches = []
        for lid in league_ids:
            logger.info(f"Fetching matches for league {lid}...")
            matches = await client.get_finished_matches(
                league_id=lid,
                season=request.season
            )
            all_matches.extend(matches)
            logger.info(f"Got {len(matches)} matches for league {lid}")
        
        if len(all_matches) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough matches for training ({len(all_matches)}). Need at least 50."
            )
        
        # Load into feature engineer
        feature_eng.load_matches(all_matches)
        
        # Prepare training data
        import pandas as pd
        matches_df = pd.DataFrame([
            {
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "league_id": m.league_id
            }
            for m in all_matches
            if m.home_goals is not None and m.away_goals is not None
        ])
        
        # Train Poisson model
        logger.info("Training Poisson model...")
        ensemble.fit_poisson(matches_df)
        
        # Prepare features for XGBoost
        logger.info("Preparing features for XGBoost...")
        X, y = feature_eng.get_training_data(all_matches)
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        ensemble.fit_xgboost(X, y)
        
        # Save models to disk for persistence
        logger.info("Saving models to disk...")
        save_models(ensemble, feature_eng)
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "details": {
                "matches_used": len(all_matches),
                "leagues": league_ids,
                "season": request.season,
                "models": ensemble.get_model_status()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-current", tags=["Model"])
async def train_model_current_season(background_tasks: BackgroundTasks):
    """
    Train models using current season (2024-2025) data from football-data.org.
    
    Uses football-data.org API which provides FREE access to current season.
    Trains on all 5 top European leagues.
    
    Returns immediately and trains in background.
    Just click Execute - no parameters needed!
    """
    settings = get_settings()
    
    # Return immediately, train in background
    background_tasks.add_task(train_models_background, settings)
    
    return {
        "status": "started",
        "message": "Model training started in background. Check /api/health to see when complete.",
        "leagues": settings.league_ids,
        "seasons": [2025, 2024]
    }


async def train_models_background(settings):
    """Background task for model training."""
    global _training_in_progress
    import asyncio
    
    _training_in_progress = True
    print("🚀 TRAINING STARTED IN BACKGROUND")
    
    fd_client = get_fd_client()
    ensemble = get_ensemble()
    feature_eng = get_feature_engineer()
    
    # Always use all configured leagues
    target_leagues = settings.league_ids
    
    # Determine current season based on date (season starts in August)
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_season = current_year if current_month >= 8 else current_year - 1
    
    # Use only football-data.org (API-Football account is suspended)
    # Fetch 2 seasons: current and previous
    fd_seasons = [current_season, current_season - 1]  # 2025, 2024
    
    try:
        all_matches = []
        
        # Fetch from football-data.org only
        for season in fd_seasons:
            for lid in target_leagues:
                try:
                    print(f"🔍 [FD.org] Fetching season {season} for league {lid}...")
                    logger.info(f"[football-data.org] Fetching season {season} for league {lid}...")
                    fd_matches = await fd_client.get_finished_matches(
                        league_id=lid,
                        season=season
                    )
                    
                    print(f"✅ [FD.org] Received {len(fd_matches)} matches for league {lid}, season {season}")
                    logger.info(f"[football-data.org] Received {len(fd_matches)} matches for league {lid}, season {season}")
                    
                    # Convert FDMatch to Match format
                    converted_count = 0
                    for fm in fd_matches:
                        if fm.home_score is not None and fm.away_score is not None:
                            match = Match(
                                fixture_id=fm.match_id,
                                league_id=fm.competition_id,
                                league_name=fm.competition_name,
                                home_team_id=fm.home_team_id,
                                home_team_name=fm.home_team_name,
                                away_team_id=fm.away_team_id,
                                away_team_name=fm.away_team_name,
                                date=fm.utc_date,
                                home_goals=fm.home_score,
                                away_goals=fm.away_score,
                                status="FT"
                            )
                            all_matches.append(match)
                            converted_count += 1
                    
                    print(f"💾 [FD.org] Converted {converted_count} matches for league {lid}, season {season}")
                    logger.info(f"[football-data.org] Converted {converted_count} matches for league {lid}, season {season}")
                    
                    # Rate limit: 10 requests per minute = 1 every 6 seconds
                    # Add small delay to be safe
                    await asyncio.sleep(7)
                except Exception as e:
                    print(f"❌ [FD.org] FAILED league {lid}, season {season}: {e}")
                    logger.error(f"[football-data.org] Failed to fetch league {lid}, season {season}: {e}")
                    # Continue with other leagues even if one fails
                    continue
        
        print(f"📊 TOTAL matches collected: {len(all_matches)}")
        logger.info(f"Total matches collected: {len(all_matches)}")
        
        if len(all_matches) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough matches for training ({len(all_matches)}). Need at least 50. Make sure football-data.org API key is configured."
            )
        
        # Load into feature engineer
        feature_eng.load_matches(all_matches)
        
        # Prepare training data
        import pandas as pd
        matches_df = pd.DataFrame([
            {
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "league_id": m.league_id
            }
            for m in all_matches
            if m.home_goals is not None and m.away_goals is not None
        ])
        
        # Train Poisson model
        print("🎯 Training Poisson model...")
        logger.info("Training Poisson model...")
        ensemble.fit_poisson(matches_df)
        
        # Prepare features for XGBoost
        print("🔧 Preparing features for XGBoost...")
        logger.info("Preparing features for XGBoost...")
        X, y = feature_eng.get_training_data(all_matches)
        
        # Train XGBoost
        print("🚀 Training XGBoost model...")
        logger.info("Training XGBoost model...")
        ensemble.fit_xgboost(X, y)
        
        # Save models to disk
        print("💾 Saving models to disk...")
        logger.info("Saving models to disk...")
        save_models(ensemble, feature_eng)
        
        print(f"✅ TRAINING COMPLETE! Models saved with {len(all_matches)} matches")
        logger.info(f"Training complete with {len(all_matches)} matches")
    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
        logger.error(f"Training failed: {e}")
    finally:
        _training_in_progress = False
        print(f"🏁 Training finished. Status: training_in_progress={_training_in_progress}")


@router.get("/upcoming-matches", tags=["Data"])
async def get_upcoming_matches(
    league_ids: Optional[str] = Query(
        default=None,
        description="Comma-separated league IDs (e.g., '39,140,135'). If not provided, uses all configured leagues."
    ),
    days_ahead: int = Query(
        default=7,
        ge=1,
        le=30,
        description="Number of days to look ahead (not used with football-data.org)"
    )
):
    """
    Get all upcoming matches for specified leagues.
    
    Uses football-data.org API for current season data (free tier supports current season).
    Returns scheduled matches that haven't been played yet.
    
    Example: /api/upcoming-matches?league_ids=39,140&days_ahead=7
    """
    settings = get_settings()
    fd_client = get_fd_client()
    
    # Parse league IDs
    if league_ids:
        target_leagues = [int(lid.strip()) for lid in league_ids.split(",")]
    else:
        target_leagues = settings.league_ids
    
    try:
        all_upcoming = []
        
        for league_id in target_leagues:
            logger.info(f"Fetching upcoming matches for league {league_id} from football-data.org...")
            
            # Use football-data.org for upcoming matches (supports current season)
            matches = await fd_client.get_upcoming_matches(
                league_id=league_id,
                limit=20  # Get up to 20 upcoming matches per league
            )
            
            for match in matches:
                all_upcoming.append({
                    "fixture_id": match.match_id,
                    "date": match.utc_date.isoformat(),
                    "league": {
                        "id": match.competition_id,
                        "name": match.competition_name
                    },
                    "home_team": {
                        "id": match.home_team_id,
                        "name": match.home_team_name
                    },
                    "away_team": {
                        "id": match.away_team_id,
                        "name": match.away_team_name
                    },
                    "status": match.status
                })
            
            logger.info(f"Found {len(matches)} upcoming matches for league {league_id}")
        
        # Sort by date
        all_upcoming.sort(key=lambda x: x["date"])
        
        return {
            "count": len(all_upcoming),
            "leagues": [
                {
                    "id": lid,
                    "name": LEAGUE_INFO.get(lid, {}).get("name", "Unknown")
                }
                for lid in target_leagues
            ],
            "days_ahead": days_ahead,
            "matches": all_upcoming
        }
    except Exception as e:
        logger.error(f"Failed to fetch upcoming matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/odds/{fixture_id}", tags=["Data"])
async def get_fixture_odds(fixture_id: int):
    """
    Get betting odds for a specific fixture.
    
    Returns odds from multiple bookmakers.
    """
    client = get_client()
    
    try:
        odds = await client.get_odds(fixture_id=fixture_id)
        
        if not odds:
            raise HTTPException(
                status_code=404,
                detail=f"No odds found for fixture {fixture_id}"
            )
        
        return {
            "fixture_id": fixture_id,
            "odds": [
                {
                    "bookmaker": o.bookmaker,
                    "home_win": o.home_win,
                    "draw": o.draw,
                    "away_win": o.away_win,
                    "updated_at": o.updated_at.isoformat()
                }
                for o in odds
            ],
            "count": len(odds)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        raise HTTPException(status_code=500, detail=str(e))
