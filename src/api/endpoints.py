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
from ..data.odds_api_client import get_odds_api_client, normalize_team
from ..features.feature_engineering import FeatureEngineer, MatchFeatures
from ..models.ensemble import EnsemblePredictor
from ..models.xgboost_model import XGBoostPredictor
from ..betting.value_bet import ValueBetAnalyzer
from ..data.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances (will be initialized on startup)
_ensemble: Optional[EnsemblePredictor] = None
_feature_engineer: Optional[FeatureEngineer] = None
_value_analyzer: Optional[ValueBetAnalyzer] = None

# Model save paths
import os
import pickle
from pathlib import Path

MODEL_DIR = Path("models")
POISSON_PATH = MODEL_DIR / "poisson_model.pkl"
XGBOOST_PATH = MODEL_DIR / "xgboost_model.pkl"
FEATURE_ENG_PATH = MODEL_DIR / "feature_engineer.pkl"
TRAINING_LOCK_PATH = MODEL_DIR / "training.lock"


def is_training() -> bool:
    """Check if training is in progress across all workers."""
    return TRAINING_LOCK_PATH.exists()


def set_training_status(status: bool) -> None:
    """Set training status across all workers."""
    MODEL_DIR.mkdir(exist_ok=True)
    if status:
        TRAINING_LOCK_PATH.touch()
    else:
        if TRAINING_LOCK_PATH.exists():
            TRAINING_LOCK_PATH.unlink()


def save_models(ensemble: EnsemblePredictor, feature_eng: FeatureEngineer) -> None:
    """Save trained models to disk."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Save Poisson model (pickle is fine as it's a simple dataclass-like object)
    with open(POISSON_PATH, "wb") as f:
        pickle.dump(ensemble.poisson, f)
    
    # Save XGBoost model using its native save method (handles state dict properly)
    # The .save() method automatically adds .pkl extension, so we strip it if present
    xgb_path_str = str(XGBOOST_PATH)
    if xgb_path_str.endswith('.pkl'):
        xgb_path_str = xgb_path_str[:-4]
    ensemble.xgboost.save(xgb_path_str)
    
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
        
        # Load XGBoost using native load method
        # We need to instantiate a fresh predictor then load state into it
        # This prevents the 'dict has no attribute is_fitted' bug
        xgb_path_str = str(XGBOOST_PATH)
        if xgb_path_str.endswith('.pkl'):
            xgb_path_str = xgb_path_str[:-4]
            
        ensemble.xgboost = XGBoostPredictor()
        ensemble.xgboost.load(xgb_path_str)
        
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
    elif not _ensemble.poisson.is_fitted or not _ensemble.xgboost.is_fitted:
        # If not fitted, try to reload (maybe another worker just finished training)
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
                
                # Migration for old models
                if not hasattr(_feature_engineer, "season_decay"):
                    _feature_engineer.season_decay = 0.85
                if not hasattr(_feature_engineer, "_elo_k_factor"):
                    _feature_engineer._elo_k_factor = 32
                if not hasattr(_feature_engineer, "_elo_home_advantage"):
                    _feature_engineer._elo_home_advantage = 50
                    
                logger.info("Feature engineer loaded from disk")
            except Exception as e:
                logger.warning(f"Failed to load feature engineer: {e}")
                _feature_engineer = FeatureEngineer(form_matches=settings.form_matches)
        else:
            _feature_engineer = FeatureEngineer(form_matches=settings.form_matches)
    elif _feature_engineer._matches_df is None and FEATURE_ENG_PATH.exists():
        # If not loaded, try to reload
        try:
            with open(FEATURE_ENG_PATH, "rb") as f:
                _feature_engineer = pickle.load(f)
            
            # Migration for reloaded model
            if not hasattr(_feature_engineer, "season_decay"):
                _feature_engineer.season_decay = 0.85
            if not hasattr(_feature_engineer, "_elo_k_factor"):
                _feature_engineer._elo_k_factor = 32
            if not hasattr(_feature_engineer, "_elo_home_advantage"):
                _feature_engineer._elo_home_advantage = 50

            logger.info("Feature engineer reloaded from disk")
        except Exception as e:
            logger.warning(f"Failed to reload feature engineer: {e}")
            
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
        default=2025,
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
    feature_engineer: Optional[Dict[str, Any]] = None
    api_configured: bool
    training_in_progress: bool = False
    timestamp: str


# ============== Endpoints ==============

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health and model status.
    
    Returns model readiness and API configuration status.
    """
    settings = get_settings()
    ensemble = get_ensemble()
    
    # Feature Engineer stats for debugging production data
    feature_eng = get_feature_engineer()
    fe_stats = {
        "loaded_matches": len(feature_eng._matches_df) if feature_eng._matches_df is not None else 0,
        "elo_ratings_count": len(feature_eng._elo_ratings),
        "season_decay": getattr(feature_eng, "season_decay", "missing"),
        "k_factor": getattr(feature_eng, "_elo_k_factor", "missing"),
        "leagues_data": feature_eng._matches_df["league_id"].value_counts().to_dict() if feature_eng._matches_df is not None and not feature_eng._matches_df.empty else {}
    }
    
    return {
        "status": "healthy",
        "models": ensemble.get_model_status(),
        "feature_engineer": fe_stats,
        "api_configured": bool(settings.api_football_key),
        "training_in_progress": is_training(),
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
        ensemble = get_ensemble()
        
        if not ensemble.poisson.is_fitted:
            if is_training():
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
        
        # Fetch ranks for display
        fd_client = get_fd_client()
        standings = await fd_client.get_standings(league_id=request.league_id)
        ranks = {s["team_id"]: s["rank"] for s in standings}
        
        result = prediction.to_dict()
        result["match"]["home_rank"] = ranks.get(request.home_team_id)
        result["match"]["away_rank"] = ranks.get(request.away_team_id)
        
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


async def _get_predictions_core(
    league_ids: str = "39",
    days: int = 2
):
    """
    Get upcoming matches with dynamic Poisson predictions and real-time odds for multiple leagues.
    Internal core function used by multiple endpoints.
    """
    # Try cache first
    from ..data.cache_client import get_cache_client
    cache = get_cache_client()
    
    cached = await cache.get_predictions(league_ids, days)
    if cached:
        logger.info(f"Cache HIT for predictions: leagues={league_ids}, days={days}")
        return cached
    
    logger.info(f"Cache MISS for predictions: leagues={league_ids}, days={days}")
    
    fd_client = get_fd_client()
    odds_client = get_odds_api_client()
    ensemble = get_ensemble()
    feature_eng = get_feature_engineer()
    
    try:
        ids = [int(i.strip()) for i in league_ids.split(",") if i.strip()]
        
        # 1. Fetch real-time odds (also check cache)
        all_league_odds = await cache.get_odds(league_ids)
        if not all_league_odds:
            all_league_odds = await odds_client.get_all_league_odds(ids)
            await cache.set_odds(league_ids, all_league_odds)
            logger.info(f"Fetched and cached odds for {league_ids}")
        else:
            logger.info(f"Using cached odds for {league_ids}")
        
        # Determine current season
        from datetime import datetime
        current_year = datetime.now().year
        season = current_year if datetime.now().month >= 8 else current_year - 1
        
        all_matches = []
        all_upcoming_fixtures = []
        standings_map = {}
        
        for lid in ids:
            # 1. Get upcoming fixtures from Football-Data.org (Reliable)
            fd_fixtures = await fd_client.get_upcoming_matches(league_id=lid, days_ahead=days)
            if fd_fixtures:
                all_upcoming_fixtures.extend(fd_fixtures)
            
            # 2. Get finished matches for team strength from Football-Data.org
            finished_matches = await fd_client.get_finished_matches(league_id=lid, season=season)
            
            # 3. Get current standings
            standings = await fd_client.get_standings(league_id=lid, season=season)
            standings_map[lid] = standings
            
            # Load finished matches into feature engineer
            try:
                feature_eng.load_matches(finished_matches)
            except Exception as e:
                logger.warning(f"Failed to load matches into feature engineer for league {lid}: {e}")
        
        for f in all_upcoming_fixtures:
            lid = getattr(f, 'league_id', getattr(f, 'competition_id', None))
            standings = standings_map.get(lid)
            
            # Predict
            try:
                home_rank = next((s["rank"] for s in standings if s["team_id"] == f.home_team_id), None) if standings else None
                away_rank = next((s["rank"] for s in standings if s["team_id"] == f.away_team_id), None) if standings else None
                
                # Build feature object
                features = None
                try:
                    features = feature_eng.calculate_features(f, None, standings, for_prediction=True)
                except Exception as e:
                    logger.warning(f"Failed to calculate features for fixture {f.match_id}: {e}")
                
                # Get recent matches for Poisson dynamic mode
                home_last_10 = feature_eng.get_last_n_matches(f.home_team_id, n=10)
                away_last_10 = feature_eng.get_last_n_matches(f.away_team_id, n=10)
                
                prediction = ensemble.predict(
                    home_team_id=f.home_team_id,
                    away_team_id=f.away_team_id,
                    home_team_name=f.home_team_name,
                    away_team_name=f.away_team_name,
                    league_id=lid,
                    features=features,
                    home_last_5=home_last_10,  # Using argument name from ensemble.py but passing 10
                    away_last_5=away_last_10
                )
                
                poisson_pred = ensemble.poisson.predict(f.home_team_id, [], f.away_team_id, [])
                
                # Get real odds from the pre-fetched all_league_odds
                odds_1x2 = {"1": 1.0, "X": 1.0, "2": 1.0}
                ou_odds = {"over": 1.0, "under": 1.0}
                btts_odds = {"yes": 1.0, "no": 1.0}
                
                # Match odds from odds-api using dictionary key lookup
                home_norm = normalize_team(f.home_team_name)
                away_norm = normalize_team(f.away_team_name)
                
                # NOTE: Key format must match odds_api_client.get_all_league_odds
                match_key = f"{home_norm}_vs_{away_norm}".lower()
                match_key_reverse = f"{away_norm}_vs_{home_norm}".lower()
                
                match_odds = all_league_odds.get(match_key) or all_league_odds.get(match_key_reverse)
                
                if match_odds:
                    if match_odds.get("1x2"):
                        odds_1x2 = match_odds["1x2"]
                    if match_odds.get("ou_2.5"):
                        ou_odds = match_odds["ou_2.5"]
                    if match_odds.get("btts"):
                        btts_odds = match_odds["btts"]
                    logger.info(f"✓ Matched odds for {f.home_team_name} vs {f.away_team_name}")
                else:
                    # Log first 5 keys in all_league_odds to see what we have
                    available_keys = list(all_league_odds.keys())[:5]
                    logger.warning(
                        f"✗ No odds found for {f.home_team_name} vs {f.away_team_name}. "
                        f"Keys tried: ['{match_key}', '{match_key_reverse}']. "
                        f"Available keys ({len(all_league_odds)} total): {available_keys}"
                    )

                match_result = {
                    "fixture_id": f.match_id,
                    "teams": {
                        "home": f.home_team_name, 
                        "away": f.away_team_name,
                        "home_elo": round(prediction.home_elo, 1) if prediction.home_elo else None,
                        "away_elo": round(prediction.away_elo, 1) if prediction.away_elo else None,
                        "home_rank": home_rank,
                        "away_rank": away_rank
                    },
                    "date": f.utc_date.isoformat() if hasattr(f.utc_date, 'isoformat') else f.utc_date,
                    "league_id": lid,
                    "predictions": {
                        "1x2": {
                            "1": round(prediction.home_win_pct / 100, 4),
                            "X": round(prediction.draw_pct / 100, 4),
                            "2": round(prediction.away_win_pct / 100, 4)
                        },
                        "goals": {
                            "over_2.5": round(poisson_pred.over_25_prob, 4),
                            "under_2.5": round(poisson_pred.under_25_prob, 4)
                        },
                        "btts": {
                            "yes": round(poisson_pred.btts_yes_prob, 4),
                            "no": round(poisson_pred.btts_no_prob, 4)
                        }
                    },
                    "odds": {
                        "1x2": odds_1x2,
                        "ou_2.5": ou_odds,
                        "btts": btts_odds,
                        "double_chance": match_odds.get("double_chance", {}) if match_odds else {}
                    },
                    "ev": {}
                }
                
                # EV calculation
                prob_home = prediction.home_win_pct / 100
                prob_draw = prediction.draw_pct / 100
                prob_away = prediction.away_win_pct / 100
                
                match_result["ev"] = {
                    "1": round(prob_home * odds_1x2.get("1", 0), 2),
                    "X": round(prob_draw * odds_1x2.get("X", 0), 2),
                    "2": round(prob_away * odds_1x2.get("2", 0), 2),
                }
                
                prob_1x = prob_home + prob_draw
                prob_x2 = prob_draw + prob_away
                prob_12 = prob_home + prob_away
                
                match_result["double_chance"] = {
                    "1X": round(prob_1x, 4),
                    "X2": round(prob_x2, 4),
                    "12": round(prob_12, 4)
                }
                
                match_result["fair_odds"] = {
                    "1x2": {
                        "1": round(1.0 / max(0.01, prob_home), 2),
                        "X": round(1.0 / max(0.01, prob_draw), 2),
                        "2": round(1.0 / max(0.01, prob_away), 2)
                    },
                    "double_chance": {
                        "1X": round(1.0 / max(0.01, prob_1x), 2),
                        "X2": round(1.0 / max(0.01, prob_x2), 2),
                        "12": round(1.0 / max(0.01, prob_12), 2)
                    }
                }
                
                import math
                p_safe = [max(0.001, prob_home), max(0.001, prob_draw), max(0.001, prob_away)]
                entropy = -sum(p * math.log(p) for p in p_safe)
                
                p_poisson = [poisson_pred.home_win_prob, poisson_pred.draw_prob, poisson_pred.away_win_prob]
                p_xgb = [prob_home, prob_draw, prob_away]
                disagreement = max(abs(p - x) for p, x in zip(p_poisson, p_xgb))
                
                max_prob = max(prob_home, prob_draw, prob_away)
                action = "BET"
                if entropy > 1.05 or disagreement > 0.20 or max_prob < 0.45:
                    action = "SKIP"
                    
                match_result["advisor"] = {
                    "action": action,
                    "entropy": round(entropy, 4),
                    "model_disagreement": round(disagreement, 4),
                    "max_probability": round(max_prob, 4)
                }

                all_matches.append(match_result)
            except Exception as fixture_error:
                logger.error(f"Error processing fixture {f.match_id}: {fixture_error}")
                continue
        
        # Cache the result before returning
        result = {"matches": all_matches}
        await cache.set_predictions(league_ids, days, result)
        logger.info(f"Cached predictions for leagues={league_ids}, days={days}")
        
        return result
    except Exception as e:
        logger.error(f"Predictions core failed: {e}")
        return {"matches": [], "error": str(e)}


@router.get("/upcoming-matches-with-predictions", tags=["Predictions"])
async def get_upcoming_matches_with_predictions(
    league_ids: str = Query(default="39", description="Comma-separated League IDs"),
    days: int = Query(default=2, ge=1, le=14, description="Days ahead")
):
    """
    Get upcoming matches with ensemble predictions and value bet calculation.
    """
    result = await _get_predictions_core(league_ids=league_ids, days=days)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/upcoming-matches", tags=["Predictions"])
async def get_upcoming_matches_v1(
    league_ids: Optional[str] = Query(default=None, description="Comma-separated League IDs"),
    days: int = Query(default=2, ge=1, le=14, description="Days ahead")
):
    """Get upcoming matches with ensemble predictions."""
    # Use all configured leagues if none provided
    if not league_ids:
        settings = get_settings()
        league_ids = ",".join(map(str, settings.league_ids))
        
    result = await _get_predictions_core(league_ids=league_ids, days=days)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@router.get("/v1/recommendations/safe-accumulator", tags=["Recommendations"])
async def get_safe_accumulator(
    league_ids: Optional[str] = Query(default=None, description="Comma-separated League IDs"),
    days: int = Query(default=7, ge=1, le=14, description="Days ahead (default 7 for next matchday)")
):
    """
    Generate a 'Safe' Accumulator betting slip with value-weighted scoring.
    
    Scoring considers:
    - Probability (safety/confidence)
    - Real odds (return potential)
    - Edge vs fair odds (value)
    
    A pick at 1.50 with 80% confidence is preferred over 1.03 with 82% confidence
    because the value-adjusted score is higher.
    """
    # Use all configured leagues if none provided
    if not league_ids:
        settings = get_settings()
        league_ids = ",".join(map(str, settings.league_ids))
        
    result = await _get_predictions_core(league_ids=league_ids, days=days)
    matches = result.get("matches", [])
    
    candidates = []
    for m in matches:
        # Filter: No SKIP matches
        if m.get("advisor", {}).get("action") == "SKIP":
            continue
            
        probs = m["predictions"]["1x2"]
        dc = m["double_chance"]
        real_odds = m.get("odds", {}).get("1x2", {})
        dc_odds = m.get("odds", {}).get("double_chance", {})
        fair_odds = m.get("fair_odds", {})
        confidence = m.get("confidence", "LOW")
        
        # Confidence multiplier: HIGH=1.2, MEDIUM=1.0, LOW=0.8
        conf_mult = {"HIGH": 1.2, "MEDIUM": 1.0, "LOW": 0.8}.get(confidence, 1.0)
        
        # Evaluate all possible selections and pick the best one
        possible_picks = []
        
        # 1X2 markets
        for sel, prob in [("1", probs.get("1", 0)), ("X", probs.get("X", 0)), ("2", probs.get("2", 0))]:
            if prob >= 0.55:  # Minimum 55% probability threshold
                real_odd = real_odds.get(sel, 1.0)
                fair_odd = fair_odds.get("1x2", {}).get(sel, 1.0)
                
                # Skip if no real odds available (fallback = 1.0)
                if real_odd <= 1.01:
                    continue
                    
                # Calculate edge: positive means real odds are better than fair
                edge = (real_odd / fair_odd) - 1 if fair_odd > 0 else 0
                
                # VALUE SCORE = probability * ln(odds) * confidence * (1 + edge)
                # Using ln(odds) gives more weight to higher odds while keeping it balanced
                import math
                value_score = prob * math.log(real_odd + 1) * conf_mult * (1 + max(0, edge))
                
                possible_picks.append({
                    "selection": sel,
                    "probability": prob,
                    "real_odd": real_odd,
                    "fair_odd": fair_odd,
                    "edge": edge,
                    "value_score": value_score,
                    "market": "1x2"
                })
        
        # Double Chance markets
        for sel, prob in [("1X", dc.get("1X", 0)), ("X2", dc.get("X2", 0)), ("12", dc.get("12", 0))]:
            if prob >= 0.75:  # Higher threshold for DC (should be very safe)
                real_odd = dc_odds.get(sel, 1.0)
                fair_odd = fair_odds.get("double_chance", {}).get(sel, 1.0)
                
                if real_odd <= 1.01:
                    continue
                    
                edge = (real_odd / fair_odd) - 1 if fair_odd > 0 else 0
                import math
                value_score = prob * math.log(real_odd + 1) * conf_mult * (1 + max(0, edge))
                
                possible_picks.append({
                    "selection": sel,
                    "probability": prob,
                    "real_odd": real_odd,
                    "fair_odd": fair_odd,
                    "edge": edge,
                    "value_score": value_score,
                    "market": "double_chance"
                })
        
        # Pick the best selection for this match (highest value score)
        if possible_picks:
            best = max(possible_picks, key=lambda x: x["value_score"])
            candidates.append({
                "fixture_id": m["fixture_id"],
                "teams": m["teams"],
                "selection": best["selection"],
                "probability": best["probability"],
                "real_odd": best["real_odd"],
                "fair_odd": best["fair_odd"],
                "edge": round(best["edge"] * 100, 1),  # As percentage
                "value_score": round(best["value_score"], 3),
                "confidence": confidence
            })
    
    # Sort by value_score descending (not just probability!)
    candidates.sort(key=lambda x: x["value_score"], reverse=True)
    top_4 = candidates[:4]
    
    if not top_4:
        return {"count": 0, "message": "No safe candidates found for the next matchday", "selections": []}
    
    # Calculate totals
    total_p = 1.0
    total_real_odd = 1.0
    total_fair_odd = 1.0
    for c in top_4:
        total_p *= c["probability"]
        total_real_odd *= c["real_odd"]
        total_fair_odd *= c["fair_odd"]
    
    return {
        "count": len(top_4),
        "total_probability": round(total_p, 4),
        "estimated_total_odd": round(total_fair_odd, 2),
        "real_total_odd": round(total_real_odd, 2),
        "avg_edge": round(sum(c["edge"] for c in top_4) / len(top_4), 1),
        "selections": top_4
    }

class BetSelection(BaseModel):
    fixture_id: int
    selection: str
    odds: float
    market: str = "Match Winner"
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    league_name: Optional[str] = None

class PlaceBetRequest(BaseModel):
    stake: float
    selections: List[BetSelection]

@router.post("/bets/place", tags=["Betting"])
async def place_bet(request: PlaceBetRequest):
    """
    Save a new bet to Supabase.
    """
    try:
        sb = get_supabase_client()
        total_odds = 1.0
        for s in request.selections:
            total_odds *= s.odds
        
        potential_win = request.stake * total_odds
        
        bet_id = sb.save_bet(
            stake=request.stake,
            total_odds=total_odds,
            potential_win=potential_win,
            selections=[s.model_dump() for s in request.selections]
        )
        
        return {"status": "success", "bet_id": bet_id, "potential_win": round(potential_win, 2)}
    except Exception as e:
        logger.error(f"Failed to place bet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bets", tags=["Betting"])
async def get_bets(limit: int = 50):
    """
    Get betting history from Supabase.
    """
    try:
        sb = get_supabase_client()
        bets = sb.get_all_bets(limit=limit)
        return {"bets": bets}
    except Exception as e:
        logger.error(f"Failed to fetch bets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bets/sync", tags=["Betting"])
async def sync_bet_results():
    """
    Sync pending bet results from Football-Data.org.
    """
    try:
        sb = get_supabase_client()
        fd_client = get_fd_client()
        
        pending_selections = sb.get_pending_selections()
        if not pending_selections:
            return {"status": "success", "message": "No pending bets to sync"}
            
        # Group selections by fixture_id to minimize API calls
        from collections import defaultdict
        fixture_to_selections = defaultdict(list)
        for s in pending_selections:
            fixture_to_selections[s["fixture_id"]].append(s)
            
        updated_count = 0
        errors = []
        
        import asyncio
        for fixture_id, selections in fixture_to_selections.items():
            try:
                # Fetch match details from Football-Data.org
                match = await fd_client.get_match(fixture_id)
                
                if not match:
                    logger.warning(f"Could not find match {fixture_id} on Football-Data.org")
                    continue
                    
                # We only sync finished matches
                if match.status not in ["FINISHED", "FT"]:
                    continue
                    
                home_goals = match.home_score
                away_goals = match.away_score
                
                if home_goals is None or away_goals is None:
                    continue
                    
                total_goals = home_goals + away_goals
                result_str = f"{home_goals}-{away_goals}"
                
                for s in selections:
                    market = (s.get("market") or "Match Winner").lower()
                    selection = s["selection"]
                    status = "lost"
                    
                    # 1X2 market
                    if market in ["match winner", "1x2"]:
                        winner = "1" if home_goals > away_goals else ("2" if home_goals < away_goals else "X")
                        # Handle different labeling
                        if (selection in ["1", "Home"] and winner == "1") or \
                           (selection in ["X", "Draw"] and winner == "X") or \
                           (selection in ["2", "Away"] and winner == "2"):
                            status = "won"
                    
                    # Over/Under 2.5 market
                    elif market in ["ou_2.5", "over/under 2.5"]:
                        if (selection.lower() == "over" and total_goals > 2.5) or \
                           (selection.lower() == "under" and total_goals < 2.5):
                            status = "won"
                            
                    # BTTS market
                    elif market in ["btts", "both teams to score"]:
                        has_btts = home_goals > 0 and away_goals > 0
                        if (selection.lower() == "yes" and has_btts) or \
                           (selection.lower() == "no" and not has_btts):
                            status = "won"
                    
                    # Update the selection in database
                    sb.update_selection_result(s["id"], status, status, result_str)
                    updated_count += 1
                
                # Small delay for rate limiting (10 req/min)
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to sync fixture {fixture_id}: {e}")
                errors.append(f"Fixture {fixture_id}: {str(e)}")
        
        # After updating selections, we should check if any bets are fully resolved
        # (This could be done in a separate trigger or function, but adding here for simplicity)
        # Note: sb.get_all_bets fetch also includes selections, so we can check those in the frontend
        # or add a backend cleanup here if needed.
        
        return {
            "status": "success", 
            "updated_selections": updated_count,
            "errors": errors if errors else None
        }
    except Exception as e:
        logger.error(f"Sync process failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    season: int = Query(default=2025, description="Season year")
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
    season: int = Query(default=2025, description="Season year")
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
    import asyncio
    
    set_training_status(True)
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
    # Fetch 3 seasons: current and two previous
    fd_seasons = [current_season, current_season - 1, current_season - 2]  # 2025, 2024, 2023
    
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
        await asyncio.to_thread(ensemble.fit_poisson, matches_df)
        
        # Prepare features for XGBoost
        print("🔧 Preparing features for XGBoost...")
        logger.info("Preparing features for XGBoost...")
        X, y = await asyncio.to_thread(feature_eng.get_training_data, all_matches)
        
        # Train XGBoost
        print("🚀 Training XGBoost model...")
        logger.info("Training XGBoost model...")
        await asyncio.to_thread(ensemble.fit_xgboost, X, y)
        
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
        set_training_status(False)
        print(f"🏁 Training finished. Status: training_in_progress={is_training()}")


# Removed duplicate /upcoming-matches endpoint to avoid conflicts with the prediction-enabled one.


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
