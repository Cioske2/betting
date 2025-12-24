"""
Ensemble Model for Football Match Prediction.

Combines Poisson and XGBoost predictions with configurable weights.
Provides unified interface for match predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from .poisson_model import PoissonModel, PoissonPrediction
from .xgboost_model import XGBoostPredictor, XGBPrediction
from ..features.feature_engineering import MatchFeatures

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """
    Combined prediction from ensemble model.
    
    Contains probabilities, confidence levels, and model-specific details.
    """
    # Match info
    home_team: str
    away_team: str
    league_id: int
    
    # Final probabilities (percentage)
    home_win_pct: float
    draw_pct: float
    away_win_pct: float
    
    # Individual model outputs
    poisson_probs: Dict[str, float]
    xgboost_probs: Dict[str, float]
    
    # Additional insights
    expected_home_goals: float
    expected_away_goals: float
    confidence: str  # "LOW", "MEDIUM", "HIGH"
    prediction_time: datetime
    
    # Most likely outcome
    predicted_outcome: str  # "1", "X", "2"
    
    # ELO and Standings
    home_elo: Optional[float] = None
    away_elo: Optional[float] = None
    home_rank: Optional[int] = None
    away_rank: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "match": {
                "home_team": self.home_team,
                "away_team": self.away_team,
                "league_id": self.league_id,
                "home_elo": round(self.home_elo, 1) if self.home_elo else None,
                "away_elo": round(self.away_elo, 1) if self.away_elo else None,
                "home_rank": self.home_rank,
                "away_rank": self.away_rank,
            },
            "probabilities": {
                "home_win": round(self.home_win_pct, 2),
                "draw": round(self.draw_pct, 2),
                "away_win": round(self.away_win_pct, 2),
            },
            "model_details": {
                "poisson": self.poisson_probs,
                "xgboost": self.xgboost_probs,
            },
            "expected_goals": {
                "home": round(self.expected_home_goals, 2),
                "away": round(self.expected_away_goals, 2),
                "total": round(self.expected_home_goals + self.expected_away_goals, 2),
            },
            "prediction": self.predicted_outcome,
            "confidence": self.confidence,
            "timestamp": self.prediction_time.isoformat(),
        }
    
    def to_percentage_string(self) -> str:
        """Get formatted percentage string."""
        return f"1: {self.home_win_pct:.1f}% | X: {self.draw_pct:.1f}% | 2: {self.away_win_pct:.1f}%"


class EnsemblePredictor:
    """
    Combines Poisson and XGBoost models for robust predictions.
    
    Features:
    - Weighted average of model predictions
    - Confidence scoring based on model agreement
    - Fallback to single model if other fails
    - Unified prediction interface
    """
    
    def __init__(
        self,
        poisson_weight: float = 0.4,
        xgboost_weight: float = 0.6,
        poisson_model: Optional[PoissonModel] = None,
        xgboost_model: Optional[XGBoostPredictor] = None
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            poisson_weight: Weight for Poisson model (0-1)
            xgboost_weight: Weight for XGBoost model (0-1)
            poisson_model: Pre-fitted Poisson model
            xgboost_model: Pre-fitted XGBoost model
        """
        # Normalize weights
        total = poisson_weight + xgboost_weight
        self.poisson_weight = poisson_weight / total
        self.xgboost_weight = xgboost_weight / total
        
        self.poisson = poisson_model or PoissonModel()
        self.xgboost = xgboost_model or XGBoostPredictor()
        
        logger.info(
            f"Ensemble initialized with weights: "
            f"Poisson={self.poisson_weight:.2f}, XGBoost={self.xgboost_weight:.2f}"
        )
    
    def fit_poisson(self, matches_df: pd.DataFrame) -> None:
        """
        Fit the Poisson model.
        
        Args:
            matches_df: DataFrame with historical matches
        """
        self.poisson.fit(matches_df)
    
    def fit_xgboost(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        self.xgboost.train(X, y)
    
    def _calculate_confidence(
        self,
        poisson_probs: Dict[str, float],
        xgb_probs: Dict[str, float],
        final_probs: np.ndarray
    ) -> str:
        """
        Calculate confidence level based on model agreement.
        
        High confidence if:
        - Both models agree on the most likely outcome
        - The probability difference between them is small
        - The winning probability is high
        
        Args:
            poisson_probs: Poisson model probabilities
            xgb_probs: XGBoost model probabilities
            final_probs: Final ensemble probabilities
            
        Returns:
            Confidence level: "LOW", "MEDIUM", "HIGH"
        """
        # Get predicted outcomes
        poisson_pred = max(poisson_probs, key=poisson_probs.get)
        xgb_pred = max(xgb_probs, key=xgb_probs.get)
        
        # Check if models agree
        models_agree = poisson_pred == xgb_pred
        
        # Maximum probability
        max_prob = np.max(final_probs)
        
        # Probability difference between models for top outcome
        prob_diff = abs(poisson_probs[poisson_pred] - xgb_probs.get(poisson_pred, 0))
        
        if models_agree and max_prob > 0.50 and prob_diff < 0.10:
            return "HIGH"
        elif models_agree and max_prob > 0.40:
            return "MEDIUM"
        elif max_prob > 0.45:
            return "MEDIUM"
        else:
            return "LOW"
    
    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
        home_team_name: str,
        away_team_name: str,
        league_id: int,
        features: Optional[MatchFeatures] = None,
        home_last_5: Optional[List[Dict]] = None,
        away_last_5: Optional[List[Dict]] = None
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction for a match.
        """
        poisson_probs = {"1": 0.33, "X": 0.33, "2": 0.34}
        xgb_probs = {"1": 0.33, "X": 0.33, "2": 0.34}
        expected_home = 1.5
        expected_away = 1.2
        
        # Get Poisson prediction
        try:
            # Poisson now requires last 5 matches
            poisson_pred = self.poisson.predict(
                home_team_id, home_last_5 or [],
                away_team_id, away_last_5 or []
            )
            data = poisson_pred.to_dict()
            poisson_probs = data["1x2"]
            expected_home = data["expected_goals"]["home"]
            expected_away = data["expected_goals"]["away"]
        except Exception as e:
            logger.warning(f"Poisson prediction failed: {e}")
        
        # Get XGBoost prediction
        if self.xgboost.is_fitted and features is not None:
            try:
                X = pd.DataFrame([features.to_dict()])[MatchFeatures.feature_names()]
                xgb_pred = self.xgboost.predict(X)
                xgb_probs = xgb_pred.to_dict()
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
        
        # Combine predictions
        final_probs = np.array([
            self.poisson_weight * poisson_probs["1"] + self.xgboost_weight * xgb_probs["1"],
            self.poisson_weight * poisson_probs["X"] + self.xgboost_weight * xgb_probs["X"],
            self.poisson_weight * poisson_probs["2"] + self.xgboost_weight * xgb_probs["2"],
        ])
        
        # Normalize
        final_probs = final_probs / final_probs.sum()
        
        # Determine predicted outcome
        outcomes = ["1", "X", "2"]
        predicted_outcome = outcomes[np.argmax(final_probs)]
        
        # Calculate confidence
        confidence = self._calculate_confidence(poisson_probs, xgb_probs, final_probs)
        
        # Extract ELO and Rank from features if available
        home_elo = features.home_elo if features else None
        away_elo = features.away_elo if features else None
        
        # Rank is tricky because position_diff = away_pos - home_pos
        # We don't have absolute ranks in features, but we can pass them if available
        # For now, let's just use ELO which we definitely have in features
        
        return EnsemblePrediction(
            home_team=home_team_name,
            away_team=away_team_name,
            league_id=league_id,
            home_win_pct=final_probs[0] * 100,
            draw_pct=final_probs[1] * 100,
            away_win_pct=final_probs[2] * 100,
            poisson_probs=poisson_probs,
            xgboost_probs=xgb_probs,
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
            confidence=confidence,
            prediction_time=datetime.now(),
            predicted_outcome=predicted_outcome,
            home_elo=home_elo,
            away_elo=away_elo
        )
    
    def predict_from_features(
        self,
        features: MatchFeatures,
        home_team_name: str,
        away_team_name: str
    ) -> EnsemblePrediction:
        """
        Predict using pre-calculated features.
        
        Args:
            features: MatchFeatures object
            home_team_name: Home team name
            away_team_name: Away team name
            
        Returns:
            EnsemblePrediction
        """
        return self.predict(
            home_team_id=features.home_team_id,
            away_team_id=features.away_team_id,
            home_team_name=home_team_name,
            away_team_name=away_team_name,
            league_id=features.league_id,
            features=features
        )
    
    def predict_batch(
        self,
        matches: List[Dict],
        features_df: pd.DataFrame
    ) -> List[EnsemblePrediction]:
        """
        Predict multiple matches.
        
        Args:
            matches: List of match dictionaries
            features_df: DataFrame with features for all matches
            
        Returns:
            List of EnsemblePrediction objects
        """
        predictions = []
        
        for i, match in enumerate(matches):
            if i < len(features_df):
                features = MatchFeatures(**features_df.iloc[i].to_dict())
            else:
                features = None
            
            pred = self.predict(
                home_team_id=match["home_team_id"],
                away_team_id=match["away_team_id"],
                home_team_name=match["home_team_name"],
                away_team_name=match["away_team_name"],
                league_id=match["league_id"],
                features=features
            )
            predictions.append(pred)
        
        return predictions
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of both models."""
        return {
            "poisson": {
                "fitted": self.poisson.is_fitted,
                "weight": self.poisson_weight,
            },
            "xgboost": {
                "fitted": self.xgboost.is_fitted,
                "weight": self.xgboost_weight,
            },
            "ensemble_ready": self.poisson.is_fitted or self.xgboost.is_fitted,
        }
    
    def update_weights(self, poisson_weight: float, xgboost_weight: float) -> None:
        """
        Update model weights.
        
        Args:
            poisson_weight: New Poisson weight
            xgboost_weight: New XGBoost weight
        """
        total = poisson_weight + xgboost_weight
        self.poisson_weight = poisson_weight / total
        self.xgboost_weight = xgboost_weight / total
        
        logger.info(
            f"Weights updated: Poisson={self.poisson_weight:.2f}, "
            f"XGBoost={self.xgboost_weight:.2f}"
        )
