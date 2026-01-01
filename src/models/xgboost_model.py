"""
XGBoost Model for Football Match Prediction.

Uses gradient boosting with engineered features to predict 1X2 outcomes.
Features probability calibration for accurate probability estimates.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class XGBPrediction:
    """Result of XGBoost model prediction."""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "1": round(self.home_win_prob, 4),
            "X": round(self.draw_prob, 4),
            "2": round(self.away_win_prob, 4),
            "confidence": round(self.confidence, 4),
        }


class XGBoostPredictor:
    """
    XGBoost-based match outcome predictor.
    
    Features:
    - Multi-class classification (Home/Draw/Away)
    - Probability calibration using isotonic regression
    - Cross-validation for robust evaluation
    - Feature importance analysis
    - Model persistence (save/load)
    
    Target encoding:
    - 0 = Home Win
    - 1 = Draw
    - 2 = Away Win
    """
    
    # Default hyperparameters optimized for football prediction
    DEFAULT_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
    }
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        calibrate: bool = True
    ):
        """
        Initialize the XGBoost predictor.
        
        Args:
            params: XGBoost parameters (uses defaults if not provided)
            calibrate: Whether to use probability calibration
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.calibrate = calibrate
        
        self._model: Optional[xgb.XGBClassifier] = None
        self._calibrated_model: Optional[CalibratedClassifierCV] = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._cv_scores: Optional[Dict] = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        verbose: bool = True
    ) -> "XGBoostPredictor":
        """
        Train the model on historical data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (0=Home, 1=Draw, 2=Away)
            eval_set: Optional validation set for early stopping
            verbose: Whether to print training progress
            
        Returns:
            self for method chaining
        """
        logger.info(f"Training XGBoost on {len(X)} samples with {len(X.columns)} features")
        
        self._feature_names = list(X.columns)
        
        # Initialize model
        self._model = xgb.XGBClassifier(**self.params)
        
        # Prepare eval set if provided
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = [(eval_set[0], eval_set[1])]
            fit_params["verbose"] = verbose
        
        # Train base model
        self._model.fit(X, y, **fit_params)
        
        # Calibrate probabilities if requested
        if self.calibrate:
            logger.info("Calibrating probabilities...")
            self._calibrated_model = CalibratedClassifierCV(
                self._model,
                method="isotonic",
                cv=3
            )
            self._calibrated_model.fit(X, y)
        
        self._is_fitted = True
        logger.info("Model training complete")
        
        # Log feature importance
        importance = self.get_feature_importance()
        top_5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top 5 features: {', '.join([f'{k}: {v:.4f}' for k, v in top_5])}")
        
        return self
    
    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance (weight, gain, cover)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted or self._model is None:
            return {}
        
        importance = self._model.get_booster().get_score(
            importance_type=importance_type
        )
        
        if not importance:
            # Fallback to sklearn property if booster score is empty
            try:
                importances = self._model.feature_importances_
                return dict(zip(self._feature_names, [round(float(x), 4) for x in importances]))
            except:
                return {name: 0.0 for name in self._feature_names}
        
        # Map to feature names and normalize
        total = sum(importance.values())
        result = {}
        
        for i, name in enumerate(self._feature_names):
            # Try both the name and the f{i} format
            score = importance.get(name, importance.get(f"f{i}", 0))
            result[name] = round(score / total, 4)
        
        # Sort by importance
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        
        return result
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "accuracy"
    ) -> Dict[str, float]:
        """
        Perform cross-validation to evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Running {cv}-fold cross-validation...")
        
        model = xgb.XGBClassifier(**self.params)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        
        self._cv_scores = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
            "cv_folds": cv,
            "metric": scoring,
        }
        
        logger.info(
            f"CV {scoring}: {self._cv_scores['mean']:.4f} "
            f"(+/- {self._cv_scores['std']:.4f})"
        )
        
        return self._cv_scores
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each outcome.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of shape (n_samples, 3) with probabilities [P(0), P(1), P(2)]
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Ensure correct feature order
        X = X[self._feature_names]
        
        if self.calibrate and self._calibrated_model is not None:
            probs = self._calibrated_model.predict_proba(X)
        else:
            probs = self._model.predict_proba(X)
        
        return probs
    
    def predict(self, X: pd.DataFrame) -> XGBPrediction:
        """
        Predict match outcome for a single match.
        
        Args:
            X: Feature DataFrame (single row or first row used)
            
        Returns:
            XGBPrediction with probabilities
        """
        if len(X) > 1:
            X = X.iloc[[0]]
        
        probs = self.predict_proba(X)[0]
        
        # Calculate confidence as the maximum probability
        confidence = float(np.max(probs))
        
        return XGBPrediction(
            home_win_prob=float(probs[0]),
            draw_prob=float(probs[1]),
            away_win_prob=float(probs[2]),
            confidence=confidence,
            feature_importance=self.get_feature_importance()
        )
    
    def predict_batch(self, X: pd.DataFrame) -> List[XGBPrediction]:
        """
        Predict outcomes for multiple matches.
        
        Args:
            X: Feature DataFrame with multiple rows
            
        Returns:
            List of XGBPrediction objects
        """
        probs = self.predict_proba(X)
        predictions = []
        
        for p in probs:
            predictions.append(XGBPrediction(
                home_win_prob=float(p[0]),
                draw_prob=float(p[1]),
                away_win_prob=float(p[2]),
                confidence=float(np.max(p))
            ))
        
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path (without extension)
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        state = {
            "model": self._model,
            "calibrated_model": self._calibrated_model,
            "feature_names": self._feature_names,
            "params": self.params,
            "cv_scores": self._cv_scores,
            "calibrate": self.calibrate,
        }
        
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Model saved to {path}.pkl")
    
    def load(self, path: str) -> "XGBoostPredictor":
        """
        Load model from disk.
        
        Args:
            path: File path (without extension)
            
        Returns:
            self for method chaining
        """
        with open(f"{path}.pkl", "rb") as f:
            state = pickle.load(f)
        
        if isinstance(state, XGBoostPredictor):
            # Handle legacy/alternative save format where entire object was pickled
            logger.info("Loaded XGBoostPredictor object directly")
            self._model = state._model
            self._calibrated_model = state._calibrated_model
            self._feature_names = state._feature_names
            self.params = state.params
            self._cv_scores = state._cv_scores
            self.calibrate = state.calibrate
            self._is_fitted = state._is_fitted
        else:
            # Handle standard save format (state dict)
            self._model = state["model"]
            self._calibrated_model = state["calibrated_model"]
            self._feature_names = state["feature_names"]
            self.params = state["params"]
            self._cv_scores = state["cv_scores"]
            self.calibrate = state["calibrate"]
            self._is_fitted = True
        
        logger.info(f"Model loaded from {path}.pkl")
        
        return self
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics and metadata."""
        return {
            "is_fitted": self._is_fitted,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
            "params": self.params,
            "cv_scores": self._cv_scores,
            "calibrated": self.calibrate,
        }
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted


def create_pretrained_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> Tuple[XGBoostPredictor, Dict]:
    """
    Convenience function to create and train a model with validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Fraction for validation set
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Create and train model
    model = XGBoostPredictor()
    model.train(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate
    cv_scores = model.cross_validate(X, y)
    
    # Validation accuracy
    val_preds = model.predict_proba(X_val)
    val_pred_classes = np.argmax(val_preds, axis=1)
    val_accuracy = (val_pred_classes == y_val).mean()
    
    metrics = {
        "cv_accuracy": cv_scores["mean"],
        "cv_std": cv_scores["std"],
        "val_accuracy": val_accuracy,
        "train_size": len(X_train),
        "val_size": len(X_val),
    }
    
    return model, metrics
