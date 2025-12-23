"""
Poisson Model for Football Match Prediction (Dynamic Version).

Uses the Poisson distribution to model the number of goals scored by each team,
with attack/defense strengths calculated dynamically from recent match form.
"""

import numpy as np
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PoissonPrediction:
    """Result of a Poisson model prediction."""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_25_prob: float = 0.0
    under_25_prob: float = 0.0
    btts_yes_prob: float = 0.0
    btts_no_prob: float = 0.0
    scoreline_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format for API responses."""
        return {
            "1x2": {
                "1": round(self.home_win_prob, 4),
                "X": round(self.draw_prob, 4),
                "2": round(self.away_win_prob, 4)
            },
            "goals": {
                "over_2.5": round(self.over_25_prob, 4),
                "under_2.5": round(self.under_25_prob, 4)
            },
            "btts": {
                "yes": round(self.btts_yes_prob, 4),
                "no": round(self.btts_no_prob, 4)
            },
            "expected_goals": {
                "home": round(self.expected_home_goals, 2),
                "away": round(self.expected_away_goals, 2)
            }
        }

class PoissonModel:
    """
    Dynamic Poisson-based goal prediction model.
    """
    
    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self._is_fitted = True  # Dynamic model doesn't need traditional fitting

    def fit(self, matches_df: any) -> None:
        """
        Placeholder fit method for training pipeline compatibility.
        The dynamic model calculates strengths on the fly.
        """
        logger.info("PoissonModel: Dynamic model used, skipping global fit.")
        pass

    def calculate_team_strength(self, team_id: int, last_5_matches: List[Dict]) -> Dict[str, float]:
        """
        Calculates: Media Goal Fatti e Media Goal Subiti nelle ultime 5 partite.
        """
        if not last_5_matches:
            return {"scored_avg": 1.3, "conceded_avg": 1.3} # Return league-like averages if no data
        
        total_scored = 0
        total_conceded = 0
        count = len(last_5_matches)
        
        for match in last_5_matches:
            # Match structure expected from API response or database
            match_home_id = match.get('home_team_id') or match.get('home', {}).get('id')
            match_home_goals = match.get('home_goals') or match.get('goals', {}).get('home')
            match_away_goals = match.get('away_goals') or match.get('goals', {}).get('away')
            
            if match_home_id == team_id:
                total_scored += match_home_goals if match_home_goals is not None else 0
                total_conceded += match_away_goals if match_away_goals is not None else 0
            else:
                total_scored += match_away_goals if match_away_goals is not None else 0
                total_conceded += match_home_goals if match_home_goals is not None else 0
                
        return {
            "scored_avg": total_scored / count,
            "conceded_avg": total_conceded / count
        }

    def predict(
        self,
        home_team_id: int,
        home_last_5: List[Dict],
        away_team_id: int,
        away_last_5: List[Dict]
    ) -> PoissonPrediction:
        """
        Predict match outcome probabilities using dynamic form.
        """
        home_stats = self.calculate_team_strength(home_team_id, home_last_5)
        away_stats = self.calculate_team_strength(away_team_id, away_last_5)
        
        # Calculate lambda_home and lambda_away
        # Expected goals = (Team A Scored Avg + Team B Conceded Avg) / 2
        lambda_home = (home_stats['scored_avg'] + away_stats['conceded_avg']) / 2
        lambda_away = (away_stats['scored_avg'] + home_stats['conceded_avg']) / 2
        
        # Clamp to reasonable values to avoid infinity/zero
        lambda_home = max(0.1, min(5.0, lambda_home))
        lambda_away = max(0.1, min(5.0, lambda_away))
        
        # Create probability matrix
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        # 1X2 Probabilities
        home_win_prob = np.sum(np.tril(matrix, -1).T) # Home goals > Away goals
        draw_prob = np.sum(np.diag(matrix))
        away_win_prob = np.sum(np.triu(matrix, 1))
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Over/Under 2.5
        under_25_prob = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                if i + j < 2.5:
                    under_25_prob += matrix[i, j]
        over_25_prob = 1.0 - under_25_prob
        
        # BTTS
        btts_no_prob = 0.0
        for i in range(self.max_goals + 1):
            btts_no_prob += matrix[i, 0] # Away 0
            btts_no_prob += matrix[0, i] # Home 0
        btts_no_prob -= matrix[0, 0] # Subtract double counted 0-0
        btts_yes_prob = 1.0 - btts_no_prob
        
        return PoissonPrediction(
            home_win_prob=float(home_win_prob),
            draw_prob=float(draw_prob),
            away_win_prob=float(away_win_prob),
            expected_home_goals=float(lambda_home),
            expected_away_goals=float(lambda_away),
            over_25_prob=float(over_25_prob),
            under_25_prob=float(under_25_prob),
            btts_yes_prob=float(btts_yes_prob),
            btts_no_prob=float(btts_no_prob),
            scoreline_matrix=matrix
        )

    @property
    def is_fitted(self) -> bool:
        return True
