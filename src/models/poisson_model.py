"""
Poisson Model for Football Match Prediction.

Uses the Poisson distribution to model the number of goals scored by each team.
Based on Dixon-Coles model principles with attack/defense strength ratings.

The model assumes:
- Goals scored by each team follow independent Poisson distributions
- Expected goals depend on attack strength of scoring team and defense strength of conceding team
- Home advantage is implicit in the league averages
"""

import numpy as np
import pandas as pd
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
    scoreline_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "1": round(self.home_win_prob, 4),
            "X": round(self.draw_prob, 4),
            "2": round(self.away_win_prob, 4),
            "expected_home_goals": round(self.expected_home_goals, 2),
            "expected_away_goals": round(self.expected_away_goals, 2),
        }


class PoissonModel:
    """
    Poisson-based goal prediction model.
    
    Calculates the probability of each scoreline using Poisson distribution,
    then aggregates to get 1X2 probabilities.
    
    Formula for expected goals:
    λ_home = home_attack_strength * away_defense_strength * league_home_avg
    λ_away = away_attack_strength * home_defense_strength * league_away_avg
    
    Where:
    - attack_strength = team's goals scored / league average
    - defense_strength = team's goals conceded / league average
    """
    
    def __init__(self, max_goals: int = 8):
        """
        Initialize the Poisson model.
        
        Args:
            max_goals: Maximum goals to consider for probability matrix
        """
        self.max_goals = max_goals
        self._team_stats: Dict[int, Dict[str, float]] = {}
        self._league_stats: Dict[int, Dict[str, float]] = {}
        self._is_fitted = False
    
    def fit(self, matches_df: pd.DataFrame) -> "PoissonModel":
        """
        Fit the model using historical match data.
        
        Calculates attack and defense strength for each team,
        and league averages for home and away goals.
        
        Args:
            matches_df: DataFrame with columns:
                - home_team_id, away_team_id
                - home_goals, away_goals
                - league_id
                
        Returns:
            self for method chaining
        """
        logger.info(f"Fitting Poisson model on {len(matches_df)} matches")
        
        # Calculate league averages
        for league_id in matches_df["league_id"].unique():
            league_df = matches_df[matches_df["league_id"] == league_id]
            
            self._league_stats[league_id] = {
                "avg_home_goals": league_df["home_goals"].mean(),
                "avg_away_goals": league_df["away_goals"].mean(),
                "matches": len(league_df),
            }
            
            logger.debug(
                f"League {league_id}: avg home={self._league_stats[league_id]['avg_home_goals']:.2f}, "
                f"avg away={self._league_stats[league_id]['avg_away_goals']:.2f}"
            )
        
        # Calculate team-level statistics
        all_teams = set(matches_df["home_team_id"]).union(set(matches_df["away_team_id"]))
        
        for team_id in all_teams:
            home_matches = matches_df[matches_df["home_team_id"] == team_id]
            away_matches = matches_df[matches_df["away_team_id"] == team_id]
            
            # Goals scored and conceded
            home_scored = home_matches["home_goals"].sum()
            home_conceded = home_matches["away_goals"].sum()
            away_scored = away_matches["away_goals"].sum()
            away_conceded = away_matches["home_goals"].sum()
            
            total_scored = home_scored + away_scored
            total_conceded = home_conceded + away_conceded
            n_matches = len(home_matches) + len(away_matches)
            
            if n_matches == 0:
                continue
            
            # Get league averages for this team's league
            if not home_matches.empty:
                league_id = home_matches["league_id"].iloc[0]
            elif not away_matches.empty:
                league_id = away_matches["league_id"].iloc[0]
            else:
                continue
                
            league_avg = self._league_stats.get(league_id, {})
            league_avg_goals = (
                league_avg.get("avg_home_goals", 1.3) + 
                league_avg.get("avg_away_goals", 1.1)
            ) / 2
            
            # Attack strength = goals scored per game / league average
            attack_strength = (total_scored / n_matches) / league_avg_goals if league_avg_goals > 0 else 1.0
            
            # Defense strength = goals conceded per game / league average
            defense_strength = (total_conceded / n_matches) / league_avg_goals if league_avg_goals > 0 else 1.0
            
            self._team_stats[team_id] = {
                "attack_strength": attack_strength,
                "defense_strength": defense_strength,
                "home_scored_avg": home_scored / len(home_matches) if len(home_matches) > 0 else 0,
                "home_conceded_avg": home_conceded / len(home_matches) if len(home_matches) > 0 else 0,
                "away_scored_avg": away_scored / len(away_matches) if len(away_matches) > 0 else 0,
                "away_conceded_avg": away_conceded / len(away_matches) if len(away_matches) > 0 else 0,
                "matches": n_matches,
                "league_id": league_id,
            }
        
        self._is_fitted = True
        logger.info(f"Model fitted with {len(self._team_stats)} teams")
        return self
    
    def _get_expected_goals(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate expected goals for each team.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: Optional league ID (uses team's league if not provided)
            
        Returns:
            Tuple of (expected_home_goals, expected_away_goals)
        """
        home_stats = self._team_stats.get(home_team_id, {})
        away_stats = self._team_stats.get(away_team_id, {})
        
        # Default values if team not found
        home_attack = home_stats.get("attack_strength", 1.0)
        home_defense = home_stats.get("defense_strength", 1.0)
        away_attack = away_stats.get("attack_strength", 1.0)
        away_defense = away_stats.get("defense_strength", 1.0)
        
        # Get league averages
        if league_id is None:
            league_id = home_stats.get("league_id", 39)  # Default to Premier League
        
        league_stats = self._league_stats.get(league_id, {
            "avg_home_goals": 1.5,
            "avg_away_goals": 1.2
        })
        
        # Expected goals formula
        # Home team expected = home attack * away defense * league home average
        expected_home = (
            home_attack * 
            away_defense * 
            league_stats["avg_home_goals"]
        )
        
        # Away team expected = away attack * home defense * league away average
        expected_away = (
            away_attack * 
            home_defense * 
            league_stats["avg_away_goals"]
        )
        
        # Clamp to reasonable values
        expected_home = max(0.3, min(4.0, expected_home))
        expected_away = max(0.2, min(3.5, expected_away))
        
        return expected_home, expected_away
    
    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None
    ) -> PoissonPrediction:
        """
        Predict match outcome probabilities.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: Optional league ID
            
        Returns:
            PoissonPrediction with probabilities and expected goals
        """
        exp_home, exp_away = self._get_expected_goals(
            home_team_id, away_team_id, league_id
        )
        
        # Create probability matrix for scorelines
        prob_matrix = self._calculate_scoreline_matrix(exp_home, exp_away)
        
        # Calculate 1X2 probabilities from matrix
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                if i > j:
                    home_win_prob += prob_matrix[i, j]
                elif i == j:
                    draw_prob += prob_matrix[i, j]
                else:
                    away_win_prob += prob_matrix[i, j]
        
        # Normalize (should be ~1.0 but ensure it)
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        return PoissonPrediction(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            expected_home_goals=exp_home,
            expected_away_goals=exp_away,
            scoreline_matrix=prob_matrix
        )
    
    def _calculate_scoreline_matrix(
        self,
        exp_home: float,
        exp_away: float
    ) -> np.ndarray:
        """
        Calculate probability matrix for all scorelines.
        
        Args:
            exp_home: Expected home goals (lambda)
            exp_away: Expected away goals (lambda)
            
        Returns:
            Matrix where [i,j] = P(home_goals=i, away_goals=j)
        """
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                # Assuming independence (basic model)
                matrix[i, j] = poisson.pmf(i, exp_home) * poisson.pmf(j, exp_away)
        
        return matrix
    
    def predict_scoreline(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get most likely scorelines.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: Optional league ID
            top_n: Number of top scorelines to return
            
        Returns:
            List of dictionaries with scoreline and probability
        """
        prediction = self.predict(home_team_id, away_team_id, league_id)
        matrix = prediction.scoreline_matrix
        
        scorelines = []
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                scorelines.append({
                    "home_goals": i,
                    "away_goals": j,
                    "scoreline": f"{i}-{j}",
                    "probability": matrix[i, j]
                })
        
        # Sort by probability descending
        scorelines.sort(key=lambda x: x["probability"], reverse=True)
        
        return scorelines[:top_n]
    
    def predict_over_under(
        self,
        home_team_id: int,
        away_team_id: int,
        line: float = 2.5,
        league_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Predict over/under probabilities for a given line.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            line: The goals line (e.g., 2.5)
            league_id: Optional league ID
            
        Returns:
            Dictionary with over and under probabilities
        """
        prediction = self.predict(home_team_id, away_team_id, league_id)
        matrix = prediction.scoreline_matrix
        
        over_prob = 0.0
        under_prob = 0.0
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                total_goals = i + j
                if total_goals > line:
                    over_prob += matrix[i, j]
                else:
                    under_prob += matrix[i, j]
        
        return {
            f"over_{line}": round(over_prob, 4),
            f"under_{line}": round(under_prob, 4)
        }
    
    def predict_btts(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Predict Both Teams To Score probability.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: Optional league ID
            
        Returns:
            Dictionary with BTTS yes/no probabilities
        """
        prediction = self.predict(home_team_id, away_team_id, league_id)
        matrix = prediction.scoreline_matrix
        
        btts_yes = 0.0
        
        for i in range(1, self.max_goals + 1):
            for j in range(1, self.max_goals + 1):
                btts_yes += matrix[i, j]
        
        return {
            "btts_yes": round(btts_yes, 4),
            "btts_no": round(1 - btts_yes, 4)
        }
    
    def get_team_stats(self, team_id: int) -> Optional[Dict]:
        """Get stored statistics for a team."""
        return self._team_stats.get(team_id)
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
