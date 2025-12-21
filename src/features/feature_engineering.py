"""
Feature Engineering for Football Match Prediction.

Calculates advanced features from historical match data including:
- Team form (recent performance)
- Attack/defense strength ratings
- Head-to-head statistics
- Home/away performance differentials
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..data.api_football_client import Match, TeamStats


@dataclass
class MatchFeatures:
    """
    Computed features for a match prediction.
    All features are normalized and ready for model input.
    """
    # Match identification
    fixture_id: int
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    league_id: int
    
    # Team form (last N matches)
    home_form_goals_scored: float = 0.0
    home_form_goals_conceded: float = 0.0
    home_form_points: float = 0.0
    home_form_win_rate: float = 0.0
    
    away_form_goals_scored: float = 0.0
    away_form_goals_conceded: float = 0.0
    away_form_points: float = 0.0
    away_form_win_rate: float = 0.0
    
    # Home/Away specific performance
    home_home_goals_scored: float = 0.0
    home_home_goals_conceded: float = 0.0
    home_home_win_rate: float = 0.0
    
    away_away_goals_scored: float = 0.0
    away_away_goals_conceded: float = 0.0
    away_away_win_rate: float = 0.0
    
    # Attack/Defense strength (relative to league average)
    home_attack_strength: float = 1.0
    home_defense_strength: float = 1.0
    away_attack_strength: float = 1.0
    away_defense_strength: float = 1.0
    
    # Head-to-head
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0
    h2h_home_goals_avg: float = 0.0
    h2h_away_goals_avg: float = 0.0
    
    # Additional features
    home_clean_sheets_rate: float = 0.0
    away_clean_sheets_rate: float = 0.0
    home_btts_rate: float = 0.0  # Both Teams To Score
    away_btts_rate: float = 0.0
    
    # League position difference (if available)
    position_diff: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "fixture_id": self.fixture_id,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "league_id": self.league_id,
            "home_form_goals_scored": self.home_form_goals_scored,
            "home_form_goals_conceded": self.home_form_goals_conceded,
            "home_form_points": self.home_form_points,
            "home_form_win_rate": self.home_form_win_rate,
            "away_form_goals_scored": self.away_form_goals_scored,
            "away_form_goals_conceded": self.away_form_goals_conceded,
            "away_form_points": self.away_form_points,
            "away_form_win_rate": self.away_form_win_rate,
            "home_home_goals_scored": self.home_home_goals_scored,
            "home_home_goals_conceded": self.home_home_goals_conceded,
            "home_home_win_rate": self.home_home_win_rate,
            "away_away_goals_scored": self.away_away_goals_scored,
            "away_away_goals_conceded": self.away_away_goals_conceded,
            "away_away_win_rate": self.away_away_win_rate,
            "home_attack_strength": self.home_attack_strength,
            "home_defense_strength": self.home_defense_strength,
            "away_attack_strength": self.away_attack_strength,
            "away_defense_strength": self.away_defense_strength,
            "h2h_home_wins": self.h2h_home_wins,
            "h2h_draws": self.h2h_draws,
            "h2h_away_wins": self.h2h_away_wins,
            "h2h_home_goals_avg": self.h2h_home_goals_avg,
            "h2h_away_goals_avg": self.h2h_away_goals_avg,
            "home_clean_sheets_rate": self.home_clean_sheets_rate,
            "away_clean_sheets_rate": self.away_clean_sheets_rate,
            "home_btts_rate": self.home_btts_rate,
            "away_btts_rate": self.away_btts_rate,
            "position_diff": self.position_diff,
        }
    
    def to_feature_vector(self) -> List[float]:
        """
        Convert to feature vector for model input.
        Excludes identification fields.
        """
        return [
            self.home_form_goals_scored,
            self.home_form_goals_conceded,
            self.home_form_points,
            self.home_form_win_rate,
            self.away_form_goals_scored,
            self.away_form_goals_conceded,
            self.away_form_points,
            self.away_form_win_rate,
            self.home_home_goals_scored,
            self.home_home_goals_conceded,
            self.home_home_win_rate,
            self.away_away_goals_scored,
            self.away_away_goals_conceded,
            self.away_away_win_rate,
            self.home_attack_strength,
            self.home_defense_strength,
            self.away_attack_strength,
            self.away_defense_strength,
            self.h2h_home_wins,
            self.h2h_draws,
            self.h2h_away_wins,
            self.h2h_home_goals_avg,
            self.h2h_away_goals_avg,
            self.home_clean_sheets_rate,
            self.away_clean_sheets_rate,
            self.home_btts_rate,
            self.away_btts_rate,
            self.position_diff,
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names for model."""
        return [
            "home_form_goals_scored",
            "home_form_goals_conceded",
            "home_form_points",
            "home_form_win_rate",
            "away_form_goals_scored",
            "away_form_goals_conceded",
            "away_form_points",
            "away_form_win_rate",
            "home_home_goals_scored",
            "home_home_goals_conceded",
            "home_home_win_rate",
            "away_away_goals_scored",
            "away_away_goals_conceded",
            "away_away_win_rate",
            "home_attack_strength",
            "home_defense_strength",
            "away_attack_strength",
            "away_defense_strength",
            "h2h_home_wins",
            "h2h_draws",
            "h2h_away_wins",
            "h2h_home_goals_avg",
            "h2h_away_goals_avg",
            "home_clean_sheets_rate",
            "away_clean_sheets_rate",
            "home_btts_rate",
            "away_btts_rate",
            "position_diff",
        ]


class FeatureEngineer:
    """
    Calculates features for match prediction from historical data.
    
    Main responsibilities:
    - Process historical matches into form statistics
    - Calculate attack/defense strength ratings
    - Compute head-to-head features
    - Generate feature vectors for model input
    """
    
    def __init__(self, form_matches: int = 5):
        """
        Initialize the feature engineer.
        
        Args:
            form_matches: Number of recent matches to use for form calculation
        """
        self.form_matches = form_matches
        self._matches_df: Optional[pd.DataFrame] = None
        self._league_stats: Dict[int, Dict] = {}
    
    def load_matches(self, matches: List[Match]) -> None:
        """
        Load historical matches for feature calculation.
        
        Args:
            matches: List of Match objects
        """
        data = []
        for m in matches:
            if m.home_goals is not None and m.away_goals is not None:
                data.append({
                    "fixture_id": m.fixture_id,
                    "league_id": m.league_id,
                    "date": m.date,
                    "home_team_id": m.home_team_id,
                    "home_team_name": m.home_team_name,
                    "away_team_id": m.away_team_id,
                    "away_team_name": m.away_team_name,
                    "home_goals": m.home_goals,
                    "away_goals": m.away_goals,
                })
        
        self._matches_df = pd.DataFrame(data)
        if not self._matches_df.empty:
            self._matches_df["date"] = pd.to_datetime(self._matches_df["date"])
            self._matches_df = self._matches_df.sort_values("date")
            self._calculate_league_averages()
    
    def _calculate_league_averages(self) -> None:
        """Calculate league-wide averages for strength ratings."""
        if self._matches_df is None or self._matches_df.empty:
            return
            
        for league_id in self._matches_df["league_id"].unique():
            league_df = self._matches_df[self._matches_df["league_id"] == league_id]
            
            self._league_stats[league_id] = {
                "avg_home_goals": league_df["home_goals"].mean(),
                "avg_away_goals": league_df["away_goals"].mean(),
                "total_matches": len(league_df),
            }
    
    def _get_team_form(
        self,
        team_id: int,
        before_date: datetime,
        n_matches: int,
        home_only: bool = False,
        away_only: bool = False
    ) -> Dict[str, float]:
        """
        Calculate team form from recent matches.
        
        Args:
            team_id: Team ID
            before_date: Only consider matches before this date
            n_matches: Number of matches to consider
            home_only: Only consider home matches
            away_only: Only consider away matches
            
        Returns:
            Dictionary with form statistics
        """
        if self._matches_df is None or self._matches_df.empty:
            return self._empty_form()
        
        # Filter for team matches first (much faster than filtering all matches by date)
        if home_only:
            team_matches = self._matches_df[self._matches_df["home_team_id"] == team_id]
        elif away_only:
            team_matches = self._matches_df[self._matches_df["away_team_id"] == team_id]
        else:
            team_matches = self._matches_df[
                (self._matches_df["home_team_id"] == team_id) | 
                (self._matches_df["away_team_id"] == team_id)
            ]
            
        if team_matches.empty:
            return self._empty_form()
            
        # Then filter by date and create a copy for calculations
        team_matches = team_matches[team_matches["date"] < before_date].copy()
        
        if team_matches.empty:
            return self._empty_form()

        # Calculate statistics
        if home_only:
            team_matches["goals_scored"] = team_matches["home_goals"]
            team_matches["goals_conceded"] = team_matches["away_goals"]
            team_matches["points"] = team_matches.apply(
                lambda x: 3 if x["home_goals"] > x["away_goals"] 
                else (1 if x["home_goals"] == x["away_goals"] else 0), axis=1
            )
            team_matches["win"] = (team_matches["home_goals"] > team_matches["away_goals"]).astype(int)
            team_matches["clean_sheet"] = (team_matches["away_goals"] == 0).astype(int)
            team_matches["btts"] = ((team_matches["home_goals"] > 0) & (team_matches["away_goals"] > 0)).astype(int)
        elif away_only:
            team_matches["goals_scored"] = team_matches["away_goals"]
            team_matches["goals_conceded"] = team_matches["home_goals"]
            team_matches["points"] = team_matches.apply(
                lambda x: 3 if x["away_goals"] > x["home_goals"] 
                else (1 if x["away_goals"] == x["home_goals"] else 0), axis=1
            )
            team_matches["win"] = (team_matches["away_goals"] > team_matches["home_goals"]).astype(int)
            team_matches["clean_sheet"] = (team_matches["home_goals"] == 0).astype(int)
            team_matches["btts"] = ((team_matches["home_goals"] > 0) & (team_matches["away_goals"] > 0)).astype(int)
        else:
            # All matches - need to handle home/away separately for goals/points
            # This part is a bit more complex, let's keep it simple but efficient
            results = []
            for _, row in team_matches.iterrows():
                if row["home_team_id"] == team_id:
                    scored = row["home_goals"]
                    conceded = row["away_goals"]
                    points = 3 if scored > conceded else (1 if scored == conceded else 0)
                    win = 1 if scored > conceded else 0
                    cs = 1 if conceded == 0 else 0
                else:
                    scored = row["away_goals"]
                    conceded = row["home_goals"]
                    points = 3 if scored > conceded else (1 if scored == conceded else 0)
                    win = 1 if scored > conceded else 0
                    cs = 1 if conceded == 0 else 0
                
                btts = 1 if row["home_goals"] > 0 and row["away_goals"] > 0 else 0
                results.append({
                    "goals_scored": scored,
                    "goals_conceded": conceded,
                    "points": points,
                    "win": win,
                    "clean_sheet": cs,
                    "btts": btts,
                    "date": row["date"]
                })
            team_matches = pd.DataFrame(results)
        
        # Take last N matches
        team_matches = team_matches.sort_values("date", ascending=False).head(n_matches)
        
        if team_matches.empty:
            return self._empty_form()
        
        return {
            "goals_scored": team_matches["goals_scored"].mean(),
            "goals_conceded": team_matches["goals_conceded"].mean(),
            "points": team_matches["points"].mean(),
            "win_rate": team_matches["win"].mean(),
            "clean_sheets_rate": team_matches["clean_sheet"].mean(),
            "btts_rate": team_matches["btts"].mean(),
            "matches": len(team_matches),
        }
    
    def _empty_form(self) -> Dict[str, float]:
        """Return empty form statistics."""
        return {
            "goals_scored": 0.0,
            "goals_conceded": 0.0,
            "points": 0.0,
            "win_rate": 0.0,
            "clean_sheets_rate": 0.0,
            "btts_rate": 0.0,
            "matches": 0,
        }
    
    def _calculate_attack_defense_strength(
        self,
        team_id: int,
        league_id: int,
        before_date: datetime
    ) -> Tuple[float, float]:
        """
        Calculate attack and defense strength relative to league average.
        
        Attack strength = team's goals scored / league average goals scored
        Defense strength = team's goals conceded / league average goals conceded
        
        Args:
            team_id: Team ID
            league_id: League ID
            before_date: Only consider matches before this date
            
        Returns:
            Tuple of (attack_strength, defense_strength)
        """
        if self._matches_df is None or self._matches_df.empty:
            return 1.0, 1.0
            
        league_stats = self._league_stats.get(league_id)
        if not league_stats:
            return 1.0, 1.0
        
        # Filter for team matches in this league before the date
        team_matches = self._matches_df[
            ((self._matches_df["home_team_id"] == team_id) | (self._matches_df["away_team_id"] == team_id)) &
            (self._matches_df["league_id"] == league_id) &
            (self._matches_df["date"] < before_date)
        ]
        
        if team_matches.empty:
            return 1.0, 1.0
            
        # Get team's home and away matches from the subset
        home_matches = team_matches[team_matches["home_team_id"] == team_id]
        away_matches = team_matches[team_matches["away_team_id"] == team_id]
        
        # Calculate team averages
        home_scored = home_matches["home_goals"].mean() if not home_matches.empty else 0
        home_conceded = home_matches["away_goals"].mean() if not home_matches.empty else 0
        away_scored = away_matches["away_goals"].mean() if not away_matches.empty else 0
        away_conceded = away_matches["home_goals"].mean() if not away_matches.empty else 0
        
        # Weighted average (considering both home and away)
        n_home = len(home_matches)
        n_away = len(away_matches)
        n_total = n_home + n_away
        
        if n_total == 0:
            return 1.0, 1.0
        
        avg_scored = (home_scored * n_home + away_scored * n_away) / n_total
        avg_conceded = (home_conceded * n_home + away_conceded * n_away) / n_total
        
        # Calculate strength relative to league average
        league_avg_goals = (league_stats["avg_home_goals"] + league_stats["avg_away_goals"]) / 2
        
        attack_strength = avg_scored / league_avg_goals if league_avg_goals > 0 else 1.0
        defense_strength = avg_conceded / league_avg_goals if league_avg_goals > 0 else 1.0
        
        return attack_strength, defense_strength
    
    def _get_h2h_stats(
        self,
        home_team_id: int,
        away_team_id: int,
        h2h_matches: List[Match]
    ) -> Dict[str, float]:
        """
        Calculate head-to-head statistics.
        
        Args:
            home_team_id: Home team ID (for this match)
            away_team_id: Away team ID (for this match)
            h2h_matches: List of historical h2h matches
            
        Returns:
            Dictionary with h2h statistics
        """
        if not h2h_matches:
            return {
                "home_wins": 0,
                "draws": 0,
                "away_wins": 0,
                "home_goals_avg": 0.0,
                "away_goals_avg": 0.0,
            }
        
        home_wins = 0
        away_wins = 0
        draws = 0
        home_team_goals = []
        away_team_goals = []
        
        for match in h2h_matches:
            if match.home_goals is None or match.away_goals is None:
                continue
                
            # Determine whose home/away from the perspective of our match
            if match.home_team_id == home_team_id:
                home_team_goals.append(match.home_goals)
                away_team_goals.append(match.away_goals)
                if match.home_goals > match.away_goals:
                    home_wins += 1
                elif match.home_goals < match.away_goals:
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_team_goals.append(match.away_goals)
                away_team_goals.append(match.home_goals)
                if match.away_goals > match.home_goals:
                    home_wins += 1
                elif match.away_goals < match.home_goals:
                    away_wins += 1
                else:
                    draws += 1
        
        return {
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "home_goals_avg": np.mean(home_team_goals) if home_team_goals else 0.0,
            "away_goals_avg": np.mean(away_team_goals) if away_team_goals else 0.0,
        }
    
    def calculate_features(
        self,
        match: Match,
        h2h_matches: Optional[List[Match]] = None,
        standings: Optional[List[Dict]] = None
    ) -> MatchFeatures:
        """
        Calculate all features for a match.
        
        Args:
            match: The match to calculate features for
            h2h_matches: Optional list of head-to-head matches
            standings: Optional current league standings
            
        Returns:
            MatchFeatures object with all calculated features
        """
        features = MatchFeatures(
            fixture_id=match.fixture_id,
            home_team_id=match.home_team_id,
            home_team_name=match.home_team_name,
            away_team_id=match.away_team_id,
            away_team_name=match.away_team_name,
            league_id=match.league_id,
        )
        
        # Overall form
        home_form = self._get_team_form(
            match.home_team_id, match.date, self.form_matches
        )
        away_form = self._get_team_form(
            match.away_team_id, match.date, self.form_matches
        )
        
        features.home_form_goals_scored = home_form["goals_scored"]
        features.home_form_goals_conceded = home_form["goals_conceded"]
        features.home_form_points = home_form["points"]
        features.home_form_win_rate = home_form["win_rate"]
        features.home_clean_sheets_rate = home_form["clean_sheets_rate"]
        features.home_btts_rate = home_form["btts_rate"]
        
        features.away_form_goals_scored = away_form["goals_scored"]
        features.away_form_goals_conceded = away_form["goals_conceded"]
        features.away_form_points = away_form["points"]
        features.away_form_win_rate = away_form["win_rate"]
        features.away_clean_sheets_rate = away_form["clean_sheets_rate"]
        features.away_btts_rate = away_form["btts_rate"]
        
        # Home/Away specific form
        home_home_form = self._get_team_form(
            match.home_team_id, match.date, self.form_matches * 2, home_only=True
        )
        away_away_form = self._get_team_form(
            match.away_team_id, match.date, self.form_matches * 2, away_only=True
        )
        
        features.home_home_goals_scored = home_home_form["goals_scored"]
        features.home_home_goals_conceded = home_home_form["goals_conceded"]
        features.home_home_win_rate = home_home_form["win_rate"]
        
        features.away_away_goals_scored = away_away_form["goals_scored"]
        features.away_away_goals_conceded = away_away_form["goals_conceded"]
        features.away_away_win_rate = away_away_form["win_rate"]
        
        # Attack/Defense strength
        home_att, home_def = self._calculate_attack_defense_strength(
            match.home_team_id, match.league_id, match.date
        )
        away_att, away_def = self._calculate_attack_defense_strength(
            match.away_team_id, match.league_id, match.date
        )
        
        features.home_attack_strength = home_att
        features.home_defense_strength = home_def
        features.away_attack_strength = away_att
        features.away_defense_strength = away_def
        
        # Head-to-head
        if h2h_matches:
            h2h_stats = self._get_h2h_stats(
                match.home_team_id, match.away_team_id, h2h_matches
            )
            features.h2h_home_wins = h2h_stats["home_wins"]
            features.h2h_draws = h2h_stats["draws"]
            features.h2h_away_wins = h2h_stats["away_wins"]
            features.h2h_home_goals_avg = h2h_stats["home_goals_avg"]
            features.h2h_away_goals_avg = h2h_stats["away_goals_avg"]
        
        # Position difference from standings
        if standings:
            home_pos = next(
                (s["rank"] for s in standings if s["team_id"] == match.home_team_id),
                10
            )
            away_pos = next(
                (s["rank"] for s in standings if s["team_id"] == match.away_team_id),
                10
            )
            features.position_diff = away_pos - home_pos  # Positive = home team higher
        
        return features
    
    def calculate_features_batch(
        self,
        matches: List[Match],
        h2h_dict: Optional[Dict[int, List[Match]]] = None,
        standings: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Calculate features for multiple matches.
        
        Args:
            matches: List of matches
            h2h_dict: Dictionary mapping fixture_id to h2h matches
            standings: League standings
            
        Returns:
            DataFrame with features for all matches
        """
        features_list = []
        
        for match in matches:
            h2h = h2h_dict.get(match.fixture_id) if h2h_dict else None
            features = self.calculate_features(match, h2h, standings)
            features_list.append(features.to_dict())
        
        return pd.DataFrame(features_list)
    
    def get_training_data(
        self,
        matches: List[Match]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from historical matches.
        
        Args:
            matches: List of finished matches
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        self.load_matches(matches)
        
        features_list = []
        targets = []
        
        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
                
            features = self.calculate_features(match)
            features_list.append(features.to_dict())
            
            # Target: 0 = Home Win, 1 = Draw, 2 = Away Win
            if match.home_goals > match.away_goals:
                targets.append(0)
            elif match.home_goals == match.away_goals:
                targets.append(1)
            else:
                targets.append(2)
        
        X = pd.DataFrame(features_list)
        y = pd.Series(targets, name="result")
        
        # Drop non-feature columns
        feature_cols = MatchFeatures.feature_names()
        X = X[feature_cols]
        
        return X, y
