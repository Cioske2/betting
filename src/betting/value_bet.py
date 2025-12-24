"""
Value Bet Detection and Analysis.

Identifies betting opportunities where the model's probability
is higher than implied by bookmaker odds (positive expected value).

Uses Kelly Criterion for optimal stake sizing.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BetConfidence(Enum):
    """Confidence levels for bet recommendations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


@dataclass
class ValueBetResult:
    """
    Result of value bet analysis for a single outcome.
    
    Attributes:
        outcome: The bet outcome ("1", "X", "2")
        model_prob: Model's probability estimate (0-1)
        bookmaker_prob: Implied probability from odds (0-1)
        odds: Bookmaker's decimal odds
        expected_value: Expected value of bet (EV = prob * odds - 1)
        edge: Percentage edge over bookmaker
        is_value: Whether this qualifies as a value bet
        kelly_stake: Recommended stake using Kelly Criterion (% of bankroll)
        confidence: Confidence level of the recommendation
    """
    outcome: str
    model_prob: float
    bookmaker_prob: float
    odds: float
    expected_value: float
    edge: float
    is_value: bool
    kelly_stake: float
    confidence: BetConfidence
    
    def to_dict(self) -> Dict:
        """Convert to API response format."""
        return {
            "outcome": self.outcome,
            "outcome_name": self._outcome_name(),
            "model_probability": round(self.model_prob * 100, 2),
            "bookmaker_probability": round(self.bookmaker_prob * 100, 2),
            "odds": self.odds,
            "expected_value": round(self.expected_value, 4),
            "edge_percent": f"{round(self.edge * 100, 2)}%",
            "is_value_bet": self.is_value,
            "suggested_stake_percent": f"{round(self.kelly_stake * 100, 2)}%",
            "confidence": self.confidence.value,
        }
    
    def _outcome_name(self) -> str:
        """Get human-readable outcome name."""
        names = {"1": "Home Win", "X": "Draw", "2": "Away Win"}
        return names.get(self.outcome, self.outcome)


@dataclass
class MatchValueAnalysis:
    """
    Complete value bet analysis for a match.
    
    Contains analysis for all outcomes and overall recommendation.
    """
    home_team: str
    away_team: str
    value_bets: List[ValueBetResult]
    best_value: Optional[ValueBetResult]
    recommendation: str
    total_edge: float
    
    def to_dict(self) -> Dict:
        """Convert to API response format."""
        return {
            "match": f"{self.home_team} vs {self.away_team}",
            "value_bets": [vb.to_dict() for vb in self.value_bets if vb.is_value],
            "all_outcomes": [vb.to_dict() for vb in self.value_bets],
            "best_value": self.best_value.to_dict() if self.best_value else None,
            "recommendation": self.recommendation,
            "total_edge_percent": f"{round(self.total_edge * 100, 2)}%",
        }
    
    def has_value(self) -> bool:
        """Check if any value bet exists."""
        return any(vb.is_value for vb in self.value_bets)


class ValueBetAnalyzer:
    """
    Analyzes betting opportunities based on model predictions vs bookmaker odds.
    
    Core concepts:
    - Value bet: When model probability > implied probability from odds
    - Expected Value (EV): probability * odds - 1
    - Edge: (model_prob - book_prob) / book_prob
    - Kelly Criterion: Optimal stake sizing for long-term growth
    
    Example:
        If model says Home Win = 45% and bookmaker offers odds of 2.50:
        - Implied prob = 1/2.50 = 40%
        - Edge = (45 - 40) / 40 = 12.5%
        - EV = 0.45 * 2.50 - 1 = 0.125 (12.5%)
        - Kelly = (0.45 * 2.50 - 1) / (2.50 - 1) = 8.3%
    """
    
    def __init__(
        self,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.25,
        max_stake: float = 0.05,
        min_odds: float = 1.20,
        max_odds: float = 10.0
    ):
        """
        Initialize the value bet analyzer.
        
        Args:
            min_edge: Minimum edge required (default 5%)
            kelly_fraction: Fraction of Kelly stake (default 25%)
            max_stake: Maximum stake as % of bankroll (default 5%)
            min_odds: Minimum acceptable odds (default 1.20)
            max_odds: Maximum acceptable odds (default 10.0)
        """
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_stake = max_stake
        self.min_odds = min_odds
        self.max_odds = max_odds
        
        logger.info(
            f"ValueBetAnalyzer initialized: min_edge={min_edge:.1%}, "
            f"kelly_fraction={kelly_fraction:.1%}"
        )
    
    @staticmethod
    def odds_to_probability(odds: float) -> float:
        """
        Convert decimal odds to implied probability.
        
        Args:
            odds: Decimal odds (e.g., 2.50)
            
        Returns:
            Implied probability (0-1)
        """
        if odds <= 1.0:
            return 1.0
        return 1.0 / odds

    @staticmethod
    def remove_margin(odds_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Remove bookmaker margin to get fair probabilities.
        Uses the proportional method (Multiplicative).
        
        Args:
            odds_dict: Dictionary of odds {"1": 2.0, "X": 3.0, "2": 4.0}
            
        Returns:
            Dictionary of fair probabilities
        """
        implied_probs = {k: 1.0 / v for k, v in odds_dict.items() if v > 0}
        total_prob = sum(implied_probs.values())
        
        if total_prob <= 0:
            return {k: 0.33 for k in odds_dict.keys()}
            
        # Normalize to sum to 1.0
        fair_probs = {k: p / total_prob for k, p in implied_probs.items()}
        return fair_probs
    
    @staticmethod
    def probability_to_odds(prob: float) -> float:
        """
        Convert probability to fair odds.
        
        Args:
            prob: Probability (0-1)
            
        Returns:
            Fair decimal odds
        """
        if prob <= 0:
            return 100.0  # Very high odds for impossible events
        if prob >= 1:
            return 1.0
        return 1.0 / prob
    
    def calculate_expected_value(self, prob: float, odds: float) -> float:
        """
        Calculate expected value of a bet.
        
        EV = (probability * odds) - 1
        
        Positive EV means profitable in the long run.
        
        Args:
            prob: Model's probability estimate
            odds: Bookmaker's decimal odds
            
        Returns:
            Expected value (-1 to +∞)
        """
        return (prob * odds) - 1.0
    
    def calculate_edge(self, model_prob: float, book_prob: float) -> float:
        """
        Calculate percentage edge over bookmaker.
        
        Edge = (model_prob - book_prob) / book_prob
        
        Args:
            model_prob: Model's probability
            book_prob: Bookmaker's implied probability
            
        Returns:
            Edge as decimal (0.10 = 10% edge)
        """
        if book_prob <= 0:
            return 0.0
        return (model_prob - book_prob) / book_prob
    
    def calculate_kelly_stake(
        self,
        prob: float,
        odds: float
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.
        
        Full Kelly: f* = (b*p - q) / b
        where:
        - b = odds - 1 (net odds)
        - p = probability of winning
        - q = 1 - p (probability of losing)
        
        We use fractional Kelly for safety.
        
        Args:
            prob: Probability of winning
            odds: Decimal odds
            
        Returns:
            Recommended stake as fraction of bankroll (0-1)
        """
        if odds <= 1.0 or prob <= 0:
            return 0.0
        
        b = odds - 1  # Net odds (profit if win)
        p = prob
        q = 1 - prob
        
        # Full Kelly
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly *= self.kelly_fraction
        
        # Clamp to max stake
        kelly = max(0.0, min(self.max_stake, kelly))
        
        return kelly
    
    def _determine_confidence(
        self,
        edge: float,
        model_prob: float,
        ev: float
    ) -> BetConfidence:
        """
        Determine confidence level for a bet.
        
        Based on:
        - Size of edge
        - Absolute probability
        - Expected value
        """
        if edge >= 0.20 and model_prob >= 0.45 and ev >= 0.15:
            return BetConfidence.VERY_HIGH
        elif edge >= 0.12 and model_prob >= 0.35:
            return BetConfidence.HIGH
        elif edge >= 0.07:
            return BetConfidence.MEDIUM
        else:
            return BetConfidence.LOW
    
    def analyze_outcome(
        self,
        outcome: str,
        model_prob: float,
        bookmaker_odds: float
    ) -> ValueBetResult:
        """
        Analyze a single betting outcome.
        
        Args:
            outcome: Outcome code ("1", "X", "2")
            model_prob: Model's probability (0-1)
            bookmaker_odds: Bookmaker's decimal odds
            
        Returns:
            ValueBetResult with full analysis
        """
        # Filter extreme odds
        if bookmaker_odds < self.min_odds or bookmaker_odds > self.max_odds:
            return ValueBetResult(
                outcome=outcome,
                model_prob=model_prob,
                bookmaker_prob=self.odds_to_probability(bookmaker_odds),
                odds=bookmaker_odds,
                expected_value=-1.0,
                edge=-1.0,
                is_value=False,
                kelly_stake=0.0,
                confidence=BetConfidence.LOW
            )
        
        book_prob = self.odds_to_probability(bookmaker_odds)
        ev = self.calculate_expected_value(model_prob, bookmaker_odds)
        edge = self.calculate_edge(model_prob, book_prob)
        
        # Determine if it's a value bet
        is_value = edge >= self.min_edge and ev > 0
        
        # Calculate Kelly stake only if value
        kelly_stake = self.calculate_kelly_stake(model_prob, bookmaker_odds) if is_value else 0.0
        
        # Determine confidence
        confidence = self._determine_confidence(edge, model_prob, ev) if is_value else BetConfidence.LOW
        
        return ValueBetResult(
            outcome=outcome,
            model_prob=model_prob,
            bookmaker_prob=book_prob,
            odds=bookmaker_odds,
            expected_value=ev,
            edge=edge,
            is_value=is_value,
            kelly_stake=kelly_stake,
            confidence=confidence
        )
    
    def analyze_match(
        self,
        model_probs: Dict[str, float],
        bookmaker_odds: Dict[str, float],
        home_team: str = "Home",
        away_team: str = "Away"
    ) -> MatchValueAnalysis:
        """
        Analyze all outcomes for a match.
        
        Args:
            model_probs: Model probabilities {"1": 0.45, "X": 0.25, "2": 0.30}
            bookmaker_odds: Bookmaker odds {"1": 2.50, "X": 3.20, "2": 2.80}
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            MatchValueAnalysis with complete analysis
        """
        results = []
        fair_probs = self.remove_margin(bookmaker_odds)
        
        for outcome in ["1", "X", "2"]:
            model_p = model_probs.get(outcome, 0.33)
            book_odds = bookmaker_odds.get(outcome, 2.0)
            fair_p = fair_probs.get(outcome, 0.33)
            
            result = self.analyze_outcome(outcome, model_p, book_odds)
            
            # Add fair probability to the result (we might need to update ValueBetResult class)
            # For now, let's just use it to refine 'is_value'
            # A "safer" bet is when Model Prob > Fair Prob, not just Model Prob > Book Prob
            if result.is_value and model_p <= fair_p:
                # If model prob is less than fair prob, it's a risky "value" 
                # because it might just be the margin we are seeing.
                result.is_value = False 
                result.kelly_stake = 0.0
                result.confidence = BetConfidence.LOW
                
            results.append(result)
        
        # Find best value bet
        value_bets = [r for r in results if r.is_value]
        best_value = max(value_bets, key=lambda x: x.edge) if value_bets else None
        
        # Calculate total edge (sum of positive edges)
        total_edge = sum(r.edge for r in results if r.edge > 0)
        
        # Generate recommendation
        if best_value:
            if best_value.confidence in [BetConfidence.HIGH, BetConfidence.VERY_HIGH]:
                recommendation = (
                    f"Strong value bet on {best_value._outcome_name()} "
                    f"@ {best_value.odds} ({best_value.edge*100:.1f}% edge)"
                )
            else:
                recommendation = (
                    f"Value bet on {best_value._outcome_name()} "
                    f"@ {best_value.odds} ({best_value.edge*100:.1f}% edge)"
                )
        else:
            recommendation = "No value bets detected - consider skipping this match"
        
        return MatchValueAnalysis(
            home_team=home_team,
            away_team=away_team,
            value_bets=results,
            best_value=best_value,
            recommendation=recommendation,
            total_edge=total_edge
        )
    
    def find_best_bets(
        self,
        matches: List[MatchValueAnalysis],
        top_n: int = 5
    ) -> List[Tuple[str, ValueBetResult]]:
        """
        Find the best value bets across multiple matches.
        
        Args:
            matches: List of match analyses
            top_n: Number of top bets to return
            
        Returns:
            List of (match_name, ValueBetResult) tuples sorted by edge
        """
        all_bets = []
        
        for match in matches:
            for vb in match.value_bets:
                if vb.is_value:
                    match_name = f"{match.home_team} vs {match.away_team}"
                    all_bets.append((match_name, vb))
        
        # Sort by edge descending
        all_bets.sort(key=lambda x: x[1].edge, reverse=True)
        
        return all_bets[:top_n]
    
    def calculate_bankroll_allocation(
        self,
        value_bets: List[ValueBetResult],
        bankroll: float = 1000.0
    ) -> List[Dict]:
        """
        Calculate stake allocation for multiple value bets.
        
        Normalizes Kelly stakes if total exceeds limit.
        
        Args:
            value_bets: List of value bets to stake
            bankroll: Total bankroll
            
        Returns:
            List of dicts with bet details and stake amounts
        """
        if not value_bets:
            return []
        
        # Calculate total Kelly stake
        total_kelly = sum(vb.kelly_stake for vb in value_bets)
        
        # Normalize if too high (max 20% of bankroll on one round)
        max_total = 0.20
        scale_factor = min(1.0, max_total / total_kelly) if total_kelly > 0 else 0
        
        allocations = []
        for vb in value_bets:
            stake_pct = vb.kelly_stake * scale_factor
            stake_amount = bankroll * stake_pct
            
            allocations.append({
                "outcome": vb.outcome,
                "outcome_name": vb._outcome_name(),
                "odds": vb.odds,
                "edge": f"{vb.edge*100:.1f}%",
                "stake_percent": f"{stake_pct*100:.2f}%",
                "stake_amount": round(stake_amount, 2),
                "potential_profit": round(stake_amount * (vb.odds - 1), 2),
                "confidence": vb.confidence.value,
            })
        
        return allocations
