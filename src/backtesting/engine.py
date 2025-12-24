
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..betting.value_bet import ValueBetAnalyzer, ValueBetResult
from ..models.ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Results of a backtesting run."""
    total_bets: int
    won_bets: int
    win_rate: float
    total_staked: float
    total_return: float
    profit: float
    roi: float
    final_bankroll: float
    max_drawdown: float
    sharpe_ratio: float
    history: pd.DataFrame

class BacktestEngine:
    """
    Engine for backtesting betting strategies.
    
    Simulates betting over historical data using model predictions.
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        analyzer: Optional[ValueBetAnalyzer] = None
    ):
        self.initial_bankroll = initial_bankroll
        self.analyzer = analyzer or ValueBetAnalyzer()
        
    def run(
        self,
        matches_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        odds_df: pd.DataFrame
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            matches_df: Historical match results (must have 'result' column: 1, X, 2)
            predictions_df: Model probabilities for each match
            odds_df: Historical odds for each match
            
        Returns:
            BacktestResult object
        """
        bankroll = self.initial_bankroll
        history = []
        
        # Merge data
        data = matches_df.merge(predictions_df, on='fixture_id')
        data = data.merge(odds_df, on='fixture_id')
        
        # Sort by date
        if 'date' in data.columns:
            data = data.sort_values('date')
            
        total_staked = 0
        total_return = 0
        won_bets = 0
        bankroll_history = [self.initial_bankroll]
        
        for _, row in data.iterrows():
            model_probs = {
                "1": row['prob_home'],
                "X": row['prob_draw'],
                "2": row['prob_away']
            }
            book_odds = {
                "1": row['odds_home'],
                "X": row['odds_draw'],
                "2": row['odds_away']
            }
            
            analysis = self.analyzer.analyze_match(model_probs, book_odds)
            
            # Check if we have a value bet
            for vb in analysis.value_bets:
                if vb.is_value:
                    stake = vb.kelly_stake * bankroll
                    if stake <= 0:
                        continue
                        
                    total_staked += stake
                    bankroll -= stake
                    
                    # Check if bet won
                    actual_result = str(row['result'])
                    is_win = (vb.outcome == actual_result)
                    
                    if is_win:
                        win_amount = stake * vb.odds
                        bankroll += win_amount
                        total_return += win_amount
                        won_bets += 1
                        profit = win_amount - stake
                    else:
                        profit = -stake
                        
                    history.append({
                        'fixture_id': row['fixture_id'],
                        'date': row.get('date'),
                        'home_team': row.get('home_team'),
                        'away_team': row.get('away_team'),
                        'outcome': vb.outcome,
                        'odds': vb.odds,
                        'prob': vb.model_prob,
                        'stake': stake,
                        'profit': profit,
                        'bankroll': bankroll,
                        'is_win': is_win
                    })
                    
                    bankroll_history.append(bankroll)
                    
        history_df = pd.DataFrame(history)
        
        if history_df.empty:
            return self._empty_result()
            
        # Calculate metrics
        total_bets = len(history_df)
        win_rate = won_bets / total_bets if total_bets > 0 else 0
        profit = bankroll - self.initial_bankroll
        roi = profit / total_staked if total_staked > 0 else 0
        
        # Max Drawdown
        peak = self.initial_bankroll
        max_dd = 0
        for b in bankroll_history:
            if b > peak:
                peak = b
            dd = (peak - b) / peak
            if dd > max_dd:
                max_dd = dd
                
        # Sharpe Ratio (simplified)
        returns = history_df['profit'] / self.initial_bankroll
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0
        
        return BacktestResult(
            total_bets=total_bets,
            won_bets=won_bets,
            win_rate=win_rate,
            total_staked=total_staked,
            total_return=total_return,
            profit=profit,
            roi=roi,
            final_bankroll=bankroll,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            history=history_df
        )
        
    def _empty_result(self) -> BacktestResult:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, self.initial_bankroll, 0, 0, pd.DataFrame())
