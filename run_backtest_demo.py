
import pandas as pd
import numpy as np
from src.backtesting.engine import BacktestEngine
from src.betting.value_bet import ValueBetAnalyzer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_demo_backtest():
    """
    Runs a demonstration backtest with synthetic data 
    to show how the engine works.
    """
    logger.info("Starting Demo Backtest...")
    
    # 1. Create synthetic data
    n_matches = 100
    fixture_ids = range(1, n_matches + 1)
    
    # Results: 1, X, 2
    results = np.random.choice(['1', 'X', '2'], size=n_matches, p=[0.45, 0.25, 0.30])
    
    matches_df = pd.DataFrame({
        'fixture_id': fixture_ids,
        'result': results,
        'home_team': [f"Team A{i}" for i in range(n_matches)],
        'away_team': [f"Team B{i}" for i in range(n_matches)],
        'date': pd.date_range(start='2025-01-01', periods=n_matches)
    })
    
    # Model predictions (slightly better than random)
    predictions_df = pd.DataFrame({
        'fixture_id': fixture_ids,
        'prob_home': np.random.uniform(0.3, 0.6, size=n_matches),
        'prob_draw': np.random.uniform(0.2, 0.3, size=n_matches),
        'prob_away': np.random.uniform(0.2, 0.4, size=n_matches)
    })
    # Normalize probabilities
    predictions_df[['prob_home', 'prob_draw', 'prob_away']] = predictions_df[['prob_home', 'prob_draw', 'prob_away']].div(predictions_df[['prob_home', 'prob_draw', 'prob_away']].sum(axis=1), axis=0)
    
    # Odds (with margin)
    odds_df = pd.DataFrame({
        'fixture_id': fixture_ids,
        'odds_home': 1.0 / (predictions_df['prob_home'] * 0.9), # 10% margin
        'odds_draw': 1.0 / (predictions_df['prob_draw'] * 0.9),
        'odds_away': 1.0 / (predictions_df['prob_away'] * 0.9)
    })
    
    # 2. Initialize Engine
    analyzer = ValueBetAnalyzer(min_edge=0.05, kelly_fraction=0.25)
    engine = BacktestEngine(initial_bankroll=1000.0, analyzer=analyzer)
    
    # 3. Run Backtest
    result = engine.run(matches_df, predictions_df, odds_df)
    
    # 4. Print Results
    print("\n" + "="*30)
    print("BACKTEST RESULTS")
    print("="*30)
    print(f"Total Bets:     {result.total_bets}")
    print(f"Win Rate:       {result.win_rate:.1%}")
    print(f"Total Staked:   ${result.total_staked:.2f}")
    print(f"Profit/Loss:    ${result.profit:.2f}")
    print(f"ROI:            {result.roi:.1%}")
    print(f"Final Bankroll: ${result.final_bankroll:.2f}")
    print(f"Max Drawdown:   {result.max_drawdown:.1%}")
    print("="*30)

if __name__ == "__main__":
    run_demo_backtest()
