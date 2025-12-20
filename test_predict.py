"""Test prediction locally after training."""
import asyncio
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.data.api_football_client import get_client
from src.features.feature_engineering import FeatureEngineer
from src.models.ensemble import EnsemblePredictor

async def test_prediction():
    print("=" * 50)
    print("Testing Full Pipeline")
    print("=" * 50)
    
    client = get_client()
    
    # 1. Fetch matches
    print("\n1. Fetching Serie A 2023 matches...")
    matches = await client.get_finished_matches(league_id=135, season=2023)
    print(f"   Got {len(matches)} matches")
    
    if len(matches) < 50:
        print("   Not enough matches!")
        return
    
    # 2. Create feature engineer
    print("\n2. Setting up feature engineer...")
    feature_eng = FeatureEngineer(form_matches=5)
    feature_eng.load_matches(matches)
    
    # 3. Prepare training data
    print("\n3. Preparing training data...")
    matches_df = pd.DataFrame([
        {
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "league_id": m.league_id
        }
        for m in matches
        if m.home_goals is not None and m.away_goals is not None
    ])
    print(f"   Training data shape: {matches_df.shape}")
    
    # 4. Train ensemble
    print("\n4. Training models...")
    ensemble = EnsemblePredictor(poisson_weight=0.4, xgboost_weight=0.6)
    
    # Fit Poisson
    print("   Training Poisson model...")
    ensemble.fit_poisson(matches_df)
    print(f"   Poisson fitted: {ensemble.poisson.is_fitted}")
    
    # Fit XGBoost
    print("   Preparing XGBoost features...")
    X, y = feature_eng.get_training_data(matches)
    print(f"   Features shape: {X.shape}, Target shape: {y.shape}")
    
    print("   Training XGBoost model...")
    ensemble.fit_xgboost(X, y)
    print(f"   XGBoost fitted: {ensemble.xgboost.is_fitted}")
    
    # 5. Make prediction
    print("\n5. Making prediction for Juventus vs Inter...")
    try:
        prediction = ensemble.predict(
            home_team_id=496,
            away_team_id=505,
            home_team_name="Juventus",
            away_team_name="Inter",
            league_id=135
        )
        
        print(f"\n{'='*50}")
        print("PREDICTION RESULT:")
        print(f"{'='*50}")
        print(f"Home Win (1): {prediction.home_win_pct:.1f}%")
        print(f"Draw (X):     {prediction.draw_pct:.1f}%")
        print(f"Away Win (2): {prediction.away_win_pct:.1f}%")
        print(f"Prediction:   {prediction.predicted_outcome}")
        print(f"Confidence:   {prediction.confidence}")
        print(f"Expected Goals: {prediction.expected_home_goals:.2f} - {prediction.expected_away_goals:.2f}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_prediction())
