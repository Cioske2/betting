import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.poisson_model import PoissonModel
import json

def test_poisson_logic():
    print("Testing Poisson Model Logic...")
    model = PoissonModel()
    
    # Mock last 5 matches for Home Team (Strong Home)
    home_last_5 = [
        {"home_team_id": 1, "away_team_id": 10, "home_goals": 3, "away_goals": 0},
        {"home_team_id": 20, "away_team_id": 1, "home_goals": 1, "away_goals": 2},
        {"home_team_id": 1, "away_team_id": 30, "home_goals": 4, "away_goals": 1},
        {"home_team_id": 40, "away_team_id": 1, "home_goals": 0, "away_goals": 1},
        {"home_team_id": 1, "away_team_id": 50, "home_goals": 2, "away_goals": 0},
    ]
    
    # Mock last 5 matches for Away Team (Weak Away)
    away_last_5 = [
        {"home_team_id": 2, "away_team_id": 60, "home_goals": 0, "away_goals": 2},
        {"home_team_id": 70, "away_team_id": 2, "home_goals": 3, "away_goals": 0},
        {"home_team_id": 2, "away_team_id": 80, "home_goals": 1, "away_goals": 1},
        {"home_team_id": 90, "away_team_id": 2, "home_goals": 2, "away_goals": 0},
        {"home_team_id": 2, "away_team_id": 100, "home_goals": 0, "away_goals": 4},
    ]
    
    prediction = model.predict(1, home_last_5, 2, away_last_5)
    result = prediction.to_dict()
    
    print(f"Prediction result: {json.dumps(result, indent=2)}")
    
    assert "1x2" in result
    assert "goals" in result
    assert "btts" in result
    assert result["expected_goals"]["home"] > result["expected_goals"]["away"]
    print("Poisson Model Logic: OK")

if __name__ == "__main__":
    try:
        test_poisson_logic()
    except Exception as e:
        print(f"Verification FAILED: {e}")
        sys.exit(1)
