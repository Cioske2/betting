# Football Betting Prediction API 🎯

A Python-based system for predicting football match outcomes (1X2) using statistical models (Poisson) and machine learning (XGBoost), with value bet detection using Kelly Criterion.

## Features

- **🔮 Match Predictions**: Accurate 1X2 probability estimates
- **📊 Dual Models**: Poisson distribution + XGBoost ensemble
- **💰 Value Bet Detection**: Find positive EV opportunities
- **📈 Kelly Criterion**: Optimal stake sizing
- **🌐 REST API**: FastAPI with OpenAPI documentation
- **⚽ Live Data**: Integration with API-Football

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure API key
copy .env.example .env
# Edit .env and add your API-Football key
```

### 2. Get API Key

1. Go to [api-football.com](https://www.api-football.com/)
2. Create a free account (100 requests/day free)
3. Copy your API key
4. Paste in `.env` file: `API_FOOTBALL_KEY=your_key_here`

### 3. Run the API

```bash
python main.py
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive documentation.

## API Endpoints

### Predictions

```bash
# Get match prediction
POST /api/predict
{
    "home_team_id": 33,
    "away_team_id": 34,
    "home_team_name": "Manchester United",
    "away_team_name": "Newcastle",
    "league_id": 39
}

# Response
{
    "probabilities": {
        "home_win": 45.2,
        "draw": 27.3,
        "away_win": 27.5
    },
    "prediction": "1",
    "confidence": "MEDIUM"
}
```

### Value Bet Analysis

```bash
# Analyze betting value
POST /api/value-bet
{
    "home_team_id": 33,
    "away_team_id": 34,
    "home_team_name": "Manchester United",
    "away_team_name": "Newcastle",
    "league_id": 39,
    "odds": {"1": 2.10, "X": 3.40, "2": 3.50}
}

# Response
{
    "recommendation": "Value bet on HOME WIN @ 2.10 (7.2% edge)",
    "best_value": {
        "outcome": "1",
        "edge_percent": "7.2%",
        "suggested_stake_percent": "3.1%"
    }
}
```

### Other Endpoints

- `GET /api/health` - System status
- `GET /api/leagues` - Available leagues
- `GET /api/fixtures/upcoming?league_id=39` - Upcoming matches
- `GET /api/standings/{league_id}` - League table
- `GET /api/team/{team_id}/stats` - Team statistics
- `GET /api/h2h/{team1_id}/{team2_id}` - Head-to-head
- `GET /api/odds/{fixture_id}` - Bookmaker odds
- `POST /api/train` - Train models

## Supported Leagues

| ID | League | Country |
|---|---|---|
| 39 | Premier League | England |
| 140 | La Liga | Spain |
| 135 | Serie A | Italy |
| 78 | Bundesliga | Germany |
| 61 | Ligue 1 | France |

## Project Structure

```
betting/
├── src/
│   ├── config.py              # Configuration
│   ├── data/
│   │   └── api_football_client.py  # API client
│   ├── features/
│   │   └── feature_engineering.py  # Feature calculation
│   ├── models/
│   │   ├── poisson_model.py   # Poisson model
│   │   ├── xgboost_model.py   # XGBoost model
│   │   └── ensemble.py        # Combined model
│   ├── betting/
│   │   └── value_bet.py       # Value bet logic
│   └── api/
│       └── endpoints.py       # FastAPI routes
├── main.py                    # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Model Details

### Poisson Model

Uses Poisson distribution to model the number of goals scored:
- λ_home = home_attack × away_defense × league_home_avg
- λ_away = away_attack × home_defense × league_away_avg

Generates probability matrix for all scorelines, then aggregates to 1X2.

### XGBoost Model

Gradient boosting classifier with:
- 28 engineered features (form, h2h, strength ratings)
- Probability calibration (isotonic regression)
- Cross-validation for robust evaluation

### Ensemble

Weighted average of both models (default: 40% Poisson, 60% XGBoost).
Confidence based on model agreement and probability margins.

## Value Bet Logic

A bet has value when:
```
model_probability > implied_probability_from_odds
```

Expected Value: `EV = (probability × odds) - 1`

Kelly Criterion for stake sizing:
```
stake = (b×p - q) / b × kelly_fraction
where b = odds-1, p = win probability, q = 1-p
```

## Configuration

Environment variables (set in `.env`):

```bash
# Required
API_FOOTBALL_KEY=your_api_key

# Optional
POISSON_WEIGHT=0.4
XGBOOST_WEIGHT=0.6
MIN_EDGE=0.05
KELLY_FRACTION=0.25
LEAGUES=39,140,135,78,61
```

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload

# Run tests
pytest tests/ -v
```

## License

MIT License - Feel free to use and modify.

## Disclaimer

⚠️ **For educational purposes only.** Sports betting involves financial risk. This tool provides statistical predictions but cannot guarantee outcomes. Bet responsibly.
