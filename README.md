# AI Football Betting Predictor

Advanced prediction system for football matches using Poisson + XGBoost ensemble models.

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-61DAFB?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

---

## Features

### Core Prediction
- **Hybrid Ensemble Model**: Poisson + XGBoost with dynamic weighting
- **Dynamic Model Weights**: Weights adjust per-match based on model confidence
- **ELO Rating System**: Real-time team strength evaluation with home advantage
- **42 Engineered Features**: Form, strength, H2H, weighted streaks, crisis index

### Betting Intelligence
- **Safe Accumulator**: Value-weighted picks for next matchday (7 days)
- **Double Chance Odds**: 1X, X2, 12 markets from The Odds API
- **Weighted Streaks**: Streak value weighted by opponent strength
- **Value Bet Detection**: Identifies positive EV opportunities
- **Kelly Criterion**: Conservative 1/4 Kelly for stake sizing

### Performance
- **Redis Caching**: Predictions (10min), Odds (5min), Standings (1hr)
- **In-Memory Fallback**: Works without Redis
- **Multi-Worker Sync**: File-based locks for Gunicorn deployments

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
python main.py
```

---

## Configuration

```env
# Required API Keys
FOOTBALL_DATA_TOKEN=your_token
ODDS_API_KEY=your_key

# Optional
REDIS_URL=redis://localhost:6379  # Leave empty for in-memory cache
POISSON_WEIGHT=0.2
XGBOOST_WEIGHT=0.8
LEAGUE_IDS=39,140,135,78,61
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/upcoming-matches` | Predictions with odds |
| `GET /api/v1/recommendations/safe-accumulator` | AI curated safe picks |
| `POST /api/train-current` | Train models (background) |
| `GET /api/health` | System & model status |
| `GET /docs` | Swagger UI |

---

## Project Structure

```
betting/
├── main.py              # FastAPI app entry
├── src/
│   ├── api/             # REST endpoints
│   ├── data/            # API clients, cache
│   ├── features/        # Feature engineering
│   ├── models/          # Poisson, XGBoost, Ensemble
│   └── betting/         # Value bet logic
├── tests/               # Test suite
└── betting-frontend/    # React UI
```

---

## Supported Leagues

| ID | League | Country |
|----|--------|---------|
| 39 | Premier League | England |
| 140 | La Liga | Spain |
| 135 | Serie A | Italy |
| 78 | Bundesliga | Germany |
| 61 | Ligue 1 | France |

---

## License

MIT License

---

## Disclaimer

**For educational purposes only.** Sports betting involves financial risk. The developers are not responsible for any losses.
