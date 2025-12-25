# AI Football Betting Predictor

An advanced prediction system for football matches using an ensemble of statistical models (Poisson) and machine learning (XGBoost).

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-61DAFB?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

---

> [!IMPORTANT]
> This repository contains the **Backend API**.
> The Frontend (React) can be found here: [betting-frontend](https://github.com/Cioske2/betting-frontend)

---

## Features

- **Hybrid Ensemble Model**: Combines Poisson distribution with XGBoost for high-precision 1X2 predictions.
- **ELO Rating System**: Dynamic rating system to evaluate the relative strength of teams in real-time.
- **League Standings**: Integration of official standings to consider positioning and motivation.
- **Margin Removal**: Algorithm to remove bookmaker overround and find "Fair Odds" (real odds).
- **Fractional Kelly Criterion**: Safe bankroll management using 1/4 Kelly to minimize risk.
- **Time-Decay Weighting**: Algorithm that gives more importance to recent results compared to past ones.
- **Value Bet Detection**: Identifies bets with positive expected value by comparing model probabilities and real odds.
- **Supabase Persistence**: Persistent storage of bets, selections, and statistics.

---

## Setup & Installation

### 1. Environment Configuration
Create a `.env` file in the root directory (see `.env.example` for reference). **Never commit your `.env` file to GitHub.**

```env
# API Keys
API_FOOTBALL_KEY=your_api_football_key_here
FOOTBALL_DATA_TOKEN=your_football_data_token_here

# Configuration
LEAGUE_IDS=39,140,135,78,61
POISSON_WEIGHT=0.4
XGBOOST_WEIGHT=0.6
```

### 2. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

## Architecture Overview

The system is built with a **FastAPI** backend, providing a robust API for match predictions and data analysis.

### Core Components

1. **Data Clients (`src/data/`)**:
   - `APIFootballClient`: Interfaces with api-football.com for historical data and odds.
   - `FootballDataClient`: Interfaces with football-data.org for current season matches and upcoming fixtures.

2. **Feature Engineering (`src/features/`)**:
   - `FeatureEngineer`: Transforms raw match data into 28+ mathematical features (form, strength, H2H).

3. **Models (`src/models/`)**:
   - `PoissonModel`: Statistical model based on goal distributions.
   - `XGBoostPredictor`: Machine learning model for complex pattern recognition.
   - `EnsemblePredictor`: Combines both models with weighted averaging.

4. **API Endpoints (`src/api/`)**:
   - RESTful endpoints for training, predictions, and bet management.
   - Background training with multi-worker synchronization.

5. **Database (`supabase/`)**:
   - PostgreSQL schema for persistence of bets and selections.
   - Automatic triggers for updating bet slip status.

---

## Prediction Logic

### 1. Poisson Distribution Model

**Concept**: Goals scored by teams follow a Poisson distribution.

**Calculation**:
- Calculates **Attack Strength** and **Defense Strength** for every team relative to league average.
- Estimates **Expected Goals (xG)** for both Home and Away teams.
- Uses the Poisson formula to calculate probability of every possible scoreline (0-0, 1-0, 0-1, etc.).
- Sums these probabilities to get 1X2 (Home/Draw/Away) outcomes.

**Formula**:
```
?_home = home_attack  away_defense  league_home_avg
?_away = away_attack  home_defense  league_away_avg
P(X = k) = (?^k  e^-?) / k!
```

**Strength**: Excellent for capturing fundamental statistical nature of football scoring.

---

### 2. XGBoost (Machine Learning)

**Concept**: Gradient Boosted Decision Tree model that analyzes over 30 advanced features.

**Features**:
- **ELO Rating**: Dynamic relative strength of teams.
- **League Rank**: Current position in the standings.
- **Time-Decay Form**: Points and goals in recent matches with higher weight on recent results.
- **H2H**: Historical performance between the two specific teams.
- **Home/Away Bias**: Performance differential between home and away.
- **Clean Sheet/BTTS rates**: Defensive and offensive consistency.

**Strength**: Captures non-linear relationships and "momentum" that simple statistics might ignore.

---

## Advanced Betting Logic

### 1. ELO Rating System
The system assigns a strength score to each team. After each match, points are exchanged between winner and loser based on the expected probability of the result. This allows identifying teams on the rise or decline before bookmakers update their odds.

### 2. Margin Removal (Fair Odds)
Bookmakers add a "margin" (overround) to the odds. Our algorithm removes this margin to find the **Real Probability** of the market, allowing an honest comparison with our AI model's probabilities.

### 3. Fractional Kelly Criterion
To protect the bankroll, the system suggests a stake based on the fractional Kelly criterion (1/4).
`Stake = (Probability * Odds - 1) / (Odds - 1) * 0.25`
This balances capital growth with protection against losing streaks.

---

### 3. The Ensemble

Final probability is a **weighted average**:
```
Final Probability = (Poisson  0.4) + (XGBoost  0.6)
```
*Weights are configurable in `.env`*

**Confidence Calculation**:
- **HIGH**: Both models agree (difference < 10%) AND max probability > 50%
- **MEDIUM**: Moderate agreement OR probability 40-50%
- **LOW**: Models disagree OR all probabilities < 40%

---

## Data Flow & Training

### Training Phase

1. System fetches **~2,500 matches** from the last 2 seasons across 5 leagues
2. `FeatureEngineer` processes matches (optimized with team-first filtering)
3. Models are trained and saved as `.pkl` files in `models/` directory
4. **Multi-Worker Sync**: Uses `training.lock` file to ensure all Gunicorn workers stay synchronized

### Prediction Phase

1. User selects leagues and timeframe
2. Backend fetches upcoming "SCHEDULED" matches
3. For each match, `EnsemblePredictor` generates detailed predictions.
4. Data is served via REST API to the integrated dashboard.

---

## Performance Optimizations

- **CPU Efficiency**: Team-first filtering reduces complexity from O(N) to O(N) for form calculations
- **Non-Blocking I/O**: Heavy training runs in separate threads (`asyncio.to_thread`) to keep API responsive
- **Caching**: API responses cached to respect rate limits (10 requests/min for football-data.org)
- **Gunicorn Timeout**: Increased to 600s for deep model training on cloud environments
- **Worker Synchronization**: File-based locks prevent race conditions in multi-worker deployments

---

## API Endpoints

### Core Endpoints

- `POST /api/predict` - Get 1X2 probabilities for a match
- `POST /api/value-bet` - Analyze betting value with Kelly Criterion
- `POST /api/train-current` - Train models on latest data (background task)
- `GET /api/health` - Check system status & model readiness
- `GET /api/upcoming-matches` - Fetch scheduled matches with filters
- `GET /api/leagues` - List supported leagues
- `GET /api/standings/{league_id}` - League table
- `GET /api/team/{team_id}/stats` - Team statistics
- `GET /api/h2h/{team1_id}/{team2_id}` - Head-to-head
- `GET /api/odds/{fixture_id}` - Bookmaker odds

### Interactive Documentation

Visit `/docs` for Swagger UI or `/redoc` for ReDoc documentation when server is running.

---

## Value Bet Detection

A bet has value when:
```
model_probability > implied_probability_from_odds
```

**Expected Value Calculation**:
```
EV = (probability  odds) - 1
```

**Kelly Criterion for Stake Sizing**:
```
stake% = [(odds  probability - 1) / (odds - 1)]  kelly_fraction
```
*Default kelly_fraction = 0.25 (conservative)*

---

## Supported Leagues

| ID  | League          | Country | Season Coverage |
|-----|----------------|---------|----------------|
| 39  | Premier League | England | 2025-2026      |
| 140 | La Liga        | Spain   | 2025-2026      |
| 135 | Serie A        | Italy   | 2025-2026      |
| 78  | Bundesliga     | Germany | 2025-2026      |
| 61  | Ligue 1        | France  | 2025-2026      |

---

## Configuration

All configuration is done via environment variables in `.env`:

```env
# Required API Keys
API_FOOTBALL_KEY=your_key_here
FOOTBALL_DATA_TOKEN=your_token_here

# Model Weights (must sum to 1.0)
POISSON_WEIGHT=0.4
XGBOOST_WEIGHT=0.6

# Value Bet Settings
MIN_EDGE=0.05           # Minimum 5% edge to recommend
KELLY_FRACTION=0.25     # Conservative Kelly (1/4)

# Leagues (comma-separated IDs)
LEAGUE_IDS=39,140,135,78,61

# Feature Engineering
FORM_MATCHES=5          # Last N matches for form
H2H_MATCHES=10          # H2H history depth

# Server
HOST=0.0.0.0
PORT=8000
```

---

## Deployment

### Railway / Render / Heroku

1. Set environment variables in platform dashboard
2. Ensure `Procfile` uses Gunicorn with 600s timeout:
   ```
   web: gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 600 --bind 0.0.0.0:$PORT
   ```
3. Deploy from GitHub repository

---

## License

MIT License - Feel free to use and modify.

---

## Disclaimer

**For educational and informational purposes only.** 

This tool provides statistical predictions based on historical data and mathematical models. It does not guarantee outcomes. Sports betting involves financial risk. Always bet responsibly and within your means.

**The developers are not responsible for any financial losses incurred from using this tool.**
