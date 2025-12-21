# ⚽ AI Football Betting Predictor

Un sistema avanzato di predizione per partite di calcio che utilizza un ensemble di modelli statistici (Poisson) e machine learning (XGBoost).

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![XGBoost](https://img.shields.io/badge/XGBoost-61DAFB?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

---

## 🚀 Features

- **🎯 Hybrid Ensemble Model**: Combina distribuzione di Poisson con XGBoost per predizioni 1X2 ad alta precisione
- **📡 Real-time Data**: Integrazione con `football-data.org` e `api-football.com`
- **🤖 Automated Training**: Processo di training in background con sincronizzazione multi-worker
- **📊 Advanced Features**: Calcola forma delle squadre, forza attacco/difesa, statistiche H2H
- **💎 Value Bet Detection**: Identifica scommesse con valore positivo atteso usando Kelly Criterion
- **🎨 Modern UI**: Dashboard responsive con React, Vite e Tailwind CSS

---## 🛠️ Setup & Installation

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

### 3. Frontend Setup
```bash
cd betting-frontend
npm install
npm run dev
```

## 🏗️ Architecture Overview

The system is built with a **FastAPI** backend and a **React (Vite)** frontend, using a hybrid modeling approach for match predictions.

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
   - RESTful endpoints for training, predicting, and health checks.
   - Background training with multi-worker synchronization via file locks.

---

## 🧠 Prediction Logic

### 1. Poisson Distribution Model

**Concept**: Goals scored by teams follow a Poisson distribution.

**Calculation**:
- Calculates **Attack Strength** and **Defense Strength** for every team relative to league average.
- Estimates **Expected Goals (xG)** for both Home and Away teams.
- Uses the Poisson formula to calculate probability of every possible scoreline (0-0, 1-0, 0-1, etc.).
- Sums these probabilities to get 1X2 (Home/Draw/Away) outcomes.

**Formula**:
```
λ_home = home_attack × away_defense × league_home_avg
λ_away = away_attack × home_defense × league_away_avg
P(X = k) = (λ^k × e^-λ) / k!
```

**Strength**: Excellent for capturing fundamental statistical nature of football scoring.

---

### 2. XGBoost (Machine Learning)

**Concept**: Gradient Boosted Decision Tree model analyzing 28+ features.

**Features**:
- **Form**: Points and goals in the last 5 matches
- **H2H**: Historical performance between the two specific teams
- **Home/Away Bias**: Performance differential by venue
- **Clean Sheet/BTTS rates**: Defensive and offensive consistency
- **Attack/Defense Strength**: Relative to league average

**Strength**: Captures non-linear relationships and momentum that simple statistics might miss.

---

### 3. The Ensemble

Final probability is a **weighted average**:
```
Final Probability = (Poisson × 0.4) + (XGBoost × 0.6)
```
*Weights are configurable in `.env`*

**Confidence Calculation**:
- **HIGH**: Both models agree (difference < 10%) AND max probability > 50%
- **MEDIUM**: Moderate agreement OR probability 40-50%
- **LOW**: Models disagree OR all probabilities < 40%

---

## 🔄 Data Flow & Training

### Training Phase

1. System fetches **~2,500 matches** from the last 2 seasons across 5 leagues
2. `FeatureEngineer` processes matches (optimized with team-first filtering)
3. Models are trained and saved as `.pkl` files in `models/` directory
4. **Multi-Worker Sync**: Uses `training.lock` file to ensure all Gunicorn workers stay synchronized

### Prediction Phase

1. User selects leagues and timeframe
2. Backend fetches upcoming "SCHEDULED" matches
3. For each match, `EnsemblePredictor` generates:
   - Win/Draw/Loss probabilities
   - Expected Goals (xG)
   - Predicted scoreline
   - Confidence level
4. Frontend displays predictions sorted by date

---

## ⚡ Performance Optimizations

- **CPU Efficiency**: Team-first filtering reduces complexity from O(N²) to O(N) for form calculations
- **Non-Blocking I/O**: Heavy training runs in separate threads (`asyncio.to_thread`) to keep API responsive
- **Caching**: API responses cached to respect rate limits (10 requests/min for football-data.org)
- **Gunicorn Timeout**: Increased to 600s for deep model training on cloud environments
- **Worker Synchronization**: File-based locks prevent race conditions in multi-worker deployments

---

## 📂 Project Structure

## 📊 API Endpoints

### Core Endpoints

- `POST /api/predict` - Get 1X2 probabilities for a match
- `POST /api/value-bet` - Analyze betting value with Kelly Criterion
- `POST /api/train-current` - Train models on latest data (background task)
- `GET /api/health` - Check system status & model readiness
- `GET /api/upcoming-matches` - Fetch scheduled matches with filters
- `GET /api/leagues` - List supported leagues

### Interactive Documentation

Visit `/docs` for Swagger UI or `/redoc` for ReDoc documentation when server is running.

---

## 🎯 Value Bet Detection

A bet has value when:
```
model_probability > implied_probability_from_odds
```

**Expected Value Calculation**:
```
EV = (probability × odds) - 1
```

**Kelly Criterion for Stake Sizing**:
```
stake% = [(odds × probability - 1) / (odds - 1)] × kelly_fraction
```
*Default kelly_fraction = 0.25 (conservative)*

---

## 🌍 Supported Leagues

| ID  | League          | Country | Season Coverage |
|-----|----------------|---------|----------------|
| 39  | Premier League | England | 2024-2025      |
| 140 | La Liga        | Spain   | 2024-2025      |
| 135 | Serie A        | Italy   | 2024-2025      |
| 78  | Bundesliga     | Germany | 2024-2025      |
| 61  | Ligue 1        | France  | 2024-2025      |

---

## 🔧 Configuration

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

## 🚀 Deployment

### Railway / Render / Heroku

1. Set environment variables in platform dashboard
2. Ensure `Procfile` uses Gunicorn with 600s timeout:
   ```
   web: gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 600 --bind 0.0.0.0:$PORT
   ```
3. Deploy from GitHub repository

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "600", "--bind", "0.0.0.0:8000"]
```

---

## 📈 Development

```bash
# Run with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests (if implemented)
pytest tests/ -v
```

---

## ⚖️ License

MIT License - Feel free to use and modify.

---

## ⚠️ Disclaimer

**For educational and informational purposes only.** 

This tool provides statistical predictions based on historical data and mathematical models. It does not guarantee outcomes. Sports betting involves financial risk. Always bet responsibly and within your means.

**The developers are not responsible for any financial losses incurred from using this tool.**
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



