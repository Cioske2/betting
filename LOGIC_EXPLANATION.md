# Football Betting Prediction API - Logic & Architecture

This document explains the internal logic, data flow, and architectural decisions of the Football Betting Prediction API.

## 🏗️ Architecture Overview

The system is built with a **FastAPI** backend and a **React (Vite)** frontend. It uses a hybrid modeling approach to predict football match outcomes.

### Core Components

1.  **Data Clients (`src/data/`)**:
    *   `APIFootballClient`: Interfaces with api-football.com (used for historical data and odds).
    *   `FootballDataClient`: Interfaces with football-data.org (used for current season matches and upcoming fixtures).
2.  **Feature Engineering (`src/features/`)**:
    *   `FeatureEngineer`: Transforms raw match data into mathematical features (form, strength, H2H).
3.  **Models (`src/models/`)**:
    *   `PoissonModel`: Statistical model based on goal distributions.
    *   `XGBoostPredictor`: Machine learning model for complex pattern recognition.
    *   `EnsemblePredictor`: Combines both models for the final prediction.
4.  **API Endpoints (`src/api/`)**:
    *   RESTful endpoints for training, predicting, and health checks.

---

## 🧠 Prediction Logic

The system uses an **Ensemble** of two distinct mathematical approaches:

### 1. Poisson Distribution Model
*   **Concept**: Assumes that goals scored by teams follow a Poisson distribution.
*   **Calculation**:
    *   Calculates **Attack Strength** and **Defense Strength** for every team relative to the league average.
    *   Estimates the "Expected Goals" (xG) for both Home and Away teams.
    *   Uses the Poisson formula to calculate the probability of every possible scoreline (0-0, 1-0, 0-1, etc.).
    *   Sums these probabilities to get the 1X2 (Home/Draw/Away) outcomes.
*   **Strength**: Excellent for capturing the fundamental statistical nature of football scoring.

### 2. XGBoost (Machine Learning)
*   **Concept**: A Gradient Boosted Decision Tree model that looks at 28+ different features.
*   **Features**:
    *   **Form**: Points and goals in the last 5 matches.
    *   **H2H**: Historical performance between the two specific teams.
    *   **Home/Away Bias**: How much better/worse a team performs at home vs. away.
    *   **Clean Sheet/BTTS rates**: Defensive and offensive consistency.
*   **Strength**: Captures non-linear relationships and "momentum" that simple statistics might miss.

### 3. The Ensemble
The final probability is a weighted average:
`Final Probability = (Poisson * 0.4) + (XGBoost * 0.6)`
*Weights are configurable in `.env`.*

---

## 🔄 Data Flow & Training

1.  **Training Phase**:
    *   The system fetches ~2,500 matches from the last 2 seasons across 5 leagues.
    *   `FeatureEngineer` processes these matches (optimized with team-first filtering).
    *   Models are trained and saved as `.pkl` files in the `models/` directory.
    *   **Multi-Worker Sync**: Uses a `training.lock` file to ensure all Gunicorn workers stay in sync during background training.

2.  **Prediction Phase**:
    *   User selects a league and timeframe.
    *   Backend fetches upcoming "SCHEDULED" matches.
    *   For each match, the `EnsemblePredictor` generates probabilities and xG.
    *   The frontend displays these with a "Predicted Score" based on the xG.

---

## 🛠️ Performance Optimizations

*   **CPU Efficiency**: Feature engineering uses a "Team-First" filtering strategy, reducing complexity from $O(N^2)$ to $O(N)$ for form calculations.
*   **Non-Blocking I/O**: Heavy training tasks run in separate threads (`asyncio.to_thread`) to keep the API responsive.
*   **Caching**: API responses from football-data.org are cached to respect rate limits (10 requests/min).
*   **Gunicorn Timeout**: Increased to 600s to allow for deep model training on cloud environments.
