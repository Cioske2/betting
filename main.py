"""
Football Betting Prediction API - Main Entry Point

Starts the FastAPI server with all endpoints.
Run with: python main.py
Or: uvicorn main:app --reload
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.config import get_settings
from src.api.endpoints import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("🚀 Starting Football Betting Prediction API...")
    settings = get_settings()
    
    if not settings.api_football_key:
        logger.warning(
            "⚠️  API_FOOTBALL_KEY not set! "
            "Data fetching will not work. "
            "Set it in .env file or environment variables."
        )
    else:
        logger.info("✅ API-Football key configured")
    
    logger.info(f"📊 Configured leagues: {settings.league_ids}")
    logger.info(f"⚖️  Model weights: Poisson={settings.poisson_weight}, XGBoost={settings.xgboost_weight}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Football Betting Prediction API",
    description="""
## 🎯 Predict football match outcomes with AI

This API provides:
- **1X2 Predictions**: Probability estimates for Home Win, Draw, and Away Win
- **Value Bet Detection**: Find bets with positive expected value
- **Multiple Models**: Poisson distribution + XGBoost ensemble
- **Live Data**: Integration with API-Football for real-time stats

### Getting Started
1. Set your `API_FOOTBALL_KEY` in `.env`
2. Train models with `POST /api/train`
3. Get predictions with `POST /api/predict`
4. Analyze value bets with `POST /api/value-bet`

### Supported Leagues
- Premier League (39)
- La Liga (140)
- Serie A (135)
- Bundesliga (78)
- Ligue 1 (61)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Football Betting Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    settings = get_settings()
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )
