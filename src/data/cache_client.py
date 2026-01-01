"""
Redis Cache Client for Betting Application.

Provides caching for:
- Odds data (TTL: 5 minutes)
- Predictions (TTL: 10 minutes)
- Standings (TTL: 1 hour)

Falls back gracefully if Redis is unavailable.
"""

import json
import logging
from typing import Optional, Any, Dict
from datetime import timedelta

logger = logging.getLogger(__name__)

# Default TTLs in seconds
TTL_ODDS = 300  # 5 minutes
TTL_PREDICTIONS = 600  # 10 minutes
TTL_STANDINGS = 3600  # 1 hour
TTL_FIXTURES = 1800  # 30 minutes


class CacheClient:
    """
    Redis-based cache client with in-memory fallback.
    
    If Redis is not available, uses a simple dict cache.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the cache client.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self._redis = None
        self._memory_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, float] = {}
        
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self._redis.ping()
                logger.info(f"Redis cache connected: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable, using memory cache: {e}")
                self._redis = None
        else:
            logger.info("No Redis URL configured, using memory cache")
    
    def _cache_key(self, prefix: str, *args) -> str:
        """Build a cache key from prefix and arguments."""
        parts = [prefix] + [str(a) for a in args]
        return ":".join(parts)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            if self._redis:
                value = self._redis.get(key)
                if value:
                    return json.loads(value)
            else:
                # Memory cache with expiry check
                import time
                if key in self._memory_cache:
                    if self._cache_expiry.get(key, 0) > time.time():
                        return self._memory_cache[key]
                    else:
                        del self._memory_cache[key]
                        del self._cache_expiry[key]
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = TTL_PREDICTIONS) -> bool:
        """Set a value in cache with TTL."""
        try:
            if self._redis:
                self._redis.setex(key, ttl, json.dumps(value))
                return True
            else:
                # Memory cache
                import time
                self._memory_cache[key] = value
                self._cache_expiry[key] = time.time() + ttl
                return True
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            if self._redis:
                self._redis.delete(key)
            elif key in self._memory_cache:
                del self._memory_cache[key]
                del self._cache_expiry[key]
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
        return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        count = 0
        try:
            if self._redis:
                for key in self._redis.scan_iter(pattern):
                    self._redis.delete(key)
                    count += 1
            else:
                keys_to_delete = [k for k in self._memory_cache if pattern.replace("*", "") in k]
                for k in keys_to_delete:
                    del self._memory_cache[k]
                    del self._cache_expiry[k]
                    count += 1
        except Exception as e:
            logger.warning(f"Cache clear pattern failed: {e}")
        return count
    
    # Convenience methods for specific data types
    
    async def get_odds(self, league_ids: str) -> Optional[Dict]:
        """Get cached odds for leagues."""
        key = self._cache_key("odds", league_ids)
        return await self.get(key)
    
    async def set_odds(self, league_ids: str, odds_data: Dict) -> bool:
        """Cache odds data."""
        key = self._cache_key("odds", league_ids)
        return await self.set(key, odds_data, TTL_ODDS)
    
    async def get_predictions(self, league_ids: str, days: int) -> Optional[Dict]:
        """Get cached predictions."""
        key = self._cache_key("predictions", league_ids, days)
        return await self.get(key)
    
    async def set_predictions(self, league_ids: str, days: int, predictions: Dict) -> bool:
        """Cache predictions."""
        key = self._cache_key("predictions", league_ids, days)
        return await self.set(key, predictions, TTL_PREDICTIONS)
    
    async def get_standings(self, league_id: int) -> Optional[list]:
        """Get cached standings."""
        key = self._cache_key("standings", league_id)
        return await self.get(key)
    
    async def set_standings(self, league_id: int, standings: list) -> bool:
        """Cache standings."""
        key = self._cache_key("standings", league_id)
        return await self.set(key, standings, TTL_STANDINGS)


# Singleton
_cache_client: Optional[CacheClient] = None


def get_cache_client() -> CacheClient:
    """Get or create the cache client singleton."""
    global _cache_client
    if _cache_client is None:
        from ..config import get_settings
        settings = get_settings()
        redis_url = getattr(settings, 'redis_url', None)
        _cache_client = CacheClient(redis_url)
    return _cache_client
