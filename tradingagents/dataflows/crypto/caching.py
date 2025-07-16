"""
Caching utilities for crypto data adapters.

Provides multi-level caching with Redis as primary and filesystem as fallback.
Supports TTL management, cache invalidation, and compression for large datasets.
"""

import os
import json
import pickle
import hashlib
import logging
import asyncio
import aiofiles
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    redis_url: str = "redis://localhost:6379/0"
    use_redis: bool = True
    filesystem_cache_dir: str = "./cache"
    max_cache_size_mb: int = 500
    default_ttl_seconds: int = 300  # 5 minutes
    compress_threshold_bytes: int = 1024  # Compress data larger than 1KB
    key_prefix: str = "tradingagents:crypto:"


class CacheKey:
    """Utility for generating consistent cache keys."""
    
    @staticmethod
    def make_key(provider: str, method: str, **params) -> str:
        """Generate a cache key from provider, method, and parameters."""
        # Sort parameters for consistent key generation
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        key_data = f"{provider}:{method}:{param_str}"
        
        # Hash long keys to avoid Redis key length limits
        if len(key_data) > 200:
            key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            key_data = f"{provider}:{method}:hash_{key_hash}"
        
        return key_data
    
    @staticmethod
    def make_ohlcv_key(provider: str, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """Generate cache key for OHLCV data."""
        return CacheKey.make_key(
            provider, "ohlcv",
            symbol=symbol,
            start=start_date,
            end=end_date,
            interval=interval
        )
    
    @staticmethod
    def make_price_key(provider: str, symbol: str) -> str:
        """Generate cache key for latest price data."""
        return CacheKey.make_key(provider, "price", symbol=symbol)
    
    @staticmethod
    def make_metadata_key(provider: str, symbol: str) -> str:
        """Generate cache key for asset metadata."""
        return CacheKey.make_key(provider, "metadata", symbol=symbol)


class FilesystemCache:
    """Filesystem-based cache implementation."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get filesystem path for cache key."""
        # Use first two characters as subdirectory to avoid too many files in one dir
        subdir = key[:2] if len(key) >= 2 else "default"
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{key}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from filesystem cache."""
        try:
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return None
            
            async with aiofiles.open(cache_path, 'rb') as f:
                data = await f.read()
            
            # Load the cached data (includes value and metadata)
            cache_data = pickle.loads(data)
            
            # Check if expired
            if cache_data.get('expires_at') and datetime.now() > cache_data['expires_at']:
                await self.delete(key)
                return None
            
            return cache_data.get('value')
            
        except Exception as e:
            logger.warning(f"Failed to read from filesystem cache for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in filesystem cache."""
        try:
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            cache_data = {
                'value': value,
                'cached_at': datetime.now(),
                'expires_at': expires_at,
                'key': key
            }
            
            cache_path = self._get_cache_path(key)
            
            # Write to temporary file first, then rename for atomicity
            temp_path = cache_path.with_suffix('.tmp')
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(pickle.dumps(cache_data))
            
            # Atomic rename
            temp_path.rename(cache_path)
            
            # Clean up old cache files if size limit exceeded
            await self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to write to filesystem cache for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from filesystem cache."""
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            return True
        except Exception as e:
            logger.warning(f"Failed to delete from filesystem cache for key {key}: {e}")
            return False
    
    async def _cleanup_if_needed(self):
        """Clean up old cache files if size limit exceeded."""
        try:
            total_size = 0
            cache_files = []
            
            for cache_file in self.cache_dir.rglob("*.cache"):
                stat = cache_file.stat()
                total_size += stat.st_size
                cache_files.append((cache_file, stat.st_mtime))
            
            # Convert to MB
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > self.max_size_mb:
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda x: x[1])
                
                # Remove oldest files until under limit
                target_size = self.max_size_mb * 0.8  # Target 80% of limit
                for cache_file, _ in cache_files:
                    cache_file.unlink()
                    stat = cache_file.stat()
                    total_size_mb -= stat.st_size / (1024 * 1024)
                    
                    if total_size_mb <= target_size:
                        break
                        
                logger.info(f"Cleaned up filesystem cache, reduced from {total_size_mb:.1f}MB to {total_size_mb:.1f}MB")
                
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str, key_prefix: str = ""):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: Optional[redis.Redis] = None
        self._connected = False
    
    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is established."""
        if not REDIS_AVAILABLE:
            return False
            
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url)
                await self._redis.ping()
                self._connected = True
                logger.debug("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._connected = False
                return False
        
        return self._connected
    
    def _make_redis_key(self, key: str) -> str:
        """Add prefix to cache key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not await self._ensure_connected():
            return None
            
        try:
            redis_key = self._make_redis_key(key)
            data = await self._redis.get(redis_key)
            
            if data is None:
                return None
            
            # Deserialize the data
            return pickle.loads(data)
            
        except Exception as e:
            logger.warning(f"Failed to read from Redis cache for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not await self._ensure_connected():
            return False
            
        try:
            redis_key = self._make_redis_key(key)
            data = pickle.dumps(value)
            
            if ttl_seconds:
                await self._redis.setex(redis_key, ttl_seconds, data)
            else:
                await self._redis.set(redis_key, data)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to write to Redis cache for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not await self._ensure_connected():
            return False
            
        try:
            redis_key = self._make_redis_key(key)
            await self._redis.delete(redis_key)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete from Redis cache for key {key}: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


class CacheManager:
    """Main cache manager with Redis primary and filesystem fallback."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Initialize Redis cache if enabled
        self.redis_cache = None
        if config.use_redis and REDIS_AVAILABLE:
            self.redis_cache = RedisCache(config.redis_url, config.key_prefix)
        
        # Always initialize filesystem cache as fallback
        self.fs_cache = FilesystemCache(config.filesystem_cache_dir, config.max_cache_size_mb)
        
        logger.info(f"Cache manager initialized - Redis: {self.redis_cache is not None}, Filesystem: {config.filesystem_cache_dir}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis first, then filesystem)."""
        # Try Redis first
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit (Redis): {key}")
                return value
        
        # Fallback to filesystem
        value = await self.fs_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit (filesystem): {key}")
            
            # Promote to Redis if available
            if self.redis_cache:
                await self.redis_cache.set(key, value, self.config.default_ttl_seconds)
        
        return value
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache (both Redis and filesystem)."""
        ttl = ttl_seconds or self.config.default_ttl_seconds
        
        # Set in both caches for redundancy
        redis_success = True
        if self.redis_cache:
            redis_success = await self.redis_cache.set(key, value, ttl)
        
        fs_success = await self.fs_cache.set(key, value, ttl)
        
        logger.debug(f"Cache set: {key} (Redis: {redis_success}, FS: {fs_success})")
        return redis_success or fs_success
    
    async def delete(self, key: str) -> bool:
        """Delete value from both caches."""
        redis_success = True
        if self.redis_cache:
            redis_success = await self.redis_cache.delete(key)
        
        fs_success = await self.fs_cache.delete(key)
        
        return redis_success and fs_success
    
    async def cached_request(
        self,
        key: str,
        fetch_func,
        ttl_seconds: Optional[int] = None,
        force_refresh: bool = False
    ) -> Any:
        """Get cached value or fetch and cache new value."""
        if not force_refresh:
            cached_value = await self.get(key)
            if cached_value is not None:
                return cached_value
        
        # Fetch new value
        try:
            new_value = await fetch_func() if asyncio.iscoroutinefunction(fetch_func) else fetch_func()
            
            # Cache the new value
            await self.set(key, new_value, ttl_seconds)
            
            return new_value
            
        except Exception as e:
            logger.error(f"Failed to fetch data for cache key {key}: {e}")
            # Return cached value if fetch fails and not force refresh
            if not force_refresh:
                cached_value = await self.get(key)
                if cached_value is not None:
                    logger.warning(f"Using stale cached data for {key} due to fetch failure")
                    return cached_value
            raise
    
    async def close(self):
        """Close cache connections."""
        if self.redis_cache:
            await self.redis_cache.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "redis_enabled": self.redis_cache is not None,
            "filesystem_cache_dir": str(self.fs_cache.cache_dir),
            "max_cache_size_mb": self.config.max_cache_size_mb,
            "default_ttl_seconds": self.config.default_ttl_seconds,
        }


# Default cache configuration
DEFAULT_CACHE_CONFIG = CacheConfig(
    redis_url="redis://localhost:6379/0",
    use_redis=True,
    filesystem_cache_dir="./cache/crypto",
    max_cache_size_mb=500,
    default_ttl_seconds=300,
    key_prefix="tradingagents:crypto:"
) 