"""
Rate limiting utilities for crypto API clients.

Provides configurable rate limiting with different strategies:
- Token bucket for burst allowance
- Fixed window for simple rate limits
- Exponential backoff for retries
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    burst_allowance: int = 5  # Extra requests allowed in burst
    retry_attempts: int = 3
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Maximum retry delay


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens from the bucket."""
        async with self._lock:
            now = time.time()
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_tokens(self, tokens: int = 1) -> float:
        """Wait until tokens are available, return wait time."""
        if await self.acquire(tokens):
            return 0.0
            
        # Calculate wait time
        shortage = tokens - self.tokens
        wait_time = shortage / self.refill_rate
        
        logger.debug(f"Rate limit hit, waiting {wait_time:.2f}s for {tokens} tokens")
        await asyncio.sleep(wait_time)
        
        # Try again after waiting
        await self.acquire(tokens)
        return wait_time


class FixedWindowLimiter:
    """Fixed window rate limiter implementation."""
    
    def __init__(self, requests_per_window: int, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire a request slot."""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests
            self.requests = [req_time for req_time in self.requests if req_time > window_start]
            
            if len(self.requests) < self.requests_per_window:
                self.requests.append(now)
                return True
            return False
    
    async def wait_for_slot(self) -> float:
        """Wait until a slot is available."""
        if await self.acquire():
            return 0.0
            
        # Calculate wait time until oldest request expires
        now = time.time()
        window_start = now - self.window_seconds
        oldest_request = min(self.requests)
        wait_time = oldest_request - window_start
        
        logger.debug(f"Rate limit hit, waiting {wait_time:.2f}s for slot")
        await asyncio.sleep(wait_time)
        
        await self.acquire()
        return wait_time


class RateLimiter:
    """Main rate limiter class with configurable strategies."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._limiter = self._create_limiter()
        self._request_history: Dict[str, float] = {}
    
    def _create_limiter(self):
        """Create the appropriate limiter based on strategy."""
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            capacity = self.config.requests_per_minute + self.config.burst_allowance
            refill_rate = self.config.requests_per_minute / 60.0  # per second
            return TokenBucket(capacity, refill_rate)
        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return FixedWindowLimiter(self.config.requests_per_minute, 60)
        else:
            raise ValueError(f"Unsupported rate limit strategy: {self.config.strategy}")
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with rate limiting and retry logic."""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Wait for rate limit clearance
                if isinstance(self._limiter, TokenBucket):
                    await self._limiter.wait_for_tokens()
                elif isinstance(self._limiter, FixedWindowLimiter):
                    await self._limiter.wait_for_slot()
                
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    wait_time = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Rate limit exceeded, backing off for {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    continue
                
                # For other errors, still backoff but log differently
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}, retrying in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.config.retry_attempts} attempts: {e}")
                    raise
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("All retry attempts failed with no exception captured")
    
    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """Check if exception indicates rate limiting."""
        error_msg = str(exception).lower()
        rate_limit_indicators = [
            "rate limit", "too many requests", "429", "quota exceeded",
            "throttled", "rate exceeded", "limit exceeded"
        ]
        return any(indicator in error_msg for indicator in rate_limit_indicators)
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.config.base_delay * (2 ** attempt)
        return min(delay, self.config.max_delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        stats: Dict[str, Any] = {
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "strategy": self.config.strategy,
                "burst_allowance": self.config.burst_allowance,
            }
        }
        
        if isinstance(self._limiter, TokenBucket):
            stats["current_tokens"] = self._limiter.tokens
            stats["capacity"] = self._limiter.capacity
        elif isinstance(self._limiter, FixedWindowLimiter):
            stats["current_requests"] = len(self._limiter.requests)
            stats["max_requests"] = self._limiter.requests_per_window
            
        return stats


# Predefined rate limit configs for crypto providers
PROVIDER_RATE_LIMITS = {
    "coingecko": RateLimitConfig(
        requests_per_minute=50,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        burst_allowance=10,
        retry_attempts=3,
        base_delay=2.0
    ),
    "binance_public": RateLimitConfig(
        requests_per_minute=1200,  # Binance allows 1200 requests per minute
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        burst_allowance=100,
        retry_attempts=3,
        base_delay=1.0
    ),
    "cryptocompare": RateLimitConfig(
        requests_per_minute=100,  # Free tier limit
        strategy=RateLimitStrategy.FIXED_WINDOW,
        burst_allowance=5,
        retry_attempts=3,
        base_delay=1.5
    ),
}


def get_rate_limiter(provider: str) -> RateLimiter:
    """Get a rate limiter for a specific provider."""
    config = PROVIDER_RATE_LIMITS.get(provider)
    if not config:
        # Default rate limiter for unknown providers
        config = RateLimitConfig(requests_per_minute=60, retry_attempts=2)
    
    return RateLimiter(config) 