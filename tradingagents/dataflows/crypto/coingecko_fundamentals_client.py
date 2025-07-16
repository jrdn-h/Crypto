"""
CoinGecko fundamentals client for crypto tokenomics data.

Implements FundamentalsClient interface to provide comprehensive crypto fundamentals
including supply metrics, protocol revenue, staking data, and treasury information.
"""

import aiohttp
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from ..base_interfaces import (
    FundamentalsClient, AssetClass, FundamentalsData, DataQuality
)
from .rate_limiter import get_rate_limiter
from .caching import CacheManager, CacheKey, DEFAULT_CACHE_CONFIG
from .fundamentals_mapping import (
    CryptoFundamentals, FundamentalsMapper, normalize_tokenomics_data,
    ProtocolRevenue, StakingMetrics, TreasuryMetrics
)

logger = logging.getLogger(__name__)


class CoinGeckoFundamentalsClient(FundamentalsClient):
    """CoinGecko fundamentals client for comprehensive crypto tokenomics."""
    
    def __init__(
        self,
        base_url: str = "https://api.coingecko.com/api/v3",
        timeout_seconds: int = 30,
        cache_manager: Optional[CacheManager] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.rate_limiter = get_rate_limiter("coingecko")
        self.cache_manager = cache_manager or CacheManager(DEFAULT_CACHE_CONFIG)
        
        # Symbol mappings (CoinGecko uses coin IDs, not symbols)
        self._symbol_to_id_cache: Dict[str, str] = {}
        self._fundamentals_cache: Dict[str, datetime] = {}
        
        logger.info(f"CoinGecko fundamentals client initialized - Base URL: {base_url}")
    
    @property
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        return AssetClass.CRYPTO
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to CoinGecko API with rate limiting."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async def _fetch():
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message="Rate limit exceeded"
                        )
                    
                    response.raise_for_status()
                    return await response.json()
        
        return await self.rate_limiter.execute_with_retry(_fetch)
    
    async def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID from symbol."""
        symbol_upper = symbol.upper()
        
        # Check cache first
        if symbol_upper in self._symbol_to_id_cache:
            return self._symbol_to_id_cache[symbol_upper]
        
        # Check cache manager
        cache_key = CacheKey.make_key("coingecko", "symbol_to_id", symbol=symbol_upper)
        cached_id = await self.cache_manager.get(cache_key)
        if cached_id:
            self._symbol_to_id_cache[symbol_upper] = cached_id
            return cached_id
        
        try:
            # Fetch coin list from CoinGecko
            coins_data = await self._make_request("/coins/list")
            
            # Build mapping
            for coin in coins_data:
                coin_symbol = coin.get('symbol', '').upper()
                coin_id = coin.get('id')
                
                if coin_symbol and coin_id:
                    self._symbol_to_id_cache[coin_symbol] = coin_id
                    
                    # Cache the mapping
                    symbol_cache_key = CacheKey.make_key("coingecko", "symbol_to_id", symbol=coin_symbol)
                    await self.cache_manager.set(symbol_cache_key, coin_id, ttl_seconds=3600)
            
            return self._symbol_to_id_cache.get(symbol_upper)
            
        except Exception as e:
            logger.error(f"Failed to get coin ID for symbol {symbol}: {e}")
            return None
    
    async def get_fundamentals(self, symbol: str, as_of_date: datetime) -> Optional[FundamentalsData]:
        """Get comprehensive fundamental data for a crypto symbol."""
        try:
            # Get CoinGecko coin ID
            coin_id = await self._get_coin_id(symbol)
            if not coin_id:
                logger.warning(f"Could not find CoinGecko ID for symbol {symbol}")
                return None
            
            # Generate cache key
            cache_key = CacheKey.make_key(
                "coingecko", "fundamentals",
                symbol=symbol,
                date=as_of_date.strftime("%Y-%m-%d")
            )
            
            # Try cache first
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached fundamentals data for {symbol}")
                return cached_data
            
            # Fetch comprehensive coin data
            coin_data = await self._fetch_comprehensive_data(coin_id)
            
            if not coin_data:
                return None
            
            # Normalize the data
            normalized_data = normalize_tokenomics_data(coin_data, "coingecko")
            
            # Add additional metrics
            enhanced_data = await self._enhance_fundamentals_data(normalized_data, coin_id, symbol)
            
            # Create crypto fundamentals object
            crypto_fundamentals = FundamentalsMapper.create_crypto_fundamentals(
                enhanced_data, symbol, "coingecko"
            )
            
            # Cache the result (cache for 1 hour for fundamentals)
            await self.cache_manager.set(cache_key, crypto_fundamentals, ttl_seconds=3600)
            
            logger.info(f"Fetched comprehensive fundamentals for {symbol}")
            return crypto_fundamentals
            
        except Exception as e:
            logger.error(f"Failed to get fundamentals for {symbol}: {e}")
            return None
    
    async def _fetch_comprehensive_data(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive data for a coin from multiple CoinGecko endpoints."""
        try:
            # Main coin data with market data
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "false",
                "sparkline": "false"
            }
            
            coin_data = await self._make_request(f"/coins/{coin_id}", params)
            
            return coin_data
            
        except Exception as e:
            logger.error(f"Failed to fetch comprehensive data for {coin_id}: {e}")
            return None
    
    async def _enhance_fundamentals_data(
        self,
        base_data: Dict[str, Any],
        coin_id: str,
        symbol: str
    ) -> Dict[str, Any]:
        """Enhance fundamentals data with additional metrics and calculations."""
        enhanced = base_data.copy()
        
        # Add current timestamp
        enhanced["timestamp"] = datetime.now()
        enhanced["data_quality"] = DataQuality.HIGH
        
        # Calculate additional metrics
        try:
            # Price-to-sales ratio (using market cap / volume as proxy)
            if enhanced.get("market_cap") and enhanced.get("volume_24h"):
                enhanced["price_to_volume_ratio"] = enhanced["market_cap"] / enhanced["volume_24h"]
            
            # Calculate supply inflation potential
            if enhanced.get("circulating_supply") and enhanced.get("max_supply"):
                circ = enhanced["circulating_supply"]
                max_supply = enhanced["max_supply"]
                if max_supply and max_supply > circ:
                    enhanced["supply_inflation_potential"] = (max_supply - circ) / circ
                    enhanced["circulating_ratio"] = circ / max_supply
            
            # Calculate market dominance (would need total crypto market cap)
            # This would require a separate API call to get global market data
            
            # Token velocity approximation
            if enhanced.get("volume_24h") and enhanced.get("market_cap"):
                enhanced["token_velocity_24h"] = enhanced["volume_24h"] / enhanced["market_cap"]
            
            # Add categories and use cases
            # These come from the coin_data directly
            
        except Exception as e:
            logger.warning(f"Failed to calculate enhanced metrics for {symbol}: {e}")
        
        return enhanced
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of symbols with fundamental data available."""
        try:
            cache_key = CacheKey.make_key("coingecko", "fundamentals_symbols")
            
            # Check cache (update daily)
            cached_symbols = await self.cache_manager.get(cache_key)
            if cached_symbols:
                return cached_symbols
            
            # Fetch coin list
            coins_data = await self._make_request("/coins/list")
            
            # Extract symbols that have market data
            symbols = []
            for coin in coins_data:
                symbol = coin.get('symbol', '').upper()
                if symbol:
                    symbols.append(symbol)
            
            symbols.sort()
            
            # Cache for 24 hours
            await self.cache_manager.set(cache_key, symbols, ttl_seconds=86400)
            
            logger.info(f"Fetched {len(symbols)} symbols with fundamentals data")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get supported symbols: {e}")
            return []
    
    async def get_protocol_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get protocol-specific metrics (fees, revenue, etc.) if available."""
        try:
            coin_id = await self._get_coin_id(symbol)
            if not coin_id:
                return None
            
            # CoinGecko doesn't provide protocol revenue data directly
            # This would need to be supplemented with other data sources like:
            # - Token Terminal
            # - DefiLlama
            # - Protocol-specific APIs
            
            # For now, return basic protocol info from CoinGecko
            coin_data = await self._make_request(f"/coins/{coin_id}")
            
            protocol_info = {
                "protocol_type": self._determine_protocol_type(coin_data),
                "categories": coin_data.get("categories", []),
                "description": coin_data.get("description", {}).get("en", ""),
                "homepage": coin_data.get("links", {}).get("homepage", []),
                "blockchain_site": coin_data.get("links", {}).get("blockchain_site", []),
            }
            
            return protocol_info
            
        except Exception as e:
            logger.error(f"Failed to get protocol metrics for {symbol}: {e}")
            return None
    
    def _determine_protocol_type(self, coin_data: Dict[str, Any]) -> str:
        """Determine protocol type based on categories and other data."""
        categories = coin_data.get("categories", [])
        
        if not categories:
            return "unknown"
        
        # Map categories to protocol types
        category_mapping = {
            "smart-contract-platform": "layer1",
            "layer-1": "layer1",
            "decentralized-exchange": "dex",
            "lending-borrowing": "lending",
            "yield-farming": "defi",
            "staking": "staking",
            "gaming": "gamefi",
            "nft": "nft",
            "metaverse": "metaverse",
            "meme-token": "meme",
            "stablecoin": "stablecoin"
        }
        
        for category in categories:
            category_lower = category.lower().replace(" ", "-")
            if category_lower in category_mapping:
                return category_mapping[category_lower]
        
        return "other"
    
    async def close(self):
        """Close client connections."""
        if self.cache_manager:
            await self.cache_manager.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": "coingecko_fundamentals",
            "asset_class": self.asset_class,
            "base_url": self.base_url,
            "rate_limiter": self.rate_limiter.get_stats(),
            "cache": self.cache_manager.get_stats() if self.cache_manager else None,
            "symbol_mappings_cached": len(self._symbol_to_id_cache),
            "fundamentals_cached": len(self._fundamentals_cache)
        } 