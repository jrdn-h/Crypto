"""
CoinGecko crypto market data client.

Provides implementation of MarketDataClient for CoinGecko API.
No API key required for basic functionality.
"""

import aiohttp
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from ..base_interfaces import (
    MarketDataClient, AssetClass, AssetMetadata, OHLCVData, DataQuality
)
from .rate_limiter import get_rate_limiter
from .caching import CacheManager, CacheKey, DEFAULT_CACHE_CONFIG

logger = logging.getLogger(__name__)


class CoinGeckoClient(MarketDataClient):
    """CoinGecko API client for cryptocurrency market data."""
    
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
        self._id_to_symbol_cache: Dict[str, str] = {}
        
        logger.info(f"CoinGecko client initialized - Base URL: {base_url}")
    
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
                    self._id_to_symbol_cache[coin_id] = coin_symbol
                    
                    # Cache the mapping
                    symbol_cache_key = CacheKey.make_key("coingecko", "symbol_to_id", symbol=coin_symbol)
                    await self.cache_manager.set(symbol_cache_key, coin_id, ttl_seconds=3600)  # Cache for 1 hour
            
            return self._symbol_to_id_cache.get(symbol_upper)
            
        except Exception as e:
            logger.error(f"Failed to get coin ID for symbol {symbol}: {e}")
            return None
    
    async def get_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[OHLCVData]:
        """Retrieve OHLCV data for a symbol."""
        try:
            # Get CoinGecko coin ID
            coin_id = await self._get_coin_id(symbol)
            if not coin_id:
                logger.warning(f"Could not find CoinGecko ID for symbol {symbol}")
                return []
            
            # Generate cache key
            cache_key = CacheKey.make_ohlcv_key(
                "coingecko", symbol, 
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval
            )
            
            # Try cache first
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached OHLCV data for {symbol}")
                return cached_data
            
            # Map interval to CoinGecko days parameter
            days = self._calculate_days_param(start_date, end_date, interval)
            
            # Fetch data from CoinGecko
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": self._map_interval(interval)
            }
            
            if days > 90:  # For longer periods, use specific date range
                params.update({
                    "from": int(start_date.timestamp()),
                    "to": int(end_date.timestamp())
                })
                endpoint = f"/coins/{coin_id}/market_chart/range"
            else:
                endpoint = f"/coins/{coin_id}/market_chart"
            
            data = await self._make_request(endpoint, params)
            
            # Parse OHLCV data
            ohlcv_data = self._parse_ohlcv_data(data, symbol, start_date, end_date)
            
            # Cache the result
            cache_ttl = 300 if interval in ["1m", "5m", "15m"] else 3600  # 5min for short intervals, 1hr for longer
            await self.cache_manager.set(cache_key, ohlcv_data, ttl_seconds=cache_ttl)
            
            logger.info(f"Fetched {len(ohlcv_data)} OHLCV data points for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return []
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        try:
            coin_id = await self._get_coin_id(symbol)
            if not coin_id:
                return None
            
            # Generate cache key
            cache_key = CacheKey.make_price_key("coingecko", symbol)
            
            # Check cache (short TTL for prices)
            cached_price = await self.cache_manager.get(cache_key)
            if cached_price:
                return cached_price
            
            # Fetch current price
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_market_cap": "false",
                "include_24hr_vol": "false",
                "include_24hr_change": "false"
            }
            
            data = await self._make_request("/simple/price", params)
            
            price = data.get(coin_id, {}).get("usd")
            if price:
                # Cache for 1 minute (prices change frequently)
                await self.cache_manager.set(cache_key, price, ttl_seconds=60)
                
            return price
            
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols."""
        try:
            cache_key = CacheKey.make_key("coingecko", "available_symbols")
            
            # Check cache (update daily)
            cached_symbols = await self.cache_manager.get(cache_key)
            if cached_symbols:
                return cached_symbols
            
            # Fetch coin list
            data = await self._make_request("/coins/list")
            
            symbols = [coin.get('symbol', '').upper() for coin in data if coin.get('symbol')]
            symbols = list(set(symbols))  # Remove duplicates
            symbols.sort()
            
            # Cache for 24 hours
            await self.cache_manager.set(cache_key, symbols, ttl_seconds=86400)
            
            logger.info(f"Fetched {len(symbols)} available symbols from CoinGecko")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
    
    async def get_asset_metadata(self, symbol: str) -> Optional[AssetMetadata]:
        """Get metadata for an asset."""
        try:
            coin_id = await self._get_coin_id(symbol)
            if not coin_id:
                return None
            
            # Generate cache key
            cache_key = CacheKey.make_metadata_key("coingecko", symbol)
            
            # Check cache
            cached_metadata = await self.cache_manager.get(cache_key)
            if cached_metadata:
                return cached_metadata
            
            # Fetch detailed coin information
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }
            
            data = await self._make_request(f"/coins/{coin_id}", params)
            
            # Parse metadata
            metadata = self._parse_asset_metadata(data, symbol)
            
            # Cache for 1 hour
            await self.cache_manager.set(cache_key, metadata, ttl_seconds=3600)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get asset metadata for {symbol}: {e}")
            return None
    
    def _calculate_days_param(self, start_date: datetime, end_date: datetime, interval: str) -> int:
        """Calculate CoinGecko days parameter from date range."""
        delta = end_date - start_date
        days = delta.days
        
        # CoinGecko has limits on historical data based on interval
        if interval in ["1m", "5m"]:
            return min(days, 1)  # 1-5 minute data only available for 1 day
        elif interval in ["15m", "30m", "1h"]:
            return min(days, 90)  # Hourly data available for 90 days
        else:
            return days  # Daily data available for longer periods
    
    def _map_interval(self, interval: str) -> str:
        """Map interval to CoinGecko format."""
        interval_mapping = {
            "1m": "minutely",
            "5m": "minutely", 
            "15m": "hourly",
            "30m": "hourly",
            "1h": "hourly",
            "4h": "hourly",
            "1d": "daily",
            "1w": "daily"
        }
        return interval_mapping.get(interval, "daily")
    
    def _parse_ohlcv_data(
        self,
        data: Dict[str, Any],
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[OHLCVData]:
        """Parse CoinGecko market chart data into OHLCV format."""
        ohlcv_list = []
        
        try:
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            # CoinGecko returns [timestamp, value] pairs
            price_dict = {int(p[0]): p[1] for p in prices}
            volume_dict = {int(v[0]): v[1] for v in volumes}
            
            # Create OHLCV data points
            # Note: CoinGecko doesn't provide OHLC, only prices, so we approximate
            timestamps = sorted(price_dict.keys())
            
            for i, timestamp in enumerate(timestamps):
                dt = datetime.fromtimestamp(timestamp / 1000)
                
                # Skip if outside requested range
                if dt < start_date or dt > end_date:
                    continue
                
                price = price_dict[timestamp]
                volume = volume_dict.get(timestamp, 0)
                
                # For OHLC, we use the price as all values (limitation of CoinGecko free API)
                # This is acceptable for daily data but less accurate for intraday
                ohlcv = OHLCVData(
                    timestamp=dt,
                    open=price,
                    high=price,
                    low=price, 
                    close=price,
                    volume=volume,
                    symbol=symbol.upper(),
                    asset_class=AssetClass.CRYPTO,
                    data_quality=DataQuality.MEDIUM  # Good quality but limited OHLC precision
                )
                
                ohlcv_list.append(ohlcv)
            
        except Exception as e:
            logger.error(f"Failed to parse OHLCV data: {e}")
        
        return ohlcv_list
    
    def _parse_asset_metadata(self, data: Dict[str, Any], symbol: str) -> AssetMetadata:
        """Parse CoinGecko coin data into AssetMetadata."""
        market_data = data.get("market_data", {})
        
        return AssetMetadata(
            symbol=symbol.upper(),
            name=data.get("name", ""),
            asset_class=AssetClass.CRYPTO,
            
            # Crypto-specific fields
            circulating_supply=market_data.get("circulating_supply"),
            max_supply=market_data.get("max_supply"),
            fully_diluted_valuation=market_data.get("fully_diluted_valuation", {}).get("usd"),
            categories=data.get("categories", []),
            
            # Universal fields
            price=market_data.get("current_price", {}).get("usd"),
            volume_24h=market_data.get("total_volume", {}).get("usd"),
            market_cap=market_data.get("market_cap", {}).get("usd"),
            currency="USD",
            data_quality=DataQuality.HIGH  # CoinGecko has high-quality metadata
        )
    
    async def close(self):
        """Close client connections."""
        if self.cache_manager:
            await self.cache_manager.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": "coingecko",
            "asset_class": self.asset_class,
            "base_url": self.base_url,
            "rate_limiter": self.rate_limiter.get_stats(),
            "cache": self.cache_manager.get_stats() if self.cache_manager else None,
            "symbol_mappings_cached": len(self._symbol_to_id_cache)
        } 