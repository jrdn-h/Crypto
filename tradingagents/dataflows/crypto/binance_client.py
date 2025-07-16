"""
Binance public API crypto market data client.

Provides implementation of MarketDataClient for Binance public API.
No API key required for market data endpoints.
High rate limits and excellent data quality.
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


class BinancePublicClient(MarketDataClient):
    """Binance public API client for cryptocurrency market data."""
    
    def __init__(
        self,
        base_url: str = "https://api.binance.com/api/v3",
        timeout_seconds: int = 30,
        cache_manager: Optional[CacheManager] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.rate_limiter = get_rate_limiter("binance_public")
        self.cache_manager = cache_manager or CacheManager(DEFAULT_CACHE_CONFIG)
        
        # Symbol mapping cache
        self._symbols_cache: Optional[Dict[str, Dict]] = None
        self._symbols_cache_time: Optional[datetime] = None
        
        logger.info(f"Binance public client initialized - Base URL: {base_url}")
    
    @property
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        return AssetClass.CRYPTO
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make HTTP request to Binance API with rate limiting."""
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
    
    async def _get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information from Binance exchange info."""
        symbol_upper = symbol.upper()
        
        # Add USDT suffix if not present (most common on Binance)
        if symbol_upper not in ["USDT", "BUSD", "BTC", "ETH"] and not any(
            symbol_upper.endswith(quote) for quote in ["USDT", "BUSD", "BTC", "ETH", "BNB"]
        ):
            symbol_upper = f"{symbol_upper}USDT"
        
        # Check if we need to refresh symbols cache
        now = datetime.now()
        if (not self._symbols_cache or 
            not self._symbols_cache_time or 
            now - self._symbols_cache_time > timedelta(hours=1)):
            
            try:
                exchange_info = await self._make_request("/exchangeInfo")
                symbols_data = exchange_info.get("symbols", [])
                
                self._symbols_cache = {s["symbol"]: s for s in symbols_data}
                self._symbols_cache_time = now
                
                logger.debug(f"Refreshed Binance symbols cache: {len(self._symbols_cache)} symbols")
                
            except Exception as e:
                logger.error(f"Failed to fetch Binance exchange info: {e}")
                return None
        
        return self._symbols_cache.get(symbol_upper) if self._symbols_cache else None
    
    async def get_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[OHLCVData]:
        """Retrieve OHLCV data for a symbol."""
        try:
            # Get symbol info and normalize symbol
            symbol_info = await self._get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"Symbol {symbol} not found on Binance")
                return []
            
            binance_symbol = symbol_info["symbol"]
            
            # Generate cache key
            cache_key = CacheKey.make_ohlcv_key(
                "binance", symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval
            )
            
            # Try cache first
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached OHLCV data for {symbol}")
                return cached_data
            
            # Map interval to Binance format
            binance_interval = self._map_interval(interval)
            if not binance_interval:
                logger.error(f"Unsupported interval: {interval}")
                return []
            
            # Prepare parameters for Binance klines API
            params = {
                "symbol": binance_symbol,
                "interval": binance_interval,
                "startTime": int(start_date.timestamp() * 1000),
                "endTime": int(end_date.timestamp() * 1000),
                "limit": 1000  # Maximum allowed by Binance
            }
            
            # Fetch klines data
            klines_data = await self._make_request("/klines", params)
            
            # Parse OHLCV data
            ohlcv_data = self._parse_klines_data(klines_data, symbol, start_date, end_date)
            
            # Cache the result
            cache_ttl = 300 if interval in ["1m", "5m", "15m"] else 3600  # 5min for short intervals, 1hr for longer
            await self.cache_manager.set(cache_key, ohlcv_data, ttl_seconds=cache_ttl)
            
            logger.info(f"Fetched {len(ohlcv_data)} OHLCV data points for {symbol} from Binance")
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return []
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        try:
            # Get symbol info
            symbol_info = await self._get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            binance_symbol = symbol_info["symbol"]
            
            # Generate cache key
            cache_key = CacheKey.make_price_key("binance", symbol)
            
            # Check cache (short TTL for prices)
            cached_price = await self.cache_manager.get(cache_key)
            if cached_price:
                return cached_price
            
            # Fetch current price
            params = {"symbol": binance_symbol}
            data = await self._make_request("/ticker/price", params)
            
            price = float(data.get("price", 0))
            if price:
                # Cache for 30 seconds (prices change frequently)
                await self.cache_manager.set(cache_key, price, ttl_seconds=30)
                
            return price
            
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols."""
        try:
            cache_key = CacheKey.make_key("binance", "available_symbols")
            
            # Check cache (update daily)
            cached_symbols = await self.cache_manager.get(cache_key)
            if cached_symbols:
                return cached_symbols
            
            # Fetch exchange info
            exchange_info = await self._make_request("/exchangeInfo")
            symbols_data = exchange_info.get("symbols", [])
            
            # Extract symbols and normalize (remove quote currency)
            symbols = set()
            for symbol_info in symbols_data:
                if symbol_info.get("status") == "TRADING":
                    symbol = symbol_info.get("baseAsset", "")
                    if symbol:
                        symbols.add(symbol.upper())
            
            symbols_list = sorted(list(symbols))
            
            # Cache for 24 hours
            await self.cache_manager.set(cache_key, symbols_list, ttl_seconds=86400)
            
            logger.info(f"Fetched {len(symbols_list)} available symbols from Binance")
            return symbols_list
            
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
    
    async def get_asset_metadata(self, symbol: str) -> Optional[AssetMetadata]:
        """Get metadata for an asset."""
        try:
            # Get symbol info
            symbol_info = await self._get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            binance_symbol = symbol_info["symbol"]
            base_asset = symbol_info.get("baseAsset", symbol).upper()
            
            # Generate cache key
            cache_key = CacheKey.make_metadata_key("binance", symbol)
            
            # Check cache
            cached_metadata = await self.cache_manager.get(cache_key)
            if cached_metadata:
                return cached_metadata
            
            # Fetch 24hr ticker statistics for additional metadata
            params = {"symbol": binance_symbol}
            ticker_data = await self._make_request("/ticker/24hr", params)
            
            # Parse metadata
            metadata = self._parse_asset_metadata(symbol_info, ticker_data, base_asset)
            
            # Cache for 1 hour
            await self.cache_manager.set(cache_key, metadata, ttl_seconds=3600)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get asset metadata for {symbol}: {e}")
            return None
    
    def _map_interval(self, interval: str) -> Optional[str]:
        """Map interval to Binance format."""
        interval_mapping = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M"
        }
        return interval_mapping.get(interval)
    
    def _parse_klines_data(
        self,
        klines_data: List[List],
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[OHLCVData]:
        """Parse Binance klines data into OHLCV format."""
        ohlcv_list = []
        
        try:
            for kline in klines_data:
                # Binance kline format:
                # [timestamp, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore]
                timestamp = int(kline[0])
                dt = datetime.fromtimestamp(timestamp / 1000)
                
                # Skip if outside requested range
                if dt < start_date or dt > end_date:
                    continue
                
                ohlcv = OHLCVData(
                    timestamp=dt,
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    symbol=symbol.upper(),
                    asset_class=AssetClass.CRYPTO,
                    data_quality=DataQuality.HIGH  # Binance provides excellent OHLCV data
                )
                
                ohlcv_list.append(ohlcv)
            
        except Exception as e:
            logger.error(f"Failed to parse klines data: {e}")
        
        return ohlcv_list
    
    def _parse_asset_metadata(
        self,
        symbol_info: Dict[str, Any],
        ticker_data: Dict[str, Any],
        symbol: str
    ) -> AssetMetadata:
        """Parse Binance symbol and ticker data into AssetMetadata."""
        return AssetMetadata(
            symbol=symbol,
            name=symbol_info.get("baseAsset", symbol),  # Binance doesn't provide full names
            asset_class=AssetClass.CRYPTO,
            
            # Universal fields from 24hr ticker
            price=float(ticker_data.get("lastPrice", 0)) or None,
            volume_24h=float(ticker_data.get("quoteVolume", 0)) or None,
            
            # Note: Binance public API doesn't provide supply or market cap data
            # These would need to be fetched from other sources like CoinGecko
            circulating_supply=None,
            max_supply=None,
            fully_diluted_valuation=None,
            market_cap=None,
            categories=None,
            
            currency="USDT",  # Most Binance pairs are quoted in USDT
            data_quality=DataQuality.HIGH  # Excellent price and volume data
        )
    
    async def close(self):
        """Close client connections."""
        if self.cache_manager:
            await self.cache_manager.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": "binance",
            "asset_class": self.asset_class,
            "base_url": self.base_url,
            "rate_limiter": self.rate_limiter.get_stats(),
            "cache": self.cache_manager.get_stats() if self.cache_manager else None,
            "symbols_cached": len(self._symbols_cache) if self._symbols_cache else 0,
            "cache_age_minutes": (
                (datetime.now() - self._symbols_cache_time).total_seconds() / 60
                if self._symbols_cache_time else None
            )
        } 