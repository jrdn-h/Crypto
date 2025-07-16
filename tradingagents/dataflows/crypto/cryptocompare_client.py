"""
CryptoCompare crypto market data client.

Provides implementation of MarketDataClient for CryptoCompare API.
Requires free API key but provides good historical data and metadata.
"""

import aiohttp
import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from ..base_interfaces import (
    MarketDataClient, AssetClass, AssetMetadata, OHLCVData, DataQuality
)
from .rate_limiter import get_rate_limiter
from .caching import CacheManager, CacheKey, DEFAULT_CACHE_CONFIG

logger = logging.getLogger(__name__)


class CryptoCompareClient(MarketDataClient):
    """CryptoCompare API client for cryptocurrency market data."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://min-api.cryptocompare.com/data",
        timeout_seconds: int = 30,
        cache_manager: Optional[CacheManager] = None
    ):
        self.api_key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.rate_limiter = get_rate_limiter("cryptocompare")
        self.cache_manager = cache_manager or CacheManager(DEFAULT_CACHE_CONFIG)
        
        # Symbol validation cache
        self._valid_symbols: Optional[set] = None
        self._symbols_cache_time: Optional[datetime] = None
        
        if not self.api_key:
            logger.warning("CryptoCompare API key not provided - using free tier with limitations")
        
        logger.info(f"CryptoCompare client initialized - Base URL: {base_url}")
    
    @property
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        return AssetClass.CRYPTO
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to CryptoCompare API with rate limiting."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Add API key to headers if available
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Apikey {self.api_key}"
        
        async def _fetch():
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 429:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message="Rate limit exceeded"
                        )
                    
                    response.raise_for_status()
                    json_data = await response.json()
                    
                    # CryptoCompare returns error info in Response field
                    if json_data.get("Response") == "Error":
                        error_msg = json_data.get("Message", "Unknown error")
                        raise Exception(f"CryptoCompare API error: {error_msg}")
                    
                    return json_data
        
        return await self.rate_limiter.execute_with_retry(_fetch)
    
    async def _validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid on CryptoCompare."""
        symbol_upper = symbol.upper()
        
        # Refresh symbols cache if needed
        now = datetime.now()
        if (not self._valid_symbols or 
            not self._symbols_cache_time or 
            now - self._symbols_cache_time > timedelta(hours=6)):
            
            try:
                # Get list of coins
                coin_list = await self._make_request("/all/coinlist")
                
                self._valid_symbols = set()
                if "Data" in coin_list:
                    for coin_id, coin_data in coin_list["Data"].items():
                        symbol = coin_data.get("Symbol", "").upper()
                        if symbol:
                            self._valid_symbols.add(symbol)
                
                self._symbols_cache_time = now
                logger.debug(f"Refreshed CryptoCompare symbols cache: {len(self._valid_symbols)} symbols")
                
            except Exception as e:
                logger.error(f"Failed to fetch CryptoCompare coin list: {e}")
                return True  # Assume valid if we can't check
        
        return symbol_upper in self._valid_symbols if self._valid_symbols else True
    
    async def get_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[OHLCVData]:
        """Retrieve OHLCV data for a symbol."""
        try:
            # Validate symbol
            if not await self._validate_symbol(symbol):
                logger.warning(f"Symbol {symbol} not found on CryptoCompare")
                return []
            
            symbol_upper = symbol.upper()
            
            # Generate cache key
            cache_key = CacheKey.make_ohlcv_key(
                "cryptocompare", symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval
            )
            
            # Try cache first
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached OHLCV data for {symbol}")
                return cached_data
            
            # Map interval and choose appropriate endpoint
            endpoint, limit_param = self._get_ohlcv_endpoint(interval)
            if not endpoint:
                logger.error(f"Unsupported interval: {interval}")
                return []
            
            # Calculate parameters
            days_diff = (end_date - start_date).days
            limit = min(days_diff + 1, 2000)  # CryptoCompare limit
            
            params = {
                "fsym": symbol_upper,
                "tsym": "USD",
                "limit": limit,
                "toTs": int(end_date.timestamp())
            }
            
            # Fetch OHLCV data
            data = await self._make_request(endpoint, params)
            
            # Parse OHLCV data
            ohlcv_data = self._parse_ohlcv_data(data, symbol, start_date, end_date)
            
            # Cache the result
            cache_ttl = 300 if interval in ["1m", "5m", "15m"] else 3600
            await self.cache_manager.set(cache_key, ohlcv_data, ttl_seconds=cache_ttl)
            
            logger.info(f"Fetched {len(ohlcv_data)} OHLCV data points for {symbol} from CryptoCompare")
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return []
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        try:
            if not await self._validate_symbol(symbol):
                return None
            
            symbol_upper = symbol.upper()
            
            # Generate cache key
            cache_key = CacheKey.make_price_key("cryptocompare", symbol)
            
            # Check cache (short TTL for prices)
            cached_price = await self.cache_manager.get(cache_key)
            if cached_price:
                return cached_price
            
            # Fetch current price
            params = {
                "fsym": symbol_upper,
                "tsyms": "USD"
            }
            
            data = await self._make_request("/price", params)
            
            price = data.get("USD")
            if price:
                # Cache for 1 minute
                await self.cache_manager.set(cache_key, price, ttl_seconds=60)
                
            return price
            
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols."""
        try:
            cache_key = CacheKey.make_key("cryptocompare", "available_symbols")
            
            # Check cache (update daily)
            cached_symbols = await self.cache_manager.get(cache_key)
            if cached_symbols:
                return cached_symbols
            
            # Fetch coin list
            coin_list = await self._make_request("/all/coinlist")
            
            symbols = []
            if "Data" in coin_list:
                for coin_id, coin_data in coin_list["Data"].items():
                    symbol = coin_data.get("Symbol", "").upper()
                    if symbol:
                        symbols.append(symbol)
            
            symbols.sort()
            
            # Cache for 24 hours
            await self.cache_manager.set(cache_key, symbols, ttl_seconds=86400)
            
            logger.info(f"Fetched {len(symbols)} available symbols from CryptoCompare")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
    
    async def get_asset_metadata(self, symbol: str) -> Optional[AssetMetadata]:
        """Get metadata for an asset."""
        try:
            if not await self._validate_symbol(symbol):
                return None
            
            symbol_upper = symbol.upper()
            
            # Generate cache key
            cache_key = CacheKey.make_metadata_key("cryptocompare", symbol)
            
            # Check cache
            cached_metadata = await self.cache_manager.get(cache_key)
            if cached_metadata:
                return cached_metadata
            
            # Fetch coin details and current price
            coin_list = await self._make_request("/all/coinlist")
            
            coin_info = None
            if "Data" in coin_list:
                for coin_id, coin_data in coin_list["Data"].items():
                    if coin_data.get("Symbol", "").upper() == symbol_upper:
                        coin_info = coin_data
                        break
            
            if not coin_info:
                return None
            
            # Get current price and volume
            price_data = await self._make_request("/pricemultifull", {
                "fsyms": symbol_upper,
                "tsyms": "USD"
            })
            
            # Parse metadata
            metadata = self._parse_asset_metadata(coin_info, price_data, symbol_upper)
            
            # Cache for 1 hour
            await self.cache_manager.set(cache_key, metadata, ttl_seconds=3600)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get asset metadata for {symbol}: {e}")
            return None
    
    def _get_ohlcv_endpoint(self, interval: str) -> tuple[Optional[str], Optional[str]]:
        """Get appropriate CryptoCompare endpoint for interval."""
        if interval in ["1m", "5m", "15m", "30m"]:
            return "/v2/histominute", "minute"
        elif interval in ["1h", "2h", "4h", "6h", "12h"]:
            return "/v2/histohour", "hour"
        elif interval in ["1d", "1w"]:
            return "/v2/histoday", "day"
        else:
            return None, None
    
    def _parse_ohlcv_data(
        self,
        data: Dict[str, Any],
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[OHLCVData]:
        """Parse CryptoCompare OHLCV data."""
        ohlcv_list = []
        
        try:
            candles = data.get("Data", {}).get("Data", [])
            
            for candle in candles:
                timestamp = candle.get("time", 0)
                dt = datetime.fromtimestamp(timestamp)
                
                # Skip if outside requested range
                if dt < start_date or dt > end_date:
                    continue
                
                ohlcv = OHLCVData(
                    timestamp=dt,
                    open=float(candle.get("open", 0)),
                    high=float(candle.get("high", 0)),
                    low=float(candle.get("low", 0)),
                    close=float(candle.get("close", 0)),
                    volume=float(candle.get("volumefrom", 0)),  # Volume in base currency
                    symbol=symbol.upper(),
                    asset_class=AssetClass.CRYPTO,
                    data_quality=DataQuality.HIGH  # CryptoCompare provides good OHLCV data
                )
                
                ohlcv_list.append(ohlcv)
            
        except Exception as e:
            logger.error(f"Failed to parse OHLCV data: {e}")
        
        return ohlcv_list
    
    def _parse_asset_metadata(
        self,
        coin_info: Dict[str, Any],
        price_data: Dict[str, Any],
        symbol: str
    ) -> AssetMetadata:
        """Parse CryptoCompare coin and price data into AssetMetadata."""
        
        # Extract price and volume data
        display_data = price_data.get("DISPLAY", {}).get(symbol, {}).get("USD", {})
        raw_data = price_data.get("RAW", {}).get(symbol, {}).get("USD", {})
        
        return AssetMetadata(
            symbol=symbol,
            name=coin_info.get("CoinName", coin_info.get("FullName", symbol)),
            asset_class=AssetClass.CRYPTO,
            
            # Price and volume data
            price=raw_data.get("PRICE"),
            volume_24h=raw_data.get("TOTALVOLUME24HTO"),  # 24h volume in USD
            market_cap=raw_data.get("MKTCAP"),
            
            # CryptoCompare doesn't provide supply data in this endpoint
            circulating_supply=None,
            max_supply=None,
            fully_diluted_valuation=None,
            categories=None,
            
            currency="USD",
            data_quality=DataQuality.HIGH  # Good metadata quality
        )
    
    async def close(self):
        """Close client connections."""
        if self.cache_manager:
            await self.cache_manager.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": "cryptocompare",
            "asset_class": self.asset_class,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "rate_limiter": self.rate_limiter.get_stats(),
            "cache": self.cache_manager.get_stats() if self.cache_manager else None,
            "valid_symbols_cached": len(self._valid_symbols) if self._valid_symbols else 0,
            "cache_age_hours": (
                (datetime.now() - self._symbols_cache_time).total_seconds() / 3600
                if self._symbols_cache_time else None
            )
        } 