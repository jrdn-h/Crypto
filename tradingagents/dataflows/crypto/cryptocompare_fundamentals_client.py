"""
CryptoCompare fundamentals client for basic crypto tokenomics.

Provides basic fundamentals data as a secondary source to complement CoinGecko,
focusing on price, volume, and market cap data with some additional metrics.
"""

import aiohttp
import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from ..base_interfaces import (
    FundamentalsClient, AssetClass, FundamentalsData, DataQuality
)
from .rate_limiter import get_rate_limiter
from .caching import CacheManager, CacheKey, DEFAULT_CACHE_CONFIG
from .fundamentals_mapping import (
    CryptoFundamentals, FundamentalsMapper, normalize_tokenomics_data
)

logger = logging.getLogger(__name__)


class CryptoCompareFundamentalsClient(FundamentalsClient):
    """CryptoCompare fundamentals client for basic crypto tokenomics."""
    
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
        
        logger.info(f"CryptoCompare fundamentals client initialized - Base URL: {base_url}")
    
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
    
    async def get_fundamentals(self, symbol: str, as_of_date: datetime) -> Optional[FundamentalsData]:
        """Get basic fundamental data for a crypto symbol."""
        try:
            # Validate symbol
            if not await self._validate_symbol(symbol):
                logger.warning(f"Symbol {symbol} not found on CryptoCompare")
                return None
            
            symbol_upper = symbol.upper()
            
            # Generate cache key
            cache_key = CacheKey.make_key(
                "cryptocompare", "fundamentals",
                symbol=symbol,
                date=as_of_date.strftime("%Y-%m-%d")
            )
            
            # Try cache first
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached fundamentals data for {symbol}")
                return cached_data
            
            # Fetch comprehensive price and volume data
            price_data = await self._make_request("/pricemultifull", {
                "fsyms": symbol_upper,
                "tsyms": "USD"
            })
            
            if "RAW" not in price_data or symbol_upper not in price_data["RAW"]:
                logger.warning(f"No fundamentals data available for {symbol}")
                return None
            
            # Extract data
            raw_data = price_data["RAW"][symbol_upper]["USD"]
            display_data = price_data.get("DISPLAY", {}).get(symbol_upper, {}).get("USD", {})
            
            # Normalize the data
            normalized_data = self._normalize_cryptocompare_data(raw_data, display_data)
            
            # Create crypto fundamentals object
            crypto_fundamentals = FundamentalsMapper.create_crypto_fundamentals(
                normalized_data, symbol, "cryptocompare"
            )
            
            # Set data quality as medium since CryptoCompare has limited fundamentals
            crypto_fundamentals.data_quality = DataQuality.MEDIUM
            
            # Cache the result (cache for 1 hour)
            await self.cache_manager.set(cache_key, crypto_fundamentals, ttl_seconds=3600)
            
            logger.info(f"Fetched basic fundamentals for {symbol} from CryptoCompare")
            return crypto_fundamentals
            
        except Exception as e:
            logger.error(f"Failed to get fundamentals for {symbol}: {e}")
            return None
    
    def _normalize_cryptocompare_data(self, raw_data: Dict[str, Any], display_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize CryptoCompare data to standard format."""
        normalized = {
            "timestamp": datetime.now(),
            "data_source": "cryptocompare",
            "data_quality": DataQuality.MEDIUM,
            
            # Basic price and volume data
            "price": raw_data.get("PRICE"),
            "volume_24h": raw_data.get("TOTALVOLUME24HTO"),  # 24h volume in USD
            "market_cap": raw_data.get("MKTCAP"),
            
            # Change metrics
            "change_24h": raw_data.get("CHANGE24HOUR"),
            "change_pct_24h": raw_data.get("CHANGEPCT24HOUR"),
            
            # Supply data (limited in CryptoCompare)
            "circulating_supply": raw_data.get("SUPPLY"),
            
            # Trading metrics
            "high_24h": raw_data.get("HIGH24HOUR"),
            "low_24h": raw_data.get("LOW24HOUR"),
            "open_24h": raw_data.get("OPEN24HOUR"),
            
            # Volume metrics
            "volume_24h_to": raw_data.get("TOTALVOLUME24H"),  # Volume in base currency
            "total_top_tier_volume": raw_data.get("TOPTIERVOLUME24HOUR"),
            
            # Last update
            "last_update": datetime.fromtimestamp(raw_data.get("LASTUPDATE", 0)) if raw_data.get("LASTUPDATE") else None,
        }
        
        # Calculate additional metrics
        if normalized.get("price") and normalized.get("volume_24h"):
            # Token velocity approximation
            if normalized.get("market_cap"):
                normalized["token_velocity_24h"] = normalized["volume_24h"] / normalized["market_cap"]
        
        # Add display data for better formatting
        normalized["display_data"] = display_data
        
        return normalized
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of symbols with fundamental data available."""
        try:
            cache_key = CacheKey.make_key("cryptocompare", "fundamentals_symbols")
            
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
            
            logger.info(f"Fetched {len(symbols)} symbols with fundamentals data")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get supported symbols: {e}")
            return []
    
    async def get_top_cryptocurrencies(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get top cryptocurrencies by market cap with basic fundamentals."""
        try:
            cache_key = CacheKey.make_key("cryptocompare", "top_cryptos", limit=str(limit))
            
            # Check cache (update every 30 minutes)
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch top cryptocurrencies
            top_list = await self._make_request("/top/mktcapfull", {
                "limit": limit,
                "tsym": "USD"
            })
            
            top_cryptos = []
            if "Data" in top_list:
                for crypto_data in top_list["Data"]:
                    coin_info = crypto_data.get("CoinInfo", {})
                    raw_data = crypto_data.get("RAW", {}).get("USD", {})
                    
                    if coin_info and raw_data:
                        crypto_info = {
                            "symbol": coin_info.get("Name", "").upper(),
                            "name": coin_info.get("FullName", ""),
                            "price": raw_data.get("PRICE"),
                            "market_cap": raw_data.get("MKTCAP"),
                            "volume_24h": raw_data.get("TOTALVOLUME24HTO"),
                            "change_pct_24h": raw_data.get("CHANGEPCT24HOUR"),
                            "rank": len(top_cryptos) + 1
                        }
                        top_cryptos.append(crypto_info)
            
            # Cache for 30 minutes
            await self.cache_manager.set(cache_key, top_cryptos, ttl_seconds=1800)
            
            logger.info(f"Fetched top {len(top_cryptos)} cryptocurrencies")
            return top_cryptos
            
        except Exception as e:
            logger.error(f"Failed to get top cryptocurrencies: {e}")
            return []
    
    async def close(self):
        """Close client connections."""
        if self.cache_manager:
            await self.cache_manager.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": "cryptocompare_fundamentals",
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