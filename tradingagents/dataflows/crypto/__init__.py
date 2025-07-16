"""
Crypto data adapters for TradingAgents.

This module provides concrete implementations of data clients for cryptocurrency markets,
supporting CoinGecko, Binance Public API, and CryptoCompare.
"""

from .coingecko_client import CoinGeckoClient
from .binance_client import BinancePublicClient
from .cryptocompare_client import CryptoCompareClient
from .coingecko_fundamentals_client import CoinGeckoFundamentalsClient
from .cryptocompare_fundamentals_client import CryptoCompareFundamentalsClient
from .caching import CacheManager
from .rate_limiter import RateLimiter
from .fundamentals_mapping import (
    CryptoFundamentals, FundamentalsMapper, TokenomicsCategory,
    ProtocolRevenue, StakingMetrics, TreasuryMetrics, TokenUnlockEvent
)

__all__ = [
    "CoinGeckoClient",
    "BinancePublicClient", 
    "CryptoCompareClient",
    "CoinGeckoFundamentalsClient",
    "CryptoCompareFundamentalsClient",
    "CacheManager",
    "RateLimiter",
    "CryptoFundamentals",
    "FundamentalsMapper",
    "TokenomicsCategory",
    "ProtocolRevenue",
    "StakingMetrics", 
    "TreasuryMetrics",
    "TokenUnlockEvent",
] 