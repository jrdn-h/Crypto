"""
Crypto data adapters for TradingAgents.

This module provides concrete implementations of data clients for cryptocurrency markets,
supporting market data (CoinGecko, Binance, CryptoCompare), fundamentals, news, and sentiment analysis.
"""

# Market Data Clients
from .coingecko_client import CoinGeckoClient
from .binance_client import BinancePublicClient
from .cryptocompare_client import CryptoCompareClient

# Fundamentals Clients
from .coingecko_fundamentals_client import CoinGeckoFundamentalsClient
from .cryptocompare_fundamentals_client import CryptoCompareFundamentalsClient

# News Clients
from .cryptopanic_client import CryptoPanicClient
from .coindesk_client import CoinDeskClient

# Sentiment Clients
from .reddit_crypto_client import RedditCryptoClient
from .twitter_sentiment_client import TwitterSentimentClient

# Utilities
from .caching import CacheManager
from .rate_limiter import RateLimiter
from .sentiment_aggregator import SentimentAggregator

# Data Models
from .fundamentals_mapping import (
    CryptoFundamentals, FundamentalsMapper, TokenomicsCategory,
    ProtocolRevenue, StakingMetrics, TreasuryMetrics, TokenUnlockEvent
)

__all__ = [
    # Market Data Clients
    "CoinGeckoClient",
    "BinancePublicClient", 
    "CryptoCompareClient",
    
    # Fundamentals Clients
    "CoinGeckoFundamentalsClient",
    "CryptoCompareFundamentalsClient",
    
    # News Clients
    "CryptoPanicClient",
    "CoinDeskClient",
    
    # Sentiment Clients
    "RedditCryptoClient",
    "TwitterSentimentClient",
    
    # Utilities
    "CacheManager",
    "RateLimiter", 
    "SentimentAggregator",
    
    # Data Models
    "CryptoFundamentals",
    "FundamentalsMapper",
    "TokenomicsCategory",
    "ProtocolRevenue",
    "StakingMetrics", 
    "TreasuryMetrics",
    "TokenUnlockEvent",
] 