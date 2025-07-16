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

# Execution Clients
from .paper_broker import CryptoPaperBroker
from .ccxt_broker import CCXTBroker, CCXTBrokerFactory
from .hyperliquid_broker import HyperliquidBroker

# Utilities
from .caching import CacheManager
from .rate_limiter import RateLimiter 
from .sentiment_aggregator import SentimentAggregator
from .crypto_stockstats import CryptoStockstatsUtils
from .crypto_technical import CryptoTechnicalAnalyzer, CryptoTechnicalConfig
from .whale_flow_tracker import WhaleFlowTracker, WhaleTransaction, ExchangeFlow, WhaleAlert
from .tokenomics_analyzer import TokenomicsAnalyzer, get_tokenomics_analysis
from .regulatory_analyzer import RegulatoryAnalyzer, get_regulatory_analysis

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
    
    # Execution Clients
    "CryptoPaperBroker",
    "CCXTBroker",
    "CCXTBrokerFactory",
    "HyperliquidBroker",
    
    # Utilities
    "CacheManager",
    "RateLimiter", 
    "SentimentAggregator",
    "CryptoStockstatsUtils",
    "CryptoTechnicalAnalyzer",
    "CryptoTechnicalConfig",
    "WhaleFlowTracker",
    "WhaleTransaction",
    "ExchangeFlow",
    "WhaleAlert",
    "TokenomicsAnalyzer",
    "get_tokenomics_analysis",
    "RegulatoryAnalyzer", 
    "get_regulatory_analysis",
    
    # Data Models
    "CryptoFundamentals",
    "FundamentalsMapper",
    "TokenomicsCategory",
    "ProtocolRevenue",
    "StakingMetrics", 
    "TreasuryMetrics",
    "TokenUnlockEvent",
] 