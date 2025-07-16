"""
Abstract interfaces for cross-asset trading data access.

These interfaces enable the TradingAgents framework to work with both equity and crypto markets
through a unified API while allowing asset-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AssetClass(str, Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"


class DataQuality(str, Enum):
    """Data quality indicators."""
    HIGH = "high"          # Official/verified sources
    MEDIUM = "medium"      # Reliable but unverified
    LOW = "low"           # Best effort/derived
    UNKNOWN = "unknown"    # Quality not assessed


# =============================================================================
# Data Models
# =============================================================================

class OHLCVData(BaseModel):
    """Open, High, Low, Close, Volume data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    asset_class: AssetClass
    data_quality: DataQuality = DataQuality.UNKNOWN


class AssetMetadata(BaseModel):
    """Metadata about a tradeable asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    
    # Equity-specific (nullable for crypto)
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    
    # Crypto-specific (nullable for equity)
    circulating_supply: Optional[float] = None
    max_supply: Optional[float] = None
    fully_diluted_valuation: Optional[float] = None
    categories: Optional[List[str]] = None
    
    # Universal fields
    price: Optional[float] = None
    volume_24h: Optional[float] = None
    currency: str = "USD"
    data_quality: DataQuality = DataQuality.UNKNOWN


class FundamentalsData(BaseModel):
    """Fundamental analysis data (equity financials or crypto tokenomics)."""
    symbol: str
    asset_class: AssetClass
    as_of_date: datetime
    
    # Universal metrics (derived for crypto)
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    revenue_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    
    # Equity-specific
    eps_ttm: Optional[float] = None
    pe_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividend_yield: Optional[float] = None
    
    # Crypto-specific  
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    max_supply: Optional[float] = None
    staking_yield: Optional[float] = None
    protocol_revenue: Optional[float] = None
    treasury_value: Optional[float] = None
    
    # Additional metadata
    data_sources: List[str] = Field(default_factory=list)
    data_quality: DataQuality = DataQuality.UNKNOWN


class NewsItem(BaseModel):
    """Individual news story."""
    title: str
    summary: str
    url: Optional[str] = None
    published_at: datetime
    source: str
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)  # -1 to 1
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)   # 0 to 1
    symbols_mentioned: List[str] = Field(default_factory=list)
    asset_class: AssetClass
    data_quality: DataQuality = DataQuality.UNKNOWN


class SentimentData(BaseModel):
    """Social sentiment aggregation."""
    symbol: str
    asset_class: AssetClass
    as_of_date: datetime
    
    # Sentiment metrics
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)  # -1 to 1
    mention_count: Optional[int] = None
    positive_mentions: Optional[int] = None
    negative_mentions: Optional[int] = None
    neutral_mentions: Optional[int] = None
    
    # Volume indicators
    social_volume_24h: Optional[int] = None
    trending_rank: Optional[int] = None
    
    # Source breakdown
    twitter_sentiment: Optional[float] = None
    reddit_sentiment: Optional[float] = None
    news_sentiment: Optional[float] = None
    
    # Metadata
    data_sources: List[str] = Field(default_factory=list)
    data_quality: DataQuality = DataQuality.UNKNOWN


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class Order(BaseModel):
    """Trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime
    updated_at: datetime
    asset_class: AssetClass


class Position(BaseModel):
    """Trading position."""
    symbol: str
    quantity: float  # Positive for long, negative for short
    average_price: float
    market_value: float
    unrealized_pnl: float
    asset_class: AssetClass
    last_updated: datetime


class Balance(BaseModel):
    """Account balance."""
    currency: str
    available: float
    total: float
    reserved: float = 0.0  # Amount tied up in orders
    last_updated: datetime


class RiskMetrics(BaseModel):
    """Risk assessment metrics."""
    symbol: str
    asset_class: AssetClass
    as_of_date: datetime
    
    # Volatility metrics
    volatility_1d: Optional[float] = None
    volatility_7d: Optional[float] = None
    volatility_30d: Optional[float] = None
    
    # Liquidity metrics
    avg_volume_7d: Optional[float] = None
    bid_ask_spread: Optional[float] = None
    market_impact: Optional[float] = None
    
    # Crypto-specific
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    
    # Risk limits
    max_position_size: Optional[float] = None
    concentration_limit: Optional[float] = None
    
    data_quality: DataQuality = DataQuality.UNKNOWN


# =============================================================================
# Abstract Interfaces  
# =============================================================================

class MarketDataClient(ABC):
    """Abstract interface for market data access."""
    
    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[OHLCVData]:
        """Retrieve OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        pass
    
    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols."""
        pass
    
    @abstractmethod  
    async def get_asset_metadata(self, symbol: str) -> Optional[AssetMetadata]:
        """Get metadata for an asset."""
        pass
    
    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        pass


class FundamentalsClient(ABC):
    """Abstract interface for fundamental data access."""
    
    @abstractmethod
    async def get_fundamentals(self, symbol: str, as_of_date: datetime) -> Optional[FundamentalsData]:
        """Get fundamental data for a symbol."""
        pass
    
    @abstractmethod
    async def get_supported_symbols(self) -> List[str]:
        """Get symbols with fundamental data available."""
        pass
    
    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        pass


class NewsClient(ABC):
    """Abstract interface for news data access."""
    
    @abstractmethod
    async def get_news(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        """Get news stories."""
        pass
    
    @abstractmethod
    async def get_global_news(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        """Get global market news."""
        pass
    
    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        pass


class SocialSentimentClient(ABC):
    """Abstract interface for social sentiment data."""
    
    @abstractmethod
    async def get_sentiment(self, symbol: str, as_of_date: datetime) -> Optional[SentimentData]:
        """Get sentiment data for a symbol."""
        pass
    
    @abstractmethod
    async def get_trending_symbols(self, limit: int = 10) -> List[str]:
        """Get trending symbols by social activity."""
        pass
    
    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        pass


class ExecutionClient(ABC):
    """Abstract interface for trade execution."""
    
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None
    ) -> Order:
        """Create a new order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        """Get account balances."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        pass
    
    @property
    @abstractmethod
    def is_paper_trading(self) -> bool:
        """Whether this is paper trading."""
        pass
    
    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        pass


class RiskMetricsClient(ABC):
    """Abstract interface for risk metrics."""
    
    @abstractmethod
    async def get_risk_metrics(self, symbol: str, as_of_date: datetime) -> Optional[RiskMetrics]:
        """Get risk metrics for a symbol."""
        pass
    
    @abstractmethod
    async def get_portfolio_risk(self, positions: List[Position]) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics."""
        pass
    
    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Asset class this client handles."""
        pass 