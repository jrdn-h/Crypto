"""
Provider registry for managing data source adapters.

This module provides centralized registration and discovery of data providers
for different asset classes, enabling automatic fallback and provider selection.
"""

from typing import Dict, List, Optional, Type, Any, Union
from enum import Enum
import logging
from dataclasses import dataclass, field

from .base_interfaces import (
    AssetClass,
    MarketDataClient,
    FundamentalsClient,
    NewsClient,
    SocialSentimentClient,
    ExecutionClient,
    RiskMetricsClient
)

logger = logging.getLogger(__name__)


class ProviderPriority(str, Enum):
    """Provider priority levels."""
    PRIMARY = "primary"      # First choice
    SECONDARY = "secondary"  # Fallback
    TERTIARY = "tertiary"   # Last resort
    DISABLED = "disabled"   # Not used


@dataclass
class ProviderConfig:
    """Configuration for a data provider."""
    name: str
    provider_class: Type[Union[MarketDataClient, FundamentalsClient, NewsClient, 
                              SocialSentimentClient, ExecutionClient, RiskMetricsClient]]
    asset_class: AssetClass
    priority: ProviderPriority
    requires_api_key: bool = False
    api_key_env_var: Optional[str] = None
    rate_limit_per_minute: Optional[int] = None
    cost_tier: str = "free"  # "free", "cheap", "premium"
    enabled: bool = True
    init_kwargs: Dict[str, Any] = field(default_factory=dict)


class ProviderRegistry:
    """Central registry for data provider management."""
    
    def __init__(self):
        self._providers: Dict[str, Dict[AssetClass, List[ProviderConfig]]] = {
            "market_data": {AssetClass.EQUITY: [], AssetClass.CRYPTO: []},
            "fundamentals": {AssetClass.EQUITY: [], AssetClass.CRYPTO: []},
            "news": {AssetClass.EQUITY: [], AssetClass.CRYPTO: []},
            "sentiment": {AssetClass.EQUITY: [], AssetClass.CRYPTO: []},
            "execution": {AssetClass.EQUITY: [], AssetClass.CRYPTO: []},
            "risk": {AssetClass.EQUITY: [], AssetClass.CRYPTO: []}
        }
        self._client_instances: Dict[str, Any] = {}
        
    def register_provider(
        self,
        provider_type: str,
        config: ProviderConfig
    ) -> None:
        """Register a data provider."""
        if provider_type not in self._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
        providers_list = self._providers[provider_type][config.asset_class]
        
        # Remove existing provider with same name
        providers_list[:] = [p for p in providers_list if p.name != config.name]
        
        # Insert in priority order
        if config.priority == ProviderPriority.PRIMARY:
            providers_list.insert(0, config)
        elif config.priority == ProviderPriority.SECONDARY:
            # Insert after primary but before tertiary
            primary_count = sum(1 for p in providers_list if p.priority == ProviderPriority.PRIMARY)
            providers_list.insert(primary_count, config)
        else:
            providers_list.append(config)
            
        logger.info(f"Registered {provider_type} provider: {config.name} for {config.asset_class}")
    
    def get_providers(
        self,
        provider_type: str,
        asset_class: AssetClass,
        enabled_only: bool = True
    ) -> List[ProviderConfig]:
        """Get providers for a specific type and asset class."""
        providers = self._providers.get(provider_type, {}).get(asset_class, [])
        
        if enabled_only:
            providers = [p for p in providers if p.enabled and p.priority != ProviderPriority.DISABLED]
            
        return sorted(providers, key=lambda p: {
            ProviderPriority.PRIMARY: 0,
            ProviderPriority.SECONDARY: 1, 
            ProviderPriority.TERTIARY: 2,
            ProviderPriority.DISABLED: 3
        }[p.priority])
    
    def get_client(
        self,
        provider_type: str,
        asset_class: AssetClass,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> Optional[Union[MarketDataClient, FundamentalsClient, NewsClient, 
                       SocialSentimentClient, ExecutionClient, RiskMetricsClient]]:
        """Get a client instance for the specified provider type."""
        providers = self.get_providers(provider_type, asset_class)
        
        if not providers:
            logger.warning(f"No providers available for {provider_type}/{asset_class}")
            return None
            
        # Filter by provider name if specified
        if provider_name:
            providers = [p for p in providers if p.name == provider_name]
            if not providers:
                logger.error(f"Provider {provider_name} not found for {provider_type}/{asset_class}")
                return None
        
        # Try providers in priority order
        for provider_config in providers:
            try:
                client = self._get_or_create_client(provider_config, **kwargs)
                if client:
                    logger.debug(f"Using {provider_config.name} for {provider_type}/{asset_class}")
                    return client
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_config.name}: {e}")
                continue
                
        logger.error(f"No working providers for {provider_type}/{asset_class}")
        return None
        
    def _get_or_create_client(
        self,
        config: ProviderConfig,
        **kwargs
    ) -> Optional[Union[MarketDataClient, FundamentalsClient, NewsClient,
                       SocialSentimentClient, ExecutionClient, RiskMetricsClient]]:
        """Get or create a client instance."""
        cache_key = f"{config.name}_{config.asset_class}"
        
        if cache_key in self._client_instances:
            return self._client_instances[cache_key]
            
        # Check API key requirement
        if config.requires_api_key and config.api_key_env_var:
            import os
            api_key = os.getenv(config.api_key_env_var)
            if not api_key:
                logger.warning(f"API key not found for {config.name} (env var: {config.api_key_env_var})")
                return None
                
        # Merge init kwargs
        init_kwargs = {**config.init_kwargs, **kwargs}
        
        try:
            client = config.provider_class(**init_kwargs)
            self._client_instances[cache_key] = client
            logger.info(f"Initialized {config.name} client for {config.asset_class}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize {config.name}: {e}")
            return None
            
    def list_available_providers(self, asset_class: Optional[AssetClass] = None) -> Dict[str, List[str]]:
        """List all available providers by type."""
        result = {}
        
        for provider_type, asset_providers in self._providers.items():
            result[provider_type] = []
            
            for ac, providers in asset_providers.items():
                if asset_class and ac != asset_class:
                    continue
                    
                for provider in providers:
                    if provider.enabled and provider.priority != ProviderPriority.DISABLED:
                        provider_info = f"{provider.name} ({ac.value})"
                        if provider_info not in result[provider_type]:
                            result[provider_type].append(provider_info)
                            
        return result
        
    def check_provider_health(self) -> Dict[str, Dict[str, str]]:
        """Check health status of all providers."""
        health_status = {}
        
        for provider_type, asset_providers in self._providers.items():
            health_status[provider_type] = {}
            
            for asset_class, providers in asset_providers.items():
                for provider in providers:
                    if not provider.enabled:
                        continue
                        
                    status = "unknown"
                    try:
                        client = self._get_or_create_client(provider)
                        if client:
                            status = "healthy"
                        else:
                            status = "failed_init"
                    except Exception as e:
                        status = f"error: {str(e)[:50]}"
                        
                    key = f"{provider.name} ({asset_class.value})"
                    health_status[provider_type][key] = status
                    
        return health_status


# Global registry instance
registry = ProviderRegistry()


def register_provider(provider_type: str, config: ProviderConfig) -> None:
    """Convenience function to register a provider."""
    registry.register_provider(provider_type, config)


def get_client(
    provider_type: str,
    asset_class: AssetClass,
    provider_name: Optional[str] = None,
    **kwargs
) -> Optional[Union[MarketDataClient, FundamentalsClient, NewsClient,
                   SocialSentimentClient, ExecutionClient, RiskMetricsClient]]:
    """Convenience function to get a client."""
    return registry.get_client(provider_type, asset_class, provider_name, **kwargs)


def register_default_equity_providers() -> None:
    """Register default equity data providers."""
    # Market data
    register_provider("market_data", ProviderConfig(
        name="yahoo_finance",
        provider_class=None,  # Will be set when implementing equity adapters
        asset_class=AssetClass.EQUITY,
        priority=ProviderPriority.PRIMARY,
        cost_tier="free"
    ))
    
    # Fundamentals  
    register_provider("fundamentals", ProviderConfig(
        name="finnhub",
        provider_class=None,  # Will be set when implementing equity adapters
        asset_class=AssetClass.EQUITY,
        priority=ProviderPriority.PRIMARY,
        requires_api_key=True,
        api_key_env_var="FINNHUB_API_KEY",
        cost_tier="free"
    ))
    
    # News
    register_provider("news", ProviderConfig(
        name="finnhub_news",
        provider_class=None,
        asset_class=AssetClass.EQUITY,
        priority=ProviderPriority.PRIMARY,
        requires_api_key=True,
        api_key_env_var="FINNHUB_API_KEY",
        cost_tier="free"
    ))
    
    register_provider("news", ProviderConfig(
        name="google_news",
        provider_class=None,
        asset_class=AssetClass.EQUITY,
        priority=ProviderPriority.SECONDARY,
        cost_tier="free"
    ))


def register_default_crypto_providers() -> None:
    """Register default crypto data providers."""
    # Market data - Free tier prioritization
    # Import crypto clients
    try:
        from .crypto import CoinGeckoClient, BinancePublicClient, CryptoCompareClient
        
        register_provider("market_data", ProviderConfig(
            name="coingecko",
            provider_class=CoinGeckoClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.PRIMARY,
            rate_limit_per_minute=50,
            cost_tier="free"
        ))
        
        register_provider("market_data", ProviderConfig(
            name="binance_public",
            provider_class=BinancePublicClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.SECONDARY,
            rate_limit_per_minute=1200,
            cost_tier="free"
        ))
        
        register_provider("market_data", ProviderConfig(
            name="cryptocompare",
            provider_class=CryptoCompareClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.TERTIARY,
            requires_api_key=True,
            api_key_env_var="CRYPTOCOMPARE_API_KEY",
            rate_limit_per_minute=100,
            cost_tier="free"
        ))
        
    except ImportError as e:
        logger.warning(f"Failed to import crypto data clients: {e}. Crypto providers will not be available.")
    
    # Fundamentals (crypto tokenomics)
    try:
        from .crypto import CoinGeckoFundamentalsClient, CryptoCompareFundamentalsClient
        
        register_provider("fundamentals", ProviderConfig(
            name="coingecko_fundamentals",
            provider_class=CoinGeckoFundamentalsClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.PRIMARY,
            rate_limit_per_minute=50,
            cost_tier="free"
        ))
        
        register_provider("fundamentals", ProviderConfig(
            name="cryptocompare_fundamentals",
            provider_class=CryptoCompareFundamentalsClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.SECONDARY,
            requires_api_key=True,
            api_key_env_var="CRYPTOCOMPARE_API_KEY",
            rate_limit_per_minute=100,
            cost_tier="free"
        ))
        
    except ImportError as e:
        logger.warning(f"Failed to import crypto fundamentals clients: {e}. Crypto fundamentals will not be available.")
    
    # News (crypto sentiment and news analysis)
    try:
        from .crypto import CryptoPanicClient, CoinDeskClient
        
        register_provider("news", ProviderConfig(
            name="cryptopanic",
            provider_class=CryptoPanicClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.PRIMARY,
            api_key_env_var="CRYPTOPANIC_API_TOKEN",
            rate_limit_per_minute=15,
            cost_tier="free"
        ))
        
        register_provider("news", ProviderConfig(
            name="coindesk_rss",
            provider_class=CoinDeskClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.SECONDARY,
            rate_limit_per_minute=30,
            cost_tier="free"
        ))
        
    except ImportError as e:
        logger.warning(f"Failed to import crypto news clients: {e}. Crypto news will not be available.")
    
    # Sentiment (social sentiment analysis)
    try:
        from .crypto import RedditCryptoClient, TwitterSentimentClient
        
        register_provider("sentiment", ProviderConfig(
            name="reddit_crypto",
            provider_class=RedditCryptoClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.PRIMARY,
            rate_limit_per_minute=60,
            cost_tier="free"
        ))
        
        register_provider("sentiment", ProviderConfig(
            name="twitter_crypto",
            provider_class=TwitterSentimentClient,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.SECONDARY,
            api_key_env_var="TWITTER_BEARER_TOKEN",
            rate_limit_per_minute=15,
            cost_tier="free"
        ))
        
    except ImportError as e:
        logger.warning(f"Failed to import crypto sentiment clients: {e}. Crypto sentiment will not be available.")
    
    # Execution (crypto trading brokers)
    try:
        from .crypto import CryptoPaperBroker, CCXTBroker, HyperliquidBroker
        
        register_provider("execution", ProviderConfig(
            name="crypto_paper_broker",
            provider_class=CryptoPaperBroker,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.PRIMARY,
            cost_tier="free",
            init_kwargs={
                "initial_balance": 100000.0,
                "base_currency": "USDT",
                "enable_perpetuals": True,
                "max_leverage": 10.0
            }
        ))
        
        register_provider("execution", ProviderConfig(
            name="ccxt_broker",
            provider_class=CCXTBroker,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.SECONDARY,
            requires_api_key=True,
            cost_tier="free",  # Depends on exchange, but many offer free API access
            init_kwargs={
                "sandbox": True,  # Default to sandbox for safety
                "enable_perpetuals": True,
                "max_leverage": 20.0
            }
        ))
        
        register_provider("execution", ProviderConfig(
            name="hyperliquid_broker",
            provider_class=HyperliquidBroker,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.TERTIARY,
            requires_api_key=True,
            cost_tier="premium",
            init_kwargs={
                "testnet": True,  # Default to testnet for safety
                "max_leverage": 50.0,
                "enable_cross_margin": True
            }
        ))
        
    except ImportError as e:
        logger.warning(f"Failed to import crypto execution clients: {e}. Crypto execution will not be available.")
    
    # Risk Management (crypto risk analysis and portfolio management)
    try:
        from .crypto import CryptoRiskManager
        
        register_provider("risk", ProviderConfig(
            name="crypto_risk_manager",
            provider_class=CryptoRiskManager,
            asset_class=AssetClass.CRYPTO,
            priority=ProviderPriority.PRIMARY,
            cost_tier="free",
            init_kwargs={
                "enable_24_7_monitoring": True,
                "volatility_lookback_days": 30,
                "correlation_lookback_days": 90
            }
        ))
        
    except ImportError as e:
        logger.warning(f"Failed to import crypto risk management clients: {e}. Crypto risk management will not be available.") 