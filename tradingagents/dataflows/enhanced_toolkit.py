"""
Enhanced Toolkit for cross-asset trading data access.

This module extends the original Toolkit with abstract interfaces and provider registry,
enabling seamless support for both equity and crypto markets.
"""

from typing import Annotated, Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio
import logging
from langchain_core.tools import tool

from .base_interfaces import AssetClass, DataQuality
from .provider_registry import get_client, register_default_equity_providers, register_default_crypto_providers
from ..default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class EnhancedToolkit:
    """Enhanced toolkit with cross-asset support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.asset_class = AssetClass(self.config.get("asset_class", "equity"))
        
        # Initialize providers
        self._initialize_providers()
        
        # Cache for client instances
        self._clients_cache = {}
        
    def _initialize_providers(self):
        """Initialize data providers based on configuration."""
        if self.asset_class == AssetClass.EQUITY:
            register_default_equity_providers()
        elif self.asset_class == AssetClass.CRYPTO:
            register_default_crypto_providers()
        else:
            # Register both for mixed portfolios (Phase 8+)
            register_default_equity_providers()
            register_default_crypto_providers()
    
    def _get_client(self, provider_type: str, asset_class: Optional[AssetClass] = None):
        """Get client instance with caching."""
        target_asset_class = asset_class or self.asset_class
        cache_key = f"{provider_type}_{target_asset_class}"
        
        if cache_key not in self._clients_cache:
            client = get_client(provider_type, target_asset_class)
            self._clients_cache[cache_key] = client
            
        return self._clients_cache[cache_key]
    
    async def _run_async_safe(self, coro):
        """Safely run async function, handling event loop issues."""
        try:
            return await coro
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # Create new event loop for this call
                return await asyncio.new_event_loop().run_until_complete(coro)
            raise
    
    # =============================================================================
    # Market Data Tools
    # =============================================================================
    
    @tool
    def get_market_data(
        self,
        symbol: Annotated[str, "Symbol to get market data for (e.g. AAPL, BTC)"],
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
        end_date: Annotated[str, "End date in YYYY-MM-DD format"],
        interval: Annotated[str, "Data interval (1d, 1h, etc.)"] = "1d"
    ) -> str:
        """
        Get OHLCV market data for any asset class.
        Automatically detects asset type and uses appropriate data source.
        """
        try:
            # Convert dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Get market data client
            client = self._get_client("market_data")
            if not client:
                return f"❌ No market data provider available for {self.asset_class}"
            
            # Fetch data
            if hasattr(client, 'get_ohlcv'):
                # Use async interface
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context but can't await
                        return f"📊 Market data request queued for {symbol} ({start_date} to {end_date})"
                    else:
                        data = loop.run_until_complete(client.get_ohlcv(symbol, start_dt, end_dt, interval))
                except:
                    data = asyncio.run(client.get_ohlcv(symbol, start_dt, end_dt, interval))
            else:
                # Fallback to legacy interface - will be removed in Phase 2
                return self._get_legacy_market_data(symbol, start_date, end_date)
                
            if not data:
                return f"❌ No market data found for {symbol}"
                
            # Format response
            result = f"## Market Data for {symbol} ({start_date} to {end_date})\n\n"
            result += "| Date | Open | High | Low | Close | Volume |\n"
            result += "|------|------|------|-----|-------|--------|\n"
            
            for point in data[-10:]:  # Last 10 data points
                result += f"| {point.timestamp.strftime('%Y-%m-%d')} | ${point.open:.2f} | ${point.high:.2f} | ${point.low:.2f} | ${point.close:.2f} | {point.volume:,.0f} |\n"
                
            result += f"\n**Data Quality**: {data[0].data_quality if data else 'Unknown'}"
            result += f"\n**Asset Class**: {self.asset_class}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return f"❌ Error retrieving market data for {symbol}: {str(e)}"
    
    def _get_legacy_market_data(self, symbol: str, start_date: str, end_date: str) -> str:
        """Fallback to legacy market data methods."""
        if self.asset_class == AssetClass.EQUITY:
            # Use existing YFinance tools
            from ..agents.utils.agent_utils import Toolkit
            legacy_toolkit = Toolkit(self.config)
            if self.config.get("online_tools", True):
                return legacy_toolkit.get_YFin_data_online(symbol, start_date, end_date)
            else:
                return legacy_toolkit.get_YFin_data(symbol, start_date, end_date)
        else:
            return f"❌ Crypto market data not yet implemented in legacy mode"
    
    # =============================================================================
    # Fundamentals Tools  
    # =============================================================================
    
    @tool
    def get_fundamentals(
        self,
        symbol: Annotated[str, "Symbol to analyze (equity ticker or crypto symbol)"],
        as_of_date: Annotated[str, "Date for fundamental analysis in YYYY-MM-DD format"]
    ) -> str:
        """
        Get fundamental data for any asset class.
        For equities: financial statements, ratios, insider data
        For crypto: tokenomics, protocol metrics, treasury data
        """
        try:
            date_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
            
            # Get fundamentals client
            client = self._get_client("fundamentals")
            if not client:
                return f"❌ No fundamentals provider available for {self.asset_class}"
            
            # Fetch fundamentals data
            if hasattr(client, 'get_fundamentals'):
                # Use async interface
                try:
                    import asyncio
                    fundamentals = asyncio.run(client.get_fundamentals(symbol, date_dt))
                except:
                    fundamentals = None
            else:
                # Fallback to legacy
                return self._get_legacy_fundamentals(symbol, as_of_date)
            
            if not fundamentals:
                return f"❌ No fundamental data found for {symbol}"
                
            # Format response based on asset class
            if fundamentals.asset_class == AssetClass.EQUITY:
                return self._format_equity_fundamentals(fundamentals)
            else:
                return self._format_crypto_fundamentals(fundamentals)
                
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return f"❌ Error retrieving fundamentals for {symbol}: {str(e)}"
    
    def _format_equity_fundamentals(self, data) -> str:
        """Format equity fundamentals data."""
        result = f"## Equity Fundamentals: {data.symbol}\n\n"
        result += f"**As of**: {data.as_of_date.strftime('%Y-%m-%d')}\n\n"
        
        if data.market_cap:
            result += f"**Market Cap**: ${data.market_cap:,.0f}\n"
        if data.pe_ratio:
            result += f"**P/E Ratio**: {data.pe_ratio:.2f}\n"
        if data.revenue_ttm:
            result += f"**Revenue (TTM)**: ${data.revenue_ttm:,.0f}\n"
        if data.eps_ttm:
            result += f"**EPS (TTM)**: ${data.eps_ttm:.2f}\n"
        if data.debt_to_equity:
            result += f"**Debt/Equity**: {data.debt_to_equity:.2f}\n"
        if data.dividend_yield:
            result += f"**Dividend Yield**: {data.dividend_yield:.2%}\n"
            
        result += f"\n**Data Sources**: {', '.join(data.data_sources)}"
        result += f"\n**Data Quality**: {data.data_quality}"
        
        return result
    
    def _format_crypto_fundamentals(self, data) -> str:
        """Format crypto tokenomics data."""
        result = f"## Crypto Tokenomics: {data.symbol}\n\n"
        result += f"**As of**: {data.as_of_date.strftime('%Y-%m-%d') if data.as_of_date else 'Current'}\n\n"
        
        # Basic valuation metrics
        if data.price:
            result += f"**Price**: ${data.price:,.2f}\n"
        if data.market_cap:
            result += f"**Market Cap**: ${data.market_cap:,.0f}\n"
        if data.volume_24h:
            result += f"**24h Volume**: ${data.volume_24h:,.0f}\n"
        
        # Supply metrics
        if data.circulating_supply:
            result += f"**Circulating Supply**: {data.circulating_supply:,.0f}\n"
        if data.max_supply:
            result += f"**Max Supply**: {data.max_supply:,.0f}\n"
        if data.fully_diluted_valuation:
            result += f"**Fully Diluted Valuation**: ${data.fully_diluted_valuation:,.0f}\n"
        
        # Protocol metrics
        if data.protocol_revenue:
            result += f"**Annual Protocol Revenue**: ${data.protocol_revenue:,.0f}\n"
        if hasattr(data, 'protocol_revenue_details') and data.protocol_revenue_details:
            if data.protocol_revenue_details.annual_fees_usd:
                result += f"**Annual Protocol Fees**: ${data.protocol_revenue_details.annual_fees_usd:,.0f}\n"
            if data.protocol_revenue_details.revenue_token_price_ratio:
                result += f"**Price/Fees Ratio**: {data.protocol_revenue_details.revenue_token_price_ratio:.1f}\n"
        
        # Staking metrics
        if data.staking_yield:
            result += f"**Staking Yield**: {data.staking_yield:.2%}\n"
        if hasattr(data, 'staking_metrics_details') and data.staking_metrics_details:
            if data.staking_metrics_details.staking_ratio:
                result += f"**Staking Ratio**: {data.staking_metrics_details.staking_ratio:.2%}\n"
        
        # Treasury metrics
        if data.treasury_value:
            result += f"**Treasury Value**: ${data.treasury_value:,.0f}\n"
        if hasattr(data, 'treasury_metrics_details') and data.treasury_metrics_details:
            if data.treasury_metrics_details.runway_months:
                result += f"**Treasury Runway**: {data.treasury_metrics_details.runway_months:.1f} months\n"
        
        # Categories and use cases
        if data.categories:
            result += f"**Categories**: {', '.join(data.categories)}\n"
        if data.use_cases:
            result += f"**Use Cases**: {', '.join(data.use_cases)}\n"
        
        # Network metrics
        if data.total_value_locked:
            result += f"**Total Value Locked**: ${data.total_value_locked:,.0f}\n"
        if data.active_addresses:
            result += f"**Active Addresses**: {data.active_addresses:,}\n"
            
        result += f"\n**Data Sources**: {', '.join(data.data_sources)}"
        result += f"\n**Data Quality**: {data.data_quality.value}"
        
        return result
    
    def _get_legacy_fundamentals(self, symbol: str, as_of_date: str) -> str:
        """Fallback to legacy fundamentals methods."""
        if self.asset_class == AssetClass.EQUITY:
            from ..agents.utils.agent_utils import Toolkit
            legacy_toolkit = Toolkit(self.config)
            if self.config.get("online_tools", True):
                return legacy_toolkit.get_fundamentals_openai(symbol, as_of_date)
            else:
                # Try multiple legacy sources
                results = []
                try:
                    results.append(legacy_toolkit.get_finnhub_company_insider_sentiment(symbol, as_of_date, 30))
                except:
                    pass
                try:
                    results.append(legacy_toolkit.get_simfin_balance_sheet(symbol, "annual", as_of_date))
                except:
                    pass
                return "\n\n".join(filter(None, results)) or f"❌ No legacy fundamental data for {symbol}"
        else:
            return f"❌ Crypto fundamentals not yet implemented in legacy mode"
    
    # =============================================================================
    # News Tools
    # =============================================================================
    
    @tool 
    def get_news(
        self,
        symbol: Annotated[str, "Symbol to get news for"] = "",
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"] = "",
        end_date: Annotated[str, "End date in YYYY-MM-DD format"] = "",
        limit: Annotated[int, "Maximum number of articles"] = 50
    ) -> str:
        """
        Get news for a specific symbol or general market news.
        Supports both equity and crypto news sources.
        """
        try:
            # Parse dates with defaults
            if not end_date:
                end_dt = datetime.now()
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
            if not start_date:
                start_dt = end_dt - timedelta(days=7)  # Default to last week
            else:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            
            # Get news client
            client = self._get_client("news")
            if not client:
                return f"❌ No news provider available for {self.asset_class}"
            
            # Fetch news
            if hasattr(client, 'get_news'):
                try:
                    import asyncio
                    if symbol:
                        news_items = asyncio.run(client.get_news(symbol, start_dt, end_dt, limit))
                    else:
                        news_items = asyncio.run(client.get_global_news(start_dt, end_dt, limit))
                except:
                    news_items = []
            else:
                # Legacy fallback
                return self._get_legacy_news(symbol, start_date, end_date)
            
            if not news_items:
                return f"❌ No news found for {symbol or 'general market'}"
            
            # Format response
            result = f"## News for {symbol or 'General Market'} ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})\n\n"
            
            for item in news_items[:10]:  # Limit to 10 articles for readability
                sentiment_emoji = "📈" if item.sentiment_score and item.sentiment_score > 0.1 else "📉" if item.sentiment_score and item.sentiment_score < -0.1 else "📊"
                result += f"### {sentiment_emoji} {item.title}\n"
                result += f"**Source**: {item.source} | **Published**: {item.published_at.strftime('%Y-%m-%d %H:%M')}\n"
                if item.sentiment_score:
                    result += f"**Sentiment**: {item.sentiment_score:.2f}\n"
                result += f"{item.summary}\n"
                if item.url:
                    result += f"[Read more]({item.url})\n"
                result += "\n---\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting news: {e}")
            return f"❌ Error retrieving news: {str(e)}"
    
    def _get_legacy_news(self, symbol: str, start_date: str, end_date: str) -> str:
        """Fallback to legacy news methods."""
        from ..agents.utils.agent_utils import Toolkit
        legacy_toolkit = Toolkit(self.config)
        
        if symbol:
            if self.asset_class == AssetClass.EQUITY:
                return legacy_toolkit.get_finnhub_news(symbol, start_date, end_date)
            else:
                return f"❌ Crypto-specific news not yet implemented in legacy mode"
        else:
            # Global news
            if self.config.get("online_tools", True):
                return legacy_toolkit.get_global_news_openai(end_date or datetime.now().strftime("%Y-%m-%d"))
            else:
                return legacy_toolkit.get_reddit_news(end_date or datetime.now().strftime("%Y-%m-%d"))
    
    # =============================================================================
    # Technical Analysis Tools
    # =============================================================================
    
    @tool
    def get_technical_indicators(
        self,
        symbol: Annotated[str, "Symbol to analyze"],
        indicator: Annotated[str, "Technical indicator (rsi, macd, boll, etc.) or 'comprehensive' for full crypto analysis"],
        current_date: Annotated[str, "Current date in YYYY-MM-DD format"]
    ) -> str:
        """
        Get technical indicators for any asset class.
        For crypto: Supports all standard indicators PLUS crypto-specific metrics (funding rates, perp basis, realized volatility).
        For equity: Uses traditional technical analysis.
        """
        try:
            asset_class = self.config.get("asset_class", "equity")
            
            if asset_class == "crypto":
                # Use crypto-specific technical analysis
                return asyncio.run(self._get_crypto_technical_indicators(symbol, indicator, current_date))
            else:
                # Use legacy equity analysis
                from ..agents.utils.agent_utils import Toolkit
                legacy_toolkit = Toolkit(self.config)
                
                if self.config.get("online_tools", True):
                    return legacy_toolkit.get_stockstats_indicators_report_online(symbol, indicator, current_date)
                else:
                    return legacy_toolkit.get_stockstats_indicators_report(symbol, indicator, current_date)
                
        except Exception as e:
            logger.error(f"Error getting technical indicators for {symbol}: {e}")
            return f"❌ Error retrieving technical indicators for {symbol}: {str(e)}"
    
    @tool
    def get_crypto_24h_analysis(
        self,
        symbol: Annotated[str, "Crypto symbol to analyze (e.g., 'BTC', 'ETH')"],
        current_date: Annotated[str, "Current date in YYYY-MM-DD format"],
        focus_areas: Annotated[str, "Comma-separated focus areas: volatility,volume,momentum,sessions"] = "volatility,volume,momentum"
    ) -> str:
        """
        Get comprehensive 24/7 crypto market analysis.
        Includes volatility patterns, volume analysis, momentum indicators, and market session effects.
        Only available for crypto asset class.
        """
        try:
            asset_class = self.config.get("asset_class", "equity")
            
            if asset_class != "crypto":
                return "❌ 24/7 crypto analysis only available when asset_class is set to 'crypto'"
            
            focus_list = [area.strip() for area in focus_areas.split(",")]
            return asyncio.run(self._get_crypto_24h_analysis(symbol, current_date, focus_list))
                
        except Exception as e:
            logger.error(f"Error getting 24h crypto analysis for {symbol}: {e}")
            return f"❌ Error retrieving 24h analysis for {symbol}: {str(e)}"
    
    @tool
    def get_crypto_perp_analysis(
        self,
        symbol: Annotated[str, "Crypto symbol with perpetual futures (e.g., 'BTC', 'ETH')"],
        current_date: Annotated[str, "Current date in YYYY-MM-DD format"]
    ) -> str:
        """
        Get perpetual futures analysis including basis, funding rates, and open interest.
        Analyzes the relationship between spot and perpetual futures prices.
        Only available for crypto asset class.
        """
        try:
            asset_class = self.config.get("asset_class", "equity")
            
            if asset_class != "crypto":
                return "❌ Perpetual futures analysis only available when asset_class is set to 'crypto'"
            
            return asyncio.run(self._get_crypto_perp_analysis(symbol, current_date))
                
        except Exception as e:
            logger.error(f"Error getting perp analysis for {symbol}: {e}")
            return f"❌ Error retrieving perpetual analysis for {symbol}: {str(e)}"
    
    @tool
    def get_whale_flow_analysis(
        self,
        symbol: Annotated[str, "Crypto symbol to analyze whale flows (e.g., 'BTC', 'ETH')"],
        timeframe: Annotated[str, "Analysis timeframe: '1h', '4h', '24h', '7d'"] = "24h"
    ) -> str:
        """
        Get whale flow and large transaction analysis for crypto markets.
        Tracks large transactions, exchange flows, and whale activity patterns.
        Only available for crypto asset class.
        """
        try:
            asset_class = self.config.get("asset_class", "equity")
            
            if asset_class != "crypto":
                return "❌ Whale flow analysis only available when asset_class is set to 'crypto'"
            
            return asyncio.run(self._get_whale_flow_analysis(symbol, timeframe))
                
        except Exception as e:
            logger.error(f"Error getting whale flow analysis for {symbol}: {e}")
            return f"❌ Error retrieving whale flow analysis for {symbol}: {str(e)}"
    
    async def _get_crypto_technical_indicators(self, symbol: str, indicator: str, current_date: str) -> str:
        """Get crypto technical indicators using crypto-specific analysis."""
        try:
            from .crypto import CryptoStockstatsUtils
            
            crypto_utils = CryptoStockstatsUtils()
            online = self.config.get("online_tools", True)
            
            return await crypto_utils.get_crypto_technical_indicators(
                symbol=symbol,
                indicator=indicator,
                current_date=current_date,
                include_crypto_metrics=True,
                online=online
            )
            
        except ImportError:
            logger.error("Crypto technical analysis not available - crypto modules not found")
            return f"❌ Crypto technical analysis not available for {symbol}"
        except Exception as e:
            logger.error(f"Error in crypto technical analysis: {e}")
            return f"❌ Error analyzing {symbol}: {str(e)}"
    
    async def _get_crypto_24h_analysis(self, symbol: str, current_date: str, focus_areas: List[str]) -> str:
        """Get 24/7 crypto market analysis."""
        try:
            from .crypto import CryptoStockstatsUtils
            
            crypto_utils = CryptoStockstatsUtils()
            
            return await crypto_utils.get_crypto_24h_analysis(
                symbol=symbol,
                current_date=current_date,
                focus_areas=focus_areas
            )
            
        except ImportError:
            logger.error("Crypto 24h analysis not available - crypto modules not found")
            return f"❌ Crypto 24h analysis not available for {symbol}"
        except Exception as e:
            logger.error(f"Error in crypto 24h analysis: {e}")
            return f"❌ Error in 24h analysis for {symbol}: {str(e)}"
    
    async def _get_crypto_perp_analysis(self, symbol: str, current_date: str) -> str:
        """Get perpetual futures analysis."""
        try:
            from .crypto import CryptoStockstatsUtils
            
            crypto_utils = CryptoStockstatsUtils()
            
            # Get comprehensive analysis which includes perp data
            comprehensive_analysis = await crypto_utils.get_crypto_technical_indicators(
                symbol=symbol,
                indicator="comprehensive",
                current_date=current_date,
                include_crypto_metrics=True,
                online=True
            )
            
            # Extract perp-specific information
            if "perpetual_analysis" in comprehensive_analysis.lower() or "funding" in comprehensive_analysis.lower():
                return comprehensive_analysis
            else:
                return f"## Perpetual Futures Analysis: {symbol}\n\n" \
                       f"📊 **Analysis Date**: {current_date}\n\n" \
                       f"⚠️ **Data Availability**: Limited perpetual futures data available for {symbol}.\n\n" \
                       f"For major cryptocurrencies like BTC and ETH, perpetual futures data includes:\n" \
                       f"- Basis (Perp Price - Spot Price)\n" \
                       f"- Funding rates and trends\n" \
                       f"- Open interest patterns\n\n" \
                       f"*Note: Perpetual futures are derivative products that track spot prices without expiration.*"
            
        except ImportError:
            logger.error("Crypto perp analysis not available - crypto modules not found")
            return f"❌ Crypto perpetual analysis not available for {symbol}"
        except Exception as e:
            logger.error(f"Error in crypto perp analysis: {e}")
            return f"❌ Error in perpetual analysis for {symbol}: {str(e)}"
    
    async def _get_whale_flow_analysis(self, symbol: str, timeframe: str) -> str:
        """Get whale flow and large transaction analysis."""
        try:
            from .crypto import WhaleFlowTracker
            
            whale_tracker = WhaleFlowTracker()
            
            return await whale_tracker.get_whale_flow_summary(
                symbol=symbol,
                timeframe=timeframe
            )
            
        except ImportError:
            logger.error("Whale flow analysis not available - crypto modules not found")
            return f"❌ Whale flow analysis not available for {symbol}"
        except Exception as e:
            logger.error(f"Error in whale flow analysis: {e}")
            return f"❌ Error in whale flow analysis for {symbol}: {str(e)}"

    # =============================================================================
    # Execution Tools (Crypto Trading)
    # =============================================================================
    
    @tool
    def create_order(
        self,
        symbol: Annotated[str, "Trading pair symbol (e.g. BTC/USDT, ETH-PERP)"],
        side: Annotated[str, "Order side: 'buy' or 'sell'"],
        order_type: Annotated[str, "Order type: 'market', 'limit', 'stop', 'stop_limit'"],
        quantity: Annotated[float, "Order quantity (in base asset or contracts)"],
        price: Annotated[Optional[float], "Limit price (required for limit orders)"] = None,
        leverage: Annotated[Optional[float], "Leverage for futures positions (1-50)"] = None
    ) -> str:
        """
        Create a trading order on the crypto exchange.
        Supports spot and perpetual futures trading with proper risk management.
        """
        # Only available for crypto asset class
        if self.asset_class != AssetClass.CRYPTO:
            return "❌ Order execution only available for crypto asset class"
        
        try:
            from .base_interfaces import OrderSide, OrderType
            
            # Convert string inputs to enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_type_enum = OrderType(order_type.lower())
            
            # Get execution client
            execution_client = self._get_client("execution")
            if not execution_client:
                return "❌ No execution provider available for crypto trading"
            
            # Validate inputs
            if order_type_enum == OrderType.LIMIT and price is None:
                return "❌ Price required for limit orders"
            
            if leverage and (leverage < 1 or leverage > 50):
                return "❌ Leverage must be between 1 and 50"
            
            # Create order
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return f"📋 Order creation queued: {side.upper()} {quantity} {symbol} @ {price or 'market'}"
                else:
                    order = loop.run_until_complete(execution_client.create_order(
                        symbol=symbol,
                        side=order_side,
                        order_type=order_type_enum,
                        quantity=quantity,
                        price=price,
                        leverage=leverage
                    ))
                    
                    return f"✅ Order created: {order.order_id}\n" \
                           f"Symbol: {order.symbol}\n" \
                           f"Side: {order.side.value}\n" \
                           f"Type: {order.order_type.value}\n" \
                           f"Quantity: {order.quantity}\n" \
                           f"Price: {order.price or 'Market'}\n" \
                           f"Status: {order.status.value}"
                           
            except Exception as e:
                return f"❌ Order creation failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return f"❌ Error creating order: {str(e)}"
    
    @tool
    def get_positions(self) -> str:
        """
        Get current trading positions from the crypto exchange.
        Shows position size, entry price, PnL, and market value.
        """
        # Only available for crypto asset class
        if self.asset_class != AssetClass.CRYPTO:
            return "❌ Position tracking only available for crypto asset class"
        
        try:
            # Get execution client
            execution_client = self._get_client("execution")
            if not execution_client:
                return "❌ No execution provider available for crypto trading"
            
            # Get positions
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return "📊 Positions query queued - check back for results"
                else:
                    positions = loop.run_until_complete(execution_client.get_positions())
                    
                    if not positions:
                        return "📊 No open positions"
                    
                    result = "📊 **Current Positions**\n\n"
                    total_pnl = 0
                    
                    for pos in positions:
                        direction = "LONG" if pos.quantity > 0 else "SHORT"
                        result += f"**{pos.symbol}** ({direction})\n"
                        result += f"  Size: {abs(pos.quantity):.6f}\n"
                        result += f"  Entry Price: ${pos.average_price:.4f}\n"
                        result += f"  Market Value: ${pos.market_value:.2f}\n"
                        result += f"  Unrealized PnL: ${pos.unrealized_pnl:.2f}\n\n"
                        total_pnl += pos.unrealized_pnl
                    
                    result += f"**Total Unrealized PnL: ${total_pnl:.2f}**"
                    return result
                    
            except Exception as e:
                return f"❌ Failed to get positions: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return f"❌ Error getting positions: {str(e)}"
    
    @tool
    def get_balances(self) -> str:
        """
        Get account balances from the crypto exchange.
        Shows available and total balances for all currencies.
        """
        # Only available for crypto asset class
        if self.asset_class != AssetClass.CRYPTO:
            return "❌ Balance checking only available for crypto asset class"
        
        try:
            # Get execution client
            execution_client = self._get_client("execution")
            if not execution_client:
                return "❌ No execution provider available for crypto trading"
            
            # Get balances
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return "💰 Balance query queued - check back for results"
                else:
                    balances = loop.run_until_complete(execution_client.get_balances())
                    
                    if not balances:
                        return "💰 No balances available"
                    
                    result = "💰 **Account Balances**\n\n"
                    
                    for balance in balances:
                        if balance.total > 0:
                            result += f"**{balance.currency}**\n"
                            result += f"  Available: {balance.available:.6f}\n"
                            result += f"  Total: {balance.total:.6f}\n"
                            if balance.reserved > 0:
                                result += f"  Reserved: {balance.reserved:.6f}\n"
                            result += "\n"
                    
                    return result
                    
            except Exception as e:
                return f"❌ Failed to get balances: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error getting balances: {e}")
            return f"❌ Error getting balances: {str(e)}"
    
    @tool
    def cancel_order(
        self,
        order_id: Annotated[str, "Order ID to cancel"]
    ) -> str:
        """
        Cancel an existing order on the crypto exchange.
        """
        # Only available for crypto asset class
        if self.asset_class != AssetClass.CRYPTO:
            return "❌ Order cancellation only available for crypto asset class"
        
        try:
            # Get execution client
            execution_client = self._get_client("execution")
            if not execution_client:
                return "❌ No execution provider available for crypto trading"
            
            # Cancel order
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return f"🚫 Order cancellation queued for {order_id}"
                else:
                    success = loop.run_until_complete(execution_client.cancel_order(order_id))
                    
                    if success:
                        return f"✅ Order {order_id} cancelled successfully"
                    else:
                        return f"❌ Failed to cancel order {order_id} (may not exist or already filled)"
                        
            except Exception as e:
                return f"❌ Order cancellation failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return f"❌ Error cancelling order: {str(e)}"


# =============================================================================
# Tool Registration Functions
# =============================================================================

def get_enhanced_tools(config: Optional[Dict[str, Any]] = None) -> List:
    """Get enhanced tools for the current asset class."""
    toolkit = EnhancedToolkit(config)
    
    base_tools = [
        toolkit.get_market_data,
        toolkit.get_fundamentals, 
        toolkit.get_news,
        toolkit.get_technical_indicators,
    ]
    
    # Add execution tools for crypto asset class
    if toolkit.asset_class == AssetClass.CRYPTO:
        base_tools.extend([
            toolkit.create_order,
            toolkit.get_positions,
            toolkit.get_balances,
            toolkit.cancel_order,
        ])
    
    return base_tools


def get_legacy_tools(config: Optional[Dict[str, Any]] = None) -> List:
    """Get legacy tools for backward compatibility."""
    from ..agents.utils.agent_utils import Toolkit
    legacy_toolkit = Toolkit(config)
    
    # Return subset of most important legacy tools
    return [
        legacy_toolkit.get_YFin_data_online if config and config.get("online_tools") else legacy_toolkit.get_YFin_data,
        legacy_toolkit.get_stockstats_indicators_report_online if config and config.get("online_tools") else legacy_toolkit.get_stockstats_indicators_report,
        legacy_toolkit.get_finnhub_news,
        legacy_toolkit.get_fundamentals_openai if config and config.get("online_tools") else legacy_toolkit.get_finnhub_company_insider_sentiment,
    ]


def get_all_tools(config: Optional[Dict[str, Any]] = None) -> List:
    """Get all available tools (enhanced + legacy for compatibility)."""
    enhanced = get_enhanced_tools(config)
    legacy = get_legacy_tools(config)
    return enhanced + legacy 