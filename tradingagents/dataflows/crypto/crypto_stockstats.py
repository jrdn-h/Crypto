"""
Crypto-specific stockstats utilities extending the existing stockstats framework.

This module provides crypto-aware technical analysis that handles 24/7 markets,
crypto-specific data sources, and integrates with the existing TradingAgents toolkit.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Annotated, Tuple
import os

try:
    from stockstats import wrap
    STOCKSTATS_AVAILABLE = True
except ImportError:
    STOCKSTATS_AVAILABLE = False

from ..base_interfaces import AssetClass
from .caching import CacheManager, CacheConfig
from .rate_limiter import RateLimiter, RateLimitConfig
from .crypto_technical import CryptoTechnicalAnalyzer, CryptoTechnicalConfig

logger = logging.getLogger(__name__)


class CryptoStockstatsUtils:
    """Crypto-specific stockstats utilities with 24/7 market support."""
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """Initialize crypto stockstats utilities."""
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        
        # Conservative rate limiting for analysis
        default_rate_config = RateLimitConfig(requests_per_minute=60)
        self.rate_limiter = rate_limiter or RateLimiter(default_rate_config)
        
        # Initialize crypto technical analyzer
        self.technical_analyzer = CryptoTechnicalAnalyzer(
            cache_manager=self.cache_manager,
            rate_limiter=self.rate_limiter
        )
    
    async def get_crypto_technical_indicators(
        self,
        symbol: Annotated[str, "Crypto symbol (e.g., 'BTC', 'ETH')"],
        indicator: Annotated[str, "Technical indicator or 'comprehensive' for full analysis"],
        current_date: Annotated[str, "Current date in YYYY-MM-DD format"],
        data_source: Annotated[str, "Data source ('coingecko', 'binance', 'cryptocompare')"] = "auto",
        include_crypto_metrics: Annotated[bool, "Include crypto-specific metrics"] = True,
        online: Annotated[bool, "Use online data sources"] = True
    ) -> str:
        """
        Get technical indicators for crypto symbols with 24/7 market analysis.
        
        Args:
            symbol: Crypto symbol to analyze
            indicator: Specific indicator or 'comprehensive' for full analysis
            current_date: Reference date for analysis
            data_source: Preferred data source for crypto data
            include_crypto_metrics: Whether to include crypto-specific metrics
            online: Whether to fetch live data
            
        Returns:
            Formatted analysis report string
        """
        try:
            # Get OHLCV data for the symbol
            ohlcv_data = await self._fetch_crypto_ohlcv_data(
                symbol, current_date, data_source, online
            )
            
            if ohlcv_data is None or len(ohlcv_data) == 0:
                return f"❌ No OHLCV data available for {symbol} on {current_date}"
            
            # Get additional crypto data if requested
            perp_data = None
            funding_data = None
            on_chain_data = None
            
            if include_crypto_metrics:
                perp_data, funding_data, on_chain_data = await self._fetch_crypto_additional_data(
                    symbol, current_date, data_source, online
                )
            
            # Perform comprehensive analysis or specific indicator
            if indicator.lower() == 'comprehensive':
                return await self._generate_comprehensive_analysis(
                    symbol, ohlcv_data, perp_data, funding_data, on_chain_data
                )
            else:
                return await self._generate_specific_indicator(
                    symbol, ohlcv_data, indicator, current_date
                )
                
        except Exception as e:
            logger.error(f"Error getting crypto technical indicators for {symbol}: {e}")
            return f"❌ Error analyzing {symbol}: {str(e)}"
    
    async def get_crypto_24h_analysis(
        self,
        symbol: Annotated[str, "Crypto symbol to analyze"],
        current_date: Annotated[str, "Current date in YYYY-MM-DD format"],
        focus_areas: Annotated[List[str], "Focus areas for analysis"] = None
    ) -> str:
        """
        Get 24/7 crypto market analysis focusing on crypto-specific patterns.
        
        Args:
            symbol: Crypto symbol to analyze
            current_date: Reference date
            focus_areas: Specific areas to focus on (volatility, volume, momentum, etc.)
            
        Returns:
            Formatted 24/7 analysis report
        """
        try:
            # Default focus areas for crypto
            if focus_areas is None:
                focus_areas = ['volatility', 'volume', 'momentum', 'sessions']
            
            # Get recent data for 24/7 analysis
            ohlcv_data = await self._fetch_crypto_ohlcv_data(
                symbol, current_date, "auto", online=True, hours_back=168  # 1 week
            )
            
            if ohlcv_data is None or len(ohlcv_data) < 24:
                return f"❌ Insufficient data for 24h analysis of {symbol}"
            
            # Perform 24/7 analysis
            analysis_results = await self.technical_analyzer.analyze_crypto_technicals(
                symbol, ohlcv_data
            )
            
            # Generate focused report
            return await self._generate_24h_focused_report(
                symbol, analysis_results, focus_areas
            )
            
        except Exception as e:
            logger.error(f"Error in 24h analysis for {symbol}: {e}")
            return f"❌ Error in 24h analysis for {symbol}: {str(e)}"
    
    def get_crypto_stats_offline(
        self,
        symbol: Annotated[str, "Crypto symbol"],
        indicator: Annotated[str, "Technical indicator"],
        curr_date: Annotated[str, "Current date YYYY-MM-DD"],
        data_dir: Annotated[str, "Data directory path"]
    ) -> Union[float, str]:
        """
        Get crypto technical indicators from offline/cached data.
        
        This method provides offline fallback compatible with existing stockstats framework.
        """
        try:
            # Look for cached crypto data
            cache_file = os.path.join(
                data_dir, f"{symbol}-crypto-ohlcv-{curr_date}.csv"
            )
            
            if not os.path.exists(cache_file):
                # Try alternative naming conventions
                alt_files = [
                    f"{symbol}-OHLCV-{curr_date}.csv",
                    f"{symbol.upper()}-crypto-{curr_date}.csv",
                    f"{symbol.lower()}-data-{curr_date}.csv"
                ]
                
                cache_file = None
                for alt_file in alt_files:
                    alt_path = os.path.join(data_dir, alt_file)
                    if os.path.exists(alt_path):
                        cache_file = alt_path
                        break
                
                if not cache_file:
                    return f"❌ No cached data found for {symbol} on {curr_date}"
            
            # Load and process cached data
            data = pd.read_csv(cache_file)
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Use stockstats if available
            if STOCKSTATS_AVAILABLE:
                df = wrap(data)
                
                # Calculate indicator
                try:
                    df[indicator]  # Trigger calculation
                    
                    # Find matching date
                    target_date = pd.to_datetime(curr_date)
                    matching_rows = df[df['Date'].dt.date == target_date.date()]
                    
                    if not matching_rows.empty:
                        indicator_value = matching_rows[indicator].values[0]
                        return float(indicator_value) if not pd.isna(indicator_value) else "N/A"
                    else:
                        return "N/A: No data for specified date"
                        
                except Exception as e:
                    return f"❌ Error calculating {indicator}: {str(e)}"
            else:
                return "❌ stockstats library not available"
                
        except Exception as e:
            logger.error(f"Error in offline crypto stats: {e}")
            return f"❌ Error processing offline data: {str(e)}"
    
    async def _fetch_crypto_ohlcv_data(
        self,
        symbol: str,
        current_date: str,
        data_source: str = "auto",
        online: bool = True,
        hours_back: int = 168  # 1 week default
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for crypto symbol."""
        try:
            if not online:
                # Try to load from cache
                return self._load_cached_ohlcv(symbol, current_date)
            
            # For now, return mock data structure
            # In a real implementation, this would integrate with the crypto data clients
            cache_key = f"ohlcv_{symbol}_{current_date}_{hours_back}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data:
                return pd.DataFrame(cached_data)
            
            # Mock data for demonstration - replace with actual crypto data client calls
            logger.info(f"Would fetch {hours_back}h of OHLCV data for {symbol} from {data_source}")
            
            # Generate sample data structure
            end_date = pd.to_datetime(current_date)
            start_date = end_date - timedelta(hours=hours_back)
            
            dates = pd.date_range(start=start_date, end=end_date, freq='h')
            
            # Mock OHLCV data
            mock_data = {
                'timestamp': dates,
                'open': np.random.uniform(50000, 60000, len(dates)),
                'high': np.random.uniform(55000, 65000, len(dates)),
                'low': np.random.uniform(45000, 55000, len(dates)),
                'close': np.random.uniform(50000, 60000, len(dates)),
                'volume': np.random.uniform(100, 1000, len(dates))
            }
            
            df = pd.DataFrame(mock_data)
            
            # Cache the result
            await self.cache_manager.set(cache_key, df.to_dict('records'), ttl_seconds=300)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return None
    
    async def _fetch_crypto_additional_data(
        self,
        symbol: str,
        current_date: str,
        data_source: str,
        online: bool
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
        """Fetch additional crypto data (perp, funding, on-chain)."""
        try:
            # Mock additional data - replace with actual implementations
            perp_data = None
            funding_data = None
            on_chain_data = None
            
            if online and symbol in ['BTC', 'ETH']:  # Major coins might have perp data
                # Mock perpetual futures data
                            dates = pd.date_range(
                start=pd.to_datetime(current_date) - timedelta(hours=24),
                end=pd.to_datetime(current_date),
                freq='h'
            )
                
                perp_data = pd.DataFrame({
                    'timestamp': dates,
                    'close': np.random.uniform(50000, 60000, len(dates))
                })
                
                # Mock funding rate data
                funding_data = pd.DataFrame({
                    'timestamp': dates,
                    'funding_rate': np.random.uniform(-0.001, 0.001, len(dates))
                })
                
                # Mock on-chain data
                on_chain_data = {
                    'large_transactions': [
                        {'value_usd': 2000000, 'timestamp': current_date},
                        {'value_usd': 5000000, 'timestamp': current_date}
                    ],
                    'exchange_flows': {
                        'net_flow': -1000000,
                        'inflow': 5000000,
                        'outflow': 6000000
                    },
                    'network_activity': {
                        'active_addresses': 1000000,
                        'transaction_count': 300000,
                        'avg_transaction_value': 1500
                    }
                }
            
            return perp_data, funding_data, on_chain_data
            
        except Exception as e:
            logger.warning(f"Error fetching additional crypto data: {e}")
            return None, None, None
    
    def _load_cached_ohlcv(self, symbol: str, current_date: str) -> Optional[pd.DataFrame]:
        """Load OHLCV data from local cache."""
        try:
            # Implementation would load from local crypto data cache
            logger.info(f"Loading cached OHLCV data for {symbol} on {current_date}")
            return None  # Placeholder
            
        except Exception as e:
            logger.warning(f"Error loading cached data: {e}")
            return None
    
    async def _generate_comprehensive_analysis(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        perp_data: Optional[pd.DataFrame],
        funding_data: Optional[pd.DataFrame],
        on_chain_data: Optional[Dict]
    ) -> str:
        """Generate comprehensive crypto technical analysis."""
        try:
            # Perform full technical analysis
            analysis_results = await self.technical_analyzer.analyze_crypto_technicals(
                symbol, ohlcv_data, perp_data, funding_data, on_chain_data
            )
            
            # Generate comprehensive report
            report = await self.technical_analyzer.generate_technical_report(
                symbol, analysis_results
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return f"❌ Error generating comprehensive analysis for {symbol}: {str(e)}"
    
    async def _generate_specific_indicator(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        indicator: str,
        current_date: str
    ) -> str:
        """Generate specific technical indicator analysis."""
        try:
            if not STOCKSTATS_AVAILABLE:
                return f"❌ stockstats library not available for {indicator} calculation"
            
            # Prepare data for stockstats
            df = ohlcv_data.copy()
            df.columns = [col.lower() for col in df.columns]
            
            # Convert timestamp to date for stockstats compatibility
            df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
            
            # Wrap with stockstats
            stock = wrap(df)
            
            # Calculate specific indicator
            try:
                stock[indicator]  # Trigger calculation
                
                # Get the latest value
                latest_value = stock[indicator].iloc[-1]
                
                if pd.isna(latest_value):
                    return f"❌ Unable to calculate {indicator} for {symbol} - insufficient data"
                
                # Format result with context
                indicator_interpretation = self._interpret_indicator(
                    indicator, latest_value, stock
                )
                
                return f"## {symbol} - {indicator.upper()}\n\n" \
                       f"**Value**: {latest_value:.6f}\n" \
                       f"**Interpretation**: {indicator_interpretation}\n" \
                       f"**Date**: {current_date}\n\n" \
                       f"*Note: Crypto markets operate 24/7. This analysis considers continuous trading patterns.*"
                
            except Exception as e:
                return f"❌ Error calculating {indicator} for {symbol}: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error generating specific indicator: {e}")
            return f"❌ Error analyzing {indicator} for {symbol}: {str(e)}"
    
    def _interpret_indicator(self, indicator: str, value: float, stock_data: pd.DataFrame) -> str:
        """Provide interpretation for technical indicators."""
        try:
            indicator_lower = indicator.lower()
            
            # RSI interpretation
            if 'rsi' in indicator_lower:
                if value > 70:
                    return f"Overbought territory ({value:.1f} > 70). Potential selling pressure."
                elif value < 30:
                    return f"Oversold territory ({value:.1f} < 30). Potential buying opportunity."
                else:
                    return f"Neutral territory ({value:.1f}). No extreme conditions."
            
            # MACD interpretation
            elif 'macd' in indicator_lower and not 'signal' in indicator_lower:
                try:
                    macd_signal = stock_data['macds'].iloc[-1]
                    if value > macd_signal:
                        return f"Above signal line ({value:.6f} > {macd_signal:.6f}). Bullish momentum."
                    else:
                        return f"Below signal line ({value:.6f} < {macd_signal:.6f}). Bearish momentum."
                except:
                    return f"MACD value: {value:.6f}. Compare with signal line for direction."
            
            # Bollinger Bands interpretation
            elif 'boll' in indicator_lower:
                if 'ub' in indicator_lower:
                    return f"Upper Bollinger Band: {value:.2f}. Price above this may indicate overbought conditions."
                elif 'lb' in indicator_lower:
                    return f"Lower Bollinger Band: {value:.2f}. Price below this may indicate oversold conditions."
                else:
                    return f"Bollinger Middle (20 SMA): {value:.2f}. Dynamic support/resistance level."
            
            # Moving averages interpretation
            elif 'sma' in indicator_lower or 'ema' in indicator_lower:
                try:
                    current_price = stock_data['close'].iloc[-1]
                    if current_price > value:
                        return f"Price ({current_price:.2f}) above moving average ({value:.2f}). Bullish signal."
                    else:
                        return f"Price ({current_price:.2f}) below moving average ({value:.2f}). Bearish signal."
                except:
                    return f"Moving average: {value:.2f}. Compare with current price for trend direction."
            
            # ATR interpretation
            elif 'atr' in indicator_lower:
                return f"Average True Range: {value:.2f}. Represents recent volatility level."
            
            # Default interpretation
            else:
                return f"Current value: {value:.6f}. Consult technical analysis resources for interpretation."
                
        except Exception as e:
            logger.warning(f"Error interpreting indicator {indicator}: {e}")
            return f"Current value: {value:.6f}"
    
    async def _generate_24h_focused_report(
        self,
        symbol: str,
        analysis_results: Dict,
        focus_areas: List[str]
    ) -> str:
        """Generate focused 24/7 analysis report."""
        try:
            report_lines = []
            
            report_lines.append(f"# 24/7 Crypto Analysis: {symbol}")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            report_lines.append("")
            
            # Market dynamics (always included)
            if 'market_24h_analysis' in analysis_results:
                market = analysis_results['market_24h_analysis']
                report_lines.append("## 24-Hour Market Summary")
                
                if 'change_24h_pct' in market:
                    change = market['change_24h_pct']
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    report_lines.append(f"- {direction} 24h Change: **{change:+.2f}%**")
                
                if 'range_24h_pct' in market:
                    range_pct = market['range_24h_pct']
                    report_lines.append(f"- 📊 24h Range: **{range_pct:.2f}%**")
                
                if 'market_session' in market:
                    session = market['market_session'].replace('_', ' ').title()
                    report_lines.append(f"- 🌍 Current Session: **{session}**")
                
                report_lines.append("")
            
            # Focus area analysis
            for focus_area in focus_areas:
                if focus_area == 'volatility' and 'volatility_analysis' in analysis_results:
                    vol_analysis = analysis_results['volatility_analysis']
                    report_lines.append("## Volatility Analysis")
                    
                    for timeframe in ['1h', '4h', '24h']:
                        key = f'volatility_{timeframe}'
                        if key in vol_analysis:
                            vol = vol_analysis[key]
                            report_lines.append(f"- {timeframe.upper()} Volatility: {vol:.1%}")
                    
                    if 'volatility_trend' in vol_analysis:
                        trend = vol_analysis['volatility_trend']
                        trend_desc = "Increasing" if trend > 0.1 else "Decreasing" if trend < -0.1 else "Stable"
                        report_lines.append(f"- Volatility Trend: {trend_desc}")
                    
                    report_lines.append("")
                
                elif focus_area == 'volume' and 'volume_analysis' in analysis_results:
                    vol_analysis = analysis_results['volume_analysis']
                    report_lines.append("## Volume Analysis")
                    
                    if 'volume_trend' in vol_analysis:
                        trend = vol_analysis['volume_trend']
                        if trend > 0.2:
                            report_lines.append("- 📈 Volume: **Strongly Increasing**")
                        elif trend > 0.1:
                            report_lines.append("- 📈 Volume: **Increasing**")
                        elif trend < -0.2:
                            report_lines.append("- 📉 Volume: **Strongly Decreasing**")
                        elif trend < -0.1:
                            report_lines.append("- 📉 Volume: **Decreasing**")
                        else:
                            report_lines.append("- ➡️ Volume: **Stable**")
                    
                    if 'unusual_volume' in vol_analysis and vol_analysis['unusual_volume']:
                        report_lines.append("- ⚠️ **Unusual volume detected** - Elevated trading activity")
                    
                    report_lines.append("")
                
                elif focus_area == 'momentum' and 'crypto_indicators' in analysis_results:
                    crypto_ind = analysis_results['crypto_indicators']
                    report_lines.append("## Momentum Analysis")
                    
                    for timeframe in ['1h', '4h', '24h']:
                        key = f'momentum_{timeframe}'
                        if key in crypto_ind and crypto_ind[key] is not None:
                            momentum = crypto_ind[key]
                            if abs(momentum) > 5:
                                strength = "Strong" if abs(momentum) > 10 else "Moderate"
                                direction = "Bullish" if momentum > 0 else "Bearish"
                                report_lines.append(f"- {timeframe.upper()}: **{strength} {direction}** ({momentum:+.2f}%)")
                            else:
                                report_lines.append(f"- {timeframe.upper()}: Neutral ({momentum:+.2f}%)")
                    
                    report_lines.append("")
            
            # Data quality note
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']['rating']
                report_lines.append(f"*Analysis based on {quality} quality data*")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating 24h focused report: {e}")
            return f"❌ Error generating 24h analysis report for {symbol}: {str(e)}"


# Convenience functions for backward compatibility
async def get_crypto_technical_indicators_async(
    symbol: str,
    indicator: str,
    current_date: str,
    data_source: str = "auto",
    include_crypto_metrics: bool = True,
    online: bool = True
) -> str:
    """Async convenience function for crypto technical indicators."""
    utils = CryptoStockstatsUtils()
    return await utils.get_crypto_technical_indicators(
        symbol, indicator, current_date, data_source, include_crypto_metrics, online
    )


def get_crypto_stats_sync(
    symbol: str,
    indicator: str,
    curr_date: str,
    data_dir: str
) -> Union[float, str]:
    """Synchronous function for offline crypto stats (backward compatibility)."""
    utils = CryptoStockstatsUtils()
    return utils.get_crypto_stats_offline(symbol, indicator, curr_date, data_dir) 