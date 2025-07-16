"""
Crypto-specific technical analysis extensions.

This module extends the existing stockstats framework with crypto-specific indicators
including funding rates, perpetual basis, realized volatility, and 24/7 market analysis.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from stockstats import wrap
    STOCKSTATS_AVAILABLE = True
except ImportError:
    STOCKSTATS_AVAILABLE = False

from ..base_interfaces import AssetClass
from .caching import CacheManager, CacheConfig
from .rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


@dataclass
class CryptoTechnicalConfig:
    """Configuration for crypto technical analysis."""
    funding_rate_lookback_hours: int = 24
    perp_basis_threshold: float = 0.01  # 1% basis threshold for alerts
    whale_flow_threshold_usd: float = 1_000_000  # $1M threshold for whale alerts
    realized_vol_window_hours: int = 24
    turnover_window_hours: int = 24
    volatility_percentile_window_days: int = 30


class CryptoTechnicalAnalyzer:
    """Crypto-specific technical analysis with 24/7 market support."""
    
    def __init__(
        self,
        config: Optional[CryptoTechnicalConfig] = None,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize crypto technical analyzer.
        
        Args:
            config: Configuration for crypto technical analysis
            cache_manager: Cache manager for storing computed indicators
            rate_limiter: Rate limiter for API calls
        """
        self.config = config or CryptoTechnicalConfig()
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        
        # Conservative rate limiting for technical analysis
        default_rate_config = RateLimitConfig(requests_per_minute=120)
        self.rate_limiter = rate_limiter or RateLimiter(default_rate_config)
        
        # Crypto-specific constants
        self.seconds_per_hour = 3600
        self.hours_per_day = 24
        
    async def analyze_crypto_technicals(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        perp_data: Optional[pd.DataFrame] = None,
        funding_data: Optional[pd.DataFrame] = None,
        on_chain_data: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict, List]]:
        """
        Comprehensive crypto technical analysis.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            ohlcv_data: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            perp_data: Optional perpetual futures data
            funding_data: Optional funding rate data
            on_chain_data: Optional on-chain metrics
            
        Returns:
            Dictionary containing all technical analysis results
        """
        cache_key = f"crypto_technical_{symbol}_{ohlcv_data.iloc[-1]['timestamp']}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached crypto technical analysis for {symbol}")
            return cached_result
        
        try:
            # Prepare data for analysis
            df = self._prepare_dataframe(ohlcv_data)
            
            # Standard technical indicators (using stockstats if available)
            standard_indicators = await self._calculate_standard_indicators(df)
            
            # Crypto-specific indicators
            crypto_indicators = await self._calculate_crypto_indicators(
                df, symbol, perp_data, funding_data, on_chain_data
            )
            
            # 24/7 market adjustments
            market_24h_analysis = await self._analyze_24h_market_dynamics(df)
            
            # Volatility analysis
            volatility_analysis = await self._analyze_crypto_volatility(df)
            
            # Volume and liquidity analysis
            volume_analysis = await self._analyze_volume_patterns(df)
            
            # Combine all analyses
            technical_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'asset_class': AssetClass.CRYPTO.value,
                'standard_indicators': standard_indicators,
                'crypto_indicators': crypto_indicators,
                'market_24h_analysis': market_24h_analysis,
                'volatility_analysis': volatility_analysis,
                'volume_analysis': volume_analysis,
                'data_quality': self._assess_data_quality(df, perp_data, funding_data)
            }
            
            # Cache the results
            await self.cache_manager.set(cache_key, technical_analysis, ttl_seconds=300)  # 5 minutes
            
            return technical_analysis
            
        except Exception as e:
            logger.error(f"Error in crypto technical analysis for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def _prepare_dataframe(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for technical analysis."""
        df = ohlcv_data.copy()
        
        # Ensure required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add common derived columns
        df['returns'] = df['close'].pct_change()
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        return df
    
    async def _calculate_standard_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate standard technical indicators using stockstats."""
        indicators = {}
        
        if not STOCKSTATS_AVAILABLE:
            logger.warning("stockstats not available, skipping standard indicators")
            return indicators
        
        try:
            # Prepare DataFrame for stockstats (needs specific column names)
            stats_df = df.copy()
            stats_df.columns = [col.lower() for col in stats_df.columns]
            
            # Wrap with stockstats
            stock = wrap(stats_df)
            
            # Calculate key indicators
            try:
                indicators['rsi_14'] = float(stock['rsi_14'].iloc[-1]) if len(stock) > 14 else None
                indicators['rsi_30'] = float(stock['rsi_30'].iloc[-1]) if len(stock) > 30 else None
            except:
                pass
                
            try:
                indicators['macd'] = float(stock['macd'].iloc[-1]) if len(stock) > 26 else None
                indicators['macd_signal'] = float(stock['macds'].iloc[-1]) if len(stock) > 26 else None
                indicators['macd_histogram'] = float(stock['macdh'].iloc[-1]) if len(stock) > 26 else None
            except:
                pass
                
            try:
                indicators['bb_upper'] = float(stock['boll_ub'].iloc[-1]) if len(stock) > 20 else None
                indicators['bb_middle'] = float(stock['boll'].iloc[-1]) if len(stock) > 20 else None
                indicators['bb_lower'] = float(stock['boll_lb'].iloc[-1]) if len(stock) > 20 else None
            except:
                pass
                
            try:
                indicators['sma_20'] = float(stock['close_20_sma'].iloc[-1]) if len(stock) > 20 else None
                indicators['sma_50'] = float(stock['close_50_sma'].iloc[-1]) if len(stock) > 50 else None
                indicators['ema_12'] = float(stock['close_12_ema'].iloc[-1]) if len(stock) > 12 else None
                indicators['ema_26'] = float(stock['close_26_ema'].iloc[-1]) if len(stock) > 26 else None
            except:
                pass
                
            try:
                indicators['atr_14'] = float(stock['atr_14'].iloc[-1]) if len(stock) > 14 else None
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Error calculating standard indicators: {e}")
        
        return indicators
    
    async def _calculate_crypto_indicators(
        self,
        df: pd.DataFrame,
        symbol: str,
        perp_data: Optional[pd.DataFrame],
        funding_data: Optional[pd.DataFrame],
        on_chain_data: Optional[Dict]
    ) -> Dict[str, Union[float, Dict]]:
        """Calculate crypto-specific indicators."""
        indicators = {}
        
        # Realized volatility (24h rolling)
        indicators['realized_volatility_24h'] = self._calculate_realized_volatility(
            df, hours=self.config.realized_vol_window_hours
        )
        
        # Price momentum indicators
        indicators['momentum_1h'] = self._calculate_momentum(df, hours=1)
        indicators['momentum_4h'] = self._calculate_momentum(df, hours=4)
        indicators['momentum_24h'] = self._calculate_momentum(df, hours=24)
        
        # Volume-weighted indicators
        indicators['vwap_24h'] = self._calculate_vwap(df, hours=24)
        indicators['volume_profile'] = self._analyze_volume_profile(df)
        
        # Perpetual futures analysis
        if perp_data is not None:
            perp_analysis = await self._analyze_perpetual_futures(df, perp_data)
            indicators['perpetual_analysis'] = perp_analysis
        
        # Funding rate analysis
        if funding_data is not None:
            funding_analysis = await self._analyze_funding_rates(funding_data)
            indicators['funding_analysis'] = funding_analysis
        
        # On-chain analysis
        if on_chain_data:
            indicators['on_chain_signals'] = await self._analyze_on_chain_signals(on_chain_data)
        
        # Liquidity and depth analysis
        indicators['liquidity_score'] = self._calculate_liquidity_score(df)
        
        return indicators
    
    def _calculate_realized_volatility(self, df: pd.DataFrame, hours: int = 24) -> Optional[float]:
        """Calculate realized volatility over specified hours."""
        try:
            if len(df) < hours:
                return None
            
            # Use log returns for realized volatility
            recent_returns = df['returns'].tail(hours).dropna()
            
            if len(recent_returns) < 2:
                return None
            
            # Annualized realized volatility
            realized_vol = recent_returns.std() * np.sqrt(365 * 24)  # 24/7 market
            return float(realized_vol)
            
        except Exception as e:
            logger.warning(f"Error calculating realized volatility: {e}")
            return None
    
    def _calculate_momentum(self, df: pd.DataFrame, hours: int) -> Optional[float]:
        """Calculate price momentum over specified hours."""
        try:
            if len(df) < hours + 1:
                return None
            
            current_price = df['close'].iloc[-1]
            past_price = df['close'].iloc[-(hours + 1)]
            
            momentum = (current_price - past_price) / past_price * 100
            return float(momentum)
            
        except Exception as e:
            logger.warning(f"Error calculating {hours}h momentum: {e}")
            return None
    
    def _calculate_vwap(self, df: pd.DataFrame, hours: int = 24) -> Optional[float]:
        """Calculate Volume Weighted Average Price."""
        try:
            if len(df) < hours:
                return None
            
            recent_data = df.tail(hours)
            typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
            
            vwap = (typical_price * recent_data['volume']).sum() / recent_data['volume'].sum()
            return float(vwap)
            
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
            return None
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume distribution patterns."""
        try:
            recent_data = df.tail(24)  # Last 24 hours
            
            # Volume statistics
            avg_volume = recent_data['volume'].mean()
            volume_std = recent_data['volume'].std()
            current_volume = recent_data['volume'].iloc[-1]
            
            # Volume relative to average
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Volume trend (increasing/decreasing)
            volume_trend = recent_data['volume'].tail(6).mean() / recent_data['volume'].head(6).mean()
            
            return {
                'avg_volume_24h': float(avg_volume),
                'current_volume_ratio': float(volume_ratio),
                'volume_trend_ratio': float(volume_trend),
                'volume_volatility': float(volume_std / avg_volume) if avg_volume > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing volume profile: {e}")
            return {}
    
    async def _analyze_perpetual_futures(
        self, 
        spot_df: pd.DataFrame, 
        perp_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze perpetual futures vs spot price dynamics."""
        try:
            # Align timestamps
            merged_df = pd.merge_asof(
                spot_df.sort_values('timestamp'),
                perp_df.sort_values('timestamp'),
                on='timestamp',
                suffixes=('_spot', '_perp')
            )
            
            if len(merged_df) == 0:
                return {}
            
            # Calculate basis (perp - spot) / spot
            merged_df['basis'] = (merged_df['close_perp'] - merged_df['close_spot']) / merged_df['close_spot']
            
            current_basis = merged_df['basis'].iloc[-1]
            avg_basis_24h = merged_df['basis'].tail(24).mean()
            basis_volatility = merged_df['basis'].tail(24).std()
            
            # Basis trend
            recent_basis = merged_df['basis'].tail(6).mean()
            past_basis = merged_df['basis'].tail(12).head(6).mean()
            basis_trend = recent_basis - past_basis
            
            return {
                'current_basis': float(current_basis),
                'avg_basis_24h': float(avg_basis_24h),
                'basis_volatility': float(basis_volatility),
                'basis_trend': float(basis_trend),
                'basis_alert': abs(current_basis) > self.config.perp_basis_threshold
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing perpetual futures: {e}")
            return {}
    
    async def _analyze_funding_rates(self, funding_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze funding rate patterns."""
        try:
            # Recent funding rates
            recent_funding = funding_df.tail(self.config.funding_rate_lookback_hours)
            
            if len(recent_funding) == 0:
                return {}
            
            current_funding = recent_funding['funding_rate'].iloc[-1]
            avg_funding_24h = recent_funding['funding_rate'].mean()
            funding_volatility = recent_funding['funding_rate'].std()
            
            # Funding rate trend
            recent_avg = recent_funding['funding_rate'].tail(6).mean()
            past_avg = recent_funding['funding_rate'].head(6).mean()
            funding_trend = recent_avg - past_avg
            
            # Extreme funding detection
            funding_percentile = (recent_funding['funding_rate'] <= current_funding).mean() * 100
            
            return {
                'current_funding_rate': float(current_funding),
                'avg_funding_24h': float(avg_funding_24h),
                'funding_volatility': float(funding_volatility),
                'funding_trend': float(funding_trend),
                'funding_percentile': float(funding_percentile),
                'extreme_funding': funding_percentile > 95 or funding_percentile < 5
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing funding rates: {e}")
            return {}
    
    async def _analyze_on_chain_signals(self, on_chain_data: Dict) -> Dict[str, Union[float, bool]]:
        """Analyze on-chain signals for whale flows and large transactions."""
        signals = {}
        
        try:
            # Whale flow analysis
            if 'large_transactions' in on_chain_data:
                large_txs = on_chain_data['large_transactions']
                
                # Count of large transactions in last 24h
                signals['whale_tx_count_24h'] = len(large_txs)
                
                # Total value of large transactions
                total_whale_volume = sum(tx.get('value_usd', 0) for tx in large_txs)
                signals['whale_volume_24h_usd'] = total_whale_volume
                
                # Whale alert
                signals['whale_alert'] = total_whale_volume > self.config.whale_flow_threshold_usd
            
            # Exchange flow analysis
            if 'exchange_flows' in on_chain_data:
                flows = on_chain_data['exchange_flows']
                signals['net_exchange_flow'] = flows.get('net_flow', 0)
                signals['exchange_inflow'] = flows.get('inflow', 0)
                signals['exchange_outflow'] = flows.get('outflow', 0)
            
            # Network activity
            if 'network_activity' in on_chain_data:
                activity = on_chain_data['network_activity']
                signals['active_addresses'] = activity.get('active_addresses', 0)
                signals['transaction_count'] = activity.get('transaction_count', 0)
                signals['avg_tx_value'] = activity.get('avg_transaction_value', 0)
            
        except Exception as e:
            logger.warning(f"Error analyzing on-chain signals: {e}")
        
        return signals
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate a liquidity score based on volume and volatility."""
        try:
            recent_data = df.tail(24)
            
            # Volume consistency (lower volatility = better liquidity)
            volume_cv = recent_data['volume'].std() / recent_data['volume'].mean()
            
            # Price stability during high volume
            volume_price_correlation = recent_data['volume'].corr(recent_data['price_range'])
            
            # Average volume
            avg_volume = recent_data['volume'].mean()
            
            # Composite liquidity score (0-100)
            liquidity_score = min(100, max(0, 
                50 * (1 - volume_cv) +  # Volume consistency
                25 * (1 - abs(volume_price_correlation)) +  # Price stability
                25 * min(1, avg_volume / 1000000)  # Volume magnitude (normalized to 1M)
            ))
            
            return float(liquidity_score)
            
        except Exception as e:
            logger.warning(f"Error calculating liquidity score: {e}")
            return 0.0
    
    async def _analyze_24h_market_dynamics(self, df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """Analyze 24/7 market dynamics specific to crypto."""
        try:
            # 24-hour statistics
            recent_24h = df.tail(24)
            
            if len(recent_24h) < 24:
                return {'warning': 'Insufficient data for 24h analysis'}
            
            # Price statistics
            high_24h = recent_24h['high'].max()
            low_24h = recent_24h['low'].min()
            open_24h = recent_24h['open'].iloc[0]
            close_24h = recent_24h['close'].iloc[-1]
            
            # 24h change
            change_24h = (close_24h - open_24h) / open_24h * 100
            
            # Volatility patterns by hour
            hourly_volatility = []
            for i in range(min(24, len(recent_24h))):
                hour_data = recent_24h.iloc[i]
                hour_vol = (hour_data['high'] - hour_data['low']) / hour_data['close']
                hourly_volatility.append(hour_vol)
            
            avg_hourly_vol = np.mean(hourly_volatility)
            max_hourly_vol = np.max(hourly_volatility)
            
            # Trading intensity (volume * volatility)
            trading_intensity = recent_24h['volume'] * (recent_24h['high'] - recent_24h['low'])
            avg_intensity = trading_intensity.mean()
            
            return {
                'high_24h': float(high_24h),
                'low_24h': float(low_24h),
                'change_24h_pct': float(change_24h),
                'range_24h_pct': float((high_24h - low_24h) / close_24h * 100),
                'avg_hourly_volatility': float(avg_hourly_vol),
                'max_hourly_volatility': float(max_hourly_vol),
                'trading_intensity': float(avg_intensity),
                'market_session': self._identify_market_session()
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing 24h market dynamics: {e}")
            return {}
    
    def _identify_market_session(self) -> str:
        """Identify current market session (crypto trades 24/7 but has patterns)."""
        try:
            current_hour_utc = datetime.utcnow().hour
            
            # Rough market session identification
            if 0 <= current_hour_utc < 8:
                return "asian_session"
            elif 8 <= current_hour_utc < 16:
                return "european_session"
            else:
                return "american_session"
                
        except:
            return "unknown_session"
    
    async def _analyze_crypto_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze crypto-specific volatility patterns."""
        try:
            # Multiple timeframe volatility
            volatilities = {}
            
            for hours in [1, 4, 12, 24]:
                if len(df) >= hours:
                    recent_data = df.tail(hours)
                    vol = recent_data['returns'].std() * np.sqrt(365 * 24)  # Annualized
                    volatilities[f'volatility_{hours}h'] = float(vol)
            
            # Volatility trend
            if len(df) >= 48:
                recent_vol = df['returns'].tail(24).std()
                past_vol = df['returns'].tail(48).head(24).std()
                vol_trend = (recent_vol - past_vol) / past_vol if past_vol > 0 else 0
                volatilities['volatility_trend'] = float(vol_trend)
            
            # Volatility percentile (relative to recent history)
            if len(df) >= self.config.volatility_percentile_window_days * 24:
                window_data = df.tail(self.config.volatility_percentile_window_days * 24)
                rolling_vols = window_data['returns'].rolling(24).std()
                current_vol = rolling_vols.iloc[-1]
                vol_percentile = (rolling_vols <= current_vol).mean() * 100
                volatilities['volatility_percentile'] = float(vol_percentile)
            
            return volatilities
            
        except Exception as e:
            logger.warning(f"Error analyzing crypto volatility: {e}")
            return {}
    
    async def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict[str, Union[float, bool]]:
        """Analyze volume patterns and turnover."""
        try:
            recent_24h = df.tail(24)
            
            if len(recent_24h) == 0:
                return {}
            
            # Volume statistics
            total_volume_24h = recent_24h['volume'].sum()
            avg_hourly_volume = recent_24h['volume'].mean()
            volume_std = recent_24h['volume'].std()
            
            # Volume trend
            first_half_vol = recent_24h['volume'].head(12).mean()
            second_half_vol = recent_24h['volume'].tail(12).mean()
            volume_trend = (second_half_vol - first_half_vol) / first_half_vol if first_half_vol > 0 else 0
            
            # Unusual volume detection
            volume_z_score = (recent_24h['volume'].iloc[-1] - avg_hourly_volume) / volume_std if volume_std > 0 else 0
            unusual_volume = abs(volume_z_score) > 2
            
            # Turnover approximation (volume / market cap proxy)
            avg_price = recent_24h['close'].mean()
            turnover_ratio = total_volume_24h / (avg_price * 1000000)  # Approximate relative turnover
            
            return {
                'volume_24h': float(total_volume_24h),
                'avg_hourly_volume': float(avg_hourly_volume),
                'volume_trend': float(volume_trend),
                'volume_z_score': float(volume_z_score),
                'unusual_volume': bool(unusual_volume),
                'turnover_ratio': float(turnover_ratio)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing volume patterns: {e}")
            return {}
    
    def _assess_data_quality(
        self,
        ohlcv_df: pd.DataFrame,
        perp_df: Optional[pd.DataFrame],
        funding_df: Optional[pd.DataFrame]
    ) -> Dict[str, Union[str, float]]:
        """Assess quality of input data for analysis."""
        quality = {}
        
        try:
            # OHLCV data quality
            quality['ohlcv_completeness'] = 1.0 - (ohlcv_df.isnull().sum().sum() / (len(ohlcv_df) * len(ohlcv_df.columns)))
            quality['ohlcv_recency_hours'] = (datetime.now() - ohlcv_df['timestamp'].max()).total_seconds() / 3600
            
            # Perp data availability
            quality['perp_data_available'] = perp_df is not None
            if perp_df is not None:
                quality['perp_completeness'] = 1.0 - (perp_df.isnull().sum().sum() / (len(perp_df) * len(perp_df.columns)))
            
            # Funding data availability
            quality['funding_data_available'] = funding_df is not None
            if funding_df is not None:
                quality['funding_completeness'] = 1.0 - (funding_df.isnull().sum().sum() / (len(funding_df) * len(funding_df.columns)))
            
            # Overall quality score
            base_score = quality['ohlcv_completeness'] * 0.6
            perp_score = quality.get('perp_completeness', 0) * 0.2 if quality['perp_data_available'] else 0
            funding_score = quality.get('funding_completeness', 0) * 0.2 if quality['funding_data_available'] else 0
            
            quality['overall_score'] = base_score + perp_score + funding_score
            
            # Quality rating
            if quality['overall_score'] >= 0.8:
                quality['rating'] = 'high'
            elif quality['overall_score'] >= 0.6:
                quality['rating'] = 'medium'
            else:
                quality['rating'] = 'low'
                
        except Exception as e:
            logger.warning(f"Error assessing data quality: {e}")
            quality['rating'] = 'unknown'
            quality['error'] = str(e)
        
        return quality
    
    async def generate_technical_report(
        self,
        symbol: str,
        analysis_results: Dict
    ) -> str:
        """Generate a comprehensive technical analysis report."""
        try:
            report_lines = []
            
            # Header
            report_lines.append(f"# Crypto Technical Analysis Report: {symbol}")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            report_lines.append("")
            
            # Standard indicators
            if 'standard_indicators' in analysis_results:
                report_lines.append("## Standard Technical Indicators")
                indicators = analysis_results['standard_indicators']
                
                if 'rsi_14' in indicators and indicators['rsi_14']:
                    rsi = indicators['rsi_14']
                    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    report_lines.append(f"- RSI (14): {rsi:.2f} ({rsi_signal})")
                
                if 'macd' in indicators and indicators['macd']:
                    report_lines.append(f"- MACD: {indicators['macd']:.6f}")
                    if 'macd_signal' in indicators:
                        signal = "Bullish" if indicators['macd'] > indicators['macd_signal'] else "Bearish"
                        report_lines.append(f"- MACD Signal: {signal}")
                
                report_lines.append("")
            
            # Crypto-specific indicators
            if 'crypto_indicators' in analysis_results:
                report_lines.append("## Crypto-Specific Indicators")
                crypto_ind = analysis_results['crypto_indicators']
                
                if 'realized_volatility_24h' in crypto_ind and crypto_ind['realized_volatility_24h']:
                    vol = crypto_ind['realized_volatility_24h']
                    report_lines.append(f"- 24h Realized Volatility: {vol:.2%}")
                
                if 'momentum_24h' in crypto_ind and crypto_ind['momentum_24h']:
                    momentum = crypto_ind['momentum_24h']
                    report_lines.append(f"- 24h Momentum: {momentum:+.2f}%")
                
                if 'perpetual_analysis' in crypto_ind:
                    perp = crypto_ind['perpetual_analysis']
                    if 'current_basis' in perp:
                        basis = perp['current_basis']
                        report_lines.append(f"- Perp Basis: {basis:+.4f} ({basis*100:+.2f}%)")
                
                if 'funding_analysis' in crypto_ind:
                    funding = crypto_ind['funding_analysis']
                    if 'current_funding_rate' in funding:
                        rate = funding['current_funding_rate']
                        report_lines.append(f"- Current Funding Rate: {rate:+.6f}")
                
                report_lines.append("")
            
            # 24/7 Market Analysis
            if 'market_24h_analysis' in analysis_results:
                report_lines.append("## 24/7 Market Dynamics")
                market = analysis_results['market_24h_analysis']
                
                if 'change_24h_pct' in market:
                    change = market['change_24h_pct']
                    report_lines.append(f"- 24h Change: {change:+.2f}%")
                
                if 'range_24h_pct' in market:
                    range_pct = market['range_24h_pct']
                    report_lines.append(f"- 24h Range: {range_pct:.2f}%")
                
                if 'market_session' in market:
                    session = market['market_session'].replace('_', ' ').title()
                    report_lines.append(f"- Current Session: {session}")
                
                report_lines.append("")
            
            # Volume Analysis
            if 'volume_analysis' in analysis_results:
                vol_analysis = analysis_results['volume_analysis']
                report_lines.append("## Volume Analysis")
                
                if 'volume_trend' in vol_analysis:
                    trend = vol_analysis['volume_trend']
                    trend_desc = "Increasing" if trend > 0.1 else "Decreasing" if trend < -0.1 else "Stable"
                    report_lines.append(f"- Volume Trend: {trend_desc} ({trend:+.2f})")
                
                if 'unusual_volume' in vol_analysis and vol_analysis['unusual_volume']:
                    report_lines.append("- ⚠️ Unusual volume detected")
                
                report_lines.append("")
            
            # Data Quality
            if 'data_quality' in analysis_results:
                quality = analysis_results['data_quality']
                if 'rating' in quality:
                    report_lines.append(f"## Data Quality: {quality['rating'].title()}")
                    if 'overall_score' in quality:
                        report_lines.append(f"Quality Score: {quality['overall_score']:.2%}")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating technical report: {e}")
            return f"Error generating technical analysis report for {symbol}: {str(e)}" 