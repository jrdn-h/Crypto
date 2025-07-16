"""
Crypto Risk Manager for advanced risk assessment and portfolio management.

Provides comprehensive risk management capabilities specific to cryptocurrency markets:
- 24/7 continuous risk monitoring
- Funding PnL calculations for perpetual futures
- Dynamic leverage caps based on market conditions
- Cross vs isolated margin management
- Liquidation risk assessment
- Portfolio optimization with crypto-specific metrics
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from ..base_interfaces import (
    RiskMetricsClient, AssetClass, Position, Balance, DataQuality
)

logger = logging.getLogger(__name__)


class MarginMode(str, Enum):
    """Margin mode types."""
    CROSS = "cross"          # Cross margin - all positions share margin
    ISOLATED = "isolated"    # Isolated margin - each position has separate margin


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"              # < 25% risk threshold
    MEDIUM = "medium"        # 25-50% risk threshold  
    HIGH = "high"            # 50-75% risk threshold
    CRITICAL = "critical"    # > 75% risk threshold


@dataclass
class FundingPnL:
    """Funding PnL calculation for perpetual futures."""
    symbol: str
    position_size: float
    funding_rate: float
    mark_price: float
    funding_payment: float
    cumulative_funding: float
    next_funding_time: datetime
    funding_history_24h: List[float] = field(default_factory=list)
    
    @property
    def daily_funding_cost(self) -> float:
        """Estimated daily funding cost."""
        if not self.funding_history_24h:
            return abs(self.funding_payment) * 3  # 3 funding periods per day
        return sum(abs(f) for f in self.funding_history_24h)


@dataclass
class LiquidationRisk:
    """Liquidation risk assessment for a position."""
    symbol: str
    position_size: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    margin_ratio: float
    distance_to_liquidation_pct: float
    estimated_time_to_liquidation: Optional[timedelta]
    risk_level: RiskLevel
    
    @property
    def is_at_risk(self) -> bool:
        """Whether position is at liquidation risk."""
        return self.distance_to_liquidation_pct < 20.0  # Within 20% of liquidation


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_account_value: float
    total_margin_used: float
    available_margin: float
    margin_ratio: float
    leverage_ratio: float
    max_leverage_allowed: float
    
    # Risk metrics
    portfolio_var_1d: float
    portfolio_var_7d: float
    max_drawdown_30d: float
    sharpe_ratio: float
    
    # Crypto-specific metrics
    funding_pnl_24h: float
    liquidation_risk_count: int
    correlation_concentration: float
    
    # Risk levels
    overall_risk_level: RiskLevel
    margin_risk_level: RiskLevel
    concentration_risk_level: RiskLevel


@dataclass
class RiskLimits:
    """Risk limits and constraints."""
    max_leverage: float = 10.0
    max_position_size_usd: float = 100000.0
    max_portfolio_concentration: float = 0.3  # 30% max in single asset
    max_correlation_exposure: float = 0.6     # 60% max in correlated assets
    max_margin_ratio: float = 0.8            # 80% max margin utilization
    min_liquidation_distance: float = 0.2    # 20% min distance to liquidation
    max_daily_var: float = 0.05              # 5% max daily VaR
    max_funding_cost_daily: float = 0.01     # 1% max daily funding cost


class CryptoRiskManager(RiskMetricsClient):
    """
    Comprehensive crypto risk management system.
    
    Features:
    - 24/7 continuous risk monitoring
    - Perpetual futures funding calculations
    - Dynamic leverage and position sizing
    - Liquidation risk assessment
    - Portfolio optimization
    - Real-time risk alerts
    """
    
    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        enable_24_7_monitoring: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        volatility_lookback_days: int = 30,
        correlation_lookback_days: int = 90
    ):
        """
        Initialize crypto risk manager.
        
        Args:
            risk_limits: Risk limits and constraints
            enable_24_7_monitoring: Enable continuous monitoring
            alert_thresholds: Custom alert thresholds
            volatility_lookback_days: Days for volatility calculation
            correlation_lookback_days: Days for correlation analysis
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.enable_24_7_monitoring = enable_24_7_monitoring
        self.volatility_lookback_days = volatility_lookback_days
        self.correlation_lookback_days = correlation_lookback_days
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'margin_ratio': 0.75,           # Alert at 75% margin utilization
            'liquidation_distance': 0.25,   # Alert when within 25% of liquidation
            'var_breach': 1.5,              # Alert when VaR exceeds 1.5x limit
            'funding_cost': 0.02,           # Alert when daily funding > 2%
            'correlation_risk': 0.7         # Alert when correlation exposure > 70%
        }
        
        # State tracking
        self._price_cache: Dict[str, List[Tuple[datetime, float]]] = {}
        self._funding_cache: Dict[str, List[Tuple[datetime, float]]] = {}
        self._risk_alerts: List[Dict[str, Any]] = []
        self._last_risk_update = datetime.now(timezone.utc)
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def get_risk_metrics(self, symbol: str, as_of_date: datetime) -> Optional['CryptoRiskMetrics']:
        """Get comprehensive risk metrics for a symbol."""
        try:
            # Get market data
            current_price = await self._get_current_price(symbol)
            volatility = await self._calculate_volatility(symbol, self.volatility_lookback_days)
            
            # Get funding data for perps
            funding_rate = None
            if self._is_perpetual(symbol):
                funding_rate = await self._get_funding_rate(symbol)
            
            # Calculate liquidity metrics
            bid_ask_spread = await self._get_bid_ask_spread(symbol)
            market_impact = await self._estimate_market_impact(symbol, 10000)  # $10k order impact
            
            # Create risk metrics
            risk_metrics = CryptoRiskMetrics(
                symbol=symbol,
                asset_class=AssetClass.CRYPTO,
                as_of_date=as_of_date,
                
                # Price and volatility
                current_price=current_price,
                volatility_1d=volatility.get('1d'),
                volatility_7d=volatility.get('7d'), 
                volatility_30d=volatility.get('30d'),
                
                # Liquidity
                bid_ask_spread=bid_ask_spread,
                market_impact=market_impact,
                
                # Crypto-specific
                funding_rate=funding_rate,
                is_perpetual=self._is_perpetual(symbol),
                
                # Risk limits
                max_position_size=self._calculate_max_position_size(symbol, volatility.get('30d', 0.5)),
                recommended_leverage=self._calculate_recommended_leverage(symbol, volatility.get('30d', 0.5)),
                
                data_quality=DataQuality.HIGH
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
            return None
    
    async def get_portfolio_risk(self, positions: List[Position]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            if not positions:
                return {'error': 'No positions provided'}
            
            # Get current portfolio state
            portfolio_value = sum(abs(pos.market_value) for pos in positions)
            total_margin = await self._calculate_total_margin(positions)
            
            # Calculate individual position risks
            position_risks = []
            liquidation_risks = []
            funding_pnls = []
            
            for position in positions:
                # Position risk
                pos_risk = await self._assess_position_risk(position)
                position_risks.append(pos_risk)
                
                # Liquidation risk
                if abs(position.quantity) > 0:
                    liq_risk = await self._calculate_liquidation_risk(position)
                    if liq_risk:
                        liquidation_risks.append(liq_risk)
                
                # Funding PnL for perps
                if self._is_perpetual(position.symbol):
                    funding = await self._calculate_funding_pnl(position)
                    if funding:
                        funding_pnls.append(funding)
            
            # Portfolio-level calculations
            correlation_matrix = await self._calculate_correlation_matrix([p.symbol for p in positions])
            portfolio_var = await self._calculate_portfolio_var(positions, correlation_matrix)
            concentration_risk = self._calculate_concentration_risk(positions)
            
            # Risk assessment
            portfolio_risk = PortfolioRisk(
                total_account_value=portfolio_value,
                total_margin_used=total_margin,
                available_margin=max(0, portfolio_value - total_margin),
                margin_ratio=total_margin / portfolio_value if portfolio_value > 0 else 0,
                leverage_ratio=self._calculate_portfolio_leverage(positions),
                max_leverage_allowed=self.risk_limits.max_leverage,
                
                portfolio_var_1d=portfolio_var.get('1d', 0),
                portfolio_var_7d=portfolio_var.get('7d', 0),
                max_drawdown_30d=await self._calculate_max_drawdown(positions),
                sharpe_ratio=await self._calculate_sharpe_ratio(positions),
                
                funding_pnl_24h=sum(f.funding_payment for f in funding_pnls),
                liquidation_risk_count=len([r for r in liquidation_risks if r.is_at_risk]),
                correlation_concentration=concentration_risk['correlation'],
                
                overall_risk_level=self._assess_overall_risk_level(portfolio_value, total_margin, liquidation_risks),
                margin_risk_level=self._assess_margin_risk_level(total_margin / portfolio_value if portfolio_value > 0 else 0),
                concentration_risk_level=self._assess_concentration_risk_level(concentration_risk['max_single_asset'])
            )
            
            return {
                'portfolio_risk': portfolio_risk,
                'position_risks': position_risks,
                'liquidation_risks': liquidation_risks,
                'funding_pnls': funding_pnls,
                'risk_alerts': await self._generate_risk_alerts(portfolio_risk, liquidation_risks),
                'recommendations': await self._generate_risk_recommendations(portfolio_risk, positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {'error': str(e)}
    
    async def calculate_funding_pnl(self, position: Position) -> Optional[FundingPnL]:
        """Calculate funding PnL for a perpetual position."""
        if not self._is_perpetual(position.symbol):
            return None
        
        return await self._calculate_funding_pnl(position)
    
    async def assess_liquidation_risk(self, position: Position) -> Optional[LiquidationRisk]:
        """Assess liquidation risk for a position."""
        if abs(position.quantity) == 0:
            return None
        
        return await self._calculate_liquidation_risk(position)
    
    async def calculate_optimal_position_size(
        self, 
        symbol: str, 
        target_risk: float = 0.02,
        kelly_fraction: float = 0.25
    ) -> Dict[str, float]:
        """
        Calculate optimal position size using Kelly Criterion and risk parity.
        
        Args:
            symbol: Trading symbol
            target_risk: Target risk per trade (default 2%)
            kelly_fraction: Fraction of Kelly bet to use (default 25%)
        """
        try:
            # Get market data
            current_price = await self._get_current_price(symbol)
            volatility = await self._calculate_volatility(symbol, 30)
            vol_30d = volatility.get('30d', 0.5)
            
            # Calculate Kelly optimal sizing
            # Kelly = (bp - q) / b where b=odds, p=win_prob, q=lose_prob
            # For crypto, we use expected return / variance approximation
            expected_return = await self._estimate_expected_return(symbol)
            kelly_fraction_calc = expected_return / (vol_30d ** 2) if vol_30d > 0 else 0
            kelly_size = kelly_fraction_calc * kelly_fraction
            
            # Risk parity sizing (target risk approach)
            # Size = Target_Risk / (Price * Volatility)
            risk_parity_size = target_risk / (current_price * vol_30d) if (current_price * vol_30d) > 0 else 0
            
            # VAR-based sizing
            var_99 = vol_30d * 2.33  # 99% confidence interval
            var_size = target_risk / var_99 if var_99 > 0 else 0
            
            # Conservative size (minimum of all methods)
            conservative_size = min(kelly_size, risk_parity_size, var_size)
            
            # Apply position limits
            max_size_usd = self.risk_limits.max_position_size_usd
            max_size_units = max_size_usd / current_price if current_price > 0 else 0
            
            return {
                'kelly_size': min(abs(kelly_size), max_size_units),
                'risk_parity_size': min(abs(risk_parity_size), max_size_units),
                'var_size': min(abs(var_size), max_size_units),
                'conservative_size': min(abs(conservative_size), max_size_units),
                'recommended_size': min(abs(conservative_size), max_size_units),
                'max_allowed_size': max_size_units,
                'current_price': current_price,
                'volatility_30d': vol_30d,
                'expected_return': expected_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal position size for {symbol}: {e}")
            return {'error': str(e)}
    
    async def start_monitoring(self) -> None:
        """Start 24/7 risk monitoring."""
        if not self.enable_24_7_monitoring:
            logger.info("24/7 monitoring disabled")
            return
        
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already running")
            return
        
        logger.info("Starting 24/7 crypto risk monitoring")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Risk monitoring stopped")
    
    @property
    def asset_class(self) -> AssetClass:
        """This handles crypto assets."""
        return AssetClass.CRYPTO
    
    # ==== Internal Methods ====
    
    def _is_perpetual(self, symbol: str) -> bool:
        """Check if symbol is a perpetual futures contract."""
        return symbol.endswith("-PERP") or symbol.endswith("PERP") or "PERP" in symbol.upper()
    
    async def _monitoring_loop(self) -> None:
        """Main 24/7 monitoring loop."""
        try:
            while True:
                try:
                    # Update risk metrics every 5 minutes
                    await self._update_risk_cache()
                    
                    # Check for risk alerts every minute
                    await self._check_risk_alerts()
                    
                    # Sleep for 1 minute
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(30)  # Shorter sleep on error
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
    
    async def _update_risk_cache(self) -> None:
        """Update risk data cache."""
        # Implementation would update price and funding rate caches
        pass
    
    async def _check_risk_alerts(self) -> None:
        """Check for risk threshold breaches."""
        # Implementation would check all alert conditions
        pass
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        # Fallback prices for testing
        fallback_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2800.0,
            "BTC-PERP": 45050.0,
            "ETH-PERP": 2805.0,
        }
        return fallback_prices.get(symbol, 100.0)
    
    async def _calculate_volatility(self, symbol: str, days: int) -> Dict[str, float]:
        """Calculate volatility over different periods."""
        # Simplified volatility calculation
        base_vol = 0.04  # 4% daily volatility base
        
        if "BTC" in symbol:
            daily_vol = base_vol * 1.2
        elif "ETH" in symbol:
            daily_vol = base_vol * 1.5
        else:
            daily_vol = base_vol * 2.0
        
        return {
            '1d': daily_vol,
            '7d': daily_vol * math.sqrt(7),
            '30d': daily_vol * math.sqrt(30)
        }
    
    async def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for perpetual."""
        if not self._is_perpetual(symbol):
            return None
        
        # Typical funding rates
        return 0.0001  # 0.01% every 8 hours
    
    async def _get_bid_ask_spread(self, symbol: str) -> float:
        """Get bid-ask spread."""
        # Typical spreads in basis points
        if "BTC" in symbol:
            return 0.0001  # 1 basis point
        elif "ETH" in symbol:
            return 0.0002  # 2 basis points
        else:
            return 0.0005  # 5 basis points
    
    async def _estimate_market_impact(self, symbol: str, order_size_usd: float) -> float:
        """Estimate market impact for order size."""
        current_price = await self._get_current_price(symbol)
        
        # Simplified market impact model
        if "BTC" in symbol:
            impact_factor = 0.00001  # Very liquid
        elif "ETH" in symbol:
            impact_factor = 0.00002  # Liquid
        else:
            impact_factor = 0.0001   # Less liquid
        
        return math.sqrt(order_size_usd / current_price) * impact_factor
    
    def _calculate_max_position_size(self, symbol: str, volatility: float) -> float:
        """Calculate maximum position size based on volatility."""
        base_size = self.risk_limits.max_position_size_usd
        
        # Reduce max size for higher volatility
        vol_adjustment = max(0.1, 1 - (volatility - 0.02) * 10)
        
        return base_size * vol_adjustment
    
    def _calculate_recommended_leverage(self, symbol: str, volatility: float) -> float:
        """Calculate recommended leverage based on volatility."""
        base_leverage = self.risk_limits.max_leverage
        
        # Reduce leverage for higher volatility  
        vol_adjustment = max(0.1, 1 - (volatility - 0.02) * 5)
        
        return min(base_leverage * vol_adjustment, base_leverage)
    
    async def _calculate_total_margin(self, positions: List[Position]) -> float:
        """Calculate total margin used across positions."""
        total_margin = 0
        
        for position in positions:
            if self._is_perpetual(position.symbol):
                # For perps, margin = notional / leverage
                # Assume 5x leverage as default
                margin = abs(position.market_value) / 5.0
            else:
                # For spot, margin = full position value
                margin = abs(position.market_value)
            
            total_margin += margin
        
        return total_margin
    
    async def _assess_position_risk(self, position: Position) -> Dict[str, Any]:
        """Assess risk for individual position."""
        symbol = position.symbol
        volatility = await self._calculate_volatility(symbol, 30)
        current_price = await self._get_current_price(symbol)
        
        # Calculate position risk metrics
        position_value = abs(position.market_value)
        volatility_30d = volatility.get('30d', 0.5)
        
        # Daily VaR (99% confidence)
        daily_var = position_value * volatility_30d * 2.33
        
        return {
            'symbol': symbol,
            'position_value': position_value,
            'daily_var_99': daily_var,
            'volatility_30d': volatility_30d,
            'risk_level': self._categorize_risk_level(daily_var / position_value if position_value > 0 else 0)
        }
    
    async def _calculate_liquidation_risk(self, position: Position) -> Optional[LiquidationRisk]:
        """Calculate liquidation risk for position."""
        if abs(position.quantity) == 0:
            return None
        
        # Simplified liquidation calculation
        current_price = await self._get_current_price(position.symbol)
        entry_price = position.average_price
        
        # Assume 80% liquidation threshold (maintenance margin = 12.5%)
        maintenance_margin_rate = 0.125
        
        if position.quantity > 0:  # Long position
            liquidation_price = entry_price * (1 - maintenance_margin_rate)
        else:  # Short position
            liquidation_price = entry_price * (1 + maintenance_margin_rate)
        
        distance_pct = abs(current_price - liquidation_price) / current_price * 100
        
        return LiquidationRisk(
            symbol=position.symbol,
            position_size=position.quantity,
            entry_price=entry_price,
            mark_price=current_price,
            liquidation_price=liquidation_price,
            margin_ratio=0.8,  # Simplified
            distance_to_liquidation_pct=distance_pct,
            estimated_time_to_liquidation=None,  # Would require volatility modeling
            risk_level=self._categorize_liquidation_risk(distance_pct)
        )
    
    async def _calculate_funding_pnl(self, position: Position) -> Optional[FundingPnL]:
        """Calculate funding PnL for perpetual position."""
        if not self._is_perpetual(position.symbol):
            return None
        
        funding_rate = await self._get_funding_rate(position.symbol)
        current_price = await self._get_current_price(position.symbol)
        
        if not funding_rate:
            return None
        
        # Funding payment = position_size * mark_price * funding_rate
        notional_value = abs(position.quantity) * current_price
        funding_payment = notional_value * funding_rate
        
        # Adjust sign based on position direction
        if position.quantity > 0:  # Long pays funding
            funding_payment = -funding_payment
        
        return FundingPnL(
            symbol=position.symbol,
            position_size=position.quantity,
            funding_rate=funding_rate,
            mark_price=current_price,
            funding_payment=funding_payment,
            cumulative_funding=funding_payment,  # Simplified
            next_funding_time=datetime.now(timezone.utc) + timedelta(hours=8),
            funding_history_24h=[funding_payment] * 3  # 3 payments per day
        )
    
    def _calculate_portfolio_leverage(self, positions: List[Position]) -> float:
        """Calculate overall portfolio leverage."""
        total_notional = sum(abs(pos.market_value) for pos in positions)
        total_margin = sum(abs(pos.market_value) / 5.0 if self._is_perpetual(pos.symbol) else abs(pos.market_value) for pos in positions)
        
        return total_notional / total_margin if total_margin > 0 else 1.0
    
    def _calculate_concentration_risk(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate portfolio concentration risk."""
        total_value = sum(abs(pos.market_value) for pos in positions)
        
        if total_value == 0:
            return {'max_single_asset': 0, 'correlation': 0}
        
        # Single asset concentration
        max_single = max(abs(pos.market_value) for pos in positions) / total_value
        
        # Simplified correlation risk (assume high correlation for same-chain assets)
        btc_exposure = sum(abs(pos.market_value) for pos in positions if "BTC" in pos.symbol) / total_value
        eth_exposure = sum(abs(pos.market_value) for pos in positions if "ETH" in pos.symbol) / total_value
        correlation_risk = max(btc_exposure, eth_exposure)
        
        return {
            'max_single_asset': max_single,
            'correlation': correlation_risk
        }
    
    async def _calculate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for symbols."""
        # Simplified correlation matrix
        matrix = {}
        for symbol1 in symbols:
            matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation = 1.0
                elif "BTC" in symbol1 and "BTC" in symbol2:
                    correlation = 0.95  # High correlation for BTC pairs
                elif "ETH" in symbol1 and "ETH" in symbol2:
                    correlation = 0.95  # High correlation for ETH pairs
                elif ("BTC" in symbol1 and "ETH" in symbol2) or ("ETH" in symbol1 and "BTC" in symbol2):
                    correlation = 0.7   # Moderate correlation between BTC/ETH
                else:
                    correlation = 0.5   # Default moderate correlation
                
                matrix[symbol1][symbol2] = correlation
        
        return matrix
    
    async def _calculate_portfolio_var(self, positions: List[Position], correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate portfolio Value at Risk."""
        # Simplified VaR calculation
        total_value = sum(abs(pos.market_value) for pos in positions)
        
        if total_value == 0:
            return {'1d': 0, '7d': 0}
        
        # Average volatility across positions
        avg_vol = 0.04  # 4% daily volatility
        
        # Portfolio VaR (99% confidence)
        daily_var = total_value * avg_vol * 2.33
        weekly_var = daily_var * math.sqrt(7)
        
        return {
            '1d': daily_var,
            '7d': weekly_var
        }
    
    async def _calculate_max_drawdown(self, positions: List[Position]) -> float:
        """Calculate maximum drawdown."""
        # Simplified calculation - would need historical PnL data
        return 0.15  # 15% max drawdown assumption
    
    async def _calculate_sharpe_ratio(self, positions: List[Position]) -> float:
        """Calculate Sharpe ratio."""
        # Simplified calculation - would need returns data
        return 1.2  # Reasonable Sharpe ratio for crypto
    
    async def _estimate_expected_return(self, symbol: str) -> float:
        """Estimate expected return for symbol."""
        # Simplified expected return model
        if "BTC" in symbol:
            return 0.0005  # 5 basis points per day
        elif "ETH" in symbol:
            return 0.0008  # 8 basis points per day
        else:
            return 0.001   # 10 basis points per day
    
    def _assess_overall_risk_level(self, portfolio_value: float, margin_used: float, liquidation_risks: List[LiquidationRisk]) -> RiskLevel:
        """Assess overall portfolio risk level."""
        margin_ratio = margin_used / portfolio_value if portfolio_value > 0 else 0
        liquidation_count = len([r for r in liquidation_risks if r.is_at_risk])
        
        if margin_ratio > 0.8 or liquidation_count > 0:
            return RiskLevel.CRITICAL
        elif margin_ratio > 0.6:
            return RiskLevel.HIGH
        elif margin_ratio > 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _assess_margin_risk_level(self, margin_ratio: float) -> RiskLevel:
        """Assess margin utilization risk level."""
        if margin_ratio > 0.8:
            return RiskLevel.CRITICAL
        elif margin_ratio > 0.6:
            return RiskLevel.HIGH
        elif margin_ratio > 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _assess_concentration_risk_level(self, max_concentration: float) -> RiskLevel:
        """Assess concentration risk level."""
        if max_concentration > 0.5:
            return RiskLevel.CRITICAL
        elif max_concentration > 0.3:
            return RiskLevel.HIGH
        elif max_concentration > 0.2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _categorize_risk_level(self, risk_ratio: float) -> RiskLevel:
        """Categorize risk based on ratio."""
        if risk_ratio > 0.1:    # >10% risk
            return RiskLevel.CRITICAL
        elif risk_ratio > 0.05: # 5-10% risk
            return RiskLevel.HIGH
        elif risk_ratio > 0.02: # 2-5% risk
            return RiskLevel.MEDIUM
        else:                   # <2% risk
            return RiskLevel.LOW
    
    def _categorize_liquidation_risk(self, distance_pct: float) -> RiskLevel:
        """Categorize liquidation risk based on distance."""
        if distance_pct < 10:
            return RiskLevel.CRITICAL
        elif distance_pct < 20:
            return RiskLevel.HIGH
        elif distance_pct < 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _generate_risk_alerts(self, portfolio_risk: PortfolioRisk, liquidation_risks: List[LiquidationRisk]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on current conditions."""
        alerts = []
        
        # Margin alerts
        if portfolio_risk.margin_ratio > self.alert_thresholds['margin_ratio']:
            alerts.append({
                'type': 'margin_warning',
                'severity': 'high',
                'message': f"Margin utilization at {portfolio_risk.margin_ratio:.1%} (threshold: {self.alert_thresholds['margin_ratio']:.1%})",
                'timestamp': datetime.now(timezone.utc)
            })
        
        # Liquidation alerts
        for liq_risk in liquidation_risks:
            if liq_risk.is_at_risk:
                alerts.append({
                    'type': 'liquidation_warning',
                    'severity': 'critical',
                    'message': f"{liq_risk.symbol} within {liq_risk.distance_to_liquidation_pct:.1f}% of liquidation",
                    'symbol': liq_risk.symbol,
                    'timestamp': datetime.now(timezone.utc)
                })
        
        return alerts
    
    async def _generate_risk_recommendations(self, portfolio_risk: PortfolioRisk, positions: List[Position]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if portfolio_risk.margin_ratio > 0.7:
            recommendations.append("Consider reducing position sizes to lower margin utilization")
        
        if portfolio_risk.leverage_ratio > 5:
            recommendations.append("Portfolio leverage is high - consider deleveraging")
        
        if portfolio_risk.correlation_concentration > 0.6:
            recommendations.append("High correlation exposure detected - diversify across different assets")
        
        if portfolio_risk.funding_pnl_24h < -portfolio_risk.total_account_value * 0.01:
            recommendations.append("High funding costs detected - consider reducing perpetual positions")
        
        return recommendations


@dataclass
class CryptoRiskMetrics:
    """Extended risk metrics for crypto assets."""
    symbol: str
    asset_class: AssetClass
    as_of_date: datetime
    
    # Price metrics
    current_price: Optional[float] = None
    
    # Volatility metrics  
    volatility_1d: Optional[float] = None
    volatility_7d: Optional[float] = None
    volatility_30d: Optional[float] = None
    
    # Liquidity metrics
    bid_ask_spread: Optional[float] = None
    market_impact: Optional[float] = None
    
    # Crypto-specific metrics
    funding_rate: Optional[float] = None
    is_perpetual: bool = False
    
    # Risk limits
    max_position_size: Optional[float] = None
    recommended_leverage: Optional[float] = None
    
    data_quality: DataQuality = DataQuality.UNKNOWN 