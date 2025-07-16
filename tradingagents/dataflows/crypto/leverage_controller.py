"""
Dynamic Leverage Controller for Crypto Trading.

Provides intelligent leverage management that adapts to market conditions:
- Dynamic leverage caps based on volatility
- Liquidity-adjusted position sizing
- Market regime detection for leverage adjustment
- Risk-based leverage optimization
- Real-time leverage monitoring and alerts
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics

from ..base_interfaces import Position
from .crypto_risk_manager import RiskLevel

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"           # Strong uptrend, low volatility
    BEAR_MARKET = "bear_market"           # Strong downtrend, moderate volatility
    SIDEWAYS = "sideways"                 # Range-bound, low volatility
    HIGH_VOLATILITY = "high_volatility"   # High volatility regardless of direction
    CRISIS = "crisis"                     # Extreme volatility, potential liquidations
    RECOVERY = "recovery"                 # Post-crisis stabilization


class LeverageClass(str, Enum):
    """Leverage classification for different assets."""
    ULTRA_CONSERVATIVE = "ultra_conservative"  # 1-2x max
    CONSERVATIVE = "conservative"              # 2-5x max
    MODERATE = "moderate"                      # 5-10x max
    AGGRESSIVE = "aggressive"                  # 10-20x max
    EXTREME = "extreme"                        # 20x+ max


@dataclass
class MarketConditions:
    """Current market conditions for leverage calculation."""
    volatility_regime: str              # "low", "medium", "high", "extreme"
    liquidity_score: float              # 0-1 scale
    trend_strength: float               # -1 to 1 (-1=strong down, 1=strong up)
    correlation_risk: float             # 0-1 scale
    funding_rate_stress: float          # 0-1 scale
    vix_equivalent: float               # Crypto volatility index equivalent
    market_cap_stability: float         # Market cap change volatility
    
    @property
    def overall_risk_score(self) -> float:
        """Combined risk score from all factors."""
        volatility_weight = 0.3
        liquidity_weight = 0.2
        correlation_weight = 0.15
        funding_weight = 0.15
        trend_weight = 0.1
        vix_weight = 0.1
        
        # Convert trend to risk (extreme moves in either direction are risky)
        trend_risk = abs(self.trend_strength)
        
        risk_score = (
            self._volatility_to_risk() * volatility_weight +
            (1 - self.liquidity_score) * liquidity_weight +
            self.correlation_risk * correlation_weight +
            self.funding_rate_stress * funding_weight +
            trend_risk * trend_weight +
            min(self.vix_equivalent / 100, 1.0) * vix_weight
        )
        
        return min(1.0, risk_score)
    
    def _volatility_to_risk(self) -> float:
        """Convert volatility regime to risk score."""
        mapping = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
            "extreme": 1.0
        }
        return mapping.get(self.volatility_regime, 0.5)


@dataclass
class LeverageRecommendation:
    """Leverage recommendation with reasoning."""
    symbol: str
    current_leverage: float
    recommended_leverage: float
    max_allowed_leverage: float
    confidence: float               # 0-1 confidence in recommendation
    reasoning: List[str]
    risk_level: RiskLevel
    market_regime: MarketRegime
    
    # Specific adjustments
    volatility_adjustment: float
    liquidity_adjustment: float
    correlation_adjustment: float
    regime_adjustment: float
    
    @property
    def leverage_change(self) -> float:
        """Change in leverage recommendation."""
        return self.recommended_leverage - self.current_leverage
    
    @property
    def should_adjust(self) -> bool:
        """Whether leverage should be adjusted."""
        return abs(self.leverage_change) > 0.5 and self.confidence > 0.6


class DynamicLeverageController:
    """
    Advanced dynamic leverage controller for crypto trading.
    
    Features:
    - Real-time market regime detection
    - Volatility-adjusted leverage caps
    - Liquidity-based position sizing
    - Cross-asset correlation adjustments
    - Risk-based leverage optimization
    """
    
    def __init__(
        self,
        base_max_leverage: float = 10.0,
        conservative_mode: bool = False,
        volatility_lookback_days: int = 30,
        liquidity_threshold: float = 0.3,
        enable_regime_detection: bool = True
    ):
        """
        Initialize dynamic leverage controller.
        
        Args:
            base_max_leverage: Base maximum leverage in normal conditions
            conservative_mode: Enable conservative leverage caps
            volatility_lookback_days: Days to look back for volatility calculation
            liquidity_threshold: Minimum liquidity score for full leverage
            enable_regime_detection: Enable market regime detection
        """
        self.base_max_leverage = base_max_leverage
        self.conservative_mode = conservative_mode
        self.volatility_lookback_days = volatility_lookback_days
        self.liquidity_threshold = liquidity_threshold
        self.enable_regime_detection = enable_regime_detection
        
        # Asset classification
        self._asset_classifications = {
            'BTC': LeverageClass.MODERATE,
            'ETH': LeverageClass.MODERATE,
            'ADA': LeverageClass.CONSERVATIVE,
            'SOL': LeverageClass.AGGRESSIVE,
            'DOGE': LeverageClass.CONSERVATIVE,
            'MATIC': LeverageClass.CONSERVATIVE,
            'LINK': LeverageClass.CONSERVATIVE,
            'AVAX': LeverageClass.AGGRESSIVE,
        }
        
        # Leverage class limits
        self._leverage_class_limits = {
            LeverageClass.ULTRA_CONSERVATIVE: 2.0,
            LeverageClass.CONSERVATIVE: 5.0,
            LeverageClass.MODERATE: 10.0,
            LeverageClass.AGGRESSIVE: 20.0,
            LeverageClass.EXTREME: 50.0
        }
        
        # Market regime adjustments
        self._regime_adjustments = {
            MarketRegime.BULL_MARKET: 1.2,      # 20% higher leverage in bull markets
            MarketRegime.BEAR_MARKET: 0.7,      # 30% lower leverage in bear markets  
            MarketRegime.SIDEWAYS: 1.0,         # Normal leverage in sideways markets
            MarketRegime.HIGH_VOLATILITY: 0.5,  # 50% lower in high vol
            MarketRegime.CRISIS: 0.2,           # 80% lower in crisis
            MarketRegime.RECOVERY: 0.8          # 20% lower in recovery
        }
        
        # State tracking
        self._market_conditions_cache: Optional[MarketConditions] = None
        self._last_market_update = datetime.now(timezone.utc)
        self._volatility_cache: Dict[str, Dict[str, float]] = {}
        
    async def calculate_optimal_leverage(
        self, 
        symbol: str, 
        position: Optional[Position] = None,
        account_balance: float = 100000.0
    ) -> LeverageRecommendation:
        """
        Calculate optimal leverage for a symbol based on current market conditions.
        
        Args:
            symbol: Trading symbol
            position: Current position (if any)
            account_balance: Total account balance
        """
        try:
            # Get current market conditions
            market_conditions = await self._get_market_conditions(symbol)
            
            # Get asset classification
            asset_class = self._classify_asset(symbol)
            base_max = self._leverage_class_limits.get(asset_class, self.base_max_leverage)
            
            # Apply conservative mode
            if self.conservative_mode:
                base_max *= 0.7
            
            # Calculate adjustments
            volatility_adj = await self._calculate_volatility_adjustment(symbol, market_conditions)
            liquidity_adj = await self._calculate_liquidity_adjustment(symbol, market_conditions)
            correlation_adj = await self._calculate_correlation_adjustment(symbol)
            regime_adj = await self._calculate_regime_adjustment(market_conditions)
            
            # Combine adjustments
            total_adjustment = volatility_adj * liquidity_adj * correlation_adj * regime_adj
            recommended_leverage = base_max * total_adjustment
            
            # Apply absolute limits
            recommended_leverage = max(1.0, min(recommended_leverage, base_max))
            
            # Get current leverage
            current_leverage = position.quantity / (position.market_value / await self._get_current_price(symbol)) if position else 1.0
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                symbol, volatility_adj, liquidity_adj, correlation_adj, regime_adj, market_conditions
            )
            
            # Assess confidence
            confidence = self._calculate_confidence(market_conditions, volatility_adj, liquidity_adj)
            
            # Detect market regime
            market_regime = await self._detect_market_regime(symbol, market_conditions)
            
            # Assess risk level
            risk_level = self._assess_risk_level(recommended_leverage, base_max, market_conditions)
            
            return LeverageRecommendation(
                symbol=symbol,
                current_leverage=current_leverage,
                recommended_leverage=recommended_leverage,
                max_allowed_leverage=base_max,
                confidence=confidence,
                reasoning=reasoning,
                risk_level=risk_level,
                market_regime=market_regime,
                volatility_adjustment=volatility_adj,
                liquidity_adjustment=liquidity_adj,
                correlation_adjustment=correlation_adj,
                regime_adjustment=regime_adj
            )
            
        except Exception as e:
            logger.error(f"Error calculating optimal leverage for {symbol}: {e}")
            # Return conservative fallback
            return LeverageRecommendation(
                symbol=symbol,
                current_leverage=1.0,
                recommended_leverage=2.0,
                max_allowed_leverage=5.0,
                confidence=0.5,
                reasoning=["Error in calculation - using conservative fallback"],
                risk_level=RiskLevel.HIGH,
                market_regime=MarketRegime.CRISIS,
                volatility_adjustment=0.5,
                liquidity_adjustment=0.5,
                correlation_adjustment=0.5,
                regime_adjustment=0.5
            )
    
    async def monitor_leverage_limits(self, positions: List[Position]) -> Dict[str, Any]:
        """
        Monitor current leverage limits across all positions.
        
        Args:
            positions: Current trading positions
        """
        try:
            perp_positions = [p for p in positions if self._is_perpetual(p.symbol)]
            
            if not perp_positions:
                return {'message': 'No perpetual positions to monitor'}
            
            leverage_analysis = []
            violations = []
            warnings = []
            
            for position in perp_positions:
                recommendation = await self.calculate_optimal_leverage(position.symbol, position)
                
                # Calculate actual leverage
                current_price = await self._get_current_price(position.symbol)
                position_value = abs(position.quantity * current_price)
                margin_used = position_value / 5.0  # Simplified: assume 5x leverage
                actual_leverage = position_value / margin_used if margin_used > 0 else 1.0
                
                leverage_analysis.append({
                    'symbol': position.symbol,
                    'current_leverage': actual_leverage,
                    'recommended_leverage': recommendation.recommended_leverage,
                    'max_allowed_leverage': recommendation.max_allowed_leverage,
                    'risk_level': recommendation.risk_level,
                    'market_regime': recommendation.market_regime,
                    'confidence': recommendation.confidence,
                    'should_adjust': recommendation.should_adjust
                })
                
                # Check for violations
                if actual_leverage > recommendation.max_allowed_leverage:
                    violations.append({
                        'symbol': position.symbol,
                        'current_leverage': actual_leverage,
                        'max_allowed': recommendation.max_allowed_leverage,
                        'excess_leverage': actual_leverage - recommendation.max_allowed_leverage,
                        'severity': 'critical'
                    })
                elif actual_leverage > recommendation.recommended_leverage * 1.2:  # 20% buffer
                    warnings.append({
                        'symbol': position.symbol,
                        'current_leverage': actual_leverage,
                        'recommended': recommendation.recommended_leverage,
                        'excess_leverage': actual_leverage - recommendation.recommended_leverage,
                        'severity': 'warning'
                    })
            
            # Calculate portfolio-level metrics
            total_notional = sum(abs(p.market_value) for p in perp_positions)
            portfolio_leverage = total_notional / sum(abs(p.market_value) / 5.0 for p in perp_positions)
            
            # Get overall market conditions
            market_conditions = await self._get_market_conditions("BTC/USDT")  # Use BTC as proxy
            overall_regime = await self._detect_market_regime("BTC/USDT", market_conditions)
            
            return {
                'total_positions': len(perp_positions),
                'portfolio_leverage': portfolio_leverage,
                'violations': violations,
                'warnings': warnings,
                'leverage_analysis': leverage_analysis,
                'market_regime': overall_regime,
                'market_risk_score': market_conditions.overall_risk_score,
                'recommendations': self._generate_portfolio_recommendations(
                    violations, warnings, portfolio_leverage, overall_regime
                )
            }
            
        except Exception as e:
            logger.error(f"Error monitoring leverage limits: {e}")
            return {'error': str(e)}
    
    async def adjust_leverage_for_regime(self, market_regime: MarketRegime) -> Dict[str, float]:
        """
        Get leverage adjustments for different market regimes.
        
        Args:
            market_regime: Current market regime
        """
        regime_multiplier = self._regime_adjustments.get(market_regime, 1.0)
        
        # Calculate adjusted limits for each asset class
        adjusted_limits = {}
        for asset_class, base_limit in self._leverage_class_limits.items():
            adjusted_limit = base_limit * regime_multiplier
            adjusted_limits[asset_class.value] = max(1.0, adjusted_limit)
        
        return {
            'market_regime': market_regime,
            'regime_multiplier': regime_multiplier,
            'adjusted_limits': adjusted_limits,
            'description': self._get_regime_description(market_regime)
        }
    
    # ==== Internal Methods ====
    
    def _is_perpetual(self, symbol: str) -> bool:
        """Check if symbol is a perpetual futures contract."""
        return symbol.endswith("-PERP") or symbol.endswith("PERP") or "PERP" in symbol.upper()
    
    def _classify_asset(self, symbol: str) -> LeverageClass:
        """Classify asset for leverage limits."""
        # Extract base asset
        base_asset = symbol.split('/')[0].split('-')[0]
        
        return self._asset_classifications.get(base_asset, LeverageClass.CONSERVATIVE)
    
    async def _get_market_conditions(self, symbol: str) -> MarketConditions:
        """Get current market conditions."""
        # Check cache
        if (self._market_conditions_cache and 
            (datetime.now(timezone.utc) - self._last_market_update).total_seconds() < 300):  # 5 min cache
            return self._market_conditions_cache
        
        # Calculate market conditions
        volatility = await self._calculate_market_volatility(symbol)
        liquidity_score = await self._calculate_liquidity_score(symbol)
        trend_strength = await self._calculate_trend_strength(symbol)
        correlation_risk = await self._calculate_market_correlation_risk()
        funding_stress = await self._calculate_funding_stress(symbol)
        vix_equivalent = volatility * 100  # Convert to VIX-like scale
        market_cap_stability = await self._calculate_market_cap_stability()
        
        # Classify volatility regime
        if volatility < 0.02:
            vol_regime = "low"
        elif volatility < 0.05:
            vol_regime = "medium"
        elif volatility < 0.10:
            vol_regime = "high"
        else:
            vol_regime = "extreme"
        
        conditions = MarketConditions(
            volatility_regime=vol_regime,
            liquidity_score=liquidity_score,
            trend_strength=trend_strength,
            correlation_risk=correlation_risk,
            funding_rate_stress=funding_stress,
            vix_equivalent=vix_equivalent,
            market_cap_stability=market_cap_stability
        )
        
        # Cache conditions
        self._market_conditions_cache = conditions
        self._last_market_update = datetime.now(timezone.utc)
        
        return conditions
    
    async def _calculate_volatility_adjustment(self, symbol: str, market_conditions: MarketConditions) -> float:
        """Calculate volatility-based leverage adjustment."""
        vol_mapping = {
            "low": 1.2,      # 20% higher leverage in low vol
            "medium": 1.0,   # Normal leverage
            "high": 0.6,     # 40% lower leverage in high vol
            "extreme": 0.3   # 70% lower leverage in extreme vol
        }
        
        return vol_mapping.get(market_conditions.volatility_regime, 1.0)
    
    async def _calculate_liquidity_adjustment(self, symbol: str, market_conditions: MarketConditions) -> float:
        """Calculate liquidity-based leverage adjustment."""
        liquidity_score = market_conditions.liquidity_score
        
        if liquidity_score >= 0.8:
            return 1.0      # Full leverage for high liquidity
        elif liquidity_score >= 0.6:
            return 0.8      # 20% reduction for medium liquidity
        elif liquidity_score >= 0.4:
            return 0.6      # 40% reduction for low liquidity
        else:
            return 0.3      # 70% reduction for very low liquidity
    
    async def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation-based leverage adjustment."""
        # Simplified correlation adjustment
        # In practice, would analyze correlation with other positions
        
        if "BTC" in symbol or "ETH" in symbol:
            return 1.0      # Major assets get full leverage
        else:
            return 0.8      # Alt coins get reduced leverage due to correlation risk
    
    async def _calculate_regime_adjustment(self, market_conditions: MarketConditions) -> float:
        """Calculate market regime-based adjustment."""
        # Detect regime based on market conditions
        if market_conditions.volatility_regime == "extreme":
            return 0.2  # Crisis mode
        elif market_conditions.overall_risk_score > 0.8:
            return 0.5  # High risk
        elif market_conditions.overall_risk_score > 0.6:
            return 0.7  # Medium risk
        else:
            return 1.0  # Normal conditions
    
    async def _detect_market_regime(self, symbol: str, market_conditions: MarketConditions) -> MarketRegime:
        """Detect current market regime."""
        if not self.enable_regime_detection:
            return MarketRegime.SIDEWAYS
        
        vol_regime = market_conditions.volatility_regime
        trend = market_conditions.trend_strength
        
        if vol_regime == "extreme":
            return MarketRegime.CRISIS
        elif vol_regime == "high":
            return MarketRegime.HIGH_VOLATILITY
        elif trend > 0.5:
            return MarketRegime.BULL_MARKET
        elif trend < -0.5:
            return MarketRegime.BEAR_MARKET
        elif abs(trend) < 0.2:
            return MarketRegime.SIDEWAYS
        else:
            return MarketRegime.RECOVERY
    
    def _assess_risk_level(self, recommended_leverage: float, max_leverage: float, market_conditions: MarketConditions) -> RiskLevel:
        """Assess risk level for leverage recommendation."""
        leverage_ratio = recommended_leverage / max_leverage
        overall_risk = market_conditions.overall_risk_score
        
        combined_risk = (leverage_ratio + overall_risk) / 2
        
        if combined_risk > 0.8:
            return RiskLevel.CRITICAL
        elif combined_risk > 0.6:
            return RiskLevel.HIGH
        elif combined_risk > 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_reasoning(
        self, 
        symbol: str, 
        vol_adj: float, 
        liq_adj: float, 
        corr_adj: float, 
        regime_adj: float,
        market_conditions: MarketConditions
    ) -> List[str]:
        """Generate reasoning for leverage recommendation."""
        reasoning = []
        
        if vol_adj < 0.8:
            reasoning.append(f"Reduced leverage due to {market_conditions.volatility_regime} volatility")
        elif vol_adj > 1.1:
            reasoning.append("Increased leverage due to low volatility environment")
        
        if liq_adj < 0.8:
            reasoning.append(f"Reduced leverage due to limited liquidity (score: {market_conditions.liquidity_score:.2f})")
        
        if corr_adj < 0.9:
            reasoning.append("Slight reduction due to correlation risk with other assets")
        
        if regime_adj < 0.8:
            reasoning.append(f"Significant reduction due to adverse market conditions (risk score: {market_conditions.overall_risk_score:.2f})")
        
        if not reasoning:
            reasoning.append("Normal market conditions allow standard leverage limits")
        
        return reasoning
    
    def _calculate_confidence(self, market_conditions: MarketConditions, vol_adj: float, liq_adj: float) -> float:
        """Calculate confidence in leverage recommendation."""
        # Higher confidence when market conditions are stable and clear
        volatility_confidence = 1.0 - min(market_conditions.overall_risk_score, 1.0)
        adjustment_confidence = 1.0 - abs(vol_adj - 1.0) - abs(liq_adj - 1.0)
        
        return max(0.1, min(1.0, (volatility_confidence + adjustment_confidence) / 2))
    
    def _generate_portfolio_recommendations(
        self, 
        violations: List[Dict], 
        warnings: List[Dict], 
        portfolio_leverage: float,
        market_regime: MarketRegime
    ) -> List[str]:
        """Generate portfolio-level recommendations."""
        recommendations = []
        
        if violations:
            recommendations.append(f"URGENT: {len(violations)} positions exceed maximum leverage limits")
            for violation in violations[:3]:  # Top 3
                recommendations.append(f"Reduce {violation['symbol']} leverage from {violation['current_leverage']:.1f}x to {violation['max_allowed']:.1f}x")
        
        if warnings:
            recommendations.append(f"WARNING: {len(warnings)} positions above recommended leverage")
        
        if portfolio_leverage > 8.0:
            recommendations.append("Portfolio leverage is high - consider reducing overall exposure")
        
        if market_regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
            recommendations.append(f"Market in {market_regime.value} - consider defensive positioning")
        
        return recommendations
    
    def _get_regime_description(self, market_regime: MarketRegime) -> str:
        """Get description for market regime."""
        descriptions = {
            MarketRegime.BULL_MARKET: "Strong uptrend with manageable volatility - favorable for higher leverage",
            MarketRegime.BEAR_MARKET: "Downtrend with elevated risk - reduce leverage exposure",
            MarketRegime.SIDEWAYS: "Range-bound market with normal volatility - standard leverage acceptable",
            MarketRegime.HIGH_VOLATILITY: "High volatility environment - significantly reduce leverage",
            MarketRegime.CRISIS: "Crisis conditions with extreme volatility - minimize leverage exposure",
            MarketRegime.RECOVERY: "Post-crisis recovery phase - gradually increase leverage as conditions stabilize"
        }
        return descriptions.get(market_regime, "Unknown market regime")
    
    # Simplified market calculation methods (in practice would use real data)
    
    async def _calculate_market_volatility(self, symbol: str) -> float:
        """Calculate market volatility."""
        # Simplified volatility calculation
        if "BTC" in symbol:
            return 0.04  # 4% daily volatility
        elif "ETH" in symbol:
            return 0.05  # 5% daily volatility
        else:
            return 0.08  # 8% daily volatility for alt coins
    
    async def _calculate_liquidity_score(self, symbol: str) -> float:
        """Calculate liquidity score."""
        # Simplified liquidity scoring
        if "BTC" in symbol:
            return 0.9  # High liquidity
        elif "ETH" in symbol:
            return 0.8  # Good liquidity
        else:
            return 0.6  # Moderate liquidity
    
    async def _calculate_trend_strength(self, symbol: str) -> float:
        """Calculate trend strength."""
        # Simplified trend calculation
        # Would use price data and technical indicators
        return 0.1  # Slight uptrend
    
    async def _calculate_market_correlation_risk(self) -> float:
        """Calculate overall market correlation risk."""
        # High correlation during market stress
        return 0.6  # 60% correlation risk
    
    async def _calculate_funding_stress(self, symbol: str) -> float:
        """Calculate funding rate stress."""
        # Simplified funding stress
        return 0.3  # 30% funding stress
    
    async def _calculate_market_cap_stability(self) -> float:
        """Calculate market cap stability."""
        # Simplified market cap stability
        return 0.7  # 70% stability
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        fallback_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2800.0,
            "BTC-PERP": 45050.0,
            "ETH-PERP": 2805.0,
        }
        return fallback_prices.get(symbol, 100.0) 