"""
Margin Manager for Crypto Trading Systems.

Provides comprehensive margin mode management for cryptocurrency derivatives:
- Cross margin vs isolated margin strategies
- Dynamic margin allocation
- Risk-based margin optimization
- Multi-asset margin management
- Margin efficiency optimization
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from ..base_interfaces import Position, Balance
from .crypto_risk_manager import MarginMode, RiskLevel

logger = logging.getLogger(__name__)


class MarginStrategy(str, Enum):
    """Margin management strategies."""
    CONSERVATIVE = "conservative"    # High margin buffer, isolated positions
    BALANCED = "balanced"           # Mix of cross and isolated based on risk
    AGGRESSIVE = "aggressive"       # Maximum leverage, cross margin
    RISK_PARITY = "risk_parity"    # Equal risk allocation across positions
    CORRELATION_AWARE = "correlation_aware"  # Adjust for asset correlations


@dataclass
class MarginAllocation:
    """Margin allocation for a position."""
    symbol: str
    margin_mode: MarginMode
    allocated_margin: float
    required_margin: float
    excess_margin: float
    margin_ratio: float
    leverage: float
    liquidation_price: float
    liquidation_distance_pct: float
    
    @property
    def is_healthy(self) -> bool:
        """Whether margin allocation is healthy."""
        return self.margin_ratio < 0.8 and self.liquidation_distance_pct > 20.0
    
    @property
    def risk_level(self) -> RiskLevel:
        """Risk level based on margin metrics."""
        if self.margin_ratio > 0.9 or self.liquidation_distance_pct < 10:
            return RiskLevel.CRITICAL
        elif self.margin_ratio > 0.75 or self.liquidation_distance_pct < 20:
            return RiskLevel.HIGH
        elif self.margin_ratio > 0.5 or self.liquidation_distance_pct < 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


@dataclass
class MarginPool:
    """Cross margin pool information."""
    total_margin: float
    used_margin: float
    available_margin: float
    total_unrealized_pnl: float
    maintenance_margin: float
    margin_ratio: float
    
    # Risk metrics
    portfolio_leverage: float
    liquidation_risk: RiskLevel
    margin_efficiency: float  # Used margin / Total possible margin
    
    # Position count
    total_positions: int
    profitable_positions: int
    losing_positions: int
    
    @property
    def is_healthy(self) -> bool:
        """Whether the margin pool is healthy."""
        return self.margin_ratio < 0.8 and self.available_margin > 0


@dataclass
class MarginOptimization:
    """Margin optimization recommendation."""
    position_symbol: str
    current_mode: MarginMode
    recommended_mode: MarginMode
    current_margin: float
    recommended_margin: float
    margin_savings: float
    risk_change: str  # "increase", "decrease", "neutral"
    leverage_change: float
    reason: str
    priority: int  # 1-5, 1 being highest priority


class MarginManager:
    """
    Advanced margin management system for crypto derivatives.
    
    Features:
    - Cross vs isolated margin optimization
    - Dynamic margin allocation based on risk
    - Correlation-aware margin efficiency
    - Real-time margin monitoring
    - Automatic rebalancing recommendations
    """
    
    def __init__(
        self,
        default_strategy: MarginStrategy = MarginStrategy.BALANCED,
        max_portfolio_leverage: float = 5.0,
        min_margin_buffer: float = 0.2,  # 20% buffer above maintenance margin
        correlation_threshold: float = 0.7,
        enable_auto_optimization: bool = True
    ):
        """
        Initialize margin manager.
        
        Args:
            default_strategy: Default margin management strategy
            max_portfolio_leverage: Maximum portfolio-wide leverage
            min_margin_buffer: Minimum margin buffer above maintenance
            correlation_threshold: Threshold for correlation-based decisions
            enable_auto_optimization: Enable automatic optimization
        """
        self.default_strategy = default_strategy
        self.max_portfolio_leverage = max_portfolio_leverage
        self.min_margin_buffer = min_margin_buffer
        self.correlation_threshold = correlation_threshold
        self.enable_auto_optimization = enable_auto_optimization
        
        # Strategy-specific settings
        self._strategy_settings = {
            MarginStrategy.CONSERVATIVE: {
                'max_leverage': 3.0,
                'margin_buffer': 0.3,
                'prefer_isolated': True,
                'max_correlation_exposure': 0.5
            },
            MarginStrategy.BALANCED: {
                'max_leverage': 5.0,
                'margin_buffer': 0.2,
                'prefer_isolated': False,
                'max_correlation_exposure': 0.7
            },
            MarginStrategy.AGGRESSIVE: {
                'max_leverage': 10.0,
                'margin_buffer': 0.1,
                'prefer_isolated': False,
                'max_correlation_exposure': 1.0
            },
            MarginStrategy.RISK_PARITY: {
                'max_leverage': 4.0,
                'margin_buffer': 0.25,
                'prefer_isolated': False,
                'max_correlation_exposure': 0.6
            },
            MarginStrategy.CORRELATION_AWARE: {
                'max_leverage': 6.0,
                'margin_buffer': 0.15,
                'prefer_isolated': False,
                'max_correlation_exposure': 0.5
            }
        }
        
        # State tracking
        self._margin_allocations: Dict[str, MarginAllocation] = {}
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._last_optimization = datetime.now(timezone.utc)
        
    async def analyze_margin_allocation(
        self, 
        positions: List[Position], 
        balances: List[Balance],
        strategy: Optional[MarginStrategy] = None
    ) -> Dict[str, Any]:
        """
        Analyze current margin allocation and provide optimization recommendations.
        
        Args:
            positions: Current trading positions
            balances: Account balances
            strategy: Margin strategy to analyze (default: instance strategy)
        """
        try:
            strategy = strategy or self.default_strategy
            strategy_config = self._strategy_settings[strategy]
            
            # Filter perpetual positions
            perp_positions = [p for p in positions if self._is_perpetual(p.symbol)]
            
            if not perp_positions:
                return {'message': 'No perpetual positions found'}
            
            # Calculate current margin allocations
            allocations = []
            total_margin_used = 0
            
            for position in perp_positions:
                allocation = await self._calculate_margin_allocation(position)
                allocations.append(allocation)
                total_margin_used += allocation.allocated_margin
            
            # Get account balance for margin calculations
            base_balance = self._get_base_balance(balances)
            total_equity = base_balance.total if base_balance else 0
            
            # Calculate cross margin pool
            margin_pool = await self._calculate_margin_pool(perp_positions, total_equity)
            
            # Generate optimization recommendations
            optimizations = await self._generate_margin_optimizations(
                allocations, margin_pool, strategy_config
            )
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(
                allocations, margin_pool, strategy_config
            )
            
            # Risk assessment
            risk_assessment = self._assess_margin_risks(allocations, margin_pool)
            
            return {
                'strategy': strategy,
                'margin_pool': margin_pool,
                'position_allocations': allocations,
                'optimization_recommendations': optimizations,
                'efficiency_metrics': efficiency_metrics,
                'risk_assessment': risk_assessment,
                'total_margin_used': total_margin_used,
                'margin_utilization': total_margin_used / total_equity if total_equity > 0 else 0,
                'recommended_actions': self._generate_action_recommendations(
                    allocations, margin_pool, optimizations
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing margin allocation: {e}")
            return {'error': str(e)}
    
    async def optimize_margin_mode(
        self, 
        position: Position, 
        target_leverage: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize margin mode for a specific position.
        
        Args:
            position: Position to optimize
            target_leverage: Target leverage (optional)
        """
        try:
            if not self._is_perpetual(position.symbol):
                return {'error': 'Position is not a perpetual future'}
            
            current_allocation = await self._calculate_margin_allocation(position)
            
            # Calculate optimal margin mode
            cross_margin_scenario = await self._calculate_cross_margin_scenario(position, target_leverage)
            isolated_margin_scenario = await self._calculate_isolated_margin_scenario(position, target_leverage)
            
            # Compare scenarios
            comparison = self._compare_margin_scenarios(
                current_allocation, cross_margin_scenario, isolated_margin_scenario
            )
            
            # Get correlation impact
            correlation_impact = await self._assess_correlation_impact(position)
            
            return {
                'symbol': position.symbol,
                'current_allocation': current_allocation,
                'cross_margin_scenario': cross_margin_scenario,
                'isolated_margin_scenario': isolated_margin_scenario,
                'comparison': comparison,
                'correlation_impact': correlation_impact,
                'recommendation': comparison['recommended_mode'],
                'optimization_benefit': comparison['benefit_summary']
            }
            
        except Exception as e:
            logger.error(f"Error optimizing margin mode for {position.symbol}: {e}")
            return {'error': str(e)}
    
    async def calculate_optimal_leverage(
        self, 
        symbol: str, 
        account_balance: float,
        risk_tolerance: float = 0.02,  # 2% risk per trade
        volatility_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal leverage for a position based on risk and volatility.
        
        Args:
            symbol: Trading symbol
            account_balance: Total account balance
            risk_tolerance: Risk per trade as fraction of account
            volatility_data: Historical volatility data
        """
        try:
            # Get or estimate volatility
            if not volatility_data:
                volatility_data = await self._estimate_volatility(symbol)
            
            daily_vol = volatility_data.get('daily_volatility', 0.04)  # 4% default
            
            # Kelly criterion for optimal sizing
            # Kelly = (bp - q) / b, simplified for crypto
            expected_return = 0.001  # 0.1% daily expected return
            kelly_fraction = expected_return / (daily_vol ** 2) if daily_vol > 0 else 0
            
            # Conservative Kelly (25% of full Kelly)
            conservative_kelly = kelly_fraction * 0.25
            
            # Risk-based leverage calculation
            # Leverage = Risk_Tolerance / (Price_Change_Risk * Position_Size)
            risk_based_leverage = risk_tolerance / daily_vol if daily_vol > 0 else 1.0
            
            # VAR-based leverage (99% confidence)
            var_99 = daily_vol * 2.33  # 99% VAR
            var_based_leverage = risk_tolerance / var_99 if var_99 > 0 else 1.0
            
            # Strategy-specific adjustments
            strategy_config = self._strategy_settings[self.default_strategy]
            max_allowed_leverage = strategy_config['max_leverage']
            
            # Take the minimum of all calculations
            optimal_leverage = min(
                conservative_kelly * 10,  # Scale Kelly to leverage
                risk_based_leverage,
                var_based_leverage,
                max_allowed_leverage
            )
            
            # Ensure minimum leverage of 1.0
            optimal_leverage = max(1.0, optimal_leverage)
            
            # Calculate margin requirements
            position_size_usd = account_balance * risk_tolerance / daily_vol
            required_margin = position_size_usd / optimal_leverage
            
            return {
                'symbol': symbol,
                'optimal_leverage': optimal_leverage,
                'max_allowed_leverage': max_allowed_leverage,
                'kelly_leverage': conservative_kelly * 10,
                'risk_based_leverage': risk_based_leverage,
                'var_based_leverage': var_based_leverage,
                'recommended_position_size_usd': position_size_usd,
                'required_margin': required_margin,
                'daily_volatility': daily_vol,
                'risk_tolerance': risk_tolerance,
                'margin_efficiency': optimal_leverage / max_allowed_leverage
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal leverage for {symbol}: {e}")
            return {'error': str(e)}
    
    async def rebalance_cross_margin(
        self, 
        positions: List[Position], 
        target_leverage: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rebalance cross margin positions for optimal efficiency.
        
        Args:
            positions: Current positions in cross margin
            target_leverage: Target portfolio leverage
        """
        try:
            cross_positions = [p for p in positions if self._is_perpetual(p.symbol)]
            
            if not cross_positions:
                return {'message': 'No cross margin positions to rebalance'}
            
            # Calculate current portfolio metrics
            total_notional = sum(abs(p.market_value) for p in cross_positions)
            current_leverage = await self._calculate_portfolio_leverage(cross_positions)
            
            target_leverage = target_leverage or self.max_portfolio_leverage
            
            # Calculate optimal allocation
            correlations = await self._get_correlation_matrix([p.symbol for p in cross_positions])
            optimal_weights = await self._calculate_optimal_weights(cross_positions, correlations)
            
            # Generate rebalancing recommendations
            rebalancing_actions = []
            
            for position in cross_positions:
                current_weight = abs(position.market_value) / total_notional if total_notional > 0 else 0
                optimal_weight = optimal_weights.get(position.symbol, current_weight)
                
                weight_diff = optimal_weight - current_weight
                
                if abs(weight_diff) > 0.05:  # 5% threshold
                    action_type = "increase" if weight_diff > 0 else "decrease"
                    adjustment_amount = abs(weight_diff) * total_notional
                    
                    rebalancing_actions.append({
                        'symbol': position.symbol,
                        'action': action_type,
                        'current_weight': current_weight,
                        'optimal_weight': optimal_weight,
                        'adjustment_amount_usd': adjustment_amount,
                        'priority': min(5, int(abs(weight_diff) * 20))  # 1-5 priority scale
                    })
            
            # Calculate expected benefits
            expected_benefits = self._calculate_rebalancing_benefits(
                cross_positions, optimal_weights, current_leverage, target_leverage
            )
            
            return {
                'current_leverage': current_leverage,
                'target_leverage': target_leverage,
                'total_notional': total_notional,
                'rebalancing_actions': rebalancing_actions,
                'expected_benefits': expected_benefits,
                'execution_priority': sorted(rebalancing_actions, key=lambda x: x['priority'], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Error rebalancing cross margin: {e}")
            return {'error': str(e)}
    
    # ==== Internal Methods ====
    
    def _is_perpetual(self, symbol: str) -> bool:
        """Check if symbol is a perpetual futures contract."""
        return symbol.endswith("-PERP") or symbol.endswith("PERP") or "PERP" in symbol.upper()
    
    def _get_base_balance(self, balances: List[Balance]) -> Optional[Balance]:
        """Get base currency balance (USDT, USD, etc.)."""
        for balance in balances:
            if balance.currency in ["USDT", "USD", "BUSD", "USDC"]:
                return balance
        
        # Return first balance if no USD-based currency found
        return balances[0] if balances else None
    
    async def _calculate_margin_allocation(self, position: Position) -> MarginAllocation:
        """Calculate margin allocation for a position."""
        # Simplified margin calculation
        position_value = abs(position.market_value)
        
        # Assume 5x leverage and 20% maintenance margin
        leverage = 5.0
        required_margin = position_value / leverage
        maintenance_margin = position_value * 0.2
        allocated_margin = required_margin * 1.2  # 20% buffer
        
        excess_margin = allocated_margin - maintenance_margin
        margin_ratio = maintenance_margin / allocated_margin if allocated_margin > 0 else 0
        
        # Calculate liquidation price
        if position.quantity > 0:  # Long
            liquidation_price = position.average_price * (1 - 0.8)  # 80% of entry
        else:  # Short
            liquidation_price = position.average_price * (1 + 0.8)
        
        current_price = await self._get_current_price(position.symbol)
        liquidation_distance = abs(current_price - liquidation_price) / current_price * 100
        
        return MarginAllocation(
            symbol=position.symbol,
            margin_mode=MarginMode.CROSS,  # Default assumption
            allocated_margin=allocated_margin,
            required_margin=required_margin,
            excess_margin=excess_margin,
            margin_ratio=margin_ratio,
            leverage=leverage,
            liquidation_price=liquidation_price,
            liquidation_distance_pct=liquidation_distance
        )
    
    async def _calculate_margin_pool(self, positions: List[Position], total_equity: float) -> MarginPool:
        """Calculate cross margin pool metrics."""
        total_margin = sum(abs(p.market_value) / 5.0 for p in positions)  # Assume 5x leverage
        maintenance_margin = sum(abs(p.market_value) * 0.1 for p in positions)  # 10% maintenance
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        used_margin = total_margin
        available_margin = total_equity - used_margin + total_unrealized_pnl
        margin_ratio = maintenance_margin / total_equity if total_equity > 0 else 0
        
        portfolio_leverage = sum(abs(p.market_value) for p in positions) / total_equity if total_equity > 0 else 0
        
        profitable_positions = sum(1 for p in positions if p.unrealized_pnl > 0)
        losing_positions = len(positions) - profitable_positions
        
        margin_efficiency = used_margin / (total_equity * 0.8) if total_equity > 0 else 0  # 80% max utilization
        
        # Assess liquidation risk
        if margin_ratio > 0.9:
            liquidation_risk = RiskLevel.CRITICAL
        elif margin_ratio > 0.75:
            liquidation_risk = RiskLevel.HIGH
        elif margin_ratio > 0.5:
            liquidation_risk = RiskLevel.MEDIUM
        else:
            liquidation_risk = RiskLevel.LOW
        
        return MarginPool(
            total_margin=total_margin,
            used_margin=used_margin,
            available_margin=available_margin,
            total_unrealized_pnl=total_unrealized_pnl,
            maintenance_margin=maintenance_margin,
            margin_ratio=margin_ratio,
            portfolio_leverage=portfolio_leverage,
            liquidation_risk=liquidation_risk,
            margin_efficiency=margin_efficiency,
            total_positions=len(positions),
            profitable_positions=profitable_positions,
            losing_positions=losing_positions
        )
    
    async def _generate_margin_optimizations(
        self, 
        allocations: List[MarginAllocation], 
        margin_pool: MarginPool,
        strategy_config: Dict[str, Any]
    ) -> List[MarginOptimization]:
        """Generate margin optimization recommendations."""
        optimizations = []
        
        for allocation in allocations:
            # Check if position should switch margin modes
            if strategy_config.get('prefer_isolated', False):
                # Strategy prefers isolated margin
                if allocation.margin_mode == MarginMode.CROSS and allocation.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    optimizations.append(MarginOptimization(
                        position_symbol=allocation.symbol,
                        current_mode=allocation.margin_mode,
                        recommended_mode=MarginMode.ISOLATED,
                        current_margin=allocation.allocated_margin,
                        recommended_margin=allocation.required_margin * 1.3,  # Higher buffer for isolated
                        margin_savings=0,  # No savings, but risk reduction
                        risk_change="decrease",
                        leverage_change=-1.0,  # Reduce leverage slightly
                        reason="High risk position should use isolated margin for better risk control",
                        priority=4
                    ))
            else:
                # Strategy prefers cross margin efficiency
                if allocation.margin_mode == MarginMode.ISOLATED and allocation.risk_level == RiskLevel.LOW:
                    # Calculate potential margin savings from cross margin
                    cross_margin_savings = allocation.allocated_margin * 0.15  # 15% efficiency gain
                    
                    optimizations.append(MarginOptimization(
                        position_symbol=allocation.symbol,
                        current_mode=allocation.margin_mode,
                        recommended_mode=MarginMode.CROSS,
                        current_margin=allocation.allocated_margin,
                        recommended_margin=allocation.allocated_margin - cross_margin_savings,
                        margin_savings=cross_margin_savings,
                        risk_change="neutral",
                        leverage_change=0.5,  # Can increase leverage slightly
                        reason="Low risk position can benefit from cross margin efficiency",
                        priority=2
                    ))
            
            # Check for over-margined positions
            if allocation.excess_margin > allocation.required_margin * 0.5:  # >50% excess
                optimizations.append(MarginOptimization(
                    position_symbol=allocation.symbol,
                    current_mode=allocation.margin_mode,
                    recommended_mode=allocation.margin_mode,  # Keep same mode
                    current_margin=allocation.allocated_margin,
                    recommended_margin=allocation.required_margin * 1.2,  # 20% buffer
                    margin_savings=allocation.excess_margin - allocation.required_margin * 0.2,
                    risk_change="neutral",
                    leverage_change=0,
                    reason="Position is over-margined, can release excess margin for other opportunities",
                    priority=3
                ))
        
        return sorted(optimizations, key=lambda x: x.priority, reverse=True)
    
    def _calculate_efficiency_metrics(
        self, 
        allocations: List[MarginAllocation], 
        margin_pool: MarginPool,
        strategy_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate margin efficiency metrics."""
        total_allocated = sum(a.allocated_margin for a in allocations)
        total_required = sum(a.required_margin for a in allocations)
        
        # Overall efficiency
        margin_efficiency = total_required / total_allocated if total_allocated > 0 else 0
        
        # Leverage efficiency
        max_leverage = strategy_config.get('max_leverage', 5.0)
        avg_leverage = sum(a.leverage for a in allocations) / len(allocations) if allocations else 0
        leverage_efficiency = avg_leverage / max_leverage
        
        # Risk-adjusted efficiency
        avg_risk_score = sum(self._risk_level_to_score(a.risk_level) for a in allocations) / len(allocations) if allocations else 0
        risk_adjusted_efficiency = margin_efficiency * (1 - avg_risk_score / 4)  # Penalty for high risk
        
        return {
            'margin_efficiency': margin_efficiency,
            'leverage_efficiency': leverage_efficiency,
            'risk_adjusted_efficiency': risk_adjusted_efficiency,
            'portfolio_efficiency': margin_pool.margin_efficiency,
            'utilization_rate': total_allocated / (margin_pool.total_margin + 1)  # Avoid division by zero
        }
    
    def _assess_margin_risks(
        self, 
        allocations: List[MarginAllocation], 
        margin_pool: MarginPool
    ) -> Dict[str, Any]:
        """Assess margin-related risks."""
        high_risk_positions = [a for a in allocations if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        # Concentration risk
        total_margin = sum(a.allocated_margin for a in allocations)
        max_single_position = max(a.allocated_margin for a in allocations) if allocations else 0
        concentration_risk = max_single_position / total_margin if total_margin > 0 else 0
        
        # Liquidation risk
        liquidation_risk_count = len([a for a in allocations if a.liquidation_distance_pct < 20])
        
        return {
            'overall_risk_level': margin_pool.liquidation_risk,
            'high_risk_position_count': len(high_risk_positions),
            'concentration_risk': concentration_risk,
            'liquidation_risk_positions': liquidation_risk_count,
            'margin_utilization_risk': margin_pool.margin_ratio,
            'portfolio_leverage_risk': margin_pool.portfolio_leverage,
            'risk_summary': self._generate_risk_summary(allocations, margin_pool)
        }
    
    def _generate_action_recommendations(
        self, 
        allocations: List[MarginAllocation], 
        margin_pool: MarginPool, 
        optimizations: List[MarginOptimization]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # High priority optimizations
        high_priority = [opt for opt in optimizations if opt.priority >= 4]
        if high_priority:
            recommendations.append(f"Urgent: {len(high_priority)} positions need immediate margin adjustments")
        
        # Margin utilization
        if margin_pool.margin_ratio > 0.8:
            recommendations.append("Consider reducing position sizes - margin utilization is high")
        elif margin_pool.margin_ratio < 0.3:
            recommendations.append("Low margin utilization - consider increasing position sizes for better capital efficiency")
        
        # Risk management
        critical_positions = [a for a in allocations if a.risk_level == RiskLevel.CRITICAL]
        if critical_positions:
            recommendations.append(f"Critical: {len(critical_positions)} positions at high liquidation risk")
        
        # Efficiency improvements
        total_savings = sum(opt.margin_savings for opt in optimizations)
        if total_savings > 0:
            recommendations.append(f"Potential margin savings: ${total_savings:.2f} from optimization")
        
        return recommendations
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> int:
        """Convert risk level to numeric score."""
        mapping = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        return mapping.get(risk_level, 2)
    
    def _generate_risk_summary(self, allocations: List[MarginAllocation], margin_pool: MarginPool) -> str:
        """Generate risk summary text."""
        if margin_pool.liquidation_risk == RiskLevel.CRITICAL:
            return "Critical risk level - immediate action required to avoid liquidation"
        elif margin_pool.liquidation_risk == RiskLevel.HIGH:
            return "High risk level - monitor positions closely and consider reducing leverage"
        elif margin_pool.liquidation_risk == RiskLevel.MEDIUM:
            return "Moderate risk level - maintain current risk controls"
        else:
            return "Low risk level - margin allocation is healthy"
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        # Fallback prices for testing
        fallback_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2800.0,
            "BTC-PERP": 45050.0,
            "ETH-PERP": 2805.0,
        }
        return fallback_prices.get(symbol, 100.0)
    
    async def _estimate_volatility(self, symbol: str) -> Dict[str, float]:
        """Estimate volatility for symbol."""
        # Simplified volatility estimates
        base_vol = 0.04  # 4% daily volatility
        
        if "BTC" in symbol:
            daily_vol = base_vol * 1.0
        elif "ETH" in symbol:
            daily_vol = base_vol * 1.3
        else:
            daily_vol = base_vol * 2.0
        
        return {
            'daily_volatility': daily_vol,
            'weekly_volatility': daily_vol * math.sqrt(7),
            'monthly_volatility': daily_vol * math.sqrt(30)
        }
    
    async def _calculate_portfolio_leverage(self, positions: List[Position]) -> float:
        """Calculate portfolio leverage."""
        total_notional = sum(abs(p.market_value) for p in positions)
        total_margin = sum(abs(p.market_value) / 5.0 for p in positions)  # Assume 5x leverage
        
        return total_notional / total_margin if total_margin > 0 else 1.0
    
    async def _get_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for symbols."""
        # Simplified correlation matrix
        matrix = {}
        for symbol1 in symbols:
            matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation = 1.0
                elif "BTC" in symbol1 and "BTC" in symbol2:
                    correlation = 0.95
                elif "ETH" in symbol1 and "ETH" in symbol2:
                    correlation = 0.95
                elif ("BTC" in symbol1 and "ETH" in symbol2) or ("ETH" in symbol1 and "BTC" in symbol2):
                    correlation = 0.7
                else:
                    correlation = 0.5
                
                matrix[symbol1][symbol2] = correlation
        
        return matrix
    
    async def _calculate_optimal_weights(
        self, 
        positions: List[Position], 
        correlations: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate optimal portfolio weights."""
        # Simplified equal-weight with correlation adjustment
        equal_weight = 1.0 / len(positions)
        weights = {}
        
        for position in positions:
            # Adjust for volatility and correlation
            symbol = position.symbol
            vol_adjustment = 1.0  # Would use actual volatility data
            
            # Correlation adjustment (reduce weight for highly correlated assets)
            correlation_penalty = 1.0
            for other_pos in positions:
                if other_pos.symbol != symbol:
                    corr = correlations.get(symbol, {}).get(other_pos.symbol, 0.5)
                    if corr > self.correlation_threshold:
                        correlation_penalty *= 0.9  # Reduce weight for high correlation
            
            adjusted_weight = equal_weight * vol_adjustment * correlation_penalty
            weights[symbol] = adjusted_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_rebalancing_benefits(
        self, 
        positions: List[Position], 
        optimal_weights: Dict[str, float],
        current_leverage: float, 
        target_leverage: float
    ) -> Dict[str, Any]:
        """Calculate expected benefits from rebalancing."""
        # Simplified benefit calculation
        leverage_improvement = target_leverage - current_leverage
        
        # Estimate risk reduction from better diversification
        current_concentration = max(abs(p.market_value) for p in positions) / sum(abs(p.market_value) for p in positions)
        optimal_concentration = max(optimal_weights.values())
        concentration_improvement = current_concentration - optimal_concentration
        
        return {
            'leverage_improvement': leverage_improvement,
            'concentration_improvement': concentration_improvement,
            'estimated_risk_reduction': concentration_improvement * 0.1,  # 10% risk reduction per 10% concentration improvement
            'margin_efficiency_gain': leverage_improvement * 0.05  # 5% efficiency gain per unit leverage
        }
    
    async def _calculate_cross_margin_scenario(self, position: Position, target_leverage: Optional[float]) -> Dict[str, Any]:
        """Calculate cross margin scenario."""
        # Simplified calculation
        return {
            'margin_mode': MarginMode.CROSS,
            'required_margin': abs(position.market_value) / (target_leverage or 5.0),
            'estimated_fees': abs(position.market_value) * 0.0001,
            'risk_level': RiskLevel.MEDIUM,
            'benefits': ['Margin efficiency', 'Portfolio netting'],
            'risks': ['Contagion risk', 'Portfolio liquidation']
        }
    
    async def _calculate_isolated_margin_scenario(self, position: Position, target_leverage: Optional[float]) -> Dict[str, Any]:
        """Calculate isolated margin scenario."""
        # Simplified calculation
        return {
            'margin_mode': MarginMode.ISOLATED,
            'required_margin': abs(position.market_value) / (target_leverage or 3.0) * 1.2,  # Higher margin requirement
            'estimated_fees': abs(position.market_value) * 0.00015,  # Slightly higher fees
            'risk_level': RiskLevel.LOW,
            'benefits': ['Risk isolation', 'Position-specific control'],
            'risks': ['Lower capital efficiency', 'Higher margin requirements']
        }
    
    def _compare_margin_scenarios(
        self, 
        current: MarginAllocation, 
        cross_scenario: Dict[str, Any], 
        isolated_scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare margin scenarios."""
        cross_cost = cross_scenario['required_margin'] + cross_scenario['estimated_fees']
        isolated_cost = isolated_scenario['required_margin'] + isolated_scenario['estimated_fees']
        
        if cross_cost < isolated_cost:
            recommended_mode = MarginMode.CROSS
            savings = isolated_cost - cross_cost
            benefit = f"Save ${savings:.2f} in margin requirements"
        else:
            recommended_mode = MarginMode.ISOLATED
            cost_increase = cross_cost - isolated_cost
            benefit = f"Better risk isolation (${cost_increase:.2f} additional cost)"
        
        return {
            'recommended_mode': recommended_mode,
            'cross_margin_cost': cross_cost,
            'isolated_margin_cost': isolated_cost,
            'benefit_summary': benefit
        }
    
    async def _assess_correlation_impact(self, position: Position) -> Dict[str, Any]:
        """Assess correlation impact for position."""
        # Simplified correlation assessment
        return {
            'correlated_assets': ['BTC-PERP'] if 'ETH' in position.symbol else ['ETH-PERP'],
            'correlation_score': 0.7,
            'diversification_benefit': 'Medium',
            'recommendation': 'Monitor correlation exposure'
        } 