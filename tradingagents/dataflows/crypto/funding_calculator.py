"""
Funding PnL Calculator for Perpetual Futures.

Provides detailed funding calculations, historical tracking, and prediction capabilities:
- Real-time funding PnL calculation
- Historical funding rate analysis
- Funding cost optimization
- Cross-exchange funding rate arbitrage
- Predictive funding models
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math

from ..base_interfaces import Position

logger = logging.getLogger(__name__)


class FundingPeriod(str, Enum):
    """Funding period intervals."""
    EIGHT_HOURS = "8h"      # Standard 8-hour funding
    FOUR_HOURS = "4h"       # Some exchanges use 4-hour
    ONE_HOUR = "1h"         # High-frequency funding
    CONTINUOUS = "continuous" # Continuous funding


@dataclass
class FundingEvent:
    """Individual funding payment event."""
    timestamp: datetime
    symbol: str
    funding_rate: float
    position_size: float
    mark_price: float
    funding_payment: float
    exchange: str = "unknown"
    
    @property
    def funding_cost_bps(self) -> float:
        """Funding cost in basis points."""
        if self.mark_price == 0:
            return 0
        return (self.funding_payment / (abs(self.position_size) * self.mark_price)) * 10000


@dataclass
class FundingStats:
    """Statistical analysis of funding rates."""
    symbol: str
    period_days: int
    
    # Rate statistics
    avg_funding_rate: float
    median_funding_rate: float
    std_funding_rate: float
    min_funding_rate: float
    max_funding_rate: float
    
    # Direction bias
    positive_rate_percentage: float
    negative_rate_percentage: float
    
    # Cost analysis
    avg_daily_cost_long: float  # Average daily cost for long positions
    avg_daily_cost_short: float # Average daily cost for short positions
    max_daily_cost: float
    
    # Trends
    funding_trend: str  # "increasing", "decreasing", "stable"
    volatility_score: float
    
    @property
    def preferred_direction(self) -> str:
        """Preferred position direction based on funding."""
        if self.avg_daily_cost_long < self.avg_daily_cost_short:
            return "long"
        else:
            return "short"


@dataclass
class FundingForecast:
    """Funding rate forecast."""
    symbol: str
    forecast_horizon_hours: int
    
    predicted_rates: List[Tuple[datetime, float]]
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper) bounds
    
    expected_funding_cost: float
    worst_case_cost: float
    best_case_cost: float
    
    model_accuracy: float
    last_updated: datetime


class FundingCalculator:
    """
    Advanced funding PnL calculator for perpetual futures.
    
    Features:
    - Real-time funding calculations
    - Historical analysis and trends
    - Multi-exchange rate comparison
    - Predictive funding models
    - Optimization recommendations
    """
    
    def __init__(
        self,
        default_funding_period: FundingPeriod = FundingPeriod.EIGHT_HOURS,
        max_history_days: int = 90,
        enable_predictions: bool = True
    ):
        """
        Initialize funding calculator.
        
        Args:
            default_funding_period: Default funding interval
            max_history_days: Maximum days of history to retain
            enable_predictions: Enable funding rate predictions
        """
        self.default_funding_period = default_funding_period
        self.max_history_days = max_history_days
        self.enable_predictions = enable_predictions
        
        # Historical data storage
        self._funding_history: Dict[str, List[FundingEvent]] = {}
        self._rate_cache: Dict[str, Tuple[datetime, float]] = {}
        
        # Prediction models (simplified)
        self._prediction_cache: Dict[str, FundingForecast] = {}
        
        # Exchange-specific settings
        self._exchange_settings = {
            'binance': {'period': FundingPeriod.EIGHT_HOURS, 'fee_rate': 0.0001},
            'okx': {'period': FundingPeriod.EIGHT_HOURS, 'fee_rate': 0.0001},
            'bybit': {'period': FundingPeriod.EIGHT_HOURS, 'fee_rate': 0.0001},
            'hyperliquid': {'period': FundingPeriod.EIGHT_HOURS, 'fee_rate': 0.0001},
            'dydx': {'period': FundingPeriod.ONE_HOUR, 'fee_rate': 0.0001},
        }
    
    async def calculate_funding_pnl(
        self, 
        position: Position,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive funding PnL for a position.
        
        Args:
            position: Trading position
            start_time: Start time for calculation (default: position start)
            end_time: End time for calculation (default: now)
        """
        try:
            if not self._is_perpetual(position.symbol):
                return {'error': 'Position is not a perpetual future'}
            
            end_time = end_time or datetime.now(timezone.utc)
            start_time = start_time or (end_time - timedelta(days=1))  # Default to 24h
            
            # Get funding events for the period
            funding_events = await self._get_funding_events(
                position.symbol, start_time, end_time
            )
            
            # Calculate PnL for each event
            total_funding_paid = 0
            funding_details = []
            
            for event in funding_events:
                # Funding payment = position_size * mark_price * funding_rate
                funding_payment = position.quantity * event.mark_price * event.funding_rate
                
                # Long positions pay positive funding, short positions receive it
                if position.quantity > 0:  # Long position
                    actual_payment = -abs(funding_payment) if event.funding_rate > 0 else abs(funding_payment)
                else:  # Short position
                    actual_payment = abs(funding_payment) if event.funding_rate > 0 else -abs(funding_payment)
                
                total_funding_paid += actual_payment
                
                funding_details.append({
                    'timestamp': event.timestamp,
                    'funding_rate': event.funding_rate,
                    'mark_price': event.mark_price,
                    'funding_payment': actual_payment,
                    'cumulative_funding': total_funding_paid
                })
            
            # Calculate metrics
            position_value = abs(position.quantity * position.average_price)
            funding_cost_percentage = (abs(total_funding_paid) / position_value * 100) if position_value > 0 else 0
            
            # Get next funding time
            next_funding = await self._get_next_funding_time(position.symbol)
            
            return {
                'symbol': position.symbol,
                'position_size': position.quantity,
                'period_start': start_time,
                'period_end': end_time,
                'total_funding_paid': total_funding_paid,
                'funding_cost_percentage': funding_cost_percentage,
                'number_of_payments': len(funding_events),
                'average_funding_rate': statistics.mean([e.funding_rate for e in funding_events]) if funding_events else 0,
                'next_funding_time': next_funding,
                'funding_details': funding_details,
                'is_profitable': total_funding_paid > 0,  # Receiving funding
                'recommendation': self._generate_funding_recommendation(position, total_funding_paid, funding_events)
            }
            
        except Exception as e:
            logger.error(f"Error calculating funding PnL: {e}")
            return {'error': str(e)}
    
    async def get_funding_stats(self, symbol: str, days: int = 30) -> Optional[FundingStats]:
        """Get statistical analysis of funding rates."""
        try:
            if not self._is_perpetual(symbol):
                return None
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            
            # Get historical funding events
            events = await self._get_funding_events(symbol, start_time, end_time)
            
            if not events:
                return None
            
            rates = [e.funding_rate for e in events]
            
            # Calculate statistics
            avg_rate = statistics.mean(rates)
            median_rate = statistics.median(rates)
            std_rate = statistics.stdev(rates) if len(rates) > 1 else 0
            min_rate = min(rates)
            max_rate = max(rates)
            
            # Direction bias
            positive_rates = sum(1 for r in rates if r > 0)
            positive_percentage = (positive_rates / len(rates)) * 100
            negative_percentage = 100 - positive_percentage
            
            # Cost analysis (assuming 3 funding periods per day)
            daily_periods = 3
            avg_daily_cost_long = abs(avg_rate) * daily_periods * 100  # As percentage
            avg_daily_cost_short = -avg_rate * daily_periods * 100     # Opposite for shorts
            max_daily_cost = max(abs(r) for r in rates) * daily_periods * 100
            
            # Trend analysis
            recent_rates = rates[-10:] if len(rates) >= 10 else rates
            older_rates = rates[:-10] if len(rates) >= 10 else []
            
            if older_rates:
                recent_avg = statistics.mean(recent_rates)
                older_avg = statistics.mean(older_rates)
                trend_change = (recent_avg - older_avg) / abs(older_avg) if older_avg != 0 else 0
                
                if trend_change > 0.1:
                    trend = "increasing"
                elif trend_change < -0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Volatility score (coefficient of variation)
            volatility_score = (std_rate / abs(avg_rate)) if avg_rate != 0 else 0
            
            return FundingStats(
                symbol=symbol,
                period_days=days,
                avg_funding_rate=avg_rate,
                median_funding_rate=median_rate,
                std_funding_rate=std_rate,
                min_funding_rate=min_rate,
                max_funding_rate=max_rate,
                positive_rate_percentage=positive_percentage,
                negative_rate_percentage=negative_percentage,
                avg_daily_cost_long=avg_daily_cost_long,
                avg_daily_cost_short=avg_daily_cost_short,
                max_daily_cost=max_daily_cost,
                funding_trend=trend,
                volatility_score=volatility_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating funding stats for {symbol}: {e}")
            return None
    
    async def predict_funding_rates(self, symbol: str, hours_ahead: int = 24) -> Optional[FundingForecast]:
        """Predict future funding rates."""
        if not self.enable_predictions:
            return None
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{hours_ahead}h"
            if cache_key in self._prediction_cache:
                forecast = self._prediction_cache[cache_key]
                if (datetime.now(timezone.utc) - forecast.last_updated).total_seconds() < 3600:  # 1 hour cache
                    return forecast
            
            # Get historical data for prediction
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30)  # Use 30 days for prediction
            
            events = await self._get_funding_events(symbol, start_time, end_time)
            
            if len(events) < 10:
                return None
            
            # Simple moving average prediction (in practice, would use more sophisticated models)
            rates = [e.funding_rate for e in events[-20:]]  # Last 20 periods
            avg_rate = statistics.mean(rates)
            std_rate = statistics.stdev(rates) if len(rates) > 1 else 0
            
            # Generate predictions
            predictions = []
            current_time = end_time
            
            for i in range(hours_ahead):
                # Simple model: trend + noise
                prediction_time = current_time + timedelta(hours=i)
                
                # Trend component (slightly mean-reverting)
                trend_factor = 0.95 ** i  # Gradual mean reversion
                predicted_rate = avg_rate * trend_factor
                
                predictions.append((prediction_time, predicted_rate))
            
            # Confidence intervals (simplified)
            confidence_intervals = []
            for _, pred_rate in predictions:
                lower = pred_rate - 1.96 * std_rate  # 95% confidence
                upper = pred_rate + 1.96 * std_rate
                confidence_intervals.append((lower, upper))
            
            # Cost projections
            avg_predicted_rate = statistics.mean([p[1] for p in predictions])
            periods_per_day = 24 / 8  # Assuming 8-hour funding
            expected_daily_cost = abs(avg_predicted_rate) * periods_per_day * 100
            
            worst_case_rate = max(abs(pred[1]) for pred in predictions)
            worst_case_cost = worst_case_rate * periods_per_day * 100
            
            best_case_rate = min(abs(pred[1]) for pred in predictions)
            best_case_cost = best_case_rate * periods_per_day * 100
            
            # Model accuracy (simplified - would use backtesting in practice)
            model_accuracy = 0.7  # 70% accuracy assumption
            
            forecast = FundingForecast(
                symbol=symbol,
                forecast_horizon_hours=hours_ahead,
                predicted_rates=predictions,
                confidence_intervals=confidence_intervals,
                expected_funding_cost=expected_daily_cost,
                worst_case_cost=worst_case_cost,
                best_case_cost=best_case_cost,
                model_accuracy=model_accuracy,
                last_updated=datetime.now(timezone.utc)
            )
            
            # Cache forecast
            self._prediction_cache[cache_key] = forecast
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error predicting funding rates for {symbol}: {e}")
            return None
    
    async def compare_cross_exchange_rates(self, symbol: str) -> Dict[str, Any]:
        """Compare funding rates across multiple exchanges."""
        try:
            exchanges = ['binance', 'okx', 'bybit', 'hyperliquid']
            current_rates = {}
            
            for exchange in exchanges:
                rate = await self._get_current_funding_rate(symbol, exchange)
                if rate is not None:
                    current_rates[exchange] = rate
            
            if not current_rates:
                return {'error': 'No funding rate data available'}
            
            # Find best rates
            best_long_exchange = min(current_rates.items(), key=lambda x: x[1])  # Lowest rate for longs
            best_short_exchange = max(current_rates.items(), key=lambda x: x[1]) # Highest rate for shorts
            
            # Calculate arbitrage opportunity
            rate_spread = best_short_exchange[1] - best_long_exchange[1]
            
            return {
                'symbol': symbol,
                'rates_by_exchange': current_rates,
                'best_for_long': {
                    'exchange': best_long_exchange[0],
                    'rate': best_long_exchange[1]
                },
                'best_for_short': {
                    'exchange': best_short_exchange[0],
                    'rate': best_short_exchange[1]
                },
                'arbitrage_spread': rate_spread,
                'arbitrage_opportunity': rate_spread > 0.0001,  # >1 basis point
                'recommendation': self._generate_arbitrage_recommendation(
                    symbol, best_long_exchange, best_short_exchange, rate_spread
                )
            }
            
        except Exception as e:
            logger.error(f"Error comparing cross-exchange rates for {symbol}: {e}")
            return {'error': str(e)}
    
    async def optimize_funding_costs(self, positions: List[Position]) -> Dict[str, Any]:
        """Optimize funding costs across all perpetual positions."""
        try:
            perp_positions = [p for p in positions if self._is_perpetual(p.symbol)]
            
            if not perp_positions:
                return {'message': 'No perpetual positions to optimize'}
            
            optimizations = []
            total_current_cost = 0
            total_optimized_cost = 0
            
            for position in perp_positions:
                # Get current funding cost
                current_funding = await self.calculate_funding_pnl(position)
                current_daily_cost = abs(current_funding.get('total_funding_paid', 0))
                total_current_cost += current_daily_cost
                
                # Get cross-exchange comparison
                exchange_rates = await self.compare_cross_exchange_rates(position.symbol)
                
                # Calculate potential savings
                best_rate = None
                current_rate = await self._get_current_funding_rate(position.symbol)
                
                if position.quantity > 0:  # Long position
                    best_exchange_data = exchange_rates.get('best_for_long', {})
                    best_rate = best_exchange_data.get('rate')
                else:  # Short position  
                    best_exchange_data = exchange_rates.get('best_for_short', {})
                    best_rate = best_exchange_data.get('rate')
                
                if best_rate is not None and current_rate is not None:
                    potential_savings = abs(current_rate - best_rate) * abs(position.market_value)
                    total_optimized_cost += abs(best_rate) * abs(position.market_value)
                    
                    optimizations.append({
                        'symbol': position.symbol,
                        'position_size': position.quantity,
                        'current_exchange': 'current',
                        'current_rate': current_rate,
                        'best_exchange': best_exchange_data.get('exchange'),
                        'best_rate': best_rate,
                        'potential_daily_savings': potential_savings * 3,  # 3 funding periods per day
                        'should_switch': potential_savings > 0.0001  # >$0.01 daily savings threshold
                    })
            
            total_potential_savings = total_current_cost - total_optimized_cost
            
            return {
                'total_positions': len(perp_positions),
                'current_daily_cost': total_current_cost,
                'optimized_daily_cost': total_optimized_cost,
                'potential_daily_savings': total_potential_savings,
                'potential_monthly_savings': total_potential_savings * 30,
                'optimizations': optimizations,
                'recommendations': self._generate_optimization_recommendations(optimizations)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing funding costs: {e}")
            return {'error': str(e)}
    
    # ==== Internal Methods ====
    
    def _is_perpetual(self, symbol: str) -> bool:
        """Check if symbol is a perpetual futures contract."""
        return symbol.endswith("-PERP") or symbol.endswith("PERP") or "PERP" in symbol.upper()
    
    async def _get_funding_events(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[FundingEvent]:
        """Get historical funding events for a symbol."""
        # In practice, this would fetch from exchange APIs or database
        # For now, simulate realistic funding events
        
        events = []
        current_time = start_time
        
        # Simulate funding events every 8 hours
        while current_time <= end_time:
            # Generate realistic funding rate
            base_rate = 0.0001  # 0.01% base rate
            
            if "BTC" in symbol:
                # BTC funding is typically more stable
                funding_rate = base_rate + (hash(str(current_time)) % 100 - 50) / 1000000
            else:
                # Alt coins have more volatile funding
                funding_rate = base_rate + (hash(str(current_time)) % 200 - 100) / 500000
            
            # Simulate mark price
            if "BTC" in symbol:
                mark_price = 45000 + (hash(str(current_time)) % 2000 - 1000)
            elif "ETH" in symbol:
                mark_price = 2800 + (hash(str(current_time)) % 200 - 100)
            else:
                mark_price = 100 + (hash(str(current_time)) % 20 - 10)
            
            events.append(FundingEvent(
                timestamp=current_time,
                symbol=symbol,
                funding_rate=funding_rate,
                position_size=1.0,  # Will be multiplied by actual position size
                mark_price=mark_price,
                funding_payment=funding_rate * mark_price,
                exchange="simulated"
            ))
            
            current_time += timedelta(hours=8)
        
        return events
    
    async def _get_next_funding_time(self, symbol: str) -> datetime:
        """Get next funding time for symbol."""
        now = datetime.now(timezone.utc)
        
        # Funding typically occurs at 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        
        for hour in funding_hours:
            next_funding = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_funding > now:
                return next_funding
        
        # If past all today's funding times, return tomorrow's first funding
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    
    async def _get_current_funding_rate(self, symbol: str, exchange: str = "default") -> Optional[float]:
        """Get current funding rate for symbol."""
        # Check cache first
        cache_key = f"{symbol}_{exchange}"
        if cache_key in self._rate_cache:
            timestamp, rate = self._rate_cache[cache_key]
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < 300:  # 5 min cache
                return rate
        
        # Simulate realistic funding rates
        if "BTC" in symbol:
            base_rate = 0.0001
        elif "ETH" in symbol:
            base_rate = 0.00015
        else:
            base_rate = 0.0002
        
        # Add some exchange-specific variation
        exchange_multipliers = {
            'binance': 1.0,
            'okx': 0.95,
            'bybit': 1.05,
            'hyperliquid': 0.9,
            'default': 1.0
        }
        
        rate = base_rate * exchange_multipliers.get(exchange, 1.0)
        
        # Cache the rate
        self._rate_cache[cache_key] = (datetime.now(timezone.utc), rate)
        
        return rate
    
    def _generate_funding_recommendation(
        self, 
        position: Position, 
        total_funding_paid: float, 
        events: List[FundingEvent]
    ) -> str:
        """Generate funding optimization recommendation."""
        if not events:
            return "Insufficient data for recommendation"
        
        avg_rate = statistics.mean([e.funding_rate for e in events])
        position_value = abs(position.quantity * position.average_price)
        cost_percentage = (abs(total_funding_paid) / position_value * 100) if position_value > 0 else 0
        
        if position.quantity > 0:  # Long position
            if avg_rate > 0.0002:  # High positive funding
                return f"Consider closing long position - high funding cost ({cost_percentage:.2f}% of position value)"
            elif avg_rate < -0.0001:  # Negative funding (receiving)
                return f"Favorable funding for long position - receiving {cost_percentage:.2f}% of position value"
            else:
                return f"Neutral funding environment - cost {cost_percentage:.2f}% of position value"
        else:  # Short position
            if avg_rate < -0.0002:  # High negative funding (paying)
                return f"Consider closing short position - high funding cost ({cost_percentage:.2f}% of position value)"
            elif avg_rate > 0.0001:  # Positive funding (receiving)
                return f"Favorable funding for short position - receiving {cost_percentage:.2f}% of position value"
            else:
                return f"Neutral funding environment - cost {cost_percentage:.2f}% of position value"
    
    def _generate_arbitrage_recommendation(
        self, 
        symbol: str, 
        best_long: Tuple[str, float], 
        best_short: Tuple[str, float], 
        spread: float
    ) -> str:
        """Generate arbitrage recommendation."""
        if spread > 0.0005:  # >5 basis points
            return f"Strong arbitrage opportunity: Long on {best_long[0]} ({best_long[1]:.4f}), Short on {best_short[0]} ({best_short[1]:.4f})"
        elif spread > 0.0001:  # >1 basis point
            return f"Moderate arbitrage opportunity: {spread:.4f} spread between exchanges"
        else:
            return "No significant arbitrage opportunity - rates are aligned across exchanges"
    
    def _generate_optimization_recommendations(self, optimizations: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        switchable_positions = [opt for opt in optimizations if opt['should_switch']]
        total_savings = sum(opt['potential_daily_savings'] for opt in switchable_positions)
        
        if switchable_positions:
            recommendations.append(f"Switch {len(switchable_positions)} positions to save ${total_savings:.2f} daily")
            
            for opt in switchable_positions[:3]:  # Top 3 recommendations
                recommendations.append(
                    f"Move {opt['symbol']} position to {opt['best_exchange']} "
                    f"(save ${opt['potential_daily_savings']:.2f}/day)"
                )
        else:
            recommendations.append("Current exchange selection is optimal for funding costs")
        
        return recommendations 