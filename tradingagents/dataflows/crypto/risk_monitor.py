"""
24/7 Risk Monitor for Crypto Trading.

Provides continuous risk monitoring and real-time alerts for cryptocurrency trading:
- 24/7 continuous risk assessment
- Real-time threshold breach alerts
- Portfolio health monitoring
- Liquidation risk tracking
- Automated risk response recommendations
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..base_interfaces import Position, Balance
from .crypto_risk_manager import CryptoRiskManager, RiskLevel, PortfolioRisk
from .leverage_controller import DynamicLeverageController, MarketRegime
from .funding_calculator import FundingCalculator

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"           # Informational alerts
    WARNING = "warning"     # Warning - attention needed
    CRITICAL = "critical"   # Critical - immediate action required
    EMERGENCY = "emergency" # Emergency - potential account loss


class AlertType(str, Enum):
    """Types of risk alerts."""
    MARGIN_WARNING = "margin_warning"
    LIQUIDATION_RISK = "liquidation_risk"
    FUNDING_COST = "funding_cost"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_RISK = "correlation_risk"
    LEVERAGE_BREACH = "leverage_breach"
    PORTFOLIO_HEALTH = "portfolio_health"
    MARKET_REGIME_CHANGE = "market_regime_change"
    SYSTEM_ERROR = "system_error"


@dataclass
class RiskAlert:
    """Risk alert with detailed information."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: datetime
    symbol: Optional[str]
    
    # Alert content
    title: str
    message: str
    current_value: Optional[float]
    threshold_value: Optional[float]
    
    # Risk metrics
    risk_level: RiskLevel
    confidence: float
    
    # Action recommendations
    recommended_actions: List[str]
    urgency_minutes: Optional[int]  # How urgent the action is
    
    # Context
    market_context: Dict[str, Any] = field(default_factory=dict)
    position_context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Whether this alert requires immediate action."""
        return self.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
    
    @property
    def age_minutes(self) -> float:
        """Age of alert in minutes."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 60


@dataclass
class MonitoringConfig:
    """Configuration for risk monitoring."""
    # Monitoring intervals
    risk_check_interval_seconds: int = 60        # 1 minute
    market_update_interval_seconds: int = 300    # 5 minutes
    alert_cleanup_interval_hours: int = 24       # 24 hours
    
    # Alert thresholds
    margin_warning_threshold: float = 0.75       # 75% margin utilization
    margin_critical_threshold: float = 0.85      # 85% margin utilization
    liquidation_warning_distance: float = 0.30   # 30% from liquidation
    liquidation_critical_distance: float = 0.15  # 15% from liquidation
    
    # Portfolio thresholds
    max_portfolio_leverage: float = 8.0
    max_single_position_percentage: float = 0.30  # 30% of portfolio
    max_correlation_exposure: float = 0.60        # 60% in correlated assets
    
    # Funding thresholds
    daily_funding_warning_percentage: float = 0.02  # 2% of position value
    daily_funding_critical_percentage: float = 0.05 # 5% of position value
    
    # Volatility thresholds
    volatility_spike_threshold: float = 2.0      # 2x normal volatility
    volatility_extreme_threshold: float = 3.0    # 3x normal volatility


class RiskMonitor:
    """
    Comprehensive 24/7 risk monitoring system for crypto trading.
    
    Features:
    - Continuous risk assessment
    - Real-time alert generation
    - Automated response recommendations
    - Portfolio health tracking
    - Market regime monitoring
    """
    
    def __init__(
        self,
        risk_manager: CryptoRiskManager,
        leverage_controller: DynamicLeverageController,
        funding_calculator: FundingCalculator,
        config: Optional[MonitoringConfig] = None,
        alert_handlers: Optional[List[Callable]] = None,
        enable_auto_actions: bool = False
    ):
        """
        Initialize 24/7 risk monitor.
        
        Args:
            risk_manager: Risk management system
            leverage_controller: Leverage control system
            funding_calculator: Funding calculation system
            config: Monitoring configuration
            alert_handlers: Custom alert handlers
            enable_auto_actions: Enable automated risk responses
        """
        self.risk_manager = risk_manager
        self.leverage_controller = leverage_controller
        self.funding_calculator = funding_calculator
        self.config = config or MonitoringConfig()
        self.alert_handlers = alert_handlers or []
        self.enable_auto_actions = enable_auto_actions
        
        # State tracking
        self._active_alerts: Dict[str, RiskAlert] = {}
        self._alert_history: List[RiskAlert] = []
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._last_risk_update = datetime.now(timezone.utc)
        self._risk_check_count = 0
        self._alert_count = 0
        
        # Current state cache
        self._current_positions: List[Position] = []
        self._current_balances: List[Balance] = []
        self._last_portfolio_risk: Optional[PortfolioRisk] = None
        self._current_market_regime: Optional[MarketRegime] = None
    
    async def start_monitoring(self, positions: List[Position], balances: List[Balance]) -> None:
        """Start 24/7 risk monitoring."""
        if self._monitoring_active:
            logger.warning("Risk monitoring already active")
            return
        
        logger.info("Starting 24/7 crypto risk monitoring")
        
        # Initialize state
        self._current_positions = positions
        self._current_balances = balances
        self._monitoring_active = True
        
        # Start monitoring loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Generate initial risk assessment
        await self._perform_risk_check()
        
        logger.info("24/7 risk monitoring started successfully")
    
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        if not self._monitoring_active:
            return
        
        logger.info("Stopping 24/7 risk monitoring")
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk monitoring stopped")
    
    async def update_positions(self, positions: List[Position]) -> None:
        """Update current positions for monitoring."""
        self._current_positions = positions
        
        # Trigger immediate risk check on position changes
        if self._monitoring_active:
            await self._perform_risk_check()
    
    async def update_balances(self, balances: List[Balance]) -> None:
        """Update current balances for monitoring."""
        self._current_balances = balances
    
    async def get_current_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[RiskAlert]:
        """Get current active alerts."""
        alerts = list(self._active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.EMERGENCY: 4,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 1
        }
        
        alerts.sort(key=lambda x: (severity_order.get(x.severity, 0), x.timestamp), reverse=True)
        return alerts
    
    async def get_portfolio_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio health summary."""
        if not self._current_positions:
            return {'status': 'no_positions', 'health_score': 1.0}
        
        # Get latest portfolio risk
        portfolio_risk = await self.risk_manager.get_portfolio_risk(self._current_positions)
        
        if 'error' in portfolio_risk:
            return {'status': 'error', 'error': portfolio_risk['error']}
        
        portfolio_risk_obj = portfolio_risk.get('portfolio_risk')
        if not portfolio_risk_obj:
            return {'status': 'error', 'error': 'No portfolio risk data'}
        
        # Calculate health score (0-1, 1 being healthiest)
        health_factors = {
            'margin_health': 1 - portfolio_risk_obj.margin_ratio,
            'leverage_health': max(0, 1 - (portfolio_risk_obj.leverage_ratio / 10)),  # Assuming 10x is max healthy
            'liquidation_health': 1 - (portfolio_risk_obj.liquidation_risk_count / len(self._current_positions)),
            'funding_health': max(0, 1 - abs(portfolio_risk_obj.funding_pnl_24h / portfolio_risk_obj.total_account_value))
        }
        
        overall_health = sum(health_factors.values()) / len(health_factors)
        
        # Determine health status
        if overall_health >= 0.8:
            health_status = "excellent"
        elif overall_health >= 0.6:
            health_status = "good"
        elif overall_health >= 0.4:
            health_status = "fair"
        elif overall_health >= 0.2:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            'status': health_status,
            'health_score': overall_health,
            'health_factors': health_factors,
            'portfolio_risk': portfolio_risk_obj,
            'active_alerts': len(self._active_alerts),
            'critical_alerts': len([a for a in self._active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            'last_update': self._last_risk_update,
            'monitoring_stats': {
                'checks_performed': self._risk_check_count,
                'alerts_generated': self._alert_count,
                'uptime_hours': (datetime.now(timezone.utc) - self._last_risk_update).total_seconds() / 3600
            }
        }
    
    async def force_risk_check(self) -> Dict[str, Any]:
        """Force an immediate comprehensive risk check."""
        logger.info("Performing forced risk check")
        
        result = await self._perform_risk_check()
        
        return {
            'timestamp': datetime.now(timezone.utc),
            'new_alerts': result.get('new_alerts', 0),
            'resolved_alerts': result.get('resolved_alerts', 0),
            'total_active_alerts': len(self._active_alerts),
            'portfolio_health': await self.get_portfolio_health_summary()
        }
    
    # ==== Internal Methods ====
    
    async def _monitoring_loop(self) -> None:
        """Main 24/7 monitoring loop."""
        logger.info("Starting 24/7 monitoring loop")
        
        try:
            while self._monitoring_active:
                try:
                    # Perform risk check
                    await self._perform_risk_check()
                    
                    # Clean up old alerts
                    await self._cleanup_old_alerts()
                    
                    # Sleep until next check
                    await asyncio.sleep(self.config.risk_check_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    
                    # Generate system error alert
                    await self._generate_alert(
                        AlertType.SYSTEM_ERROR,
                        AlertSeverity.WARNING,
                        "Risk Monitor Error",
                        f"Error in monitoring loop: {str(e)}",
                        None,
                        {"error": str(e)},
                        {"action": "Check system logs"},
                        []
                    )
                    
                    # Sleep before retrying
                    await asyncio.sleep(min(self.config.risk_check_interval_seconds, 30))
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
            self._monitoring_active = False
    
    async def _perform_risk_check(self) -> Dict[str, Any]:
        """Perform comprehensive risk check."""
        self._risk_check_count += 1
        self._last_risk_update = datetime.now(timezone.utc)
        
        new_alerts = 0
        resolved_alerts = 0
        
        try:
            # Check portfolio-level risks
            if self._current_positions:
                portfolio_risk = await self.risk_manager.get_portfolio_risk(self._current_positions)
                
                if 'error' not in portfolio_risk:
                    # Store for health summary
                    self._last_portfolio_risk = portfolio_risk.get('portfolio_risk')
                    
                    # Check margin risks
                    new_alerts += await self._check_margin_risks(portfolio_risk)
                    
                    # Check liquidation risks
                    new_alerts += await self._check_liquidation_risks(portfolio_risk)
                    
                    # Check funding costs
                    new_alerts += await self._check_funding_risks(portfolio_risk)
                    
                    # Check leverage limits
                    new_alerts += await self._check_leverage_risks()
                    
                    # Check correlation risks
                    new_alerts += await self._check_correlation_risks(portfolio_risk)
                    
                    # Check market regime changes
                    new_alerts += await self._check_market_regime_changes()
            
            # Resolve alerts that are no longer valid
            resolved_alerts = await self._resolve_outdated_alerts()
            
            return {
                'new_alerts': new_alerts,
                'resolved_alerts': resolved_alerts,
                'total_alerts': len(self._active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error performing risk check: {e}")
            return {'error': str(e)}
    
    async def _check_margin_risks(self, portfolio_risk: Dict[str, Any]) -> int:
        """Check for margin-related risks."""
        portfolio_risk_obj = portfolio_risk.get('portfolio_risk')
        if not portfolio_risk_obj:
            return 0
        
        new_alerts = 0
        margin_ratio = portfolio_risk_obj.margin_ratio
        
        # Critical margin alert
        if margin_ratio >= self.config.margin_critical_threshold:
            await self._generate_alert(
                AlertType.MARGIN_WARNING,
                AlertSeverity.CRITICAL,
                "Critical Margin Level",
                f"Margin utilization at {margin_ratio:.1%} - immediate action required",
                margin_ratio,
                self.config.margin_critical_threshold,
                {"margin_ratio": margin_ratio, "available_margin": portfolio_risk_obj.available_margin},
                {},
                ["Reduce position sizes immediately", "Add additional margin", "Close high-risk positions"],
                15  # 15 minutes urgency
            )
            new_alerts += 1
            
        # Warning margin alert
        elif margin_ratio >= self.config.margin_warning_threshold:
            await self._generate_alert(
                AlertType.MARGIN_WARNING,
                AlertSeverity.WARNING,
                "High Margin Utilization",
                f"Margin utilization at {margin_ratio:.1%} - monitor closely",
                margin_ratio,
                self.config.margin_warning_threshold,
                {"margin_ratio": margin_ratio, "available_margin": portfolio_risk_obj.available_margin},
                {},
                ["Monitor positions closely", "Consider reducing leverage", "Prepare to add margin if needed"],
                60  # 60 minutes urgency
            )
            new_alerts += 1
        
        return new_alerts
    
    async def _check_liquidation_risks(self, portfolio_risk: Dict[str, Any]) -> int:
        """Check for liquidation risks."""
        liquidation_risks = portfolio_risk.get('liquidation_risks', [])
        new_alerts = 0
        
        for liq_risk in liquidation_risks:
            distance = liq_risk.distance_to_liquidation_pct
            
            if distance <= self.config.liquidation_critical_distance * 100:
                await self._generate_alert(
                    AlertType.LIQUIDATION_RISK,
                    AlertSeverity.EMERGENCY,
                    f"Liquidation Risk - {liq_risk.symbol}",
                    f"{liq_risk.symbol} within {distance:.1f}% of liquidation",
                    distance,
                    self.config.liquidation_critical_distance * 100,
                    {},
                    {"symbol": liq_risk.symbol, "liquidation_price": liq_risk.liquidation_price},
                    [f"Close {liq_risk.symbol} position immediately", "Add margin urgently", "Reduce position size"],
                    5  # 5 minutes urgency
                )
                new_alerts += 1
                
            elif distance <= self.config.liquidation_warning_distance * 100:
                await self._generate_alert(
                    AlertType.LIQUIDATION_RISK,
                    AlertSeverity.CRITICAL,
                    f"Liquidation Warning - {liq_risk.symbol}",
                    f"{liq_risk.symbol} within {distance:.1f}% of liquidation",
                    distance,
                    self.config.liquidation_warning_distance * 100,
                    {},
                    {"symbol": liq_risk.symbol, "liquidation_price": liq_risk.liquidation_price},
                    [f"Reduce {liq_risk.symbol} position size", "Add margin", "Monitor price closely"],
                    30  # 30 minutes urgency
                )
                new_alerts += 1
        
        return new_alerts
    
    async def _check_funding_risks(self, portfolio_risk: Dict[str, Any]) -> int:
        """Check for funding cost risks."""
        portfolio_risk_obj = portfolio_risk.get('portfolio_risk')
        if not portfolio_risk_obj:
            return 0
        
        new_alerts = 0
        funding_pnl = portfolio_risk_obj.funding_pnl_24h
        account_value = portfolio_risk_obj.total_account_value
        
        if account_value > 0:
            funding_percentage = abs(funding_pnl) / account_value
            
            if funding_percentage >= self.config.daily_funding_critical_percentage:
                await self._generate_alert(
                    AlertType.FUNDING_COST,
                    AlertSeverity.CRITICAL,
                    "High Funding Costs",
                    f"Daily funding costs at {funding_percentage:.2%} of account value",
                    funding_percentage,
                    self.config.daily_funding_critical_percentage,
                    {"daily_funding": funding_pnl, "account_value": account_value},
                    {},
                    ["Reduce perpetual positions", "Switch to spot trading", "Find better funding rates"],
                    120  # 2 hours urgency
                )
                new_alerts += 1
                
            elif funding_percentage >= self.config.daily_funding_warning_percentage:
                await self._generate_alert(
                    AlertType.FUNDING_COST,
                    AlertSeverity.WARNING,
                    "Elevated Funding Costs",
                    f"Daily funding costs at {funding_percentage:.2%} of account value",
                    funding_percentage,
                    self.config.daily_funding_warning_percentage,
                    {"daily_funding": funding_pnl, "account_value": account_value},
                    {},
                    ["Monitor funding rates", "Consider position adjustments", "Optimize funding strategy"],
                    360  # 6 hours urgency
                )
                new_alerts += 1
        
        return new_alerts
    
    async def _check_leverage_risks(self) -> int:
        """Check for leverage limit breaches."""
        new_alerts = 0
        
        leverage_monitor = await self.leverage_controller.monitor_leverage_limits(self._current_positions)
        
        if 'error' in leverage_monitor:
            return 0
        
        violations = leverage_monitor.get('violations', [])
        for violation in violations:
            await self._generate_alert(
                AlertType.LEVERAGE_BREACH,
                AlertSeverity.CRITICAL,
                f"Leverage Limit Breach - {violation['symbol']}",
                f"{violation['symbol']} leverage at {violation['current_leverage']:.1f}x exceeds limit of {violation['max_allowed']:.1f}x",
                violation['current_leverage'],
                violation['max_allowed'],
                {},
                {"symbol": violation['symbol'], "excess_leverage": violation['excess_leverage']},
                [f"Reduce {violation['symbol']} position size", "Adjust leverage settings", "Monitor market conditions"],
                45  # 45 minutes urgency
            )
            new_alerts += 1
        
        return new_alerts
    
    async def _check_correlation_risks(self, portfolio_risk: Dict[str, Any]) -> int:
        """Check for correlation concentration risks."""
        portfolio_risk_obj = portfolio_risk.get('portfolio_risk')
        if not portfolio_risk_obj:
            return 0
        
        new_alerts = 0
        correlation_concentration = portfolio_risk_obj.correlation_concentration
        
        if correlation_concentration >= self.config.max_correlation_exposure:
            await self._generate_alert(
                AlertType.CORRELATION_RISK,
                AlertSeverity.WARNING,
                "High Correlation Exposure",
                f"Portfolio correlation exposure at {correlation_concentration:.1%}",
                correlation_concentration,
                self.config.max_correlation_exposure,
                {"correlation_exposure": correlation_concentration},
                {},
                ["Diversify across different asset classes", "Reduce correlated positions", "Add uncorrelated assets"],
                240  # 4 hours urgency
            )
            new_alerts += 1
        
        return new_alerts
    
    async def _check_market_regime_changes(self) -> int:
        """Check for significant market regime changes."""
        # This is a simplified implementation
        # In practice, would compare current regime with stored previous regime
        return 0
    
    async def _generate_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        current_value: Optional[float],
        threshold_value: Optional[float],
        market_context: Dict[str, Any],
        position_context: Dict[str, Any],
        recommended_actions: List[str],
        urgency_minutes: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> None:
        """Generate a new risk alert."""
        
        # Create unique alert ID
        alert_id = f"{alert_type.value}_{symbol or 'portfolio'}_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Check if similar alert already exists
        existing_similar = [
            alert for alert in self._active_alerts.values()
            if (alert.alert_type == alert_type and 
                alert.symbol == symbol and 
                alert.severity == severity)
        ]
        
        if existing_similar:
            # Update existing alert instead of creating duplicate
            existing_alert = existing_similar[0]
            existing_alert.timestamp = datetime.now(timezone.utc)
            existing_alert.message = message
            existing_alert.current_value = current_value
            return
        
        # Determine risk level
        if severity == AlertSeverity.EMERGENCY:
            risk_level = RiskLevel.CRITICAL
        elif severity == AlertSeverity.CRITICAL:
            risk_level = RiskLevel.HIGH
        elif severity == AlertSeverity.WARNING:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Create new alert
        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            title=title,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            risk_level=risk_level,
            confidence=0.8,  # Default confidence
            recommended_actions=recommended_actions,
            urgency_minutes=urgency_minutes,
            market_context=market_context,
            position_context=position_context
        )
        
        # Store alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        self._alert_count += 1
        
        # Send to alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.info(f"Generated {severity.value} alert: {title}")
    
    async def _resolve_outdated_alerts(self) -> int:
        """Resolve alerts that are no longer valid."""
        resolved_count = 0
        alerts_to_remove = []
        
        for alert_id, alert in self._active_alerts.items():
            # Check if alert should be resolved based on current conditions
            should_resolve = False
            
            if alert.alert_type == AlertType.MARGIN_WARNING and self._last_portfolio_risk:
                if self._last_portfolio_risk.margin_ratio < self.config.margin_warning_threshold:
                    should_resolve = True
            
            elif alert.alert_type == AlertType.LIQUIDATION_RISK:
                # Would check if position is no longer at risk
                should_resolve = False  # Simplified for now
            
            # Auto-resolve old alerts
            if alert.age_minutes > 60:  # Auto-resolve after 1 hour
                should_resolve = True
            
            if should_resolve:
                alerts_to_remove.append(alert_id)
                resolved_count += 1
        
        # Remove resolved alerts
        for alert_id in alerts_to_remove:
            del self._active_alerts[alert_id]
        
        return resolved_count
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts from history."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config.alert_cleanup_interval_hours)
        
        self._alert_history = [
            alert for alert in self._alert_history 
            if alert.timestamp > cutoff_time
        ] 