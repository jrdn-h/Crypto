"""
Whale flow tracking and large transaction analysis for crypto markets.

This module provides on-chain analysis capabilities including whale transaction detection,
exchange flow monitoring, and large holder movement tracking.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..base_interfaces import AssetClass, DataQuality
from .caching import CacheManager, CacheConfig
from .rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


class TransactionType(str, Enum):
    """Types of whale transactions."""
    LARGE_TRANSFER = "large_transfer"
    EXCHANGE_DEPOSIT = "exchange_deposit"
    EXCHANGE_WITHDRAWAL = "exchange_withdrawal"
    WHALE_ACCUMULATION = "whale_accumulation"
    WHALE_DISTRIBUTION = "whale_distribution"
    UNKNOWN = "unknown"


class FlowDirection(str, Enum):
    """Direction of fund flows."""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NEUTRAL = "neutral"


@dataclass
class WhaleTransaction:
    """Individual whale transaction data."""
    transaction_hash: str
    timestamp: datetime
    from_address: str
    to_address: str
    amount: float
    amount_usd: float
    symbol: str
    transaction_type: TransactionType
    exchange_involved: Optional[str] = None
    is_internal: bool = False
    fee_usd: Optional[float] = None
    block_number: Optional[int] = None


@dataclass
class ExchangeFlow:
    """Exchange flow analysis data."""
    exchange: str
    symbol: str
    timestamp: datetime
    inflow_24h: float
    outflow_24h: float
    net_flow_24h: float
    inflow_7d: float
    outflow_7d: float
    net_flow_7d: float
    large_deposits_count: int
    large_withdrawals_count: int


@dataclass
class WhaleAlert:
    """Whale activity alert."""
    alert_id: str
    timestamp: datetime
    symbol: str
    alert_type: str
    amount_usd: float
    description: str
    confidence_score: float
    related_addresses: List[str]
    exchange_involved: Optional[str] = None


class WhaleFlowTracker:
    """Whale flow tracking and analysis system."""
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
        whale_threshold_usd: float = 1_000_000,
        large_tx_threshold_usd: float = 100_000
    ):
        """
        Initialize whale flow tracker.
        
        Args:
            cache_manager: Cache manager for storing analysis results
            rate_limiter: Rate limiter for API calls
            whale_threshold_usd: USD threshold for whale transactions
            large_tx_threshold_usd: USD threshold for large transactions
        """
        self.cache_manager = cache_manager or CacheManager(CacheConfig())
        
        # Conservative rate limiting for on-chain APIs
        default_rate_config = RateLimitConfig(requests_per_minute=30)
        self.rate_limiter = rate_limiter or RateLimiter(default_rate_config)
        
        self.whale_threshold_usd = whale_threshold_usd
        self.large_tx_threshold_usd = large_tx_threshold_usd
        
        # Known exchange addresses (would be expanded with real data)
        self.known_exchanges = {
            'binance': ['1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s', 'bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h'],
            'coinbase': ['3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64', 'bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv1yk59smf3m2q'],
            'kraken': ['3BMEXvBCKzG8QfYyS3qQ9Gs68XjGsKF3kM', 'bc1qjasf9z3h7w3jjvgk6f6jh9f6gf4h3f2hjf4h3f'],
            'okx': ['1KzTSfqjF2iKCduwz59nv2uqh1W2JdaTYe', 'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh'],
            'bybit': ['1Ey4pCZXjY8HxYg5sE9aHjBfQ9FhY8zQZr', 'bc1qh4pz9nz5nh7j7h6pz9nz5nh7j7h6pz9nz5nh7j7']
        }
        
        # Known whale addresses (would be populated with real data)
        self.known_whales = {
            'tesla': '1Nc6XCUKgY5JVpBj6hbhPcHNJ8H2cKjqeH',
            'microstrategy': '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ',
            'satoshi': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'
        }
    
    async def analyze_whale_activity(
        self,
        symbol: str,
        hours_back: int = 24,
        include_exchange_flows: bool = True
    ) -> Dict[str, Union[List, Dict, float]]:
        """
        Analyze whale activity for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            hours_back: Hours to look back for analysis
            include_exchange_flows: Whether to include exchange flow analysis
            
        Returns:
            Dictionary containing whale activity analysis
        """
        cache_key = f"whale_activity_{symbol}_{hours_back}_{include_exchange_flows}"
        
        # Try cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Using cached whale activity for {symbol}")
            return cached_result
        
        try:
            # Fetch whale transactions
            whale_transactions = await self._fetch_whale_transactions(symbol, hours_back)
            
            # Analyze transaction patterns
            transaction_analysis = self._analyze_transaction_patterns(whale_transactions)
            
            # Exchange flow analysis
            exchange_flows = []
            if include_exchange_flows:
                exchange_flows = await self._analyze_exchange_flows(symbol, hours_back)
            
            # Generate whale alerts
            whale_alerts = self._generate_whale_alerts(whale_transactions, exchange_flows)
            
            # Calculate summary metrics
            summary_metrics = self._calculate_summary_metrics(
                whale_transactions, exchange_flows, hours_back
            )
            
            # Compile results
            analysis_results = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'hours_analyzed': hours_back,
                'whale_transactions': [self._transaction_to_dict(tx) for tx in whale_transactions],
                'transaction_analysis': transaction_analysis,
                'exchange_flows': [self._exchange_flow_to_dict(flow) for flow in exchange_flows],
                'whale_alerts': [self._alert_to_dict(alert) for alert in whale_alerts],
                'summary_metrics': summary_metrics,
                'data_quality': self._assess_whale_data_quality(whale_transactions, exchange_flows)
            }
            
            # Cache the results
            await self.cache_manager.set(cache_key, analysis_results, ttl_seconds=600)  # 10 minutes
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing whale activity for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def get_whale_flow_summary(
        self,
        symbol: str,
        timeframe: str = "24h"
    ) -> str:
        """
        Get a formatted summary of whale flow activity.
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Analysis timeframe ('1h', '4h', '24h', '7d')
            
        Returns:
            Formatted whale flow summary
        """
        try:
            # Convert timeframe to hours
            timeframe_hours = {
                '1h': 1, '4h': 4, '24h': 24, '7d': 168
            }.get(timeframe, 24)
            
            # Get whale activity analysis
            whale_data = await self.analyze_whale_activity(
                symbol, hours_back=timeframe_hours, include_exchange_flows=True
            )
            
            # Generate formatted report
            return self._generate_whale_flow_report(whale_data, timeframe)
            
        except Exception as e:
            logger.error(f"Error generating whale flow summary: {e}")
            return f"❌ Error generating whale flow summary for {symbol}: {str(e)}"
    
    async def _fetch_whale_transactions(
        self,
        symbol: str,
        hours_back: int
    ) -> List[WhaleTransaction]:
        """Fetch whale transactions from on-chain data sources."""
        try:
            await self.rate_limiter.acquire()
            
            # Mock whale transaction data - replace with actual on-chain API calls
            transactions = []
            
            # Generate mock whale transactions
            current_time = datetime.now()
            for i in range(random.randint(3, 15)):  # 3-15 whale transactions
                tx_time = current_time - timedelta(hours=random.uniform(0, hours_back))
                
                # Mock transaction amounts
                amount_usd = random.uniform(self.large_tx_threshold_usd, self.whale_threshold_usd * 5)
                
                # Determine transaction type
                tx_type = random.choice([
                    TransactionType.LARGE_TRANSFER,
                    TransactionType.EXCHANGE_DEPOSIT,
                    TransactionType.EXCHANGE_WITHDRAWAL,
                    TransactionType.WHALE_ACCUMULATION
                ])
                
                # Mock addresses
                from_addr = f"1{''.join(random.choices('23456789ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=25))}"
                to_addr = f"1{''.join(random.choices('23456789ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=25))}"
                
                # Mock exchange involvement
                exchange = None
                if tx_type in [TransactionType.EXCHANGE_DEPOSIT, TransactionType.EXCHANGE_WITHDRAWAL]:
                    exchange = random.choice(['Binance', 'Coinbase', 'Kraken', 'OKX'])
                
                whale_tx = WhaleTransaction(
                    transaction_hash=f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
                    timestamp=tx_time,
                    from_address=from_addr,
                    to_address=to_addr,
                    amount=amount_usd / 50000,  # Mock price conversion
                    amount_usd=amount_usd,
                    symbol=symbol,
                    transaction_type=tx_type,
                    exchange_involved=exchange,
                    fee_usd=random.uniform(10, 100)
                )
                
                transactions.append(whale_tx)
            
            # Sort by timestamp (newest first)
            transactions.sort(key=lambda x: x.timestamp, reverse=True)
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error fetching whale transactions: {e}")
            return []
    
    def _analyze_transaction_patterns(
        self,
        transactions: List[WhaleTransaction]
    ) -> Dict[str, Union[int, float, str]]:
        """Analyze patterns in whale transactions."""
        if not transactions:
            return {}
        
        try:
            total_volume = sum(tx.amount_usd for tx in transactions)
            avg_transaction_size = total_volume / len(transactions)
            
            # Count by transaction type
            type_counts = {}
            for tx in transactions:
                tx_type = tx.transaction_type.value
                type_counts[tx_type] = type_counts.get(tx_type, 0) + 1
            
            # Exchange activity
            exchange_transactions = [tx for tx in transactions if tx.exchange_involved]
            exchange_volume = sum(tx.amount_usd for tx in exchange_transactions)
            
            # Timing analysis
            recent_1h = [tx for tx in transactions if 
                        (datetime.now() - tx.timestamp).total_seconds() < 3600]
            recent_4h = [tx for tx in transactions if 
                        (datetime.now() - tx.timestamp).total_seconds() < 14400]
            
            # Determine dominant pattern
            if len(recent_1h) > len(transactions) * 0.5:
                activity_pattern = "high_recent_activity"
            elif type_counts.get('exchange_deposit', 0) > type_counts.get('exchange_withdrawal', 0):
                activity_pattern = "net_exchange_inflow"
            elif type_counts.get('exchange_withdrawal', 0) > type_counts.get('exchange_deposit', 0):
                activity_pattern = "net_exchange_outflow"
            else:
                activity_pattern = "balanced_activity"
            
            return {
                'total_transactions': len(transactions),
                'total_volume_usd': total_volume,
                'average_transaction_usd': avg_transaction_size,
                'largest_transaction_usd': max(tx.amount_usd for tx in transactions),
                'transaction_types': type_counts,
                'exchange_transactions': len(exchange_transactions),
                'exchange_volume_usd': exchange_volume,
                'recent_1h_count': len(recent_1h),
                'recent_4h_count': len(recent_4h),
                'activity_pattern': activity_pattern
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transaction patterns: {e}")
            return {}
    
    async def _analyze_exchange_flows(
        self,
        symbol: str,
        hours_back: int
    ) -> List[ExchangeFlow]:
        """Analyze exchange flow patterns."""
        try:
            flows = []
            
            # Mock exchange flow data for major exchanges
            for exchange in ['Binance', 'Coinbase', 'Kraken', 'OKX', 'Bybit']:
                # Mock flow data
                inflow_24h = random.uniform(1_000_000, 50_000_000)
                outflow_24h = random.uniform(1_000_000, 50_000_000)
                
                flow = ExchangeFlow(
                    exchange=exchange,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    inflow_24h=inflow_24h,
                    outflow_24h=outflow_24h,
                    net_flow_24h=inflow_24h - outflow_24h,
                    inflow_7d=inflow_24h * random.uniform(6, 8),
                    outflow_7d=outflow_24h * random.uniform(6, 8),
                    net_flow_7d=(inflow_24h - outflow_24h) * random.uniform(6, 8),
                    large_deposits_count=random.randint(5, 25),
                    large_withdrawals_count=random.randint(5, 25)
                )
                
                flows.append(flow)
            
            return flows
            
        except Exception as e:
            logger.error(f"Error analyzing exchange flows: {e}")
            return []
    
    def _generate_whale_alerts(
        self,
        transactions: List[WhaleTransaction],
        exchange_flows: List[ExchangeFlow]
    ) -> List[WhaleAlert]:
        """Generate whale activity alerts."""
        alerts = []
        
        try:
            # Large transaction alerts
            for tx in transactions:
                if tx.amount_usd > self.whale_threshold_usd:
                    alert = WhaleAlert(
                        alert_id=f"whale_tx_{tx.transaction_hash[:8]}",
                        timestamp=tx.timestamp,
                        symbol=tx.symbol,
                        alert_type="large_transaction",
                        amount_usd=tx.amount_usd,
                        description=f"Large {tx.transaction_type.value} of ${tx.amount_usd:,.0f}",
                        confidence_score=0.9,
                        related_addresses=[tx.from_address, tx.to_address],
                        exchange_involved=tx.exchange_involved
                    )
                    alerts.append(alert)
            
            # Exchange flow alerts
            for flow in exchange_flows:
                if abs(flow.net_flow_24h) > 10_000_000:  # $10M threshold
                    flow_type = "inflow" if flow.net_flow_24h > 0 else "outflow"
                    alert = WhaleAlert(
                        alert_id=f"exchange_flow_{flow.exchange.lower()}_{flow_type}",
                        timestamp=flow.timestamp,
                        symbol=flow.symbol,
                        alert_type=f"large_exchange_{flow_type}",
                        amount_usd=abs(flow.net_flow_24h),
                        description=f"Large {flow_type} at {flow.exchange}: ${abs(flow.net_flow_24h):,.0f}",
                        confidence_score=0.8,
                        related_addresses=[],
                        exchange_involved=flow.exchange
                    )
                    alerts.append(alert)
            
            # Sort alerts by amount (largest first)
            alerts.sort(key=lambda x: x.amount_usd, reverse=True)
            
            return alerts[:10]  # Return top 10 alerts
            
        except Exception as e:
            logger.error(f"Error generating whale alerts: {e}")
            return []
    
    def _calculate_summary_metrics(
        self,
        transactions: List[WhaleTransaction],
        exchange_flows: List[ExchangeFlow],
        hours_back: int
    ) -> Dict[str, Union[float, int, str]]:
        """Calculate summary metrics for whale activity."""
        try:
            # Transaction metrics
            total_whale_volume = sum(tx.amount_usd for tx in transactions 
                                   if tx.amount_usd >= self.whale_threshold_usd)
            
            # Exchange metrics
            total_exchange_inflow = sum(flow.inflow_24h for flow in exchange_flows)
            total_exchange_outflow = sum(flow.outflow_24h for flow in exchange_flows)
            net_exchange_flow = total_exchange_inflow - total_exchange_outflow
            
            # Activity metrics
            whale_transaction_count = len([tx for tx in transactions 
                                         if tx.amount_usd >= self.whale_threshold_usd])
            
            # Flow direction analysis
            if net_exchange_flow > 5_000_000:
                flow_sentiment = "bearish"  # Money flowing to exchanges
            elif net_exchange_flow < -5_000_000:
                flow_sentiment = "bullish"  # Money flowing from exchanges
            else:
                flow_sentiment = "neutral"
            
            # Activity intensity
            activity_per_hour = len(transactions) / hours_back
            if activity_per_hour > 2:
                activity_level = "high"
            elif activity_per_hour > 0.5:
                activity_level = "moderate"
            else:
                activity_level = "low"
            
            return {
                'total_whale_volume_usd': total_whale_volume,
                'whale_transaction_count': whale_transaction_count,
                'total_exchange_inflow_usd': total_exchange_inflow,
                'total_exchange_outflow_usd': total_exchange_outflow,
                'net_exchange_flow_usd': net_exchange_flow,
                'flow_sentiment': flow_sentiment,
                'activity_level': activity_level,
                'activity_per_hour': activity_per_hour,
                'largest_single_transaction_usd': max((tx.amount_usd for tx in transactions), default=0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary metrics: {e}")
            return {}
    
    def _assess_whale_data_quality(
        self,
        transactions: List[WhaleTransaction],
        exchange_flows: List[ExchangeFlow]
    ) -> Dict[str, Union[str, float, bool]]:
        """Assess quality of whale flow data."""
        try:
            # Data completeness
            transaction_completeness = 1.0 if transactions else 0.0
            exchange_flow_completeness = 1.0 if exchange_flows else 0.0
            
            # Data recency
            if transactions:
                latest_tx = max(transactions, key=lambda x: x.timestamp)
                data_age_hours = (datetime.now() - latest_tx.timestamp).total_seconds() / 3600
                recency_score = max(0, 1 - (data_age_hours / 24))  # Decay over 24 hours
            else:
                recency_score = 0.0
            
            # Overall quality score
            overall_score = (transaction_completeness * 0.5 + 
                           exchange_flow_completeness * 0.3 + 
                           recency_score * 0.2)
            
            # Quality rating
            if overall_score >= 0.8:
                rating = "high"
            elif overall_score >= 0.6:
                rating = "medium"
            else:
                rating = "low"
            
            return {
                'transaction_data_available': len(transactions) > 0,
                'exchange_flow_data_available': len(exchange_flows) > 0,
                'data_recency_score': recency_score,
                'overall_quality_score': overall_score,
                'quality_rating': rating
            }
            
        except Exception as e:
            logger.error(f"Error assessing whale data quality: {e}")
            return {'quality_rating': 'unknown', 'error': str(e)}
    
    def _generate_whale_flow_report(
        self,
        whale_data: Dict,
        timeframe: str
    ) -> str:
        """Generate formatted whale flow report."""
        try:
            symbol = whale_data.get('symbol', 'Unknown')
            summary = whale_data.get('summary_metrics', {})
            alerts = whale_data.get('whale_alerts', [])
            
            report_lines = []
            
            # Header
            report_lines.append(f"# 🐋 Whale Flow Analysis: {symbol}")
            report_lines.append(f"**Timeframe**: {timeframe}")
            report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            report_lines.append("")
            
            # Summary metrics
            if summary:
                report_lines.append("## 📊 Summary Metrics")
                
                if 'total_whale_volume_usd' in summary:
                    volume = summary['total_whale_volume_usd']
                    report_lines.append(f"- 💰 Total Whale Volume: **${volume:,.0f}**")
                
                if 'whale_transaction_count' in summary:
                    count = summary['whale_transaction_count']
                    report_lines.append(f"- 🔢 Whale Transactions: **{count}**")
                
                if 'net_exchange_flow_usd' in summary:
                    net_flow = summary['net_exchange_flow_usd']
                    flow_emoji = "📈" if net_flow > 0 else "📉" if net_flow < 0 else "➡️"
                    report_lines.append(f"- {flow_emoji} Net Exchange Flow: **${net_flow:+,.0f}**")
                
                if 'flow_sentiment' in summary:
                    sentiment = summary['flow_sentiment']
                    sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(sentiment, "")
                    report_lines.append(f"- {sentiment_emoji} Flow Sentiment: **{sentiment.title()}**")
                
                if 'activity_level' in summary:
                    activity = summary['activity_level']
                    activity_emoji = {"high": "🔥", "moderate": "📊", "low": "😴"}.get(activity, "")
                    report_lines.append(f"- {activity_emoji} Activity Level: **{activity.title()}**")
                
                report_lines.append("")
            
            # Top alerts
            if alerts:
                report_lines.append("## 🚨 Top Whale Alerts")
                
                for i, alert in enumerate(alerts[:5]):  # Top 5 alerts
                    alert_emoji = {"large_transaction": "💸", 
                                 "large_exchange_inflow": "📥", 
                                 "large_exchange_outflow": "📤"}.get(alert['alert_type'], "⚠️")
                    
                    amount = alert['amount_usd']
                    description = alert['description']
                    report_lines.append(f"- {alert_emoji} **${amount:,.0f}**: {description}")
                
                report_lines.append("")
            
            # Data quality
            data_quality = whale_data.get('data_quality', {})
            if 'quality_rating' in data_quality:
                quality = data_quality['quality_rating']
                quality_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(quality, "")
                report_lines.append(f"**Data Quality**: {quality_emoji} {quality.title()}")
            
            report_lines.append("")
            report_lines.append("*Whale analysis based on on-chain transaction data and exchange flow monitoring.*")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating whale flow report: {e}")
            return f"❌ Error generating whale flow report: {str(e)}"
    
    def _transaction_to_dict(self, tx: WhaleTransaction) -> Dict:
        """Convert WhaleTransaction to dictionary."""
        return {
            'transaction_hash': tx.transaction_hash,
            'timestamp': tx.timestamp.isoformat(),
            'from_address': tx.from_address,
            'to_address': tx.to_address,
            'amount': tx.amount,
            'amount_usd': tx.amount_usd,
            'symbol': tx.symbol,
            'transaction_type': tx.transaction_type.value,
            'exchange_involved': tx.exchange_involved,
            'is_internal': tx.is_internal,
            'fee_usd': tx.fee_usd
        }
    
    def _exchange_flow_to_dict(self, flow: ExchangeFlow) -> Dict:
        """Convert ExchangeFlow to dictionary."""
        return {
            'exchange': flow.exchange,
            'symbol': flow.symbol,
            'timestamp': flow.timestamp.isoformat(),
            'inflow_24h': flow.inflow_24h,
            'outflow_24h': flow.outflow_24h,
            'net_flow_24h': flow.net_flow_24h,
            'inflow_7d': flow.inflow_7d,
            'outflow_7d': flow.outflow_7d,
            'net_flow_7d': flow.net_flow_7d,
            'large_deposits_count': flow.large_deposits_count,
            'large_withdrawals_count': flow.large_withdrawals_count
        }
    
    def _alert_to_dict(self, alert: WhaleAlert) -> Dict:
        """Convert WhaleAlert to dictionary."""
        return {
            'alert_id': alert.alert_id,
            'timestamp': alert.timestamp.isoformat(),
            'symbol': alert.symbol,
            'alert_type': alert.alert_type,
            'amount_usd': alert.amount_usd,
            'description': alert.description,
            'confidence_score': alert.confidence_score,
            'related_addresses': alert.related_addresses,
            'exchange_involved': alert.exchange_involved
        }


# For mock data generation
import random 