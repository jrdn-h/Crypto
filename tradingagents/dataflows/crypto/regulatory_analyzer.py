"""
Regulatory risk analysis module for crypto assets.

This module provides comprehensive regulatory analysis including compliance risks,
regulatory clarity assessment, and government action monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RegulatoryStatus(str, Enum):
    """Regulatory status levels."""
    CLEAR = "clear"
    PARTIALLY_CLEAR = "partially_clear"
    UNCLEAR = "unclear"
    RESTRICTED = "restricted"
    BANNED = "banned"
    UNKNOWN = "unknown"


class RegulatoryRisk(str, Enum):
    """Regulatory risk levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ComplianceStatus(str, Enum):
    """Compliance status options."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    UNKNOWN = "unknown"


@dataclass
class RegulatoryJurisdiction:
    """Regulatory status for a specific jurisdiction."""
    jurisdiction: str
    status: RegulatoryStatus
    compliance_status: ComplianceStatus
    last_updated: datetime
    key_regulations: List[str]
    pending_legislation: List[str]
    enforcement_actions: List[str]


@dataclass
class RegulatoryEvent:
    """Regulatory event or development."""
    date: datetime
    jurisdiction: str
    event_type: str  # "legislation", "enforcement", "guidance", "court_ruling"
    description: str
    impact_assessment: str  # "positive", "negative", "neutral"
    confidence_level: float


class RegulatoryAnalyzer:
    """Comprehensive regulatory analysis for crypto assets."""
    
    def __init__(self):
        """Initialize regulatory analyzer."""
        pass
    
    async def analyze_regulatory_environment(
        self,
        symbol: str,
        focus_jurisdictions: Optional[List[str]] = None
    ) -> Dict[str, Union[Dict, List, str, float]]:
        """
        Perform comprehensive regulatory analysis.
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH', 'XRP')
            focus_jurisdictions: List of jurisdictions to focus on
            
        Returns:
            Dictionary containing regulatory analysis
        """
        try:
            if focus_jurisdictions is None:
                focus_jurisdictions = ['US', 'EU', 'UK', 'Japan', 'Singapore', 'China']
            
            # Analyze regulatory status by jurisdiction
            jurisdictions = []
            for jurisdiction in focus_jurisdictions:
                jurisdiction_analysis = await self._analyze_jurisdiction(symbol, jurisdiction)
                jurisdictions.append(jurisdiction_analysis)
            
            # Get recent regulatory events
            recent_events = await self._get_recent_regulatory_events(symbol)
            
            # Assess overall regulatory risk
            risk_assessment = self._assess_regulatory_risk(jurisdictions, recent_events)
            
            # Generate regulatory bull/bear points
            bull_points = self._generate_regulatory_bull_points(jurisdictions, recent_events)
            bear_points = self._generate_regulatory_bear_points(jurisdictions, recent_events)
            
            # Identify key risks and opportunities
            key_risks = self._identify_key_regulatory_risks(jurisdictions, recent_events)
            opportunities = self._identify_regulatory_opportunities(jurisdictions, recent_events)
            
            return {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'jurisdictions': [self._jurisdiction_to_dict(j) for j in jurisdictions],
                'recent_events': [self._event_to_dict(e) for e in recent_events],
                'risk_assessment': risk_assessment,
                'bull_points': bull_points,
                'bear_points': bear_points,
                'key_risks': key_risks,
                'opportunities': opportunities,
                'overall_regulatory_score': self._calculate_regulatory_score(jurisdictions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regulatory environment for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def _analyze_jurisdiction(self, symbol: str, jurisdiction: str) -> RegulatoryJurisdiction:
        """Analyze regulatory status for a specific jurisdiction."""
        try:
            # Mock regulatory data - in real implementation, this would fetch from regulatory APIs
            regulatory_data = self._get_mock_regulatory_data(symbol, jurisdiction)
            
            return RegulatoryJurisdiction(
                jurisdiction=jurisdiction,
                status=RegulatoryStatus(regulatory_data.get('status', 'unknown')),
                compliance_status=ComplianceStatus(regulatory_data.get('compliance', 'unknown')),
                last_updated=datetime.now() - timedelta(days=regulatory_data.get('days_since_update', 30)),
                key_regulations=regulatory_data.get('regulations', []),
                pending_legislation=regulatory_data.get('pending', []),
                enforcement_actions=regulatory_data.get('enforcement', [])
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing jurisdiction {jurisdiction}: {e}")
            return RegulatoryJurisdiction(
                jurisdiction=jurisdiction,
                status=RegulatoryStatus.UNKNOWN,
                compliance_status=ComplianceStatus.UNKNOWN,
                last_updated=datetime.now(),
                key_regulations=[],
                pending_legislation=[],
                enforcement_actions=[]
            )
    
    def _get_mock_regulatory_data(self, symbol: str, jurisdiction: str) -> Dict:
        """Get mock regulatory data for testing purposes."""
        # Mock data based on real-world regulatory situations
        data = {
            ('BTC', 'US'): {
                'status': 'partially_clear',
                'compliance': 'partially_compliant',
                'regulations': ['Bank Secrecy Act', 'SEC Guidance on Digital Assets'],
                'pending': ['Cryptocurrency Market Structure Bill'],
                'enforcement': ['Recent SEC enforcement on unregistered securities'],
                'days_since_update': 15
            },
            ('ETH', 'US'): {
                'status': 'partially_clear',
                'compliance': 'under_review',
                'regulations': ['CFTC commodity classification'],
                'pending': ['Ethereum 2.0 staking guidance'],
                'enforcement': [],
                'days_since_update': 30
            },
            ('XRP', 'US'): {
                'status': 'unclear',
                'compliance': 'under_review',
                'regulations': [],
                'pending': ['SEC vs Ripple lawsuit resolution'],
                'enforcement': ['SEC lawsuit alleging unregistered security'],
                'days_since_update': 7
            },
            ('BTC', 'EU'): {
                'status': 'clear',
                'compliance': 'compliant',
                'regulations': ['MiCA Regulation', 'AMLD5'],
                'pending': ['MiCA implementation guidelines'],
                'enforcement': [],
                'days_since_update': 20
            },
            ('BTC', 'China'): {
                'status': 'banned',
                'compliance': 'non_compliant',
                'regulations': ['Trading and mining ban'],
                'pending': [],
                'enforcement': ['Exchange shutdowns', 'Mining farm closures'],
                'days_since_update': 45
            },
            ('BTC', 'Japan'): {
                'status': 'clear',
                'compliance': 'compliant',
                'regulations': ['Virtual Currency Act', 'FSA licensing'],
                'pending': [],
                'enforcement': [],
                'days_since_update': 10
            }
        }
        
        return data.get((symbol, jurisdiction), {
            'status': 'unknown',
            'compliance': 'unknown',
            'regulations': [],
            'pending': [],
            'enforcement': [],
            'days_since_update': 90
        })
    
    async def _get_recent_regulatory_events(self, symbol: str) -> List[RegulatoryEvent]:
        """Get recent regulatory events affecting the token."""
        try:
            # Mock recent events - would fetch from regulatory news APIs
            mock_events = [
                RegulatoryEvent(
                    date=datetime.now() - timedelta(days=5),
                    jurisdiction="US",
                    event_type="guidance",
                    description="SEC Chair clarifies stance on cryptocurrency regulation",
                    impact_assessment="positive",
                    confidence_level=0.8
                ),
                RegulatoryEvent(
                    date=datetime.now() - timedelta(days=12),
                    jurisdiction="EU",
                    event_type="legislation",
                    description="European Parliament approves MiCA implementation timeline",
                    impact_assessment="positive",
                    confidence_level=0.9
                ),
                RegulatoryEvent(
                    date=datetime.now() - timedelta(days=18),
                    jurisdiction="UK",
                    event_type="guidance",
                    description="FCA publishes updated cryptocurrency guidance",
                    impact_assessment="neutral",
                    confidence_level=0.7
                )
            ]
            
            # Filter events relevant to the symbol
            if symbol == 'XRP':
                mock_events.append(RegulatoryEvent(
                    date=datetime.now() - timedelta(days=3),
                    jurisdiction="US",
                    event_type="court_ruling",
                    description="Partial victory in SEC vs Ripple case",
                    impact_assessment="positive",
                    confidence_level=0.95
                ))
            
            return mock_events
            
        except Exception as e:
            logger.warning(f"Error getting regulatory events: {e}")
            return []
    
    def _assess_regulatory_risk(
        self,
        jurisdictions: List[RegulatoryJurisdiction],
        events: List[RegulatoryEvent]
    ) -> Dict[str, Union[str, float, List[str]]]:
        """Assess overall regulatory risk."""
        try:
            risk_factors = []
            risk_score = 0
            
            # Analyze jurisdiction risks
            for jurisdiction in jurisdictions:
                if jurisdiction.status == RegulatoryStatus.BANNED:
                    risk_factors.append(f"{jurisdiction.jurisdiction}: Complete ban on crypto activities")
                    risk_score += 30
                elif jurisdiction.status == RegulatoryStatus.RESTRICTED:
                    risk_factors.append(f"{jurisdiction.jurisdiction}: Significant regulatory restrictions")
                    risk_score += 20
                elif jurisdiction.status == RegulatoryStatus.UNCLEAR:
                    risk_factors.append(f"{jurisdiction.jurisdiction}: Regulatory uncertainty")
                    risk_score += 15
                
                if jurisdiction.enforcement_actions:
                    risk_factors.append(f"{jurisdiction.jurisdiction}: Active enforcement actions")
                    risk_score += 10
                
                if jurisdiction.pending_legislation:
                    risk_factors.append(f"{jurisdiction.jurisdiction}: Pending legislation may change rules")
                    risk_score += 5
            
            # Analyze recent events
            negative_events = [e for e in events if e.impact_assessment == "negative"]
            if len(negative_events) > 2:
                risk_factors.append("Multiple recent negative regulatory developments")
                risk_score += 15
            
            # Determine risk level
            if risk_score >= 80:
                risk_level = RegulatoryRisk.VERY_HIGH
            elif risk_score >= 60:
                risk_level = RegulatoryRisk.HIGH
            elif risk_score >= 40:
                risk_level = RegulatoryRisk.MEDIUM
            elif risk_score >= 20:
                risk_level = RegulatoryRisk.LOW
            else:
                risk_level = RegulatoryRisk.VERY_LOW
            
            return {
                'risk_level': risk_level.value,
                'risk_score': min(risk_score, 100),
                'risk_factors': risk_factors,
                'major_jurisdictions_at_risk': [
                    j.jurisdiction for j in jurisdictions 
                    if j.status in [RegulatoryStatus.BANNED, RegulatoryStatus.RESTRICTED, RegulatoryStatus.UNCLEAR]
                ]
            }
            
        except Exception as e:
            logger.warning(f"Error assessing regulatory risk: {e}")
            return {
                'risk_level': 'unknown',
                'risk_score': 50,
                'risk_factors': [],
                'major_jurisdictions_at_risk': []
            }
    
    def _generate_regulatory_bull_points(
        self,
        jurisdictions: List[RegulatoryJurisdiction],
        events: List[RegulatoryEvent]
    ) -> List[str]:
        """Generate bullish regulatory points."""
        bull_points = []
        
        # Clear regulatory status
        clear_jurisdictions = [j for j in jurisdictions if j.status == RegulatoryStatus.CLEAR]
        if clear_jurisdictions:
            major_clear = [j.jurisdiction for j in clear_jurisdictions if j.jurisdiction in ['US', 'EU', 'Japan']]
            if major_clear:
                bull_points.append(f"Clear regulatory framework in major markets: {', '.join(major_clear)}")
        
        # Positive recent events
        positive_events = [e for e in events if e.impact_assessment == "positive"]
        if len(positive_events) >= 2:
            bull_points.append("Multiple recent positive regulatory developments")
        
        # Strong compliance status
        compliant_jurisdictions = [j for j in jurisdictions if j.compliance_status == ComplianceStatus.COMPLIANT]
        if len(compliant_jurisdictions) >= 3:
            bull_points.append("Strong compliance track record across multiple jurisdictions")
        
        # Pending favorable legislation
        favorable_pending = []
        for jurisdiction in jurisdictions:
            for pending in jurisdiction.pending_legislation:
                if any(word in pending.lower() for word in ['approval', 'clarity', 'framework', 'guidelines']):
                    favorable_pending.append(f"{jurisdiction.jurisdiction}: {pending}")
        
        if favorable_pending:
            bull_points.append("Pending legislation likely to provide regulatory clarity")
        
        # Lack of enforcement actions
        no_enforcement = [j for j in jurisdictions if not j.enforcement_actions]
        if len(no_enforcement) >= 4:
            bull_points.append("No significant enforcement actions in major markets")
        
        return bull_points
    
    def _generate_regulatory_bear_points(
        self,
        jurisdictions: List[RegulatoryJurisdiction],
        events: List[RegulatoryEvent]
    ) -> List[str]:
        """Generate bearish regulatory points."""
        bear_points = []
        
        # Banned or restricted jurisdictions
        banned_jurisdictions = [j for j in jurisdictions if j.status == RegulatoryStatus.BANNED]
        if banned_jurisdictions:
            bear_points.append(f"Banned in major markets: {', '.join([j.jurisdiction for j in banned_jurisdictions])}")
        
        restricted_jurisdictions = [j for j in jurisdictions if j.status == RegulatoryStatus.RESTRICTED]
        if restricted_jurisdictions:
            bear_points.append(f"Restricted access in: {', '.join([j.jurisdiction for j in restricted_jurisdictions])}")
        
        # Regulatory uncertainty
        unclear_jurisdictions = [j for j in jurisdictions if j.status == RegulatoryStatus.UNCLEAR]
        major_unclear = [j.jurisdiction for j in unclear_jurisdictions if j.jurisdiction in ['US', 'EU', 'UK']]
        if major_unclear:
            bear_points.append(f"Regulatory uncertainty in key markets: {', '.join(major_unclear)}")
        
        # Enforcement actions
        enforcement_jurisdictions = [j for j in jurisdictions if j.enforcement_actions]
        if enforcement_jurisdictions:
            bear_points.append("Active enforcement actions creating compliance risks")
        
        # Negative recent events
        negative_events = [e for e in events if e.impact_assessment == "negative"]
        if negative_events:
            bear_points.append("Recent negative regulatory developments")
        
        # Non-compliance issues
        non_compliant = [j for j in jurisdictions if j.compliance_status == ComplianceStatus.NON_COMPLIANT]
        if non_compliant:
            bear_points.append(f"Non-compliance issues in: {', '.join([j.jurisdiction for j in non_compliant])}")
        
        # Pending restrictive legislation
        restrictive_pending = []
        for jurisdiction in jurisdictions:
            for pending in jurisdiction.pending_legislation:
                if any(word in pending.lower() for word in ['ban', 'restriction', 'prohibition', 'crackdown']):
                    restrictive_pending.append(f"{jurisdiction.jurisdiction}: {pending}")
        
        if restrictive_pending:
            bear_points.append("Pending legislation may impose new restrictions")
        
        return bear_points
    
    def _identify_key_regulatory_risks(
        self,
        jurisdictions: List[RegulatoryJurisdiction],
        events: List[RegulatoryEvent]
    ) -> List[str]:
        """Identify key regulatory risks."""
        risks = []
        
        # Major market risks
        us_jurisdiction = next((j for j in jurisdictions if j.jurisdiction == 'US'), None)
        if us_jurisdiction and us_jurisdiction.status in [RegulatoryStatus.UNCLEAR, RegulatoryStatus.RESTRICTED]:
            risks.append("US regulatory uncertainty poses significant market access risk")
        
        eu_jurisdiction = next((j for j in jurisdictions if j.jurisdiction == 'EU'), None)
        if eu_jurisdiction and eu_jurisdiction.status == RegulatoryStatus.UNCLEAR:
            risks.append("EU regulatory ambiguity may limit European market access")
        
        # Enforcement risk
        enforcement_count = sum(1 for j in jurisdictions if j.enforcement_actions)
        if enforcement_count >= 2:
            risks.append("Multiple enforcement actions suggest heightened regulatory scrutiny")
        
        # Compliance risk
        under_review_count = sum(1 for j in jurisdictions if j.compliance_status == ComplianceStatus.UNDER_REVIEW)
        if under_review_count >= 2:
            risks.append("Ongoing compliance reviews may result in operational restrictions")
        
        # Legislative risk
        pending_count = sum(len(j.pending_legislation) for j in jurisdictions)
        if pending_count >= 3:
            risks.append("Multiple pending regulations create policy uncertainty")
        
        return risks
    
    def _identify_regulatory_opportunities(
        self,
        jurisdictions: List[RegulatoryJurisdiction],
        events: List[RegulatoryEvent]
    ) -> List[str]:
        """Identify regulatory opportunities."""
        opportunities = []
        
        # Clarity opportunities
        positive_events = [e for e in events if e.impact_assessment == "positive"]
        if len(positive_events) >= 2:
            opportunities.append("Recent positive regulatory developments may improve market sentiment")
        
        # Market access opportunities
        clear_jurisdictions = [j for j in jurisdictions if j.status == RegulatoryStatus.CLEAR]
        if len(clear_jurisdictions) >= 3:
            opportunities.append("Clear regulatory status in multiple markets enables broader adoption")
        
        # Compliance advantage
        compliant_jurisdictions = [j for j in jurisdictions if j.compliance_status == ComplianceStatus.COMPLIANT]
        if len(compliant_jurisdictions) >= 3:
            opportunities.append("Strong compliance record provides competitive advantage")
        
        # First-mover advantage in regulated markets
        regulated_markets = [j for j in jurisdictions if j.status == RegulatoryStatus.CLEAR and j.compliance_status == ComplianceStatus.COMPLIANT]
        if len(regulated_markets) >= 2:
            opportunities.append("Early compliance in regulated markets creates barriers for competitors")
        
        return opportunities
    
    def _calculate_regulatory_score(self, jurisdictions: List[RegulatoryJurisdiction]) -> float:
        """Calculate overall regulatory score (0-100)."""
        try:
            total_score = 0
            jurisdiction_count = len(jurisdictions)
            
            if jurisdiction_count == 0:
                return 50  # Neutral score if no data
            
            for jurisdiction in jurisdictions:
                # Weight major markets more heavily
                weight = 2 if jurisdiction.jurisdiction in ['US', 'EU', 'China'] else 1
                
                # Score based on regulatory status
                if jurisdiction.status == RegulatoryStatus.CLEAR:
                    status_score = 100
                elif jurisdiction.status == RegulatoryStatus.PARTIALLY_CLEAR:
                    status_score = 70
                elif jurisdiction.status == RegulatoryStatus.UNCLEAR:
                    status_score = 40
                elif jurisdiction.status == RegulatoryStatus.RESTRICTED:
                    status_score = 20
                elif jurisdiction.status == RegulatoryStatus.BANNED:
                    status_score = 0
                else:
                    status_score = 50
                
                # Adjust for compliance status
                if jurisdiction.compliance_status == ComplianceStatus.COMPLIANT:
                    compliance_bonus = 10
                elif jurisdiction.compliance_status == ComplianceStatus.PARTIALLY_COMPLIANT:
                    compliance_bonus = 5
                elif jurisdiction.compliance_status == ComplianceStatus.NON_COMPLIANT:
                    compliance_bonus = -20
                else:
                    compliance_bonus = 0
                
                # Adjust for enforcement actions
                enforcement_penalty = len(jurisdiction.enforcement_actions) * 5
                
                jurisdiction_score = max(0, status_score + compliance_bonus - enforcement_penalty)
                total_score += jurisdiction_score * weight
            
            # Calculate weighted average
            total_weight = sum(2 if j.jurisdiction in ['US', 'EU', 'China'] else 1 for j in jurisdictions)
            return min(100, total_score / total_weight)
            
        except Exception as e:
            logger.warning(f"Error calculating regulatory score: {e}")
            return 50
    
    def _jurisdiction_to_dict(self, jurisdiction: RegulatoryJurisdiction) -> Dict:
        """Convert RegulatoryJurisdiction to dictionary."""
        return {
            'jurisdiction': jurisdiction.jurisdiction,
            'status': jurisdiction.status.value,
            'compliance_status': jurisdiction.compliance_status.value,
            'last_updated': jurisdiction.last_updated.isoformat(),
            'key_regulations': jurisdiction.key_regulations,
            'pending_legislation': jurisdiction.pending_legislation,
            'enforcement_actions': jurisdiction.enforcement_actions
        }
    
    def _event_to_dict(self, event: RegulatoryEvent) -> Dict:
        """Convert RegulatoryEvent to dictionary."""
        return {
            'date': event.date.isoformat(),
            'jurisdiction': event.jurisdiction,
            'event_type': event.event_type,
            'description': event.description,
            'impact_assessment': event.impact_assessment,
            'confidence_level': event.confidence_level
        }


async def get_regulatory_analysis(symbol: str, jurisdictions: Optional[List[str]] = None) -> str:
    """
    Get formatted regulatory analysis for crypto researchers.
    
    Args:
        symbol: Crypto symbol to analyze
        jurisdictions: Optional list of jurisdictions to focus on
        
    Returns:
        Formatted regulatory analysis string
    """
    try:
        analyzer = RegulatoryAnalyzer()
        results = await analyzer.analyze_regulatory_environment(symbol, jurisdictions)
        
        if 'error' in results:
            return f"❌ Error analyzing regulatory environment for {symbol}: {results['error']}"
        
        return _format_regulatory_report(results)
        
    except Exception as e:
        logger.error(f"Error getting regulatory analysis: {e}")
        return f"❌ Error analyzing regulatory environment for {symbol}: {str(e)}"


def _format_regulatory_report(results: Dict) -> str:
    """Format regulatory analysis into readable report."""
    try:
        symbol = results['symbol']
        jurisdictions = results['jurisdictions']
        events = results['recent_events']
        risk_assessment = results['risk_assessment']
        bull_points = results['bull_points']
        bear_points = results['bear_points']
        score = results['overall_regulatory_score']
        
        report = [
            f"# ⚖️ Regulatory Analysis: {symbol}",
            f"**Regulatory Score**: {score:.0f}/100",
            f"**Risk Level**: {risk_assessment['risk_level'].replace('_', ' ').title()}",
            "",
            "## 🌍 Jurisdiction Status"
        ]
        
        # Jurisdiction breakdown
        for jurisdiction in jurisdictions:
            status = jurisdiction['status'].replace('_', ' ').title()
            compliance = jurisdiction['compliance_status'].replace('_', ' ').title()
            
            status_emoji = {
                'Clear': '🟢',
                'Partially Clear': '🟡',
                'Unclear': '🟠', 
                'Restricted': '🔴',
                'Banned': '⛔',
                'Unknown': '⚪'
            }.get(status, '⚪')
            
            report.append(f"### {status_emoji} {jurisdiction['jurisdiction']}")
            report.append(f"- **Status**: {status}")
            report.append(f"- **Compliance**: {compliance}")
            
            if jurisdiction['key_regulations']:
                report.append(f"- **Key Regulations**: {', '.join(jurisdiction['key_regulations'])}")
            
            if jurisdiction['enforcement_actions']:
                report.append(f"- **Enforcement**: {', '.join(jurisdiction['enforcement_actions'])}")
            
            report.append("")
        
        # Recent events
        if events:
            report.extend([
                "## 📰 Recent Regulatory Events"
            ])
            for event in events[-3:]:  # Show last 3 events
                impact_emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(event['impact_assessment'], '⚪')
                date = datetime.fromisoformat(event['date']).strftime('%Y-%m-%d')
                report.append(f"- {impact_emoji} **{event['jurisdiction']}** ({date}): {event['description']}")
            report.append("")
        
        # Risk factors
        if risk_assessment['risk_factors']:
            report.extend([
                "## ⚠️ Key Regulatory Risks"
            ])
            for risk in risk_assessment['risk_factors']:
                report.append(f"- {risk}")
            report.append("")
        
        # Bull points
        if bull_points:
            report.extend([
                "## 🟢 Bullish Regulatory Points"
            ])
            for point in bull_points:
                report.append(f"- {point}")
            report.append("")
        
        # Bear points
        if bear_points:
            report.extend([
                "## 🔴 Bearish Regulatory Points"
            ])
            for point in bear_points:
                report.append(f"- {point}")
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error formatting regulatory report: {e}")
        return f"❌ Error formatting regulatory analysis: {str(e)}" 