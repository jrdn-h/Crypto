"""
Tokenomics analysis module for crypto assets.

This module provides comprehensive tokenomics analysis including supply mechanics,
vesting schedules, inflation/deflation analysis, and distribution assessment.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TokenSupplyType(str, Enum):
    """Types of token supply models."""
    FIXED = "fixed"
    INFLATIONARY = "inflationary"
    DEFLATIONARY = "deflationary"
    ELASTIC = "elastic"
    UNKNOWN = "unknown"


class TokenDistributionType(str, Enum):
    """Types of token distribution."""
    FAIR_LAUNCH = "fair_launch"
    ICO = "ico"
    IDO = "ido"
    AIRDROP = "airdrop"
    MINING = "mining"
    PREMINE = "premine"
    UNKNOWN = "unknown"


@dataclass
class TokenSupplyMetrics:
    """Token supply and inflation metrics."""
    total_supply: Optional[float]
    circulating_supply: Optional[float]
    max_supply: Optional[float]
    inflation_rate_annual: Optional[float]
    supply_type: TokenSupplyType
    burn_rate: Optional[float]
    emission_schedule: Optional[str]
    supply_growth_rate: Optional[float]


@dataclass
class VestingSchedule:
    """Token vesting and unlock schedule."""
    total_vested_tokens: Optional[float]
    next_unlock_date: Optional[datetime]
    next_unlock_amount: Optional[float]
    vesting_cliff_months: Optional[int]
    vesting_period_months: Optional[int]
    unlock_percentage_remaining: Optional[float]
    monthly_unlock_rate: Optional[float]


@dataclass
class TokenDistribution:
    """Token distribution and allocation."""
    team_allocation_pct: Optional[float]
    investors_allocation_pct: Optional[float]
    public_allocation_pct: Optional[float]
    treasury_allocation_pct: Optional[float]
    ecosystem_allocation_pct: Optional[float]
    staking_rewards_pct: Optional[float]
    distribution_type: TokenDistributionType
    insider_concentration: Optional[float]


@dataclass
class TokenUtility:
    """Token utility and value accrual mechanisms."""
    governance_voting: bool
    staking_rewards: bool
    fee_reduction: bool
    protocol_revenue_share: bool
    burning_mechanism: bool
    collateral_use: bool
    utility_score: Optional[float]
    value_accrual_mechanisms: List[str]


class TokenomicsAnalyzer:
    """Comprehensive tokenomics analysis for crypto assets."""
    
    def __init__(self):
        """Initialize tokenomics analyzer."""
        pass
    
    async def analyze_tokenomics(
        self,
        symbol: str,
        token_data: Optional[Dict] = None
    ) -> Dict[str, Union[Dict, str, float]]:
        """
        Perform comprehensive tokenomics analysis.
        
        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH', 'SOL')
            token_data: Optional token data dictionary
            
        Returns:
            Dictionary containing tokenomics analysis
        """
        try:
            # Fetch or use provided token data
            if token_data is None:
                token_data = await self._fetch_token_data(symbol)
            
            # Analyze different aspects of tokenomics
            supply_metrics = self._analyze_supply_mechanics(token_data, symbol)
            vesting_schedule = self._analyze_vesting_schedule(token_data, symbol)
            distribution = self._analyze_token_distribution(token_data, symbol)
            utility = self._analyze_token_utility(token_data, symbol)
            
            # Generate risk assessment
            tokenomics_risks = self._assess_tokenomics_risks(
                supply_metrics, vesting_schedule, distribution, utility
            )
            
            # Generate bull/bear analysis points
            bull_points = self._generate_bull_points(
                supply_metrics, vesting_schedule, distribution, utility
            )
            bear_points = self._generate_bear_points(
                supply_metrics, vesting_schedule, distribution, utility
            )
            
            return {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'supply_metrics': self._metrics_to_dict(supply_metrics),
                'vesting_schedule': self._vesting_to_dict(vesting_schedule),
                'distribution': self._distribution_to_dict(distribution),
                'utility': self._utility_to_dict(utility),
                'tokenomics_risks': tokenomics_risks,
                'bull_points': bull_points,
                'bear_points': bear_points,
                'overall_score': self._calculate_overall_score(
                    supply_metrics, vesting_schedule, distribution, utility
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tokenomics for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def _fetch_token_data(self, symbol: str) -> Dict:
        """Fetch token data from various sources."""
        # Mock token data - in real implementation, this would fetch from APIs
        mock_data = {
            'BTC': {
                'total_supply': 21_000_000,
                'circulating_supply': 19_800_000,
                'max_supply': 21_000_000,
                'inflation_rate': 0.65,  # Current Bitcoin inflation
                'distribution_type': 'mining',
                'governance': False,
                'staking': False,
                'burn_mechanism': False
            },
            'ETH': {
                'total_supply': 120_300_000,
                'circulating_supply': 120_300_000,
                'max_supply': None,
                'inflation_rate': -0.2,  # Deflationary post-merge
                'distribution_type': 'premine',
                'governance': True,
                'staking': True,
                'burn_mechanism': True
            },
            'SOL': {
                'total_supply': 580_000_000,
                'circulating_supply': 470_000_000,
                'max_supply': None,
                'inflation_rate': 5.2,
                'distribution_type': 'ico',
                'governance': True,
                'staking': True,
                'burn_mechanism': False,
                'team_allocation': 0.125,  # 12.5%
                'investors_allocation': 0.178,  # 17.8%
                'public_allocation': 0.697  # 69.7%
            }
        }.get(symbol, {})
        
        return mock_data
    
    def _analyze_supply_mechanics(self, token_data: Dict, symbol: str) -> TokenSupplyMetrics:
        """Analyze token supply mechanics."""
        try:
            total_supply = token_data.get('total_supply')
            circulating_supply = token_data.get('circulating_supply')
            max_supply = token_data.get('max_supply')
            inflation_rate = token_data.get('inflation_rate', 0)
            
            # Determine supply type
            if max_supply and total_supply and total_supply >= max_supply:
                supply_type = TokenSupplyType.FIXED
            elif inflation_rate < 0:
                supply_type = TokenSupplyType.DEFLATIONARY
            elif inflation_rate > 0:
                supply_type = TokenSupplyType.INFLATIONARY
            else:
                supply_type = TokenSupplyType.UNKNOWN
            
            # Calculate supply growth rate
            supply_growth_rate = None
            if circulating_supply and total_supply:
                supply_growth_rate = (total_supply - circulating_supply) / circulating_supply
            
            return TokenSupplyMetrics(
                total_supply=total_supply,
                circulating_supply=circulating_supply,
                max_supply=max_supply,
                inflation_rate_annual=inflation_rate,
                supply_type=supply_type,
                burn_rate=token_data.get('burn_rate'),
                emission_schedule=token_data.get('emission_schedule'),
                supply_growth_rate=supply_growth_rate
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing supply mechanics: {e}")
            return TokenSupplyMetrics(
                total_supply=None, circulating_supply=None, max_supply=None,
                inflation_rate_annual=None, supply_type=TokenSupplyType.UNKNOWN,
                burn_rate=None, emission_schedule=None, supply_growth_rate=None
            )
    
    def _analyze_vesting_schedule(self, token_data: Dict, symbol: str) -> VestingSchedule:
        """Analyze token vesting and unlock schedules."""
        try:
            # Mock vesting analysis - would be fetched from vesting APIs
            if symbol == 'SOL':
                return VestingSchedule(
                    total_vested_tokens=110_000_000,  # Tokens still vesting
                    next_unlock_date=datetime.now() + timedelta(days=30),
                    next_unlock_amount=5_000_000,
                    vesting_cliff_months=12,
                    vesting_period_months=48,
                    unlock_percentage_remaining=0.19,  # 19% still locked
                    monthly_unlock_rate=0.02  # 2% per month
                )
            else:
                return VestingSchedule(
                    total_vested_tokens=None,
                    next_unlock_date=None,
                    next_unlock_amount=None,
                    vesting_cliff_months=None,
                    vesting_period_months=None,
                    unlock_percentage_remaining=None,
                    monthly_unlock_rate=None
                )
                
        except Exception as e:
            logger.warning(f"Error analyzing vesting schedule: {e}")
            return VestingSchedule(
                total_vested_tokens=None, next_unlock_date=None, next_unlock_amount=None,
                vesting_cliff_months=None, vesting_period_months=None,
                unlock_percentage_remaining=None, monthly_unlock_rate=None
            )
    
    def _analyze_token_distribution(self, token_data: Dict, symbol: str) -> TokenDistribution:
        """Analyze token distribution and allocation."""
        try:
            team_pct = token_data.get('team_allocation', 0)
            investors_pct = token_data.get('investors_allocation', 0)
            public_pct = token_data.get('public_allocation', 0)
            
            # Calculate insider concentration
            insider_concentration = team_pct + investors_pct
            
            # Determine distribution type
            dist_type_str = token_data.get('distribution_type', 'unknown')
            distribution_type = TokenDistributionType(dist_type_str)
            
            return TokenDistribution(
                team_allocation_pct=team_pct,
                investors_allocation_pct=investors_pct,
                public_allocation_pct=public_pct,
                treasury_allocation_pct=token_data.get('treasury_allocation'),
                ecosystem_allocation_pct=token_data.get('ecosystem_allocation'),
                staking_rewards_pct=token_data.get('staking_rewards'),
                distribution_type=distribution_type,
                insider_concentration=insider_concentration
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing token distribution: {e}")
            return TokenDistribution(
                team_allocation_pct=None, investors_allocation_pct=None,
                public_allocation_pct=None, treasury_allocation_pct=None,
                ecosystem_allocation_pct=None, staking_rewards_pct=None,
                distribution_type=TokenDistributionType.UNKNOWN,
                insider_concentration=None
            )
    
    def _analyze_token_utility(self, token_data: Dict, symbol: str) -> TokenUtility:
        """Analyze token utility and value accrual mechanisms."""
        try:
            governance = token_data.get('governance', False)
            staking = token_data.get('staking', False)
            fee_reduction = token_data.get('fee_reduction', False)
            revenue_share = token_data.get('revenue_share', False)
            burn_mechanism = token_data.get('burn_mechanism', False)
            collateral = token_data.get('collateral_use', False)
            
            # Build list of value accrual mechanisms
            mechanisms = []
            if governance:
                mechanisms.append("Governance voting rights")
            if staking:
                mechanisms.append("Staking rewards")
            if fee_reduction:
                mechanisms.append("Fee discounts")
            if revenue_share:
                mechanisms.append("Protocol revenue sharing")
            if burn_mechanism:
                mechanisms.append("Token burning")
            if collateral:
                mechanisms.append("Collateral usage")
            
            # Calculate utility score (0-100)
            utility_score = (
                (governance * 20) + 
                (staking * 25) + 
                (fee_reduction * 15) + 
                (revenue_share * 20) + 
                (burn_mechanism * 15) + 
                (collateral * 5)
            )
            
            return TokenUtility(
                governance_voting=governance,
                staking_rewards=staking,
                fee_reduction=fee_reduction,
                protocol_revenue_share=revenue_share,
                burning_mechanism=burn_mechanism,
                collateral_use=collateral,
                utility_score=utility_score,
                value_accrual_mechanisms=mechanisms
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing token utility: {e}")
            return TokenUtility(
                governance_voting=False, staking_rewards=False, fee_reduction=False,
                protocol_revenue_share=False, burning_mechanism=False, collateral_use=False,
                utility_score=0, value_accrual_mechanisms=[]
            )
    
    def _assess_tokenomics_risks(
        self,
        supply: TokenSupplyMetrics,
        vesting: VestingSchedule,
        distribution: TokenDistribution,
        utility: TokenUtility
    ) -> Dict[str, Union[str, float, List[str]]]:
        """Assess tokenomics-related risks."""
        risks = []
        risk_score = 0
        
        # Supply-related risks
        if supply.inflation_rate_annual and supply.inflation_rate_annual > 10:
            risks.append("High inflation rate may depress token price")
            risk_score += 25
        
        if supply.supply_type == TokenSupplyType.INFLATIONARY and not supply.burn_rate:
            risks.append("Inflationary supply without offsetting burn mechanism")
            risk_score += 20
        
        # Vesting-related risks
        if vesting.unlock_percentage_remaining and vesting.unlock_percentage_remaining > 0.3:
            risks.append("Significant token unlocks pending (>30% of supply)")
            risk_score += 30
        
        if vesting.monthly_unlock_rate and vesting.monthly_unlock_rate > 0.05:
            risks.append("High monthly unlock rate may create selling pressure")
            risk_score += 20
        
        # Distribution-related risks
        if distribution.insider_concentration and distribution.insider_concentration > 0.5:
            risks.append("High insider concentration (>50%) creates centralization risk")
            risk_score += 35
        
        # Utility-related risks
        if utility.utility_score and utility.utility_score < 20:
            risks.append("Limited token utility may reduce long-term value")
            risk_score += 25
        
        # Overall risk level
        if risk_score >= 80:
            risk_level = "very_high"
        elif risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        elif risk_score >= 20:
            risk_level = "low"
        else:
            risk_level = "very_low"
        
        return {
            'risk_factors': risks,
            'risk_score': min(risk_score, 100),
            'risk_level': risk_level,
            'major_concerns': [risk for risk in risks if any(word in risk.lower() 
                             for word in ['high', 'significant', 'limited'])]
        }
    
    def _generate_bull_points(
        self,
        supply: TokenSupplyMetrics,
        vesting: VestingSchedule,
        distribution: TokenDistribution,
        utility: TokenUtility
    ) -> List[str]:
        """Generate bullish tokenomics points."""
        bull_points = []
        
        # Supply-related bull points
        if supply.supply_type == TokenSupplyType.DEFLATIONARY:
            bull_points.append("Deflationary tokenomics create scarcity and potential price appreciation")
        
        if supply.supply_type == TokenSupplyType.FIXED:
            bull_points.append("Fixed supply cap provides predictable monetary policy")
        
        if supply.burn_rate and supply.burn_rate > 0:
            bull_points.append("Token burning mechanism reduces circulating supply over time")
        
        # Vesting-related bull points
        if vesting.unlock_percentage_remaining and vesting.unlock_percentage_remaining < 0.1:
            bull_points.append("Minimal remaining token unlocks reduce future selling pressure")
        
        # Distribution-related bull points
        if distribution.public_allocation_pct and distribution.public_allocation_pct > 0.6:
            bull_points.append("Strong public allocation promotes decentralization")
        
        if distribution.insider_concentration and distribution.insider_concentration < 0.3:
            bull_points.append("Low insider concentration reduces centralization risks")
        
        # Utility-related bull points
        if utility.staking_rewards:
            bull_points.append("Staking rewards provide yield opportunity for long-term holders")
        
        if utility.burning_mechanism:
            bull_points.append("Fee burning creates deflationary pressure on token supply")
        
        if utility.governance_voting:
            bull_points.append("Governance rights give token holders protocol control")
        
        if utility.utility_score and utility.utility_score > 60:
            bull_points.append("Strong token utility across multiple use cases")
        
        return bull_points
    
    def _generate_bear_points(
        self,
        supply: TokenSupplyMetrics,
        vesting: VestingSchedule,
        distribution: TokenDistribution,
        utility: TokenUtility
    ) -> List[str]:
        """Generate bearish tokenomics points."""
        bear_points = []
        
        # Supply-related bear points
        if supply.inflation_rate_annual and supply.inflation_rate_annual > 5:
            bear_points.append(f"High inflation rate ({supply.inflation_rate_annual:.1f}%) creates constant selling pressure")
        
        if supply.supply_type == TokenSupplyType.INFLATIONARY and not supply.burn_rate:
            bear_points.append("Inflationary supply without burn mechanism dilutes existing holders")
        
        # Vesting-related bear points
        if vesting.unlock_percentage_remaining and vesting.unlock_percentage_remaining > 0.2:
            bear_points.append(f"Significant token unlocks pending ({vesting.unlock_percentage_remaining:.1%}) may depress price")
        
        if vesting.next_unlock_amount and vesting.total_vested_tokens:
            pct_unlock = vesting.next_unlock_amount / vesting.total_vested_tokens
            if pct_unlock > 0.05:
                bear_points.append("Large upcoming token unlock may create selling pressure")
        
        # Distribution-related bear points
        if distribution.insider_concentration and distribution.insider_concentration > 0.4:
            bear_points.append(f"High insider concentration ({distribution.insider_concentration:.1%}) creates centralization risk")
        
        if distribution.team_allocation_pct and distribution.team_allocation_pct > 0.2:
            bear_points.append("Large team allocation may incentivize early selling")
        
        # Utility-related bear points
        if utility.utility_score and utility.utility_score < 30:
            bear_points.append("Limited token utility may not support long-term value")
        
        if not utility.staking_rewards and not utility.burning_mechanism:
            bear_points.append("Lack of yield opportunities or deflationary mechanisms")
        
        if len(utility.value_accrual_mechanisms) < 2:
            bear_points.append("Few value accrual mechanisms limit token appreciation potential")
        
        return bear_points
    
    def _calculate_overall_score(
        self,
        supply: TokenSupplyMetrics,
        vesting: VestingSchedule,
        distribution: TokenDistribution,
        utility: TokenUtility
    ) -> float:
        """Calculate overall tokenomics score (0-100)."""
        score = 0
        
        # Supply score (25 points)
        if supply.supply_type == TokenSupplyType.DEFLATIONARY:
            score += 25
        elif supply.supply_type == TokenSupplyType.FIXED:
            score += 20
        elif supply.inflation_rate_annual and supply.inflation_rate_annual < 5:
            score += 15
        elif supply.inflation_rate_annual and supply.inflation_rate_annual < 10:
            score += 10
        
        # Vesting score (25 points)
        if vesting.unlock_percentage_remaining:
            if vesting.unlock_percentage_remaining < 0.1:
                score += 25
            elif vesting.unlock_percentage_remaining < 0.2:
                score += 20
            elif vesting.unlock_percentage_remaining < 0.3:
                score += 15
            else:
                score += 5
        else:
            score += 20  # No vesting concerns
        
        # Distribution score (25 points)
        if distribution.insider_concentration:
            if distribution.insider_concentration < 0.2:
                score += 25
            elif distribution.insider_concentration < 0.3:
                score += 20
            elif distribution.insider_concentration < 0.4:
                score += 15
            elif distribution.insider_concentration < 0.5:
                score += 10
            else:
                score += 5
        else:
            score += 15  # Unknown distribution
        
        # Utility score (25 points)
        if utility.utility_score:
            score += min(utility.utility_score / 4, 25)  # Scale to 25 max
        
        return min(score, 100)
    
    def _metrics_to_dict(self, metrics: TokenSupplyMetrics) -> Dict:
        """Convert TokenSupplyMetrics to dictionary."""
        return {
            'total_supply': metrics.total_supply,
            'circulating_supply': metrics.circulating_supply,
            'max_supply': metrics.max_supply,
            'inflation_rate_annual': metrics.inflation_rate_annual,
            'supply_type': metrics.supply_type.value,
            'burn_rate': metrics.burn_rate,
            'emission_schedule': metrics.emission_schedule,
            'supply_growth_rate': metrics.supply_growth_rate
        }
    
    def _vesting_to_dict(self, vesting: VestingSchedule) -> Dict:
        """Convert VestingSchedule to dictionary."""
        return {
            'total_vested_tokens': vesting.total_vested_tokens,
            'next_unlock_date': vesting.next_unlock_date.isoformat() if vesting.next_unlock_date else None,
            'next_unlock_amount': vesting.next_unlock_amount,
            'vesting_cliff_months': vesting.vesting_cliff_months,
            'vesting_period_months': vesting.vesting_period_months,
            'unlock_percentage_remaining': vesting.unlock_percentage_remaining,
            'monthly_unlock_rate': vesting.monthly_unlock_rate
        }
    
    def _distribution_to_dict(self, distribution: TokenDistribution) -> Dict:
        """Convert TokenDistribution to dictionary."""
        return {
            'team_allocation_pct': distribution.team_allocation_pct,
            'investors_allocation_pct': distribution.investors_allocation_pct,
            'public_allocation_pct': distribution.public_allocation_pct,
            'treasury_allocation_pct': distribution.treasury_allocation_pct,
            'ecosystem_allocation_pct': distribution.ecosystem_allocation_pct,
            'staking_rewards_pct': distribution.staking_rewards_pct,
            'distribution_type': distribution.distribution_type.value,
            'insider_concentration': distribution.insider_concentration
        }
    
    def _utility_to_dict(self, utility: TokenUtility) -> Dict:
        """Convert TokenUtility to dictionary."""
        return {
            'governance_voting': utility.governance_voting,
            'staking_rewards': utility.staking_rewards,
            'fee_reduction': utility.fee_reduction,
            'protocol_revenue_share': utility.protocol_revenue_share,
            'burning_mechanism': utility.burning_mechanism,
            'collateral_use': utility.collateral_use,
            'utility_score': utility.utility_score,
            'value_accrual_mechanisms': utility.value_accrual_mechanisms
        }


async def get_tokenomics_analysis(symbol: str) -> str:
    """
    Get formatted tokenomics analysis for crypto researchers.
    
    Args:
        symbol: Crypto symbol to analyze
        
    Returns:
        Formatted tokenomics analysis string
    """
    try:
        analyzer = TokenomicsAnalyzer()
        results = await analyzer.analyze_tokenomics(symbol)
        
        if 'error' in results:
            return f"❌ Error analyzing tokenomics for {symbol}: {results['error']}"
        
        return _format_tokenomics_report(results)
        
    except Exception as e:
        logger.error(f"Error getting tokenomics analysis: {e}")
        return f"❌ Error analyzing tokenomics for {symbol}: {str(e)}"


def _format_tokenomics_report(results: Dict) -> str:
    """Format tokenomics analysis into readable report."""
    try:
        symbol = results['symbol']
        supply = results['supply_metrics']
        vesting = results['vesting_schedule']
        distribution = results['distribution']
        utility = results['utility']
        risks = results['tokenomics_risks']
        bull_points = results['bull_points']
        bear_points = results['bear_points']
        overall_score = results['overall_score']
        
        report = [
            f"# 🪙 Tokenomics Analysis: {symbol}",
            f"**Overall Score**: {overall_score:.0f}/100",
            f"**Risk Level**: {risks['risk_level'].replace('_', ' ').title()}",
            "",
            "## 📊 Supply Mechanics",
            f"- **Total Supply**: {supply['total_supply']:,}" if supply['total_supply'] else "- **Total Supply**: Unknown",
            f"- **Circulating Supply**: {supply['circulating_supply']:,}" if supply['circulating_supply'] else "- **Circulating Supply**: Unknown",
            f"- **Max Supply**: {supply['max_supply']:,}" if supply['max_supply'] else "- **Max Supply**: Unlimited",
            f"- **Supply Type**: {supply['supply_type'].replace('_', ' ').title()}",
            f"- **Annual Inflation**: {supply['inflation_rate_annual']:+.2f}%" if supply['inflation_rate_annual'] is not None else "- **Annual Inflation**: Unknown",
            ""
        ]
        
        # Vesting information
        if vesting['unlock_percentage_remaining']:
            report.extend([
                "## 🔒 Vesting Schedule",
                f"- **Tokens Still Vesting**: {vesting['unlock_percentage_remaining']:.1%} of supply",
                f"- **Next Unlock**: {vesting['next_unlock_amount']:,} tokens" if vesting['next_unlock_amount'] else "",
                f"- **Monthly Unlock Rate**: {vesting['monthly_unlock_rate']:.1%}" if vesting['monthly_unlock_rate'] else "",
                ""
            ])
        
        # Distribution
        if distribution['insider_concentration']:
            report.extend([
                "## 🏢 Token Distribution",
                f"- **Team Allocation**: {distribution['team_allocation_pct']:.1%}" if distribution['team_allocation_pct'] else "",
                f"- **Investor Allocation**: {distribution['investors_allocation_pct']:.1%}" if distribution['investors_allocation_pct'] else "",
                f"- **Public Allocation**: {distribution['public_allocation_pct']:.1%}" if distribution['public_allocation_pct'] else "",
                f"- **Insider Concentration**: {distribution['insider_concentration']:.1%}",
                ""
            ])
        
        # Utility
        if utility['value_accrual_mechanisms']:
            report.extend([
                "## ⚡ Token Utility",
                f"- **Utility Score**: {utility['utility_score']:.0f}/100",
                "- **Value Accrual Mechanisms**:"
            ])
            for mechanism in utility['value_accrual_mechanisms']:
                report.append(f"  - {mechanism}")
            report.append("")
        
        # Risk factors
        if risks['risk_factors']:
            report.extend([
                "## ⚠️ Risk Factors"
            ])
            for risk in risks['risk_factors']:
                report.append(f"- {risk}")
            report.append("")
        
        # Bull points
        if bull_points:
            report.extend([
                "## 🟢 Bullish Tokenomics Points"
            ])
            for point in bull_points:
                report.append(f"- {point}")
            report.append("")
        
        # Bear points
        if bear_points:
            report.extend([
                "## 🔴 Bearish Tokenomics Points"
            ])
            for point in bear_points:
                report.append(f"- {point}")
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error formatting tokenomics report: {e}")
        return f"❌ Error formatting tokenomics analysis: {str(e)}" 