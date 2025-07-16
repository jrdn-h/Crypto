"""
Fundamentals mapping between equity metrics and crypto tokenomics.

This module provides the mapping layer to convert traditional equity fundamentals
into crypto-equivalent tokenomics metrics, enabling cross-asset analysis.
"""

from typing import Dict, Optional, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

from ..base_interfaces import FundamentalsData, AssetClass, DataQuality


class TokenomicsCategory(str, Enum):
    """Categories of tokenomics data."""
    SUPPLY_METRICS = "supply_metrics"
    VALUATION = "valuation"
    REVENUE = "revenue"
    TREASURY = "treasury"
    GOVERNANCE = "governance"
    STAKING = "staking"
    UNLOCK_SCHEDULE = "unlock_schedule"


class TokenUnlockEvent(BaseModel):
    """Represents a token unlock event."""
    date: datetime
    amount: float
    recipient: str  # "team", "investors", "public", "treasury"
    percentage_of_total: float
    description: Optional[str] = None


class StakingMetrics(BaseModel):
    """Staking-related metrics."""
    total_staked: Optional[float] = None
    staking_ratio: Optional[float] = None  # % of supply staked
    annual_yield: Optional[float] = None  # APY
    validator_count: Optional[int] = None
    delegation_count: Optional[int] = None


class TreasuryMetrics(BaseModel):
    """Treasury and governance token holdings."""
    total_treasury_value_usd: Optional[float] = None
    native_token_holdings: Optional[float] = None
    stablecoin_holdings: Optional[float] = None
    other_token_holdings: Optional[Dict[str, float]] = None
    runway_months: Optional[float] = None  # At current burn rate


class ProtocolRevenue(BaseModel):
    """Protocol revenue and fee metrics."""
    daily_fees_usd: Optional[float] = None
    monthly_fees_usd: Optional[float] = None
    annual_fees_usd: Optional[float] = None
    fee_sources: Optional[Dict[str, float]] = None  # Trading, staking, etc.
    token_burn_rate: Optional[float] = None  # Tokens burned per period
    revenue_token_price_ratio: Optional[float] = None  # P/S equivalent


class CryptoFundamentals(FundamentalsData):
    """
    Extended fundamentals data for crypto assets.
    
    Inherits universal fields from FundamentalsData and adds crypto-specific metrics.
    """
    
    # Token supply metrics
    circulating_supply: Optional[float] = None
    max_supply: Optional[float] = None
    total_supply: Optional[float] = None
    fully_diluted_valuation: Optional[float] = None
    free_float_ratio: Optional[float] = None  # Excluding locked/vested tokens
    
    # Unlock schedule
    unlock_events: Optional[List[TokenUnlockEvent]] = None
    next_unlock_date: Optional[datetime] = None
    next_unlock_amount: Optional[float] = None
    
    # Protocol metrics (extended objects)
    protocol_revenue_details: Optional[ProtocolRevenue] = None
    treasury_metrics_details: Optional[TreasuryMetrics] = None
    staking_metrics_details: Optional[StakingMetrics] = None
    
    # Governance and decentralization
    governance_token: Optional[bool] = None
    voting_power_distribution: Optional[Dict[str, float]] = None
    
    # Network metrics (for L1s)
    active_addresses: Optional[int] = None
    transaction_count_24h: Optional[int] = None
    total_value_locked: Optional[float] = None  # For DeFi protocols
    
    # Token categorization
    categories: Optional[List[str]] = None  # ["DeFi", "Layer 1", "GameFi", etc.]
    use_cases: Optional[List[str]] = None


class FundamentalsMapper:
    """Maps equity fundamentals concepts to crypto tokenomics."""
    
    EQUITY_TO_CRYPTO_MAPPING = {
        # Core valuation metrics
        "market_cap": "circulating_market_cap",
        "enterprise_value": "fully_diluted_valuation",
        "shares_outstanding": "circulating_supply", 
        "float": "free_float_supply",
        
        # Revenue and profitability
        "revenue": "protocol_fees",
        "net_income": "protocol_revenue",
        "earnings_per_share": "revenue_per_token",
        
        # Valuation ratios
        "price_to_earnings": "price_to_fees_ratio",
        "price_to_sales": "price_to_revenue_ratio",
        "price_to_book": "price_to_treasury_ratio",
        
        # Cash and debt
        "cash_and_equivalents": "treasury_stablecoin_holdings",
        "total_debt": "token_emissions_liability",
        "debt_to_equity": "inflation_rate",
        
        # Returns to shareholders
        "dividend_yield": "staking_yield",
        "dividend_per_share": "rewards_per_token",
        "buyback_yield": "token_burn_rate",
        
        # Growth and efficiency
        "revenue_growth": "protocol_growth_rate",
        "roe": "return_on_treasury",
        "asset_turnover": "fee_efficiency_ratio",
    }
    
    @classmethod
    def map_equity_to_crypto(cls, equity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map equity fundamentals data to crypto tokenomics format."""
        crypto_data = {}
        
        for equity_field, crypto_field in cls.EQUITY_TO_CRYPTO_MAPPING.items():
            if equity_field in equity_data and equity_data[equity_field] is not None:
                crypto_data[crypto_field] = equity_data[equity_field]
        
        return crypto_data
    
    @classmethod
    def calculate_derived_metrics(cls, crypto_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived crypto metrics from basic data."""
        derived = {}
        
        # Calculate price-to-fees ratio (crypto P/E equivalent)
        if crypto_data.get("price") and crypto_data.get("annual_fees_usd"):
            total_fees = crypto_data["annual_fees_usd"]
            market_cap = crypto_data.get("circulating_market_cap")
            if market_cap and total_fees > 0:
                derived["price_to_fees_ratio"] = market_cap / total_fees
        
        # Calculate fully diluted valuation
        if crypto_data.get("price") and crypto_data.get("max_supply"):
            derived["fully_diluted_valuation"] = crypto_data["price"] * crypto_data["max_supply"]
        
        # Calculate inflation rate
        if crypto_data.get("circulating_supply") and crypto_data.get("max_supply"):
            circ = crypto_data["circulating_supply"]
            max_supply = crypto_data["max_supply"]
            if max_supply > 0:
                derived["supply_inflation_potential"] = (max_supply - circ) / circ
        
        # Calculate token velocity (if transaction volume available)
        if crypto_data.get("volume_24h") and crypto_data.get("market_cap"):
            derived["token_velocity"] = crypto_data["volume_24h"] / crypto_data["market_cap"]
        elif crypto_data.get("volume_24h") and crypto_data.get("circulating_market_cap"):
            derived["token_velocity"] = crypto_data["volume_24h"] / crypto_data["circulating_market_cap"]
        
        # Calculate staking ratio
        if crypto_data.get("total_staked") and crypto_data.get("circulating_supply"):
            derived["staking_ratio"] = crypto_data["total_staked"] / crypto_data["circulating_supply"]
        
        return derived
    
    @classmethod
    def create_crypto_fundamentals(
        cls,
        basic_data: Dict[str, Any],
        symbol: str,
        source: str = "unknown"
    ) -> CryptoFundamentals:
        """Create a CryptoFundamentals object from raw data."""
        
        # Map equity-style fields to crypto equivalents
        mapped_data = cls.map_equity_to_crypto(basic_data)
        
        # Calculate derived metrics
        derived_data = cls.calculate_derived_metrics({**basic_data, **mapped_data})
        
        # Combine all data
        all_data = {**basic_data, **mapped_data, **derived_data}
        
        # Create protocol revenue object
        protocol_revenue_details = None
        if any(key in all_data for key in ["daily_fees_usd", "monthly_fees_usd", "annual_fees_usd"]):
            protocol_revenue_details = ProtocolRevenue(
                daily_fees_usd=all_data.get("daily_fees_usd"),
                monthly_fees_usd=all_data.get("monthly_fees_usd"),
                annual_fees_usd=all_data.get("annual_fees_usd"),
                fee_sources=all_data.get("fee_sources"),
                token_burn_rate=all_data.get("token_burn_rate"),
                revenue_token_price_ratio=all_data.get("price_to_fees_ratio")
            )
        
        # Create staking metrics
        staking_metrics_details = None
        if any(key in all_data for key in ["total_staked", "staking_ratio", "annual_yield"]):
            staking_metrics_details = StakingMetrics(
                total_staked=all_data.get("total_staked"),
                staking_ratio=all_data.get("staking_ratio"),
                annual_yield=all_data.get("annual_yield"),
                validator_count=all_data.get("validator_count"),
                delegation_count=all_data.get("delegation_count")
            )
        
        # Create treasury metrics
        treasury_metrics_details = None
        if any(key in all_data for key in ["total_treasury_value_usd", "treasury_stablecoin_holdings"]):
            treasury_metrics_details = TreasuryMetrics(
                total_treasury_value_usd=all_data.get("total_treasury_value_usd"),
                native_token_holdings=all_data.get("native_token_holdings"),
                stablecoin_holdings=all_data.get("treasury_stablecoin_holdings"),
                other_token_holdings=all_data.get("other_token_holdings"),
                runway_months=all_data.get("runway_months")
            )
        
        # Create the crypto fundamentals object
        crypto_fundamentals = CryptoFundamentals(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            as_of_date=all_data.get("timestamp", datetime.now()),
            data_quality=all_data.get("data_quality", DataQuality.MEDIUM),
            
            # Universal fundamentals fields
            market_cap=all_data.get("market_cap") or all_data.get("circulating_market_cap"),
            data_sources=[source],
            
            # Base class crypto fields
            circulating_supply=all_data.get("circulating_supply"),
            max_supply=all_data.get("max_supply"),
            total_supply=all_data.get("total_supply"),
            staking_yield=staking_metrics_details.annual_yield if staking_metrics_details else None,
            protocol_revenue=protocol_revenue_details.annual_fees_usd if protocol_revenue_details else None,
            treasury_value=treasury_metrics_details.total_treasury_value_usd if treasury_metrics_details else None,
            
            # Extended crypto-specific fields
            fully_diluted_valuation=all_data.get("fully_diluted_valuation"),
            free_float_ratio=all_data.get("free_float_ratio"),
            
            protocol_revenue_details=protocol_revenue_details,
            staking_metrics_details=staking_metrics_details,
            treasury_metrics_details=treasury_metrics_details,
            
            governance_token=all_data.get("governance_token"),
            voting_power_distribution=all_data.get("voting_power_distribution"),
            
            active_addresses=all_data.get("active_addresses"),
            transaction_count_24h=all_data.get("transaction_count_24h"),
            total_value_locked=all_data.get("total_value_locked"),
            
            categories=all_data.get("categories"),
            use_cases=all_data.get("use_cases")
        )
        
        return crypto_fundamentals


def get_equity_crypto_field_mapping() -> Dict[str, str]:
    """Get the complete mapping of equity fields to crypto equivalents."""
    return FundamentalsMapper.EQUITY_TO_CRYPTO_MAPPING.copy()


def normalize_tokenomics_data(raw_data: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Normalize tokenomics data from different sources to a standard format."""
    normalized = {}
    
    # Source-specific normalization
    if source.lower() == "coingecko":
        # CoinGecko field mappings
        field_map = {
            "market_data.current_price.usd": "price",
            "market_data.market_cap.usd": "market_cap",
            "market_data.total_volume.usd": "volume_24h",
            "market_data.circulating_supply": "circulating_supply",
            "market_data.max_supply": "max_supply",
            "market_data.total_supply": "total_supply",
            "market_data.fully_diluted_valuation.usd": "fully_diluted_valuation",
            "categories": "categories"
        }
        
        for source_field, target_field in field_map.items():
            # Handle nested field access
            value = raw_data
            for key in source_field.split("."):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            if value is not None:
                normalized[target_field] = value
    
    elif source.lower() == "cryptocompare":
        # CryptoCompare field mappings
        field_map = {
            "RAW.USD.PRICE": "price",
            "RAW.USD.MKTCAP": "market_cap",
            "RAW.USD.TOTALVOLUME24HTO": "volume_24h",
            "DISPLAY.USD.SUPPLY": "circulating_supply"
        }
        
        for source_field, target_field in field_map.items():
            value = raw_data
            for key in source_field.split("."):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            if value is not None:
                normalized[target_field] = value
    
    elif source.lower() == "binance":
        # Binance has limited fundamentals data
        field_map = {
            "price": "price",
            "volume": "volume_24h"
        }
        
        for source_field, target_field in field_map.items():
            if source_field in raw_data:
                normalized[target_field] = raw_data[source_field]
    
    # Add source metadata
    normalized["data_source"] = source
    normalized["timestamp"] = datetime.now()
    
    return normalized 