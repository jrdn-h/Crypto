# TradingAgents Crypto Extension

<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center">
  🚀 <strong>Comprehensive Crypto Trading Framework</strong> 🚀<br>
  24/7 Markets • Advanced Derivatives • Multi-Exchange Support • AI-Powered Analysis
</div>

---

## Overview

The TradingAgents Crypto Extension brings the power of multi-agent AI trading to cryptocurrency markets. Building on the proven TradingAgents framework, this extension provides:

- **24/7 Market Analysis** - No market hours, continuous trading
- **Advanced Crypto Features** - Perpetual futures, funding rates, on-chain analysis
- **Multi-Exchange Support** - Binance, Coinbase, Kraken, Hyperliquid, and more
- **Risk Management** - Crypto-specific portfolio optimization and risk controls
- **Cost-Optimized** - Free-tier data providers with premium upgrades available

## Quick Start

### 1. Basic Setup

```bash
# Clone and setup
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
conda create -n tradingagents python=3.13
conda activate tradingagents
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp example.env .env
# Edit .env and add your API keys (see Configuration section below)
```

### 3. Enable Crypto Support

```bash
# Set in your .env file:
DEFAULT_ASSET_CLASS=crypto
CRYPTO_SUPPORT=true
OPENAI_API_KEY=your_openai_key_here
```

### 4. Run Your First Crypto Analysis

```bash
# Interactive CLI
python -m cli.main

# Direct command line
python -m cli.main analyze --asset-class crypto --ticker BTC/USDT
```

## Features Overview

### 🔄 24/7 Market Operations
- Continuous market analysis without market hours
- Real-time price feeds and order execution
- Global crypto market coverage

### 📊 Advanced Technical Analysis
- Crypto-specific indicators (funding rates, perp basis, realized volatility)
- On-chain whale tracking and exchange flow analysis
- Multi-timeframe momentum analysis (1h, 4h, 24h)

### 🏛️ Institutional-Grade Fundamentals
- Tokenomics analysis (supply mechanics, vesting schedules)
- Multi-jurisdiction regulatory monitoring
- Protocol revenue and treasury analysis

### 📰 Comprehensive Sentiment Analysis
- Multi-source news aggregation (CryptoPanic, CoinDesk)
- Social sentiment tracking (Reddit, Twitter/X)
- Crypto-specific sentiment scoring

### 💼 Professional Trading Infrastructure
- Multi-exchange execution (spot and perpetual futures)
- Advanced order types (market, limit, stop, bracket)
- Paper trading with realistic simulation

### ⚠️ Advanced Risk Management
- 24/7 portfolio monitoring and alerts
- Dynamic leverage controls and liquidation tracking
- Funding PnL optimization across exchanges

## Configuration

### Required API Keys

**Minimum Setup (Free Tier):**
```bash
# In your .env file:
OPENAI_API_KEY=your_openai_key_here
DEFAULT_ASSET_CLASS=crypto
CRYPTO_SUPPORT=true
```

**Enhanced Setup (Recommended):**
```bash
# Core LLM
OPENAI_API_KEY=your_openai_key_here

# Crypto Data (Optional - Free tiers available)
COINGECKO_API_KEY=your_coingecko_key_here
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key_here

# Exchange Trading (Optional - for live trading)
BINANCE_API_KEY=your_binance_key_here
BINANCE_SECRET_KEY=your_binance_secret_here
```

### Data Provider Tiers

**Free Tier (No API Keys Required):**
- CoinGecko public endpoints (50 req/min)
- Binance public data (1200 req/min)
- CryptoPanic RSS feeds
- CoinDesk news feeds

**Premium Tier (API Keys):**
- CoinGecko Pro (10,000 req/min)
- CryptoCompare (100,000 req/min)
- Twitter/X sentiment analysis
- Exchange trading APIs

## CLI Usage

### Basic Commands

```bash
# Interactive mode
python -m cli.main

# Crypto analysis
python -m cli.main analyze --asset-class crypto --ticker BTC/USDT

# Provider status check
python -m cli.main providers --asset-class crypto --status

# Configuration management
python -m cli.main config --show --validate
```

### Crypto-Specific Commands

```bash
# Crypto market analysis
python -m cli.main crypto analyze --ticker BTC/USDT

# Perpetual futures analysis
python -m cli.main crypto analyze --ticker BTC-PERP

# Risk assessment
python -m cli.main crypto risk --ticker ETH-PERP

# Funding rate analysis
python -m cli.main crypto funding --ticker BTC-PERP

# Multi-exchange trading
python -m cli.main crypto trade --ticker ETH/USDT --exchange binance
```

### Command Arguments

```bash
# Asset class selection
--asset-class crypto|equity

# Symbols and tickers
--ticker BTC/USDT        # Spot pairs
--ticker BTC-PERP        # Perpetual futures
--ticker ETH/USDT,SOL/USDT  # Multiple symbols

# Provider management
--provider-preset free|premium|enterprise
--cost-preset cheap|balanced|premium

# Configuration
--config path/to/config.json
--non-interactive        # For automation/scripting
```

## Python API Usage

### Basic Crypto Analysis

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Configure for crypto
config = DEFAULT_CONFIG.copy()
config.update({
    "asset_class": "crypto",
    "crypto_providers": ["coingecko", "binance", "cryptocompare"],
    "execution_mode": "paper"
})

# Initialize framework
ta = TradingAgentsGraph(debug=True, config=config)

# Run analysis
_, decision = ta.propagate("BTC/USDT", "2024-12-15")
print(f"Trading Decision: {decision}")
```

### Advanced Configuration

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Advanced crypto configuration
config = DEFAULT_CONFIG.copy()
config.update({
    "asset_class": "crypto",
    "crypto_providers": ["coingecko", "binance", "cryptocompare"],
    "crypto_quote_currency": "USDT",
    "use_perps": True,
    "execution_mode": "ccxt",  # Live trading
    "risk.max_notional_pct": 0.10,
    "risk.max_leverage": 3,
    "deep_think_llm": "gpt-4o",
    "quick_think_llm": "gpt-4o-mini",
    "max_debate_rounds": 2,
    "online_tools": True
})

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("ETH-PERP", "2024-12-15")
```

### Direct Provider Usage

```python
from tradingagents.dataflows.crypto import (
    CoinGeckoClient, BinancePublicClient, CryptoPaperBroker
)

# Market data client
client = CoinGeckoClient()
price_data = await client.get_price("bitcoin", "2024-12-15")

# Paper trading broker
broker = CryptoPaperBroker(initial_balance={"USDT": 10000})
order = await broker.create_order(
    symbol="BTC/USDT",
    side="BUY",
    order_type="MARKET",
    quantity=0.1
)
```

## Data Providers

### Market Data Sources

| Provider | Cost | Rate Limits | Features |
|----------|------|-------------|----------|
| **CoinGecko** | Free/Premium | 50-10k req/min | Comprehensive market data, no API key for basic |
| **Binance Public** | Free | 1200 req/min | High-quality OHLCV, real-time prices |
| **CryptoCompare** | Free/Premium | 100-100k req/min | Historical data, multi-exchange |

### News & Sentiment Sources

| Provider | Cost | Coverage | Features |
|----------|------|----------|----------|
| **CryptoPanic** | Free/Premium | Crypto news aggregation | RSS feeds, sentiment scoring |
| **CoinDesk** | Free | Professional crypto news | Market analysis, regulatory updates |
| **Reddit** | Free | Social sentiment | r/CryptoCurrency analysis |
| **Twitter/X** | Premium | Real-time sentiment | Bearer token required |

### Exchange Support

| Exchange | Spot Trading | Perpetuals | Features |
|----------|--------------|------------|----------|
| **Paper Broker** | ✅ | ✅ | 24/7 simulation, realistic fees |
| **Binance** | ✅ | ✅ | High liquidity, advanced orders |
| **Coinbase** | ✅ | ❌ | Institutional-grade |
| **Kraken** | ✅ | ✅ | European focus |
| **Hyperliquid** | ✅ | ✅ | Advanced perpetuals |

## Trading Examples

### Spot Trading Example

```python
import asyncio
from tradingagents.dataflows.crypto import CryptoPaperBroker

async def spot_trading_example():
    # Initialize paper broker with USDT balance
    broker = CryptoPaperBroker(initial_balance={"USDT": 10000})
    
    # Market buy order
    buy_order = await broker.create_order(
        symbol="BTC/USDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.1  # 0.1 BTC
    )
    
    # Check positions
    positions = await broker.get_positions()
    print(f"BTC Position: {positions.get('BTC', 0)}")
    
    # Limit sell order
    sell_order = await broker.create_order(
        symbol="BTC/USDT",
        side="SELL",
        order_type="LIMIT",
        quantity=0.05,
        price=45000  # Sell at $45,000
    )

asyncio.run(spot_trading_example())
```

### Perpetual Futures Example

```python
import asyncio
from tradingagents.dataflows.crypto import CryptoPaperBroker

async def perp_trading_example():
    broker = CryptoPaperBroker(
        initial_balance={"USDT": 10000},
        enable_perps=True
    )
    
    # Long position on BTC perpetual
    long_order = await broker.create_order(
        symbol="BTC-PERP",
        side="BUY",
        order_type="MARKET",
        notional=1000,  # $1000 position
        leverage=3      # 3x leverage
    )
    
    # Check funding PnL
    funding_pnl = await broker.get_funding_pnl("BTC-PERP")
    print(f"Funding PnL: {funding_pnl}")

asyncio.run(perp_trading_example())
```

### Risk Management Example

```python
from tradingagents.dataflows.crypto import CryptoRiskManager, RiskLimits

# Configure risk limits
risk_limits = RiskLimits(
    max_portfolio_risk=0.15,      # 15% max portfolio risk
    max_position_size=0.10,       # 10% max single position
    max_leverage=5,               # 5x max leverage
    max_concentration=0.30        # 30% max in single asset
)

# Initialize risk manager
risk_manager = CryptoRiskManager(risk_limits=risk_limits)

# Assess portfolio risk
portfolio_risk = await risk_manager.assess_portfolio_risk(positions, prices)
print(f"Portfolio Risk Score: {portfolio_risk.risk_score}")
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing crypto dependencies
pip install python-binance web3 ccxt aiohttp redis
```

**2. API Key Issues**
```bash
# Check .env file configuration
python -m cli.main config --validate

# Test provider connections
python -m cli.main providers --status
```

**3. Rate Limiting**
```bash
# Increase cache TTL in config
config["cache"]["ttl_fast"] = 300  # 5 minutes
config["cache"]["ttl_slow"] = 3600  # 1 hour
```

**4. Memory Issues with Large Analysis**
```bash
# Reduce analysis scope
config["max_debate_rounds"] = 1
config["research_depth"] = "low"
```

### Performance Optimization

**Cost Optimization:**
```python
config.update({
    "deep_think_llm": "gpt-4o-mini",     # Cheaper model
    "quick_think_llm": "gpt-4o-mini",    # Consistent model
    "provider_preset": "free",            # Free-tier providers
    "cache_enabled": True                 # Enable aggressive caching
})
```

**Speed Optimization:**
```python
config.update({
    "max_debate_rounds": 1,              # Reduce debate rounds
    "research_depth": "low",             # Faster analysis
    "parallel_analysis": True,           # Parallel processing
    "use_cached_data": True              # Prefer cached results
})
```

## Advanced Features

### Multi-Exchange Arbitrage

```python
from tradingagents.dataflows.crypto import CCXTBrokerFactory

# Initialize multiple exchanges
binance = CCXTBrokerFactory.create_broker("binance", api_key, secret)
coinbase = CCXTBrokerFactory.create_broker("coinbase", api_key, secret)

# Check price differences
btc_binance = await binance.get_price("BTC/USDT")
btc_coinbase = await coinbase.get_price("BTC/USD")

spread = abs(btc_binance - btc_coinbase) / btc_binance
if spread > 0.005:  # 0.5% arbitrage opportunity
    print(f"Arbitrage opportunity: {spread:.2%}")
```

### On-Chain Whale Tracking

```python
from tradingagents.dataflows.crypto import WhaleFlowTracker

tracker = WhaleFlowTracker()
whale_alerts = await tracker.get_whale_alerts("BTC", hours=24)

for alert in whale_alerts:
    print(f"Whale Alert: {alert.amount} BTC moved to {alert.exchange}")
```

### Funding Rate Optimization

```python
from tradingagents.dataflows.crypto import FundingCalculator

calculator = FundingCalculator()
funding_forecast = await calculator.get_funding_forecast("BTC-PERP", hours=48)

print(f"Expected funding cost: {funding_forecast.total_cost:.2f} USDT")
print(f"Optimal rebalance time: {funding_forecast.optimal_rebalance_time}")
```

## Support & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/TauricResearch/TradingAgents/issues)
- **Discord**: [Join our trading community](https://discord.com/invite/hk9PGKShPK)
- **Documentation**: [Full API documentation](https://docs.tauric.ai)
- **Research**: [Tauric Research Community](https://tauric.ai/)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <strong>🚀 Start trading crypto with AI-powered insights today! 🚀</strong><br>
  <em>Built by researchers, for traders</em>
</div> 