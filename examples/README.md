# TradingAgents Crypto Examples

This directory contains practical examples demonstrating the key features and capabilities of the TradingAgents crypto extension.

## 📁 Example Scripts

### 🚀 `run_crypto_demo.py` - Main Demo Script
**Purpose**: Comprehensive introduction to the crypto extension  
**Features**: 
- Market data provider testing
- Paper trading examples
- Risk management demos
- Whale tracking demonstrations
- Full AI-powered crypto analysis

**Usage**:
```bash
python examples/run_crypto_demo.py
```

**Requirements**: OpenAI API key (optional: CoinGecko/CryptoCompare keys)

---

### 📈 `crypto_trading_strategies.py` - Advanced Trading Strategies  
**Purpose**: Demonstrates sophisticated crypto trading patterns  
**Features**:
- Dollar Cost Averaging (DCA)
- Grid trading for range-bound markets
- Momentum trading with technical indicators
- Funding rate arbitrage
- Integrated risk management

**Usage**:
```bash
python examples/crypto_trading_strategies.py
```

**Requirements**: Basic setup (no API keys required for demo mode)

---

## 🛠️ Setup Instructions

### 1. Basic Setup
```bash
# Ensure you're in the TradingAgents directory
cd TradingAgents

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp example.env .env
```

### 2. Configure Environment
Edit your `.env` file with the following minimum configuration:

```bash
# Required for AI analysis
OPENAI_API_KEY=your_openai_key_here

# Enable crypto support
DEFAULT_ASSET_CLASS=crypto
CRYPTO_SUPPORT=true

# Optional: Enhanced data providers
COINGECKO_API_KEY=your_coingecko_key_here
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key_here
```

### 3. Run Examples
```bash
# Start with the main demo
python examples/run_crypto_demo.py

# Explore advanced strategies
python examples/crypto_trading_strategies.py
```

## 🎯 What Each Example Teaches

### Main Demo (`run_crypto_demo.py`)
- **Data Provider Integration**: How to use CoinGecko, Binance, and CryptoCompare clients
- **Paper Trading**: Setting up and executing crypto trades in simulation
- **Portfolio Management**: Balance tracking and position management
- **Risk Assessment**: Basic risk management principles
- **AI Analysis**: Full TradingAgents analysis workflow

### Trading Strategies (`crypto_trading_strategies.py`)
- **DCA Strategy**: Systematic accumulation over time
- **Grid Trading**: Profit from range-bound price action
- **Momentum Trading**: Technical indicator-based decisions
- **Arbitrage**: Funding rate and cross-exchange opportunities
- **Risk Integration**: How strategies work with risk management

## 🔧 Customization Guide

### Modifying Trading Parameters
```python
# In any trading example, you can adjust:
initial_balance = {"USDT": 20000}  # Starting capital
position_size = 1000               # Position sizes
risk_limits = RiskLimits(
    max_portfolio_risk=0.15,       # Risk tolerance
    max_leverage=3                 # Leverage limits
)
```

### Adding Your Own Strategies
```python
# Template for new strategy class
class MyCustomStrategy:
    def __init__(self, broker, symbol):
        self.broker = broker
        self.symbol = symbol
    
    async def analyze_market(self):
        # Your analysis logic here
        pass
    
    async def execute_trades(self):
        # Your execution logic here
        pass
```

### Testing with Different Assets
```python
# Easy to test with different crypto pairs
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

for symbol in symbols:
    strategy = YourStrategy(broker, symbol)
    await strategy.run()
```

## 📊 Understanding the Output

### Market Data Examples
```
✅ CoinGecko:
  Response time: 0.45s
  Price: $43,250.00
  Status: Available
```

### Trading Examples
```
✅ Buy order created: Order(id='abc123', symbol='BTC/USDT', 
    side='BUY', quantity=0.1, status='FILLED')
📍 Current positions: {'BTC': 0.1, 'USDT': 5675.0}
```

### Risk Management Output
```
📊 Portfolio Risk Assessment:
  Current portfolio risk: 0.12 (60% of limit)
  Largest position: BTC at 8% (53% of limit)
  Risk level: MODERATE
```

## ⚠️ Important Notes

### Paper Trading vs Live Trading
- **All examples use paper trading by default** for safety
- Paper trading simulates real market conditions without real money
- Switch to live trading only after thorough testing
- Live trading requires exchange API keys and real funds

### Cost Considerations
- **Data providers**: Most examples use free tiers
- **AI analysis**: Uses OpenAI API (costs API credits)
- **Exchange APIs**: Free for paper trading, real trading has fees

### Risk Management
- Examples include risk management demonstrations
- **Always use appropriate position sizing**
- **Set stop losses and risk limits**
- **Never risk more than you can afford to lose**

## 🔗 Next Steps

After running these examples:

1. **Explore the CLI**: `python -m cli.main crypto analyze --ticker BTC/USDT`
2. **Read the full documentation**: `CRYPTO_README.md`
3. **Join the community**: [Discord](https://discord.com/invite/hk9PGKShPK)
4. **Contribute**: Submit your own examples or improvements

## 🆘 Troubleshooting

### Common Issues

**Import Errors**:
```bash
pip install python-binance web3 ccxt aiohttp redis
```

**API Key Issues**:
```bash
python -m cli.main config --validate
```

**Performance Issues**:
```python
# Use cheaper models in config
config["deep_think_llm"] = "gpt-4o-mini"
config["max_debate_rounds"] = 1
```

### Getting Help
- Check the main `CRYPTO_README.md` for detailed documentation
- Review error messages carefully - they often contain helpful hints
- Test with minimal examples first before complex strategies
- Join our Discord community for support

---

**Happy Trading! 🚀**

*Remember: These examples are for educational purposes. Always do your own research and never invest more than you can afford to lose.* 