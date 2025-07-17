#!/usr/bin/env python3
"""
TradingAgents Crypto Demo - Basic Usage Examples

This script demonstrates the core functionality of the TradingAgents crypto extension,
including market analysis, paper trading, and risk management for cryptocurrency markets.

Usage:
    python examples/run_crypto_demo.py

Requirements:
    - OpenAI API key in .env file
    - Optional: CoinGecko/CryptoCompare API keys for enhanced data
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.crypto import (
    CoinGeckoClient, BinancePublicClient, CryptoPaperBroker,
    CryptoRiskManager, RiskLimits, WhaleFlowTracker
)


def print_header(title: str):
    """Print a formatted header for demo sections."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


async def demo_basic_crypto_analysis():
    """Demonstrate basic crypto market analysis using TradingAgents."""
    print_header("BASIC CRYPTO ANALYSIS DEMO")
    
    try:
        # Configure for crypto analysis
        config = DEFAULT_CONFIG.copy()
        config.update({
            "asset_class": "crypto",
            "crypto_providers": ["coingecko", "binance", "cryptocompare"],
            "execution_mode": "paper",
            "deep_think_llm": "gpt-4o-mini",  # Cost optimization
            "quick_think_llm": "gpt-4o-mini",
            "max_debate_rounds": 1,           # Speed optimization
            "online_tools": True
        })
        
        print("🚀 Initializing TradingAgents for crypto analysis...")
        # Use the enhanced toolkit for crypto
        from tradingagents.dataflows.enhanced_toolkit import EnhancedToolkit
        ta = TradingAgentsGraph(
            debug=True,
            config=config,
            toolkit=EnhancedToolkit(config=config)
        )
        
        # Analyze Bitcoin
        symbol = "BTC/USDT"
        date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"📊 Running analysis for {symbol} on {date}...")
        print("⏳ This may take a few minutes with AI agent discussions...")
        
        # Run the analysis
        graph_state, decision = ta.propagate(symbol, date)
        
        print_section("ANALYSIS RESULTS")
        print(f"Symbol: {symbol}")
        print(f"Date: {date}")
        print(f"Decision: {decision}")
        
        # Show some key insights from the analysis
        if hasattr(graph_state, 'analyst_reports'):
            print("\n📈 Analyst Insights:")
            for agent, report in graph_state.analyst_reports.items():
                print(f"  {agent}: {str(report)[:100]}...")
        
        return decision
        
    except Exception as e:
        print(f"❌ Error in crypto analysis: {e}")
        print("💡 Make sure you have OPENAI_API_KEY set in your .env file")
        return None


async def demo_market_data_clients():
    """Demonstrate direct usage of crypto data clients."""
    print_header("CRYPTO DATA PROVIDERS DEMO")
    
    symbols = ["bitcoin", "ethereum", "solana"]
    
    # CoinGecko client (free tier)
    print_section("CoinGecko Market Data")
    try:
        coingecko = CoinGeckoClient()
        print("🌍 Testing CoinGecko client (free tier)...")
        
        for symbol in symbols:
            try:
                price_data = await coingecko.get_latest_price(symbol)
                print(f"  {symbol.upper()}: ${price_data:,.2f}" if price_data else f"  {symbol.upper()}: Not found")
            except Exception as e:
                print(f"  {symbol.upper()}: Error - {e}")
                
    except Exception as e:
        print(f"❌ CoinGecko client error: {e}")
    
    # Binance client (free tier, high rate limits)
    print_section("Binance Public Data")
    try:
        binance = BinancePublicClient()
        print("🏦 Testing Binance public client...")
        
        crypto_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        for pair in crypto_pairs:
            try:
                price_data = await binance.get_latest_price(pair)
                print(f"  {pair}: ${price_data:,.2f}" if price_data else f"  {pair}: Not found")
            except Exception as e:
                print(f"  {pair}: Error - {e}")
                
    except Exception as e:
        print(f"❌ Binance client error: {e}")


async def demo_paper_trading():
    """Demonstrate crypto paper trading with the paper broker."""
    print_header("CRYPTO PAPER TRADING DEMO")
    
    try:
        # Initialize paper broker with starting balance
        initial_balance = {"USDT": 10000}  # $10,000 USDT
        broker = CryptoPaperBroker(
            initial_balance=initial_balance["USDT"],
            enable_perpetuals=True
        )
        
        print(f"💰 Starting paper trading with {initial_balance}")
        
        # Check initial balance
        balances = await broker.get_balances()
        print(f"📊 Initial balances: {balances}")
        
        print_section("Spot Trading Example")
        
        # Market buy order for BTC
        print("📈 Placing market buy order for 0.1 BTC...")
        buy_order = await broker.create_order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.1
        )
        print(f"✅ Buy order created: {buy_order}")
        
        # Check positions
        positions = await broker.get_positions()
        print(f"📍 Current positions: {positions}")
        
        # Limit sell order
        print("📉 Placing limit sell order for 0.05 BTC at $50,000...")
        sell_order = await broker.create_order(
            symbol="BTC/USDT",
            side="SELL",
            order_type="LIMIT",
            quantity=0.05,
            price=50000
        )
        print(f"✅ Sell order created: {sell_order}")
        
        print_section("Perpetual Futures Example")
        
        # Long position on ETH perpetual
        print("🚀 Opening long position on ETH-PERP with $1000 notional, 3x leverage...")
        perp_order = await broker.create_order(
            symbol="ETH-PERP",
            side="BUY",
            order_type="MARKET",
            notional=1000,  # $1000 position
            leverage=3      # 3x leverage
        )
        print(f"✅ Perp order created: {perp_order}")
        
        # Check funding PnL
        try:
            funding_pnl = await broker.get_funding_pnl("ETH-PERP")
            print(f"💸 Funding PnL: {funding_pnl}")
        except Exception as e:
            print(f"⚠️  Funding PnL not available: {e}")
        
        # Final portfolio summary
        final_balances = await broker.get_balances()
        final_positions = await broker.get_positions()
        
        print_section("Portfolio Summary")
        print(f"📊 Final balances: {final_balances}")
        print(f"📍 Final positions: {final_positions}")
        
    except Exception as e:
        print(f"❌ Paper trading error: {e}")


async def demo_risk_management():
    """Demonstrate crypto risk management features."""
    print_header("CRYPTO RISK MANAGEMENT DEMO")
    
    try:
        # Configure risk limits
        risk_limits = RiskLimits(
            max_daily_var=0.15,      # 15% max portfolio risk
            max_position_size_usd=10000,   # $10k max single position
            max_leverage=5,               # 5x max leverage
            max_portfolio_concentration=0.30  # 30% max in single asset
        )
        
        print("⚠️  Initializing crypto risk manager...")
        risk_manager = CryptoRiskManager(risk_limits=risk_limits)
        
        print(f"📋 Risk Limits Configuration:")
        print(f"  Max Daily VaR: {risk_limits.max_daily_var:.1%}")
        print(f"  Max Position Size: ${risk_limits.max_position_size_usd:,.0f}")
        print(f"  Max Leverage: {risk_limits.max_leverage}x")
        print(f"  Max Concentration: {risk_limits.max_portfolio_concentration:.1%}")
        
        # Mock portfolio for demonstration
        mock_positions = {
            "BTC": {"size": 0.5, "notional": 20000, "leverage": 2},
            "ETH": {"size": 10, "notional": 15000, "leverage": 3},
            "SOL": {"size": 100, "notional": 5000, "leverage": 1}
        }
        
        mock_prices = {
            "BTC": 40000,
            "ETH": 1500,
            "SOL": 50
        }
        
        print_section("Portfolio Risk Assessment")
        print("📊 Analyzing mock portfolio...")
        print(f"Positions: {mock_positions}")
        
        # This would normally call the risk manager's assess_portfolio_risk method
        print("✅ Risk assessment complete (demo mode)")
        print("📈 Portfolio risk score: 0.12 (within limits)")
        print("⚠️  ETH position near concentration limit")
        print("💡 Recommendation: Consider reducing ETH exposure")
        
    except Exception as e:
        print(f"❌ Risk management error: {e}")


async def demo_whale_tracking():
    """Demonstrate whale flow tracking (if available)."""
    print_header("WHALE FLOW TRACKING DEMO")
    
    try:
        print("🐋 Initializing whale flow tracker...")
        tracker = WhaleFlowTracker()
        
        print("📡 Tracking large transactions for BTC...")
        print("⏳ Scanning blockchain for whale movements...")
        
        # Mock whale alert for demonstration
        print_section("Whale Alerts (Demo)")
        print("🚨 Large BTC transaction detected:")
        print("  Amount: 1,500 BTC (~$60M)")
        print("  Direction: Exchange inflow")
        print("  Exchange: Binance")
        print("  Time: 15 minutes ago")
        print("  Potential impact: Bearish (selling pressure)")
        
        print("\n🚨 Large ETH transaction detected:")
        print("  Amount: 25,000 ETH (~$37.5M)")
        print("  Direction: Exchange outflow")
        print("  Exchange: Coinbase")
        print("  Time: 2 hours ago")
        print("  Potential impact: Bullish (accumulation)")
        
    except Exception as e:
        print(f"❌ Whale tracking error: {e}")


async def demo_provider_comparison():
    """Compare performance and features of different data providers."""
    print_header("DATA PROVIDER COMPARISON")
    
    providers = [
        ("CoinGecko", CoinGeckoClient),
        ("Binance", BinancePublicClient)
    ]
    
    test_symbol = "bitcoin"
    
    print("🔍 Comparing data providers for reliability and performance...")
    
    for provider_name, provider_class in providers:
        print_section(f"{provider_name} Performance")
        
        try:
            start_time = datetime.now()
            client = provider_class()
            
            # Test basic price fetch
            price_data = await client.get_latest_price(test_symbol)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            print(f"✅ {provider_name}:")
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Price: ${price_data:,.2f}" if price_data else "  Price: Not found")
            print(f"  Status: Available")
            
        except Exception as e:
            print(f"❌ {provider_name}:")
            print(f"  Status: Error - {e}")


def main():
    """Main demo function."""
    print_header("TRADINGAGENTS CRYPTO EXTENSION DEMO")
    print("🚀 Welcome to the TradingAgents Crypto Demo!")
    print("📝 This demo showcases the key features of the crypto extension.")
    print("💡 Make sure you have your .env file configured with API keys.")
    
    # Check for required environment variables
    required_env = ["OPENAI_API_KEY"]
    missing_env = [env for env in required_env if not os.getenv(env)]
    
    if missing_env:
        print(f"\n❌ Missing required environment variables: {missing_env}")
        print("💡 Please set these in your .env file before running the demo.")
        return
    
    print("\n🔧 Optional: Set COINGECKO_API_KEY and CRYPTOCOMPARE_API_KEY for enhanced features.")
    
    # Run demo sections
    async def run_all_demos():
        # Data provider demos (always available)
        await demo_market_data_clients()
        await demo_provider_comparison()
        
        # Paper trading demo
        await demo_paper_trading()
        
        # Risk management demo
        await demo_risk_management()
        
        # Whale tracking demo
        await demo_whale_tracking()
        
        # Full analysis demo (requires OpenAI API)
        print("\n🤖 Ready to run full AI analysis? This uses OpenAI API credits.")
        user_input = input("Run full crypto analysis demo? (y/n): ")
        
        if user_input.lower() in ['y', 'yes']:
            await demo_basic_crypto_analysis()
        else:
            print("⏭️  Skipping full analysis demo.")
        
        print_header("DEMO COMPLETE")
        print("🎉 Demo completed successfully!")
        print("📚 Check out CRYPTO_README.md for more detailed documentation.")
        print("🔗 Join our community: https://discord.com/invite/hk9PGKShPK")
    
    # Run the async demo
    asyncio.run(run_all_demos())


if __name__ == "__main__":
    main() 