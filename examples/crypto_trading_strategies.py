#!/usr/bin/env python3
"""
Advanced Crypto Trading Strategies Examples

This script demonstrates advanced trading strategies and patterns specific to
cryptocurrency markets using the TradingAgents framework.

Strategies covered:
- DCA (Dollar Cost Averaging)
- Grid Trading
- Momentum Trading with Technical Indicators
- Funding Rate Arbitrage
- Cross-Exchange Arbitrage

Usage:
    python examples/crypto_trading_strategies.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tradingagents.dataflows.crypto import (
    CryptoPaperBroker, CoinGeckoClient, BinancePublicClient,
    CryptoTechnicalAnalyzer, FundingCalculator, CryptoRiskManager, RiskLimits
)
from tradingagents.dataflows.base_interfaces import OrderSide, OrderType


class CryptoDCAStrategy:
    """Dollar Cost Averaging strategy for crypto accumulation."""
    
    def __init__(self, broker: CryptoPaperBroker, symbol: str, amount_per_buy: float):
        self.broker = broker
        self.symbol = symbol
        self.amount_per_buy = amount_per_buy
        self.trades = []
    
    async def execute_dca_buy(self) -> Dict:
        """Execute a single DCA buy order."""
        print(f"💰 DCA: Buying ${self.amount_per_buy} worth of {self.symbol}")
        
        try:
            order = await self.broker.create_order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                notional=self.amount_per_buy
            )
            
            self.trades.append({
                "timestamp": datetime.now(),
                "amount": self.amount_per_buy,
                "order": order
            })
            
            print(f"✅ DCA buy executed: {order}")
            return order
            
        except Exception as e:
            print(f"❌ DCA buy failed: {e}")
            return None
    
    async def run_dca_schedule(self, intervals: int, interval_hours: int = 24):
        """Run DCA strategy over multiple intervals."""
        print(f"🔄 Starting DCA strategy: {intervals} buys every {interval_hours} hours")
        
        for i in range(intervals):
            print(f"\n--- DCA Interval {i+1}/{intervals} ---")
            await self.execute_dca_buy()
            
            # In real implementation, you'd wait for the actual interval
            # For demo purposes, we'll just simulate the passage of time
            if i < intervals - 1:
                print(f"⏳ Waiting {interval_hours} hours for next DCA...")
        
        # Summary
        total_invested = len(self.trades) * self.amount_per_buy
        print(f"\n📊 DCA Summary:")
        print(f"  Total intervals: {len(self.trades)}")
        print(f"  Total invested: ${total_invested:,.2f}")
        print(f"  Average buy amount: ${self.amount_per_buy:,.2f}")


class CryptoGridStrategy:
    """Grid trading strategy for range-bound markets."""
    
    def __init__(self, broker: CryptoPaperBroker, symbol: str, 
                 base_price: float, grid_spacing: float, grid_levels: int):
        self.broker = broker
        self.symbol = symbol
        self.base_price = base_price
        self.grid_spacing = grid_spacing
        self.grid_levels = grid_levels
        self.active_orders = []
    
    async def setup_grid(self):
        """Setup buy and sell orders across the grid."""
        print(f"🕸️  Setting up grid for {self.symbol}")
        print(f"  Base price: ${self.base_price:,.2f}")
        print(f"  Grid spacing: {self.grid_spacing:.2%}")
        print(f"  Grid levels: {self.grid_levels}")
        
        # Create buy orders below current price
        for i in range(1, self.grid_levels + 1):
            buy_price = self.base_price * (1 - self.grid_spacing * i)
            
            try:
                order = await self.broker.create_order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=0.01,  # Fixed quantity for demo
                    price=buy_price
                )
                
                self.active_orders.append(order)
                print(f"📉 Buy order placed at ${buy_price:,.2f}")
                
            except Exception as e:
                print(f"❌ Failed to place buy order at ${buy_price:,.2f}: {e}")
        
        # Create sell orders above current price
        for i in range(1, self.grid_levels + 1):
            sell_price = self.base_price * (1 + self.grid_spacing * i)
            
            try:
                order = await self.broker.create_order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=0.01,  # Fixed quantity for demo
                    price=sell_price
                )
                
                self.active_orders.append(order)
                print(f"📈 Sell order placed at ${sell_price:,.2f}")
                
            except Exception as e:
                print(f"❌ Failed to place sell order at ${sell_price:,.2f}: {e}")
        
        print(f"✅ Grid setup complete: {len(self.active_orders)} orders placed")


class CryptoMomentumStrategy:
    """Momentum-based trading strategy using technical indicators."""
    
    def __init__(self, broker: CryptoPaperBroker, symbol: str):
        self.broker = broker
        self.symbol = symbol
        self.analyzer = CryptoTechnicalAnalyzer()
        self.position_size = 1000  # $1000 positions
    
    async def analyze_momentum(self) -> Dict:
        """Analyze momentum indicators for trading signals."""
        print(f"📊 Analyzing momentum for {self.symbol}")
        
        # Mock technical analysis (in real implementation, this would use actual data)
        analysis = {
            "rsi": 65,  # RSI above 50 indicates bullish momentum
            "macd_signal": "bullish",  # MACD crossover signal
            "trend": "upward",
            "volatility": "normal",
            "signal_strength": 0.7
        }
        
        print(f"  RSI: {analysis['rsi']}")
        print(f"  MACD: {analysis['macd_signal']}")
        print(f"  Trend: {analysis['trend']}")
        print(f"  Signal strength: {analysis['signal_strength']:.1%}")
        
        return analysis
    
    async def execute_momentum_trade(self, analysis: Dict):
        """Execute trade based on momentum analysis."""
        signal_strength = analysis['signal_strength']
        
        if signal_strength > 0.6 and analysis['macd_signal'] == 'bullish':
            print("🚀 Strong bullish signal detected - Going LONG")
            
            order = await self.broker.create_order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                notional=self.position_size
            )
            
            print(f"✅ Long position opened: {order}")
            
        elif signal_strength < 0.4 and analysis['macd_signal'] == 'bearish':
            print("📉 Strong bearish signal detected - Going SHORT")
            
            # For demo purposes (short would require margin/perp trading)
            print("📝 Short signal logged (requires perpetual futures)")
            
        else:
            print("➡️  Neutral signal - Holding position")


class CryptoFundingArbitrageStrategy:
    """Funding rate arbitrage strategy for perpetual futures."""
    
    def __init__(self, broker: CryptoPaperBroker):
        self.broker = broker
        self.funding_calc = FundingCalculator()
    
    async def find_funding_opportunities(self, symbols: List[str]) -> List[Dict]:
        """Find funding rate arbitrage opportunities."""
        print("🔍 Scanning for funding rate arbitrage opportunities...")
        
        opportunities = []
        
        for symbol in symbols:
            # Mock funding rate analysis
            funding_rate = 0.01  # 1% per day (high funding rate)
            
            if abs(funding_rate) > 0.005:  # 0.5% threshold
                direction = "short" if funding_rate > 0 else "long"
                opportunity = {
                    "symbol": symbol,
                    "funding_rate": funding_rate,
                    "direction": direction,
                    "expected_profit": abs(funding_rate) * 365 * 100,  # Annualized %
                }
                opportunities.append(opportunity)
                
                print(f"💰 Opportunity found: {symbol}")
                print(f"  Funding rate: {funding_rate:.3%} per day")
                print(f"  Strategy: {direction.upper()} perp, LONG spot")
                print(f"  Expected annual profit: {opportunity['expected_profit']:.1f}%")
        
        return opportunities
    
    async def execute_funding_arbitrage(self, opportunity: Dict):
        """Execute funding rate arbitrage trade."""
        symbol = opportunity['symbol']
        direction = opportunity['direction']
        
        print(f"⚡ Executing funding arbitrage for {symbol}")
        
        if direction == "short":
            # Short perp, long spot
            print("📉 Shorting perpetual future...")
            print("📈 Longing spot position...")
            
            # Mock execution
            print("✅ Arbitrage position established")
            print("💰 Collecting positive funding payments")
            
        else:
            # Long perp, short spot
            print("📈 Longing perpetual future...")
            print("📉 Shorting spot position...")
            
            # Mock execution
            print("✅ Arbitrage position established")
            print("💰 Paying negative funding to earn on spread")


async def demo_dca_strategy():
    """Demonstrate DCA trading strategy."""
    print("\n" + "="*60)
    print("  DOLLAR COST AVERAGING (DCA) STRATEGY")
    print("="*60)
    
    broker = CryptoPaperBroker(initial_balance={"USDT": 5000})
    dca = CryptoDCAStrategy(broker, "BTC/USDT", amount_per_buy=100)
    
    # Run DCA for 5 intervals
    await dca.run_dca_schedule(intervals=5, interval_hours=24)


async def demo_grid_strategy():
    """Demonstrate grid trading strategy."""
    print("\n" + "="*60)
    print("  GRID TRADING STRATEGY")
    print("="*60)
    
    broker = CryptoPaperBroker(initial_balance={"USDT": 10000, "BTC": 0.1})
    
    # Setup grid around current BTC price
    grid = CryptoGridStrategy(
        broker=broker,
        symbol="BTC/USDT",
        base_price=45000,      # Base price around $45k
        grid_spacing=0.02,     # 2% spacing between levels
        grid_levels=5          # 5 levels each direction
    )
    
    await grid.setup_grid()


async def demo_momentum_strategy():
    """Demonstrate momentum trading strategy."""
    print("\n" + "="*60)
    print("  MOMENTUM TRADING STRATEGY")
    print("="*60)
    
    broker = CryptoPaperBroker(initial_balance={"USDT": 10000})
    momentum = CryptoMomentumStrategy(broker, "ETH/USDT")
    
    # Analyze and trade
    analysis = await momentum.analyze_momentum()
    await momentum.execute_momentum_trade(analysis)


async def demo_funding_arbitrage():
    """Demonstrate funding rate arbitrage."""
    print("\n" + "="*60)
    print("  FUNDING RATE ARBITRAGE STRATEGY")
    print("="*60)
    
    broker = CryptoPaperBroker(initial_balance={"USDT": 20000}, enable_perps=True)
    arbitrage = CryptoFundingArbitrageStrategy(broker)
    
    # Scan for opportunities
    symbols = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
    opportunities = await arbitrage.find_funding_opportunities(symbols)
    
    if opportunities:
        # Execute first opportunity
        await arbitrage.execute_funding_arbitrage(opportunities[0])
    else:
        print("❌ No funding arbitrage opportunities found")


async def demo_risk_management_integration():
    """Demonstrate risk management across strategies."""
    print("\n" + "="*60)
    print("  INTEGRATED RISK MANAGEMENT")
    print("="*60)
    
    # Setup risk manager
    risk_limits = RiskLimits(
        max_portfolio_risk=0.20,
        max_position_size=0.15,
        max_leverage=3,
        max_concentration=0.40
    )
    
    risk_manager = CryptoRiskManager(risk_limits=risk_limits)
    
    print("⚠️  Risk Management Configuration:")
    print(f"  Max Portfolio Risk: {risk_limits.max_portfolio_risk:.1%}")
    print(f"  Max Position Size: {risk_limits.max_position_size:.1%}")
    print(f"  Max Leverage: {risk_limits.max_leverage}x")
    print(f"  Max Concentration: {risk_limits.max_concentration:.1%}")
    
    # Mock portfolio analysis
    print("\n📊 Portfolio Risk Assessment:")
    print("  Current portfolio risk: 0.12 (60% of limit)")
    print("  Largest position: BTC at 8% (53% of limit)")
    print("  Average leverage: 2.1x (70% of limit)")
    print("  Risk level: MODERATE")
    
    print("\n💡 Risk Management Recommendations:")
    print("  ✅ Portfolio within all risk limits")
    print("  ⚠️  Consider reducing position sizes before major news events")
    print("  📈 Opportunity to increase allocation if strong signals emerge")


def main():
    """Main function to run all strategy demos."""
    print("🚀 TradingAgents Crypto Trading Strategies Demo")
    print("🎯 Demonstrating advanced trading patterns for crypto markets")
    
    async def run_all_strategies():
        await demo_dca_strategy()
        await demo_grid_strategy()
        await demo_momentum_strategy()
        await demo_funding_arbitrage()
        await demo_risk_management_integration()
        
        print("\n" + "="*60)
        print("  ALL STRATEGIES DEMO COMPLETE")
        print("="*60)
        print("🎉 All trading strategy demos completed!")
        print("📚 These examples show the flexibility of the TradingAgents framework")
        print("⚠️  Remember: Past performance doesn't guarantee future results")
        print("💡 Always test strategies with paper trading before going live")
    
    # Run all demos
    asyncio.run(run_all_strategies())


if __name__ == "__main__":
    main() 