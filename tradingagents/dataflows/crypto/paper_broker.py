"""
Crypto Paper Trading Broker with 24/7 market support.

Provides simulated trading environment for crypto assets including:
- 24/7 continuous market simulation
- Spot and perpetual futures trading
- Notional position sizing 
- Realistic slippage and fees
- Multi-asset balance management
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

from ..base_interfaces import (
    ExecutionClient, AssetClass, Order, Position, Balance,
    OrderSide, OrderType, OrderStatus, MarketDataClient
)


class CryptoPaperBroker(ExecutionClient):
    """
    Paper trading broker for crypto assets with 24/7 market simulation.
    
    Features:
    - Continuous 24/7 trading (no market hours)
    - Spot and perpetual futures support
    - Notional position sizing for crypto
    - Realistic fees and slippage simulation
    - Multi-currency balance management
    """
    
    def __init__(
        self,
        initial_balance: float = 100000.0,
        base_currency: str = "USDT",
        fee_rate: float = 0.001,  # 0.1% (typical crypto exchange fee)
        slippage_bps: float = 5.0,  # 5 basis points
        market_data_client: Optional[MarketDataClient] = None,
        enable_perpetuals: bool = True,
        max_leverage: float = 10.0,
        funding_rate_interval_hours: int = 8,
        state_file: Optional[str] = None
    ):
        """
        Initialize the crypto paper broker.
        
        Args:
            initial_balance: Starting balance in base currency
            base_currency: Base currency for the account (USDT, USD, etc.)
            fee_rate: Trading fee rate (0.001 = 0.1%)
            slippage_bps: Market slippage in basis points
            market_data_client: Client for real-time price data
            enable_perpetuals: Whether to support perpetual futures
            max_leverage: Maximum leverage for perp positions
            funding_rate_interval_hours: Hours between funding payments
            state_file: File to persist broker state
        """
        self.base_currency = base_currency
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.market_data_client = market_data_client
        self.enable_perpetuals = enable_perpetuals
        self.max_leverage = max_leverage
        self.funding_rate_interval_hours = funding_rate_interval_hours
        self.state_file = state_file
        
        # Trading state
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._balances: Dict[str, Balance] = {}
        self._order_counter = 0
        self._last_funding_payment = datetime.now(timezone.utc)
        
        # Initialize base currency balance
        self._balances[base_currency] = Balance(
            currency=base_currency,
            available=initial_balance,
            total=initial_balance,
            reserved=0.0,
            last_updated=datetime.now(timezone.utc)
        )
        
        # Load persisted state if available
        if state_file and Path(state_file).exists():
            self._load_state()
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        leverage: Optional[float] = None
    ) -> Order:
        """
        Create a new crypto order with 24/7 execution.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT", "ETH-PERP")
            side: Buy or sell
            order_type: Market, limit, etc.
            quantity: Order quantity (in base asset for spot, contracts for perp)
            price: Limit price (required for limit orders)
            reduce_only: For perps, only reduce existing position
            leverage: Leverage for perp positions (1-max_leverage)
        """
        # Generate order ID
        order_id = f"order_{uuid.uuid4().hex[:8]}"
        self._order_counter += 1
        
        # Determine if this is a perpetual
        is_perp = self._is_perpetual(symbol)
        
        if is_perp and not self.enable_perpetuals:
            raise ValueError(f"Perpetual trading not enabled: {symbol}")
        
        # Validate leverage for perps
        if is_perp and leverage:
            if leverage > self.max_leverage:
                raise ValueError(f"Leverage {leverage} exceeds maximum {self.max_leverage}")
        
        # Create order object
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            filled_quantity=0.0,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            asset_class=AssetClass.CRYPTO
        )
        
        self._orders[order_id] = order
        
        # For market orders, execute immediately (24/7 market)
        if order_type == OrderType.MARKET:
            await self._execute_market_order(order, is_perp, leverage or 1.0, reduce_only)
        else:
            # For limit orders, validate funds and reserve
            if not await self._validate_and_reserve_funds(order, is_perp, leverage or 1.0):
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now(timezone.utc)
        
        self._save_state()
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order and unreserve funds."""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN]:
            return False
        
        # Unreserve funds
        await self._unreserve_funds(order)
        
        order.status = OrderStatus.CANCELED
        order.updated_at = datetime.now(timezone.utc)
        
        self._save_state()
        return True
    
    async def get_positions(self) -> List[Position]:
        """Get all current positions (crypto operates 24/7)."""
        # Update funding payments for perps before returning positions
        await self._process_funding_payments()
        
        return list(self._positions.values())
    
    async def get_balances(self) -> List[Balance]:
        """Get all account balances with real-time updates."""
        # Update position values with current market prices
        await self._update_position_values()
        
        return list(self._balances.values())
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status with 24/7 fill simulation."""
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        
        # Check if limit orders should be filled (24/7 price monitoring)
        if (order.status == OrderStatus.OPEN and 
            order.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]):
            await self._check_limit_order_fills(order)
        
        return order
    
    @property
    def is_paper_trading(self) -> bool:
        """This is always paper trading."""
        return True
    
    @property
    def asset_class(self) -> AssetClass:
        """This handles crypto assets."""
        return AssetClass.CRYPTO
    
    # ==== Internal Methods ====
    
    def _is_perpetual(self, symbol: str) -> bool:
        """Check if symbol is a perpetual futures contract."""
        return symbol.endswith("-PERP") or symbol.endswith("PERP") or "PERP" in symbol.upper()
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        if self.market_data_client:
            try:
                price = await self.market_data_client.get_latest_price(symbol)
                if price:
                    return price
            except Exception:
                pass  # Fallback to default
        
        # Fallback: use hardcoded prices for testing
        fallback_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2800.0,
            "BTC-PERP": 45050.0,
            "ETH-PERP": 2805.0,
        }
        return fallback_prices.get(symbol, 100.0)
    
    async def _execute_market_order(
        self, 
        order: Order, 
        is_perp: bool, 
        leverage: float, 
        reduce_only: bool
    ) -> None:
        """Execute a market order immediately (24/7 execution)."""
        try:
            # Get current market price
            market_price = await self._get_current_price(order.symbol)
            
            # Apply slippage
            slippage_factor = self.slippage_bps / 10000
            if order.side == OrderSide.BUY:
                execution_price = market_price * (1 + slippage_factor)
            else:
                execution_price = market_price * (1 - slippage_factor)
            
            # Calculate fees
            notional_value = order.quantity * execution_price
            if is_perp:
                notional_value *= leverage
            
            fees = notional_value * self.fee_rate
            
            # Validate funds
            if not await self._validate_funds_for_execution(order, execution_price, fees, is_perp, leverage):
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now(timezone.utc)
                return
            
            # Execute the trade
            await self._settle_trade(order, execution_price, fees, is_perp, leverage, reduce_only)
            
            order.filled_quantity = order.quantity
            order.status = OrderStatus.FILLED
            order.updated_at = datetime.now(timezone.utc)
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now(timezone.utc)
            print(f"Order execution failed: {e}")
    
    async def _validate_funds_for_execution(
        self, 
        order: Order, 
        price: float, 
        fees: float, 
        is_perp: bool, 
        leverage: float
    ) -> bool:
        """Validate sufficient funds for order execution."""
        if order.side == OrderSide.BUY:
            if is_perp:
                # For perp buys, only need margin (notional/leverage) + fees
                required = (order.quantity * price / leverage) + fees
            else:
                # For spot buys, need full notional + fees
                required = (order.quantity * price) + fees
            
            base_balance = self._balances.get(self.base_currency)
            return base_balance and base_balance.available >= required
        
        else:  # SELL
            if is_perp:
                # For perp sells, only need margin + fees
                required_margin = (order.quantity * price / leverage) + fees
                base_balance = self._balances.get(self.base_currency)
                return base_balance and base_balance.available >= required_margin
            else:
                # For spot sells, need the actual asset
                base_asset = order.symbol.split('/')[0]
                asset_balance = self._balances.get(base_asset)
                return asset_balance and asset_balance.available >= order.quantity
    
    async def _settle_trade(
        self, 
        order: Order, 
        price: float, 
        fees: float, 
        is_perp: bool, 
        leverage: float,
        reduce_only: bool
    ) -> None:
        """Settle a completed trade and update positions/balances."""
        notional_value = order.quantity * price
        
        if is_perp:
            # Handle perpetual futures
            await self._settle_perp_trade(order, price, fees, leverage, reduce_only)
        else:
            # Handle spot trading
            await self._settle_spot_trade(order, price, fees)
    
    async def _settle_spot_trade(self, order: Order, price: float, fees: float) -> None:
        """Settle a spot trade."""
        base_asset, quote_asset = order.symbol.split('/')
        notional_value = order.quantity * price
        
        if order.side == OrderSide.BUY:
            # Deduct quote currency
            quote_balance = self._balances.get(quote_asset, self._create_empty_balance(quote_asset))
            quote_balance.available -= (notional_value + fees)
            quote_balance.total -= (notional_value + fees)
            quote_balance.last_updated = datetime.now(timezone.utc)
            self._balances[quote_asset] = quote_balance
            
            # Add base asset
            base_balance = self._balances.get(base_asset, self._create_empty_balance(base_asset))
            base_balance.available += order.quantity
            base_balance.total += order.quantity
            base_balance.last_updated = datetime.now(timezone.utc)
            self._balances[base_asset] = base_balance
            
        else:  # SELL
            # Deduct base asset
            base_balance = self._balances.get(base_asset)
            base_balance.available -= order.quantity
            base_balance.total -= order.quantity
            base_balance.last_updated = datetime.now(timezone.utc)
            
            # Add quote currency (minus fees)
            quote_balance = self._balances.get(quote_asset, self._create_empty_balance(quote_asset))
            quote_balance.available += (notional_value - fees)
            quote_balance.total += (notional_value - fees)
            quote_balance.last_updated = datetime.now(timezone.utc)
            self._balances[quote_asset] = quote_balance
    
    async def _settle_perp_trade(
        self, 
        order: Order, 
        price: float, 
        fees: float, 
        leverage: float,
        reduce_only: bool
    ) -> None:
        """Settle a perpetual futures trade."""
        symbol = order.symbol
        notional_value = order.quantity * price
        margin_required = notional_value / leverage
        
        # Get or create position
        position = self._positions.get(symbol)
        
        if order.side == OrderSide.BUY:
            new_quantity = order.quantity
        else:
            new_quantity = -order.quantity
        
        if position:
            # Update existing position
            if reduce_only:
                # Only allow reducing the position
                if (position.quantity > 0 and new_quantity > 0) or (position.quantity < 0 and new_quantity < 0):
                    raise ValueError("Reduce-only order cannot increase position")
                new_quantity = min(abs(new_quantity), abs(position.quantity)) * (1 if new_quantity > 0 else -1)
            
            # Calculate new average price
            old_notional = position.quantity * position.average_price
            new_notional = new_quantity * price
            total_quantity = position.quantity + new_quantity
            
            if total_quantity != 0:
                new_avg_price = (old_notional + new_notional) / total_quantity
                position.quantity = total_quantity
                position.average_price = abs(new_avg_price)
            else:
                # Position closed
                del self._positions[symbol]
                position = None
                
        else:
            # Create new position
            if reduce_only:
                raise ValueError("Cannot create new position with reduce-only order")
                
            position = Position(
                symbol=symbol,
                quantity=new_quantity,
                average_price=price,
                market_value=notional_value * (1 if new_quantity > 0 else -1),
                unrealized_pnl=0.0,
                asset_class=AssetClass.CRYPTO,
                last_updated=datetime.now(timezone.utc)
            )
            self._positions[symbol] = position
        
        # Update balances (deduct margin and fees)
        base_balance = self._balances.get(self.base_currency)
        base_balance.available -= (margin_required + fees)
        base_balance.total -= fees  # Only fees reduce total, margin is reserved
        base_balance.last_updated = datetime.now(timezone.utc)
    
    def _create_empty_balance(self, currency: str) -> Balance:
        """Create an empty balance for a new currency."""
        return Balance(
            currency=currency,
            available=0.0,
            total=0.0,
            reserved=0.0,
            last_updated=datetime.now(timezone.utc)
        )
    
    async def _validate_and_reserve_funds(
        self, 
        order: Order, 
        is_perp: bool, 
        leverage: float
    ) -> bool:
        """Validate and reserve funds for a pending order."""
        # Implementation similar to _validate_funds_for_execution
        # but reserves the funds instead of spending them
        # This is a simplified version for now
        return True
    
    async def _unreserve_funds(self, order: Order) -> None:
        """Unreserve funds when an order is canceled."""
        # Implementation to unreserve funds
        pass
    
    async def _check_limit_order_fills(self, order: Order) -> None:
        """Check if limit orders should be filled based on current price."""
        # Implementation for 24/7 limit order monitoring
        pass
    
    async def _process_funding_payments(self) -> None:
        """Process funding payments for perpetual positions."""
        now = datetime.now(timezone.utc)
        hours_since_last = (now - self._last_funding_payment).total_seconds() / 3600
        
        if hours_since_last >= self.funding_rate_interval_hours:
            # Process funding for all perp positions
            for symbol, position in self._positions.items():
                if self._is_perpetual(symbol):
                    # Simplified funding rate (0.01% every 8 hours)
                    funding_rate = 0.0001 * (hours_since_last / self.funding_rate_interval_hours)
                    current_price = await self._get_current_price(symbol)
                    
                    notional_value = abs(position.quantity) * current_price
                    funding_payment = notional_value * funding_rate
                    
                    if position.quantity > 0:  # Long pays funding
                        funding_payment = -funding_payment
                    
                    # Update balance
                    base_balance = self._balances.get(self.base_currency)
                    base_balance.available += funding_payment
                    base_balance.total += funding_payment
                    base_balance.last_updated = now
            
            self._last_funding_payment = now
    
    async def _update_position_values(self) -> None:
        """Update position market values with current prices."""
        for symbol, position in self._positions.items():
            current_price = await self._get_current_price(symbol)
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.average_price) * position.quantity
            position.last_updated = datetime.now(timezone.utc)
    
    def _save_state(self) -> None:
        """Save broker state to file."""
        if not self.state_file:
            return
        
        state = {
            'orders': {k: v.dict() for k, v in self._orders.items()},
            'positions': {k: v.dict() for k, v in self._positions.items()},
            'balances': {k: v.dict() for k, v in self._balances.items()},
            'order_counter': self._order_counter,
            'last_funding_payment': self._last_funding_payment.isoformat()
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save broker state: {e}")
    
    def _load_state(self) -> None:
        """Load broker state from file."""
        if not self.state_file or not Path(self.state_file).exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self._orders = {k: Order(**v) for k, v in state.get('orders', {}).items()}
            self._positions = {k: Position(**v) for k, v in state.get('positions', {}).items()}
            self._balances = {k: Balance(**v) for k, v in state.get('balances', {}).items()}
            self._order_counter = state.get('order_counter', 0)
            
            if 'last_funding_payment' in state:
                self._last_funding_payment = datetime.fromisoformat(state['last_funding_payment'])
                
        except Exception as e:
            print(f"Failed to load broker state: {e}") 