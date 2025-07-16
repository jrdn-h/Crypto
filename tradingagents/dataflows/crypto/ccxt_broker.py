"""
CCXT-based crypto exchange broker for real trading.

Provides live trading capabilities across multiple crypto exchanges using the CCXT library:
- Multi-exchange support (Binance, Coinbase, Kraken, etc.)
- Spot and perpetual futures trading
- Real-time order management
- Live position tracking
- Portfolio management
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import json

# CCXT is an optional dependency for real trading
try:
    import ccxt.pro as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    try:
        import ccxt
        CCXT_AVAILABLE = True
    except ImportError:
        CCXT_AVAILABLE = False

from ..base_interfaces import (
    ExecutionClient, AssetClass, Order, Position, Balance,
    OrderSide, OrderType, OrderStatus
)


class CCXTBroker(ExecutionClient):
    """
    Live crypto exchange broker using CCXT library.
    
    Features:
    - Multi-exchange support via CCXT
    - Spot and perpetual futures trading
    - Real-time order execution
    - Live position and balance tracking
    - Cross-exchange compatibility
    """
    
    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        sandbox: bool = True,
        enable_rate_limit: bool = True,
        timeout: int = 30000,
        enable_perpetuals: bool = True,
        default_leverage: float = 1.0,
        max_leverage: float = 20.0
    ):
        """
        Initialize CCXT broker for a specific exchange.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase', 'kraken')
            api_key: Exchange API key
            api_secret: Exchange API secret
            passphrase: API passphrase (required for some exchanges)
            sandbox: Use sandbox/testnet environment
            enable_rate_limit: Enable built-in rate limiting
            timeout: Request timeout in milliseconds
            enable_perpetuals: Enable perpetual futures trading
            default_leverage: Default leverage for new positions
            max_leverage: Maximum allowed leverage
        """
        if not CCXT_AVAILABLE:
            raise ImportError(
                "CCXT library not available. Install with: pip install ccxt"
            )
        
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self.enable_perpetuals = enable_perpetuals
        self.default_leverage = default_leverage
        self.max_leverage = max_leverage
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'timeout': timeout,
            'enableRateLimit': enable_rate_limit,
            'sandbox': sandbox,
        }
        
        if passphrase:
            config['passphrase'] = passphrase
        
        self.exchange = exchange_class(config)
        
        # Cache for exchange info
        self._markets_cache = {}
        self._positions_cache = {}
        self._balances_cache = {}
        self._last_cache_update = None
        self._cache_ttl = 30  # seconds
        
        # Initialize exchange capabilities
        self._exchange_capabilities = {}
    
    async def initialize(self) -> None:
        """Initialize exchange connection and load markets."""
        try:
            # Load markets
            await self.exchange.load_markets()
            
            # Cache exchange capabilities
            self._exchange_capabilities = {
                'spot': self.exchange.has.get('spot', False),
                'futures': self.exchange.has.get('future', False),
                'margin': self.exchange.has.get('margin', False),
                'derivatives': self.exchange.has.get('derivative', False),
                'createMarketOrder': self.exchange.has.get('createMarketOrder', False),
                'createLimitOrder': self.exchange.has.get('createLimitOrder', False),
                'fetchPositions': self.exchange.has.get('fetchPositions', False),
                'fetchBalance': self.exchange.has.get('fetchBalance', False),
                'cancelOrder': self.exchange.has.get('cancelOrder', False),
            }
            
            # Set default leverage if supported
            if self.exchange.has.get('setLeverage', False) and self.enable_perpetuals:
                try:
                    # This is exchange-specific, may need customization
                    pass  # Implementation depends on specific exchange
                except Exception:
                    pass  # Not all exchanges support setting default leverage
                    
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {self.exchange_id}: {e}")
    
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
        Create a live order on the exchange.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT", "ETH/USD:USD")
            side: Buy or sell
            order_type: Market, limit, etc.
            quantity: Order quantity
            price: Limit price (required for limit orders)
            reduce_only: For futures, only reduce existing position
            leverage: Leverage for futures positions
        """
        await self._ensure_initialized()
        
        # Normalize symbol for exchange
        normalized_symbol = await self._normalize_symbol(symbol)
        
        # Determine market type (spot vs futures)
        market = self.exchange.markets.get(normalized_symbol)
        if not market:
            raise ValueError(f"Symbol {symbol} not found on {self.exchange_id}")
        
        is_futures = market.get('type') == 'future' or market.get('derivative', False)
        
        if is_futures and not self.enable_perpetuals:
            raise ValueError(f"Futures trading not enabled: {symbol}")
        
        # Set leverage for futures if specified
        if is_futures and leverage and self.exchange.has.get('setLeverage', False):
            try:
                await self.exchange.set_leverage(leverage, normalized_symbol)
            except Exception as e:
                print(f"Warning: Could not set leverage {leverage} for {symbol}: {e}")
        
        # Prepare order parameters
        order_params = {}
        
        if reduce_only and is_futures:
            order_params['reduceOnly'] = True
        
        # Convert order type
        ccxt_order_type = self._convert_order_type(order_type)
        ccxt_side = side.value
        
        try:
            # Create order on exchange
            if order_type == OrderType.MARKET:
                ccxt_order = await self.exchange.create_market_order(
                    normalized_symbol, ccxt_side, quantity, None, None, order_params
                )
            elif order_type == OrderType.LIMIT:
                if not price:
                    raise ValueError("Price required for limit orders")
                ccxt_order = await self.exchange.create_limit_order(
                    normalized_symbol, ccxt_side, quantity, price, None, order_params
                )
            else:
                # For stop orders, use generic create_order
                ccxt_order = await self.exchange.create_order(
                    normalized_symbol, ccxt_order_type, ccxt_side, quantity, price, order_params
                )
            
            # Convert CCXT order to our Order model
            order = self._convert_ccxt_order(ccxt_order, symbol)
            
            return order
            
        except Exception as e:
            # Create rejected order
            order = Order(
                order_id=f"rejected_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                filled_quantity=0.0,
                status=OrderStatus.REJECTED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                asset_class=AssetClass.CRYPTO
            )
            
            raise RuntimeError(f"Order creation failed: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order on the exchange."""
        await self._ensure_initialized()
        
        try:
            # Cancel order on exchange
            result = await self.exchange.cancel_order(order_id)
            return result is not None
            
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        """Get current positions from the exchange."""
        await self._ensure_initialized()
        
        if not self.exchange.has.get('fetchPositions', False):
            return []  # Exchange doesn't support position tracking
        
        try:
            # Check cache first
            if self._is_cache_valid():
                return list(self._positions_cache.values())
            
            # Fetch fresh positions
            ccxt_positions = await self.exchange.fetch_positions()
            
            positions = []
            for ccxt_pos in ccxt_positions:
                if ccxt_pos['contracts'] and ccxt_pos['contracts'] != 0:
                    position = self._convert_ccxt_position(ccxt_pos)
                    positions.append(position)
                    self._positions_cache[position.symbol] = position
            
            self._update_cache_timestamp()
            return positions
            
        except Exception as e:
            print(f"Failed to fetch positions: {e}")
            return []
    
    async def get_balances(self) -> List[Balance]:
        """Get account balances from the exchange."""
        await self._ensure_initialized()
        
        try:
            # Check cache first
            if self._is_cache_valid():
                return list(self._balances_cache.values())
            
            # Fetch fresh balances
            ccxt_balance = await self.exchange.fetch_balance()
            
            balances = []
            for currency, balance_info in ccxt_balance.items():
                if currency in ['info', 'free', 'used', 'total']:
                    continue  # Skip metadata fields
                
                if isinstance(balance_info, dict) and balance_info.get('total', 0) > 0:
                    balance = Balance(
                        currency=currency,
                        available=balance_info.get('free', 0.0),
                        total=balance_info.get('total', 0.0),
                        reserved=balance_info.get('used', 0.0),
                        last_updated=datetime.now(timezone.utc)
                    )
                    balances.append(balance)
                    self._balances_cache[currency] = balance
            
            self._update_cache_timestamp()
            return balances
            
        except Exception as e:
            print(f"Failed to fetch balances: {e}")
            return []
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from the exchange."""
        await self._ensure_initialized()
        
        try:
            ccxt_order = await self.exchange.fetch_order(order_id)
            return self._convert_ccxt_order(ccxt_order)
            
        except Exception as e:
            print(f"Failed to fetch order {order_id}: {e}")
            return None
    
    @property
    def is_paper_trading(self) -> bool:
        """Whether this is paper trading (sandbox mode)."""
        return self.sandbox
    
    @property
    def asset_class(self) -> AssetClass:
        """This handles crypto assets."""
        return AssetClass.CRYPTO
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        await self._ensure_initialized()
        
        symbols = []
        for symbol, market in self.exchange.markets.items():
            # Include spot markets
            if market.get('spot', False):
                symbols.append(symbol)
            
            # Include futures if enabled
            if (self.enable_perpetuals and 
                (market.get('future', False) or market.get('derivative', False))):
                symbols.append(symbol)
        
        return symbols
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and capabilities."""
        await self._ensure_initialized()
        
        return {
            'exchange_id': self.exchange_id,
            'sandbox': self.sandbox,
            'capabilities': self._exchange_capabilities,
            'markets_count': len(self.exchange.markets),
            'rate_limit': self.exchange.rateLimit,
            'timeout': self.exchange.timeout,
            'enable_perpetuals': self.enable_perpetuals,
            'max_leverage': self.max_leverage,
        }
    
    # ==== Internal Methods ====
    
    async def _ensure_initialized(self) -> None:
        """Ensure exchange is initialized."""
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            await self.initialize()
    
    async def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for the specific exchange."""
        # This may need exchange-specific customization
        # For now, return as-is and let CCXT handle it
        return symbol
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType enum to CCXT order type string."""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop_market',
            OrderType.STOP_LIMIT: 'stop_limit',
        }
        return mapping.get(order_type, 'market')
    
    def _convert_ccxt_order(self, ccxt_order: Dict[str, Any], symbol: Optional[str] = None) -> Order:
        """Convert CCXT order to our Order model."""
        # Map CCXT order status to our enum
        status_mapping = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'cancelled': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'pending': OrderStatus.PENDING,
        }
        
        # Map order side
        side_mapping = {
            'buy': OrderSide.BUY,
            'sell': OrderSide.SELL,
        }
        
        # Map order type
        type_mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'stop_limit': OrderType.STOP_LIMIT,
            'stop_market': OrderType.STOP,
        }
        
        return Order(
            order_id=str(ccxt_order['id']),
            symbol=symbol or ccxt_order['symbol'],
            side=side_mapping.get(ccxt_order['side'], OrderSide.BUY),
            order_type=type_mapping.get(ccxt_order['type'], OrderType.MARKET),
            quantity=ccxt_order['amount'] or 0.0,
            price=ccxt_order['price'],
            filled_quantity=ccxt_order['filled'] or 0.0,
            status=status_mapping.get(ccxt_order['status'], OrderStatus.PENDING),
            created_at=datetime.fromtimestamp(ccxt_order['timestamp'] / 1000, timezone.utc) if ccxt_order['timestamp'] else datetime.now(timezone.utc),
            updated_at=datetime.fromtimestamp(ccxt_order['lastTradeTimestamp'] / 1000, timezone.utc) if ccxt_order['lastTradeTimestamp'] else datetime.now(timezone.utc),
            asset_class=AssetClass.CRYPTO
        )
    
    def _convert_ccxt_position(self, ccxt_position: Dict[str, Any]) -> Position:
        """Convert CCXT position to our Position model."""
        return Position(
            symbol=ccxt_position['symbol'],
            quantity=ccxt_position['contracts'] or 0.0,
            average_price=ccxt_position['entryPrice'] or 0.0,
            market_value=ccxt_position['notional'] or 0.0,
            unrealized_pnl=ccxt_position['unrealizedPnl'] or 0.0,
            asset_class=AssetClass.CRYPTO,
            last_updated=datetime.now(timezone.utc)
        )
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._last_cache_update:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self._last_cache_update).total_seconds()
        return elapsed < self._cache_ttl
    
    def _update_cache_timestamp(self) -> None:
        """Update cache timestamp."""
        self._last_cache_update = datetime.now(timezone.utc)


class CCXTBrokerFactory:
    """Factory for creating CCXT brokers for different exchanges."""
    
    @staticmethod
    def create_binance_broker(
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        **kwargs
    ) -> CCXTBroker:
        """Create Binance broker instance."""
        return CCXTBroker(
            exchange_id='binance',
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            enable_perpetuals=True,
            max_leverage=20.0,
            **kwargs
        )
    
    @staticmethod
    def create_coinbase_broker(
        api_key: str,
        api_secret: str,
        passphrase: str,
        sandbox: bool = True,
        **kwargs
    ) -> CCXTBroker:
        """Create Coinbase Pro broker instance."""
        return CCXTBroker(
            exchange_id='coinbasepro',
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            sandbox=sandbox,
            enable_perpetuals=False,  # Coinbase Pro doesn't support futures
            **kwargs
        )
    
    @staticmethod
    def create_kraken_broker(
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        **kwargs
    ) -> CCXTBroker:
        """Create Kraken broker instance."""
        return CCXTBroker(
            exchange_id='kraken',
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            enable_perpetuals=False,  # Kraken has limited futures support
            **kwargs
        )
    
    @staticmethod
    def create_bybit_broker(
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        **kwargs
    ) -> CCXTBroker:
        """Create Bybit broker instance."""
        return CCXTBroker(
            exchange_id='bybit',
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            enable_perpetuals=True,
            max_leverage=100.0,  # Bybit supports high leverage
            **kwargs
        )
    
    @staticmethod
    def get_supported_exchanges() -> List[str]:
        """Get list of supported exchanges."""
        return ['binance', 'coinbasepro', 'kraken', 'bybit', 'okx', 'huobi', 'ftx'] 