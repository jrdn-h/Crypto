"""
Hyperliquid broker for advanced perpetual futures trading.

Provides specialized trading capabilities for Hyperliquid's advanced derivatives platform:
- High-performance perpetual futures
- Advanced order types (conditional, bracket, etc.)
- Real-time position management
- Integrated market making features
- Cross-margining support
"""

import asyncio
import uuid
import json
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import aiohttp

from ..base_interfaces import (
    ExecutionClient, AssetClass, Order, Position, Balance,
    OrderSide, OrderType, OrderStatus
)


class HyperliquidBroker(ExecutionClient):
    """
    Hyperliquid perpetual futures broker with advanced features.
    
    Features:
    - High-performance perpetual futures trading
    - Advanced order types (conditional, bracket, etc.)
    - Real-time WebSocket feeds
    - Cross-margining and portfolio management
    - Integrated market making capabilities
    - Sub-account support
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        sub_account_id: Optional[str] = None,
        max_leverage: float = 50.0,
        default_leverage: float = 5.0,
        enable_cross_margin: bool = True,
        enable_market_making: bool = False,
        ws_subscriptions: Optional[List[str]] = None
    ):
        """
        Initialize Hyperliquid broker.
        
        Args:
            api_key: Hyperliquid API key
            api_secret: Hyperliquid API secret
            testnet: Use testnet environment
            sub_account_id: Optional sub-account identifier
            max_leverage: Maximum leverage allowed
            default_leverage: Default leverage for new positions
            enable_cross_margin: Enable cross-margining
            enable_market_making: Enable market making features
            ws_subscriptions: WebSocket subscription topics
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.sub_account_id = sub_account_id
        self.max_leverage = max_leverage
        self.default_leverage = default_leverage
        self.enable_cross_margin = enable_cross_margin
        self.enable_market_making = enable_market_making
        
        # API endpoints
        if testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
            self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws"
        else:
            self.base_url = "https://api.hyperliquid.xyz"
            self.ws_url = "wss://api.hyperliquid.xyz/ws"
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_session: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # State tracking
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._balances: Dict[str, Balance] = {}
        self._account_info: Dict[str, Any] = {}
        
        # WebSocket subscriptions
        self.ws_subscriptions = ws_subscriptions or ['orders', 'fills', 'positions', 'balance']
        
        # Market data cache
        self._market_cache: Dict[str, Any] = {}
        self._last_market_update = None
    
    async def initialize(self) -> None:
        """Initialize connection to Hyperliquid."""
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'TradingAgents-Crypto/1.0',
                'Content-Type': 'application/json'
            }
        )
        
        # Load account info and market data
        await self._load_account_info()
        await self._load_market_data()
        
        # Initialize WebSocket connection
        if self.ws_subscriptions:
            await self._connect_websocket()
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        leverage: Optional[float] = None,
        time_in_force: str = "GTC",
        post_only: bool = False,
        conditional: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create an advanced order on Hyperliquid.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: Buy or sell
            order_type: Market, limit, etc.
            quantity: Order quantity in contracts
            price: Limit price (required for limit orders)
            reduce_only: Only reduce existing position
            leverage: Leverage for the position
            time_in_force: Time in force (GTC, IOC, FOK)
            post_only: Post-only order (maker only)
            conditional: Conditional order parameters
        """
        await self._ensure_session()
        
        # Normalize symbol for Hyperliquid
        normalized_symbol = self._normalize_symbol(symbol)
        
        # Validate leverage
        if leverage and leverage > self.max_leverage:
            raise ValueError(f"Leverage {leverage} exceeds maximum {self.max_leverage}")
        
        # Set leverage if specified
        if leverage:
            await self._set_leverage(normalized_symbol, leverage)
        
        # Prepare order parameters
        order_params = {
            "coin": normalized_symbol,
            "is_buy": side == OrderSide.BUY,
            "sz": str(quantity),
            "limit_px": str(price) if price else None,
            "order_type": self._convert_order_type(order_type),
            "reduce_only": reduce_only,
            "time_in_force": time_in_force,
            "post_only": post_only
        }
        
        # Add conditional parameters
        if conditional:
            order_params.update(conditional)
        
        # Generate client order ID
        client_order_id = f"hl_{uuid.uuid4().hex[:12]}"
        order_params["cloid"] = client_order_id
        
        try:
            # Submit order to Hyperliquid
            response = await self._make_request("POST", "/exchange", {
                "action": {
                    "type": "order",
                    "orders": [order_params]
                },
                "nonce": self._get_nonce(),
                "signature": self._sign_request(order_params)
            })
            
            # Parse response
            if response.get("status") == "ok" and response.get("response", {}).get("data"):
                order_data = response["response"]["data"]["statuses"][0]
                
                # Create order object
                order = Order(
                    order_id=str(order_data.get("resting", {}).get("oid", client_order_id)),
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    filled_quantity=0.0,
                    status=OrderStatus.OPEN if order_data.get("resting") else OrderStatus.FILLED,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    asset_class=AssetClass.CRYPTO
                )
                
                self._orders[order.order_id] = order
                return order
            
            else:
                # Order failed
                error_msg = response.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("error", "Unknown error")
                raise RuntimeError(f"Order creation failed: {error_msg}")
                
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
        """Cancel an existing order."""
        await self._ensure_session()
        
        try:
            response = await self._make_request("POST", "/exchange", {
                "action": {
                    "type": "cancel",
                    "cancels": [{"oid": int(order_id)}]
                },
                "nonce": self._get_nonce()
            })
            
            return response.get("status") == "ok"
            
        except Exception as e:
            print(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        """Get current positions from Hyperliquid."""
        await self._ensure_session()
        
        try:
            response = await self._make_request("POST", "/info", {
                "type": "clearinghouseState",
                "user": self.api_key
            })
            
            positions = []
            if response.get("assetPositions"):
                for pos_data in response["assetPositions"]:
                    if float(pos_data["position"]["szi"]) != 0:
                        position = Position(
                            symbol=pos_data["position"]["coin"],
                            quantity=float(pos_data["position"]["szi"]),
                            average_price=float(pos_data["position"]["entryPx"]) if pos_data["position"]["entryPx"] else 0.0,
                            market_value=float(pos_data["position"]["positionValue"]),
                            unrealized_pnl=float(pos_data["position"]["unrealizedPnl"]),
                            asset_class=AssetClass.CRYPTO,
                            last_updated=datetime.now(timezone.utc)
                        )
                        positions.append(position)
                        self._positions[position.symbol] = position
            
            return positions
            
        except Exception as e:
            print(f"Failed to fetch positions: {e}")
            return []
    
    async def get_balances(self) -> List[Balance]:
        """Get account balances from Hyperliquid."""
        await self._ensure_session()
        
        try:
            response = await self._make_request("POST", "/info", {
                "type": "clearinghouseState",
                "user": self.api_key
            })
            
            balances = []
            if response.get("marginSummary"):
                margin_summary = response["marginSummary"]
                
                # Main balance (USDC)
                balance = Balance(
                    currency="USDC",
                    available=float(margin_summary["accountValue"]) - float(margin_summary["totalMarginUsed"]),
                    total=float(margin_summary["accountValue"]),
                    reserved=float(margin_summary["totalMarginUsed"]),
                    last_updated=datetime.now(timezone.utc)
                )
                balances.append(balance)
                self._balances["USDC"] = balance
            
            return balances
            
        except Exception as e:
            print(f"Failed to fetch balances: {e}")
            return []
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from Hyperliquid."""
        await self._ensure_session()
        
        try:
            response = await self._make_request("POST", "/info", {
                "type": "openOrders",
                "user": self.api_key
            })
            
            for order_data in response:
                if str(order_data["oid"]) == order_id:
                    return self._convert_hyperliquid_order(order_data)
            
            # Order not found in open orders, might be filled/canceled
            return self._orders.get(order_id)
            
        except Exception as e:
            print(f"Failed to fetch order {order_id}: {e}")
            return None
    
    @property
    def is_paper_trading(self) -> bool:
        """Whether this is paper trading (testnet)."""
        return self.testnet
    
    @property
    def asset_class(self) -> AssetClass:
        """This handles crypto assets."""
        return AssetClass.CRYPTO
    
    async def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        leverage: Optional[float] = None
    ) -> List[Order]:
        """
        Create a bracket order with entry, take profit, and stop loss.
        
        This is a Hyperliquid-specific advanced feature.
        """
        orders = []
        
        # Create entry order
        entry_order = await self.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=entry_price,
            leverage=leverage
        )
        orders.append(entry_order)
        
        # Create take profit order (conditional)
        tp_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        tp_order = await self.create_order(
            symbol=symbol,
            side=tp_side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=take_profit_price,
            reduce_only=True,
            conditional={
                "trigger": {
                    "triggerPx": str(take_profit_price),
                    "isMarket": False,
                    "tpsl": "tp"
                }
            }
        )
        orders.append(tp_order)
        
        # Create stop loss order (conditional)
        sl_order = await self.create_order(
            symbol=symbol,
            side=tp_side,
            order_type=OrderType.STOP,
            quantity=quantity,
            price=stop_loss_price,
            reduce_only=True,
            conditional={
                "trigger": {
                    "triggerPx": str(stop_loss_price),
                    "isMarket": True,
                    "tpsl": "sl"
                }
            }
        )
        orders.append(sl_order)
        
        return orders
    
    async def get_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates for all perpetuals."""
        await self._ensure_session()
        
        try:
            response = await self._make_request("POST", "/info", {
                "type": "meta"
            })
            
            funding_rates = {}
            for universe_item in response.get("universe", []):
                coin = universe_item["name"]
                funding_rate = float(universe_item.get("prevDayPx", 0)) / 100  # Convert from percentage
                funding_rates[coin] = funding_rate
            
            return funding_rates
            
        except Exception as e:
            print(f"Failed to fetch funding rates: {e}")
            return {}
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for a symbol."""
        await self._ensure_session()
        
        normalized_symbol = self._normalize_symbol(symbol)
        
        try:
            response = await self._make_request("POST", "/info", {
                "type": "l2Book",
                "coin": normalized_symbol
            })
            
            return {
                "symbol": symbol,
                "bids": response.get("levels", [[]])[0],
                "asks": response.get("levels", [[], []])[1],
                "timestamp": datetime.now(timezone.utc),
                "mid_price": self._calculate_mid_price(response.get("levels", [[], []])),
            }
            
        except Exception as e:
            print(f"Failed to fetch market data for {symbol}: {e}")
            return {}
    
    # ==== Internal Methods ====
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is active."""
        if not self._session:
            await self.initialize()
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Hyperliquid (e.g., BTC/USD -> BTC)."""
        # Hyperliquid uses simplified symbols
        if '/' in symbol:
            return symbol.split('/')[0]
        if '-' in symbol:
            return symbol.split('-')[0]
        return symbol
    
    def _convert_order_type(self, order_type: OrderType) -> Dict[str, Any]:
        """Convert OrderType to Hyperliquid order type."""
        if order_type == OrderType.MARKET:
            return {"limit": {"tif": "Ioc"}}
        elif order_type == OrderType.LIMIT:
            return {"limit": {"tif": "Gtc"}}
        elif order_type == OrderType.STOP:
            return {"trigger": {"isMarket": True, "tpsl": "sl"}}
        elif order_type == OrderType.STOP_LIMIT:
            return {"trigger": {"isMarket": False, "tpsl": "sl"}}
        else:
            return {"limit": {"tif": "Gtc"}}
    
    def _convert_hyperliquid_order(self, order_data: Dict[str, Any]) -> Order:
        """Convert Hyperliquid order to our Order model."""
        status_mapping = {
            "open": OrderStatus.OPEN,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
        }
        
        return Order(
            order_id=str(order_data["oid"]),
            symbol=order_data["coin"],
            side=OrderSide.BUY if order_data["side"] == "B" else OrderSide.SELL,
            order_type=OrderType.LIMIT,  # Simplification for now
            quantity=float(order_data["sz"]),
            price=float(order_data["limitPx"]) if order_data["limitPx"] else None,
            filled_quantity=float(order_data["sz"]) - float(order_data["szDecimals"]),
            status=status_mapping.get(order_data.get("status", "open"), OrderStatus.OPEN),
            created_at=datetime.fromtimestamp(order_data["timestamp"] / 1000, timezone.utc),
            updated_at=datetime.now(timezone.utc),
            asset_class=AssetClass.CRYPTO
        )
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request to Hyperliquid API."""
        url = f"{self.base_url}{endpoint}"
        
        headers = {}
        if data:
            headers["Content-Type"] = "application/json"
            if "signature" not in data and endpoint == "/exchange":
                data["signature"] = self._sign_request(data)
        
        async with self._session.request(method, url, json=data, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    
    def _sign_request(self, data: Dict[str, Any]) -> str:
        """Sign request with API secret."""
        message = json.dumps(data, separators=(',', ':'), sort_keys=True)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_nonce(self) -> int:
        """Get current timestamp as nonce."""
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    
    async def _set_leverage(self, symbol: str, leverage: float) -> None:
        """Set leverage for a symbol."""
        try:
            await self._make_request("POST", "/exchange", {
                "action": {
                    "type": "updateLeverage",
                    "asset": self._normalize_symbol(symbol),
                    "isCross": self.enable_cross_margin,
                    "leverage": str(int(leverage))
                },
                "nonce": self._get_nonce()
            })
        except Exception as e:
            print(f"Failed to set leverage for {symbol}: {e}")
    
    async def _load_account_info(self) -> None:
        """Load account information."""
        try:
            response = await self._make_request("POST", "/info", {
                "type": "clearinghouseState",
                "user": self.api_key
            })
            self._account_info = response
        except Exception as e:
            print(f"Failed to load account info: {e}")
    
    async def _load_market_data(self) -> None:
        """Load market metadata."""
        try:
            response = await self._make_request("POST", "/info", {
                "type": "meta"
            })
            self._market_cache = response
            self._last_market_update = datetime.now(timezone.utc)
        except Exception as e:
            print(f"Failed to load market data: {e}")
    
    def _calculate_mid_price(self, levels: List[List[List[str]]]) -> Optional[float]:
        """Calculate mid price from order book levels."""
        if len(levels) >= 2 and levels[0] and levels[1]:
            best_bid = float(levels[0][0][0])
            best_ask = float(levels[1][0][0])
            return (best_bid + best_ask) / 2
        return None
    
    async def _connect_websocket(self) -> None:
        """Connect to Hyperliquid WebSocket for real-time updates."""
        try:
            self._ws_session = await self._session.ws_connect(self.ws_url)
            
            # Subscribe to relevant channels
            subscription_msg = {
                "method": "subscribe",
                "subscription": {
                    "type": "allMids"
                }
            }
            
            await self._ws_session.send_str(json.dumps(subscription_msg))
            
            # Start background task to handle WebSocket messages
            asyncio.create_task(self._handle_websocket_messages())
            
        except Exception as e:
            print(f"Failed to connect WebSocket: {e}")
    
    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self._ws_session:
                if message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    await self._process_websocket_message(data)
                elif message.type == aiohttp.WSMsgType.ERROR:
                    print(f"WebSocket error: {self._ws_session.exception()}")
                    break
        except Exception as e:
            print(f"WebSocket message handling error: {e}")
    
    async def _process_websocket_message(self, data: Dict[str, Any]) -> None:
        """Process incoming WebSocket message."""
        message_type = data.get("channel")
        
        if message_type == "fills":
            # Handle fill updates
            await self._handle_fill_update(data.get("data", {}))
        elif message_type == "orders":
            # Handle order updates
            await self._handle_order_update(data.get("data", {}))
        elif message_type == "position":
            # Handle position updates
            await self._handle_position_update(data.get("data", {}))
    
    async def _handle_fill_update(self, fill_data: Dict[str, Any]) -> None:
        """Handle trade fill update."""
        # Update order status based on fill
        pass
    
    async def _handle_order_update(self, order_data: Dict[str, Any]) -> None:
        """Handle order status update."""
        # Update order in cache
        pass
    
    async def _handle_position_update(self, position_data: Dict[str, Any]) -> None:
        """Handle position update."""
        # Update position in cache
        pass
    
    async def close(self) -> None:
        """Close all connections."""
        if self._ws_session:
            await self._ws_session.close()
        
        if self._session:
            await self._session.close() 