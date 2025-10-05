"""
Paper Trading Engine for Order Simulation and Portfolio Management
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from config import TradingConfig

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    fill_timestamp: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'timestamp': self.timestamp.isoformat(),
            'fill_timestamp': self.fill_timestamp.isoformat() if self.fill_timestamp else None,
            'commission': self.commission,
            'slippage': self.slippage,
            'notes': self.notes
        }

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price > 0:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        return 0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'entry_time': self.entry_time.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }

class Portfolio:
    """Portfolio management class"""
    
    def __init__(self, initial_capital: float = None):
        if initial_capital is None:
            initial_capital = TradingConfig.INITIAL_CAPITAL
            
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.orders: List[Order] = []
        self.order_history: List[Order] = []
        self.transaction_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L"""
        return self.total_value - self.initial_capital
    
    @property
    def total_pnl_pct(self) -> float:
        """Calculate total P&L percentage"""
        if self.initial_capital > 0:
            return (self.total_pnl / self.initial_capital) * 100
        return 0
    
    @property
    def positions_count(self) -> int:
        """Number of open positions"""
        return len(self.positions)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def add_position(self, symbol: str, quantity: int, price: float,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> bool:
        """Add or update a position"""
        if symbol in self.positions:
            # Update existing position (averaging)
            pos = self.positions[symbol]
            total_cost = (pos.quantity * pos.entry_price) + (quantity * price)
            pos.quantity += quantity
            pos.entry_price = total_cost / pos.quantity if pos.quantity > 0 else price
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        return True
    
    def close_position(self, symbol: str, price: float) -> Dict:
        """Close a position"""
        if symbol not in self.positions:
            return {'success': False, 'error': 'Position not found'}
        
        pos = self.positions[symbol]
        realized_pnl = pos.quantity * (price - pos.entry_price)
        
        # Record closed position
        closed = {
            'symbol': symbol,
            'quantity': pos.quantity,
            'entry_price': pos.entry_price,
            'exit_price': price,
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': ((price - pos.entry_price) / pos.entry_price) * 100,
            'entry_time': pos.entry_time.isoformat(),
            'exit_time': datetime.now().isoformat(),
            'holding_period': (datetime.now() - pos.entry_time).total_seconds() / 3600  # hours
        }
        
        self.closed_positions.append(closed)
        
        # Update cash
        self.cash += pos.quantity * price
        
        # Remove position
        del self.positions[symbol]
        
        return {'success': True, 'closed_position': closed}
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def check_stop_loss_take_profit(self, prices: Dict[str, float]) -> List[Dict]:
        """Check if any positions hit stop loss or take profit"""
        triggered = []
        
        for symbol, pos in list(self.positions.items()):
            if symbol in prices:
                price = prices[symbol]
                
                # Check stop loss
                if pos.stop_loss and price <= pos.stop_loss:
                    result = self.close_position(symbol, price)
                    if result['success']:
                        triggered.append({
                            'symbol': symbol,
                            'type': 'STOP_LOSS',
                            'price': price,
                            **result['closed_position']
                        })
                
                # Check take profit
                elif pos.take_profit and price >= pos.take_profit:
                    result = self.close_position(symbol, price)
                    if result['success']:
                        triggered.append({
                            'symbol': symbol,
                            'type': 'TAKE_PROFIT',
                            'price': price,
                            **result['closed_position']
                        })
        
        return triggered
    
    def get_summary(self) -> Dict:
        """Get portfolio summary"""
        open_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        closed_pnl = sum(cp['realized_pnl'] for cp in self.closed_positions)
        
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': self.total_value - self.cash,
            'initial_capital': self.initial_capital,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'open_positions': self.positions_count,
            'unrealized_pnl': open_pnl,
            'realized_pnl': closed_pnl,
            'total_trades': len(self.closed_positions),
            'winning_trades': sum(1 for cp in self.closed_positions if cp['realized_pnl'] > 0),
            'losing_trades': sum(1 for cp in self.closed_positions if cp['realized_pnl'] < 0),
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
        }


class PaperTradingEngine:
    """Main paper trading engine"""
    
    def __init__(self, mode: str = TradingConfig.INTRADAY, 
                 initial_capital: float = None):
        self.mode = mode
        self.config = TradingConfig.get_config(mode)
        self.portfolio = Portfolio(initial_capital)
        self.metrics_config = TradingConfig.METRICS_CONFIG
        self.order_counter = 0
        self.execution_log = []
        
        logger.info(f"Paper Trading Engine initialized in {mode} mode")
    
    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"ORD_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
    
    def place_order(self, symbol: str, side: OrderSide, quantity: int,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   notes: str = "") -> Order:
        """Place a new order"""
        order = Order(
            order_id=self.generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price or 0,
            stop_price=stop_price,
            notes=notes
        )
        
        self.portfolio.orders.append(order)
        
        # Log order placement
        logger.info(f"Order placed: {order.order_id} - {side.value} {quantity} {symbol} @ {price}")
        
        return order
    
    def execute_order(self, order: Order, market_price: float) -> Dict:
        """Execute an order"""
        try:
            # Calculate execution price with slippage
            slippage_pct = self.metrics_config['slippage_pct'] / 100
            
            if order.side == OrderSide.BUY:
                execution_price = market_price * (1 + slippage_pct)
            else:
                execution_price = market_price * (1 - slippage_pct)
            
            # Calculate commission
            commission_pct = self.metrics_config['commission_pct'] / 100
            commission = order.quantity * execution_price * commission_pct
            
            # Check if we have enough cash for buy orders
            if order.side == OrderSide.BUY:
                required_cash = (order.quantity * execution_price) + commission
                if required_cash > self.portfolio.cash:
                    order.status = OrderStatus.REJECTED
                    return {
                        'success': False,
                        'error': f'Insufficient funds. Required: {required_cash:.2f}, Available: {self.portfolio.cash:.2f}'
                    }
                
                # Deduct cash
                self.portfolio.cash -= required_cash
                
                # Add position
                stop_loss = execution_price * (1 - self.config['stop_loss_pct'] / 100)
                take_profit = execution_price * (1 + self.config['target_pct'] / 100)
                
                self.portfolio.add_position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=execution_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
            else:  # SELL order
                # Check if we have the position
                position = self.portfolio.get_position(order.symbol)
                if not position or position.quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    return {
                        'success': False,
                        'error': f'Insufficient position. Required: {order.quantity}, Available: {position.quantity if position else 0}'
                    }
                
                # Close or reduce position
                if position.quantity == order.quantity:
                    result = self.portfolio.close_position(order.symbol, execution_price)
                else:
                    # Partial sell
                    position.quantity -= order.quantity
                    self.portfolio.cash += (order.quantity * execution_price) - commission
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.fill_timestamp = datetime.now()
            order.commission = commission
            order.slippage = execution_price - market_price
            
            # Add to history
            self.portfolio.order_history.append(order)
            
            # Record transaction
            transaction = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': execution_price,
                'commission': commission,
                'slippage': order.slippage
            }
            self.portfolio.transaction_history.append(transaction)
            
            # Log execution
            logger.info(f"Order executed: {order.order_id} - Filled {order.quantity} @ {execution_price:.2f}")
            
            return {
                'success': True,
                'order': order.to_dict(),
                'execution_price': execution_price,
                'commission': commission
            }
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return {'success': False, 'error': str(e)}
    
    def process_signal(self, symbol: str, signal: Dict, 
                      market_data: Dict) -> Optional[Order]:
        """Process a trading signal and place order if appropriate"""
        try:
            # Check if signal is actionable
            if signal['signal'] == 'HOLD':
                return None
            
            current_price = market_data.get('current_price', 0)
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return None
            
            # Check position limits
            if self.portfolio.positions_count >= self.config['max_positions']:
                logger.info(f"Max positions reached ({self.config['max_positions']})")
                return None
            
            # Calculate position size
            position_value = self.portfolio.total_value * \
                           (self.config['position_size_pct'] / 100) * \
                           signal.get('position_size', 1.0)
            
            quantity = int(position_value / current_price)
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity is 0 for {symbol}")
                return None
            
            # Place order based on signal
            if signal['signal'] == 'BUY':
                # Check if we already have a position
                if symbol in self.portfolio.positions:
                    logger.info(f"Already have position in {symbol}")
                    return None
                
                order = self.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    price=current_price,
                    notes=signal.get('reasoning', '')
                )
                
                # Execute immediately for market orders
                self.execute_order(order, current_price)
                return order
                
            elif signal['signal'] == 'SELL':
                # Check if we have a position to sell
                position = self.portfolio.get_position(symbol)
                if not position:
                    logger.info(f"No position to sell for {symbol}")
                    return None
                
                order = self.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    price=current_price,
                    notes=signal.get('reasoning', '')
                )
                
                # Execute immediately for market orders
                self.execute_order(order, current_price)
                return order
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
            return None
    
    def update_portfolio(self, market_data: Dict):
        """Update portfolio with latest market data"""
        # Extract prices
        prices = {}
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'current_price' in data:
                prices[symbol] = data['current_price']
        
        # Update position prices
        self.portfolio.update_prices(prices)
        
        # Check stop loss and take profit
        triggered = self.portfolio.check_stop_loss_take_profit(prices)
        
        if triggered:
            for trigger in triggered:
                logger.info(f"{trigger['type']} triggered for {trigger['symbol']} @ {trigger['price']}")
        
        # Record portfolio snapshot
        snapshot = self.portfolio.get_summary()
        snapshot['timestamp'] = datetime.now().isoformat()
        self.portfolio.portfolio_history.append(snapshot)
    
    def get_performance_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio.portfolio_history:
            return {}
        
        # Create DataFrame from portfolio history
        df = pd.DataFrame(self.portfolio.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        df['returns'] = df['total_value'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Basic metrics
        total_return = self.portfolio.total_pnl_pct / 100
        
        # Calculate daily returns
        if len(df) > 1:
            daily_returns = df['returns'].dropna()
            
            # Sharpe Ratio
            risk_free_rate = self.metrics_config['risk_free_rate'] / self.metrics_config['trading_days']
            excess_returns = daily_returns - risk_free_rate
            
            if daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(self.metrics_config['trading_days']) * \
                              (excess_returns.mean() / daily_returns.std())
            else:
                sharpe_ratio = 0
            
            # Max Drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Win rate
            closed = self.portfolio.closed_positions
            if closed:
                win_rate = len([p for p in closed if p['realized_pnl'] > 0]) / len(closed)
                avg_win = np.mean([p['realized_pnl'] for p in closed if p['realized_pnl'] > 0]) if any(p['realized_pnl'] > 0 for p in closed) else 0
                avg_loss = np.mean([abs(p['realized_pnl']) for p in closed if p['realized_pnl'] < 0]) if any(p['realized_pnl'] < 0 for p in closed) else 0
                profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0
            
            # Calmar Ratio
            if max_drawdown < 0:
                calmar_ratio = total_return / abs(max_drawdown)
            else:
                calmar_ratio = 0
            
            # Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    sortino_ratio = np.sqrt(self.metrics_config['trading_days']) * \
                                  (excess_returns.mean() / downside_std)
                else:
                    sortino_ratio = 0
            else:
                sortino_ratio = float('inf')
            
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
            profit_factor = 0
            calmar_ratio = 0
            sortino_ratio = 0
        
        metrics = {
            'total_return_pct': self.portfolio.total_pnl_pct,
            'total_pnl': self.portfolio.total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': len(self.portfolio.closed_positions),
            'open_positions': self.portfolio.positions_count,
            'best_trade': max([p['realized_pnl'] for p in self.portfolio.closed_positions], default=0),
            'worst_trade': min([p['realized_pnl'] for p in self.portfolio.closed_positions], default=0),
            'avg_trade_pnl': np.mean([p['realized_pnl'] for p in self.portfolio.closed_positions]) if self.portfolio.closed_positions else 0,
            'current_value': self.portfolio.total_value
        }
        
        # Compare with benchmark if provided
        if benchmark_returns is not None and len(benchmark_returns) == len(daily_returns):
            metrics['alpha'] = (daily_returns.mean() - benchmark_returns.mean()) * \
                              self.metrics_config['trading_days']
            
            # Beta
            if benchmark_returns.std() > 0:
                covariance = daily_returns.cov(benchmark_returns)
                metrics['beta'] = covariance / (benchmark_returns.std() ** 2)
            else:
                metrics['beta'] = 0
            
            # Information Ratio
            tracking_error = (daily_returns - benchmark_returns).std()
            if tracking_error > 0:
                metrics['information_ratio'] = \
                    (daily_returns.mean() - benchmark_returns.mean()) / tracking_error * \
                    np.sqrt(self.metrics_config['trading_days'])
            else:
                metrics['information_ratio'] = 0
        
        return metrics
    
    def save_state(self, filepath: str):
        """Save trading engine state to file"""
        state = {
            'mode': self.mode,
            'portfolio': self.portfolio.get_summary(),
            'closed_positions': self.portfolio.closed_positions,
            'transaction_history': self.portfolio.transaction_history,
            'order_history': [order.to_dict() for order in self.portfolio.order_history],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Trading state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load trading engine state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore portfolio state
            self.portfolio.closed_positions = state['closed_positions']
            self.portfolio.transaction_history = state['transaction_history']
            
            logger.info(f"Trading state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
