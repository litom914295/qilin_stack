"""
麒麟量化系统 - 回测引擎
支持策略回测、性能评估、风险分析
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"


@dataclass
class Order:
    """订单类"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    slippage: float = 0
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d%H%M%S%f')}"


@dataclass
class Trade:
    """成交记录"""
    order: Order
    execution_price: float
    execution_quantity: float
    execution_time: datetime
    commission: float
    slippage: float
    
    @property
    def value(self) -> float:
        """成交金额"""
        return self.execution_price * self.execution_quantity


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    
    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """成本"""
        return self.quantity * self.avg_price
    
    def update_price(self, price: float):
        """更新价格和未实现盈亏"""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity


class Portfolio:
    """投资组合"""
    
    def __init__(self, initial_capital: float = 1000000):
        """
        初始化投资组合
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        
    @property
    def total_value(self) -> float:
        """总资产"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        """总收益率"""
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    def add_trade(self, trade: Trade):
        """添加成交记录"""
        self.trades.append(trade)
        
        # 更新现金
        if trade.order.side == OrderSide.BUY:
            self.cash -= trade.value + trade.commission
        else:
            self.cash += trade.value - trade.commission
        
        # 更新持仓
        symbol = trade.order.symbol
        
        if symbol not in self.positions:
            if trade.order.side == OrderSide.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.execution_quantity,
                    avg_price=trade.execution_price,
                    current_price=trade.execution_price
        else:
            position = self.positions[symbol]
            
            if trade.order.side == OrderSide.BUY:
                # 加仓
                total_cost = (position.quantity * position.avg_price + 
                            trade.execution_quantity * trade.execution_price)
                position.quantity += trade.execution_quantity
                position.avg_price = total_cost / position.quantity
            else:
                # 减仓
                position.quantity -= trade.execution_quantity
                
                # 计算已实现盈亏
                realized = (trade.execution_price - position.avg_price) * trade.execution_quantity
                position.realized_pnl += realized
                
                # 如果全部卖出，删除持仓
                if position.quantity <= 0:
                    del self.positions[symbol]
    
    def update_prices(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
    
    def record_equity(self, timestamp: datetime):
        """记录权益曲线"""
        self.timestamps.append(timestamp)
        self.equity_curve.append(self.total_value)
        
        # 计算日收益率
        if len(self.equity_curve) >= 2:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        commission: float = 0.0003,
        slippage: float = 0.001,
        min_commission: float = 5
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点
            min_commission: 最小手续费
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission
        self.slippage_rate = slippage
        self.min_commission = min_commission
        
        self.portfolio = Portfolio(initial_capital)
        self.pending_orders: List[Order] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_date: Optional[datetime] = None
        self.strategy = None
        
    def load_data(self, symbol: str, data: pd.DataFrame):
        """
        加载历史数据
        
        Args:
            symbol: 股票代码
            data: 历史数据
        """
        self.historical_data[symbol] = data
        logger.info(f"加载{symbol}数据: {len(data)}条")
    
    def set_strategy(self, strategy):
        """设置策略"""
        self.strategy = strategy
        strategy.set_engine(self)
    
    def submit_order(self, order: Order):
        """提交订单"""
        self.pending_orders.append(order)
        logger.debug(f"订单提交: {order.order_id}")
    
    def _execute_order(self, order: Order, current_bar: pd.Series) -> Optional[Trade]:
        """
        执行订单
        
        Args:
            order: 订单
            current_bar: 当前K线
            
        Returns:
            成交记录
        """
        # 计算执行价格
        if order.order_type == OrderType.MARKET:
            # 市价单
            if order.side == OrderSide.BUY:
                exec_price = current_bar['open'] * (1 + self.slippage_rate)
            else:
                exec_price = current_bar['open'] * (1 - self.slippage_rate)
        
        elif order.order_type == OrderType.LIMIT:
            # 限价单
            if order.side == OrderSide.BUY:
                if current_bar['low'] <= order.price:
                    exec_price = min(order.price, current_bar['open'])
                else:
                    return None  # 未成交
            else:
                if current_bar['high'] >= order.price:
                    exec_price = max(order.price, current_bar['open'])
                else:
                    return None  # 未成交
        
        elif order.order_type == OrderType.STOP:
            # 止损单
            if order.side == OrderSide.BUY:
                if current_bar['high'] >= order.stop_price:
                    exec_price = max(order.stop_price, current_bar['open']) * (1 + self.slippage_rate)
                else:
                    return None
            else:
                if current_bar['low'] <= order.stop_price:
                    exec_price = min(order.stop_price, current_bar['open']) * (1 - self.slippage_rate)
                else:
                    return None
        else:
            return None
        
        # 计算手续费
        commission = max(
            exec_price * order.quantity * self.commission_rate,
            self.min_commission
        
        # 检查资金是否充足
        if order.side == OrderSide.BUY:
            required_capital = exec_price * order.quantity + commission
            if self.portfolio.cash < required_capital:
                logger.warning(f"资金不足: 需要{required_capital}, 可用{self.portfolio.cash}")
                return None
        
        # 创建成交记录
        trade = Trade(
            order=order,
            execution_price=exec_price,
            execution_quantity=order.quantity,
            execution_time=self.current_date,
            commission=commission,
            slippage=abs(exec_price - current_bar['open'])
        
        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = exec_price
        
        return trade
    
    def run(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbols: 股票列表
            
        Returns:
            回测结果
        """
        if not self.strategy:
            raise ValueError("未设置策略")
        
        if symbols is None:
            symbols = list(self.historical_data.keys())
        
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 获取所有交易日期
        all_dates = set()
        for symbol in symbols:
            if symbol in self.historical_data:
                dates = self.historical_data[symbol].index
                all_dates.update(dates[(dates >= start_date) & (dates <= end_date)])
        
        all_dates = sorted(all_dates)
        
        logger.info(f"开始回测: {start_date} - {end_date}, 共{len(all_dates)}个交易日")
        
        # 遍历每个交易日
        for date in all_dates:
            self.current_date = date
            
            # 获取当日数据
            current_data = {}
            for symbol in symbols:
                if symbol in self.historical_data:
                    df = self.historical_data[symbol]
                    if date in df.index:
                        current_data[symbol] = df.loc[date]
            
            # 处理待执行订单
            new_pending = []
            for order in self.pending_orders:
                if order.symbol in current_data:
                    trade = self._execute_order(order, current_data[order.symbol])
                    if trade:
                        self.portfolio.add_trade(trade)
                        logger.debug(f"订单成交: {order.order_id} @ {trade.execution_price}")
                    else:
                        new_pending.append(order)
                else:
                    new_pending.append(order)
            
            self.pending_orders = new_pending
            
            # 更新持仓价格
            prices = {symbol: data['close'] for symbol, data in current_data.items()}
            self.portfolio.update_prices(prices)
            
            # 调用策略
            self.strategy.on_bar(date, current_data)
            
            # 记录权益
            self.portfolio.record_equity(date)
        
        # 生成回测报告
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        equity_curve = pd.Series(
            self.portfolio.equity_curve[1:],
            index=self.portfolio.timestamps
        
        daily_returns = pd.Series(self.portfolio.daily_returns)
        
        # 计算性能指标
        metrics = PerformanceMetrics.calculate(
            equity_curve,
            daily_returns,
            self.initial_capital
        
        # 生成报告
        report = {
            'metrics': metrics,
            'equity_curve': equity_curve.to_dict(),
            'trades': [self._trade_to_dict(t) for t in self.portfolio.trades],
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.portfolio.positions.items()
            },
            'final_value': self.portfolio.total_value,
            'total_return': self.portfolio.total_return
        }
        
        return report
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """将交易转换为字典"""
        return {
            'symbol': trade.order.symbol,
            'side': trade.order.side.value,
            'quantity': trade.execution_quantity,
            'price': trade.execution_price,
            'time': trade.execution_time.isoformat(),
            'commission': trade.commission,
            'slippage': trade.slippage,
            'value': trade.value
        }


class PerformanceMetrics:
    """性能指标计算"""
    
    @staticmethod
    def calculate(
        equity_curve: pd.Series,
        returns: pd.Series,
        initial_capital: float,
        risk_free_rate: float = 0.03
    ) -> Dict[str, float]:
        """
        计算性能指标
        
        Args:
            equity_curve: 权益曲线
            returns: 日收益率
            initial_capital: 初始资金
            risk_free_rate: 无风险利率
            
        Returns:
            性能指标字典
        """
        # 基础指标
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        
        # 年化指标
        days = len(returns)
        years = days / 252
        
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0
        
        # 波动率
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0
        
        # 夏普比率
        if volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        # Calmar比率
        if abs(max_drawdown) > 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # 胜率
        winning_trades = [r for r in returns if r > 0]
        if len(returns) > 0:
            win_rate = len(winning_trades) / len(returns)
        else:
            win_rate = 0
        
        # 盈亏比
        if winning_trades:
            avg_win = np.mean(winning_trades)
            losing_trades = [r for r in returns if r < 0]
            if losing_trades:
                avg_loss = abs(np.mean(losing_trades))
                profit_factor = avg_win / avg_loss
            else:
                profit_factor = float('inf')
        else:
            profit_factor = 0
        
        # Sortino比率
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std > 0:
                sortino_ratio = (annual_return - risk_free_rate) / downside_std
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = sharpe_ratio  # 没有负收益时等于Sharpe
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(returns)
        }
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """计算最大回撤"""
        if len(equity_curve) < 2:
            return 0
        
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        
        return float(drawdown.min())


class BaseStrategy:
    """策略基类"""
    
    def __init__(self):
        self.engine: Optional[BacktestEngine] = None
        
    def set_engine(self, engine: BacktestEngine):
        """设置回测引擎"""
        self.engine = engine
    
    def on_bar(self, timestamp: datetime, data: Dict[str, pd.Series]):
        """K线回调"""
        raise NotImplementedError
    
    def buy(self, symbol: str, quantity: float, order_type: OrderType = OrderType.MARKET, **kwargs):
        """买入"""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            **kwargs
        self.engine.submit_order(order)
    
    def sell(self, symbol: str, quantity: float, order_type: OrderType = OrderType.MARKET, **kwargs):
        """卖出"""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            **kwargs
        self.engine.submit_order(order)


# 示例策略
class SimpleMovingAverageStrategy(BaseStrategy):
    """简单均线策略"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.positions = set()
        
    def on_bar(self, timestamp: datetime, data: Dict[str, pd.Series]):
        """处理K线"""
        for symbol, bar in data.items():
            # 获取历史数据
            if symbol not in self.engine.historical_data:
                continue
            
            hist = self.engine.historical_data[symbol]
            
            # 确保有足够的历史数据
            current_idx = hist.index.get_loc(timestamp)
            if current_idx < self.slow_period:
                continue
            
            # 计算均线
            recent_data = hist.iloc[max(0, current_idx - self.slow_period):current_idx + 1]
            fast_ma = recent_data['close'].tail(self.fast_period).mean()
            slow_ma = recent_data['close'].mean()
            
            current_price = bar['close']
            
            # 生成信号
            if fast_ma > slow_ma and symbol not in self.positions:
                # 金叉买入
                quantity = 1000  # 固定数量
                self.buy(symbol, quantity)
                self.positions.add(symbol)
                logger.info(f"{timestamp}: 买入 {symbol} @ {current_price}")
                
            elif fast_ma < slow_ma and symbol in self.positions:
                # 死叉卖出
                if symbol in self.engine.portfolio.positions:
                    quantity = self.engine.portfolio.positions[symbol].quantity
                    self.sell(symbol, quantity)
                    self.positions.discard(symbol)
                    logger.info(f"{timestamp}: 卖出 {symbol} @ {current_price}")


async def run_example():
    """运行示例"""
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=1000000,
        commission=0.0003,
        slippage=0.001
    
    # 生成模拟数据
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    for symbol in ['000001', '000002']:
        prices = 100 + np.random.randn(len(dates)).cumsum()
        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        engine.load_data(symbol, df)
    
    # 创建策略
    strategy = SimpleMovingAverageStrategy(fast_period=10, slow_period=30)
    engine.set_strategy(strategy)
    
    # 运行回测
    report = engine.run(
        start_date='2023-01-01',
        end_date='2024-01-01'
    
    # 打印结果
    print("\n=== 回测结果 ===")
    print(f"总收益率: {report['total_return']:.2%}")
    print(f"年化收益率: {report['metrics']['annual_return']:.2%}")
    print(f"最大回撤: {report['metrics']['max_drawdown']:.2%}")
    print(f"夏普比率: {report['metrics']['sharpe_ratio']:.2f}")
    print(f"胜率: {report['metrics']['win_rate']:.2%}")
    print(f"交易次数: {report['metrics']['total_trades']}")
    
    # 保存报告
    with open('backtest_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    
    # 运行示例
    asyncio.run(run_example())