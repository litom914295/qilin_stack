"""
回测引擎模块
提供策略回测、性能评估和风险分析功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)


# 复用trade_executor中的枚举，避免枚举不一致导致的比较失败
from app.core.trade_executor import OrderType, OrderSide, OrderStatus






@dataclass
class Order:
    """订单数据结构"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    slippage: float = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Trade:
    """成交记录"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    pnl: float = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """持仓信息 (支持T+1规则)"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float
    cost_basis: float
    last_update: datetime
    # T+1相关字段
    purchase_date: datetime  # 购入日期
    available_quantity: float = 0  # 可卖数量 (除当日买入外)
    frozen_quantity: float = 0  # 冻结数量 (当日买入,不可卖)
    metadata: Dict = field(default_factory=dict)


class Portfolio:
    """投资组合管理 (支持T+1规则)"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.current_date: Optional[datetime] = None  # 当前交易日
        
    def update_position(self, symbol: str, quantity: float, price: float, timestamp: datetime):
        """更新持仓 (支持T+1规则)"""
        if symbol in self.positions:
            position = self.positions[symbol]
            if quantity > 0:  # 买入
                total_cost = position.avg_price * position.quantity + price * quantity
                position.quantity += quantity
                position.avg_price = total_cost / position.quantity if position.quantity > 0 else 0
                
                # T+1: 当日买入的股票冻结,次日才能卖出
                position.frozen_quantity += quantity
                position.purchase_date = timestamp
            else:  # 卖出
                # 验证T+1规则：只能卖出可用数量
                sell_qty = abs(quantity)
                if sell_qty > position.available_quantity:
                    raise ValueError(
                        f"T+1限制: {symbol} 可卖数量={position.available_quantity}, "
                        f"请求卖出={sell_qty}, 冻结数量={position.frozen_quantity}"
                    )
                
                position.quantity += quantity  # quantity为负数
                position.available_quantity -= sell_qty
                
                if position.quantity <= 0:
                    del self.positions[symbol]
            
            if symbol in self.positions:
                position.last_update = timestamp
                position.current_price = price
                position.market_value = position.quantity * price
                position.unrealized_pnl = (price - position.avg_price) * position.quantity
        else:
            if quantity > 0:
                # T+1: 新买入的股票全部冻结
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    market_value=quantity * price,
                    cost_basis=quantity * price,
                    last_update=timestamp,
                    purchase_date=timestamp,
                    available_quantity=0,  # 当日买入,不可卖
                    frozen_quantity=quantity
                )
    
    def unfreeze_positions(self, current_date: datetime):
        """
        解冻持仓 (次日调用)
        T+1规则: 将上个交易日买入的股票转为可用
        
        Args:
            current_date: 当前日期
        """
        for symbol, position in self.positions.items():
            # 如果不是同一天,解冻之前的冻结数量
            if position.purchase_date and position.purchase_date.date() < current_date.date():
                position.available_quantity += position.frozen_quantity
                position.frozen_quantity = 0
    def get_total_value(self) -> float:
        """获取总资产价值"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_returns(self) -> float:
        """获取收益率"""
        total_value = self.get_total_value()
        return (total_value - self.initial_capital) / self.initial_capital


class BacktestEngine:
    """回测引擎主类"""
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.0001,
                 min_commission: float = 5):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            min_commission: 最低手续费
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission
        
        self.portfolio = Portfolio(initial_capital)
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.performance_metrics: Dict = {}
        
        self.current_timestamp: Optional[datetime] = None
        self.data: Optional[pd.DataFrame] = None
        
    def set_data(self, data: pd.DataFrame):
        """设置回测数据"""
        self.data = data
        logger.info(f"加载回测数据: {len(data)}条记录")
        
    def place_order(self, order: Order) -> str:
        """下单"""
        import uuid
        order.order_id = str(uuid.uuid4())
        order.timestamp = self.current_timestamp
        
        # 检查订单有效性
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return order.order_id
        
        # 市价单立即执行
        if order.order_type == OrderType.MARKET:
            self._execute_order(order)
        else:
            # 限价单等待执行
            self.pending_orders.append(order)
        
        self.order_history.append(order)
        return order.order_id
    
    def _validate_order(self, order: Order) -> bool:
        """验证订单有效性 (含 T+1 规则)"""
        if order.side == OrderSide.BUY:
            # 检查资金是否充足
            required_cash = order.quantity * (order.price or self._get_current_price(order.symbol))
            required_cash *= (1 + self.commission_rate + self.slippage_rate)
            
            if required_cash > self.portfolio.cash:
                logger.warning(f"资金不足: 需要{required_cash:.2f}, 可用{self.portfolio.cash:.2f}")
                return False
        else:
            # 检查持仓是否充足
            if order.symbol not in self.portfolio.positions:
                logger.warning(f"无持仓: {order.symbol}")
                return False
            
            position = self.portfolio.positions[order.symbol]
            
            # T+1规则: 只能卖出可用数量 (不包含当日买入)
            if position.available_quantity < order.quantity:
                logger.warning(
                    f"T+1限制: {order.symbol} 可卖数量={position.available_quantity}, "
                    f"请求卖出={order.quantity}, "
                    f"冻结数量={position.frozen_quantity} (当日买入不可卖)"
                )
                return False
            
            if position.quantity < order.quantity:
                logger.warning(f"持仓不足: {order.symbol}, 需要{order.quantity}, 可用{position.quantity}")
                return False
        
        return True
    
    def _execute_order(self, order: Order):
        """执行订单"""
        current_price = self._get_current_price(order.symbol)
        
        # 计算滑点
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + self.slippage_rate)
        else:
            execution_price = current_price * (1 - self.slippage_rate)
        
        # 计算手续费
        commission = max(
            order.quantity * execution_price * self.commission_rate,
            self.min_commission
        )
        
        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission
        order.slippage = abs(execution_price - current_price) * order.quantity
        
        # 更新资金和持仓
        if order.side == OrderSide.BUY:
            total_cost = order.quantity * execution_price + commission
            self.portfolio.cash -= total_cost
            self.portfolio.update_position(
                order.symbol, 
                order.quantity, 
                execution_price, 
                self.current_timestamp
            )
        else:
            total_proceeds = order.quantity * execution_price - commission
            self.portfolio.cash += total_proceeds
            self.portfolio.update_position(
                order.symbol, 
                -order.quantity, 
                execution_price, 
                self.current_timestamp
            )
        
        # 记录成交
        trade = Trade(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=self.current_timestamp,
            commission=commission,
            slippage=order.slippage
        )
        self.portfolio.trades.append(trade)
        
        logger.debug(f"订单执行: {order.symbol} {order.side.value} {order.quantity}@{execution_price:.2f}")
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        if self.data is not None and self.current_timestamp:
            # 从数据中获取价格
            mask = (self.data['symbol'] == symbol) & (self.data.index == self.current_timestamp)
            if mask.any():
                return self.data.loc[mask, 'close'].iloc[0]
        
        # 返回默认价格或抛出异常
        raise ValueError(f"无法获取{symbol}在{self.current_timestamp}的价格")
    
    def run_backtest(self, strategy_func, start_date: datetime, end_date: datetime):
        """
        运行回测
        
        Args:
            strategy_func: 策略函数，接收当前数据和portfolio，返回订单列表
            start_date: 开始日期
            end_date: 结束日期
        """
        logger.info(f"开始回测: {start_date} - {end_date}")
        
        # 获取回测期间的交易日
        trading_days = pd.date_range(start_date, end_date, freq='B')
        
        for date in trading_days:
            self.current_timestamp = date
            self.portfolio.current_date = date
            
            # T+1规则: 每日开盘前解冻上个交易日买入的股票
            self.portfolio.unfreeze_positions(date)
            
            # 更新持仓市值
            self._update_positions()
            
            # 处理挂单
            self._process_pending_orders()
            
            # 获取当天数据
            daily_data = self._get_daily_data(date)
            
            # 执行策略
            orders = strategy_func(daily_data, self.portfolio)
            
            # 处理策略产生的订单
            for order in orders:
                self.place_order(order)
            
            # 记录每日净值
            total_value = self.portfolio.get_total_value()
            self.portfolio.equity_curve.append((date, total_value))
            
            # 计算日收益率
            if len(self.portfolio.equity_curve) > 1:
                prev_value = self.portfolio.equity_curve[-2][1]
                daily_return = (total_value - prev_value) / prev_value
                self.portfolio.daily_returns.append(daily_return)
        
        # 计算性能指标
        self._calculate_performance_metrics()
        
        logger.info(f"回测完成，总收益率: {self.performance_metrics.get('total_return', 0):.2%}")
    
    def _update_positions(self):
        """更新持仓市值"""
        for symbol, position in self.portfolio.positions.items():
            try:
                current_price = self._get_current_price(symbol)
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                position.last_update = self.current_timestamp
            except ValueError:
                logger.warning(f"无法更新{symbol}的持仓市值")
    
    def _process_pending_orders(self):
        """处理挂单"""
        filled_orders = []
        
        for order in self.pending_orders:
            current_price = self._get_current_price(order.symbol)
            
            # 检查限价单是否可以成交
            if order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    self._execute_order(order)
                    filled_orders.append(order)
            
            # 检查止损单
            elif order.order_type == OrderType.STOP:
                if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
                   (order.side == OrderSide.SELL and current_price <= order.stop_price):
                    self._execute_order(order)
                    filled_orders.append(order)
        
        # 移除已成交的订单
        for order in filled_orders:
            self.pending_orders.remove(order)
    
    def _get_daily_data(self, date: datetime) -> pd.DataFrame:
        """获取指定日期的数据"""
        if self.data is not None:
            return self.data[self.data.index == date]
        return pd.DataFrame()
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.portfolio.equity_curve:
            return
        
        # 转换为DataFrame便于计算
        equity_df = pd.DataFrame(
            self.portfolio.equity_curve, 
            columns=['date', 'equity']
        ).set_index('date')
        
        # 总收益率
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        days = (equity_df.index[-1] - equity_df.index[0]).days
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算日收益率统计
        returns = pd.Series(self.portfolio.daily_returns)
        
        # Sharpe比率（假设无风险利率为3%）
        risk_free_rate = 0.03 / 252
        sharpe_ratio = np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std() if returns.std() > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        trades = self.portfolio.trades
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # 盈亏比
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in trades if t.pnl < 0]
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 保存指标
        self.performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(trades),
            'avg_daily_return': returns.mean(),
            'daily_volatility': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        return {
            '基本指标': {
                '总收益率': f"{self.performance_metrics.get('total_return', 0):.2%}",
                '年化收益率': f"{self.performance_metrics.get('annual_return', 0):.2%}",
                '最大回撤': f"{self.performance_metrics.get('max_drawdown', 0):.2%}",
                'Sharpe比率': f"{self.performance_metrics.get('sharpe_ratio', 0):.3f}",
                'Calmar比率': f"{self.performance_metrics.get('calmar_ratio', 0):.3f}"
            },
            '交易统计': {
                '总交易次数': self.performance_metrics.get('total_trades', 0),
                '胜率': f"{self.performance_metrics.get('win_rate', 0):.2%}",
                '盈亏比': f"{self.performance_metrics.get('profit_factor', 0):.2f}"
            },
            '风险指标': {
                '日均收益率': f"{self.performance_metrics.get('avg_daily_return', 0):.3%}",
                '日波动率': f"{self.performance_metrics.get('daily_volatility', 0):.3%}",
                '偏度': f"{self.performance_metrics.get('skewness', 0):.3f}",
                '峰度': f"{self.performance_metrics.get('kurtosis', 0):.3f}"
            },
            '资金状况': {
                '初始资金': f"{self.initial_capital:,.2f}",
                '最终资金': f"{self.portfolio.get_total_value():,.2f}",
                '当前现金': f"{self.portfolio.cash:,.2f}",
                '持仓市值': f"{sum(p.market_value for p in self.portfolio.positions.values()):,.2f}"
            }
        }
    
    def export_results(self, filepath: str):
        """导出回测结果"""
        results = {
            'performance_metrics': self.performance_metrics,
            'equity_curve': [(str(d), v) for d, v in self.portfolio.equity_curve],
            'trades': [
                {
                    'order_id': t.order_id,
                    'symbol': t.symbol,
                    'side': t.side.value,
                    'quantity': t.quantity,
                    'price': t.price,
                    'timestamp': str(t.timestamp),
                    'commission': t.commission,
                    'slippage': t.slippage,
                    'pnl': t.pnl
                }
                for t in self.portfolio.trades
            ],
            'final_positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'market_value': pos.market_value
                }
                for symbol, pos in self.portfolio.positions.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"回测结果已导出至: {filepath}")


class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
        self.optimization_results = []
    
    def optimize_parameters(self, 
                           strategy_class,
                           param_ranges: Dict[str, List],
                           optimization_target: str = 'sharpe_ratio',
                           n_trials: int = 100):
        """
        优化策略参数
        
        Args:
            strategy_class: 策略类
            param_ranges: 参数范围
            optimization_target: 优化目标
            n_trials: 优化次数
        """
        try:
            import optuna
            
            def objective(trial):
                # 采样参数
                params = {}
                for param_name, param_range in param_ranges.items():
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, *param_range)
                    elif isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name, *param_range)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                
                # 创建策略实例
                strategy = strategy_class(**params)
                
                # 运行回测
                self.backtest_engine.run_backtest(
                    strategy.generate_signals,
                    strategy.start_date,
                    strategy.end_date
                )
                
                # 获取优化目标值
                target_value = self.backtest_engine.performance_metrics.get(optimization_target, 0)
                
                # 记录结果
                self.optimization_results.append({
                    'params': params,
                    'target_value': target_value,
                    'metrics': self.backtest_engine.performance_metrics.copy()
                })
                
                return target_value
            
            # 创建优化研究
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # 返回最佳参数
            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'all_results': self.optimization_results
            }
            
        except ImportError:
            logger.error("需要安装optuna库: pip install optuna")
            return None


# 示例策略
class SimpleMovingAverageStrategy:
    """简单移动平均策略示例"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        self.positions = {}
    
    def generate_signals(self, data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        """生成交易信号"""
        orders = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            
            if len(symbol_data) < self.long_window:
                continue
            
            # 计算移动平均
            short_ma = symbol_data['close'].rolling(window=self.short_window).mean().iloc[-1]
            long_ma = symbol_data['close'].rolling(window=self.long_window).mean().iloc[-1]
            current_price = symbol_data['close'].iloc[-1]
            
            # 生成信号
            if short_ma > long_ma and symbol not in portfolio.positions:
                # 金叉买入
                quantity = int(portfolio.cash * 0.1 / current_price)  # 使用10%的现金
                if quantity > 0:
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity
                    ))
            
            elif short_ma < long_ma and symbol in portfolio.positions:
                # 死叉卖出
                position = portfolio.positions[symbol]
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity
                ))
        
        return orders


if __name__ == "__main__":
    # 示例用法
    engine = BacktestEngine(
        initial_capital=1000000,
        commission_rate=0.0003,
        slippage_rate=0.0001
    )
    
    # 加载数据（示例）
    # data = pd.read_csv("historical_data.csv", index_col='date', parse_dates=True)
    # engine.set_data(data)
    
    # 创建策略
    strategy = SimpleMovingAverageStrategy(short_window=20, long_window=50)
    
    # 运行回测
    # engine.run_backtest(
    #     strategy.generate_signals,
    #     start_date=datetime(2023, 1, 1),
    #     end_date=datetime(2023, 12, 31)
    # )
    
    # 获取性能报告
    # report = engine.get_performance_report()
    # print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 导出结果
    # engine.export_results("backtest_results.json")