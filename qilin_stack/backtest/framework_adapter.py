"""
回测框架集成适配器 (Backtest Framework Adapter)
统一接口对接主流回测框架

支持的框架：
1. Backtrader - 成熟的Python回测框架
2. VectorBT - 高性能向量化回测
3. Zipline - Quantopian开源框架
4. Custom - 自定义框架
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class FrameworkType(Enum):
    """框架类型"""
    BACKTRADER = "Backtrader"
    VECTORBT = "VectorBT"
    ZIPLINE = "Zipline"
    CUSTOM = "Custom"


class OrderSide(Enum):
    """订单方向"""
    BUY = "买入"
    SELL = "卖出"


class OrderType(Enum):
    """订单类型"""
    MARKET = "市价单"
    LIMIT = "限价单"
    STOP = "止损单"
    STOP_LIMIT = "止损限价单"


@dataclass
class Order:
    """统一订单结构"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None      # 限价单需要
    stop_price: Optional[float] = None # 止损单需要
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    status: str = "pending"
    filled_quantity: float = 0.0
    filled_price: float = 0.0


@dataclass
class Position:
    """统一持仓结构"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class PerformanceMetrics:
    """统一绩效指标"""
    # 收益指标
    total_return: float           # 总收益率
    annual_return: float          # 年化收益率
    cumulative_return: float      # 累计收益率
    
    # 风险指标
    volatility: float             # 波动率
    sharpe_ratio: float           # 夏普比率
    sortino_ratio: float          # 索提诺比率
    max_drawdown: float           # 最大回撤
    max_drawdown_duration: int    # 最大回撤持续期
    
    # 交易指标
    total_trades: int             # 总交易次数
    win_rate: float               # 胜率
    profit_factor: float          # 盈亏比
    avg_win: float                # 平均盈利
    avg_loss: float               # 平均亏损
    
    # 日期范围
    start_date: datetime
    end_date: datetime
    
    # 原始数据
    equity_curve: pd.Series       # 权益曲线
    trades: List[Dict]            # 交易记录


class BacktestAdapter(ABC):
    """回测框架适配器抽象基类"""
    
    def __init__(self, framework_type: FrameworkType):
        self.framework_type = framework_type
        self.engine = None
        self._initialized = False
    
    @abstractmethod
    def initialize(self, **kwargs):
        """初始化框架"""
        pass
    
    @abstractmethod
    def add_data(self, data: pd.DataFrame, symbol: str):
        """添加数据"""
        pass
    
    @abstractmethod
    def set_strategy(self, strategy_func: Callable, params: Dict = None):
        """设置策略"""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """下单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取持仓"""
        pass
    
    @abstractmethod
    def get_account_value(self) -> float:
        """获取账户价值"""
        pass
    
    @abstractmethod
    def run(self) -> PerformanceMetrics:
        """运行回测"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取绩效指标"""
        pass


class BacktraderAdapter(BacktestAdapter):
    """Backtrader框架适配器"""
    
    def __init__(self):
        super().__init__(FrameworkType.BACKTRADER)
        self.cerebro = None
        self.strategy_class = None
        self.data_feeds = {}
    
    def initialize(self, **kwargs):
        """初始化Backtrader"""
        try:
            import backtrader as bt
        except ImportError:
            raise ImportError("需要安装 backtrader: pip install backtrader")
        
        self.cerebro = bt.Cerebro()
        
        # 设置初始资金
        initial_cash = kwargs.get('initial_cash', 1000000)
        self.cerebro.broker.setcash(initial_cash)
        
        # 设置手续费
        commission = kwargs.get('commission', 0.001)
        self.cerebro.broker.setcommission(commission=commission)
        
        self._initialized = True
        print(f"✅ Backtrader初始化完成（初始资金: {initial_cash:,.0f}）")
    
    def add_data(self, data: pd.DataFrame, symbol: str):
        """添加数据"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize()")
        
        import backtrader as bt
        
        # 确保索引为日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"数据必须包含列: {required_cols}")
        
        # 创建数据源
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # 使用索引作为日期
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data_feed, name=symbol)
        self.data_feeds[symbol] = data
        
        print(f"✅ 添加数据: {symbol} (共{len(data)}条)")
    
    def set_strategy(self, strategy_func: Callable, params: Dict = None):
        """设置策略"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize()")
        
        import backtrader as bt
        
        # 动态创建策略类
        class AdapterStrategy(bt.Strategy):
            def __init__(self):
                self.strategy_func = strategy_func
                self.params = params or {}
            
            def next(self):
                # 调用用户策略函数
                self.strategy_func(self, self.params)
        
        self.strategy_class = AdapterStrategy
        self.cerebro.addstrategy(AdapterStrategy)
        
        print(f"✅ 策略设置完成")
    
    def place_order(self, order: Order) -> str:
        """下单（在策略中调用）"""
        # Backtrader的下单通常在策略内部调用
        # 这里提供一个简化的接口
        raise NotImplementedError("Backtrader的下单需要在策略内部调用 self.buy() 或 self.sell()")
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        raise NotImplementedError("Backtrader的撤单需要在策略内部调用 self.cancel()")
    
    def get_positions(self) -> List[Position]:
        """获取持仓"""
        # 需要在策略运行时获取
        raise NotImplementedError("请在策略内部使用 self.getposition()")
    
    def get_account_value(self) -> float:
        """获取账户价值"""
        if self.cerebro:
            return self.cerebro.broker.getvalue()
        return 0.0
    
    def run(self) -> PerformanceMetrics:
        """运行回测"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize()")
        
        print("\n🚀 开始回测...")
        initial_value = self.cerebro.broker.getvalue()
        print(f"初始资金: {initial_value:,.2f}")
        
        # 运行回测
        results = self.cerebro.run()
        
        final_value = self.cerebro.broker.getvalue()
        print(f"最终资金: {final_value:,.2f}")
        print(f"收益: {(final_value - initial_value):,.2f} ({(final_value/initial_value - 1)*100:.2f}%)")
        
        # 获取绩效指标
        metrics = self.get_performance_metrics()
        
        return metrics
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取绩效指标"""
        # Backtrader没有内置的完整绩效分析
        # 这里提供一个简化版本
        
        initial_value = 1000000  # 假设初始资金
        final_value = self.get_account_value()
        
        total_return = (final_value / initial_value) - 1
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=0.0,  # 需要计算
            cumulative_return=total_return,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            equity_curve=pd.Series(),
            trades=[]
        )
    
    def plot(self):
        """绘制回测结果"""
        if self.cerebro:
            self.cerebro.plot()


class VectorBTAdapter(BacktestAdapter):
    """VectorBT框架适配器（向量化高性能）"""
    
    def __init__(self):
        super().__init__(FrameworkType.VECTORBT)
        self.portfolio = None
        self.data = {}
    
    def initialize(self, **kwargs):
        """初始化VectorBT"""
        try:
            import vectorbt as vbt
        except ImportError:
            raise ImportError("需要安装 vectorbt: pip install vectorbt")
        
        self.initial_cash = kwargs.get('initial_cash', 1000000)
        self.commission = kwargs.get('commission', 0.001)
        
        self._initialized = True
        print(f"✅ VectorBT初始化完成（初始资金: {self.initial_cash:,.0f}）")
    
    def add_data(self, data: pd.DataFrame, symbol: str):
        """添加数据"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize()")
        
        self.data[symbol] = data
        print(f"✅ 添加数据: {symbol} (共{len(data)}条)")
    
    def set_strategy(self, strategy_func: Callable, params: Dict = None):
        """设置策略（VectorBT使用向量化信号）"""
        # VectorBT需要预先计算所有信号
        # 这里strategy_func应该返回买入/卖出信号
        self.strategy_func = strategy_func
        self.strategy_params = params or {}
        
        print(f"✅ 策略设置完成")
    
    def place_order(self, order: Order) -> str:
        """下单"""
        raise NotImplementedError("VectorBT使用向量化信号，不支持单次下单")
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        raise NotImplementedError("VectorBT使用向量化信号，不支持撤单")
    
    def get_positions(self) -> List[Position]:
        """获取持仓"""
        if self.portfolio is None:
            return []
        
        # 需要从portfolio中提取
        return []
    
    def get_account_value(self) -> float:
        """获取账户价值"""
        if self.portfolio is None:
            return self.initial_cash
        
        return self.portfolio.value().iloc[-1]
    
    def run(self) -> PerformanceMetrics:
        """运行回测"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize()")
        
        import vectorbt as vbt
        
        print("\n🚀 开始向量化回测...")
        
        # 获取数据
        if len(self.data) == 0:
            raise ValueError("请先添加数据")
        
        # 假设只有一个标的
        symbol = list(self.data.keys())[0]
        data = self.data[symbol]
        
        # 调用策略生成信号
        signals = self.strategy_func(data, self.strategy_params)
        entries = signals['entries']
        exits = signals['exits']
        
        # 创建投资组合
        self.portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.commission
        )
        
        print(f"初始资金: {self.initial_cash:,.2f}")
        final_value = self.portfolio.value().iloc[-1]
        print(f"最终资金: {final_value:,.2f}")
        print(f"收益: {(final_value - self.initial_cash):,.2f} "
              f"({(final_value/self.initial_cash - 1)*100:.2f}%)")
        
        # 获取绩效指标
        metrics = self.get_performance_metrics()
        
        return metrics
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取绩效指标"""
        if self.portfolio is None:
            raise RuntimeError("请先运行回测")
        
        stats = self.portfolio.stats()
        
        return PerformanceMetrics(
            total_return=stats['Total Return [%]'] / 100,
            annual_return=stats.get('Annual Return [%]', 0) / 100,
            cumulative_return=stats['Total Return [%]'] / 100,
            volatility=stats.get('Volatility', 0),
            sharpe_ratio=stats.get('Sharpe Ratio', 0),
            sortino_ratio=stats.get('Sortino Ratio', 0),
            max_drawdown=stats['Max Drawdown [%]'] / 100,
            max_drawdown_duration=stats.get('Max Drawdown Duration', 0),
            total_trades=stats['Total Trades'],
            win_rate=stats['Win Rate [%]'] / 100,
            profit_factor=stats.get('Profit Factor', 0),
            avg_win=stats.get('Avg Winning Trade [%]', 0) / 100,
            avg_loss=stats.get('Avg Losing Trade [%]', 0) / 100,
            start_date=self.portfolio.wrapper.index[0],
            end_date=self.portfolio.wrapper.index[-1],
            equity_curve=self.portfolio.value(),
            trades=self.portfolio.trades.records_readable.to_dict('records')
        )


class CustomAdapter(BacktestAdapter):
    """自定义框架适配器（简化实现）"""
    
    def __init__(self):
        super().__init__(FrameworkType.CUSTOM)
        self.data = {}
        self.positions = {}
        self.cash = 0
        self.orders = []
        self.trades = []
        self.equity_curve = []
    
    def initialize(self, **kwargs):
        """初始化自定义框架"""
        self.cash = kwargs.get('initial_cash', 1000000)
        self.initial_cash = self.cash
        self.commission = kwargs.get('commission', 0.001)
        
        self._initialized = True
        print(f"✅ 自定义框架初始化完成（初始资金: {self.cash:,.0f}）")
    
    def add_data(self, data: pd.DataFrame, symbol: str):
        """添加数据"""
        self.data[symbol] = data
        print(f"✅ 添加数据: {symbol} (共{len(data)}条)")
    
    def set_strategy(self, strategy_func: Callable, params: Dict = None):
        """设置策略"""
        self.strategy_func = strategy_func
        self.strategy_params = params or {}
        print(f"✅ 策略设置完成")
    
    def place_order(self, order: Order) -> str:
        """下单"""
        order.order_id = f"ORDER_{len(self.orders)+1}"
        order.timestamp = datetime.now()
        self.orders.append(order)
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        for order in self.orders:
            if order.order_id == order_id and order.status == "pending":
                order.status = "cancelled"
                return True
        return False
    
    def get_positions(self) -> List[Position]:
        """获取持仓"""
        positions = []
        for symbol, qty in self.positions.items():
            if symbol in self.data and len(self.data[symbol]) > 0:
                current_price = self.data[symbol]['close'].iloc[-1]
                # 简化计算
                positions.append(Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_price=current_price,
                    market_value=qty * current_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0
                ))
        return positions
    
    def get_account_value(self) -> float:
        """获取账户价值"""
        total = self.cash
        for position in self.get_positions():
            total += position.market_value
        return total
    
    def run(self) -> PerformanceMetrics:
        """运行回测（简化实现）"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize()")
        
        print("\n🚀 开始自定义回测...")
        print(f"初始资金: {self.cash:,.2f}")
        
        # 简化回测逻辑：遍历每个时间点
        # 实际应该更复杂
        for symbol, data in self.data.items():
            for i, (timestamp, row) in enumerate(data.iterrows()):
                # 调用策略
                signals = self.strategy_func(data.iloc[:i+1], self.strategy_params)
                
                # 处理信号（简化）
                # ...
                
                # 记录权益
                self.equity_curve.append(self.get_account_value())
        
        final_value = self.get_account_value()
        print(f"最终资金: {final_value:,.2f}")
        print(f"收益: {(final_value - self.initial_cash):,.2f} "
              f"({(final_value/self.initial_cash - 1)*100:.2f}%)")
        
        return self.get_performance_metrics()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取绩效指标（简化实现）"""
        final_value = self.get_account_value()
        total_return = (final_value / self.initial_cash) - 1
        
        # 计算其他指标（简化）
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大回撤
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_dd = drawdown.min()
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=0.0,
            cumulative_return=total_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=0.0,
            max_drawdown=max_dd,
            max_drawdown_duration=0,
            total_trades=len(self.trades),
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            equity_curve=equity_series,
            trades=self.trades
        )


class UnifiedBacktester:
    """统一回测器（门面模式）"""
    
    def __init__(self, framework_type: FrameworkType = FrameworkType.CUSTOM):
        """
        初始化统一回测器
        
        Args:
            framework_type: 使用的框架类型
        """
        self.framework_type = framework_type
        self.adapter = self._create_adapter(framework_type)
    
    def _create_adapter(self, framework_type: FrameworkType) -> BacktestAdapter:
        """创建适配器"""
        adapters = {
            FrameworkType.BACKTRADER: BacktraderAdapter,
            FrameworkType.VECTORBT: VectorBTAdapter,
            FrameworkType.CUSTOM: CustomAdapter
        }
        
        if framework_type not in adapters:
            raise ValueError(f"不支持的框架类型: {framework_type}")
        
        return adapters[framework_type]()
    
    def initialize(self, **kwargs):
        """初始化框架"""
        self.adapter.initialize(**kwargs)
    
    def add_data(self, data: pd.DataFrame, symbol: str):
        """添加数据"""
        self.adapter.add_data(data, symbol)
    
    def set_strategy(self, strategy_func: Callable, params: Dict = None):
        """设置策略"""
        self.adapter.set_strategy(strategy_func, params)
    
    def run(self) -> PerformanceMetrics:
        """运行回测"""
        return self.adapter.run()
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取绩效指标"""
        return self.adapter.get_performance_metrics()
    
    def print_summary(self, metrics: PerformanceMetrics):
        """打印绩效摘要"""
        print("\n" + "="*60)
        print("📊 回测绩效报告")
        print("="*60)
        print(f"\n框架: {self.framework_type.value}")
        print(f"回测周期: {metrics.start_date.date()} ~ {metrics.end_date.date()}")
        
        print(f"\n📈 收益指标:")
        print(f"  总收益率: {metrics.total_return:.2%}")
        print(f"  年化收益率: {metrics.annual_return:.2%}")
        print(f"  累计收益率: {metrics.cumulative_return:.2%}")
        
        print(f"\n⚠️  风险指标:")
        print(f"  波动率: {metrics.volatility:.2%}")
        print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
        print(f"  索提诺比率: {metrics.sortino_ratio:.2f}")
        print(f"  最大回撤: {metrics.max_drawdown:.2%}")
        
        print(f"\n💼 交易指标:")
        print(f"  总交易次数: {metrics.total_trades}")
        print(f"  胜率: {metrics.win_rate:.2%}")
        print(f"  盈亏比: {metrics.profit_factor:.2f}")
        print(f"  平均盈利: {metrics.avg_win:.2%}")
        print(f"  平均亏损: {metrics.avg_loss:.2%}")
        
        print("\n" + "="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # 确保high >= close >= low
    data['high'] = data[['open', 'close']].max(axis=1) + 1
    data['low'] = data[['open', 'close']].min(axis=1) - 1
    
    # 示例策略：简单的均线交叉
    def simple_ma_strategy(data, params):
        """简单移动平均线策略"""
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        data['ma_short'] = data['close'].rolling(short_window).mean()
        data['ma_long'] = data['close'].rolling(long_window).mean()
        
        # 生成信号
        entries = (data['ma_short'] > data['ma_long']) & (data['ma_short'].shift(1) <= data['ma_long'].shift(1))
        exits = (data['ma_short'] < data['ma_long']) & (data['ma_short'].shift(1) >= data['ma_long'].shift(1))
        
        return {'entries': entries, 'exits': exits}
    
    # 测试自定义框架
    print("测试自定义框架:")
    backtester = UnifiedBacktester(FrameworkType.CUSTOM)
    backtester.initialize(initial_cash=1000000, commission=0.001)
    backtester.add_data(data, '000001.SZ')
    backtester.set_strategy(simple_ma_strategy, {'short_window': 20, 'long_window': 50})
    metrics = backtester.run()
    backtester.print_summary(metrics)
    
    print("\n✅ 完成")
