"""
P2-9: 高频交易引擎 (High-Frequency Trading Engine)
实现订单簿分析、微观结构信号、延迟优化等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    volume: int
    orders: int = 1


@dataclass
class Tick:
    """Tick数据"""
    timestamp: float
    price: float
    volume: int
    side: str  # 'buy' or 'sell'


class OrderBook:
    """限价订单簿"""
    
    def __init__(self, symbol: str, depth: int = 10):
        """
        初始化订单簿
        
        Args:
            symbol: 交易标的
            depth: 订单簿深度
        """
        self.symbol = symbol
        self.depth = depth
        self.bids: List[OrderBookLevel] = []  # 买盘 (价格降序)
        self.asks: List[OrderBookLevel] = []  # 卖盘 (价格升序)
        self.last_update = time.time()
        
        logger.info(f"订单簿初始化: {symbol}, 深度={depth}")
    
    def update(self, bids: List[Tuple], asks: List[Tuple]):
        """
        更新订单簿
        
        Args:
            bids: 买盘列表 [(price, volume), ...]
            asks: 卖盘列表 [(price, volume), ...]
        """
        self.bids = [OrderBookLevel(p, v) for p, v in bids[:self.depth]]
        self.asks = [OrderBookLevel(p, v) for p, v in asks[:self.depth]]
        self.last_update = time.time()
    
    def get_spread(self) -> float:
        """获取买卖价差"""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0
    
    def get_mid_price(self) -> float:
        """获取中间价"""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0
    
    def get_weighted_mid_price(self) -> float:
        """获取加权中间价"""
        if self.bids and self.asks:
            bid_weight = self.bids[0].volume
            ask_weight = self.asks[0].volume
            total_weight = bid_weight + ask_weight
            
            if total_weight > 0:
                return (self.bids[0].price * bid_weight + 
                       self.asks[0].price * ask_weight) / total_weight
        return self.get_mid_price()
    
    def get_order_imbalance(self) -> float:
        """
        计算订单不平衡度
        
        Returns:
            不平衡度 [-1, 1], 正值表示买盘强
        """
        if not self.bids or not self.asks:
            return 0.0
        
        bid_volume = sum(level.volume for level in self.bids)
        ask_volume = sum(level.volume for level in self.asks)
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0.0


class MicrostructureSignals:
    """微观结构信号生成器"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化信号生成器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.tick_buffer = deque(maxlen=window_size)
        
        logger.info(f"微观结构信号初始化 (window={window_size})")
    
    def add_tick(self, tick: Tick):
        """添加新tick"""
        self.tick_buffer.append(tick)
    
    def compute_vwap(self) -> float:
        """计算成交量加权均价 (VWAP)"""
        if not self.tick_buffer:
            return 0.0
        
        total_value = sum(t.price * t.volume for t in self.tick_buffer)
        total_volume = sum(t.volume for t in self.tick_buffer)
        
        return total_value / total_volume if total_volume > 0 else 0.0
    
    def compute_realized_volatility(self) -> float:
        """计算实现波动率"""
        if len(self.tick_buffer) < 2:
            return 0.0
        
        prices = [t.price for t in self.tick_buffer]
        returns = np.diff(np.log(prices))
        
        return np.std(returns) * np.sqrt(len(returns))
    
    def compute_order_flow(self) -> float:
        """
        计算订单流不平衡
        
        Returns:
            净买入量
        """
        if not self.tick_buffer:
            return 0.0
        
        buy_volume = sum(t.volume for t in self.tick_buffer if t.side == 'buy')
        sell_volume = sum(t.volume for t in self.tick_buffer if t.side == 'sell')
        
        return buy_volume - sell_volume
    
    def compute_trade_intensity(self) -> float:
        """计算交易强度 (每秒交易次数)"""
        if len(self.tick_buffer) < 2:
            return 0.0
        
        time_span = self.tick_buffer[-1].timestamp - self.tick_buffer[0].timestamp
        
        return len(self.tick_buffer) / time_span if time_span > 0 else 0.0
    
    def get_all_signals(self) -> Dict[str, float]:
        """获取所有微观结构信号"""
        return {
            'vwap': self.compute_vwap(),
            'realized_vol': self.compute_realized_volatility(),
            'order_flow': self.compute_order_flow(),
            'trade_intensity': self.compute_trade_intensity()
        }


class LatencyOptimizer:
    """延迟优化器"""
    
    def __init__(self):
        self.latency_history = deque(maxlen=1000)
        self.start_time = None
        
        logger.info("延迟优化器初始化")
    
    def start_measurement(self):
        """开始延迟测量"""
        self.start_time = time.perf_counter()
    
    def end_measurement(self) -> float:
        """
        结束延迟测量
        
        Returns:
            延迟(微秒)
        """
        if self.start_time is None:
            return 0.0
        
        latency = (time.perf_counter() - self.start_time) * 1_000_000
        self.latency_history.append(latency)
        self.start_time = None
        
        return latency
    
    def get_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        if not self.latency_history:
            return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0}
        
        latencies = np.array(self.latency_history)
        
        return {
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': np.max(latencies)
        }


class HighFrequencyBacktester:
    """高频回测引擎"""
    
    def __init__(self, commission: float = 0.0001, slippage: float = 0.0001):
        """
        初始化回测引擎
        
        Args:
            commission: 手续费率
            slippage: 滑点
        """
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.pnl_history = []
        
        logger.info(f"高频回测引擎初始化 (commission={commission}, slippage={slippage})")
    
    def execute_trade(self, price: float, volume: int, side: str, 
                     timestamp: float) -> Dict:
        """
        执行交易
        
        Args:
            price: 价格
            volume: 数量
            side: 方向 ('buy' or 'sell')
            timestamp: 时间戳
        
        Returns:
            交易记录
        """
        # 考虑滑点
        if side == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # 计算成本
        cost = execution_price * volume * self.commission
        
        trade = {
            'timestamp': timestamp,
            'price': execution_price,
            'volume': volume,
            'side': side,
            'cost': cost
        }
        
        self.trades.append(trade)
        return trade
    
    def calculate_pnl(self) -> float:
        """计算总盈亏"""
        if not self.trades:
            return 0.0
        
        # 简化计算：假设买入后平仓
        total_pnl = 0.0
        position = 0
        avg_price = 0.0
        
        for trade in self.trades:
            if trade['side'] == 'buy':
                position += trade['volume']
                avg_price = (avg_price * (position - trade['volume']) + 
                           trade['price'] * trade['volume']) / position
            else:  # sell
                if position > 0:
                    pnl = (trade['price'] - avg_price) * trade['volume']
                    total_pnl += pnl - trade['cost']
                    position -= trade['volume']
        
        return total_pnl
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取绩效指标"""
        if not self.trades:
            return {'total_pnl': 0, 'num_trades': 0, 'win_rate': 0}
        
        total_pnl = self.calculate_pnl()
        
        return {
            'total_pnl': total_pnl,
            'num_trades': len(self.trades),
            'avg_pnl_per_trade': total_pnl / len(self.trades),
            'total_cost': sum(t['cost'] for t in self.trades)
        }


def create_sample_orderbook() -> OrderBook:
    """创建示例订单簿"""
    book = OrderBook("AAPL", depth=5)
    
    # 模拟订单簿数据
    bids = [
        (150.05, 100),
        (150.04, 200),
        (150.03, 150),
        (150.02, 300),
        (150.01, 250)
    ]
    
    asks = [
        (150.06, 120),
        (150.07, 180),
        (150.08, 160),
        (150.09, 220),
        (150.10, 200)
    ]
    
    book.update(bids, asks)
    return book


def create_sample_ticks(num_ticks: int = 100) -> List[Tick]:
    """创建示例tick数据"""
    np.random.seed(42)
    
    ticks = []
    base_price = 150.0
    current_time = time.time()
    
    for i in range(num_ticks):
        price = base_price + np.random.normal(0, 0.1)
        volume = np.random.randint(10, 200)
        side = np.random.choice(['buy', 'sell'])
        timestamp = current_time + i * 0.001  # 1ms间隔
        
        ticks.append(Tick(timestamp, price, volume, side))
    
    return ticks


def main():
    """示例: 高频交易引擎"""
    print("=" * 80)
    print("P2-9: 高频交易引擎 - 示例")
    print("=" * 80)
    
    # 1. 订单簿分析
    print("\n📖 订单簿分析...")
    
    book = create_sample_orderbook()
    
    print(f"  标的: {book.symbol}")
    print(f"  买一价: {book.bids[0].price:.2f} (量: {book.bids[0].volume})")
    print(f"  卖一价: {book.asks[0].price:.2f} (量: {book.asks[0].volume})")
    print(f"  价差: {book.get_spread():.4f}")
    print(f"  中间价: {book.get_mid_price():.4f}")
    print(f"  加权中间价: {book.get_weighted_mid_price():.4f}")
    print(f"  订单不平衡: {book.get_order_imbalance():.4f}")
    
    print("✅ 订单簿分析完成")
    
    # 2. 微观结构信号
    print("\n📊 微观结构信号...")
    
    signals = MicrostructureSignals(window_size=50)
    ticks = create_sample_ticks(100)
    
    for tick in ticks:
        signals.add_tick(tick)
    
    all_signals = signals.get_all_signals()
    print(f"  VWAP: {all_signals['vwap']:.4f}")
    print(f"  实现波动率: {all_signals['realized_vol']:.4f}")
    print(f"  订单流: {all_signals['order_flow']:.0f}")
    print(f"  交易强度: {all_signals['trade_intensity']:.2f} trades/s")
    
    print("✅ 微观结构信号完成")
    
    # 3. 延迟优化
    print("\n⚡ 延迟测量...")
    
    optimizer = LatencyOptimizer()
    
    # 模拟100次操作
    for _ in range(100):
        optimizer.start_measurement()
        # 模拟一些计算
        _ = np.random.rand(100).sum()
        optimizer.end_measurement()
    
    stats = optimizer.get_stats()
    print(f"  平均延迟: {stats['mean']:.2f} μs")
    print(f"  P50延迟: {stats['p50']:.2f} μs")
    print(f"  P95延迟: {stats['p95']:.2f} μs")
    print(f"  P99延迟: {stats['p99']:.2f} μs")
    print(f"  最大延迟: {stats['max']:.2f} μs")
    
    print("✅ 延迟测量完成")
    
    # 4. 高频回测
    print("\n🔄 高频回测...")
    
    backtester = HighFrequencyBacktester(
        commission=0.0001,
        slippage=0.0001
    )
    
    # 模拟交易
    for i in range(20):
        side = 'buy' if i % 2 == 0 else 'sell'
        backtester.execute_trade(
            price=150.0 + np.random.normal(0, 0.1),
            volume=100,
            side=side,
            timestamp=time.time() + i * 0.1
        )
    
    metrics = backtester.get_performance_metrics()
    print(f"  总交易数: {metrics['num_trades']}")
    print(f"  总盈亏: ${metrics['total_pnl']:.2f}")
    print(f"  平均单笔盈亏: ${metrics['avg_pnl_per_trade']:.4f}")
    print(f"  总成本: ${metrics['total_cost']:.4f}")
    
    print("✅ 高频回测完成")
    
    print("\n" + "=" * 80)
    print("✅ 所有高频交易功能演示完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
