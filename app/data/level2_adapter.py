"""
麒麟量化系统 - Level2数据适配层
统一接口适配多家券商的Level2行情数据

核心功能：
1. 统一Level2数据接口
2. 逐笔成交解析
3. 委托队列分析
4. 大单追踪
5. 盘口深度分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, time
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    BUY = "买入"
    SELL = "卖出"
    CANCEL = "撤单"


class OrderSize(Enum):
    """订单规模"""
    SUPER_LARGE = "超大单"  # >100万
    LARGE = "大单"          # 20-100万
    MEDIUM = "中单"         # 5-20万
    SMALL = "小单"          # <5万


@dataclass
class TickData:
    """逐笔成交数据"""
    timestamp: datetime     # 时间戳
    price: float           # 成交价
    volume: int            # 成交量
    turnover: float        # 成交额
    order_type: OrderType  # 买卖方向
    order_size: OrderSize  # 单量级别
    
    # 扩展信息
    bs_flag: str = ""      # B/S标志
    seq_no: int = 0        # 序号
    

@dataclass
class OrderQueueItem:
    """委托队列单项"""
    price: float           # 价格
    volume: int            # 总量
    order_count: int       # 委托笔数
    orders: List[int] = field(default_factory=list)  # 前50笔委托量


@dataclass
class OrderBook:
    """盘口数据（10档）"""
    timestamp: datetime
    
    # 买盘（按价格降序）
    bid_prices: List[float]     # 买价
    bid_volumes: List[int]      # 买量
    bid_orders: List[int]       # 买单笔数
    
    # 卖盘（按价格升序）
    ask_prices: List[float]     # 卖价
    ask_volumes: List[int]      # 卖量
    ask_orders: List[int]       # 卖单笔数
    
    # 汇总
    total_bid_volume: int = 0   # 总买量
    total_ask_volume: int = 0   # 总卖量
    

@dataclass
class Level2Snapshot:
    """Level2完整快照"""
    symbol: str
    timestamp: datetime
    
    # 基础行情
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    turnover: float
    
    # 盘口数据
    order_book: OrderBook
    
    # 队列数据
    bid_queue: Optional[OrderQueueItem] = None
    ask_queue: Optional[OrderQueueItem] = None
    
    # 统计数据
    total_bid_vol: int = 0      # 内盘
    total_ask_vol: int = 0      # 外盘
    avg_price: float = 0.0      # 均价
    

class Level2DataSource(ABC):
    """Level2数据源抽象基类"""
    
    @abstractmethod
    def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[TickData]:
        """获取逐笔成交数据"""
        pass
    
    @abstractmethod
    def fetch_order_book(
        self, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> OrderBook:
        """获取盘口数据"""
        pass
    
    @abstractmethod
    def fetch_order_queue(
        self, 
        symbol: str
    ) -> Tuple[Optional[OrderQueueItem], Optional[OrderQueueItem]]:
        """获取委托队列"""
        pass


class MockLevel2Source(Level2DataSource):
    """模拟Level2数据源（用于测试）"""
    
    def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[TickData]:
        """生成模拟逐笔数据"""
        ticks = []
        current_time = start_time
        base_price = 10.0
        
        # 生成100笔模拟成交
        for i in range(100):
            # 随机价格波动
            price = base_price + np.random.randn() * 0.1
            volume = np.random.randint(100, 10000)
            turnover = price * volume
            
            # 随机买卖方向
            order_type = OrderType.BUY if np.random.rand() > 0.5 else OrderType.SELL
            
            # 判断单量级别
            amount = turnover / 10000  # 万元
            if amount > 100:
                order_size = OrderSize.SUPER_LARGE
            elif amount > 20:
                order_size = OrderSize.LARGE
            elif amount > 5:
                order_size = OrderSize.MEDIUM
            else:
                order_size = OrderSize.SMALL
            
            tick = TickData(
                timestamp=current_time,
                price=price,
                volume=volume,
                turnover=turnover,
                order_type=order_type,
                order_size=order_size,
                bs_flag="B" if order_type == OrderType.BUY else "S",
                seq_no=i
            )
            ticks.append(tick)
            
            # 时间递增（每3秒一笔）
            current_time = current_time + pd.Timedelta(seconds=3)
        
        return ticks
    
    def fetch_order_book(
        self, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> OrderBook:
        """生成模拟盘口数据"""
        base_price = 10.0
        
        # 生成10档买盘
        bid_prices = [base_price - i * 0.01 for i in range(10)]
        bid_volumes = [np.random.randint(1000, 50000) for _ in range(10)]
        bid_orders = [np.random.randint(10, 200) for _ in range(10)]
        
        # 生成10档卖盘
        ask_prices = [base_price + (i + 1) * 0.01 for i in range(10)]
        ask_volumes = [np.random.randint(1000, 50000) for _ in range(10)]
        ask_orders = [np.random.randint(10, 200) for _ in range(10)]
        
        return OrderBook(
            timestamp=timestamp or datetime.now(),
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            bid_orders=bid_orders,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            ask_orders=ask_orders,
            total_bid_volume=sum(bid_volumes),
            total_ask_volume=sum(ask_volumes)
        )
    
    def fetch_order_queue(
        self, 
        symbol: str
    ) -> Tuple[Optional[OrderQueueItem], Optional[OrderQueueItem]]:
        """生成模拟委托队列"""
        # 买一队列
        bid_queue = OrderQueueItem(
            price=10.0,
            volume=100000,
            order_count=50,
            orders=[np.random.randint(500, 5000) for _ in range(50)]
        )
        
        # 卖一队列
        ask_queue = OrderQueueItem(
            price=10.01,
            volume=80000,
            order_count=45,
            orders=[np.random.randint(500, 4000) for _ in range(45)]
        )
        
        return bid_queue, ask_queue


class Level2Adapter:
    """Level2数据适配器"""
    
    def __init__(self, data_source: Level2DataSource):
        """
        初始化适配器
        
        Args:
            data_source: Level2数据源实例
        """
        self.data_source = data_source
        logger.info(f"Level2适配器初始化完成，数据源: {type(data_source).__name__}")
    
    def get_snapshot(self, symbol: str) -> Level2Snapshot:
        """
        获取完整Level2快照
        
        Args:
            symbol: 股票代码
            
        Returns:
            Level2快照
        """
        # 获取盘口数据
        order_book = self.data_source.fetch_order_book(symbol)
        
        # 获取委托队列
        bid_queue, ask_queue = self.data_source.fetch_order_queue(symbol)
        
        # 构建快照
        snapshot = Level2Snapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=order_book.bid_prices[0] if order_book.bid_prices else 0,
            open_price=0,  # 需要从其他数据源补充
            high_price=0,
            low_price=0,
            volume=0,
            turnover=0,
            order_book=order_book,
            bid_queue=bid_queue,
            ask_queue=ask_queue,
            total_bid_vol=order_book.total_bid_volume,
            total_ask_vol=order_book.total_ask_volume
        )
        
        return snapshot
    
    def analyze_order_flow(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        分析订单流
        
        Args:
            symbol: 股票代码
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            订单流分析结果
        """
        # 获取逐笔数据
        ticks = self.data_source.fetch_tick_data(symbol, start_time, end_time)
        
        if not ticks:
            return {"error": "无逐笔数据"}
        
        # 统计分析
        buy_ticks = [t for t in ticks if t.order_type == OrderType.BUY]
        sell_ticks = [t for t in ticks if t.order_type == OrderType.SELL]
        
        # 大单统计
        large_buy = [t for t in buy_ticks if t.order_size in [OrderSize.LARGE, OrderSize.SUPER_LARGE]]
        large_sell = [t for t in sell_ticks if t.order_size in [OrderSize.LARGE, OrderSize.SUPER_LARGE]]
        
        # 计算指标
        buy_volume = sum(t.volume for t in buy_ticks)
        sell_volume = sum(t.volume for t in sell_ticks)
        total_volume = buy_volume + sell_volume
        
        buy_turnover = sum(t.turnover for t in buy_ticks)
        sell_turnover = sum(t.turnover for t in sell_ticks)
        
        # 大单金额
        large_buy_amount = sum(t.turnover for t in large_buy) / 10000  # 万元
        large_sell_amount = sum(t.turnover for t in large_sell) / 10000
        
        # 订单流不平衡
        order_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        return {
            "total_ticks": len(ticks),
            "buy_ticks": len(buy_ticks),
            "sell_ticks": len(sell_ticks),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_turnover": buy_turnover / 10000,  # 万元
            "sell_turnover": sell_turnover / 10000,
            "large_buy_count": len(large_buy),
            "large_sell_count": len(large_sell),
            "large_buy_amount": large_buy_amount,
            "large_sell_amount": large_sell_amount,
            "order_imbalance": order_imbalance,
            "avg_buy_price": buy_turnover / buy_volume if buy_volume > 0 else 0,
            "avg_sell_price": sell_turnover / sell_volume if sell_volume > 0 else 0,
            "start_time": start_time,
            "end_time": end_time
        }
    
    def analyze_order_book_depth(self, symbol: str) -> Dict[str, Any]:
        """
        分析盘口深度
        
        Args:
            symbol: 股票代码
            
        Returns:
            盘口深度分析
        """
        order_book = self.data_source.fetch_order_book(symbol)
        
        # 计算买卖压力
        total_bid = order_book.total_bid_volume
        total_ask = order_book.total_ask_volume
        
        # 前3档占比
        bid_top3 = sum(order_book.bid_volumes[:3])
        ask_top3 = sum(order_book.ask_volumes[:3])
        
        bid_top3_ratio = bid_top3 / total_bid if total_bid > 0 else 0
        ask_top3_ratio = ask_top3 / total_ask if total_ask > 0 else 0
        
        # 买卖比
        bid_ask_ratio = total_bid / total_ask if total_ask > 0 else 0
        
        # 价差
        spread = order_book.ask_prices[0] - order_book.bid_prices[0]
        spread_pct = spread / order_book.bid_prices[0] * 100 if order_book.bid_prices[0] > 0 else 0
        
        # 盘口压力评分
        pressure_score = 0
        if bid_ask_ratio > 2:
            pressure_score = 80  # 买盘强势
        elif bid_ask_ratio > 1.5:
            pressure_score = 60
        elif bid_ask_ratio > 1:
            pressure_score = 40
        elif bid_ask_ratio > 0.5:
            pressure_score = 20
        else:
            pressure_score = 10  # 卖盘强势
        
        return {
            "total_bid_volume": total_bid,
            "total_ask_volume": total_ask,
            "bid_ask_ratio": bid_ask_ratio,
            "bid_top3_ratio": bid_top3_ratio,
            "ask_top3_ratio": ask_top3_ratio,
            "spread": spread,
            "spread_pct": spread_pct,
            "pressure_score": pressure_score,
            "timestamp": order_book.timestamp.isoformat()
        }
    
    def analyze_queue(self, symbol: str) -> Dict[str, Any]:
        """
        分析委托队列
        
        实战要点：
        - 涨停时看买一队列，排队越多越难买到
        - 跌停时看卖一队列，卖盘越多越难卖出
        """
        bid_queue, ask_queue = self.data_source.fetch_order_queue(symbol)
        
        if not bid_queue or not ask_queue:
            return {"error": "无队列数据"}
        
        # 分析买一队列
        bid_analysis = {
            "price": bid_queue.price,
            "total_volume": bid_queue.volume,
            "order_count": bid_queue.order_count,
            "avg_order_size": bid_queue.volume / bid_queue.order_count if bid_queue.order_count > 0 else 0,
            "top10_volume": sum(bid_queue.orders[:10]),
            "concentration": sum(bid_queue.orders[:10]) / bid_queue.volume if bid_queue.volume > 0 else 0
        }
        
        # 分析卖一队列
        ask_analysis = {
            "price": ask_queue.price,
            "total_volume": ask_queue.volume,
            "order_count": ask_queue.order_count,
            "avg_order_size": ask_queue.volume / ask_queue.order_count if ask_queue.order_count > 0 else 0,
            "top10_volume": sum(ask_queue.orders[:10]),
            "concentration": sum(ask_queue.orders[:10]) / ask_queue.volume if ask_queue.volume > 0 else 0
        }
        
        return {
            "bid_queue": bid_analysis,
            "ask_queue": ask_analysis,
            "queue_ratio": bid_queue.volume / ask_queue.volume if ask_queue.volume > 0 else 0
        }
    
    def detect_large_orders(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        threshold: float = 50  # 万元
    ) -> List[Dict]:
        """
        检测大单
        
        Args:
            symbol: 股票代码
            start_time: 开始时间
            end_time: 结束时间
            threshold: 大单阈值（万元）
            
        Returns:
            大单列表
        """
        ticks = self.data_source.fetch_tick_data(symbol, start_time, end_time)
        
        large_orders = []
        for tick in ticks:
            amount = tick.turnover / 10000  # 万元
            if amount >= threshold:
                large_orders.append({
                    "timestamp": tick.timestamp.isoformat(),
                    "price": tick.price,
                    "volume": tick.volume,
                    "amount": amount,
                    "direction": tick.order_type.value,
                    "size_level": tick.order_size.value
                })
        
        return large_orders


# 使用示例
if __name__ == "__main__":
    # 创建模拟数据源
    mock_source = MockLevel2Source()
    
    # 创建适配器
    adapter = Level2Adapter(mock_source)
    
    # 获取快照
    snapshot = adapter.get_snapshot("000001")
    print("=" * 60)
    print("Level2快照")
    print("=" * 60)
    print(f"股票: {snapshot.symbol}")
    print(f"当前价: {snapshot.current_price:.2f}")
    print(f"买一: {snapshot.order_book.bid_prices[0]:.2f} x {snapshot.order_book.bid_volumes[0]}")
    print(f"卖一: {snapshot.order_book.ask_prices[0]:.2f} x {snapshot.order_book.ask_volumes[0]}")
    
    # 分析订单流
    start = datetime.now().replace(hour=9, minute=30, second=0)
    end = datetime.now().replace(hour=9, minute=35, second=0)
    flow = adapter.analyze_order_flow("000001", start, end)
    
    print("\n" + "=" * 60)
    print("订单流分析")
    print("=" * 60)
    print(f"总成交笔数: {flow['total_ticks']}")
    print(f"买入笔数: {flow['buy_ticks']}")
    print(f"卖出笔数: {flow['sell_ticks']}")
    print(f"大单买入: {flow['large_buy_count']}笔, {flow['large_buy_amount']:.2f}万")
    print(f"大单卖出: {flow['large_sell_count']}笔, {flow['large_sell_amount']:.2f}万")
    print(f"订单不平衡度: {flow['order_imbalance']:.2f}")
    
    # 分析盘口深度
    depth = adapter.analyze_order_book_depth("000001")
    print("\n" + "=" * 60)
    print("盘口深度分析")
    print("=" * 60)
    print(f"买卖比: {depth['bid_ask_ratio']:.2f}")
    print(f"买盘压力评分: {depth['pressure_score']}分")
    print(f"价差: {depth['spread']:.4f} ({depth['spread_pct']:.2f}%)")
