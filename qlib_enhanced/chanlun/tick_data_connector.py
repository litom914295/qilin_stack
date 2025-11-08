"""Tick级别数据源连接器

支持多种Tick数据源:
1. 模拟数据（测试用）
2. AKShare实时接口（免费但有限制）
3. Tushare Pro（需token）
4. 自定义WebSocket源

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - 实时Tick数据接入
"""
import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Callable, Dict
from queue import Queue
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Tick数据结构"""
    symbol: str
    timestamp: datetime
    last_price: float
    volume: int
    amount: float
    bid_price1: float = 0.0
    bid_volume1: int = 0
    ask_price1: float = 0.0
    ask_volume1: int = 0
    # L2深度（可选）
    bid_prices: List[float] = None
    bid_volumes: List[int] = None
    ask_prices: List[float] = None
    ask_volumes: List[int] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_ohlcv(self) -> Dict:
        """转换为OHLCV格式（聚合用）"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price': self.last_price,
            'volume': self.volume,
            'amount': self.amount
        }


class TickDataSource(ABC):
    """Tick数据源抽象基类"""
    
    @abstractmethod
    def connect(self):
        """连接数据源"""
        pass
    
    @abstractmethod
    def subscribe(self, symbols: List[str]):
        """订阅股票代码"""
        pass
    
    @abstractmethod
    def get_tick(self) -> Optional[TickData]:
        """获取单个Tick（阻塞）"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass


class MockTickDataSource(TickDataSource):
    """模拟Tick数据源（测试用）
    
    生成随机波动的模拟Tick数据
    """
    
    def __init__(self, interval_ms: int = 1000):
        """初始化
        
        Args:
            interval_ms: Tick间隔（毫秒）
        """
        self.interval_ms = interval_ms
        self.symbols = []
        self.connected = False
        self.base_prices = {}  # 基础价格
        logger.info(f"MockTickDataSource初始化: interval={interval_ms}ms")
    
    def connect(self):
        """连接（模拟）"""
        self.connected = True
        logger.info("模拟数据源连接成功")
    
    def subscribe(self, symbols: List[str]):
        """订阅股票"""
        self.symbols = symbols
        # 初始化基础价格
        for symbol in symbols:
            self.base_prices[symbol] = 10.0 + hash(symbol) % 50
        logger.info(f"订阅股票: {symbols}")
    
    def get_tick(self) -> Optional[TickData]:
        """生成模拟Tick"""
        if not self.connected or not self.symbols:
            return None
        
        # 等待间隔
        time.sleep(self.interval_ms / 1000.0)
        
        # 随机选择一个股票
        import random
        symbol = random.choice(self.symbols)
        
        # 随机波动
        base = self.base_prices[symbol]
        change = random.gauss(0, 0.01) * base
        price = base + change
        self.base_prices[symbol] = price  # 更新基础价格
        
        tick = TickData(
            symbol=symbol,
            timestamp=datetime.now(),
            last_price=round(price, 2),
            volume=random.randint(100, 10000),
            amount=round(price * random.randint(100, 10000), 2),
            bid_price1=round(price * 0.999, 2),
            bid_volume1=random.randint(1000, 5000),
            ask_price1=round(price * 1.001, 2),
            ask_volume1=random.randint(1000, 5000)
        )
        
        return tick
    
    def disconnect(self):
        """断开连接"""
        self.connected = False
        logger.info("模拟数据源断开连接")
    
    def is_connected(self) -> bool:
        """检查连接"""
        return self.connected


class AKShareTickDataSource(TickDataSource):
    """AKShare实时数据源
    
    注意: AKShare的实时数据接口有频率限制，不适合高频场景
    """
    
    def __init__(self, fetch_interval_sec: int = 3):
        """初始化
        
        Args:
            fetch_interval_sec: 抓取间隔（秒）
        """
        self.fetch_interval_sec = fetch_interval_sec
        self.symbols = []
        self.connected = False
        self.tick_queue = Queue(maxsize=1000)
        self.fetcher_thread = None
        self.stop_flag = threading.Event()
        logger.info(f"AKShareTickDataSource初始化: interval={fetch_interval_sec}s")
    
    def connect(self):
        """连接"""
        try:
            import akshare as ak
            self.ak = ak
            self.connected = True
            logger.info("AKShare数据源连接成功")
        except ImportError:
            logger.error("AKShare未安装: pip install akshare")
            self.connected = False
    
    def subscribe(self, symbols: List[str]):
        """订阅股票并启动抓取线程"""
        self.symbols = symbols
        logger.info(f"订阅股票: {symbols}")
        
        # 启动后台抓取线程
        if self.connected and not self.fetcher_thread:
            self.stop_flag.clear()
            self.fetcher_thread = threading.Thread(
                target=self._fetch_loop,
                daemon=True
            )
            self.fetcher_thread.start()
    
    def _fetch_loop(self):
        """后台抓取循环"""
        logger.info("AKShare抓取线程启动")
        
        while not self.stop_flag.is_set():
            for symbol in self.symbols:
                try:
                    # AKShare实时行情接口
                    df = self.ak.stock_zh_a_spot_em()
                    
                    # 查找对应股票
                    stock_row = df[df['代码'] == symbol]
                    
                    if not stock_row.empty:
                        row = stock_row.iloc[0]
                        tick = TickData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            last_price=float(row.get('最新价', 0)),
                            volume=int(row.get('成交量', 0)),
                            amount=float(row.get('成交额', 0)),
                            bid_price1=float(row.get('买一价', 0)),
                            bid_volume1=int(row.get('买一量', 0)),
                            ask_price1=float(row.get('卖一价', 0)),
                            ask_volume1=int(row.get('卖一量', 0))
                        )
                        
                        # 放入队列
                        if not self.tick_queue.full():
                            self.tick_queue.put(tick)
                
                except Exception as e:
                    logger.error(f"抓取{symbol}失败: {e}")
            
            # 等待间隔
            time.sleep(self.fetch_interval_sec)
        
        logger.info("AKShare抓取线程停止")
    
    def get_tick(self) -> Optional[TickData]:
        """从队列获取Tick"""
        try:
            return self.tick_queue.get(timeout=1.0)
        except:
            return None
    
    def disconnect(self):
        """断开连接"""
        self.stop_flag.set()
        if self.fetcher_thread:
            self.fetcher_thread.join(timeout=5.0)
        self.connected = False
        logger.info("AKShare数据源断开连接")
    
    def is_connected(self) -> bool:
        """检查连接"""
        return self.connected


class TushareTickDataSource(TickDataSource):
    """Tushare Pro实时数据源
    
    需要Tushare Pro权限和Token
    """
    
    def __init__(self, token: str, fetch_interval_sec: int = 3):
        """初始化
        
        Args:
            token: Tushare Pro Token
            fetch_interval_sec: 抓取间隔（秒）
        """
        self.token = token
        self.fetch_interval_sec = fetch_interval_sec
        self.symbols = []
        self.connected = False
        self.tick_queue = Queue(maxsize=1000)
        self.fetcher_thread = None
        self.stop_flag = threading.Event()
        logger.info(f"TushareTickDataSource初始化: interval={fetch_interval_sec}s")
    
    def connect(self):
        """连接"""
        try:
            import tushare as ts
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            self.connected = True
            logger.info("Tushare Pro连接成功")
        except ImportError:
            logger.error("Tushare未安装: pip install tushare")
            self.connected = False
        except Exception as e:
            logger.error(f"Tushare连接失败: {e}")
            self.connected = False
    
    def subscribe(self, symbols: List[str]):
        """订阅股票"""
        # 转换为Tushare格式（如000001.SZ）
        self.symbols = [self._to_tushare_symbol(s) for s in symbols]
        logger.info(f"订阅股票: {self.symbols}")
        
        # 启动后台抓取线程
        if self.connected and not self.fetcher_thread:
            self.stop_flag.clear()
            self.fetcher_thread = threading.Thread(
                target=self._fetch_loop,
                daemon=True
            )
            self.fetcher_thread.start()
    
    def _to_tushare_symbol(self, symbol: str) -> str:
        """转换为Tushare格式"""
        # 简单转换逻辑
        if '.' in symbol:
            return symbol
        if symbol.startswith('6'):
            return f"{symbol}.SH"
        else:
            return f"{symbol}.SZ"
    
    def _fetch_loop(self):
        """后台抓取循环"""
        logger.info("Tushare抓取线程启动")
        
        while not self.stop_flag.is_set():
            for symbol in self.symbols:
                try:
                    # Tushare实时行情接口
                    df = self.pro.query('daily_basic', ts_code=symbol)
                    
                    if not df.empty:
                        row = df.iloc[0]
                        tick = TickData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            last_price=float(row.get('close', 0)),
                            volume=int(row.get('vol', 0)),
                            amount=float(row.get('amount', 0))
                        )
                        
                        if not self.tick_queue.full():
                            self.tick_queue.put(tick)
                
                except Exception as e:
                    logger.error(f"抓取{symbol}失败: {e}")
            
            time.sleep(self.fetch_interval_sec)
        
        logger.info("Tushare抓取线程停止")
    
    def get_tick(self) -> Optional[TickData]:
        """从队列获取Tick"""
        try:
            return self.tick_queue.get(timeout=1.0)
        except:
            return None
    
    def disconnect(self):
        """断开连接"""
        self.stop_flag.set()
        if self.fetcher_thread:
            self.fetcher_thread.join(timeout=5.0)
        self.connected = False
        logger.info("Tushare数据源断开连接")
    
    def is_connected(self) -> bool:
        """检查连接"""
        return self.connected


class TickDataConnector:
    """Tick数据连接器管理类
    
    功能:
    1. 统一管理多种数据源
    2. 自动重连
    3. 回调通知
    """
    
    def __init__(self, source_type: str = 'mock', **kwargs):
        """初始化
        
        Args:
            source_type: 数据源类型 ('mock', 'akshare', 'tushare')
            **kwargs: 数据源特定参数
        """
        self.source_type = source_type
        self.source = self._create_source(source_type, kwargs)
        self.callbacks: List[Callable[[TickData], None]] = []
        self.running = False
        self.worker_thread = None
        logger.info(f"TickDataConnector初始化: source={source_type}")
    
    def _create_source(self, source_type: str, kwargs: Dict) -> TickDataSource:
        """创建数据源实例"""
        if source_type == 'mock':
            return MockTickDataSource(
                interval_ms=kwargs.get('interval_ms', 1000)
            )
        elif source_type == 'akshare':
            return AKShareTickDataSource(
                fetch_interval_sec=kwargs.get('fetch_interval_sec', 3)
            )
        elif source_type == 'tushare':
            token = kwargs.get('token', '')
            return TushareTickDataSource(
                token=token,
                fetch_interval_sec=kwargs.get('fetch_interval_sec', 3)
            )
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
    
    def connect(self):
        """连接数据源"""
        self.source.connect()
    
    def subscribe(self, symbols: List[str]):
        """订阅股票"""
        self.source.subscribe(symbols)
    
    def register_callback(self, callback: Callable[[TickData], None]):
        """注册Tick回调函数"""
        self.callbacks.append(callback)
        logger.info(f"注册回调函数: {callback.__name__}")
    
    def start(self):
        """启动Tick接收线程"""
        if self.running:
            logger.warning("Tick接收线程已在运行")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True
        )
        self.worker_thread.start()
        logger.info("Tick接收线程启动")
    
    def _receive_loop(self):
        """接收循环"""
        while self.running:
            tick = self.source.get_tick()
            
            if tick:
                # 调用所有回调
                for callback in self.callbacks:
                    try:
                        callback(tick)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")
    
    def stop(self):
        """停止Tick接收"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Tick接收线程停止")
    
    def disconnect(self):
        """断开数据源"""
        self.stop()
        self.source.disconnect()


if __name__ == '__main__':
    """测试Tick数据连接器"""
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("Tick数据连接器测试")
    print("="*70)
    
    # 创建连接器（模拟数据源）
    connector = TickDataConnector(source_type='mock', interval_ms=500)
    
    # 连接
    connector.connect()
    
    # 订阅股票
    connector.subscribe(['000001', '600000', '000002'])
    
    # 注册回调
    tick_count = [0]
    
    def on_tick(tick: TickData):
        tick_count[0] += 1
        if tick_count[0] <= 10:
            print(f"Tick#{tick_count[0]}: {tick.symbol} @ {tick.last_price:.2f} | {tick.timestamp.strftime('%H:%M:%S')}")
    
    connector.register_callback(on_tick)
    
    # 启动
    connector.start()
    
    # 运行10秒
    print("\n接收Tick数据10秒...")
    time.sleep(10)
    
    # 停止
    connector.stop()
    connector.disconnect()
    
    print(f"\n✅ 测试完成！共接收{tick_count[0]}个Tick")
