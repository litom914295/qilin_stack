"""
实时数据流管理器 (Real-time Stream Manager)
统一管理多数据源的实时数据流

核心功能：
1. 多数据源统一订阅
2. 数据流缓冲和合并
3. 心跳检测和自动重连
4. 数据质量监控
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import deque
import threading
import time


class DataSourceType(Enum):
    """数据源类型"""
    LEVEL2 = "Level2行情"
    TICK = "逐笔成交"
    ORDER_BOOK = "盘口快照"
    LHB = "龙虎榜"
    NEWS = "资讯"


class StreamStatus(Enum):
    """数据流状态"""
    CONNECTED = "已连接"
    DISCONNECTED = "已断开"
    RECONNECTING = "重连中"
    ERROR = "错误"


@dataclass
class StreamData:
    """数据流数据"""
    source_type: DataSourceType
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    sequence: int              # 序列号（检测丢包）


@dataclass
class StreamHealth:
    """数据流健康状态"""
    source_type: DataSourceType
    status: StreamStatus
    last_update: datetime
    data_count: int            # 数据计数
    error_count: int           # 错误计数
    latency_ms: float          # 延迟（毫秒）
    packet_loss_rate: float    # 丢包率
    warnings: List[str]


class DataStreamSource:
    """数据流源抽象基类"""
    
    def __init__(self, source_type: DataSourceType):
        self.source_type = source_type
        self.status = StreamStatus.DISCONNECTED
        self.callbacks: List[Callable] = []
        self._last_sequence = 0
        self._data_count = 0
        self._error_count = 0
        self._latencies: deque = deque(maxlen=100)  # 保留最近100次延迟
        self._lost_packets = 0
    
    def connect(self) -> bool:
        """连接数据源"""
        raise NotImplementedError
    
    def disconnect(self):
        """断开连接"""
        raise NotImplementedError
    
    def subscribe(self, symbols: List[str]):
        """订阅股票"""
        raise NotImplementedError
    
    def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        raise NotImplementedError
    
    def add_callback(self, callback: Callable[[StreamData], None]):
        """添加数据回调"""
        self.callbacks.append(callback)
    
    def _on_data(self, data: StreamData):
        """数据到达处理"""
        # 检测丢包
        if data.sequence > self._last_sequence + 1:
            self._lost_packets += (data.sequence - self._last_sequence - 1)
        self._last_sequence = data.sequence
        
        # 计算延迟
        latency = (datetime.now() - data.timestamp).total_seconds() * 1000
        self._latencies.append(latency)
        
        self._data_count += 1
        
        # 调用回调函数
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                self._error_count += 1
                print(f"回调错误: {e}")
    
    def get_health(self) -> StreamHealth:
        """获取健康状态"""
        warnings = []
        
        # 计算平均延迟
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        
        # 计算丢包率
        total_expected = self._last_sequence if self._last_sequence > 0 else 1
        packet_loss_rate = self._lost_packets / total_expected
        
        # 生成警告
        if avg_latency > 500:
            warnings.append(f"延迟过高: {avg_latency:.0f}ms")
        
        if packet_loss_rate > 0.01:
            warnings.append(f"丢包率过高: {packet_loss_rate:.2%}")
        
        if self.status != StreamStatus.CONNECTED:
            warnings.append(f"连接状态异常: {self.status.value}")
        
        return StreamHealth(
            source_type=self.source_type,
            status=self.status,
            last_update=datetime.now(),
            data_count=self._data_count,
            error_count=self._error_count,
            latency_ms=avg_latency,
            packet_loss_rate=packet_loss_rate,
            warnings=warnings
        )


class RealStreamSource(DataStreamSource):
    """真实数据流源"""
    
    def __init__(self, source_type: DataSourceType, config: Dict[str, Any]):
        super().__init__(source_type)
        self.config = config
        self.api_url = config.get('api_url', 'ws://localhost:8080/stream')
        self.api_key = config.get('api_key', '')
        self._running = False
        self._thread = None
        self._subscribed_symbols = []
        self._ws = None  # WebSocket连接
    
    def connect(self) -> bool:
        """连接数据源"""
        try:
            import websocket
            
            # 创建WebSocket连接
            self._ws = websocket.WebSocketApp(
                self.api_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # 在新线程中运行
            self._running = True
            self._thread = threading.Thread(target=self._ws.run_forever)
            self._thread.daemon = True
            self._thread.start()
            
            # 等待连接成功
            time.sleep(1)
            
            if self.status == StreamStatus.CONNECTED:
                print(f"[{self.source_type.value}] 连接成功")
                return True
            else:
                print(f"[{self.source_type.value}] 连接失败")
                return False
                
        except ImportError:
            print("请安装websocket-client: pip install websocket-client")
            return False
        except Exception as e:
            print(f"[{self.source_type.value}] 连接错误: {e}")
            self._error_count += 1
            return False
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)
        self.status = StreamStatus.DISCONNECTED
        print(f"[{self.source_type.value}] 已断开")
    
    def subscribe(self, symbols: List[str]):
        """订阅股票"""
        if not self._ws:
            print(f"[{self.source_type.value}] 未连接，无法订阅")
            return
        
        try:
            self._subscribed_symbols.extend(symbols)
            # 发送订阅请求
            subscribe_msg = {
                "action": "subscribe",
                "source": self.source_type.value,
                "symbols": symbols,
                "api_key": self.api_key
            }
            
            import json
            self._ws.send(json.dumps(subscribe_msg))
            print(f"[{self.source_type.value}] 订阅: {symbols}")
            
        except Exception as e:
            print(f"[{self.source_type.value}] 订阅失败: {e}")
            self._error_count += 1
    
    def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        if not self._ws:
            return
        
        try:
            for symbol in symbols:
                if symbol in self._subscribed_symbols:
                    self._subscribed_symbols.remove(symbol)
            
            # 发送取消订阅请求
            unsubscribe_msg = {
                "action": "unsubscribe",
                "source": self.source_type.value,
                "symbols": symbols,
                "api_key": self.api_key
            }
            
            import json
            self._ws.send(json.dumps(unsubscribe_msg))
            print(f"[{self.source_type.value}] 取消订阅: {symbols}")
            
        except Exception as e:
            print(f"[{self.source_type.value}] 取消订阅失败: {e}")
            self._error_count += 1
    
    def _on_open(self, ws):
        """连接已建立"""
        self.status = StreamStatus.CONNECTED
        print(f"[{self.source_type.value}] WebSocket已开启")
    
    def _on_message(self, ws, message):
        """收到消息"""
        try:
            import json
            msg_data = json.loads(message)
            
            # 解析为StreamData
            data = StreamData(
                source_type=self.source_type,
                symbol=msg_data.get('symbol', ''),
                timestamp=datetime.fromisoformat(msg_data.get('timestamp', datetime.now().isoformat())),
                data=msg_data.get('data', {}),
                sequence=msg_data.get('sequence', 0)
            )
            
            # 触发数据处理
            self._on_data(data)
            
        except json.JSONDecodeError as e:
            print(f"[{self.source_type.value}] JSON解析错误: {e}")
            self._error_count += 1
        except Exception as e:
            print(f"[{self.source_type.value}] 消息处理错误: {e}")
            self._error_count += 1
    
    def _on_error(self, ws, error):
        """发生错误"""
        self.status = StreamStatus.ERROR
        self._error_count += 1
        print(f"[{self.source_type.value}] 错误: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """连接关闭"""
        self.status = StreamStatus.DISCONNECTED
        print(f"[{self.source_type.value}] 连接关闭: {close_status_code} - {close_msg}")
        
        # 如果还在运行，尝试重连
        if self._running:
            self.status = StreamStatus.RECONNECTING
            print(f"[{self.source_type.value}] 5秒后重连...")
            time.sleep(5)
            if self._running:
                self.connect()


class MockStreamSource(DataStreamSource):
    """模拟数据流源（用于测试）"""
    
    def __init__(self, source_type: DataSourceType):
        super().__init__(source_type)
        self._running = False
        self._thread = None
        self._subscribed_symbols = []
    
    def connect(self) -> bool:
        """连接数据源"""
        print(f"[{self.source_type.value}] 连接成功")
        self.status = StreamStatus.CONNECTED
        return True
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        if self._thread:
            self._thread.join()
        self.status = StreamStatus.DISCONNECTED
        print(f"[{self.source_type.value}] 已断开")
    
    def subscribe(self, symbols: List[str]):
        """订阅股票"""
        self._subscribed_symbols.extend(symbols)
        print(f"[{self.source_type.value}] 订阅: {symbols}")
        
        # 启动数据生成线程
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._generate_data)
            self._thread.daemon = True
            self._thread.start()
    
    def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                self._subscribed_symbols.remove(symbol)
        print(f"[{self.source_type.value}] 取消订阅: {symbols}")
    
    def _generate_data(self):
        """生成模拟数据"""
        sequence = 0
        while self._running:
            for symbol in self._subscribed_symbols:
                sequence += 1
                data = StreamData(
                    source_type=self.source_type,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    data={
                        "price": 10.50 + (sequence % 10) * 0.01,
                        "volume": 1000 * (sequence % 5 + 1),
                        "bid": 10.49,
                        "ask": 10.51
                    },
                    sequence=sequence
                )
                self._on_data(data)
            
            time.sleep(1)  # 每秒推送一次


class StreamManager:
    """数据流管理器"""
    
    def __init__(self):
        self.sources: Dict[DataSourceType, DataStreamSource] = {}
        self.data_buffer: deque = deque(maxlen=1000)  # 数据缓冲区
        self.subscribers: Dict[str, List[Callable]] = {}  # {symbol: [callbacks]}
        self._monitor_thread = None
        self._monitoring = False
    
    def add_source(self, source: DataStreamSource):
        """添加数据源"""
        self.sources[source.source_type] = source
        # 添加数据回调
        source.add_callback(self._on_source_data)
        print(f"✅ 添加数据源: {source.source_type.value}")
    
    def connect_all(self) -> Dict[DataSourceType, bool]:
        """连接所有数据源"""
        results = {}
        for source_type, source in self.sources.items():
            results[source_type] = source.connect()
        return results
    
    def disconnect_all(self):
        """断开所有数据源"""
        for source in self.sources.values():
            source.disconnect()
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def subscribe(self, symbols: List[str], callback: Callable[[StreamData], None]):
        """订阅股票数据"""
        for symbol in symbols:
            if symbol not in self.subscribers:
                self.subscribers[symbol] = []
            self.subscribers[symbol].append(callback)
            
            # 通知所有数据源订阅
            for source in self.sources.values():
                source.subscribe([symbol])
        
        print(f"📡 订阅成功: {symbols}")
    
    def unsubscribe(self, symbols: List[str], callback: Optional[Callable] = None):
        """取消订阅"""
        for symbol in symbols:
            if symbol in self.subscribers:
                if callback:
                    if callback in self.subscribers[symbol]:
                        self.subscribers[symbol].remove(callback)
                else:
                    self.subscribers[symbol] = []
                
                # 如果没有订阅者了，通知数据源取消订阅
                if not self.subscribers[symbol]:
                    for source in self.sources.values():
                        source.unsubscribe([symbol])
                    del self.subscribers[symbol]
        
        print(f"📡 取消订阅: {symbols}")
    
    def _on_source_data(self, data: StreamData):
        """数据源数据到达"""
        # 存入缓冲区
        self.data_buffer.append(data)
        
        # 分发给订阅者
        if data.symbol in self.subscribers:
            for callback in self.subscribers[data.symbol]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"订阅者回调错误: {e}")
    
    def start_monitoring(self, interval: int = 10):
        """启动健康监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        print(f"🔍 启动健康监控（间隔{interval}秒）")
    
    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self._monitoring:
            time.sleep(interval)
            
            print("\n" + "="*50)
            print("📊 数据流健康检查")
            print("="*50)
            
            for source_type, source in self.sources.items():
                health = source.get_health()
                
                status_icon = "✅" if health.status == StreamStatus.CONNECTED else "❌"
                print(f"\n{status_icon} {source_type.value}")
                print(f"  状态: {health.status.value}")
                print(f"  数据量: {health.data_count}")
                print(f"  延迟: {health.latency_ms:.1f}ms")
                print(f"  丢包率: {health.packet_loss_rate:.2%}")
                print(f"  错误数: {health.error_count}")
                
                if health.warnings:
                    print("  ⚠️  警告:")
                    for warning in health.warnings:
                        print(f"    - {warning}")
            
            print(f"\n缓冲区: {len(self.data_buffer)}/1000")
            print(f"订阅数: {len(self.subscribers)}")
            print("="*50 + "\n")
    
    def get_all_health(self) -> Dict[DataSourceType, StreamHealth]:
        """获取所有数据源健康状态"""
        return {
            source_type: source.get_health()
            for source_type, source in self.sources.items()
        }
    
    def get_buffer_snapshot(self, limit: int = 10) -> List[StreamData]:
        """获取缓冲区快照"""
        return list(self.data_buffer)[-limit:]


# 使用示例
if __name__ == "__main__":
    # 创建数据流管理器
    manager = StreamManager()
    
    # 添加模拟数据源
    level2_source = MockStreamSource(DataSourceType.LEVEL2)
    tick_source = MockStreamSource(DataSourceType.TICK)
    
    manager.add_source(level2_source)
    manager.add_source(tick_source)
    
    # 连接所有数据源
    print("\n🔌 连接数据源...")
    results = manager.connect_all()
    for source_type, success in results.items():
        print(f"  {source_type.value}: {'成功' if success else '失败'}")
    
    # 定义数据回调
    def on_data(data: StreamData):
        print(f"📨 [{data.source_type.value}] {data.symbol} @ {data.timestamp.strftime('%H:%M:%S')}: "
              f"价格={data.data.get('price'):.2f}, 量={data.data.get('volume')}")
    
    # 订阅股票
    print("\n📡 订阅股票...")
    manager.subscribe(["000001.SZ", "600000.SH"], on_data)
    
    # 启动健康监控
    manager.start_monitoring(interval=5)
    
    # 运行30秒
    print("\n▶️  数据流运行中（30秒）...\n")
    time.sleep(30)
    
    # 断开所有连接
    print("\n🔌 断开所有数据源...")
    manager.disconnect_all()
    
    # 获取最终健康报告
    print("\n📊 最终健康报告:")
    for source_type, health in manager.get_all_health().items():
        print(f"\n{source_type.value}:")
        print(f"  总数据量: {health.data_count}")
        print(f"  总错误数: {health.error_count}")
        print(f"  平均延迟: {health.latency_ms:.1f}ms")
        print(f"  丢包率: {health.packet_loss_rate:.2%}")
    
    print("\n✅ 完成")
