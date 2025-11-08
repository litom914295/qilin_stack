"""Tick数据后台Worker - P0-3完整实现

功能:
1. 持续接收Tick数据源(AKShare/Tushare/Mock)
2. 实时计算Tick级别缠论特征
3. 写入SQLite信号存储
4. 支持多股票并发监控

用法:

from web.services.tick_data_worker import TickDataWorker

worker = TickDataWorker(
    symbols=['000001', '600000'],
    source_type='akshare',
    store_path='data/chanlun_signals.sqlite'
)
worker.start()

# 停止
worker.stop()
"""
import os
import sys
import time
import threading
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qlib_enhanced.chanlun.tick_data_connector import TickDataConnector, TickData
from qlib_enhanced.chanlun.tick_chanlun import TickLevelChanLun
from web.services.chanlun_signal_store import ChanLunSignalStore

logger = logging.getLogger(__name__)


@dataclass
class TickDataWorker:
    """Tick数据后台Worker"""
    
    symbols: List[str]
    source_type: str = 'mock'  # 'mock' / 'akshare' / 'tushare'
    store_path: str = 'data/chanlun_signals.sqlite'
    interval_ms: int = 1000  # Mock数据源更新间隔
    tushare_token: Optional[str] = None
    enable_chanlun_analysis: bool = True
    max_tick_buffer: int = 200  # 每只股票最多缓存200个Tick
    
    def __post_init__(self):
        """初始化"""
        self.connector: Optional[TickDataConnector] = None
        self.signal_store = ChanLunSignalStore(db_path=self.store_path)
        self.signal_store.init()
        
        # Tick缓冲区: {symbol: [TickData]}
        self.tick_buffers: Dict[str, List[TickData]] = {s: [] for s in self.symbols}
        
        # Tick级别缠论分析器
        self.chanlun_analyzers: Dict[str, TickLevelChanLun] = {}
        if self.enable_chanlun_analysis:
            for symbol in self.symbols:
                self.chanlun_analyzers[symbol] = TickLevelChanLun(
                    code=symbol,
                    window_size=100
                )
        
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        logger.info(f"TickDataWorker初始化: {len(self.symbols)}只股票, 数据源={self.source_type}")
    
    def start(self):
        """启动Worker"""
        if self.running:
            logger.warning("Worker已经在运行")
            return
        
        # 创建连接器
        kwargs = {'source_type': self.source_type}
        if self.source_type == 'mock':
            kwargs['interval_ms'] = self.interval_ms
        elif self.source_type == 'tushare' and self.tushare_token:
            kwargs['tushare_token'] = self.tushare_token
        
        self.connector = TickDataConnector(**kwargs)
        
        # 连接并订阅
        if not self.connector.connect():
            logger.error("连接Tick数据源失败")
            return
        
        self.connector.subscribe(self.symbols)
        
        # 注册回调
        self.connector.register_callback(self._on_tick_received)
        
        # 启动连接器
        self.connector.start()
        
        # 启动后台处理线程
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("✅ TickDataWorker启动成功")
    
    def stop(self):
        """停止Worker"""
        if not self.running:
            return
        
        logger.info("正在停止TickDataWorker...")
        self.running = False
        
        if self.connector:
            self.connector.stop()
            self.connector.disconnect()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        
        logger.info("✅ TickDataWorker已停止")
    
    def _on_tick_received(self, tick: TickData):
        """接收到Tick数据的回调"""
        symbol = tick.symbol
        
        if symbol not in self.tick_buffers:
            logger.debug(f"忽略未订阅股票: {symbol}")
            return
        
        # 添加到缓冲区
        self.tick_buffers[symbol].append(tick)
        
        # 限制缓冲区大小
        if len(self.tick_buffers[symbol]) > self.max_tick_buffer:
            self.tick_buffers[symbol] = self.tick_buffers[symbol][-self.max_tick_buffer:]
        
        # 实时缠论分析
        if self.enable_chanlun_analysis and symbol in self.chanlun_analyzers:
            self._analyze_tick_chanlun(symbol, tick)
    
    def _analyze_tick_chanlun(self, symbol: str, tick: TickData):
        """分析Tick级别缠论"""
        analyzer = self.chanlun_analyzers[symbol]
        
        # 更新分析器
        analyzer.update(
            timestamp=tick.timestamp,
            price=tick.last_price,
            volume=tick.volume
        )
        
        # 检查买卖点信号
        signals = analyzer.get_recent_signals(limit=1)
        
        if signals and len(signals) > 0:
            latest_signal = signals[0]
            
            # 写入SQLite
            signal_df = pd.DataFrame([{
                'time': datetime.fromtimestamp(tick.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'signal_type': latest_signal['type'],
                'price': tick.last_price,
                'score': latest_signal.get('score', 0),
                'status': '实时'
            }])
            
            try:
                self.signal_store.save_signals(signal_df)
                logger.info(f"🔴 {symbol} Tick信号: {latest_signal['type']} @ {tick.last_price:.2f}")
            except Exception as e:
                logger.error(f"{symbol} 保存信号失败: {e}")
    
    def _worker_loop(self):
        """后台处理循环"""
        logger.info("后台Worker线程启动")
        
        while self.running:
            try:
                # 定期统计缓冲区状态
                total_ticks = sum(len(buf) for buf in self.tick_buffers.values())
                if total_ticks > 0:
                    logger.debug(f"Tick缓冲区: {total_ticks}条, "
                               f"分布={[(s, len(b)) for s, b in self.tick_buffers.items()]}")
                
                # 等待1秒
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Worker循环异常: {e}")
                time.sleep(5)  # 异常后等待5秒
        
        logger.info("后台Worker线程退出")
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """获取缓冲区统计"""
        return {symbol: len(buffer) for symbol, buffer in self.tick_buffers.items()}
    
    def get_latest_ticks(self, symbol: str, limit: int = 10) -> List[TickData]:
        """获取最近的Tick数据"""
        if symbol not in self.tick_buffers:
            return []
        return self.tick_buffers[symbol][-limit:]
    
    def clear_buffer(self, symbol: Optional[str] = None):
        """清空缓冲区"""
        if symbol:
            if symbol in self.tick_buffers:
                self.tick_buffers[symbol].clear()
                logger.info(f"已清空 {symbol} 缓冲区")
        else:
            for s in self.tick_buffers:
                self.tick_buffers[s].clear()
            logger.info("已清空所有缓冲区")


def run_demo():
    """演示运行"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 创建Worker
    worker = TickDataWorker(
        symbols=['000001', '600000', '000002'],
        source_type='mock',
        interval_ms=500,
        store_path='data/chanlun_signals.sqlite'
    )
    
    print("启动TickDataWorker...")
    worker.start()
    
    try:
        # 运行30秒
        for i in range(30):
            time.sleep(1)
            stats = worker.get_buffer_stats()
            print(f"[{i+1}s] 缓冲区状态: {stats}")
            
            # 每10秒显示最近的信号
            if (i + 1) % 10 == 0:
                recent_signals = worker.signal_store.load_signals(limit=5)
                if len(recent_signals) > 0:
                    print("\n最近5条信号:")
                    print(recent_signals[['time', 'symbol', 'signal_type', 'price', 'status']])
                    print()
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        print("停止Worker...")
        worker.stop()
        print("✅ 演示完成")


if __name__ == '__main__':
    run_demo()
