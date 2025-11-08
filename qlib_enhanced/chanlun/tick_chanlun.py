"""盘口级别缠论分析 - P1-3

功能:
- Tick级别实时处理
- 1分钟K线聚合
- L2行情分析 (大单/委买委卖)
- 实时信号生成

应用场景:
- 日内T+0交易
- 高频策略
- 盘口异动监控

作者: Warp AI Assistant
日期: 2025-01
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class FenxingSignal:
    """分型信号"""
    type: str  # 'top' / 'bottom'
    price: float
    time: datetime
    confidence: float


@dataclass
class BuySignal:
    """买卖点信号"""
    type: str  # '1买' / '2买' / '3买'
    price: float
    time: datetime
    confidence: float
    l2_support: bool = False  # L2行情支持


class TickLevelChanLun:
    """Tick级别缠论分析"""
    
    def __init__(
        self,
        agg_period: str = '1min',
        tick_buffer_size: int = 1000,
        use_l2: bool = False
    ):
        """初始化
        
        Args:
            agg_period: 聚合周期 ('1min' / '5min')
            tick_buffer_size: Tick缓存大小
            use_l2: 是否使用L2行情
        """
        self.agg_period = agg_period
        self.tick_buffer = deque(maxlen=tick_buffer_size)
        self.kline_buffer = []
        self.use_l2 = use_l2
        
        # 导入ChanPy特征生成器
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from features.chanlun.chanpy_features import ChanPyFeatureGenerator
            self.chanpy_gen = ChanPyFeatureGenerator()
        except Exception as e:
            logger.warning(f"无法导入ChanPyFeatureGenerator: {e}")
            self.chanpy_gen = None
        
        logger.info(f"Tick级别缠论初始化: {agg_period}, L2={use_l2}")
    
    def process_tick(self, tick_data: Dict) -> Optional[object]:
        """实时处理Tick数据
        
        Args:
            tick_data: {
                'symbol': str,
                'time': datetime,
                'price': float,
                'volume': int,
                'bid': float,
                'ask': float
            }
        
        Returns:
            信号对象 (FenxingSignal / BuySignal) or None
        """
        try:
            # 1. 缓存Tick
            self.tick_buffer.append(tick_data)
            
            # 2. 检查是否到达聚合时间点
            if not self._should_aggregate(tick_data['time']):
                return None
            
            # 3. 聚合为K线
            kline_1m = self._aggregate_ticks()
            if kline_1m is None:
                return None
            
            self.kline_buffer.append(kline_1m)
            
            # 4. 缠论分析 (需要至少20根K线)
            if len(self.kline_buffer) < 20:
                return None
            
            df = pd.DataFrame(self.kline_buffer[-100:])  # 最近100根
            
            # 5. 生成缠论特征
            if self.chanpy_gen:
                features = self.chanpy_gen.generate_features(df, tick_data['symbol'])
                
                # 6. 检测分型
                if 'fx_mark' in features.columns:
                    last_fx = features['fx_mark'].iloc[-1]
                    if last_fx != 0:
                        return FenxingSignal(
                            type='top' if last_fx == 1 else 'bottom',
                            price=kline_1m['close'],
                            time=tick_data['time'],
                            confidence=0.75
                        )
                
                # 7. 检测买卖点
                if 'is_buy_point' in features.columns:
                    if features['is_buy_point'].iloc[-1] == 1:
                        bsp_type = features.get('bsp_type', pd.Series([0])).iloc[-1]
                        return BuySignal(
                            type=f"{int(bsp_type)}买" if bsp_type > 0 else "买点",
                            price=kline_1m['close'],
                            time=tick_data['time'],
                            confidence=0.80
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"处理Tick失败: {e}")
            return None
    
    def analyze_order_book(self, l2_data: Dict) -> Dict:
        """分析L2行情 (委买委卖盘口)
        
        Args:
            l2_data: {
                'bid_prices': [p1, p2, ...],   # 买1-10价
                'bid_volumes': [v1, v2, ...],  # 买1-10量
                'ask_prices': [p1, p2, ...],   # 卖1-10价
                'ask_volumes': [v1, v2, ...]   # 卖1-10量
            }
        
        Returns:
            {
                'order_book_pressure': float,  # >0.3=多头占优
                'large_order_support': float,  # 大单支撑位
                'large_order_resistance': float,  # 大单压力位
                'imbalance_ratio': float       # 买卖失衡比率
            }
        """
        try:
            bid_volumes = np.array(l2_data.get('bid_volumes', []))
            ask_volumes = np.array(l2_data.get('ask_volumes', []))
            bid_prices = np.array(l2_data.get('bid_prices', []))
            ask_prices = np.array(l2_data.get('ask_prices', []))
            
            # 1. 计算买卖压力
            total_bid = bid_volumes.sum()
            total_ask = ask_volumes.sum()
            
            if total_bid + total_ask > 0:
                pressure = (total_bid - total_ask) / (total_bid + total_ask)
            else:
                pressure = 0
            
            # 2. 识别大单
            avg_bid_vol = bid_volumes.mean() if len(bid_volumes) > 0 else 0
            avg_ask_vol = ask_volumes.mean() if len(ask_volumes) > 0 else 0
            
            large_bid_threshold = avg_bid_vol * 3  # 3倍平均为大单
            large_ask_threshold = avg_ask_vol * 3
            
            large_bids = [(bid_prices[i], bid_volumes[i]) 
                         for i in range(len(bid_volumes)) 
                         if bid_volumes[i] > large_bid_threshold]
            
            large_asks = [(ask_prices[i], ask_volumes[i]) 
                         for i in range(len(ask_volumes)) 
                         if ask_volumes[i] > large_ask_threshold]
            
            # 3. 大单支撑/压力位
            support = large_bids[0][0] if large_bids else (bid_prices[0] if len(bid_prices) > 0 else 0)
            resistance = large_asks[0][0] if large_asks else (ask_prices[0] if len(ask_prices) > 0 else 0)
            
            # 4. 失衡比率 (买1/卖1)
            if len(bid_volumes) > 0 and len(ask_volumes) > 0:
                imbalance = bid_volumes[0] / ask_volumes[0] if ask_volumes[0] > 0 else 1
            else:
                imbalance = 1
            
            return {
                'order_book_pressure': pressure,
                'large_order_support': support,
                'large_order_resistance': resistance,
                'imbalance_ratio': imbalance,
                'large_bid_count': len(large_bids),
                'large_ask_count': len(large_asks)
            }
            
        except Exception as e:
            logger.error(f"L2行情分析失败: {e}")
            return {
                'order_book_pressure': 0,
                'large_order_support': 0,
                'large_order_resistance': 0,
                'imbalance_ratio': 1
            }
    
    def _should_aggregate(self, current_time: datetime) -> bool:
        """判断是否应该聚合K线"""
        if len(self.tick_buffer) < 2:
            return False
        
        # 检查是否跨越分钟边界
        if self.agg_period == '1min':
            last_minute = self.tick_buffer[-2]['time'].minute
            current_minute = current_time.minute
            return last_minute != current_minute
        elif self.agg_period == '5min':
            last_5min = self.tick_buffer[-2]['time'].minute // 5
            current_5min = current_time.minute // 5
            return last_5min != current_5min
        
        return False
    
    def _aggregate_ticks(self) -> Optional[Dict]:
        """聚合Tick为K线"""
        if len(self.tick_buffer) == 0:
            return None
        
        try:
            # 获取当前周期内的ticks
            current_period = self._get_current_period()
            period_ticks = [t for t in self.tick_buffer 
                           if self._get_period(t['time']) == current_period]
            
            if len(period_ticks) == 0:
                return None
            
            # 聚合OHLCV
            prices = [t['price'] for t in period_ticks]
            volumes = [t.get('volume', 0) for t in period_ticks]
            
            kline = {
                'datetime': period_ticks[-1]['time'],
                'open': prices[0],
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],
                'volume': sum(volumes)
            }
            
            return kline
            
        except Exception as e:
            logger.error(f"聚合K线失败: {e}")
            return None
    
    def _get_period(self, dt: datetime) -> str:
        """获取时间所属周期"""
        if self.agg_period == '1min':
            return dt.strftime('%Y%m%d%H%M')
        elif self.agg_period == '5min':
            minute_5 = (dt.minute // 5) * 5
            return dt.strftime('%Y%m%d%H') + f"{minute_5:02d}"
        return dt.strftime('%Y%m%d%H%M')
    
    def _get_current_period(self) -> str:
        """获取当前周期"""
        if len(self.tick_buffer) > 0:
            return self._get_period(self.tick_buffer[-1]['time'])
        return ""


class RealtimeChanLunTrader:
    """实时缠论交易器"""
    
    def __init__(self):
        self.tick_chanlun = TickLevelChanLun(use_l2=True)
        self.positions = {}
        
    def on_tick(self, tick: Dict, l2_data: Optional[Dict] = None):
        """Tick回调
        
        Args:
            tick: Tick数据
            l2_data: L2行情数据 (可选)
        """
        # 1. 缠论分析
        signal = self.tick_chanlun.process_tick(tick)
        
        if signal and isinstance(signal, BuySignal):
            # 2. L2行情确认
            if l2_data:
                l2_analysis = self.tick_chanlun.analyze_order_book(l2_data)
                
                # 多头占优 + 大单支撑
                if l2_analysis['order_book_pressure'] > 0.3:
                    signal.l2_support = True
                    logger.info(f"✅ 缠论买点+L2支持: {signal.type} @{signal.price:.2f}")
                    
                    # 3. 执行买入 (需接入实际交易API)
                    # self.execute_order(...)
                else:
                    logger.info(f"⚠️ 缠论买点但L2不支持: {signal.type}")
            else:
                logger.info(f"✅ 缠论买点: {signal.type} @{signal.price:.2f}")


if __name__ == '__main__':
    print("="*60)
    print("P1-3: 盘口级别缠论分析测试")
    print("="*60)
    
    tick_chan = TickLevelChanLun(agg_period='1min')
    
    # 模拟Tick流
    print("\n测试1: Tick处理")
    base_time = datetime.now().replace(second=0, microsecond=0)
    
    for i in range(65):  # 跨越1分钟
        tick = {
            'symbol': 'TEST001',
            'time': base_time + timedelta(seconds=i),
            'price': 10.0 + np.random.randn() * 0.01,
            'volume': 100,
            'bid': 9.99,
            'ask': 10.01
        }
        
        signal = tick_chan.process_tick(tick)
        if signal:
            print(f"✅ 检测到信号: {signal}")
    
    # 测试2: L2行情分析
    print("\n测试2: L2行情分析")
    l2_data = {
        'bid_prices': [10.00, 9.99, 9.98, 9.97, 9.96],
        'bid_volumes': [1000, 800, 600, 400, 300],
        'ask_prices': [10.01, 10.02, 10.03, 10.04, 10.05],
        'ask_volumes': [500, 600, 700, 400, 300]
    }
    
    l2_analysis = tick_chan.analyze_order_book(l2_data)
    print(f"✅ 买卖压力: {l2_analysis['order_book_pressure']:.2f}")
    print(f"   大单支撑: {l2_analysis['large_order_support']:.2f}")
    print(f"   买卖失衡: {l2_analysis['imbalance_ratio']:.2f}")
    
    if l2_analysis['order_book_pressure'] > 0.3:
        print("   ✅ 多头占优")
    else:
        print("   ⚠️ 空头占优")
    
    print("\n✅ P1-3测试完成!")
    print("⚠️  实盘使用需要:")
    print("   1. 接入实时Tick行情源")
    print("   2. 接入L2行情数据 (付费)")
    print("   3. 接入交易API")
