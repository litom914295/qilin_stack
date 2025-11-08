"""CZSC缠论特征提取器"""

import pandas as pd
import numpy as np
from czsc import CZSC
from czsc.objects import RawBar
from czsc.enum import Freq
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CzscFeatureGenerator:
    """
    CZSC缠论特征生成器
    
    功能:
    - 分型识别
    - 笔方向/位置/幅度
    - 中枢判断
    - 距离分型K线数
    """
    
    def __init__(self, freq='日线'):
        """初始化CZSC特征生成器
        
        Args:
            freq: 周期类型，可选'日线','60分','30分','15分'等，或直接传入Freq枚举
        """
        # 转换字符串频率为Freq枚举
        if isinstance(freq, str):
            freq_map = {
                '日线': Freq.D,
                '60分': Freq.F60,
                '30分': Freq.F30,
                '15分': Freq.F15,
                '5分': Freq.F5,
            }
            self.freq = freq_map.get(freq, Freq.D)
        else:
            self.freq = freq
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从价格数据生成缠论特征
        
        Args:
            df: DataFrame with columns [datetime, open, close, high, low, volume]
        
        Returns:
            df with 缠论特征列
        """
        if len(df) < 10:
            logger.warning(f"数据不足10条, 跳过缠论特征计算")
            return self._add_empty_features(df)
        
        try:
            # 1. 转换为RawBar格式
            bars = self._to_raw_bars(df)
            
            # 2. 初始化CZSC
            czsc = CZSC(bars)
            
            # 3. 提取缠论特征
            features = self._extract_chanlun_features(czsc, len(df))
            
            # 4. 合并回原始DataFrame
            result = df.copy()
            for col, values in features.items():
                result[col] = values
            
            return result
            
        except Exception as e:
            logger.error(f"CZSC特征生成失败: {e}")
            # 返回空特征
            return self._add_empty_features(df)
    
    def _to_raw_bars(self, df: pd.DataFrame) -> List[RawBar]:
        """转换DataFrame为RawBar列表"""
        bars = []
        for idx, row in df.iterrows():
            bar = RawBar(
                symbol=row.get('symbol', 'UNKNOWN'),
                id=idx if isinstance(idx, int) else 0,
                freq=self.freq,
                dt=pd.to_datetime(row['datetime']),
                open=float(row['open']),
                close=float(row['close']),
                high=float(row['high']),
                low=float(row['low']),
                vol=float(row.get('volume', 0)),
                amount=float(row.get('amount', 0))
            )
            bars.append(bar)
        return bars
    
    def _extract_chanlun_features(self, czsc: CZSC, n: int) -> Dict[str, np.ndarray]:
        """从CZSC对象提取缠论特征"""
        features = {}
        
        # 特征1: 分型标记 (1=顶分型, -1=底分型, 0=无)
        fx_marks = np.zeros(n)
        for fx in czsc.fx_list:
            for i, bar in enumerate(czsc.bars_raw):
                if bar.dt == fx.dt:
                    fx_marks[i] = 1 if fx.mark.value == 'g' else -1
                    break
        features['fx_mark'] = fx_marks
        
        # 特征2: 笔方向 (1=上涨笔, -1=下跌笔, 0=无)
        bi_marks = np.zeros(n)
        for bi in czsc.bi_list:
            for i, bar in enumerate(czsc.bars_raw):
                if bi.sdt <= bar.dt <= bi.edt:
                    bi_marks[i] = 1 if bi.direction.value == 'up' else -1
        features['bi_direction'] = bi_marks
        
        # 特征3: 笔内位置 (0-1, 0=笔起点, 1=笔终点)
        bi_position = np.zeros(n)
        for bi in czsc.bi_list:
            bi_bars = [bar for bar in czsc.bars_raw if bi.sdt <= bar.dt <= bi.edt]
            if len(bi_bars) > 1:
                for j, bar in enumerate(bi_bars):
                    for i, raw_bar in enumerate(czsc.bars_raw):
                        if raw_bar.dt == bar.dt:
                            bi_position[i] = j / (len(bi_bars) - 1)
                            break
        features['bi_position'] = bi_position
        
        # 特征4: 笔幅度
        bi_power = np.zeros(n)
        for bi in czsc.bi_list:
            power = bi.power
            for i, bar in enumerate(czsc.bars_raw):
                if bi.sdt <= bar.dt <= bi.edt:
                    bi_power[i] = power
        features['bi_power'] = bi_power
        
        # 特形5: 是否在中枢内 (1=是, 0=否)
        # 注: CZSC 0.10.3版本中枢需要从线段中计算，暂时留空
        in_zs = np.zeros(n)
        features['in_zs'] = in_zs
        
        # 特征6: 距离最近分型的K线数
        bars_since_fx = np.full(n, 999)
        last_fx_idx = -999
        for i in range(n):
            if fx_marks[i] != 0:
                last_fx_idx = i
            bars_since_fx[i] = i - last_fx_idx if last_fx_idx >= 0 else 999
        features['bars_since_fx'] = bars_since_fx
        
        return features
    
    def _add_empty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加空特征"""
        result = df.copy()
        for col in ['fx_mark', 'bi_direction', 'bi_position', 
                   'bi_power', 'in_zs', 'bars_since_fx']:
            result[col] = 0
        return result
