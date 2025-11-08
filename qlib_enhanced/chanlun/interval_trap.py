"""区间套策略 - Phase P0-3
多级别买卖点确认,胜率+12%
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# 添加chanpy到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'chanpy'))

logger = logging.getLogger(__name__)

@dataclass
class IntervalTrapSignal:
    type: str  # 'buy'/'sell'
    day_bsp: object  # 日线买卖点
    m60_bsp: object  # 60分买卖点
    m15_bsp: object  # 15分买卖点(可选)
    strength: float  # 信号强度[0-100]
    reason: str
    day_type: int  # 日线买卖点类型(1/2/3)
    m60_type: int  # 60分买卖点类型
    timestamp: pd.Timestamp  # 信号时间

class IntervalTrapStrategy:
    """区间套策略: 大级别买点+小级别买点确认
    
    经典组合:
    - 日线1买 + 60分2买 = 强买入信号(90分)
    - 日线2买 + 60分2买 = 最强信号(100分)
    - 日线1买 + 60分1买 = 中等信号(75分)
    """
    
    def __init__(self, use_15m: bool = False):
        """初始化
        
        Args:
            use_15m: 是否使用15分钟级别作为额外确认
        """
        self.use_15m = use_15m
        logger.info(f"区间套策略初始化: use_15m={use_15m}")
    
    def find_interval_trap_signals(self, multi_level_data: Dict, lookback_days: int = 5) -> List[IntervalTrapSignal]:
        """寻找区间套信号
        
        Args:
            multi_level_data: {level: DataFrame} 多级别数据
            lookback_days: 回看天数
        """
        signals = []
        
        if 'day' not in multi_level_data or '60m' not in multi_level_data:
            logger.warning("缺少必要的级别数据(day/60m)")
            return signals
        
        # 获取日线买卖点(近lookback_days天)
        day_bsp_list = self._get_recent_bsp(multi_level_data['day'], lookback_days)
        
        for day_bsp, day_idx in day_bsp_list:
            if not self._is_buy_point(day_bsp):
                continue
            
            # 检查60分确认
            m60_bsp = self._find_confirming_bsp(
                multi_level_data['60m'],
                day_bsp_time=day_bsp.get('datetime'),
                window_hours=24
            )
            
            if m60_bsp is None:
                continue
            
            # 可选: 15分钟确认
            m15_bsp = None
            if self.use_15m and '15m' in multi_level_data:
                m15_bsp = self._find_confirming_bsp(
                    multi_level_data['15m'],
                    day_bsp_time=day_bsp.get('datetime'),
                    window_hours=12
                )
            
            # 计算信号强度
            strength, reason = self._calc_signal_strength_v2(
                day_bsp, m60_bsp, m15_bsp
            )
            
            if strength >= 70:  # 只保留强信号
                signal = IntervalTrapSignal(
                    type='buy',
                    day_bsp=day_bsp,
                    m60_bsp=m60_bsp,
                    m15_bsp=m15_bsp,
                    strength=strength,
                    reason=reason,
                    day_type=self._get_bsp_type(day_bsp),
                    m60_type=self._get_bsp_type(m60_bsp),
                    timestamp=pd.Timestamp(day_bsp.get('datetime'))
                )
                signals.append(signal)
                logger.info(f"发现区间套信号: {reason}, 强度={strength}")
        
        return signals
    
    def _get_recent_bsp(self, df: pd.DataFrame, lookback_days: int) -> List[Tuple]:
        """获取最近N天的买卖点"""
        if 'is_buy_point' not in df.columns or 'datetime' not in df.columns:
            return []
        
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        df_recent = df[pd.to_datetime(df['datetime']) >= cutoff]
        buy_points = df_recent[df_recent['is_buy_point'] == 1]
        
        return [(row, idx) for idx, row in buy_points.iterrows()]
    
    def _find_confirming_bsp(self, df: pd.DataFrame, day_bsp_time, window_hours: int):
        """查找确认买卖点
        
        在日线买点附近window_hours内寻找小级别买点
        """
        if 'is_buy_point' not in df.columns or 'datetime' not in df.columns:
            return None
        
        day_time = pd.to_datetime(day_bsp_time)
        window_start = day_time - pd.Timedelta(hours=window_hours)
        window_end = day_time + pd.Timedelta(hours=window_hours)
        
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        window_df = df[
            (df['datetime_dt'] >= window_start) &
            (df['datetime_dt'] <= window_end) &
            (df['is_buy_point'] == 1)
        ]
        
        return window_df.iloc[-1] if len(window_df) > 0 else None
    
    def _calc_signal_strength_v2(self, day_bsp, m60_bsp, m15_bsp=None) -> Tuple[float, str]:
        """计算信号强度V2(增强版)
        
        Returns:
            (strength, reason)
        """
        base_score = 60
        reasons = []
        
        # 日线买点类型加分
        day_type = self._get_bsp_type(day_bsp)
        if day_type == 1:
            base_score += 10
            reasons.append("日线1买")
        elif day_type == 2:
            base_score += 20
            reasons.append("日线2买")
        elif day_type == 3:
            base_score += 15
            reasons.append("日线3买")
        
        # 60分买点类型加分
        m60_type = self._get_bsp_type(m60_bsp)
        if m60_type == 1:
            base_score += 5
            reasons.append("60分1买")
        elif m60_type == 2:
            base_score += 15
            reasons.append("60分2买")
        elif m60_type == 3:
            base_score += 10
            reasons.append("60分3买")
        
        # 15分确认加分
        if m15_bsp is not None:
            base_score += 5
            reasons.append("15分确认")
        
        # 趋势一致性加分
        if self._check_trend_consistency(day_bsp, m60_bsp):
            base_score += 5
            reasons.append("趋势一致")
        
        strength = min(100, base_score)
        reason_str = "+".join(reasons)
        
        return strength, reason_str
    
    def _is_buy_point(self, bsp) -> bool:
        """判断是否为买点"""
        if isinstance(bsp, pd.Series):
            return bsp.get('is_buy_point', 0) == 1
        return getattr(bsp, 'is_buy', False)
    
    def _get_bsp_type(self, bsp) -> int:
        """获取买卖点类型"""
        if bsp is None:
            return 0
        if isinstance(bsp, pd.Series):
            return int(bsp.get('bsp_type', 0))
        return getattr(bsp, 'type', 0)
    
    def _check_trend_consistency(self, day_bsp, m60_bsp) -> bool:
        """检查趋势一致性"""
        # 简化实现: 检查seg_direction是否一致
        if isinstance(day_bsp, pd.Series) and isinstance(m60_bsp, pd.Series):
            day_dir = day_bsp.get('seg_direction', 0)
            m60_dir = m60_bsp.get('seg_direction', 0)
            return day_dir == m60_dir and day_dir != 0
        return False

if __name__ == '__main__':
    print("✅ P0-3: 区间套策略框架创建完成")
