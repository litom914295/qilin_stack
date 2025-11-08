"""
涨停板策略增强 - Phase 5.2
功能: 涨停买点识别、封板强度分析、涨停预测
双模式复用: Qlib涨停因子 + 独立系统涨停信号
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LimitUpSignal:
    """涨停信号"""
    symbol: str
    signal_type: str  # 'pre_limit'|'limit'|'limit_break'
    seal_strength: float  # 封板强度[0-1]
    chanlun_buy: bool  # 是否缠论买点
    buy_point_type: Optional[str] = None
    reason: str = ""


class LimitUpAnalyzer:
    """涨停板分析器"""
    
    def __init__(self, limit_pct: float = 0.099):
        self.limit_pct = limit_pct  # 涨停阈值(9.9%)
    
    def analyze(self, df: pd.DataFrame, symbol: str = "unknown") -> Optional[LimitUpSignal]:
        """分析涨停"""
        if len(df) < 5:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 计算涨幅
        pct_change = (latest['close'] - prev['close']) / prev['close']
        
        # 判断涨停状态
        if pct_change >= self.limit_pct * 0.99:
            signal_type = 'limit'
        elif pct_change >= self.limit_pct * 0.8:
            signal_type = 'pre_limit'
        else:
            return None
        
        # 封板强度
        seal_strength = self._calc_seal_strength(df)
        
        # 缠论买点
        chanlun_buy = bool(latest.get('buy_point'))
        buy_point_type = str(latest['buy_point']) if chanlun_buy else None
        
        reason = f"涨幅{pct_change:.1%}, 封板强度{seal_strength:.2f}"
        if chanlun_buy:
            reason += f", 缠论{buy_point_type}"
        
        return LimitUpSignal(symbol, signal_type, seal_strength, chanlun_buy, buy_point_type, reason)
    
    def _calc_seal_strength(self, df: pd.DataFrame) -> float:
        """计算封板强度"""
        latest = df.iloc[-1]
        
        # 方法1: 成交量萎缩程度
        vol_ratio = latest['volume'] / df['volume'].tail(5).mean() if df['volume'].tail(5).mean() > 0 else 1
        vol_score = max(0, 1 - vol_ratio)  # 成交量越小越强
        
        # 方法2: 振幅
        amplitude = (latest['high'] - latest['low']) / latest['close'] if latest['close'] > 0 else 0
        amp_score = max(0, 1 - amplitude * 10)  # 振幅越小越强
        
        # 综合
        strength = vol_score * 0.6 + amp_score * 0.4
        return max(0.0, min(1.0, strength))


def generate_limit_up_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成涨停特征"""
    df = df.copy()
    
    # 涨幅
    df['pct_change'] = df['close'].pct_change()
    
    # 是否涨停
    df['is_limit_up'] = (df['pct_change'] >= 0.099).astype(int)
    
    # 封板强度 (简化版)
    df['seal_strength'] = 0.0
    
    # 涨停后N日表现
    df['limit_up_next_return'] = 0.0
    limit_idx = df[df['is_limit_up'] == 1].index
    for idx in limit_idx:
        next_idx = df.index.get_loc(idx) + 1
        if next_idx < len(df):
            df.loc[idx, 'limit_up_next_return'] = df.iloc[next_idx]['pct_change']
    
    return df
