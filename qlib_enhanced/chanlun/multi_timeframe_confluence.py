"""多周期共振评分器 - P2-1

目标:
- 统一在多时间框架(日/周/月或分钟级)上评估方向一致性
- 为交易信号提供共振加权分数,用于过滤与排序

用法:

from qlib_enhanced.chanlun.multi_timeframe_confluence import (
    resample_ohlc, compute_direction, compute_confluence_score
)

# 1) 准备不同周期的OHLC数据(DataFrame需含: datetime, open, high, low, close, volume)
#    假设 df_day, df_week, df_month 已就绪

# 2) 计算方向 (1=多, -1=空, 0=震荡/不确定)
dirs = {
    'D': compute_direction(df_day),
    'W': compute_direction(df_week),
    'M': compute_direction(df_month),
}

# 3) 计算共振分数 (默认权重: D=1, W=1.3, M=1.6)
score = compute_confluence_score(dirs)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd


DEFAULT_WEIGHTS: Dict[str, float] = {
    '1m': 0.6,
    '5m': 0.8,
    '15m': 1.0,
    '30m': 1.1,
    '60m': 1.2,
    'D': 1.0,
    'W': 1.3,
    'M': 1.6,
}


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """按给定周期重采样OHLCV

    Args:
        df: 列包含 ['datetime','open','high','low','close','volume']
        rule: 采样规则, 如 'W'/'M' 或 '15min'
    """
    if df is None or df.empty:
        return df
    if 'datetime' in df.columns:
        x = df.set_index(pd.to_datetime(df['datetime']))
    else:
        x = df.copy()
        x.index = pd.to_datetime(x.index)
    o = x['open'].resample(rule).first()
    h = x['high'].resample(rule).max()
    l = x['low'].resample(rule).min()
    c = x['close'].resample(rule).last()
    v = x['volume'].resample(rule).sum()
    y = pd.concat([o, h, l, c, v], axis=1)
    y.columns = ['open','high','low','close','volume']
    y = y.dropna(how='any')
    y = y.reset_index().rename(columns={'index':'datetime'})
    return y


def _slope(series: pd.Series, last_n: int = 12) -> float:
    """近段线性斜率(用于判定方向)
    返回值>0表多,<0表空
    """
    n = min(last_n, len(series))
    if n < 3:
        return 0.0
    y = series.values[-n:]
    x = np.arange(n)
    k, b = np.polyfit(x, y, 1)
    return float(k)


def _volatility(series: pd.Series, last_n: int = 12) -> float:
    n = min(last_n, len(series))
    if n < 3:
        return 0.0
    y = series.values[-n:]
    return float(np.std(np.diff(y)))


def compute_direction(df: pd.DataFrame, last_n: int = 20) -> int:
    """根据价格斜率与波动度粗略判定方向
    1=多头, -1=空头, 0=震荡/不确定
    """
    if df is None or df.empty:
        return 0
    s = _slope(df['close'], last_n=last_n)
    vol = _volatility(df['close'], last_n=last_n)

    # 自适应阈值: 用价格量级和波动衡量
    level = max(1e-6, abs(df['close'].tail(last_n).mean()) * 0.0005)  # 0.05%
    thresh = max(level, vol * 0.05)

    if s > thresh:
        return 1
    if s < -thresh:
        return -1
    return 0


def compute_confluence_score(directions: Dict[str, int], weights: Dict[str, float] | None = None) -> float:
    """计算多周期共振分数

    Args:
        directions: {'D':1, 'W':1, 'M':-1, '60m':0, ...}
        weights: 每个周期的权重, 默认为 DEFAULT_WEIGHTS

    Returns:
        共振分数: [-sum(w), +sum(w)] 区间, 绝对值越大代表一致性越强
    """
    if not directions:
        return 0.0
    w = (weights or DEFAULT_WEIGHTS).copy()
    score = 0.0
    for tf, d in directions.items():
        if d == 0:
            continue
        score += float(w.get(tf, 1.0)) * float(np.sign(d))
    return score


@dataclass
class ConfluenceReport:
    score: float
    aligned: Dict[str, int]  # 各周期方向
    majority: int            # 主方向: 1 / -1 / 0


def summarize_confluence(directions: Dict[str, int], weights: Dict[str, float] | None = None) -> ConfluenceReport:
    score = compute_confluence_score(directions, weights)
    nonzero = [d for d in directions.values() if d != 0]
    majority = int(np.sign(np.sum(nonzero))) if nonzero else 0
    return ConfluenceReport(score=score, aligned=directions, majority=majority)
