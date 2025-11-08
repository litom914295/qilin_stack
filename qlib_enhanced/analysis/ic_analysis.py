"""
IC分析核心算法
Phase 5.3 实现

提供：
- Qlib数据加载（表达式）
- IC/Rank IC 时间序列计算
- 月度IC热力图
- 分层（分位）收益分析
- 统计摘要
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import qlib
    from qlib.data import D
except Exception:
    qlib = None
    D = None
    logger.warning("Qlib 未可用，Qlib模式将不可用")


@dataclass
class ICResult:
    ic_series: pd.Series
    rank_ic_series: Optional[pd.Series]
    stats: Dict[str, float]
    monthly_ic: pd.DataFrame
    layered_returns: pd.DataFrame


def init_qlib_if_needed(provider_uri: Optional[str] = None, region: str = "cn") -> None:
    if qlib is None:
        raise RuntimeError("未安装Qlib，无法使用Qlib模式")
    try:
        # 多次init也安全
        qlib.init(provider_uri=provider_uri, region=region)
    except Exception as e:
        logger.warning(f"Qlib初始化警告: {e}")


def load_factor_from_qlib(
    instruments: str = "csi300",
    start: str = "2018-01-01",
    end: str = "2021-12-31",
    factor_expr: str = "Ref($close, 0) / Ref($close, 1) - 1",
    label_expr: str = "Ref($close, -1) / $close - 1",
    provider_uri: Optional[str] = None,
) -> pd.DataFrame:
    """从Qlib加载因子与标签数据
    返回列: ['factor','label']，MultiIndex: [datetime, instrument]
    """
    init_qlib_if_needed(provider_uri)
    df = D.features(
        instruments=instruments,
        fields=[factor_expr, label_expr],
        start_time=start,
        end_time=end,
        freq="day",
    )
    df.columns = ["factor", "label"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _by_date_corr(series_x: pd.Series, series_y: pd.Series, method: str = "spearman") -> float:
    if len(series_x) < 3:
        return np.nan
    if method == "spearman":
        from scipy.stats import spearmanr
        return spearmanr(series_x, series_y, nan_policy="omit").correlation
    else:
        # pearson
        return np.corrcoef(series_x, series_y)[0, 1]


def compute_ic_timeseries(
    df: pd.DataFrame,
    factor_col: str = "factor",
    label_col: str = "label",
    method: str = "spearman",
) -> pd.Series:
    """按日期横截面计算IC时间序列"""
    assert isinstance(df.index, pd.MultiIndex) and len(df.index.names) >= 2, "需要MultiIndex[datetime,instrument]"
    by_date = df[[factor_col, label_col]].groupby(level=0)
    ic_series = by_date.apply(lambda x: _by_date_corr(x[factor_col], x[label_col], method=method))
    ic_series.name = "IC"
    return ic_series


def compute_monthly_ic_heatmap(ic_series: pd.Series) -> pd.DataFrame:
    """计算月度平均IC，返回形如 Year x Month 的矩阵"""
    s = ic_series.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    dfm = (
        s.to_frame("IC")
        .assign(Year=lambda d: d.index.year, Month=lambda d: d.index.month)
        .groupby(["Year", "Month"]).mean()["IC"]
        .unstack("Month")
        .sort_index()
    )
    # 补全1-12月列
    for m in range(1, 13):
        if m not in dfm.columns:
            dfm[m] = np.nan
    dfm = dfm[sorted(dfm.columns)]
    return dfm


def layered_return_analysis(
    df: pd.DataFrame,
    factor_col: str = "factor",
    label_col: str = "label",
    quantiles: int = 5,
) -> pd.DataFrame:
    """分层收益分析：将横截面按分位分层，计算各层的平均label
    返回列: ['layer','mean_ret','count']
    """
    idx_names = df.index.names
    by_date = df[[factor_col, label_col]].dropna().groupby(level=0)

    records: List[Tuple[int, float, int]] = []
    for dt, x in by_date:
        try:
            q = pd.qcut(x[factor_col].rank(method="first"), q=quantiles, labels=False) + 1
        except Exception:
            # 数据不足无法分位
            continue
        tmp = x.assign(layer=q)
        g = tmp.groupby("layer")[label_col].agg(["mean", "count"]).reset_index()
        for _, row in g.iterrows():
            records.append((int(row["layer"]), float(row["mean"]), int(row["count"])) )

    res = pd.DataFrame(records, columns=["layer", "mean_ret", "count"]) if records else pd.DataFrame(columns=["layer","mean_ret","count"])
    if len(res) == 0:
        return res
    layered = res.groupby("layer").agg({"mean_ret": "mean", "count": "sum"}).reset_index()
    layered["layer"] = layered["layer"].astype(int)
    layered = layered.sort_values("layer")
    layered["long_short"] = layered["mean_ret"].iloc[-1] - layered["mean_ret"].iloc[0]
    return layered


def ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
    s = ic_series.dropna()
    if len(s) == 0:
        return {k: 0.0 for k in ["ic_mean","ic_std","icir","pos_rate","t_stat","p05","p95"]}
    ic_mean = float(s.mean())
    ic_std = float(s.std(ddof=1))
    icir = float(ic_mean / (ic_std + 1e-12))
    pos_rate = float((s > 0).mean())
    # 简易t统计
    t_stat = float(ic_mean / (ic_std / np.sqrt(len(s) + 1e-12)))
    p05 = float(np.percentile(s, 5))
    p95 = float(np.percentile(s, 95))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "pos_rate": pos_rate,
        "t_stat": t_stat,
        "p05": p05,
        "p95": p95,
    }


def run_ic_pipeline(
    df: pd.DataFrame,
    factor_col: str = "factor",
    label_col: str = "label",
    quantiles: int = 5,
    method: str = "spearman",
) -> ICResult:
    ic_series = compute_ic_timeseries(df, factor_col=factor_col, label_col=label_col, method=method)
    monthly_ic = compute_monthly_ic_heatmap(ic_series)
    layered = layered_return_analysis(df, factor_col=factor_col, label_col=label_col, quantiles=quantiles)
    stats = ic_statistics(ic_series)
    return ICResult(ic_series=ic_series, rank_ic_series=None, stats=stats, monthly_ic=monthly_ic, layered_returns=layered)
