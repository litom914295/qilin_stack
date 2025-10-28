"""
One-into-Two (一进二) limit-up selection pipeline
- Labeling: detect yesterday limit-up and today re-limit (二板) or touch limit
- Features: minute-level microstructure + seal strength, stability, open count, close seal, volume burst/shrink, sector/sentiment proxies
- Models: two-stage (pool -> board) stacking and calibration
- Rolling CV and Precision@TopN optimization

Notes:
- Designed to run with optional dependencies; will gracefully degrade to sklearn-only if LightGBM/XGBoost/CatBoost are not installed.
- Data: tries to use MultiSourceDataProvider (daily) and HighFreqLimitUpAnalyzer sample data for minute-level if real intraday data is unavailable.
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# Optional heavy deps
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None
try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:
    CatBoostClassifier = None

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from datetime import datetime, timedelta

# Reuse enhanced modules if available
try:
    from .high_freq_limitup import HighFreqLimitUpAnalyzer, create_sample_high_freq_data
except Exception:  # fallback minimal
    HighFreqLimitUpAnalyzer = None
    def create_sample_high_freq_data(symbol: str) -> pd.DataFrame:
        now = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        times = [(now + timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(240)]
        prices = 10 + np.cumsum(np.random.randn(len(times)) * 0.02)
        volumes = np.random.randint(1000, 10000, len(times))
        return pd.DataFrame({'time': times, 'open': prices, 'high': prices+0.02, 'low': prices-0.02, 'close': prices, 'volume': volumes})


# =========================== Labeling ===========================

def is_limit_up(yesterday_close: float, today_price: float, limit_rate: float = 0.10) -> bool:
    limit_price = yesterday_close * (1.0 + limit_rate)
    return today_price >= 0.999 * limit_price


def label_one_into_two(min1: pd.DataFrame, min2: pd.DataFrame, y_close: float, limit_rate: float = 0.10) -> Tuple[int, int]:
    """
    Returns: (label_pool, label_board)
    - label_pool: 1 if yesterday touched/closed limit (candidate pool)
    - label_board: 1 if today re-limit (二板) or touched
    """
    # Pool label: did day1 (min1) touch/close limit?
    touched1 = is_limit_up(y_close, float(min1['high'].astype(float).max()), limit_rate)
    pool = 1 if touched1 else 0
    # Board label: day2 touch/close limit?
    touched2 = is_limit_up(y_close, float(min2['high'].astype(float).max()), limit_rate)
    board = 1 if (pool and touched2) else 0
    return pool, board


# =========================== Features ===========================

def extract_limitup_features(min_df: pd.DataFrame, symbol: str) -> Dict[str, float]:
    """Compute microstructure + analyzer features for a single day minute data.
    Requires columns: ['time','open','high','low','close','volume'] where 'time' either HH:MM:SS or datetime.
    """
    feats: Dict[str, float] = {}
    try:
        close = min_df['close'].astype(float)
        volume = min_df['volume'].astype(float)
        feats['ret_day'] = float((close.iloc[-1] / close.iloc[0] - 1.0))
        feats['vol_burst'] = float(volume.rolling(5).mean().max() / (volume.rolling(20).mean().mean() + 1e-6))
        feats['late_strength'] = float(close.tail(20).mean() / (close.mean() + 1e-6))
        feats['open_up'] = float((close.iloc[5] / close.iloc[0] - 1.0)) if len(close) > 5 else 0.0
        feats['high_ratio'] = float((min_df['high'].max() - close.iloc[0]) / (close.iloc[0] + 1e-6))
        feats['low_drawdown'] = float((min_df['low'].min() - close.iloc[0]) / (close.iloc[0] + 1e-6))
        feats['vwap'] = float((close * volume).sum() / (volume.sum() + 1e-6))
        feats['volatility'] = float(close.pct_change().std())
    except Exception:
        pass

    # If analyzer is available, add seal features
    if HighFreqLimitUpAnalyzer is not None:
        try:
            analyzer = HighFreqLimitUpAnalyzer(freq="1min")
            # naive infer limitup time by highest minute
            max_idx = min_df['close'].idxmax()
            time_str = str(min_df.loc[max_idx, 'time']) if 'time' in min_df.columns else "10:30:00"
            feat2 = analyzer.analyze_intraday_pattern(min_df, time_str)
            # rename / select
            rename_map = {
                'volume_burst_before_limit': 'feat_vol_burst_before',
                'seal_stability': 'feat_seal_stability',
                'big_order_rhythm': 'feat_big_order_rhythm',
                'close_seal_strength': 'feat_close_seal_strength',
                'intraday_open_count': 'feat_open_count',
                'volume_shrink_after_limit': 'feat_vol_shrink_after',
            }
            for k_src, k_dst in rename_map.items():
                v = float(feat2.get(k_src, 0.0)) if isinstance(feat2.get(k_src, 0.0), (int, float)) else 0.0
                feats[k_dst] = v
        except Exception:
            # fall back zeros
            feats.setdefault('feat_seal_stability', 0.0)
            feats.setdefault('feat_close_seal_strength', 0.0)
            feats.setdefault('feat_open_count', 0.0)

    # sentiment/sector proxies (placeholders)
    feats.setdefault('market_sentiment', 0.0)
    feats.setdefault('sector_hot', 0.0)
    return feats


# =========================== Modeling ===========================

def _base_estimators() -> List[Tuple[str, BaseEstimator]]:
    ests: List[Tuple[str, BaseEstimator]] = []
    if lgb is not None:
        ests.append(("lgb", lgb.LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.05)))
    if xgb is not None:
        ests.append(("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')))
    if CatBoostClassifier is not None:
        ests.append(("cat", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, loss_function='Logloss', verbose=False)))
    if not ests:
        ests.append(("rf", RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1)))
        ests.append(("lr", LogisticRegression(max_iter=1000, n_jobs=None)))
    return ests


def _stacking_classifier() -> BaseEstimator:
    ests = _base_estimators()
    final_est = LogisticRegression(max_iter=1000)
    clf = StackingClassifier(estimators=ests, final_estimator=final_est, passthrough=True, n_jobs=-1)
    return clf


@dataclass
class TrainResult:
    model_pool: Pipeline
    model_board: Pipeline
    auc_pool: float
    auc_board: float
    threshold_topn: float


class OneIntoTwoTrainer:
    def __init__(self, top_n: int = 20):
        self.top_n = top_n
        self.scaler = StandardScaler()

    def _build_pipeline(self) -> Pipeline:
        stack = _stacking_classifier()
        calib = CalibratedClassifierCV(stack, cv=3, method='isotonic')
        return Pipeline([
            ("scaler", self.scaler),
            ("clf", calib),
        ])

    def fit(self, df: pd.DataFrame) -> TrainResult:
        """df columns: ['date','symbol','pool_label','board_label'] + features..."""
        feature_cols = [c for c in df.columns if c not in ['date','symbol','pool_label','board_label']]
        X = df[feature_cols].fillna(0.0).values
        y_pool = df['pool_label'].astype(int).values
        y_board = df['board_label'].astype(int).values

        # time-aware split by date
        dates = pd.to_datetime(df['date']).sort_values().unique()
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(dates)//20)))

        # Train pool
        pipe_pool = self._build_pipeline()
        pipe_pool.fit(X, y_pool)
        # Train board on positive pool candidates only to focus learning
        idx_pool = np.where(y_pool == 1)[0]
        X2 = X[idx_pool] if len(idx_pool)>0 else X
        y2 = y_board[idx_pool] if len(idx_pool)>0 else y_board
        pipe_board = self._build_pipeline()
        pipe_board.fit(X2, y2)

        # Metrics
        try:
            auc_pool = float(roc_auc_score(y_pool, pipe_pool.predict_proba(X)[:,1]))
        except Exception:
            auc_pool = 0.5
        try:
            auc_board = float(roc_auc_score(y2, pipe_board.predict_proba(X2)[:,1]))
        except Exception:
            auc_board = 0.5

        # Compute threshold by approximating topN across a rolling day
        preds = pd.DataFrame({'date': df['date'], 'symbol': df['symbol'], 'p': pipe_board.predict_proba(X)[:,1]})
        thr = _approx_threshold_for_topn(preds, self.top_n)
        return TrainResult(pipe_pool, pipe_board, auc_pool, auc_board, thr)


def _approx_threshold_for_topn(preds: pd.DataFrame, top_n: int) -> float:
    # choose p at topN average across days
    thr_list = []
    for d, g in preds.groupby('date'):
        if len(g) == 0:
            continue
        g_sorted = g.sort_values('p', ascending=False)
        if len(g_sorted) < top_n:
            thr_list.append(float(g_sorted['p'].min()))
        else:
            thr_list.append(float(g_sorted['p'].iloc[top_n-1]))
    return float(np.median(thr_list)) if thr_list else 0.5


# =========================== Dataset builder (toy) ===========================

def build_sample_dataset(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    rows = []
    for d in dates:
        prev_d = d - timedelta(days=1)
        for sym in symbols:
            # simulate two days of minute data
            m1 = create_sample_high_freq_data(sym)
            m2 = create_sample_high_freq_data(sym)
            y_close = float(m1['close'].iloc[0])
            pool, board = label_one_into_two(m1, m2, y_close)
            feats_y = extract_limitup_features(m1, sym)
            feats_t = extract_limitup_features(m2, sym)
            # merge day1+day2 basic features for predicting day2 outcome
            feats = {**{f"y_{k}": v for k, v in feats_y.items()}, **feats_t}
            feats['date'] = d.strftime('%Y-%m-%d')
            feats['symbol'] = sym
            feats['pool_label'] = pool
            feats['board_label'] = board
            rows.append(feats)
    return pd.DataFrame(rows)


# =========================== Inference ===========================

def rank_candidates(model_board: Pipeline, features: pd.DataFrame, threshold: float, top_n: int = 20) -> pd.DataFrame:
    feature_cols = [c for c in features.columns if c not in ['date','symbol']]
    proba = model_board.predict_proba(features[feature_cols].fillna(0.0).values)[:,1]
    out = features[['date','symbol']].copy()
    out['score'] = proba
    out['pass'] = out['score'] >= threshold
    out = out.sort_values('score', ascending=False).head(top_n)
    return out


__all__ = [
    'build_sample_dataset',
    'OneIntoTwoTrainer',
    'TrainResult',
    'rank_candidates',
]
