"""
TechnicalIndicators: common TA indicators used by tests.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Basic technical indicators for tests."""

    @staticmethod
    def sma(series: pd.Series, window: int = 20) -> pd.Series:
        return pd.Series(series, copy=False).rolling(window=window, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, span: int = 20) -> pd.Series:
        return pd.Series(series, copy=False).ewm(span=span, adjust=False).mean()

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        close = pd.Series(series, copy=False)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': hist,
        })

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = pd.Series(series, copy=False).diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method='bfill').fillna(0)

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
        high_k = pd.Series(high).rolling(window=k, min_periods=1).max()
        low_k = pd.Series(low).rolling(window=k, min_periods=1).min()
        percent_k = (pd.Series(close) - low_k) / (high_k - low_k).replace(0, np.nan) * 100
        percent_d = percent_k.rolling(window=d, min_periods=1).mean()
        return pd.DataFrame({'K': percent_k.fillna(0), 'D': percent_d.fillna(0)})

    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        mid = pd.Series(series).rolling(window=window, min_periods=1).mean()
        std = pd.Series(series).rolling(window=window, min_periods=1).std(ddof=0)
        upper = mid + num_std * std
        lower = mid - num_std * std
        return pd.DataFrame({'upper': upper, 'middle': mid, 'lower': lower})

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        close = pd.Series(close)
        volume = pd.Series(volume)
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum().fillna(0)

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3.0
        vol = pd.Series(volume)
        cum_vp = (typical_price * vol).cumsum()
        cum_vol = vol.cumsum().replace(0, np.nan)
        return (cum_vp / cum_vol).fillna(method='bfill').fillna(0)
