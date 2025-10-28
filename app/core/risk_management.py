"""
Risk management utilities for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


class PositionSizer:
    def kelly_criterion(self, win_prob: float, reward_risk: float) -> float:
        # f* = p - (1-p)/R
        p = float(win_prob)
        R = max(1e-9, float(reward_risk))
        f = p - (1 - p) / R
        return float(max(0.0, min(1.0, f)))

    def fixed_fractional(self, equity: float, risk_pct: float, entry_price: float, stop_price: float) -> float:
        risk_per_share = max(1e-9, abs(entry_price - stop_price))
        capital_at_risk = max(0.0, equity * max(0.0, risk_pct))
        return capital_at_risk / risk_per_share

    def volatility_based(self, equity: float, risk_pct: float, volatility: float) -> float:
        vol = max(1e-9, float(volatility))
        return max(0.0, equity * max(0.0, risk_pct) / vol)


class StopLossManager:
    def __init__(self):
        self.trailing_stops: dict[str, float] = {}

    def calculate_stop_loss(self, price: float, method: str = 'fixed', **kwargs) -> float:
        if method == 'fixed':
            stop_pct = float(kwargs.get('stop_pct', 0.02))
            return price * (1 - stop_pct)
        if method == 'atr':
            atr_value = float(kwargs.get('atr_value', 1.0))
            atr_multiplier = float(kwargs.get('atr_multiplier', 1.5))
            return price - atr_value * atr_multiplier
        # default
        return price * 0.98

    def update_trailing_stop(self, price: float, current_stop: float, trail_pct: float) -> float:
        new_stop = max(current_stop, price * (1 - trail_pct))
        self.trailing_stops['default'] = new_stop
        return new_stop


class RiskManager:
    def __init__(self, config: dict):
        self.config = config or {}

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        # Historical VaR (lower quantile)
        q = np.quantile(returns.dropna(), 1 - confidence) if len(returns.dropna()) else 0.0
        return float(q)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        r = returns.dropna()
        if r.empty:
            return 0.0
        var = self.calculate_var(r, confidence)
        tail = r[r <= var]
        return float(tail.mean()) if not tail.empty else float(var)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        r = returns.dropna() - risk_free
        if r.std() == 0 or len(r) == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))
