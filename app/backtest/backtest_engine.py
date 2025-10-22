"""
Deprecated backtest engine module (app/backtest/backtest_engine.py)

This module has been deprecated to avoid duplication and syntax issues.
Please use one of the maintained engines instead:
- backtest/engine.py               (simple async engine used in README)
- app/core/backtest_engine.py      (feature-rich engine used by test_suite)
"""
from __future__ import annotations
import warnings
warnings.warn(
    "app/backtest/backtest_engine.py is deprecated. Use backtest/engine.py or app/core/backtest_engine.py.",
    DeprecationWarning,
)
class BacktestEngine:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Deprecated: import BacktestEngine from backtest.engine or app.core.backtest_engine"
        )
