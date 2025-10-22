import asyncio
import pandas as pd
import numpy as np
import pytest

from backtest.engine import BacktestEngine, BacktestConfig

class _StubDecisionEngine:
    async def make_decisions(self, symbols, date):
        # 始终给出 BUY 信号的最小桩
        from decision_engine.core import FusedSignal, SignalType
        import datetime
        return [
            FusedSignal(
                symbol=s,
                final_signal=SignalType.BUY,
                confidence=0.8,
                strength=0.3,
                component_signals=[],
                weights={},
                reasoning="stub",
                timestamp=datetime.datetime.now(),
            )
            for s in symbols
        ]

@pytest.mark.asyncio
async def test_backtest_engine_smoke():
    # 构造极简日线数据
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    symbol = "000001.SZ"
    data = pd.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "date": dates,
            "close": 10 + np.random.randn(len(dates)) * 0.1,
        }
    )

    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    # 注入桩引擎，避免真实依赖
    engine.decision_engine = _StubDecisionEngine()

    metrics = await engine.run_backtest([symbol], str(dates.min().date()), str(dates.max().date()), data)

    # 关键字段存在且类型正确
    assert "final_equity" in metrics and isinstance(metrics["final_equity"], (int, float))
    assert "total_trades" in metrics and isinstance(metrics["total_trades"], int)
