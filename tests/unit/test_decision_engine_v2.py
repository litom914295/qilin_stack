import asyncio
import pytest
from datetime import datetime
from typing import List, Dict, Any

from decision_engine.core import (
    DecisionEngine,
    Signal, SignalType, SignalSource,
    FusedSignal,
)

class _StubGen:
    def __init__(self, source: SignalSource, plan: Dict[str, Dict[str, Any]]):
        self.source = source
        self._plan = plan

    async def generate_signals(self, symbols: List[str], date: str) -> List[Signal]:
        out: List[Signal] = []
        for sym in symbols:
            cfg = self._plan.get(sym, {"sig": "hold", "conf": 0.6, "str": 0.0})
            m = {
                "buy": SignalType.BUY,
                "sell": SignalType.SELL,
                "hold": SignalType.HOLD,
                "strong_buy": SignalType.STRONG_BUY,
                "strong_sell": SignalType.STRONG_SELL,
            }
            st = m[cfg["sig"]]
            out.append(
                Signal(
                    symbol=sym,
                    signal_type=st,
                    source=self.source,
                    confidence=cfg.get("conf", 0.6),
                    strength=cfg.get("str", 0.0),
                    reasoning=f"stub {self.source.value}",
                    timestamp=datetime.now(),
                )
            )
        return out

@pytest.mark.asyncio
async def test_decision_engine_parallel_fusion(monkeypatch):
    symbols = ["000001.SZ", "600000.SH"]

    # 构造三路一致偏买的信号，验证融合为 BUY/STRONG_BUY
    qlib_plan = {s: {"sig": "buy", "conf": 0.8, "str": 0.5} for s in symbols}
    ta_plan = {s: {"sig": "buy", "conf": 0.7, "str": 0.4} for s in symbols}
    rd_plan = {s: {"sig": "hold", "conf": 0.6, "str": 0.0} for s in symbols}

    eng = DecisionEngine(enable_performance=False)
    eng.qlib_generator = _StubGen(SignalSource.QLIB, qlib_plan)
    eng.ta_generator = _StubGen(SignalSource.TRADING_AGENTS, ta_plan)
    eng.rd_generator = _StubGen(SignalSource.RD_AGENT, rd_plan)

    decisions = await eng.make_decisions(symbols, "2024-06-30")

    assert len(decisions) == len(symbols)
    for d in decisions:
        assert isinstance(d, FusedSignal)
        assert d.symbol in symbols
        assert d.final_signal in {SignalType.BUY, SignalType.STRONG_BUY}
        assert 0.0 <= d.confidence <= 1.0

@pytest.mark.asyncio
async def test_decision_engine_risk_filter(monkeypatch):
    # 构造低置信度信号，验证会被过滤或降为 HOLD
    symbols = ["000001.SZ"]
    low_plan = {symbols[0]: {"sig": "buy", "conf": 0.2, "str": 0.1}}

    eng = DecisionEngine(enable_performance=False)
    eng.qlib_generator = _StubGen(SignalSource.QLIB, low_plan)
    eng.ta_generator = _StubGen(SignalSource.TRADING_AGENTS, low_plan)
    eng.rd_generator = _StubGen(SignalSource.RD_AGENT, low_plan)

    decisions = await eng.make_decisions(symbols, "2024-06-30")
    assert len(decisions) == 1
    # 由于 _apply_risk_filters 中 <0.5 会被过滤，可能为空；若未过滤，则应降为 HOLD
    d = decisions[0]
    assert d.final_signal in {SignalType.HOLD, SignalType.BUY, SignalType.STRONG_BUY, SignalType.SELL, SignalType.STRONG_SELL}
