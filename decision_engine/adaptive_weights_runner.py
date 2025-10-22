"""
自适应权重服务运行器：连接回测/实盘收益回放存储并周期性更新权重
用法：
  python -m decision_engine.adaptive_weights_runner --interval 3600 --lookback 3
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from .adaptive_weights import AdaptiveWeightsService
from .core import get_decision_engine, Signal
from persistence.returns_store import get_returns_store


async def _fetch_signals(date: Optional[str] = None) -> List[Signal]:
    engine = get_decision_engine()
    # 使用最近1000条源信号做评估
    return engine.signal_history[-1000:]


async def _fetch_actual_returns(date: Optional[str] = None, lookback_days: int = 3) -> Dict[str, float]:
    store = get_returns_store()
    return store.get_recent(days=lookback_days)


def _apply_weights(new_weights):
    engine = get_decision_engine()
    engine.fuser.update_weights(new_weights)


async def main(interval: int, lookback: int):
    service = AdaptiveWeightsService(
        fetch_signals=_fetch_signals,
        fetch_actual_returns=lambda d=None: _fetch_actual_returns(d, lookback),
        update_weights=_apply_weights,
    )
    await service.run_forever(interval_seconds=interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=3600, help="更新间隔（秒）")
    parser.add_argument("--lookback", type=int, default=3, help="收益回放回看天数")
    args = parser.parse_args()

    asyncio.run(main(args.interval, args.lookback))
