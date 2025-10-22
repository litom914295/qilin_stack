"""
自适应权重自动执行服务
对接实盘/回测结果，周期性评估三套系统表现并更新融合权重；将权重轨迹写入监控看板。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional

from monitoring.metrics import get_monitor
from .weight_optimizer import AdaptiveWeightStrategy, WeightOptimizer
from .core import Signal, SignalSource

logger = logging.getLogger(__name__)


FetchSignalsFn = Callable[[Optional[str]], Awaitable[List[Signal]]]
FetchReturnsFn = Callable[[Optional[str]], Awaitable[Dict[str, float]]]
UpdateWeightsFn = Callable[[Dict[SignalSource, float]], None]


@dataclass
class AdaptiveWeightsService:
    """滚动权重自适应服务。

    依赖注入三类能力：
    - fetch_signals(date) -> 最近的源信号（用于评估性能）
    - fetch_actual_returns(date) -> {symbol: return}
    - update_weights(new_weights) -> 将新权重写回决策引擎/配置
    """

    fetch_signals: FetchSignalsFn
    fetch_actual_returns: FetchReturnsFn
    update_weights: UpdateWeightsFn
    strategy: AdaptiveWeightStrategy = AdaptiveWeightStrategy()

    async def run_once(self, date: Optional[str] = None) -> Dict[str, float]:
        """执行一次权重评估与更新。"""
        # 1) 拉取数据
        signals = await self.fetch_signals(date)
        actual_returns = await self.fetch_actual_returns(date)
        if not signals or not actual_returns:
            logger.warning("自适应权重：缺少信号或收益数据，跳过本次更新")
            return {}

        # 2) 评估表现
        optimizer: WeightOptimizer = self.strategy.optimizer
        perf = optimizer.evaluate_performance(signals, actual_returns)

        # 3) 计算新权重
        new_weights = optimizer.optimize_weights(perf)

        # 4) 应用并记录到看板
        self.update_weights(new_weights)
        for src, w in new_weights.items():
            get_monitor().record_weight(src.value, w)
        get_monitor().collector.increment_counter("weights_updated_total", labels={"mode": "adaptive"})

        self.strategy.last_update = datetime.now()
        logger.info(f"自适应权重完成：{ {k.value: v for k, v in new_weights.items()} }")
        return {k.value: v for k, v in new_weights.items()}

    async def run_forever(self, interval_seconds: int = 3600, date_fn: Optional[Callable[[], str]] = None):
        """常驻服务：到点自动评估并更新。"""
        while True:
            try:
                if self.strategy.optimizer.should_update(self.strategy.last_update):
                    date = date_fn() if date_fn else None
                    await self.run_once(date)
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.exception(f"自适应权重服务异常: {e}")
                await asyncio.sleep(interval_seconds)
