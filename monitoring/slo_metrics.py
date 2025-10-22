"""
SLO 指标模块
- 提供端到端延迟直方图与恢复时间 Gauge
- 支持注入自定义 Prometheus CollectorRegistry，便于与业务/连接池指标共享
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional, Dict

from prometheus_client import CollectorRegistry, Histogram, Gauge


class SLOMetrics:
    """SLO 指标集合"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        # 端到端延迟（毫秒）直方图
        # buckets 以毫秒为单位，覆盖常见范围
        self.e2e_latency_ms = Histogram(
            'qilin_e2e_latency_ms',
            'End-to-end latency (milliseconds) for core analysis pipeline',
            buckets=[10, 25, 50, 100, 200, 400, 600, 800, 1000, 2000, 5000],
            registry=self.registry,
        )
        # 故障恢复时间（秒）
        self.failover_recovery_seconds = Gauge(
            'qilin_failover_recovery_seconds',
            'Measured failover recovery time in seconds',
            registry=self.registry,
        )

    def observe_latency_ms(self, value_ms: float) -> None:
        self.e2e_latency_ms.observe(float(value_ms))

    def set_recovery_time(self, seconds: float) -> None:
        self.failover_recovery_seconds.set(float(seconds))

    @contextmanager
    def timer_ms(self):
        """上下文计时器：退出时记录毫秒延迟"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.observe_latency_ms(elapsed_ms)


# 基于 registry 的缓存，避免重复创建指标实例
_registry_cache: Dict[int, SLOMetrics] = {}

def get_slo_metrics(registry: Optional[CollectorRegistry] = None) -> SLOMetrics:
    if registry is None:
        # 单独使用：每次返回独立实例
        return SLOMetrics()
    key = id(registry)
    if key not in _registry_cache:
        _registry_cache[key] = SLOMetrics(registry=registry)
    return _registry_cache[key]