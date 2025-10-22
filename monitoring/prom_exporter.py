"""
Prometheus exporter：将项目内部自定义指标同步到Prometheus默认注册表并暴露/metrics
- 读取 monitoring.metrics.SystemMonitor 的 collector 数据
- 映射到 Prometheus 指标（Gauge/Counter）
"""
from __future__ import annotations

import time
from typing import Dict
from collections import defaultdict

from prometheus_client import Gauge, Counter, start_http_server
from monitoring.metrics import get_monitor

# Prometheus 指标（默认注册表）
SYSTEM_WEIGHT = Gauge('system_weight', 'Fused weights by source', ['source'])
WEIGHTS_UPDATED = Counter('weights_updated_total', 'Weights updated count', ['mode'])

# 可选：门槛拒绝（若已在项目内计入自定义采集器）
GATE_REJECT = Counter('gate_reject_total', 'Gate rejection count by reason', ['reason'])


def _sync_system_weight():
    monitor = get_monitor()
    metrics = monitor.collector.get_metrics('system_weight')
    latest_by_source: Dict[str, float] = {}
    for m in reversed(metrics):  # 逆序找最新值
        source = m.labels.get('source', 'unknown')
        if source not in latest_by_source:
            latest_by_source[source] = float(m.value)
    for src, val in latest_by_source.items():
        SYSTEM_WEIGHT.labels(source=src).set(val)


_last_weights_updated = defaultdict(float)

def _sync_weights_updated():
    monitor = get_monitor()
    metrics = monitor.collector.get_metrics('weights_updated_total')
    # 聚合不同label组合，按mode维度增量推进Counter
    latest_by_mode: Dict[str, float] = {}
    for m in metrics:
        mode = m.labels.get('mode', 'unknown')
        latest_by_mode[mode] = float(m.value)
    for mode, current in latest_by_mode.items():
        prev = _last_weights_updated[mode]
        delta = max(0.0, current - prev)
        if delta > 0:
            WEIGHTS_UPDATED.labels(mode=mode).inc(delta)
            _last_weights_updated[mode] = current


def _sync_gate_reject():
    monitor = get_monitor()
    metrics = monitor.collector.get_metrics('gate_reject_total')
    latest_by_reason: Dict[str, float] = {}
    for m in metrics:
        reason = m.labels.get('reason', 'unknown')
        latest_by_reason[reason] = float(m.value)
    # 使用增量推进Counter
    if not hasattr(_sync_gate_reject, 'last'):  # type: ignore
        _sync_gate_reject.last = defaultdict(float)  # type: ignore
    last = _sync_gate_reject.last  # type: ignore
    for reason, current in latest_by_reason.items():
        delta = max(0.0, current - last[reason])
        if delta > 0:
            GATE_REJECT.labels(reason=reason).inc(delta)
            last[reason] = current


def main(port: int = 9100, interval_sec: int = 5):
    start_http_server(port)
    while True:
        try:
            _sync_system_weight()
            _sync_weights_updated()
            _sync_gate_reject()
        except Exception:
            pass
        time.sleep(interval_sec)


if __name__ == '__main__':
    main()
