"""
连接池指标采集器
- 周期性从 app.pool.pool_manager 采样数据库/HTTP连接池统计
- 将指标写入 Prometheus CollectorRegistry（与业务指标共用）
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Dict, Optional

from prometheus_client import Gauge, Counter, CollectorRegistry

try:
    # 运行时依赖于上层应用的连接池实现
    from app.pool import pool_manager  # type: ignore
except Exception as e:  # pragma: no cover
    pool_manager = None  # 延迟导入失败时，仍允许模块被导入

logger = logging.getLogger(__name__)


class PoolMetricsCollector:
    """连接池指标采集器（Prometheus）"""

    def __init__(self, registry: Optional[CollectorRegistry] = None, interval_seconds: float = 10.0):
        self.registry = registry or CollectorRegistry()
        self.interval_seconds = max(1.0, float(interval_seconds))

        # 指标定义（低基数标签）
        # 数据库连接池 Gauges
        self.db_current_size = Gauge(
            "qilin_db_pool_current_size",
            "Database pool current size",
            ["pool"],
            registry=self.registry,
        )
        self.db_available = Gauge(
            "qilin_db_pool_available",
            "Database pool available connections",
            ["pool"],
            registry=self.registry,
        )
        self.db_in_use = Gauge(
            "qilin_db_pool_in_use",
            "Database pool in-use connections",
            ["pool"],
            registry=self.registry,
        )
        self.db_peak_size = Gauge(
            "qilin_db_pool_peak_size",
            "Database pool peak size",
            ["pool"],
            registry=self.registry,
        )

        # 数据库累计事件 Counters（用delta累加，保持单调递增）
        self.db_created_total = Counter(
            "qilin_db_pool_created_total",
            "Total database connections created",
            ["pool"],
            registry=self.registry,
        )
        self.db_closed_total = Counter(
            "qilin_db_pool_closed_total",
            "Total database connections closed",
            ["pool"],
            registry=self.registry,
        )
        self.db_wait_total = Counter(
            "qilin_db_pool_wait_total",
            "Total waits when acquiring a DB connection",
            ["pool"],
            registry=self.registry,
        )
        self.db_timeout_total = Counter(
            "qilin_db_pool_timeouts_total",
            "Total timeouts when acquiring a DB connection",
            ["pool"],
            registry=self.registry,
        )

        # HTTP 统计（避免高基数，不采集URL/方法等维度）
        self.http_requests_total = Counter(
            "qilin_http_requests_total",
            "Total HTTP requests via HTTPPool",
            ["status"],  # success|failure
            registry=self.registry,
        )
        self.http_bytes_total = Counter(
            "qilin_http_bytes_total",
            "Total HTTP bytes (sent/received)",
            ["direction"],  # sent|received
            registry=self.registry,
        )
        self.http_success_rate = Gauge(
            "qilin_http_success_rate",
            "HTTP success rate (0-1)",
            registry=self.registry,
        )
        self.http_latency_avg_seconds = Gauge(
            "qilin_http_request_seconds_avg",
            "Average HTTP request duration (seconds)",
            registry=self.registry,
        )
        self.http_latency_max_seconds = Gauge(
            "qilin_http_request_seconds_max",
            "Max HTTP request duration (seconds)",
            registry=self.registry,
        )

        # 内部状态
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_db_counters: Dict[str, Dict[str, int]] = {}
        self._last_http_counters: Dict[str, int] = {}

    def start(self):
        """启动后台采集线程"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="PoolMetricsCollector", daemon=True)
        self._thread.start()
        logger.info("PoolMetricsCollector started")

    def stop(self):
        """停止后台采集线程"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            logger.info("PoolMetricsCollector stopped")

    def collect_once(self):
        """执行一次采样（便于测试调用）"""
        if pool_manager is None:
            logger.debug("pool_manager not available; skip collection")
            return

        try:
            # 采集数据库连接池
            for pool_name, stats in (pool_manager.get_all_stats().get("database_pools", {}) or {}).items():
                # Gauges（直接设置当前值）
                self.db_current_size.labels(pool=pool_name).set(float(stats.get("current_size", 0)))
                self.db_available.labels(pool=pool_name).set(float(stats.get("pool_available", 0)))
                self.db_in_use.labels(pool=pool_name).set(float(stats.get("pool_in_use", 0)))
                self.db_peak_size.labels(pool=pool_name).set(float(stats.get("peak_size", 0)))

                # Counters（增量）
                last = self._last_db_counters.get(pool_name, {"created": 0, "closed": 0, "wait": 0, "timeout": 0})
                cur_created = int(stats.get("total_created", 0))
                cur_closed = int(stats.get("total_closed", 0))
                cur_wait = int(stats.get("wait_count", 0))
                cur_timeout = int(stats.get("timeout_count", 0))

                if (delta := cur_created - last.get("created", 0)) > 0:
                    self.db_created_total.labels(pool=pool_name).inc(delta)
                if (delta := cur_closed - last.get("closed", 0)) > 0:
                    self.db_closed_total.labels(pool=pool_name).inc(delta)
                if (delta := cur_wait - last.get("wait", 0)) > 0:
                    self.db_wait_total.labels(pool=pool_name).inc(delta)
                if (delta := cur_timeout - last.get("timeout", 0)) > 0:
                    self.db_timeout_total.labels(pool=pool_name).inc(delta)

                self._last_db_counters[pool_name] = {
                    "created": cur_created,
                    "closed": cur_closed,
                    "wait": cur_wait,
                    "timeout": cur_timeout,
                }

            # 采集HTTP连接池
            http_stats = (pool_manager.get_all_stats() or {}).get("http_pool") if pool_manager else None
            if http_stats:
                last = self._last_http_counters or {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "sent": 0,
                    "recv": 0,
                }
                cur_total = int(http_stats.get("total_requests", 0))
                cur_success = int(http_stats.get("successful_requests", 0))
                cur_failed = int(http_stats.get("failed_requests", 0))
                cur_sent = int(http_stats.get("total_bytes_sent", 0))
                cur_recv = int(http_stats.get("total_bytes_received", 0))

                # Counters（增量）
                if (delta := cur_success - last.get("success", 0)) > 0:
                    self.http_requests_total.labels(status="success").inc(delta)
                if (delta := cur_failed - last.get("failed", 0)) > 0:
                    self.http_requests_total.labels(status="failure").inc(delta)
                if (delta := cur_sent - last.get("sent", 0)) > 0:
                    self.http_bytes_total.labels(direction="sent").inc(delta)
                if (delta := cur_recv - last.get("recv", 0)) > 0:
                    self.http_bytes_total.labels(direction="received").inc(delta)

                # Gauges
                self.http_success_rate.set(float(http_stats.get("success_rate", 0.0)))
                self.http_latency_avg_seconds.set(float(http_stats.get("avg_request_time", 0.0)))
                self.http_latency_max_seconds.set(float(http_stats.get("max_request_time", 0.0)))

                self._last_http_counters = {
                    "total": cur_total,
                    "success": cur_success,
                    "failed": cur_failed,
                    "sent": cur_sent,
                    "recv": cur_recv,
                }
        except Exception as e:  # pragma: no cover
            logger.error(f"Pool metrics collection error: {e}")

    # 内部执行循环
    def _run_loop(self):
        while not self._stop.is_set():
            self.collect_once()
            self._stop.wait(self.interval_seconds)


# 便捷工厂
_def_collector: Optional[PoolMetricsCollector] = None

def start_pool_metrics_collector(registry: Optional[CollectorRegistry] = None, interval_seconds: float = 10.0) -> PoolMetricsCollector:
    global _def_collector
    if _def_collector is None:
        _def_collector = PoolMetricsCollector(registry=registry, interval_seconds=interval_seconds)
        _def_collector.start()
    return _def_collector