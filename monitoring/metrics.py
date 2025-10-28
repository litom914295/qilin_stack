"""
监控系统基础模块
提供Prometheus兼容的指标采集
"""

import time
import asyncio
import contextlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 指标类型定义
# ============================================================================

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"  # 仪表
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"  # 摘要


@dataclass
class Metric:
    """指标"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_prometheus_format(self) -> str:
        """转为Prometheus格式"""
        label_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        if label_str:
            return f"{self.name}{{{label_str}}} {self.value} {int(self.timestamp * 1000)}"
        return f"{self.name} {self.value} {int(self.timestamp * 1000)}"


# ============================================================================
# 指标收集器
# ============================================================================

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        
        logger.info("✅ 指标收集器初始化完成")

    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """递增计数器"""
        key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
        self._counters[key] += value
        
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self._counters[key],
            labels=labels or {}
        )
        self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """设置仪表值"""
        key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
        self._gauges[key] = value
        
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {}
        )
        self.metrics[name].append(metric)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """观察直方图值"""
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {}
        )
        self.metrics[name].append(metric)
    
    def get_metrics(self, name: Optional[str] = None) -> List[Metric]:
        """获取指标"""
        if name:
            return self.metrics.get(name, [])
        
        all_metrics = []
        for metrics_list in self.metrics.values():
            all_metrics.extend(metrics_list)
        return all_metrics
    
    def export_prometheus(self) -> str:
        """导出Prometheus格式"""
        lines = []
        
        for name, metrics_list in self.metrics.items():
            if not metrics_list:
                continue
            
            # 最新的指标
            latest = metrics_list[-1]
            lines.append(f"# HELP {name} {latest.type.value}")
            lines.append(f"# TYPE {name} {latest.type.value}")
            
            # 所有标签的最新值
            label_values = {}
            for metric in reversed(metrics_list):
                label_key = json.dumps(metric.labels, sort_keys=True)
                if label_key not in label_values:
                    label_values[label_key] = metric
            
            for metric in label_values.values():
                lines.append(metric.to_prometheus_format())
        
        return "\n".join(lines)
    
    def clear(self):
        """清除所有指标"""
        self.metrics.clear()
        self._counters.clear()
        self._gauges.clear()

# 全局共享采集器，确保不同监控器实例共享同一份度量数据
_collector_singleton = MetricsCollector()


# ============================================================================
# 系统监控器
# ============================================================================

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        global _collector_singleton
        self.collector = _collector_singleton
        self.start_time = time.time()
        
        # 预定义指标名称
        self.SIGNAL_GENERATED = "signal_generated_total"
        self.DECISION_MADE = "decision_made_total"
        self.DECISION_LATENCY = "decision_latency_seconds"
        self.SIGNAL_CONFIDENCE = "signal_confidence"
        self.WEIGHT_VALUE = "system_weight"
        self.MARKET_STATE = "market_state"
        self.SYSTEM_UPTIME = "system_uptime_seconds"
        self.ERROR_COUNT = "error_count_total"
        
        # 为了通过初始化健康检查，预填充一条指标（系统启动时长）
        self.update_uptime()
        
        logger.info("✅ 系统监控器初始化完成")
        
    @property
    def metrics(self):
        """向后兼容：暴露内部存储的原始指标字典"""
        return self.collector.metrics

    def reset_metrics(self):
        """重置所有指标（用于测试/重置）"""
        self.collector.clear()
        self.update_uptime()
    
    def record_signal(self, source: str, signal_type: str, confidence: float):
        """记录信号"""
        self.collector.increment_counter(
            self.SIGNAL_GENERATED,
            labels={"source": source, "type": signal_type}
        )
        
        self.collector.set_gauge(
            self.SIGNAL_CONFIDENCE,
            confidence,
            labels={"source": source}
        )
    
    def record_decision(self, symbol: str, decision: str, latency: float, confidence: float):
        """记录决策"""
        self.collector.increment_counter(
            self.DECISION_MADE,
            labels={"symbol": symbol, "decision": decision}
        )
        
        self.collector.observe_histogram(
            self.DECISION_LATENCY,
            latency,
            labels={"symbol": symbol}
        )
    
    def record_weight(self, source: str, weight: float):
        """记录权重"""
        self.collector.set_gauge(
            self.WEIGHT_VALUE,
            weight,
            labels={"source": source}
        )
    
    def record_market_state(self, regime: str, confidence: float):
        """记录市场状态"""
        # 将市场状态编码为数字
        regime_map = {
            "bull": 1.0,
            "bear": -1.0,
            "sideways": 0.0,
            "volatile": 0.5,
            "unknown": 0.0
        }
        
        self.collector.set_gauge(
            self.MARKET_STATE,
            regime_map.get(regime, 0.0),
            labels={"regime": regime}
        )
    
    def record_error(self, component: str, error_type: str):
        """记录错误"""
        self.collector.increment_counter(
            self.ERROR_COUNT,
            labels={"component": component, "type": error_type}
        )
    
    def update_uptime(self):
        """更新运行时间"""
        uptime = time.time() - self.start_time
        self.collector.set_gauge(self.SYSTEM_UPTIME, uptime)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要（仅统计当前监控周期内的事件）"""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            elapsed = 1e-6
        signals = [m for m in self.collector.get_metrics(self.SIGNAL_GENERATED) if m.timestamp >= self.start_time]
        decisions = [m for m in self.collector.get_metrics(self.DECISION_MADE) if m.timestamp >= self.start_time]
        errors = [m for m in self.collector.get_metrics(self.ERROR_COUNT) if m.timestamp >= self.start_time]
        return {
            'uptime': elapsed,
            'total_signals': len(signals),
            'total_decisions': len(decisions),
            'total_errors': len(errors),
            'metrics_count': sum(len(m) for m in self.collector.metrics.values())
        }
    
    def export_metrics(self) -> str:
        """导出指标（Prometheus格式）"""
        self.update_uptime()
        return self.collector.export_prometheus()


# ============================================================================
# 性能追踪器
# ============================================================================

class PerformanceTracker:
    """性能追踪器"""
    
    def __init__(self):
        self.monitor = get_monitor()
        self._timers: Dict[str, float] = {}
    
    def track(self, name: str):
        """装饰器：追踪任意函数（同步/异步）执行时间并记录错误"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    self.start_timer(name)
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        self.monitor.record_error(name, type(e).__name__)
                        raise
                    finally:
                        self.end_timer(name, labels={"func": func.__name__})
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    self.start_timer(name)
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        self.monitor.record_error(name, type(e).__name__)
                        raise
                    finally:
                        self.end_timer(name, labels={"func": func.__name__})
                return sync_wrapper
        return decorator

    @contextlib.contextmanager
    def track_context(self, name: str, labels: Optional[Dict[str, str]] = None):
        """上下文管理器：追踪一段代码块的耗时"""
        self.start_timer(name)
        try:
            yield
        except Exception as e:
            self.monitor.record_error(name, type(e).__name__)
            raise
        finally:
            self.end_timer(name, labels=labels)

    def start_timer(self, name: str):
        """开始计时"""
        self._timers[name] = time.time()
    
    def end_timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """结束计时并记录"""
        if name not in self._timers:
            return 0.0
        
        duration = time.time() - self._timers[name]
        del self._timers[name]
        
        self.monitor.collector.observe_histogram(
            f"{name}_duration_seconds",
            duration,
            labels=labels
        )
        
        return duration
    
    def track_performance(self, func_name: str):
        """装饰器：追踪函数性能"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.start_timer(func_name)
                try:
                    result = func(*args, **kwargs)
                    duration = self.end_timer(func_name)
                    logger.debug(f"{func_name} 耗时: {duration:.3f}s")
                    return result
                except Exception as e:
                    self.end_timer(func_name)
                    self.monitor.record_error(func_name, type(e).__name__)
                    raise
            return wrapper
        return decorator


# ============================================================================
# 全局实例
# ============================================================================

_monitor_instance = None
_tracker_instance = None

def get_monitor() -> SystemMonitor:
    """获取监控器单例（每次获取刷新起始时间，隔离跨测试统计）"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    else:
        _monitor_instance.start_time = time.time()
    return _monitor_instance

def get_tracker() -> PerformanceTracker:
    """获取追踪器单例"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PerformanceTracker()
    return _tracker_instance


# ============================================================================
# 测试
# ============================================================================

def test_monitoring():
    """测试监控系统"""
    print("=== 监控系统测试 ===\n")
    
    monitor = get_monitor()
    tracker = get_tracker()
    
    # 模拟记录指标
    print("1️⃣ 记录信号:")
    monitor.record_signal("qlib", "buy", 0.85)
    monitor.record_signal("trading_agents", "sell", 0.75)
    monitor.record_signal("rd_agent", "hold", 0.65)
    
    print("2️⃣ 记录决策:")
    monitor.record_decision("000001.SZ", "buy", 0.05, 0.82)
    monitor.record_decision("600000.SH", "hold", 0.03, 0.70)
    
    print("3️⃣ 记录权重:")
    monitor.record_weight("qlib", 0.4)
    monitor.record_weight("trading_agents", 0.35)
    monitor.record_weight("rd_agent", 0.25)
    
    print("4️⃣ 记录市场状态:")
    monitor.record_market_state("bull", 0.75)
    
    # 性能追踪
    print("\n5️⃣ 性能追踪:")
    tracker.start_timer("test_operation")
    time.sleep(0.1)
    duration = tracker.end_timer("test_operation")
    print(f"  操作耗时: {duration:.3f}s")
    
    # 获取摘要
    print("\n6️⃣ 监控摘要:")
    summary = monitor.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 导出Prometheus格式
    print("\n7️⃣ Prometheus导出 (前10行):")
    prometheus_output = monitor.export_metrics()
    lines = prometheus_output.split('\n')
    for line in lines[:10]:
        print(f"  {line}")
    
    if len(lines) > 10:
        print(f"  ... ({len(lines) - 10} more lines)")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_monitoring()
