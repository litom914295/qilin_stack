"""
P2-10: 实时监控与告警系统 (Realtime Monitoring & Alerting)
实现系统健康检查、性能监控、告警系统等功能
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警信息"""
    level: AlertLevel
    message: str
    timestamp: float
    category: str
    value: Optional[float] = None


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_sent_mbps: float
    network_recv_mbps: float
    process_count: int
    timestamp: float


class HealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.last_network_io = psutil.net_io_counters()
        self.last_check_time = time.time()
        
        logger.info("健康检查器初始化完成")
    
    def check_cpu(self, threshold: float = 80.0) -> Optional[Alert]:
        """
        检查CPU使用率
        
        Args:
            threshold: 阈值百分比
        
        Returns:
            告警信息(如果超过阈值)
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if cpu_percent > threshold:
            return Alert(
                level=AlertLevel.WARNING if cpu_percent < 90 else AlertLevel.ERROR,
                message=f"CPU使用率过高: {cpu_percent:.1f}%",
                timestamp=time.time(),
                category="cpu",
                value=cpu_percent
            )
        return None
    
    def check_memory(self, threshold: float = 85.0) -> Optional[Alert]:
        """
        检查内存使用率
        
        Args:
            threshold: 阈值百分比
        
        Returns:
            告警信息(如果超过阈值)
        """
        memory = psutil.virtual_memory()
        
        if memory.percent > threshold:
            return Alert(
                level=AlertLevel.WARNING if memory.percent < 95 else AlertLevel.CRITICAL,
                message=f"内存使用率过高: {memory.percent:.1f}%",
                timestamp=time.time(),
                category="memory",
                value=memory.percent
            )
        return None
    
    def check_disk(self, threshold: float = 90.0) -> Optional[Alert]:
        """
        检查磁盘使用率
        
        Args:
            threshold: 阈值百分比
        
        Returns:
            告警信息(如果超过阈值)
        """
        disk = psutil.disk_usage('/')
        
        if disk.percent > threshold:
            return Alert(
                level=AlertLevel.WARNING if disk.percent < 95 else AlertLevel.ERROR,
                message=f"磁盘空间不足: {disk.percent:.1f}%",
                timestamp=time.time(),
                category="disk",
                value=disk.percent
            )
        return None
    
    def check_network(self, max_mbps: float = 100.0) -> Optional[Alert]:
        """
        检查网络流量
        
        Args:
            max_mbps: 最大流量(Mbps)
        
        Returns:
            告警信息(如果超过阈值)
        """
        current_io = psutil.net_io_counters()
        current_time = time.time()
        
        # 计算速率
        time_delta = current_time - self.last_check_time
        sent_bytes = current_io.bytes_sent - self.last_network_io.bytes_sent
        recv_bytes = current_io.bytes_recv - self.last_network_io.bytes_recv
        
        sent_mbps = (sent_bytes * 8) / (time_delta * 1_000_000)
        recv_mbps = (recv_bytes * 8) / (time_delta * 1_000_000)
        
        # 更新记录
        self.last_network_io = current_io
        self.last_check_time = current_time
        
        total_mbps = sent_mbps + recv_mbps
        
        if total_mbps > max_mbps:
            return Alert(
                level=AlertLevel.WARNING,
                message=f"网络流量过高: {total_mbps:.2f} Mbps",
                timestamp=time.time(),
                category="network",
                value=total_mbps
            )
        return None
    
    def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        current_io = psutil.net_io_counters()
        current_time = time.time()
        
        time_delta = current_time - self.last_check_time
        sent_mbps = 0.0
        recv_mbps = 0.0
        
        if time_delta > 0:
            sent_bytes = current_io.bytes_sent - self.last_network_io.bytes_sent
            recv_bytes = current_io.bytes_recv - self.last_network_io.bytes_recv
            sent_mbps = (sent_bytes * 8) / (time_delta * 1_000_000)
            recv_mbps = (recv_bytes * 8) / (time_delta * 1_000_000)
        
        self.last_network_io = current_io
        self.last_check_time = current_time
        
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            network_sent_mbps=sent_mbps,
            network_recv_mbps=recv_mbps,
            process_count=len(psutil.pids()),
            timestamp=current_time
        )


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, history_size: int = 1000):
        """
        初始化性能监控器
        
        Args:
            history_size: 历史数据缓冲区大小
        """
        self.history_size = history_size
        self.latency_history = deque(maxlen=history_size)
        self.throughput_history = deque(maxlen=history_size)
        self.error_count = 0
        self.success_count = 0
        
        logger.info(f"性能监控器初始化 (history_size={history_size})")
    
    def record_latency(self, latency_ms: float):
        """记录延迟"""
        self.latency_history.append((time.time(), latency_ms))
    
    def record_throughput(self, ops_per_sec: float):
        """记录吞吐量"""
        self.throughput_history.append((time.time(), ops_per_sec))
    
    def record_success(self):
        """记录成功操作"""
        self.success_count += 1
    
    def record_error(self):
        """记录错误操作"""
        self.error_count += 1
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        if not self.latency_history:
            return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0}
        
        latencies = [lat for _, lat in self.latency_history]
        
        import numpy as np
        return {
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': np.max(latencies)
        }
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """获取吞吐量统计"""
        if not self.throughput_history:
            return {'current': 0, 'avg': 0, 'peak': 0}
        
        throughputs = [thr for _, thr in self.throughput_history]
        
        import numpy as np
        return {
            'current': throughputs[-1] if throughputs else 0,
            'avg': np.mean(throughputs),
            'peak': np.max(throughputs)
        }
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        total = self.success_count + self.error_count
        return self.error_count / total if total > 0 else 0.0


class AlertManager:
    """告警管理器"""
    
    def __init__(self, max_alerts: int = 100):
        """
        初始化告警管理器
        
        Args:
            max_alerts: 最大告警数量
        """
        self.max_alerts = max_alerts
        self.alerts = deque(maxlen=max_alerts)
        self.handlers: List[Callable[[Alert], None]] = []
        
        logger.info(f"告警管理器初始化 (max_alerts={max_alerts})")
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self.handlers.append(handler)
    
    def emit_alert(self, alert: Alert):
        """发送告警"""
        self.alerts.append(alert)
        
        # 调用所有处理器
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器错误: {e}")
    
    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """获取最近的告警"""
        return list(self.alerts)[-count:]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """按级别获取告警"""
        return [a for a in self.alerts if a.level == level]
    
    def clear_alerts(self):
        """清除所有告警"""
        self.alerts.clear()


class RealtimeMonitor:
    """实时监控系统"""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        
        self.is_running = False
        self.monitor_thread = None
        
        logger.info("实时监控系统初始化完成")
    
    def start(self, interval: float = 5.0):
        """
        启动监控
        
        Args:
            interval: 监控间隔(秒)
        """
        if self.is_running:
            logger.warning("监控系统已在运行")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"监控系统已启动 (interval={interval}s)")
    
    def stop(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("监控系统已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.is_running:
            try:
                # 执行健康检查
                alerts = [
                    self.health_checker.check_cpu(),
                    self.health_checker.check_memory(),
                    self.health_checker.check_disk(),
                    self.health_checker.check_network()
                ]
                
                # 发送告警
                for alert in alerts:
                    if alert:
                        self.alert_manager.emit_alert(alert)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(interval)
    
    def get_dashboard_data(self) -> Dict:
        """获取Dashboard数据"""
        return {
            'system_metrics': self.health_checker.get_system_metrics(),
            'performance_stats': {
                'latency': self.performance_monitor.get_latency_stats(),
                'throughput': self.performance_monitor.get_throughput_stats(),
                'error_rate': self.performance_monitor.get_error_rate()
            },
            'recent_alerts': self.alert_manager.get_recent_alerts(10)
        }


def main():
    """示例: 实时监控系统"""
    print("=" * 80)
    print("P2-10: 实时监控与告警系统 - 示例")
    print("=" * 80)
    
    # 1. 系统健康检查
    print("\n🏥 系统健康检查...")
    
    health_checker = HealthChecker()
    
    cpu_alert = health_checker.check_cpu(threshold=50.0)
    if cpu_alert:
        print(f"  ⚠️  {cpu_alert.message}")
    else:
        print(f"  ✅ CPU使用正常")
    
    memory_alert = health_checker.check_memory(threshold=50.0)
    if memory_alert:
        print(f"  ⚠️  {memory_alert.message}")
    else:
        print(f"  ✅ 内存使用正常")
    
    print("✅ 健康检查完成")
    
    # 2. 系统指标
    print("\n📊 系统指标...")
    
    metrics = health_checker.get_system_metrics()
    print(f"  CPU: {metrics.cpu_percent:.1f}%")
    print(f"  内存: {metrics.memory_percent:.1f}%")
    print(f"  磁盘: {metrics.disk_usage_percent:.1f}%")
    print(f"  网络发送: {metrics.network_sent_mbps:.2f} Mbps")
    print(f"  网络接收: {metrics.network_recv_mbps:.2f} Mbps")
    print(f"  进程数: {metrics.process_count}")
    
    print("✅ 系统指标获取完成")
    
    # 3. 性能监控
    print("\n⚡ 性能监控...")
    
    perf_monitor = PerformanceMonitor()
    
    # 模拟记录一些数据
    import numpy as np
    for _ in range(100):
        perf_monitor.record_latency(np.random.uniform(1, 10))
        perf_monitor.record_throughput(np.random.uniform(100, 1000))
        if np.random.rand() > 0.95:
            perf_monitor.record_error()
        else:
            perf_monitor.record_success()
    
    lat_stats = perf_monitor.get_latency_stats()
    thr_stats = perf_monitor.get_throughput_stats()
    
    print(f"  延迟统计:")
    print(f"    平均: {lat_stats['mean']:.2f} ms")
    print(f"    P95: {lat_stats['p95']:.2f} ms")
    print(f"    P99: {lat_stats['p99']:.2f} ms")
    
    print(f"  吞吐量统计:")
    print(f"    当前: {thr_stats['current']:.2f} ops/s")
    print(f"    平均: {thr_stats['avg']:.2f} ops/s")
    print(f"    峰值: {thr_stats['peak']:.2f} ops/s")
    
    print(f"  错误率: {perf_monitor.get_error_rate():.2%}")
    
    print("✅ 性能监控完成")
    
    # 4. 告警管理
    print("\n🔔 告警管理...")
    
    alert_manager = AlertManager()
    
    # 添加告警处理器
    def print_alert(alert: Alert):
        print(f"  [{alert.level.value.upper()}] {alert.message}")
    
    alert_manager.add_handler(print_alert)
    
    # 发送一些告警
    alert_manager.emit_alert(Alert(
        level=AlertLevel.WARNING,
        message="CPU使用率较高",
        timestamp=time.time(),
        category="cpu",
        value=75.5
    ))
    
    alert_manager.emit_alert(Alert(
        level=AlertLevel.ERROR,
        message="内存不足",
        timestamp=time.time(),
        category="memory",
        value=92.3
    ))
    
    recent = alert_manager.get_recent_alerts(5)
    print(f"\n  最近告警数: {len(recent)}")
    
    print("✅ 告警管理完成")
    
    # 5. 完整监控系统
    print("\n🚀 完整监控系统测试...")
    
    monitor = RealtimeMonitor()
    
    # 启动监控(短时间测试)
    monitor.start(interval=1.0)
    print("  监控系统已启动...")
    
    time.sleep(3)
    
    # 获取Dashboard数据
    dashboard_data = monitor.get_dashboard_data()
    print(f"  系统CPU: {dashboard_data['system_metrics'].cpu_percent:.1f}%")
    print(f"  告警数: {len(dashboard_data['recent_alerts'])}")
    
    monitor.stop()
    print("  监控系统已停止")
    
    print("✅ 完整监控测试完成")
    
    print("\n" + "=" * 80)
    print("✅ 所有实时监控功能演示完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
