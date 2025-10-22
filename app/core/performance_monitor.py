"""
性能监控系统模块
实现系统性能监控、Agent表现跟踪、交易执行监控等功能
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import warnings
import os
import sys

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"  # 仪表盘（当前值）
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"  # 摘要统计


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_usage: float  # CPU使用率
    memory_usage: float  # 内存使用率
    memory_available: float  # 可用内存(GB)
    disk_usage: float  # 磁盘使用率
    network_io: Dict[str, float]  # 网络IO
    process_count: int  # 进程数
    thread_count: int  # 线程数
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_available': self.memory_available,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'process_count': self.process_count,
            'thread_count': self.thread_count
        }


@dataclass
class TradingMetrics:
    """交易指标"""
    timestamp: datetime
    total_trades: int  # 总交易数
    successful_trades: int  # 成功交易数
    failed_trades: int  # 失败交易数
    avg_execution_time: float  # 平均执行时间
    total_volume: float  # 总成交量
    total_value: float  # 总成交额
    slippage: float  # 滑点
    commission: float  # 手续费
    pnl: float  # 盈亏
    win_rate: float  # 胜率
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'avg_execution_time': self.avg_execution_time,
            'total_volume': self.total_volume,
            'total_value': self.total_value,
            'slippage': self.slippage,
            'commission': self.commission,
            'pnl': self.pnl,
            'win_rate': self.win_rate
        }


@dataclass
class AgentMetrics:
    """Agent性能指标"""
    agent_id: str
    timestamp: datetime
    response_time: float  # 响应时间
    accuracy: float  # 准确率
    confidence: float  # 置信度
    signal_count: int  # 信号数
    error_count: int  # 错误数
    memory_usage: float  # 内存使用
    cpu_time: float  # CPU时间
    
    def to_dict(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'response_time': self.response_time,
            'accuracy': self.accuracy,
            'confidence': self.confidence,
            'signal_count': self.signal_count,
            'error_count': self.error_count,
            'memory_usage': self.memory_usage,
            'cpu_time': self.cpu_time
        }


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, float] = defaultdict(float)
        self.lock = threading.Lock()
    
    def record(self, metric_name: str, value: float, metric_type: MetricType = MetricType.GAUGE):
        """记录指标"""
        with self.lock:
            timestamp = datetime.now()
            
            if metric_type == MetricType.COUNTER:
                self.counters[metric_name] += value
                self.metrics[metric_name].append((timestamp, self.counters[metric_name]))
            else:
                self.metrics[metric_name].append((timestamp, value))
    
    def get_metric(self, metric_name: str) -> List[Tuple[datetime, float]]:
        """获取指标历史"""
        with self.lock:
            return list(self.metrics.get(metric_name, []))
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """获取最新值"""
        with self.lock:
            if metric_name in self.metrics and self.metrics[metric_name]:
                return self.metrics[metric_name][-1][1]
        return None
    
    def get_statistics(self, metric_name: str) -> Dict:
        """获取统计信息"""
        with self.lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return {}
            
            values = [v for _, v in self.metrics[metric_name]]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'count': len(values)
            }


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, interval: float = 1.0):
        """
        初始化系统监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.collector = MetricsCollector()
        self.running = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = self._collect_system_metrics()
                
                # 记录各项指标
                self.collector.record('cpu_usage', metrics.cpu_usage)
                self.collector.record('memory_usage', metrics.memory_usage)
                self.collector.record('disk_usage', metrics.disk_usage)
                self.collector.record('thread_count', metrics.thread_count)
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"系统监控错误: {str(e)}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # 转换为GB
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # 网络IO
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # 进程信息
        process_count = len(psutil.pids())
        
        # 当前进程线程数
        thread_count = self.process.num_threads()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            thread_count=thread_count
    
    def get_current_metrics(self) -> SystemMetrics:
        """获取当前系统指标"""
        return self._collect_system_metrics()


class TradingMonitor:
    """交易监控器"""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.trade_history = deque(maxlen=10000)
        self.execution_times = deque(maxlen=1000)
        
    def record_trade(self, trade: Dict):
        """记录交易"""
        self.trade_history.append({
            **trade,
            'timestamp': datetime.now()
        })
        
        # 更新计数器
        self.collector.record('total_trades', 1, MetricType.COUNTER)
        
        if trade.get('status') == 'success':
            self.collector.record('successful_trades', 1, MetricType.COUNTER)
        else:
            self.collector.record('failed_trades', 1, MetricType.COUNTER)
        
        # 记录执行时间
        if 'execution_time' in trade:
            self.execution_times.append(trade['execution_time'])
            self.collector.record('execution_time', trade['execution_time'])
        
        # 记录其他指标
        if 'volume' in trade:
            self.collector.record('trade_volume', trade['volume'], MetricType.COUNTER)
        
        if 'value' in trade:
            self.collector.record('trade_value', trade['value'], MetricType.COUNTER)
        
        if 'slippage' in trade:
            self.collector.record('slippage', trade['slippage'])
        
        if 'pnl' in trade:
            self.collector.record('pnl', trade['pnl'], MetricType.COUNTER)
    
    def get_trading_metrics(self) -> TradingMetrics:
        """获取交易指标"""
        total_trades = int(self.collector.get_latest('total_trades') or 0)
        successful_trades = int(self.collector.get_latest('successful_trades') or 0)
        failed_trades = int(self.collector.get_latest('failed_trades') or 0)
        
        # 计算平均执行时间
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
        
        # 获取累计值
        total_volume = self.collector.get_latest('trade_volume') or 0
        total_value = self.collector.get_latest('trade_value') or 0
        total_pnl = self.collector.get_latest('pnl') or 0
        
        # 计算滑点统计
        slippage_stats = self.collector.get_statistics('slippage')
        avg_slippage = slippage_stats.get('mean', 0) if slippage_stats else 0
        
        # 计算胜率
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        return TradingMetrics(
            timestamp=datetime.now(),
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            avg_execution_time=avg_execution_time,
            total_volume=total_volume,
            total_value=total_value,
            slippage=avg_slippage,
            commission=0,  # 需要从其他地方获取
            pnl=total_pnl,
            win_rate=win_rate
    
    def get_trade_statistics(self) -> Dict:
        """获取交易统计"""
        if not self.trade_history:
            return {}
        
        df = pd.DataFrame(list(self.trade_history))
        
        stats = {
            'total_trades': len(df),
            'by_status': df['status'].value_counts().to_dict() if 'status' in df else {},
            'by_symbol': df['symbol'].value_counts().to_dict() if 'symbol' in df else {},
            'hourly_distribution': self._get_hourly_distribution(df),
            'execution_time': self.collector.get_statistics('execution_time'),
            'slippage': self.collector.get_statistics('slippage'),
            'volume': self.collector.get_statistics('trade_volume')
        }
        
        return stats
    
    def _get_hourly_distribution(self, df: pd.DataFrame) -> Dict:
        """获取小时分布"""
        if 'timestamp' not in df:
            return {}
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        return df['hour'].value_counts().sort_index().to_dict()


class AgentMonitor:
    """Agent监控器"""
    
    def __init__(self):
        self.collectors: Dict[str, MetricsCollector] = defaultdict(MetricsCollector)
        self.agent_profiles: Dict[str, Dict] = {}
        
    def record_agent_call(self, agent_id: str, response_time: float, success: bool = True):
        """记录Agent调用"""
        collector = self.collectors[agent_id]
        
        # 记录响应时间
        collector.record('response_time', response_time)
        
        # 记录调用次数
        collector.record('call_count', 1, MetricType.COUNTER)
        
        if success:
            collector.record('success_count', 1, MetricType.COUNTER)
        else:
            collector.record('error_count', 1, MetricType.COUNTER)
    
    def record_agent_performance(self, agent_id: str, accuracy: float, confidence: float):
        """记录Agent性能"""
        collector = self.collectors[agent_id]
        collector.record('accuracy', accuracy)
        collector.record('confidence', confidence)
    
    def record_agent_resources(self, agent_id: str, memory_mb: float, cpu_seconds: float):
        """记录Agent资源使用"""
        collector = self.collectors[agent_id]
        collector.record('memory_usage', memory_mb)
        collector.record('cpu_time', cpu_seconds)
    
    def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """获取Agent指标"""
        collector = self.collectors[agent_id]
        
        # 获取最新值
        response_time = collector.get_latest('response_time') or 0
        accuracy = collector.get_latest('accuracy') or 0
        confidence = collector.get_latest('confidence') or 0
        signal_count = int(collector.get_latest('call_count') or 0)
        error_count = int(collector.get_latest('error_count') or 0)
        memory_usage = collector.get_latest('memory_usage') or 0
        cpu_time = collector.get_latest('cpu_time') or 0
        
        return AgentMetrics(
            agent_id=agent_id,
            timestamp=datetime.now(),
            response_time=response_time,
            accuracy=accuracy,
            confidence=confidence,
            signal_count=signal_count,
            error_count=error_count,
            memory_usage=memory_usage,
            cpu_time=cpu_time
    
    def get_all_agent_metrics(self) -> List[AgentMetrics]:
        """获取所有Agent指标"""
        return [self.get_agent_metrics(agent_id) for agent_id in self.collectors.keys()]
    
    def get_agent_statistics(self, agent_id: str) -> Dict:
        """获取Agent统计信息"""
        if agent_id not in self.collectors:
            return {}
        
        collector = self.collectors[agent_id]
        
        stats = {
            'response_time': collector.get_statistics('response_time'),
            'accuracy': collector.get_statistics('accuracy'),
            'confidence': collector.get_statistics('confidence'),
            'call_count': collector.get_latest('call_count') or 0,
            'success_count': collector.get_latest('success_count') or 0,
            'error_count': collector.get_latest('error_count') or 0,
            'error_rate': self._calculate_error_rate(agent_id),
            'availability': self._calculate_availability(agent_id)
        }
        
        return stats
    
    def _calculate_error_rate(self, agent_id: str) -> float:
        """计算错误率"""
        collector = self.collectors[agent_id]
        total_calls = collector.get_latest('call_count') or 0
        errors = collector.get_latest('error_count') or 0
        
        return errors / total_calls if total_calls > 0 else 0
    
    def _calculate_availability(self, agent_id: str) -> float:
        """计算可用率"""
        collector = self.collectors[agent_id]
        total_calls = collector.get_latest('call_count') or 0
        success = collector.get_latest('success_count') or 0
        
        return success / total_calls if total_calls > 0 else 0


class PerformanceMonitor:
    """性能监控主类"""
    
    def __init__(self, 
                 enable_system_monitor: bool = True,
                 system_monitor_interval: float = 1.0):
        """
        初始化性能监控器
        
        Args:
            enable_system_monitor: 是否启用系统监控
            system_monitor_interval: 系统监控间隔
        """
        # 初始化各个监控器
        self.system_monitor = SystemMonitor(interval=system_monitor_interval) if enable_system_monitor else None
        self.trading_monitor = TradingMonitor()
        self.agent_monitor = AgentMonitor()
        
        # 警报管理
        self.alert_manager = AlertManager()
        
        # 性能报告
        self.report_generator = ReportGenerator(self)
        
        # 启动系统监控
        if self.system_monitor:
            self.system_monitor.start()
        
        logger.info("性能监控系统初始化完成")
    
    def shutdown(self):
        """关闭监控"""
        if self.system_monitor:
            self.system_monitor.stop()
        logger.info("性能监控系统已关闭")
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表板数据"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'system': self.system_monitor.get_current_metrics().to_dict() if self.system_monitor else {},
            'trading': self.trading_monitor.get_trading_metrics().to_dict(),
            'agents': [m.to_dict() for m in self.agent_monitor.get_all_agent_metrics()],
            'alerts': self.alert_manager.get_active_alerts()
        }
        
        return data
    
    def check_health(self) -> Dict:
        """健康检查"""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 检查系统资源
        if self.system_monitor:
            system_metrics = self.system_monitor.get_current_metrics()
            
            # CPU检查
            if system_metrics.cpu_usage > 80:
                health_status['checks']['cpu'] = 'warning'
                if system_metrics.cpu_usage > 95:
                    health_status['checks']['cpu'] = 'critical'
                    health_status['status'] = 'unhealthy'
            else:
                health_status['checks']['cpu'] = 'ok'
            
            # 内存检查
            if system_metrics.memory_usage > 80:
                health_status['checks']['memory'] = 'warning'
                if system_metrics.memory_usage > 95:
                    health_status['checks']['memory'] = 'critical'
                    health_status['status'] = 'unhealthy'
            else:
                health_status['checks']['memory'] = 'ok'
            
            # 磁盘检查
            if system_metrics.disk_usage > 85:
                health_status['checks']['disk'] = 'warning'
                if system_metrics.disk_usage > 95:
                    health_status['checks']['disk'] = 'critical'
                    health_status['status'] = 'unhealthy'
            else:
                health_status['checks']['disk'] = 'ok'
        
        # 检查交易系统
        trading_metrics = self.trading_monitor.get_trading_metrics()
        if trading_metrics.failed_trades > trading_metrics.successful_trades:
            health_status['checks']['trading'] = 'warning'
            health_status['status'] = 'degraded' if health_status['status'] == 'healthy' else health_status['status']
        else:
            health_status['checks']['trading'] = 'ok'
        
        # 检查Agents
        for agent_id in self.agent_monitor.collectors.keys():
            agent_stats = self.agent_monitor.get_agent_statistics(agent_id)
            if agent_stats.get('error_rate', 0) > 0.1:  # 错误率超过10%
                health_status['checks'][f'agent_{agent_id}'] = 'warning'
                if agent_stats.get('error_rate', 0) > 0.3:  # 错误率超过30%
                    health_status['checks'][f'agent_{agent_id}'] = 'critical'
                    health_status['status'] = 'unhealthy'
            else:
                health_status['checks'][f'agent_{agent_id}'] = 'ok'
        
        return health_status


class AlertManager:
    """警报管理器"""
    
    def __init__(self):
        self.alerts: List[Dict] = []
        self.alert_rules: List[Dict] = []
        self.active_alerts: Dict[str, Dict] = {}
        
    def add_rule(self, rule: Dict):
        """添加警报规则"""
        self.alert_rules.append(rule)
    
    def check_and_trigger(self, metric_name: str, value: float):
        """检查并触发警报"""
        for rule in self.alert_rules:
            if rule['metric'] == metric_name:
                if self._evaluate_rule(rule, value):
                    self._trigger_alert(rule, value)
    
    def _evaluate_rule(self, rule: Dict, value: float) -> bool:
        """评估规则"""
        operator = rule.get('operator', '>')
        threshold = rule.get('threshold', 0)
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        else:
            return False
    
    def _trigger_alert(self, rule: Dict, value: float):
        """触发警报"""
        alert_id = f"{rule['metric']}_{datetime.now().timestamp()}"
        
        alert = {
            'id': alert_id,
            'metric': rule['metric'],
            'value': value,
            'threshold': rule['threshold'],
            'severity': rule.get('severity', 'warning'),
            'message': rule.get('message', f"{rule['metric']} exceeded threshold"),
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        self.active_alerts[alert_id] = alert
        
        logger.warning(f"Alert triggered: {alert['message']} (value={value}, threshold={rule['threshold']})")
    
    def get_active_alerts(self) -> List[Dict]:
        """获取活跃警报"""
        return list(self.active_alerts.values())
    
    def acknowledge_alert(self, alert_id: str):
        """确认警报"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def generate_daily_report(self) -> Dict:
        """生成日报"""
        report = {
            'date': datetime.now().date().isoformat(),
            'summary': self._generate_summary(),
            'system': self._generate_system_report(),
            'trading': self._generate_trading_report(),
            'agents': self._generate_agent_report(),
            'alerts': self._generate_alert_report()
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """生成摘要"""
        trading_metrics = self.monitor.trading_monitor.get_trading_metrics()
        
        return {
            'total_trades': trading_metrics.total_trades,
            'success_rate': trading_metrics.win_rate,
            'total_pnl': trading_metrics.pnl,
            'system_health': self.monitor.check_health()['status']
        }
    
    def _generate_system_report(self) -> Dict:
        """生成系统报告"""
        if not self.monitor.system_monitor:
            return {}
        
        collector = self.monitor.system_monitor.collector
        
        return {
            'cpu': collector.get_statistics('cpu_usage'),
            'memory': collector.get_statistics('memory_usage'),
            'disk': collector.get_statistics('disk_usage'),
            'thread_count': collector.get_statistics('thread_count')
        }
    
    def _generate_trading_report(self) -> Dict:
        """生成交易报告"""
        return self.monitor.trading_monitor.get_trade_statistics()
    
    def _generate_agent_report(self) -> Dict:
        """生成Agent报告"""
        report = {}
        
        for agent_id in self.monitor.agent_monitor.collectors.keys():
            report[agent_id] = self.monitor.agent_monitor.get_agent_statistics(agent_id)
        
        return report
    
    def _generate_alert_report(self) -> Dict:
        """生成警报报告"""
        alert_manager = self.monitor.alert_manager
        
        return {
            'total_alerts': len(alert_manager.alerts),
            'active_alerts': len(alert_manager.active_alerts),
            'alerts_by_severity': self._count_by_severity(alert_manager.alerts)
        }
    
    def _count_by_severity(self, alerts: List[Dict]) -> Dict:
        """按严重程度统计"""
        counts = defaultdict(int)
        for alert in alerts:
            counts[alert.get('severity', 'info')] += 1
        return dict(counts)
    
    def export_report(self, filepath: str, report: Dict):
        """导出报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"报告已导出至: {filepath}")


if __name__ == "__main__":
    # 示例用法
    monitor = PerformanceMonitor(enable_system_monitor=True, system_monitor_interval=1.0)
    
    try:
        # 添加警报规则
        monitor.alert_manager.add_rule({
            'metric': 'cpu_usage',
            'operator': '>',
            'threshold': 80,
            'severity': 'warning',
            'message': 'CPU使用率过高'
        })
        
        # 模拟记录交易
        for i in range(10):
            monitor.trading_monitor.record_trade({
                'symbol': f'STOCK_{i}',
                'status': 'success' if i % 2 == 0 else 'failed',
                'execution_time': 0.1 + i * 0.01,
                'volume': 100 * (i + 1),
                'value': 1000 * (i + 1),
                'slippage': 0.01 * i,
                'pnl': 10 * (i - 5)
            })
        
        # 模拟记录Agent调用
        for i in range(5):
            agent_id = f'agent_{i}'
            monitor.agent_monitor.record_agent_call(agent_id, 0.5 + i * 0.1, success=i != 2)
            monitor.agent_monitor.record_agent_performance(agent_id, 0.8 - i * 0.1, 0.7 + i * 0.05)
        
        # 获取仪表板数据
        # dashboard = monitor.get_dashboard_data()
        # print(json.dumps(dashboard, indent=2, ensure_ascii=False))
        
        # 健康检查
        # health = monitor.check_health()
        # print(json.dumps(health, indent=2, ensure_ascii=False))
        
        # 生成日报
        # report = monitor.report_generator.generate_daily_report()
        # print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        
    finally:
        monitor.shutdown()