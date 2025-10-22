"""
实时监控和预警系统
集成Prometheus监控、Grafana仪表板、实时价格预警、异常检测
支持系统性能监控、策略监控、风险监控
"""

import logging
from typing import Dict, List, Optional, Any, Callable
import time
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# 告警级别枚举
# ============================================================================

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    PRICE = "price"           # 价格告警
    VOLATILITY = "volatility" # 波动率告警
    VOLUME = "volume"         # 成交量告警
    DRAWDOWN = "drawdown"     # 回撤告警
    PERFORMANCE = "performance" # 性能告警
    SYSTEM = "system"         # 系统告警


@dataclass
class Alert:
    """告警信息"""
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType
    symbol: str
    message: str
    value: float
    threshold: float


# ============================================================================
# Prometheus监控指标
# ============================================================================

class PrometheusMetrics:
    """Prometheus监控指标管理"""
    
    def __init__(self, port: int = 8000):
        """
        初始化Prometheus指标
        
        Args:
            port: HTTP服务端口
        """
        self.port = port
        self.metrics = {}
        self._init_metrics()
        self._start_server()
        
        logger.info(f"Prometheus指标服务已启动: http://localhost:{port}/metrics")
    
    def _init_metrics(self):
        """初始化指标"""
        try:
            from prometheus_client import Counter, Gauge, Histogram, Summary
            
            # 交易相关指标
            self.metrics['trades_total'] = Counter(
                'trading_trades_total',
                'Total number of trades',
                ['symbol', 'direction']
            )
            
            self.metrics['portfolio_value'] = Gauge(
                'trading_portfolio_value',
                'Current portfolio value'
            )
            
            self.metrics['position_size'] = Gauge(
                'trading_position_size',
                'Current position size',
                ['symbol']
            )
            
            # 性能指标
            self.metrics['backtest_duration'] = Histogram(
                'trading_backtest_duration_seconds',
                'Backtest execution time'
            )
            
            self.metrics['prediction_latency'] = Summary(
                'trading_prediction_latency_seconds',
                'Model prediction latency'
            )
            
            # 策略指标
            self.metrics['strategy_return'] = Gauge(
                'trading_strategy_return',
                'Strategy cumulative return',
                ['strategy_name']
            )
            
            self.metrics['strategy_sharpe'] = Gauge(
                'trading_strategy_sharpe',
                'Strategy Sharpe ratio',
                ['strategy_name']
            )
            
            self.metrics['strategy_drawdown'] = Gauge(
                'trading_strategy_drawdown',
                'Strategy maximum drawdown',
                ['strategy_name']
            )
            
            # 系统指标
            self.metrics['data_points_processed'] = Counter(
                'trading_data_points_processed_total',
                'Total data points processed'
            )
            
            self.metrics['errors_total'] = Counter(
                'trading_errors_total',
                'Total errors',
                ['error_type']
            )
            
            logger.info(f"初始化了 {len(self.metrics)} 个Prometheus指标")
            
        except ImportError:
            logger.error("prometheus_client未安装，请运行: pip install prometheus-client")
            raise
    
    def _start_server(self):
        """启动HTTP服务器"""
        try:
            from prometheus_client import start_http_server
            start_http_server(self.port)
        except Exception as e:
            logger.error(f"无法启动Prometheus服务器: {e}")
    
    def record_trade(self, symbol: str, direction: str):
        """记录交易"""
        self.metrics['trades_total'].labels(symbol=symbol, direction=direction).inc()
    
    def update_portfolio_value(self, value: float):
        """更新组合价值"""
        self.metrics['portfolio_value'].set(value)
    
    def update_position(self, symbol: str, size: float):
        """更新持仓"""
        self.metrics['position_size'].labels(symbol=symbol).set(size)
    
    def record_backtest_duration(self, duration: float):
        """记录回测时长"""
        self.metrics['backtest_duration'].observe(duration)
    
    def record_prediction_latency(self, latency: float):
        """记录预测延迟"""
        self.metrics['prediction_latency'].observe(latency)
    
    def update_strategy_metrics(self, strategy_name: str, 
                               return_: float, sharpe: float, drawdown: float):
        """更新策略指标"""
        self.metrics['strategy_return'].labels(strategy_name=strategy_name).set(return_)
        self.metrics['strategy_sharpe'].labels(strategy_name=strategy_name).set(sharpe)
        self.metrics['strategy_drawdown'].labels(strategy_name=strategy_name).set(drawdown)
    
    def record_data_processed(self, count: int = 1):
        """记录处理的数据点数"""
        self.metrics['data_points_processed'].inc(count)
    
    def record_error(self, error_type: str):
        """记录错误"""
        self.metrics['errors_total'].labels(error_type=error_type).inc()


# ============================================================================
# 价格监控和预警
# ============================================================================

class PriceMonitor:
    """实时价格监控"""
    
    def __init__(self, 
                 symbols: List[str],
                 alert_callback: Optional[Callable] = None):
        """
        初始化价格监控
        
        Args:
            symbols: 监控的股票代码列表
            alert_callback: 告警回调函数
        """
        self.symbols = symbols
        self.alert_callback = alert_callback
        self.price_history = {symbol: [] for symbol in symbols}
        self.alert_rules = {}
        
        logger.info(f"价格监控初始化: {len(symbols)} 只股票")
    
    def add_price_alert(self, 
                       symbol: str,
                       threshold_high: Optional[float] = None,
                       threshold_low: Optional[float] = None):
        """
        添加价格告警规则
        
        Args:
            symbol: 股票代码
            threshold_high: 价格上限
            threshold_low: 价格下限
        """
        self.alert_rules[symbol] = {
            'high': threshold_high,
            'low': threshold_low
        }
        logger.info(f"添加价格告警: {symbol}, 上限={threshold_high}, 下限={threshold_low}")
    
    def update_price(self, symbol: str, price: float):
        """
        更新价格并检查告警
        
        Args:
            symbol: 股票代码
            price: 最新价格
        """
        # 记录价格历史
        if symbol in self.price_history:
            self.price_history[symbol].append({
                'timestamp': datetime.now(),
                'price': price
            })
            
            # 限制历史长度
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        # 检查告警规则
        if symbol in self.alert_rules:
            rule = self.alert_rules[symbol]
            
            if rule['high'] and price > rule['high']:
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.PRICE,
                    symbol=symbol,
                    message=f"价格突破上限: {price:.2f} > {rule['high']:.2f}",
                    value=price,
                    threshold=rule['high']
                )
                self._trigger_alert(alert)
            
            if rule['low'] and price < rule['low']:
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.PRICE,
                    symbol=symbol,
                    message=f"价格跌破下限: {price:.2f} < {rule['low']:.2f}",
                    value=price,
                    threshold=rule['low']
                )
                self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Alert):
        """触发告警"""
        logger.warning(f"[{alert.level.value.upper()}] {alert.symbol}: {alert.message}")
        
        if self.alert_callback:
            self.alert_callback(alert)


# ============================================================================
# 异常检测
# ============================================================================

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 20, threshold_sigma: float = 3.0):
        """
        初始化异常检测器
        
        Args:
            window_size: 滑动窗口大小
            threshold_sigma: 异常阈值（标准差倍数）
        """
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        self.data_buffer = {}
        
        logger.info(f"异常检测器初始化: 窗口={window_size}, 阈值={threshold_sigma}σ")
    
    def detect_price_anomaly(self, symbol: str, price: float) -> Optional[Alert]:
        """
        检测价格异常
        
        Args:
            symbol: 股票代码
            price: 当前价格
            
        Returns:
            如果检测到异常返回Alert，否则返回None
        """
        import numpy as np
        
        # 初始化缓冲区
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        buffer = self.data_buffer[symbol]
        buffer.append(price)
        
        # 保持窗口大小
        if len(buffer) > self.window_size:
            buffer.pop(0)
        
        # 需要足够的数据点
        if len(buffer) < self.window_size:
            return None
        
        # 计算统计量
        mean = np.mean(buffer[:-1])  # 不包括当前点
        std = np.std(buffer[:-1])
        
        # Z-score检测
        if std > 0:
            z_score = abs((price - mean) / std)
            
            if z_score > self.threshold_sigma:
                return Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.CRITICAL,
                    alert_type=AlertType.PRICE,
                    symbol=symbol,
                    message=f"检测到价格异常: {price:.2f} (Z-score={z_score:.2f})",
                    value=price,
                    threshold=mean + self.threshold_sigma * std
                )
        
        return None
    
    def detect_volatility_spike(self, symbol: str, returns: List[float]) -> Optional[Alert]:
        """
        检测波动率突增
        
        Args:
            symbol: 股票代码
            returns: 收益率序列
            
        Returns:
            如果检测到异常返回Alert
        """
        import numpy as np
        
        if len(returns) < 2:
            return None
        
        # 计算滚动波动率
        recent_vol = np.std(returns[-5:]) if len(returns) >= 5 else np.std(returns)
        historical_vol = np.std(returns[:-5]) if len(returns) > 5 else recent_vol
        
        # 检测突增
        if historical_vol > 0 and recent_vol > historical_vol * 2.0:
            return Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                alert_type=AlertType.VOLATILITY,
                symbol=symbol,
                message=f"波动率突增: {recent_vol:.4f} (历史={historical_vol:.4f})",
                value=recent_vol,
                threshold=historical_vol * 2.0
            )
        
        return None


# ============================================================================
# 性能监控
# ============================================================================

class PerformanceMonitor:
    """系统性能监控"""
    
    def __init__(self):
        """初始化性能监控"""
        self.start_times = {}
        self.metrics_history = []
        
        logger.info("性能监控初始化")
    
    def start_timer(self, task_name: str):
        """开始计时"""
        self.start_times[task_name] = time.time()
    
    def end_timer(self, task_name: str) -> float:
        """
        结束计时
        
        Args:
            task_name: 任务名称
            
        Returns:
            执行时间（秒）
        """
        if task_name not in self.start_times:
            logger.warning(f"任务 {task_name} 未开始计时")
            return 0.0
        
        duration = time.time() - self.start_times[task_name]
        del self.start_times[task_name]
        
        # 记录历史
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'task': task_name,
            'duration': duration
        })
        
        logger.info(f"{task_name} 执行时间: {duration:.4f}秒")
        return duration
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / 1024**3,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except ImportError:
            logger.warning("psutil未安装，无法获取系统指标")
            return {}
    
    def check_system_health(self) -> List[Alert]:
        """检查系统健康状态"""
        alerts = []
        metrics = self.get_system_metrics()
        
        if not metrics:
            return alerts
        
        # CPU告警
        if metrics['cpu_percent'] > 90:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.SYSTEM,
                symbol="SYSTEM",
                message=f"CPU使用率过高: {metrics['cpu_percent']:.1f}%",
                value=metrics['cpu_percent'],
                threshold=90.0
            ))
        
        # 内存告警
        if metrics['memory_percent'] > 90:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.SYSTEM,
                symbol="SYSTEM",
                message=f"内存使用率过高: {metrics['memory_percent']:.1f}%",
                value=metrics['memory_percent'],
                threshold=90.0
            ))
        
        # 磁盘告警
        if metrics['disk_usage_percent'] > 90:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                alert_type=AlertType.SYSTEM,
                symbol="SYSTEM",
                message=f"磁盘使用率过高: {metrics['disk_usage_percent']:.1f}%",
                value=metrics['disk_usage_percent'],
                threshold=90.0
            ))
        
        return alerts


# ============================================================================
# 综合监控管理器
# ============================================================================

class MonitoringManager:
    """综合监控管理器"""
    
    def __init__(self, prometheus_port: int = 8000):
        """
        初始化监控管理器
        
        Args:
            prometheus_port: Prometheus端口
        """
        self.prometheus = PrometheusMetrics(port=prometheus_port)
        self.price_monitor = None
        self.anomaly_detector = AnomalyDetector()
        self.performance_monitor = PerformanceMonitor()
        self.alert_handlers = []
        
        logger.info("监控管理器初始化完成")
    
    def init_price_monitor(self, symbols: List[str]):
        """
        初始化价格监控
        
        Args:
            symbols: 监控股票列表
        """
        self.price_monitor = PriceMonitor(
            symbols=symbols,
            alert_callback=self._handle_alert
        )
    
    def add_alert_handler(self, handler: Callable):
        """
        添加告警处理器
        
        Args:
            handler: 告警处理函数
        """
        self.alert_handlers.append(handler)
        logger.info(f"添加告警处理器: {handler.__name__}")
    
    def _handle_alert(self, alert: Alert):
        """处理告警"""
        # 调用所有处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
    
    def start_health_check(self, interval: int = 60):
        """
        启动健康检查
        
        Args:
            interval: 检查间隔（秒）
        """
        def health_check_loop():
            while True:
                alerts = self.performance_monitor.check_system_health()
                for alert in alerts:
                    self._handle_alert(alert)
                time.sleep(interval)
        
        thread = threading.Thread(target=health_check_loop, daemon=True)
        thread.start()
        logger.info(f"健康检查已启动，间隔={interval}秒")


# ============================================================================
# 使用示例
# ============================================================================

def example_monitoring():
    """监控系统示例"""
    print("=== 实时监控和预警系统示例 ===\n")
    
    # 1. 初始化监控
    print("1. 初始化监控管理器")
    manager = MonitoringManager(prometheus_port=8000)
    print(f"  Prometheus: http://localhost:8000/metrics")
    
    # 2. 添加告警处理器
    def alert_handler(alert: Alert):
        """告警处理函数"""
        print(f"  [{alert.level.value.upper()}] {alert.message}")
    
    manager.add_alert_handler(alert_handler)
    
    # 3. 价格监控
    print("\n2. 价格监控和告警")
    symbols = ['600519.SH', '000001.SZ']
    manager.init_price_monitor(symbols)
    
    # 设置告警规则
    manager.price_monitor.add_price_alert('600519.SH', threshold_high=200.0, threshold_low=150.0)
    
    # 模拟价格更新
    import numpy as np
    for _ in range(5):
        price = np.random.uniform(160, 210)
        manager.price_monitor.update_price('600519.SH', price)
        print(f"  更新价格: 600519.SH = {price:.2f}")
    
    # 4. 异常检测
    print("\n3. 异常检测")
    prices = [100, 101, 102, 103, 102, 101, 150]  # 最后一个是异常值
    for i, price in enumerate(prices):
        anomaly = manager.anomaly_detector.detect_price_anomaly('000001.SZ', price)
        if anomaly:
            print(f"  检测到异常: {anomaly.message}")
    
    # 5. 性能监控
    print("\n4. 性能监控")
    manager.performance_monitor.start_timer('test_task')
    time.sleep(0.1)
    duration = manager.performance_monitor.end_timer('test_task')
    print(f"  任务执行时间: {duration:.4f}秒")
    
    # 系统指标
    metrics = manager.performance_monitor.get_system_metrics()
    print(f"  CPU: {metrics.get('cpu_percent', 0):.1f}%")
    print(f"  内存: {metrics.get('memory_percent', 0):.1f}%")
    
    # 6. Prometheus指标
    print("\n5. Prometheus指标更新")
    manager.prometheus.record_trade('600519.SH', 'buy')
    manager.prometheus.update_portfolio_value(1050000.0)
    manager.prometheus.update_strategy_metrics('momentum', return_=0.15, sharpe=1.8, drawdown=-0.08)
    print("  指标已更新，可访问 http://localhost:8000/metrics 查看")


if __name__ == "__main__":
    example_monitoring()
