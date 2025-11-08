"""
监控告警系统
P2-4任务: 监控告警系统 (48h estimated, ROI 160%)

功能:
1. Prometheus指标采集
2. 系统健康监控
3. 数据质量监控
4. 模型性能监控
5. 交易执行监控
6. 风险指标监控

作者: Qilin Stack Team
日期: 2025-11-07
"""

from typing import Dict, List, Optional, Any
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 尝试导入Prometheus客户端
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info
    from prometheus_client import start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
    logger.info("✅ Prometheus客户端可用")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("⚠️ prometheus_client未安装,请安装: pip install prometheus-client")
    # 创建虚拟类
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    Summary = Histogram
    Info = Counter


# ==================== 指标定义 ====================

class QilinMetrics:
    """
    Qilin Stack Prometheus指标集合
    
    指标类型:
    - Counter: 只增不减的计数器
    - Gauge: 可增可减的仪表
    - Histogram: 直方图(观察值分布)
    - Summary: 摘要(分位数)
    """
    
    def __init__(self, namespace: str = 'qilin'):
        """
        初始化指标集合
        
        Args:
            namespace: 指标命名空间
        """
        self.namespace = namespace
        
        # === 1. 系统健康指标 ===
        
        # CPU使用率
        self.cpu_usage = Gauge(
            f'{namespace}_system_cpu_usage_percent',
            'CPU使用率百分比',
            ['host']
        )
        
        # 内存使用率
        self.memory_usage = Gauge(
            f'{namespace}_system_memory_usage_percent',
            '内存使用率百分比',
            ['host']
        )
        
        # 磁盘使用率
        self.disk_usage = Gauge(
            f'{namespace}_system_disk_usage_percent',
            '磁盘使用率百分比',
            ['host', 'mount_point']
        )
        
        # 系统运行时间
        self.uptime = Gauge(
            f'{namespace}_system_uptime_seconds',
            '系统运行时间(秒)',
            ['host']
        )
        
        # === 2. 数据质量指标 ===
        
        # 数据缺失率
        self.data_missing_rate = Gauge(
            f'{namespace}_data_missing_rate',
            '数据缺失率',
            ['symbol', 'field']
        )
        
        # 数据延迟
        self.data_latency = Histogram(
            f'{namespace}_data_latency_seconds',
            '数据延迟(秒)',
            ['source'],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60]
        )
        
        # 数据更新次数
        self.data_updates = Counter(
            f'{namespace}_data_updates_total',
            '数据更新总次数',
            ['source', 'status']
        )
        
        # === 3. 模型性能指标 ===
        
        # 模型IC
        self.model_ic = Gauge(
            f'{namespace}_model_ic',
            '模型信息系数(IC)',
            ['model_id', 'period']
        )
        
        # 模型Sharpe比率
        self.model_sharpe = Gauge(
            f'{namespace}_model_sharpe_ratio',
            '模型Sharpe比率',
            ['model_id']
        )
        
        # 模型预测次数
        self.model_predictions = Counter(
            f'{namespace}_model_predictions_total',
            '模型预测总次数',
            ['model_id', 'status']
        )
        
        # 模型训练时间
        self.model_train_duration = Histogram(
            f'{namespace}_model_train_duration_seconds',
            '模型训练时间(秒)',
            ['model_id'],
            buckets=[10, 30, 60, 300, 600, 1800, 3600]
        )
        
        # === 4. 交易执行指标 ===
        
        # 订单总数
        self.orders_total = Counter(
            f'{namespace}_orders_total',
            '订单总数',
            ['status', 'side']
        )
        
        # 订单成功率
        self.order_success_rate = Gauge(
            f'{namespace}_order_success_rate',
            '订单成功率',
            ['broker']
        )
        
        # 订单执行延迟
        self.order_execution_latency = Histogram(
            f'{namespace}_order_execution_latency_seconds',
            '订单执行延迟(秒)',
            ['broker'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        )
        
        # 成交金额
        self.trade_amount = Counter(
            f'{namespace}_trade_amount_total',
            '成交总金额',
            ['side']
        )
        
        # === 5. 风险指标 ===
        
        # 组合净值
        self.portfolio_value = Gauge(
            f'{namespace}_portfolio_value',
            '组合净值',
            ['account']
        )
        
        # 最大回撤
        self.max_drawdown = Gauge(
            f'{namespace}_max_drawdown',
            '最大回撤',
            ['account']
        )
        
        # 持仓集中度
        self.position_concentration = Gauge(
            f'{namespace}_position_concentration',
            '持仓集中度(单票最大占比)',
            ['account']
        )
        
        # 日内交易次数
        self.daily_trades = Counter(
            f'{namespace}_daily_trades_total',
            '日内交易总次数',
            ['account']
        )
        
        # 风险检查失败次数
        self.risk_check_failures = Counter(
            f'{namespace}_risk_check_failures_total',
            '风险检查失败次数',
            ['reason']
        )
        
        # 熔断触发次数
        self.circuit_breaker_triggers = Counter(
            f'{namespace}_circuit_breaker_triggers_total',
            '熔断触发次数',
            ['account']
        )
        
        logger.info(f"✅ Qilin指标集合初始化完成: 命名空间={namespace}")
    
    def record_system_metrics(self, cpu: float, memory: float, disk: float, uptime: float, host: str = 'localhost'):
        """记录系统指标"""
        self.cpu_usage.labels(host=host).set(cpu)
        self.memory_usage.labels(host=host).set(memory)
        self.disk_usage.labels(host=host, mount_point='/').set(disk)
        self.uptime.labels(host=host).set(uptime)
    
    def record_data_quality(self, symbol: str, field: str, missing_rate: float):
        """记录数据质量"""
        self.data_missing_rate.labels(symbol=symbol, field=field).set(missing_rate)
    
    def record_model_performance(self, model_id: str, ic: float, sharpe: float):
        """记录模型性能"""
        self.model_ic.labels(model_id=model_id, period='daily').set(ic)
        self.model_sharpe.labels(model_id=model_id).set(sharpe)
    
    def record_order(self, status: str, side: str, broker: str = 'default'):
        """记录订单"""
        self.orders_total.labels(status=status, side=side).inc()
    
    def record_risk_metrics(self, account: str, value: float, drawdown: float, concentration: float):
        """记录风险指标"""
        self.portfolio_value.labels(account=account).set(value)
        self.max_drawdown.labels(account=account).set(drawdown)
        self.position_concentration.labels(account=account).set(concentration)


# ==================== 告警规则 ====================

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric: str
    condition: str  # '>', '<', '>=', '<=', '=='
    threshold: float
    duration: int = 60  # 持续时间(秒)
    severity: str = 'warning'  # 'critical', 'warning', 'info'
    message: str = ""
    enabled: bool = True


class AlertManager:
    """
    告警管理器
    
    功能:
    1. 定义告警规则
    2. 评估告警条件
    3. 触发告警通知
    4. 告警历史记录
    """
    
    def __init__(self):
        """初始化告警管理器"""
        self.rules: List[AlertRule] = []
        self.alert_history: List[Dict] = []
        self.active_alerts: Dict[str, datetime] = {}
        
        # 初始化默认告警规则
        self._init_default_rules()
        
        logger.info("✅ 告警管理器初始化完成")
    
    def _init_default_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            # 系统告警
            AlertRule(
                name='cpu_high',
                metric='qilin_system_cpu_usage_percent',
                condition='>',
                threshold=80.0,
                severity='warning',
                message='CPU使用率过高'
            ),
            AlertRule(
                name='memory_high',
                metric='qilin_system_memory_usage_percent',
                condition='>',
                threshold=85.0,
                severity='warning',
                message='内存使用率过高'
            ),
            AlertRule(
                name='disk_high',
                metric='qilin_system_disk_usage_percent',
                condition='>',
                threshold=90.0,
                severity='critical',
                message='磁盘使用率严重过高'
            ),
            
            # 数据质量告警
            AlertRule(
                name='data_missing_high',
                metric='qilin_data_missing_rate',
                condition='>',
                threshold=0.1,  # 10%
                severity='warning',
                message='数据缺失率过高'
            ),
            
            # 模型性能告警
            AlertRule(
                name='model_ic_low',
                metric='qilin_model_ic',
                condition='<',
                threshold=0.02,
                severity='warning',
                message='模型IC过低'
            ),
            
            # 交易执行告警
            AlertRule(
                name='order_success_rate_low',
                metric='qilin_order_success_rate',
                condition='<',
                threshold=0.95,  # 95%
                severity='critical',
                message='订单成功率过低'
            ),
            
            # 风险告警
            AlertRule(
                name='drawdown_high',
                metric='qilin_max_drawdown',
                condition='>',
                threshold=0.1,  # 10%
                severity='critical',
                message='最大回撤超限,触发熔断'
            ),
            AlertRule(
                name='concentration_high',
                metric='qilin_position_concentration',
                condition='>',
                threshold=0.2,  # 20%
                severity='warning',
                message='持仓集中度过高'
            ),
        ]
        
        self.rules = default_rules
        logger.info(f"已加载{len(default_rules)}条默认告警规则")
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")
    
    def evaluate_rule(self, rule: AlertRule, current_value: float) -> bool:
        """
        评估告警规则
        
        Args:
            rule: 告警规则
            current_value: 当前值
            
        Returns:
            triggered: 是否触发告警
        """
        if not rule.enabled:
            return False
        
        condition = rule.condition
        threshold = rule.threshold
        
        if condition == '>':
            return current_value > threshold
        elif condition == '<':
            return current_value < threshold
        elif condition == '>=':
            return current_value >= threshold
        elif condition == '<=':
            return current_value <= threshold
        elif condition == '==':
            return abs(current_value - threshold) < 1e-6
        else:
            return False
    
    def trigger_alert(self, rule: AlertRule, current_value: float):
        """
        触发告警
        
        Args:
            rule: 告警规则
            current_value: 当前值
        """
        # 检查是否在活跃告警中
        if rule.name in self.active_alerts:
            # 检查持续时间
            elapsed = (datetime.now() - self.active_alerts[rule.name]).total_seconds()
            if elapsed < rule.duration:
                return  # 还未达到持续时间
        else:
            # 首次触发,记录时间
            self.active_alerts[rule.name] = datetime.now()
            return
        
        # 达到持续时间,发送告警
        alert = {
            'rule_name': rule.name,
            'metric': rule.metric,
            'severity': rule.severity,
            'message': rule.message,
            'current_value': current_value,
            'threshold': rule.threshold,
            'timestamp': datetime.now()
        }
        
        self.alert_history.append(alert)
        
        # 发送告警通知
        self._send_notification(alert)
        
        # 从活跃告警中移除(避免重复发送)
        if rule.name in self.active_alerts:
            del self.active_alerts[rule.name]
    
    def _send_notification(self, alert: Dict):
        """
        发送告警通知
        
        Args:
            alert: 告警信息
        """
        severity_emoji = {
            'critical': '🔴',
            'warning': '🟠',
            'info': '🟡'
        }
        
        emoji = severity_emoji.get(alert['severity'], '⚪')
        
        message = (
            f"{emoji} **Qilin告警** [{alert['severity'].upper()}]\n"
            f"规则: {alert['rule_name']}\n"
            f"指标: {alert['metric']}\n"
            f"当前值: {alert['current_value']:.4f}\n"
            f"阈值: {alert['threshold']:.4f}\n"
            f"消息: {alert['message']}\n"
            f"时间: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        logger.warning(f"\n{'='*60}\n{message}\n{'='*60}")
        
        # TODO: 集成实际通知渠道
        # - 邮件: SMTP
        # - 短信: Twilio/阿里云
        # - 微信: 企业微信机器人
        # - 钉钉: 钉钉机器人
    
    def get_active_alerts(self) -> List[str]:
        """获取活跃告警列表"""
        return list(self.active_alerts.keys())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """获取告警历史"""
        return self.alert_history[-limit:]


# ==================== 监控服务 ====================

class MonitoringService:
    """
    监控服务
    
    功能:
    1. 启动Prometheus HTTP服务器
    2. 定期采集指标
    3. 评估告警规则
    4. 生成监控报告
    """
    
    def __init__(
        self,
        metrics: QilinMetrics,
        alert_manager: AlertManager,
        port: int = 9090
    ):
        """
        初始化监控服务
        
        Args:
            metrics: 指标集合
            alert_manager: 告警管理器
            port: Prometheus HTTP端口
        """
        self.metrics = metrics
        self.alert_manager = alert_manager
        self.port = port
        self.running = False
        
        logger.info(f"✅ 监控服务初始化完成: 端口={port}")
    
    def start(self):
        """启动监控服务"""
        if not PROMETHEUS_AVAILABLE:
            logger.error("❌ Prometheus客户端未安装,无法启动监控服务")
            return False
        
        try:
            # 启动HTTP服务器
            start_http_server(self.port)
            self.running = True
            
            logger.info(f"✅ Prometheus HTTP服务器已启动: http://localhost:{self.port}/metrics")
            logger.info(f"   访问 http://localhost:{self.port}/metrics 查看所有指标")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动监控服务失败: {e}")
            return False
    
    def stop(self):
        """停止监控服务"""
        self.running = False
        logger.info("监控服务已停止")
    
    def collect_system_metrics(self):
        """采集系统指标"""
        try:
            import psutil
            
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            uptime = time.time() - psutil.boot_time()
            
            self.metrics.record_system_metrics(cpu, memory, disk, uptime)
            
            logger.debug(f"系统指标: CPU={cpu:.1f}% 内存={memory:.1f}% 磁盘={disk:.1f}%")
            
        except ImportError:
            logger.warning("psutil未安装,无法采集系统指标")
    
    def generate_report(self) -> Dict:
        """生成监控报告"""
        return {
            'timestamp': datetime.now(),
            'service_status': 'running' if self.running else 'stopped',
            'active_alerts': self.alert_manager.get_active_alerts(),
            'alert_count': len(self.alert_manager.alert_history),
            'metrics_endpoint': f'http://localhost:{self.port}/metrics'
        }


# ==================== 便捷创建函数 ====================

def create_monitoring_service(port: int = 9090) -> MonitoringService:
    """
    创建监控服务的便捷函数
    
    Args:
        port: Prometheus HTTP端口
        
    Returns:
        service: 监控服务实例
    """
    metrics = QilinMetrics(namespace='qilin')
    alert_manager = AlertManager()
    service = MonitoringService(metrics, alert_manager, port)
    
    return service


# ==================== 使用示例 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("监控告警系统示例")
    print("=" * 60)
    
    # 1. 创建监控服务
    service = create_monitoring_service(port=9090)
    
    # 2. 启动服务
    if service.start():
        print(f"\n✅ 监控服务已启动")
        print(f"   Prometheus端点: http://localhost:9090/metrics")
        print(f"\n⚠️ 在生产环境中:")
        print(f"   1. 安装Prometheus服务器")
        print(f"   2. 配置抓取目标: localhost:9090")
        print(f"   3. 安装Grafana")
        print(f"   4. 导入Qilin Dashboard")
    
    # 3. 模拟采集指标
    print("\n" + "=" * 60)
    print("模拟指标采集")
    print("=" * 60)
    
    # 系统指标
    service.collect_system_metrics()
    
    # 数据质量指标
    service.metrics.record_data_quality('000001.SZ', 'close', 0.05)
    print("✅ 数据质量指标已记录")
    
    # 模型性能指标
    service.metrics.record_model_performance('lgb_model_v1', ic=0.03, sharpe=1.5)
    print("✅ 模型性能指标已记录")
    
    # 交易执行指标
    service.metrics.record_order('filled', 'buy', 'ptrade')
    service.metrics.record_order('filled', 'sell', 'ptrade')
    service.metrics.record_order('failed', 'buy', 'ptrade')
    print("✅ 交易执行指标已记录")
    
    # 风险指标
    service.metrics.record_risk_metrics(
        account='test_account',
        value=1050000.0,
        drawdown=0.05,
        concentration=0.15
    )
    print("✅ 风险指标已记录")
    
    # 4. 测试告警
    print("\n" + "=" * 60)
    print("测试告警系统")
    print("=" * 60)
    
    # 模拟触发告警
    cpu_rule = service.alert_manager.rules[0]  # CPU告警
    if service.alert_manager.evaluate_rule(cpu_rule, 85.0):
        print(f"⚠️ 告警触发: {cpu_rule.name}")
        service.alert_manager.trigger_alert(cpu_rule, 85.0)
    
    # 5. 生成报告
    report = service.generate_report()
    print("\n" + "=" * 60)
    print("监控报告")
    print("=" * 60)
    print(f"服务状态: {report['service_status']}")
    print(f"活跃告警: {len(report['active_alerts'])}个")
    print(f"历史告警: {report['alert_count']}条")
    print(f"指标端点: {report['metrics_endpoint']}")
    
    print("\n✅ 监控告警系统演示完成!")
    print("\n📊 Grafana Dashboard模板:")
    print("   - CPU/内存/磁盘使用率")
    print("   - 数据质量大盘")
    print("   - 模型性能趋势")
    print("   - 订单执行统计")
    print("   - 风险指标监控")
