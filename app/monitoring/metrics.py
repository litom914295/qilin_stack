"""
Prometheus监控指标采集器
"""

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any, Optional
import time
import functools
import logging

logger = logging.getLogger(__name__)


class SystemMetrics:
    """系统级监控指标"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # 系统信息
        self.system_info = Info(
            'qilin_system',
            'System information',
            registry=self.registry
        )
        
        # 请求指标
        self.request_count = Counter(
            'qilin_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        self.request_duration = Histogram(
            'qilin_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Agent相关指标
        self.active_agents = Gauge(
            'qilin_active_agents',
            'Number of active agents',
            registry=self.registry
        )
        self.agent_analysis_duration = Histogram(
            'qilin_agent_analysis_duration_seconds',
            'Agent analysis duration',
            ['agent_name'],
            registry=self.registry
        )
        
        self.agent_errors = Counter(
            'qilin_agent_errors_total',
            'Total agent errors',
            ['agent_name', 'error_type'],
            registry=self.registry
        )
        
        # 业务指标
        self.stocks_analyzed = Counter(
            'qilin_stocks_analyzed_total',
            'Total stocks analyzed',
            registry=self.registry
        )
        self.trading_signals = Counter(
            'qilin_trading_signals_total',
            'Total trading signals generated',
            ['signal_type'],
            registry=self.registry
        )
        
        self.recommendations = Gauge(
            'qilin_daily_recommendations',
            'Number of daily recommendations',
            registry=self.registry
        )
        
        # 数据质量指标
        self.data_quality_score = Gauge(
            'qilin_data_quality_score',
            'Data quality score (0-1)',
            ['data_source'],
            registry=self.registry
        )
        self.data_fetch_errors = Counter(
            'qilin_data_fetch_errors_total',
            'Total data fetch errors',
            ['data_source', 'error_type'],
            registry=self.registry
        )
        
        # 性能指标
        self.cache_hits = Counter(
            'qilin_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        self.cache_misses = Counter(
            'qilin_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # 系统健康
        self.system_health = Gauge(
            'qilin_system_health',
            'System health status (0=unhealthy, 1=healthy)',
            registry=self.registry
        )
    
    def track_request(self, method: str, endpoint: str):
        """请求追踪装饰器"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = 'success'
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = 'error'
                    raise
                finally:
                    duration = time.time() - start_time
                    self.request_count.labels(
                        method=method,
                        endpoint=endpoint,
                        status=status
                    ).inc()
                    self.request_duration.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
            
            return wrapper
        return decorator
    
    def track_agent_execution(self, agent_name: str):
        """Agent执行追踪装饰器"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.agent_errors.labels(
                        agent_name=agent_name,
                        error_type=type(e).__name__
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.agent_analysis_duration.labels(
                        agent_name=agent_name
                    ).observe(duration)
            
            return wrapper
        return decorator
    
    def set_system_info(self, info: Dict[str, str]):
        """设置系统信息"""
        self.system_info.info(info)
    
    def update_health_status(self, is_healthy: bool):
        """更新系统健康状态"""
        self.system_health.set(1 if is_healthy else 0)
    
    def get_metrics(self) -> bytes:
        """获取Prometheus格式的指标"""
        return generate_latest(self.registry)


# 全局metrics实例
metrics = SystemMetrics()


def start_metrics_server(port: int = 9090):
    """启动Prometheus metrics HTTP服务器"""
    try:
        start_http_server(port, registry=metrics.registry)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
