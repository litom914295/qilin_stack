"""
依赖健康探针系统（P0-12）
监控外部依赖：数据库、Redis、Kafka、外部API、MLflow等
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import redis.asyncio as aioredis
from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class DependencyType(str, Enum):
    """依赖类型"""
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    ML_SERVICE = "ml_service"
    STORAGE = "storage"

@dataclass
class ProbeResult:
    """探针结果"""
    dependency_name: str
    dependency_type: DependencyType
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class DependencyConfig:
    """依赖配置"""
    name: str
    type: DependencyType
    endpoint: str
    timeout_seconds: float = 5.0
    critical: bool = True  # 是否为关键依赖

class DependencyHealthProbe:
    """依赖健康探针"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._init_prometheus_metrics()
        
        # 依赖配置
        self.dependencies: Dict[str, DependencyConfig] = {}
        
        # 探针结果缓存
        self.probe_results: Dict[str, ProbeResult] = {}
        
        # 会话管理
        self.http_session: Optional[aiohttp.ClientSession] = None
    
    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        
        # 依赖健康状态
        self.dependency_health = Gauge(
            'qilin_dependency_health_status',
            'Dependency health status (0=unhealthy, 0.5=degraded, 1=healthy)',
            ['dependency', 'type'],
            registry=self.registry
        )
        # 探针响应时间
        self.probe_response_time = Histogram(
            'qilin_dependency_probe_duration_seconds',
            'Dependency health probe duration',
            ['dependency', 'type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # 探针失败次数
        self.probe_failures = Counter(
            'qilin_dependency_probe_failures_total',
            'Total number of dependency probe failures',
            ['dependency', 'type'],
            registry=self.registry
        )
        # 依赖可用性
        self.dependency_availability = Gauge(
            'qilin_dependency_availability',
            'Dependency availability over time window',
            ['dependency', 'type'],
            registry=self.registry
        )
    
    def register_dependency(self, config: DependencyConfig):
        """
        注册依赖
        
        Args:
            config: 依赖配置
        """
        self.dependencies[config.name] = config
        logger.info(f"Registered dependency: {config.name} ({config.type.value})")
    
    async def _ensure_http_session(self):
        """确保HTTP会话存在"""
        if self.http_session is None or self.http_session.closed:
            timeout = aiohttp.ClientTimeout(total=5)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
    
    async def probe_database(self, config: DependencyConfig) -> ProbeResult:
        """
        探测数据库健康
        
        Args:
            config: 数据库配置
            
        Returns:
            探针结果
        """
        start_time = time.time()
        
        try:
            # 这里应该根据实际数据库类型实现
            # 示例：PostgreSQL健康检查
            import asyncpg
            
            conn = await asyncio.wait_for(
                asyncpg.connect(config.endpoint),
                timeout=config.timeout_seconds
            )

            # 执行简单查询
            await conn.fetchval('SELECT 1')
            await conn.close()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now()
            )
        
        except asyncio.TimeoutError:
            response_time_ms = config.timeout_seconds * 1000
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error_message="Connection timeout"
            )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    async def probe_redis(self, config: DependencyConfig) -> ProbeResult:
        """
        探测Redis健康
        
        Args:
            config: Redis配置
            
        Returns:
            探针结果
        """
        start_time = time.time()
        
        try:
            redis_client = await asyncio.wait_for(
                aioredis.from_url(config.endpoint),
                timeout=config.timeout_seconds
            )

            # PING测试
            await redis_client.ping()
            
            # 获取信息
            info = await redis_client.info()
            
            await redis_client.close()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now()
            )
        
        except asyncio.TimeoutError:
            response_time_ms = config.timeout_seconds * 1000
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error_message="Connection timeout"
            )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error_message=str(e)
            )
        
    async def probe_http_endpoint(self, config: DependencyConfig) -> ProbeResult:
        """
        探测HTTP端点健康
        
        Args:
            config: 端点配置
            
        Returns:
            探针结果
        """
        await self._ensure_http_session()
        start_time = time.time()
        
        try:
            async with asyncio.timeout(config.timeout_seconds):
                async with self.http_session.get(config.endpoint) as response:
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                    elif 200 <= response.status < 300:
                        status = HealthStatus.HEALTHY
                    elif response.status == 503:
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.UNHEALTHY
                    
                    return ProbeResult(
                        dependency_name=config.name,
                        dependency_type=config.type,
                        status=status,
                        response_time_ms=response_time_ms,
                        timestamp=datetime.now(),
                        details={'http_status': response.status}
                    )
                    
        except asyncio.TimeoutError:
            response_time_ms = config.timeout_seconds * 1000
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error_message="Request timeout"
            )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ProbeResult(
                dependency_name=config.name,
                dependency_type=config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    async def probe_dependency(self, dependency_name: str) -> ProbeResult:
        """
        探测单个依赖
        
        Args:
            dependency_name: 依赖名称
            
        Returns:
            探针结果
        """
        if dependency_name not in self.dependencies:
            raise ValueError(f"Dependency {dependency_name} not registered")
        
        config = self.dependencies[dependency_name]
        
        # 根据类型选择探针方法
        if config.type == DependencyType.DATABASE:
            result = await self.probe_database(config)
        elif config.type == DependencyType.CACHE:
            result = await self.probe_redis(config)
        elif config.type in [DependencyType.EXTERNAL_API, DependencyType.ML_SERVICE]:
            result = await self.probe_http_endpoint(config)
        else:
            result = await self.probe_http_endpoint(config)
        
        # 更新缓存
        self.probe_results[dependency_name] = result
        
        # 更新Prometheus指标
        self._update_metrics(result, config)
        
        return result
    
    def _update_metrics(self, result: ProbeResult, config: DependencyConfig):
        """更新Prometheus指标"""
        
        # 健康状态
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.UNKNOWN: 0.0
        }[result.status]
        
        self.dependency_health.labels(
            dependency=result.dependency_name,
            type=result.dependency_type.value
        ).set(status_value)
        
        # 响应时间
        self.probe_response_time.labels(
            dependency=result.dependency_name,
            type=result.dependency_type.value
        ).observe(result.response_time_ms / 1000)
        
        # 失败计数
        if result.status == HealthStatus.UNHEALTHY:
            self.probe_failures.labels(
                dependency=result.dependency_name,
                type=result.dependency_type.value
            ).inc()
    
    async def probe_all(self) -> Dict[str, ProbeResult]:
        """
        探测所有依赖
        
        Returns:
            所有依赖的探针结果
        """
        tasks = [
            self.probe_dependency(name)
            for name in self.dependencies.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        probe_results = {}
        for i, (name, result) in enumerate(zip(self.dependencies.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Failed to probe {name}: {result}")
                probe_results[name] = ProbeResult(
                    dependency_name=name,
                    dependency_type=self.dependencies[name].type,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    timestamp=datetime.now(),
                    error_message=str(result)
                )
            else:
                probe_results[name] = result
        
        return probe_results
    
    def get_overall_health(self) -> HealthStatus:
        """
        获取整体健康状态
        
        Returns:
            整体健康状态
        """
        if not self.probe_results:
            return HealthStatus.UNKNOWN
        
        # 检查关键依赖
        critical_deps = [
            name for name, config in self.dependencies.items()
            if config.critical
        ]
        
        for dep_name in critical_deps:
            if dep_name in self.probe_results:
                result = self.probe_results[dep_name]
                if result.status == HealthStatus.UNHEALTHY:
                    return HealthStatus.UNHEALTHY
        
        # 检查是否有降级依赖
        has_degraded = any(
            result.status == HealthStatus.DEGRADED
            for result in self.probe_results.values()
        )
        
        if has_degraded:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        获取健康摘要
        
        Returns:
            健康摘要字典
        """
        overall_status = self.get_overall_health()
        
        summary = {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'dependencies': {}
        }
        
        for name, result in self.probe_results.items():
            summary['dependencies'][name] = {
                'status': result.status.value,
                'type': result.dependency_type.value,
                'response_time_ms': result.response_time_ms,
                'critical': self.dependencies[name].critical,
                'error': result.error_message
            }
        
        return summary
    
    def export_metrics(self) -> str:
        """导出Prometheus指标"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def start_periodic_probe(self, interval_seconds: int = 30):
        """
        启动周期性探测
        
        Args:
            interval_seconds: 探测间隔（秒）
        """
        logger.info(f"Starting periodic probe with {interval_seconds}s interval")
        
        while True:
            try:
                await self.probe_all()
                logger.debug("Periodic probe completed")
            except Exception as e:
                logger.error(f"Periodic probe failed: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def cleanup(self):
        """清理资源"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

async def main():
    """示例用法"""
    logging.basicConfig(level=logging.INFO)
    
    probe = DependencyHealthProbe()
    
    # 注册依赖
    probe.register_dependency(DependencyConfig(
        name="postgres",
        type=DependencyType.DATABASE,
        endpoint="postgresql://user:pass@localhost:5432/qilin",
        timeout_seconds=5.0,
        critical=True
    ))
    
    probe.register_dependency(DependencyConfig(
        name="redis",
        type=DependencyType.CACHE,
        endpoint="redis://localhost:6379",
        timeout_seconds=3.0,
        critical=True
    ))
    
    probe.register_dependency(DependencyConfig(
        name="mlflow",
        type=DependencyType.ML_SERVICE,
        endpoint="http://localhost:5000/health",
        timeout_seconds=5.0,
        critical=False
    ))
    
    # 执行探测
    results = await probe.probe_all()
    
    print("\n=== Health Probe Results ===")
    for name, result in results.items():
        status_icon = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.UNKNOWN: "❓"
        }[result.status]
        
        print(f"{status_icon} {name}: {result.status.value} ({result.response_time_ms:.2f}ms)")
        if result.error_message:
            print(f"   Error: {result.error_message}")
    
    print(f"\nOverall Status: {probe.get_overall_health().value}")
    
    # 导出指标
    print("\n=== Prometheus Metrics ===")
    print(probe.export_metrics())
    
    await probe.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

