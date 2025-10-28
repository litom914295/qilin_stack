"""
故障切换控制器（P0-15.2）
实现自动/手动故障切换，健康检查，切换决策逻辑
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)


class AZStatus(str, Enum):
    """可用区状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailoverState(str, Enum):
    """切换状态"""
    NORMAL = "normal"
    PREPARING = "preparing"
    SWITCHING = "switching"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service: str
    endpoint: str
    status: bool
    response_time: float
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AZHealth:
    """可用区健康度"""
    az_name: str
    status: AZStatus
    healthy_services: int
    total_services: int
    health_score: float
    last_check: datetime
    details: List[HealthCheckResult]


@dataclass
class FailoverDecision:
    """切换决策"""
    should_failover: bool
    reason: str
    primary_az: str
    target_az: str
    confidence: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FailoverController:
    """故障切换控制器"""
    
    def __init__(
        self,
        primary_az: str = "az-a",
        secondary_az: str = "az-b",
        health_check_interval: int = 30,
        failure_threshold: int = 3
    ):
        """
        初始化切换控制器
        
        Args:
            primary_az: 主可用区
            secondary_az: 备可用区
            health_check_interval: 健康检查间隔（秒）
            failure_threshold: 故障阈值（连续失败次数）
        """
        self.primary_az = primary_az
        self.secondary_az = secondary_az
        self.current_az = primary_az
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        
        self.state = FailoverState.NORMAL
        self.failure_count = 0
        self.last_failover = None
        self.health_history = []
        
        # 服务端点配置
        self.service_endpoints = {
            "api": {
                "primary": f"http://{primary_az}.api.qilin:8000/health",
                "secondary": f"http://{secondary_az}.api.qilin:8000/health"
            },
            "database": {
                "primary": f"postgresql://{primary_az}.db.qilin:5432",
                "secondary": f"postgresql://{secondary_az}.db.qilin:5432"
            },
            "redis": {
                "primary": f"redis://{primary_az}.redis.qilin:6379",
                "secondary": f"redis://{secondary_az}.redis.qilin:6379"
            },
            "kafka": {
                "primary": f"{primary_az}.kafka.qilin:9092",
                "secondary": f"{secondary_az}.kafka.qilin:9092"
            }
        }
        
        logger.info(f"Failover controller initialized: primary={primary_az}, secondary={secondary_az}")
    
    async def check_service_health(
        self,
        service: str,
        endpoint: str
    ) -> HealthCheckResult:
        """
        检查服务健康状态
        
        Args:
            service: 服务名称
            endpoint: 服务端点
            
        Returns:
            健康检查结果
        """
        import aiohttp
        start_time = time.time()
        
        try:
            if service == "api":
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, timeout=5) as response:
                        response_time = time.time() - start_time
                        return HealthCheckResult(
                            service=service,
                            endpoint=endpoint,
                            status=response.status == 200,
                            response_time=response_time
                        )
            
            elif service == "database":
                import asyncpg
                conn = await asyncpg.connect(endpoint, timeout=5)
                result = await conn.fetchval("SELECT 1")
                await conn.close()
                response_time = time.time() - start_time
                return HealthCheckResult(
                    service=service,
                    endpoint=endpoint,
                    status=result == 1,
                    response_time=response_time
                )
            
            elif service == "redis":
                import aioredis
                redis = await aioredis.create_redis_pool(endpoint)
                pong = await redis.ping()
                redis.close()
                await redis.wait_closed()
                response_time = time.time() - start_time
                return HealthCheckResult(
                    service=service,
                    endpoint=endpoint,
                    status=pong == b'PONG',
                    response_time=response_time
                )
            
            elif service == "kafka":
                from aiokafka import AIOKafkaProducer
                producer = AIOKafkaProducer(bootstrap_servers=endpoint)
                await producer.start()
                await producer.stop()
                response_time = time.time() - start_time
                return HealthCheckResult(
                    service=service,
                    endpoint=endpoint,
                    status=True,
                    response_time=response_time
                )
            
            else:
                return HealthCheckResult(
                    service=service,
                    endpoint=endpoint,
                    status=False,
                    response_time=0,
                    error=f"Unknown service type: {service}"
                )
        
        except Exception as e:
            return HealthCheckResult(
                service=service,
                endpoint=endpoint,
                status=False,
                response_time=response_time,
                error=str(e)
            )
    
    async def check_az_health(self, az: str) -> AZHealth:
        """
        检查可用区健康度
        
        Args:
            az: 可用区名称
            
        Returns:
            可用区健康状态
        """
        results = []
        
        for service, endpoints in self.service_endpoints.items():
            endpoint = endpoints.get("primary" if az == self.primary_az else "secondary")
            if endpoint:
                result = await self.check_service_health(service, endpoint)
                results.append(result)
        
        healthy_count = sum(1 for r in results if r.status)
        total_count = len(results)
        health_score = healthy_count / total_count if total_count > 0 else 0
        
        # 判断AZ状态
        if health_score >= 0.9:
            status = AZStatus.HEALTHY
        elif health_score >= 0.5:
            status = AZStatus.DEGRADED
        elif health_score > 0:
            status = AZStatus.UNHEALTHY
        else:
            status = AZStatus.UNKNOWN
        
        return AZHealth(
            az_name=az,
            status=status,
            healthy_services=healthy_count,
            total_services=total_count,
            health_score=health_score,
            last_check=datetime.now(),
            details=results
        )
    
    def analyze_health_trend(self) -> Dict:
        """
        分析健康趋势
        
        Returns:
            趋势分析结果
        """
        if len(self.health_history) < 3:
            return {
                "trend": "unknown",
                "confidence": 0
            }
        
        recent_scores = [h.health_score for h in self.health_history[-5:]]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # 计算趋势
        if all(s > 0.9 for s in recent_scores):
            trend = "stable"
            confidence = 0.95
        elif recent_scores[-1] < recent_scores[0]:
            trend = "degrading"
            confidence = 0.8
        elif recent_scores[-1] > recent_scores[0]:
            trend = "improving"
            confidence = 0.7
        else:
            trend = "fluctuating"
            confidence = 0.6
        
        return {
            "trend": trend,
            "confidence": confidence,
            "average_score": avg_score,
            "recent_scores": recent_scores
        }
    
    async def make_failover_decision(
        self,
        primary_health: AZHealth,
        secondary_health: AZHealth
    ) -> FailoverDecision:
        """
        做出切换决策
        
        Args:
            primary_health: 主AZ健康状态
            secondary_health: 备AZ健康状态
            
        Returns:
            切换决策
        """
        # 分析健康趋势
        trend_analysis = self.analyze_health_trend()
        
        # 决策逻辑
        should_failover = False
        reason = ""
        confidence = 0.0
        
        # 场景1：主AZ完全不可用
        if primary_health.status == AZStatus.UNHEALTHY:
            if secondary_health.status in [AZStatus.HEALTHY, AZStatus.DEGRADED]:
                should_failover = True
                reason = f"Primary AZ is unhealthy ({primary_health.health_score:.2%})"
                confidence = 0.95
        
        # 场景2：主AZ严重降级且持续恶化
        elif primary_health.status == AZStatus.DEGRADED:
            if trend_analysis["trend"] == "degrading" and self.failure_count >= self.failure_threshold:
                if secondary_health.status == AZStatus.HEALTHY:
                    should_failover = True
                    reason = f"Primary AZ is degrading (trend: {trend_analysis['trend']})"
                    confidence = 0.8
        
        # 场景3：关键服务不可用
        critical_services = ["database", "kafka"]
        critical_failures = [
            r for r in primary_health.details
            if r.service in critical_services and not r.status
        ]
        if len(critical_failures) > 0:
            if secondary_health.status == AZStatus.HEALTHY:
                should_failover = True
                reason = f"Critical service failure: {[f.service for f in critical_failures]}"
                confidence = 0.9
        
        # 防止频繁切换
        if self.last_failover:
            time_since_last = datetime.now() - self.last_failover
            if time_since_last < timedelta(minutes=30):
                should_failover = False
                reason = f"Too soon since last failover ({time_since_last.total_seconds()//60:.0f} minutes ago)"
                confidence = 0.3
        
        return FailoverDecision(
            should_failover=should_failover,
            reason=reason,
            primary_az=self.primary_az,
            target_az=self.secondary_az if should_failover else self.primary_az,
            confidence=confidence
        )
    
    async def execute_failover(self, decision: FailoverDecision) -> bool:
        """
        执行故障切换
        
        Args:
            decision: 切换决策
            
        Returns:
            是否成功
        """
        logger.info(f"Executing failover: {decision.reason}")
        self.state = FailoverState.PREPARING
        
        try:
            # 1. 准备阶段
            logger.info("Phase 1: Preparing failover")
            await self._prepare_failover(decision.target_az)
            
            # 2. 切换阶段
            self.state = FailoverState.SWITCHING
            logger.info("Phase 2: Switching traffic")
            await self._switch_traffic(decision.target_az)
            
            # 3. 验证阶段
            self.state = FailoverState.VALIDATING
            logger.info("Phase 3: Validating services")
            validation_result = await self._validate_services(decision.target_az)
            
            if validation_result:
                self.state = FailoverState.COMPLETED
                self.current_az = decision.target_az
                self.last_failover = datetime.now()
                self.failure_count = 0
                logger.info(f"Failover completed successfully to {decision.target_az}")
                return True
            else:
                logger.error("Service validation failed")
                await self.rollback_failover()
                return False
                
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            self.state = FailoverState.FAILED
            await self.rollback_failover()
            return False
    
    async def _prepare_failover(self, target_az: str):
        """准备切换"""
        # 1. 检查目标AZ资源
        logger.info(f"Checking {target_az} resources")
        
        # 2. 停止主AZ写入操作
        logger.info("Stopping write operations on primary")
        
        # 3. 等待数据同步
        logger.info("Waiting for data sync completion")
        await asyncio.sleep(2)  # 模拟等待
    
    async def _switch_traffic(self, target_az: str):
        """切换流量"""
        # 1. 更新DNS记录
        logger.info(f"Updating DNS to point to {target_az}")
        
        # 2. 更新负载均衡器
        logger.info("Updating load balancer configuration")
        
        # 3. 更新服务发现
        logger.info("Updating service discovery")
        await asyncio.sleep(1)  # 模拟切换
    
    async def _validate_services(self, target_az: str) -> bool:
        """验证服务"""
        logger.info(f"Validating services in {target_az}")
        
        # 执行健康检查
        health = await self.check_az_health(target_az)
        
        # 验证标准
        if health.health_score >= 0.8:
            logger.info(f"Validation passed: health_score={health.health_score:.2%}")
            return True
        else:
            logger.error(f"Validation failed: health_score={health.health_score:.2%}")
            return False
    
    async def rollback_failover(self):
        """回滚切换"""
        logger.info("Rolling back failover")
        self.state = FailoverState.ROLLING_BACK
        
        try:
            # 1. 恢复原始流量
            await self._switch_traffic(self.primary_az)
            
            # 2. 验证回滚
            validation = await self._validate_services(self.primary_az)
            
            if validation:
                self.state = FailoverState.NORMAL
                logger.info("Rollback completed successfully")
            else:
                self.state = FailoverState.FAILED
                logger.error("Rollback validation failed")
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.state = FailoverState.FAILED
    
    async def monitor_loop(self):
        """监控循环"""
        logger.info("Starting failover monitor loop")
        
        while True:
            try:
                # 检查主备AZ健康状态
                primary_health = await self.check_az_health(self.primary_az)
                secondary_health = await self.check_az_health(self.secondary_az)
                
                # 记录历史
                self.health_history.append(primary_health)
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # 更新失败计数
                if primary_health.status in [AZStatus.UNHEALTHY, AZStatus.DEGRADED]:
                    self.failure_count += 1
                else:
                    self.failure_count = 0
                
                # 做出切换决策
                if self.state == FailoverState.NORMAL:
                    decision = await self.make_failover_decision(
                        primary_health,
                        secondary_health
                    )
                    
                    if decision.should_failover and decision.confidence >= 0.7:
                        logger.warning(f"Failover decision: {decision.reason} (confidence: {decision.confidence:.2%})")
                        await self.execute_failover(decision)
                
                # 记录监控指标
                logger.debug(f"Health check - Primary: {primary_health.health_score:.2%}, Secondary: {secondary_health.health_score:.2%}")
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            await asyncio.sleep(self.health_check_interval)
    
    def get_status(self) -> Dict:
        """
        获取控制器状态
        
        Returns:
            状态信息
        """
        return {
            "state": self.state.value,
            "current_az": self.current_az,
            "primary_az": self.primary_az,
            "secondary_az": self.secondary_az,
            "failure_count": self.failure_count,
            "last_failover": self.last_failover.isoformat() if self.last_failover else None,
            "health_history_size": len(self.health_history)
        }
    
    async def manual_failover(self, target_az: str, reason: str = "Manual failover") -> bool:
        """
        手动故障切换
        
        Args:
            target_az: 目标AZ
            reason: 切换原因
            
        Returns:
            是否成功
        """
        logger.info(f"Manual failover requested to {target_az}: {reason}")
        
        # 创建手动切换决策
        decision = FailoverDecision(
            should_failover=True,
            reason=reason,
            primary_az=self.current_az,
            target_az=target_az,
            confidence=1.0
        )
        
        return await self.execute_failover(decision)


async def main():
    """示例用法"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建控制器
    controller = FailoverController(
        primary_az="az-a",
        secondary_az="az-b",
        health_check_interval=30,
        failure_threshold=3
    )
    
    # 测试健康检查
    primary_health = await controller.check_az_health("az-a")
    print(f"Primary AZ Health: {primary_health.status.value}, Score: {primary_health.health_score:.2%}")
    
    secondary_health = await controller.check_az_health("az-b")
    print(f"Secondary AZ Health: {secondary_health.status.value}, Score: {secondary_health.health_score:.2%}")
    
    # 测试决策
    decision = await controller.make_failover_decision(primary_health, secondary_health)
    print(f"Failover Decision: {decision.should_failover}, Reason: {decision.reason}")
    
    # 启动监控（注释掉避免无限循环）
    # await controller.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())