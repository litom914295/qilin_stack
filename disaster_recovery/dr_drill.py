"""
容灾演练脚本（P0-15.4）
模拟各种故障场景，验证RTO/RPO
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json
import time

from disaster_recovery.failover_controller import FailoverController, AZStatus
from disaster_recovery.data_sync_manager import DataSyncManager, DataSource

logger = logging.getLogger(__name__)


class DrillScenario(str, Enum):
    """演练场景"""
    NETWORK_PARTITION = "network_partition"
    PRIMARY_DB_FAILURE = "primary_db_failure"
    COMPLETE_AZ_FAILURE = "complete_az_failure"
    PARTIAL_SERVICE_FAILURE = "partial_service_failure"
    DATA_CORRUPTION = "data_corruption"
    CASCADING_FAILURE = "cascading_failure"


@dataclass
class DrillResult:
    """演练结果"""
    scenario: DrillScenario
    start_time: datetime
    end_time: datetime
    rto_seconds: float
    rpo_seconds: float
    data_loss: bool
    service_available: bool
    success: bool
    errors: List[str]
    metrics: Dict[str, Any]


class DisasterRecoveryDrill:
    """容灾演练"""
    
    def __init__(
        self,
        primary_az: str = "az-a",
        secondary_az: str = "az-b"
    ):
        """
        初始化演练
        
        Args:
            primary_az: 主可用区
            secondary_az: 备可用区
        """
        self.primary_az = primary_az
        self.secondary_az = secondary_az
        
        self.failover_controller = FailoverController(
            primary_az=primary_az,
            secondary_az=secondary_az
        
        self.sync_manager = DataSyncManager(
            primary_az=primary_az,
            secondary_az=secondary_az
        
        self.drill_results = []
        
        logger.info(f"DR drill initialized: primary={primary_az}, secondary={secondary_az}")
    
    async def simulate_network_partition(self) -> Dict:
        """
        模拟网络分区
        
        Returns:
            故障影响
        """
        logger.info("🔥 Simulating network partition between AZs")
        
        impacts = {
            "affected_services": ["all"],
            "connectivity": False,
            "data_sync": "interrupted",
            "expected_behavior": "failover_to_secondary"
        }
        
        # 模拟网络中断
        await asyncio.sleep(1)
        
        # 设置主AZ不可达
        self.failover_controller.failure_count = 5
        
        return impacts
    
    async def simulate_database_failure(self) -> Dict:
        """
        模拟数据库故障
        
        Returns:
            故障影响
        """
        logger.info("🔥 Simulating primary database failure")
        
        impacts = {
            "affected_services": ["postgresql"],
            "data_availability": False,
            "write_operations": "blocked",
            "expected_behavior": "promote_standby"
        }
        
        # 模拟数据库宕机
        await asyncio.sleep(1)
        
        return impacts
    
    async def simulate_complete_az_failure(self) -> Dict:
        """
        模拟完整AZ故障
        
        Returns:
            故障影响
        """
        logger.info("🔥 Simulating complete AZ failure")
        
        impacts = {
            "affected_services": ["all"],
            "az_status": "offline",
            "data_availability": False,
            "expected_behavior": "full_failover"
        }
        
        # 模拟AZ完全不可用
        await asyncio.sleep(2)
        
        return impacts
    
    async def simulate_partial_service_failure(self, services: List[str]) -> Dict:
        """
        模拟部分服务故障
        
        Args:
            services: 故障服务列表
            
        Returns:
            故障影响
        """
        logger.info(f"🔥 Simulating partial service failure: {services}")
        
        impacts = {
            "affected_services": services,
            "degraded_performance": True,
            "partial_availability": True,
            "expected_behavior": "service_degradation"
        }
        
        # 模拟服务故障
        await asyncio.sleep(1)
        
        return impacts
    
    async def simulate_data_corruption(self) -> Dict:
        """
        模拟数据损坏
        
        Returns:
            故障影响
        """
        logger.info("🔥 Simulating data corruption")
        
        impacts = {
            "affected_data": ["trades", "positions"],
            "data_integrity": False,
            "recovery_required": True,
            "expected_behavior": "restore_from_backup"
        }
        
        # 模拟数据损坏
        await asyncio.sleep(1)
        
        return impacts
    
    async def inject_failure(self, scenario: DrillScenario) -> Dict:
        """
        注入故障
        
        Args:
            scenario: 故障场景
            
        Returns:
            故障影响
        """
        logger.info(f"💉 Injecting failure: {scenario.value}")
        
        if scenario == DrillScenario.NETWORK_PARTITION:
            return await self.simulate_network_partition()
        elif scenario == DrillScenario.PRIMARY_DB_FAILURE:
            return await self.simulate_database_failure()
        elif scenario == DrillScenario.COMPLETE_AZ_FAILURE:
            return await self.simulate_complete_az_failure()
        elif scenario == DrillScenario.PARTIAL_SERVICE_FAILURE:
            return await self.simulate_partial_service_failure(["redis", "kafka"])
        elif scenario == DrillScenario.DATA_CORRUPTION:
            return await self.simulate_data_corruption()
        else:
            return {}
    
    async def measure_recovery_time(self, start_time: datetime) -> float:
        """
        测量恢复时间
        
        Args:
            start_time: 故障开始时间
            
        Returns:
            RTO（秒）
        """
        # 等待服务恢复
        max_wait = 300  # 5分钟超时
        check_interval = 5
        
        elapsed = 0
        while elapsed < max_wait:
            # 检查服务状态
            secondary_health = await self.failover_controller.check_az_health(self.secondary_az)
            
            if secondary_health.status == AZStatus.HEALTHY:
                recovery_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Service recovered in {recovery_time:.1f} seconds")
                return recovery_time
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
        logger.error("❌ Recovery timeout")
        return -1
    
    async def measure_data_loss(self) -> float:
        """
        测量数据丢失
        
        Returns:
            RPO（秒）
        """
        # 检查各数据源的复制延迟
        total_lag = 0
        
        for source in DataSource:
            metrics = await self.sync_manager.get_sync_metrics(source)
            total_lag = max(total_lag, metrics.lag_seconds)
        
        logger.info(f"📊 Maximum data lag: {total_lag:.1f} seconds")
        return total_lag
    
    async def validate_service_availability(self) -> bool:
        """
        验证服务可用性
        
        Returns:
            是否可用
        """
        # 检查关键服务
        critical_services = ["api", "database", "redis", "kafka"]
        
        for service in critical_services:
            endpoint = f"http://{self.secondary_az}.{service}.qilin:8000/health"
            result = await self.failover_controller.check_service_health(service, endpoint)
            
            if not result.status:
                logger.error(f"❌ Service {service} is not available")
                return False
        
        logger.info("✅ All critical services are available")
        return True
    
    async def validate_data_consistency(self) -> bool:
        """
        验证数据一致性
        
        Returns:
            是否一致
        """
        consistency = await self.sync_manager.validate_data_consistency()
        
        for source, is_consistent in consistency.items():
            if not is_consistent:
                logger.error(f"❌ Data inconsistency detected in {source}")
                return False
        
        logger.info("✅ Data consistency validated")
        return True
    
    async def run_scenario(self, scenario: DrillScenario) -> DrillResult:
        """
        运行演练场景
        
        Args:
            scenario: 演练场景
            
        Returns:
            演练结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎭 Starting drill scenario: {scenario.value}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        errors = []
        
        try:
            # 1. 记录初始状态
            logger.info("📸 Recording initial state")
            initial_sync_status = self.sync_manager.get_sync_status()
            
            # 2. 注入故障
            failure_impact = await self.inject_failure(scenario)
            failure_time = datetime.now()
            
            # 3. 等待故障检测
            logger.info("🔍 Waiting for failure detection")
            await asyncio.sleep(10)
            
            # 4. 检查是否触发切换
            primary_health = await self.failover_controller.check_az_health(self.primary_az)
            secondary_health = await self.failover_controller.check_az_health(self.secondary_az)
            
            decision = await self.failover_controller.make_failover_decision(
                primary_health,
                secondary_health
            
            if decision.should_failover:
                logger.info(f"🔄 Triggering failover: {decision.reason}")
                
                # 5. 执行故障切换
                failover_success = await self.failover_controller.execute_failover(decision)
                
                if not failover_success:
                    errors.append("Failover execution failed")
            else:
                logger.info("⚠️ No failover triggered")
            
            # 6. 测量恢复时间
            rto_seconds = await self.measure_recovery_time(failure_time)
            
            # 7. 测量数据丢失
            rpo_seconds = await self.measure_data_loss()
            
            # 8. 验证服务可用性
            service_available = await self.validate_service_availability()
            
            # 9. 验证数据一致性
            data_consistent = await self.validate_data_consistency()
            
            end_time = datetime.now()
            
            # 10. 收集指标
            metrics = {
                "initial_sync_status": initial_sync_status,
                "failure_impact": failure_impact,
                "failover_decision": {
                    "should_failover": decision.should_failover,
                    "confidence": decision.confidence,
                    "reason": decision.reason
                },
                "primary_health": primary_health.health_score,
                "secondary_health": secondary_health.health_score
            }
            
            success = (
                rto_seconds > 0 and
                rto_seconds <= 300 and  # 5分钟内恢复
                rpo_seconds <= 60 and   # 数据丢失少于1分钟
                service_available and
                data_consistent
            
            result = DrillResult(
                scenario=scenario,
                start_time=start_time,
                end_time=end_time,
                rto_seconds=rto_seconds,
                rpo_seconds=rpo_seconds,
                data_loss=rpo_seconds > 0,
                service_available=service_available,
                success=success,
                errors=errors,
                metrics=metrics
            
            self.drill_results.append(result)
            
            # 打印结果摘要
            self._print_result_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Drill scenario failed: {e}")
            errors.append(str(e))
            
            return DrillResult(
                scenario=scenario,
                start_time=start_time,
                end_time=datetime.now(),
                rto_seconds=-1,
                rpo_seconds=-1,
                data_loss=True,
                service_available=False,
                success=False,
                errors=errors,
                metrics={}
    
    def _print_result_summary(self, result: DrillResult):
        """打印结果摘要"""
        logger.info(f"\n{'='*60}")
        logger.info("📋 DRILL RESULT SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Scenario: {result.scenario.value}")
        logger.info(f"Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
        logger.info(f"RTO: {result.rto_seconds:.1f}s {'✅' if result.rto_seconds <= 300 else '❌'}")
        logger.info(f"RPO: {result.rpo_seconds:.1f}s {'✅' if result.rpo_seconds <= 60 else '❌'}")
        logger.info(f"Service Available: {'✅' if result.service_available else '❌'}")
        logger.info(f"Data Loss: {'❌ Yes' if result.data_loss else '✅ No'}")
        logger.info(f"Overall: {'✅ PASSED' if result.success else '❌ FAILED'}")
        
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        
        logger.info(f"{'='*60}\n")
    
    async def run_all_scenarios(self) -> Dict:
        """
        运行所有演练场景
        
        Returns:
            演练报告
        """
        logger.info("🚀 Starting comprehensive DR drill")
        
        scenarios = [
            DrillScenario.NETWORK_PARTITION,
            DrillScenario.PRIMARY_DB_FAILURE,
            DrillScenario.COMPLETE_AZ_FAILURE,
            DrillScenario.PARTIAL_SERVICE_FAILURE,
            DrillScenario.DATA_CORRUPTION
        ]
        
        for scenario in scenarios:
            await self.run_scenario(scenario)
            
            # 恢复到初始状态
            logger.info("🔧 Restoring to initial state")
            await self.restore_initial_state()
            await asyncio.sleep(10)
        
        return self.generate_report()
    
    async def restore_initial_state(self):
        """恢复初始状态"""
        # 切换回主AZ
        if self.failover_controller.current_az != self.primary_az:
            await self.failover_controller.manual_failover(
                self.primary_az,
                "Restore to primary after drill"
        
        # 重置故障计数
        self.failover_controller.failure_count = 0
        
        # 等待数据同步
        await self.sync_manager.wait_for_sync(max_lag_seconds=5, timeout=30)
    
    def generate_report(self) -> Dict:
        """
        生成演练报告
        
        Returns:
            演练报告
        """
        successful_scenarios = sum(1 for r in self.drill_results if r.success)
        total_scenarios = len(self.drill_results)
        
        avg_rto = sum(r.rto_seconds for r in self.drill_results if r.rto_seconds > 0) / total_scenarios
        avg_rpo = sum(r.rpo_seconds for r in self.drill_results if r.rpo_seconds > 0) / total_scenarios
        
        report = {
            "drill_date": datetime.now().isoformat(),
            "primary_az": self.primary_az,
            "secondary_az": self.secondary_az,
            "summary": {
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "success_rate": successful_scenarios / total_scenarios * 100,
                "average_rto": avg_rto,
                "average_rpo": avg_rpo,
                "meets_sla": avg_rto <= 300 and avg_rpo <= 60
            },
            "scenarios": [
                {
                    "scenario": r.scenario.value,
                    "success": r.success,
                    "rto": r.rto_seconds,
                    "rpo": r.rpo_seconds,
                    "errors": r.errors
                }
                for r in self.drill_results
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        for result in self.drill_results:
            if not result.success:
                if result.rto_seconds > 300:
                    recommendations.append(f"Improve failover speed for {result.scenario.value}")
                if result.rpo_seconds > 60:
                    recommendations.append(f"Reduce replication lag for {result.scenario.value}")
                if not result.service_available:
                    recommendations.append(f"Ensure service availability after {result.scenario.value}")
        
        # 去重
        recommendations = list(set(recommendations))
        
        if not recommendations:
            recommendations.append("All scenarios passed. Consider more complex failure scenarios.")
        
        return recommendations
    
    def save_report(self, filepath: str = "dr_drill_report.json"):
        """
        保存报告到文件
        
        Args:
            filepath: 文件路径
        """
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Report saved to {filepath}")


async def main():
    """示例用法"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建演练实例
    drill = DisasterRecoveryDrill(
        primary_az="az-a",
        secondary_az="az-b"
    
    # 运行单个场景
    result = await drill.run_scenario(DrillScenario.NETWORK_PARTITION)
    print(f"Scenario result: {result.success}")
    
    # 运行所有场景
    # report = await drill.run_all_scenarios()
    # print(json.dumps(report, indent=2, default=str))
    
    # 保存报告
    # drill.save_report()


if __name__ == "__main__":
    asyncio.run(main())