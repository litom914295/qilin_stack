"""
数据降级管理器
实现数据质量<0.8时的自动降级机制、备用数据源切换、数据补数
"""

import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """降级级别"""
    NORMAL = "normal"           # 正常运行
    WARNING = "warning"         # 警告状态，使用缓存
    DEGRADED = "degraded"       # 降级状态，使用备用源
    EMERGENCY = "emergency"     # 紧急状态，停止服务


class DataSourceType(Enum):
    """数据源类型"""
    PRIMARY = "primary"         # 主数据源
    BACKUP = "backup"           # 备用数据源
    CACHE = "cache"             # 缓存数据


@dataclass
class DegradationEvent:
    """降级事件"""
    event_id: str
    data_source: str
    quality_score: float
    threshold: float
    level: DegradationLevel
    timestamp: datetime
    trigger_reason: str
    recovery_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    type: DataSourceType
    priority: int  # 优先级，数字越小优先级越高
    enabled: bool = True
    health_check_url: Optional[str] = None
    timeout_seconds: int = 30
    retry_count: int = 3
    circuit_breaker_threshold: int = 5  # 连续失败次数阈值


class DataDegradationManager:
    """数据降级管理器"""
    
    def __init__(self):
        self.data_sources: Dict[str, List[DataSourceConfig]] = defaultdict(list)
        self.current_sources: Dict[str, DataSourceConfig] = {}
        self.degradation_history: List[DegradationEvent] = []
        self.circuit_breakers: Dict[str, int] = defaultdict(int)  # 失败计数
        self.quality_threshold = 0.8
        self.callbacks: List[Callable] = []
        
    def register_data_source(
        self,
        data_type: str,
        config: DataSourceConfig
    ):
        """
        注册数据源
        
        Args:
            data_type: 数据类型（market/capital/news/longhu）
            config: 数据源配置
        """
        self.data_sources[data_type].append(config)
        # 按优先级排序
        self.data_sources[data_type].sort(key=lambda x: x.priority)
        
        # 设置当前活跃数据源（如果没有）
        if data_type not in self.current_sources:
            self.current_sources[data_type] = config
            
        logger.info(f"Registered data source: {config.name} for {data_type}")
    
    def evaluate_quality(
        self,
        data_type: str,
        quality_score: float
    ) -> DegradationLevel:
        """
        评估数据质量并确定降级级别
        
        Args:
            data_type: 数据类型
            quality_score: 质量分数
            
        Returns:
            降级级别
        """
        if quality_score >= self.quality_threshold:
            return DegradationLevel.NORMAL
        elif quality_score >= 0.5:
            return DegradationLevel.WARNING
        elif quality_score >= 0.3:
            return DegradationLevel.DEGRADED
        else:
            return DegradationLevel.EMERGENCY
    
    def handle_degradation(
        self,
        data_type: str,
        quality_score: float,
        reason: str = ""
    ) -> DegradationEvent:
        """
        处理降级事件
        
        Args:
            data_type: 数据类型
            quality_score: 质量分数
            reason: 降级原因
            
        Returns:
            降级事件
        """
        level = self.evaluate_quality(data_type, quality_score)
        
        # 创建降级事件
        event = DegradationEvent(
            event_id=self._generate_event_id(),
            data_source=data_type,
            quality_score=quality_score,
            threshold=self.quality_threshold,
            level=level,
            timestamp=datetime.now(),
            trigger_reason=reason or f"Quality score {quality_score} below threshold {self.quality_threshold}"
        
        # 执行降级策略
        if level == DegradationLevel.WARNING:
            event.recovery_actions = self._handle_warning(data_type, event)
        elif level == DegradationLevel.DEGRADED:
            event.recovery_actions = self._handle_degraded(data_type, event)
        elif level == DegradationLevel.EMERGENCY:
            event.recovery_actions = self._handle_emergency(data_type, event)
        
        # 记录事件
        self.degradation_history.append(event)
        
        # 触发回调
        self._trigger_callbacks(event)
        
        logger.critical(f"Degradation event: {json.dumps(event.__dict__, default=str)}")
        
        return event
    
    def _handle_warning(
        self,
        data_type: str,
        event: DegradationEvent
    ) -> List[str]:
        """处理WARNING级别降级"""
        actions = []
        
        # 1. 使用缓存数据
        actions.append(f"Using cached data for {data_type}")
        logger.warning(f"Quality warning for {data_type}, using cache")
        
        # 2. 增加数据源健康检查频率
        actions.append("Increasing health check frequency")
        
        # 3. 发送告警通知
        actions.append("Sending alert notification")
        
        return actions
    
    def _handle_degraded(
        self,
        data_type: str,
        event: DegradationEvent
    ) -> List[str]:
        """处理DEGRADED级别降级"""
        actions = []
        
        # 1. 切换到备用数据源
        backup_source = self._find_backup_source(data_type)
        if backup_source:
            self._switch_data_source(data_type, backup_source)
            actions.append(f"Switched to backup source: {backup_source.name}")
            logger.warning(f"Switched {data_type} to backup source: {backup_source.name}")
        else:
            actions.append("No backup source available, using cache")
            logger.error(f"No backup source for {data_type}, using cache")
        
        # 2. 熔断主数据源
        primary_source = self._find_primary_source(data_type)
        if primary_source:
            self.circuit_breakers[primary_source.name] += 1
            actions.append(f"Circuit breaker triggered for {primary_source.name}")
        
        # 3. 启动数据补数任务
        actions.append("Scheduling data backfill task")
        
        # 4. 发送紧急告警
        actions.append("Sending critical alert")
        
        return actions
    
    def _handle_emergency(
        self,
        data_type: str,
        event: DegradationEvent
    ) -> List[str]:
        """处理EMERGENCY级别降级"""
        actions = []
        
        # 1. 停止使用该数据源
        actions.append(f"Stopping {data_type} data source")
        logger.critical(f"Emergency: Stopping {data_type} data source")
        
        # 2. 熔断所有相关数据源
        for source in self.data_sources.get(data_type, []):
            self.circuit_breakers[source.name] = source.circuit_breaker_threshold
            actions.append(f"Circuit breaker opened for {source.name}")
        
        # 3. 触发人工介入
        actions.append("Manual intervention required")
        
        # 4. 发送紧急告警到PagerDuty
        actions.append("Paging on-call engineer")
        
        return actions
    
    def _find_backup_source(self, data_type: str) -> Optional[DataSourceConfig]:
        """查找备用数据源"""
        sources = self.data_sources.get(data_type, [])
        for source in sources:
            if (source.type == DataSourceType.BACKUP and 
                source.enabled and 
                self.circuit_breakers[source.name] < source.circuit_breaker_threshold):
                return source
        return None
    
    def _find_primary_source(self, data_type: str) -> Optional[DataSourceConfig]:
        """查找主数据源"""
        sources = self.data_sources.get(data_type, [])
        for source in sources:
            if source.type == DataSourceType.PRIMARY:
                return source
        return None
    
    def _switch_data_source(self, data_type: str, new_source: DataSourceConfig):
        """切换数据源"""
        old_source = self.current_sources.get(data_type)
        self.current_sources[data_type] = new_source
        
        logger.info(
            f"Data source switched for {data_type}: "
            f"{old_source.name if old_source else 'None'} -> {new_source.name}"
    
    def recover_data_source(self, data_type: str) -> bool:
        """
        尝试恢复数据源
        
        Args:
            data_type: 数据类型
            
        Returns:
            是否成功恢复
        """
        primary_source = self._find_primary_source(data_type)
        if not primary_source:
            return False
        
        # 检查熔断器状态
        if self.circuit_breakers[primary_source.name] >= primary_source.circuit_breaker_threshold:
            logger.info(f"Circuit breaker still open for {primary_source.name}")
            return False
        
        # 尝试切换回主数据源
        self._switch_data_source(data_type, primary_source)
        logger.info(f"Recovered primary source for {data_type}")
        
        return True
    
    def reset_circuit_breaker(self, source_name: str):
        """重置熔断器"""
        self.circuit_breakers[source_name] = 0
        logger.info(f"Circuit breaker reset for {source_name}")
    
    def register_callback(self, callback: Callable):
        """注册降级回调"""
        self.callbacks.append(callback)
    
    def _trigger_callbacks(self, event: DegradationEvent):
        """触发所有回调"""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback execution failed: {e}")
    
    def _generate_event_id(self) -> str:
        """生成事件ID"""
        import uuid
        return str(uuid.uuid4())
    
    def get_status(self) -> Dict[str, Any]:
        """获取降级管理器状态"""
        return {
            'current_sources': {
                data_type: source.name 
                for data_type, source in self.current_sources.items()
            },
            'circuit_breakers': dict(self.circuit_breakers),
            'recent_events': [
                {
                    'event_id': e.event_id,
                    'data_source': e.data_source,
                    'level': e.level.value,
                    'quality_score': e.quality_score,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in self.degradation_history[-10:]
            ]
        }


class DataBackfillManager:
    """数据补数管理器"""
    
    def __init__(self):
        self.backfill_queue: List[Dict] = []
        self.completed_backfills: List[Dict] = []
    
    def schedule_backfill(
        self,
        data_type: str,
        start_time: datetime,
        end_time: datetime,
        priority: int = 1
    ) -> str:
        """
        调度数据补数任务
        
        Args:
            data_type: 数据类型
            start_time: 开始时间
            end_time: 结束时间
            priority: 优先级
            
        Returns:
            任务ID
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        task = {
            'task_id': task_id,
            'data_type': data_type,
            'start_time': start_time,
            'end_time': end_time,
            'priority': priority,
            'status': 'pending',
            'created_at': datetime.now()
        }
        
        self.backfill_queue.append(task)
        # 按优先级排序
        self.backfill_queue.sort(key=lambda x: x['priority'])
        
        logger.info(f"Scheduled backfill task: {task_id} for {data_type}")
        return task_id
    
    def execute_backfill(self, task_id: str) -> bool:
        """
        执行补数任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功
        """
        task = next((t for t in self.backfill_queue if t['task_id'] == task_id), None)
        if not task:
            logger.error(f"Backfill task not found: {task_id}")
            return False
        
        try:
            logger.info(f"Executing backfill task: {task_id}")
            task['status'] = 'running'
            
            # 实际的补数逻辑在这里实现
            # 这里只是示例框架
            data_type = task['data_type']
            start_time = task['start_time']
            end_time = task['end_time']
            
            # 模拟数据获取
            logger.info(
                f"Backfilling {data_type} from {start_time} to {end_time}"
            
            # 标记完成
            task['status'] = 'completed'
            task['completed_at'] = datetime.now()
            self.backfill_queue.remove(task)
            self.completed_backfills.append(task)
            
            logger.info(f"Backfill task completed: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backfill task failed: {task_id} - {e}")
            task['status'] = 'failed'
            task['error'] = str(e)
            return False
    
    def get_pending_tasks(self) -> List[Dict]:
        """获取待处理任务"""
        return [t for t in self.backfill_queue if t['status'] == 'pending']
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        # 在队列中查找
        task = next((t for t in self.backfill_queue if t['task_id'] == task_id), None)
        if task:
            return task
        
        # 在已完成中查找
        task = next((t for t in self.completed_backfills if t['task_id'] == task_id), None)
        return task


# 示例使用
if __name__ == "__main__":
    # 初始化降级管理器
    manager = DataDegradationManager()
    
    # 注册数据源
    # 主数据源 - AkShare
    manager.register_data_source(
        "market",
        DataSourceConfig(
            name="akshare_primary",
            type=DataSourceType.PRIMARY,
            priority=1,
            health_check_url="https://akshare.example.com/health"
    
    # 备用数据源 - TuShare
    manager.register_data_source(
        "market",
        DataSourceConfig(
            name="tushare_backup",
            type=DataSourceType.BACKUP,
            priority=2,
            health_check_url="https://tushare.example.com/health"
    
    # 注册降级回调
    def on_degradation(event: DegradationEvent):
        print(f"Degradation callback: {event.data_source} - {event.level.value}")
    
    manager.register_callback(on_degradation)
    
    # 模拟质量下降
    event = manager.handle_degradation(
        data_type="market",
        quality_score=0.6,
        reason="Data completeness below threshold"
    
    print(f"Degradation Event:")
    print(f"  Level: {event.level.value}")
    print(f"  Actions: {event.recovery_actions}")
    
    # 获取状态
    status = manager.get_status()
    print(f"\nManager Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # 补数示例
    backfill_mgr = DataBackfillManager()
    task_id = backfill_mgr.schedule_backfill(
        data_type="market",
        start_time=datetime.now() - timedelta(hours=2),
        end_time=datetime.now(),
        priority=1
    print(f"\nScheduled backfill task: {task_id}")
