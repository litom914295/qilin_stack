"""
异常行为检测器（P0-5.2）
检测并告警异常审计行为
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import asyncio

from security.audit_enhanced import AuditEventType
from monitoring.audit_metrics import get_audit_metrics

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRule:
    """异常检测规则"""
    rule_id: str
    name: str
    description: str
    severity: str  # low/medium/high/critical
    threshold: Any
    time_window: int  # 时间窗口（秒）
    enabled: bool = True


@dataclass
class AnomalyAlert:
    """异常告警"""
    alert_id: str
    rule_id: str
    timestamp: datetime
    severity: str
    title: str
    description: str
    user_id: Optional[str]
    metadata: Dict[str, Any]
    resolved: bool = False


class AnomalyDetector:
    """异常行为检测器"""
    
    def __init__(
        self,
        alert_callback: Optional[callable] = None
    ):
        """
        初始化异常检测器
        
        Args:
            alert_callback: 告警回调函数
        """
        self.alert_callback = alert_callback
        self.metrics = get_audit_metrics()
        
        # 事件缓存（用于时间窗口检测）
        self.event_cache: Dict[str, List[Dict]] = defaultdict(list)
        
        # 活跃告警
        self.active_alerts: Dict[str, AnomalyAlert] = {}
        
        # 初始化规则
        self.rules = self._init_rules()
        
        logger.info(f"Anomaly detector initialized with {len(self.rules)} rules")
    
    def _init_rules(self) -> Dict[str, AnomalyRule]:
        """初始化检测规则"""
        rules = {
            # 登录失败规则
            "login_failure": AnomalyRule(
                rule_id="login_failure",
                name="频繁登录失败",
                description="5分钟内登录失败超过5次",
                severity="high",
                threshold=5,
                time_window=300  # 5分钟
            ),
            
            # 批量数据导出
            "bulk_export": AnomalyRule(
                rule_id="bulk_export",
                name="批量数据导出",
                description="单次导出记录数超过10000",
                severity="medium",
                threshold=10000,
                time_window=0
            ),
            
            # 非工作时间访问
            "off_hours_access": AnomalyRule(
                rule_id="off_hours_access",
                name="非工作时间访问",
                description="非工作时间（22:00-06:00）进行敏感操作",
                severity="medium",
                threshold=None,
                time_window=0
            ),
            
            # 频繁API调用
            "api_rate_limit": AnomalyRule(
                rule_id="api_rate_limit",
                name="API调用频率过高",
                description="1分钟内API调用超过100次",
                severity="medium",
                threshold=100,
                time_window=60  # 1分钟
            ),
            
            # 越权访问尝试
            "unauthorized_access": AnomalyRule(
                rule_id="unauthorized_access",
                name="越权访问尝试",
                description="尝试访问未授权资源",
                severity="high",
                threshold=3,
                time_window=300  # 5分钟
            ),
            
            # 异常IP访问
            "suspicious_ip": AnomalyRule(
                rule_id="suspicious_ip",
                name="可疑IP访问",
                description="非白名单IP访问敏感资源",
                severity="high",
                threshold=None,
                time_window=0
            ),
            
            # 数据导出频率
            "export_frequency": AnomalyRule(
                rule_id="export_frequency",
                name="频繁数据导出",
                description="1小时内导出操作超过10次",
                severity="medium",
                threshold=10,
                time_window=3600  # 1小时
            ),
            
            # 配置变更
            "config_change": AnomalyRule(
                rule_id="config_change",
                name="配置变更检测",
                description="系统配置发生变更",
                severity="high",
                threshold=None,
                time_window=0
            ),
            
            # 同一用户多地点登录
            "multi_location_login": AnomalyRule(
                rule_id="multi_location_login",
                name="异地登录",
                description="短时间内从不同地点登录",
                severity="high",
                threshold=2,
                time_window=600  # 10分钟
            ),
        }
        
        return rules
    
    async def analyze_event(
        self,
        event_type: str,
        user_id: str,
        action: str,
        result: str,
        ip_address: str,
        metadata: Dict[str, Any]
    ) -> List[AnomalyAlert]:
        """
        分析审计事件，检测异常
        
        Args:
            event_type: 事件类型
            user_id: 用户ID
            action: 操作
            result: 结果
            ip_address: IP地址
            metadata: 元数据
            
        Returns:
            检测到的异常告警列表
        """
        alerts = []
        
        event = {
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "result": result,
            "ip_address": ip_address,
            "metadata": metadata,
            "timestamp": datetime.now()
        }
        
        # 登录失败检测
        if event_type == AuditEventType.USER_LOGIN.value and result == "failure":
            alert = await self._check_login_failures(user_id, event)
            if alert:
                alerts.append(alert)
        
        # 批量导出检测
        if event_type == AuditEventType.DATA_EXPORT.value:
            alert = await self._check_bulk_export(user_id, event)
            if alert:
                alerts.append(alert)
            
            # 导出频率检测
            alert = await self._check_export_frequency(user_id, event)
            if alert:
                alerts.append(alert)
        
        # 非工作时间检测
        if event_type in [
            AuditEventType.DATA_EXPORT.value,
            AuditEventType.CONFIG_CHANGE.value
        ]:
            alert = await self._check_off_hours_access(user_id, event)
            if alert:
                alerts.append(alert)
        
        # API频率检测
        if event_type == AuditEventType.API_CALL.value:
            alert = await self._check_api_rate_limit(user_id, event)
            if alert:
                alerts.append(alert)
        
        # 越权访问检测
        if result == "failure" and "unauthorized" in metadata.get("error", "").lower():
            alert = await self._check_unauthorized_access(user_id, event)
            if alert:
                alerts.append(alert)
        
        # 配置变更检测
        if event_type == AuditEventType.CONFIG_CHANGE.value:
            alert = await self._check_config_change(user_id, event)
            if alert:
                alerts.append(alert)
        
        # 处理告警
        for alert in alerts:
            await self._handle_alert(alert)
        
        return alerts
    
    async def _check_login_failures(
        self,
        user_id: str,
        event: Dict
    ) -> Optional[AnomalyAlert]:
        """检测登录失败"""
        rule = self.rules["login_failure"]
        if not rule.enabled:
            return None
        
        cache_key = f"login_failure_{user_id}"
        self._add_to_cache(cache_key, event, rule.time_window)
        
        events = self._get_from_cache(cache_key, rule.time_window)
        
        if len(events) >= rule.threshold:
            return AnomalyAlert(
                alert_id=f"{rule.rule_id}_{user_id}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                severity=rule.severity,
                title=rule.name,
                description=f"用户 {user_id} 在 {rule.time_window}秒内登录失败{len(events)}次",
                user_id=user_id,
                metadata={"failure_count": len(events), "last_ip": event["ip_address"]}
            )
        
        return None
    
    async def _check_bulk_export(
        self,
        user_id: str,
        event: Dict
    ) -> Optional[AnomalyAlert]:
        """检测批量导出"""
        rule = self.rules["bulk_export"]
        if not rule.enabled:
            return None
        
        records = event["metadata"].get("records", 0)
        
        if records > rule.threshold:
            return AnomalyAlert(
                alert_id=f"{rule.rule_id}_{user_id}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                severity=rule.severity,
                title=rule.name,
                description=f"用户 {user_id} 单次导出 {records} 条记录",
                user_id=user_id,
                metadata={"records": records, "export_type": event["metadata"].get("export_type")}
            )
        
        return None
    
    async def _check_off_hours_access(
        self,
        user_id: str,
        event: Dict
    ) -> Optional[AnomalyAlert]:
        """检测非工作时间访问"""
        rule = self.rules["off_hours_access"]
        if not rule.enabled:
            return None
        
        current_hour = datetime.now().hour
        
        # 非工作时间：22:00-06:00
        if current_hour >= 22 or current_hour < 6:
            return AnomalyAlert(
                alert_id=f"{rule.rule_id}_{user_id}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                severity=rule.severity,
                title=rule.name,
                description=f"用户 {user_id} 在非工作时间（{current_hour}:00）进行了 {event['action']} 操作",
                user_id=user_id,
                metadata={"hour": current_hour, "action": event["action"]}
            )
        
        return None
    
    async def _check_api_rate_limit(
        self,
        user_id: str,
        event: Dict
    ) -> Optional[AnomalyAlert]:
        """检测API调用频率"""
        rule = self.rules["api_rate_limit"]
        if not rule.enabled:
            return None
        
        cache_key = f"api_calls_{user_id}"
        self._add_to_cache(cache_key, event, rule.time_window)
        
        events = self._get_from_cache(cache_key, rule.time_window)
        
        if len(events) >= rule.threshold:
            return AnomalyAlert(
                alert_id=f"{rule.rule_id}_{user_id}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                severity=rule.severity,
                title=rule.name,
                description=f"用户 {user_id} 在 {rule.time_window}秒内调用API {len(events)}次",
                user_id=user_id,
                metadata={"call_count": len(events)}
            )
        
        return None
    
    async def _check_unauthorized_access(
        self,
        user_id: str,
        event: Dict
    ) -> Optional[AnomalyAlert]:
        """检测越权访问"""
        rule = self.rules["unauthorized_access"]
        if not rule.enabled:
            return None
        
        cache_key = f"unauthorized_{user_id}"
        self._add_to_cache(cache_key, event, rule.time_window)
        
        events = self._get_from_cache(cache_key, rule.time_window)
        
        if len(events) >= rule.threshold:
            return AnomalyAlert(
                alert_id=f"{rule.rule_id}_{user_id}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                severity=rule.severity,
                title=rule.name,
                description=f"用户 {user_id} 在 {rule.time_window}秒内尝试越权访问{len(events)}次",
                user_id=user_id,
                metadata={"attempt_count": len(events)}
            )
        
        return None
    
    async def _check_export_frequency(
        self,
        user_id: str,
        event: Dict
    ) -> Optional[AnomalyAlert]:
        """检测导出频率"""
        rule = self.rules["export_frequency"]
        if not rule.enabled:
            return None
        
        cache_key = f"exports_{user_id}"
        self._add_to_cache(cache_key, event, rule.time_window)
        
        events = self._get_from_cache(cache_key, rule.time_window)
        
        if len(events) >= rule.threshold:
            return AnomalyAlert(
                alert_id=f"{rule.rule_id}_{user_id}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                severity=rule.severity,
                title=rule.name,
                description=f"用户 {user_id} 在 {rule.time_window}秒内导出数据{len(events)}次",
                user_id=user_id,
                metadata={"export_count": len(events)}
            )
        
        return None
    
    async def _check_config_change(
        self,
        user_id: str,
        event: Dict
    ) -> Optional[AnomalyAlert]:
        """检测配置变更"""
        rule = self.rules["config_change"]
        if not rule.enabled:
            return None
        
        return AnomalyAlert(
            alert_id=f"{rule.rule_id}_{user_id}_{int(datetime.now().timestamp())}",
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            severity=rule.severity,
            title=rule.name,
            description=f"用户 {user_id} 修改了系统配置",
            user_id=user_id,
            metadata=event["metadata"]
        )
    
    def _add_to_cache(self, key: str, event: Dict, ttl: int):
        """添加事件到缓存"""
        self.event_cache[key].append(event)
        
        # 清理过期事件
        if ttl > 0:
            cutoff_time = datetime.now() - timedelta(seconds=ttl)
            self.event_cache[key] = [
                e for e in self.event_cache[key]
                if e["timestamp"] > cutoff_time
            ]
    
    def _get_from_cache(self, key: str, ttl: int) -> List[Dict]:
        """从缓存获取事件"""
        if ttl > 0:
            cutoff_time = datetime.now() - timedelta(seconds=ttl)
            return [
                e for e in self.event_cache.get(key, [])
                if e["timestamp"] > cutoff_time
            ]
        return self.event_cache.get(key, [])
    
    async def _handle_alert(self, alert: AnomalyAlert):
        """处理告警"""
        # 记录到Prometheus
        self.metrics.record_security_event(
            event_subtype=alert.rule_id,
            severity=alert.severity
        )
        
        # 记录到日志
        logger.warning(
            f"Anomaly detected: {alert.title} | "
            f"User: {alert.user_id} | "
            f"Severity: {alert.severity} | "
            f"{alert.description}"
        )
        
        # 保存到活跃告警
        self.active_alerts[alert.alert_id] = alert
        
        # 调用回调
        if self.alert_callback:
            try:
                await self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_active_alerts(
        self,
        severity: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[AnomalyAlert]:
        """
        获取活跃告警
        
        Args:
            severity: 严重程度过滤
            user_id: 用户ID过滤
            
        Returns:
            告警列表
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]
        
        return alerts
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"Alert {alert_id} resolved")


# 全局单例
_anomaly_detector = None


def get_anomaly_detector(alert_callback: Optional[callable] = None) -> AnomalyDetector:
    """获取异常检测器单例"""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector(alert_callback=alert_callback)
    return _anomaly_detector
