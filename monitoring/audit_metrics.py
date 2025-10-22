"""
审计指标Prometheus导出（P0-5.1）
导出审计事件统计指标到Prometheus
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AuditMetrics:
    """审计指标收集器"""
    
    def __init__(self):
        # 审计事件总数
        self.audit_events_total = Counter(
            'audit_events_total',
            'Total number of audit events',
            ['event_type', 'result', 'user_role']
        
        # PII检测事件数
        self.pii_detected_total = Counter(
            'audit_pii_detected_total',
            'Total number of events with PII detected',
            ['pii_type', 'masked']
        
        # 用户行为统计
        self.user_actions_total = Counter(
            'audit_user_actions_total',
            'Total user actions by type',
            ['user_id', 'action', 'result']
        
        # 失败事件统计
        self.audit_failures_total = Counter(
            'audit_failures_total',
            'Total number of failed audit events',
            ['event_type', 'user_role']
        
        # 数据导出统计
        self.data_exports_total = Counter(
            'audit_data_exports_total',
            'Total number of data export events',
            ['user_id', 'export_type', 'contains_pii']
        
        # 安全事件统计
        self.security_events_total = Counter(
            'audit_security_events_total',
            'Total number of security events',
            ['event_subtype', 'severity']
        
        # 当前活跃用户数
        self.active_users = Gauge(
            'audit_active_users',
            'Number of currently active users'
        
        # 审计事件处理延迟
        self.event_processing_duration = Histogram(
            'audit_event_processing_duration_seconds',
            'Time spent processing audit events',
            ['event_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        
        # 日志文件大小
        self.log_file_size_bytes = Gauge(
            'audit_log_file_size_bytes',
            'Current audit log file size in bytes',
            ['log_date']
        
        # PII脱敏成功率
        self.pii_masking_success_rate = Gauge(
            'audit_pii_masking_success_rate',
            'Success rate of PII masking operations'
        
        logger.info("Audit metrics initialized")
    
    def record_event(
        self,
        event_type: str,
        result: str,
        user_role: Optional[str] = None,
        duration: Optional[float] = None
    ):
        """
        记录审计事件
        
        Args:
            event_type: 事件类型
            result: 结果（success/failure/error）
            user_role: 用户角色
            duration: 处理时长（秒）
        """
        role = user_role or "unknown"
        self.audit_events_total.labels(
            event_type=event_type,
            result=result,
            user_role=role
        ).inc()
        
        if result in ["failure", "error"]:
            self.audit_failures_total.labels(
                event_type=event_type,
                user_role=role
            ).inc()
        
        if duration is not None:
            self.event_processing_duration.labels(
                event_type=event_type
            ).observe(duration)
    
    def record_pii_detection(
        self,
        pii_types: list,
        masked: bool = True
    ):
        """
        记录PII检测
        
        Args:
            pii_types: 检测到的PII类型列表
            masked: 是否已脱敏
        """
        for pii_type in pii_types:
            self.pii_detected_total.labels(
                pii_type=pii_type,
                masked=str(masked).lower()
            ).inc()
    
    def record_user_action(
        self,
        user_id: str,
        action: str,
        result: str
    ):
        """
        记录用户行为
        
        Args:
            user_id: 用户ID
            action: 操作
            result: 结果
        """
        self.user_actions_total.labels(
            user_id=user_id,
            action=action,
            result=result
        ).inc()
    
    def record_data_export(
        self,
        user_id: str,
        export_type: str,
        contains_pii: bool
    ):
        """
        记录数据导出
        
        Args:
            user_id: 用户ID
            export_type: 导出类型（csv/json/excel等）
            contains_pii: 是否包含PII
        """
        self.data_exports_total.labels(
            user_id=user_id,
            export_type=export_type,
            contains_pii=str(contains_pii).lower()
        ).inc()
    
    def record_security_event(
        self,
        event_subtype: str,
        severity: str = "medium"
    ):
        """
        记录安全事件
        
        Args:
            event_subtype: 事件子类型（login_failure/unauthorized_access等）
            severity: 严重程度（low/medium/high/critical）
        """
        self.security_events_total.labels(
            event_subtype=event_subtype,
            severity=severity
        ).inc()
    
    def update_active_users(self, count: int):
        """
        更新活跃用户数
        
        Args:
            count: 活跃用户数
        """
        self.active_users.set(count)
    
    def update_log_file_size(self, log_date: str, size_bytes: int):
        """
        更新日志文件大小
        
        Args:
            log_date: 日志日期（YYYYMMDD）
            size_bytes: 文件大小（字节）
        """
        self.log_file_size_bytes.labels(log_date=log_date).set(size_bytes)
    
    def update_masking_success_rate(self, rate: float):
        """
        更新PII脱敏成功率
        
        Args:
            rate: 成功率（0-1）
        """
        self.pii_masking_success_rate.set(rate)


# 全局单例
_audit_metrics = None


def get_audit_metrics() -> AuditMetrics:
    """获取审计指标收集器单例"""
    global _audit_metrics
    if _audit_metrics is None:
        _audit_metrics = AuditMetrics()
    return _audit_metrics
