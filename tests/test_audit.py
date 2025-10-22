"""
审计系统单元测试（P0-5.6）
测试PII脱敏、审计日志、异常检测
"""

import pytest
import asyncio
from datetime import datetime
from security.audit_enhanced import (
    PIIMasker, AuditLogger, AuditEventType, AuditEvent
from security.anomaly_detector import AnomalyDetector, AnomalyRule
from security.audit_report import AuditReportGenerator
)

class TestPIIMasker:
    """PII脱敏测试"""
    
    def setup_method(self):
        self.masker = PIIMasker()
    
    def test_mask_phone(self):
        """测试手机号脱敏"""
        phone = "13812345678"
        masked = self.masker.mask_phone(phone)
        assert masked == "138****5678"
        assert len(masked) == 11
    
    def test_mask_email(self):
        """测试邮箱脱敏"""
        email = "user@example.com"
        masked = self.masker.mask_email(email)
        assert masked.startswith("u")
        assert masked.endswith("r@example.com")
        assert "@" in masked
    
    def test_mask_id_card(self):
        """测试身份证脱敏"""
        id_card = "110101199001011234"
        masked = self.masker.mask_id_card(id_card)
        assert masked == "110***********1234"
        assert len(masked) == 18
    
    def test_mask_bank_card(self):
        """测试银行卡脱敏"""
        bank_card = "6222021234567890123"
        masked = self.masker.mask_bank_card(bank_card)
        assert masked.startswith("6222")
        assert masked.endswith("0123")
        assert "*" in masked
    
    def test_mask_ip_address(self):
        """测试IP地址脱敏"""
        ip = "192.168.1.100"
        masked = self.masker.mask_ip_address(ip)
        assert masked == "192.168.*.*"
    
    def test_detect_and_mask(self):
        """测试PII检测和脱敏"""

        masked_text, detected_types = self.masker.detect_and_mask(text)
        
        assert "138****5678" in masked_text
        assert "u***r@example.com" in masked_text
        assert len(detected_types) == 2
    
    def test_mask_dict(self):
        """测试字典脱敏"""
        data = {
            "user_id": "user123",
            "phone": "13812345678",
            "email": "test@example.com",
            "nested": {
                "bank_card": "6222021234567890123"
            }
        }
        
        masked = self.masker.mask_dict(data)
        
        assert masked["user_id"] == "user123"
        assert masked["phone"] == "138****5678"
        assert "t***t@example.com" in masked["email"]
        assert "6222" in masked["nested"]["bank_card"]

class TestAuditLogger:
    """审计日志测试"""
    
    def setup_method(self):
        self.logger = AuditLogger(
            log_dir="tests/logs/audit",
            enable_pii_masking=True,
            enable_console=False
    
    @pytest.mark.asyncio
    async def test_log_event(self):
        """测试记录审计事件"""
        event = await self.logger.log_event(
            event_type=AuditEventType.USER_LOGIN,
            user_id="test_user",
            action="login",
            resource="/api/auth/login",
            result="success",
            ip_address="192.168.1.100",
            metadata={
                "phone": "13812345678"
            }
        
        assert event.user_id == "test_user"
        assert event.event_type == AuditEventType.USER_LOGIN
        assert event.result == "success"
        assert event.pii_masked is True
    
    @pytest.mark.asyncio
    async def test_log_event_with_pii(self):
        """测试包含PII的审计事件"""
        event = await self.logger.log_event(
            event_type=AuditEventType.DATA_EXPORT,
            user_id="test_user",
            action="export",
            resource="/api/data/export",
            result="success",
            ip_address="192.168.1.100",
            metadata={
                "phone": "13812345678",
                "email": "test@example.com",
                "bank_card": "6222021234567890123"
            }
        
        assert event.pii_detected is True
        assert event.pii_masked is True
        assert "138****5678" in str(event.metadata)
    
    def test_query_events(self):
        """测试查询审计事件"""
        # 查询所有事件
        events = self.logger.query_events()
        assert isinstance(events, list)
        
        # 按用户ID查询
        events = self.logger.query_events(user_id="test_user")
        for event in events:
            assert event["user_id"] == "test_user"

class TestAnomalyDetector:
    """异常检测测试"""
    
    def setup_method(self):
        self.detector = AnomalyDetector()
    
    @pytest.mark.asyncio
    async def test_login_failure_detection(self):
        """测试登录失败检测"""
        # 模拟多次登录失败
        for i in range(6):
            alerts = await self.detector.analyze_event(
                event_type="user_login",
                user_id="test_user",
                action="login",
                result="failure",
                ip_address="192.168.1.100",
                metadata={}
        
        # 第6次应该触发告警
        assert len(alerts) > 0
        alert = alerts[0]
        assert alert.rule_id == "login_failure"
        assert alert.severity == "high"
    
    @pytest.mark.asyncio
    async def test_bulk_export_detection(self):
        """测试批量导出检测"""
        alerts = await self.detector.analyze_event(
            event_type="data_export",
            user_id="test_user",
            action="export",
            result="success",
            ip_address="192.168.1.100",
            metadata={
                "records": 15000,  # 超过阈值10000
                "export_type": "csv"
            }
        
        assert len(alerts) > 0
        alert = [a for a in alerts if a.rule_id == "bulk_export"][0]
        assert alert.severity == "medium"
    
    @pytest.mark.asyncio
    async def test_off_hours_access_detection(self):
        """测试非工作时间访问检测"""
        current_hour = datetime.now().hour
        
        # 仅在非工作时间测试
        if current_hour >= 22 or current_hour < 6:
            alerts = await self.detector.analyze_event(
                event_type="data_export",
                user_id="test_user",
                action="export",
                result="success",
                ip_address="192.168.1.100",
                metadata={}
            
            off_hours_alerts = [a for a in alerts if a.rule_id == "off_hours_access"]
            assert len(off_hours_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_detection(self):
        """测试API频率限制检测"""
        # 模拟大量API调用
        for i in range(101):
            alerts = await self.detector.analyze_event(
                event_type="api_call",
                user_id="test_user",
                action="GET",
                result="success",
                ip_address="192.168.1.100",
                metadata={}
        
        # 应该触发频率限制告警
        assert len(alerts) > 0
        rate_limit_alerts = [a for a in alerts if a.rule_id == "api_rate_limit"]
        assert len(rate_limit_alerts) > 0
    
    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        alerts = self.detector.get_active_alerts()
        assert isinstance(alerts, list)
        
        # 按严重程度过滤
        high_alerts = self.detector.get_active_alerts(severity="high")
        for alert in high_alerts:
            assert alert.severity == "high"

class TestAuditReportGenerator:
    """审计报告生成测试"""
    
    def setup_method(self):
        self.audit_logger = AuditLogger(
            log_dir="tests/logs/audit",
            enable_pii_masking=True
        self.generator = AuditReportGenerator(
            audit_logger=self.audit_logger,
            output_dir="tests/reports/audit"
    
    @pytest.mark.asyncio
    async def test_generate_daily_report(self):
        """测试生成日报"""
        # 创建测试事件
        await self.audit_logger.log_event(
            event_type=AuditEventType.API_CALL,
            user_id="test_user",
            action="GET",
            resource="/api/test",
            result="success",
            ip_address="192.168.1.100",
            metadata={}
        
        report = self.generator.generate_daily_report()
        
        assert report["report_type"] == "日报"
        assert "statistics" in report
        assert "events" in report
    
    @pytest.mark.asyncio
    async def test_generate_weekly_report(self):
        """测试生成周报"""
        report = self.generator.generate_weekly_report()
        
        assert report["report_type"] == "周报"
        assert "statistics" in report
    
    @pytest.mark.asyncio
    async def test_generate_monthly_report(self):
        """测试生成月报"""
        report = self.generator.generate_monthly_report()
        
        assert report["report_type"] == "月报"
        assert "statistics" in report
    
    @pytest.mark.asyncio
    async def test_export_html(self):
        """测试导出HTML报告"""
        report = self.generator.generate_daily_report()
        output_path = self.generator.export_html(report)
        
        assert output_path.endswith(".html")
        
        # 验证文件存在
        from pathlib import Path
        assert Path(output_path).exists()
    
    @pytest.mark.asyncio
    async def test_export_json(self):
        """测试导出JSON报告"""
        report = self.generator.generate_daily_report()
        output_path = self.generator.export_json(report)
        
        assert output_path.endswith(".json")
        
        # 验证文件存在
        from pathlib import Path
        assert Path(output_path).exists()
    
    def test_analyze_events(self):
        """测试事件分析"""
        events = [
            {
                "event_type": "user_login",
                "user_id": "user1",
                "action": "login",
                "result": "success",
                "timestamp": datetime.now().isoformat(),
                "pii_detected": False,
                "pii_masked": False
            },
            {
                "event_type": "data_export",
                "user_id": "user1",
                "action": "export",
                "result": "success",
                "timestamp": datetime.now().isoformat(),
                "pii_detected": True,
                "pii_masked": True
            },
            {
                "event_type": "api_call",
                "user_id": "user2",
                "action": "GET",
                "result": "failure",
                "timestamp": datetime.now().isoformat(),
                "pii_detected": False,
                "pii_masked": False
            }
        ]
        
        stats = self.generator._analyze_events(events)
        
        assert stats["total_events"] == 3
        assert stats["pii_detected"] == 1
        assert stats["pii_masked"] == 1
        assert stats["failed_count"] == 1
        assert stats["success_rate"] < 1.0

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

