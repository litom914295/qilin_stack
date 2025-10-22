"""
审计增强系统（P0-5）
PII脱敏、审计日志、合规追踪
"""

import logging
import json
import hashlib
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """审计事件类型"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    CONFIG_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"
    API_CALL = "api_call"


class PIIType(str, Enum):
    """PII类型"""
    PHONE = "phone"
    EMAIL = "email"
    ID_CARD = "id_card"
    BANK_CARD = "bank_card"
    ADDRESS = "address"
    NAME = "name"
    IP_ADDRESS = "ip_address"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    user_role: Optional[str]
    action: str
    resource: str
    result: str  # success/failure
    ip_address: str
    metadata: Dict[str, Any]
    pii_detected: bool = False
    pii_masked: bool = False


class PIIMasker:
    """PII脱敏器"""
    
    # 正则模式
    PATTERNS = {
        PIIType.PHONE: r'1[3-9]\d{9}',
        PIIType.EMAIL: r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        PIIType.ID_CARD: r'\d{17}[\dXx]',
        PIIType.BANK_CARD: r'\d{16,19}',
        PIIType.IP_ADDRESS: r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    }
    
    def __init__(self, mask_char: str = "*"):
        self.mask_char = mask_char
        self.compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in self.PATTERNS.items()
        }
    
    def mask_phone(self, phone: str) -> str:
        """
        脱敏手机号: 138****5678
        
        Args:
            phone: 手机号
            
        Returns:
            脱敏后的手机号
        """
        if len(phone) != 11:
            return phone
        return f"{phone[:3]}****{phone[-4:]}"
    
    def mask_email(self, email: str) -> str:
        """
        脱敏邮箱: u***r@example.com
        
        Args:
            email: 邮箱地址
            
        Returns:
            脱敏后的邮箱
        """
        parts = email.split('@')
        if len(parts) != 2:
            return email
        
        username = parts[0]
        if len(username) <= 2:
            masked_username = self.mask_char * len(username)
        else:
            masked_username = f"{username[0]}{self.mask_char * (len(username) - 2)}{username[-1]}"
        
        return f"{masked_username}@{parts[1]}"
    
    def mask_id_card(self, id_card: str) -> str:
        """
        脱敏身份证: 110***********1234
        
        Args:
            id_card: 身份证号
            
        Returns:
            脱敏后的身份证号
        """
        if len(id_card) != 18:
            return id_card
        return f"{id_card[:3]}***********{id_card[-4:]}"
    
    def mask_bank_card(self, bank_card: str) -> str:
        """
        脱敏银行卡: 6222************3456
        
        Args:
            bank_card: 银行卡号
            
        Returns:
            脱敏后的银行卡号
        """
        if len(bank_card) < 16:
            return bank_card
        return f"{bank_card[:4]}{'*' * (len(bank_card) - 8)}{bank_card[-4:]}"
    
    def mask_ip_address(self, ip: str) -> str:
        """
        脱敏IP地址: 192.168.*.*
        
        Args:
            ip: IP地址
            
        Returns:
            脱敏后的IP
        """
        parts = ip.split('.')
        if len(parts) != 4:
            return ip
        return f"{parts[0]}.{parts[1]}.*.*"
    
    def detect_and_mask(self, text: str) -> tuple[str, List[PIIType]]:
        """
        检测并脱敏文本中的PII
        
        Args:
            text: 原始文本
            
        Returns:
            (脱敏后的文本, 检测到的PII类型列表)
        """
        masked_text = text
        detected_types = []
        
        # 手机号
        if self.compiled_patterns[PIIType.PHONE].search(masked_text):
            detected_types.append(PIIType.PHONE)
            masked_text = self.compiled_patterns[PIIType.PHONE].sub(
                lambda m: self.mask_phone(m.group()),
                masked_text
        
        # 邮箱
        if self.compiled_patterns[PIIType.EMAIL].search(masked_text):
            detected_types.append(PIIType.EMAIL)
            masked_text = self.compiled_patterns[PIIType.EMAIL].sub(
                lambda m: self.mask_email(m.group()),
                masked_text
        
        # 身份证
        if self.compiled_patterns[PIIType.ID_CARD].search(masked_text):
            detected_types.append(PIIType.ID_CARD)
            masked_text = self.compiled_patterns[PIIType.ID_CARD].sub(
                lambda m: self.mask_id_card(m.group()),
                masked_text
        
        # 银行卡
        if self.compiled_patterns[PIIType.BANK_CARD].search(masked_text):
            detected_types.append(PIIType.BANK_CARD)
            masked_text = self.compiled_patterns[PIIType.BANK_CARD].sub(
                lambda m: self.mask_bank_card(m.group()),
                masked_text
        
        # IP地址
        if self.compiled_patterns[PIIType.IP_ADDRESS].search(masked_text):
            detected_types.append(PIIType.IP_ADDRESS)
            masked_text = self.compiled_patterns[PIIType.IP_ADDRESS].sub(
                lambda m: self.mask_ip_address(m.group()),
                masked_text
        
        return masked_text, detected_types
    
    def mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        脱敏字典中的PII
        
        Args:
            data: 原始字典
            
        Returns:
            脱敏后的字典
        """
        masked_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                masked_value, _ = self.detect_and_mask(value)
                masked_data[key] = masked_value
            elif isinstance(value, dict):
                masked_data[key] = self.mask_dict(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    self.mask_dict(item) if isinstance(item, dict)
                    else self.detect_and_mask(item)[0] if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked_data[key] = value
        
        return masked_data


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(
        self,
        log_dir: str = "logs/audit",
        enable_pii_masking: bool = True,
        enable_console: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_pii_masking = enable_pii_masking
        self.enable_console = enable_console
        
        self.pii_masker = PIIMasker()
        
        # 配置文件日志
        self.file_logger = logging.getLogger("audit")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.propagate = False
        
        # 日志文件按日期滚动
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(message)s')  # 纯JSON格式
        self.file_logger.addHandler(file_handler)
    
    def _generate_event_id(self, event: AuditEvent) -> str:
        """生成事件ID"""
        data = f"{event.timestamp.isoformat()}{event.user_id}{event.action}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        ip_address: str,
        user_role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """
        记录审计事件
        
        Args:
            event_type: 事件类型
            user_id: 用户ID
            action: 操作动作
            resource: 资源
            result: 结果（success/failure）
            ip_address: IP地址
            user_role: 用户角色
            metadata: 元数据
            
        Returns:
            审计事件对象
        """
        # 创建事件
        event = AuditEvent(
            event_id="",  # 稍后生成
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            user_role=user_role,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            metadata=metadata or {}
        
        event.event_id = self._generate_event_id(event)
        
        # PII检测和脱敏
        if self.enable_pii_masking:
            # 脱敏metadata
            event.metadata = self.pii_masker.mask_dict(event.metadata)
            
            # 脱敏IP地址
            event.ip_address = self.pii_masker.mask_ip_address(event.ip_address)
            
            # 检测是否包含PII
            _, pii_types = self.pii_masker.detect_and_mask(json.dumps(event.metadata))
            if pii_types:
                event.pii_detected = True
                event.pii_masked = True
        
        # 记录日志
        log_entry = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "user_role": event.user_role,
            "action": action,
            "resource": resource,
            "result": result,
            "ip_address": event.ip_address,
            "metadata": event.metadata,
            "pii_detected": event.pii_detected,
            "pii_masked": event.pii_masked
        }
        
        self.file_logger.info(json.dumps(log_entry, ensure_ascii=False))
        
        if self.enable_console:
            logger.info(f"Audit: {event.event_type.value} | {user_id} | {action} | {result}")
        
        return event
    
    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None
    ) -> List[Dict[str, Any]]:
        """
        查询审计事件
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            user_id: 用户ID
            event_type: 事件类型
            
        Returns:
            事件列表
        """
        events = []
        
        # 遍历日志文件
        for log_file in sorted(self.log_dir.glob("audit_*.log")):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        
                        # 过滤条件
                        event_time = datetime.fromisoformat(event['timestamp'])
                        
                        if start_date and event_time < start_date:
                            continue
                        if end_date and event_time > end_date:
                            continue
                        if user_id and event['user_id'] != user_id:
                            continue
                        if event_type and event['event_type'] != event_type.value:
                            continue
                        
                        events.append(event)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse log line: {line}")
                        continue
        
        return events


async def main():
    """示例用法"""
    logging.basicConfig(level=logging.INFO)
    
    audit_logger = AuditLogger(enable_console=True)
    
    # 记录登录事件
    await audit_logger.log_event(
        event_type=AuditEventType.USER_LOGIN,
        user_id="user123",
        action="login",
        resource="/api/auth/login",
        result="success",
        ip_address="192.168.1.100",
        user_role="trader",
        metadata={
            "phone": "13812345678",
            "email": "user@example.com"
        }
    
    # 记录数据导出事件（包含敏感信息）
    await audit_logger.log_event(
        event_type=AuditEventType.DATA_EXPORT,
        user_id="user123",
        action="export_user_data",
        resource="/api/data/export",
        result="success",
        ip_address="192.168.1.100",
        metadata={
            "export_type": "csv",
            "records": 1000,
            "contains_pii": True,
            "bank_card": "6222021234567890123"
        }
    
    print("\n=== Querying Events ===")
    events = audit_logger.query_events(user_id="user123")
    for event in events:
        print(json.dumps(event, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
