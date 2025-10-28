"""
Web应用防火墙（WAF）规则引擎
实现OWASP Top 10防护规则
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """威胁级别"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """攻击类型"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    XXE = "xxe"
    SSRF = "ssrf"
    LDAP_INJECTION = "ldap_injection"
    FILE_UPLOAD = "file_upload"
    BUFFER_OVERFLOW = "buffer_overflow"


@dataclass
class WAFRule:
    """WAF规则"""
    rule_id: str
    name: str
    attack_type: AttackType
    threat_level: ThreatLevel
    patterns: List[str]
    enabled: bool = True
    description: str = ""
    false_positive_score: float = 0.0


@dataclass
class WAFDetectionResult:
    """WAF检测结果"""
    blocked: bool
    threat_level: ThreatLevel
    matched_rules: List[WAFRule]
    attack_types: List[AttackType]
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class WAFRuleEngine:
    """WAF规则引擎"""
    
    def __init__(self):
        self.rules: List[WAFRule] = []
        self.blocked_ips: set = set()
        self.rate_limiter = RateLimiter()
        self.attack_stats = defaultdict(int)
        self._initialize_owasp_rules()
    
    def _initialize_owasp_rules(self):
        """初始化OWASP Top 10规则"""
        
        # 1. SQL注入防护
        sql_injection_patterns = [
            r"(\bunion\b.*\bselect\b)",
            r"(\bselect\b.*\bfrom\b.*\bwhere\b)",
            r"(\bdrop\b.*\btable\b)",
            r"(\binsert\b.*\binto\b.*\bvalues\b)",
            r"(\bdelete\b.*\bfrom\b)",
            r"(\bupdate\b.*\bset\b)",
            r"(\bexec\b.*\bsp_)",
            r"(\bor\b.*\b1\s*=\s*1\b)",
            r"(\band\b.*\b1\s*=\s*1\b)",
            r"(--\s*$)",
            r"(;\s*drop\s+)",
            r"(\bwaitfor\b.*\bdelay\b)",
            r"(\bexec\s*\()",
            r"(\bexecute\s*\()"
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-001",
            name="SQL Injection Protection",
            attack_type=AttackType.SQL_INJECTION,
            threat_level=ThreatLevel.CRITICAL,
            patterns=sql_injection_patterns,
            description="Detects SQL injection attempts"
        ))
        
        # 2. XSS防护
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"onerror\s*=",
            r"onload\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<iframe[^>]*>",
            r"<embed[^>]*>",
            r"<object[^>]*>",
            r"document\.cookie",
            r"document\.write",
            r"eval\s*\(",
            r"expression\s*\(",
            r"vbscript:"
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-002",
            name="XSS Protection",
            attack_type=AttackType.XSS,
            threat_level=ThreatLevel.HIGH,
            patterns=xss_patterns,
            description="Detects Cross-Site Scripting attempts"
        ))
        
        # 3. 路径遍历防护
        path_traversal_patterns = [
            r"\.\./",
            r"\.\.\%2f",
            r"\.\.\%5c",
            r"%2e%2e/",
            r"\.\.\\",
            r"/etc/passwd",
            r"/etc/shadow",
            r"c:\\windows",
            r"c:/windows"
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-003",
            name="Path Traversal Protection",
            attack_type=AttackType.PATH_TRAVERSAL,
            threat_level=ThreatLevel.HIGH,
            patterns=path_traversal_patterns,
            description="Detects directory traversal attempts"
        ))
        
        # 4. 命令注入防护
        command_injection_patterns = [
            r";\s*(ls|cat|wget|curl|nc|netcat|bash|sh)",
            r"\|\s*(ls|cat|wget|curl|nc|netcat|bash|sh)",
            r"&&\s*(ls|cat|wget|curl|nc|netcat|bash|sh)",
            r"`.*`",
            r"\$\(.*\)",
            r">\s*/dev/",
            r"<\s*/dev/"
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-004",
            name="Command Injection Protection",
            attack_type=AttackType.COMMAND_INJECTION,
            threat_level=ThreatLevel.CRITICAL,
            patterns=command_injection_patterns,
            description="Detects OS command injection attempts"
        ))
        
        # 5. XXE防护
        xxe_patterns = [
            r"<!ENTITY",
            r"<!DOCTYPE.*ENTITY",
            r"SYSTEM\s+[\"']file://",
            r"PUBLIC\s+[\"']-//",
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-005",
            name="XXE Protection",
            attack_type=AttackType.XXE,
            threat_level=ThreatLevel.HIGH,
            patterns=xxe_patterns,
            description="Detects XML External Entity attacks"
        ))
        
        # 6. SSRF防护
        ssrf_patterns = [
            r"(http|https|ftp)://127\.0\.0\.1",
            r"(http|https|ftp)://localhost",
            r"(http|https|ftp)://192\.168\.",
            r"(http|https|ftp)://10\.",
            r"(http|https|ftp)://172\.(1[6-9]|2[0-9]|3[01])\.",
            r"file://",
            r"gopher://",
            r"dict://"
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-006",
            name="SSRF Protection",
            attack_type=AttackType.SSRF,
            threat_level=ThreatLevel.HIGH,
            patterns=ssrf_patterns,
            description="Detects Server-Side Request Forgery"
        ))
        
        # 7. LDAP注入防护
        ldap_patterns = [
            r"\*\)\s*\(",
            r"\)\s*\|\s*\(",
            r"\)\s*&\s*\(",
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-007",
            name="LDAP Injection Protection",
            attack_type=AttackType.LDAP_INJECTION,
            threat_level=ThreatLevel.MEDIUM,
            patterns=ldap_patterns,
            description="Detects LDAP injection attempts"
        ))
        
        # 8. 文件上传防护
        file_upload_patterns = [
            r"\.php\d?$",
            r"\.jsp$",
            r"\.asp$",
            r"\.aspx$",
            r"\.sh$",
            r"\.bat$",
            r"\.cmd$",
            r"\.exe$",
            r"\.dll$"
        ]
        self.add_rule(WAFRule(
            rule_id="OWASP-008",
            name="Malicious File Upload Protection",
            attack_type=AttackType.FILE_UPLOAD,
            threat_level=ThreatLevel.HIGH,
            patterns=file_upload_patterns,
            description="Detects malicious file upload attempts"
        ))
    
    def add_rule(self, rule: WAFRule):
        """添加规则"""
        self.rules.append(rule)
        logger.info(f"Added WAF rule: {rule.rule_id} - {rule.name}")
    
    def check_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        client_ip: str = ""
    ) -> WAFDetectionResult:
        """
        检查HTTP请求
        
        Args:
            method: HTTP方法
            path: 请求路径
            headers: 请求头
            body: 请求体
            params: 请求参数
            client_ip: 客户端IP
            
        Returns:
            检测结果
        """
        matched_rules = []
        attack_types = set()
        max_threat_level = ThreatLevel.INFO
        
        # 检查IP黑名单
        if client_ip in self.blocked_ips:
            return WAFDetectionResult(
                blocked=True,
                threat_level=ThreatLevel.CRITICAL,
                matched_rules=[],
                attack_types=[],
                details={'reason': 'IP in blacklist', 'ip': client_ip}
            )
        
        # 检查速率限制
        if not self.rate_limiter.allow(client_ip):
            return WAFDetectionResult(
                blocked=True,
                threat_level=ThreatLevel.MEDIUM,
                matched_rules=[],
                attack_types=[],
                details={'reason': 'Rate limit exceeded', 'ip': client_ip}
            )
        
        # 合并所有需要检查的内容
        check_content = self._prepare_check_content(
            path, headers, body, params
        )
        
        # 执行规则匹配
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if self._match_rule(rule, check_content):
                matched_rules.append(rule)
                attack_types.add(rule.attack_type)
                
                # 更新最高威胁级别
                if self._threat_level_value(rule.threat_level) > \
                   self._threat_level_value(max_threat_level):
                    max_threat_level = rule.threat_level
        
        # 决定是否阻断
        should_block = len(matched_rules) > 0 and \
                      max_threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        
        # 记录统计
        if should_block:
            for attack_type in attack_types:
                self.attack_stats[attack_type.value] += 1
            
            # 如果是严重攻击，加入IP黑名单
            if max_threat_level == ThreatLevel.CRITICAL:
                self.blocked_ips.add(client_ip)
                logger.critical(f"IP {client_ip} added to blacklist")
        
        result = WAFDetectionResult(
            blocked=should_block,
            threat_level=max_threat_level,
            matched_rules=matched_rules,
            attack_types=list(attack_types),
            details={
                'client_ip': client_ip,
                'method': method,
                'path': path,
                'matched_rule_ids': [r.rule_id for r in matched_rules]
            }
        )
        
        # 记录日志
        if should_block:
            logger.warning(
                f"WAF BLOCKED: {client_ip} - {method} {path} - "
                f"Threat: {max_threat_level.value} - "
                f"Rules: {[r.rule_id for r in matched_rules]}"
            )
        
        return result
    
    def _prepare_check_content(
        self,
        path: str,
        headers: Dict[str, str],
        body: Optional[str],
        params: Optional[Dict[str, str]]
    ) -> str:
        """准备待检查内容"""
        content_parts = [path]
        
        # 添加headers
        for key, value in headers.items():
            content_parts.append(f"{key}: {value}")
        
        # 添加参数
        if params:
            for key, value in params.items():
                content_parts.append(f"{key}={value}")
        
        # 添加body
        if body:
            content_parts.append(body)
        
        return " ".join(content_parts).lower()
    
    def _match_rule(self, rule: WAFRule, content: str) -> bool:
        """匹配规则"""
        for pattern in rule.patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            except re.error as e:
                logger.error(f"Invalid regex pattern in rule {rule.rule_id}: {e}")
        return False
    
    def _threat_level_value(self, level: ThreatLevel) -> int:
        """威胁级别数值"""
        mapping = {
            ThreatLevel.INFO: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return mapping.get(level, 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'blocked_ips': len(self.blocked_ips),
            'attack_stats': dict(self.attack_stats),
            'top_attacks': sorted(
                self.attack_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def remove_from_blacklist(self, ip: str):
        """从黑名单移除IP"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"IP {ip} removed from blacklist")
    
    def export_rules(self) -> List[Dict]:
        """导出规则配置"""
        return [
            {
                'rule_id': r.rule_id,
                'name': r.name,
                'attack_type': r.attack_type.value,
                'threat_level': r.threat_level.value,
                'enabled': r.enabled,
                'patterns_count': len(r.patterns)
            }
            for r in self.rules
        ]


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def allow(self, client_id: str) -> bool:
        """检查是否允许请求"""
        import time
        current_time = time.time()
        
        # 清理过期记录
        if client_id in self.requests:
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if current_time - t < self.window_seconds
            ]
        
        # 检查速率
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # 记录本次请求
        self.requests[client_id].append(current_time)
        return True


class WAFAnalyzer:
    """WAF日志分析器"""
    
    def __init__(self):
        self.false_positives: List[Dict] = []
        self.true_positives: List[Dict] = []
    
    def analyze_logs(self, logs: List[Dict]) -> Dict[str, Any]:
        """分析WAF日志"""
        total = len(logs)
        blocked = len([l for l in logs if l.get('blocked')])
        
        # 按攻击类型统计
        attack_type_stats = defaultdict(int)
        for log in logs:
            for attack_type in log.get('attack_types', []):
                attack_type_stats[attack_type] += 1
        
        # 按威胁级别统计
        threat_level_stats = defaultdict(int)
        for log in logs:
            threat_level = log.get('threat_level')
            if threat_level:
                threat_level_stats[threat_level] += 1
        
        # 计算误报率（需要人工标注）
        false_positive_rate = len(self.false_positives) / total if total > 0 else 0
        
        return {
            'total_requests': total,
            'blocked_requests': blocked,
            'block_rate': blocked / total if total > 0 else 0,
            'attack_type_distribution': dict(attack_type_stats),
            'threat_level_distribution': dict(threat_level_stats),
            'false_positive_rate': false_positive_rate,
            'top_attack_types': sorted(
                attack_type_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def mark_false_positive(self, log_entry: Dict):
        """标记误报"""
        self.false_positives.append(log_entry)
    
    def mark_true_positive(self, log_entry: Dict):
        """标记正确拦截"""
        self.true_positives.append(log_entry)


if __name__ == "__main__":
    # 测试WAF规则
    waf = WAFRuleEngine()
    
    # 测试SQL注入
    result = waf.check_request(
        method="GET",
        path="/api/users?id=1' OR '1'='1",
        headers={"User-Agent": "TestBot"},
        client_ip="192.168.1.100"
    )
    print(f"SQL Injection Test - Blocked: {result.blocked}")
    print(f"Threat Level: {result.threat_level.value}")
    print(f"Matched Rules: {[r.rule_id for r in result.matched_rules]}")
    
    # 测试XSS
    result = waf.check_request(
        method="POST",
        path="/api/comment",
        headers={"Content-Type": "application/json"},
        body='{"comment": "<script>alert(1)</script>"}',
        client_ip="192.168.1.101"
    )
    print(f"\nXSS Test - Blocked: {result.blocked}")
    
    # 获取统计
    stats = waf.get_statistics()
    print(f"\nWAF Statistics: {json.dumps(stats, indent=2)}")
