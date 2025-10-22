# 审计增强与合规（P0-5）

## 概述
建立完整的审计日志系统，包含PII脱敏、合规追踪、安全事件记录。

## PII脱敏

### 支持的PII类型
- **手机号**: 138****5678
- **邮箱**: u***r@example.com
- **身份证**: 110***********1234
- **银行卡**: 6222************3456
- **IP地址**: 192.168.*.*

### 脱敏规则
1. **手机号**: 保留前3位和后4位
2. **邮箱**: 保留首尾字符，中间用*替代
3. **身份证**: 保留前3位和后4位
4. **银行卡**: 保留前4位和后4位
5. **IP地址**: 保留前两段，后两段用*替代

## 审计事件类型

### 用户行为
- **USER_LOGIN**: 用户登录
- **USER_LOGOUT**: 用户登出
- **DATA_ACCESS**: 数据访问
- **DATA_EXPORT**: 数据导出

### 系统操作
- **MODEL_TRAINING**: 模型训练
- **MODEL_INFERENCE**: 模型推理
- **CONFIG_CHANGE**: 配置变更
- **API_CALL**: API调用

### 安全事件
- **SECURITY_EVENT**: 安全事件（登录失败、越权访问等）

## 审计日志格式

### JSON格式
```json
{
  "event_id": "a1b2c3d4e5f6g7h8",
  "event_type": "user_login",
  "timestamp": "2025-10-16T10:00:00Z",
  "user_id": "user123",
  "user_role": "trader",
  "action": "login",
  "resource": "/api/auth/login",
  "result": "success",
  "ip_address": "192.168.*.*",
  "metadata": {
    "phone": "138****5678",
    "email": "u***r@example.com"
  },
  "pii_detected": true,
  "pii_masked": true
}
```

## 使用示例

### 直接记录审计事件
```python
from security.audit_enhanced import AuditLogger, AuditEventType

audit_logger = AuditLogger()

await audit_logger.log_event(
    event_type=AuditEventType.DATA_EXPORT,
    user_id="user123",
    action="export_data",
    resource="/api/data/export",
    result="success",
    ip_address="192.168.1.100",
    metadata={
        "export_type": "csv",
        "records": 1000
    }
)
```

### FastAPI中间件自动审计
```python
from fastapi import FastAPI
from api.audit_middleware import setup_audit_middleware

app = FastAPI()
setup_audit_middleware(app)

# 所有API请求自动记录审计日志
```

### 查询审计日志
```python
from datetime import datetime, timedelta

# 查询最近7天的审计事件
events = audit_logger.query_events(
    start_date=datetime.now() - timedelta(days=7),
    user_id="user123"
)
```

## 日志存储

### 文件存储
- **路径**: `logs/audit/`
- **命名**: `audit_YYYYMMDD.log`
- **格式**: 每行一条JSON记录
- **滚动**: 按日期自动滚动

### 日志保留策略
- **生产环境**: 保留90天
- **测试环境**: 保留30天
- **归档**: 超过保留期的日志压缩归档到S3/MinIO

## 合规要求

### GDPR合规
- ✅ PII自动脱敏
- ✅ 用户数据访问可追溯
- ✅ 数据导出记录审计
- ✅ 用户请求历史可查询

### 等保2.0要求
- ✅ 用户登录/登出审计
- ✅ 重要操作审计
- ✅ 安全事件记录
- ✅ 审计日志完整性保护

### 金融行业要求
- ✅ 交易操作全程可追溯
- ✅ 敏感数据访问审计
- ✅ 异常行为检测支持
- ✅ 定期审计报告生成

## 最佳实践

1. **PII脱敏**: 默认启用，所有日志自动脱敏
2. **最小记录原则**: 只记录必要信息
3. **访问控制**: 审计日志仅授权人员可访问
4. **完整性保护**: 使用哈希校验防止篡改
5. **定期审查**: 每月审查审计日志，检测异常

## 告警集成

### 异常行为告警
- 短时间内多次登录失败
- 非工作时间数据导出
- 越权访问尝试
- 批量数据访问

### 告警配置
```python
# 配置异常行为检测规则
# 5分钟内登录失败>5次 → 告警
# 单次导出记录>10000 → 告警
# 非白名单IP访问 → 告警
```

## 故障排查

### 审计日志丢失
1. 检查磁盘空间
2. 验证日志目录权限
3. 查看应用日志错误

### PII脱敏失效
1. 确认enable_pii_masking=True
2. 检查PIIMasker正则表达式
3. 验证脱敏逻辑

## 部署

### 环境变量
```bash
AUDIT_LOG_DIR=/var/log/qilin/audit
AUDIT_ENABLE_PII_MASKING=true
AUDIT_LOG_RETENTION_DAYS=90
```

### K8s部署
```yaml
# 使用PersistentVolume存储审计日志
volumeMounts:
  - name: audit-logs
    mountPath: /var/log/qilin/audit
```

## 参考资料
- [GDPR合规指南](https://gdpr.eu/)
- [等保2.0审计要求](https://www.djbh.net/)
- [金融数据安全规范](https://www.pbc.gov.cn/)
