# 麒麟量化系统 P0问题修复行动计划

## 文档信息
- **版本**: v1.0
- **创建日期**: 2025-10-16
- **基于评审**: ChatGPT评审报告 - Technical_Architecture_v2.1_Final.md
- **状态**: 🟡 执行中

---

## 一、评审结论摘要

### 可用性结论
✅ **达到"受控MVP可用/灰度可用"门槛**
- 建议先在模拟盘/小流量灰度运行
- 对外宣称"生产可用"需先关闭若干P0阻断项

### 关键矛盾
⚠️ **文档"总结"宣称生产可用 vs. "实施路线图"P0项未勾选**
- 存在"设计完备 vs. 落地未闭环"的不一致
- 需优先校正P0阻断项

---

## 二、P0阻断项清单（17项）

### 🔐 安全零信任落地（P0-1 至 P0-5）

#### P0-1: 密钥轮换机制 ✅ 已完成
**状态**: 已实现
**文件**: `security/key_rotation.py`
**完成内容**:
- ✅ JWT密钥自动轮换（30天周期）
- ✅ API密钥轮换（90天周期）
- ✅ 加密密钥轮换（180天周期）
- ✅ 签名密钥轮换（60天周期）
- ✅ 数据库密码轮换（90天周期）
- ✅ 轮换调度器（每小时检查）
- ✅ 密钥宽限期机制（7天）
- ✅ 轮换历史记录
- ✅ 密钥备份与恢复

**验证步骤**:
```bash
# 测试密钥生成
python security/key_rotation.py

# 检查Redis中的密钥
redis-cli KEYS "qilin:keys:*"

# 启动轮换调度器
# 在deployment中配置环境变量：
# MASTER_ENCRYPTION_KEY=<strong-key>
# REDIS_URL=redis://redis-service:6379
```

#### P0-2: mTLS与证书管理 ⏳ 待实施
**优先级**: Critical
**预计耗时**: 2天

**实施任务**:
1. **证书生成脚本**
   - 使用cert-manager自动化证书管理
   - CA证书、服务器证书、客户端证书
   - 证书轮换策略（90天）

2. **Envoy配置**
   - 在deployment中添加Envoy sidecar
   - 配置mTLS通信
   - 证书挂载与热重载

3. **监控告警**
   - 证书过期监控（提前30天告警）
   - 证书验证失败告警

**文件清单**:
- `deploy/k8s/security/cert-manager.yaml`
- `deploy/k8s/security/certificates.yaml`
- `deploy/k8s/security/envoy-config.yaml`
- `scripts/generate-certs.sh`

#### P0-3: WAF规则库 ⏳ 待实施
**优先级**: High
**预计耗时**: 1.5天

**实施任务**:
1. **OWASP Top 10防护**
   - SQL注入防护规则
   - XSS防护规则
   - CSRF防护规则
   - 文件上传防护
   - 命令注入防护

2. **误报率监控**
   - WAF日志收集与分析
   - 误报率仪表盘
   - 规则动态调优

3. **攻击日志分析**
   - 攻击类型统计
   - 攻击源IP黑名单
   - 实时告警

**文件清单**:
- `security/waf_rules.py`
- `security/waf_analyzer.py`
- `deploy/k8s/security/waf-config.yaml`

#### P0-4: K8s安全基线 ✅ 部分完成
**状态**: 已实现RBAC和NetworkPolicy
**文件**: 
- ✅ `deploy/k8s/security/rbac.yaml`
- ✅ `deploy/k8s/security/network-policy.yaml`
- ⏳ `deploy/k8s/security/pod-security-policy.yaml` (待实施)
- ⏳ `deploy/k8s/security/opa-policies/` (待实施)

**已完成**:
- ✅ RBAC策略（应用、管理员、只读三级权限）
- ✅ NetworkPolicy（入站/出站流量控制）
- ✅ 最小权限原则

**待完成**:
- ⏳ PodSecurityPolicy/PodSecurity Standards
- ⏳ OPA/Gatekeeper策略
- ⏳ CIS基线检查脚本

**部署指令**:
```bash
# 应用RBAC策略
kubectl apply -f deploy/k8s/security/rbac.yaml

# 应用网络策略
kubectl apply -f deploy/k8s/security/network-policy.yaml

# 验证
kubectl get rolebindings -n production
kubectl get networkpolicies -n production
```

#### P0-5: 审计增强 ⏳ 待实施
**优先级**: High
**预计耗时**: 2天

**实施任务**:
1. **PII识别与脱敏**
   - 扩展PIIDetector规则
   - 自动脱敏中间件
   - PII字段配置化

2. **审计日志留存**
   - 90天保留策略
   - 日志归档到对象存储
   - 合规查询接口

3. **合规报告**
   - 日度/周度/月度审计报告
   - 异常行为统计
   - 合规仪表盘

**文件清单**:
- `security/audit_enhanced.py`
- `security/pii_detector_v2.py`
- `security/compliance_reporter.py`

---

### 📊 数据质量门禁（P0-6 至 P0-8）

#### P0-6: Great Expectations集成 ⏳ 待实施
**优先级**: Critical
**预计耗时**: 2天

**实施任务**:
1. **Expectation Suite编制**
   - 行情数据：价格合理性、成交量范围、时间连续性
   - 资金流数据：净流入范围、大单占比
   - 新闻数据：时效性、完整性
   - 龙虎榜数据：封单金额、涨停时间

2. **CI集成**
   - GitHub Actions步骤
   - 失败阻断构建
   - 质量报告artifact

3. **生产门禁**
   - 数据摄取前验证
   - 失败触发降级
   - 告警通知

**文件清单**:
- `data_quality/expectations/market_data_suite.json`
- `data_quality/expectations/capital_flow_suite.json`
- `data_quality/expectations/news_suite.json`
- `data_quality/expectations/longhu_suite.json`
- `data_quality/ge_integration.py`

**示例Expectation**:
```python
# 行情数据Expectation Suite
{
  "expectation_suite_name": "market_data_suite",
  "expectations": [
    {
      "expectation_type": "expect_column_values_to_be_between",
      "kwargs": {
        "column": "close",
        "min_value": 0,
        "max_value": 10000
      }
    },
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {
        "column": "timestamp",
        "mostly": 0.99
      }
    }
  ]
}
```

#### P0-7: 降级与补数流程 ⏳ 待实施
**优先级**: Critical
**预计耗时**: 2天

**实施任务**:
1. **降级机制**
   - 质量分数<0.8触发
   - 使用缓存数据
   - 降级状态监控

2. **备用数据源**
   - 主备数据源配置
   - 自动切换逻辑
   - 健康检查探针

3. **补数策略**
   - 数据缺失检测
   - 历史数据回填
   - 补数任务队列

4. **Runbook编制**
   - 故障分类与处置流程
   - 联系人与升级路径
   - 恢复验证清单

**文件清单**:
- `data_quality/degradation_manager.py`
- `data_quality/data_recovery.py`
- `docs/runbooks/DATA_QUALITY_INCIDENT.md`

#### P0-8: 质量报表与告警 ⏳ 待实施
**优先级**: High
**预计耗时**: 1.5天

**实施任务**:
1. **质量仪表盘**
   - Grafana仪表盘
   - 实时质量分数
   - 趋势分析图表

2. **告警路由**
   - PagerDuty集成
   - 告警级别分类
   - 告警聚合与去重

3. **阈值配置**
   - 不同数据源独立阈值
   - 豁免机制（维护窗口）
   - 动态阈值调整

**文件清单**:
- `monitoring/grafana-dashboards/data-quality.json`
- `monitoring/alertmanager-config.yaml`
- `data_quality/alerting.py`

---

### 🎯 端到端SLA与监控（P0-9 至 P0-12）

#### P0-9: 端到端SLA与验收 ⏳ 待实施
**优先级**: Critical
**预计耗时**: 2天

**SLO定义**:
```yaml
MVP闭环SLO:
  - 指标: P95延迟
    目标: < 1秒
    范围: 从数据采集到生成推荐
  
  - 指标: 信号覆盖率
    目标: ≥ 80%
    说明: 每日至少80%的目标股票有分析信号
  
  - 指标: 回退时间
    目标: ≤ 5分钟
    说明: 故障检测到系统恢复正常
  
  - 指标: 推荐准确率
    目标: ≥ 70%
    说明: T+1日涨幅>3%视为成功
  
  - 指标: 系统可用性
    目标: ≥ 99.9%
    说明: 月度计算，排除计划维护
```

**验收用例**:
1. **端到端流程测试**
   - 数据采集→特征工程→Agent分析→生成推荐→风控检查
   - 验证各环节耗时
   - 验证数据完整性

2. **故障恢复测试**
   - 模拟数据源故障
   - 验证降级机制
   - 验证恢复时间

3. **负载测试**
   - 并发分析100支股票
   - 验证P95延迟<1s
   - 验证资源使用率

**可观测面板**:
- Grafana SLA仪表盘
- Jaeger分布式追踪
- ELK日志聚合

**文件清单**:
- `tests/e2e/test_mvp_slo.py`
- `tests/e2e/test_failover.py`
- `monitoring/grafana-dashboards/sla-overview.json`
- `docs/SLA_VALIDATION_REPORT.md`

#### P0-10: SLO与告警矩阵 ⏳ 待实施
**优先级**: Critical
**预计耗时**: 1.5天

**告警矩阵（红黄绿三级）**:
```yaml
红色告警 (Critical):
  系统:
    - API不可用 (1分钟内无响应)
    - 数据库连接失败
    - Redis连接失败
  数据:
    - 数据质量<0.5
    - 行情数据延迟>10分钟
  业务:
    - 推荐生成失败率>20%
    - 交易信号错误率>10%
  
黄色告警 (Warning):
  系统:
    - CPU使用率>80%
    - 内存使用率>85%
    - P95延迟>500ms
  数据:
    - 数据质量0.5-0.8
    - 行情数据延迟5-10分钟
  业务:
    - 推荐生成失败率10-20%
    - 单日推荐数量<1
  
绿色告警 (Info):
  系统:
    - 服务重启
    - 配置变更
  数据:
    - 数据质量0.8-0.9
  业务:
    - 新增推荐生成
```

**Pager通道配置**:
- 红色告警 → PagerDuty（24/7值班）
- 黄色告警 → Slack + Email
- 绿色告警 → Slack

**文件清单**:
- `monitoring/alertmanager/alert-rules.yaml`
- `monitoring/pagerduty-config.yaml`
- `docs/ALERT_RUNBOOK.md`

#### P0-11: 业务金指标 ⏳ 待实施
**优先级**: High
**预计耗时**: 2天

**金指标清单**:
```yaml
推荐质量指标:
  - 推荐命中率: T+1日涨幅>3%的比例
  - 推荐覆盖率: 有推荐的交易日占比
  - 平均收益率: 推荐股票的平均涨幅
  
风险指标:
  - 最大回撤: 单日最大亏损比例
  - 夏普比率: 收益风险比
  - 胜率: 盈利次数/总次数
  
系统性能指标:
  - 分析失败率: 分析任务失败占比
  - P95延迟: 95%分位延迟
  - Agent并发数: 同时运行的Agent数量
  
降级触发条件:
  - 推荐命中率连续3天<50% → 触发人工审核
  - 系统失败率>10% → 触发限流降级
  - 数据质量<0.7 → 停止推荐生成
```

**实现**:
- Prometheus自定义metrics
- Grafana业务仪表盘
- 实时告警规则

**文件清单**:
- `monitoring/business_metrics.py`
- `monitoring/grafana-dashboards/business-metrics.json`
- `monitoring/degradation_triggers.py`

#### P0-12: 依赖探针与合成监控 ⏳ 待实施
**优先级**: High
**预计耗时**: 2天

**依赖服务**:
- Redis: 缓存服务
- PostgreSQL: 数据存储
- Kafka: 消息队列
- AkShare: 行情数据源
- TuShare: 备用数据源
- LLM API: OpenAI/DeepSeek

**健康探针**:
```python
# 依赖健康检查
dependencies = {
    'redis': check_redis_health(),
    'postgres': check_postgres_health(),
    'kafka': check_kafka_health(),
    'akshare': check_akshare_api(),
    'llm': check_llm_api()
}
```

**合成监控**:
- 模拟完整交易流程
- 每5分钟执行一次
- 端到端延迟监控

**依赖地图**:
- Service Mesh可视化
- 调用链路追踪
- 级联故障预警

**文件清单**:
- `monitoring/dependency_probes.py`
- `monitoring/synthetic_monitor.py`
- `monitoring/service-map-config.yaml`

---

### 🔄 灾备与回滚（P0-13 至 P0-15）

#### P0-13: 数据库迁移策略 ⏳ 待实施
**优先级**: Critical
**预计耗时**: 2天

**RPO/RTO目标**:
- RPO (Recovery Point Objective): < 5分钟
- RTO (Recovery Time Objective): < 15分钟

**实施任务**:
1. **数据库变更策略**
   - Flyway/Alembic迁移工具
   - 版本化SQL脚本
   - 回滚脚本配对

2. **快照备份**
   - 每日全量备份
   - 每小时增量备份
   - 备份验证脚本

3. **回滚程序**
   - 自动回滚触发器
   - 数据一致性检查
   - 回滚验证测试

**文件清单**:
- `scripts/db-backup.sh`
- `scripts/db-restore.sh`
- `scripts/db-rollback.sh`
- `migrations/` (Alembic)
- `tests/test_db_migration.py`

**备份策略**:
```bash
# 全量备份（每日0点）
0 0 * * * /app/scripts/db-backup.sh --type=full

# 增量备份（每小时）
0 * * * * /app/scripts/db-backup.sh --type=incremental

# 备份验证（每日6点）
0 6 * * * /app/scripts/db-backup.sh --verify
```

#### P0-14: 消息幂等与状态回滚 ⏳ 待实施
**优先级**: High
**预计耗时**: 2天

**实施任务**:
1. **消息幂等性**
   - 消息去重（基于消息ID）
   - 幂等消费者实现
   - 幂等性测试

2. **消息重放**
   - Kafka offset管理
   - 手动/自动重放
   - 重放监控

3. **状态回滚**
   - 分布式事务补偿（SAGA）
   - 状态机回滚
   - 回滚审计日志

**文件清单**:
- `streaming/idempotent_consumer.py`
- `streaming/message_replay.py`
- `trading/transaction_saga.py`
- `tests/test_idempotency.py`

#### P0-15: 跨AZ演练 ⏳ 待实施
**优先级**: High
**预计耗时**: 1天

**演练方案**:
1. **故障注入**
   - 模拟单AZ网络故障
   - 模拟数据库主节点故障
   - 模拟Redis集群故障

2. **自动切换**
   - DNS切换脚本
   - 流量迁移验证
   - 数据一致性检查

3. **演练记录**
   - 演练日志
   - RTO/RPO实测值
   - 改进建议

**文件清单**:
- `scripts/az-failover.sh`
- `scripts/chaos-injection.sh`
- `docs/runbooks/AZ_FAILOVER.md`
- `docs/DISASTER_RECOVERY_DRILL_REPORT.md`

---

### 📄 文档与报告（P0-16 至 P0-17）

#### P0-16: 更新技术文档 ⏳ 待实施
**优先级**: Medium
**预计耗时**: 0.5天

**更新内容**:
1. 修正路线图勾选状态
2. 补充落地证据链接
3. 添加环境参数化清单
4. 更新部署架构图

**文件**: `docs/Technical_Architecture_v2.1_Final.md`

#### P0-17: 生成安全落地清单 ⏳ 待实施
**优先级**: Medium
**预计耗时**: 1天

**报告内容**:
1. 安全措施验证证据
2. 攻防演练记录
3. 合规检查结果
4. 漏洞扫描报告
5. 渗透测试报告

**文件**: `docs/SECURITY_VALIDATION_REPORT.md`

---

## 三、实施时间表

### 第一周 (Days 1-7)
**焦点**: 安全加固
- Day 1-2: P0-2 mTLS证书管理
- Day 2-3: P0-3 WAF规则库
- Day 4-5: P0-4 完成K8s安全基线 (PSP + OPA)
- Day 5-7: P0-5 审计增强

### 第二周 (Days 8-14)
**焦点**: 数据质量与监控
- Day 8-9: P0-6 Great Expectations集成
- Day 10-11: P0-7 降级与补数流程
- Day 11-12: P0-8 质量报表与告警
- Day 12-13: P0-9 端到端SLA验收
- Day 13-14: P0-10 SLO与告警矩阵

### 第三周 (Days 15-21)
**焦点**: 监控优化与灾备
- Day 15-16: P0-11 业务金指标
- Day 16-17: P0-12 依赖探针
- Day 18-19: P0-13 数据库迁移策略
- Day 19-20: P0-14 消息幂等与回滚
- Day 20-21: P0-15 跨AZ演练

### 第四周 (Days 22-28)
**焦点**: 文档与验证
- Day 22: P0-16 更新技术文档
- Day 23-24: P0-17 安全落地报告
- Day 25-28: 全面测试与验证

---

## 四、验收标准

### 安全验收
- [ ] 所有密钥均已配置自动轮换
- [ ] mTLS在所有服务间通信启用
- [ ] WAF规则覆盖OWASP Top 10
- [ ] K8s安全基线通过CIS扫描
- [ ] 审计日志留存90天可查

### 数据质量验收
- [ ] 所有数据源均有Expectation Suite
- [ ] 质量门禁集成到CI/CD
- [ ] 质量<0.8时自动降级生效
- [ ] 质量仪表盘实时更新

### 监控验收
- [ ] SLA指标<红色告警阈值
- [ ] 告警矩阵完整配置
- [ ] 业务金指标采集正常
- [ ] 依赖探针覆盖所有外部服务

### 灾备验收
- [ ] 数据库RPO<5min, RTO<15min
- [ ] 消息幂等性测试通过
- [ ] 跨AZ切换演练成功
- [ ] 灾备文档完整

---

## 五、风险与缓解

### 高风险项
1. **mTLS实施复杂度高**
   - 缓解: 分阶段实施，先内部服务
   - 备选: 仅外部API使用mTLS

2. **Great Expectations性能影响**
   - 缓解: 异步验证，不阻塞主流程
   - 备选: 采样验证（10%数据）

3. **跨AZ演练影响业务**
   - 缓解: 在非交易时段演练
   - 备选: 在staging环境先行

### 依赖项
- cert-manager (K8s插件)
- Prometheus + Grafana
- PagerDuty账号
- Kafka集群
- 数据库副本集

---

## 六、责任分配（建议）

| 任务组 | 负责人角色 | 协助角色 |
|--------|------------|----------|
| 安全加固 (P0-1~5) | 安全工程师 | DevOps, 后端 |
| 数据质量 (P0-6~8) | 数据工程师 | 后端, QA |
| 监控告警 (P0-9~12) | SRE | 后端, DevOps |
| 灾备回滚 (P0-13~15) | DBA, DevOps | 后端, SRE |
| 文档报告 (P0-16~17) | 技术写作 | 架构师 |

---

## 七、成功指标

### 技术指标
- 系统可用性: ≥ 99.9%
- P95延迟: < 1秒
- 数据质量平均分: ≥ 0.9
- 安全漏洞数: 0个Critical/High

### 业务指标
- 推荐命中率: ≥ 70%
- 日推荐覆盖率: ≥ 80%
- 故障恢复时间: ≤ 5分钟

### 合规指标
- 审计日志完整性: 100%
- 安全扫描通过率: 100%
- 灾备演练成功率: 100%

---

## 八、后续优化（P1/P2）

### P1优先级（1-2周内）
- MLOps准入门槛明确化
- 性能压测与容量规划
- 合规文档完善（A股监管）
- 接口契约与错误码体系

### P2优先级（持续优化）
- 序列图/时序图补充
- 故障排查手册
- 运营飞轮建立
- 知识库沉淀

---

## 附录

### A. 部署检查清单
```bash
# 安全检查
kubectl get sa -n production
kubectl get rolebindings -n production
kubectl get networkpolicies -n production

# 监控检查
curl http://qilin-service:9090/metrics
kubectl logs -n monitoring prometheus-0

# 数据质量检查
python -m data_quality.ge_integration --validate-all

# 灾备检查
./scripts/db-backup.sh --verify
./scripts/az-failover.sh --dry-run
```

### B. 关键配置参数
```yaml
# 密钥轮换周期
key_rotation:
  jwt_secret: 30  # 天
  api_key: 90
  encryption_key: 180

# 数据质量阈值
data_quality:
  critical_threshold: 0.5  # 红色告警
  warning_threshold: 0.8   # 黄色告警
  grace_period_minutes: 5  # 降级宽限期

# SLA目标
sla:
  p95_latency_ms: 1000
  availability_percent: 99.9
  mttr_minutes: 5  # Mean Time To Recovery
```

### C. 快速参考链接
- [技术架构文档](./Technical_Architecture_v2.1_Final.md)
- [进度报告](./IMPROVEMENT_PROGRESS.md)
- [部署指南](./DEPLOYMENT_GUIDE.md)
- [安全框架](../security/zero_trust_framework.py)
- [数据质量检查](../app/data/data_quality.py)

---

**文档状态**: 🟡 执行中
**最后更新**: 2025-10-16
**审阅者**: 待指定
**批准者**: 待指定
