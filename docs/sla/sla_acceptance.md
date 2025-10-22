# SLA/SLO 验收标准（P0-9）

## 验收目标
- 端到端 P95 延迟 ≤ 1000ms（交易时段内）
  - 指标：`qilin_e2e_latency_ms_bucket`（Prometheus 直方图）
- 服务可用性 ≥ 99.9%
- 信号覆盖率 ≥ 80%
- 故障恢复时间（RTO）≤ 5 分钟
  - 指标：`qilin_failover_recovery_seconds`（Gauge）
- 推荐准确率（T+1）≥ 70%

## 验收步骤（MVP）
1) 运行 E2E 与并发/恢复测试：
   - PowerShell: `./scripts/ci/run_slo_tests.ps1`
   - 或手动：`python -m pytest tests/e2e/test_mvp_slo.py -q`
2) 生成 SLO 报告（控制台输出）；记录到发布工单。
3) 检查 Prometheus 告警无红线（详见 `monitoring/prometheus/rules/slo_alerts.yaml`）。
   - /metrics 由 `qilin_stack_with_ta/api/metrics_endpoint.py` 暴露，已集成业务、连接池与 SLO 指标（共享 registry）。
4) Grafana 仪表盘“Qilin SLO Overview”各核心面板达标。

## 退出准入条件
- 所有 E2E/SLO 测试用例通过。
- 近 24h 未触发 SLO P0 告警（page）且错误预算在阈内。
- 有 Runbook（排障手册）与回滚预案链接；演练记录通过。

## 备注
- 指标名/维度需与实际导出的 Prometheus 指标对齐。
- Windows 环境建议使用 PowerShell 脚本以统一入口。
