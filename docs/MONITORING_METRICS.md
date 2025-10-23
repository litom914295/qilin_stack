# 监控指标（Prometheus）

本项目关键指标（导出/可视化请参考 docker/grafana 配置或自建看板）。

## 决策与信号
- `signal_generated_total{source,type}`：各来源信号计数
- `decision_made_total{symbol,decision}`：生成的最终决策计数
- `decision_latency_seconds{symbol}`：单标的决策耗时直方图

## 权重与 Gates
- `system_weight{source}`：融合权重（若启用自适应会更新）
- `gate_reject_total{reason}`：Gates 拦截计数（如 `mcap/turnover/concept_heat/first_touch/open_count/volume_surge/seal_quality/price/low_confidence`）

## Qlib 通道
- `qlib_cache_hit_total`：Qlib 预测缓存命中次数
- `qlib_pred_latency_seconds{symbol}`：单标预测时延（简化动量）
- `qlib_batch_pred_latency_seconds{count}`：批量预取时延（count=批量标的数）

## 使用建议
- 将上述指标抓取到 Prometheus，并在 Grafana 中建立：
  - 权重轨迹（QLib/TradingAgents/RD-Agent）
  - 拒绝原因 TopN（漏斗诊断）
  - 决策耗时分布（性能健康）
  - Qlib 命中率与时延（数据通道健康）
