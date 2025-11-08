# 改进计划与进度（2025-10-23）

本页跟踪“一进二量化”项目的阶段性改造计划与落地进度。

## P0 稳定性与口径修正（本周）
- [x] 决策引擎健壮性：
  - logger 提前初始化，避免 ImportError 分支引用未定义 logger
  - 并发示例修正为 `generate_signals` 批量调用
  - Qlib 一次性初始化（幂等），简化预测路径容错
- [x] RD-Agent 字段兼容层：统一/适配 `seal_strength→seal_quality`、`board_height→continuous_board` 等别名，降低口径差异风险
- [x] 权重优化器与单测接口对齐：
  - 新增 `SystemPerformance`，`evaluate_performance` 双形态（数组/信号）兼容
  - `_calculate_metrics/_calculate_sharpe_ratio`、`get_performance_summary`、`optimize_weights`（字符串键）完善
- [x] 回测成交口径增强：
  - 新增 `trade_at='next_open'`（T+1开盘成交）；默认保持 `same_day_close` 不破坏现有示例
  - 增加“开盘涨停（近似≥+9.8%）不可成交”判定
- [x] 冒烟检查：
  - `import decision_engine.core` / `decision_engine.weight_optimizer` / `backtest.engine` 通过

已追加（P0）
- [x] README 增补 T+1 用法示例
- [x] 默认成交口径切换为 `next_open`（2025-10-23）

## P1 策略与执行深化（1–2 周）
- [x] Gates 配置化（first_touch/open_count/volume_surge/price/mcap 等）+ Prometheus `gate_reject_total{reason=...}` 指标
- [x] 执行撮合细化（基础）：deterministic/prob 成交比例、T+1 开盘一字不可成交、无法成交判定、无法成交率统计与回测报告输出
- [x] 题材热度/龙虎榜：注入至决策上下文，LeaderAgent 融合（基础）
- [x] Qlib 批量/缓存与监控：`qlib_cache_hit_total`、`qlib_pred_latency_seconds`、`qlib_batch_pred_latency_seconds`；决策耗时 `decision_latency_seconds`

## 使用提示
- 回测 T+1 示例：
  ```python
  # backtest/engine.py
  metrics = await engine.run_backtest(
      symbols, '2024-01-01','2024-06-30', data_source,
      trade_at='next_open', avoid_limit_up_unfillable=True,
  )
  ```

## 里程碑
- 2025-10-23：完成 P0 主线修复、T+1 默认化与冒烟；README 已补充。
