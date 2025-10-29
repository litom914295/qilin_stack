# AGENT.md — 多智能体协作规范（TradingAgents + RD-Agent）

## 角色（10）
1. 市场生态（Regime）
2. 竞价博弈（Auction）
3. 仓位控制（Position）
4. 成交量（Volume）
5. 技术指标（TA）
6. 市场情绪（Sentiment）
7. 风险管理（Risk）
8. K线形态（Pattern）
9. 宏观经济（Macro）
10. 套利机会（Arb）

## 决策
- 双阈值门锁：`score > threshold` 且 `confidence > threshold`
- 输出可解释：各 Agent 贡献度汇总到 Dashboard

## 流程
1. 研究员：产出 `factors/*.py` 与 `config/factor_xxx.yaml`
2. 回测员：`main.py --mode simulation` 批跑，指标：IC/IR、TopK、回撤、稳定性
3. 风控官：审查持仓/下行风险/流动性与约束
4. 执行员：接入 `executors/order_gateway/` 撮合模拟，生成盘前/盘后报告
