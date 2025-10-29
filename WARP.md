# WARP Rules — 麒麟（Qilin）一进二量化作战平台（定制版）
_Last updated: 2025-10-29 04:36:56_

> 本规则文件**专为**仓库 `qilin_stack` 设计，对齐现有目录、脚本与运行方式：
> - 入口：`main.py`（`--mode replay/simulation`，`--action prepare_data`）
> - Web：`web/unified_dashboard.py`（或 `start_web.py`）
> - 数据：`scripts/get_data.py`（Qlib cn_data 下载/校验）
> - 配置：`config/default.yaml`（含 `market_regimes` 权重/风控）
> - 可观测：`docker compose --profile observability up -d`（Prometheus/Grafana）
> - 研究/多智能体：`rd_agent/`、`integrations/tradingagents_cn/`、`tradingagents_integration/`
> - 因子/策略：`factors/`、`strategy/`
> - 执行/网关：`executors/order_gateway/`（仅模拟）

## 规则加载
- 根目录 `WARP.md` 为**项目规则**；如子目录存在 `WARP.md`，**子目录优先**。
- 需要特化 Agent 行为时，可在 `web/`、`docs/` 等目录放置轻量规则。

## 工程规范
- Python 3.10/3.11；使用 `ruff`/`black`/`isort`，类型检查 `pyright`。
- 目录职责：
  - `data_pipeline/`：数据落地、校验、缓存（含 `qlib_enhanced/`）
  - `factors/`：一进二特征（涨停强度/竞价/扩散/情绪/生态位）
  - `strategy/`：信号→仓位→风控（**双阈值门锁：score & confidence**）
  - `rd_agent/`：因子研发循环（RD-Agent）
  - `tradingagents_*`：多智能体协作（TradingAgents）
  - `executors/order_gateway/`：撮合与订单模拟（**严禁实盘**）
- 提交前：`ruff check && ruff format && pyright` + 最小回测单测。

## Agent 执行蓝图（Checklist）
### A. 数据准备与验证
1. `python scripts/get_data.py --source qlib`（首次）或 `python main.py --action prepare_data`
2. 校验 `~/.qlib/qlib_data/cn_data` 覆盖范围
3. `docker compose --profile observability up -d` 启动监控栈

### B. 一进二因子（factors/）
- 标签：`next_day_second_limit`（T+1 是否二板）或事件收益
- 特征族：涨停强度、竞价换手、板块扩散/生态位、情绪、前日行为
- 产物：`factors/onein2.py`、`config/factor_onein2.yaml`

### C. 回测与报告（simulation/replay）
- 训练/验证/测试：示例 2019–2022 / 2023 / 2024–2025
- 命令：
  - 复盘：`python main.py --mode replay --date YYYY-MM-DD`
  - 区间：`python main.py --mode simulation --start 2024-01-01 --end 2024-06-30`
- 指标：TopK 命中/收益、IC/IR、年化/回撤/胜率/盈亏比、稳定性（月度）

### D. RD-Agent 循环（rd_agent/）
- 候选因子→实现与单测→对比实验→归档复现

### E. TradingAgents 协作（integrations/tradingagents_cn/ 等）
- 典型角色：市场生态、竞价、仓位、量能、TA、情绪、风险、形态、宏观、套利
- 由 `market_regimes` 动态调权与风控

### F. 执行与风险（executors/order_gateway/）
- 仅模拟：涨跌停不可交易、滑点/手续费、板块 20% 限制、单票/单日资金上限

## 常用命令
- 准备数据：`python main.py --action prepare_data`
- 复盘一天：`python main.py --mode replay --date 2025-10-22`
- 区间模拟：`python main.py --mode simulation --start 2024-01-01 --end 2024-06-30`
- Web 面板：`streamlit run web/unified_dashboard.py`
- 监控：`cd docker && docker compose --profile observability up -d`

## 约束
- 禁止实盘下单；严格时间切分避免未来数据；支持一键复现实验。
