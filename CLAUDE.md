# CLAUDE.md — 项目说明与操作手册（qilin_stack 定制版）

## 环境
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# 需要高级能力时：
pip install -r requirements-extra.txt
```

## 数据（Qlib）
```bash
python scripts/get_data.py --source qlib
# 或
python main.py --action prepare_data
```
默认路径：`~/.qlib/qlib_data/cn_data`（请预留足够磁盘）。

## 运行
- 盘后复盘：`python main.py --mode replay --date YYYY-MM-DD`
- 区间模拟：`python main.py --mode simulation --start 2024-01-01 --end 2024-06-30`
- Web 面板：`python start_web.py` 或 `streamlit run web/unified_dashboard.py`

## 配置
- 全局：`config/default.yaml`
  - `market_regimes`：按牛/熊/震荡切换 Agent 权重与风控
- 因子与策略：
  - `factors/` 新增 `onein2.py`
  - `config/factor_onein2.yaml` 参数网格/过滤开关
  - `strategy/` 遵循“score & confidence”双阈值门锁

## 多智能体
- TradingAgents（`integrations/tradingagents_cn/`, `tradingagents_integration/`）
- RD-Agent（`rd_agent/`）：因子发现→实现→单测→对比→归档

## 观测
```bash
cd docker
docker compose --profile observability up -d
# Prometheus: :9090, Grafana: :3000（默认端口）
```

## 质量与复现
- `ruff`/`black`/`isort`/`pyright`
- 输出落地 `reports/`，保留复现脚本
- 研究与教育用途，严禁实盘
