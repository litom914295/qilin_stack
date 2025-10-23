# 执行口径与撮合模型说明

本项目的回测执行支持更贴近实盘的 T+1 开盘成交与撮合建模。

## 成交口径（trade_at）
- `next_open`（默认）：次日开盘价撮合；如开盘近似涨停（一字/≥+9.8%），判定不可成交。
- `same_day_close`：同日收盘成交（理想化，便于对比）。

## 成交模型（fill_model）
- `deterministic`（默认）：
  - 基于前一日特征（封板质量/量能突增/题材热度/连板高度）计算确定性成交比例 `fill_ratio∈[0,1]`。
  - 无随机性，回测结果可重复。
- `prob`（概率模型，可复现）：
  - 在确定性基础上引入稳定噪声（MD5 哈希 → [0,1)），实现更贴近真实的“部分成交/滑点”效果，同时保持可复现。

## 示例
```python
from backtest.engine import BacktestEngine, BacktestConfig
engine = BacktestEngine(BacktestConfig(initial_capital=1_000_000, fill_model='prob'))
metrics = await engine.run_backtest(
    symbols=['000001.SZ','600519.SH'],
    start_date='2024-01-01', end_date='2024-06-30', data_source=df,
    trade_at='next_open', avoid_limit_up_unfillable=True,
)
```
