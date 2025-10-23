import asyncio
import json
from datetime import datetime
from typing import List

from backtest.engine import BacktestEngine, BacktestConfig

async def main(symbols: List[str], start: str, end: str, data_source):
    engine = BacktestEngine(BacktestConfig(initial_capital=1_000_000, fill_model='prob'))
    metrics = await engine.run_backtest(
        symbols=symbols,
        start_date=start,
        end_date=end,
        data_source=data_source,
        trade_at='next_open',
        avoid_limit_up_unfillable=True,
    )
    engine.print_summary(metrics)
    # 保存报告
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'params': {'symbols': symbols, 'start': start, 'end': end},
        'metrics': metrics,
    }
    with open(f'reports/range_backtest_{ts}.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # 示例：用户可自行替换 data_source（DataFrame，包含 date/symbol/close/open/high/low/volume）
    import pandas as pd
    # 这里简单合成两只股票的示例数据（实际可接入你的数据管道）
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
    def synth(sym):
        import numpy as np
        price = 10 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
        return pd.DataFrame({
            'symbol': sym,
            'date': dates,
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': np.random.randint(1_000_000, 5_000_000, len(dates)),
        })
    df = pd.concat([synth('000001.SZ'), synth('600519.SH')])
    asyncio.run(main(['000001.SZ','600519.SH'], '2024-01-01', '2024-06-30', df))
