param(
  [string]$Start = '2024-01-01',
  [string]$End   = '2024-06-30',
  [string[]]$Symbols = @('000001.SZ','600519.SH')
)

Write-Host "Running range backtest from $Start to $End for: $($Symbols -join ', ')" -ForegroundColor Cyan

# 合成示例数据（如果你有真实数据，请改为你的数据加载逻辑）
$py = @'
import asyncio, sys, json
import pandas as pd, numpy as np
from backtest.engine import BacktestEngine, BacktestConfig

symbols = json.loads(sys.argv[1])
start = sys.argv[2]
end = sys.argv[3]

# synth data
dates = pd.date_range(start, end, freq='B')

def synth(sym):
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

df = pd.concat([synth(s) for s in symbols])

async def main():
    engine = BacktestEngine(BacktestConfig(initial_capital=1_000_000, fill_model='prob'))
    m = await engine.run_backtest(symbols, start, end, df, trade_at='next_open', avoid_limit_up_unfillable=True)
    engine.print_summary(m)

asyncio.run(main())
'@

python - <<PY @($Symbols | ConvertTo-Json -Compress) $Start $End
$py
PY
