
from pathlib import Path
import pandas as pd
def load_preds(yyyymmdd: str) -> dict:
    p = Path('layer2_qlib/artifacts')/f'preds_{yyyymmdd}.csv'
    if not p.exists(): return {}
    df = pd.read_csv(p)
    return {str(r.symbol): float(r.score) for r in df.itertuples()}
