
import yaml, os
from . import akshare_adapter as ak_ad
from . import tushare_adapter as ts_ad
from layer3_online.data_stub import get_daily_ohlc as stub_ohlc

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
_cfg = yaml.safe_load(open(_cfg_path,"r",encoding="utf-8")) if os.path.exists(_cfg_path) else {"source":"stub"}

def get_ohlc(symbol: str, start: str="2024-01-01"):
    src = _cfg.get("source","auto")
    if src in ("akshare","auto"):
        df = ak_ad.get_daily_ohlc(symbol, start=start)
        if len(df)>0: return df
    if src in ("tushare","auto"):
        df = ts_ad.get_daily_ohlc(symbol, start=start)
        if len(df)>0: return df
    return stub_ohlc(symbol, start=start)
