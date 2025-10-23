
import os
import json
import time
import pandas as pd
from typing import Optional

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "adapters", "akshare")
os.makedirs(_CACHE_DIR, exist_ok=True)
_last_call_ts = {}
_RATE_LIMIT_SEC = 0.5


def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_").replace(":", "_").replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.json")


def _cache_get(key: str):
    try:
        p = _cache_path(key)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None


def _cache_set(key: str, value):
    try:
        p = _cache_path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except Exception:
        pass


def _rate_limit(tag: str):
    now = time.time()
    last = _last_call_ts.get(tag, 0)
    if now - last < _RATE_LIMIT_SEC:
        time.sleep(_RATE_LIMIT_SEC - (now - last))
    _last_call_ts[tag] = time.time()

def get_daily_ohlc(symbol: str, start: str = "2024-01-01", end: str = None) -> pd.DataFrame:
    # 缓存键
    key = f"ohlc:{symbol}:{start}:{end}"
    cached = _cache_get(key)
    if cached is not None:
        try:
            return pd.DataFrame(cached)
        except Exception:
            pass
    try:
        import akshare as ak
        _rate_limit("ohlc")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start.replace("-",""), end_date=None if end is None else end.replace("-",""), adjust="qfq")
        df = df.rename(columns={"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"vol"})
        df["date"] = pd.to_datetime(df["date"])
        out = df[["date","open","high","low","close","vol"]]
        _cache_set(key, out.to_dict(orient="list"))
        return out
    except Exception:
        return pd.DataFrame(columns=["date","open","high","low","close","vol"])

def get_limit_up_pool(the_date: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol"])

def get_lhb(the_date: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol","netbuy"])

# 题材热度（概念热度）
# 返回该 symbol 当日所属概念中“涨停家数”的近似总和（无数据则返回0）
def get_concept_heat_for_symbol(symbol: str, the_date: str) -> float:
    key = f"concept_heat:{symbol}:{the_date}"
    c = _cache_get(key)
    if c is not None:
        try:
            return float(c)
        except Exception:
            pass
    try:
        import akshare as ak
        _rate_limit("concept")
        # 获取当日涨停池
        try:
            zt_df = ak.stock_zt_pool_em(date=the_date.replace("-",""))
            zt_syms = set(str(x) for x in zt_df.get('代码', []))
        except Exception:
            zt_syms = set()
        # 获取 symbol 概念列表
        try:
            cons = ak.stock_board_concept_cons_em()
        except Exception:
            cons = None
        if cons is None or cons.empty:
            _cache_set(key, 0.0)
            return 0.0
        heat = 0
        cons = cons[['板块代码','板块名称','股票代码']].rename(columns={'股票代码':'ts_code'})
        sym_cons = cons[cons['ts_code'].astype(str).str.contains(symbol.split('.')[0])]
        for _, row in sym_cons.iterrows():
            code = row['板块代码']
            try:
                cons_one = cons[cons['板块代码'] == code]
                cnt = sum(1 for s in cons_one['ts_code'].astype(str) if s in zt_syms)
                heat += cnt
            except Exception:
                continue
        _cache_set(key, float(heat))
        return float(heat)
    except Exception:
        _cache_set(key, 0.0)
        return 0.0

# 龙虎榜净买入（单位：元，近似）
def get_lhb_netbuy(symbol: str, the_date: str) -> float:
    key = f"lhb_netbuy:{symbol}:{the_date}"
    c = _cache_get(key)
    if c is not None:
        try:
            return float(c)
        except Exception:
            pass
    try:
        import akshare as ak
        _rate_limit("lhb")
        df = ak.stock_lhb_detail_em(date=the_date.replace("-",""))
        if df is None or df.empty:
            _cache_set(key, 0.0)
            return 0.0
        sym = symbol.split('.')[0]
        sub = df[df['代码'].astype(str) == sym]
        if sub.empty:
            _cache_set(key, 0.0)
            return 0.0
        for col in ['净额','净买入']:
            if col in sub.columns:
                try:
                    val = float(sub[col].astype(str).str.replace(',','').astype(float).sum())
                    _cache_set(key, val)
                    return val
                except Exception:
                    continue
        _cache_set(key, 0.0)
        return 0.0
    except Exception:
        _cache_set(key, 0.0)
        return 0.0

def get_news_titles(symbol: str, days: int = 3) -> list:
    return []
