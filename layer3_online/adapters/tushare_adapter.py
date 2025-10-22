
import pandas as pd, os
def _ts():
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("请设置环境变量 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()

def get_daily_ohlc(symbol: str, start: str = "20240101", end: str = None) -> pd.DataFrame:
    try:
        pro = _ts()
        ts_code = symbol
        df = pro.daily(ts_code=ts_code, start_date=start.replace("-",""), end_date=None if end is None else end.replace("-",""))
        df = df.rename(columns={"trade_date":"date","open":"open","high":"high","low":"low","close":"close","vol":"vol"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date","open","high","low","close","vol"]].sort_values("date")
    except Exception:
        return pd.DataFrame(columns=["date","open","high","low","close","vol"])

def get_limit_up_pool(the_date: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol"])

def get_lhb(the_date: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol","netbuy"])

def get_news_titles(symbol: str, days: int = 3) -> list:
    return []
