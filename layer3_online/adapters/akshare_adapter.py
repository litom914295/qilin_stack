
import pandas as pd
def get_daily_ohlc(symbol: str, start: str = "2024-01-01", end: str = None) -> pd.DataFrame:
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start.replace("-",""), end_date=None if end is None else end.replace("-",""), adjust="qfq")
        df = df.rename(columns={"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"vol"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date","open","high","low","close","vol"]]
    except Exception:
        return pd.DataFrame(columns=["date","open","high","low","close","vol"])

def get_limit_up_pool(the_date: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol"])

def get_lhb(the_date: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol","netbuy"])

def get_news_titles(symbol: str, days: int = 3) -> list:
    return []
