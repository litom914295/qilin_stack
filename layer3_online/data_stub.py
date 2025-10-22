import pandas as pd, numpy as np
def get_daily_ohlc(symbol,start='2024-01-01'):
 d=pd.date_range(start, periods=120);
 price=10+np.cumsum(np.random.normal(0,0.1,len(d)));
 high=price+np.random.rand(len(d))*0.2; low=price-np.random.rand(len(d))*0.2; vol=np.random.randint(1e6,5e6,len(d));
 return pd.DataFrame({'date':d,'open':price,'high':high,'low':low,'close':price,'vol':vol})
