
import os, json, pandas as pd, hashlib
from datetime import datetime
from layer3_online.adapters.selector import get_ohlc

def _hash(obj_str: str)->str:
    return hashlib.sha256(obj_str.encode('utf-8')).hexdigest()

def main():
    day=datetime.now().strftime('%Y-%m-%d')
    pre=f'output/preopen_{day}.csv'
    if os.path.exists(pre):
        syms=list(pd.read_csv(pre)['symbol'].astype(str).unique())[:50]
    else:
        syms=['SZ000001','SH600000']

    # adapter close
    rows=[]
    for s in syms:
        d=get_ohlc(s).tail(1)
        if len(d)>0: rows.append({'symbol':s,'close':float(d['close'].iloc[-1])})
    h_adapter=_hash(pd.DataFrame(rows).to_csv(index=False))

    # qlib preds
    preds=f'layer2_qlib/artifacts/preds_{datetime.now():%Y%m%d}.csv'
    if os.path.exists(preds):
        df=pd.read_csv(preds)
        df=df[df['symbol'].isin(syms)]
        h_qlib=_hash(df.to_csv(index=False))
    else:
        h_qlib='EMPTY'

    # rd-agent registry
    reg='layer2_qlib/artifacts/model_registry.json'
    if os.path.exists(reg):
        h_rd=_hash(open(reg,'r',encoding='utf-8').read())
    else:
        h_rd='EMPTY'

    os.makedirs('tools/data_consistency/logs',exist_ok=True)
    out={'date':day,'hash_adapter':h_adapter,'hash_qlib':h_qlib,'hash_rd':h_rd}
    open(f'tools/data_consistency/logs/check_{day}.json','w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=2))
    print(out)

if __name__=='__main__':
    main()
