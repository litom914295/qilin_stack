
import os, csv, time, requests, hmac, hashlib, json, secrets
from dataclasses import dataclass
@dataclass
class Order:
    symbol: str; side: str; qty: int; price: float

class OrderGateway:
    def __init__(self,cfg): self.cfg=cfg; self.mode=cfg.get('mode','csv')
    def place(self,o:Order):
        if self.mode=='csv': self._csv(o)
        elif self.mode=='http': self._http(o)
    def _csv(self,o):
        outdir=self.cfg.get('csv',{}).get('dir','orders'); os.makedirs(outdir,exist_ok=True)
        day=time.strftime('%Y-%m-%d'); path=os.path.join(outdir,day,'orders.csv'); os.makedirs(os.path.dirname(path),exist_ok=True)
        new=not os.path.exists(path)
        with open(path,'a',newline='',encoding='utf-8') as f:
            w=csv.writer(f); 
            if new: w.writerow(['ts','symbol','side','qty','price'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'),o.symbol,o.side,o.qty,f'{o.price:.3f}'])
        if self.cfg.get('security',{}).get('enable_sig',True):
            sig_path=path+'.sig'
            with open(path,'rb') as f: data=f.read()
            sha=hashlib.sha256(data).hexdigest()
            open(sig_path,'w',encoding='utf-8').write(sha)
    def _http(self,o):
        ep=self.cfg.get('http',{}).get('endpoint'); key=self.cfg.get('http',{}).get('api_key',''); sec=self.cfg.get('http',{}).get('api_secret','')
        payload={'symbol':o.symbol,'side':o.side,'qty':o.qty,'price':o.price}
        headers={'Authorization':f'Bearer {key}'} if key else {}
        if self.cfg.get('security',{}).get('enable_sig',True) and sec:
            ts=str(int(time.time())); nonce=secrets.token_hex(8); body=json.dumps(payload,separators=(',',':'))
            msg='|'.join([ts,nonce,body]).encode('utf-8'); sign=hmac.new(sec.encode('utf-8'),msg,hashlib.sha256).hexdigest()
            headers.update({'X-Timestamp':ts,'X-Nonce':nonce,'X-Signature':sign,'Content-Type':'application/json'})
        r=requests.post(ep,json=payload,headers=headers,timeout=5); r.raise_for_status()
