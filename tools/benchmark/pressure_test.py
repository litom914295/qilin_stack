
import time, subprocess, sys, os, pandas as pd
from datetime import datetime
def main():
    day=datetime.now().strftime('%Y%m%d')
    src=f'layer2_qlib/artifacts/preds_{day}.csv'
    if not os.path.exists(src):
        print('请先生成 preds：python layer2_qlib/scripts/predict_online_qlib.py'); return
    df=pd.read_csv(src); big=pd.concat([df]*10,ignore_index=True); big.to_csv(src.replace('.csv','_x10.csv'),index=False)
    t0=time.time(); code=subprocess.call([sys.executable,'integrations/tradingagents_cn/run_workflow.py','--topk','2']); dt=time.time()-t0
    print('run_workflow 耗时：', round(dt,3),'秒；返回码=',code)
if __name__=='__main__': main()
