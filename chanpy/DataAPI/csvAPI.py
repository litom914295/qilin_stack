"""CSV数据源适配器 - 用于Chan.py集成"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataAPI.CommonStockAPI import CCommonStockApi
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit
import pandas as pd
import os

class CSV_API(CCommonStockApi):
    """CSV文件数据源"""
    
    def __init__(self, code, k_type, begin_date, end_date, autype):
        super().__init__(code, k_type, begin_date, end_date, autype)
        # 使用临时目录存储CSV
        self.csv_path = f'G:/test/qilin_stack/temp/chanpy_{code}.csv'
    
    @classmethod
    def do_init(cls):
        """初始化"""
        # 确保临时目录存在
        os.makedirs('G:/test/qilin_stack/temp', exist_ok=True)
    
    @classmethod
    def do_close(cls):
        """关闭"""
        pass
    
    def get_kl_data(self):
        """读取CSV数据"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        for idx, row in df.iterrows():
            try:
                # 解析时间
                time = CTime.from_str(str(row['datetime']))
                
                # 创建K线单元
                klu = CKLine_Unit({
                    'time': time,
                    'open': float(row['open']),
                    'close': float(row['close']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'volume': float(row.get('volume', 0)),
                    'turnover': float(row.get('amount', 0)),
                })
                
                yield klu
                
            except Exception as e:
                print(f"Warning: 跳过行 {idx}, 错误: {e}")
                continue
