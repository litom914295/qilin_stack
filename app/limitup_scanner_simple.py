"""
简化版涨停股扫描脚本
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# 清除所有代理设置
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_limitup_stocks():
    """扫描涨停股"""
    try:
        import requests
        import json
        
        logger.info("正在获取实时行情...")
        
        # 直接使用可用的新浪 API
        all_stocks = []
        
        # 分页获取所有股票
        for page in range(1, 100):  # 最多100页
            url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
            params = {
                "page": page,
                "num": 80,
                "sort": "changepercent",
                "asc": 0,
                "node": "hs_a"
            }
            
            session = requests.Session()
            session.trust_env = False  # 忽略系统代理
            
            response = session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                break
                
            # 解析JSON
            text = response.text
            if not text or text == "null":
                break
                
            # 新浪返回的是不标准的JSON，需要处理
            try:
                # 使用eval来解析（注意：生产环境应使用更安全的方法）
                import ast
                data = ast.literal_eval(text)
            except:
                # 如果不能直接eval，尝试修复并解析
                import re
                # 将单引号替换为双引号
                text = text.replace("'", '"')
                # 处理可能的中文字符
                data = json.loads(text)
            
            if not data:
                break
                
            all_stocks.extend(data)
            
            # 如果返回的数据少于请求的数量，说明已经到最后一页
            if len(data) < 80:
                break
        
        logger.info(f"获取到 {len(all_stocks)} 只股票")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_stocks)
        
        # 筛选涨停股 (涨幅 >= 9.5%)
        df['changepercent'] = pd.to_numeric(df['changepercent'], errors='coerce')
        limitup = df[df['changepercent'] >= 9.5].copy()
        
        # 过滤ST股
        limitup = limitup[~limitup['name'].str.contains('ST|退', na=False)]
        
        logger.info(f"找到 {len(limitup)} 只涨停股")
        
        # 转换为统一格式
        results = []
        for _, row in limitup.iterrows():
            results.append({
                'symbol': row.get('code', row.get('symbol', '')),
                'name': row.get('name', ''),
                'price': float(row.get('trade', 0)),
                'change_pct': float(row.get('changepercent', 0)),
                'volume': int(float(row.get('volume', 0))),
                'amount': float(row.get('amount', 0)),
                'total_score': np.random.randint(60, 95),  # 模拟评分
                'rating': '⚠️ 一般',
                'recommendation': '谨慎观望',
                'scores': {
                    'time_score': np.random.randint(50, 100),
                    'seal_score': np.random.randint(50, 100),
                    'open_score': np.random.randint(50, 100),
                    'volume_score': np.random.randint(50, 100)
                }
            })
        
        return pd.DataFrame(results).sort_values('total_score', ascending=False)
        
    except Exception as e:
        logger.error(f"扫描失败: {e}")
        # 抛出异常，让调用方知道出错了
        raise

def scan_and_analyze_today():
    """主函数"""
    return scan_limitup_stocks()

if __name__ == "__main__":
    print("=" * 60)
    print("涨停股扫描")
    print("=" * 60)
    
    df = scan_and_analyze_today()
    
    if not df.empty:
        print(df.to_json(orient='records', force_ascii=False))