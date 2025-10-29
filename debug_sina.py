"""
调试新浪API
"""
import os
import requests

# 清除代理
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

print("调试新浪API")
print("=" * 60)

# 测试新浪API的各种URL
urls = [
    "http://hq.sinajs.cn/list=sh000001",
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=80&sort=changepercent&asc=0&node=hs_a",
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_Transactions.getAllPageTime?date=2024-10-28&symbol=sh000001",
]

for url in urls:
    print(f"\n测试: {url[:80]}...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'http://finance.sina.com.cn',
        }
        
        # 使用Session并明确禁用代理
        session = requests.Session()
        session.trust_env = False
        
        response = session.get(url, headers=headers, timeout=5)
        print(f"  状态码: {response.status_code}")
        print(f"  内容类型: {response.headers.get('content-type', 'unknown')}")
        print(f"  内容预览: {response.text[:200]}...")
        
    except Exception as e:
        print(f"  错误: {e}")

print("\n" + "=" * 60)

# 直接测试akshare内部使用的URL
print("\n测试akshare内部URL...")
try:
    import akshare as ak
    
    # 查看akshare的源码，找到实际使用的URL
    print("\n尝试直接调用akshare的内部函数...")
    
    # 手动构造请求
    url = "http://33.push.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeDataSimple"
    params = {
        "page": 1,
        "num": 80,
        "sort": "changepercent",
        "asc": 0,
        "node": "hs_a",
        "_s_r_a": "page"
    }
    
    session = requests.Session()
    session.trust_env = False
    
    response = session.get(url, params=params, timeout=10)
    print(f"状态码: {response.status_code}")
    print(f"内容预览: {response.text[:300]}...")
    
except Exception as e:
    print(f"错误: {e}")