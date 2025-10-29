"""
测试网络连接和akshare
"""
import os
import sys

print("=" * 60)
print("网络连接测试")
print("=" * 60)

# 1. 显示当前代理设置
print("\n1. 当前环境变量中的代理设置:")
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    value = os.environ.get(key, "未设置")
    print(f"   {key}: {value}")

# 2. 测试基本的网络连接
print("\n2. 测试基本网络连接:")
import requests

# 清除代理
print("   清除所有代理设置...")
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

# 测试连接百度
print("\n3. 测试连接百度 (国内网站):")
try:
    response = requests.get("https://www.baidu.com", timeout=5, proxies={'http': None, 'https': None})
    print(f"   ✓ 成功! 状态码: {response.status_code}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 测试连接东方财富
print("\n4. 测试连接东方财富API:")
try:
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://quote.eastmoney.com/',
    }
    response = requests.get(url, headers=headers, timeout=5, proxies={'http': None, 'https': None})
    print(f"   ✓ 成功! 状态码: {response.status_code}")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 5. 测试 akshare
print("\n5. 测试 akshare 库:")
try:
    import akshare as ak
    print("   akshare 已安装")
    
    # 尝试获取一个简单的数据
    print("   尝试获取股票列表...")
    df = ak.stock_zh_a_spot_em()
    print(f"   ✓ 成功! 获取到 {len(df)} 条数据")
except ImportError:
    print("   ✗ akshare 未安装")
except Exception as e:
    print(f"   ✗ 获取数据失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)