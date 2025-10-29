"""
测试使用同花顺数据源获取涨停板数据
"""
import os
import sys

# 清除所有代理
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

print("测试同花顺数据源...")
print("=" * 60)

try:
    import akshare as ak
    
    # 方法1：尝试获取涨停板统计
    print("1. 尝试获取涨停板统计数据...")
    try:
        # 获取涨停板行情
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20241001", end_date="20241028", adjust="")
        print(f"   ✓ 获取历史数据成功: {len(df)} 条")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # 方法2：尝试使用新浪数据源
    print("\n2. 尝试使用新浪数据源...")
    try:
        df = ak.stock_zh_a_spot()  # 新浪实时行情
        print(f"   ✓ 获取实时行情成功: {len(df)} 条")
        # 筛选涨停股（涨幅>=9.5%）
        limitup = df[df['涨跌幅'] >= 9.5]
        print(f"   ✓ 找到涨停股: {len(limitup)} 只")
        if len(limitup) > 0:
            print("   前5只涨停股:")
            for idx, row in limitup.head(5).iterrows():
                print(f"      - {row['名称']} ({row['代码']}): {row['涨跌幅']:.2f}%")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # 方法3：使用腾讯数据源
    print("\n3. 尝试使用腾讯数据源...")
    try:
        import requests
        # 腾讯实时行情接口
        url = "http://qt.gtimg.cn/q=sh000001"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, proxies={'http': '', 'https': ''})
        print(f"   ✓ 腾讯接口响应: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        
except ImportError:
    print("akshare 未安装")
except Exception as e:
    print(f"错误: {e}")

print("=" * 60)