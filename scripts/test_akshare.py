"""
AKShare数据测试脚本
"""
import sys
import time

try:
    import akshare as ak
    print("✅ AKShare导入成功")
    print(f"   版本: {ak.__version__}")
except ImportError:
    print("❌ AKShare未安装，请运行: pip install akshare")
    sys.exit(1)


def test_akshare():
    """测试AKShare功能"""
    print(f"\n{'='*60}")
    print("AKShare功能测试")
    print(f"{'='*60}\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # 测试1: 实时行情
    print("1️⃣ 测试实时行情...")
    try:
        df = ak.stock_zh_a_spot_em()
        print(f"   ✅ 获取成功，股票数量: {len(df)}")
        print(f"   数据列: {list(df.columns)}")
        print(f"\n   示例数据:")
        print(df[['代码', '名称', '最新价', '涨跌幅', '成交量']].head(3))
        tests_passed += 1
        time.sleep(1)  # 避免请求过快
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        tests_failed += 1
    
    # 测试2: 历史数据
    print("\n2️⃣ 测试历史数据...")
    try:
        df = ak.stock_zh_a_hist(
            symbol="000001",
            period="daily",
            start_date="20240101",
            end_date="20240630",
            adjust="qfq"
        )
        print(f"   ✅ 获取成功，数据量: {len(df)}")
        print(f"   日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
        print(f"\n   最近5天:")
        print(df[['日期', '开盘', '收盘', '最高', '最低', '成交量']].tail())
        tests_passed += 1
        time.sleep(1)
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        tests_failed += 1
    
    # 测试3: 指数数据
    print("\n3️⃣ 测试指数数据...")
    try:
        df = ak.stock_zh_index_spot()
        print(f"   ✅ 获取成功，指数数量: {len(df)}")
        # 查找主要指数
        major_indices = df[df['代码'].isin(['000001', '399001', '399006'])]
        print(f"\n   主要指数:")
        print(major_indices[['代码', '名称', '最新价', '涨跌幅']])
        tests_passed += 1
        time.sleep(1)
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        tests_failed += 1
    
    # 测试4: 涨停板数据
    print("\n4️⃣ 测试涨停板数据...")
    try:
        df = ak.stock_zt_pool_em(date="20240630")
        print(f"   ✅ 获取成功，涨停股票: {len(df)}")
        if len(df) > 0:
            print(f"\n   示例:")
            print(df[['代码', '名称', '涨停价', '首次封板时间', '封板资金']].head(3))
        tests_passed += 1
        time.sleep(1)
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        tests_failed += 1
    
    # 测试5: 个股资金流
    print("\n5️⃣ 测试个股资金流...")
    try:
        df = ak.stock_individual_fund_flow(symbol="000001", market="sz")
        print(f"   ✅ 获取成功，数据量: {len(df)}")
        if len(df) > 0:
            print(f"\n   最近数据:")
            print(df[['日期', '收盘价', '主力净流入', '超大单净流入', '大单净流入']].head(3))
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        tests_failed += 1
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"测试完成: {tests_passed} 通过, {tests_failed} 失败")
    
    if tests_failed == 0:
        print("✅ 所有测试通过！AKShare工作正常")
    else:
        print("⚠️ 部分测试失败，可能是网络问题或API限流")
        print("   建议: 稍后重试或检查网络连接")
    print(f"{'='*60}\n")
    
    return tests_failed == 0


def test_rate_limiting():
    """测试API限流"""
    print("\n6️⃣ 测试API限流...")
    print("   连续请求10次，观察响应...")
    
    success_count = 0
    start_time = time.time()
    
    for i in range(10):
        try:
            df = ak.stock_zh_a_spot_em()
            success_count += 1
            print(f"   请求 {i+1}/10: ✅ 成功 (数据量: {len(df)})")
            time.sleep(1)  # 控制频率
        except Exception as e:
            print(f"   请求 {i+1}/10: ❌ 失败 ({e})")
    
    elapsed = time.time() - start_time
    print(f"\n   总耗时: {elapsed:.2f}秒")
    print(f"   成功率: {success_count}/10 ({success_count*10}%)")
    
    if success_count >= 8:
        print("   ✅ API稳定性良好")
    else:
        print("   ⚠️ API可能不稳定或触发限流")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AKShare功能测试')
    parser.add_argument('--rate-limit', action='store_true', help='测试API限流')
    
    args = parser.parse_args()
    
    if test_akshare():
        if args.rate_limit:
            test_rate_limiting()
    else:
        print("\n💡 提示:")
        print("1. 检查网络连接")
        print("2. 确保AKShare版本最新: pip install --upgrade akshare")
        print("3. 如果持续失败，可能是API暂时不可用")
