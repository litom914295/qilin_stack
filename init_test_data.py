#!/usr/bin/env python
"""
测试数据初始化脚本
为因子库生成10个Mock因子用于测试
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from web.tabs.rdagent.factor_library import FactorLibraryDB
from datetime import datetime, timedelta
import random
import numpy as np


def generate_mock_factors(num_factors: int = 10):
    """生成Mock测试因子"""
    
    factor_names = [
        "momentum_ma20", "volume_price_corr", "rsi_divergence",
        "bollinger_width", "macd_signal", "atr_ratio",
        "volume_momentum", "price_acceleration", "liquidity_factor",
        "sentiment_score", "volatility_ratio", "bid_ask_spread",
        "order_imbalance", "tick_direction", "volume_weighted_price"
    ]
    
    factor_types = ["技术因子", "基本面因子", "量价因子", "情绪因子", "混合因子"]
    
    formulations = [
        "(close - ma(close, 20)) / ma(close, 20)",
        "corr(volume, close, 10)",
        "rsi(14) - rsi(7)",
        "(upper_band - lower_band) / close",
        "macd(12, 26, 9)",
        "atr(14) / close",
        "volume / ma(volume, 20)",
        "delta(close, 2) / delta(close, 1)",
        "volume / volatility(20)"
    ]
    
    factors = []
    
    for i in range(num_factors):
        factor = {
            'name': factor_names[i % len(factor_names)] + f"_v{i//len(factor_names) + 1}",
            'type': random.choice(factor_types),
            'description': f"自动生成的测试因子 #{i+1}",
            'formulation': formulations[i % len(formulations)],
            'code': f"""def factor_{i}(data):
    import pandas as pd
    import numpy as np
    
    # 计算{factor_names[i % len(factor_names)]}
    result = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
    return result.fillna(0)
""",
            'ic': round(random.uniform(0.03, 0.18), 3),
            'ir': round(random.uniform(0.2, 0.9), 3),
            'sharpe': round(random.uniform(0.8, 2.5), 3),
            'annual_return': round(random.uniform(0.05, 0.35), 3),
            'max_drawdown': round(random.uniform(0.05, 0.25), 3),
            'turnover': round(random.uniform(0.1, 0.8), 3),
            'valid': random.random() > 0.2,  # 80%有效
            'tags': [
                'test_data',
                factor_types[i % len(factor_types)].replace('因子', ''),
                random.choice(['momentum', 'reversal', 'volume', 'volatility'])
            ],
            'metadata': {
                'test_version': '1.0',
                'generated_by': 'init_test_data.py',
                'test_date': str(datetime.now()),
                'sample_data': True
            }
        }
        
        factors.append(factor)
    
    return factors


def init_database():
    """初始化数据库并添加测试数据"""
    
    print("=" * 60)
    print("🚀 初始化因子库测试数据")
    print("=" * 60)
    
    try:
        # 创建数据库实例
        print("\n📂 创建数据库连接...")
        db = FactorLibraryDB()
        print("✅ 数据库连接成功!")
        
        # 检查现有因子数量
        stats = db.get_factor_stats()
        print(f"\n📊 当前因子库状态:")
        print(f"  - 总因子数: {stats['total']}")
        print(f"  - 有效因子: {stats['valid']}")
        print(f"  - 平均IC: {stats['avg_ic']:.3f}")
        
        # 生成测试因子
        print("\n🎲 生成10个测试因子...")
        factors = generate_mock_factors(10)
        print(f"✅ 生成完成! 共 {len(factors)} 个因子")
        
        # 保存到数据库
        print("\n💾 保存因子到数据库...")
        saved_count = 0
        failed_count = 0
        
        for i, factor in enumerate(factors, 1):
            try:
                factor_id = db.save_factor(factor)
                saved_count += 1
                status = "✅" if factor['valid'] else "⚠️"
                print(f"  {status} [{i}/10] {factor['name']}: IC={factor['ic']:.3f}, IR={factor['ir']:.3f}")
            except Exception as e:
                failed_count += 1
                print(f"  ❌ [{i}/10] {factor['name']}: 保存失败 - {str(e)[:50]}")
        
        # 显示结果
        print("\n" + "=" * 60)
        print("📊 初始化完成!")
        print("=" * 60)
        print(f"✅ 成功保存: {saved_count} 个")
        if failed_count > 0:
            print(f"❌ 保存失败: {failed_count} 个")
        
        # 显示更新后的统计
        stats_after = db.get_factor_stats()
        print(f"\n📈 更新后的因子库状态:")
        print(f"  - 总因子数: {stats_after['total']} (+{stats_after['total'] - stats['total']})")
        print(f"  - 有效因子: {stats_after['valid']} (+{stats_after['valid'] - stats['valid']})")
        print(f"  - 平均IC: {stats_after['avg_ic']:.3f}")
        print(f"  - 最佳IC: {stats_after['max_ic']:.3f}")
        
        # 显示因子类型分布
        if stats_after['type_distribution']:
            print(f"\n📊 因子类型分布:")
            for factor_type, count in stats_after['type_distribution'].items():
                print(f"  - {factor_type}: {count}个")
        
        print("\n💡 提示: 现在可以启动Web界面查看因子库!")
        print("   运行命令: python start_web.py")
        print("   访问: http://localhost:8501")
        print("   导航: RD-Agent → 因子挖掘 → 📚 因子库管理")
        
        print("\n" + "=" * 60)
        print("🎉 测试数据初始化完成!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def clear_test_data():
    """清除测试数据"""
    
    print("=" * 60)
    print("🗑️  清除测试数据")
    print("=" * 60)
    
    try:
        db = FactorLibraryDB()
        
        # 查找测试数据
        factors = db.get_factors(limit=1000)
        test_factors = [f for f in factors if 'test_data' in f.get('tags', [])]
        
        if not test_factors:
            print("\n✅ 没有找到测试数据")
            return True
        
        print(f"\n📋 找到 {len(test_factors)} 个测试因子")
        
        # 确认删除
        response = input(f"\n⚠️  确认删除这些测试因子? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ 取消删除")
            return False
        
        # 删除测试因子
        deleted_count = 0
        for factor in test_factors:
            try:
                db.delete_factor(factor['id'])
                deleted_count += 1
                print(f"  ✅ 删除: {factor['name']}")
            except Exception as e:
                print(f"  ❌ 删除失败 {factor['name']}: {e}")
        
        print(f"\n✅ 成功删除 {deleted_count} 个测试因子")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 清除失败: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="因子库测试数据初始化")
    parser.add_argument('--clear', action='store_true', help='清除测试数据')
    parser.add_argument('--num', type=int, default=10, help='生成因子数量 (默认10)')
    
    args = parser.parse_args()
    
    if args.clear:
        clear_test_data()
    else:
        # 如果指定了数量,使用自定义数量
        if args.num != 10:
            def generate_custom_factors():
                return generate_mock_factors(args.num)
            globals()['generate_mock_factors'] = generate_custom_factors
        
        init_database()
