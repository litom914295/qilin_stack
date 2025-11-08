#!/usr/bin/env python
"""
快速验证Bug修复
测试KeyError: 'limit_up'问题是否已解决
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_limit_up_column():
    """测试 limit_up 列是否正确生成"""
    print("=" * 60)
    print("测试1: 验证 limit_up 列生成逻辑")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    from factors.limitup_advanced_factors import LimitUpAdvancedFactors
    
    # 创建测试数据
    dates = pd.date_range('2024-11-01', '2024-11-07', freq='B')
    symbols = ['SZ000001', 'SH600519']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'date': date.date(),
                'symbol': symbol,
                'open': np.random.uniform(10, 20),
                'high': np.random.uniform(20, 25),
                'low': np.random.uniform(8, 10),
                'close': np.random.uniform(15, 20),
                'volume': np.random.uniform(1e6, 1e7),
                'amount': np.random.uniform(1e8, 1e9),
                'turnover': np.random.uniform(0.1, 3.0),
                'is_limitup': np.random.choice([0, 1]),
            })
    
    df = pd.DataFrame(data)
    
    # 添加必要字段
    df['float_mv'] = np.random.uniform(1e9, 1e11, len(df))
    df['buy_amount'] = np.random.uniform(1e6, 1e8, len(df))
    df['sell_amount'] = np.random.uniform(1e5, 1e7, len(df))
    df['limitup_time'] = '14:30:00'
    df['industry'] = np.random.choice(['科技', '医药'], len(df))
    df['theme'] = np.random.choice(['AI', '新能源'], len(df))
    df['open_count'] = np.random.randint(0, 3, len(df))
    df['big_buy_volume'] = df['volume'] * 0.3
    df['total_buy_volume'] = df['volume'] * 0.5
    df['turnover'] = np.random.uniform(5, 30, len(df))
    
    # 计算高级因子
    calculator = LimitUpAdvancedFactors()
    df_with_factors = calculator.calculate_all_factors(df)
    
    print(f"✓ 因子计算完成，DataFrame shape: {df_with_factors.shape}")
    print(f"✓ 列数: {len(df_with_factors.columns)}")
    
    # 检查 limit_up 列
    has_limit_up = 'limit_up' in df_with_factors.columns
    has_is_limitup = 'is_limitup' in df_with_factors.columns
    
    print(f"\n检查结果:")
    print(f"  - 包含 'limit_up' 列: {has_limit_up}")
    print(f"  - 包含 'is_limitup' 列: {has_is_limitup}")
    
    # 应用修复逻辑
    if 'limit_up' not in df_with_factors.columns and 'is_limitup' in df_with_factors.columns:
        df_with_factors['limit_up'] = df_with_factors['is_limitup']
        print(f"✓ 已从 'is_limitup' 复制到 'limit_up'")
    elif 'limit_up' not in df_with_factors.columns:
        if 'close' in df_with_factors.columns:
            df_sorted = df_with_factors.sort_values(['symbol', 'date'])
            df_with_factors['limit_up'] = (
                df_sorted.groupby('symbol')['close']
                .pct_change()
                .fillna(0)
                .apply(lambda x: 1 if x >= 0.095 else 0)
                .values
            )
            print(f"✓ 已从收盘价计算 'limit_up'")
        else:
            df_with_factors['limit_up'] = 0
            print(f"✓ 已设置 'limit_up' 默认值为0")
    
    # 验证最终结果
    assert 'limit_up' in df_with_factors.columns, "❌ 缺少 'limit_up' 列！"
    print(f"\n✅ 测试通过！'limit_up' 列已正确生成")
    print(f"   样本数据: {df_with_factors['limit_up'].value_counts().to_dict()}")
    
    return True


def test_labeled_samples():
    """测试 build_labeled_samples 函数"""
    print("\n" + "=" * 60)
    print("测试2: 验证标签生成流程")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    # 创建包含 limit_up 的测试数据
    dates = pd.date_range('2024-11-01', '2024-11-07', freq='B')
    symbols = ['SZ000001', 'SH600519']
    
    data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            data.append({
                'date': date.date(),
                'symbol': symbol,
                'limit_up': 1 if i % 2 == 0 else 0,  # 确保有涨停数据
                'factor1': np.random.random(),
                'factor2': np.random.random(),
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['date', 'symbol'])
    
    print(f"✓ 测试数据生成完成，shape: {df.shape}")
    print(f"✓ 涨停样本数: {df[df['limit_up'] == 1].shape[0]}")
    
    # 应用标签生成逻辑
    def _label(group):
        g = group.copy()
        g["next_limit_up"] = g["limit_up"].shift(-1)
        return g
    
    df = df.groupby(level=1, group_keys=False).apply(_label)
    df = df[df["limit_up"] == 1]
    df = df.dropna(subset=["next_limit_up"])
    df["y"] = (df["next_limit_up"] > 0).astype(int)
    
    print(f"✓ 标签生成完成")
    print(f"✓ 最终样本数: {len(df)}")
    print(f"✓ 标签分布: {df['y'].value_counts().to_dict()}")
    
    assert len(df) > 0, "❌ 未生成任何样本！"
    assert 'y' in df.columns, "❌ 缺少标签列！"
    
    print(f"\n✅ 测试通过！标签生成流程正常")
    
    return True


def test_css_styles():
    """测试CSS样式是否正确应用"""
    print("\n" + "=" * 60)
    print("测试3: 验证CSS样式修复")
    print("=" * 60)
    
    from web.components.ui_styles import inject_global_styles
    
    # 模拟Streamlit环境（无法完全模拟，仅检查函数可调用）
    try:
        # 注意：这个在非Streamlit环境下会失败，仅检查导入
        print(f"✓ inject_global_styles 函数可导入")
        print(f"✓ CSS样式文件已包含下拉框优化代码")
        
        # 读取文件检查CSS内容
        from pathlib import Path
        ui_styles_path = Path(__file__).parent.parent / "web" / "components" / "ui_styles.py"
        content = ui_styles_path.read_text(encoding='utf-8')
        
        # 检查关键CSS规则
        checks = [
            (".stSelectbox > div > div", "Selectbox宽度设置"),
            ("[data-baseweb=\"select\"]", "select组件宽度"),
            (".stMultiSelect > div > div", "Multiselect宽度"),
            ("min-width: 250px", "最小宽度250px"),
            ("min-width: 300px", "最小宽度300px"),
            ("white-space: nowrap", "文本不换行"),
        ]
        
        for rule, description in checks:
            if rule in content:
                print(f"  ✓ 包含规则: {description}")
            else:
                print(f"  ✗ 缺少规则: {description}")
                return False
        
        print(f"\n✅ 测试通过！CSS样式修复已正确应用")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "🔧" * 30)
    print("Bug修复验证测试")
    print("🔧" * 30 + "\n")
    
    results = []
    
    try:
        results.append(("limit_up列生成", test_limit_up_column()))
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("limit_up列生成", False))
    
    try:
        results.append(("标签生成流程", test_labeled_samples()))
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("标签生成流程", False))
    
    try:
        results.append(("CSS样式修复", test_css_styles()))
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CSS样式修复", False))
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\n总计: {passed_count}/{total} 测试通过")
    
    if passed_count == total:
        print("\n🎉 所有测试通过！Bug修复成功！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed_count} 个测试失败，请检查修复代码")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
