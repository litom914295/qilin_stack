"""
P2-7 绩效归因分析系统 - 快速验证脚本
一键验证所有功能是否正常工作
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "qlib_enhanced"))

def verify_imports():
    """验证模块导入"""
    print("🔍 验证模块导入...")
    try:
        from performance_attribution import (
            BrinsonAttribution,
            FactorAttribution,
            TransactionCostAnalysis,
            create_sample_attribution_data,
            AttributionResult
        )
        print("  ✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def verify_brinson():
    """验证Brinson归因"""
    print("\n🔍 验证Brinson归因...")
    try:
        from performance_attribution import BrinsonAttribution, create_sample_attribution_data
        
        pw, pr, bw, br = create_sample_attribution_data()
        brinson = BrinsonAttribution(pw, pr, bw, br)
        result = brinson.analyze()
        
        # 验证结果类型
        assert hasattr(result, 'allocation_effect'), "缺少配置效应"
        assert hasattr(result, 'selection_effect'), "缺少选择效应"
        assert hasattr(result, 'interaction_effect'), "缺少交互效应"
        assert hasattr(result, 'total_active_return'), "缺少总超额收益"
        
        # 验证一致性
        total = result.allocation_effect + result.selection_effect + result.interaction_effect
        assert abs(total - result.total_active_return) < 1e-6, "归因分解不一致"
        
        print(f"  ✅ Brinson归因正常")
        print(f"     配置效应: {result.allocation_effect:.4f}")
        print(f"     选择效应: {result.selection_effect:.4f}")
        print(f"     交互效应: {result.interaction_effect:.4f}")
        return True
        
    except Exception as e:
        print(f"  ❌ Brinson归因失败: {e}")
        return False


def verify_factor_attribution():
    """验证因子归因"""
    print("\n🔍 验证因子归因...")
    try:
        import pandas as pd
        import numpy as np
        from performance_attribution import FactorAttribution
        
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.02, 50))
        factors = pd.DataFrame({
            'Market': np.random.normal(0.008, 0.015, 50),
            'Size': np.random.normal(0.002, 0.01, 50),
            'Value': np.random.normal(0.003, 0.01, 50)
        })
        
        factor_attr = FactorAttribution(returns, factors)
        contributions = factor_attr.analyze()
        
        # 验证结果
        assert 'Market' in contributions, "缺少市场因子"
        assert 'Size' in contributions, "缺少规模因子"
        assert 'Value' in contributions, "缺少价值因子"
        assert 'Residual' in contributions, "缺少残差项"
        
        print(f"  ✅ 因子归因正常")
        print(f"     因子数量: {len(contributions)}")
        return True
        
    except Exception as e:
        print(f"  ❌ 因子归因失败: {e}")
        return False


def verify_transaction_cost():
    """验证交易成本"""
    print("\n🔍 验证交易成本分析...")
    try:
        import pandas as pd
        import numpy as np
        from performance_attribution import TransactionCostAnalysis
        
        np.random.seed(42)
        trades = pd.DataFrame({
            'symbol': ['A', 'B', 'C'] * 30,
            'quantity': np.random.randint(100, 1000, 90),
            'price': np.random.uniform(50, 200, 90),
            'timestamp': pd.date_range('2024-01-01', periods=90, freq='H')
        })
        
        cost_analysis = TransactionCostAnalysis(trades)
        costs = cost_analysis.analyze()
        
        # 验证结果
        assert 'total_cost' in costs, "缺少总成本"
        assert 'commission_cost' in costs, "缺少佣金成本"
        assert 'slippage_cost' in costs, "缺少滑点成本"
        assert 'market_impact_cost' in costs, "缺少市场冲击"
        assert 'cost_as_pct_of_value' in costs, "缺少成本占比"
        
        # 验证数值合理性
        assert costs['total_cost'] > 0, "总成本应为正数"
        assert 0 < costs['cost_as_pct_of_value'] < 0.1, "成本占比应在合理范围"
        
        print(f"  ✅ 交易成本分析正常")
        print(f"     总成本: ¥{costs['total_cost']:,.2f}")
        print(f"     成本占比: {costs['cost_as_pct_of_value']:.3%}")
        return True
        
    except Exception as e:
        print(f"  ❌ 交易成本分析失败: {e}")
        return False


def verify_file_structure():
    """验证文件结构"""
    print("\n🔍 验证文件结构...")
    
    files_to_check = [
        "qlib_enhanced/performance_attribution.py",
        "tests/test_attribution_integration.py",
        "docs/P2-7_Attribution_Analysis_README.md",
        "docs/P2-7_COMPLETION_SUMMARY.md"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist


def main():
    """主验证流程"""
    print("=" * 80)
    print("P2-7 绩效归因分析系统 - 快速验证")
    print("=" * 80)
    
    results = {
        '模块导入': verify_imports(),
        'Brinson归因': verify_brinson(),
        '因子归因': verify_factor_attribution(),
        '交易成本分析': verify_transaction_cost(),
        '文件结构': verify_file_structure()
    }
    
    print("\n" + "=" * 80)
    print("验证结果汇总")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:<15} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有验证通过！P2-7归因分析系统运行正常")
        print("=" * 80)
        print("\n📚 下一步:")
        print("  1. 运行完整测试: python tests/test_attribution_integration.py")
        print("  2. 启动Web界面: streamlit run web/unified_dashboard.py")
        print("  3. 查看文档: docs/P2-7_Attribution_Analysis_README.md")
        return 0
    else:
        print("❌ 部分验证失败，请检查上述错误信息")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
