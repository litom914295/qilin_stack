"""
P2-7 归因分析系统集成测试
测试Brinson归因、因子归因和交易成本分析的完整流程
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "qlib_enhanced"))

from performance_attribution import (
    BrinsonAttribution,
    FactorAttribution,
    TransactionCostAnalysis,
    create_sample_attribution_data
)


def test_brinson_attribution():
    """测试Brinson归因"""
    print("\n" + "="*80)
    print("TEST 1: Brinson归因模型")
    print("="*80)
    
    pw, pr, bw, br = create_sample_attribution_data()
    
    brinson = BrinsonAttribution(pw, pr, bw, br)
    result = brinson.analyze()
    
    print(f"✓ 配置效应: {result.allocation_effect:.4f}")
    print(f"✓ 选择效应: {result.selection_effect:.4f}")
    print(f"✓ 交互效应: {result.interaction_effect:.4f}")
    print(f"✓ 总超额收益: {result.total_active_return:.4f}")
    
    # 验证一致性
    assert abs(result.total_active_return - 
               (result.allocation_effect + result.selection_effect + result.interaction_effect)) < 1e-6, \
           "归因分解不一致！"
    
    print("✅ Brinson归因测试通过")
    return result


def test_factor_attribution():
    """测试因子归因"""
    print("\n" + "="*80)
    print("TEST 2: 因子归因分析")
    print("="*80)
    
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.01, 0.02, 100))
    factors = pd.DataFrame({
        'Market': np.random.normal(0.008, 0.015, 100),
        'Size': np.random.normal(0.002, 0.01, 100),
        'Value': np.random.normal(0.003, 0.01, 100),
        'Momentum': np.random.normal(0.004, 0.012, 100)
    })
    
    factor_attr = FactorAttribution(returns, factors)
    contributions = factor_attr.analyze()
    
    print("\n因子贡献:")
    for factor, contrib in contributions.items():
        print(f"  ✓ {factor}: {contrib:.4f}")
    
    print("✅ 因子归因测试通过")
    return contributions


def test_transaction_cost_analysis():
    """测试交易成本分析"""
    print("\n" + "="*80)
    print("TEST 3: 交易成本分析")
    print("="*80)
    
    np.random.seed(42)
    trades = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'] * 50,
        'quantity': np.random.randint(100, 1000, 150),
        'price': np.random.uniform(50, 200, 150),
        'timestamp': pd.date_range('2024-01-01', periods=150, freq='H')
    })
    
    cost_analysis = TransactionCostAnalysis(trades)
    
    # 测试不同佣金率
    for commission_rate in [0.001, 0.002, 0.003]:
        costs = cost_analysis.analyze(
            commission_rate=commission_rate,
            slippage_bps=5.0
        )
        
        print(f"\n佣金率 {commission_rate*100}%:")
        print(f"  ✓ 总成本: ¥{costs['total_cost']:,.2f}")
        print(f"  ✓ 佣金占比: {costs['commission_cost']/costs['total_cost']:.1%}")
        print(f"  ✓ 滑点占比: {costs['slippage_cost']/costs['total_cost']:.1%}")
        print(f"  ✓ 成本率: {costs['cost_as_pct_of_value']:.3%}")
        
        # 验证成本合理性
        assert costs['total_cost'] > 0, "总成本应为正数"
        assert costs['commission_cost'] > 0, "佣金成本应为正数"
        assert costs['slippage_cost'] > 0, "滑点成本应为正数"
        assert costs['cost_as_pct_of_value'] < 0.02, "成本率不应超过2%"
    
    print("\n✅ 交易成本分析测试通过")
    return costs


def test_integrated_workflow():
    """测试完整归因工作流"""
    print("\n" + "="*80)
    print("TEST 4: 完整归因工作流")
    print("="*80)
    
    # 1. 生成模拟投资组合数据
    np.random.seed(42)
    periods = 12
    
    # 组合数据
    portfolio_weights = pd.DataFrame({
        'Stock_A': np.random.uniform(0.2, 0.4, periods),
        'Stock_B': np.random.uniform(0.3, 0.5, periods),
        'Stock_C': np.random.uniform(0.2, 0.3, periods)
    })
    portfolio_weights = portfolio_weights.div(portfolio_weights.sum(axis=1), axis=0)
    
    portfolio_returns = pd.DataFrame({
        'Stock_A': np.random.normal(0.01, 0.02, periods),
        'Stock_B': np.random.normal(0.012, 0.025, periods),
        'Stock_C': np.random.normal(0.008, 0.015, periods)
    })
    
    # 基准数据
    benchmark_weights = pd.DataFrame({
        'Stock_A': [1/3] * periods,
        'Stock_B': [1/3] * periods,
        'Stock_C': [1/3] * periods
    })
    
    benchmark_returns = pd.DataFrame({
        'Stock_A': np.random.normal(0.009, 0.018, periods),
        'Stock_B': np.random.normal(0.010, 0.020, periods),
        'Stock_C': np.random.normal(0.009, 0.018, periods)
    })
    
    # 2. Brinson归因
    print("\n步骤 1: Brinson归因分析")
    brinson = BrinsonAttribution(
        portfolio_weights, portfolio_returns,
        benchmark_weights, benchmark_returns
    )
    brinson_result = brinson.analyze()
    print(f"  ✓ 总超额收益: {brinson_result.total_active_return:.2%}")
    
    # 3. 计算组合总收益
    portfolio_return = (portfolio_weights * portfolio_returns).sum(axis=1)
    
    # 4. 因子归因
    print("\n步骤 2: 因子归因分析")
    factors = pd.DataFrame({
        'Market': np.random.normal(0.008, 0.015, periods),
        'Value': np.random.normal(0.003, 0.01, periods),
        'Growth': np.random.normal(0.004, 0.012, periods)
    })
    
    factor_attr = FactorAttribution(portfolio_return, factors)
    factor_contrib = factor_attr.analyze()
    print(f"  ✓ 因子个数: {len(factor_contrib)}")
    
    # 5. 模拟交易成本
    print("\n步骤 3: 交易成本分析")
    trades = pd.DataFrame({
        'symbol': ['Stock_A', 'Stock_B', 'Stock_C'] * 20,
        'quantity': np.random.randint(100, 1000, 60),
        'price': np.random.uniform(50, 150, 60),
        'timestamp': pd.date_range('2024-01-01', periods=60, freq='D')
    })
    
    cost_analysis = TransactionCostAnalysis(trades)
    costs = cost_analysis.analyze()
    print(f"  ✓ 总交易成本: ¥{costs['total_cost']:,.2f}")
    print(f"  ✓ 成本占比: {costs['cost_as_pct_of_value']:.3%}")
    
    # 6. 综合报告
    print("\n步骤 4: 生成综合归因报告")
    print("\n" + "-"*80)
    print("归因分析综合报告")
    print("-"*80)
    print(f"组合表现:")
    print(f"  平均月收益: {portfolio_return.mean():.2%}")
    print(f"  收益波动率: {portfolio_return.std():.2%}")
    print(f"\nBrinson归因:")
    print(f"  配置效应: {brinson_result.allocation_effect:.2%}")
    print(f"  选择效应: {brinson_result.selection_effect:.2%}")
    print(f"  交互效应: {brinson_result.interaction_effect:.2%}")
    print(f"\n主要因子贡献:")
    for factor, contrib in list(factor_contrib.items())[:3]:
        print(f"  {factor}: {contrib:.4f}")
    print(f"\n交易成本影响:")
    print(f"  成本占比: {costs['cost_as_pct_of_value']:.3%}")
    print(f"  年化成本: {costs['cost_as_pct_of_value'] * 12:.2%}")
    print("-"*80)
    
    print("\n✅ 完整工作流测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🚀"*40)
    print("P2-7 归因分析系统 - 完整测试套件")
    print("🚀"*40)
    
    try:
        test_brinson_attribution()
        test_factor_attribution()
        test_transaction_cost_analysis()
        test_integrated_workflow()
        
        print("\n" + "="*80)
        print("🎉 所有测试通过！归因分析系统运行正常")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
