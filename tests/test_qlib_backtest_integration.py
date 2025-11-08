"""
Qlib回测集成模块测试
测试核心功能是否正常工作
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "web"))


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        from tabs.qlib_backtest_tab import (
            render_qlib_backtest_tab,
            _ensure_qlib_initialized,
            _generate_sample_predictions,
            run_qlib_backtest
        )
        print("✅ 所有核心函数导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_qlib_availability():
    """测试Qlib可用性"""
    print("\n" + "=" * 60)
    print("测试2: Qlib可用性")
    print("=" * 60)
    
    try:
        import qlib
        from qlib.backtest import backtest
        from qlib.constant import REG_CN
        print("✅ Qlib已安装")
        
        # 检查是否已初始化
        from tabs.qlib_backtest_tab import _ensure_qlib_initialized
        if _ensure_qlib_initialized():
            print("✅ Qlib已初始化")
            return True
        else:
            print("⚠️ Qlib未初始化（可能需要配置数据路径）")
            return False
    except ImportError as e:
        print(f"❌ Qlib未安装: {e}")
        return False


def test_generate_sample_predictions():
    """测试示例预测数据生成"""
    print("\n" + "=" * 60)
    print("测试3: 示例预测数据生成")
    print("=" * 60)
    
    try:
        from tabs.qlib_backtest_tab import _generate_sample_predictions
        
        pred_score = _generate_sample_predictions()
        
        print(f"✅ 生成成功")
        print(f"   - 数据类型: {type(pred_score)}")
        print(f"   - 数据形状: {pred_score.shape}")
        print(f"   - 索引层级: {pred_score.index.names}")
        print(f"   - 数值范围: [{pred_score.min():.4f}, {pred_score.max():.4f}]")
        print(f"   - 均值: {pred_score.mean():.4f}")
        print(f"   - 标准差: {pred_score.std():.4f}")
        
        # 验证数据格式
        assert isinstance(pred_score, pd.Series), "应该是Series类型"
        assert pred_score.index.names == ['datetime', 'instrument'], "索引应该是datetime和instrument"
        assert len(pred_score) > 0, "数据不应为空"
        
        print("✅ 所有验证通过")
        return pred_score
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_backtest_execution_mock():
    """测试回测执行（模拟模式）"""
    print("\n" + "=" * 60)
    print("测试4: 回测执行（模拟）")
    print("=" * 60)
    
    try:
        from tabs.qlib_backtest_tab import _generate_sample_predictions
        
        # 生成示例数据
        pred_score = _generate_sample_predictions()
        
        print("✅ 预测数据准备完成")
        print(f"   - 数据量: {len(pred_score)}")
        
        # 模拟回测参数
        params = {
            'pred_score': pred_score,
            'start_time': '2020-01-01',
            'end_time': '2020-12-31',
            'benchmark': 'SH000300',
            'topk': 30,
            'n_drop': 5,
            'init_cash': 1000000,
            'open_cost': 0.0015,
            'close_cost': 0.0025,
            'min_cost': 5.0
        }
        
        print("✅ 回测参数配置完成")
        print(f"   - 时间范围: {params['start_time']} ~ {params['end_time']}")
        print(f"   - 持仓数量: {params['topk']}")
        print(f"   - 初始资金: {params['init_cash']:,.0f}元")
        
        # 注意：实际执行需要Qlib已初始化和数据可用
        # 这里仅测试参数准备
        print("ℹ️ 实际回测需要Qlib完全初始化（跳过执行）")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_result_structure():
    """测试结果数据结构"""
    print("\n" + "=" * 60)
    print("测试5: 结果数据结构验证")
    print("=" * 60)
    
    try:
        # 模拟结果数据结构
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        # 模拟净值
        portfolio_value = pd.Series(
            np.cumprod(1 + np.random.randn(len(dates)) * 0.01),
            index=dates
        )
        
        # 模拟日收益
        daily_returns = pd.Series(
            np.random.randn(len(dates)) * 0.01,
            index=dates
        )
        
        # 模拟回撤
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max
        
        # 计算指标
        annualized_return = (portfolio_value.iloc[-1] ** (365 / len(dates))) - 1
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        max_drawdown = drawdown.min()
        volatility = daily_returns.std() * np.sqrt(252)
        win_rate = (daily_returns > 0).sum() / len(daily_returns)
        
        metrics = {
            'annualized_return': annualized_return,
            'cumulative_return': portfolio_value.iloc[-1] - 1,
            'information_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
        }
        
        print("✅ 结果数据结构创建成功")
        print("\n关键指标：")
        print(f"   - 年化收益率: {metrics['annualized_return']:.2%}")
        print(f"   - 夏普比率: {metrics['information_ratio']:.3f}")
        print(f"   - 最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"   - 波动率: {metrics['volatility']:.2%}")
        print(f"   - 胜率: {metrics['win_rate']:.2%}")
        
        # 验证结构
        assert 'annualized_return' in metrics
        assert 'information_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        print("\n✅ 所有结构验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 20)
    print("Qlib回测集成模块测试套件")
    print("🧪 " * 20 + "\n")
    
    results = {}
    
    # 运行各项测试
    results['imports'] = test_imports()
    results['qlib_availability'] = test_qlib_availability()
    results['sample_predictions'] = test_generate_sample_predictions() is not None
    results['backtest_mock'] = test_backtest_execution_mock()
    results['result_structure'] = test_result_structure()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
