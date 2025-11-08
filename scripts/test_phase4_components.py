"""
Phase 4组件测试脚本
测试模拟交易、策略回测、数据导出功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

# 导入Phase 4组件
from web.components.advanced_features import (
    SimulatedTrading,
    StrategyBacktest,
    ExportManager
)


def test_simulated_trading():
    """测试模拟交易系统"""
    print("=" * 60)
    print("测试 1: 模拟交易系统")
    print("=" * 60)
    
    try:
        # 创建模拟交易系统（使用独立的session模拟）
        class MockSession(dict):
            def __init__(self):
                super().__init__()
                self['simulated_positions'] = []
                self['simulated_history'] = []
                self['simulated_capital'] = 100000
            
            def __getattr__(self, key):
                return self[key]
            
            def __setattr__(self, key, value):
                self[key] = value
        
        mock_session = MockSession()
        
        # 手动模拟st.session_state
        import web.components.advanced_features as af_module
        original_st = af_module.st
        
        class MockSt:
            session_state = mock_session
        
        af_module.st = MockSt()
        
        trading = SimulatedTrading()
        print("✅ SimulatedTrading初始化正确")
        
        # 测试买入
        result = trading.buy('000001', 10.0, 1000, '2024-01-01')
        assert result['success'] == True
        assert '成功买入' in result['message']
        print(f"✅ 买入测试通过: {result['message']}")
        
        # 测试资金检查
        result2 = trading.buy('000002', 10000, 1000, '2024-01-02')
        assert result2['success'] == False
        assert '资金不足' in result2['message']
        print("✅ 资金检查测试通过")
        
        # 测试持仓查询
        positions = trading.get_positions()
        assert len(positions) == 1
        assert positions.iloc[0]['symbol'] == '000001'
        print(f"✅ 持仓查询测试通过 (持仓数: {len(positions)})")
        
        # 测试卖出
        result3 = trading.sell('000001', 11.0, None, '2024-01-03')
        assert result3['success'] == True
        assert '成功卖出' in result3['message']
        assert result3['profit'] > 0
        print(f"✅ 卖出测试通过: {result3['message']}")
        
        # 测试统计
        stats = trading.get_statistics()
        assert stats['total_trades'] == 2
        assert stats['win_trades'] == 1
        assert stats['win_rate'] == 100
        print(f"✅ 统计测试通过 (胜率: {stats['win_rate']}%)")
        
        # 恢复原始st模块
        af_module.st = original_st
        
        print("✅ 通过 - 模拟交易系统\n")
        return True
        
    except Exception as e:
        print(f"❌ 失败 - 模拟交易系统: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_backtest():
    """测试策略回测引擎"""
    print("=" * 60)
    print("测试 2: 策略回测引擎")
    print("=" * 60)
    
    try:
        backtest = StrategyBacktest()
        print("✅ StrategyBacktest初始化正确")
        
        # 创建测试信号
        signals = [
            {'date': '2024-01-01', 'symbol': '000001', 'action': 'buy', 'price': 10.0},
            {'date': '2024-01-05', 'symbol': '000001', 'action': 'sell', 'price': 11.0},
            {'date': '2024-01-10', 'symbol': '000002', 'action': 'buy', 'price': 20.0},
            {'date': '2024-01-15', 'symbol': '000002', 'action': 'sell', 'price': 19.0},
        ]
        signals_df = pd.DataFrame(signals)
        
        # 执行回测
        result = backtest.backtest(signals_df)
        
        # 验证结果
        assert 'equity_curve' in result
        assert 'statistics' in result
        assert len(result['equity_curve']) > 0
        print("✅ 回测执行成功")
        
        # 验证统计指标
        stats = result['statistics']
        assert 'total_return' in stats
        assert 'win_rate' in stats
        assert 'total_trades' in stats
        assert stats['total_trades'] == 2
        print(f"✅ 统计指标正确 (交易次数: {stats['total_trades']}, 胜率: {stats['win_rate']:.1f}%)")
        
        # 测试权益曲线绘制
        fig = backtest.plot_equity_curve(result)
        assert fig is not None
        assert hasattr(fig, 'data')
        print("✅ 权益曲线绘制正确")
        
        print("✅ 通过 - 策略回测引擎\n")
        return True
        
    except Exception as e:
        print(f"❌ 失败 - 策略回测引擎: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_export_manager():
    """测试数据导出管理器"""
    print("=" * 60)
    print("测试 3: 数据导出管理器")
    print("=" * 60)
    
    try:
        # 创建测试数据
        test_df = pd.DataFrame({
            'symbol': ['000001', '000002', '000003'],
            'name': ['平安银行', '万科A', '国农科技'],
            'price': [10.0, 20.0, 30.0]
        })
        
        test_stats = {
            'total_count': 3,
            'avg_price': 20.0
        }
        
        # 测试CSV导出
        csv_data = ExportManager.export_to_csv(test_df)
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0
        print(f"✅ CSV导出成功 (大小: {len(csv_data)} bytes)")
        
        # 测试JSON导出
        json_data = ExportManager.export_to_json({'test': 'data'})
        assert isinstance(json_data, bytes)
        assert b'test' in json_data
        print(f"✅ JSON导出成功 (大小: {len(json_data)} bytes)")
        
        # 测试Excel导出
        excel_data = ExportManager.export_to_excel({
            'Sheet1': test_df,
            'Sheet2': pd.DataFrame([test_stats])
        })
        assert isinstance(excel_data, bytes)
        assert len(excel_data) > 0
        print(f"✅ Excel导出成功 (大小: {len(excel_data)} bytes)")
        
        # 测试完整报告生成
        for fmt in ['excel', 'csv', 'json']:
            report_data = ExportManager.create_report(test_df, test_stats, fmt)
            assert isinstance(report_data, bytes)
            assert len(report_data) > 0
            print(f"✅ {fmt.upper()}报告生成成功")
        
        print("✅ 通过 - 数据导出管理器\n")
        return True
        
    except Exception as e:
        print(f"❌ 失败 - 数据导出管理器: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Phase 4 组件测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 执行所有测试
    results.append(("模拟交易系统", test_simulated_trading()))
    results.append(("策略回测引擎", test_strategy_backtest()))
    results.append(("数据导出管理器", test_export_manager()))
    
    # 统计结果
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有Phase 4组件测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
