"""
测试 Phase 1 新组件的导入和基本功能
验证所有文件都能正确加载
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'web'))
sys.path.insert(0, str(project_root / 'web' / 'components'))


def test_stage_indicator():
    """测试阶段识别组件"""
    print("=" * 60)
    print("测试 1: 阶段识别组件 (stage_indicator.py)")
    print("=" * 60)
    
    try:
        from web.components.stage_indicator import StageIndicator
        
        indicator = StageIndicator()
        stage_name, description, suggestion = indicator.get_current_stage()
        
        print(f"✅ 导入成功")
        print(f"当前阶段: {stage_name}")
        print(f"阶段描述: {description}")
        print(f"操作建议: {suggestion}")
        
        # 测试倒计时
        countdown = indicator.get_countdown()
        if countdown['show']:
            print(f"倒计时: 距离{countdown['target']} - {countdown['display']}")
        
        # 测试颜色
        color = indicator.get_stage_color()
        print(f"阶段颜色: {color}")
        
        # 测试提示
        tips = indicator.get_stage_tips({'candidate_count': 10, 'limitup_count': 50})
        print(f"智能提示: {len(tips)} 条")
        for tip in tips[:3]:
            print(f"  - {tip}")
        
        print("\n✅ 阶段识别组件测试通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 阶段识别组件测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_dashboard():
    """测试指标仪表盘组件"""
    print("=" * 60)
    print("测试 2: 指标仪表盘组件 (metrics_dashboard.py)")
    print("=" * 60)
    
    try:
        from web.components.metrics_dashboard import MetricsDashboard, create_metrics_from_data
        
        dashboard = MetricsDashboard()
        
        # 创建测试指标
        test_metrics = {
            'candidate_count': 15,
            'monitor_count': 8,
            'position_count': 5,
            'position_value': 123456.78,
            'total_profit': 5678.90,
            'profit_rate': 4.6
        }
        
        print(f"✅ 导入成功")
        print(f"测试指标:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value}")
        
        # 测试从数据创建指标
        import pandas as pd
        test_position_df = pd.DataFrame({
            'symbol': ['000001', '000002'],
            'current_value': [50000, 30000],
            'cost_value': [48000, 32000],
            'profit': [2000, -2000]
        })
        
        metrics = create_metrics_from_data(position_df=test_position_df)
        print(f"\n从DataFrame创建的指标:")
        print(f"  持仓数量: {metrics['position_count']}")
        print(f"  持仓市值: {metrics['position_value']:.2f}")
        print(f"  总盈亏: {metrics['total_profit']:.2f}")
        
        print("\n✅ 指标仪表盘组件测试通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 指标仪表盘组件测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_limitup_monitor_unified():
    """测试统一涨停监控视图"""
    print("=" * 60)
    print("测试 3: 统一涨停监控视图 (limitup_monitor_unified.py)")
    print("=" * 60)
    
    try:
        from web.tabs.limitup_monitor_unified import (
            get_available_dates,
            load_auction_report,
            load_rl_decision,
            load_rl_weights
        )
        
        print(f"✅ 导入成功")
        
        # 测试辅助函数
        dates = get_available_dates("reports")
        print(f"可用日期: {len(dates)} 个")
        if dates:
            print(f"  最新日期: {dates[0]}")
        
        # 测试数据加载函数（即使文件不存在，函数也应该正常返回None）
        test_date = "2024-01-01"
        auction_data = load_auction_report("reports", test_date)
        print(f"竞价报告加载: {'成功' if auction_data else '未找到（正常）'}")
        
        rl_data = load_rl_decision("reports", test_date)
        print(f"RL决策加载: {'成功' if rl_data else '未找到（正常）'}")
        
        weights = load_rl_weights("config")
        print(f"RL权重加载: {'成功' if weights else '未找到（正常）'}")
        
        print("\n✅ 统一涨停监控视图测试通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 统一涨停监控视图测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_unified_dashboard_integration():
    """测试unified_dashboard集成"""
    print("=" * 60)
    print("测试 4: unified_dashboard 集成")
    print("=" * 60)
    
    try:
        # 只测试能否导入，不实际运行streamlit
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "unified_dashboard",
            project_root / "web" / "unified_dashboard.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        print(f"✅ unified_dashboard.py 文件检查通过")
        
        # 检查是否有render_limitup_monitor_unified方法
        with open(project_root / "web" / "unified_dashboard.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if 'render_limitup_monitor_unified' in content:
                print(f"✅ 找到 render_limitup_monitor_unified 方法")
            else:
                print(f"❌ 未找到 render_limitup_monitor_unified 方法")
                return False
            
            if '🎯 一进二涨停监控' in content:
                print(f"✅ 找到新的主标签页入口")
            else:
                print(f"❌ 未找到新的主标签页入口")
                return False
        
        print("\n✅ unified_dashboard 集成测试通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ unified_dashboard 集成测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Phase 1 组件测试开始")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("阶段识别组件", test_stage_indicator()))
    results.append(("指标仪表盘组件", test_metrics_dashboard()))
    results.append(("统一涨停监控视图", test_limitup_monitor_unified()))
    results.append(("unified_dashboard集成", test_unified_dashboard_integration()))
    
    # 输出总结
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Phase 1 组件已准备就绪。")
        print("\n下一步:")
        print("  1. 运行: streamlit run web/unified_dashboard.py")
        print("  2. 进入 '🎯 一进二涨停监控' 标签页")
        print("  3. 验证所有功能正常工作")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查上述错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
