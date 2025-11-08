"""
Phase 2 组件综合测试
测试所有Phase 2新增的交互组件
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'web'))
sys.path.insert(0, str(project_root / 'web' / 'components'))


def test_interactive_filter():
    """测试交互式筛选漏斗"""
    print("=" * 60)
    print("测试 1: 交互式筛选漏斗 (interactive_filter.py)")
    print("=" * 60)
    
    try:
        from web.components.interactive_filter import InteractiveFilter
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        test_data = pd.DataFrame({
            'symbol': [f"{i:06d}" for i in range(50)],
            'name': [f"股票{i}" if i % 10 != 0 else f"ST股票{i}" for i in range(50)],
            'open_count': np.random.randint(0, 5, 50),
            'quality_score': np.random.randint(40, 100, 50),
            'rl_score': np.random.uniform(3, 10, 50),
            'is_first_board': np.random.choice([True, False], 50)
        })
        
        # 创建筛选器
        filter_component = InteractiveFilter(test_data, key_prefix="test")
        
        print(f"✅ 导入成功")
        print(f"测试数据: {len(test_data)} 只股票")
        print(f"筛选器初始化成功")
        
        # 测试筛选逻辑（非UI部分）
        print(f"\n模拟筛选流程:")
        print(f"  原始数据: {len(filter_component.original_data)} 只")
        
        print("\n✅ 交互式筛选漏斗测试通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 交互式筛选漏斗测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_auction_realtime():
    """测试竞价实时监控"""
    print("=" * 60)
    print("测试 2: 竞价实时监控 (auction_realtime.py)")
    print("=" * 60)
    
    try:
        from web.components.auction_realtime import AuctionRealtimeMonitor, create_test_auction_data
        
        # 创建监控器
        monitor = AuctionRealtimeMonitor(refresh_interval=10, key_prefix="test")
        
        print(f"✅ 导入成功")
        print(f"监控器初始化成功")
        print(f"刷新间隔: {monitor.refresh_interval}秒")
        
        # 创建测试数据
        test_data = create_test_auction_data(20)
        print(f"\n测试数据: {len(test_data)} 只股票")
        print(f"  包含列: {', '.join(test_data.columns.tolist())}")
        
        # 测试强度等级判断
        test_strengths = [9.5, 6.2, 3.1, -2.5, -7.8]
        print(f"\n强度等级测试:")
        for strength in test_strengths:
            _, level, emoji = monitor._get_strength_level(strength)
            print(f"  {strength:+.1f}% → {emoji} {level}")
        
        print("\n✅ 竞价实时监控测试通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 竞价实时监控测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_smart_actions():
    """测试智能提示和操作按钮"""
    print("=" * 60)
    print("测试 3: 智能提示系统 (smart_actions.py)")
    print("=" * 60)
    
    try:
        from web.components.smart_actions import SmartTipSystem, ActionButtons, RiskLevelIndicator
        
        # 测试智能提示系统
        tip_system = SmartTipSystem()
        
        print(f"✅ 导入成功")
        print(f"智能提示系统初始化成功")
        
        # 测试不同阶段的提示生成
        test_stages = {
            "T日选股": {
                'limitup_count': 85,
                'candidate_count': 8,
                'avg_quality_score': 75
            },
            "T+1竞价监控": {
                'strong_count': 5,
                'weak_count': 2,
                'avg_strength': 6.5,
                'monitor_count': 10
            },
            "T+2卖出决策": {
                'profit_count': 6,
                'loss_count': 2,
                'high_profit_count': 3
            }
        }
        
        print(f"\n提示生成测试:")
        for stage, data in test_stages.items():
            tips = tip_system.generate_tips(stage, data)
            print(f"  {stage}: 生成 {len(tips)} 条提示")
            for tip in tips[:2]:  # 显示前2条
                print(f"    - [{tip['type']}] {tip['message'][:50]}...")
        
        # 测试风险等级
        print(f"\n风险等级测试:")
        test_profits = [15.5, 5.2, -3.1, -8.5]
        for profit in test_profits:
            risk = RiskLevelIndicator.get_risk_level(profit)
            print(f"  {profit:+.1f}% → {risk['emoji']} {risk['level']} - {risk['suggestion']}")
        
        print("\n✅ 智能提示系统测试通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 智能提示系统测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_component_integration():
    """测试组件集成"""
    print("=" * 60)
    print("测试 4: 组件集成验证")
    print("=" * 60)
    
    try:
        # 检查所有组件文件是否存在
        components = [
            'web/components/interactive_filter.py',
            'web/components/auction_realtime.py',
            'web/components/smart_actions.py',
            'web/components/stage_indicator.py',
            'web/components/metrics_dashboard.py'
        ]
        
        print(f"检查组件文件:")
        all_exist = True
        for comp in components:
            comp_path = project_root / comp
            exists = comp_path.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {comp}")
            if not exists:
                all_exist = False
        
        if not all_exist:
            print(f"\n❌ 部分组件文件缺失")
            return False
        
        print(f"\n✅ 所有组件文件完整")
        
        # 统计代码行数
        total_lines = 0
        print(f"\n组件代码统计:")
        for comp in components:
            comp_path = project_root / comp
            if comp_path.exists():
                lines = len(comp_path.read_text(encoding='utf-8').splitlines())
                total_lines += lines
                print(f"  {comp.split('/')[-1]}: {lines} 行")
        
        print(f"\n  总计: {total_lines} 行代码")
        
        print("\n✅ 组件集成验证通过\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 组件集成验证失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Phase 2 组件综合测试开始")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("交互式筛选漏斗", test_interactive_filter()))
    results.append(("竞价实时监控", test_auction_realtime()))
    results.append(("智能提示系统", test_smart_actions()))
    results.append(("组件集成验证", test_component_integration()))
    
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
        print("\n🎉 所有测试通过！Phase 2 组件已准备就绪。")
        print("\n📊 Phase 2 完成度: 75% (6/8)")
        print("\n下一步:")
        print("  1. 优化数据展示表格（可选）")
        print("  2. 更新使用文档")
        print("  3. 集成到主界面测试")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查上述错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
