"""
集成测试运行脚本
Run All Tests Script

功能:
1. 一键运行所有测试
2. 自动生成测试报告
3. 显示测试摘要

Author: Qilin Stack Team
Date: 2025-11-07
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def run_live_trading_test():
    """运行实盘交易测试"""
    print("\n" + "="*80)
    print("🔄 步骤 1/3: 运行实盘交易测试")
    print("="*80)
    
    try:
        from tests.live_trading_test import LiveTradingTestSuite
        
        # 使用Mock券商进行测试
        test_suite = LiveTradingTestSuite(broker_name='mock', broker_config={
            'initial_cash': 1000000,
            'commission_rate': 0.0003
        })
        
        summary = await test_suite.run_all_tests()
        
        # 保存结果
        import json
        output_file = f"live_trading_test_results_mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 实盘交易测试完成")
        print(f"   成功率: {summary['success_rate']*100:.1f}%")
        print(f"   结果文件: {output_file}")
        
        return summary
        
    except Exception as e:
        print(f"\n❌ 实盘交易测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_stress_test():
    """运行性能压力测试"""
    print("\n" + "="*80)
    print("🏋️ 步骤 2/3: 运行性能压力测试")
    print("="*80)
    
    try:
        from tests.performance_stress_test import PerformanceStressTestSuite
        
        test_suite = PerformanceStressTestSuite()
        summary = await test_suite.run_all_tests()
        
        # 保存结果
        import json
        output_file = f"performance_stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 性能压力测试完成")
        print(f"   成功率: {summary['success_rate']*100:.1f}%")
        print(f"   结果文件: {output_file}")
        
        return summary
        
    except Exception as e:
        print(f"\n❌ 性能压力测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_report():
    """生成测试报告"""
    print("\n" + "="*80)
    print("📝 步骤 3/3: 生成测试报告")
    print("="*80)
    
    try:
        from tests.test_report_generator import TestReportGenerator
        import json
        
        generator = TestReportGenerator()
        
        # 查找最新的测试结果文件
        current_dir = Path('.')
        
        live_trading_files = list(current_dir.glob('live_trading_test_results_*.json'))
        stress_test_files = list(current_dir.glob('performance_stress_test_results_*.json'))
        
        live_results = None
        stress_results = None
        
        if live_trading_files:
            latest_live = max(live_trading_files, key=lambda p: p.stat().st_mtime)
            print(f"\n✅ 找到实盘测试结果: {latest_live.name}")
            with open(latest_live, 'r', encoding='utf-8') as f:
                live_results = json.load(f)
        
        if stress_test_files:
            latest_stress = max(stress_test_files, key=lambda p: p.stat().st_mtime)
            print(f"✅ 找到压力测试结果: {latest_stress.name}")
            with open(latest_stress, 'r', encoding='utf-8') as f:
                stress_results = json.load(f)
        
        if live_results or stress_results:
            report_file = generator.generate_markdown_report(live_results, stress_results)
            print(f"\n✅ 测试报告生成成功")
            print(f"   报告文件: {report_file}")
            return report_file
        else:
            print("\n⚠️ 未找到测试结果文件")
            return None
            
    except Exception as e:
        print(f"\n❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """主函数"""
    print("\n" + "="*80)
    print("🧪 Qilin Stack 集成测试套件")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    start_time = datetime.now()
    
    # 运行测试
    live_summary = await run_live_trading_test()
    stress_summary = await run_stress_test()
    report_file = generate_report()
    
    # 生成最终摘要
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("📊 最终测试摘要")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    if live_summary:
        total_tests += live_summary.get('total_tests', 0)
        passed_tests += live_summary.get('passed_tests', 0)
        print(f"\n🔄 实盘交易测试:")
        print(f"   测试数: {live_summary.get('total_tests', 0)}")
        print(f"   通过: {live_summary.get('passed_tests', 0)}")
        print(f"   成功率: {live_summary.get('success_rate', 0)*100:.1f}%")
    
    if stress_summary:
        total_tests += stress_summary.get('total_tests', 0)
        passed_tests += stress_summary.get('passed_tests', 0)
        print(f"\n🏋️ 性能压力测试:")
        print(f"   测试数: {stress_summary.get('total_tests', 0)}")
        print(f"   通过: {stress_summary.get('passed_tests', 0)}")
        print(f"   成功率: {stress_summary.get('success_rate', 0)*100:.1f}%")
    
    if total_tests > 0:
        overall_success_rate = (passed_tests / total_tests) * 100
        print(f"\n🎯 总体统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {total_tests - passed_tests}")
        print(f"   成功率: {overall_success_rate:.1f}%")
        print(f"   总耗时: {duration:.1f}秒")
        
        # 状态评价
        if overall_success_rate >= 95:
            print(f"\n✅ 测试状态: 优秀 - 系统稳定可靠!")
        elif overall_success_rate >= 80:
            print(f"\n⚠️ 测试状态: 良好 - 存在少量问题")
        else:
            print(f"\n❌ 测试状态: 需要改进 - 存在较多问题")
    
    if report_file:
        print(f"\n📄 完整测试报告: {report_file}")
    
    print("\n" + "="*80)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return {
        'live_summary': live_summary,
        'stress_summary': stress_summary,
        'report_file': report_file,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'duration': duration
    }


if __name__ == '__main__':
    try:
        result = asyncio.run(main())
        
        # 根据测试结果设置退出码
        if result['total_tests'] > 0:
            success_rate = result['passed_tests'] / result['total_tests']
            if success_rate < 0.8:
                sys.exit(1)  # 失败
            else:
                sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 没有测试运行
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
