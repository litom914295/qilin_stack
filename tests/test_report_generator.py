"""
测试报告生成器
Test Report Generator

功能:
1. 自动生成测试报告
2. 包含性能指标、成功率、问题诊断
3. 生成Markdown和HTML格式报告

Author: Qilin Stack Team
Date: 2025-11-07
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, output_dir: str = '.'):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_markdown_report(self, 
                                 live_trading_results: Optional[Dict] = None,
                                 stress_test_results: Optional[Dict] = None) -> str:
        """
        生成Markdown格式报告
        
        Args:
            live_trading_results: 实盘测试结果
            stress_test_results: 压力测试结果
            
        Returns:
            报告文件路径
        """
        report_lines = []
        
        # 标题
        report_lines.append("# 🧪 Qilin Stack 测试报告")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 执行摘要
        report_lines.append("## 📊 执行摘要")
        report_lines.append("")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        if live_trading_results:
            total_tests += live_trading_results.get('total_tests', 0)
            passed_tests += live_trading_results.get('passed_tests', 0)
            failed_tests += live_trading_results.get('failed_tests', 0)
        
        if stress_test_results:
            total_tests += stress_test_results.get('total_tests', 0)
            passed_tests += stress_test_results.get('passed_tests', 0)
            failed_tests += stress_test_results.get('failed_tests', 0)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report_lines.append(f"- **总测试数**: {total_tests}")
        report_lines.append(f"- **通过**: {passed_tests} ✅")
        report_lines.append(f"- **失败**: {failed_tests} ❌")
        report_lines.append(f"- **成功率**: {success_rate:.1f}%")
        report_lines.append("")
        
        # 状态指示
        if success_rate >= 95:
            report_lines.append("### ✅ 测试状态: 优秀")
        elif success_rate >= 80:
            report_lines.append("### ⚠️ 测试状态: 良好")
        else:
            report_lines.append("### ❌ 测试状态: 需要改进")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 实盘交易测试部分
        if live_trading_results:
            report_lines.extend(self._generate_live_trading_section(live_trading_results))
        
        # 性能压力测试部分
        if stress_test_results:
            report_lines.extend(self._generate_stress_test_section(stress_test_results))
        
        # 问题诊断
        report_lines.extend(self._generate_diagnosis_section(live_trading_results, stress_test_results))
        
        # 建议
        report_lines.extend(self._generate_recommendations_section(live_trading_results, stress_test_results))
        
        # 附录
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## 📎 附录")
        report_lines.append("")
        report_lines.append("### 测试环境")
        report_lines.append("")
        report_lines.append("- **操作系统**: Windows")
        report_lines.append("- **Python版本**: 3.8+")
        report_lines.append("- **项目路径**: G:\\test\\qilin_stack")
        report_lines.append("")
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"test_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return str(report_file)
    
    def _generate_live_trading_section(self, results: Dict) -> List[str]:
        """生成实盘交易测试部分"""
        lines = []
        
        lines.append("## 🔄 实盘交易测试")
        lines.append("")
        lines.append(f"**券商**: {results.get('broker', 'N/A').upper()}")
        lines.append(f"**测试时间**: {results.get('timestamp', 'N/A')}")
        lines.append("")
        
        # 测试结果表格
        lines.append("### 测试结果")
        lines.append("")
        lines.append("| 测试项 | 状态 | 耗时 | 备注 |")
        lines.append("|--------|------|------|------|")
        
        for test in results.get('test_results', []):
            status = "✅ 通过" if test['success'] else "❌ 失败"
            duration = f"{test['duration']:.2f}s"
            note = ""
            
            # 提取关键信息
            details = test.get('details', {})
            if 'success_rate' in details:
                note = f"成功率: {details['success_rate']*100:.1f}%"
            elif 'statistics' in details and 'avg_latency' in details['statistics']:
                note = f"平均延迟: {details['statistics']['avg_latency']:.2f}ms"
            
            lines.append(f"| {test['test_name']} | {status} | {duration} | {note} |")
        
        lines.append("")
        
        # 性能指标
        lines.append("### 性能指标")
        lines.append("")
        
        # 提取延迟测量数据
        for test in results.get('test_results', []):
            if '延迟测量' in test['test_name'] and 'statistics' in test.get('details', {}):
                stats = test['details']['statistics']
                lines.append(f"- **平均延迟**: {stats['avg_latency']:.2f}ms")
                lines.append(f"- **最小延迟**: {stats['min_latency']:.2f}ms")
                lines.append(f"- **最大延迟**: {stats['max_latency']:.2f}ms")
                lines.append(f"- **成功率**: {stats['success_rate']*100:.1f}%")
                break
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_stress_test_section(self, results: Dict) -> List[str]:
        """生成性能压力测试部分"""
        lines = []
        
        lines.append("## 🏋️ 性能压力测试")
        lines.append("")
        lines.append(f"**测试时间**: {results.get('timestamp', 'N/A')}")
        lines.append("")
        
        # 测试结果表格
        lines.append("### 测试结果")
        lines.append("")
        lines.append("| 测试项 | 状态 | 耗时 | 关键指标 |")
        lines.append("|--------|------|------|----------|")
        
        for test in results.get('test_results', []):
            status = "✅ 通过" if test['success'] else "❌ 失败"
            duration = f"{test['duration']:.2f}s"
            key_metric = ""
            
            details = test.get('details', {})
            
            # 并发订单测试
            if '并发订单' in test['test_name']:
                if 'throughput' in details:
                    key_metric = f"吞吐量: {details['throughput']:.1f} 订单/秒"
            
            # 稳定性测试
            elif '稳定性' in test['test_name']:
                if 'error_rate' in details:
                    key_metric = f"错误率: {details['error_rate']*100:.2f}%"
            
            # 内存泄漏测试
            elif '内存泄漏' in test['test_name']:
                if 'avg_increase' in details:
                    key_metric = f"平均增长: {details['avg_increase']:.2f}MB/轮"
            
            # 异常恢复测试
            elif '异常恢复' in test['test_name']:
                scenarios = details.get('scenarios', [])
                passed = sum(1 for s in scenarios if s.get('success'))
                key_metric = f"场景通过: {passed}/{len(scenarios)}"
            
            # 因子计算性能
            elif '因子计算' in test['test_name']:
                if 'size_10000' in details and 'ma_throughput' in details['size_10000']:
                    key_metric = f"MA吞吐量: {details['size_10000']['ma_throughput']:.0f} 样本/秒"
            
            lines.append(f"| {test['test_name']} | {status} | {duration} | {key_metric} |")
        
        lines.append("")
        
        # 详细性能指标
        lines.append("### 详细性能指标")
        lines.append("")
        
        for test in results.get('test_results', []):
            details = test.get('details', {})
            
            # 并发订单详情
            if '并发订单' in test['test_name']:
                lines.append("#### 并发订单测试")
                lines.append("")
                if 'throughput' in details:
                    lines.append(f"- **吞吐量**: {details['throughput']:.1f} 订单/秒")
                if 'avg_latency' in details:
                    lines.append(f"- **平均延迟**: {details['avg_latency']:.2f}ms")
                if 'success_rate' in details:
                    lines.append(f"- **成功率**: {details['success_rate']*100:.1f}%")
                if 'memory_increase' in details:
                    lines.append(f"- **内存增长**: {details['memory_increase']:.2f}MB")
                lines.append("")
            
            # 稳定性测试详情
            elif '稳定性' in test['test_name']:
                lines.append("#### 长时间稳定性测试")
                lines.append("")
                if 'total_orders' in details:
                    lines.append(f"- **总订单数**: {details['total_orders']}")
                if 'error_rate' in details:
                    lines.append(f"- **错误率**: {details['error_rate']*100:.2f}%")
                if 'memory_increase' in details:
                    lines.append(f"- **内存增长**: {details['memory_increase']:.2f}MB")
                if 'memory_growth_rate' in details:
                    lines.append(f"- **内存增长率**: {details['memory_growth_rate']:.4f}MB/秒")
                lines.append("")
            
            # 内存泄漏详情
            elif '内存泄漏' in test['test_name']:
                lines.append("#### 内存泄漏检测")
                lines.append("")
                if 'avg_increase' in details:
                    lines.append(f"- **平均增长**: {details['avg_increase']:.2f}MB/轮")
                if 'max_increase' in details:
                    lines.append(f"- **最大增长**: {details['max_increase']:.2f}MB")
                if 'total_increase' in details:
                    lines.append(f"- **总增长**: {details['total_increase']:.2f}MB")
                lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_diagnosis_section(self, 
                                    live_results: Optional[Dict],
                                    stress_results: Optional[Dict]) -> List[str]:
        """生成问题诊断部分"""
        lines = []
        
        lines.append("## 🔍 问题诊断")
        lines.append("")
        
        issues = []
        warnings = []
        
        # 检查实盘测试问题
        if live_results:
            for test in live_results.get('test_results', []):
                if not test['success']:
                    issues.append(f"实盘测试失败: {test['test_name']}")
                    if 'error' in test.get('details', {}):
                        issues.append(f"  错误: {test['details']['error']}")
        
        # 检查压力测试问题
        if stress_results:
            for test in stress_results.get('test_results', []):
                if not test['success']:
                    issues.append(f"压力测试失败: {test['test_name']}")
                
                details = test.get('details', {})
                
                # 检查性能警告
                if 'avg_latency' in details and details['avg_latency'] > 200:
                    warnings.append(f"延迟较高: {test['test_name']} - {details['avg_latency']:.2f}ms")
                
                if 'error_rate' in details and details['error_rate'] > 0.01:
                    warnings.append(f"错误率偏高: {test['test_name']} - {details['error_rate']*100:.2f}%")
                
                if 'memory_growth_rate' in details and details['memory_growth_rate'] > 0.05:
                    warnings.append(f"可能存在内存泄漏: {test['test_name']} - {details['memory_growth_rate']:.4f}MB/秒")
        
        # 输出问题
        if issues:
            lines.append("### ❌ 发现的问题")
            lines.append("")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        # 输出警告
        if warnings:
            lines.append("### ⚠️ 警告")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        # 无问题
        if not issues and not warnings:
            lines.append("### ✅ 未发现明显问题")
            lines.append("")
            lines.append("所有测试正常通过,系统运行良好。")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _generate_recommendations_section(self,
                                         live_results: Optional[Dict],
                                         stress_results: Optional[Dict]) -> List[str]:
        """生成建议部分"""
        lines = []
        
        lines.append("## 💡 改进建议")
        lines.append("")
        
        recommendations = []
        
        # 基于测试结果生成建议
        if stress_results:
            for test in stress_results.get('test_results', []):
                details = test.get('details', {})
                
                # 性能优化建议
                if 'avg_latency' in details and details['avg_latency'] > 100:
                    recommendations.append({
                        'category': '性能优化',
                        'priority': '中',
                        'content': f"考虑优化订单处理流程,当前平均延迟 {details['avg_latency']:.2f}ms"
                    })
                
                # 内存优化建议
                if 'memory_growth_rate' in details and details['memory_growth_rate'] > 0.05:
                    recommendations.append({
                        'category': '内存管理',
                        'priority': '高',
                        'content': f"排查潜在内存泄漏,增长率 {details['memory_growth_rate']:.4f}MB/秒"
                    })
                
                # 稳定性建议
                if 'error_rate' in details and details['error_rate'] > 0:
                    recommendations.append({
                        'category': '稳定性',
                        'priority': '高' if details['error_rate'] > 0.05 else '中',
                        'content': f"降低错误率,当前为 {details['error_rate']*100:.2f}%"
                    })
        
        # 通用建议
        if not recommendations:
            recommendations.append({
                'category': '持续优化',
                'priority': '低',
                'content': '系统表现良好,建议持续监控和优化'
            })
            recommendations.append({
                'category': '生产部署',
                'priority': '中',
                'content': '可以考虑进行小规模生产环境测试'
            })
        
        # 按优先级分类
        high_priority = [r for r in recommendations if r['priority'] == '高']
        medium_priority = [r for r in recommendations if r['priority'] == '中']
        low_priority = [r for r in recommendations if r['priority'] == '低']
        
        if high_priority:
            lines.append("### 🔴 高优先级")
            lines.append("")
            for rec in high_priority:
                lines.append(f"- **[{rec['category']}]** {rec['content']}")
            lines.append("")
        
        if medium_priority:
            lines.append("### 🟡 中优先级")
            lines.append("")
            for rec in medium_priority:
                lines.append(f"- **[{rec['category']}]** {rec['content']}")
            lines.append("")
        
        if low_priority:
            lines.append("### 🟢 低优先级")
            lines.append("")
            for rec in low_priority:
                lines.append(f"- **[{rec['category']}]** {rec['content']}")
            lines.append("")
        
        return lines
    
    def load_test_results(self, result_file: str) -> Dict:
        """加载测试结果文件"""
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("📝 Qilin Stack 测试报告生成器")
    print("="*60)
    
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
        live_results = generator.load_test_results(str(latest_live))
    else:
        print("\n⚠️ 未找到实盘测试结果文件")
    
    if stress_test_files:
        latest_stress = max(stress_test_files, key=lambda p: p.stat().st_mtime)
        print(f"✅ 找到压力测试结果: {latest_stress.name}")
        stress_results = generator.load_test_results(str(latest_stress))
    else:
        print("⚠️ 未找到压力测试结果文件")
    
    if not live_results and not stress_results:
        print("\n❌ 没有找到任何测试结果文件")
        print("请先运行测试:")
        print("  python tests/live_trading_test.py")
        print("  python tests/performance_stress_test.py")
        return
    
    # 生成报告
    print("\n📝 生成测试报告...")
    report_file = generator.generate_markdown_report(live_results, stress_results)
    
    print(f"\n✅ 报告生成成功!")
    print(f"📄 报告路径: {report_file}")
    print("\n提示: 可以使用Markdown阅读器查看报告")


if __name__ == '__main__':
    main()
