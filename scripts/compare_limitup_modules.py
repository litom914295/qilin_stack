"""
涨停监控模块功能对比分析脚本
对比 limitup_dashboard.py 和 limitup_monitor.py 的功能实现
"""

import ast
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import defaultdict


class ModuleAnalyzer:
    """模块分析器"""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.content = filepath.read_text(encoding='utf-8')
        self.tree = ast.parse(self.content)
        
    def get_imports(self) -> Dict[str, List[str]]:
        """获取导入的模块"""
        imports = defaultdict(list)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports['standard'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports['from'].append(f"{module}.{alias.name}")
        return dict(imports)
    
    def get_functions(self) -> List[Dict[str, Any]]:
        """获取所有函数定义"""
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # 获取函数参数
                args = [arg.arg for arg in node.args.args]
                # 获取文档字符串
                docstring = ast.get_docstring(node) or ''
                
                functions.append({
                    'name': node.name,
                    'args': args,
                    'lineno': node.lineno,
                    'docstring': docstring.split('\n')[0] if docstring else '',
                    'is_render': 'render' in node.name.lower(),
                    'lines': node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                })
        return functions
    
    def get_streamlit_components(self) -> Dict[str, int]:
        """统计使用的Streamlit组件"""
        components = defaultdict(int)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == 'st':
                            components[node.func.attr] += 1
        return dict(components)
    
    def get_tabs_definition(self) -> List[str]:
        """提取标签页定义"""
        tabs = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                # 查找 st.tabs(...) 调用
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute):
                        if (isinstance(node.value.func.value, ast.Name) and 
                            node.value.func.value.id == 'st' and 
                            node.value.func.attr == 'tabs'):
                            # 提取标签页名称
                            if node.value.args:
                                arg = node.value.args[0]
                                if isinstance(arg, ast.List):
                                    for elt in arg.elts:
                                        if isinstance(elt, ast.Constant):
                                            tabs.append(elt.value)
        return tabs
    
    def get_data_sources(self) -> List[str]:
        """提取数据源文件路径"""
        sources = []
        for node in ast.walk(self.tree):
            # 查找字符串常量中的文件路径模式
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    if any(keyword in node.value for keyword in ['.json', '.csv', 'report', 'backtest']):
                        if node.value not in sources:
                            sources.append(node.value)
        return sources
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取代码指标"""
        return {
            'total_lines': len(self.content.splitlines()),
            'total_functions': len([n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)]),
            'total_classes': len([n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]),
            'render_functions': len([n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef) and 'render' in n.name.lower()])
        }


def compare_modules(dashboard_path: Path, monitor_path: Path) -> Dict[str, Any]:
    """对比两个模块"""
    
    print("📊 开始分析模块...")
    
    # 创建分析器
    dashboard_analyzer = ModuleAnalyzer(dashboard_path)
    monitor_analyzer = ModuleAnalyzer(monitor_path)
    
    # 获取基本信息
    dashboard_info = {
        'imports': dashboard_analyzer.get_imports(),
        'functions': dashboard_analyzer.get_functions(),
        'components': dashboard_analyzer.get_streamlit_components(),
        'tabs': dashboard_analyzer.get_tabs_definition(),
        'data_sources': dashboard_analyzer.get_data_sources(),
        'metrics': dashboard_analyzer.get_metrics()
    }
    
    monitor_info = {
        'imports': monitor_analyzer.get_imports(),
        'functions': monitor_analyzer.get_functions(),
        'components': monitor_analyzer.get_streamlit_components(),
        'tabs': monitor_analyzer.get_tabs_definition(),
        'data_sources': monitor_analyzer.get_data_sources(),
        'metrics': monitor_analyzer.get_metrics()
    }
    
    # 对比分析
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'dashboard': dashboard_info,
        'monitor': monitor_info,
        'differences': analyze_differences(dashboard_info, monitor_info)
    }
    
    return comparison


def analyze_differences(dashboard_info: Dict, monitor_info: Dict) -> Dict[str, Any]:
    """分析差异"""
    
    # 标签页对比
    dashboard_tabs = set(dashboard_info['tabs'])
    monitor_tabs = set(monitor_info['tabs'])
    
    # 函数对比
    dashboard_funcs = {f['name'] for f in dashboard_info['functions']}
    monitor_funcs = {f['name'] for f in monitor_info['functions']}
    
    # Streamlit组件对比
    all_components = set(dashboard_info['components'].keys()) | set(monitor_info['components'].keys())
    component_diff = {
        comp: {
            'dashboard': dashboard_info['components'].get(comp, 0),
            'monitor': monitor_info['components'].get(comp, 0),
            'diff': monitor_info['components'].get(comp, 0) - dashboard_info['components'].get(comp, 0)
        }
        for comp in all_components
    }
    
    return {
        'tabs': {
            'common': list(dashboard_tabs & monitor_tabs),
            'dashboard_only': list(dashboard_tabs - monitor_tabs),
            'monitor_only': list(monitor_tabs - dashboard_tabs),
            'coverage_rate': len(dashboard_tabs & monitor_tabs) / len(dashboard_tabs) if dashboard_tabs else 0
        },
        'functions': {
            'common': list(dashboard_funcs & monitor_funcs),
            'dashboard_only': list(dashboard_funcs - monitor_funcs),
            'monitor_only': list(monitor_funcs - dashboard_funcs),
            'coverage_rate': len(dashboard_funcs & monitor_funcs) / len(dashboard_funcs) if dashboard_funcs else 0
        },
        'components': component_diff,
        'metrics_comparison': {
            'dashboard': dashboard_info['metrics'],
            'monitor': monitor_info['metrics']
        }
    }


def generate_markdown_report(comparison: Dict[str, Any], output_path: Path):
    """生成Markdown对比报告"""
    
    report = f"""# 涨停监控模块功能对比报告

**生成时间**: {comparison['timestamp']}

## 📊 执行摘要

### 总体覆盖率

| 项目 | 覆盖率 | 说明 |
|------|--------|------|
| 标签页 | {comparison['differences']['tabs']['coverage_rate']:.1%} | limitup_monitor.py 覆盖 limitup_dashboard.py 的标签页功能 |
| 函数 | {comparison['differences']['functions']['coverage_rate']:.1%} | 核心函数的实现覆盖率 |

### 关键发现

- **limitup_dashboard.py**: {comparison['dashboard']['metrics']['total_lines']} 行代码, {comparison['dashboard']['metrics']['total_functions']} 个函数
- **limitup_monitor.py**: {comparison['monitor']['metrics']['total_lines']} 行代码, {comparison['monitor']['metrics']['total_functions']} 个函数

---

## 🏷️ 标签页对比

### 共同标签页 ({len(comparison['differences']['tabs']['common'])}个)

"""
    
    for tab in comparison['differences']['tabs']['common']:
        report += f"- ✅ {tab}\n"
    
    report += f"""
### limitup_dashboard.py 独有标签页 ({len(comparison['differences']['tabs']['dashboard_only'])}个)

"""
    
    if comparison['differences']['tabs']['dashboard_only']:
        for tab in comparison['differences']['tabs']['dashboard_only']:
            report += f"- ⚠️ {tab}\n"
    else:
        report += "- *无*\n"
    
    report += f"""
### limitup_monitor.py 独有标签页 ({len(comparison['differences']['tabs']['monitor_only'])}个)

"""
    
    if comparison['differences']['tabs']['monitor_only']:
        for tab in comparison['differences']['tabs']['monitor_only']:
            report += f"- ➕ {tab}\n"
    else:
        report += "- *无*\n"
    
    report += """
---

## 🔧 函数对比

### 共同函数

"""
    
    common_funcs = comparison['differences']['functions']['common']
    if common_funcs:
        for func in sorted(common_funcs):
            report += f"- ✅ `{func}()`\n"
    else:
        report += "- *无共同函数*\n"
    
    report += f"""
### limitup_dashboard.py 独有函数 ({len(comparison['differences']['functions']['dashboard_only'])}个)

"""
    
    dashboard_only_funcs = comparison['differences']['functions']['dashboard_only']
    if dashboard_only_funcs:
        # 获取详细函数信息
        dashboard_func_details = {f['name']: f for f in comparison['dashboard']['functions']}
        for func in sorted(dashboard_only_funcs):
            details = dashboard_func_details.get(func, {})
            args = ', '.join(details.get('args', []))
            docstring = details.get('docstring', '')
            report += f"- ⚠️ `{func}({args})` - {docstring}\n"
    else:
        report += "- *无*\n"
    
    report += f"""
### limitup_monitor.py 独有函数 ({len(comparison['differences']['functions']['monitor_only'])}个)

"""
    
    monitor_only_funcs = comparison['differences']['functions']['monitor_only']
    if monitor_only_funcs:
        monitor_func_details = {f['name']: f for f in comparison['monitor']['functions']}
        for func in sorted(monitor_only_funcs):
            details = monitor_func_details.get(func, {})
            args = ', '.join(details.get('args', []))
            docstring = details.get('docstring', '')
            report += f"- ➕ `{func}({args})` - {docstring}\n"
    else:
        report += "- *无*\n"
    
    report += """
---

## 📊 Streamlit组件使用对比

| 组件 | limitup_dashboard.py | limitup_monitor.py | 差异 |
|------|---------------------|-------------------|------|
"""
    
    for comp, diff in sorted(comparison['differences']['components'].items()):
        report += f"| `st.{comp}()` | {diff['dashboard']} | {diff['monitor']} | {diff['diff']:+d} |\n"
    
    report += """
---

## 📈 代码指标对比

| 指标 | limitup_dashboard.py | limitup_monitor.py |
|------|---------------------|-------------------|
"""
    
    dashboard_metrics = comparison['differences']['metrics_comparison']['dashboard']
    monitor_metrics = comparison['differences']['metrics_comparison']['monitor']
    
    for key in dashboard_metrics:
        report += f"| {key} | {dashboard_metrics[key]} | {monitor_metrics[key]} |\n"
    
    report += """
---

## 🎯 结论与建议

### 功能覆盖情况

"""
    
    coverage = comparison['differences']['tabs']['coverage_rate']
    
    if coverage >= 0.9:
        report += """
✅ **功能基本一致** (覆盖率 ≥ 90%)

limitup_monitor.py 已经实现了 limitup_dashboard.py 的绝大部分功能，可以安全地替代使用。

**建议：**
1. 确认 limitup_monitor.py 已正确集成到 unified_dashboard.py
2. 将 limitup_dashboard.py 标记为已归档或删除
3. 更新相关文档，统一使用 unified_dashboard.py 作为主入口

"""
    elif coverage >= 0.7:
        report += """
⚠️ **大部分功能已覆盖** (覆盖率 70-90%)

limitup_monitor.py 实现了大部分功能，但仍有少量功能缺失。

**建议：**
1. 检查缺失的功能是否必要
2. 如必要，需要将缺失功能迁移到 limitup_monitor.py
3. 暂时保留 limitup_dashboard.py 作为功能完整的备份

"""
    else:
        report += """
❌ **存在显著功能差异** (覆盖率 < 70%)

两个模块存在显著差异，需要进一步分析和整合。

**建议：**
1. 详细评估缺失功能的重要性
2. 制定功能迁移计划
3. 分阶段实施整合
4. 完整测试所有功能

"""
    
    # 添加数据源对比
    report += """
### 数据源对比

#### limitup_dashboard.py 使用的数据源：

"""
    
    for source in comparison['dashboard']['data_sources']:
        report += f"- `{source}`\n"
    
    report += """
#### limitup_monitor.py 使用的数据源：

"""
    
    for source in comparison['monitor']['data_sources']:
        report += f"- `{source}`\n"
    
    report += """
---

## 📝 附录

### 详细函数列表

#### limitup_dashboard.py 函数列表

"""
    
    for func in sorted(comparison['dashboard']['functions'], key=lambda x: x['lineno']):
        args = ', '.join(func['args'])
        report += f"- **{func['name']}**`({args})` (第{func['lineno']}行, {func['lines']}行代码)\n"
        if func['docstring']:
            report += f"  - {func['docstring']}\n"
    
    report += """
#### limitup_monitor.py 函数列表

"""
    
    for func in sorted(comparison['monitor']['functions'], key=lambda x: x['lineno']):
        args = ', '.join(func['args'])
        report += f"- **{func['name']}**`({args})` (第{func['lineno']}行, {func['lines']}行代码)\n"
        if func['docstring']:
            report += f"  - {func['docstring']}\n"
    
    report += """
---

*本报告由自动化脚本生成*
"""
    
    # 保存报告
    output_path.write_text(report, encoding='utf-8')
    print(f"\n✅ 报告已生成: {output_path}")


def main():
    """主函数"""
    
    # 项目根目录
    root_dir = Path(__file__).parent.parent
    
    # 文件路径
    dashboard_path = root_dir / 'web' / 'limitup_dashboard.py'
    monitor_path = root_dir / 'web' / 'tabs' / 'rdagent' / 'limitup_monitor.py'
    
    # 检查文件是否存在
    if not dashboard_path.exists():
        print(f"❌ 文件不存在: {dashboard_path}")
        return
    
    if not monitor_path.exists():
        print(f"❌ 文件不存在: {monitor_path}")
        return
    
    print("🔍 正在对比以下两个模块:")
    print(f"   1. {dashboard_path.relative_to(root_dir)}")
    print(f"   2. {monitor_path.relative_to(root_dir)}")
    print()
    
    # 执行对比
    comparison = compare_modules(dashboard_path, monitor_path)
    
    # 保存JSON结果
    json_output = root_dir / 'output' / 'limitup_modules_comparison.json'
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ JSON数据已保存: {json_output}")
    
    # 生成Markdown报告
    report_output = root_dir / 'docs' / 'LIMITUP_MODULES_COMPARISON_REPORT.md'
    report_output.parent.mkdir(parents=True, exist_ok=True)
    generate_markdown_report(comparison, report_output)
    
    # 打印摘要
    print("\n" + "="*60)
    print("📊 对比摘要")
    print("="*60)
    print(f"\n标签页覆盖率: {comparison['differences']['tabs']['coverage_rate']:.1%}")
    print(f"函数覆盖率: {comparison['differences']['functions']['coverage_rate']:.1%}")
    print(f"\nlimitup_dashboard.py: {comparison['dashboard']['metrics']['total_lines']} 行代码")
    print(f"limitup_monitor.py: {comparison['monitor']['metrics']['total_lines']} 行代码")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
