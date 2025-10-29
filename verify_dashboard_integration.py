#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dashboard整合验证脚本
快速检查涨停板监控模块是否成功整合到统一Dashboard
"""

import sys
from pathlib import Path


def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if file_path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} 不存在: {file_path}")
        return False


def check_module_import(module_path, module_name):
    """检查模块是否可导入"""
    try:
        sys.path.insert(0, str(module_path.parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        print(f"✅ 模块可导入: {module_name}")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {module_name}")
        print(f"   错误: {e}")
        return False


def check_unified_dashboard_integration():
    """检查unified_dashboard是否正确整合"""
    unified_path = Path("web/unified_dashboard.py")
    
    if not unified_path.exists():
        print("❌ unified_dashboard.py 不存在")
        return False
    
    content = unified_path.read_text(encoding='utf-8')
    
    checks = [
        ("limitup_monitor import", "from tabs.rdagent import limitup_monitor" in content),
        ("limitup_monitor.render()", "limitup_monitor.render()" in content),
        ("涨停板监控tab", "涨停板监控" in content or "🎯" in content),
    ]
    
    all_passed = True
    for check_name, result in checks:
        if result:
            print(f"✅ {check_name}: 已整合")
        else:
            print(f"❌ {check_name}: 未找到")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 60)
    print("Dashboard整合验证")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent
    
    # 1. 检查关键文件
    print("📁 检查关键文件...")
    print()
    
    files_to_check = [
        (project_root / "web" / "unified_dashboard.py", "统一Dashboard"),
        (project_root / "web" / "limitup_dashboard.py", "独立Dashboard"),
        (project_root / "web" / "tabs" / "rdagent" / "limitup_monitor.py", "涨停板监控模块"),
        (project_root / "start_dashboard.bat", "Windows启动脚本"),
        (project_root / "start_dashboard.sh", "Linux/Mac启动脚本"),
        (project_root / "WEB_DASHBOARD_GUIDE.md", "使用指南"),
        (project_root / "README_DASHBOARD.md", "快速入门"),
        (project_root / "DASHBOARD_INTEGRATION_NOTES.md", "整合说明"),
    ]
    
    files_ok = True
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            files_ok = False
    
    print()
    
    # 2. 检查模块导入
    print("🔍 检查模块导入...")
    print()
    
    limitup_monitor_path = project_root / "web" / "tabs" / "rdagent" / "limitup_monitor.py"
    module_ok = check_module_import(limitup_monitor_path, "limitup_monitor")
    
    print()
    
    # 3. 检查unified_dashboard整合
    print("🔗 检查unified_dashboard整合...")
    print()
    
    sys.path.insert(0, str(project_root / "web"))
    integration_ok = check_unified_dashboard_integration()
    
    print()
    
    # 4. 检查依赖
    print("📦 检查依赖...")
    print()
    
    dependencies = [
        "streamlit",
        "matplotlib",
        "pandas",
        "numpy",
    ]
    
    deps_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: 已安装")
        except ImportError:
            print(f"❌ {dep}: 未安装")
            deps_ok = False
    
    print()
    print("=" * 60)
    
    # 5. 总结
    if files_ok and module_ok and integration_ok and deps_ok:
        print("✅ 所有检查通过！Dashboard整合成功！")
        print()
        print("🚀 启动命令:")
        print("   - Windows: start_dashboard.bat")
        print("   - Linux/Mac: ./start_dashboard.sh")
        print("   - 手动: streamlit run web/unified_dashboard.py")
        print()
        print("📍 访问路径:")
        print("   Qlib → 数据管理 → 🎯涨停板监控")
        return 0
    else:
        print("❌ 部分检查未通过，请检查以上错误信息")
        print()
        if not deps_ok:
            print("💡 安装依赖:")
            print("   pip install streamlit matplotlib pandas numpy")
        return 1


if __name__ == "__main__":
    sys.exit(main())
