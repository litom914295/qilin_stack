"""
策略优化闭环集成验证脚本
验证策略优化闭环是否正确集成到unified_dashboard.py
"""

import sys
from pathlib import Path

def verify_integration():
    """验证集成完整性"""
    
    print("=" * 70)
    print("策略优化闭环集成验证")
    print("=" * 70)
    print()
    
    results = []
    
    # 1. 检查后端模块
    print("📝 [1/5] 检查后端模块...")
    backend_file = Path("strategy/strategy_feedback_loop.py")
    if backend_file.exists():
        print(f"  ✅ 后端模块存在: {backend_file}")
        with open(backend_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class StrategyFeedbackLoop' in content:
                print("  ✅ StrategyFeedbackLoop 类存在")
                results.append(True)
            else:
                print("  ❌ StrategyFeedbackLoop 类未找到")
                results.append(False)
    else:
        print(f"  ❌ 后端模块不存在: {backend_file}")
        results.append(False)
    print()
    
    # 2. 检查UI组件
    print("📝 [2/5] 检查UI组件...")
    ui_file = Path("web/components/strategy_loop_ui.py")
    if ui_file.exists():
        print(f"  ✅ UI组件存在: {ui_file}")
        with open(ui_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class StrategyLoopUI' in content and 'def render_strategy_loop_ui' in content:
                print("  ✅ StrategyLoopUI 类和 render_strategy_loop_ui 函数存在")
                results.append(True)
            else:
                print("  ❌ 必需的类或函数未找到")
                results.append(False)
    else:
        print(f"  ❌ UI组件不存在: {ui_file}")
        results.append(False)
    print()
    
    # 3. 检查集成点 - advanced_features_tab.py
    print("📝 [3/5] 检查集成入口...")
    integration_file = Path("web/tabs/advanced_features_tab.py")
    if integration_file.exists():
        print(f"  ✅ 集成文件存在: {integration_file}")
        with open(integration_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            checks = [
                ('from components.strategy_loop_ui import render_strategy_loop_ui', '导入语句'),
                ('STRATEGY_LOOP_AVAILABLE', '可用性标志'),
                ('"🔥 策略优化闭环"', 'Tab标签'),
                ('render_strategy_loop_ui()', '渲染函数调用')
            ]
            
            all_passed = True
            for check_str, desc in checks:
                if check_str in content:
                    print(f"  ✅ {desc} 存在")
                else:
                    print(f"  ❌ {desc} 未找到: {check_str}")
                    all_passed = False
            
            results.append(all_passed)
    else:
        print(f"  ❌ 集成文件不存在: {integration_file}")
        results.append(False)
    print()
    
    # 4. 检查文档
    print("📝 [4/5] 检查文档...")
    docs = [
        ('docs/STRATEGY_LOOP_INTEGRATION.md', '集成说明文档'),
        ('docs/STRATEGY_FEEDBACK_LOOP.md', '完整指南'),
        ('STRATEGY_LOOP_QUICKSTART.md', '快速开始'),
        ('strategy/README.md', '模块说明')
    ]
    
    doc_results = []
    for doc_path, doc_name in docs:
        if Path(doc_path).exists():
            print(f"  ✅ {doc_name}: {doc_path}")
            doc_results.append(True)
        else:
            print(f"  ❌ {doc_name}不存在: {doc_path}")
            doc_results.append(False)
    
    results.append(all(doc_results))
    print()
    
    # 5. 检查README更新
    print("📝 [5/5] 检查README更新...")
    readme_file = Path("README.md")
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            checks = [
                ('Web Dashboard', 'Web Dashboard章节'),
                ('策略优化闭环', '策略优化闭环提及'),
                ('高级功能', '高级功能提及'),
                ('STRATEGY_LOOP_INTEGRATION.md', '文档链接')
            ]
            
            readme_passed = True
            for check_str, desc in checks:
                if check_str in content:
                    print(f"  ✅ {desc} 已更新")
                else:
                    print(f"  ⚠️  {desc} 未找到 (可选)")
                    # 不标记为失败,因为README可能有不同格式
            
            results.append(True)  # README检查作为可选项
    else:
        print(f"  ❌ README.md 不存在")
        results.append(False)
    print()
    
    # 总结
    print("=" * 70)
    print("验证结果汇总")
    print("=" * 70)
    
    test_names = [
        "后端模块 (strategy_feedback_loop.py)",
        "UI组件 (strategy_loop_ui.py)",
        "集成入口 (advanced_features_tab.py)",
        "文档完整性",
        "README更新"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results), 1):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"[{i}/5] {name}: {status}")
    
    print()
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100
    
    print(f"总体通过率: {passed}/{total} ({percentage:.0f}%)")
    
    if all(results):
        print()
        print("🎉 恭喜! 策略优化闭环已成功集成到麒麟系统!")
        print()
        print("✨ 下一步:")
        print("   1. 启动Dashboard: streamlit run web/unified_dashboard.py")
        print("   2. 访问浏览器: http://localhost:8501")
        print("   3. 导航到: 🚀 高级功能 → 🔥 策略优化闭环")
        print("   4. 阅读文档: docs/STRATEGY_LOOP_INTEGRATION.md")
        print()
        return 0
    else:
        print()
        print("⚠️  部分检查未通过,请检查上述失败项")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(verify_integration())
