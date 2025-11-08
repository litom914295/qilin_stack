"""
Phase 1 Web集成测试脚本
验证所有组件是否正确集成
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有必需模块是否可导入"""
    print("=" * 60)
    print("测试1: 检查模块导入")
    print("=" * 60)
    
    tests = []
    
    # 测试Phase 1 Pipeline组件
    try:
        from web.components.phase1_pipeline_panel import Phase1PipelinePanel, show_phase1_pipeline_panel
        print("✅ Phase1PipelinePanel 导入成功")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Phase1PipelinePanel 导入失败: {e}")
        tests.append(False)
    
    # 测试集成面板
    try:
        from web.components.auction_integration import show_integration_panel
        print("✅ auction_integration 导入成功")
        tests.append(True)
    except ImportError as e:
        print(f"❌ auction_integration 导入失败: {e}")
        tests.append(False)
    
    # 测试主视图
    try:
        from web.auction_decision_view import AuctionDecisionView
        print("✅ AuctionDecisionView 导入成功")
        tests.append(True)
    except ImportError as e:
        print(f"❌ AuctionDecisionView 导入失败: {e}")
        tests.append(False)
    
    # 测试UnifiedPhase1Pipeline
    try:
        from qlib_enhanced.unified_phase1_pipeline import UnifiedPhase1Pipeline
        print("✅ UnifiedPhase1Pipeline 导入成功")
        tests.append(True)
    except ImportError as e:
        print(f"⚠️ UnifiedPhase1Pipeline 导入失败: {e}")
        print("   (这是可选的，Panel会使用演示模式)")
        tests.append(True)  # 不算失败
    
    return all(tests)


def test_file_existence():
    """测试必需文件是否存在"""
    print("\n" + "=" * 60)
    print("测试2: 检查文件存在性")
    print("=" * 60)
    
    required_files = [
        "web/components/phase1_pipeline_panel.py",
        "web/components/auction_integration.py",
        "web/auction_decision_view.py",
        "docs/PHASE1_USAGE_GUIDE.md",
        "qlib_enhanced/unified_phase1_pipeline.py"
    ]
    
    tests = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path} 存在")
            tests.append(True)
        else:
            print(f"❌ {file_path} 不存在")
            tests.append(False)
    
    return all(tests)


def test_panel_instantiation():
    """测试组件实例化"""
    print("\n" + "=" * 60)
    print("测试3: 测试组件实例化")
    print("=" * 60)
    
    tests = []
    
    try:
        from web.components.phase1_pipeline_panel import Phase1PipelinePanel
        panel = Phase1PipelinePanel()
        print("✅ Phase1PipelinePanel 实例化成功")
        tests.append(True)
    except Exception as e:
        print(f"❌ Phase1PipelinePanel 实例化失败: {e}")
        tests.append(False)
    
    try:
        from web.auction_decision_view import AuctionDecisionView
        view = AuctionDecisionView()
        print("✅ AuctionDecisionView 实例化成功")
        tests.append(True)
    except Exception as e:
        print(f"❌ AuctionDecisionView 实例化失败: {e}")
        tests.append(False)
    
    return all(tests)


def test_documentation():
    """测试文档内容"""
    print("\n" + "=" * 60)
    print("测试4: 检查文档内容")
    print("=" * 60)
    
    doc_path = project_root / "docs" / "PHASE1_USAGE_GUIDE.md"
    
    if doc_path.exists():
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键章节
        key_sections = [
            "快速开始",
            "核心模块介绍",
            "完整Pipeline使用",
            "最佳实践",
            "常见问题"
        ]
        
        tests = []
        for section in key_sections:
            if section in content:
                print(f"✅ 文档包含「{section}」章节")
                tests.append(True)
            else:
                print(f"⚠️ 文档缺少「{section}」章节")
                tests.append(False)
        
        print(f"\n📄 文档总长度: {len(content)} 字符")
        return all(tests)
    else:
        print("❌ 文档不存在")
        return False


def test_integration_features():
    """测试集成特性"""
    print("\n" + "=" * 60)
    print("测试5: 检查集成特性")
    print("=" * 60)
    
    # 检查auction_decision_view.py是否包含Phase 1相关代码
    view_path = project_root / "web" / "auction_decision_view.py"
    
    with open(view_path, 'r', encoding='utf-8') as f:
        view_content = f.read()
    
    features = [
        ("Phase 1 Pipeline标签页", "Phase 1 Pipeline"),
        ("_render_phase1_pipeline方法", "_render_phase1_pipeline"),
        ("show_phase1_pipeline_panel导入", "show_phase1_pipeline_panel")
    ]
    
    tests = []
    for feature_name, keyword in features:
        if keyword in view_content:
            print(f"✅ 包含{feature_name}")
            tests.append(True)
        else:
            print(f"❌ 缺少{feature_name}")
            tests.append(False)
    
    # 检查auction_integration.py是否突出Phase 1
    integration_path = project_root / "web" / "components" / "auction_integration.py"
    
    with open(integration_path, 'r', encoding='utf-8') as f:
        integration_content = f.read()
    
    if "Phase 1 完整集成模块" in integration_content:
        print("✅ 集成面板突出显示Phase 1")
        tests.append(True)
    else:
        print("❌ 集成面板未突出Phase 1")
        tests.append(False)
    
    return all(tests)


def main():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("Phase 1 Web集成测试")
    print("🧪" * 30 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("模块导入", test_imports()))
    results.append(("文件存在性", test_file_existence()))
    results.append(("组件实例化", test_panel_instantiation()))
    results.append(("文档内容", test_documentation()))
    results.append(("集成特性", test_integration_features()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print("\n" + "=" * 60)
    print(f"总计: {total_passed}/{total_tests} 测试通过")
    print("=" * 60)
    
    if total_passed == total_tests:
        print("\n🎉 所有测试通过！Phase 1已成功集成到Web界面。")
        print("\n下一步：")
        print("1. 启动Streamlit应用: streamlit run web/unified_dashboard.py")
        print("2. 在浏览器中打开应用")
        print("3. 切换到「🚀 Phase 1 Pipeline」标签页")
        print("4. 尝试运行演示Pipeline")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
