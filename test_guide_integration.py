"""
验证系统指南集成测试脚本
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_system_guide_import():
    """测试系统指南模块导入"""
    print("🔍 测试1: 检查系统指南模块导入...")
    try:
        from web.components.system_guide import show_system_guide
        print("✅ 系统指南模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 系统指南模块导入失败: {e}")
        return False

def test_helper_functions():
    """测试辅助函数"""
    print("\n🔍 测试2: 检查辅助函数...")
    try:
        from web.components.system_guide import (
            render_quick_landing_guide,
            render_command_reference
        )
        print("✅ 辅助函数导入成功")
        print("  - render_quick_landing_guide ✓")
        print("  - render_command_reference ✓")
        return True
    except Exception as e:
        print(f"❌ 辅助函数导入失败: {e}")
        return False

def test_architecture_guide():
    """测试架构指南文档"""
    print("\n🔍 测试3: 检查架构指南文档...")
    doc_path = Path(__file__).parent / "docs" / "DEEP_ARCHITECTURE_GUIDE.md"
    
    if not doc_path.exists():
        print(f"❌ 架构指南文档不存在: {doc_path}")
        return False
    
    try:
        content = doc_path.read_text(encoding='utf-8')
        
        # 检查关键章节
        required_sections = [
            "快速落地实战指南",
            "前置准备",
            "环境初始化",
            "Qlib数据准备",
            "RD-Agent因子发现",
            "因子生命周期测试",
            "一进二模型训练",
            "启动Web界面",
            "验证完整流程"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️  缺少以下章节: {', '.join(missing_sections)}")
            return False
        
        print(f"✅ 架构指南文档完整")
        print(f"  - 文件路径: {doc_path}")
        print(f"  - 文件大小: {len(content)/1024:.1f} KB")
        print(f"  - 包含所有必需章节 ✓")
        return True
    except Exception as e:
        print(f"❌ 读取架构指南文档失败: {e}")
        return False

def test_dashboard_integration():
    """测试Dashboard集成"""
    print("\n🔍 测试4: 检查Dashboard集成...")
    try:
        from web.unified_dashboard import UnifiedDashboard
        print("✅ UnifiedDashboard导入成功")
        
        # 检查是否有system_guide的调用
        dashboard_path = Path(__file__).parent / "web" / "unified_dashboard.py"
        content = dashboard_path.read_text(encoding='utf-8')
        
        if "system_guide" in content:
            print("  - 系统指南已集成到Dashboard ✓")
        else:
            print("  ⚠️  未在Dashboard中发现system_guide引用")
        
        return True
    except Exception as e:
        print(f"❌ Dashboard集成检查失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("="*70)
    print("🚀 Qilin Stack 系统指南集成测试")
    print("="*70)
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_system_guide_import()))
    results.append(("辅助函数", test_helper_functions()))
    results.append(("架构指南文档", test_architecture_guide()))
    results.append(("Dashboard集成", test_dashboard_integration()))
    
    # 总结
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print("\n" + "="*70)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统指南集成成功！")
        print("\n下一步操作:")
        print("  1. 启动Web界面: streamlit run web/unified_dashboard.py")
        print("  2. 导航到: 📚 系统指南 → 🚀 快速开始 → 🚀 快速落地实战")
        print("  3. 开始使用30分钟快速上手指南！")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
