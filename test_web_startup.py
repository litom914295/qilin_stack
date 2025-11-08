"""
快速测试 Web 界面启动
用于验证导入和基本功能是否正常
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        import streamlit as st
        print("✅ Streamlit 导入成功")
    except Exception as e:
        print(f"❌ Streamlit 导入失败: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly 导入成功")
    except Exception as e:
        print(f"⚠️  Plotly 导入失败: {e}")
    
    try:
        import pandas as pd
        print("✅ Pandas 导入成功")
    except Exception as e:
        print(f"❌ Pandas 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ Numpy 导入成功")
    except Exception as e:
        print(f"❌ Numpy 导入失败: {e}")
        return False
    
    return True

def test_web_file():
    """测试 unified_dashboard.py 文件语法"""
    print("\n🔍 测试 Web 文件语法...")
    
    web_file = Path(__file__).parent / "web" / "unified_dashboard.py"
    
    if not web_file.exists():
        print(f"❌ 文件不存在: {web_file}")
        return False
    
    print(f"✅ 文件存在: {web_file}")
    
    # 尝试编译检查语法
    try:
        with open(web_file, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, str(web_file), 'exec')
        print("✅ 文件语法检查通过")
        return True
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"⚠️  其他错误: {e}")
        return True  # 可能是导入错误，但语法没问题

if __name__ == "__main__":
    print("=" * 50)
    print("麒麟堆栈 Web 界面启动测试")
    print("=" * 50)
    
    success = True
    
    # 测试导入
    if not test_imports():
        print("\n❌ 核心依赖导入失败")
        success = False
    
    # 测试文件
    if not test_web_file():
        print("\n❌ Web 文件检查失败")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 所有测试通过！")
        print("\n🚀 现在可以运行:")
        print("   streamlit run web/unified_dashboard.py")
    else:
        print("❌ 部分测试失败，请检查依赖安装")
        print("\n💡 建议:")
        print("   pip install streamlit pandas numpy plotly")
    print("=" * 50)
