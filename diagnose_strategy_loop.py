"""
策略优化闭环依赖诊断脚本
快速检测并提供修复建议
"""

import sys

def diagnose():
    """诊断依赖问题"""
    
    print("=" * 70)
    print("策略优化闭环 - 依赖诊断")
    print("=" * 70)
    print()
    
    # Python版本
    print(f"Python版本: {sys.version}")
    print()
    
    issues = []
    
    # 1. 检查streamlit
    print("📝 [1/4] 检查 Streamlit...")
    try:
        import streamlit as st
        print(f"  ✅ streamlit {st.__version__}")
    except ImportError as e:
        print(f"  ❌ streamlit 未安装: {e}")
        issues.append("streamlit")
    print()
    
    # 2. 检查pandas
    print("📝 [2/4] 检查 pandas...")
    try:
        import pandas as pd
        print(f"  ✅ pandas {pd.__version__}")
    except Exception as e:
        print(f"  ❌ pandas 导入失败: {e}")
        issues.append("pandas")
    print()
    
    # 3. 检查pyarrow
    print("📝 [3/4] 检查 pyarrow...")
    try:
        import pyarrow as pa
        print(f"  ✅ pyarrow {pa.__version__}")
    except Exception as e:
        print(f"  ❌ pyarrow 导入失败: {e}")
        issues.append("pyarrow")
    print()
    
    # 4. 检查后端模块
    print("📝 [4/4] 检查 策略闭环后端...")
    sys.path.insert(0, r'G:\test\qilin_stack')
    try:
        from strategy.strategy_feedback_loop import StrategyFeedbackLoop
        print(f"  ✅ strategy_feedback_loop 正常")
    except Exception as e:
        print(f"  ❌ 后端模块导入失败: {e}")
        issues.append("backend")
    print()
    
    # 总结
    print("=" * 70)
    print("诊断结果")
    print("=" * 70)
    
    if not issues:
        print()
        print("🎉 所有依赖正常! 策略优化闭环应该可以使用。")
        print()
        print("✨ 启动Dashboard:")
        print("   streamlit run web/unified_dashboard.py")
        print()
        return 0
    else:
        print()
        print(f"⚠️  发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        
        # 提供修复建议
        if "pandas" in issues or "pyarrow" in issues:
            print("🔧 修复pandas/pyarrow问题:")
            print()
            print("   方法1 (推荐):")
            print("   pip uninstall pyarrow pandas -y")
            print("   pip install pandas pyarrow")
            print()
            print("   方法2:")
            print("   pip install --upgrade pandas pyarrow")
            print()
            print("   方法3 (conda用户):")
            print("   conda install pandas pyarrow -c conda-forge")
            print()
        
        if "streamlit" in issues:
            print("🔧 安装streamlit:")
            print("   pip install streamlit")
            print()
        
        if "backend" in issues:
            print("🔧 后端模块问题:")
            print("   检查文件是否存在: G:\\test\\qilin_stack\\strategy\\strategy_feedback_loop.py")
            print()
        
        print("📖 详细文档: fix_pandas_pyarrow.md")
        print()
        
        return 1


if __name__ == "__main__":
    sys.exit(diagnose())
