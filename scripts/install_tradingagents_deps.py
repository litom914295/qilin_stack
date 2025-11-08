"""
TradingAgents-CN-Plus 依赖安装脚本
快速安装所需的依赖包
"""

import subprocess
import sys
from pathlib import Path


def check_module(module_name: str) -> bool:
    """检查模块是否已安装"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def install_package(package: str):
    """安装单个包"""
    print(f"📦 正在安装 {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 TradingAgents-CN-Plus 依赖安装工具")
    print("=" * 60)
    
    # 核心依赖列表
    core_deps = [
        ("langgraph", "langgraph"),
        ("langchain_anthropic", "langchain-anthropic"),
        ("langchain_openai", "langchain-openai"),
        ("langchain_google_genai", "langchain-google-genai"),
        ("akshare", "akshare"),
        ("yfinance", "yfinance"),
        ("pandas", "pandas"),
        ("openai", "openai>=1.0.0"),
        ("google.generativeai", "google-generativeai>=0.8.0"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
    ]
    
    # 检查当前状态
    print("\n📊 检查当前依赖状态...")
    missing = []
    installed = []
    
    for module_name, package_name in core_deps:
        if check_module(module_name):
            print(f"✅ {module_name:30s} - 已安装")
            installed.append(module_name)
        else:
            print(f"❌ {module_name:30s} - 未安装")
            missing.append((module_name, package_name))
    
    print(f"\n📈 统计: {len(installed)}/{len(core_deps)} 已安装")
    
    if not missing:
        print("\n🎉 所有依赖都已安装！")
        return
    
    print(f"\n⚠️  缺少 {len(missing)} 个依赖包")
    print("\n是否开始安装? (y/n): ", end="")
    
    choice = input().strip().lower()
    if choice != 'y':
        print("❌ 取消安装")
        return
    
    # 开始安装
    print("\n🔧 开始安装依赖...")
    success_count = 0
    failed = []
    
    for module_name, package_name in missing:
        if install_package(package_name):
            success_count += 1
        else:
            failed.append(package_name)
    
    # 总结
    print("\n" + "=" * 60)
    print(f"✅ 成功安装: {success_count}/{len(missing)}")
    
    if failed:
        print(f"❌ 安装失败: {len(failed)}")
        print("失败的包:")
        for pkg in failed:
            print(f"  - {pkg}")
        print("\n💡 请手动安装失败的包:")
        print(f"pip install {' '.join(failed)}")
    else:
        print("\n🎉 所有依赖安装完成！")
        print("\n📝 下一步:")
        print("1. 配置环境变量 (LLM API密钥)")
        print("2. 运行 streamlit 应用")
        print("3. 在决策分析tab中选择 '完整' 深度进行分析")


if __name__ == "__main__":
    main()
