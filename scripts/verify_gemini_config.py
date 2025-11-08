"""
快速验证 Gemini 配置
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("🔍 验证 Gemini 配置")
print("=" * 70)

# 1. 检查环境变量
print("\n📋 步骤1: 检查环境变量")
import os
from dotenv import load_dotenv

env_file = Path(__file__).parent.parent / ".env"
load_dotenv(env_file)

llm_provider = os.getenv("LLM_PROVIDER")
google_key = os.getenv("GOOGLE_API_KEY")
deep_think = os.getenv("DEEP_THINK_LLM")
quick_think = os.getenv("QUICK_THINK_LLM")

print(f"   Provider: {llm_provider}")
print(f"   Google Key: {'✅ 已配置' if google_key else '❌ 未配置'}")
print(f"   深度模型: {deep_think}")
print(f"   快速模型: {quick_think}")

if llm_provider != "google":
    print("\n❌ LLM_PROVIDER 不是 google")
    sys.exit(1)

if not google_key or "your_" in google_key:
    print("\n❌ GOOGLE_API_KEY 未正确配置")
    sys.exit(1)

print("✅ 环境变量配置正确")

# 2. 测试适配器导入和初始化
print("\n📋 步骤2: 测试适配器初始化")

try:
    from tradingagents_integration.tradingagents_cn_plus_adapter import create_tradingagents_cn_plus_adapter
    print("✅ 适配器导入成功")
    
    adapter = create_tradingagents_cn_plus_adapter()
    print("✅ 适配器实例创建成功")
    
    status = adapter.get_status()
    print(f"\n适配器状态:")
    print(f"   可用: {status.get('available')}")
    print(f"   模式: {status.get('mode')}")
    
    if status.get('error'):
        print(f"   ❌ 错误: {status['error']}")
        sys.exit(1)
    
    if not status.get('available'):
        print("   ❌ 适配器不可用")
        sys.exit(1)
    
    print("✅ 适配器完全可用")
    
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 检查 LangChain Google Genai
print("\n📋 步骤3: 检查 Google Genai 集成")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✅ langchain_google_genai 已安装")
    
    # 测试是否能创建实例（不调用API）
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_key,
        temperature=0.7
    )
    print("✅ ChatGoogleGenerativeAI 实例创建成功")
    
except Exception as e:
    print(f"❌ Google Genai 集成失败: {e}")
    sys.exit(1)

# 总结
print("\n" + "=" * 70)
print("🎉 所有检查通过！")
print("=" * 70)
print("\n✅ Gemini 配置完全正确")
print("✅ TradingAgents-CN-Plus 适配器就绪")
print("✅ 可以开始进行深度分析")

print("\n📝 下一步:")
print("   1. 启动 Streamlit: streamlit run web/main.py")
print("   2. 进入 TradingAgents → 决策分析")
print("   3. 输入股票代码并选择 '完整' 深度")
print("   4. 开始分析！")
print("\n" + "=" * 70)
