"""
环境变量配置检查脚本
检查LLM API密钥和TradingAgents配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def check_api_key(name: str, required: bool = False) -> bool:
    """检查API密钥是否配置"""
    value = os.getenv(name, "")
    
    # 检查是否为占位符
    placeholder_keywords = ["your_", "YOUR_", "here"]
    is_placeholder = any(kw in value for kw in placeholder_keywords)
    
    if not value or is_placeholder:
        status = "❌ 未配置" if required else "⚠️  未配置（可选）"
        return False, status
    else:
        # 隐藏密钥，只显示前后几位
        if len(value) > 10:
            masked = f"{value[:8]}...{value[-4:]}"
        else:
            masked = "***"
        return True, f"✅ 已配置: {masked}"


def main():
    print("=" * 70)
    print("🔍 环境变量配置检查")
    print("=" * 70)
    
    # 检查.env文件
    print(f"\n📄 配置文件: {env_path}")
    if env_path.exists():
        print("✅ .env 文件存在")
    else:
        print("❌ .env 文件不存在")
        print("💡 请复制 .env.example 为 .env 并填写API密钥")
        return
    
    print("\n" + "=" * 70)
    print("🔑 API 密钥配置")
    print("=" * 70)
    
    # 核心API密钥
    api_keys = [
        ("GOOGLE_API_KEY", "Google Gemini", True),
        ("OPENAI_API_KEY", "OpenAI/DeepSeek", False),
        ("ANTHROPIC_API_KEY", "Anthropic Claude", False),
    ]
    
    any_configured = False
    for key_name, provider, required in api_keys:
        configured, status = check_api_key(key_name, required)
        print(f"\n{provider:20s}: {status}")
        if configured:
            any_configured = True
    
    if not any_configured:
        print("\n❌ 警告: 没有配置任何API密钥！")
        print("💡 至少需要配置一个LLM提供商的API密钥")
    
    # LLM Provider配置
    print("\n" + "=" * 70)
    print("⚙️  LLM Provider 配置")
    print("=" * 70)
    
    llm_provider = os.getenv("LLM_PROVIDER", "未设置")
    llm_model = os.getenv("LLM_MODEL", "未设置")
    api_base = os.getenv("OPENAI_API_BASE", "默认")
    
    print(f"\nLLM Provider: {llm_provider}")
    print(f"LLM Model:    {llm_model}")
    print(f"API Base:     {api_base}")
    
    # TradingAgents配置
    print("\n" + "=" * 70)
    print("🤖 TradingAgents-CN-Plus 配置")
    print("=" * 70)
    
    ta_path = os.getenv("TRADINGAGENTS_PATH", "")
    deep_think = os.getenv("DEEP_THINK_LLM", "未设置")
    quick_think = os.getenv("QUICK_THINK_LLM", "未设置")
    
    print(f"\n项目路径:     {ta_path}")
    if ta_path:
        if Path(ta_path).exists():
            print("              ✅ 路径存在")
        else:
            print("              ❌ 路径不存在")
    
    print(f"深度思考模型: {deep_think}")
    print(f"快速思考模型: {quick_think}")
    
    # 依赖检查
    print("\n" + "=" * 70)
    print("📦 关键依赖包检查")
    print("=" * 70)
    
    deps = [
        "langgraph",
        "langchain_anthropic",
        "langchain_openai",
        "langchain_google_genai",
        "akshare",
        "yfinance",
        "pandas",
        "streamlit"
    ]
    
    installed = []
    missing = []
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
            installed.append(dep)
        except ImportError:
            print(f"❌ {dep}")
            missing.append(dep)
    
    print(f"\n📊 统计: {len(installed)}/{len(deps)} 已安装")
    
    if missing:
        print(f"\n⚠️  缺少 {len(missing)} 个依赖包")
        print("💡 运行以下命令安装:")
        print(f"   python scripts/install_tradingagents_deps.py")
    
    # 总结
    print("\n" + "=" * 70)
    print("📋 配置总结")
    print("=" * 70)
    
    issues = []
    
    if not any_configured:
        issues.append("❌ 没有配置API密钥")
    
    if ta_path and not Path(ta_path).exists():
        issues.append("❌ TradingAgents项目路径不存在")
    
    if missing:
        issues.append(f"❌ 缺少 {len(missing)} 个依赖包")
    
    if issues:
        print("\n发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 请按照上述提示解决问题")
    else:
        print("\n✅ 所有配置正常！")
        print("🚀 可以开始使用 TradingAgents-CN-Plus 进行深度分析")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
