#!/usr/bin/env python
"""
检查并建议修复 RD-Agent 的环境变量配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path("G:/test/qilin_stack/.env")
load_dotenv(env_path)


def check_rdagent_env():
    """检查 RD-Agent 所需的环境变量"""
    
    print("=" * 70)
    print("🔍 检查 RD-Agent 环境变量配置")
    print("=" * 70)
    
    # RD-Agent 需要的环境变量
    required_vars = {
        "CHAT_MODEL": "聊天模型 (必需)",
        "EMBEDDING_MODEL": "嵌入模型 (必需)",
    }
    
    # DeepSeek 相关环境变量
    deepseek_vars = {
        "DEEPSEEK_API_KEY": "DeepSeek API密钥",
        "OPENAI_API_KEY": "OpenAI API密钥 (或DeepSeek密钥)",
        "OPENAI_API_BASE": "API基础URL",
    }
    
    # 其他可选环境变量
    optional_vars = {
        "LITELLM_PROXY_API_KEY": "Embedding代理密钥 (可选)",
        "LITELLM_PROXY_API_BASE": "Embedding代理URL (可选)",
        "REASONING_THINK_RM": "推理思考模式 (可选)",
    }
    
    print("\n📋 当前环境变量状态:\n")
    
    # 检查必需变量
    print("🔴 必需变量:")
    missing_required = []
    for var, desc in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {desc}")
            print(f"     当前值: {value}")
        else:
            print(f"  ❌ {var}: {desc} - 未设置")
            missing_required.append(var)
    
    # 检查 DeepSeek 变量
    print("\n🟡 DeepSeek 相关变量:")
    for var, desc in deepseek_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            # 部分隐藏密钥
            if "KEY" in var and len(value) > 8:
                display_value = value[:8] + "..." + value[-4:]
            else:
                display_value = value
            print(f"  ✅ {var}: {desc}")
            print(f"     当前值: {display_value}")
        else:
            print(f"  ⚠️  {var}: {desc} - 未设置或为默认值")
    
    # 检查可选变量
    print("\n🟢 可选变量:")
    for var, desc in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {desc}")
            print(f"     当前值: {value}")
        else:
            print(f"  ⚪ {var}: {desc} - 未设置 (可选)")
    
    # 分析配置问题
    print("\n" + "=" * 70)
    print("📊 配置分析")
    print("=" * 70)
    
    issues = []
    suggestions = []
    
    # 检查 CHAT_MODEL
    chat_model = os.getenv("CHAT_MODEL")
    if not chat_model:
        issues.append("❌ 缺少 CHAT_MODEL")
        suggestions.append("需要设置 CHAT_MODEL=deepseek/deepseek-chat")
    elif chat_model == "deepseek-chat":
        issues.append("⚠️  CHAT_MODEL 格式不正确")
        suggestions.append("RD-Agent 使用 LiteLLM，需要改为: CHAT_MODEL=deepseek/deepseek-chat")
    
    # 检查 EMBEDDING_MODEL
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        issues.append("❌ 缺少 EMBEDDING_MODEL")
        suggestions.append("DeepSeek 没有 embedding 模型，需要使用第三方")
        suggestions.append("推荐: EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3")
        suggestions.append("并配置 LITELLM_PROXY_API_KEY 和 LITELLM_PROXY_API_BASE")
    
    # 检查 DEEPSEEK_API_KEY
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not deepseek_key:
        if openai_key and openai_key.startswith("sk-"):
            issues.append("⚠️  使用 OPENAI_API_KEY 存储 DeepSeek 密钥")
            suggestions.append("建议改为使用 DEEPSEEK_API_KEY 更清晰")
        else:
            issues.append("❌ 缺少 DeepSeek API 密钥")
    
    # 输出问题和建议
    if issues:
        print("\n⚠️  发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    
    if suggestions:
        print("\n💡 修复建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    # 生成推荐配置
    print("\n" + "=" * 70)
    print("📝 推荐的 .env 配置 (针对 RD-Agent + DeepSeek)")
    print("=" * 70)
    print("""
# RD-Agent Chat Model (使用 DeepSeek)
CHAT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-04104c2d50864c30b307e6f6cfdf8fb4

# RD-Agent Embedding Model (DeepSeek没有embedding，使用SiliconFlow)
EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
LITELLM_PROXY_API_KEY=<你的SiliconFlow密钥>
LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1

# DeepSeek 推理模型设置 (如果使用 deepseek-reasoner)
# REASONING_THINK_RM=True

# Clash 代理配置 (如需要)
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
ALL_PROXY=http://127.0.0.1:7890

# 保留原有配置用于其他系统
OPENAI_API_KEY=sk-04104c2d50864c30b307e6f6cfdf8fb4
OPENAI_API_BASE=https://api.deepseek.com
LLM_PROVIDER=openai
LLM_MODEL=deepseek-chat
""")
    
    print("\n" + "=" * 70)
    print("⚠️  重要说明")
    print("=" * 70)
    print("""
1. RD-Agent 使用 LiteLLM 后端，需要特定的模型格式:
   - DeepSeek: deepseek/deepseek-chat (不是 deepseek-chat)
   - OpenAI: gpt-4o (直接写模型名)

2. DeepSeek 没有 embedding 模型，需要配置第三方:
   - 推荐使用 SiliconFlow 的 BAAI/bge-m3 模型
   - 需要注册 SiliconFlow 并获取 API key

3. 当前配置保留了原有变量，不会影响其他系统使用

4. 添加新配置后，运行测试:
   rdagent health_check
""")
    
    # 检查是否有 SiliconFlow 密钥
    if not os.getenv("LITELLM_PROXY_API_KEY"):
        print("\n🔗 获取 SiliconFlow API Key:")
        print("   访问: https://cloud.siliconflow.cn/")
        print("   注册并在控制台获取 API Key")


if __name__ == '__main__':
    check_rdagent_env()
