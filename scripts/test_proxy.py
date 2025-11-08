"""
检测Clash代理并测试DeepSeek连接
"""

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
env_file = Path(__file__).parent.parent / ".env"
load_dotenv(env_file)

print("=" * 70)
print("🔍 Clash代理和DeepSeek连接测试")
print("=" * 70)

# 1. 检查代理配置
print("\n📋 步骤1: 检查代理配置")
http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

if http_proxy:
    print(f"✅ HTTP_PROXY: {http_proxy}")
else:
    print("❌ HTTP_PROXY 未配置")

if https_proxy:
    print(f"✅ HTTPS_PROXY: {https_proxy}")
else:
    print("❌ HTTPS_PROXY 未配置")

# 2. 测试代理连接
print("\n📋 步骤2: 测试代理连接")

if http_proxy:
    proxies = {
        'http': http_proxy,
        'https': https_proxy or http_proxy
    }
    
    try:
        print(f"⏳ 测试代理连接: {http_proxy}")
        response = requests.get(
            "https://www.google.com",
            proxies=proxies,
            timeout=5
        )
        if response.status_code == 200:
            print("✅ 代理连接正常")
        else:
            print(f"⚠️ 代理返回状态码: {response.status_code}")
    except requests.exceptions.ProxyError as e:
        print(f"❌ 代理连接失败: {e}")
        print("\n💡 可能的原因:")
        print("   1. Clash未启动")
        print("   2. 代理端口错误")
        print("   3. 系统代理未启用")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
else:
    print("⚠️ 未配置代理，将尝试直连")
    proxies = None

# 3. 测试DeepSeek API
print("\n📋 步骤3: 测试DeepSeek API连接")

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")

if not api_key or "your_" in api_key:
    print("❌ DeepSeek API密钥未配置")
else:
    print(f"✅ API密钥: {api_key[:15]}...")
    print(f"✅ API基地址: {api_base}")
    
    try:
        print("⏳ 测试DeepSeek API连接...")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "你好"}
            ],
            "max_tokens": 10
        }
        
        response = requests.post(
            f"{api_base}/v1/chat/completions",
            headers=headers,
            json=data,
            proxies=proxies,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ DeepSeek API连接成功！")
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']['content']
                print(f"✅ API响应: {message}")
        elif response.status_code == 401:
            print("❌ API密钥无效")
            print("💡 请检查OPENAI_API_KEY配置")
        else:
            print(f"❌ API返回错误: {response.status_code}")
            print(f"   响应: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("❌ API请求超时")
        print("💡 可能需要配置代理")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接失败: {e}")
        print("💡 可能的原因:")
        print("   1. 网络问题")
        print("   2. 需要代理但未配置")
        print("   3. 代理配置错误")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

# 4. 给出建议
print("\n" + "=" * 70)
print("📋 配置建议")
print("=" * 70)

if not http_proxy:
    print("\n💡 如果需要通过Clash代理访问DeepSeek:")
    print("   1. 启动Clash")
    print("   2. 在.env文件中添加:")
    print("      HTTP_PROXY=http://127.0.0.1:7890")
    print("      HTTPS_PROXY=http://127.0.0.1:7890")
    print("   3. 如果Clash端口不是7890，请修改端口号")

print("\n💡 检查Clash端口的方法:")
print("   打开Clash → Settings → Port")
print("   常见端口: 7890, 7891, 10808")

print("\n" + "=" * 70)
