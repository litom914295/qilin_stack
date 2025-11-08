"""
LLM配置管理界面

功能:
1. 模型提供商选择 (OpenAI/Claude/本地模型/其他)
2. API Key安全管理
3. 模型参数配置
4. 配置保存和加载
5. 连接测试
6. 使用统计
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import os


class LLMConfigManager:
    """LLM配置管理器"""
    
    CONFIG_FILE = Path("config/llm_config.json")
    
    def __init__(self):
        self.init_session_state()
        self.ensure_config_dir()
    
    def init_session_state(self):
        """初始化session状态"""
        if 'llm_config' not in st.session_state:
            st.session_state.llm_config = self.load_config()
        if 'llm_test_result' not in st.session_state:
            st.session_state.llm_test_result = None
    
    def ensure_config_dir(self):
        """确保配置目录存在"""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict:
        """加载配置"""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"配置加载失败: {e}")
        
        # 默认配置
        return {
            'provider': 'OpenAI',
            'openai': {
                'api_key': '',
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 4096,
                'top_p': 0.9
            },
            'claude': {
                'api_key': '',
                'model': 'claude-3-5-sonnet-20241022',
                'temperature': 0.7,
                'max_tokens': 8192
            },
            'local': {
                'base_url': 'http://localhost:11434',
                'model': 'llama2',
                'temperature': 0.7,
                'max_tokens': 2048
            },
            'azure': {
                'api_key': '',
                'endpoint': '',
                'deployment_name': '',
                'api_version': '2024-02-15-preview',
                'temperature': 0.7,
                'max_tokens': 4096
            },
            'usage_stats': {
                'total_requests': 0,
                'total_tokens': 0,
                'last_used': None
            }
        }
    
    def save_config(self, config: Dict):
        """保存配置"""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"配置保存失败: {e}")
            return False
    
    def mask_api_key(self, api_key: str) -> str:
        """遮盖API Key"""
        if not api_key or len(api_key) < 8:
            return api_key
        return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"
    
    def test_connection(self, provider: str, config: Dict) -> Dict:
        """测试连接"""
        try:
            if provider == 'OpenAI':
                return self._test_openai(config)
            elif provider == 'Claude':
                return self._test_claude(config)
            elif provider == 'Local':
                return self._test_local(config)
            elif provider == 'Azure':
                return self._test_azure(config)
            else:
                return {'success': False, 'message': '不支持的提供商'}
        except Exception as e:
            return {'success': False, 'message': f'连接失败: {str(e)}'}
    
    def _test_openai(self, config: Dict) -> Dict:
        """测试OpenAI连接"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=config.get('api_key'),
                base_url=config.get('base_url')
            )
            
            # 简单的测试请求
            response = client.chat.completions.create(
                model=config.get('model', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                'success': True,
                'message': '连接成功!',
                'model': response.model,
                'tokens_used': response.usage.total_tokens
            }
        except Exception as e:
            return {'success': False, 'message': f'OpenAI连接失败: {str(e)}'}
    
    def _test_claude(self, config: Dict) -> Dict:
        """测试Claude连接"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.get('api_key'))
            
            message = client.messages.create(
                model=config.get('model', 'claude-3-5-sonnet-20241022'),
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            return {
                'success': True,
                'message': '连接成功!',
                'model': message.model,
                'tokens_used': message.usage.input_tokens + message.usage.output_tokens
            }
        except Exception as e:
            return {'success': False, 'message': f'Claude连接失败: {str(e)}'}
    
    def _test_local(self, config: Dict) -> Dict:
        """测试本地模型连接"""
        try:
            import requests
            base_url = config.get('base_url', 'http://localhost:11434')
            
            # 测试Ollama API
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'success': True,
                    'message': f'连接成功! 发现 {len(models)} 个模型',
                    'available_models': [m.get('name') for m in models]
                }
            else:
                return {'success': False, 'message': f'连接失败: HTTP {response.status_code}'}
        except Exception as e:
            return {'success': False, 'message': f'本地模型连接失败: {str(e)}'}
    
    def _test_azure(self, config: Dict) -> Dict:
        """测试Azure OpenAI连接"""
        try:
            import openai
            client = openai.AzureOpenAI(
                api_key=config.get('api_key'),
                azure_endpoint=config.get('endpoint'),
                api_version=config.get('api_version')
            )
            
            response = client.chat.completions.create(
                model=config.get('deployment_name'),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                'success': True,
                'message': '连接成功!',
                'deployment': config.get('deployment_name'),
                'tokens_used': response.usage.total_tokens
            }
        except Exception as e:
            return {'success': False, 'message': f'Azure连接失败: {str(e)}'}
    
    def render_provider_selection(self):
        """渲染提供商选择"""
        st.subheader("🤖 模型提供商")
        
        providers = ['OpenAI', 'Claude', 'Azure', 'Local', '其他']
        
        current_provider = st.session_state.llm_config.get('provider', 'OpenAI')
        
        selected = st.selectbox(
            "选择LLM提供商",
            providers,
            index=providers.index(current_provider) if current_provider in providers else 0,
            help="选择要使用的大语言模型提供商"
        )
        
        st.session_state.llm_config['provider'] = selected
        
        # 提供商描述
        descriptions = {
            'OpenAI': '🔵 OpenAI官方API (GPT-3.5/GPT-4)',
            'Claude': '🟣 Anthropic Claude (Claude 3/3.5)',
            'Azure': '🔷 Azure OpenAI Service',
            'Local': '🟢 本地部署模型 (Ollama/LM Studio)',
            '其他': '⚪ 其他兼容OpenAI API的服务'
        }
        
        st.info(descriptions.get(selected, ''))
    
    def render_openai_config(self):
        """渲染OpenAI配置"""
        st.subheader("🔵 OpenAI 配置")
        
        config = st.session_state.llm_config.get('openai', {})
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            api_key = st.text_input(
                "API Key",
                value=config.get('api_key', ''),
                type="password",
                help="从 https://platform.openai.com/api-keys 获取"
            )
            config['api_key'] = api_key
        
        with col2:
            if api_key:
                st.text("已设置")
                st.caption(self.mask_api_key(api_key))
        
        base_url = st.text_input(
            "Base URL",
            value=config.get('base_url', 'https://api.openai.com/v1'),
            help="API基础URL,使用代理时可修改"
        )
        config['base_url'] = base_url
        
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.selectbox(
                "模型",
                ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
                index=['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'].index(config.get('model', 'gpt-4'))
            )
            config['model'] = model
        
        with col2:
            temperature = st.slider(
                "Temperature",
                0.0, 2.0,
                float(config.get('temperature', 0.7)),
                0.1,
                help="控制输出随机性,越高越随机"
            )
            config['temperature'] = temperature
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_tokens = st.number_input(
                "Max Tokens",
                100, 128000,
                int(config.get('max_tokens', 4096)),
                100,
                help="最大生成token数"
            )
            config['max_tokens'] = max_tokens
        
        with col2:
            top_p = st.slider(
                "Top P",
                0.0, 1.0,
                float(config.get('top_p', 0.9)),
                0.05,
                help="核采样参数"
            )
            config['top_p'] = top_p
        
        st.session_state.llm_config['openai'] = config
    
    def render_claude_config(self):
        """渲染Claude配置"""
        st.subheader("🟣 Claude 配置")
        
        config = st.session_state.llm_config.get('claude', {})
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            api_key = st.text_input(
                "API Key",
                value=config.get('api_key', ''),
                type="password",
                help="从 https://console.anthropic.com 获取"
            )
            config['api_key'] = api_key
        
        with col2:
            if api_key:
                st.text("已设置")
                st.caption(self.mask_api_key(api_key))
        
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.selectbox(
                "模型",
                ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                index=0
            )
            config['model'] = model
        
        with col2:
            temperature = st.slider(
                "Temperature",
                0.0, 1.0,
                float(config.get('temperature', 0.7)),
                0.1
            )
            config['temperature'] = temperature
        
        max_tokens = st.number_input(
            "Max Tokens",
            100, 200000,
            int(config.get('max_tokens', 8192)),
            100
        )
        config['max_tokens'] = max_tokens
        
        st.session_state.llm_config['claude'] = config
    
    def render_azure_config(self):
        """渲染Azure配置"""
        st.subheader("🔷 Azure OpenAI 配置")
        
        config = st.session_state.llm_config.get('azure', {})
        
        api_key = st.text_input(
            "API Key",
            value=config.get('api_key', ''),
            type="password"
        )
        config['api_key'] = api_key
        
        endpoint = st.text_input(
            "Endpoint",
            value=config.get('endpoint', ''),
            placeholder="https://your-resource.openai.azure.com/",
            help="Azure资源的端点URL"
        )
        config['endpoint'] = endpoint
        
        col1, col2 = st.columns(2)
        
        with col1:
            deployment_name = st.text_input(
                "Deployment Name",
                value=config.get('deployment_name', ''),
                help="部署的模型名称"
            )
            config['deployment_name'] = deployment_name
        
        with col2:
            api_version = st.selectbox(
                "API Version",
                ['2024-02-15-preview', '2023-12-01-preview', '2023-05-15'],
                index=0
            )
            config['api_version'] = api_version
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                0.0, 2.0,
                float(config.get('temperature', 0.7)),
                0.1
            )
            config['temperature'] = temperature
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                100, 128000,
                int(config.get('max_tokens', 4096)),
                100
            )
            config['max_tokens'] = max_tokens
        
        st.session_state.llm_config['azure'] = config
    
    def render_local_config(self):
        """渲染本地模型配置"""
        st.subheader("🟢 本地模型配置")
        
        config = st.session_state.llm_config.get('local', {})
        
        st.info("💡 支持 Ollama, LM Studio 等本地模型服务")
        
        base_url = st.text_input(
            "Base URL",
            value=config.get('base_url', 'http://localhost:11434'),
            help="本地模型服务地址"
        )
        config['base_url'] = base_url
        
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.text_input(
                "模型名称",
                value=config.get('model', 'llama2'),
                help="例如: llama2, mistral, codellama"
            )
            config['model'] = model
        
        with col2:
            temperature = st.slider(
                "Temperature",
                0.0, 2.0,
                float(config.get('temperature', 0.7)),
                0.1
            )
            config['temperature'] = temperature
        
        max_tokens = st.number_input(
            "Max Tokens",
            100, 32000,
            int(config.get('max_tokens', 2048)),
            100
        )
        config['max_tokens'] = max_tokens
        
        # Ollama安装指南
        with st.expander("📖 Ollama 安装指南"):
            st.markdown("""
            ### 安装 Ollama
            
            **Windows:**
            ```bash
            # 下载安装包
            https://ollama.ai/download/windows
            ```
            
            **Linux/Mac:**
            ```bash
            curl -fsSL https://ollama.ai/install.sh | sh
            ```
            
            ### 下载模型
            ```bash
            # Llama 2
            ollama pull llama2
            
            # Mistral
            ollama pull mistral
            
            # Code Llama
            ollama pull codellama
            ```
            
            ### 启动服务
            ```bash
            ollama serve
            ```
            """)
        
        st.session_state.llm_config['local'] = config
    
    def render_test_connection(self):
        """渲染连接测试"""
        st.subheader("🔌 连接测试")
        
        provider = st.session_state.llm_config.get('provider')
        
        if st.button("🧪 测试连接", type="primary"):
            with st.spinner("测试中..."):
                config_key = provider.lower()
                config = st.session_state.llm_config.get(config_key, {})
                
                result = self.test_connection(provider, config)
                st.session_state.llm_test_result = result
        
        if st.session_state.llm_test_result:
            result = st.session_state.llm_test_result
            
            if result['success']:
                st.success(f"✅ {result['message']}")
                
                if 'model' in result:
                    st.info(f"📋 使用模型: {result['model']}")
                if 'tokens_used' in result:
                    st.info(f"🎫 测试消耗token: {result['tokens_used']}")
                if 'available_models' in result:
                    st.info(f"📦 可用模型: {', '.join(result['available_models'][:5])}")
            else:
                st.error(f"❌ {result['message']}")
    
    def render_usage_stats(self):
        """渲染使用统计"""
        st.subheader("📊 使用统计")
        
        stats = st.session_state.llm_config.get('usage_stats', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总请求数", f"{stats.get('total_requests', 0):,}")
        
        with col2:
            st.metric("总Token数", f"{stats.get('total_tokens', 0):,}")
        
        with col3:
            last_used = stats.get('last_used')
            if last_used:
                st.metric("最后使用", last_used)
            else:
                st.metric("最后使用", "未使用")
        
        st.info("💡 使用统计将在实际调用LLM时自动更新")
    
    def render_save_load(self):
        """渲染保存/加载"""
        st.subheader("💾 配置管理")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 保存配置", type="primary", use_container_width=True):
                if self.save_config(st.session_state.llm_config):
                    st.success("✅ 配置已保存!")
                    st.balloons()
        
        with col2:
            if st.button("🔄 重新加载", use_container_width=True):
                st.session_state.llm_config = self.load_config()
                st.success("✅ 配置已重新加载!")
                st.rerun()
        
        with col3:
            if st.button("🗑️ 重置为默认", use_container_width=True):
                if st.button("确认重置?", key="confirm_reset"):
                    st.session_state.llm_config = self.load_config()
                    st.success("✅ 已重置为默认配置!")
                    st.rerun()
        
        st.caption(f"📁 配置文件路径: `{self.CONFIG_FILE.absolute()}`")
    
    def render(self):
        """主渲染函数"""
        st.title("⚙️ LLM配置管理")
        
        st.markdown("""
        配置大语言模型连接参数,支持多种提供商。
        所有配置将安全保存在本地。
        """)
        
        st.divider()
        
        # 提供商选择
        self.render_provider_selection()
        
        st.divider()
        
        # 根据选择的提供商显示配置
        provider = st.session_state.llm_config.get('provider')
        
        if provider == 'OpenAI':
            self.render_openai_config()
        elif provider == 'Claude':
            self.render_claude_config()
        elif provider == 'Azure':
            self.render_azure_config()
        elif provider == 'Local':
            self.render_local_config()
        else:
            st.info("该提供商配置界面开发中...")
        
        st.divider()
        
        # 连接测试
        self.render_test_connection()
        
        st.divider()
        
        # 使用统计
        self.render_usage_stats()
        
        st.divider()
        
        # 保存/加载
        self.render_save_load()


def main():
    """主函数"""
    manager = LLMConfigManager()
    manager.render()


if __name__ == "__main__":
    main()
