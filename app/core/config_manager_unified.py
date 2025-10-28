"""
统一配置管理系统
管理三个项目的配置参数
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        # Qlib配置
        'qlib': {
            'path': 'G:\\test\\qlib',
            'data_path': '~/.qlib/qlib_data/cn_data',
            'region': 'cn',
            'enabled': True,
        },
        
        # RD-Agent配置
        'rdagent': {
            'path': 'G:\\test\\RD-Agent',
            'workspace': './rdagent_workspace',
            'llm_provider': 'openai',  # openai, deepseek, azure等
            'llm_model': 'gpt-4',
            'api_key': '',  # 从环境变量读取
            'enabled': True,
        },
        
        # TradingAgents配置
        'tradingagents': {
            'path': 'G:\\test\\tradingagents-cn-plus',
            'llm_provider': 'openai',
            'api_key': '',
            'enabled': True,
        },
        
        # 数据共享配置
        'data_bridge': {
            'cache_dir': './shared_cache',
            'enable_auto_sync': True,
        },
        
        # Web界面配置
        'web': {
            'host': 'localhost',
            'port': 8501,
            'theme': 'light',  # light, dark
            'page_title': '麒麟量化统一平台',
        },
        
        # 日志配置
        'logging': {
            'level': 'INFO',
            'file': './logs/qilin_stack.log',
            'max_size': '10MB',
            'backup_count': 5,
        },
        
        # 性能配置
        'performance': {
            'enable_monitoring': True,
            'cache_enabled': True,
            'max_workers': 4,
        },
        
        # 导出配置
        'export': {
            'default_format': 'excel',  # excel, csv, json
            'output_dir': './exports',
        },
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file or './config.yaml'
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            config_path = Path(self.config_file)
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix in ['.yaml', '.yml']:
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # 合并配置
                self._merge_config(self.config, user_config)
                logger.info(f"配置文件加载成功: {config_path}")
            else:
                logger.warning(f"配置文件不存在，使用默认配置: {config_path}")
                self.save_config()  # 保存默认配置
            
            # 从环境变量读取敏感信息
            self._load_from_env()
            
            return True
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return False
    
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
                else:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置保存成功: {config_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def _merge_config(self, base: Dict, update: Dict):
        """递归合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_from_env(self):
        """从环境变量加载敏感配置"""
        # LLM API Keys
        if 'OPENAI_API_KEY' in os.environ:
            self.config['rdagent']['api_key'] = os.environ['OPENAI_API_KEY']
            self.config['tradingagents']['api_key'] = os.environ['OPENAI_API_KEY']
        
        if 'DEEPSEEK_API_KEY' in os.environ:
            if self.config['rdagent']['llm_provider'] == 'deepseek':
                self.config['rdagent']['api_key'] = os.environ['DEEPSEEK_API_KEY']
        
        # 其他环境变量
        env_mappings = {
            'QLIB_DATA_PATH': ['qlib', 'data_path'],
            'RDAGENT_WORKSPACE': ['rdagent', 'workspace'],
            'WEB_PORT': ['web', 'port'],
        }
        
        for env_key, config_path in env_mappings.items():
            if env_key in os.environ:
                self._set_nested_value(self.config, config_path, os.environ[env_key])
    
    def _set_nested_value(self, d: Dict, path: list, value: Any):
        """设置嵌套字典的值"""
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = value
    
    def get(self, *keys, default=None) -> Any:
        """
        获取配置值
        
        Args:
            *keys: 配置路径，如 'qlib', 'data_path'
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, *keys, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            *keys: 配置路径
            value: 配置值
            
        Returns:
            是否成功
        """
        try:
            d = self.config
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value
            return True
        except Exception as e:
            logger.error(f"设置配置失败: {e}")
            return False
    
    def get_qlib_config(self) -> Dict[str, Any]:
        """获取Qlib配置"""
        return self.config.get('qlib', {})
    
    def get_rdagent_config(self) -> Dict[str, Any]:
        """获取RD-Agent配置"""
        return self.config.get('rdagent', {})
    
    def get_tradingagents_config(self) -> Dict[str, Any]:
        """获取TradingAgents配置"""
        return self.config.get('tradingagents', {})
    
    def is_module_enabled(self, module: str) -> bool:
        """
        检查模块是否启用
        
        Args:
            module: 模块名 (qlib, rdagent, tradingagents)
            
        Returns:
            是否启用
        """
        return self.get(module, 'enabled', default=True)
    
    def validate_config(self) -> tuple[bool, list]:
        """
        验证配置
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查必需的路径
        required_paths = [
            ('qlib', 'path'),
            ('rdagent', 'path'),
            ('tradingagents', 'path'),
        ]
        
        for module, key in required_paths:
            path = self.get(module, key)
            if not path or not Path(path).exists():
                errors.append(f"{module}.{key} 路径不存在: {path}")
        
        # 检查API密钥
        if self.is_module_enabled('rdagent'):
            if not self.get('rdagent', 'api_key'):
                errors.append("rdagent.api_key 未配置")
        
        if self.is_module_enabled('tradingagents'):
            if not self.get('tradingagents', 'api_key'):
                errors.append("tradingagents.api_key 未配置")
        
        return len(errors) == 0, errors
    
    def export_config(self, output_file: str) -> bool:
        """导出配置到文件"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 移除敏感信息
            safe_config = self._remove_sensitive(self.config)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.suffix in ['.yaml', '.yml']:
                    yaml.dump(safe_config, f, allow_unicode=True)
                else:
                    json.dump(safe_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置导出成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"配置导出失败: {e}")
            return False
    
    def _remove_sensitive(self, config: Dict) -> Dict:
        """移除敏感信息"""
        safe_config = config.copy()
        
        # 移除API密钥
        if 'rdagent' in safe_config:
            safe_config['rdagent']['api_key'] = '***'
        if 'tradingagents' in safe_config:
            safe_config['tradingagents']['api_key'] = '***'
        
        return safe_config


# 全局配置管理器实例
config_manager = ConfigManager()
