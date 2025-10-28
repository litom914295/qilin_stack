"""
环境配置管理器 (Environment Configuration Manager)
统一管理所有环境变量和配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum


class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class EnvironmentConfig:
    """环境配置类"""
    
    # ============================================================================
    # 基础配置
    # ============================================================================
    env: str = os.getenv('QILIN_ENV', 'development')
    debug: bool = os.getenv('QILIN_DEBUG', 'True').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # ============================================================================
    # 数据路径配置
    # ============================================================================
    data_path: str = os.getenv('QILIN_DATA_PATH', './data')
    qlib_data_path: str = os.getenv('QLIB_DATA_PATH', './qlib_data')
    model_path: str = os.getenv('MODEL_PATH', './models')
    log_path: str = os.getenv('LOG_PATH', './logs')
    cache_path: str = os.getenv('CACHE_PATH', './cache')
    
    # ============================================================================
    # API配置
    # ============================================================================
    # RD-Agent API
    rdagent_api_url: str = os.getenv('RDAGENT_API_URL', 'http://localhost:9000')
    rdagent_enabled: bool = os.getenv('RDAGENT_ENABLED', 'True').lower() == 'true'
    
    # Qlib API
    qlib_api_url: str = os.getenv('QLIB_API_URL', 'http://localhost:5000')
    qlib_serving_url: str = os.getenv('QLIB_SERVING_URL', 'http://localhost:9710')
    
    # Trading Agents API
    tradingagents_api_url: str = os.getenv('TRADINGAGENTS_API_URL', 'http://localhost:8000')
    
    # 数据流API
    stream_api_url: str = os.getenv('STREAM_API_URL', 'ws://localhost:8080/stream')
    
    # ============================================================================
    # 数据库配置
    # ============================================================================
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///./qilin.db')
    database_type: str = os.getenv('DATABASE_TYPE', 'sqlite')  # sqlite, postgresql
    
    # PostgreSQL配置
    postgres_host: str = os.getenv('POSTGRES_HOST', 'localhost')
    postgres_port: int = int(os.getenv('POSTGRES_PORT', '5432'))
    postgres_user: str = os.getenv('POSTGRES_USER', 'qilin')
    postgres_password: str = os.getenv('POSTGRES_PASSWORD', '')
    postgres_database: str = os.getenv('POSTGRES_DB', 'qilin_stack')
    
    # ============================================================================
    # Redis配置
    # ============================================================================
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    redis_password: Optional[str] = os.getenv('REDIS_PASSWORD')
    redis_db: int = int(os.getenv('REDIS_DB', '0'))
    redis_enabled: bool = os.getenv('REDIS_ENABLED', 'True').lower() == 'true'
    
    # ============================================================================
    # 券商交易配置
    # ============================================================================
    broker_type: str = os.getenv('BROKER_TYPE', 'simulated')  # simulated, real
    broker_api_url: str = os.getenv('BROKER_API_URL', 'http://localhost:8000')
    broker_api_key: Optional[str] = os.getenv('BROKER_API_KEY')
    broker_api_secret: Optional[str] = os.getenv('BROKER_API_SECRET')
    broker_name: str = os.getenv('BROKER_NAME', 'Simulated Broker')
    
    # 交易参数
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '1000000'))
    commission_rate: float = float(os.getenv('COMMISSION_RATE', '0.0003'))
    min_commission: float = float(os.getenv('MIN_COMMISSION', '5'))
    slippage: float = float(os.getenv('SLIPPAGE', '0.001'))
    
    # ============================================================================
    # LLM配置
    # ============================================================================
    llm_provider: str = os.getenv('LLM_PROVIDER', 'openai')  # openai, anthropic, azure
    llm_model: str = os.getenv('LLM_MODEL', 'gpt-4')
    llm_api_key: Optional[str] = os.getenv('LLM_API_KEY')
    llm_api_base: Optional[str] = os.getenv('LLM_API_BASE')
    llm_temperature: float = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    llm_max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', '2000'))
    
    # ============================================================================
    # MLflow配置
    # ============================================================================
    mlflow_tracking_uri: str = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
    mlflow_enabled: bool = os.getenv('MLFLOW_ENABLED', 'False').lower() == 'true'
    
    # ============================================================================
    # 监控配置
    # ============================================================================
    prometheus_enabled: bool = os.getenv('PROMETHEUS_ENABLED', 'False').lower() == 'true'
    prometheus_port: int = int(os.getenv('PROMETHEUS_PORT', '9090'))
    
    grafana_enabled: bool = os.getenv('GRAFANA_ENABLED', 'False').lower() == 'true'
    grafana_url: str = os.getenv('GRAFANA_URL', 'http://localhost:3000')
    
    # ============================================================================
    # 安全配置
    # ============================================================================
    secret_key: str = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    jwt_secret: str = os.getenv('JWT_SECRET', 'your-jwt-secret-change-in-production')
    jwt_expire_minutes: int = int(os.getenv('JWT_EXPIRE_MINUTES', '1440'))
    
    # ============================================================================
    # Web UI配置
    # ============================================================================
    web_host: str = os.getenv('WEB_HOST', '0.0.0.0')
    web_port: int = int(os.getenv('WEB_PORT', '8501'))
    web_debug: bool = os.getenv('WEB_DEBUG', 'True').lower() == 'true'
    
    # ============================================================================
    # 数据源配置
    # ============================================================================
    # AKShare
    akshare_enabled: bool = os.getenv('AKSHARE_ENABLED', 'True').lower() == 'true'
    
    # TuShare
    tushare_enabled: bool = os.getenv('TUSHARE_ENABLED', 'False').lower() == 'true'
    tushare_token: Optional[str] = os.getenv('TUSHARE_TOKEN')
    
    # Wind
    wind_enabled: bool = os.getenv('WIND_ENABLED', 'False').lower() == 'true'
    wind_username: Optional[str] = os.getenv('WIND_USERNAME')
    wind_password: Optional[str] = os.getenv('WIND_PASSWORD')
    
    # ============================================================================
    # 性能配置
    # ============================================================================
    max_workers: int = int(os.getenv('MAX_WORKERS', '4'))
    batch_size: int = int(os.getenv('BATCH_SIZE', '100'))
    cache_ttl: int = int(os.getenv('CACHE_TTL', '3600'))  # 秒
    
    # ============================================================================
    # 方法
    # ============================================================================
    @classmethod
    def from_env(cls, env: str = None):
        """
        从环境加载配置
        
        Args:
            env: 环境类型 (development, testing, production)
        """
        if env:
            os.environ['QILIN_ENV'] = env
        
        # 加载.env文件
        env_file = f'.env.{os.getenv("QILIN_ENV", "development")}'
        if os.path.exists(env_file):
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"✅ 已加载配置文件: {env_file}")
        else:
            print(f"⚠️  配置文件不存在: {env_file}，使用默认配置")
        
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)
    
    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        if self.database_type == 'postgresql':
            return (f"postgresql://{self.postgres_user}:{self.postgres_password}"
                   f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}")
        else:
            return self.database_url
    
    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        验证配置
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查必需的API密钥
        if self.broker_type == 'real':
            if not self.broker_api_key:
                errors.append("真实交易模式需要配置 BROKER_API_KEY")
            if not self.broker_api_secret:
                errors.append("真实交易模式需要配置 BROKER_API_SECRET")
        
        if self.llm_provider in ['openai', 'anthropic'] and not self.llm_api_key:
            errors.append(f"{self.llm_provider} 需要配置 LLM_API_KEY")
        
        # 检查路径
        paths_to_check = [
            ('data_path', self.data_path),
            ('model_path', self.model_path),
            ('log_path', self.log_path),
        ]
        
        for name, path in paths_to_check:
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                    print(f"✅ 创建目录: {path}")
                except Exception as e:
                    errors.append(f"无法创建目录 {name}: {path} - {e}")
        
        return len(errors) == 0, errors
    
    def print_config(self, show_secrets: bool = False):
        """
        打印配置信息
        
        Args:
            show_secrets: 是否显示敏感信息
        """
        print("\n" + "="*60)
        print("📋 Qilin Stack 配置信息")
        print("="*60)
        
        print(f"\n🌍 环境: {self.env}")
        print(f"🐛 调试模式: {self.debug}")
        print(f"📝 日志级别: {self.log_level}")
        
        print(f"\n📁 路径配置:")
        print(f"  数据路径: {self.data_path}")
        print(f"  模型路径: {self.model_path}")
        print(f"  日志路径: {self.log_path}")
        
        print(f"\n🔌 API配置:")
        print(f"  RD-Agent: {self.rdagent_api_url} ({'启用' if self.rdagent_enabled else '禁用'})")
        print(f"  Qlib: {self.qlib_api_url}")
        print(f"  Trading Agents: {self.tradingagents_api_url}")
        
        print(f"\n💾 数据库:")
        print(f"  类型: {self.database_type}")
        if not show_secrets:
            print(f"  URL: {self.get_database_url().split('@')[0]}@***")
        else:
            print(f"  URL: {self.get_database_url()}")
        
        print(f"\n💰 交易配置:")
        print(f"  券商类型: {self.broker_type}")
        print(f"  初始资金: {self.initial_capital:,.0f}")
        print(f"  手续费率: {self.commission_rate:.4f}")
        
        print(f"\n🤖 LLM配置:")
        print(f"  提供商: {self.llm_provider}")
        print(f"  模型: {self.llm_model}")
        if self.llm_api_key and not show_secrets:
            print(f"  API Key: {self.llm_api_key[:8]}***")
        
        print("\n" + "="*60 + "\n")


# 全局配置实例
_config_instance: Optional[EnvironmentConfig] = None


def get_config() -> EnvironmentConfig:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = EnvironmentConfig.from_env()
    return _config_instance


def reload_config(env: str = None):
    """重新加载配置"""
    global _config_instance
    _config_instance = EnvironmentConfig.from_env(env)
    return _config_instance


if __name__ == "__main__":
    # 测试配置
    config = EnvironmentConfig.from_env()
    config.print_config(show_secrets=False)
    
    # 验证配置
    is_valid, errors = config.validate()
    if is_valid:
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
