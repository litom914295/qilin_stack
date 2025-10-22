"""
TradingAgents集成配置管理
支持环境变量和配置文件
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingAgentsConfig:
    """TradingAgents配置类"""
    
    # 路径配置
    tradingagents_path: str = field(
        default_factory=lambda: os.getenv(
            "TRADINGAGENTS_PATH",
            "D:/test/Qlib/tradingagents"
        )
    )
    
    # LLM配置
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4-turbo")
    )
    
    # 模式控制
    force_official: bool = field(
        default_factory=lambda: os.getenv("FORCE_TA_OFFICIAL", "false").lower() in ("1","true","yes")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    llm_api_base: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE")
    )
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # 智能体配置（基础4 + 一进二专用6 共10个）
    enable_market_analyst: bool = True
    enable_fundamental_analyst: bool = True
    enable_technical_analyst: bool = True
    enable_sentiment_analyst: bool = True
    enable_news_analyst: bool = True
    # 一进二专用
    enable_limitup_validator: bool = True
    enable_seal_quality: bool = True
    enable_volume_surge: bool = True
    enable_board_continuity: bool = True
    enable_qlib_momentum: bool = True
    enable_rd_composite: bool = True
    
    # 工具配置
    news_api_key: str = field(
        default_factory=lambda: os.getenv("NEWS_API_KEY", "")
    )
    alpha_vantage_key: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", "")
    )
    
    # 共识机制配置
    consensus_method: str = "weighted_vote"  # weighted_vote, simple_vote, confidence_based
    agent_weights: Dict[str, float] = field(default_factory=lambda: {
        # 基础4
        "market_analyst": 0.12,
        "fundamental_analyst": 0.10,
        "technical_analyst": 0.10,
        "sentiment_analyst": 0.08,
        # 一进二专用6（合计权重更高）
        "limitup_validator": 0.15,
        "seal_quality": 0.12,
        "volume_surge": 0.10,
        "board_continuity": 0.08,
        "qlib_momentum": 0.08,
        "rd_composite": 0.07,
    })
    
    # 一进二硬性门槛（可调）
    min_seal_quality: float = 6.0
    min_volume_surge: float = 1.8
    max_limit_up_minutes: int = 60
    max_open_count: int = 3
    min_price: float = 3.0
    max_price: float = 50.0
    
    # 性能配置
    timeout: int = 30  # 秒
    max_retries: int = 3
    enable_cache: bool = True
    cache_ttl: int = 300  # 秒
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """后初始化验证"""
        # 验证路径
        if not Path(self.tradingagents_path).exists():
            logger.warning(f"TradingAgents路径不存在: {self.tradingagents_path}")
        
        # 验证API密钥
        if not self.llm_api_key:
            logger.warning("LLM API密钥未配置，部分功能将不可用")
    
    @classmethod
    def from_yaml(cls, config_file: str) -> 'TradingAgentsConfig':
        """从YAML文件加载配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_file}, 使用默认配置")
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict.get('tradingagents', {}))
    
    @classmethod
    def from_json(cls, config_file: str) -> 'TradingAgentsConfig':
        """从JSON文件加载配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_file}, 使用默认配置")
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict.get('tradingagents', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'tradingagents_path': self.tradingagents_path,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'llm_max_tokens': self.llm_max_tokens,
            'enable_market_analyst': self.enable_market_analyst,
            'enable_fundamental_analyst': self.enable_fundamental_analyst,
            'enable_technical_analyst': self.enable_technical_analyst,
            'enable_sentiment_analyst': self.enable_sentiment_analyst,
            'enable_news_analyst': self.enable_news_analyst,
            'consensus_method': self.consensus_method,
'agent_weights': self.agent_weights,
            'force_official': self.force_official,
            'min_seal_quality': self.min_seal_quality,
            'min_volume_surge': self.min_volume_surge,
            'max_limit_up_minutes': self.max_limit_up_minutes,
            'max_open_count': self.max_open_count,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'enable_cache': self.enable_cache,
            'cache_ttl': self.cache_ttl,
            'log_level': self.log_level,
        }
    
    def validate(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 验证路径
        if not Path(self.tradingagents_path).exists():
            errors.append(f"TradingAgents路径不存在: {self.tradingagents_path}")
        
        # 验证LLM配置
        if self.llm_provider not in ['openai', 'azure', 'anthropic', 'local']:
            errors.append(f"不支持的LLM提供商: {self.llm_provider}")
        
        if not self.llm_api_key and self.llm_provider != 'local':
            errors.append(f"LLM API密钥未配置")
        
        # 验证权重和
        weight_sum = sum(self.agent_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"智能体权重之和应为1.0，当前为{weight_sum}")
        
        if errors:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True
    
    def get_enabled_agents(self) -> list[str]:
        """获取启用的智能体列表"""
        agents = []
        if self.enable_market_analyst:
            agents.append("market_analyst")
        if self.enable_fundamental_analyst:
            agents.append("fundamental_analyst")
        if self.enable_technical_analyst:
            agents.append("technical_analyst")
        if self.enable_sentiment_analyst:
            agents.append("sentiment_analyst")
        if self.enable_news_analyst:
            agents.append("news_analyst")
        if self.enable_limitup_validator:
            agents.append("limitup_validator")
        if self.enable_seal_quality:
            agents.append("seal_quality")
        if self.enable_volume_surge:
            agents.append("volume_surge")
        if self.enable_board_continuity:
            agents.append("board_continuity")
        if self.enable_qlib_momentum:
            agents.append("qlib_momentum")
        if self.enable_rd_composite:
            agents.append("rd_composite")
        return agents


def load_config(config_file: Optional[str] = None) -> TradingAgentsConfig:
    """
    加载配置
    
    优先级：
    1. 指定的配置文件
    2. 环境变量
    3. 默认配置
    """
    if config_file:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return TradingAgentsConfig.from_yaml(config_file)
        elif config_file.endswith('.json'):
            return TradingAgentsConfig.from_json(config_file)
    
    # 使用默认配置（会从环境变量读取）
    return TradingAgentsConfig()


# 示例配置模板
CONFIG_TEMPLATE = """
# TradingAgents集成配置

tradingagents:
  # TradingAgents项目路径
  tradingagents_path: "D:/test/Qlib/tradingagents"
  
  # LLM配置
  llm_provider: "openai"  # openai, azure, anthropic, local
  llm_model: "gpt-4-turbo"
  llm_temperature: 0.7
  llm_max_tokens: 2000
  
  # 智能体启用配置
  enable_market_analyst: true
  enable_fundamental_analyst: true
  enable_technical_analyst: true
  enable_sentiment_analyst: true
  enable_news_analyst: true
  
  # 共识机制
  consensus_method: "weighted_vote"
  agent_weights:
    market_analyst: 0.25
    fundamental_analyst: 0.25
    technical_analyst: 0.20
    sentiment_analyst: 0.15
    news_analyst: 0.15
  
  # 性能配置
  timeout: 30
  max_retries: 3
  enable_cache: true
  cache_ttl: 300
  
  # 日志配置
  log_level: "INFO"
  log_file: null
"""


if __name__ == "__main__":
    # 测试配置
    config = load_config()
    print("=== TradingAgents配置 ===")
    print(f"TradingAgents路径: {config.tradingagents_path}")
    print(f"LLM提供商: {config.llm_provider}")
    print(f"LLM模型: {config.llm_model}")
    print(f"启用的智能体: {config.get_enabled_agents()}")
    print(f"配置有效性: {config.validate()}")
    
    # 保存示例配置
    example_config_path = Path(__file__).parent / "config.example.yaml"
    with open(example_config_path, 'w', encoding='utf-8') as f:
        f.write(CONFIG_TEMPLATE)
    print(f"\n示例配置已保存到: {example_config_path}")
