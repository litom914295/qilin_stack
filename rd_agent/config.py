"""
RD-Agent集成配置管理
支持环境变量和配置文件
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RDAgentConfig:
    """RD-Agent配置类"""
    
    # 路径配置
    rdagent_path: str = field(
        default_factory=lambda: os.getenv(
            "RDAGENT_PATH",
            "D:/test/Qlib/RD-Agent"
        )
    )
    
    # LLM配置
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4-turbo")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    llm_api_base: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE")
    )
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4000
    
    # 研究配置
    research_mode: str = "factor"  # factor, model, strategy
    max_iterations: int = 10
    parallel_tasks: int = 3
    enable_cache: bool = True
    cache_dir: str = "./workspace/rdagent_cache"
    
    # 因子研究配置
    factor_pool_size: int = 20
    factor_selection_top_k: int = 5
    factor_ic_threshold: float = 0.03
    factor_ir_threshold: float = 0.5
    
    # 模型研究配置
    model_types: List[str] = field(default_factory=lambda: [
        "lightgbm", "xgboost", "catboost", "tabnet"
    ])
    model_selection_metric: str = "sharpe_ratio"
    model_cv_folds: int = 5
    
    # 优化配置
    optim_method: str = "optuna"  # optuna, grid, random
    optim_trials: int = 100
    optim_timeout: int = 3600  # 秒
    
    # 数据配置
    data_path: Optional[str] = None
    data_provider: str = "qlib"  # qlib, custom
    train_start: str = "2020-01-01"
    train_end: str = "2022-12-31"
    valid_start: str = "2023-01-01"
    valid_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str = "2023-12-31"
    
    # 报告配置
    enable_report: bool = True
    report_format: str = "markdown"  # markdown, html, pdf
    report_dir: str = "./reports/rdagent"
    
    # 性能配置
    timeout: int = 300  # 单次任务超时（秒）
    max_retries: int = 3
    enable_parallel: bool = True
    num_workers: int = 4
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    verbose: bool = True
    
    def __post_init__(self):
        """后初始化验证"""
        # 验证路径
        if not Path(self.rdagent_path).exists():
            logger.warning(f"RD-Agent路径不存在: {self.rdagent_path}")
        
        # 验证API密钥
        if not self.llm_api_key:
            logger.warning("LLM API密钥未配置，部分功能将不可用")
        
        # 创建必要目录
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_file: str) -> 'RDAgentConfig':
        """从YAML文件加载配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_file}, 使用默认配置")
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 只提取dataclass字段
        rdagent_config = config_dict.get('rdagent', {})
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in rdagent_config.items() if k in valid_fields}
        
        # 保存额外字段
        instance = cls(**filtered_config)
        for key, value in rdagent_config.items():
            if key not in valid_fields:
                setattr(instance, key, value)
        
        return instance
    
    @classmethod
    def from_json(cls, config_file: str) -> 'RDAgentConfig':
        """从JSON文件加载配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_file}, 使用默认配置")
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict.get('rdagent', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rdagent_path': self.rdagent_path,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'llm_max_tokens': self.llm_max_tokens,
            'research_mode': self.research_mode,
            'max_iterations': self.max_iterations,
            'parallel_tasks': self.parallel_tasks,
            'enable_cache': self.enable_cache,
            'factor_pool_size': self.factor_pool_size,
            'factor_selection_top_k': self.factor_selection_top_k,
            'model_types': self.model_types,
            'model_selection_metric': self.model_selection_metric,
            'optim_method': self.optim_method,
            'optim_trials': self.optim_trials,
            'enable_report': self.enable_report,
            'report_format': self.report_format,
            'enable_parallel': self.enable_parallel,
            'num_workers': self.num_workers,
            'log_level': self.log_level,
            'verbose': self.verbose,
        }
    
    def validate(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 验证路径
        if not Path(self.rdagent_path).exists():
            errors.append(f"RD-Agent路径不存在: {self.rdagent_path}")
        
        # 验证LLM配置
        if self.llm_provider not in ['openai', 'azure', 'anthropic', 'local']:
            errors.append(f"不支持的LLM提供商: {self.llm_provider}")
        
        if not self.llm_api_key and self.llm_provider != 'local':
            errors.append("LLM API密钥未配置")
        
        # 验证研究模式
        if self.research_mode not in ['factor', 'model', 'strategy']:
            errors.append(f"不支持的研究模式: {self.research_mode}")
        
        # 验证迭代次数
        if self.max_iterations < 1 or self.max_iterations > 100:
            errors.append(f"迭代次数应在1-100之间: {self.max_iterations}")
        
        # 验证阈值
        if self.factor_ic_threshold < 0 or self.factor_ic_threshold > 1:
            errors.append(f"IC阈值应在0-1之间: {self.factor_ic_threshold}")
        
        if errors:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True
    
    def get_qlib_config(self) -> Dict[str, Any]:
        """获取Qlib相关配置"""
        return {
            "data_provider": self.data_provider,
            "train_period": (self.train_start, self.train_end),
            "valid_period": (self.valid_start, self.valid_end),
            "test_period": (self.test_start, self.test_end),
        }


def load_config(config_file: Optional[str] = None) -> RDAgentConfig:
    """
    加载配置
    
    优先级：
    1. 指定的配置文件
    2. 环境变量
    3. 默认配置
    """
    if config_file:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return RDAgentConfig.from_yaml(config_file)
        elif config_file.endswith('.json'):
            return RDAgentConfig.from_json(config_file)
    
    # 使用默认配置（会从环境变量读取）
    return RDAgentConfig()


# 示例配置模板
CONFIG_TEMPLATE = """
# RD-Agent集成配置

rdagent:
  # RD-Agent项目路径
  rdagent_path: "D:/test/Qlib/RD-Agent"
  
  # LLM配置
  llm_provider: "openai"  # openai, azure, anthropic, local
  llm_model: "gpt-4-turbo"
  llm_temperature: 0.7
  llm_max_tokens: 4000
  
  # 研究配置
  research_mode: "factor"  # factor, model, strategy
  max_iterations: 10
  parallel_tasks: 3
  enable_cache: true
  cache_dir: "./workspace/rdagent_cache"
  
  # 因子研究配置
  factor_pool_size: 20
  factor_selection_top_k: 5
  factor_ic_threshold: 0.03
  factor_ir_threshold: 0.5
  
  # 模型研究配置
  model_types:
    - "lightgbm"
    - "xgboost"
    - "catboost"
  model_selection_metric: "sharpe_ratio"
  model_cv_folds: 5
  
  # 优化配置
  optim_method: "optuna"  # optuna, grid, random
  optim_trials: 100
  optim_timeout: 3600
  
  # 数据配置
  data_provider: "qlib"
  train_start: "2020-01-01"
  train_end: "2022-12-31"
  valid_start: "2023-01-01"
  valid_end: "2023-06-30"
  test_start: "2023-07-01"
  test_end: "2023-12-31"
  
  # 报告配置
  enable_report: true
  report_format: "markdown"
  report_dir: "./reports/rdagent"
  
  # 性能配置
  timeout: 300
  max_retries: 3
  enable_parallel: true
  num_workers: 4
  
  # 日志配置
  log_level: "INFO"
  verbose: true
"""


if __name__ == "__main__":
    # 测试配置
    config = load_config()
    print("=== RD-Agent配置 ===")
    print(f"RD-Agent路径: {config.rdagent_path}")
    print(f"LLM提供商: {config.llm_provider}")
    print(f"LLM模型: {config.llm_model}")
    print(f"研究模式: {config.research_mode}")
    print(f"最大迭代: {config.max_iterations}")
    print(f"配置有效性: {config.validate()}")
    
    # 保存示例配置
    example_config_path = Path(__file__).parent / "config.example.yaml"
    with open(example_config_path, 'w', encoding='utf-8') as f:
        f.write(CONFIG_TEMPLATE)
    print(f"\n示例配置已保存到: {example_config_path}")
