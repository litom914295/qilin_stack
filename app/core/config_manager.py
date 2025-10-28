"""
统一配置管理模块
使用Pydantic进行配置验证和管理
支持环境变量、YAML文件和默认值的多层次配置
"""

from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum
import os
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class MarketType(str, Enum):
    """市场类型"""
    CN = "cn"  # 中国A股
    US = "us"  # 美股
    HK = "hk"  # 港股


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BacktestConfig(BaseModel):
    """回测配置"""
    model_config = ConfigDict(extra='allow', use_enum_values=True)  # 允许额外字段
    
    initial_capital: float = Field(1000000, ge=10000, description="初始资金")
    commission_rate: float = Field(0.0003, ge=0, le=0.01, description="手续费率")
    slippage_rate: float = Field(0.0001, ge=0, le=0.01, description="滑点率")
    min_commission: float = Field(5.0, ge=0, description="最低手续费")
    
    # T+1规则
    enable_t_plus_1: bool = Field(True, description="启用T+1交易规则")
    
    # 涨停板规则
    enable_limit_up_restriction: bool = Field(True, description="启用涨停板限制")
    one_word_block_strict: bool = Field(True, description="一字板严格模式(完全无法成交)")


class RiskConfig(BaseModel):
    """风险管理配置"""
    model_config = ConfigDict(extra='allow')  # 允许额外字段
    
    max_position_ratio: float = Field(0.3, ge=0, le=1, description="单票最大仓位比例")
    max_total_position_ratio: float = Field(0.95, ge=0, le=1, description="总仓位上限")
    stop_loss_ratio: float = Field(0.05, ge=0, le=0.5, description="止损比例")
    max_drawdown: float = Field(0.20, ge=0, le=1, description="最大回撤限制")
    
    @model_validator(mode='after')
    def validate_position_ratio(self):
        """验证单票仓位不能超过总仓位"""
        if self.max_position_ratio > self.max_total_position_ratio:
            raise ValueError(
                f"单票仓位{self.max_position_ratio}不能超过总仓位限制{self.max_total_position_ratio}"
            )
        return self


class DataConfig(BaseModel):
    """数据配置"""
    model_config = ConfigDict(extra='allow')  # 允许额外字段
    
    qlib_data_path: str = Field("~/.qlib/qlib_data/cn_data", description="Qlib数据路径")
    cache_dir: str = Field("./cache", description="缓存目录")
    enable_cache: bool = Field(True, description="启用数据缓存")
    cache_expire_days: int = Field(7, ge=1, le=365, description="缓存过期天数")
    
    @field_validator('qlib_data_path', 'cache_dir')
    @classmethod
    def expand_path(cls, v):
        """展开路径中的~和环境变量"""
        return str(Path(v).expanduser().resolve())


class StrategyConfig(BaseModel):
    """策略配置"""
    model_config = ConfigDict(extra='allow')  # 允许额外字段
    
    name: str = Field("qilin_limitup", description="策略名称")
    topk: int = Field(5, ge=1, le=20, description="选股数量")
    min_score: float = Field(0.6, ge=0, le=1, description="最低评分阈值")
    min_confidence: float = Field(0.7, ge=0, le=1, description="最低置信度阈值")
    
    # 一进二专项参数
    min_seal_amount: float = Field(10_000_000, ge=0, description="最小封单金额(元)")
    max_continuous_boards: int = Field(5, ge=1, le=20, description="最大连板天数")


class AgentConfig(BaseModel):
    """Agent配置"""
    model_config = ConfigDict(extra='allow')  # 允许额外字段
    
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "market_tech": 0.3,
            "fundamental": 0.2,
            "sentiment": 0.25,
            "risk": 0.25
        },
        description="各Agent权重"
    )
    
    max_runtime_sec: int = Field(60, ge=10, le=600, description="最大运行时间(秒)")
    enable_parallel: bool = Field(False, description="启用并行执行")
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """验证权重和为1"""
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):  # 允许5%误差
            raise ValueError(f"权重总和{total}应该接近1.0")
        return v


class RDAgentConfig(BaseModel):
    """RD-Agent配置"""
    model_config = ConfigDict(extra='allow')  # 允许额外字段
    
    enable: bool = Field(False, description="启用RD-Agent")
    rdagent_path: Optional[str] = Field(None, description="RD-Agent安装路径")
    llm_provider: str = Field("openai", description="LLM提供商")
    api_key: Optional[str] = Field(None, description="API密钥")
    max_iterations: int = Field(5, ge=1, le=50, description="最大迭代次数")
    
    @model_validator(mode='after')
    def validate_rdagent_path(self):
        """验证RD-Agent路径"""
        if not self.enable:
            return self
        
        v = self.rdagent_path
        
        if v is None:
            # 尝试从环境变量获取
            v = os.environ.get('RDAGENT_PATH')
            if v is None:
                raise ValueError(
                    "RD-Agent已启用但未指定路径。\n"
                    "请设置:\n"
                    "  1. 配置文件中的 rdagent_path\n"
                    "  2. 或环境变量 RDAGENT_PATH"
                )
        
        path = Path(v).expanduser().resolve()
        if not path.exists():
            raise ValueError(
                f"RD-Agent路径不存在: {path}\n"
                f"请确认:\n"
                f"  1. 路径是否正确\n"
                f"  2. RD-Agent是否已安装\n"
                f"  3. 或使用 'pip install rdagent' 安装"
            )
        
        self.rdagent_path = str(path)
        return self


class LoggingConfig(BaseModel):
    """日志配置"""
    model_config = ConfigDict(extra='allow')  # 允许额外字段
    
    level: LogLevel = Field(LogLevel.INFO, description="日志级别")
    log_dir: str = Field("./logs", description="日志目录")
    max_file_size_mb: int = Field(100, ge=1, le=1000, description="单个日志文件最大大小(MB)")
    backup_count: int = Field(10, ge=1, le=100, description="日志文件保留数量")
    
    @field_validator('log_dir')
    @classmethod
    def create_log_dir(cls, v):
        """确保日志目录存在"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())


class QilinConfig(BaseSettings):
    """
    Qilin Stack 主配置
    支持从多个来源加载:
    1. 默认值
    2. YAML配置文件
    3. 环境变量 (最高优先级)
    """
    
    # 基础配置
    project_name: str = Field("Qilin Stack", description="项目名称")
    version: str = Field("2.1", description="版本号")
    environment: str = Field("production", description="运行环境")
    market: MarketType = Field(MarketType.CN, description="市场类型")
    
    # 子配置
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rdagent: RDAgentConfig = Field(default_factory=RDAgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # 额外配置 (存储未定义的字段)
    extra_fields: Dict[str, Any] = Field(default_factory=dict, description="额外配置")
    
    model_config = ConfigDict(
        extra='allow',  # 允许额外字段
        env_prefix="QILIN_",
        env_nested_delimiter="__",
        case_sensitive=False,
        use_enum_values=True
    )
        
    @model_validator(mode='after')
    def validate_config(self):
        """全局配置验证"""
        # 检查关键路径
        data_path = Path(self.data.qlib_data_path)
        if not data_path.exists():
            logger.warning(
                f"Qlib数据路径不存在: {data_path}\n"
                f"请运行: python scripts/get_data.py --source qlib"
            )
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()
    
    def save_to_yaml(self, path: str):
        """保存到YAML文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
        logger.info(f"配置已保存到: {path}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config: Optional[QilinConfig] = None
        self._config_file: Optional[str] = None
    
    def load_from_yaml(self, config_file: str) -> QilinConfig:
        """
        从YAML文件加载配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            配置对象
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_file}, 使用默认配置")
            self._config = QilinConfig()
            return self._config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f) or {}
            
            # Pydantic会自动合并环境变量
            self._config = QilinConfig(**config_dict)
            self._config_file = str(config_path.resolve())
            
            logger.info(f"✅ 配置加载成功: {config_file}")
            return self._config
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise
    
    def load_from_env(self) -> QilinConfig:
        """
        从环境变量加载配置
        
        Returns:
            配置对象
        """
        self._config = QilinConfig()
        logger.info("✅ 从环境变量加载配置")
        return self._config
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   **kwargs) -> QilinConfig:
        """
        加载配置(智能选择加载方式)
        
        优先级: 环境变量 > 传入参数 > YAML文件 > 默认值
        
        Args:
            config_file: 配置文件路径
            **kwargs: 额外配置参数(会覆盖文件配置)
            
        Returns:
            配置对象
        """
        # 1. 先加载配置文件或使用默认配置
        if config_file:
            config = self.load_from_yaml(config_file)
        else:
            # 尝试查找默认配置文件
            default_configs = [
                'config/default.yaml',
                'config.yaml',
                'config.example.yaml'
            ]
            
            config_loaded = False
            for default_config in default_configs:
                if Path(default_config).exists():
                    config = self.load_from_yaml(default_config)
                    config_loaded = True
                    break
            
            if not config_loaded:
                logger.info("未找到配置文件,使用默认配置和环境变量")
                config = self.load_from_env()
        
        # 2. 应用额外参数
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = QilinConfig(**config_dict)
            logger.info(f"应用了 {len(kwargs)} 个额外配置参数")
        
        self._config = config
        return config
    
    @property
    def config(self) -> QilinConfig:
        """获取当前配置"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload(self):
        """重新加载配置"""
        if self._config_file:
            return self.load_from_yaml(self._config_file)
        else:
            return self.load_from_env()


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_config() -> QilinConfig:
    """
    获取全局配置
    
    Returns:
        配置对象
    """
    return _config_manager.config


def load_config(config_file: Optional[str] = None, **kwargs) -> QilinConfig:
    """
    加载配置
    
    Args:
        config_file: 配置文件路径
        **kwargs: 额外配置参数
        
    Returns:
        配置对象
    """
    return _config_manager.load_config(config_file, **kwargs)


# 使用示例
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== 配置管理器测试 ===\n")
    
    # 1. 使用默认配置
    print("1. 加载默认配置:")
    config = QilinConfig()
    print(f"   项目: {config.project_name} v{config.version}")
    print(f"   市场: {config.market.value}")
    print(f"   初始资金: {config.backtest.initial_capital:,.0f}元")
    print(f"   T+1规则: {'启用' if config.backtest.enable_t_plus_1 else '禁用'}")
    print(f"   一字板严格模式: {'启用' if config.backtest.one_word_block_strict else '禁用'}")
    
    # 2. 测试验证
    print("\n2. 测试配置验证:")
    try:
        bad_config = QilinConfig(
            risk=RiskConfig(max_position_ratio=0.5, max_total_position_ratio=0.3)
        )
    except Exception as e:
        print(f"   ✅ 捕获到验证错误: {str(e)[:60]}...")
    
    # 3. 测试环境变量覆盖
    print("\n3. 环境变量覆盖测试:")
    os.environ['QILIN_STRATEGY__TOPK'] = '10'
    config_with_env = QilinConfig()
    print(f"   TOPK (环境变量): {config_with_env.strategy.topk}")
    
    # 4. 保存配置
    print("\n4. 保存配置到文件:")
    output_path = "config_test_output.yaml"
    config.save_to_yaml(output_path)
    print(f"   ✅ 已保存到: {output_path}")
    
    # 5. 测试配置管理器
    print("\n5. 配置管理器测试:")
    manager = ConfigManager()
    loaded_config = manager.load_config()
    print(f"   ✅ 配置加载成功")
    print(f"   初始资金: {loaded_config.backtest.initial_capital:,.0f}元")
    
    # 6. RD-Agent配置验证
    print("\n6. RD-Agent配置验证:")
    try:
        rd_config = RDAgentConfig(enable=True, rdagent_path=None)
    except Exception as e:
        print(f"   ✅ 捕获到RD-Agent路径验证错误")
        print(f"   错误信息包含设置指引: {'RDAGENT_PATH' in str(e)}")
    
    print("\n=== 测试完成 ===")
