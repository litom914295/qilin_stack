"""
配置管理模块
提供YAML配置文件加载、环境变量覆盖、配置验证

功能：
- YAML配置文件加载
- 环境变量覆盖
- 配置验证和类型转换
- 默认配置
- 多环境配置支持
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """
    配置管理器
    提供统一的配置加载和管理
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 系统配置
        'system': {
            'project_name': 'qilin_limitup_ai',
            'version': '1.0.0',
            'log_level': 'INFO',
            'debug': False
        },
        
        # 数据配置
        'data': {
            'data_dir': './data',
            'cache_dir': './cache',
            'output_dir': './output'
        },
        
        # 模型配置
        'model': {
            'model_dir': './models',
            'model_type': 'lightgbm',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 7,
            'random_state': 42
        },
        
        # 特征配置
        'features': {
            'feature_version': 'v1',
            'n_features': 100,
            'feature_selection': True,
            'cache_features': True
        },
        
        # 训练配置
        'training': {
            'train_start_date': '2020-01-01',
            'train_end_date': '2023-12-31',
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'cv_folds': 5
        },
        
        # 预测配置
        'prediction': {
            'confidence_threshold': 0.5,
            'top_k': 10,
            'enable_risk_filter': True
        },
        
        # 回测配置
        'backtest': {
            'initial_capital': 1000000.0,
            'commission': 0.0003,
            'slippage': 0.001,
            'position_size': 0.1
        },
        
        # 风控配置
        'risk': {
            'max_position_pct': 0.3,
            'max_drawdown_pct': 0.2,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15
        },
        
        # MLflow配置
        'mlflow': {
            'tracking_uri': './mlruns',
            'experiment_name': 'limitup_ai',
            'enable_logging': True
        },
        
        # 缓存配置
        'cache': {
            'cache_dir': './feature_cache',
            'ttl_hours': 24,
            'max_size_gb': 10.0
        },
        
        # 工作流配置
        'workflow': {
            'enable_t_day_screening': True,
            'enable_t1_auction_monitor': True,
            'enable_t1_buy': True,
            'enable_t2_sell': True,
            'auto_mode': False
        },
        
        # 筛选配置
        'screening': {
            'min_seal_strength': 3.0,
            'min_prediction_score': 0.6,
            'max_candidates': 30,
            'min_volume': 100000000,
            'exclude_st': True,
            'exclude_new_stock_days': 60
        },
        
        # 竞价配置
        'auction': {
            'min_auction_strength': 0.6,
            'monitor_start_time': '09:15',
            'monitor_end_time': '09:25',
            'min_buy_ratio': 0.5,
            'max_price_premium': 0.03
        },
        
        # 买入配置
        'buy': {
            'total_capital': 1000000,
            'max_position_per_stock': 0.10,
            'min_position_per_stock': 0.02,
            'max_total_position': 0.80,
            'enable_layered_buy': True,
            'layer_count': 3
        },
        
        # 卖出配置
        'sell': {
            'enable_partial_sell': True,
            'profit_target': 0.05,
            'stop_loss': -0.03,
            'trailing_stop': True,
            'trailing_stop_trigger': 0.03,
            'trailing_stop_distance': 0.02
        },
        
        # Kelly仓位管理配置
        'kelly': {
            'enable_kelly': True,
            'kelly_fraction': 0.5,
            'max_kelly_position': 0.15,
            'min_kelly_position': 0.02,
            'confidence_weight': 0.3
        },
        
        # 市场熔断配置
        'market_breaker': {
            'enable_breaker': True,
            'index_drop_threshold': -0.02,
            'limit_down_ratio_threshold': 0.05,
            'continuous_loss_threshold': 3,
            'max_drawdown_threshold': -0.10
        },
        
        # 消息推送配置
        'notification': {
            'enable_notification': True,
            'channels': [],
            'wechat_webhook': '',
            'dingtalk_webhook': '',
            'email_smtp_server': '',
            'email_smtp_port': 587,
            'email_from': '',
            'email_to': []
        },
        
        # 定时任务配置
        'scheduler': {
            'enable_scheduler': True,
            't_day_screening_time': '15:30',
            't1_auction_monitor_time': '09:15',
            't2_sell_time': '09:30',
            'timezone': 'Asia/Shanghai'
        },
        
        # 交易日志配置
        'journal': {
            'enable_journal': True,
            'db_path': 'data/trading_journal.db',
            'auto_backup': True,
            'backup_days': 30
        }
    }
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        env: str = 'default',
        override_from_env: bool = True
    ):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
            env: 环境名称 ('default', 'dev', 'prod', 'test')
            override_from_env: 是否从环境变量覆盖配置
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.env = env
        
        # 加载配置文件
        if config_file:
            self.load_from_file(config_file)
        
        # 从环境变量覆盖
        if override_from_env:
            self.override_from_env()
    
    def load_from_file(self, config_file: str):
        """
        从YAML文件加载配置
        
        Args:
            config_file: 配置文件路径
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            warnings.warn(f"Config file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            if loaded_config:
                # 深度合并配置
                self._deep_update(self.config, loaded_config)
        
        except Exception as e:
            warnings.warn(f"Failed to load config file: {e}")
    
    def override_from_env(self):
        """从环境变量覆盖配置"""
        # 环境变量前缀
        prefix = 'QILIN_'
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # 移除前缀并转换为小写
            config_key = env_key[len(prefix):].lower()
            
            # 解析嵌套键（例如：QILIN_MODEL_TYPE -> model.type）
            parts = config_key.split('_')
            
            if len(parts) >= 2:
                section = parts[0]
                key = '_'.join(parts[1:])
                
                if section in self.config:
                    # 自动类型转换
                    typed_value = self._convert_type(env_value)
                    self.config[section][key] = typed_value
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """
        深度更新字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _convert_type(self, value: str) -> Any:
        """
        自动类型转换
        
        Args:
            value: 字符串值
        
        Returns:
            转换后的值
        """
        # 布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 数值
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 字符串
        return value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持嵌套键）
        
        Args:
            key_path: 配置键路径（例如：'model.learning_rate'）
            default: 默认值
        
        Returns:
            配置值
        """
        parts = key_path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值（支持嵌套键）
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        parts = key_path.split('.')
        config = self.config
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def get_section(self, section: str) -> Dict:
        """
        获取配置节
        
        Args:
            section: 配置节名称
        
        Returns:
            配置节字典
        """
        return self.config.get(section, {})
    
    def validate(self) -> bool:
        """
        验证配置
        
        Returns:
            是否有效
        """
        required_sections = ['system', 'model', 'data']
        
        for section in required_sections:
            if section not in self.config:
                warnings.warn(f"Missing required config section: {section}")
                return False
        
        # 验证数据路径
        if not isinstance(self.config['data'].get('data_dir'), str):
            warnings.warn("Invalid data_dir configuration")
            return False
        
        # 验证模型参数
        if self.config['model'].get('learning_rate', 0) <= 0:
            warnings.warn("Invalid learning_rate: must be positive")
            return False
        
        # 验证策略配置（如果存在）
        if 'screening' in self.config:
            screening = self.config['screening']
            if screening.get('min_seal_strength', 0) < 0:
                warnings.warn("Invalid min_seal_strength: must be >= 0")
                return False
            if not 0 <= screening.get('min_prediction_score', 0) <= 1:
                warnings.warn("Invalid min_prediction_score: must be in [0, 1]")
                return False
        
        if 'buy' in self.config:
            buy = self.config['buy']
            if buy.get('total_capital', 0) <= 0:
                warnings.warn("Invalid total_capital: must be > 0")
                return False
            if not 0 < buy.get('max_position_per_stock', 0) <= 1:
                warnings.warn("Invalid max_position_per_stock: must be in (0, 1]")
                return False
        
        if 'sell' in self.config:
            sell = self.config['sell']
            if sell.get('stop_loss', 0) >= 0:
                warnings.warn("Invalid stop_loss: must be < 0")
                return False
            if sell.get('profit_target', 0) <= 0:
                warnings.warn("Invalid profit_target: must be > 0")
                return False
        
        return True
    
    def save_to_file(self, output_file: str):
        """
        保存配置到文件
        
        Args:
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            warnings.warn(f"Failed to save config: {e}")
    
    def to_dict(self) -> Dict:
        """返回配置字典"""
        return self.config.copy()
    
    def __repr__(self) -> str:
        return f"ConfigManager(env='{self.env}', sections={list(self.config.keys())})"


# 全局配置实例（单例模式）
_global_config: Optional[ConfigManager] = None


def get_config(
    config_file: Optional[str] = None,
    env: str = 'default',
    reload: bool = False
) -> ConfigManager:
    """
    获取全局配置实例
    
    Args:
        config_file: 配置文件路径
        env: 环境名称
        reload: 是否重新加载
    
    Returns:
        配置管理器实例
    """
    global _global_config
    
    if _global_config is None or reload:
        _global_config = ConfigManager(config_file=config_file, env=env)
    
    return _global_config


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试配置管理器"""
    
    print("=" * 60)
    print("配置管理模块测试")
    print("=" * 60)
    
    # 创建配置管理器
    config = ConfigManager()
    
    print(f"配置管理器: {config}")
    print(f"环境: {config.env}")
    
    # 测试获取配置
    print("\n" + "=" * 60)
    print("测试获取配置...")
    print("=" * 60)
    
    print(f"\n项目名称: {config.get('system.project_name')}")
    print(f"模型类型: {config.get('model.model_type')}")
    print(f"学习率: {config.get('model.learning_rate')}")
    print(f"数据目录: {config.get('data.data_dir')}")
    print(f"不存在的键: {config.get('unknown.key', '默认值')}")
    
    # 测试设置配置
    print("\n" + "=" * 60)
    print("测试设置配置...")
    print("=" * 60)
    
    config.set('model.learning_rate', 0.1)
    print(f"更新后的学习率: {config.get('model.learning_rate')}")
    
    config.set('custom.new_param', 'test_value')
    print(f"新增参数: {config.get('custom.new_param')}")
    
    # 测试获取配置节
    print("\n" + "=" * 60)
    print("测试获取配置节...")
    print("=" * 60)
    
    model_config = config.get_section('model')
    print("\n模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    risk_config = config.get_section('risk')
    print("\n风控配置:")
    for key, value in risk_config.items():
        print(f"  {key}: {value}")
    
    # 测试配置验证
    print("\n" + "=" * 60)
    print("测试配置验证...")
    print("=" * 60)
    
    is_valid = config.validate()
    print(f"\n配置验证结果: {'通过' if is_valid else '失败'}")
    
    # 测试保存配置
    print("\n" + "=" * 60)
    print("测试保存配置...")
    print("=" * 60)
    
    output_file = "./test_config.yaml"
    config.save_to_file(output_file)
    print(f"\n✓ 配置已保存到: {output_file}")
    
    # 测试从文件加载
    print("\n" + "=" * 60)
    print("测试从文件加载配置...")
    print("=" * 60)
    
    config2 = ConfigManager(config_file=output_file)
    print(f"\n从文件加载的学习率: {config2.get('model.learning_rate')}")
    print(f"从文件加载的项目名称: {config2.get('system.project_name')}")
    
    # 测试环境变量覆盖
    print("\n" + "=" * 60)
    print("测试环境变量覆盖...")
    print("=" * 60)
    
    os.environ['QILIN_MODEL_LEARNING_RATE'] = '0.08'
    os.environ['QILIN_SYSTEM_DEBUG'] = 'true'
    
    config3 = ConfigManager(override_from_env=True)
    print(f"\n环境变量覆盖后的学习率: {config3.get('model.learning_rate')}")
    print(f"环境变量覆盖后的debug: {config3.get('system.debug')}")
    
    # 测试全局配置
    print("\n" + "=" * 60)
    print("测试全局配置...")
    print("=" * 60)
    
    global_config = get_config()
    print(f"\n全局配置实例: {global_config}")
    print(f"项目名称: {global_config.get('system.project_name')}")
    
    # 验证单例模式
    global_config2 = get_config()
    print(f"是否为同一实例: {global_config is global_config2}")
    
    # 完整配置示例
    print("\n" + "=" * 60)
    print("完整配置示例 (部分):")
    print("=" * 60)
    
    print("\n系统配置:")
    for key, value in config.get_section('system').items():
        print(f"  {key}: {value}")
    
    print("\n训练配置:")
    for key, value in config.get_section('training').items():
        print(f"  {key}: {value}")
    
    print("\n回测配置:")
    for key, value in config.get_section('backtest').items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    # 清理测试文件
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"\n✓ 清理测试文件: {output_file}")
