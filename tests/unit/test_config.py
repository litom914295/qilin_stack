"""
测试Pydantic配置管理系统
"""

import pytest
from pathlib import Path
import yaml

from config.settings import (
    Settings,
    SystemConfig,
    TradingConfig,
    AgentWeights,
    get_settings,
    TradingMode,
    LogLevel
)


class TestSystemConfig:
    """系统配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SystemConfig()
        assert config.name == "麒麟量化系统"
        assert config.version == "1.0.0"
        assert config.mode == TradingMode.SIMULATION
        assert config.timezone == "Asia/Shanghai"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SystemConfig(
            name="Test System",
            version="2.0.0",
            mode=TradingMode.LIVE
        )
        assert config.name == "Test System"
        assert config.version == "2.0.0"
        assert config.mode == TradingMode.LIVE


class TestTradingConfig:
    """交易配置测试"""
    
    def test_default_trading_config(self):
        """测试默认交易配置"""
        config = TradingConfig()
        assert len(config.symbols) == 6
        assert config.max_positions == 5
        assert config.position_size == 0.2
        assert config.stop_loss == 0.05
        assert config.take_profit == 0.10
    
    def test_empty_symbols_validation(self):
        """测试空股票池验证"""
        with pytest.raises(ValueError, match="股票池不能为空"):
            TradingConfig(symbols=[])
    
    def test_position_size_validation(self):
        """测试仓位配置验证"""
        # 有效配置: 0.2 * 5 = 1.0
        config = TradingConfig(position_size=0.2, max_positions=5)
        assert config.position_size == 0.2
        
        # 无效配置: 0.3 * 5 = 1.5 > 1.0
        with pytest.raises(ValueError, match="总仓位.*不能超过100%"):
            TradingConfig(position_size=0.3, max_positions=5)
    
    def test_risk_params_bounds(self):
        """测试风控参数边界"""
        # 止损不能为负
        with pytest.raises(ValueError):
            TradingConfig(stop_loss=-0.1)
        
        # 止损不能超过0.5
        with pytest.raises(ValueError):
            TradingConfig(stop_loss=0.6)
        
        # 最大回撤不能超过0.5
        with pytest.raises(ValueError):
            TradingConfig(max_drawdown=0.6)


class TestAgentWeights:
    """Agent权重测试"""
    
    def test_default_weights(self):
        """测试默认权重"""
        weights = AgentWeights()
        total = sum(weights.model_dump().values())
        assert 0.99 <= total <= 1.01, f"权重总和应为1.0,实际为{total}"
    
    def test_weights_sum_validation(self):
        """测试权重总和验证"""
        # 有效权重(总和为1.0)
        weights = AgentWeights(
            market_ecology=0.1,
            auction_game=0.1,
            money_nature=0.1,
            zt_quality=0.1,
            leader=0.1,
            emotion=0.1,
            technical=0.1,
            position=0.1,
            risk=0.1,
            news=0.1
        )
        assert weights is not None
        
        # 无效权重(总和不为1.0)
        with pytest.raises(ValueError, match="Agent权重总和必须为1.0"):
            AgentWeights(
                market_ecology=0.2,
                auction_game=0.2,
                money_nature=0.2,
                zt_quality=0.2,
                leader=0.2,
                emotion=0.0,
                technical=0.0,
                position=0.0,
                risk=0.0,
                news=0.0
            )


class TestSettings:
    """统一配置测试"""
    
    def test_from_yaml(self, tmp_path):
        """测试从YAML加载配置"""
        # 创建临时YAML文件
        config_data = {
            "system": {
                "name": "Test System",
                "version": "1.0.0",
                "mode": "simulation"
            },
            "trading": {
                "symbols": ["000001", "000002"],
                "max_positions": 3,
                "position_size": 0.3,
                "stop_loss": 0.05,
                "take_profit": 0.10
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        # 加载配置
        settings = Settings.from_yaml(str(config_file))
        
        assert settings.system.name == "Test System"
        assert settings.trading.symbols == ["000001", "000002"]
        assert settings.trading.max_positions == 3
    
    def test_to_dict(self):
        """测试转换为字典"""
        settings = Settings()
        config_dict = settings.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "system" in config_dict
        assert "trading" in config_dict
        assert "agents" in config_dict
    
    def test_get_settings_singleton(self):
        """测试单例模式"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # 应该是同一个实例
        assert settings1 is settings2


class TestDatabaseConfig:
    """数据库配置测试"""
    
    def test_port_validation(self):
        """测试端口号验证"""
        from config.settings import DatabaseConfig
        
        # 有效端口
        config = DatabaseConfig(mongodb_port=27017)
        assert config.mongodb_port == 27017
        
        # 无效端口(超出范围)
        with pytest.raises(ValueError):
            DatabaseConfig(mongodb_port=70000)
        
        # 无效端口(负数)
        with pytest.raises(ValueError):
            DatabaseConfig(mongodb_port=-1)


class TestMonitoringConfig:
    """监控配置测试"""
    
    def test_threshold_validation(self):
        """测试阈值验证"""
        from config.settings import MonitoringConfig
        
        # 有效阈值
        config = MonitoringConfig(error_rate_threshold=0.05)
        assert config.error_rate_threshold == 0.05
        
        # 错误率不能超过1
        with pytest.raises(ValueError):
            MonitoringConfig(error_rate_threshold=1.5)
        
        # 内存使用率不能超过1
        with pytest.raises(ValueError):
            MonitoringConfig(memory_usage_threshold=1.2)


class TestBacktestConfig:
    """回测配置测试"""
    
    def test_commission_validation(self):
        """测试手续费率验证"""
        from config.settings import BacktestConfig
        
        # 有效手续费率
        config = BacktestConfig(commission=0.0003)
        assert config.commission == 0.0003
        
        # 手续费率不能为负
        with pytest.raises(ValueError):
            BacktestConfig(commission=-0.001)
        
        # 手续费率不能超过1%
        with pytest.raises(ValueError):
            BacktestConfig(commission=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
