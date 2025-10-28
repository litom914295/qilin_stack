"""
改进功能测试套件
测试所有Critical和High级别的改进
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.validators import Validator, ValidationError, RiskValidator
from app.core.backtest_engine import BacktestEngine, Portfolio, Position, Order, OrderSide, OrderType
from app.core.config_manager import (
    QilinConfig, ConfigManager, BacktestConfig, RiskConfig,
    RDAgentConfig, StrategyConfig
)
from qilin_stack.backtest.limit_up_backtest_adapter import LimitUpBacktestAdapter


class TestValidatorImprovements:
    """测试验证器改进 (C1, H2)"""
    
    def test_normalize_symbol_sh_to_standard(self):
        """测试SH格式转标准格式"""
        result = Validator.normalize_symbol("SH600000", "standard")
        assert result == "600000.SH"
    
    def test_normalize_symbol_standard_to_qlib(self):
        """测试标准格式转qlib格式"""
        result = Validator.normalize_symbol("600000.SH", "qlib")
        assert result == "SH600000"
    
    def test_normalize_symbol_auto_detect(self):
        """测试自动检测交易所"""
        # 测试沪市
        result_sh = Validator.normalize_symbol("600000", "qlib")
        assert result_sh == "SH600000"
        
        # 测试深市
        result_sz = Validator.normalize_symbol("000001", "qlib")
        assert result_sz == "SZ000001"
        
        # 测试创业板
        result_cy = Validator.normalize_symbol("300750", "qlib")
        assert result_cy == "SZ300750"
    
    def test_normalize_symbol_invalid(self):
        """测试无效的股票代码"""
        with pytest.raises(ValidationError):
            Validator.normalize_symbol("invalid.XX")
    
    def test_validate_parameter_min_max(self):
        """测试配置驱动的参数验证"""
        # 正常范围
        result = Validator.validate_parameter("topk", 5, min_val=1, max_val=10)
        assert result == 5
        
        # 超出最大值
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_parameter("topk", 15, min_val=1, max_val=10)
        assert "大于最大值" in str(exc_info.value)
        
        # 低于最小值
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_parameter("topk", 0, min_val=1, max_val=10)
        assert "小于最小值" in str(exc_info.value)
    
    def test_validate_parameter_allowed_values(self):
        """测试允许值列表验证"""
        # 正常值
        result = Validator.validate_parameter(
            "market", "cn", allowed_values=["cn", "us", "hk"]
        )
        assert result == "cn"
        
        # 不允许的值
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_parameter(
                "market", "jp", allowed_values=["cn", "us", "hk"]
            )
        assert "不在允许的值列表中" in str(exc_info.value)
    
    def test_validate_config_with_schema(self):
        """测试配置模式验证"""
        config_schema = {
            'topk': {'min': 1, 'max': 10, 'type': int, 'required': True},
            'max_runtime_sec': {'min': 10, 'max': 300, 'type': int, 'default': 45}
        }
        
        # 正常配置
        config = {'topk': 5}
        validated = Validator.validate_config(config, config_schema)
        assert validated['topk'] == 5
        assert validated['max_runtime_sec'] == 45  # 默认值
        
        # 缺少必需配置
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_config({}, config_schema)
        assert "缺少必需的配置项" in str(exc_info.value)


class TestTPlusOneRule:
    """测试T+1交易规则 (C2)"""
    
    def test_position_creation_with_frozen(self):
        """测试持仓创建时的冻结数量"""
        portfolio = Portfolio(initial_capital=1000000)
        timestamp = datetime.now()
        
        # 买入
        portfolio.update_position("SH600000", 1000, 10.0, timestamp)
        
        position = portfolio.positions["SH600000"]
        assert position.quantity == 1000
        assert position.available_quantity == 0  # 当日买入,不可卖
        assert position.frozen_quantity == 1000  # 全部冻结
    
    def test_unfreeze_positions_next_day(self):
        """测试次日解冻持仓"""
        portfolio = Portfolio(initial_capital=1000000)
        day1 = datetime(2024, 1, 15, 10, 0)
        day2 = datetime(2024, 1, 16, 9, 30)
        
        # 第1天买入
        portfolio.update_position("SH600000", 1000, 10.0, day1)
        position = portfolio.positions["SH600000"]
        assert position.frozen_quantity == 1000
        assert position.available_quantity == 0
        
        # 第2天解冻
        portfolio.unfreeze_positions(day2)
        assert position.frozen_quantity == 0
        assert position.available_quantity == 1000
    
    def test_cannot_sell_same_day(self):
        """测试当日买入不能卖出"""
        portfolio = Portfolio(initial_capital=1000000)
        timestamp = datetime.now()
        
        # 买入
        portfolio.update_position("SH600000", 1000, 10.0, timestamp)
        
        # 尝试当日卖出
        with pytest.raises(ValueError) as exc_info:
            portfolio.update_position("SH600000", -500, 10.5, timestamp)
        assert "T+1限制" in str(exc_info.value)
    
    def test_can_sell_next_day(self):
        """测试次日可以卖出"""
        portfolio = Portfolio(initial_capital=1000000)
        day1 = datetime(2024, 1, 15, 10, 0)
        day2 = datetime(2024, 1, 16, 10, 0)
        
        # 第1天买入
        portfolio.update_position("SH600000", 1000, 10.0, day1)
        
        # 第2天解冻
        portfolio.unfreeze_positions(day2)
        
        # 第2天卖出
        portfolio.update_position("SH600000", -500, 10.5, day2)
        
        position = portfolio.positions["SH600000"]
        assert position.quantity == 500
        assert position.available_quantity == 500
    
    def test_backtest_engine_validates_t_plus_1(self):
        """测试回测引擎验证T+1规则"""
        engine = BacktestEngine(initial_capital=1000000)
        engine.current_timestamp = datetime(2024, 1, 15, 10, 0)
        
        # 买入
        engine.portfolio.update_position("SH600000", 1000, 10.0, engine.current_timestamp)
        
        # 创建卖出订单(当日)
        sell_order = Order(
            symbol="SH600000",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=500,
            timestamp=engine.current_timestamp
        )
        
        # 验证应该失败
        is_valid = engine._validate_order(sell_order)
        assert not is_valid  # T+1规则阻止


class TestLimitUpRestriction:
    """测试涨停板限制 (C3)"""
    
    def test_calculate_limit_price(self):
        """测试涨停价计算"""
        adapter = LimitUpBacktestAdapter()
        
        # 主板10%
        limit_price = adapter.calculate_limit_price(10.0, 0.10)
        assert limit_price == 11.0
        
        # 科创板20%
        limit_price_sci = adapter.calculate_limit_price(20.0, 0.20)
        assert limit_price_sci == 24.0
    
    def test_is_limit_up(self):
        """测试涨停判断"""
        adapter = LimitUpBacktestAdapter()
        
        # 正好涨停
        assert adapter.is_limit_up("SH600000", 11.0, 10.0, 0.10)
        
        # 未涨停
        assert not adapter.is_limit_up("SH600000", 10.5, 10.0, 0.10)
        
        # 允许1分钱误差
        assert adapter.is_limit_up("SH600000", 11.01, 10.0, 0.10)
    
    def test_get_limit_up_ratio(self):
        """测试涨停幅度识别"""
        adapter = LimitUpBacktestAdapter()
        
        # 主板
        assert adapter.get_limit_up_ratio("SH600000") == 0.10
        
        # 科创板
        assert adapter.get_limit_up_ratio("SH688001") == 0.20
        
        # 创业板
        assert adapter.get_limit_up_ratio("SZ300001") == 0.20
        
        # ST股票
        assert adapter.get_limit_up_ratio("SHST0001") == 0.05
    
    def test_one_word_board_strict_mode(self):
        """测试一字板严格模式"""
        adapter = LimitUpBacktestAdapter(
            enable_one_word_block=True,
            strict_mode=True
        )
        
        # 一字板 (开盘即封)
        can_fill, execution = adapter.can_buy_at_limit_up(
            symbol="SH600000",
            order_time=datetime(2024, 1, 15, 9, 40),
            target_shares=10000,
            limit_price=11.0,
            seal_amount=100_000_000,  # 1亿封单
            seal_time=datetime(2024, 1, 15, 9, 30),  # 开盘即封
            open_times=0
        )
        
        # 严格模式下不能成交
        assert not can_fill
        assert execution.filled_shares == 0
        assert "一字板" in execution.execution_reason
    
    def test_mid_seal_can_fill(self):
        """测试盘中封板可能成交"""
        adapter = LimitUpBacktestAdapter(strict_mode=True)
        
        # 盘中封板
        can_fill, execution = adapter.can_buy_at_limit_up(
            symbol="SH600000",
            order_time=datetime(2024, 1, 15, 11, 0),
            target_shares=10000,
            limit_price=11.0,
            seal_amount=30_000_000,  # 3000万封单
            seal_time=datetime(2024, 1, 15, 10, 30),
            open_times=0
        )
        
        # 盘中封板有成交概率(可能成交或不成交)
        assert execution is not None
        if can_fill:
            assert execution.filled_shares > 0


class TestConfigManagement:
    """测试配置管理 (H1)"""
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        config = QilinConfig()
        
        assert config.project_name == "Qilin Stack"
        assert config.version == "2.1"
        # use_enum_values=True 使得 market 直接是字符串
        assert config.market == "cn"
        assert config.backtest.initial_capital == 1000000
        assert config.backtest.enable_t_plus_1 is True
    
    def test_backtest_config_validation(self):
        """测试回测配置验证"""
        # 正常配置
        config = BacktestConfig(initial_capital=100000)
        assert config.initial_capital == 100000
        
        # 资金过低
        with pytest.raises(ValueError):
            BacktestConfig(initial_capital=5000)  # 低于最小值10000
    
    def test_risk_config_validation(self):
        """测试风险配置验证"""
        # 正常配置
        config = RiskConfig(max_position_ratio=0.2)
        assert config.max_position_ratio == 0.2
        
        # 单票仓位超过总仓位
        with pytest.raises(ValueError) as exc_info:
            RiskConfig(
                max_position_ratio=0.5,
                max_total_position_ratio=0.3
            )
        assert "不能超过总仓位" in str(exc_info.value)
    
    def test_rdagent_config_validation(self):
        """测试RD-Agent配置验证 (H3)"""
        # 禁用RD-Agent,路径可以为空
        config = RDAgentConfig(enable=False, rdagent_path=None)
        assert config.enable is False
        
        # 启用RD-Agent但未提供路径
        with pytest.raises(ValueError) as exc_info:
            RDAgentConfig(enable=True, rdagent_path=None)
        
        error_msg = str(exc_info.value)
        assert "RD-Agent已启用但未指定路径" in error_msg
        assert "RDAGENT_PATH" in error_msg  # 应包含环境变量提示
    
    def test_strategy_config_bounds(self):
        """测试策略配置边界"""
        # 正常值
        config = StrategyConfig(topk=5)
        assert config.topk == 5
        
        # 超出上限
        with pytest.raises(ValueError):
            StrategyConfig(topk=25)  # 最大20
        
        # 低于下限
        with pytest.raises(ValueError):
            StrategyConfig(topk=0)  # 最小1
    
    def test_config_manager_load(self):
        """测试配置管理器加载"""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert config is not None
        assert isinstance(config, QilinConfig)
    
    def test_environment_variable_override(self):
        """测试环境变量覆盖"""
        # 设置环境变量
        os.environ['QILIN_STRATEGY__TOPK'] = '8'
        
        config = QilinConfig()
        assert config.strategy.topk == 8
        
        # 清理
        os.environ.pop('QILIN_STRATEGY__TOPK', None)
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = QilinConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'backtest' in config_dict
        assert 'risk' in config_dict
        assert config_dict['project_name'] == "Qilin Stack"


class TestIntegration:
    """集成测试"""
    
    def test_full_backtest_flow_with_t_plus_1(self):
        """测试完整回测流程(含T+1)"""
        engine = BacktestEngine(initial_capital=1000000)
        
        # 模拟两天的交易
        day1 = datetime(2024, 1, 15, 10, 0)
        day2 = datetime(2024, 1, 16, 10, 0)
        
        engine.current_timestamp = day1
        engine.portfolio.current_date = day1
        
        # 第1天买入
        engine.portfolio.update_position("SH600000", 1000, 10.0, day1)
        
        # 切换到第2天
        engine.current_timestamp = day2
        engine.portfolio.current_date = day2
        engine.portfolio.unfreeze_positions(day2)
        
        # 第2天卖出
        engine.portfolio.update_position("SH600000", -500, 10.5, day2)
        
        # 验证最终持仓
        position = engine.portfolio.positions["SH600000"]
        assert position.quantity == 500
    
    def test_config_with_backtest_engine(self):
        """测试配置与回测引擎集成"""
        config = QilinConfig()
        
        # 使用配置创建回测引擎
        engine = BacktestEngine(
            initial_capital=config.backtest.initial_capital,
            commission_rate=config.backtest.commission_rate,
            slippage_rate=config.backtest.slippage_rate,
            min_commission=config.backtest.min_commission
        )
        
        assert engine.initial_capital == config.backtest.initial_capital
        assert engine.commission_rate == config.backtest.commission_rate


# Pytest配置
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
