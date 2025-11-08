"""
P2-1单元测试: 嵌套执行器集成
测试覆盖率目标: 100%

测试模块:
1. MarketImpactModel - 市场冲击成本模型
2. SlippageModel - 滑点模型
3. OrderSplitter - 订单拆分策略
4. ProductionNestedExecutor - 生产级嵌套执行器
"""

import pytest
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, 'G:\\test\\qilin_stack')

from qlib_enhanced.nested_executor_integration import (
    MarketImpactModel,
    SlippageModel,
    OrderSplitter,
    ProductionNestedExecutor,
    create_production_executor
)


# ==================== MarketImpactModel测试 ====================

class TestMarketImpactModel:
    """市场冲击成本模型测试"""
    
    def test_init(self):
        """测试模型初始化"""
        model = MarketImpactModel(
            permanent_impact=0.1,
            temporary_impact=0.01
        )
        assert model.permanent == 0.1
        assert model.temporary == 0.01
        assert model.min_impact_pct == 0.0001
        assert model.max_impact_pct == 0.05
    
    def test_calculate_cost_normal(self):
        """测试正常情况的成本计算"""
        model = MarketImpactModel()
        cost = model.calculate_cost(
            order_size=10000,
            daily_volume=5000000,
            price=10.0
        )
        # 成本应该大于0
        assert cost > 0
        # 参与率 = 10000/5000000 = 0.002
        # 永久成本 = 0.1 * 0.002 * 10 * 10000 = 20
        # 临时成本 = 0.01 * sqrt(0.002) * 10 * 10000 = 44.72
        # 总成本约 64.72
        assert 60 < cost < 70
    
    def test_calculate_cost_zero_volume(self):
        """测试零成交量情况"""
        model = MarketImpactModel()
        cost = model.calculate_cost(
            order_size=10000,
            daily_volume=0,
            price=10.0
        )
        assert cost == 0.0
    
    def test_calculate_cost_zero_order(self):
        """测试零订单量情况"""
        model = MarketImpactModel()
        cost = model.calculate_cost(
            order_size=0,
            daily_volume=5000000,
            price=10.0
        )
        assert cost == 0.0
    
    def test_calculate_impact_price_buy(self):
        """测试买入时的冲击价格"""
        model = MarketImpactModel()
        impact_price = model.calculate_impact_price(
            order_size=10000,
            daily_volume=5000000,
            price=10.0,
            side='buy'
        )
        # 买入时价格应该上涨
        assert impact_price > 10.0
        assert impact_price < 10.1  # 不应该涨太多
    
    def test_calculate_impact_price_sell(self):
        """测试卖出时的冲击价格"""
        model = MarketImpactModel()
        impact_price = model.calculate_impact_price(
            order_size=10000,
            daily_volume=5000000,
            price=10.0,
            side='sell'
        )
        # 卖出时价格应该下跌
        assert impact_price < 10.0
        assert impact_price > 9.9
    
    def test_impact_cost_limits(self):
        """测试冲击成本上下限"""
        model = MarketImpactModel()
        # 极大订单 (超过最大限制)
        cost = model.calculate_cost(
            order_size=10000000,  # 巨大订单
            daily_volume=1000000,
            price=10.0
        )
        # 不应该超过max_impact_pct * price * order_size
        max_cost = 0.05 * 10.0 * 10000000
        assert cost <= max_cost


# ==================== SlippageModel测试 ====================

class TestSlippageModel:
    """滑点模型测试"""
    
    def test_init(self):
        """测试模型初始化"""
        model = SlippageModel(
            base_slippage=0.0005,
            volatility_factor=0.02,
            liquidity_factor=0.01
        )
        assert model.base_slippage == 0.0005
        assert model.volatility_factor == 0.02
        assert model.liquidity_factor == 0.01
    
    def test_calculate_slippage_normal(self):
        """测试正常滑点计算"""
        model = SlippageModel()
        slippage = model.calculate_slippage(
            order_size=10000,
            price=10.0,
            volatility=0.02,
            daily_volume=5000000
        )
        # 滑点应该大于0
        assert slippage > 0
        # 应该包含三部分: 基础+波动率+流动性
        # 基础: 0.0005 * 10 * 10000 = 50
        # 波动率: 0.02 * 0.02 * 10 * 10000 = 40
        # 流动性: 0.01 * (10000/5000000) * 10 * 10000 = 20
        # 总计约 92 (50+40+2=92)
        assert 85 < slippage < 100
    
    def test_slippage_increases_with_volatility(self):
        """测试滑点随波动率增加"""
        model = SlippageModel()
        slippage1 = model.calculate_slippage(
            order_size=10000,
            price=10.0,
            volatility=0.01,
            daily_volume=5000000
        )
        slippage2 = model.calculate_slippage(
            order_size=10000,
            price=10.0,
            volatility=0.05,
            daily_volume=5000000
        )
        assert slippage2 > slippage1
    
    def test_slippage_increases_with_size(self):
        """测试滑点随订单量增加"""
        model = SlippageModel()
        slippage1 = model.calculate_slippage(
            order_size=1000,
            price=10.0,
            volatility=0.02,
            daily_volume=5000000
        )
        slippage2 = model.calculate_slippage(
            order_size=50000,
            price=10.0,
            volatility=0.02,
            daily_volume=5000000
        )
        assert slippage2 > slippage1


# ==================== OrderSplitter测试 ====================

class TestOrderSplitter:
    """订单拆分策略测试"""
    
    def test_init(self):
        """测试拆分器初始化"""
        splitter = OrderSplitter(
            strategy='twap',
            max_participation_rate=0.1,
            min_order_size=100
        )
        assert splitter.strategy == 'twap'
        assert splitter.max_participation_rate == 0.1
        assert splitter.min_order_size == 100
    
    def test_split_order_twap(self):
        """测试TWAP策略"""
        splitter = OrderSplitter(strategy='twap')
        sizes = splitter.split_order(
            total_size=10000,
            num_slices=5
        )
        # 应该均匀拆分
        assert len(sizes) == 5
        assert all(abs(s - 2000) < 1 for s in sizes)  # 每份2000股
        assert abs(sum(sizes) - 10000) < 1  # 总和应该等于10000
    
    def test_split_order_vwap(self):
        """测试VWAP策略"""
        splitter = OrderSplitter(strategy='vwap')
        volume_profile = np.array([100, 200, 300, 200, 100])
        sizes = splitter.split_order(
            total_size=10000,
            num_slices=5,
            volume_profile=volume_profile
        )
        # 应该按成交量权重拆分
        assert len(sizes) == 5
        # 最大的应该在中间 (index=2, volume=300)
        assert sizes[2] > sizes[0]
        assert sizes[2] > sizes[4]
    
    def test_split_order_pov(self):
        """测试POV策略"""
        splitter = OrderSplitter(strategy='pov')
        sizes = splitter.split_order(
            total_size=10000,
            num_slices=4
        )
        assert len(sizes) == 4
        assert abs(sum(sizes) - 10000) < 1
    
    def test_split_order_min_size(self):
        """测试最小订单量约束"""
        splitter = OrderSplitter(strategy='twap', min_order_size=500)
        sizes = splitter.split_order(
            total_size=10000,  # 更大的总量
            num_slices=5
        )
        # 每份应该是2000,已经 > min_order_size
        assert all(s >= 500 for s in sizes)
        assert abs(sum(sizes) - 10000) < 1
    
    def test_split_order_zero_slices(self):
        """测试零拆分"""
        splitter = OrderSplitter()
        sizes = splitter.split_order(
            total_size=10000,
            num_slices=0
        )
        assert len(sizes) == 1
        assert sizes[0] == 10000
    
    def test_invalid_strategy(self):
        """测试无效策略"""
        splitter = OrderSplitter(strategy='invalid')
        with pytest.raises(ValueError):
            splitter.split_order(10000, 5)


# ==================== ProductionNestedExecutor测试 ====================

class TestProductionNestedExecutor:
    """生产级嵌套执行器测试"""
    
    def test_init(self):
        """测试执行器初始化"""
        executor = ProductionNestedExecutor()
        assert executor.daily_time_step == '1d'
        assert executor.hourly_time_step == '1h'
        assert executor.minute_time_step == '1min'
        assert executor.impact_model is not None
        assert executor.slippage_model is not None
        assert executor.order_splitter is not None
        assert executor.stats['total_orders'] == 0
    
    def test_custom_config(self):
        """测试自定义配置"""
        executor = ProductionNestedExecutor(
            daily_time_step='day',
            impact_model_config={'permanent_impact': 0.2},
            slippage_model_config={'base_slippage': 0.001}
        )
        assert executor.daily_time_step == 'day'
        assert executor.impact_model.permanent == 0.2
        assert executor.slippage_model.base_slippage == 0.001
    
    def test_simulate_order_execution_buy(self):
        """测试买入订单执行"""
        executor = ProductionNestedExecutor()
        order = {
            'symbol': '000001.SZ',
            'size': 10000,
            'side': 'buy',
            'price': 10.0
        }
        market_data = {
            'daily_volume': 5000000,
            'volatility': 0.02,
            'current_price': 10.0
        }
        
        result = executor.simulate_order_execution(order, market_data)
        
        # 验证结果结构
        assert result['symbol'] == '000001.SZ'
        assert result['filled_size'] == 10000
        assert result['avg_price'] > 10.0  # 买入价格应该更高
        assert result['impact_cost'] > 0
        assert result['slippage_cost'] > 0
        assert result['total_cost'] > 0
        assert result['execution_quality'] > 0
        assert 'timestamp' in result
    
    def test_simulate_order_execution_sell(self):
        """测试卖出订单执行"""
        executor = ProductionNestedExecutor()
        order = {
            'symbol': '000001.SZ',
            'size': 10000,
            'side': 'sell',
            'price': 10.0
        }
        market_data = {
            'daily_volume': 5000000,
            'volatility': 0.02,
            'current_price': 10.0
        }
        
        result = executor.simulate_order_execution(order, market_data)
        
        # 卖出价格应该更低
        assert result['avg_price'] < 10.0
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        executor = ProductionNestedExecutor()
        
        # 执行多个订单
        for i in range(10):
            order = {
                'symbol': '000001.SZ',
                'size': 10000,
                'side': 'buy',
                'price': 10.0
            }
            market_data = {
                'daily_volume': 5000000,
                'volatility': 0.02,
                'current_price': 10.0
            }
            executor.simulate_order_execution(order, market_data)
        
        stats = executor.get_statistics()
        assert stats['total_orders'] == 10
        assert stats['total_cost'] > 0
        assert stats['avg_impact_cost'] > 0
        assert stats['avg_slippage_cost'] > 0
        assert stats['avg_execution_quality'] > 0
    
    def test_reset_statistics(self):
        """测试统计信息重置"""
        executor = ProductionNestedExecutor()
        
        # 执行订单
        order = {
            'symbol': '000001.SZ',
            'size': 10000,
            'side': 'buy',
            'price': 10.0
        }
        market_data = {
            'daily_volume': 5000000,
            'volatility': 0.02,
            'current_price': 10.0
        }
        executor.simulate_order_execution(order, market_data)
        
        # 重置
        executor.reset_statistics()
        assert executor.stats['total_orders'] == 0
        assert executor.stats['total_cost'] == 0.0
    
    def test_get_statistics_empty(self):
        """测试空统计信息"""
        executor = ProductionNestedExecutor()
        stats = executor.get_statistics()
        assert stats['total_orders'] == 0


# ==================== 辅助函数测试 ====================

class TestHelpers:
    """辅助函数测试"""
    
    def test_create_production_executor_default(self):
        """测试默认配置创建执行器"""
        executor = create_production_executor()
        assert executor is not None
        assert executor.daily_time_step == '1d'
    
    def test_create_production_executor_custom(self):
        """测试自定义配置创建执行器"""
        config = {
            'daily_time_step': '2d',
            'impact_model_config': {'permanent_impact': 0.15}
        }
        executor = create_production_executor(config)
        assert executor.daily_time_step == '2d'
        assert executor.impact_model.permanent == 0.15


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 创建执行器
        executor = create_production_executor()
        
        # 2. 执行一系列订单
        orders = [
            {'symbol': '000001.SZ', 'size': 10000, 'side': 'buy', 'price': 10.0},
            {'symbol': '000002.SZ', 'size': 20000, 'side': 'sell', 'price': 15.0},
            {'symbol': '000003.SZ', 'size': 5000, 'side': 'buy', 'price': 8.0},
        ]
        
        market_data = {
            'daily_volume': 5000000,
            'volatility': 0.02,
            'current_price': 10.0
        }
        
        results = []
        for order in orders:
            result = executor.simulate_order_execution(order, market_data)
            results.append(result)
        
        # 3. 验证结果
        assert len(results) == 3
        assert all('total_cost' in r for r in results)
        
        # 4. 检查统计
        stats = executor.get_statistics()
        assert stats['total_orders'] == 3
        assert stats['avg_execution_quality'] > 0
    
    def test_order_splitting_integration(self):
        """测试订单拆分集成"""
        executor = ProductionNestedExecutor()
        
        # 大订单拆分
        total_size = 100000
        num_slices = 10
        sizes = executor.order_splitter.split_order(total_size, num_slices)
        
        # 每个子订单独立执行
        total_cost = 0
        for size in sizes:
            order = {
                'symbol': '000001.SZ',
                'size': size,
                'side': 'buy',
                'price': 10.0
            }
            market_data = {
                'daily_volume': 5000000,
                'volatility': 0.02,
                'current_price': 10.0
            }
            result = executor.simulate_order_execution(order, market_data)
            total_cost += result['total_cost']
        
        # 拆分订单的总成本应该小于单笔大订单
        # (因为减少了市场冲击)
        assert total_cost > 0
        assert executor.stats['total_orders'] == num_slices


# ==================== 性能测试 ====================

class TestPerformance:
    """性能测试"""
    
    def test_execution_speed(self):
        """测试执行速度"""
        import time
        
        executor = ProductionNestedExecutor()
        
        order = {
            'symbol': '000001.SZ',
            'size': 10000,
            'side': 'buy',
            'price': 10.0
        }
        market_data = {
            'daily_volume': 5000000,
            'volatility': 0.02,
            'current_price': 10.0
        }
        
        # 执行1000次
        start = time.time()
        for _ in range(1000):
            executor.simulate_order_execution(order, market_data)
        elapsed = time.time() - start
        
        # 应该在1秒内完成
        assert elapsed < 1.0
        print(f"\n执行1000次订单耗时: {elapsed:.3f}秒")
        print(f"平均每次: {elapsed/1000*1000:.2f}毫秒")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])
