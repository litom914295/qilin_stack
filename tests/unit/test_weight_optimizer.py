"""
权重优化器单元测试
"""
import pytest
import numpy as np
from decision_engine.weight_optimizer import (
    WeightOptimizer,
    PerformanceMetrics,
    SystemPerformance
)


class TestPerformanceMetrics:
    """测试性能指标数据类"""
    
    def test_metrics_creation(self):
        """测试指标创建"""
        metrics = PerformanceMetrics(
            accuracy=0.65,
            precision=0.70,
            recall=0.60,
            f1_score=0.65,
            sharpe_ratio=1.5,
            win_rate=0.55
        )
        assert metrics.accuracy == 0.65
        assert metrics.f1_score == 0.65
        assert metrics.sharpe_ratio == 1.5
    
    def test_metrics_validation(self):
        """测试指标验证"""
        with pytest.raises(ValueError):
            PerformanceMetrics(accuracy=1.5, precision=0.7, recall=0.6)


class TestSystemPerformance:
    """测试系统性能类"""
    
    def test_system_performance_creation(self):
        """测试系统性能创建"""
        metrics = PerformanceMetrics(0.65, 0.70, 0.60, 0.65, 1.5, 0.55)
        perf = SystemPerformance(
            system_name='qlib',
            metrics=metrics,
            sample_size=100
        )
        assert perf.system_name == 'qlib'
        assert perf.sample_size == 100


class TestWeightOptimizer:
    """测试权重优化器"""
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        optimizer = WeightOptimizer()
        assert optimizer.systems == ['qlib', 'trading_agents', 'rd_agent']
        assert len(optimizer.performance_history) == 0
    
    def test_evaluate_performance(self):
        """测试性能评估"""
        optimizer = WeightOptimizer()
        
        # 模拟预测和真实值
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        actuals = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        returns = np.random.randn(10) * 0.02
        
        metrics = optimizer.evaluate_performance(
            system_name='qlib',
            predictions=predictions,
            actuals=actuals,
            returns=returns
        )
        
        assert isinstance(metrics, SystemPerformance)
        assert metrics.system_name == 'qlib'
        assert 0 <= metrics.metrics.accuracy <= 1
        assert 0 <= metrics.metrics.f1_score <= 1
    
    def test_calculate_metrics(self):
        """测试指标计算"""
        optimizer = WeightOptimizer()
        
        # 完美预测
        predictions = np.array([1, 0, 1, 0])
        actuals = np.array([1, 0, 1, 0])
        
        metrics = optimizer._calculate_metrics(predictions, actuals)
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
    
    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        optimizer = WeightOptimizer()
        
        # 正收益
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = optimizer._calculate_sharpe_ratio(returns)
        assert sharpe > 0
        
        # 负收益
        returns = np.array([-0.01, -0.02, 0.01, -0.03, -0.01])
        sharpe = optimizer._calculate_sharpe_ratio(returns)
        assert sharpe < 0
        
        # 零收益
        returns = np.array([0.0, 0.0, 0.0])
        sharpe = optimizer._calculate_sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_optimize_weights_basic(self):
        """测试基本权重优化"""
        optimizer = WeightOptimizer()
        
        # 添加性能数据
        for system in ['qlib', 'trading_agents', 'rd_agent']:
            predictions = np.random.randint(0, 2, 100)
            actuals = np.random.randint(0, 2, 100)
            returns = np.random.randn(100) * 0.02
            
            optimizer.evaluate_performance(system, predictions, actuals, returns)
        
        # 优化权重
        new_weights = optimizer.optimize_weights()
        
        assert len(new_weights) == 3
        assert 'qlib' in new_weights
        assert 'trading_agents' in new_weights
        assert 'rd_agent' in new_weights
        assert abs(sum(new_weights.values()) - 1.0) < 0.01  # 总和为1
        
        # 检查权重范围
        for weight in new_weights.values():
            assert 0.1 <= weight <= 0.6
    
    def test_optimize_weights_with_constraints(self):
        """测试带约束的权重优化"""
        optimizer = WeightOptimizer(
            min_weight=0.2,
            max_weight=0.5
        )
        
        # 添加性能数据
        for system in ['qlib', 'trading_agents', 'rd_agent']:
            predictions = np.random.randint(0, 2, 100)
            actuals = np.random.randint(0, 2, 100)
            returns = np.random.randn(100) * 0.02
            
            optimizer.evaluate_performance(system, predictions, actuals, returns)
        
        new_weights = optimizer.optimize_weights()
        
        # 检查约束
        for weight in new_weights.values():
            assert 0.2 <= weight <= 0.5
    
    def test_optimize_weights_insufficient_data(self):
        """测试数据不足时的权重优化"""
        optimizer = WeightOptimizer()
        
        # 只添加一个系统的数据
        predictions = np.random.randint(0, 2, 100)
        actuals = np.random.randint(0, 2, 100)
        returns = np.random.randn(100) * 0.02
        
        optimizer.evaluate_performance('qlib', predictions, actuals, returns)
        
        # 应返回默认权重
        new_weights = optimizer.optimize_weights()
        assert new_weights == optimizer.default_weights
    
    def test_get_performance_summary(self):
        """测试性能摘要"""
        optimizer = WeightOptimizer()
        
        # 添加性能数据
        predictions = np.random.randint(0, 2, 100)
        actuals = np.random.randint(0, 2, 100)
        returns = np.random.randn(100) * 0.02
        
        optimizer.evaluate_performance('qlib', predictions, actuals, returns)
        optimizer.evaluate_performance('trading_agents', predictions, actuals, returns)
        
        summary = optimizer.get_performance_summary()
        
        assert 'qlib' in summary
        assert 'trading_agents' in summary
        assert isinstance(summary['qlib'], dict)
        assert 'accuracy' in summary['qlib']
        assert 'f1_score' in summary['qlib']
    
    def test_adaptive_update_daily(self):
        """测试每日自适应更新"""
        optimizer = WeightOptimizer()
        
        # 模拟连续多天的性能数据
        for day in range(5):
            for system in ['qlib', 'trading_agents', 'rd_agent']:
                predictions = np.random.randint(0, 2, 20)
                actuals = np.random.randint(0, 2, 20)
                returns = np.random.randn(20) * 0.02
                
                optimizer.evaluate_performance(system, predictions, actuals, returns)
        
        # 执行自适应更新
        new_weights = optimizer.adaptive_update(strategy='daily')
        
        assert len(new_weights) == 3
        assert abs(sum(new_weights.values()) - 1.0) < 0.01
    
    def test_compare_systems(self):
        """测试系统比较"""
        optimizer = WeightOptimizer()
        
        # 添加不同质量的性能数据
        # qlib - 高准确率
        optimizer.evaluate_performance(
            'qlib',
            np.array([1, 1, 1, 1, 0, 0, 0, 0]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0]),
            np.random.randn(8) * 0.02
        )
        
        # trading_agents - 低准确率
        optimizer.evaluate_performance(
            'trading_agents',
            np.array([1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            np.random.randn(8) * 0.02
        )
        
        summary = optimizer.get_performance_summary()
        
        # qlib应该有更高的准确率
        assert summary['qlib']['accuracy'] > summary['trading_agents']['accuracy']


@pytest.mark.integration
class TestWeightOptimizerIntegration:
    """权重优化器集成测试"""
    
    def test_full_optimization_cycle(self):
        """测试完整优化周期"""
        optimizer = WeightOptimizer()
        
        # 模拟一周的交易数据
        for day in range(7):
            for system in ['qlib', 'trading_agents', 'rd_agent']:
                # 生成模拟数据
                predictions = np.random.randint(0, 2, 50)
                actuals = np.random.randint(0, 2, 50)
                returns = np.random.randn(50) * 0.02
                
                optimizer.evaluate_performance(system, predictions, actuals, returns)
        
        # 获取性能摘要
        summary = optimizer.get_performance_summary()
        assert len(summary) == 3
        
        # 优化权重
        new_weights = optimizer.optimize_weights()
        assert len(new_weights) == 3
        assert abs(sum(new_weights.values()) - 1.0) < 0.01
        
        # 每个系统都应该有权重
        for system in ['qlib', 'trading_agents', 'rd_agent']:
            assert system in new_weights
            assert new_weights[system] > 0
