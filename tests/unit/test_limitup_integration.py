"""
测试 limitup_integration.py 中的 P0-3 和 P0-6 功能

测试覆盖:
1. P0-3: 真实因子评估逻辑
2. P0-6: 配置驱动的因子类别 (factor_categories)
3. P0-6: 配置驱动的预测目标 (prediction_targets)
4. 动态指标计算 (next_day_limit_up_rate, open_premium, continuous_probability)
5. 预定义因子加载
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path


@pytest.fixture
def mock_config():
    """Mock 配置对象"""
    config = Mock()
    config.rdagent_path = "/fake/rdagent"
    config.checkpoint_path = None
    # P0-6: 因子类别
    config.factor_categories = [
        'seal_strength',
        'continuous_board',
        'concept_synergy',
        'timing'
    ]
    # P0-6: 预测目标
    config.prediction_targets = [
        'next_day_limit_up',
        'open_premium',
        'continuous_probability'
    ]
    return config


@pytest.fixture
def mock_limit_up_data():
    """Mock 涨停股票数据"""
    symbols = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    
    # 特征数据
    features_df = pd.DataFrame({
        'seal_amount': [1000, 1500, 800, 2000, 1200],
        'continuous_board': [1, 2, 1, 3, 2],
        'concept_heat': [5, 8, 3, 10, 6],
        'limit_up_strength': [0.8, 0.9, 0.7, 0.95, 0.85],
        'limit_up_minutes': [30, 60, 90, 20, 45],
        'volume': [10000, 15000, 8000, 20000, 12000],
        'volume_ma20': [8000, 10000, 7000, 15000, 10000],
        'volume_ratio': [1.25, 1.5, 1.14, 1.33, 1.2],
        'market_cap': [50000, 60000, 40000, 80000, 55000],
        'large_buy': [2000, 3000, 1500, 4000, 2500],
        'large_sell': [1000, 1200, 800, 1500, 1000],
        'amount': [100000, 150000, 80000, 200000, 120000]
    }, index=symbols)
    
    # 次日结果数据
    next_day_df = pd.DataFrame({
        'next_return': [0.05, 0.10, -0.02, 0.12, 0.08],
        'next_limit_up': [0, 1, 0, 1, 1],
        'open_premium': [0.02, 0.05, -0.01, 0.08, 0.04],
        'continuous_board': [1, 3, 1, 4, 3]
    }, index=symbols)
    
    return features_df, next_day_df


class TestPredefinedFactorLoading:
    """测试 P0-6: 配置驱动的预定义因子加载"""
    
    def test_load_factors_from_config_categories(self, mock_config):
        """测试从配置加载因子类别"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpDataInterface'), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            factors = integration._get_predefined_limit_up_factors()
            
            # 验证加载的因子数量和类别
            assert len(factors) == 4  # seal_strength, continuous_board, concept_synergy, timing
            
            factor_names = [f['name'] for f in factors]
            assert 'seal_strength' in factor_names
            assert 'continuous_momentum' in factor_names
            assert 'concept_synergy' in factor_names
            assert 'early_limit_up' in factor_names
    
    def test_factor_categories_filtering(self, mock_config):
        """测试因子类别过滤"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        # 只保留 2 个类别
        mock_config.factor_categories = ['seal_strength', 'continuous_board']
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpDataInterface'), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            factors = integration._get_predefined_limit_up_factors()
            
            # 应该只加载 2 个因子
            assert len(factors) == 2
            categories = [f['category'] for f in factors]
            assert 'seal_strength' in categories
            assert 'continuous_board' in categories
            assert 'concept_synergy' not in categories
    
    def test_all_factor_categories_loaded(self):
        """测试加载所有因子类别"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        config = Mock()
        config.rdagent_path = "/fake"
        config.checkpoint_path = None
        config.factor_categories = [
            'seal_strength',
            'continuous_board',
            'concept_synergy',
            'timing',
            'volume_pattern',
            'order_flow'
        ]
        config.prediction_targets = ['next_day_limit_up']
        
        with patch('rd_agent.limitup_integration.load_config', return_value=config), \
             patch('rd_agent.limitup_integration.LimitUpDataInterface'), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            factors = integration._get_predefined_limit_up_factors()
            
            # 应该加载所有 6 个因子
            assert len(factors) == 6
            
            expected_names = [
                'seal_strength',
                'continuous_momentum',
                'concept_synergy',
                'early_limit_up',
                'volume_explosion',
                'large_order_net'
            ]
            factor_names = [f['name'] for f in factors]
            for name in expected_names:
                assert name in factor_names


class TestFactorEvaluationP03:
    """测试 P0-3: 真实因子评估逻辑"""
    
    @pytest.mark.asyncio
    async def test_real_factor_evaluation_with_data(self, mock_config, mock_limit_up_data):
        """测试真实因子评估流程"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        # Mock 涨停股票
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        # Mock 数据接口
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            # 测试因子
            factors = [{
                'name': 'test_factor',
                'code': 'lambda df: df["seal_amount"] / df["market_cap"]',
                'category': 'test',
                'description': 'Test factor'
            }]
            
            # 执行评估
            evaluated = await integration._evaluate_factors(
                factors,
                start_date='2024-01-15',
                end_date='2024-01-15'
            )
            
            # 验证
            assert len(evaluated) >= 1
            factor = evaluated[0]
            
            # 验证性能指标
            assert 'performance' in factor
            assert 'ic' in factor['performance']
            assert 'ir' in factor['performance']
            assert 'sharpe' in factor['performance']
            assert 'sample_count' in factor['performance']
            
            # IC 应该在合理范围内
            assert -1 <= factor['performance']['ic'] <= 1
            assert factor['performance']['sample_count'] == 5  # 5只股票
    
    @pytest.mark.asyncio
    async def test_ic_calculation_correctness(self, mock_config):
        """测试 IC 计算正确性"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        # 构造完美正相关的数据
        symbols = ['A', 'B', 'C', 'D', 'E']
        features_df = pd.DataFrame({
            'factor_value': [1, 2, 3, 4, 5],
            'seal_amount': [100, 200, 300, 400, 500],
            'market_cap': [1000, 1000, 1000, 1000, 1000]
        }, index=symbols)
        
        next_day_df = pd.DataFrame({
            'next_return': [0.01, 0.02, 0.03, 0.04, 0.05],  # 完美正相关
            'next_limit_up': [0, 0, 1, 1, 1]
        }, index=symbols)
        
        limit_up_stocks = [Mock(symbol=s) for s in symbols]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'perfect_corr_factor',
                'code': 'lambda df: df["factor_value"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # IC 应该接近 1.0 (完美正相关)
            assert evaluated[0]['performance']['ic'] > 0.95
    
    @pytest.mark.asyncio
    async def test_insufficient_samples_handling(self, mock_config):
        """测试样本不足时的处理"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        # 只有 5 只股票，但有 NaN 值导致有效样本 < 10
        symbols = ['A', 'B', 'C']
        features_df = pd.DataFrame({
            'seal_amount': [100, np.nan, 300],
            'market_cap': [1000, 1000, np.nan]
        }, index=symbols)
        
        next_day_df = pd.DataFrame({
            'next_return': [0.01, 0.02, 0.03],
            'next_limit_up': [0, 1, 1]
        }, index=symbols)
        
        limit_up_stocks = [Mock(symbol=s) for s in symbols]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'sparse_factor',
                'code': 'lambda df: df["seal_amount"] / df["market_cap"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 有效样本 < 10，应该被跳过
            assert len(evaluated) == 0


class TestDynamicMetricsP06:
    """测试 P0-6: 配置驱动的动态指标计算"""
    
    @pytest.mark.asyncio
    async def test_next_day_limit_up_rate_calculation(self, mock_config, mock_limit_up_data):
        """测试次日涨停率计算"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        # P0-6: 配置包含 next_day_limit_up
        mock_config.prediction_targets = ['next_day_limit_up']
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'momentum_factor',
                'code': 'lambda df: df["seal_amount"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 验证 next_day_limit_up_rate 存在
            assert 'next_day_limit_up_rate' in evaluated[0]['performance']
            
            # 次日涨停率应该在 0-1 之间
            rate = evaluated[0]['performance']['next_day_limit_up_rate']
            assert 0 <= rate <= 1
    
    @pytest.mark.asyncio
    async def test_open_premium_calculation(self, mock_config, mock_limit_up_data):
        """测试开盘溢价计算"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        # P0-6: 配置包含 open_premium
        mock_config.prediction_targets = ['open_premium']
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'seal_factor',
                'code': 'lambda df: df["seal_amount"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 验证 open_premium 存在
            assert 'open_premium' in evaluated[0]['performance']
    
    @pytest.mark.asyncio
    async def test_continuous_probability_calculation(self, mock_config, mock_limit_up_data):
        """测试连板概率计算"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        # P0-6: 配置包含 continuous_probability
        mock_config.prediction_targets = ['continuous_probability']
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'board_factor',
                'code': 'lambda df: df["continuous_board"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 验证 continuous_probability 存在
            assert 'continuous_probability' in evaluated[0]['performance']
            
            # 连板概率应该在 0-1 之间
            prob = evaluated[0]['performance']['continuous_probability']
            assert 0 <= prob <= 1
    
    @pytest.mark.asyncio
    async def test_multiple_prediction_targets(self, mock_config, mock_limit_up_data):
        """测试多个预测目标同时计算"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        # P0-6: 配置包含所有 3 个目标
        mock_config.prediction_targets = [
            'next_day_limit_up',
            'open_premium',
            'continuous_probability'
        ]
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'multi_target_factor',
                'code': 'lambda df: df["seal_amount"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 验证所有 3 个指标都存在
            perf = evaluated[0]['performance']
            assert 'next_day_limit_up_rate' in perf
            assert 'open_premium' in perf
            assert 'continuous_probability' in perf
    
    @pytest.mark.asyncio
    async def test_missing_data_column_handling(self, mock_config, mock_limit_up_data):
        """测试缺少数据列时的处理"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        # 移除 continuous_board 列
        next_day_df_incomplete = next_day_df.drop(columns=['continuous_board'])
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df_incomplete
        
        # 配置需要 continuous_probability
        mock_config.prediction_targets = ['continuous_probability']
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'test_factor',
                'code': 'lambda df: df["seal_amount"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 应该返回 0.0 作为默认值
            assert evaluated[0]['performance']['continuous_probability'] == 0.0


class TestFactorCodeExecution:
    """测试不同类型的因子代码执行"""
    
    @pytest.mark.asyncio
    async def test_lambda_expression_factor(self, mock_config, mock_limit_up_data):
        """测试 lambda 表达式因子"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'lambda_factor',
                'code': 'lambda df: df["seal_amount"] * df["volume_ratio"]',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 应该成功执行
            assert len(evaluated) > 0
            assert 'performance' in evaluated[0]
    
    @pytest.mark.asyncio
    async def test_python_function_factor(self, mock_config, mock_limit_up_data):
        """测试 Python 函数定义因子"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'function_factor',
                'code': '''
def calculate(df):
    return df["seal_amount"] / df["market_cap"]

result = calculate(df)
''',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 应该成功执行
            assert len(evaluated) > 0
    
    @pytest.mark.asyncio
    async def test_invalid_factor_code_handling(self, mock_config, mock_limit_up_data):
        """测试无效因子代码的处理"""
        from rd_agent.limitup_integration import LimitUpRDAgentIntegration
        
        features_df, next_day_df = mock_limit_up_data
        
        limit_up_stocks = [Mock(symbol=s) for s in features_df.index]
        
        mock_data_interface = Mock()
        mock_data_interface.get_limit_up_stocks.return_value = limit_up_stocks
        mock_data_interface.get_limit_up_features.return_value = features_df
        mock_data_interface.get_next_day_result.return_value = next_day_df
        
        with patch('rd_agent.limitup_integration.load_config', return_value=mock_config), \
             patch('rd_agent.limitup_integration.LimitUpFactorLibrary'), \
             patch.object(LimitUpRDAgentIntegration, '_initialize_rdagent'):
            
            integration = LimitUpRDAgentIntegration()
            integration.data_interface = mock_data_interface
            
            factors = [{
                'name': 'invalid_factor',
                'code': 'this is not valid python code!',
                'category': 'test'
            }]
            
            evaluated = await integration._evaluate_factors(factors, '2024-01-15', '2024-01-15')
            
            # 应该跳过无效因子
            assert len(evaluated) == 0


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
