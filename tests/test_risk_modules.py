"""
风控模块单元测试
测试市场择时门控和流动性风险过滤器
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.market_timing_gate import MarketTimingGate, create_mock_market_data
from risk.liquidity_risk_filter import LiquidityRiskFilter, create_mock_stocks_data


class TestMarketTimingGate(unittest.TestCase):
    """市场择时门控测试"""
    
    def setUp(self):
        """测试前准备"""
        self.gate = MarketTimingGate(
            enable_timing=True,
            risk_threshold=0.5,
            sentiment_window=20
        )
        self.market_data = create_mock_market_data(n_days=60)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.gate)
        self.assertEqual(self.gate.risk_threshold, 0.5)
        self.assertEqual(self.gate.sentiment_window, 20)
        self.assertTrue(self.gate.enable_timing)
    
    def test_compute_market_sentiment(self):
        """测试市场情绪计算"""
        sentiment = self.gate.compute_market_sentiment(self.market_data)
        
        # 验证返回字段
        self.assertIn('limitup_avg', sentiment)
        self.assertIn('limitdown_avg', sentiment)
        self.assertIn('overall_score', sentiment)
        
        # 验证数值范围
        self.assertGreaterEqual(sentiment['overall_score'], 0)
        self.assertLessEqual(sentiment['overall_score'], 1)
        self.assertGreaterEqual(sentiment['limitup_avg'], 0)
    
    def test_assess_market_risk(self):
        """测试市场风险评估"""
        sentiment = self.gate.compute_market_sentiment(self.market_data)
        risk = self.gate.assess_market_risk(sentiment)
        
        # 验证返回字段
        self.assertIn('risk_score', risk)
        self.assertIn('risk_level', risk)
        self.assertIn('risk_desc', risk)
        
        # 验证风险等级
        self.assertIn(risk['risk_level'], ['low', 'medium', 'high'])
        
        # 验证风险分数范围
        self.assertGreaterEqual(risk['risk_score'], 0)
        self.assertLessEqual(risk['risk_score'], 1)
    
    def test_generate_timing_signal(self):
        """测试择时信号生成"""
        signal = self.gate.generate_timing_signal(self.market_data)
        
        # 验证返回字段
        self.assertIn('signal', signal)
        self.assertIn('gate_status', signal)
        self.assertIn('sentiment', signal)
        self.assertIn('risk', signal)
        
        # 验证信号类型
        self.assertIn(signal['signal'], ['bullish', 'neutral', 'caution', 'avoid'])
        self.assertIn(signal['gate_status'], ['open', 'restricted', 'closed'])
    
    def test_should_trade(self):
        """测试交易决策"""
        should_trade, reason = self.gate.should_trade(self.market_data)
        
        # 验证返回类型
        self.assertIsInstance(should_trade, bool)
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)
    
    def test_position_size_factor(self):
        """测试仓位调整因子"""
        factor = self.gate.get_position_size_factor(self.market_data)
        
        # 验证因子范围
        self.assertGreaterEqual(factor, 0)
        self.assertLessEqual(factor, 1)
    
    def test_disabled_timing(self):
        """测试禁用择时功能"""
        gate_disabled = MarketTimingGate(enable_timing=False)
        signal = gate_disabled.generate_timing_signal(self.market_data)
        
        self.assertEqual(signal['signal'], 'neutral')
        self.assertEqual(signal['gate_status'], 'open')


class TestLiquidityRiskFilter(unittest.TestCase):
    """流动性风险过滤器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.filter = LiquidityRiskFilter(
            min_volume=1e8,
            min_turnover=0.02,
            max_volatility=0.15,
            min_price=5.0,
            filter_st=True,
            filter_suspended=True
        )
        self.stocks_data = create_mock_stocks_data(n_stocks=100)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.filter)
        self.assertEqual(self.filter.min_volume, 1e8)
        self.assertEqual(self.filter.min_turnover, 0.02)
        self.assertTrue(self.filter.filter_st)
    
    def test_check_volume(self):
        """测试成交量检查"""
        stock = self.stocks_data.iloc[0]
        passed, reason = self.filter.check_volume(stock)
        
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)
    
    def test_check_turnover(self):
        """测试换手率检查"""
        stock = self.stocks_data.iloc[0]
        passed, reason = self.filter.check_turnover(stock)
        
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(reason, str)
    
    def test_check_price(self):
        """测试价格检查"""
        # 测试正常价格
        stock_normal = pd.Series({'close': 20.0})
        passed, _ = self.filter.check_price(stock_normal)
        self.assertTrue(passed)
        
        # 测试过低价格
        stock_low = pd.Series({'close': 2.0})
        passed, _ = self.filter.check_price(stock_low)
        self.assertFalse(passed)
    
    def test_check_st_status(self):
        """测试ST状态检查"""
        # 测试ST股票
        stock_st = pd.Series({'is_st': True})
        passed, reason = self.filter.check_st_status(stock_st)
        self.assertFalse(passed)
        self.assertIn('ST', reason)
        
        # 测试非ST股票
        stock_normal = pd.Series({'is_st': False})
        passed, _ = self.filter.check_st_status(stock_normal)
        self.assertTrue(passed)
    
    def test_filter_stock(self):
        """测试单只股票过滤"""
        stock = self.stocks_data.iloc[0]
        result = self.filter.filter_stock(stock)
        
        # 验证返回字段
        self.assertIn('symbol', result)
        self.assertIn('passed', result)
        self.assertIn('risk_score', result)
        self.assertIn('checks', result)
        
        # 验证风险分数范围
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 1)
    
    def test_filter_stocks_batch(self):
        """测试批量过滤"""
        results = self.filter.filter_stocks(self.stocks_data)
        
        # 验证返回DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), len(self.stocks_data))
        
        # 验证列
        self.assertIn('symbol', results.columns)
        self.assertIn('passed', results.columns)
        self.assertIn('risk_score', results.columns)
    
    def test_get_passed_stocks(self):
        """测试获取通过的股票"""
        passed = self.filter.get_passed_stocks(self.stocks_data)
        
        # 验证返回DataFrame
        self.assertIsInstance(passed, pd.DataFrame)
        self.assertLessEqual(len(passed), len(self.stocks_data))
    
    def test_get_filter_stats(self):
        """测试过滤统计"""
        stats = self.filter.get_filter_stats(self.stocks_data)
        
        # 验证统计字段
        self.assertIn('total', stats)
        self.assertIn('passed', stats)
        self.assertIn('failed', stats)
        self.assertIn('pass_rate', stats)
        
        # 验证数量一致性
        self.assertEqual(stats['total'], stats['passed'] + stats['failed'])
        
        # 验证通过率范围
        self.assertGreaterEqual(stats['pass_rate'], 0)
        self.assertLessEqual(stats['pass_rate'], 1)
    
    def test_compute_risk_score(self):
        """测试风险评分计算"""
        stock = self.stocks_data.iloc[0]
        risk_score = self.filter.compute_risk_score(stock)
        
        # 验证范围
        self.assertGreaterEqual(risk_score, 0)
        self.assertLessEqual(risk_score, 1)
        self.assertIsInstance(risk_score, float)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_combined_workflow(self):
        """测试组合工作流"""
        # 创建组件
        gate = MarketTimingGate(enable_timing=True, risk_threshold=0.5)
        filter_engine = LiquidityRiskFilter(min_volume=5e7, filter_st=True)
        
        # 准备数据
        market_data = create_mock_market_data(n_days=60)
        stocks_data = create_mock_stocks_data(n_stocks=50)
        
        # 1. 检查市场条件
        should_trade, reason, signal = gate.should_trade(market_data), *gate.generate_timing_signal(market_data).values()
        
        if should_trade[0]:  # 如果市场允许交易
            # 2. 过滤股票
            passed_stocks = filter_engine.get_passed_stocks(stocks_data)
            
            # 验证结果
            self.assertIsInstance(passed_stocks, pd.DataFrame)
            self.assertLessEqual(len(passed_stocks), len(stocks_data))
            
            # 3. 应用仓位调整
            position_factor = gate.get_position_size_factor(market_data)
            self.assertGreaterEqual(position_factor, 0)
            self.assertLessEqual(position_factor, 1)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
