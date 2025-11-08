"""
核心模块单元测试
测试配置管理、Kelly仓位管理、市场熔断等核心功能
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from risk.kelly_position_manager import KellyPositionManager
from risk.market_circuit_breaker import MarketCircuitBreaker
from notification.notifier import Notifier, NotificationLevel


class TestConfigManager(unittest.TestCase):
    """配置管理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = ConfigManager()
    
    def test_default_config_loaded(self):
        """测试默认配置是否加载"""
        self.assertIsNotNone(self.config.config)
        self.assertIn('system', self.config.config)
        self.assertIn('screening', self.config.config)
        self.assertIn('buy', self.config.config)
    
    def test_get_config_value(self):
        """测试获取配置值"""
        # 测试嵌套键访问
        project_name = self.config.get('system.project_name')
        self.assertIsNotNone(project_name)
        
        # 测试不存在的键
        non_existent = self.config.get('non.existent.key', 'default')
        self.assertEqual(non_existent, 'default')
    
    def test_set_config_value(self):
        """测试设置配置值"""
        # 设置新值
        self.config.set('test.key', 'test_value')
        value = self.config.get('test.key')
        self.assertEqual(value, 'test_value')
        
        # 修改现有值
        original = self.config.get('buy.total_capital')
        self.config.set('buy.total_capital', 2000000)
        new_value = self.config.get('buy.total_capital')
        self.assertEqual(new_value, 2000000)
    
    def test_validate_config(self):
        """测试配置验证"""
        # 默认配置应该是有效的
        is_valid = self.config.validate()
        self.assertTrue(is_valid)
        
        # 测试无效配置
        self.config.set('screening.min_seal_strength', -1)
        is_valid = self.config.validate()
        self.assertFalse(is_valid)
        
        # 恢复有效值
        self.config.set('screening.min_seal_strength', 3.0)
    
    def test_get_section(self):
        """测试获取配置节"""
        screening_config = self.config.get_section('screening')
        self.assertIsInstance(screening_config, dict)
        self.assertIn('min_seal_strength', screening_config)
        self.assertIn('max_candidates', screening_config)


class TestKellyPositionManager(unittest.TestCase):
    """Kelly仓位管理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.manager = KellyPositionManager(total_capital=1000000)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.manager.total_capital, 1000000)
        self.assertIsNotNone(self.manager.config)
    
    def test_calculate_kelly_fraction(self):
        """测试Kelly分数计算"""
        # 高胜率高赔率
        kelly = self.manager.calculate_kelly_fraction(
            win_rate=0.7,
            reward_ratio=2.0,
            risk_ratio=1.0
        )
        self.assertGreater(kelly, 0)
        self.assertLessEqual(kelly, 1.0)
        
        # 低胜率低赔率
        kelly_low = self.manager.calculate_kelly_fraction(
            win_rate=0.4,
            reward_ratio=1.2,
            risk_ratio=1.0
        )
        self.assertLessEqual(kelly_low, kelly)
        
        # 胜率过低应该返回0
        kelly_zero = self.manager.calculate_kelly_fraction(
            win_rate=0.3,
            reward_ratio=1.0,
            risk_ratio=1.0
        )
        self.assertEqual(kelly_zero, 0)
    
    def test_calculate_positions(self):
        """测试仓位计算"""
        # 创建候选数据
        candidates = pd.DataFrame({
            'symbol': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'prediction_score': [0.8, 0.7, 0.6],
            'seal_strength': [8.0, 6.0, 4.0]
        })
        
        # 计算仓位
        positions = self.manager.calculate_positions(
            candidates=candidates,
            historical_performance={}
        )
        
        # 验证结果
        self.assertEqual(len(positions), 3)
        
        # 验证总仓位不超过1
        total_position = sum(p['position_size'] for p in positions.values())
        self.assertLessEqual(total_position, 1.0)
        
        # 验证每个仓位都有正确的字段
        for symbol, pos in positions.items():
            self.assertIn('position_size', pos)
            self.assertIn('capital', pos)
            self.assertIn('kelly_fraction', pos)
    
    def test_apply_constraints(self):
        """测试约束条件"""
        candidates = pd.DataFrame({
            'symbol': ['000001.SZ'],
            'prediction_score': [0.95],
            'seal_strength': [10.0]
        })
        
        positions = self.manager.calculate_positions(candidates, {})
        
        # 验证不超过最大仓位
        max_position = self.manager.config['max_position']
        for pos in positions.values():
            self.assertLessEqual(pos['position_size'], max_position)


class TestMarketCircuitBreaker(unittest.TestCase):
    """市场熔断机制测试"""
    
    def setUp(self):
        """测试前准备"""
        self.breaker = MarketCircuitBreaker()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.breaker.config)
        self.assertEqual(len(self.breaker.history), 0)
    
    def test_normal_market_condition(self):
        """测试正常市场条件"""
        market_data = {
            'index_changes': {'sh': 0.5, 'sz': 0.3, 'cyb': 0.8},
            'limit_up_count': 80,
            'limit_down_count': 30,
            'total_stocks': 4800,
            'avg_turnover': 2.5,
            'northbound_flow': 30,
            'daily_pnl_ratio': 0.02,
            'continuous_loss_days': 0,
            'max_drawdown': -0.05
        }
        
        signal = self.breaker.check_market_condition(market_data)
        
        # 正常市场应该允许交易
        self.assertEqual(signal.level, 0)  # NORMAL
        self.assertTrue(signal.allow_new_positions)
        self.assertEqual(signal.position_adjustment, 1.0)
    
    def test_warning_market_condition(self):
        """测试预警市场条件"""
        market_data = {
            'index_changes': {'sh': -1.5, 'sz': -1.0, 'cyb': -2.0},
            'limit_up_count': 50,
            'limit_down_count': 100,
            'total_stocks': 4800,
            'avg_turnover': 2.5,
            'northbound_flow': -20,
            'daily_pnl_ratio': -0.02,
            'continuous_loss_days': 1,
            'max_drawdown': -0.08
        }
        
        signal = self.breaker.check_market_condition(market_data)
        
        # 应该进入预警状态
        self.assertGreaterEqual(signal.level, 1)  # WARNING or higher
        self.assertLess(signal.position_adjustment, 1.0)
    
    def test_halt_market_condition(self):
        """测试停止市场条件"""
        market_data = {
            'index_changes': {'sh': -3.0, 'sz': -3.5, 'cyb': -4.0},
            'limit_up_count': 20,
            'limit_down_count': 200,
            'total_stocks': 4800,
            'avg_turnover': 5.0,
            'northbound_flow': -100,
            'daily_pnl_ratio': -0.05,
            'continuous_loss_days': 5,
            'max_drawdown': -0.15
        }
        
        signal = self.breaker.check_market_condition(market_data)
        
        # 应该停止交易
        self.assertEqual(signal.level, 4)  # HALT
        self.assertFalse(signal.allow_new_positions)
        self.assertEqual(signal.position_adjustment, 0.0)
    
    def test_history_recording(self):
        """测试历史记录"""
        market_data = {
            'index_changes': {'sh': 0.5, 'sz': 0.3, 'cyb': 0.8},
            'limit_up_count': 80,
            'limit_down_count': 30,
            'total_stocks': 4800,
            'avg_turnover': 2.5,
            'northbound_flow': 30,
            'daily_pnl_ratio': 0.02,
            'continuous_loss_days': 0,
            'max_drawdown': -0.05
        }
        
        initial_count = len(self.breaker.history)
        self.breaker.check_market_condition(market_data)
        
        # 验证历史记录增加
        self.assertEqual(len(self.breaker.history), initial_count + 1)


class TestNotifier(unittest.TestCase):
    """消息推送器测试"""
    
    def setUp(self):
        """测试前准备"""
        # 使用测试配置（不实际发送）
        test_config = {
            'enable_notification': True,
            'channels': [],
            'wechat_webhook': '',
            'dingtalk_webhook': '',
            'email_smtp_server': '',
            'email_from': '',
            'email_to': []
        }
        self.notifier = Notifier(config=test_config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.notifier.enabled)
        self.assertEqual(len(self.notifier.history), 0)
    
    def test_send_basic_message(self):
        """测试基本消息发送"""
        result = self.notifier.send(
            title="测试消息",
            content="测试内容",
            level=NotificationLevel.INFO
        )
        
        # 验证返回结果
        self.assertIn('status', result)
        self.assertIn('results', result)
        
        # 验证历史记录
        self.assertEqual(len(self.notifier.history), 1)
        self.assertEqual(self.notifier.history[0]['title'], "测试消息")
    
    def test_send_auction_signal(self):
        """测试竞价信号推送"""
        signals = [
            {'symbol': '000001.SZ', 'name': '平安银行', 'auction_strength': 0.85, 'auction_price': 12.50},
            {'symbol': '600519.SH', 'name': '贵州茅台', 'auction_strength': 0.92, 'auction_price': 1680.00}
        ]
        
        result = self.notifier.send_auction_signal(signals)
        
        # 验证结果
        self.assertIn('status', result)
        
        # 空信号应该返回特殊状态
        empty_result = self.notifier.send_auction_signal([])
        self.assertEqual(empty_result['status'], 'no_signals')
    
    def test_send_buy_notification(self):
        """测试买入通知"""
        orders = [
            {'symbol': '000001.SZ', 'price': 12.50, 'volume': 1000, 'amount': 12500}
        ]
        
        result = self.notifier.send_buy_notification(orders)
        self.assertIn('status', result)
    
    def test_send_sell_notification(self):
        """测试卖出通知"""
        orders = [
            {'symbol': '000001.SZ', 'sell_price': 13.00, 'profit': 500, 'profit_rate': 0.04}
        ]
        
        result = self.notifier.send_sell_notification(orders)
        self.assertIn('status', result)
    
    def test_send_daily_report(self):
        """测试每日报告"""
        report = {
            'date': '2024-11-01',
            'candidates': 23,
            'buy_orders': 12,
            'sell_orders': 8,
            'profit': 3240.50,
            'profit_rate': 0.0254
        }
        
        result = self.notifier.send_daily_report(report)
        self.assertIn('status', result)
    
    def test_get_history(self):
        """测试获取历史记录"""
        # 发送多条消息
        for i in range(5):
            self.notifier.send(f"测试{i}", f"内容{i}")
        
        # 获取历史
        history = self.notifier.get_history(limit=3)
        self.assertEqual(len(history), 3)
        
        # 获取全部历史
        all_history = self.notifier.get_history(limit=100)
        self.assertEqual(len(all_history), 5)


class TestWorkflowIntegration(unittest.TestCase):
    """工作流集成测试"""
    
    def test_config_workflow_integration(self):
        """测试配置与工作流集成"""
        config = ConfigManager()
        
        # 验证工作流相关配置存在
        workflow_config = config.get_section('workflow')
        self.assertIn('enable_t_day_screening', workflow_config)
        self.assertIn('enable_t1_auction_monitor', workflow_config)
        
        # 验证策略配置存在
        screening_config = config.get_section('screening')
        self.assertIn('min_seal_strength', screening_config)
    
    def test_kelly_breaker_integration(self):
        """测试Kelly与熔断器集成"""
        kelly = KellyPositionManager(total_capital=1000000)
        breaker = MarketCircuitBreaker()
        
        # 正常市场
        market_data = {
            'index_changes': {'sh': 0.5, 'sz': 0.3, 'cyb': 0.8},
            'limit_up_count': 80,
            'limit_down_count': 30,
            'total_stocks': 4800,
            'avg_turnover': 2.5,
            'northbound_flow': 30,
            'daily_pnl_ratio': 0.02,
            'continuous_loss_days': 0,
            'max_drawdown': -0.05
        }
        signal = breaker.check_market_condition(market_data)
        
        # 根据熔断信号调整Kelly仓位
        candidates = pd.DataFrame({
            'symbol': ['000001.SZ', '000002.SZ'],
            'prediction_score': [0.8, 0.7],
            'seal_strength': [8.0, 6.0]
        })
        
        positions = kelly.calculate_positions(candidates, {})
        
        # 在正常市场下应该有仓位
        self.assertGreater(len(positions), 0)
        
        # 在熔断状态下应该调整仓位
        if not signal.allow_new_positions:
            # 如果不允许新仓位，Kelly应该返回空或降低仓位
            pass


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestKellyPositionManager))
    suite.addTests(loader.loadTestsFromTestCase(TestMarketCircuitBreaker))
    suite.addTests(loader.loadTestsFromTestCase(TestNotifier))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Qilin量化交易系统 - 核心模块单元测试")
    print("="*80 + "\n")
    
    result = run_tests()
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查上方输出")
    
    print("="*80 + "\n")
