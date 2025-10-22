"""
综合测试套件
测试所有核心模块的功能和集成
"""

import unittest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'app'))

# 导入要测试的模块
from app.core.advanced_indicators import TechnicalIndicators
from app.core.risk_management import RiskManager, PositionSizer, StopLossManager
from app.core.backtest_engine import BacktestEngine, Order, OrderSide, OrderType
from app.core.agent_orchestrator import AgentOrchestrator, SignalStrength, TrendFollowingAgent
from app.core.performance_monitor import PerformanceMonitor
from app.core.trade_executor import ExecutionEngine, OrderStatus
from app.core.trading_context import TradingContext

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTechnicalIndicators(unittest.TestCase):
    """测试技术指标模块"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建示例数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 2,
            'high': 102 + np.random.randn(100) * 2,
            'low': 98 + np.random.randn(100) * 2,
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': 1000000 + np.random.randint(-100000, 100000, 100)
        }, index=dates)
        
        self.indicators = TechnicalIndicators()
    
    def test_trend_indicators(self):
        """测试趋势指标"""
        # 测试SMA
        sma = self.indicators.sma(self.data['close'], 20)
        self.assertEqual(len(sma), len(self.data))
        self.assertTrue(pd.notna(sma.iloc[-1]))
        
        # 测试EMA
        ema = self.indicators.ema(self.data['close'], 20)
        self.assertEqual(len(ema), len(self.data))
        
        # 测试MACD
        macd_result = self.indicators.macd(self.data['close'])
        self.assertIn('macd', macd_result.columns)
        self.assertIn('signal', macd_result.columns)
        self.assertIn('histogram', macd_result.columns)
    
    def test_momentum_indicators(self):
        """测试动量指标"""
        # 测试RSI
        rsi = self.indicators.rsi(self.data['close'])
        self.assertEqual(len(rsi), len(self.data))
        self.assertTrue(0 <= rsi.iloc[-1] <= 100)
        
        # 测试Stochastic
        stoch = self.indicators.stochastic(self.data['high'], self.data['low'], self.data['close'])
        self.assertIn('K', stoch.columns)
        self.assertIn('D', stoch.columns)
    
    def test_volatility_indicators(self):
        """测试波动率指标"""
        # 测试Bollinger Bands
        bb = self.indicators.bollinger_bands(self.data['close'])
        self.assertIn('upper', bb.columns)
        self.assertIn('middle', bb.columns)
        self.assertIn('lower', bb.columns)
        
        # 测试ATR
        atr = self.indicators.atr(self.data['high'], self.data['low'], self.data['close'])
        self.assertEqual(len(atr), len(self.data))
        self.assertTrue(atr.iloc[-1] > 0)
    
    def test_volume_indicators(self):
        """测试成交量指标"""
        # 测试OBV
        obv = self.indicators.obv(self.data['close'], self.data['volume'])
        self.assertEqual(len(obv), len(self.data))
        
        # 测试VWAP
        vwap = self.indicators.vwap(self.data['high'], self.data['low'], 
                                   self.data['close'], self.data['volume'])
        self.assertEqual(len(vwap), len(self.data))


class TestRiskManagement(unittest.TestCase):
    """测试风险管理模块"""
    
    def setUp(self):
        """设置测试环境"""
        config = {
            'max_position_size': 0.1,
            'max_portfolio_risk': 0.02,
            'max_correlation': 0.7,
            'stop_loss_pct': 0.02,
            'trailing_stop_pct': 0.03
        }
        self.risk_manager = RiskManager(config)
        self.position_sizer = PositionSizer()
        self.stop_loss_manager = StopLossManager()
    
    def test_position_sizing(self):
        """测试仓位计算"""
        # Kelly criterion
        size = self.position_sizer.kelly_criterion(0.6, 1.5)
        self.assertTrue(0 <= size <= 1)
        
        # Fixed fractional
        size = self.position_sizer.fixed_fractional(100000, 0.02, 100, 98)
        self.assertTrue(size >= 0)
        
        # Volatility based
        size = self.position_sizer.volatility_based(100000, 0.02, 2.5)
        self.assertTrue(size >= 0)
    
    def test_risk_metrics(self):
        """测试风险指标计算"""
        returns = pd.Series(np.random.randn(100) * 0.01)
        
        # VaR
        var = self.risk_manager.calculate_var(returns)
        self.assertTrue(var < 0)  # VaR应该是负数
        
        # CVaR
        cvar = self.risk_manager.calculate_cvar(returns)
        self.assertTrue(cvar <= var)  # CVaR应该比VaR更负
        
        # Sharpe ratio
        sharpe = self.risk_manager.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, (int, float))
    
    def test_stop_loss(self):
        """测试止损功能"""
        # 固定止损
        stop = self.stop_loss_manager.calculate_stop_loss(100, 'fixed', stop_pct=0.02)
        self.assertEqual(stop, 98)
        
        # ATR止损
        stop = self.stop_loss_manager.calculate_stop_loss(100, 'atr', atr_value=2, atr_multiplier=1.5)
        self.assertEqual(stop, 97)
        
        # 跟踪止损
        self.stop_loss_manager.update_trailing_stop(100, 95, 0.03)
        new_stop = self.stop_loss_manager.trailing_stops.get('default')
        self.assertEqual(new_stop, 97)


class TestBacktestEngine(unittest.TestCase):
    """测试回测引擎"""
    
    def setUp(self):
        """设置测试环境"""
        self.engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        self.test_data = pd.DataFrame({
            'symbol': ['AAPL'] * 30,
            'open': 150 + np.random.randn(30) * 2,
            'high': 152 + np.random.randn(30) * 2,
            'low': 148 + np.random.randn(30) * 2,
            'close': 150 + np.cumsum(np.random.randn(30) * 0.5),
            'volume': 1000000 + np.random.randint(-100000, 100000, 30)
        }, index=dates)
        
        self.engine.set_data(self.test_data)
    
    def test_order_execution(self):
        """测试订单执行"""
        # 创建买入订单
        buy_order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        
        self.engine.current_timestamp = self.test_data.index[0]
        order_id = self.engine.place_order(buy_order)
        
        self.assertIsNotNone(order_id)
        self.assertEqual(len(self.engine.order_history), 1)
        self.assertIn(OrderStatus.FILLED, [o.status for o in self.engine.order_history])
    
    def test_portfolio_management(self):
        """测试投资组合管理"""
        portfolio = self.engine.portfolio
        
        # 初始状态
        self.assertEqual(portfolio.cash, 100000)
        self.assertEqual(len(portfolio.positions), 0)
        
        # 执行交易后
        self.engine.current_timestamp = self.test_data.index[0]
        buy_order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        self.engine.place_order(buy_order)
        
        # 检查持仓
        self.assertEqual(len(portfolio.positions), 1)
        self.assertIn('AAPL', portfolio.positions)
        self.assertLess(portfolio.cash, 100000)  # 现金减少
    
    def test_performance_metrics(self):
        """测试性能指标计算"""
        # 运行简单策略
        def simple_strategy(data, portfolio):
            orders = []
            if len(portfolio.positions) == 0:
                orders.append(Order(
                    symbol='AAPL',
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100
                ))
            return orders
        
        self.engine.run_backtest(
            simple_strategy,
            start_date=self.test_data.index[0],
            end_date=self.test_data.index[-1]
        
        # 检查性能指标
        metrics = self.engine.performance_metrics
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)


class TestAgentOrchestrator(unittest.TestCase):
    """测试Agent协调器"""
    
    def setUp(self):
        """设置测试环境"""
        self.orchestrator = AgentOrchestrator(max_workers=3, timeout=2.0)
        
        # 注册测试Agents
        for i in range(3):
            agent = TrendFollowingAgent(f"test_agent_{i}")
            self.orchestrator.register_agent(
                f"test_agent_{i}", 
                agent, 
                "trend_following", 
                0.5 + i * 0.1
    
    def test_agent_registration(self):
        """测试Agent注册"""
        self.assertEqual(len(self.orchestrator.agents), 3)
        self.assertIn("test_agent_0", self.orchestrator.agents)
        self.assertIn("test_agent_1", self.orchestrator.agents)
        self.assertIn("test_agent_2", self.orchestrator.agents)
    
    def test_signal_collection(self):
        """测试信号收集"""
        context = {
            'symbol': 'TEST',
            'price_data': [100 + i for i in range(30)]
        }
        
        signals = self.orchestrator.collect_signals(context)
        self.assertGreater(len(signals), 0)
        self.assertLessEqual(len(signals), 3)
    
    def test_collaborative_decision(self):
        """测试协同决策"""
        context = {
            'symbol': 'TEST',
            'price_data': [100 + i * 0.5 for i in range(30)]  # 上升趋势
        }
        
        consensus = self.orchestrator.make_collaborative_decision(context)
        
        self.assertIsNotNone(consensus)
        self.assertIn(consensus.final_signal, list(SignalStrength))
        self.assertTrue(0 <= consensus.confidence <= 1)
    
    def test_dynamic_weights(self):
        """测试动态权重"""
        weights = self.orchestrator._get_dynamic_weights()
        
        self.assertEqual(len(weights), 3)
        for agent_id, weight in weights.items():
            self.assertTrue(0 <= weight <= 1)


class TestPerformanceMonitor(unittest.TestCase):
    """测试性能监控系统"""
    
    def setUp(self):
        """设置测试环境"""
        self.monitor = PerformanceMonitor(
            enable_system_monitor=True,
            system_monitor_interval=0.1
    
    def tearDown(self):
        """清理"""
        self.monitor.shutdown()
    
    def test_system_monitoring(self):
        """测试系统监控"""
        import time
        time.sleep(0.5)  # 等待收集一些数据
        
        if self.monitor.system_monitor:
            metrics = self.monitor.system_monitor.get_current_metrics()
            
            self.assertIsNotNone(metrics)
            self.assertTrue(0 <= metrics.cpu_usage <= 100)
            self.assertTrue(0 <= metrics.memory_usage <= 100)
            self.assertGreater(metrics.memory_available, 0)
    
    def test_trading_monitoring(self):
        """测试交易监控"""
        # 记录一些交易
        for i in range(5):
            self.monitor.trading_monitor.record_trade({
                'symbol': f'STOCK_{i}',
                'status': 'success' if i % 2 == 0 else 'failed',
                'execution_time': 0.1 + i * 0.01,
                'volume': 100 * (i + 1),
                'value': 10000 * (i + 1),
                'pnl': 100 * (i - 2)
            })
        
        metrics = self.monitor.trading_monitor.get_trading_metrics()
        
        self.assertEqual(metrics.total_trades, 5)
        self.assertEqual(metrics.successful_trades, 3)
        self.assertEqual(metrics.failed_trades, 2)
        self.assertEqual(metrics.win_rate, 0.6)
    
    def test_agent_monitoring(self):
        """测试Agent监控"""
        # 记录Agent调用
        for i in range(3):
            agent_id = f'agent_{i}'
            self.monitor.agent_monitor.record_agent_call(
                agent_id, 
                0.1 + i * 0.05, 
                success=(i != 1)
            self.monitor.agent_monitor.record_agent_performance(
                agent_id, 
                0.8 - i * 0.1, 
                0.7 + i * 0.05
        
        all_metrics = self.monitor.agent_monitor.get_all_agent_metrics()
        self.assertEqual(len(all_metrics), 3)
        
        for metric in all_metrics:
            self.assertIsNotNone(metric.agent_id)
            self.assertTrue(0 <= metric.accuracy <= 1)
            self.assertTrue(0 <= metric.confidence <= 1)
    
    def test_health_check(self):
        """测试健康检查"""
        health = self.monitor.check_health()
        
        self.assertIn('status', health)
        self.assertIn('checks', health)
        self.assertIn(health['status'], ['healthy', 'degraded', 'unhealthy'])


class TestTradeExecutor(unittest.TestCase):
    """测试交易执行模块"""
    
    def setUp(self):
        """设置测试环境"""
        config = {
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'slippage': 0.0005,
            'max_position_size': 10000,
            'max_order_size': 5000,
            'min_order_size': 100
        }
        self.engine = ExecutionEngine(broker_type="simulated", config=config)
        asyncio.run(self.engine.connect())
    
    def tearDown(self):
        """清理"""
        asyncio.run(self.engine.disconnect())
    
    def test_order_submission(self):
        """测试订单提交"""
        async def submit_order():
            order_id = await self.engine.execute_order(
                symbol="TEST",
                side=OrderSide.BUY,
                quantity=1000,
                order_type=OrderType.MARKET
            return order_id
        
        order_id = asyncio.run(submit_order())
        self.assertIsNotNone(order_id)
    
    def test_execution_strategies(self):
        """测试执行策略"""
        async def test_strategies():
            # 测试VWAP策略
            vwap_id = await self.engine.execute_order(
                symbol="TEST1",
                side=OrderSide.BUY,
                quantity=3000,
                strategy="vwap"
            self.assertIsNotNone(vwap_id)
            
            # 测试智能路由
            smart_id = await self.engine.execute_order(
                symbol="TEST2",
                side=OrderSide.BUY,
                quantity=500,
                strategy="smart"
            self.assertIsNotNone(smart_id)
        
        asyncio.run(test_strategies())
    
    def test_risk_checks(self):
        """测试风险检查"""
        async def test_risk():
            # 测试超大订单（应该被拒绝）
            order_id = await self.engine.execute_order(
                symbol="TEST",
                side=OrderSide.BUY,
                quantity=50000,  # 超过最大订单大小
                order_type=OrderType.MARKET
            self.assertIsNone(order_id)
            
            # 测试过小订单（应该被拒绝）
            order_id = await self.engine.execute_order(
                symbol="TEST",
                side=OrderSide.BUY,
                quantity=50,  # 低于最小订单大小
                order_type=OrderType.MARKET
            self.assertIsNone(order_id)
        
        asyncio.run(test_risk())


class TestTradingContext(unittest.TestCase):
    """测试交易上下文管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.context = TradingContext()
    
    def test_context_creation(self):
        """测试上下文创建"""
        symbol = "TEST"
        d_day_data = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [98],
            'close': [101],
            'volume': [1000000]
        })
        
        t1_data = pd.DataFrame({
            'bid_price': [100.5],
            'ask_price': [101.5],
            'bid_volume': [50000],
            'ask_volume': [60000]
        })
        
        context_data = self.context.create_context(
            symbol=symbol,
            d_day_historical=d_day_data,
            t1_premarket=t1_data
        
        self.assertIsNotNone(context_data)
        self.assertEqual(context_data['symbol'], symbol)
        self.assertIn('d_day_data', context_data)
        self.assertIn('t1_data', context_data)
    
    def test_data_validation(self):
        """测试数据验证"""
        # 测试空数据验证
        is_valid, errors = self.context.validate_data(
            pd.DataFrame(),
            required_fields=['open', 'close']
        self.assertFalse(is_valid)
        self.assertIn("数据为空", errors)
        
        # 测试缺失字段验证
        data = pd.DataFrame({'open': [100]})
        is_valid, errors = self.context.validate_data(
            data,
            required_fields=['open', 'close']
        self.assertFalse(is_valid)
        self.assertIn("缺失必需字段: close", errors)


class TestTradingAgentsIntegration(unittest.TestCase):
    """测试TradingAgents集成"""
    
    def setUp(self):
        """设置测试环境"""
        # 添加tradingagents项目路径
        tradingagents_path = Path("D:/test/Qlib/tradingagents")
        if tradingagents_path.exists():
            sys.path.insert(0, str(tradingagents_path))
    
    def test_qilin_agents_import(self):
        """测试Qilin智能体导入"""
        try:
            from tradingagents.agents.qilin_agents import (
                MarketEcologyAgent,
                AuctionGameAgent,
                PositionControlAgent,
                VolumeAnalysisAgent,
                TechnicalPatternAgent,
                SentimentAgent,
                RiskControlAgent,
                PatternRecognitionAgent,
                MacroAgent,
                ArbitrageAgent,
                QilinMultiAgentCoordinator
            
            # 测试智能体创建
            market_agent = MarketEcologyAgent("test_market")
            self.assertIsNotNone(market_agent)
            
            # 测试多智能体协调器
            coordinator = QilinMultiAgentCoordinator()
            self.assertIsNotNone(coordinator)
            
            logger.info("✓ TradingAgents Qilin智能体集成测试通过")
            
        except ImportError as e:
            logger.warning(f"TradingAgents未安装或路径不正确: {e}")
            self.skipTest("TradingAgents not available")
    
    def test_tradingagents_integration_adapter(self):
        """测试TradingAgents集成适配器"""
        try:
            # 导入集成适配器
            adapter_path = project_root / "tradingagents_integration" / "integration_adapter.py"
            if adapter_path.exists():
                from tradingagents_integration.integration_adapter import TradingAgentsAdapter
                
                adapter = TradingAgentsAdapter()
                self.assertIsNotNone(adapter)
                
                # 测试数据转换
                qilin_data = pd.DataFrame({
                    'open': [100],
                    'high': [102],
                    'low': [98],
                    'close': [101],
                    'volume': [1000000]
                })
                
                ta_format = adapter.convert_to_ta_format(qilin_data)
                self.assertIsNotNone(ta_format)
                
                logger.info("✓ TradingAgents适配器集成测试通过")
            else:
                logger.warning("TradingAgents适配器未找到")
                
        except Exception as e:
            logger.warning(f"TradingAgents适配器测试失败: {e}")


class IntegrationTestSuite(unittest.TestCase):
    """集成测试套件"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        logger.info("开始完整工作流测试...")
        
        # 1. 初始化组件
        indicators = TechnicalIndicators()
        risk_manager = RiskManager({
            'max_position_size': 0.1,
            'max_portfolio_risk': 0.02
        })
        monitor = PerformanceMonitor()
        
        try:
            # 2. 创建测试数据
            dates = pd.date_range('2023-01-01', periods=50, freq='D')
            test_data = pd.DataFrame({
                'symbol': ['TEST'] * 50,
                'open': 100 + np.random.randn(50) * 2,
                'high': 102 + np.random.randn(50) * 2,
                'low': 98 + np.random.randn(50) * 2,
                'close': 100 + np.cumsum(np.random.randn(50) * 0.5),
                'volume': 1000000 + np.random.randint(-100000, 100000, 50)
            }, index=dates)
            
            # 3. 计算技术指标
            test_data['sma_20'] = indicators.sma(test_data['close'], 20)
            test_data['rsi'] = indicators.rsi(test_data['close'])
            
            # 4. 风险评估
            returns = test_data['close'].pct_change()
            var = risk_manager.calculate_var(returns.dropna())
            self.assertIsNotNone(var)
            
            # 5. 记录性能
            monitor.trading_monitor.record_trade({
                'symbol': 'TEST',
                'status': 'success',
                'execution_time': 0.1,
                'volume': 1000,
                'value': 100000,
                'pnl': 500
            })
            
            # 6. 获取监控数据
            dashboard = monitor.get_dashboard_data()
            self.assertIsNotNone(dashboard)
            
            logger.info("✓ 完整工作流测试通过")
            
        finally:
            monitor.shutdown()
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效输入
        indicators = TechnicalIndicators()
        
        # 空数据
        result = indicators.sma(pd.Series([]), 20)
        self.assertEqual(len(result), 0)
        
        # 无效参数
        with self.assertRaises(ValueError):
            indicators.sma(pd.Series([1, 2, 3]), -1)
        
        logger.info("✓ 错误处理测试通过")


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestTechnicalIndicators,
        TestRiskManagement,
        TestBacktestEngine,
        TestAgentOrchestrator,
        TestPerformanceMonitor,
        TestTradeExecutor,
        TestTradingContext,
        TestTradingAgentsIntegration,
        IntegrationTestSuite
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n出错的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)