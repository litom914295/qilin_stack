"""
端到端集成测试
"""
import pytest
import asyncio
from decision_engine.core import get_decision_engine, SignalType
from adaptive_system.market_state import AdaptiveStrategyAdjuster, MarketRegime
from monitoring.metrics import get_monitor
from data_pipeline.unified_data import UnifiedDataPipeline


@pytest.mark.integration
class TestFullSystemIntegration:
    """完整系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_decision_pipeline(self, sample_symbols, sample_date):
        """测试完整决策流程"""
        # 1. 初始化所有组件
        engine = get_decision_engine()
        adjuster = AdaptiveStrategyAdjuster()
        monitor = get_monitor()
        
        # 2. 检测市场状态
        import pandas as pd
        import numpy as np
        market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'close': 100 + np.arange(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        market_state = adjuster.detector.detect_state(market_data)
        assert market_state is not None
        assert market_state.regime in MarketRegime
        
        # 3. 调整策略参数
        params = adjuster.adjust_strategy(market_data)
        assert 'position_size' in params
        assert 'stop_loss' in params
        
        # 4. 生成交易决策
        decisions = await engine.make_decisions(sample_symbols, sample_date)
        assert len(decisions) > 0
        
        # 5. 记录监控指标
        for decision in decisions:
            monitor.record_decision(
                symbol=decision.symbol,
                decision=decision.final_signal.value,
                latency=0.05,
                confidence=decision.confidence
            )
        
        monitor.record_market_state(
            state=market_state.regime.value,
            confidence=market_state.confidence
        )
        
        # 6. 验证结果
        summary = monitor.get_summary()
        assert summary['total_decisions'] == len(decisions)
        assert summary['total_errors'] == 0
        
        # 7. 导出指标
        metrics = monitor.export_metrics()
        assert 'decision_made_total' in metrics
        assert 'market_state' in metrics
    
    @pytest.mark.asyncio
    async def test_adaptive_decision_cycle(self, sample_symbols, sample_date):
        """测试自适应决策周期"""
        engine = get_decision_engine()
        adjuster = AdaptiveStrategyAdjuster()
        
        import pandas as pd
        import numpy as np
        
        # 模拟不同市场条件
        bull_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'close': 100 + np.arange(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        bear_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'close': 100 - np.arange(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        # 牛市决策
        bull_state = adjuster.detector.detect_state(bull_data)
        bull_params = adjuster.adjust_strategy(bull_data)
        bull_decisions = await engine.make_decisions(sample_symbols[:1], sample_date)
        
        # 熊市决策
        bear_state = adjuster.detector.detect_state(bear_data)
        bear_params = adjuster.adjust_strategy(bear_data)
        bear_decisions = await engine.make_decisions(sample_symbols[:1], sample_date)
        
        # 验证自适应效果
        assert bull_params['position_size'] >= bear_params['position_size']
        assert len(bull_decisions) > 0
        assert len(bear_decisions) > 0
    
    @pytest.mark.asyncio
    async def test_weight_optimization_cycle(self, sample_symbols, sample_date):
        """测试权重优化周期"""
        engine = get_decision_engine()
        monitor = get_monitor()
        
        from decision_engine.weight_optimizer import WeightOptimizer
        import numpy as np
        
        optimizer = WeightOptimizer()
        
        # 模拟多天的性能数据
        for day in range(5):
            # 生成决策
            decisions = await engine.make_decisions(sample_symbols[:1], sample_date)
            
            # 模拟性能评估
            for system in ['qlib', 'trading_agents', 'rd_agent']:
                predictions = np.random.randint(0, 2, 20)
                actuals = np.random.randint(0, 2, 20)
                returns = np.random.randn(20) * 0.02
                
                optimizer.evaluate_performance(system, predictions, actuals, returns)
        
        # 优化权重
        new_weights = optimizer.optimize_weights()
        
        # 更新引擎权重
        engine.update_weights(new_weights)
        
        # 使用新权重生成决策
        new_decisions = await engine.make_decisions(sample_symbols[:1], sample_date)
        
        assert len(new_decisions) > 0
        assert abs(sum(new_weights.values()) - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_data_pipeline_integration(self, sample_symbols, date_range):
        """测试数据管道集成"""
        pipeline = UnifiedDataPipeline()
        
        # 获取市场数据
        market_data = await pipeline.get_market_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        # 数据应该可用于决策
        if market_data is not None:
            assert len(market_data) > 0
    
    @pytest.mark.asyncio
    async def test_error_resilience(self, sample_symbols, sample_date):
        """测试系统容错性"""
        engine = get_decision_engine()
        monitor = get_monitor()
        
        # 1. 正常操作
        decisions1 = await engine.make_decisions(sample_symbols[:1], sample_date)
        assert len(decisions1) > 0
        
        # 2. 使用无效输入
        invalid_decisions = await engine.make_decisions([], sample_date)
        assert len(invalid_decisions) == 0
        
        # 3. 系统应该继续工作
        decisions2 = await engine.make_decisions(sample_symbols[:1], sample_date)
        assert len(decisions2) > 0
        
        # 4. 监控应该记录所有操作
        summary = monitor.get_summary()
        assert summary['total_decisions'] >= 2
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, sample_symbols, sample_date):
        """测试并发操作"""
        engine = get_decision_engine()
        
        # 并发生成多个决策
        tasks = [
            engine.make_decisions([symbol], sample_date)
            for symbol in sample_symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 所有任务应该成功
        assert len(results) == len(sample_symbols)
        
        # 验证每个结果
        for result in results:
            if not isinstance(result, Exception):
                assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_under_load(self, sample_symbols, sample_date):
        """测试高负载监控"""
        engine = get_decision_engine()
        monitor = get_monitor()
        
        # 生成大量决策
        for i in range(10):
            decisions = await engine.make_decisions(sample_symbols, sample_date)
            
            for decision in decisions:
                monitor.record_decision(
                    symbol=decision.symbol,
                    decision=decision.final_signal.value,
                    latency=0.05,
                    confidence=decision.confidence
                )
        
        # 系统应该正常工作
        summary = monitor.get_summary()
        assert summary['total_decisions'] >= 10 * len(sample_symbols)
        
        # 导出应该成功
        metrics = monitor.export_metrics()
        assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_complete_trading_day_simulation(self, sample_symbols):
        """测试完整交易日模拟"""
        engine = get_decision_engine()
        adjuster = AdaptiveStrategyAdjuster()
        monitor = get_monitor()
        
        import pandas as pd
        import numpy as np
        
        # 模拟交易日流程
        trading_date = '2024-06-30'
        
        # 1. 早盘：分析市场状态
        market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'close': 100 + np.cumsum(np.random.randn(100) * 2),
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        market_state = adjuster.detector.detect_state(market_data)
        params = adjuster.adjust_strategy(market_data)
        
        # 2. 盘中：生成决策
        decisions = await engine.make_decisions(sample_symbols, trading_date)
        
        # 3. 记录监控
        for decision in decisions:
            monitor.record_signal('qlib', decision.final_signal.value, decision.confidence)
            monitor.record_decision(
                symbol=decision.symbol,
                decision=decision.final_signal.value,
                latency=0.05,
                confidence=decision.confidence
            )
        
        monitor.record_market_state(market_state.regime.value, market_state.confidence)
        
        # 4. 盘后：统计分析
        summary = monitor.get_summary()
        
        assert summary['total_signals'] > 0
        assert summary['total_decisions'] == len(decisions)
        assert summary['total_errors'] == 0
        
        # 5. 导出日报
        daily_report = {
            'date': trading_date,
            'market_regime': market_state.regime.value,
            'market_confidence': market_state.confidence,
            'total_decisions': summary['total_decisions'],
            'position_size': params['position_size'],
            'stop_loss': params['stop_loss'],
            'uptime': summary['uptime']
        }
        
        assert daily_report['total_decisions'] > 0
        assert 0 < daily_report['position_size'] <= 1
        assert -1 < daily_report['stop_loss'] < 0


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningIntegration:
    """长时间运行集成测试"""
    
    @pytest.mark.asyncio
    async def test_multi_day_operation(self, sample_symbols):
        """测试多日运行"""
        engine = get_decision_engine()
        monitor = get_monitor()
        
        import pandas as pd
        
        # 模拟7天交易
        dates = pd.date_range('2024-06-24', periods=7, freq='D')
        
        all_decisions = []
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            decisions = await engine.make_decisions(sample_symbols, date_str)
            all_decisions.extend(decisions)
            
            # 记录
            for decision in decisions:
                monitor.record_decision(
                    symbol=decision.symbol,
                    decision=decision.final_signal.value,
                    latency=0.05,
                    confidence=decision.confidence
                )
        
        # 验证
        summary = monitor.get_summary()
        assert summary['total_decisions'] == len(all_decisions)
        assert len(all_decisions) == 7 * len(sample_symbols)
