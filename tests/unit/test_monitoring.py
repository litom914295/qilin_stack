"""
监控系统单元测试
"""
import pytest
import time
from monitoring.metrics import (
    SystemMonitor,
    PerformanceTracker,
    get_monitor
)


class TestSystemMonitor:
    """测试系统监控器"""
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        monitor = SystemMonitor()
        assert monitor.start_time is not None
        assert len(monitor.metrics) > 0
    
    def test_record_signal(self):
        """测试记录信号"""
        monitor = SystemMonitor()
        
        monitor.record_signal('qlib', 'buy', 0.85)
        
        summary = monitor.get_summary()
        assert summary['total_signals'] == 1
    
    def test_record_multiple_signals(self):
        """测试记录多个信号"""
        monitor = SystemMonitor()
        
        monitor.record_signal('qlib', 'buy', 0.85)
        monitor.record_signal('trading_agents', 'sell', 0.75)
        monitor.record_signal('rd_agent', 'hold', 0.65)
        
        summary = monitor.get_summary()
        assert summary['total_signals'] == 3
    
    def test_record_decision(self):
        """测试记录决策"""
        monitor = SystemMonitor()
        
        monitor.record_decision(
            symbol='000001.SZ',
            decision='buy',
            latency=0.05,
            confidence=0.80
        )
        
        summary = monitor.get_summary()
        assert summary['total_decisions'] == 1
    
    def test_record_weight(self):
        """测试记录权重"""
        monitor = SystemMonitor()
        
        monitor.record_weight('qlib', 0.45)
        monitor.record_weight('trading_agents', 0.35)
        monitor.record_weight('rd_agent', 0.20)
        
        # 权重应该被记录
        metrics = monitor.export_metrics()
        assert 'system_weight' in metrics
    
    def test_record_market_state(self):
        """测试记录市场状态"""
        monitor = SystemMonitor()
        
        monitor.record_market_state('bull', 0.85)
        
        metrics = monitor.export_metrics()
        assert 'market_state' in metrics
    
    def test_record_error(self):
        """测试记录错误"""
        monitor = SystemMonitor()
        
        initial_summary = monitor.get_summary()
        initial_errors = initial_summary['total_errors']
        
        monitor.record_error('test_error', 'Test error message')
        
        summary = monitor.get_summary()
        assert summary['total_errors'] == initial_errors + 1
    
    def test_get_summary(self):
        """测试获取摘要"""
        monitor = SystemMonitor()
        
        # 记录一些数据
        monitor.record_signal('qlib', 'buy', 0.85)
        monitor.record_decision('000001.SZ', 'buy', 0.05, 0.80)
        
        summary = monitor.get_summary()
        
        assert 'uptime' in summary
        assert 'total_signals' in summary
        assert 'total_decisions' in summary
        assert 'total_errors' in summary
        assert 'metrics_count' in summary
        
        assert isinstance(summary['uptime'], float)
        assert summary['uptime'] > 0
    
    def test_export_metrics_prometheus(self):
        """测试Prometheus格式导出"""
        monitor = SystemMonitor()
        
        # 记录数据
        monitor.record_signal('qlib', 'buy', 0.85)
        monitor.record_decision('000001.SZ', 'buy', 0.05, 0.80)
        
        metrics = monitor.export_metrics()
        
        assert isinstance(metrics, str)
        assert '# HELP' in metrics
        assert '# TYPE' in metrics
        assert 'signal_generated_total' in metrics
        assert 'decision_made_total' in metrics
    
    def test_reset_metrics(self):
        """测试重置指标"""
        monitor = SystemMonitor()
        
        # 记录数据
        monitor.record_signal('qlib', 'buy', 0.85)
        monitor.record_decision('000001.SZ', 'buy', 0.05, 0.80)
        
        # 重置
        monitor.reset_metrics()
        
        summary = monitor.get_summary()
        assert summary['total_signals'] == 0
        assert summary['total_decisions'] == 0
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        
        assert monitor1 is monitor2


class TestPerformanceTracker:
    """测试性能追踪器"""
    
    def test_tracker_initialization(self):
        """测试追踪器初始化"""
        tracker = PerformanceTracker()
        assert tracker.monitor is not None
    
    def test_track_sync_function(self):
        """测试追踪同步函数"""
        tracker = PerformanceTracker()
        
        @tracker.track('test_function')
        def test_func():
            time.sleep(0.01)
            return 'result'
        
        result = test_func()
        assert result == 'result'
    
    @pytest.mark.asyncio
    async def test_track_async_function(self):
        """测试追踪异步函数"""
        tracker = PerformanceTracker()
        
        @tracker.track('async_test_function')
        async def async_test_func():
            await asyncio.sleep(0.01)
            return 'async_result'
        
        import asyncio
        result = await async_test_func()
        assert result == 'async_result'
    
    def test_track_function_with_error(self):
        """测试追踪带错误的函数"""
        tracker = PerformanceTracker()
        
        @tracker.track('error_function')
        def error_func():
            raise ValueError('Test error')
        
        with pytest.raises(ValueError):
            error_func()
        
        # 错误应该被记录
        summary = tracker.monitor.get_summary()
        assert summary['total_errors'] > 0
    
    def test_context_manager(self):
        """测试上下文管理器"""
        tracker = PerformanceTracker()
        
        with tracker.track_context('test_operation'):
            time.sleep(0.01)
        
        # 操作应该被追踪（通过时间验证）
        summary = tracker.monitor.get_summary()
        assert summary['uptime'] > 0


@pytest.mark.integration
class TestMonitoringIntegration:
    """监控系统集成测试"""
    
    def test_full_monitoring_cycle(self):
        """测试完整监控周期"""
        monitor = get_monitor()
        tracker = PerformanceTracker()
        
        # 1. 记录信号生成
        monitor.record_signal('qlib', 'buy', 0.85)
        monitor.record_signal('trading_agents', 'sell', 0.75)
        monitor.record_signal('rd_agent', 'hold', 0.65)
        
        # 2. 记录决策
        @tracker.track('make_decision')
        def make_decision():
            time.sleep(0.01)
            return 'buy'
        
        decision = make_decision()
        monitor.record_decision('000001.SZ', decision, 0.01, 0.80)
        
        # 3. 记录权重
        monitor.record_weight('qlib', 0.40)
        monitor.record_weight('trading_agents', 0.35)
        monitor.record_weight('rd_agent', 0.25)
        
        # 4. 记录市场状态
        monitor.record_market_state('bull', 0.85)
        
        # 5. 获取摘要
        summary = monitor.get_summary()
        
        assert summary['total_signals'] == 3
        assert summary['total_decisions'] == 1
        assert summary['total_errors'] == 0
        assert summary['uptime'] > 0
        
        # 6. 导出Prometheus指标
        metrics = monitor.export_metrics()
        
        assert 'signal_generated_total' in metrics
        assert 'decision_made_total' in metrics
        assert 'signal_confidence' in metrics
        assert 'system_weight' in metrics
        assert 'market_state' in metrics
    
    def test_monitoring_under_load(self):
        """测试高负载下的监控"""
        monitor = SystemMonitor()
        
        # 模拟大量操作
        for i in range(100):
            monitor.record_signal('qlib', 'buy', 0.85)
            monitor.record_decision(f'00000{i}.SZ', 'buy', 0.05, 0.80)
        
        summary = monitor.get_summary()
        
        assert summary['total_signals'] == 100
        assert summary['total_decisions'] == 100
        
        # 导出应该成功
        metrics = monitor.export_metrics()
        assert len(metrics) > 0
    
    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        monitor = SystemMonitor()
        tracker = PerformanceTracker()
        
        # 记录正常操作
        monitor.record_signal('qlib', 'buy', 0.85)
        
        # 模拟错误
        @tracker.track('failing_operation')
        def failing_op():
            raise RuntimeError('Simulated error')
        
        with pytest.raises(RuntimeError):
            failing_op()
        
        # 继续正常操作
        monitor.record_signal('trading_agents', 'sell', 0.75)
        
        summary = monitor.get_summary()
        
        assert summary['total_signals'] == 2
        assert summary['total_errors'] > 0
        
        # 系统应该继续工作
        monitor.record_decision('000001.SZ', 'buy', 0.05, 0.80)
        assert monitor.get_summary()['total_decisions'] == 1
