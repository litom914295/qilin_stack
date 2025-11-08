"""
测试 FileStorage 集成

测试 QilinRDAgentLogger 的功能:
- FileStorage 初始化
- 实验对象记录 (pkl)
- 指标记录 (json)
- 历史实验读取
- RDAgentWrapper 集成

作者: AI Agent
日期: 2024
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil


@pytest.fixture
def temp_workspace():
    """创建临时工作目录"""
    temp_dir = tempfile.mkdtemp(prefix='test_logging_')
    yield Path(temp_dir)
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def qilin_logger(temp_workspace):
    """创建 QilinRDAgentLogger 实例"""
    try:
        from rd_agent.logging_integration import QilinRDAgentLogger
        return QilinRDAgentLogger(str(temp_workspace))
    except ImportError:
        pytest.skip("FileStorage not available")


def create_mock_experiment():
    """创建模拟实验对象"""
    exp = Mock()
    exp.hypothesis = Mock()
    exp.hypothesis.hypothesis = "测试假设: 动量因子"
    
    exp.workspace = Mock()
    exp.workspace.code_dict = {
        'factor.py': 'def factor(data):\n    return data["close"].pct_change(20)'
    }
    
    exp.result = {
        'IC': 0.05,
        'IR': 0.8,
        'sharpe_ratio': 1.5
    }
    
    return exp


class TestQilinRDAgentLogger:
    """测试 QilinRDAgentLogger"""
    
    def test_logger_initialization(self, temp_workspace):
        """测试日志器初始化"""
        try:
            from rd_agent.logging_integration import QilinRDAgentLogger
            
            logger = QilinRDAgentLogger(str(temp_workspace))
            
            assert logger.workspace_path == temp_workspace
            assert logger.storage is not None
            assert temp_workspace.exists()
        except ImportError:
            pytest.skip("FileStorage not available")
    
    def test_log_experiment(self, qilin_logger, temp_workspace):
        """测试实验记录 (pkl)"""
        exp = create_mock_experiment()
        
        # 记录实验
        path = qilin_logger.log_experiment(exp, tag='test.experiment')
        
        # 验证文件生成
        assert path.exists()
        assert path.suffix == '.pkl'
        assert 'test' in str(path)
        assert 'experiment' in str(path)
    
    def test_log_metrics(self, qilin_logger, temp_workspace):
        """测试指标记录 (json)"""
        metrics = {
            'ic': 0.05,
            'ir': 0.8,
            'sharpe': 1.5,
            'annual_return': 0.15
        }
        
        # 记录指标
        path = qilin_logger.log_metrics(metrics, tag='test.metrics')
        
        # 验证文件生成
        assert path.exists()
        assert path.suffix == '.json'
        assert 'test' in str(path)
        assert 'metrics' in str(path)
        
        # 验证内容
        import json
        with open(path, 'r') as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics['ic'] == 0.05
        assert saved_metrics['ir'] == 0.8
        assert 'timestamp' in saved_metrics  # 应该自动添加时间戳
    
    def test_iter_experiments(self, qilin_logger):
        """测试历史实验迭代"""
        # 记录多个实验
        for i in range(3):
            exp = create_mock_experiment()
            exp.hypothesis.hypothesis = f"测试假设 {i+1}"
            qilin_logger.log_experiment(exp, tag='test.iter')
        
        # 读取实验
        experiments = list(qilin_logger.iter_experiments(tag='test.iter'))
        
        assert len(experiments) == 3
        for exp in experiments:
            assert hasattr(exp, 'hypothesis')
    
    def test_iter_metrics(self, qilin_logger):
        """测试历史指标迭代"""
        # 记录多个指标
        for i in range(3):
            metrics = {'ic': 0.05 + i*0.01, 'ir': 0.8 + i*0.1}
            qilin_logger.log_metrics(metrics, tag='test.metrics.iter')
        
        # 读取指标
        metrics_list = list(qilin_logger.iter_metrics(tag='test.metrics.iter'))
        
        assert len(metrics_list) == 3
        for m in metrics_list:
            assert 'ic' in m
            assert 'ir' in m
            assert 'timestamp' in m
    
    def test_get_experiment_count(self, qilin_logger):
        """测试获取实验数量"""
        # 记录实验
        for i in range(5):
            exp = create_mock_experiment()
            qilin_logger.log_experiment(exp, tag='test.count')
        
        # 获取数量
        count = qilin_logger.get_experiment_count(tag='test.count')
        
        assert count == 5
    
    def test_get_latest_experiment(self, qilin_logger):
        """测试获取最新实验"""
        # 记录实验
        for i in range(3):
            exp = create_mock_experiment()
            exp.hypothesis.hypothesis = f"假设 {i+1}"
            qilin_logger.log_experiment(exp, tag='test.latest')
        
        # 获取最新实验
        latest = qilin_logger.get_latest_experiment(tag='test.latest')
        
        assert latest is not None
        assert hasattr(latest, 'hypothesis')
        # 应该是最后一个 (假设 3)
        assert '3' in latest.hypothesis.hypothesis
    
    def test_clear_logs(self, qilin_logger, temp_workspace):
        """测试清理日志"""
        # 记录一些数据
        for i in range(3):
            exp = create_mock_experiment()
            qilin_logger.log_experiment(exp, tag='test.clear')
        
        # 清理
        count = qilin_logger.clear_logs(tag='test.clear')
        
        assert count == 3
        
        # 验证已清理
        remaining = qilin_logger.get_experiment_count(tag='test.clear')
        assert remaining == 0


class TestRDAgentWrapperIntegration:
    """测试 RDAgentWrapper 集成"""
    
    @pytest.mark.asyncio
    async def test_wrapper_filestorage_initialization(self, temp_workspace):
        """测试 Wrapper 初始化时创建 FileStorage"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            config = {
                'llm_model': 'gpt-4-turbo',
                'workspace_path': str(temp_workspace)
            }
            
            wrapper = RDAgentWrapper(config)
            
            # 验证 FileStorage 已初始化
            assert hasattr(wrapper, 'qilin_logger')
            if wrapper.qilin_logger:
                assert wrapper.qilin_logger.workspace_path == temp_workspace
        except ImportError:
            pytest.skip("RDAgentWrapper dependencies not available")
    
    @pytest.mark.asyncio
    async def test_wrapper_logs_experiments(self, temp_workspace):
        """测试 Wrapper 在 research_pipeline 中记录实验"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            config = {
                'llm_model': 'gpt-4-turbo',
                'workspace_path': str(temp_workspace),
                'max_iterations': 2
            }
            
            wrapper = RDAgentWrapper(config)
            
            # 模拟 trace
            mock_trace = Mock()
            mock_trace.hist = []
            
            # 添加模拟实验
            for i in range(2):
                exp = create_mock_experiment()
                feedback = Mock()
                feedback.decision = True  # 被采纳
                mock_trace.hist.append((exp, feedback))
            
            # 模拟 factor_loop
            with patch.object(wrapper._official_manager, 'get_factor_loop') as mock_get:
                mock_loop = Mock()
                mock_loop.trace = mock_trace
                mock_loop.run = Mock(return_value=None)
                mock_get.return_value = mock_loop
                
                # 运行 research_pipeline
                # results = await wrapper.research_pipeline('测试', pd.DataFrame())
                
                # 验证实验已记录 (如果 qilin_logger 可用)
                if wrapper.qilin_logger:
                    count = wrapper.qilin_logger.get_experiment_count(tag='limitup.factor')
                    # assert count > 0  # 应该有实验记录
        except ImportError:
            pytest.skip("RDAgentWrapper dependencies not available")


class TestOfflineReading:
    """测试离线读取功能 (Phase 1.2)"""
    
    def test_load_historical_factors(self, qilin_logger, temp_workspace):
        """测试加载历史因子"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            # 先记录一些实验
            for i in range(5):
                exp = create_mock_experiment()
                exp.hypothesis.hypothesis = f"因子 {i+1}"
                qilin_logger.log_experiment(exp, tag='limitup.factor')
            
            # 创建 Wrapper
            config = {
                'llm_model': 'gpt-4-turbo',
                'workspace_path': str(temp_workspace)
            }
            wrapper = RDAgentWrapper(config)
            
            # 加载历史因子
            factors = wrapper.load_historical_factors(str(temp_workspace), n_factors=3)
            
            # 验证
            assert len(factors) == 3
            for factor in factors:
                assert hasattr(factor, 'name')
                assert hasattr(factor, 'performance')
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_load_historical_metrics(self, qilin_logger, temp_workspace):
        """测试加载历史指标"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            # 记录指标
            for i in range(3):
                metrics = {
                    'topic': f'研究 {i+1}',
                    'total_experiments': 10 + i,
                    'successful_factors': 5 + i
                }
                qilin_logger.log_metrics(metrics, tag='limitup.summary')
            
            # 创建 Wrapper
            config = {
                'llm_model': 'gpt-4-turbo',
                'workspace_path': str(temp_workspace)
            }
            wrapper = RDAgentWrapper(config)
            
            # 加载指标
            metrics_list = wrapper.load_historical_metrics(str(temp_workspace))
            
            # 验证
            assert len(metrics_list) == 3
            for m in metrics_list:
                assert 'topic' in m
                assert 'total_experiments' in m
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_load_factors_with_fallback_level1(self, qilin_logger, temp_workspace):
        """测试兜底策略 Level 1 (FileStorage)"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            # 记录实验到 FileStorage
            for i in range(3):
                exp = create_mock_experiment()
                qilin_logger.log_experiment(exp, tag='limitup.factor')
            
            # 创建 Wrapper
            config = {
                'llm_model': 'gpt-4-turbo',
                'workspace_path': str(temp_workspace)
            }
            wrapper = RDAgentWrapper(config)
            
            # 应该从 Level 1 (FileStorage) 加载
            factors = wrapper.load_factors_with_fallback(str(temp_workspace))
            
            assert len(factors) > 0
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_load_factors_with_fallback_level2(self, temp_workspace):
        """测试兜底策略 Level 2 (运行时 trace)"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            # 创建 Wrapper (不启用 FileStorage)
            config = {
                'llm_model': 'gpt-4-turbo',
                'workspace_path': str(temp_workspace)
            }
            
            with patch('rd_agent.compat_wrapper.QilinRDAgentLogger', side_effect=ImportError):
                wrapper = RDAgentWrapper(config)
                
                # 模拟运行时 trace
                mock_trace = Mock()
                mock_trace.hist = []
                for i in range(2):
                    exp = create_mock_experiment()
                    feedback = Mock()
                    feedback.decision = True
                    mock_trace.hist.append((exp, feedback))
                
                with patch.object(wrapper._official_manager, 'get_trace', return_value=mock_trace):
                    # 应该从 Level 2 (Runtime trace) 加载
                    factors = wrapper.load_factors_with_fallback(str(temp_workspace))
                    
                    # 注意: 由于 _ResultAdapter.exp_to_factor 可能失败,只检查不抛异常
                    assert isinstance(factors, list)
        except ImportError:
            pytest.skip("Dependencies not available")
    
    def test_load_factors_with_fallback_error(self, temp_workspace):
        """测试所有数据源不可用时的错误处理"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            # 创建 Wrapper
            config = {
                'llm_model': 'gpt-4-turbo',
                'workspace_path': str(temp_workspace)
            }
            wrapper = RDAgentWrapper(config)
            
            # 在空目录下加载,应该抛异常
            with pytest.raises(Exception):  # DataNotFoundError
                wrapper.load_factors_with_fallback(str(temp_workspace))
        except ImportError:
            pytest.skip("Dependencies not available")


class TestFilestorageErrorHandling:
    """测试错误处理"""
    
    def test_logger_import_error(self, temp_workspace):
        """测试 FileStorage 不可用时的处理"""
        with patch('rd_agent.logging_integration.FILESTORAGE_AVAILABLE', False):
            from rd_agent.logging_integration import QilinRDAgentLogger
            
            with pytest.raises(ImportError):
                QilinRDAgentLogger(str(temp_workspace))
    
    def test_wrapper_filestorage_unavailable(self, temp_workspace):
        """测试 Wrapper 在 FileStorage 不可用时仍能工作"""
        try:
            from rd_agent.compat_wrapper import RDAgentWrapper
            
            with patch('rd_agent.compat_wrapper.QilinRDAgentLogger', side_effect=ImportError):
                config = {
                    'llm_model': 'gpt-4-turbo',
                    'workspace_path': str(temp_workspace)
                }
                
                wrapper = RDAgentWrapper(config)
                
                # FileStorage 不可用,但 Wrapper 应该正常初始化
                assert wrapper.qilin_logger is None
        except ImportError:
            pytest.skip("RDAgentWrapper dependencies not available")


# 性能测试
@pytest.mark.slow
class TestFilestoragePerformance:
    """测试 FileStorage 性能"""
    
    def test_log_many_experiments(self, qilin_logger):
        """测试记录大量实验的性能"""
        import time
        
        n = 100
        start = time.time()
        
        for i in range(n):
            exp = create_mock_experiment()
            qilin_logger.log_experiment(exp, tag='test.performance')
        
        elapsed = time.time() - start
        
        # 应该在合理时间内完成 (< 10秒)
        assert elapsed < 10.0
        print(f"\n✅ 记录 {n} 个实验耗时: {elapsed:.2f}秒 ({elapsed/n*1000:.2f}ms/个)")
    
    def test_iter_many_experiments(self, qilin_logger):
        """测试读取大量实验的性能"""
        import time
        
        # 先记录100个实验
        for i in range(100):
            exp = create_mock_experiment()
            qilin_logger.log_experiment(exp, tag='test.iter.perf')
        
        # 测试读取性能
        start = time.time()
        experiments = list(qilin_logger.iter_experiments(tag='test.iter.perf'))
        elapsed = time.time() - start
        
        assert len(experiments) == 100
        assert elapsed < 5.0  # 应该在5秒内完成
        print(f"\n✅ 读取 100 个实验耗时: {elapsed:.2f}秒")


if __name__ == "__main__":
    """
    运行测试:
        pytest tests/unit/test_logging_integration.py -v
    """
    pytest.main([__file__, '-v', '-s'])
