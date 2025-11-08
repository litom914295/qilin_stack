"""
测试 compat_wrapper.py 中的 P0-4 健壮性增强

测试覆盖:
1. workspace 提取的多路径尝试 (sub_workspace_list, workspace, sub_workspace)
2. 多文件名候选 (factor.py, code.py, main.py, implementation.py)
3. 多指标键名处理 (IC/ic/information_coefficient, IR/ir, etc.)
4. dict 和 DataFrame 结果格式兼容性
5. 错误处理和日志
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from rd_agent.compat_wrapper import (
    _ResultAdapter,
    _ConfigAdapter,
    ResultConversionError
)
from rd_agent.research_agent import FactorDefinition


class TestResultAdapterWorkspaceExtraction:
    """测试 workspace 提取的多路径尝试"""
    
    def test_workspace_from_sub_workspace_list(self):
        """测试从 sub_workspace_list[0] 提取 workspace"""
        # 创建 mock experiment
        exp = Mock()
        exp.sub_workspace_list = [
            Mock(file_dict={'factor.py': 'code_content'})
        ]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test hypothesis")
        
        # 执行转换
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # 验证
        assert factor.expression == 'code_content'
        assert 'factor.py' in factor.parameters['code_file']
    
    def test_workspace_from_workspace_attribute(self):
        """测试从 workspace 属性提取"""
        exp = Mock()
        exp.sub_workspace_list = []
        exp.workspace = Mock(file_dict={'code.py': 'workspace_code'})
        exp.result = pd.DataFrame({'ic': [0.03]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.expression == 'workspace_code'
        assert 'code.py' in factor.parameters['code_file']
    
    def test_workspace_from_sub_workspace_singular(self):
        """测试从 sub_workspace 单数形式提取"""
        exp = Mock()
        # 不设置 sub_workspace_list 和 workspace
        delattr(exp, 'sub_workspace_list') if hasattr(exp, 'sub_workspace_list') else None
        delattr(exp, 'workspace') if hasattr(exp, 'workspace') else None
        
        exp.sub_workspace = Mock(file_dict={'main.py': 'sub_workspace_code'})
        exp.result = pd.DataFrame({'IC': [0.04]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.expression == 'sub_workspace_code'
        assert 'main.py' in factor.parameters['code_file']
    
    def test_no_workspace_raises_error(self):
        """测试没有任何 workspace 时抛出错误"""
        exp = Mock(spec=['result', 'hypothesis'])  # 只有这两个属性
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        with pytest.raises(ResultConversionError, match="No workspace found"):
            _ResultAdapter.exp_to_factor(exp)


class TestResultAdapterFileNameCandidates:
    """测试多文件名候选"""
    
    @pytest.mark.parametrize("filename", [
        'factor.py',
        'code.py',
        'main.py',
        'implementation.py',
        'factor_code.py'
    ])
    def test_file_candidates_priority(self, filename):
        """测试文件名候选优先级"""
        exp = Mock()
        exp.sub_workspace_list = [
            Mock(file_dict={filename: f'code_from_{filename}'})
        ]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.expression == f'code_from_{filename}'
        assert filename in factor.parameters['code_file']
    
    def test_multiple_files_uses_first_candidate(self):
        """测试多个候选文件时使用第一个"""
        exp = Mock()
        exp.sub_workspace_list = [
            Mock(file_dict={
                'implementation.py': 'impl_code',
                'factor.py': 'factor_code',
                'code.py': 'code_code'
            })
        ]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # factor.py 优先级最高 (在候选列表中排第一)
        assert factor.expression == 'factor_code'
        assert 'factor.py' in factor.parameters['code_file']
    
    def test_fallback_to_first_py_file(self):
        """测试回退到第一个 .py 文件"""
        exp = Mock()
        exp.sub_workspace_list = [
            Mock(file_dict={
                'custom_file.py': 'custom_code',
                'another.py': 'another_code',
                'readme.txt': 'not python'
            })
        ]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # 应该使用第一个 .py 文件
        assert factor.expression in ['custom_code', 'another_code']
        assert factor.parameters['code_file'].endswith('.py')
    
    def test_no_python_files_uses_placeholder(self):
        """测试没有 Python 文件时使用占位符"""
        exp = Mock()
        exp.sub_workspace_list = [
            Mock(file_dict={
                'readme.txt': 'readme content',
                'data.csv': 'csv data'
            })
        ]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # 应该使用占位符
        assert factor.expression == '# Factor code not available'
        assert factor.parameters['code_file'] == 'unknown.py'


class TestResultAdapterMetricsExtraction:
    """测试多指标键名处理"""
    
    @pytest.mark.parametrize("ic_key,ic_value", [
        ('IC', 0.05),
        ('ic', 0.04),
        ('information_coefficient', 0.06),
        ('IC_mean', 0.03),
        ('ic_mean', 0.07)
    ])
    def test_ic_extraction_with_various_keys(self, ic_key, ic_value):
        """测试 IC 提取支持多种键名"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = pd.DataFrame({ic_key: [ic_value]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.performance['ic'] == pytest.approx(ic_value)
    
    @pytest.mark.parametrize("ir_key,ir_value", [
        ('IR', 0.8),
        ('ir', 0.7),
        ('information_ratio', 0.9),
        ('IC_IR', 0.6),
        ('ic_ir', 1.0)
    ])
    def test_ir_extraction_with_various_keys(self, ir_key, ir_value):
        """测试 IR 提取支持多种键名"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = pd.DataFrame({
            'IC': [0.05],
            ir_key: [ir_value]
        })
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.performance['ir'] == pytest.approx(ir_value)
    
    @pytest.mark.parametrize("return_key,return_value", [
        ('1day.excess_return_with_cost.annualized_return', 0.15),
        ('annualized_return', 0.12),
        ('annual_return', 0.18),
        ('excess_return_with_cost.annualized_return', 0.14)
    ])
    def test_annual_return_extraction(self, return_key, return_value):
        """测试年化收益提取"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = pd.DataFrame({
            'IC': [0.05],
            return_key: [return_value]
        })
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.performance['annual_return'] == pytest.approx(return_value)
    
    @pytest.mark.parametrize("dd_key,dd_value", [
        ('1day.excess_return_with_cost.max_drawdown', -0.12),
        ('max_drawdown', -0.10),
        ('maximum_drawdown', -0.15),
        ('excess_return_with_cost.max_drawdown', -0.08)
    ])
    def test_max_drawdown_extraction(self, dd_key, dd_value):
        """测试最大回撤提取"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = pd.DataFrame({
            'IC': [0.05],
            dd_key: [dd_value]
        })
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.performance['max_drawdown'] == pytest.approx(dd_value)
    
    def test_missing_metrics_do_not_raise_error(self):
        """测试缺少某些指标不会导致错误"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        # 只提供 IC，缺少 IR, annual_return, max_drawdown
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # IC 应该存在
        assert factor.performance['ic'] == pytest.approx(0.05)
        # 其他指标应该缺失或为0
        assert 'ir' not in factor.performance or factor.performance['ir'] == 0
        assert 'annual_return' not in factor.performance
        assert 'max_drawdown' not in factor.performance


class TestResultAdapterResultFormats:
    """测试 dict 和 DataFrame 结果格式兼容性"""
    
    def test_dataframe_result_format(self):
        """测试 DataFrame 格式的结果"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = pd.DataFrame({
            'IC': [0.05],
            'IR': [0.8],
            'annualized_return': [0.15]
        })
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.performance['ic'] == pytest.approx(0.05)
        assert factor.performance['ir'] == pytest.approx(0.8)
        assert factor.performance['annual_return'] == pytest.approx(0.15)
    
    def test_dict_result_format(self):
        """测试 dict 格式的结果"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = {
            'IC': 0.06,
            'IR': 0.9,
            'ic': 0.05  # 小写应该也能识别
        }
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # 应该优先使用大写 IC
        assert factor.performance['ic'] == pytest.approx(0.06)
        assert factor.performance['ir'] == pytest.approx(0.9)
    
    def test_dict_with_lowercase_keys(self):
        """测试 dict 中只有小写键名"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = {
            'ic': 0.04,
            'ir': 0.7
        }
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.performance['ic'] == pytest.approx(0.04)
        assert factor.performance['ir'] == pytest.approx(0.7)
    
    def test_empty_result_dict(self):
        """测试空结果 dict"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = {}
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # 应该不抛出错误，但 performance 为空或有默认值
        assert isinstance(factor.performance, dict)


class TestResultAdapterEdgeCases:
    """测试边缘情况和错误处理"""
    
    def test_workspace_as_dict(self):
        """测试 workspace 为 dict 类型"""
        exp = Mock()
        exp.sub_workspace_list = [
            {
                'file_dict': {'factor.py': 'dict_code'},
                'version': 'v1.0'
            }
        ]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.expression == 'dict_code'
        assert factor.parameters['version'] == 'v1.0'
    
    def test_files_instead_of_file_dict(self):
        """测试 workspace.files 而非 workspace.file_dict"""
        workspace = Mock()
        workspace.files = {'factor.py': 'files_code'}
        delattr(workspace, 'file_dict') if hasattr(workspace, 'file_dict') else None
        
        exp = Mock()
        exp.sub_workspace_list = [workspace]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.expression == 'files_code'
    
    def test_hypothesis_as_string(self):
        """测试 hypothesis 为字符串"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = pd.DataFrame({'IC': [0.05]})
        exp.hypothesis = "Direct string hypothesis"
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        assert factor.description == "Direct string hypothesis"
    
    def test_invalid_metric_values_are_handled(self):
        """测试无效的指标值被正确处理"""
        exp = Mock()
        exp.sub_workspace_list = [Mock(file_dict={'factor.py': 'code'})]
        exp.result = pd.DataFrame({
            'IC': [np.nan],
            'IR': [np.inf],
            'annual_return': [-np.inf]
        })
        exp.hypothesis = Mock(hypothesis="Test")
        
        factor = _ResultAdapter.exp_to_factor(exp)
        
        # NaN 应该转换为 0.0
        assert factor.performance['ic'] == 0.0


class TestConfigAdapter:
    """测试配置适配器"""
    
    def test_basic_config_conversion(self):
        """测试基础配置转换"""
        config = {
            "llm_model": "gpt-4-turbo",
            "llm_api_key": "sk-test",
            "max_iterations": 10
        }
        
        official_config = _ConfigAdapter.to_official_config(config)
        
        assert official_config["llm_model"] == "gpt-4-turbo"
        assert official_config["llm_api_key"] == "sk-test"
        assert official_config["max_iterations"] == 10
    
    def test_model_name_alias(self):
        """测试 model 别名转换为 llm_model"""
        config = {"model": "gpt-4"}
        
        official_config = _ConfigAdapter.to_official_config(config)
        
        assert official_config["llm_model"] == "gpt-4"
    
    def test_provider_inference_from_model(self):
        """测试从模型名推断 provider"""
        # OpenAI
        config = {"llm_model": "gpt-4-turbo"}
        official = _ConfigAdapter.to_official_config(config)
        assert official["llm_provider"] == "openai"
        
        # Anthropic
        config = {"llm_model": "claude-3"}
        official = _ConfigAdapter.to_official_config(config)
        assert official["llm_provider"] == "anthropic"
    
    def test_temperature_alias(self):
        """测试 temperature 别名转换"""
        config = {"temperature": 0.8}
        
        official_config = _ConfigAdapter.to_official_config(config)
        
        assert official_config["llm_temperature"] == 0.8


class TestExperimentsToFactors:
    """测试批量转换 experiments → factors"""
    
    def test_extract_multiple_factors(self):
        """测试提取多个因子"""
        # 创建 mock trace
        trace = Mock()
        trace.hist = []
        
        for i in range(5):
            exp = Mock()
            exp.sub_workspace_list = [Mock(file_dict={'factor.py': f'code_{i}'})]
            exp.result = pd.DataFrame({'IC': [0.05 + i*0.01]})
            exp.hypothesis = Mock(hypothesis=f"Hypothesis {i}")
            
            feedback = Mock(decision=True)
            trace.hist.append((exp, feedback))
        
        # 执行转换
        factors = _ResultAdapter.experiments_to_factors(trace, n_factors=3)
        
        # 验证
        assert len(factors) == 3
        for i, factor in enumerate(factors):
            assert factor.expression == f'code_{i}'
            assert factor.performance['ic'] == pytest.approx(0.05 + i*0.01)
    
    def test_skip_rejected_experiments(self):
        """测试跳过被拒绝的实验"""
        trace = Mock()
        trace.hist = []
        
        # 3个成功，2个拒绝
        for i in range(5):
            exp = Mock()
            exp.sub_workspace_list = [Mock(file_dict={'factor.py': f'code_{i}'})]
            exp.result = pd.DataFrame({'IC': [0.05]}) if i % 2 == 0 else None
            exp.hypothesis = Mock(hypothesis=f"Hypothesis {i}")
            
            feedback = Mock(decision=(i % 2 == 0))  # 0,2,4 成功
            trace.hist.append((exp, feedback))
        
        factors = _ResultAdapter.experiments_to_factors(trace, n_factors=10)
        
        # 应该只有3个成功的因子
        assert len(factors) == 3
    
    def test_handle_conversion_errors_gracefully(self):
        """测试优雅处理转换错误"""
        trace = Mock()
        trace.hist = []
        
        # 第1个实验正常
        exp1 = Mock()
        exp1.sub_workspace_list = [Mock(file_dict={'factor.py': 'code_1'})]
        exp1.result = pd.DataFrame({'IC': [0.05]})
        exp1.hypothesis = Mock(hypothesis="Good")
        trace.hist.append((exp1, Mock(decision=True)))
        
        # 第2个实验会导致错误 (no workspace)
        exp2 = Mock(spec=['result', 'hypothesis'])
        exp2.result = pd.DataFrame({'IC': [0.06]})
        exp2.hypothesis = Mock(hypothesis="Bad")
        trace.hist.append((exp2, Mock(decision=True)))
        
        # 第3个实验正常
        exp3 = Mock()
        exp3.sub_workspace_list = [Mock(file_dict={'factor.py': 'code_3'})]
        exp3.result = pd.DataFrame({'IC': [0.04]})
        exp3.hypothesis = Mock(hypothesis="Good")
        trace.hist.append((exp3, Mock(decision=True)))
        
        # 执行转换 (不应该抛出异常)
        factors = _ResultAdapter.experiments_to_factors(trace, n_factors=10)
        
        # 应该跳过错误的，返回2个成功的
        assert len(factors) == 2
        assert factors[0].expression == 'code_1'
        assert factors[1].expression == 'code_3'


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
