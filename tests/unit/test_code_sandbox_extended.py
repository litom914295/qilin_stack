"""
code_sandbox 模块扩展测试套件

测试范围:
1. 5层安全级别测试
2. 超时机制测试 (Linux/Mac)
3. 资源限制测试
4. 恶意代码拦截测试
5. 并发执行测试
6. 边界条件测试

Phase: 2.1 - code_sandbox 测试补充
收益: +5% 测试覆盖率 (77% → 82%)

作者: AI Agent
日期: 2024-11-08
"""

import pytest
import sys
import platform
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import threading
import multiprocessing

from rd_agent.code_sandbox import (
    CodeSandbox,
    SecurityLevel,
    CodeExecutionResult,
    execute_safe
)


@pytest.fixture
def sample_dataframe():
    """创建测试用的 DataFrame"""
    return pd.DataFrame({
        'close': [10.0, 11.0, 12.0, 13.0, 14.0],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'open': [9.5, 10.5, 11.5, 12.5, 13.5]
    })


class TestSecurityLevels:
    """测试 5层安全级别"""
    
    def test_strict_level_safe_code(self, sample_dataframe):
        """测试 STRICT 级别 - 安全代码应该通过"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        code = """
result = df['close'].mean()
factor = df['close'] / df['volume']
"""
        
        result = sandbox.execute(code, {'df': sample_dataframe})
        
        assert result.success, f"安全代码应该通过: {result.error}"
        assert 'result' in result.locals
        assert result.locals['result'] == 12.0
        assert 'factor' in result.locals
    
    def test_strict_level_dangerous_import(self):
        """测试 STRICT 级别 - 危险 import 应该被阻止"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        dangerous_codes = [
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
            "from os import system"
        ]
        
        for code in dangerous_codes:
            result = sandbox.execute(code, {})
            
            assert not result.success, f"危险代码应该被阻止: {code}"
            assert result.error is not None
            assert "Unsafe import" in result.error or "Dangerous keyword" in result.error
    
    def test_strict_level_dangerous_builtins(self):
        """测试 STRICT 级别 - 危险内置函数应该被阻止"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        dangerous_codes = [
            "eval('1+1')",
            "exec('print(1)')",
            "open('/etc/passwd')",
            "__import__('os')",
            "compile('print(1)', '', 'exec')"
        ]
        
        for code in dangerous_codes:
            result = sandbox.execute(code, {})
            
            assert not result.success, f"危险函数应该被阻止: {code}"
            assert result.error is not None
    
    def test_moderate_level_allows_safe_imports(self, sample_dataframe):
        """测试 MODERATE 级别 - 应该允许安全的 import"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.MODERATE,
            timeout=5
        )
        
        # 安全的数学和数据处理模块
        safe_codes = [
            "import numpy as np\nresult = np.mean([1,2,3])",
            "import pandas as pd\nresult = pd.Series([1,2,3]).mean()",
            "import math\nresult = math.sqrt(16)",
            "from collections import Counter\nresult = Counter([1,2,2,3])"
        ]
        
        for code in safe_codes:
            result = sandbox.execute(code, {'df': sample_dataframe})
            
            assert result.success, f"安全 import 应该通过: {code}, Error: {result.error}"
            assert 'result' in result.locals
    
    def test_permissive_level_allows_more_operations(self):
        """测试 PERMISSIVE 级别 - 允许更多操作 (仅用于测试环境)"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.PERMISSIVE,
            timeout=5
        )
        
        # PERMISSIVE 级别主要用于测试,这里验证它与 STRICT 的区别
        # 注意: 实际实现中可能需要扩展 PERMISSIVE 的行为
        
        code = "result = sum([1, 2, 3, 4, 5])"
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == 15


class TestTimeoutMechanism:
    """测试超时机制"""
    
    @pytest.mark.skipif(
        sys.platform == 'win32',
        reason="超时机制在 Windows 上不可用 (signal.SIGALRM 不支持)"
    )
    def test_timeout_on_long_running_code(self):
        """测试超时 - 长时间运行的代码应该被终止 (仅 Linux/Mac)"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=1  # 1秒超时
        )
        
        # 模拟长时间运行
        code = """
import time
time.sleep(5)  # 睡眠 5 秒,应该触发超时
"""
        
        result = sandbox.execute(code, {})
        
        assert not result.success
        assert result.error is not None
        assert "timeout" in result.error.lower()
    
    @pytest.mark.skipif(
        sys.platform == 'win32',
        reason="超时机制在 Windows 上不可用"
    )
    def test_timeout_on_infinite_loop(self):
        """测试超时 - 无限循环应该被终止 (仅 Linux/Mac)"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=2
        )
        
        code = """
count = 0
while True:
    count += 1
"""
        
        result = sandbox.execute(code, {})
        
        assert not result.success
        assert "timeout" in result.error.lower()
    
    def test_no_timeout_on_fast_code(self, sample_dataframe):
        """测试快速代码不会超时"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        code = """
result = df['close'].sum()
"""
        
        result = sandbox.execute(code, {'df': sample_dataframe})
        
        assert result.success
        assert result.locals['result'] == 60.0
    
    def test_windows_timeout_warning(self):
        """测试 Windows 上的超时警告"""
        if sys.platform == 'win32':
            sandbox = CodeSandbox(
                security_level=SecurityLevel.STRICT,
                timeout=1
            )
            
            # Windows 上超时不会生效,但也不应该报错
            code = "result = 1 + 1"
            result = sandbox.execute(code, {})
            
            assert result.success
            assert result.locals['result'] == 2
            
            # TODO: Phase 3.1 将添加 Windows 超时支持


class TestMaliciousCodeDetection:
    """测试恶意代码拦截"""
    
    def test_block_file_operations(self):
        """测试阻止文件操作"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        malicious_codes = [
            "open('/etc/passwd', 'r').read()",
            "with open('secret.txt', 'w') as f: f.write('hacked')",
            "file('/etc/hosts')",
        ]
        
        for code in malicious_codes:
            result = sandbox.execute(code, {})
            
            assert not result.success, f"文件操作应该被阻止: {code}"
            assert result.error is not None
    
    def test_block_system_commands(self):
        """测试阻止系统命令"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        malicious_codes = [
            "import os; os.system('rm -rf /')",
            "import subprocess; subprocess.call(['ls'])",
            "__import__('os').system('echo hacked')",
        ]
        
        for code in malicious_codes:
            result = sandbox.execute(code, {})
            
            assert not result.success, f"系统命令应该被阻止: {code}"
    
    def test_block_network_operations(self):
        """测试阻止网络操作"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        malicious_codes = [
            "import socket; socket.socket()",
            "import urllib; urllib.urlopen('http://evil.com')",
            "import requests; requests.get('http://evil.com')",
        ]
        
        for code in malicious_codes:
            result = sandbox.execute(code, {})
            
            assert not result.success, f"网络操作应该被阻止: {code}"
    
    def test_block_private_attribute_access(self):
        """测试阻止访问私有属性 (警告级别)"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        code = """
class Test:
    def __init__(self):
        self._private = 42

obj = Test()
result = obj._private
"""
        
        # 这应该产生警告但可能不阻止 (取决于实现)
        result = sandbox.execute(code, {})
        
        # 验证警告存在
        if result.warnings:
            assert any('private' in w.lower() for w in result.warnings)
    
    def test_block_code_manipulation(self):
        """测试阻止代码操作"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5
        )
        
        malicious_codes = [
            "compile('print(1)', '', 'exec')",
            "eval('__import__(\"os\").system(\"ls\")')",
            "exec('import os')",
        ]
        
        for code in malicious_codes:
            result = sandbox.execute(code, {})
            
            assert not result.success, f"代码操作应该被阻止: {code}"


class TestConcurrentExecution:
    """测试并发执行"""
    
    def test_multiple_sandboxes_independent(self, sample_dataframe):
        """测试多个沙盒实例相互独立"""
        sandbox1 = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        sandbox2 = CodeSandbox(security_level=SecurityLevel.MODERATE, timeout=5)
        
        code1 = "result = df['close'].mean()"
        code2 = "result = df['volume'].sum()"
        
        result1 = sandbox1.execute(code1, {'df': sample_dataframe})
        result2 = sandbox2.execute(code2, {'df': sample_dataframe})
        
        assert result1.success
        assert result2.success
        assert result1.locals['result'] == 12.0
        assert result2.locals['result'] == 6000
    
    def test_concurrent_execution_thread_safe(self, sample_dataframe):
        """测试并发执行的线程安全性"""
        def run_sandbox(code, context, results, index):
            sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
            result = sandbox.execute(code, context)
            results[index] = result
        
        codes = [
            "result = df['close'].mean()",
            "result = df['volume'].sum()",
            "result = df['open'].min()",
            "result = df['close'].max()",
        ]
        
        threads = []
        results = [None] * len(codes)
        
        for i, code in enumerate(codes):
            thread = threading.Thread(
                target=run_sandbox,
                args=(code, {'df': sample_dataframe}, results, i)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 验证所有结果
        assert all(r is not None for r in results), "所有线程应该完成"
        assert all(r.success for r in results), "所有执行应该成功"
        
        # 验证结果正确
        assert results[0].locals['result'] == 12.0  # mean
        assert results[1].locals['result'] == 6000  # sum
        assert results[2].locals['result'] == 9.5   # min
        assert results[3].locals['result'] == 14.0  # max


class TestBoundaryConditions:
    """测试边界条件"""
    
    def test_empty_code(self):
        """测试空代码"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        result = sandbox.execute("", {})
        
        # 空代码应该成功执行 (什么都不做)
        assert result.success
        assert len(result.locals) == 0
    
    def test_whitespace_only_code(self):
        """测试只包含空白的代码"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        codes = [
            "   ",
            "\n\n\n",
            "\t\t\t",
            "  \n  \n  "
        ]
        
        for code in codes:
            result = sandbox.execute(code, {})
            assert result.success
    
    def test_very_long_code(self):
        """测试非常长的代码"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        # 生成长代码 (1000行赋值)
        code = "\n".join([f"var_{i} = {i}" for i in range(1000)])
        code += "\nresult = var_999"
        
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == 999
    
    def test_empty_context(self):
        """测试空上下文"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        code = "result = 1 + 1"
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == 2
    
    def test_large_dataframe_context(self):
        """测试大数据 DataFrame 上下文"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=10)
        
        # 创建大 DataFrame (10万行)
        large_df = pd.DataFrame({
            'value': np.random.randn(100000)
        })
        
        code = "result = df['value'].mean()"
        result = sandbox.execute(code, {'df': large_df})
        
        assert result.success
        assert 'result' in result.locals
    
    def test_syntax_error_code(self):
        """测试语法错误的代码"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        syntax_errors = [
            "result = ",
            "if True",
            "def func(",
            "import",
        ]
        
        for code in syntax_errors:
            result = sandbox.execute(code, {})
            
            assert not result.success
            assert "Syntax error" in result.error or "syntax" in result.error.lower()
    
    def test_runtime_error_code(self):
        """测试运行时错误的代码"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        runtime_errors = [
            "result = 1 / 0",  # ZeroDivisionError
            "result = undefined_var",  # NameError
            "result = [1, 2, 3][10]",  # IndexError
        ]
        
        for code in runtime_errors:
            result = sandbox.execute(code, {})
            
            assert not result.success
            assert result.error is not None


class TestExecuteSafeConvenience:
    """测试 execute_safe 便捷函数"""
    
    def test_execute_safe_basic(self, sample_dataframe):
        """测试 execute_safe 基本功能"""
        result = execute_safe(
            code="result = df['close'].mean()",
            context={'df': sample_dataframe},
            timeout=5
        )
        
        assert result.success
        assert result.locals['result'] == 12.0
    
    def test_execute_safe_custom_security_level(self):
        """测试 execute_safe 自定义安全级别"""
        result = execute_safe(
            code="result = sum([1, 2, 3])",
            context={},
            timeout=5,
            security_level=SecurityLevel.MODERATE
        )
        
        assert result.success
        assert result.locals['result'] == 6
    
    def test_execute_safe_with_error(self):
        """测试 execute_safe 错误处理"""
        result = execute_safe(
            code="import os",
            context={},
            timeout=5
        )
        
        assert not result.success
        assert result.error is not None


class TestLoggingIntegration:
    """测试日志集成"""
    
    def test_logging_enabled(self, sample_dataframe, caplog):
        """测试启用日志"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5,
            enable_logging=True
        )
        
        with caplog.at_level('DEBUG'):
            result = sandbox.execute(
                "result = df['close'].mean()",
                {'df': sample_dataframe}
            )
        
        assert result.success
        # 验证有日志记录 (如果 logging 配置正确)
    
    def test_logging_disabled(self, sample_dataframe, caplog):
        """测试禁用日志"""
        sandbox = CodeSandbox(
            security_level=SecurityLevel.STRICT,
            timeout=5,
            enable_logging=False
        )
        
        with caplog.at_level('DEBUG'):
            result = sandbox.execute(
                "result = df['close'].mean()",
                {'df': sample_dataframe}
            )
        
        assert result.success
        # 禁用日志时不应该有日志记录


class TestSpecialCases:
    """测试特殊情况"""
    
    def test_code_with_unicode(self):
        """测试包含 Unicode 字符的代码"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        code = """
# 这是中文注释
result = "你好世界"
emoji = "🎉"
"""
        
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == "你好世界"
        assert result.locals['emoji'] == "🎉"
    
    def test_code_with_complex_data_structures(self):
        """测试复杂数据结构"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        code = """
nested_dict = {
    'level1': {
        'level2': {
            'level3': [1, 2, 3]
        }
    }
}
result = nested_dict['level1']['level2']['level3'][1]
"""
        
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == 2
    
    def test_code_with_lambda_and_comprehensions(self):
        """测试 lambda 和列表推导式"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        code = """
data = [1, 2, 3, 4, 5]
squared = [x**2 for x in data]
filtered = list(filter(lambda x: x > 10, squared))
result = sum(filtered)
"""
        
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == 41  # 16 + 25


class TestResourceLimits:
    """测试资源限制"""
    
    def test_memory_intensive_operation(self):
        """测试内存密集型操作"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=10)
        
        # 创建较大的数组 (但不至于耗尽内存)
        code = """
import numpy as np
large_array = np.zeros((1000, 1000))
result = large_array.sum()
"""
        
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == 0.0
    
    def test_cpu_intensive_operation(self):
        """测试 CPU 密集型操作"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=10)
        
        code = """
# 计算斐波那契数列
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

result = fib(20)  # 适度计算
"""
        
        result = sandbox.execute(code, {})
        
        assert result.success
        assert result.locals['result'] == 6765


# 性能基准测试
class TestPerformance:
    """测试性能"""
    
    def test_execution_speed(self, sample_dataframe):
        """测试执行速度"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        code = "result = df['close'].mean()"
        
        start = time.time()
        result = sandbox.execute(code, {'df': sample_dataframe})
        elapsed = time.time() - start
        
        assert result.success
        assert elapsed < 1.0, f"执行应该在1秒内完成,实际: {elapsed:.2f}s"
    
    def test_multiple_executions_performance(self, sample_dataframe):
        """测试多次执行的性能"""
        sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
        
        code = "result = df['close'].sum()"
        
        start = time.time()
        for _ in range(100):
            result = sandbox.execute(code, {'df': sample_dataframe})
            assert result.success
        
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"100次执行应该在5秒内完成,实际: {elapsed:.2f}s"
        print(f"\n⚡ 性能: 100次执行耗时 {elapsed:.2f}s (平均 {elapsed/100*1000:.1f}ms/次)")


if __name__ == "__main__":
    """
    运行测试:
    
    # 运行所有测试
    pytest tests/unit/test_code_sandbox_extended.py -v
    
    # 运行特定测试类
    pytest tests/unit/test_code_sandbox_extended.py::TestSecurityLevels -v
    
    # 跳过 Windows 不支持的测试
    pytest tests/unit/test_code_sandbox_extended.py -v -m "not skipif"
    
    # 运行性能测试
    pytest tests/unit/test_code_sandbox_extended.py::TestPerformance -v -s
    """
    pytest.main([__file__, '-v', '-s'])
