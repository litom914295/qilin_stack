"""
代码沙盒模块 - 安全执行用户/LLM生成的代码

功能:
1. AST 静态分析 (白名单检查)
2. 限定命名空间 (safe builtins + 指定模块)
3. 执行超时控制
4. 异常捕获与日志记录

优先级: P1 (安全加固)
"""

import ast
import logging
import signal
import sys
from typing import Dict, Any, List, Set, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """安全级别"""
    STRICT = "strict"      # 严格模式: 只允许安全的数学/数据操作
    MODERATE = "moderate"  # 中等模式: 允许部分 import 和文件读取
    PERMISSIVE = "permissive"  # 宽松模式: 允许大部分操作 (仅用于测试)


@dataclass
class CodeExecutionResult:
    """代码执行结果"""
    success: bool
    locals: Dict[str, Any]  # 执行后的局部变量
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CodeSandbox:
    """
    安全代码沙盒
    
    使用方法:
    ```python
    sandbox = CodeSandbox(security_level=SecurityLevel.STRICT, timeout=5)
    
    result = sandbox.execute(
        code="result = df['close'].mean()",
        context={'df': dataframe}
    )
    
    if result.success:
        print(result.locals['result'])
    else:
        print(f"Error: {result.error}")
    ```
    """
    
    # 危险关键字 (禁止出现在代码中)
    DANGEROUS_KEYWORDS = {
        'import os', 'import sys', 'import subprocess', 'import socket',
        '__import__', '__builtins__', '__globals__', '__locals__',
        'compile', 'execfile', 'file', 'input', 'raw_input',
        'reload', 'system', 'popen', 'rmdir', 'remove', 'unlink',
    }
    
    # 危险内置函数
    DANGEROUS_BUILTINS = {
        'open', 'exec', 'eval', '__import__', 'compile',
        'execfile', 'reload', 'input', 'raw_input'
    }
    
    # 安全的内置函数白名单
    SAFE_BUILTINS = {
        # 基础类型
        'int', 'float', 'str', 'bool', 'list', 'tuple', 'dict', 'set',
        
        # 数学运算
        'abs', 'round', 'min', 'max', 'sum', 'pow',
        
        # 迭代器
        'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
        
        # 类型检查
        'len', 'type', 'isinstance', 'hasattr', 'getattr',
        
        # 其他安全函数
        'print',  # 允许 print (用于调试)
    }
    
    # 安全的模块白名单
    SAFE_MODULES = {
        'numpy': ['np'],
        'pandas': ['pd'],
        'scipy': ['scipy'],
        'sklearn': ['sklearn'],
        'math': ['math'],
        'datetime': ['datetime'],
        'collections': ['collections'],
        'itertools': ['itertools'],
        'functools': ['functools'],
    }
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STRICT,
        timeout: int = 5,
        enable_logging: bool = True
    ):
        """
        初始化代码沙盒
        
        Args:
            security_level: 安全级别
            timeout: 执行超时时间 (秒)
            enable_logging: 是否启用日志
        """
        self.security_level = security_level
        self.timeout = timeout
        self.enable_logging = enable_logging
    
    def execute(
        self,
        code: str,
        context: Dict[str, Any],
        allowed_modules: Optional[List[str]] = None
    ) -> CodeExecutionResult:
        """
        安全执行代码
        
        Args:
            code: 要执行的 Python 代码
            context: 执行上下文 (可用变量)
            allowed_modules: 额外允许的模块列表
        
        Returns:
            CodeExecutionResult 执行结果
        """
        # 1. 静态分析 (AST 检查)
        is_safe, warnings = self._validate_code_ast(code, allowed_modules)
        if not is_safe:
            error_msg = f"Code validation failed: {'; '.join(warnings)}"
            if self.enable_logging:
                logger.error(error_msg)
            return CodeExecutionResult(
                success=False,
                locals={},
                error=error_msg,
                warnings=warnings
            )
        
        # 2. 关键字检查
        keyword_warnings = self._check_dangerous_keywords(code)
        if keyword_warnings:
            warnings.extend(keyword_warnings)
            error_msg = f"Code contains dangerous keywords: {'; '.join(keyword_warnings)}"
            if self.enable_logging:
                logger.error(error_msg)
            return CodeExecutionResult(
                success=False,
                locals={},
                error=error_msg,
                warnings=warnings
            )
        
        # 3. 构建安全执行环境
        safe_globals = self._build_safe_globals(context, allowed_modules)
        safe_locals = {}
        
        # 4. 执行代码 (带超时)
        try:
            # 使用 timeout (仅在 Unix 系统上有效)
            if sys.platform != 'win32' and self.timeout > 0:
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.timeout)
            
            # 执行代码
            exec(code, safe_globals, safe_locals)
            
            # 取消 timeout
            if sys.platform != 'win32' and self.timeout > 0:
                signal.alarm(0)
            
            if self.enable_logging:
                logger.debug(f"Code executed successfully. Locals: {list(safe_locals.keys())}")
            
            return CodeExecutionResult(
                success=True,
                locals=safe_locals,
                warnings=warnings
            )
        
        except TimeoutError:
            error_msg = f"Code execution timeout ({self.timeout}s)"
            if self.enable_logging:
                logger.error(error_msg)
            return CodeExecutionResult(
                success=False,
                locals={},
                error=error_msg,
                warnings=warnings
            )
        
        except Exception as e:
            error_msg = f"Code execution failed: {type(e).__name__}: {str(e)}"
            if self.enable_logging:
                logger.error(error_msg)
            return CodeExecutionResult(
                success=False,
                locals={},
                error=error_msg,
                warnings=warnings
            )
    
    def _validate_code_ast(
        self,
        code: str,
        allowed_modules: Optional[List[str]] = None
    ) -> tuple[bool, List[str]]:
        """
        使用 AST 验证代码安全性
        
        Returns:
            (is_safe, warnings)
        """
        warnings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            warnings.append(f"Syntax error: {e}")
            return False, warnings
        
        # 遍历 AST 节点
        for node in ast.walk(tree):
            # 检查 import
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                is_safe_import, import_warning = self._check_import_safety(
                    node, allowed_modules
                )
                if not is_safe_import:
                    warnings.append(import_warning)
                    return False, warnings
            
            # 检查函数调用
            if isinstance(node, ast.Call):
                is_safe_call, call_warning = self._check_call_safety(node)
                if not is_safe_call:
                    warnings.append(call_warning)
                    return False, warnings
            
            # 检查属性访问 (防止访问危险属性)
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    warnings.append(f"Access to private attribute: {node.attr}")
                    # 警告但不阻止 (某些合法用途需要访问私有属性)
        
        return True, warnings
    
    def _check_import_safety(
        self,
        node: ast.AST,
        allowed_modules: Optional[List[str]] = None
    ) -> tuple[bool, str]:
        """检查 import 是否安全"""
        allowed = set(self.SAFE_MODULES.keys())
        if allowed_modules:
            allowed.update(allowed_modules)
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name not in allowed:
                    return False, f"Unsafe import: {alias.name}"
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name not in allowed:
                    return False, f"Unsafe import from: {node.module}"
        
        return True, ""
    
    def _check_call_safety(self, node: ast.Call) -> tuple[bool, str]:
        """检查函数调用是否安全"""
        # 检查直接调用危险函数
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.DANGEROUS_BUILTINS:
                return False, f"Dangerous function call: {func_name}"
        
        return True, ""
    
    def _check_dangerous_keywords(self, code: str) -> List[str]:
        """检查代码中是否包含危险关键字"""
        warnings = []
        code_lower = code.lower()
        
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in code_lower:
                warnings.append(f"Dangerous keyword: {keyword}")
        
        return warnings
    
    def _build_safe_globals(
        self,
        context: Dict[str, Any],
        allowed_modules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """构建安全的全局命名空间"""
        # 安全的内置函数
        safe_builtins = {
            name: __builtins__[name]
            for name in self.SAFE_BUILTINS
            if name in __builtins__
        }
        
        # 构建 globals
        safe_globals = {
            '__builtins__': safe_builtins,
        }
        
        # 添加安全模块
        import numpy as np
        import pandas as pd
        
        safe_globals['np'] = np
        safe_globals['pd'] = pd
        
        # 添加用户提供的上下文
        safe_globals.update(context)
        
        return safe_globals
    
    def _timeout_handler(self, signum, frame):
        """超时处理器 (仅 Unix)"""
        raise TimeoutError("Code execution timeout")


# 便捷函数
def execute_safe(
    code: str,
    context: Dict[str, Any],
    timeout: int = 5,
    security_level: SecurityLevel = SecurityLevel.STRICT
) -> CodeExecutionResult:
    """
    便捷函数: 安全执行代码
    
    示例:
    ```python
    result = execute_safe(
        code="result = df['close'].mean()",
        context={'df': dataframe},
        timeout=5
    )
    ```
    """
    sandbox = CodeSandbox(
        security_level=security_level,
        timeout=timeout
    )
    return sandbox.execute(code, context)


if __name__ == "__main__":
    # 测试示例
    import pandas as pd
    import numpy as np
    
    # 示例 1: 安全代码
    test_df = pd.DataFrame({
        'close': [10.0, 11.0, 12.0],
        'volume': [1000, 1100, 1200]
    })
    
    result = execute_safe(
        code="""
result = df['close'].mean()
factor = df['close'] / df['volume']
""",
        context={'df': test_df},
        timeout=5
    )
    
    print("=== Test 1: Safe Code ===")
    print(f"Success: {result.success}")
    print(f"Result: {result.locals.get('result')}")
    print(f"Factor: {result.locals.get('factor')}")
    print()
    
    # 示例 2: 危险代码 (应该被阻止)
    result2 = execute_safe(
        code="""
import os
os.system('echo dangerous')
""",
        context={},
        timeout=5
    )
    
    print("=== Test 2: Dangerous Code ===")
    print(f"Success: {result2.success}")
    print(f"Error: {result2.error}")
    print(f"Warnings: {result2.warnings}")
    print()
    
    # 示例 3: 超时测试 (仅在 Unix 上有效)
    result3 = execute_safe(
        code="""
import time
time.sleep(10)  # 会超时
""",
        context={},
        timeout=2
    )
    
    print("=== Test 3: Timeout ===")
    print(f"Success: {result3.success}")
    print(f"Error: {result3.error}")
