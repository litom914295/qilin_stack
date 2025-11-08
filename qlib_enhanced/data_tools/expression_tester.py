"""
Qlib表达式引擎测试器
提供表达式语法验证、计算测试、结果可视化功能
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpressionResult:
    """表达式计算结果"""
    success: bool
    data: Optional[pd.DataFrame]
    error_message: str
    statistics: Optional[Dict[str, float]]
    
    def to_dict(self):
        """转换为字典"""
        return {
            'success': self.success,
            'error_message': self.error_message,
            'statistics': self.statistics,
            'data_shape': self.data.shape if self.data is not None else None
        }


class ExpressionTester:
    """Qlib表达式测试器"""
    
    def __init__(self):
        """初始化"""
        self.qlib_initialized = False
        try:
            import qlib
            from qlib.data import D
            self.qlib = qlib
            self.D = D
        except ImportError:
            logger.error("Qlib未安装")
            raise ImportError("请先安装Qlib: pip install pyqlib")
    
    def ensure_qlib_init(self, provider_uri: str = None):
        """确保Qlib已初始化"""
        if not self.qlib_initialized:
            try:
                if provider_uri is None:
                    from pathlib import Path
                    provider_uri = str(Path.home() / '.qlib' / 'qlib_data' / 'cn_data')
                
                self.qlib.init(provider_uri=provider_uri)
                self.qlib_initialized = True
                logger.info(f"Qlib初始化成功: {provider_uri}")
            except Exception as e:
                logger.error(f"Qlib初始化失败: {e}")
                raise
    
    def validate_syntax(self, expression: str) -> Tuple[bool, str]:
        """
        验证表达式语法
        
        Args:
            expression: Qlib表达式字符串
            
        Returns:
            (是否有效, 错误信息)
        """
        if not expression or not expression.strip():
            return False, "表达式不能为空"
        
        # 基础语法检查
        if expression.count('(') != expression.count(')'):
            return False, "括号不匹配"
        
        # 检查是否包含Qlib字段标识符
        qlib_fields = ['$open', '$close', '$high', '$low', '$volume', '$amount', '$vwap']
        has_field = any(field in expression.lower() for field in qlib_fields)
        
        if not has_field:
            return False, f"表达式必须包含至少一个Qlib字段 (如 {', '.join(qlib_fields)})"
        
        # 检查常见函数
        valid_functions = [
            'Ref', 'Mean', 'Std', 'Max', 'Min', 'Sum', 'Rank',
            'Corr', 'Cov', 'Delta', 'EMA', 'WMA', 'Log', 'Sign',
            'Abs', 'Pow', 'Greater', 'Less', 'And', 'Or', 'Not',
            'If', 'Rolling'
        ]
        
        # 尝试模拟解析（基础检查）
        try:
            # 检查是否有未知函数
            import re
            functions_in_expr = re.findall(r'\b([A-Z][a-zA-Z]+)\s*\(', expression)
            unknown_funcs = [f for f in functions_in_expr if f not in valid_functions]
            
            if unknown_funcs:
                return False, f"未知函数: {', '.join(unknown_funcs)}"
            
            return True, "语法检查通过"
            
        except Exception as e:
            return False, f"语法错误: {str(e)}"
    
    def test_expression(
        self,
        expression: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        provider_uri: str = None
    ) -> ExpressionResult:
        """
        测试表达式计算
        
        Args:
            expression: Qlib表达式
            symbols: 股票代码列表
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期
            provider_uri: Qlib数据路径
            
        Returns:
            ExpressionResult
        """
        # 1. 语法验证
        is_valid, error_msg = self.validate_syntax(expression)
        if not is_valid:
            return ExpressionResult(
                success=False,
                data=None,
                error_message=error_msg,
                statistics=None
            )
        
        # 2. 初始化Qlib
        try:
            self.ensure_qlib_init(provider_uri)
        except Exception as e:
            return ExpressionResult(
                success=False,
                data=None,
                error_message=f"Qlib初始化失败: {str(e)}",
                statistics=None
            )
        
        # 3. 计算表达式
        try:
            # 使用Qlib D.features计算
            result_df = self.D.features(
                symbols,
                [expression],
                start_time=start_date,
                end_time=end_date
            )
            
            # 4. 计算统计信息
            stats = self._calculate_statistics(result_df, expression)
            
            return ExpressionResult(
                success=True,
                data=result_df,
                error_message="",
                statistics=stats
            )
            
        except Exception as e:
            logger.error(f"表达式计算失败: {e}")
            return ExpressionResult(
                success=False,
                data=None,
                error_message=f"计算失败: {str(e)}",
                statistics=None
            )
    
    def _calculate_statistics(self, df: pd.DataFrame, expr_col: str) -> Dict[str, float]:
        """计算统计信息"""
        try:
            # DataFrame可能是MultiIndex (datetime, instrument)
            values = df[expr_col].dropna()
            
            return {
                'count': int(len(values)),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'q50': float(values.quantile(0.50)),
                'q75': float(values.quantile(0.75)),
                'missing_count': int(df[expr_col].isnull().sum()),
                'missing_rate': float(df[expr_col].isnull().sum() / len(df))
            }
        except Exception as e:
            logger.error(f"统计计算失败: {e}")
            return {}
    
    def get_example_expressions(self) -> Dict[str, List[str]]:
        """获取示例表达式"""
        return {
            "基础价格因子": [
                "Ref($close, 0) / Ref($close, 1) - 1",  # 日收益率
                "Ref($close, 0) / Ref($close, 5) - 1",  # 5日收益率
                "($high - $low) / $close",  # 日内振幅
                "($close - $open) / $open",  # 日内涨幅
            ],
            "动量因子": [
                "Ref($close, 0) / Ref($close, 20) - 1",  # 20日动量
                "Mean($close, 5) / Mean($close, 20) - 1",  # 均线差
                "Rank(Ref($close, 0) / Ref($close, 10) - 1)",  # 动量排名
            ],
            "波动率因子": [
                "Std($close / Ref($close, 1) - 1, 20)",  # 20日收益率标准差
                "Std($close, 10) / Mean($close, 10)",  # 变异系数
                "($high - $low) / $open",  # 波动率
            ],
            "量价因子": [
                "Corr($close, $volume, 20)",  # 价量相关性
                "$volume / Mean($volume, 5) - 1",  # 成交量异动
                "Corr($close / Ref($close, 1) - 1, Log($volume), 10)",  # 收益与成交量相关
            ],
            "高级因子": [
                "Rank($close) - Rank($open)",  # 价格排名差
                "Delta($close, 5) / Delta($close, 10)",  # 变化率比值
                "If($close > $open, $volume, -$volume)",  # 条件因子
            ]
        }


# 便捷函数
def validate_expression_syntax(expression: str) -> Tuple[bool, str]:
    """验证表达式语法（快捷函数）"""
    tester = ExpressionTester()
    return tester.validate_syntax(expression)


def test_expression(
    expression: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    provider_uri: str = None
) -> ExpressionResult:
    """测试表达式计算（快捷函数）"""
    tester = ExpressionTester()
    return tester.test_expression(expression, symbols, start_date, end_date, provider_uri)


# 导出
__all__ = [
    'ExpressionTester',
    'ExpressionResult',
    'validate_expression_syntax',
    'test_expression'
]
