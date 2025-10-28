"""
统一异常处理系统
"""

from typing import Dict, Any, Optional
from enum import Enum
import traceback
import logging

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """错误代码枚举"""
    # 通用错误 (1000-1999)
    UNKNOWN_ERROR = 1000
    VALIDATION_ERROR = 1001
    CONFIGURATION_ERROR = 1002
    
    # 数据相关错误 (2000-2999)
    DATA_FETCH_ERROR = 2000
    DATA_QUALITY_ERROR = 2001
    DATA_NOT_FOUND = 2002
    DATA_FORMAT_ERROR = 2003
    
    # Agent相关错误 (3000-3999)
    AGENT_EXECUTION_ERROR = 3000
    AGENT_TIMEOUT_ERROR = 3001
    AGENT_NOT_FOUND = 3002
    
    # 交易相关错误 (4000-4999)
    TRADING_ERROR = 4000
    ORDER_EXECUTION_ERROR = 4001
    POSITION_ERROR = 4002
    RISK_LIMIT_ERROR = 4003
    
    # 系统相关错误 (5000-5999)
    SYSTEM_ERROR = 5000
    RESOURCE_ERROR = 5001
    TIMEOUT_ERROR = 5002
    NETWORK_ERROR = 5003
    
    # 安全相关错误 (6000-6999)
    AUTHENTICATION_ERROR = 6000
    AUTHORIZATION_ERROR = 6001
    RATE_LIMIT_ERROR = 6002


class QilinException(Exception):
    """基础异常类"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'error_code': self.error_code.value,
            'error_name': self.error_code.name,
            'message': self.message,
            'details': self.details,
            'recoverable': self.recoverable
        }


class DataException(QilinException):
    """数据相关异常"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.DATA_FETCH_ERROR,
            **kwargs
        )


class DataQualityException(QilinException):
    """数据质量异常"""
    def __init__(self, message: str, quality_issues: Dict[str, Any], **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.DATA_QUALITY_ERROR,
            details={'quality_issues': quality_issues},
            **kwargs
        )


class AgentException(QilinException):
    """Agent执行异常"""
    def __init__(self, agent_name: str, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.AGENT_EXECUTION_ERROR,
            details={'agent_name': agent_name},
            **kwargs
        )


class TradingException(QilinException):
    """交易异常"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.TRADING_ERROR,
            **kwargs
        )


class RiskLimitException(QilinException):
    """风险限制异常"""
    def __init__(self, message: str, limit_type: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.RISK_LIMIT_ERROR,
            details={'limit_type': limit_type},
            recoverable=False,  # 风险限制通常不可恢复
            **kwargs
        )


class SecurityException(QilinException):
    """安全异常"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            recoverable=False,  # 安全异常不可恢复
            **kwargs
        )


class ExceptionHandler:
    """全局异常处理器"""
    
    def __init__(self):
        self.handlers = {}
        self.fallback_handler = self._default_fallback
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self.register(DataException, self._handle_data_exception)
        self.register(AgentException, self._handle_agent_exception)
        self.register(TradingException, self._handle_trading_exception)
        self.register(RiskLimitException, self._handle_risk_exception)
        self.register(SecurityException, self._handle_security_exception)
    
    def register(self, exception_type: type, handler: callable):
        """注册异常处理器"""
        self.handlers[exception_type] = handler
    
    async def handle(self, exception: Exception) -> Dict[str, Any]:
        """处理异常"""
        # 获取对应的处理器
        handler = None
        for exc_type, exc_handler in self.handlers.items():
            if isinstance(exception, exc_type):
                handler = exc_handler
                break
        
        if handler is None:
            handler = self.fallback_handler
        
        # 执行处理
        try:
            return await handler(exception)
        except Exception as e:
            logger.error(f"Error in exception handler: {e}")
            return await self._default_fallback(exception)
    
    async def _handle_data_exception(self, exc: DataException) -> Dict[str, Any]:
        """处理数据异常"""
        logger.error(f"Data exception: {exc.message}", extra=exc.details)
        
        # 尝试降级处理
        if exc.recoverable:
            logger.info("Attempting to use cached data...")
            # 这里可以实现从缓存获取数据的逻辑
        
        return {
            'handled': True,
            'strategy': 'use_cache',
            'error': exc.to_dict()
        }
    
    async def _handle_agent_exception(self, exc: AgentException) -> Dict[str, Any]:
        """处理Agent异常"""
        logger.error(f"Agent exception: {exc.message}", extra=exc.details)
        
        # 降级：使用默认评分
        return {
            'handled': True,
            'strategy': 'default_score',
            'error': exc.to_dict()
        }
    
    async def _handle_trading_exception(self, exc: TradingException) -> Dict[str, Any]:
        """处理交易异常"""
        logger.error(f"Trading exception: {exc.message}", extra=exc.details)
        
        # 交易异常需要人工介入
        return {
            'handled': True,
            'strategy': 'manual_intervention_required',
            'error': exc.to_dict()
        }
    
    async def _handle_risk_exception(self, exc: RiskLimitException) -> Dict[str, Any]:
        """处理风险限制异常"""
        logger.critical(f"Risk limit exceeded: {exc.message}", extra=exc.details)
        
        # 风险异常：立即停止交易
        return {
            'handled': True,
            'strategy': 'stop_trading',
            'error': exc.to_dict()
        }
    
    async def _handle_security_exception(self, exc: SecurityException) -> Dict[str, Any]:
        """处理安全异常"""
        logger.critical(f"Security exception: {exc.message}", extra=exc.details)
        
        # 安全异常：拒绝请求
        return {
            'handled': True,
            'strategy': 'deny_request',
            'error': exc.to_dict()
        }
    
    async def _default_fallback(self, exception: Exception) -> Dict[str, Any]:
        """默认回退处理"""
        logger.error(
            f"Unhandled exception: {type(exception).__name__}: {str(exception)}",
            extra={'traceback': traceback.format_exc()}
        )
        
        return {
            'handled': False,
            'strategy': 'log_and_continue',
            'error': {
                'type': type(exception).__name__,
                'message': str(exception)
            }
        }


# 全局异常处理器实例
exception_handler = ExceptionHandler()
