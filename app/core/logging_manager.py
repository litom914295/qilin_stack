"""
统一日志管理模块
提供全局日志配置和管理功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


class LoggingManager:
    """日志管理器 - 统一配置和管理日志"""
    
    _initialized = False
    _loggers = {}
    
    @classmethod
    def setup_logging(
        cls,
        log_dir: str = "./logs",
        log_level: str = "INFO",
        max_file_size_mb: int = 100,
        backup_count: int = 10,
        console_output: bool = True,
        log_format: Optional[str] = None
    ) -> None:
        """
        设置全局日志配置
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
            max_file_size_mb: 单个日志文件最大大小(MB)
            backup_count: 日志文件保留数量
            console_output: 是否输出到控制台
            log_format: 自定义日志格式
        """
        if cls._initialized:
            return
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志格式
        if log_format is None:
            log_format = (
                '%(asctime)s [%(levelname)s] '
                '%(name)s:%(funcName)s:%(lineno)d - '
                '%(message)s'
            )
        
        formatter = logging.Formatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 获取根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有的处理器
        root_logger.handlers.clear()
        
        # 1. 文件处理器 - 所有日志
        all_log_file = log_path / f"qilin_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            all_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 2. 错误日志文件处理器
        error_log_file = log_path / f"qilin_error_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # 3. 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            # 控制台使用简化格式
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        cls._initialized = True
        
        # 记录初始化信息
        logging.info(f"✅ 日志系统初始化完成")
        logging.info(f"   日志目录: {log_path.absolute()}")
        logging.info(f"   日志级别: {log_level}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称 (通常使用 __name__)
            
        Returns:
            Logger实例
        """
        if not cls._initialized:
            cls.setup_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: str, logger_name: Optional[str] = None) -> None:
        """
        动态设置日志级别
        
        Args:
            level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
            logger_name: 指定的日志记录器名称，None表示根日志记录器
        """
        log_level = getattr(logging, level.upper())
        
        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
        else:
            logging.getLogger().setLevel(log_level)
    
    @classmethod
    def disable_logger(cls, logger_name: str) -> None:
        """禁用指定的日志记录器"""
        logging.getLogger(logger_name).disabled = True
    
    @classmethod
    def enable_logger(cls, logger_name: str) -> None:
        """启用指定的日志记录器"""
        logging.getLogger(logger_name).disabled = False


class SensitiveDataFilter(logging.Filter):
    """敏感数据过滤器 - 脱敏处理"""
    
    SENSITIVE_PATTERNS = [
        # API密钥
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', 'API_KEY'),
        # 密码
        (r'password["\']?\s*[:=]\s*["\']?(\S+)', 'PASSWORD'),
        # Token
        (r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', 'TOKEN'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤并脱敏敏感信息"""
        import re
        
        message = record.getMessage()
        
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, f'{replacement}=***', message, flags=re.IGNORECASE)
        
        record.msg = message
        record.args = ()
        
        return True


def get_logger(name: str) -> logging.Logger:
    """
    便捷函数 - 获取日志记录器
    
    Usage:
        from app.core.logging_manager import get_logger
        
        logger = get_logger(__name__)
        logger.info("This is an info message")
    """
    return LoggingManager.get_logger(name)


def setup_logging_from_config(config: 'QilinConfig') -> None:
    """
    从配置对象初始化日志系统
    
    Args:
        config: QilinConfig配置对象
    """
    LoggingManager.setup_logging(
        log_dir=config.logging.log_dir,
        log_level=config.logging.level.value,
        max_file_size_mb=config.logging.max_file_size_mb,
        backup_count=config.logging.backup_count,
        console_output=True
    )


# 便捷导出
__all__ = [
    'LoggingManager',
    'get_logger',
    'setup_logging_from_config',
    'SensitiveDataFilter'
]
