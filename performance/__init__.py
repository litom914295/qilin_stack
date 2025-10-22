"""
性能优化模块

提供并发优化和多级缓存功能，提升决策引擎性能
"""

__version__ = '1.0.0'

from .concurrency import (
    ConcurrencyOptimizer,
    get_optimizer,
    parallel_task
)

from .cache import (
    MemoryCache,
    RedisCache,
    MultiLevelCache,
    get_cache,
    cached
)

__all__ = [
    # 并发优化
    'ConcurrencyOptimizer',
    'get_optimizer',
    'parallel_task',
    
    # 缓存
    'MemoryCache',
    'RedisCache',
    'MultiLevelCache',
    'get_cache',
    'cached',
]
