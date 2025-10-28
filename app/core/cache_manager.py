"""
缓存管理模块
提供多级缓存支持,优化数据访问性能
"""

import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import threading


class CacheManager:
    """
    缓存管理器
    支持内存缓存和磁盘缓存
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        default_ttl: int = 3600,  # 默认1小时
        max_memory_items: int = 1000
    ):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 磁盘缓存目录
            default_ttl: 默认过期时间(秒)
            max_memory_items: 内存缓存最大项数
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items
        
        # 内存缓存 {key: (value, expire_time)}
        self._memory_cache = {}
        self._lock = threading.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_str = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str, use_disk: bool = True) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
            use_disk: 是否使用磁盘缓存
            
        Returns:
            缓存值,不存在或已过期返回None
        """
        # 1. 尝试从内存获取
        with self._lock:
            if key in self._memory_cache:
                value, expire_time = self._memory_cache[key]
                if expire_time is None or datetime.now() < expire_time:
                    return value
                else:
                    # 已过期,删除
                    del self._memory_cache[key]
        
        # 2. 尝试从磁盘获取
        if use_disk:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    value, expire_time = data['value'], data['expire_time']
                    
                    if expire_time is None or datetime.now() < expire_time:
                        # 加载到内存
                        with self._lock:
                            self._memory_cache[key] = (value, expire_time)
                        return value
                    else:
                        # 已过期,删除
                        cache_file.unlink()
                except Exception:
                    pass
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_disk: bool = True
    ) -> None:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒),None表示永不过期
            use_disk: 是否同时保存到磁盘
        """
        ttl = ttl if ttl is not None else self.default_ttl
        expire_time = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
        
        # 1. 保存到内存
        with self._lock:
            # LRU策略: 内存满时删除最旧的
            if len(self._memory_cache) >= self.max_memory_items:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
            
            self._memory_cache[key] = (value, expire_time)
        
        # 2. 保存到磁盘
        if use_disk:
            cache_file = self.cache_dir / f"{key}.cache"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'value': value,
                        'expire_time': expire_time
                    }, f)
            except Exception:
                pass
    
    def delete(self, key: str) -> None:
        """删除缓存"""
        # 从内存删除
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
        
        # 从磁盘删除
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self, memory_only: bool = False) -> None:
        """清空缓存"""
        # 清空内存
        with self._lock:
            self._memory_cache.clear()
        
        # 清空磁盘
        if not memory_only:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
    
    def cleanup_expired(self) -> int:
        """清理过期缓存,返回清理数量"""
        count = 0
        now = datetime.now()
        
        # 清理内存
        with self._lock:
            expired_keys = [
                key for key, (_, expire_time) in self._memory_cache.items()
                if expire_time and now >= expire_time
            ]
            for key in expired_keys:
                del self._memory_cache[key]
                count += 1
        
        # 清理磁盘
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                expire_time = data.get('expire_time')
                if expire_time and now >= expire_time:
                    cache_file.unlink()
                    count += 1
            except Exception:
                pass
        
        return count


# 全局缓存实例
_global_cache = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def cached(
    ttl: Optional[int] = None,
    use_disk: bool = True,
    key_func: Optional[Callable] = None
):
    """
    缓存装饰器
    
    Args:
        ttl: 过期时间(秒)
        use_disk: 是否使用磁盘缓存
        key_func: 自定义键生成函数
    
    Usage:
        @cached(ttl=3600)
        def expensive_function(param1, param2):
            # 耗时计算
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}_{cache._generate_key(*args, **kwargs)}"
            
            # 尝试从缓存获取
            result = cache.get(cache_key, use_disk=use_disk)
            if result is not None:
                return result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 保存到缓存
            cache.set(cache_key, result, ttl=ttl, use_disk=use_disk)
            
            return result
        
        # 添加清除缓存的方法
        def clear_cache(*args, **kwargs):
            cache = get_cache_manager()
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}_{cache._generate_key(*args, **kwargs)}"
            cache.delete(cache_key)
        
        wrapper.clear_cache = clear_cache
        return wrapper
    
    return decorator


def memoize(func):
    """
    简单的记忆化装饰器 - 仅使用内存缓存
    适合纯函数的结果缓存
    
    Usage:
        @memoize
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    wrapper.cache = cache
    wrapper.clear = cache.clear
    
    return wrapper


# 便捷导出
__all__ = [
    'CacheManager',
    'get_cache_manager',
    'cached',
    'memoize'
]
