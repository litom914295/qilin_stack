"""
缓存管理器单元测试
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# 假设 cache_manager 在 app.core 模块中
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.cache_manager import (
    CacheManager,
    get_cache_manager,
    cached,
    memoize
)


class TestCacheManager:
    """测试缓存管理器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.test_cache_dir = tempfile.mkdtemp()
        self.cache = CacheManager(
            cache_dir=self.test_cache_dir,
            default_ttl=10,
            max_memory_items=5
        )
    
    def teardown_method(self):
        """每个测试后的清理"""
        if Path(self.test_cache_dir).exists():
            shutil.rmtree(self.test_cache_dir)
    
    def test_basic_set_get(self):
        """测试基本的设置和获取"""
        # 设置缓存
        self.cache.set('key1', 'value1', ttl=60)
        
        # 获取缓存
        value = self.cache.get('key1')
        assert value == 'value1'
    
    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        value = self.cache.get('nonexistent_key')
        assert value is None
    
    def test_ttl_expiration(self):
        """测试缓存过期"""
        # 设置1秒过期
        self.cache.set('key_expire', 'value', ttl=1)
        
        # 立即获取应该存在
        value = self.cache.get('key_expire')
        assert value == 'value'
        
        # 等待过期
        time.sleep(1.5)
        
        # 过期后应该返回None
        value = self.cache.get('key_expire')
        assert value is None
    
    def test_disk_cache(self):
        """测试磁盘缓存"""
        # 设置带磁盘缓存
        self.cache.set('disk_key', {'data': 'complex_object'}, ttl=60, use_disk=True)
        
        # 清空内存缓存
        self.cache.clear(memory_only=True)
        
        # 从磁盘恢复
        value = self.cache.get('disk_key', use_disk=True)
        assert value == {'data': 'complex_object'}
    
    def test_delete(self):
        """测试删除缓存"""
        self.cache.set('key_delete', 'value')
        
        # 确认存在
        assert self.cache.get('key_delete') == 'value'
        
        # 删除
        self.cache.delete('key_delete')
        
        # 确认已删除
        assert self.cache.get('key_delete') is None
    
    def test_clear(self):
        """测试清空所有缓存"""
        # 设置多个缓存
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')
        self.cache.set('key3', 'value3')
        
        # 清空
        self.cache.clear()
        
        # 确认全部清空
        assert self.cache.get('key1') is None
        assert self.cache.get('key2') is None
        assert self.cache.get('key3') is None
    
    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        # 缓存容量为5
        for i in range(7):
            self.cache.set(f'key{i}', f'value{i}', use_disk=False)
        
        # 由于LRU策略,旧的可能会被淘汰(但从磁盘还能加载)
        # 我们直接测试内存缓存大小
        assert len(self.cache._memory_cache) <= 5
        
        # 最新的应该还在内存
        assert self.cache.get('key6', use_disk=False) == 'value6'
    
    def test_cleanup_expired(self):
        """测试清理过期缓存"""
        # 设置多个缓存,部分过期
        self.cache.set('key_short', 'value1', ttl=1)
        self.cache.set('key_long', 'value2', ttl=60)
        
        # 等待短期缓存过期
        time.sleep(1.5)
        
        # 清理过期
        count = self.cache.cleanup_expired()
        
        # 应该清理了至少1个(可能包括内存和磁盘)
        assert count >= 1
        
        # 长期的还在
        assert self.cache.get('key_long') == 'value2'


class TestCachedDecorator:
    """测试缓存装饰器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 清空全局缓存
        cache = get_cache_manager()
        cache.clear()
        self.call_count = 0
    
    def test_cached_decorator(self):
        """测试缓存装饰器基本功能"""
        
        @cached(ttl=60)
        def expensive_function(x, y):
            self.call_count += 1
            return x + y
        
        # 第一次调用
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert self.call_count == 1
        
        # 第二次调用应该从缓存获取
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert self.call_count == 1  # 没有增加
        
        # 不同参数应该重新计算
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert self.call_count == 2
    
    def test_clear_cache(self):
        """测试清除缓存方法"""
        
        @cached(ttl=60)
        def cached_func(x):
            self.call_count += 1
            return x * 2
        
        # 调用并缓存
        result1 = cached_func(5)
        assert result1 == 10
        assert self.call_count == 1
        
        # 清除缓存
        cached_func.clear_cache(5)
        
        # 再次调用应该重新计算
        result2 = cached_func(5)
        assert result2 == 10
        assert self.call_count == 2


class TestMemoizeDecorator:
    """测试记忆化装饰器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.call_count = 0
    
    def test_memoize(self):
        """测试记忆化装饰器"""
        
        @memoize
        def fibonacci(n):
            self.call_count += 1
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        # 计算 fibonacci(5)
        result = fibonacci(5)
        assert result == 5
        
        # 由于缓存,调用次数应该很少
        # fibonacci(5) = fibonacci(4) + fibonacci(3)
        # fibonacci(4) = fibonacci(3) + fibonacci(2)
        # fibonacci(3) = fibonacci(2) + fibonacci(1)
        # fibonacci(2) = fibonacci(1) + fibonacci(0)
        # 总共调用: 0,1,1,2,3,5 = 6次(没有重复计算)
        assert self.call_count == 6
    
    def test_memoize_clear(self):
        """测试清除记忆化缓存"""
        
        @memoize
        def expensive_func(x):
            self.call_count += 1
            return x ** 2
        
        # 第一次调用
        result1 = expensive_func(10)
        assert result1 == 100
        assert self.call_count == 1
        
        # 第二次调用(缓存)
        result2 = expensive_func(10)
        assert result2 == 100
        assert self.call_count == 1
        
        # 清除缓存
        expensive_func.clear()
        
        # 第三次调用(重新计算)
        result3 = expensive_func(10)
        assert result3 == 100
        assert self.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
