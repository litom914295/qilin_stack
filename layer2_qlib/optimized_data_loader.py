"""
优化的数据加载器 (Optimized Data Loader)
使用Redis缓存提升数据加载性能
"""

import os
import hashlib
import pickle
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  Redis未安装，缓存功能将被禁用。安装方法: pip install redis")

logger = logging.getLogger(__name__)


class CacheConfig:
    """缓存配置"""
    def __init__(self):
        self.enabled = os.getenv('REDIS_ENABLED', 'True').lower() == 'true' and REDIS_AVAILABLE
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', '6379'))
        self.db = int(os.getenv('REDIS_DB', '0'))
        self.password = os.getenv('REDIS_PASSWORD')
        self.default_ttl = int(os.getenv('CACHE_TTL', '3600'))  # 默认1小时
        self.prefix = 'qilin:data:'


class OptimizedDataLoader:
    """
    优化的数据加载器
    
    特性:
    1. Redis缓存机制
    2. 批量加载优化
    3. 内存缓存（LRU）
    4. 懒加载
    5. 异步加载支持
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        self.redis_client = None
        self._memory_cache = {}  # 简单的内存缓存
        self._memory_cache_size = 100
        
        if self.cache_config.enabled:
            self._init_redis()
    
    def _init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.Redis(
                host=self.cache_config.host,
                port=self.cache_config.port,
                db=self.cache_config.db,
                password=self.cache_config.password,
                decode_responses=False,  # 我们会存储二进制数据
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 测试连接
            self.redis_client.ping()
            logger.info(f"✅ Redis缓存已连接: {self.cache_config.host}:{self.cache_config.port}")
        except redis.ConnectionError as e:
            logger.warning(f"⚠️  Redis连接失败，缓存功能将被禁用: {e}")
            self.redis_client = None
        except Exception as e:
            logger.error(f"Redis初始化错误: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 创建唯一的键
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)
        
        # 使用MD5生成哈希
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{self.cache_config.prefix}{key_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        # 先检查内存缓存
        if cache_key in self._memory_cache:
            logger.debug(f"💾 从内存缓存命中: {cache_key[:20]}...")
            return self._memory_cache[cache_key]
        
        # 再检查Redis缓存
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = pickle.loads(cached_data)
                    # 更新到内存缓存
                    self._update_memory_cache(cache_key, data)
                    logger.debug(f"💾 从Redis缓存命中: {cache_key[:20]}...")
                    return data
            except Exception as e:
                logger.error(f"从Redis读取缓存失败: {e}")
        
        return None
    
    def _set_to_cache(self, cache_key: str, data: Any, ttl: Optional[int] = None):
        """存储数据到缓存"""
        ttl = ttl or self.cache_config.default_ttl
        
        # 存储到内存缓存
        self._update_memory_cache(cache_key, data)
        
        # 存储到Redis
        if self.redis_client:
            try:
                serialized_data = pickle.dumps(data)
                self.redis_client.setex(cache_key, ttl, serialized_data)
                logger.debug(f"💾 数据已缓存 (TTL={ttl}s): {cache_key[:20]}...")
            except Exception as e:
                logger.error(f"存储到Redis失败: {e}")
    
    def _update_memory_cache(self, key: str, data: Any):
        """更新内存缓存（LRU）"""
        if len(self._memory_cache) >= self._memory_cache_size:
            # 移除最旧的项
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = data
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        清除缓存
        
        Args:
            pattern: 匹配模式，如 "qilin:data:*"
        """
        # 清除内存缓存
        self._memory_cache.clear()
        
        # 清除Redis缓存
        if self.redis_client:
            try:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.info(f"✅ 已清除 {len(keys)} 个缓存项")
                else:
                    # 清除所有qilin相关的缓存
                    keys = self.redis_client.keys(f"{self.cache_config.prefix}*")
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.info(f"✅ 已清除所有缓存 ({len(keys)} 项)")
            except Exception as e:
                logger.error(f"清除缓存失败: {e}")
    
    def get_stock_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取股票数据（带缓存）
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            fields: 字段列表
            use_cache: 是否使用缓存
            
        Returns:
            数据DataFrame
        """
        cache_key = self._generate_cache_key(
            'stock_data',
            symbols=','.join(sorted(symbols)),
            start_date=start_date,
            end_date=end_date,
            fields=','.join(sorted(fields)) if fields else 'all'
        )
        
        # 尝试从缓存获取
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # 从数据源加载
        logger.info(f"📥 正在加载股票数据: {len(symbols)} 只股票")
        data = self._load_from_source(symbols, start_date, end_date, fields)
        
        # 存储到缓存
        if use_cache:
            self._set_to_cache(cache_key, data)
        
        return data
    
    def _load_from_source(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """从数据源加载数据"""
        try:
            # 尝试使用Qlib
            import qlib
            from qlib.data import D
            
            fields = fields or ['$open', '$high', '$low', '$close', '$volume', '$factor']
            
            # 批量加载数据
            data_dict = {}
            for symbol in symbols:
                try:
                    df = D.features(
                        [symbol],
                        fields,
                        start_date,
                        end_date
                    )
                    if not df.empty:
                        data_dict[symbol] = df
                except Exception as e:
                    logger.warning(f"加载 {symbol} 失败: {e}")
            
            if not data_dict:
                logger.warning("没有成功加载任何数据")
                return pd.DataFrame()
            
            # 合并数据
            result = pd.concat(data_dict, names=['symbol', 'datetime'])
            return result
            
        except ImportError:
            logger.warning("Qlib未安装，使用模拟数据")
            return self._generate_mock_data(symbols, start_date, end_date)
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def _generate_mock_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """生成模拟数据"""
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        data_list = []
        for symbol in symbols:
            # 生成随机价格数据
            base_price = 10 + np.random.randn() * 2
            prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.1)
            
            df = pd.DataFrame({
                'symbol': symbol,
                'datetime': dates,
                'open': prices * (1 + np.random.randn(len(dates)) * 0.01),
                'high': prices * (1 + abs(np.random.randn(len(dates))) * 0.02),
                'low': prices * (1 - abs(np.random.randn(len(dates))) * 0.02),
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates)),
                'factor': 1.0
            })
            data_list.append(df)
        
        result = pd.concat(data_list, ignore_index=False)
        result.set_index(['symbol', 'datetime'], inplace=True)
        return result
    
    def batch_load(
        self,
        tasks: List[Dict[str, Any]],
        use_cache: bool = True,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载数据
        
        Args:
            tasks: 任务列表，每个任务是一个字典，包含symbols, start_date, end_date等
            use_cache: 是否使用缓存
            parallel: 是否并行加载
            
        Returns:
            结果字典 {task_id: DataFrame}
        """
        results = {}
        
        if parallel:
            # 并行加载
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(
                        self.get_stock_data,
                        task['symbols'],
                        task['start_date'],
                        task['end_date'],
                        task.get('fields'),
                        use_cache
                    ): task.get('task_id', f"task_{i}")
                    for i, task in enumerate(tasks)
                }
                
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    try:
                        results[task_id] = future.result()
                    except Exception as e:
                        logger.error(f"任务 {task_id} 失败: {e}")
                        results[task_id] = pd.DataFrame()
        else:
            # 顺序加载
            for i, task in enumerate(tasks):
                task_id = task.get('task_id', f"task_{i}")
                try:
                    results[task_id] = self.get_stock_data(
                        task['symbols'],
                        task['start_date'],
                        task['end_date'],
                        task.get('fields'),
                        use_cache
                    )
                except Exception as e:
                    logger.error(f"任务 {task_id} 失败: {e}")
                    results[task_id] = pd.DataFrame()
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'memory_cache_size': len(self._memory_cache),
            'memory_cache_max': self._memory_cache_size,
            'redis_enabled': self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info('stats')
                keys_count = len(self.redis_client.keys(f"{self.cache_config.prefix}*"))
                stats['redis_keys'] = keys_count
                stats['redis_hits'] = info.get('keyspace_hits', 0)
                stats['redis_misses'] = info.get('keyspace_misses', 0)
                if stats['redis_hits'] + stats['redis_misses'] > 0:
                    stats['redis_hit_rate'] = stats['redis_hits'] / (stats['redis_hits'] + stats['redis_misses'])
            except Exception as e:
                logger.error(f"获取Redis统计信息失败: {e}")
        
        return stats


def cached_data_loader(ttl: Optional[int] = None):
    """
    数据加载装饰器（带缓存）
    
    Args:
        ttl: 缓存时间（秒）
    """
    def decorator(func):
        loader = OptimizedDataLoader()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = loader._generate_cache_key(
                func.__name__,
                *args,
                **kwargs
            )
            
            # 尝试从缓存获取
            cached_data = loader._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # 调用原函数
            result = func(*args, **kwargs)
            
            # 存储到缓存
            loader._set_to_cache(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据加载器
    loader = OptimizedDataLoader()
    
    print("\n📊 测试优化的数据加载器\n")
    
    # 测试单次加载
    symbols = ['000001.SZ', '600000.SH', '600519.SH']
    start_date = '2024-01-01'
    end_date = '2024-03-01'
    
    print(f"第一次加载 ({len(symbols)} 只股票)...")
    import time
    start = time.time()
    data1 = loader.get_stock_data(symbols, start_date, end_date)
    elapsed1 = time.time() - start
    print(f"✅ 完成，耗时: {elapsed1:.2f}秒")
    print(f"   数据shape: {data1.shape}")
    
    print(f"\n第二次加载（应该命中缓存）...")
    start = time.time()
    data2 = loader.get_stock_data(symbols, start_date, end_date)
    elapsed2 = time.time() - start
    print(f"✅ 完成，耗时: {elapsed2:.2f}秒")
    print(f"   加速比: {elapsed1/elapsed2:.1f}x")
    
    # 缓存统计
    print("\n📈 缓存统计:")
    stats = loader.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 清除缓存
    print("\n🧹 清除缓存...")
    loader.clear_cache()
    print("✅ 完成")
