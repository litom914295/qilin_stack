"""
特征缓存系统
提供特征计算结果的磁盘缓存和智能失效机制

功能：
- 特征计算结果缓存
- 基于时间和版本的缓存失效
- 缓存命中率统计
- 自动清理过期缓存
- 支持多种缓存后端
"""

import os
import pickle
import json
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import time

import pandas as pd


class FeatureCache:
    """
    特征缓存管理器
    提供特征计算结果的高效缓存和检索
    """
    
    def __init__(
        self,
        cache_dir: str = "./feature_cache",
        ttl_hours: int = 24,  # 缓存有效期（小时）
        max_cache_size_gb: float = 10.0,  # 最大缓存大小（GB）
        enable_compression: bool = True
    ):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            ttl_hours: 缓存有效期（小时）
            max_cache_size_gb: 最大缓存大小（GB）
            enable_compression: 是否启用压缩
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ttl_hours = ttl_hours
        self.max_cache_size_gb = max_cache_size_gb
        self.enable_compression = enable_compression
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0
        }
        
        # 元数据文件
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载缓存元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_key(
        self,
        feature_name: str,
        params: Optional[Dict] = None,
        version: str = "v1"
    ) -> str:
        """
        计算缓存键
        
        Args:
            feature_name: 特征名称
            params: 参数字典
            version: 版本号
        
        Returns:
            缓存键
        """
        # 构建键字符串
        key_parts = [feature_name, version]
        
        if params:
            # 对参数排序以确保一致性
            param_str = json.dumps(params, sort_keys=True)
            key_parts.append(param_str)
        
        key_string = '|'.join(key_parts)
        
        # 使用hash避免过长的文件名
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{feature_name}_{key_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_expired(self, cache_key: str) -> bool:
        """
        检查缓存是否过期
        
        Args:
            cache_key: 缓存键
        
        Returns:
            是否过期
        """
        if cache_key not in self.metadata:
            return True
        
        cache_info = self.metadata[cache_key]
        created_time = datetime.fromisoformat(cache_info['created_at'])
        
        age = datetime.now() - created_time
        
        return age.total_seconds() > self.ttl_hours * 3600
    
    def get(
        self,
        feature_name: str,
        params: Optional[Dict] = None,
        version: str = "v1"
    ) -> Optional[Any]:
        """
        从缓存获取特征
        
        Args:
            feature_name: 特征名称
            params: 参数字典
            version: 版本号
        
        Returns:
            缓存的特征数据，如果不存在或过期则返回None
        """
        cache_key = self._compute_key(feature_name, params, version)
        cache_path = self._get_cache_path(cache_key)
        
        # 检查缓存是否存在
        if not cache_path.exists():
            self.stats['misses'] += 1
            return None
        
        # 检查是否过期
        if self._is_expired(cache_key):
            self.stats['misses'] += 1
            # 删除过期缓存
            self._delete_cache(cache_key)
            return None
        
        # 加载缓存
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            self.stats['hits'] += 1
            
            # 更新访问时间
            if cache_key in self.metadata:
                self.metadata[cache_key]['last_accessed'] = datetime.now().isoformat()
                self._save_metadata()
            
            return data
        
        except Exception as e:
            warnings.warn(f"Failed to load cache {cache_key}: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(
        self,
        feature_name: str,
        data: Any,
        params: Optional[Dict] = None,
        version: str = "v1"
    ):
        """
        设置缓存
        
        Args:
            feature_name: 特征名称
            data: 特征数据
            params: 参数字典
            version: 版本号
        """
        cache_key = self._compute_key(feature_name, params, version)
        cache_path = self._get_cache_path(cache_key)
        
        # 检查缓存大小限制
        self._check_cache_size()
        
        # 保存缓存
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 更新元数据
            file_size = cache_path.stat().st_size
            self.metadata[cache_key] = {
                'feature_name': feature_name,
                'params': params,
                'version': version,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'size_bytes': file_size
            }
            self._save_metadata()
            
            self.stats['writes'] += 1
        
        except Exception as e:
            warnings.warn(f"Failed to save cache {cache_key}: {e}")
    
    def _delete_cache(self, cache_key: str):
        """删除指定缓存"""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            cache_path.unlink()
        
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
        
        self.stats['evictions'] += 1
    
    def _check_cache_size(self):
        """检查并控制缓存大小"""
        # 计算当前缓存总大小
        total_size = sum(
            info['size_bytes'] for info in self.metadata.values()
        )
        
        max_size_bytes = self.max_cache_size_gb * 1024**3
        
        # 如果超过限制，删除最旧的缓存
        if total_size > max_size_bytes:
            # 按最后访问时间排序
            sorted_keys = sorted(
                self.metadata.keys(),
                key=lambda k: self.metadata[k]['last_accessed']
            )
            
            # 删除最旧的10%缓存
            n_to_delete = max(1, len(sorted_keys) // 10)
            
            for cache_key in sorted_keys[:n_to_delete]:
                self._delete_cache(cache_key)
    
    def clear_expired(self):
        """清理所有过期缓存"""
        expired_keys = [
            key for key in self.metadata.keys()
            if self._is_expired(key)
        ]
        
        for key in expired_keys:
            self._delete_cache(key)
        
        return len(expired_keys)
    
    def clear_all(self):
        """清空所有缓存"""
        for cache_key in list(self.metadata.keys()):
            self._delete_cache(cache_key)
        
        self.metadata = {}
        self._save_metadata()
    
    def invalidate(
        self,
        feature_name: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        使指定缓存失效
        
        Args:
            feature_name: 特征名称（如果为None则失效所有）
            version: 版本号（如果为None则失效所有版本）
        """
        keys_to_delete = []
        
        for cache_key, info in self.metadata.items():
            if feature_name and info['feature_name'] != feature_name:
                continue
            if version and info['version'] != version:
                continue
            keys_to_delete.append(cache_key)
        
        for key in keys_to_delete:
            self._delete_cache(key)
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        total_size = sum(
            info['size_bytes'] for info in self.metadata.values()
        )
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'writes': self.stats['writes'],
            'evictions': self.stats['evictions'],
            'cached_items': len(self.metadata),
            'total_size_mb': total_size / (1024**2),
            'total_size_gb': total_size / (1024**3)
        }
    
    def list_cached_features(self) -> pd.DataFrame:
        """列出所有缓存的特征"""
        records = []
        
        for cache_key, info in self.metadata.items():
            records.append({
                'cache_key': cache_key,
                'feature_name': info['feature_name'],
                'version': info['version'],
                'created_at': info['created_at'],
                'last_accessed': info['last_accessed'],
                'size_mb': info['size_bytes'] / (1024**2),
                'is_expired': self._is_expired(cache_key)
            })
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            df = df.sort_values('last_accessed', ascending=False)
        
        return df
    
    def cached(
        self,
        feature_name: str,
        params: Optional[Dict] = None,
        version: str = "v1"
    ):
        """
        装饰器：自动缓存函数结果
        
        使用示例:
            @cache.cached("my_feature", version="v1")
            def compute_feature(data):
                # 耗时计算
                return result
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # 尝试从缓存获取
                cached_result = self.get(feature_name, params, version)
                
                if cached_result is not None:
                    return cached_result
                
                # 计算结果
                result = func(*args, **kwargs)
                
                # 缓存结果
                self.set(feature_name, result, params, version)
                
                return result
            
            return wrapper
        return decorator


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试特征缓存系统"""
    
    print("=" * 60)
    print("特征缓存系统测试")
    print("=" * 60)
    
    # 创建缓存管理器
    cache = FeatureCache(
        cache_dir="./test_feature_cache",
        ttl_hours=1,
        max_cache_size_gb=1.0
    )
    
    print("✓ 缓存管理器已创建")
    
    # 测试基本缓存操作
    print("\n" + "=" * 60)
    print("测试基本缓存操作...")
    print("=" * 60)
    
    # 创建测试数据
    import numpy as np
    test_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'label': np.random.randint(0, 2, 1000)
    })
    
    feature_name = "test_feature"
    params = {'window': 20, 'method': 'mean'}
    
    # 第一次：缓存未命中
    print(f"\n第一次获取 {feature_name}...")
    result = cache.get(feature_name, params)
    print(f"结果: {result} (期望为None)")
    
    # 设置缓存
    print(f"\n设置缓存...")
    cache.set(feature_name, test_data, params)
    print("✓ 缓存已设置")
    
    # 第二次：缓存命中
    print(f"\n第二次获取 {feature_name}...")
    result = cache.get(feature_name, params)
    print(f"结果: DataFrame shape = {result.shape} (期望为(1000, 3))")
    
    # 测试缓存统计
    print("\n" + "=" * 60)
    print("缓存统计信息:")
    print("=" * 60)
    
    stats = cache.get_cache_stats()
    print(f"\n命中次数: {stats['hits']}")
    print(f"未命中次数: {stats['misses']}")
    print(f"命中率: {stats['hit_rate']:.1%}")
    print(f"写入次数: {stats['writes']}")
    print(f"缓存项数: {stats['cached_items']}")
    print(f"缓存大小: {stats['total_size_mb']:.2f} MB")
    
    # 测试装饰器
    print("\n" + "=" * 60)
    print("测试装饰器...")
    print("=" * 60)
    
    @cache.cached("expensive_feature", params={'param1': 10}, version="v1")
    def expensive_computation():
        print("  执行耗时计算...")
        time.sleep(0.5)  # 模拟耗时操作
        return np.random.randn(100, 10)
    
    print("\n第一次调用（计算）:")
    start = time.time()
    result1 = expensive_computation()
    elapsed1 = time.time() - start
    print(f"  耗时: {elapsed1:.2f} 秒")
    
    print("\n第二次调用（从缓存）:")
    start = time.time()
    result2 = expensive_computation()
    elapsed2 = time.time() - start
    print(f"  耗时: {elapsed2:.2f} 秒 (快 {elapsed1/elapsed2:.1f}x)")
    
    # 列出缓存
    print("\n" + "=" * 60)
    print("列出所有缓存:")
    print("=" * 60)
    
    cached_features = cache.list_cached_features()
    print("\n" + cached_features.to_string(index=False))
    
    # 测试缓存失效
    print("\n" + "=" * 60)
    print("测试缓存失效...")
    print("=" * 60)
    
    print(f"\n当前缓存项数: {len(cache.metadata)}")
    
    cache.invalidate(feature_name="test_feature")
    print(f"失效 test_feature 后: {len(cache.metadata)}")
    
    # 清理过期缓存
    print("\n" + "=" * 60)
    print("清理过期缓存...")
    print("=" * 60)
    
    n_expired = cache.clear_expired()
    print(f"清理了 {n_expired} 个过期缓存")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("最终统计:")
    print("=" * 60)
    
    final_stats = cache.get_cache_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
