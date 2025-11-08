"""
配置管理和缓存系统单元测试
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager, get_config
from cache.feature_cache import FeatureCache


class TestConfigManager(unittest.TestCase):
    """配置管理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.config)
        self.assertIsNotNone(self.config.config)
        self.assertIn('system', self.config.config)
        self.assertIn('model', self.config.config)
    
    def test_get_config_value(self):
        """测试获取配置值"""
        # 测试存在的配置
        project_name = self.config.get('system.project_name')
        self.assertEqual(project_name, 'qilin_limitup_ai')
        
        # 测试不存在的配置
        unknown = self.config.get('unknown.key', 'default')
        self.assertEqual(unknown, 'default')
    
    def test_set_config_value(self):
        """测试设置配置值"""
        # 设置新值
        self.config.set('model.learning_rate', 0.08)
        self.assertEqual(self.config.get('model.learning_rate'), 0.08)
        
        # 设置嵌套新值
        self.config.set('custom.nested.value', 'test')
        self.assertEqual(self.config.get('custom.nested.value'), 'test')
    
    def test_get_section(self):
        """测试获取配置节"""
        model_config = self.config.get_section('model')
        
        self.assertIsInstance(model_config, dict)
        self.assertIn('model_type', model_config)
        self.assertIn('learning_rate', model_config)
    
    def test_validate(self):
        """测试配置验证"""
        is_valid = self.config.validate()
        self.assertTrue(is_valid)
        
        # 测试无效配置
        invalid_config = ConfigManager()
        invalid_config.config['model']['learning_rate'] = -0.1
        is_valid = invalid_config.validate()
        self.assertFalse(is_valid)
    
    def test_save_and_load(self):
        """测试保存和加载配置"""
        # 修改配置
        self.config.set('model.learning_rate', 0.12)
        
        # 保存
        self.config.save_to_file(self.config_file)
        self.assertTrue(os.path.exists(self.config_file))
        
        # 加载
        loaded_config = ConfigManager(config_file=self.config_file)
        self.assertEqual(loaded_config.get('model.learning_rate'), 0.12)
    
    def test_type_conversion(self):
        """测试类型转换"""
        # 布尔值
        self.assertEqual(self.config._convert_type('true'), True)
        self.assertEqual(self.config._convert_type('false'), False)
        
        # 整数
        self.assertEqual(self.config._convert_type('123'), 123)
        
        # 浮点数
        self.assertEqual(self.config._convert_type('1.23'), 1.23)
        
        # 字符串
        self.assertEqual(self.config._convert_type('test'), 'test')
    
    def test_env_override(self):
        """测试环境变量覆盖"""
        # 设置环境变量
        os.environ['QILIN_MODEL_LEARNING_RATE'] = '0.15'
        os.environ['QILIN_SYSTEM_DEBUG'] = 'true'
        
        # 创建新配置
        config = ConfigManager(override_from_env=True)
        
        # 验证覆盖
        self.assertEqual(config.get('model.learning_rate'), 0.15)
        self.assertEqual(config.get('system.debug'), True)
        
        # 清理环境变量
        del os.environ['QILIN_MODEL_LEARNING_RATE']
        del os.environ['QILIN_SYSTEM_DEBUG']
    
    def test_global_config_singleton(self):
        """测试全局配置单例"""
        config1 = get_config()
        config2 = get_config()
        
        # 验证是同一实例
        self.assertIs(config1, config2)
        
        # 验证reload
        config3 = get_config(reload=True)
        self.assertIsNot(config1, config3)


class TestFeatureCache(unittest.TestCase):
    """特征缓存测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = FeatureCache(
            cache_dir=self.temp_dir,
            ttl_hours=1,
            max_cache_size_gb=0.1
        )
        
        # 测试数据
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.cache)
        self.assertTrue(self.cache.cache_dir.exists())
        self.assertEqual(self.cache.ttl_hours, 1)
    
    def test_set_and_get(self):
        """测试设置和获取缓存"""
        # 第一次获取（缺失）
        result = self.cache.get('test_feature', {'param': 1}, 'v1')
        self.assertIsNone(result)
        
        # 设置缓存
        self.cache.set('test_feature', self.test_data, {'param': 1}, 'v1')
        
        # 第二次获取（命中）
        result = self.cache.get('test_feature', {'param': 1}, 'v1')
        self.assertIsNotNone(result)
        self.assertTrue(result.equals(self.test_data))
    
    def test_cache_key_computation(self):
        """测试缓存键计算"""
        key1 = self.cache._compute_key('feature', {'a': 1, 'b': 2}, 'v1')
        key2 = self.cache._compute_key('feature', {'b': 2, 'a': 1}, 'v1')
        
        # 参数顺序不同但键相同
        self.assertEqual(key1, key2)
        
        # 不同参数的键不同
        key3 = self.cache._compute_key('feature', {'a': 1, 'b': 3}, 'v1')
        self.assertNotEqual(key1, key3)
    
    def test_cache_stats(self):
        """测试缓存统计"""
        # 初始统计
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        
        # 设置和获取
        self.cache.set('feature1', self.test_data, version='v1')
        self.cache.get('feature1', version='v1')  # 命中
        self.cache.get('feature2', version='v1')  # 未命中
        
        # 验证统计
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['hit_rate'], 0)
    
    def test_invalidate(self):
        """测试缓存失效"""
        # 设置多个缓存
        self.cache.set('feature1', self.test_data, version='v1')
        self.cache.set('feature1', self.test_data, version='v2')
        self.cache.set('feature2', self.test_data, version='v1')
        
        # 失效特定特征
        self.cache.invalidate(feature_name='feature1')
        
        # 验证失效
        self.assertIsNone(self.cache.get('feature1', version='v1'))
        self.assertIsNotNone(self.cache.get('feature2', version='v1'))
    
    def test_clear_all(self):
        """测试清空所有缓存"""
        # 设置缓存
        self.cache.set('feature1', self.test_data)
        self.cache.set('feature2', self.test_data)
        
        # 清空
        self.cache.clear_all()
        
        # 验证清空
        self.assertEqual(len(self.cache.metadata), 0)
        self.assertIsNone(self.cache.get('feature1'))
    
    def test_list_cached_features(self):
        """测试列出缓存特征"""
        # 设置缓存
        self.cache.set('feature1', self.test_data, version='v1')
        self.cache.set('feature2', self.test_data, version='v2')
        
        # 列出
        cached_list = self.cache.list_cached_features()
        
        # 验证
        self.assertIsInstance(cached_list, pd.DataFrame)
        self.assertEqual(len(cached_list), 2)
        self.assertIn('feature_name', cached_list.columns)
        self.assertIn('version', cached_list.columns)
    
    def test_cached_decorator(self):
        """测试缓存装饰器"""
        call_count = [0]  # 使用列表以便在闭包中修改
        
        @self.cache.cached('expensive_feature', version='v1')
        def expensive_function():
            call_count[0] += 1
            return self.test_data.copy()
        
        # 第一次调用（执行函数）
        result1 = expensive_function()
        self.assertEqual(call_count[0], 1)
        
        # 第二次调用（从缓存）
        result2 = expensive_function()
        self.assertEqual(call_count[0], 1)  # 调用次数未增加
        
        # 验证结果相同
        self.assertTrue(result1.equals(result2))


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_config_with_cache(self):
        """测试配置与缓存集成"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 从配置创建缓存
            config = ConfigManager()
            cache_dir = config.get('cache.cache_dir', temp_dir)
            ttl_hours = config.get('cache.ttl_hours', 24)
            
            cache = FeatureCache(
                cache_dir=cache_dir,
                ttl_hours=ttl_hours
            )
            
            # 验证缓存配置
            self.assertEqual(cache.ttl_hours, ttl_hours)
            
            # 使用缓存
            test_data = pd.DataFrame({'col': [1, 2, 3]})
            cache.set('test', test_data)
            result = cache.get('test')
            
            self.assertIsNotNone(result)
            self.assertTrue(result.equals(test_data))
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
