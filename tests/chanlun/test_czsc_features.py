"""测试CZSC特征提取器"""

import unittest
import pandas as pd
import numpy as np
from features.chanlun.czsc_features import CzscFeatureGenerator

class TestCzscFeatureGenerator(unittest.TestCase):
    
    def setUp(self):
        """准备测试数据"""
        # 生成模拟数据
        dates = pd.date_range('2023-01-01', periods=100)
        np.random.seed(42)
        
        self.df = pd.DataFrame({
            'datetime': dates,
            'open': 10 + np.random.randn(100).cumsum() * 0.5,
            'close': 10 + np.random.randn(100).cumsum() * 0.5,
            'high': 10.5 + np.random.randn(100).cumsum() * 0.5,
            'low': 9.5 + np.random.randn(100).cumsum() * 0.5,
            'volume': np.random.randint(1000, 10000, 100),
        })
        
        self.generator = CzscFeatureGenerator()
    
    def test_generate_features(self):
        """测试特征生成"""
        result = self.generator.generate_features(self.df)
        
        # 检查特征列是否存在
        self.assertIn('fx_mark', result.columns)
        self.assertIn('bi_direction', result.columns)
        self.assertIn('bi_position', result.columns)
        self.assertIn('bi_power', result.columns)
        self.assertIn('in_zs', result.columns)
        self.assertIn('bars_since_fx', result.columns)
        
        # 检查行数不变
        self.assertEqual(len(result), len(self.df))
        
        print(f"✅ 特征生成测试通过")
        print(f"   生成特征数: {len([c for c in result.columns if c.startswith(('fx_', 'bi_', 'in_'))])}")
    
    def test_feature_values(self):
        """测试特征值范围"""
        result = self.generator.generate_features(self.df)
        
        # fx_mark应该在[-1, 0, 1]
        self.assertTrue(result['fx_mark'].isin([-1, 0, 1]).all())
        
        # bi_direction应该在[-1, 0, 1]
        self.assertTrue(result['bi_direction'].isin([-1, 0, 1]).all())
        
        # bi_position应该在[0, 1]
        self.assertTrue((result['bi_position'] >= 0).all())
        self.assertTrue((result['bi_position'] <= 1).all())
        
        print(f"✅ 特征值范围测试通过")
    
    def test_empty_data(self):
        """测试空数据"""
        empty_df = pd.DataFrame(columns=['datetime', 'open', 'close', 'high', 'low', 'volume'])
        result = self.generator.generate_features(empty_df)
        
        self.assertEqual(len(result), 0)
        print(f"✅ 空数据测试通过")
    
    def test_insufficient_data(self):
        """测试数据不足场景"""
        small_df = self.df.head(5)  # 只有5条数据
        result = self.generator.generate_features(small_df)
        
        # 应该返回空特征
        self.assertIn('fx_mark', result.columns)
        self.assertEqual(len(result), 5)
        print(f"✅ 数据不足测试通过")

if __name__ == '__main__':
    unittest.main()
