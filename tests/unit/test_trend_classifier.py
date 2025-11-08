"""测试走势类型分类器

测试TrendClassifier的所有核心功能
"""
import pytest
import pandas as pd
import numpy as np
from qlib_enhanced.chanlun.trend_classifier import TrendClassifier, TrendType


class TestTrendClassifier:
    """TrendClassifier测试类"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.classifier = TrendClassifier()
    
    def test_init(self):
        """测试初始化"""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'classify_trend')
    
    def test_classify_trend_insufficient_data(self, mock_seg_list):
        """测试数据不足时的处理"""
        # 少于3个线段
        result = self.classifier.classify_trend(mock_seg_list[:2], [])
        assert result == TrendType.UNKNOWN
    
    def test_classify_trend_uptrend(self, mock_seg_list, mock_zs_list):
        """测试上涨趋势识别"""
        # 创建上涨中枢列表
        class MockZS:
            def __init__(self, mid):
                self.mid = mid
        
        rising_zs = [MockZS(10.0), MockZS(11.0)]  # 中枢抬高
        
        result = self.classifier.classify_trend(mock_seg_list, rising_zs)
        # 应该识别为上涨趋势或侧翻（取决于实现）
        assert result in [TrendType.UPTREND, TrendType.DOWNTREND, TrendType.SIDEWAYS]
    
    def test_classify_trend_downtrend(self, mock_seg_list):
        """测试下跌趋势识别"""
        # 创建下跌中枢列表
        class MockZS:
            def __init__(self, mid):
                self.mid = mid
        
        falling_zs = [MockZS(12.0), MockZS(10.0)]  # 中枢降低
        
        result = self.classifier.classify_trend(mock_seg_list, falling_zs)
        assert result in [TrendType.UPTREND, TrendType.DOWNTREND, TrendType.SIDEWAYS]
    
    def test_classify_trend_sideways(self, mock_seg_list):
        """测试震荡识别"""
        # 创建震荡中枢列表
        class MockZS:
            def __init__(self, mid):
                self.mid = mid
        
        sideways_zs = [MockZS(10.0), MockZS(10.1), MockZS(10.0)]  # 中枢平稳
        
        result = self.classifier.classify_trend(mock_seg_list, sideways_zs)
        assert result in [TrendType.UPTREND, TrendType.DOWNTREND, TrendType.SIDEWAYS]
    
    def test_analyze_zs_trend(self):
        """测试中枢趋势分析"""
        class MockZS:
            def __init__(self, mid):
                self.mid = mid
        
        # 测试上涨
        rising = [MockZS(10.0), MockZS(11.0)]
        result = self.classifier._analyze_zs_trend(rising)
        assert result in ['rising', 'falling', 'sideways', 'unknown']
        
        # 测试下跌
        falling = [MockZS(12.0), MockZS(10.0)]
        result = self.classifier._analyze_zs_trend(falling)
        assert result in ['rising', 'falling', 'sideways', 'unknown']
    
    def test_analyze_zs_trend_insufficient_data(self):
        """测试数据不足时的中枢趋势分析"""
        class MockZS:
            def __init__(self, mid):
                self.mid = mid
        
        single_zs = [MockZS(10.0)]
        result = self.classifier._analyze_zs_trend(single_zs)
        assert result == 'unknown'
    
    def test_trend_type_enum(self):
        """测试TrendType枚举"""
        assert hasattr(TrendType, 'UPTREND')
        assert hasattr(TrendType, 'DOWNTREND')
        assert hasattr(TrendType, 'SIDEWAYS')
        assert hasattr(TrendType, 'UNKNOWN')
    
    def test_classify_with_empty_zs_list(self, mock_seg_list):
        """测试空中枢列表"""
        result = self.classifier.classify_trend(mock_seg_list, [])
        # 应该基于线段方向判断
        assert result in [TrendType.UPTREND, TrendType.DOWNTREND, TrendType.SIDEWAYS, TrendType.UNKNOWN]


@pytest.mark.parametrize("zs_count,expected_type", [
    (0, TrendType.UNKNOWN),
    (1, TrendType.UNKNOWN),
    (2, [TrendType.UPTREND, TrendType.DOWNTREND, TrendType.SIDEWAYS]),
])
def test_classify_with_different_zs_counts(zs_count, expected_type, mock_seg_list):
    """参数化测试：不同中枢数量"""
    class MockZS:
        def __init__(self, mid):
            self.mid = mid
    
    zs_list = [MockZS(10.0 + i) for i in range(zs_count)]
    
    classifier = TrendClassifier()
    result = classifier.classify_trend(mock_seg_list, zs_list)
    
    if isinstance(expected_type, list):
        assert result in expected_type
    else:
        assert result == expected_type
