"""
数据质量监控单元测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.data import (
    DataQualityChecker,
    QualityMonitor,
    QualityCheckType,
    SeverityLevel
)

@pytest.mark.unit
class TestDataQualityChecker:
    """数据质量检查器测试"""
    
    @pytest.fixture
    def checker(self):
        """创建检查器实例"""
        return DataQualityChecker()
    
    @pytest.fixture
    def good_data(self):
        """正常的数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(100, 120, 100),
            'low': np.random.uniform(80, 100, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_completeness_check_missing_fields(self, checker):
        """测试完整性检查 - 缺少字段"""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103]
        })
        
        report = checker.check_all(data, 'test_source')
        
        # 应该报告缺少字段
        assert len(report.issues) > 0
        missing_field_issues = [
            i for i in report.issues
            if i.check_type == QualityCheckType.COMPLETENESS
        ]
        assert len(missing_field_issues) > 0
    
    def test_completeness_check_null_values(self, checker):
        """测试完整性检查 - 空值"""
        data = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [101, 102, 103],
            'volume': [1000000, 1100000, 1200000]
        })
        
        report = checker.check_all(data, 'test_source')
        
        # 应该检测到空值
        null_issues = [
            i for i in report.issues
            if 'null' in i.message.lower() or '空值' in i.message
        ]
        assert len(null_issues) > 0
    
    def test_consistency_check_high_low(self, checker):
        """测试一致性检查 - high/low关系"""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 95, 107],  # 第二行high < low
            'low': [95, 96, 97],
            'close': [101, 102, 103],
            'volume': [1000000, 1100000, 1200000]
        })
        
        report = checker.check_all(data, 'test_source')
        
        # 应该检测到high < low的问题
        consistency_issues = [
            i for i in report.issues
            if i.check_type == QualityCheckType.CONSISTENCY
        ]
        assert len(consistency_issues) > 0
    
    def test_accuracy_check_outliers(self, checker, good_data):
        """测试准确性检查 - 异常值"""
        # 添加异常值
        good_data.iloc[50, good_data.columns.get_loc('close')] = 5000
        
        report = checker.check_all(good_data, 'test_source')
        
        # 应该检测到异常值
        accuracy_issues = [
            i for i in report.issues
            if i.check_type == QualityCheckType.ACCURACY
        ]
        assert len(accuracy_issues) > 0
    
    def test_validity_check_negative_values(self, checker):
        """测试有效性检查 - 负值"""
        data = pd.DataFrame({
            'open': [100, -101, 102],  # 负值
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [101, 102, 103],
            'volume': [1000000, 1100000, 1200000]
        })
        
        report = checker.check_all(data, 'test_source')
        
        # 应该检测到负值
        validity_issues = [
            i for i in report.issues
            if i.check_type == QualityCheckType.VALIDITY
        ]
        assert len(validity_issues) > 0
    
    def test_good_data_quality_score(self, checker, good_data):
        """测试正常数据的质量分数"""
        report = checker.check_all(good_data, 'test_source')
        
        # 正常数据应该有高质量分数
        assert report.quality_score >= 80
        assert report.total_records == 100
    
    def test_quality_metrics_calculation(self, checker, good_data):
        """测试质量指标计算"""
        report = checker.check_all(good_data, 'test_source')
        
        # 应该计算各种指标
        assert 'field_completeness' in report.metrics
        assert 'data_completeness' in report.metrics
        assert 'high_low_consistency' in report.metrics
        
        # 完整性应该是1.0
        assert report.metrics['field_completeness'] == 1.0
        assert report.metrics['data_completeness'] > 0.99


@pytest.mark.unit
class TestQualityMonitor:
    """质量监控器测试"""
    
    @pytest.fixture
    def monitor(self):
        """创建监控器实例"""
        return QualityMonitor(alert_threshold=80.0)
    
    @pytest.fixture
    def good_data(self):
        """正常数据"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(100, 120, 50),
            'low': np.random.uniform(80, 100, 50),
            'close': np.random.uniform(90, 110, 50),
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
    
    @pytest.fixture
    def bad_data(self):
        """质量差的数据"""
        return pd.DataFrame({
            'open': [100, -50, np.nan],  # 负值和空值
            'high': [105, 40, 107],      # high < low
            'low': [95, 45, 97],
            'close': [101, 42, 103],
            'volume': [1000000, -1000, 1200000]  # 负值
        })
    
    def test_monitor_good_data(self, monitor, good_data):
        """测试监控正常数据"""
        report, needs_alert = monitor.monitor(good_data, 'test_source')
        
        assert report is not None
        assert not needs_alert  # 正常数据不应该触发告警
        assert len(monitor.history) == 1
    
    def test_monitor_bad_data(self, monitor, bad_data):
        """测试监控低质量数据"""
        report, needs_alert = monitor.monitor(bad_data, 'test_source')
        
        assert report is not None
        assert needs_alert  # 低质量数据应该触发告警
        assert report.quality_score < 80
    
    def test_quality_trend(self, monitor, good_data, bad_data):
        """测试质量趋势"""
        # 监控多次
        monitor.monitor(good_data, 'source1')
        monitor.monitor(bad_data, 'source2')
        monitor.monitor(good_data, 'source1')
        
        # 获取趋势
        trend = monitor.get_quality_trend()
        
        assert not trend.empty
        assert len(trend) == 3
        assert 'quality_score' in trend.columns
    
    def test_generate_summary(self, monitor, good_data):
        """测试生成摘要"""
        # 添加一些历史记录
        for _ in range(5):
            monitor.monitor(good_data, 'test_source')
        
        summary = monitor.generate_summary()
        
        assert 'status' in summary
        assert 'recent_avg_score' in summary
        assert 'total_issues' in summary
        assert 'severity_distribution' in summary
        assert summary['reports_count'] == 5


@pytest.mark.integration
class TestDataQualityIntegration:
    """数据质量集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 1. 创建监控器
        monitor = QualityMonitor(alert_threshold=75.0)
        
        # 2. 准备测试数据
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 30),
            'high': np.random.uniform(100, 120, 30),
            'low': np.random.uniform(80, 100, 30),
            'close': np.random.uniform(90, 110, 30),
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)
        
        # 3. 执行监控
        report, needs_alert = monitor.monitor(data, 'integration_test')
        
        # 4. 验证报告
        assert report is not None
        assert report.total_records == 30
        assert report.quality_score >= 0
        assert report.quality_score <= 100
        
        # 5. 检查指标
        assert 'field_completeness' in report.metrics
        assert 'data_completeness' in report.metrics
        
        # 6. 生成摘要
        summary = monitor.generate_summary()
        assert summary['status'] in ['ok', 'degraded', 'no_data']
        
        # 7. 获取趋势
        trend = monitor.get_quality_trend()
        assert not trend.empty
    
    def test_multiple_data_sources(self):
        """测试多数据源监控"""
        monitor = QualityMonitor()
        
        # 创建不同数据源的数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        for source in ['source_a', 'source_b', 'source_c']:
            data = pd.DataFrame({
                'open': np.random.uniform(90, 110, 20),
                'high': np.random.uniform(100, 120, 20),
                'low': np.random.uniform(80, 100, 20),
                'close': np.random.uniform(90, 110, 20),
                'volume': np.random.randint(1000000, 5000000, 20)
            }, index=dates)
            
            monitor.monitor(data, source)
        
        # 验证所有数据源都被监控
        assert len(monitor.history) == 3
        
        # 获取特定数据源的趋势
        trend_a = monitor.get_quality_trend(data_source='source_a')
        assert not trend_a.empty
        assert (trend_a['data_source'] == 'source_a').all()
    
    def test_quality_degradation_detection(self):
        """测试质量下降检测"""
        monitor = QualityMonitor(alert_threshold=80.0)
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # 先添加高质量数据
        good_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(100, 120, 50),
            'low': np.random.uniform(80, 100, 50),
            'close': np.random.uniform(90, 110, 50),
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        report1, alert1 = monitor.monitor(good_data, 'test')
        assert not alert1
        
        # 然后添加低质量数据
        bad_data = pd.DataFrame({
            'open': [100, -50, np.nan],
            'high': [105, 40, 107],
            'low': [95, 45, 97],
            'close': [101, 42, 103],
            'volume': [1000000, -1000, 1200000]
        })
        
        report2, alert2 = monitor.monitor(bad_data, 'test')
        assert alert2
        
        # 质量应该明显下降
        assert report2.quality_score < report1.quality_score
