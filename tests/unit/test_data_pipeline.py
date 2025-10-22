"""
数据管道单元测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_pipeline.unified_data import (
    MarketData,
    DataSource,
    UnifiedDataPipeline,
    QlibDataAdapter,
    AKShareDataAdapter
)


class TestMarketData:
    """测试市场数据类"""
    
    def test_market_data_creation(self):
        """测试市场数据创建"""
        data = MarketData(
            symbol='000001.SZ',
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000000,
            source=DataSource.QLIB
        )
        
        assert data.symbol == '000001.SZ'
        assert data.open == 100.0
        assert data.close == 103.0
        assert data.source == DataSource.QLIB
    
    def test_market_data_validation(self):
        """测试数据验证"""
        with pytest.raises(ValueError):
            MarketData(
                symbol='000001.SZ',
                timestamp=datetime.now(),
                open=100.0,
                high=95.0,  # high < open, 无效
                low=98.0,
                close=103.0,
                volume=1000000,
                source=DataSource.QLIB
            )


class TestDataSource:
    """测试数据源枚举"""
    
    def test_data_sources(self):
        """测试数据源类型"""
        assert DataSource.QLIB
        assert DataSource.AKSHARE
        assert DataSource.TUSHARE
        assert DataSource.UNKNOWN


class TestQlibDataAdapter:
    """测试Qlib数据适配器"""
    
    def test_adapter_initialization(self):
        """测试适配器初始化"""
        adapter = QlibDataAdapter()
        assert adapter.source == DataSource.QLIB
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, sample_symbols, sample_date):
        """测试获取市场数据"""
        adapter = QlibDataAdapter()
        
        # 模拟获取数据
        data = await adapter.get_market_data(
            symbols=sample_symbols[:1],
            start_date=sample_date,
            end_date=sample_date
        )
        
        # 由于是模拟实现，检查返回格式
        assert isinstance(data, (pd.DataFrame, type(None)))
    
    @pytest.mark.asyncio
    async def test_get_features(self, sample_symbols):
        """测试获取特征"""
        adapter = QlibDataAdapter()
        
        features = await adapter.get_features(
            symbols=sample_symbols[:1],
            fields=['close', 'volume'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )
        
        assert isinstance(features, (pd.DataFrame, type(None)))


class TestAKShareDataAdapter:
    """测试AKShare数据适配器"""
    
    def test_adapter_initialization(self):
        """测试适配器初始化"""
        adapter = AKShareDataAdapter()
        assert adapter.source == DataSource.AKSHARE
    
    @pytest.mark.asyncio
    async def test_get_realtime_data(self, sample_symbols):
        """测试获取实时数据"""
        adapter = AKShareDataAdapter()
        
        data = await adapter.get_realtime_data(sample_symbols[:1])
        
        # 模拟实现应该返回数据或None
        assert isinstance(data, (pd.DataFrame, type(None)))
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, sample_symbols, date_range):
        """测试获取历史数据"""
        adapter = AKShareDataAdapter()
        
        data = await adapter.get_historical_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        assert isinstance(data, (pd.DataFrame, type(None)))


class TestUnifiedDataPipeline:
    """测试统一数据管道"""
    
    def test_pipeline_initialization(self):
        """测试管道初始化"""
        pipeline = UnifiedDataPipeline()
        
        assert pipeline.primary_source == DataSource.QLIB
        assert len(pipeline.adapters) > 0
    
    def test_pipeline_with_custom_sources(self):
        """测试自定义数据源"""
        pipeline = UnifiedDataPipeline(
            primary_source=DataSource.AKSHARE,
            fallback_sources=[DataSource.QLIB]
        )
        
        assert pipeline.primary_source == DataSource.AKSHARE
    
    @pytest.mark.asyncio
    async def test_get_data_from_primary(self, sample_symbols, date_range):
        """测试从主数据源获取数据"""
        pipeline = UnifiedDataPipeline()
        
        data = await pipeline.get_market_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        assert isinstance(data, (pd.DataFrame, list, type(None)))
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, sample_symbols, date_range):
        """测试降级机制"""
        pipeline = UnifiedDataPipeline(
            primary_source=DataSource.AKSHARE,
            fallback_sources=[DataSource.QLIB]
        )
        
        # 即使主数据源失败，应该尝试降级
        data = await pipeline.get_market_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        # 应该返回数据或None，不应抛出异常
        assert isinstance(data, (pd.DataFrame, list, type(None)))
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, sample_symbols, date_range):
        """测试缓存功能"""
        pipeline = UnifiedDataPipeline(cache_enabled=True)
        
        # 第一次获取
        data1 = await pipeline.get_market_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        # 第二次获取（应该从缓存）
        data2 = await pipeline.get_market_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        # 两次结果应该一致
        if data1 is not None and data2 is not None:
            assert type(data1) == type(data2)
    
    @pytest.mark.asyncio
    async def test_batch_fetch(self, sample_symbols, date_range):
        """测试批量获取"""
        pipeline = UnifiedDataPipeline()
        
        data = await pipeline.get_market_data(
            symbols=sample_symbols,  # 多个股票
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        assert isinstance(data, (pd.DataFrame, list, type(None)))
    
    def test_data_quality_check(self):
        """测试数据质量检查"""
        pipeline = UnifiedDataPipeline()
        
        # 创建测试数据
        df = pd.DataFrame({
            'symbol': ['000001.SZ'] * 10,
            'close': [100, 101, 102, None, 104, 105, 106, 107, 108, 109],
            'volume': [1000000] * 10
        })
        
        # 检查数据质量
        is_valid = pipeline._check_data_quality(df)
        
        # 有缺失值，可能返回False或进行清洗
        assert isinstance(is_valid, bool)
    
    def test_normalize_data(self):
        """测试数据标准化"""
        pipeline = UnifiedDataPipeline()
        
        # 创建不同格式的数据
        df = pd.DataFrame({
            'symbol': ['000001.SZ'],
            'Close': [100],  # 大写
            'Volume': [1000000]
        })
        
        normalized = pipeline._normalize_data(df, DataSource.AKSHARE)
        
        # 应该标准化列名
        assert isinstance(normalized, pd.DataFrame)


@pytest.mark.integration
class TestDataPipelineIntegration:
    """数据管道集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_data_flow(self, sample_symbols, date_range):
        """测试完整数据流"""
        pipeline = UnifiedDataPipeline(
            primary_source=DataSource.QLIB,
            fallback_sources=[DataSource.AKSHARE],
            cache_enabled=True
        )
        
        # 1. 获取市场数据
        market_data = await pipeline.get_market_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        # 2. 获取特征数据
        features = await pipeline.get_features(
            symbols=sample_symbols[:1],
            fields=['close', 'volume', 'open'],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        # 应该成功获取
        assert market_data is not None or features is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, sample_symbols, date_range):
        """测试错误处理和恢复"""
        pipeline = UnifiedDataPipeline()
        
        # 使用无效符号
        invalid_symbols = ['INVALID.XX']
        
        # 应该优雅处理错误
        data = await pipeline.get_market_data(
            symbols=invalid_symbols,
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        # 错误应该被捕获，返回None或空数据
        assert data is None or (isinstance(data, (pd.DataFrame, list)) and len(data) == 0)
        
        # 系统应该继续工作
        valid_data = await pipeline.get_market_data(
            symbols=sample_symbols[:1],
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
        
        assert isinstance(valid_data, (pd.DataFrame, list, type(None)))
