"""
统一数据流管道
融合Qlib、TradingAgents、RD-Agent的数据需求
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import pickle
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类型定义
# ============================================================================

class DataFrequency(Enum):
    """数据频率"""
    TICK = "tick"           # 逐笔
    MIN_1 = "1min"          # 1分钟
    MIN_5 = "5min"          # 5分钟
    MIN_15 = "15min"        # 15分钟
    MIN_30 = "30min"        # 30分钟
    HOUR_1 = "1hour"        # 1小时
    DAY = "day"             # 日线
    WEEK = "week"           # 周线
    MONTH = "month"         # 月线


class DataSource(Enum):
    """数据源"""
    QLIB = "qlib"
    AKSHARE = "akshare"
    TUSHARE = "tushare"
    JOINQUANT = "joinquant"
    WIND = "wind"
    CUSTOM = "custom"


@dataclass
class MarketData:
    """市场数据标准格式"""
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    frequency: DataFrequency
    source: DataSource
    
    # 扩展字段
    turnover_rate: Optional[float] = None
    vwap: Optional[float] = None
    num_trades: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            'symbol': self.symbol,
            'datetime': self.datetime,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'amount': self.amount,
            'frequency': self.frequency.value,
            'source': self.source.value,
            'turnover_rate': self.turnover_rate,
            'vwap': self.vwap,
            'num_trades': self.num_trades,
            'metadata': self.metadata
        }
    
    def to_series(self) -> pd.Series:
        """转为Series"""
        return pd.Series(self.to_dict())


# ============================================================================
# 数据源抽象接口
# ============================================================================

class DataSourceAdapter(ABC):
    """数据源适配器基类"""
    
    def __init__(self, source: DataSource):
        self.source = source
        self.cache_enabled = True
        self.cache_dir = Path("./cache/data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fetch_bars(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   frequency: DataFrequency = DataFrequency.DAY) -> pd.DataFrame:
        """获取K线数据"""
        pass
    
    @abstractmethod
    def fetch_ticks(self,
                    symbol: str,
                    date: str) -> pd.DataFrame:
        """获取tick数据"""
        pass
    
    @abstractmethod
    def fetch_fundamentals(self,
                          symbols: Union[str, List[str]],
                          date: str) -> pd.DataFrame:
        """获取基本面数据"""
        pass
    
    def _get_cache_key(self, **kwargs) -> str:
        """生成缓存键"""
        return "_".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存加载"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存到缓存"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            data.to_pickle(cache_file)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")


# ============================================================================
# Qlib数据源适配器
# ============================================================================

class QlibDataAdapter(DataSourceAdapter):
    """Qlib数据源适配器"""
    
    def __init__(self):
        super().__init__(DataSource.QLIB)
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保Qlib已初始化"""
        if self._initialized:
            return
        
        try:
            import qlib
            from qlib.config import REG_CN
            
            # 初始化Qlib
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            self._initialized = True
            logger.info("✅ Qlib初始化成功")
        except Exception as e:
            logger.warning(f"Qlib初始化失败: {e}")
    
    def fetch_bars(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   frequency: DataFrequency = DataFrequency.DAY) -> pd.DataFrame:
        """获取K线数据"""
        self._ensure_initialized()
        
        # 检查缓存
        cache_key = self._get_cache_key(
            symbols=str(symbols),
            start=start_date,
            end=end_date,
            freq=frequency.value,
            source="qlib"
        )
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            from qlib.data import D
            
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Qlib字段映射
            fields = ['$open', '$high', '$low', '$close', '$volume', '$amount']
            
            # 获取数据
            data = D.features(
                instruments=symbols,
                fields=fields,
                start_time=start_date,
                end_time=end_date,
                freq=frequency.value if frequency == DataFrequency.DAY else 'day'
            )
            
            # 标准化列名
            data.columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            
            # 保存缓存
            self._save_to_cache(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Qlib数据获取失败: {e}")
            return pd.DataFrame()
    
    def fetch_ticks(self, symbol: str, date: str) -> pd.DataFrame:
        """Qlib不支持tick数据"""
        logger.warning("Qlib不支持tick数据，返回空DataFrame")
        return pd.DataFrame()
    
    def fetch_fundamentals(self,
                          symbols: Union[str, List[str]],
                          date: str) -> pd.DataFrame:
        """获取基本面数据"""
        self._ensure_initialized()
        
        try:
            from qlib.data import D
            
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # 基本面字段
            fields = ['$market_cap', '$pe_ratio', '$pb_ratio', '$ps_ratio']
            
            data = D.features(
                instruments=symbols,
                fields=fields,
                start_time=date,
                end_time=date,
                freq='day'
            )
            
            return data
            
        except Exception as e:
            logger.error(f"基本面数据获取失败: {e}")
            return pd.DataFrame()


# ============================================================================
# AKShare数据源适配器
# ============================================================================

class AKShareDataAdapter(DataSourceAdapter):
    """AKShare数据源适配器"""
    
    def __init__(self):
        super().__init__(DataSource.AKSHARE)
    
    def fetch_bars(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   frequency: DataFrequency = DataFrequency.DAY) -> pd.DataFrame:
        """获取K线数据"""
        cache_key = self._get_cache_key(
            symbols=str(symbols),
            start=start_date,
            end=end_date,
            freq=frequency.value,
            source="akshare"
        )
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            import akshare as ak
            
            if isinstance(symbols, str):
                symbols = [symbols]
            
            all_data = []
            
            for symbol in symbols:
                # AKShare股票代码格式转换
                ak_symbol = symbol.replace('.SH', '').replace('.SZ', '')
                
                # 获取日线数据
                df = ak.stock_zh_a_hist(
                    symbol=ak_symbol,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust="qfq"  # 前复权
                )
                
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                
                # 标准化列名
                result = result.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '换手率': 'turnover_rate'
                })
                
                # 设置索引
                result['date'] = pd.to_datetime(result['date'])
                result = result.set_index(['symbol', 'date'])
                
                self._save_to_cache(cache_key, result)
                return result
            
        except Exception as e:
            logger.error(f"AKShare数据获取失败: {e}")
        
        return pd.DataFrame()
    
    def fetch_ticks(self, symbol: str, date: str) -> pd.DataFrame:
        """获取分时数据"""
        try:
            import akshare as ak
            
            ak_symbol = symbol.replace('.SH', '').replace('.SZ', '')
            
            # 获取分时数据
            df = ak.stock_zh_a_hist_min_em(
                symbol=ak_symbol,
                period='1',
                start_date=date.replace('-', ''),
                end_date=date.replace('-', '')
            )
            
            return df
            
        except Exception as e:
            logger.error(f"AKShare分时数据获取失败: {e}")
            return pd.DataFrame()
    
    def fetch_fundamentals(self,
                          symbols: Union[str, List[str]],
                          date: str) -> pd.DataFrame:
        """获取基本面数据"""
        try:
            import akshare as ak
            
            # 获取A股实时行情
            df = ak.stock_zh_a_spot_em()
            
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # 过滤指定股票
            ak_symbols = [s.replace('.SH', '').replace('.SZ', '') for s in symbols]
            result = df[df['代码'].isin(ak_symbols)]
            
            return result
            
        except Exception as e:
            logger.error(f"基本面数据获取失败: {e}")
            return pd.DataFrame()


# ============================================================================
# 统一数据管道
# ============================================================================

class UnifiedDataPipeline:
    """统一数据管道"""
    
    def __init__(self):
        self.adapters: Dict[DataSource, DataSourceAdapter] = {}
        self.primary_source = DataSource.QLIB
        self.fallback_sources = [DataSource.AKSHARE, DataSource.TUSHARE]
        
        # 初始化适配器
        self._init_adapters()
    
    def _init_adapters(self):
        """初始化数据源适配器"""
        # Qlib适配器
        try:
            self.adapters[DataSource.QLIB] = QlibDataAdapter()
            logger.info("✅ Qlib适配器初始化成功")
        except Exception as e:
            logger.warning(f"Qlib适配器初始化失败: {e}")
        
        # AKShare适配器
        try:
            self.adapters[DataSource.AKSHARE] = AKShareDataAdapter()
            logger.info("✅ AKShare适配器初始化成功")
        except Exception as e:
            logger.warning(f"AKShare适配器初始化失败: {e}")
    
    def get_bars(self,
                 symbols: Union[str, List[str]],
                 start_date: str,
                 end_date: str,
                 frequency: DataFrequency = DataFrequency.DAY,
                 source: Optional[DataSource] = None) -> pd.DataFrame:
        """
        获取K线数据（支持多数据源降级）
        
        Args:
            symbols: 股票代码或列表
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            frequency: 数据频率
            source: 指定数据源（None=自动选择）
        
        Returns:
            标准化的DataFrame
        """
        # 确定数据源优先级
        if source:
            sources = [source]
        else:
            sources = [self.primary_source] + self.fallback_sources
        
        # 尝试从各数据源获取
        for src in sources:
            if src not in self.adapters:
                continue
            
            adapter = self.adapters[src]
            
            try:
                logger.info(f"尝试从 {src.value} 获取数据...")
                data = adapter.fetch_bars(symbols, start_date, end_date, frequency)
                
                if not data.empty:
                    logger.info(f"✅ 从 {src.value} 获取到 {len(data)} 条数据")
                    return data
                
            except Exception as e:
                logger.warning(f"{src.value} 数据获取失败: {e}")
        
        logger.error("所有数据源都获取失败")
        return pd.DataFrame()
    
    def get_ticks(self,
                  symbol: str,
                  date: str,
                  source: Optional[DataSource] = None) -> pd.DataFrame:
        """获取tick数据"""
        if source and source in self.adapters:
            return self.adapters[source].fetch_ticks(symbol, date)
        
        # 优先使用AKShare（Qlib不支持tick）
        if DataSource.AKSHARE in self.adapters:
            return self.adapters[DataSource.AKSHARE].fetch_ticks(symbol, date)
        
        return pd.DataFrame()
    
    def get_fundamentals(self,
                        symbols: Union[str, List[str]],
                        date: str,
                        source: Optional[DataSource] = None) -> pd.DataFrame:
        """获取基本面数据"""
        sources = [source] if source else [self.primary_source] + self.fallback_sources
        
        for src in sources:
            if src not in self.adapters:
                continue
            
            try:
                data = self.adapters[src].fetch_fundamentals(symbols, date)
                if not data.empty:
                    return data
            except Exception as e:
                logger.warning(f"{src.value} 基本面获取失败: {e}")
        
        return pd.DataFrame()
    
    def get_realtime_quote(self, symbols: Union[str, List[str]]) -> pd.DataFrame:
        """获取实时行情"""
        # 使用AKShare获取实时数据
        if DataSource.AKSHARE in self.adapters:
            adapter = self.adapters[DataSource.AKSHARE]
            return adapter.fetch_fundamentals(symbols, datetime.now().strftime('%Y-%m-%d'))
        
        return pd.DataFrame()
    
    def get_available_sources(self) -> List[DataSource]:
        """获取可用数据源列表"""
        return list(self.adapters.keys())
    
    def test_connectivity(self) -> Dict[str, bool]:
        """测试数据源连通性"""
        results = {}
        
        for source, adapter in self.adapters.items():
            try:
                # 简单测试：获取一个股票的最近一天数据
                test_data = adapter.fetch_bars(
                    '000001.SZ',
                    '2024-01-01',
                    '2024-01-02',
                    DataFrequency.DAY
                )
                results[source.value] = not test_data.empty
            except Exception as e:
                logger.error(f"{source.value} 连通性测试失败: {e}")
                results[source.value] = False
        
        return results


# ============================================================================
# 工厂函数
# ============================================================================

_pipeline_instance = None

def get_unified_pipeline() -> UnifiedDataPipeline:
    """获取统一数据管道单例"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = UnifiedDataPipeline()
    return _pipeline_instance


# ============================================================================
# 测试
# ============================================================================

def test_unified_pipeline():
    """测试统一数据管道"""
    print("=== 统一数据管道测试 ===\n")
    
    pipeline = get_unified_pipeline()
    
    # 测试连通性
    print("1️⃣ 测试数据源连通性:")
    connectivity = pipeline.test_connectivity()
    for source, status in connectivity.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {source}: {'可用' if status else '不可用'}")
    
    # 测试获取K线数据
    print("\n2️⃣ 测试获取K线数据:")
    data = pipeline.get_bars(
        symbols=['000001.SZ', '600000.SH'],
        start_date='2024-01-01',
        end_date='2024-01-10',
        frequency=DataFrequency.DAY
    )
    
    if not data.empty:
        print(f"  获取到 {len(data)} 条数据")
        print(f"  数据列: {list(data.columns)}")
        print(f"\n  前5条数据:")
        print(data.head())
    else:
        print("  ❌ 未获取到数据")
    
    # 测试可用数据源
    print("\n3️⃣ 可用数据源:")
    sources = pipeline.get_available_sources()
    for src in sources:
        print(f"  - {src.value}")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_unified_pipeline()
