"""
Qlib多数据源集成
支持Qlib、Yahoo Finance、Tushare、AKShare等多个数据源
实现自动降级和数据融合
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# 数据源枚举
# ============================================================================

class DataSource(Enum):
    """数据源类型"""
    QLIB = "qlib"
    YAHOO = "yahoo"
    TUSHARE = "tushare"
    AKSHARE = "akshare"


@dataclass
class DataSourceStatus:
    """数据源状态"""
    source: DataSource
    available: bool
    latency_ms: float
    last_check: datetime
    error_message: str = ""


# ============================================================================
# 多数据源管理器
# ============================================================================

class MultiSourceDataProvider:
    """多数据源数据提供者"""
    
    def __init__(self, 
                 primary_source: DataSource = DataSource.QLIB,
                 fallback_sources: List[DataSource] = None,
                 auto_fallback: bool = True):
        """
        初始化多数据源提供者
        
        Args:
            primary_source: 主数据源
            fallback_sources: 备用数据源列表
            auto_fallback: 是否自动降级
        """
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources or [
            DataSource.AKSHARE,
            DataSource.YAHOO,
            DataSource.TUSHARE
        ]
        self.auto_fallback = auto_fallback
        
        # 初始化数据源适配器
        self.adapters = {}
        self._init_adapters()
        
        # 数据源状态
        self.source_status = {}
        
        logger.info(f"多数据源初始化: 主={primary_source.value}, 备用={[s.value for s in self.fallback_sources]}")
    
    def _init_adapters(self):
        """初始化各个数据源适配器"""
        # Qlib适配器
        try:
            self.adapters[DataSource.QLIB] = QlibAdapter()
        except Exception as e:
            logger.warning(f"Qlib适配器初始化失败: {e}")
        
        # AKShare适配器
        try:
            self.adapters[DataSource.AKSHARE] = AKShareAdapter()
        except Exception as e:
            logger.warning(f"AKShare适配器初始化失败: {e}")
        
        # Yahoo适配器
        try:
            self.adapters[DataSource.YAHOO] = YahooAdapter()
        except Exception as e:
            logger.warning(f"Yahoo适配器初始化失败: {e}")
        
        # Tushare适配器
        try:
            self.adapters[DataSource.TUSHARE] = TushareAdapter()
        except Exception as e:
            logger.warning(f"Tushare适配器初始化失败: {e}")
    
    async def get_data(self, 
                      symbols: List[str],
                      start_date: str,
                      end_date: str,
                      fields: List[str] = None) -> pd.DataFrame:
        """
        获取数据（自动降级）
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表
            
        Returns:
            数据DataFrame
        """
        # 尝试主数据源
        try:
            adapter = self.adapters.get(self.primary_source)
            if adapter:
                logger.info(f"使用主数据源: {self.primary_source.value}")
                data = await adapter.fetch(symbols, start_date, end_date, fields)
                
                if data is not None and not data.empty:
                    self._update_status(self.primary_source, True, "")
                    return data
        except Exception as e:
            logger.warning(f"主数据源失败: {e}")
            self._update_status(self.primary_source, False, str(e))
        
        # 自动降级到备用数据源
        if self.auto_fallback:
            for fallback_source in self.fallback_sources:
                try:
                    adapter = self.adapters.get(fallback_source)
                    if adapter:
                        logger.info(f"降级到备用数据源: {fallback_source.value}")
                        data = await adapter.fetch(symbols, start_date, end_date, fields)
                        
                        if data is not None and not data.empty:
                            self._update_status(fallback_source, True, "")
                            return data
                except Exception as e:
                    logger.warning(f"备用数据源{fallback_source.value}失败: {e}")
                    self._update_status(fallback_source, False, str(e))
        
        # 所有数据源都失败
        logger.error("所有数据源都失败")
        return pd.DataFrame()
    
    def _update_status(self, source: DataSource, available: bool, error: str = ""):
        """更新数据源状态"""
        self.source_status[source] = DataSourceStatus(
            source=source,
            available=available,
            latency_ms=0.0,
            last_check=datetime.now(),
            error_message=error
        )
    
    def get_source_status(self) -> Dict[DataSource, DataSourceStatus]:
        """获取所有数据源状态"""
        return self.source_status


# ============================================================================
# Qlib适配器
# ============================================================================

class QlibAdapter:
    """Qlib数据适配器"""
    
    def __init__(self):
        self.provider_uri = "~/.qlib/qlib_data/cn_data"
        self._init_qlib()
    
    def _init_qlib(self):
        """初始化Qlib"""
        try:
            import qlib
            from qlib.config import REG_CN
            qlib.init(provider_uri=self.provider_uri, region=REG_CN)
            logger.info("Qlib初始化成功")
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            raise
    
    async def fetch(self, 
                   symbols: List[str],
                   start_date: str,
                   end_date: str,
                   fields: List[str] = None) -> pd.DataFrame:
        """获取数据"""
        from qlib.data import D
        
        # 默认字段
        if fields is None:
            fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
        
        # 获取数据
        data = D.features(
            instruments=symbols,
            fields=fields,
            start_time=start_date,
            end_time=end_date
        )
        
        return data


# ============================================================================
# AKShare适配器
# ============================================================================

class AKShareAdapter:
    """AKShare数据适配器"""
    
    def __init__(self):
        try:
            import akshare as ak
            self.ak = ak
            logger.info("AKShare初始化成功")
        except ImportError:
            logger.error("AKShare未安装")
            raise
    
    async def fetch(self,
                   symbols: List[str],
                   start_date: str,
                   end_date: str,
                   fields: List[str] = None) -> pd.DataFrame:
        """获取数据"""
        all_data = []
        
        for symbol in symbols:
            # 转换股票代码格式
            stock_code = self._convert_symbol(symbol)
            
            # 获取日线数据
            df = self.ak.stock_zh_a_hist(
                symbol=stock_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )
            
            # 重命名列
            df = self._rename_columns(df)
            df['symbol'] = symbol
            all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码格式"""
        # 000001.SZ -> 000001
        return symbol.split('.')[0]
    
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """重命名列以匹配标准格式"""
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount'
        }
        df = df.rename(columns=column_mapping)
        return df


# ============================================================================
# Yahoo Finance适配器
# ============================================================================

class YahooAdapter:
    """Yahoo Finance数据适配器"""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("Yahoo Finance初始化成功")
        except ImportError:
            logger.error("yfinance未安装")
            raise
    
    async def fetch(self,
                   symbols: List[str],
                   start_date: str,
                   end_date: str,
                   fields: List[str] = None) -> pd.DataFrame:
        """获取数据"""
        # 转换股票代码
        yahoo_symbols = [self._convert_symbol(s) for s in symbols]
        
        # 批量获取
        data = self.yf.download(
            yahoo_symbols,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # 标准化格式
        return self._standardize_format(data, symbols)
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换为Yahoo格式"""
        # 000001.SZ -> 000001.SZ (已经是Yahoo格式)
        # 600000.SH -> 600000.SS (上海交易所)
        if '.SH' in symbol:
            return symbol.replace('.SH', '.SS')
        return symbol
    
    def _standardize_format(self, data: pd.DataFrame, original_symbols: List[str]) -> pd.DataFrame:
        """标准化数据格式"""
        # 实现略
        return data


# ============================================================================
# Tushare适配器
# ============================================================================

class TushareAdapter:
    """Tushare数据适配器"""
    
    def __init__(self):
        try:
            import tushare as ts
            token = os.getenv('TUSHARE_TOKEN', '')
            if not token:
                raise ValueError("TUSHARE_TOKEN环境变量未设置")
            
            self.ts = ts
            self.pro = ts.pro_api(token)
            logger.info("Tushare初始化成功")
        except Exception as e:
            logger.error(f"Tushare初始化失败: {e}")
            raise
    
    async def fetch(self,
                   symbols: List[str],
                   start_date: str,
                   end_date: str,
                   fields: List[str] = None) -> pd.DataFrame:
        """获取数据"""
        all_data = []
        
        for symbol in symbols:
            # 转换格式
            ts_code = self._convert_symbol(symbol)
            
            # 获取数据
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            df['symbol'] = symbol
            all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换为Tushare格式"""
        # 000001.SZ -> 000001.SZ (已经是Tushare格式)
        return symbol


# ============================================================================
# 使用示例
# ============================================================================

async def example_multi_source():
    """多数据源示例"""
    print("=== 多数据源集成示例 ===\n")
    
    provider = MultiSourceDataProvider(
        primary_source=DataSource.QLIB,
        fallback_sources=[DataSource.AKSHARE, DataSource.YAHOO],
        auto_fallback=True
    )
    
    # 获取数据
    data = await provider.get_data(
        symbols=['000001.SZ', '600519.SH'],
        start_date='2024-01-01',
        end_date='2024-06-30'
    )
    
    print(f"获取数据: {len(data)} 行")
    print(f"\n数据源状态:")
    for source, status in provider.get_source_status().items():
        print(f"  {source.value}: {'可用' if status.available else '不可用'}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_multi_source())
