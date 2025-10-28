"""
Qlib多数据源支持模块
支持从多个数据源获取行情数据: Yahoo Finance, Tushare, AKShare, CSV等
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 数据源基类
# ============================================================================

class DataSource(ABC):
    """数据源基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.cache = {}
        self.available = self._check_availability()
    
    @abstractmethod
    def _check_availability(self) -> bool:
        """检查数据源是否可用"""
        pass
    
    @abstractmethod
    def fetch_data(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取数据
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            fields: 字段列表
            
        Returns:
            数据DataFrame
        """
        pass
    
    def clear_cache(self):
        """清空缓存"""
        self.cache = {}
        logger.info(f"{self.name} 缓存已清空")


# ============================================================================
# Yahoo Finance 数据源
# ============================================================================

class YahooFinanceProvider(DataSource):
    """Yahoo Finance 数据提供者"""
    
    def __init__(self):
        super().__init__("YahooFinance")
        self.yf = None
        
    def _check_availability(self) -> bool:
        """检查yfinance是否安装"""
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("✅ Yahoo Finance 数据源可用")
            return True
        except ImportError:
            logger.warning("❌ yfinance未安装,Yahoo Finance数据源不可用")
            logger.info("安装: pip install yfinance")
            return False
    
    def fetch_data(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取Yahoo Finance数据"""
        if not self.available:
            raise RuntimeError("Yahoo Finance数据源不可用")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"从Yahoo Finance获取数据: {symbols}, {start_date} to {end_date}")
        
        all_data = []
        for symbol in symbols:
            try:
                # 转换为Yahoo Finance格式
                yf_symbol = self._convert_symbol(symbol)
                
                # 获取数据
                ticker = self.yf.Ticker(yf_symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"未获取到 {symbol} 的数据")
                    continue
                
                # 标准化列名
                df.columns = [col.lower() for col in df.columns]
                df['symbol'] = symbol
                df.reset_index(inplace=True)
                df.rename(columns={'date': 'datetime'}, inplace=True)
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 过滤字段
        if fields:
            available_fields = [f for f in fields if f in result.columns]
            result = result[['symbol', 'datetime'] + available_fields]
        
        return result
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码为Yahoo Finance格式"""
        # SH600000 -> 600000.SS
        # SZ000001 -> 000001.SZ
        if symbol.startswith('SH'):
            return f"{symbol[2:]}.SS"
        elif symbol.startswith('SZ'):
            return f"{symbol[2:]}.SZ"
        return symbol


# ============================================================================
# Tushare 数据源
# ============================================================================

class TushareProvider(DataSource):
    """Tushare 数据提供者"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.ts = None
        super().__init__("Tushare")
        
    def _check_availability(self) -> bool:
        """检查tushare是否安装和配置"""
        try:
            import tushare as ts
            self.ts = ts
            
            if self.token:
                ts.set_token(self.token)
            
            # 测试连接
            pro = ts.pro_api()
            logger.info("✅ Tushare 数据源可用")
            return True
            
        except ImportError:
            logger.warning("❌ tushare未安装,Tushare数据源不可用")
            logger.info("安装: pip install tushare")
            return False
        except Exception as e:
            logger.warning(f"❌ Tushare配置失败: {e}")
            logger.info("请设置Tushare Token")
            return False
    
    def fetch_data(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取Tushare数据"""
        if not self.available:
            raise RuntimeError("Tushare数据源不可用")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"从Tushare获取数据: {symbols}, {start_date} to {end_date}")
        
        pro = self.ts.pro_api()
        all_data = []
        
        for symbol in symbols:
            try:
                # 转换为Tushare格式
                ts_symbol = self._convert_symbol(symbol)
                
                # 获取日线数据
                df = pro.daily(
                    ts_code=ts_symbol,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
                
                if df.empty:
                    logger.warning(f"未获取到 {symbol} 的数据")
                    continue
                
                # 标准化
                df['symbol'] = symbol
                df['datetime'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('datetime')
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 重命名列以匹配Qlib格式
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount'
        }
        
        result.rename(columns=column_mapping, inplace=True)
        
        if fields:
            available_fields = [f for f in fields if f in result.columns]
            result = result[['symbol', 'datetime'] + available_fields]
        
        return result
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换股票代码为Tushare格式"""
        # SH600000 -> 600000.SH
        # SZ000001 -> 000001.SZ
        if symbol.startswith('SH'):
            return f"{symbol[2:]}.SH"
        elif symbol.startswith('SZ'):
            return f"{symbol[2:]}.SZ"
        return symbol


# ============================================================================
# AKShare 数据源
# ============================================================================

class AKShareProvider(DataSource):
    """AKShare 数据提供者"""
    
    def __init__(self):
        super().__init__("AKShare")
        self.ak = None
        
    def _check_availability(self) -> bool:
        """检查akshare是否安装"""
        try:
            import akshare as ak
            self.ak = ak
            logger.info("✅ AKShare 数据源可用")
            return True
        except ImportError:
            logger.warning("❌ akshare未安装,AKShare数据源不可用")
            logger.info("安装: pip install akshare")
            return False
    
    def fetch_data(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取AKShare数据"""
        if not self.available:
            raise RuntimeError("AKShare数据源不可用")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"从AKShare获取数据: {symbols}, {start_date} to {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # 提取代码(去掉前缀)
                code = symbol[2:] if symbol[:2] in ['SH', 'SZ'] else symbol
                
                # 获取历史数据
                df = self.ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust="qfq"  # 前复权
                )
                
                if df.empty:
                    logger.warning(f"未获取到 {symbol} 的数据")
                    continue
                
                # 标准化
                df['symbol'] = symbol
                df.rename(columns={'日期': 'datetime'}, inplace=True)
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 重命名列
        column_mapping = {
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount'
        }
        
        result.rename(columns=column_mapping, inplace=True)
        
        if fields:
            available_fields = [f for f in fields if f in result.columns]
            result = result[['symbol', 'datetime'] + available_fields]
        
        return result


# ============================================================================
# CSV 数据源
# ============================================================================

class CSVProvider(DataSource):
    """CSV文件数据提供者"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        super().__init__("CSV")
        
    def _check_availability(self) -> bool:
        """检查CSV数据目录是否存在"""
        if self.data_dir.exists():
            logger.info(f"✅ CSV 数据源可用: {self.data_dir}")
            return True
        else:
            logger.warning(f"❌ CSV数据目录不存在: {self.data_dir}")
            return False
    
    def fetch_data(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
        """从CSV文件读取数据"""
        if not self.available:
            raise RuntimeError("CSV数据源不可用")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"从CSV读取数据: {symbols}, {start_date} to {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            csv_file = self.data_dir / f"{symbol}.csv"
            
            if not csv_file.exists():
                logger.warning(f"CSV文件不存在: {csv_file}")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # 确保有datetime列
                if 'datetime' not in df.columns and 'date' in df.columns:
                    df.rename(columns={'date': 'datetime'}, inplace=True)
                
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['symbol'] = symbol
                
                # 过滤日期
                mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
                df = df[mask]
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"读取 {symbol} CSV失败: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        if fields:
            available_fields = [f for f in fields if f in result.columns]
            result = result[['symbol', 'datetime'] + available_fields]
        
        return result


# ============================================================================
# 多数据源管理器
# ============================================================================

class MultiSourceProvider:
    """多数据源管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多数据源管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.providers: Dict[str, DataSource] = {}
        
        # 初始化数据源
        self._init_providers()
        
    def _init_providers(self):
        """初始化所有数据源"""
        # Yahoo Finance
        self.providers['yahoo'] = YahooFinanceProvider()
        
        # Tushare
        tushare_token = self.config.get('tushare_token')
        self.providers['tushare'] = TushareProvider(token=tushare_token)
        
        # AKShare
        self.providers['akshare'] = AKShareProvider()
        
        # CSV
        csv_dir = self.config.get('csv_data_dir', 'data/csv')
        self.providers['csv'] = CSVProvider(data_dir=csv_dir)
        
        # 输出可用的数据源
        available = [name for name, provider in self.providers.items() if provider.available]
        logger.info(f"可用数据源: {', '.join(available)}")
    
    def get_provider(self, source: str) -> Optional[DataSource]:
        """
        获取指定数据源
        
        Args:
            source: 数据源名称
            
        Returns:
            数据源实例
        """
        provider = self.providers.get(source)
        if provider and provider.available:
            return provider
        return None
    
    def fetch_data(self,
                   symbols: Union[str, List[str]],
                   start_date: str,
                   end_date: str,
                   source: str = 'auto',
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从指定数据源获取数据
        
        Args:
            symbols: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源 ('auto', 'yahoo', 'tushare', 'akshare', 'csv')
            fields: 字段列表
            
        Returns:
            数据DataFrame
        """
        # 自动选择数据源
        if source == 'auto':
            source = self._select_best_source()
        
        provider = self.get_provider(source)
        
        if provider is None:
            raise RuntimeError(f"数据源 '{source}' 不可用")
        
        return provider.fetch_data(symbols, start_date, end_date, fields)
    
    def _select_best_source(self) -> str:
        """自动选择最佳数据源"""
        # 优先级: Tushare > AKShare > Yahoo Finance > CSV
        priority = ['tushare', 'akshare', 'yahoo', 'csv']
        
        for source in priority:
            if self.providers.get(source) and self.providers[source].available:
                logger.info(f"自动选择数据源: {source}")
                return source
        
        raise RuntimeError("没有可用的数据源")
    
    def get_available_sources(self) -> List[str]:
        """获取可用数据源列表"""
        return [name for name, provider in self.providers.items() if provider.available]
    
    def clear_all_cache(self):
        """清空所有缓存"""
        for provider in self.providers.values():
            provider.clear_cache()
        logger.info("所有数据源缓存已清空")


# ============================================================================
# 工具函数
# ============================================================================

def create_multi_source_provider(config_file: Optional[str] = None) -> MultiSourceProvider:
    """
    创建多数据源提供者
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        MultiSourceProvider实例
    """
    config = {}
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return MultiSourceProvider(config)


# ============================================================================
# 示例
# ============================================================================

def example_usage():
    """使用示例"""
    
    # 1. 创建多数据源管理器
    provider = MultiSourceProvider(config={
        'tushare_token': 'YOUR_TOKEN',  # 替换为实际token
        'csv_data_dir': 'data/csv'
    })
    
    # 2. 查看可用数据源
    print("可用数据源:", provider.get_available_sources())
    
    # 3. 自动获取数据
    symbols = ['SH600000', 'SZ000001']
    df = provider.fetch_data(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-12-31',
        source='auto',
        fields=['open', 'close', 'high', 'low', 'volume']
    )
    
    print("\n获取的数据:")
    print(df.head())
    
    # 4. 指定数据源
    try:
        df_yahoo = provider.fetch_data(
            symbols='SH600000',
            start_date='2024-01-01',
            end_date='2024-12-31',
            source='yahoo'
        )
        print("\nYahoo Finance数据:")
        print(df_yahoo.head())
    except Exception as e:
        print(f"Yahoo Finance获取失败: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Qlib多数据源模块示例")
    print("=" * 70)
    
    example_usage()
