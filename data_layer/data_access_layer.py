"""
多数据源统一数据接入层
支持行情数据、财务数据、新闻资讯等多种数据源
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import aiohttp
import akshare as ak
import tushare as ts
import yfinance as yf
from dataclasses import dataclass
import json
import logging
from enum import Enum
import redis
from motor.motor_asyncio import AsyncIOMotorClient
import clickhouse_driver
from kafka import KafkaProducer, KafkaConsumer
import pyarrow.parquet as pq
import pyarrow as pa

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """数据源枚举"""
    AKSHARE = "akshare"
    TUSHARE = "tushare"
    YAHOO = "yahoo"
    EASTMONEY = "eastmoney"
    SINA = "sina"
    CUSTOM = "custom"


class DataType(Enum):
    """数据类型枚举"""
    KLINE = "kline"          # K线数据
    TICK = "tick"            # 分笔数据
    DEPTH = "depth"          # 深度数据
    TRADE = "trade"          # 成交数据
    FINANCIAL = "financial"  # 财务数据
    NEWS = "news"            # 新闻数据
    ANNOUNCEMENT = "announcement"  # 公告数据
    DRAGON_TIGER = "dragon_tiger"  # 龙虎榜数据


@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    data_type: DataType
    source: DataSource
    metadata: Dict[str, Any]


class DataAccessLayer:
    """数据接入层主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据接入层
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.providers = {}
        self.cache = DataCache(config.get('redis', {}))
        self.storage = DataStorage(config.get('storage', {}))
        self.stream_processor = StreamProcessor(config.get('kafka', {}))
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化数据提供者"""
        # AkShare数据源
        if self.config.get('akshare', {}).get('enabled', True):
            self.providers[DataSource.AKSHARE] = AkShareProvider()
        
        # Tushare数据源
        if self.config.get('tushare', {}).get('enabled', False):
            ts_token = self.config.get('tushare', {}).get('token')
            if ts_token:
                self.providers[DataSource.TUSHARE] = TushareProvider(ts_token)
        
        # Yahoo Finance数据源
        if self.config.get('yahoo', {}).get('enabled', False):
            self.providers[DataSource.YAHOO] = YahooProvider()
        
        logger.info(f"Initialized {len(self.providers)} data providers")
    
    async def get_realtime_data(self, 
                                symbols: List[str], 
                                data_type: DataType = DataType.KLINE) -> pd.DataFrame:
        """
        获取实时数据
        
        Args:
            symbols: 股票代码列表
            data_type: 数据类型
            
        Returns:
            实时数据DataFrame
        """
        # 先尝试从缓存获取
        cached_data = await self.cache.get_batch(symbols, data_type)
        if cached_data is not None:
            return cached_data
        
        # 从数据源获取
        tasks = []
        for symbol in symbols:
            provider = self._select_best_provider(symbol, data_type)
            if provider:
                tasks.append(provider.get_realtime(symbol, data_type))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        valid_results = [r for r in results if not isinstance(r, Exception)]
        if valid_results:
            df = pd.concat(valid_results, ignore_index=True)
            
            # 缓存结果
            await self.cache.set_batch(df, data_type)
            
            # 发送到流处理
            await self.stream_processor.send(df)
            
            return df
        
        return pd.DataFrame()
    
    async def get_historical_data(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str,
                                 frequency: str = "1d",
                                 data_type: DataType = DataType.KLINE) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            data_type: 数据类型
            
        Returns:
            历史数据DataFrame
        """
        # 先从存储获取
        stored_data = await self.storage.query_historical(
            symbol, start_date, end_date, frequency, data_type
        )
        
        if not stored_data.empty:
            return stored_data
        
        # 从数据源获取
        provider = self._select_best_provider(symbol, data_type)
        if provider:
            df = await provider.get_historical(
                symbol, start_date, end_date, frequency, data_type
            )
            
            # 存储数据
            if not df.empty:
                await self.storage.save_historical(df, data_type)
            
            return df
        
        return pd.DataFrame()
    
    async def get_financial_data(self, 
                                symbol: str,
                                report_type: str = "income") -> Dict[str, Any]:
        """
        获取财务数据
        
        Args:
            symbol: 股票代码
            report_type: 报表类型 (income/balance/cashflow)
            
        Returns:
            财务数据字典
        """
        provider = self._select_best_provider(symbol, DataType.FINANCIAL)
        if provider:
            return await provider.get_financial(symbol, report_type)
        return {}
    
    async def get_news_data(self,
                           keywords: List[str] = None,
                           start_date: str = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取新闻数据
        
        Args:
            keywords: 关键词列表
            start_date: 开始日期
            limit: 返回条数限制
            
        Returns:
            新闻数据列表
        """
        news_data = []
        
        for source, provider in self.providers.items():
            if hasattr(provider, 'get_news'):
                data = await provider.get_news(keywords, start_date, limit)
                news_data.extend(data)
        
        # 按时间排序并去重
        news_data = self._deduplicate_news(news_data)
        news_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return news_data[:limit]
    
    async def get_dragon_tiger_data(self, 
                                   date: str = None) -> pd.DataFrame:
        """
        获取龙虎榜数据
        
        Args:
            date: 日期
            
        Returns:
            龙虎榜数据DataFrame
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        provider = self._select_best_provider('', DataType.DRAGON_TIGER)
        if provider:
            return await provider.get_dragon_tiger(date)
        
        return pd.DataFrame()
    
    def _select_best_provider(self, symbol: str, data_type: DataType):
        """选择最佳数据提供者"""
        # 根据数据类型和股票代码选择最佳提供者
        if symbol.startswith('6') or symbol.startswith('0') or symbol.startswith('3'):
            # A股
            if DataSource.AKSHARE in self.providers:
                return self.providers[DataSource.AKSHARE]
            elif DataSource.TUSHARE in self.providers:
                return self.providers[DataSource.TUSHARE]
        else:
            # 美股或其他
            if DataSource.YAHOO in self.providers:
                return self.providers[DataSource.YAHOO]
        
        # 返回第一个可用的提供者
        return next(iter(self.providers.values())) if self.providers else None
    
    def _deduplicate_news(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """新闻去重"""
        seen = set()
        unique_news = []
        
        for news in news_list:
            title = news.get('title', '')
            if title and title not in seen:
                seen.add(title)
                unique_news.append(news)
        
        return unique_news


class DataProvider(ABC):
    """数据提供者基类"""
    
    @abstractmethod
    async def get_realtime(self, symbol: str, data_type: DataType) -> pd.DataFrame:
        """获取实时数据"""
        pass
    
    @abstractmethod
    async def get_historical(self, symbol: str, start_date: str, end_date: str, 
                            frequency: str, data_type: DataType) -> pd.DataFrame:
        """获取历史数据"""
        pass


class AkShareProvider(DataProvider):
    """AkShare数据提供者"""
    
    async def get_realtime(self, symbol: str, data_type: DataType) -> pd.DataFrame:
        """获取实时数据"""
        try:
            if data_type == DataType.KLINE:
                # 获取实时行情
                df = ak.stock_zh_a_spot_em()
                df = df[df['代码'] == symbol]
                
                # 标准化列名
                df = df.rename(columns={
                    '代码': 'symbol',
                    '名称': 'name',
                    '最新价': 'close',
                    '涨跌幅': 'change_pct',
                    '涨跌额': 'change',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '最高': 'high',
                    '最低': 'low',
                    '今开': 'open',
                    '昨收': 'pre_close'
                })
                
                df['timestamp'] = datetime.now()
                df['data_type'] = data_type.value
                df['source'] = DataSource.AKSHARE.value
                
                return df
            
        except Exception as e:
            logger.error(f"AkShare get_realtime error: {e}")
            return pd.DataFrame()
    
    async def get_historical(self, symbol: str, start_date: str, end_date: str,
                            frequency: str, data_type: DataType) -> pd.DataFrame:
        """获取历史数据"""
        try:
            if data_type == DataType.KLINE:
                # 转换频率参数
                period_map = {
                    '1m': '1',
                    '5m': '5', 
                    '15m': '15',
                    '30m': '30',
                    '60m': '60',
                    '1d': 'daily'
                }
                
                period = period_map.get(frequency, 'daily')
                
                # 获取历史K线
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period=period,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust='qfq'  # 前复权
                )
                
                # 标准化列名
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                })
                
                df['symbol'] = symbol
                df['data_type'] = data_type.value
                df['source'] = DataSource.AKSHARE.value
                
                return df
                
        except Exception as e:
            logger.error(f"AkShare get_historical error: {e}")
            return pd.DataFrame()
    
    async def get_financial(self, symbol: str, report_type: str) -> Dict[str, Any]:
        """获取财务数据"""
        try:
            if report_type == 'income':
                # 利润表
                df = ak.stock_financial_report_sina(stock=symbol, symbol='利润表')
            elif report_type == 'balance':
                # 资产负债表
                df = ak.stock_financial_report_sina(stock=symbol, symbol='资产负债表')
            elif report_type == 'cashflow':
                # 现金流量表
                df = ak.stock_financial_report_sina(stock=symbol, symbol='现金流量表')
            else:
                return {}
            
            # 转换为字典格式
            return df.to_dict('records')[-1] if not df.empty else {}
            
        except Exception as e:
            logger.error(f"AkShare get_financial error: {e}")
            return {}
    
    async def get_dragon_tiger(self, date: str) -> pd.DataFrame:
        """获取龙虎榜数据"""
        try:
            df = ak.stock_lhb_detail_daily_sina(date=date.replace('-', ''))
            return df
        except Exception as e:
            logger.error(f"AkShare get_dragon_tiger error: {e}")
            return pd.DataFrame()


class TushareProvider(DataProvider):
    """Tushare数据提供者"""
    
    def __init__(self, token: str):
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    async def get_realtime(self, symbol: str, data_type: DataType) -> pd.DataFrame:
        """获取实时数据"""
        try:
            df = ts.get_realtime_quotes(symbol)
            df['timestamp'] = datetime.now()
            df['data_type'] = data_type.value
            df['source'] = DataSource.TUSHARE.value
            return df
        except Exception as e:
            logger.error(f"Tushare get_realtime error: {e}")
            return pd.DataFrame()
    
    async def get_historical(self, symbol: str, start_date: str, end_date: str,
                            frequency: str, data_type: DataType) -> pd.DataFrame:
        """获取历史数据"""
        try:
            df = self.pro.daily(
                ts_code=symbol,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            df['data_type'] = data_type.value
            df['source'] = DataSource.TUSHARE.value
            return df
        except Exception as e:
            logger.error(f"Tushare get_historical error: {e}")
            return pd.DataFrame()


class YahooProvider(DataProvider):
    """Yahoo Finance数据提供者"""
    
    async def get_realtime(self, symbol: str, data_type: DataType) -> pd.DataFrame:
        """获取实时数据"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            df = pd.DataFrame([{
                'symbol': symbol,
                'open': info.get('open'),
                'high': info.get('dayHigh'),
                'low': info.get('dayLow'),
                'close': info.get('currentPrice', info.get('previousClose')),
                'volume': info.get('volume'),
                'timestamp': datetime.now(),
                'data_type': data_type.value,
                'source': DataSource.YAHOO.value
            }])
            
            return df
        except Exception as e:
            logger.error(f"Yahoo get_realtime error: {e}")
            return pd.DataFrame()
    
    async def get_historical(self, symbol: str, start_date: str, end_date: str,
                            frequency: str, data_type: DataType) -> pd.DataFrame:
        """获取历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=frequency)
            df['symbol'] = symbol
            df['data_type'] = data_type.value
            df['source'] = DataSource.YAHOO.value
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Yahoo get_historical error: {e}")
            return pd.DataFrame()


class DataCache:
    """数据缓存"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.enabled = False
        try:
            self.redis_client = redis.Redis(
                host=config.get('host', 'localhost'),
                port=config.get('port', 6379),
                db=config.get('db', 0),
                decode_responses=True,
                socket_connect_timeout=3
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available, cache disabled: {e}")
        self.ttl = config.get('ttl', 60)  # 默认60秒过期
    
    async def get_batch(self, symbols: List[str], data_type: DataType) -> Optional[pd.DataFrame]:
        """批量获取缓存"""
        if not self.enabled or self.redis_client is None:
            return None
            
        try:
            keys = [f"{data_type.value}:{symbol}" for symbol in symbols]
            values = self.redis_client.mget(keys)
            
            if all(values):
                dfs = [pd.read_json(v) for v in values if v]
                return pd.concat(dfs, ignore_index=True)
                
        except Exception as e:
            logger.error(f"Cache get_batch error: {e}")
        
        return None
    
    async def set_batch(self, df: pd.DataFrame, data_type: DataType):
        """批量设置缓存"""
        if not self.enabled or self.redis_client is None:
            return
            
        try:
            grouped = df.groupby('symbol')
            
            for symbol, group_df in grouped:
                key = f"{data_type.value}:{symbol}"
                value = group_df.to_json()
                self.redis_client.setex(key, self.ttl, value)
                
        except Exception as e:
            logger.error(f"Cache set_batch error: {e}")


class DataStorage:
    """数据存储"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_type = config.get('type', 'clickhouse')
        self.client = None
        self.enabled = False
        
        try:
            if self.storage_type == 'clickhouse':
                self.client = clickhouse_driver.Client(
                    host=config.get('host', 'localhost'),
                    port=config.get('port', 9000),
                    database=config.get('database', 'qilin'),
                    connect_timeout=3
                )
                # Test connection
                self.client.execute('SELECT 1')
                self.enabled = True
                logger.info("ClickHouse storage initialized successfully")
            elif self.storage_type == 'mongodb':
                self.client = AsyncIOMotorClient(
                    config.get('url', 'mongodb://localhost:27017'),
                    serverSelectionTimeoutMS=3000
                )
                self.db = self.client[config.get('database', 'qilin')]
                self.enabled = True
                logger.info("MongoDB storage initialized successfully")
        except Exception as e:
            logger.warning(f"Storage not available ({self.storage_type}): {e}")
    
    async def save_historical(self, df: pd.DataFrame, data_type: DataType):
        """保存历史数据"""
        try:
            if self.storage_type == 'clickhouse':
                # ClickHouse存储
                table_name = f"{data_type.value}_historical"
                self.client.insert_dataframe(
                    f'INSERT INTO {table_name} VALUES',
                    df
                )
            elif self.storage_type == 'mongodb':
                # MongoDB存储
                collection = self.db[f"{data_type.value}_historical"]
                records = df.to_dict('records')
                await collection.insert_many(records)
                
        except Exception as e:
            logger.error(f"Storage save_historical error: {e}")
    
    async def query_historical(self, symbol: str, start_date: str, end_date: str,
                              frequency: str, data_type: DataType) -> pd.DataFrame:
        """查询历史数据"""
        try:
            if self.storage_type == 'clickhouse':
                query = f"""
                SELECT * FROM {data_type.value}_historical
                WHERE symbol = '{symbol}'
                AND date >= '{start_date}'
                AND date <= '{end_date}'
                ORDER BY date
                """
                df = self.client.query_dataframe(query)
                return df
            elif self.storage_type == 'mongodb':
                collection = self.db[f"{data_type.value}_historical"]
                cursor = collection.find({
                    'symbol': symbol,
                    'date': {'$gte': start_date, '$lte': end_date}
                }).sort('date', 1)
                
                records = await cursor.to_list(None)
                return pd.DataFrame(records)
                
        except Exception as e:
            logger.error(f"Storage query_historical error: {e}")
            return pd.DataFrame()


class StreamProcessor:
    """流数据处理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = None
        self.enabled = False
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=config.get('brokers', ['localhost:9092']),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(0, 10, 1),
                request_timeout_ms=5000,
                max_block_ms=5000
            )
            self.enabled = True
            logger.info("Kafka StreamProcessor initialized successfully")
        except Exception as e:
            logger.warning(f"Kafka not available, StreamProcessor disabled: {e}")
        self.topic = config.get('topic', 'market_data')
    
    async def send(self, df: pd.DataFrame):
        """发送数据到Kafka"""
        if not self.enabled or self.producer is None:
            logger.debug("StreamProcessor is disabled, skipping send")
            return
            
        try:
            records = df.to_dict('records')
            for record in records:
                # 转换时间戳为字符串
                if 'timestamp' in record and isinstance(record['timestamp'], datetime):
                    record['timestamp'] = record['timestamp'].isoformat()
                
                self.producer.send(self.topic, record)
            
            self.producer.flush()
            
        except Exception as e:
            logger.error(f"StreamProcessor send error: {e}")


# 数据质量检查
class DataQualityChecker:
    """数据质量检查器"""
    
    @staticmethod
    def check_completeness(df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据完整性"""
        missing_ratio = df.isnull().sum() / len(df)
        return {
            'total_rows': len(df),
            'missing_ratio': missing_ratio.to_dict(),
            'is_complete': missing_ratio.max() < 0.1  # 缺失率小于10%
        }
    
    @staticmethod
    def check_consistency(df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据一致性"""
        issues = []
        
        # 检查价格一致性
        if 'high' in df.columns and 'low' in df.columns:
            invalid = df[df['high'] < df['low']]
            if not invalid.empty:
                issues.append(f"Found {len(invalid)} rows where high < low")
        
        # 检查成交量
        if 'volume' in df.columns:
            negative = df[df['volume'] < 0]
            if not negative.empty:
                issues.append(f"Found {len(negative)} rows with negative volume")
        
        return {
            'issues': issues,
            'is_consistent': len(issues) == 0
        }
    
    @staticmethod
    def check_timeliness(df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据时效性"""
        if 'timestamp' in df.columns:
            latest = pd.to_datetime(df['timestamp']).max()
            delay = datetime.now() - latest
            
            return {
                'latest_timestamp': latest,
                'delay_seconds': delay.total_seconds(),
                'is_timely': delay.total_seconds() < 60  # 延迟小于60秒
            }
        
        return {'is_timely': False}


if __name__ == "__main__":
    # 测试代码
    config = {
        'akshare': {'enabled': True},
        'redis': {'host': 'localhost', 'port': 6379},
        'storage': {'type': 'clickhouse', 'host': 'localhost'},
        'kafka': {'brokers': ['localhost:9092']}
    }
    
    # 创建数据接入层实例
    dal = DataAccessLayer(config)
    
    # 测试获取实时数据
    async def test():
        df = await dal.get_realtime_data(['000001', '000002'])
        print(df)
        
        # 数据质量检查
        checker = DataQualityChecker()
        print(checker.check_completeness(df))
        print(checker.check_consistency(df))
        print(checker.check_timeliness(df))
    
    # asyncio.run(test())