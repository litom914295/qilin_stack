"""
麒麟量化系统 - 真实数据接入模块
支持Tushare、AkShare等多数据源
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from functools import lru_cache
import json
import os
from pathlib import Path

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 300):
        """
        初始化缓存
        
        Args:
            redis_url: Redis连接URL
            ttl: 缓存过期时间（秒）
        """
        self.ttl = ttl
        self.memory_cache = {}
        self.redis_client = None
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis缓存已连接")
            except Exception as e:
                logger.warning(f"Redis连接失败，使用内存缓存: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        # 先查内存缓存
        if key in self.memory_cache:
            data, expire_time = self.memory_cache[key]
            if datetime.now() < expire_time:
                return data
            else:
                del self.memory_cache[key]
        
        # 再查Redis缓存
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"Redis读取失败: {e}")
        
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存数据"""
        # 内存缓存
        expire_time = datetime.now() + timedelta(seconds=self.ttl)
        self.memory_cache[key] = (value, expire_time)
        
        # Redis缓存
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key, 
                    self.ttl, 
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.debug(f"Redis写入失败: {e}")
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        if self.redis_client:
            self.redis_client.flushdb()


class RealTimeDataFetcher:
    """实时数据获取器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据获取器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.cache = DataCache(
            redis_url=self.config.get('redis_url'),
            ttl=self.config.get('cache_ttl', 300)
        )
        
        # 初始化数据源
        self._init_tushare()
        self._init_akshare()
        
        # 数据存储路径
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.data_dir.mkdir(exist_ok=True)
        
    def _init_tushare(self):
        """初始化Tushare"""
        if TUSHARE_AVAILABLE:
            token = os.getenv('TUSHARE_TOKEN') or self.config.get('tushare_token')
            if token:
                try:
                    ts.set_token(token)
                    self.ts_api = ts.pro_api()
                    logger.info("Tushare初始化成功")
                except Exception as e:
                    logger.error(f"Tushare初始化失败: {e}")
                    self.ts_api = None
            else:
                logger.warning("未配置Tushare Token")
                self.ts_api = None
        else:
            logger.warning("Tushare未安装")
            self.ts_api = None
    
    def _init_akshare(self):
        """初始化AkShare"""
        if AKSHARE_AVAILABLE:
            self.ak = ak
            logger.info("AkShare初始化成功")
        else:
            logger.warning("AkShare未安装")
            self.ak = None
    
    async def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        获取实时行情数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            股票行情数据字典
        """
        quotes = {}
        
        for symbol in symbols:
            # 检查缓存
            cache_key = f"quote:{symbol}:{datetime.now().strftime('%Y%m%d%H%M')}"
            cached_data = self.cache.get(cache_key)
            
            if cached_data:
                quotes[symbol] = pd.DataFrame(cached_data)
                continue
            
            # 获取实时数据
            try:
                if self.ts_api:
                    quote = await self._get_tushare_realtime(symbol)
                elif self.ak:
                    quote = await self._get_akshare_realtime(symbol)
                else:
                    quote = self._get_simulated_quote(symbol)
                
                quotes[symbol] = quote
                
                # 缓存数据
                self.cache.set(cache_key, quote.to_dict('records'))
                
            except Exception as e:
                logger.error(f"获取{symbol}实时行情失败: {e}")
                quotes[symbol] = self._get_simulated_quote(symbol)
        
        return quotes
    
    async def _get_tushare_realtime(self, symbol: str) -> pd.DataFrame:
        """通过Tushare获取实时行情"""
        try:
            # Tushare实时行情接口
            df = self.ts_api.daily(
                ts_code=self._format_ts_code(symbol),
                start_date=datetime.now().strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d')
            )
            
            if df.empty:
                # 如果当天没有数据，获取最近一天
                df = self.ts_api.daily(
                    ts_code=self._format_ts_code(symbol),
                    limit=1
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Tushare获取{symbol}数据失败: {e}")
            raise
    
    async def _get_akshare_realtime(self, symbol: str) -> pd.DataFrame:
        """通过AkShare获取实时行情"""
        try:
            # AkShare实时行情接口
            df = self.ak.stock_zh_a_spot_em()
            
            # 筛选指定股票
            df = df[df['代码'] == symbol]
            
            if df.empty:
                raise ValueError(f"未找到股票{symbol}")
            
            # 转换字段名
            df = df.rename(columns={
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'close',
                '涨跌幅': 'pct_change',
                '成交量': 'volume',
                '成交额': 'amount',
                '换手率': 'turnover_rate',
                '市盈率': 'pe',
                '市净率': 'pb'
            })
            
            return df
            
        except Exception as e:
            logger.error(f"AkShare获取{symbol}数据失败: {e}")
            raise
    
    def _get_simulated_quote(self, symbol: str) -> pd.DataFrame:
        """获取模拟行情数据"""
        base_price = 10 + hash(symbol) % 90
        
        return pd.DataFrame({
            'symbol': [symbol],
            'open': [base_price * 0.98],
            'high': [base_price * 1.05],
            'low': [base_price * 0.95],
            'close': [base_price],
            'volume': [np.random.randint(1000000, 10000000)],
            'amount': [np.random.randint(10000000, 100000000)],
            'turnover_rate': [np.random.uniform(1, 20)],
            'pct_change': [np.random.uniform(-10, 10)],
            'timestamp': [datetime.now()]
        })
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期
            freq: 频率 D/W/M
            
        Returns:
            历史数据DataFrame
        """
        # 检查缓存
        cache_key = f"hist:{symbol}:{start_date}:{end_date}:{freq}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            if self.ts_api:
                data = await self._get_tushare_history(symbol, start_date, end_date, freq)
            elif self.ak:
                data = await self._get_akshare_history(symbol, start_date, end_date, freq)
            else:
                data = self._get_simulated_history(symbol, start_date, end_date, freq)
            
            # 缓存数据
            self.cache.set(cache_key, data.to_dict('records'))
            
            # 保存到本地
            self._save_historical_data(symbol, data)
            
            return data
            
        except Exception as e:
            logger.error(f"获取{symbol}历史数据失败: {e}")
            return self._get_simulated_history(symbol, start_date, end_date, freq)
    
    async def _get_tushare_history(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        freq: str
    ) -> pd.DataFrame:
        """通过Tushare获取历史数据"""
        try:
            # 格式化日期
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            
            if freq == 'D':
                df = self.ts_api.daily(
                    ts_code=self._format_ts_code(symbol),
                    start_date=start,
                    end_date=end
                )
            elif freq == 'W':
                df = self.ts_api.weekly(
                    ts_code=self._format_ts_code(symbol),
                    start_date=start,
                    end_date=end
                )
            elif freq == 'M':
                df = self.ts_api.monthly(
                    ts_code=self._format_ts_code(symbol),
                    start_date=start,
                    end_date=end
                )
            else:
                raise ValueError(f"不支持的频率: {freq}")
            
            # 排序
            df = df.sort_values('trade_date')
            df.index = pd.to_datetime(df['trade_date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Tushare获取历史数据失败: {e}")
            raise
    
    async def _get_akshare_history(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        freq: str
    ) -> pd.DataFrame:
        """通过AkShare获取历史数据"""
        try:
            # AkShare历史数据接口
            df = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period=self._freq_to_akshare(freq),
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust='qfq'  # 前复权
            )
            
            # 转换字段名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '换手率': 'turnover_rate',
                '涨跌幅': 'pct_change'
            })
            
            df.index = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"AkShare获取历史数据失败: {e}")
            raise
    
    def _get_simulated_history(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        freq: str
    ) -> pd.DataFrame:
        """生成模拟历史数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        base_price = 10 + hash(symbol) % 90
        prices = []
        
        for i, date in enumerate(date_range):
            # 模拟随机游走
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + np.random.randn() * 0.02)
            
            prices.append(price)
        
        df = pd.DataFrame({
            'date': date_range,
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(date_range)),
            'amount': np.random.randint(10000000, 100000000, len(date_range)),
            'turnover_rate': np.random.uniform(1, 10, len(date_range))
        })
        
        df.index = df['date']
        
        return df
    
    async def get_market_data(self) -> Dict[str, Any]:
        """
        获取市场整体数据
        
        Returns:
            市场数据字典
        """
        market_data = {}
        
        try:
            # 获取指数数据
            indices = await self.get_index_data(['000001.SH', '399001.SZ', '399006.SZ'])
            market_data['indices'] = indices
            
            # 获取涨跌统计
            stats = await self.get_market_stats()
            market_data['stats'] = stats
            
            # 获取板块数据
            sectors = await self.get_sector_data()
            market_data['sectors'] = sectors
            
            # 获取龙虎榜
            lhb = await self.get_lhb_data()
            market_data['lhb'] = lhb
            
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            market_data = self._get_simulated_market_data()
        
        return market_data
    
    async def get_index_data(self, indices: List[str]) -> Dict[str, pd.DataFrame]:
        """获取指数数据"""
        index_data = {}
        
        for index in indices:
            try:
                if self.ts_api:
                    df = self.ts_api.index_daily(ts_code=index, limit=1)
                    index_data[index] = df
                else:
                    # 模拟数据
                    index_data[index] = pd.DataFrame({
                        'close': [3000 + np.random.randn() * 100],
                        'pct_change': [np.random.randn() * 2]
                    })
            except Exception as e:
                logger.error(f"获取指数{index}失败: {e}")
                
        return index_data
    
    async def get_market_stats(self) -> Dict[str, int]:
        """获取市场涨跌统计"""
        try:
            if self.ak:
                # 使用AkShare获取
                df = self.ak.stock_zh_a_spot_em()
                
                stats = {
                    'total': len(df),
                    'up': len(df[df['涨跌幅'] > 0]),
                    'down': len(df[df['涨跌幅'] < 0]),
                    'flat': len(df[df['涨跌幅'] == 0]),
                    'limit_up': len(df[df['涨跌幅'] >= 9.5]),
                    'limit_down': len(df[df['涨跌幅'] <= -9.5])
                }
            else:
                # 模拟数据
                stats = {
                    'total': 5000,
                    'up': 2800,
                    'down': 2000,
                    'flat': 200,
                    'limit_up': 120,
                    'limit_down': 10
                }
                
            return stats
            
        except Exception as e:
            logger.error(f"获取市场统计失败: {e}")
            return {'total': 0, 'up': 0, 'down': 0}
    
    async def get_sector_data(self) -> List[Dict]:
        """获取板块数据"""
        try:
            if self.ak:
                # 获取板块涨跌幅
                df = self.ak.stock_board_industry_name_em()
                
                sectors = []
                for _, row in df.head(10).iterrows():
                    sectors.append({
                        'name': row['板块名称'],
                        'change': row['涨跌幅'],
                        'leader': row['领涨股票']
                    })
                    
                return sectors
            else:
                # 模拟数据
                return [
                    {'name': '新能源', 'change': 3.5, 'leader': '300750'},
                    {'name': '半导体', 'change': 2.8, 'leader': '603986'},
                    {'name': '军工', 'change': 2.1, 'leader': '600760'}
                ]
                
        except Exception as e:
            logger.error(f"获取板块数据失败: {e}")
            return []
    
    async def get_lhb_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取龙虎榜数据"""
        try:
            if self.ak:
                # 获取龙虎榜
                date = date or datetime.now().strftime('%Y%m%d')
                df = self.ak.stock_lhb_detail_daily_sina(date=date)
                return df
            else:
                # 模拟数据
                return pd.DataFrame({
                    'symbol': ['000001', '000002'],
                    'name': ['平安银行', '万科A'],
                    'net_buy': [5000000, 3000000]
                })
                
        except Exception as e:
            logger.error(f"获取龙虎榜失败: {e}")
            return pd.DataFrame()
    
    def _format_ts_code(self, symbol: str) -> str:
        """格式化为Tushare代码"""
        if '.' in symbol:
            return symbol
        
        # 判断交易所
        if symbol.startswith('6'):
            return f"{symbol}.SH"
        elif symbol.startswith('0') or symbol.startswith('3'):
            return f"{symbol}.SZ"
        else:
            return symbol
    
    def _freq_to_akshare(self, freq: str) -> str:
        """频率转换为AkShare格式"""
        mapping = {
            'D': 'daily',
            'W': 'weekly',
            'M': 'monthly'
        }
        return mapping.get(freq, 'daily')
    
    def _save_historical_data(self, symbol: str, data: pd.DataFrame):
        """保存历史数据到本地"""
        try:
            file_path = self.data_dir / f"{symbol}_history.parquet"
            data.to_parquet(file_path)
            logger.debug(f"历史数据已保存: {file_path}")
        except Exception as e:
            logger.error(f"保存历史数据失败: {e}")
    
    def _get_simulated_market_data(self) -> Dict[str, Any]:
        """生成模拟市场数据"""
        return {
            'indices': {
                '000001.SH': pd.DataFrame({'close': [3100], 'pct_change': [0.5]})
            },
            'stats': {
                'total': 5000,
                'up': 2800,
                'down': 2000,
                'limit_up': 100
            },
            'sectors': [
                {'name': '新能源', 'change': 3.5}
            ],
            'lhb': pd.DataFrame()
        }


# 导出便捷函数
async def fetch_realtime_quotes(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """获取实时行情的便捷函数"""
    fetcher = RealTimeDataFetcher()
    return await fetcher.get_realtime_quotes(symbols)


async def fetch_historical_data(
    symbol: str,
    start_date: str,
    end_date: str,
    freq: str = 'D'
) -> pd.DataFrame:
    """获取历史数据的便捷函数"""
    fetcher = RealTimeDataFetcher()
    return await fetcher.get_historical_data(symbol, start_date, end_date, freq)


if __name__ == "__main__":
    # 测试代码
    async def test():
        fetcher = RealTimeDataFetcher()
        
        # 测试实时行情
        quotes = await fetcher.get_realtime_quotes(['000001', '000002'])
        print("实时行情:")
        for symbol, df in quotes.items():
            print(f"{symbol}: {df.iloc[0]['close'] if not df.empty else 'N/A'}")
        
        # 测试历史数据
        hist = await fetcher.get_historical_data(
            '000001',
            '2024-01-01',
            '2024-01-31'
        )
        print(f"\n历史数据: {len(hist)}条")
        
        # 测试市场数据
        market = await fetcher.get_market_data()
        print(f"\n市场统计: {market['stats']}")
    
    asyncio.run(test())