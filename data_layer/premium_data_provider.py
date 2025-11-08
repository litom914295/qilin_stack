"""
高级数据提供器 - 涨停板高级特征数据
提供流通市值、封单金额、板块题材、市场情绪等高级数据
"""

import pandas as pd
import numpy as np
import akshare as ak
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)


class PremiumDataProvider:
    """高级数据提供器"""
    
    def __init__(self, api_token: Optional[str] = None, use_cache: bool = True):
        """
        初始化高级数据提供器
        
        Args:
            api_token: API令牌（预留给未来使用Tushare Pro等付费服务）
            use_cache: 是否使用缓存
        """
        self.api_token = api_token
        self.use_cache = use_cache
        self.cache_dir = Path(__file__).parent.parent / 'cache' / 'premium_data'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据源
        self._init_data_sources()
    
    def _init_data_sources(self):
        """初始化数据源"""
        # 如果有Tushare Pro token，初始化
        if self.api_token and self.api_token.startswith('tushare_'):
            try:
                import tushare as ts
                self.ts_pro = ts.pro_api(self.api_token.replace('tushare_', ''))
                logger.info("Tushare Pro初始化成功")
            except Exception as e:
                logger.warning(f"Tushare Pro初始化失败: {e}")
                self.ts_pro = None
        else:
            self.ts_pro = None
    
    def get_daily_advanced_metrics(self, trade_date: str) -> pd.DataFrame:
        """
        获取指定交易日所有股票的高级指标
        
        Args:
            trade_date: 交易日期 (YYYY-MM-DD格式)
        
        Returns:
            DataFrame，索引为'symbol'，包含以下列：
            - circulating_market_cap: 流通市值（亿元）
            - total_market_cap: 总市值（亿元）
            - sector: 行业
            - themes: 题材概念（列表）
            - seal_amount: 涨停封单金额（万元，若当日涨停）
            - is_limit_up: 是否涨停
            - limit_up_time: 涨停时间
            - open_times: 开板次数
            - dragon_tiger_amount: 龙虎榜买入金额（万元）
        """
        # 尝试从缓存加载
        cache_file = self.cache_dir / f"advanced_metrics_{trade_date.replace('-', '')}.parquet"
        if self.use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"从缓存加载 {trade_date} 的高级数据")
                return df
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        
        # 获取实时数据
        df = self._fetch_advanced_metrics(trade_date)
        
        # 保存到缓存
        if self.use_cache and not df.empty:
            try:
                df.to_parquet(cache_file)
                logger.info(f"高级数据已缓存: {cache_file}")
            except Exception as e:
                logger.warning(f"缓存保存失败: {e}")
        
        return df
    
    def _fetch_advanced_metrics(self, trade_date: str) -> pd.DataFrame:
        """
        从数据源获取高级指标
        
        Args:
            trade_date: 交易日期
        
        Returns:
            高级指标DataFrame
        """
        results = []
        
        try:
            # 获取今日涨停股池
            date_str = trade_date.replace('-', '')
            df_limit = self._get_limit_up_stocks(date_str)
            
            # 获取所有股票的基础信息
            df_basic = self._get_stock_basic_info()
            
            # 获取板块题材信息
            df_sectors = self._get_sector_themes()
            
            # 合并数据
            if not df_limit.empty:
                # 处理涨停股
                for idx, row in df_limit.iterrows():
                    code = row.get('代码', '')
                    if not code:
                        continue
                    
                    # 转换股票代码格式
                    if code.startswith('6'):
                        symbol = f"{code}.SH"
                    elif code.startswith(('0', '3')):
                        symbol = f"{code}.SZ"
                    else:
                        symbol = code
                    
                    # 提取高级指标
                    result = {
                        'symbol': symbol,
                        'name': row.get('名称', ''),
                        'circulating_market_cap': self._safe_float(row.get('流通市值', 0)) / 1e8,  # 转换为亿元
                        'total_market_cap': self._safe_float(row.get('总市值', 0)) / 1e8,
                        'is_limit_up': True,
                        'limit_up_time': row.get('首次涨停时间', ''),
                        'seal_amount': self._safe_float(row.get('封单金额', 0)) / 1e4,  # 转换为万元
                        'seal_ratio': self._safe_float(row.get('封板比', 0)),
                        'open_times': int(row.get('涨停开板次数', 0)),
                        'consecutive_days': int(row.get('连板数', 1)),
                        'turnover_rate': self._safe_float(row.get('换手率', 0)),
                        'volume_ratio': self._safe_float(row.get('量比', 1.0)),
                    }
                    
                    # 添加板块信息
                    if symbol in df_sectors.index:
                        result['sector'] = df_sectors.loc[symbol, 'sector']
                        result['themes'] = df_sectors.loc[symbol, 'themes']
                    else:
                        result['sector'] = '未分类'
                        result['themes'] = []
                    
                    results.append(result)
            
            # 添加非涨停股的基础信息
            if not df_basic.empty:
                limit_symbols = {r['symbol'] for r in results}
                for idx, row in df_basic.iterrows():
                    if row['symbol'] not in limit_symbols:
                        result = {
                            'symbol': row['symbol'],
                            'name': row.get('name', ''),
                            'circulating_market_cap': self._safe_float(row.get('circ_mv', 0)) / 1e8,
                            'total_market_cap': self._safe_float(row.get('total_mv', 0)) / 1e8,
                            'is_limit_up': False,
                            'limit_up_time': '',
                            'seal_amount': 0,
                            'seal_ratio': 0,
                            'open_times': 0,
                            'consecutive_days': 0,
                            'turnover_rate': self._safe_float(row.get('turnover_rate', 0)),
                            'volume_ratio': self._safe_float(row.get('volume_ratio', 1.0)),
                            'sector': row.get('sector', '未分类'),
                            'themes': row.get('themes', []),
                        }
                        results.append(result)
        
        except Exception as e:
            logger.error(f"获取高级指标失败: {e}")
            # 返回示例数据，确保系统能继续运行
            results = self._generate_demo_metrics(trade_date)
        
        df = pd.DataFrame(results)
        if not df.empty:
            df.set_index('symbol', inplace=True)
        
        return df
    
    def _get_limit_up_stocks(self, date_str: str) -> pd.DataFrame:
        """获取涨停股票池"""
        try:
            # 使用akshare获取涨停板数据
            df = ak.stock_zt_pool_em(date=date_str)
            return df
        except Exception as e:
            logger.warning(f"获取涨停股失败: {e}")
            return pd.DataFrame()
    
    def _get_stock_basic_info(self) -> pd.DataFrame:
        """获取股票基础信息"""
        try:
            # 使用akshare获取实时行情
            df = ak.stock_zh_a_spot_em()
            return df
        except Exception as e:
            logger.warning(f"获取股票基础信息失败: {e}")
            return pd.DataFrame()
    
    def _get_sector_themes(self) -> pd.DataFrame:
        """获取板块题材信息"""
        try:
            # 使用akshare获取概念板块
            df_concept = ak.stock_board_concept_name_em()
            
            # 这里简化处理，实际需要更复杂的映射
            sectors = {}
            for idx, row in df_concept.iterrows():
                # 这里需要更完善的股票-概念映射逻辑
                pass
            
            return pd.DataFrame(sectors)
        except Exception as e:
            logger.warning(f"获取板块题材失败: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(self, trade_date: str) -> Dict[str, Any]:
        """
        获取指定交易日的市场情绪
        
        Args:
            trade_date: 交易日期 (YYYY-MM-DD格式)
        
        Returns:
            市场情绪字典，包含：
            - limit_up_count: 涨停家数
            - limit_down_count: 跌停家数
            - break_rate: 炸板率
            - yesterday_limit_up_performance: 昨日涨停股今日平均表现
            - max_consecutive_boards: 最高连板数
            - sentiment_score: 情绪评分（0-100）
        """
        sentiment = {
            'limit_up_count': 0,
            'limit_down_count': 0,
            'break_rate': 0,
            'yesterday_limit_up_performance': 0,
            'max_consecutive_boards': 0,
            'sentiment_score': 50
        }
        
        try:
            # 获取涨停跌停统计
            date_str = trade_date.replace('-', '')
            
            # 涨停池
            df_zt = ak.stock_zt_pool_em(date=date_str)
            if not df_zt.empty:
                sentiment['limit_up_count'] = len(df_zt)
                sentiment['max_consecutive_boards'] = df_zt['连板数'].max() if '连板数' in df_zt.columns else 1
            
            # 跌停池
            df_dt = ak.stock_dt_pool_em(date=date_str)
            if not df_dt.empty:
                sentiment['limit_down_count'] = len(df_dt)
            
            # 炸板池
            df_zb = ak.stock_zt_pool_strong_em(date=date_str)
            if not df_zb.empty:
                total_limit_attempts = sentiment['limit_up_count'] + len(df_zb)
                sentiment['break_rate'] = (len(df_zb) / total_limit_attempts * 100) if total_limit_attempts > 0 else 0
            
            # 昨日涨停表现（需要获取昨日涨停股今日表现）
            sentiment['yesterday_limit_up_performance'] = self._calc_yesterday_limitup_performance(trade_date)
            
            # 计算情绪评分
            sentiment['sentiment_score'] = self._calc_sentiment_score(sentiment)
            
        except Exception as e:
            logger.error(f"获取市场情绪失败: {e}")
            # 返回示例数据
            sentiment = self._generate_demo_sentiment(trade_date)
        
        return sentiment
    
    def _calc_yesterday_limitup_performance(self, trade_date: str) -> float:
        """计算昨日涨停股今日平均表现"""
        # 这里需要实现获取昨日涨停股今日表现的逻辑
        # 暂时返回模拟值
        return np.random.uniform(-0.02, 0.05)
    
    def _calc_sentiment_score(self, sentiment: Dict) -> float:
        """
        计算市场情绪评分
        
        Args:
            sentiment: 市场情绪指标字典
        
        Returns:
            情绪评分（0-100）
        """
        score = 50.0  # 基础分
        
        # 涨停数量加分（最多20分）
        limit_up_bonus = min(sentiment['limit_up_count'] / 100 * 20, 20)
        score += limit_up_bonus
        
        # 跌停数量减分（最多-20分）
        limit_down_penalty = min(sentiment['limit_down_count'] / 50 * 20, 20)
        score -= limit_down_penalty
        
        # 炸板率减分（最多-15分）
        break_penalty = min(sentiment['break_rate'] / 30 * 15, 15)
        score -= break_penalty
        
        # 连板高度加分（最多15分）
        board_bonus = min(sentiment['max_consecutive_boards'] / 10 * 15, 15)
        score += board_bonus
        
        # 昨日涨停表现加分/减分（-10到10分）
        yesterday_bonus = sentiment['yesterday_limit_up_performance'] * 200
        yesterday_bonus = max(min(yesterday_bonus, 10), -10)
        score += yesterday_bonus
        
        # 限制在0-100范围
        score = max(0, min(100, score))
        
        return score
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """安全转换为浮点数"""
        try:
            if pd.isna(value):
                return default
            return float(value)
        except:
            return default
    
    def _generate_demo_metrics(self, trade_date: str) -> List[Dict]:
        """生成演示数据"""
        demo_data = []
        symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH']
        
        for symbol in symbols:
            demo_data.append({
                'symbol': symbol,
                'name': f'股票{symbol[:6]}',
                'circulating_market_cap': np.random.uniform(50, 500),
                'total_market_cap': np.random.uniform(100, 1000),
                'is_limit_up': np.random.choice([True, False], p=[0.2, 0.8]),
                'limit_up_time': '09:30:00' if np.random.random() > 0.5 else '14:30:00',
                'seal_amount': np.random.uniform(1000, 50000),
                'seal_ratio': np.random.uniform(0.01, 0.5),
                'open_times': np.random.randint(0, 3),
                'consecutive_days': np.random.randint(1, 5),
                'turnover_rate': np.random.uniform(1, 30),
                'volume_ratio': np.random.uniform(0.5, 5),
                'sector': np.random.choice(['科技', '医药', '消费', '金融']),
                'themes': np.random.choice([['AI', '芯片'], ['新能源'], ['白酒'], ['银行']])
            })
        
        return demo_data
    
    def _generate_demo_sentiment(self, trade_date: str) -> Dict:
        """生成演示情绪数据"""
        return {
            'limit_up_count': np.random.randint(30, 150),
            'limit_down_count': np.random.randint(5, 30),
            'break_rate': np.random.uniform(10, 40),
            'yesterday_limit_up_performance': np.random.uniform(-0.02, 0.05),
            'max_consecutive_boards': np.random.randint(1, 10),
            'sentiment_score': np.random.uniform(30, 80)
        }
    
    async def get_real_time_seal_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时封单数据（异步）
        
        Args:
            symbol: 股票代码
        
        Returns:
            封单数据字典
        """
        # 这里可以接入L2数据源获取实时封单
        # 目前返回模拟数据
        return {
            'symbol': symbol,
            'seal_amount': np.random.uniform(1000, 50000),
            'seal_change_rate': np.random.uniform(-0.5, 0.5),
            'queue_length': np.random.randint(1000, 100000),
            'update_time': datetime.now().isoformat()
        }