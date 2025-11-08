"""
竞价特征提取模块
用于预测T+1竞价强度和日内表现
专注于从T日数据预测T+1早盘竞价表现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings


class AuctionFeatureExtractor:
    """
    竞价特征提取器
    
    核心目标：从T日数据预测T+1竞价强度
    应用场景：T日盘后分析，预测次日竞价表现
    """
    
    def __init__(self):
        pass
    
    def extract_auction_predictive_features(self, 
                                           panel: pd.DataFrame,
                                           lookback_days: int = 5) -> pd.DataFrame:
        """
        提取预测竞价强度的特征
        
        Parameters:
        -----------
        panel: DataFrame
            必须包含列: date, symbol, open, high, low, close, volume
            按(symbol, date)排序
        lookback_days: int
            历史回溯天数
            
        Returns:
        --------
        DataFrame: 竞价预测特征
        """
        if panel.empty:
            return pd.DataFrame()
        
        # 确保按股票和日期排序
        panel = panel.sort_values(['symbol', 'date']).copy()
        
        features = pd.DataFrame(index=panel.index)
        features['date'] = panel['date']
        features['symbol'] = panel['symbol']
        
        # ========== 第一类：T日封单特征 ==========
        print("提取T日封单特征...")
        features = self._extract_t_day_features(features, panel)
        
        # ========== 第二类：历史竞价表现 ==========
        print("提取历史竞价表现...")
        features = self._extract_historical_auction_features(features, panel, lookback_days)
        
        # ========== 第三类：市场环境特征 ==========
        print("提取市场环境特征...")
        features = self._extract_market_context_features(features, panel)
        
        # ========== 第四类：连续性特征 ==========
        print("提取连续性特征...")
        features = self._extract_continuity_features(features, panel, lookback_days)
        
        print(f"竞价特征提取完成，共 {len(features.columns) - 2} 个特征")
        
        return features
    
    def _extract_t_day_features(self, features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        """提取T日封单相关特征"""
        
        # T日涨跌幅
        panel['prev_close'] = panel.groupby('symbol')['close'].shift(1)
        features['t_day_return'] = (panel['close'] / panel['prev_close'] - 1).fillna(0)
        
        # T日是否涨停（粗略判断，涨幅>9.5%）
        features['t_day_is_limitup'] = (features['t_day_return'] > 0.095).astype(int)
        
        # T日收盘强度（收盘价相对高低点位置）
        price_range = panel['high'] - panel['low']
        price_range = price_range.replace(0, 1)  # 避免除零
        features['t_day_close_strength'] = (panel['close'] - panel['low']) / price_range
        
        # T日量比（相对5日均量）
        panel['vol_ma5'] = panel.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        features['t_day_volume_ratio'] = (panel['volume'] / panel['vol_ma5']).fillna(1.0)
        
        # T日振幅
        features['t_day_amplitude'] = ((panel['high'] - panel['low']) / panel['prev_close']).fillna(0)
        
        # T日换手率（如果有数据）
        if 'turnover' in panel.columns:
            features['t_day_turnover'] = panel['turnover']
        else:
            features['t_day_turnover'] = features['t_day_volume_ratio'] * 5  # 近似值
        
        return features
    
    def _extract_historical_auction_features(self, 
                                            features: pd.DataFrame, 
                                            panel: pd.DataFrame,
                                            lookback_days: int) -> pd.DataFrame:
        """提取历史竞价表现特征"""
        
        # 计算每日竞价涨幅（开盘价相对前收）
        panel['prev_close'] = panel.groupby('symbol')['close'].shift(1)
        panel['auction_gap'] = (panel['open'] / panel['prev_close'] - 1).fillna(0)
        
        # 过去N天平均竞价涨幅
        features['avg_auction_gap_5d'] = panel.groupby('symbol')['auction_gap'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).mean()
        )
        
        # 过去N天竞价涨幅标准差（波动率）
        features['auction_volatility'] = panel.groupby('symbol')['auction_gap'].transform(
            lambda x: x.rolling(lookback_days, min_periods=2).std()
        ).fillna(0.01)
        
        # 过去N天竞价高开次数比例
        panel['auction_high_open'] = (panel['auction_gap'] > 0.03).astype(int)
        features['auction_high_open_ratio'] = panel.groupby('symbol')['auction_high_open'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).mean()
        )
        
        # 过去N天竞价低开次数比例
        panel['auction_low_open'] = (panel['auction_gap'] < -0.01).astype(int)
        features['auction_low_open_ratio'] = panel.groupby('symbol')['auction_low_open'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).mean()
        )
        
        # 竞价一致性（最近5天竞价方向一致性）
        panel['auction_direction'] = np.sign(panel['auction_gap'])
        features['auction_consistency'] = panel.groupby('symbol')['auction_direction'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).apply(
                lambda y: np.abs(y.sum()) / len(y) if len(y) > 0 else 0
            )
        )
        
        return features
    
    def _extract_market_context_features(self, 
                                        features: pd.DataFrame, 
                                        panel: pd.DataFrame) -> pd.DataFrame:
        """提取市场环境特征"""
        
        # 按日期统计市场整体涨停数（模拟，实际需要从外部数据源获取）
        # 这里用涨幅>9.5%的股票数近似
        panel['is_limitup'] = (panel['close'] / panel.groupby('symbol')['close'].shift(1) - 1 > 0.095).astype(int)
        market_stats = panel.groupby('date').agg({
            'is_limitup': 'sum',  # 涨停数
            'close': 'count'  # 总股票数
        }).rename(columns={'is_limitup': 'market_limitup_count', 'close': 'market_total_stocks'})
        
        # 合并市场统计
        features = features.merge(market_stats, left_on='date', right_index=True, how='left')
        
        # 市场涨停比例
        features['market_limitup_ratio'] = (
            features['market_limitup_count'] / features['market_total_stocks']
        ).fillna(0)
        
        # 市场情绪分类（根据涨停数）
        features['market_sentiment'] = pd.cut(
            features['market_limitup_count'],
            bins=[0, 30, 50, 100, 1000],
            labels=[0, 1, 2, 3],  # 0-低迷, 1-正常, 2-活跃, 3-火热
            include_lowest=True
        ).astype(float).fillna(1)
        
        # 板块强度（同板块涨停数，需要板块信息）
        if 'sector' in panel.columns:
            sector_stats = panel[panel['is_limitup'] == 1].groupby(['date', 'sector']).size()
            sector_stats = sector_stats.reset_index(name='sector_limitup_count')
            
            features = features.merge(
                sector_stats, 
                left_on=['date', panel['sector']], 
                right_on=['date', 'sector'],
                how='left'
            )
            features['sector_limitup_count'] = features['sector_limitup_count'].fillna(0)
        else:
            features['sector_limitup_count'] = 0
        
        return features
    
    def _extract_continuity_features(self,
                                    features: pd.DataFrame,
                                    panel: pd.DataFrame,
                                    lookback_days: int) -> pd.DataFrame:
        """提取连续性特征（判断趋势延续性）"""
        
        # 连续上涨天数
        panel['is_up'] = (panel['close'] > panel.groupby('symbol')['close'].shift(1)).astype(int)
        features['continuous_up_days'] = panel.groupby('symbol')['is_up'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).sum()
        )
        
        # 连续涨停天数
        panel['is_limitup'] = (
            panel['close'] / panel.groupby('symbol')['close'].shift(1) - 1 > 0.095
        ).astype(int)
        features['continuous_limitup_days'] = panel.groupby('symbol')['is_limitup'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).sum()
        )
        
        # 最近N天最高涨幅
        panel['daily_return'] = (panel['close'] / panel.groupby('symbol')['close'].shift(1) - 1).fillna(0)
        features['max_return_Nd'] = panel.groupby('symbol')['daily_return'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).max()
        )
        
        # 最近N天累计涨幅
        features['cum_return_Nd'] = panel.groupby('symbol')['daily_return'].transform(
            lambda x: (1 + x).rolling(lookback_days, min_periods=1).apply(
                lambda y: y.prod() - 1, raw=True
            )
        )
        
        # 价格动量（当前价相对N天前）
        panel['close_Nd_ago'] = panel.groupby('symbol')['close'].shift(lookback_days)
        features['price_momentum_Nd'] = (panel['close'] / panel['close_Nd_ago'] - 1).fillna(0)
        
        # 量能趋势（当前量相对N天平均）
        panel['volume_maN'] = panel.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(lookback_days, min_periods=1).mean()
        )
        features['volume_trend'] = (panel['volume'] / panel['volume_maN'] - 1).fillna(0)
        
        return features
    
    def extract_intraday_auction_features(self, 
                                         auction_data: pd.DataFrame) -> pd.DataFrame:
        """
        提取实时竞价特征（9:15-9:25实时数据）
        
        Parameters:
        -----------
        auction_data: DataFrame
            实时竞价数据，包含：time, symbol, price, volume, buy_volume, sell_volume
            
        Returns:
        --------
        DataFrame: 实时竞价特征
        """
        if auction_data.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame()
        
        # 按股票分组计算
        for symbol, group in auction_data.groupby('symbol'):
            symbol_features = {}
            symbol_features['symbol'] = symbol
            
            # 1. 竞价涨幅（最新价相对昨收）
            if 'prev_close' in group.columns:
                symbol_features['auction_gap'] = (
                    group['price'].iloc[-1] / group['prev_close'].iloc[-1] - 1
                )
            
            # 2. 竞价量能比（竞价量相对昨日首5分钟）
            if 'prev_5min_volume' in group.columns:
                symbol_features['auction_volume_ratio'] = (
                    group['volume'].sum() / group['prev_5min_volume'].iloc[-1]
                )
            else:
                symbol_features['auction_volume_ratio'] = 1.0
            
            # 3. 买卖单比
            if 'buy_volume' in group.columns and 'sell_volume' in group.columns:
                total_sell = group['sell_volume'].sum()
                symbol_features['buy_sell_ratio'] = (
                    group['buy_volume'].sum() / total_sell if total_sell > 0 else 1.0
                )
            
            # 4. 大单占比（>10万的单子）
            if 'order_size' in group.columns:
                total_orders = len(group)
                big_orders = len(group[group['order_size'] > 100000])
                symbol_features['big_order_ratio'] = big_orders / total_orders if total_orders > 0 else 0
            
            # 5. 价格稳定性（9:20-9:25波动率）
            if len(group) >= 2:
                price_std = group['price'].std()
                price_mean = group['price'].mean()
                symbol_features['price_stability'] = 1 - (price_std / price_mean if price_mean > 0 else 0)
            else:
                symbol_features['price_stability'] = 1.0
            
            # 6. 最后冲刺（9:24-9:25增量）
            if len(group) >= 2:
                last_min_volume = group.iloc[-1]['volume']
                prev_avg_volume = group.iloc[:-1]['volume'].mean()
                symbol_features['last_minute_surge'] = (
                    last_min_volume / prev_avg_volume if prev_avg_volume > 0 else 1.0
                )
            else:
                symbol_features['last_minute_surge'] = 1.0
            
            features = pd.concat([features, pd.DataFrame([symbol_features])], ignore_index=True)
        
        return features
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        获取所有特征名称（分类）
        
        Returns:
        --------
        Dict: 特征名称字典
        """
        return {
            'T日封单特征': [
                't_day_return', 't_day_is_limitup', 't_day_close_strength',
                't_day_volume_ratio', 't_day_amplitude', 't_day_turnover'
            ],
            '历史竞价表现': [
                'avg_auction_gap_5d', 'auction_volatility', 'auction_high_open_ratio',
                'auction_low_open_ratio', 'auction_consistency'
            ],
            '市场环境': [
                'market_limitup_count', 'market_limitup_ratio', 'market_sentiment',
                'sector_limitup_count'
            ],
            '连续性特征': [
                'continuous_up_days', 'continuous_limitup_days', 'max_return_Nd',
                'cum_return_Nd', 'price_momentum_Nd', 'volume_trend'
            ]
        }
