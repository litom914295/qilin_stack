"""
一进二特征构建器
统一训练和推理的特征构建流程，确保特征口径一致
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 复用现有模块
from factors.limitup_advanced_factors import LimitUpAdvancedFactors
from qlib_enhanced.one_into_two_pipeline import extract_limitup_features

# 导入竞价特征提取器
try:
    from features.auction_features import AuctionFeatureExtractor
except ImportError:
    AuctionFeatureExtractor = None


class OneIntoTwoFeatureBuilder:
    """一进二特征构建器（增强竞价特征）"""
    
    def __init__(self, premium_provider=None):
        """
        Parameters:
        -----------
        premium_provider: PremiumDataProvider实例（可选）
            用于获取高级数据（封单、题材等）
        """
        self.premium_provider = premium_provider
        self.factor_calculator = LimitUpAdvancedFactors()
        
        # 竞价特征提取器
        if AuctionFeatureExtractor is not None:
            self.auction_extractor = AuctionFeatureExtractor()
        else:
            self.auction_extractor = None
        
        # 定义特征列表（确保训练和推理一致）
        self.base_features = [
            'ret_day', 'vol_burst', 'late_strength', 'open_up',
            'high_ratio', 'low_drawdown', 'vwap', 'volatility'
        ]
        
        self.advanced_features = [
            'seal_strength', 'open_count', 'limitup_time_score',
            'board_height', 'market_sentiment', 'leader_score',
            'big_order_ratio', 'theme_decay'
        ]
        
        self.all_features = self.base_features + self.advanced_features
        
    def build_train_features(self, 
                            panel: pd.DataFrame,
                            premium_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        构建训练特征
        
        Parameters:
        -----------
        panel: 基础行情数据（OHLCV）
        premium_data: 高级数据（封单、题材等）
        
        Returns:
        --------
        包含所有特征的DataFrame
        """
        result = panel.copy()
        
        # 1. 合并premium数据（如果有）
        if premium_data is not None:
            # 确保索引对齐
            merge_keys = ['date', 'symbol'] if 'date' in premium_data.columns else ['symbol']
            result = result.merge(premium_data, on=merge_keys, how='left')
        
        # 2. 如果有premium_provider，尝试获取缺失的高级数据
        if self.premium_provider is not None:
            try:
                # 获取所有日期的高级数据
                for date in result['date'].unique():
                    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                    daily_metrics = self.premium_provider.get_daily_advanced_metrics(date_str)
                    
                    # 合并到结果中
                    date_mask = result['date'] == date
                    for col in daily_metrics.columns:
                        if col not in result.columns:
                            result.loc[date_mask, col] = result.loc[date_mask, 'symbol'].map(
                                daily_metrics.set_index('symbol')[col].to_dict()
                            )
            except Exception as e:
                print(f"获取premium数据失败，使用默认值: {e}")
        
        # 3. 计算高级因子
        # 补充缺失的必要字段（使用默认值或代理）
        self._fill_missing_fields(result)
        
        # 调用高级因子计算器
        result = self.factor_calculator.calculate_all_factors(result)
        
        # 4. 计算基础特征（如果有分钟数据）
        # 这里简化处理，使用日线数据计算代理特征
        if 'ret_day' not in result.columns:
            result = self._calculate_base_features(result)
        
        # 5. 确保所有特征都存在（缺失的填0）
        for feat in self.all_features:
            if feat not in result.columns:
                result[feat] = 0.0
        
        return result
    
    def build_infer_features(self, 
                            today_limitups: pd.DataFrame,
                            minute_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        构建推理特征（T日涨停池 -> T+1预测）
        
        Parameters:
        -----------
        today_limitups: 今日涨停股票数据
        minute_data: 分钟级数据字典（可选）
        
        Returns:
        --------
        推理用特征DataFrame
        """
        result = today_limitups.copy()
        
        # 1. 如果有分钟数据，计算微观结构特征
        if minute_data:
            for symbol in result['symbol'].unique():
                if symbol in minute_data:
                    # 使用现有的extract_limitup_features
                    features = extract_limitup_features(minute_data[symbol], symbol)
                    
                    # 更新到结果中
                    mask = result['symbol'] == symbol
                    for key, value in features.items():
                        if key not in result.columns:
                            result.loc[mask, key] = value
        
        # 2. 获取今日的高级数据
        if self.premium_provider is not None:
            try:
                from datetime import datetime
                today = datetime.now().strftime('%Y-%m-%d')
                daily_metrics = self.premium_provider.get_daily_advanced_metrics(today)
                market_sentiment = self.premium_provider.get_market_sentiment(today)
                
                # 合并数据
                if not daily_metrics.empty:
                    result = result.merge(daily_metrics, on='symbol', how='left')
                
                # 添加市场情绪
                result['total_limitup'] = market_sentiment.get('limit_up_count', 0)
                result['market_sentiment'] = self._classify_market_sentiment(
                    market_sentiment.get('limit_up_count', 0)
                )
            except Exception as e:
                print(f"获取今日premium数据失败: {e}")
        
        # 3. 补充缺失字段并计算高级因子
        self._fill_missing_fields(result)
        result = self.factor_calculator.calculate_all_factors(result)
        
        # 4. 计算基础特征
        if 'ret_day' not in result.columns:
            result = self._calculate_base_features(result)
        
        # 5. 提取竞价特征（推理时）
        if self.auction_extractor is not None:
            try:
                print("提取竞价预测特征（推理模式）...")
                # 构建历史数据panel用于特征提取
                # 这里简化处理，实际应该传入完整的历史数据
                auction_features = self.auction_extractor.extract_auction_predictive_features(
                    panel=result, lookback_days=5
                )
                # 合并竞价特征
                if not auction_features.empty:
                    result = result.merge(
                        auction_features, 
                        on=['date', 'symbol'], 
                        how='left',
                        suffixes=('', '_auction')
                    )
            except Exception as e:
                print(f"提取竞价特征失败（推理）: {e}")
        
        # 6. 确保特征完整性（与训练一致）
        for feat in self.all_features:
            if feat not in result.columns:
                result[feat] = 0.0
        
        # 只返回需要的特征列
        feature_cols = ['date', 'symbol'] + self.all_features
        available_cols = [col for col in feature_cols if col in result.columns]
        
        return result[available_cols]
    
    def _fill_missing_fields(self, df: pd.DataFrame):
        """补充缺失的必要字段"""
        
        # 涨停相关
        if 'is_limitup' not in df.columns:
            df['is_limitup'] = 1  # 推理时默认都是涨停股
        
        if 'limitup_time' not in df.columns:
            df['limitup_time'] = '10:00:00'  # 默认10点涨停
        
        if 'buy_amount' not in df.columns and 'seal_amount' in df.columns:
            df['buy_amount'] = df['seal_amount']
        elif 'buy_amount' not in df.columns:
            # 使用成交额的一定比例作为代理
            if 'amount' in df.columns:
                df['buy_amount'] = df['amount'] * 0.2
            else:
                df['buy_amount'] = 1e7  # 默认1000万
        
        # 流通市值
        if 'float_mv' not in df.columns:
            if 'circulating_market_cap' in df.columns:
                df['float_mv'] = df['circulating_market_cap']
            else:
                # 使用价格和成交量估算
                if 'close' in df.columns and 'volume' in df.columns:
                    df['float_mv'] = df['close'] * df['volume'] * 100  # 粗略估算
                else:
                    df['float_mv'] = 1e10  # 默认100亿
        
        # 大单相关
        if 'big_buy_volume' not in df.columns:
            if 'volume' in df.columns:
                df['big_buy_volume'] = df['volume'] * 0.3  # 假设30%是大单
            else:
                df['big_buy_volume'] = 1e6
        
        if 'total_buy_volume' not in df.columns:
            if 'volume' in df.columns:
                df['total_buy_volume'] = df['volume'] * 0.6  # 假设60%是买入
            else:
                df['total_buy_volume'] = 3e6
        
        # 换手率
        if 'turnover' not in df.columns:
            df['turnover'] = 10.0  # 默认10%换手率
        
        # 板块和题材
        if 'industry' not in df.columns and 'sector' in df.columns:
            df['industry'] = df['sector']
        elif 'industry' not in df.columns:
            df['industry'] = 'unknown'
        
        if 'theme' not in df.columns and 'themes' in df.columns:
            df['theme'] = df['themes'].apply(
                lambda x: x.split(',')[0] if isinstance(x, str) else 'unknown'
            )
        elif 'theme' not in df.columns:
            df['theme'] = 'unknown'
    
    def _calculate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础特征（使用日线数据）"""
        
        # 日收益率
        if 'ret_day' not in df.columns:
            df['ret_day'] = df.groupby('symbol')['close'].pct_change()
        
        # 成交量爆发
        if 'vol_burst' not in df.columns:
            df['vol_burst'] = df.groupby('symbol')['volume'].transform(
                lambda x: x / x.rolling(5, min_periods=1).mean()
            ).fillna(1.0)
        
        # 尾盘强度（简化：使用收盘价相对位置）
        if 'late_strength' not in df.columns:
            df['late_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # 开盘涨幅
        if 'open_up' not in df.columns:
            df['open_up'] = (df['open'] / df.groupby('symbol')['close'].shift(1) - 1).fillna(0)
        
        # 最高点相对强度
        if 'high_ratio' not in df.columns:
            df['high_ratio'] = (df['high'] / df['open'] - 1).fillna(0)
        
        # 最低点回撤
        if 'low_drawdown' not in df.columns:
            df['low_drawdown'] = (df['low'] / df['open'] - 1).fillna(0)
        
        # VWAP（简化：使用均价）
        if 'vwap' not in df.columns:
            if 'amount' in df.columns and 'volume' in df.columns:
                df['vwap'] = df['amount'] / (df['volume'] + 1e-8)
            else:
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 波动率
        if 'volatility' not in df.columns:
            df['volatility'] = df.groupby('symbol')['ret_day'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            ).fillna(0.02)
        
        return df
    
    def _classify_market_sentiment(self, limitup_count: int) -> str:
        """根据涨停数量判断市场情绪"""
        if limitup_count >= 100:
            return 'strong'
        elif limitup_count >= 50:
            return 'neutral'
        else:
            return 'weak'
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.all_features