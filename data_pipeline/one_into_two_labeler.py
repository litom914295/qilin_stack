"""
一进二标签生成器
统一生成pool_label和board_label，确保仅使用T日可得信息
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta


class OneIntoTwoLabeler:
    """一进二标签生成器"""
    
    def __init__(self, 
                 limit_rate_10: float = 0.10,  # 主板涨跌停限制
                 limit_rate_20: float = 0.20,  # 科创板/创业板限制
                 limit_rate_5: float = 0.05):   # ST股票限制
        self.limit_rate_10 = limit_rate_10
        self.limit_rate_20 = limit_rate_20
        self.limit_rate_5 = limit_rate_5
    
    def get_limit_rate(self, symbol: str) -> float:
        """根据股票代码判断涨跌停限制"""
        # ST股票
        if 'ST' in symbol.upper() or '*ST' in symbol.upper():
            return self.limit_rate_5
        
        # 科创板（688开头）
        if symbol.startswith('688'):
            return self.limit_rate_20
        
        # 创业板（300开头）- 注册制后也是20%
        if symbol.startswith('300'):
            return self.limit_rate_20
        
        # 北交所（8开头，暂按30%，可调整）
        if symbol.startswith('8') and len(symbol) == 6:
            return 0.30
        
        # 默认主板10%
        return self.limit_rate_10
    
    def is_limit_up(self, 
                     yesterday_close: float, 
                     today_price: float, 
                     limit_rate: float) -> bool:
        """判断是否触及涨停（含一定容差）"""
        if yesterday_close <= 0:
            return False
        
        limit_price = yesterday_close * (1.0 + limit_rate)
        # 允许0.1%的容差（应对浮点数精度问题）
        return today_price >= limit_price * 0.999
    
    def make_labels(self, 
                   panel: pd.DataFrame,
                   touch_or_close: str = 'close') -> pd.DataFrame:
        """
        生成pool和board标签
        
        Parameters:
        -----------
        panel: DataFrame
            必须包含列: date, symbol, open, high, low, close
            按(date, symbol)排序
        touch_or_close: str
            'touch': 触及涨停即算（使用high）
            'close': 收盘涨停才算（使用close）
            
        Returns:
        --------
        DataFrame: 包含[date, symbol, pool_label, board_label]
        """
        if panel.empty:
            return pd.DataFrame(columns=['date', 'symbol', 'pool_label', 'board_label'])
        
        # 确保按日期和股票排序
        panel = panel.sort_values(['symbol', 'date']).copy()
        
        # 计算前一日收盘价
        panel['prev_close'] = panel.groupby('symbol')['close'].shift(1)
        
        # 根据股票代码获取涨跌停限制
        panel['limit_rate'] = panel['symbol'].apply(self.get_limit_rate)
        
        # 判断涨停价格字段
        price_field = 'high' if touch_or_close == 'touch' else 'close'
        
        # 计算pool_label: T日是否涨停（首板候选池）
        panel['pool_label'] = panel.apply(
            lambda row: 1 if self.is_limit_up(
                row['prev_close'], 
                row[price_field], 
                row['limit_rate']
            ) else 0,
            axis=1
        )
        
        # 计算board_label: T日涨停且T+1日继续涨停（二板）
        # 需要前移pool_label并与下一日涨停判断结合
        panel['prev_pool'] = panel.groupby('symbol')['pool_label'].shift(1)
        panel['next_limitup'] = panel.groupby('symbol')['pool_label'].shift(-1)
        
        # board_label = 1 当且仅当：T-1日涨停(prev_pool=1) 且 T日涨停(pool_label=1)
        # 这样T日预测的是"T+1是否继续涨停"
        panel['board_label'] = 0
        mask = (panel['prev_pool'] == 1) & (panel['pool_label'] == 1)
        panel.loc[mask, 'board_label'] = 1
        
        # 清理临时列，只保留需要的
        result = panel[['date', 'symbol', 'pool_label', 'board_label']].copy()
        
        # 统计信息
        pool_ratio = result['pool_label'].mean()
        board_ratio = result['board_label'].mean()
        print(f"标签生成完成: pool_label比例={pool_ratio:.2%}, board_label比例={board_ratio:.2%}")
        
        return result
    
    def make_labels_with_features(self,
                                  panel: pd.DataFrame,
                                  feature_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        生成标签并合并特征
        
        Parameters:
        -----------
        panel: 基础行情数据
        feature_df: 额外特征数据（可选）
        
        Returns:
        --------
        包含标签和特征的完整DataFrame
        """
        # 生成标签
        labels = self.make_labels(panel)
        
        # 如果有额外特征，合并
        if feature_df is not None:
            # 确保索引对齐
            labels = labels.set_index(['date', 'symbol'])
            feature_df = feature_df.set_index(['date', 'symbol'])
            
            # 左连接，保留所有标签样本
            result = labels.join(feature_df, how='left')
            result = result.reset_index()
        else:
            result = labels
        
        return result