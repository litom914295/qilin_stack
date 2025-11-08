"""
一进二标签生成器 - T+1制度适配版
生成多维度标签：T+1收盘收益、日内最高/最低、T+2最佳卖出收益、竞价强度
适配A股T+1交易制度：买入后必须持有到次日才能卖出
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta
import warnings


class OneIntoTwoLabeler:
    """
    一进二标签生成器 - T+1制度适配版
    
    核心标签：
    - t1_close_return: T+1收盘收益率（相对买入价）
    - t1_max_return: T+1日内最高收益率
    - t1_min_return: T+1日内最低收益率（风险指标）
    - t2_best_sell_return: T+2最佳卖出收益率
    - auction_strength: T+1竞价强度等级（0-3）
    - pool_label: T日是否涨停（兼容旧版）
    - board_label: T+1是否涨停（兼容旧版）
    """
    
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
                   touch_or_close: str = 'close',
                   include_advanced: bool = True) -> pd.DataFrame:
        """
        生成多维度标签（T+1制度适配）
        
        Parameters:
        -----------
        panel: DataFrame
            必须包含列: date, symbol, open, high, low, close
            按(date, symbol)排序
        touch_or_close: str
            'touch': 触及涨停即算（使用high）
            'close': 收盘涨停才算（使用close）
        include_advanced: bool
            是否生成高级标签（T+1收益、竞价强度等）
            
        Returns:
        --------
        DataFrame: 包含多维度标签的数据
        """
        if panel.empty:
            base_cols = ['date', 'symbol', 'pool_label', 'board_label']
            if include_advanced:
                base_cols.extend(['t1_close_return', 't1_max_return', 't1_min_return', 
                                 't2_best_sell_return', 'auction_strength'])
            return pd.DataFrame(columns=base_cols)
        
        # 确保按日期和股票排序
        panel = panel.sort_values(['symbol', 'date']).copy()
        
        # 计算前一日收盘价
        panel['prev_close'] = panel.groupby('symbol')['close'].shift(1)
        
        # 根据股票代码获取涨跌停限制
        panel['limit_rate'] = panel['symbol'].apply(self.get_limit_rate)
        
        # 判断涨停价格字段
        price_field = 'high' if touch_or_close == 'touch' else 'close'
        
        # ========== 基础标签（兼容旧版）==========
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
        panel['prev_pool'] = panel.groupby('symbol')['pool_label'].shift(1)
        panel['next_limitup'] = panel.groupby('symbol')['pool_label'].shift(-1)
        
        panel['board_label'] = 0
        mask = (panel['prev_pool'] == 1) & (panel['pool_label'] == 1)
        panel.loc[mask, 'board_label'] = 1
        
        # ========== 高级标签（T+1制度）==========
        if include_advanced:
            # 获取T+1和T+2的数据
            panel['t1_open'] = panel.groupby('symbol')['open'].shift(-1)
            panel['t1_high'] = panel.groupby('symbol')['high'].shift(-1)
            panel['t1_low'] = panel.groupby('symbol')['low'].shift(-1)
            panel['t1_close'] = panel.groupby('symbol')['close'].shift(-1)
            
            panel['t2_open'] = panel.groupby('symbol')['open'].shift(-2)
            panel['t2_high'] = panel.groupby('symbol')['high'].shift(-2)
            panel['t2_close'] = panel.groupby('symbol')['close'].shift(-2)
            
            # 计算买入价格（T+1开盘价或竞价价，这里简化为开盘价）
            panel['buy_price'] = panel['t1_open']
            
            # 1. T+1收盘收益率（核心指标）
            panel['t1_close_return'] = (panel['t1_close'] / panel['buy_price'] - 1).fillna(0)
            
            # 2. T+1日内最高收益率
            panel['t1_max_return'] = (panel['t1_high'] / panel['buy_price'] - 1).fillna(0)
            
            # 3. T+1日内最低收益率（风险指标）
            panel['t1_min_return'] = (panel['t1_low'] / panel['buy_price'] - 1).fillna(0)
            
            # 4. T+2最佳卖出收益率（考虑开盘和日内最高）
            t2_open_return = (panel['t2_open'] / panel['buy_price'] - 1).fillna(0)
            t2_high_return = (panel['t2_high'] / panel['buy_price'] - 1).fillna(0)
            panel['t2_best_sell_return'] = np.maximum(t2_open_return, t2_high_return)
            
            # 5. T+1竞价强度等级（根据T+1开盘涨幅分级）
            panel['auction_gap'] = (panel['t1_open'] / panel['close'] - 1).fillna(0)
            panel['auction_strength'] = pd.cut(
                panel['auction_gap'],
                bins=[-np.inf, 0.01, 0.03, 0.05, np.inf],
                labels=[0, 1, 2, 3],  # 0-弱势, 1-中等, 2-强势, 3-超强
                include_lowest=True
            ).astype(float).fillna(0)
        
        # 清理临时列，只保留需要的
        keep_cols = ['date', 'symbol', 'pool_label', 'board_label']
        if include_advanced:
            keep_cols.extend(['t1_close_return', 't1_max_return', 't1_min_return', 
                            't2_best_sell_return', 'auction_strength'])
        
        result = panel[keep_cols].copy()
        
        # 统计信息
        pool_ratio = result['pool_label'].mean()
        board_ratio = result['board_label'].mean()
        print(f"\n{'='*60}")
        print(f"标签生成完成 (T+1制度适配版)")
        print(f"{'='*60}")
        print(f"样本总数: {len(result):,}")
        print(f"Pool标签比例: {pool_ratio:.2%} (T日涨停)")
        print(f"Board标签比例: {board_ratio:.2%} (T+1继续涨停)")
        
        if include_advanced and len(result[result['pool_label'] == 1]) > 0:
            success_cases = result[result['pool_label'] == 1]
            print(f"\n--- T+1表现统计 (基于T日涨停的{len(success_cases)}个样本) ---")
            print(f"T+1平均收盘收益: {success_cases['t1_close_return'].mean():.2%}")
            print(f"T+1收盘盈利率: {(success_cases['t1_close_return'] > 0).mean():.2%}")
            print(f"T+1平均最大浮亏: {success_cases['t1_min_return'].mean():.2%}")
            print(f"T+2平均最佳收益: {success_cases['t2_best_sell_return'].mean():.2%}")
            
            # 竞价强度分布
            auction_dist = success_cases['auction_strength'].value_counts(normalize=True).sort_index()
            print(f"\n--- T+1竞价强度分布 ---")
            strength_labels = {0: '弱势(<1%)', 1: '中等(1-3%)', 2: '强势(3-5%)', 3: '超强(>5%)'}
            for level, ratio in auction_dist.items():
                print(f"{strength_labels.get(level, level)}: {ratio:.2%}")
        
        print(f"{'='*60}\n")
        
        return result
    
    def make_labels_with_features(self,
                                  panel: pd.DataFrame,
                                  feature_df: Optional[pd.DataFrame] = None,
                                  include_advanced: bool = True) -> pd.DataFrame:
        """
        生成标签并合并特征
        
        Parameters:
        -----------
        panel: 基础行情数据
        feature_df: 额外特征数据（可选）
        include_advanced: 是否生成高级标签
        
        Returns:
        --------
        包含标签和特征的完整DataFrame
        """
        # 生成标签
        labels = self.make_labels(panel, include_advanced=include_advanced)
        
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
    
    def get_label_statistics(self, labeled_data: pd.DataFrame) -> Dict[str, float]:
        """
        获取标签统计信息
        
        Parameters:
        -----------
        labeled_data: 已标注的数据
        
        Returns:
        --------
        Dict: 统计指标字典
        """
        stats = {
            'total_samples': len(labeled_data),
            'pool_ratio': labeled_data['pool_label'].mean() if 'pool_label' in labeled_data else 0,
            'board_ratio': labeled_data['board_label'].mean() if 'board_label' in labeled_data else 0,
        }
        
        # 如果有高级标签，计算更多统计
        if 't1_close_return' in labeled_data.columns:
            pool_cases = labeled_data[labeled_data['pool_label'] == 1]
            if len(pool_cases) > 0:
                stats.update({
                    't1_avg_return': pool_cases['t1_close_return'].mean(),
                    't1_positive_rate': (pool_cases['t1_close_return'] > 0).mean(),
                    't1_avg_max_return': pool_cases['t1_max_return'].mean(),
                    't1_avg_min_return': pool_cases['t1_min_return'].mean(),
                    't2_avg_best_return': pool_cases['t2_best_sell_return'].mean(),
                    'auction_strong_ratio': (pool_cases['auction_strength'] >= 2).mean(),
                })
        
        return stats
