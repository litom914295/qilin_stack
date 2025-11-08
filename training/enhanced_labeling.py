#!/usr/bin/env python
"""
增强标签生成器
专注于首板次日大涨/涨停/连板的多维度标签
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class EnhancedLabeling:
    """增强标签生成器 - 多维度标签体系"""
    
    def __init__(self):
        # 成功标准（重点！）
        self.SUCCESS_THRESHOLDS = {
            'excellent': 0.095,  # 涨停
            'great': 0.05,       # 大涨
            'good': 0.02,        # 上涨
            'neutral': 0.0,      # 平衡
            'bad': -0.02         # 下跌
        }
    
    def create_enhanced_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建增强标签
        
        核心目标：准确标记首板次日大涨/涨停的成功案例
        """
        
        data = data.copy()
        
        # 1. 计算未来收益率
        data = self._calculate_future_returns(data)
        
        # 2. 主标签（4分类）- 最重要！
        data = self._create_main_label(data)
        
        # 3. 持续性标签
        data = self._create_sustainability_label(data)
        
        # 4. 成功概率标签
        data = self._create_success_probability_label(data)
        
        # 5. 最大收益和回撤标签
        data = self._create_maxmin_labels(data)
        
        # 6. 连板成功标签
        data = self._create_consecutive_board_label(data)
        
        return data
    
    def _calculate_future_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算未来N日收益率"""
        
        for days in [1, 2, 3, 5, 10]:
            data[f'return_{days}d'] = (
                data.groupby('code')['close']
                .pct_change(days)
                .shift(-days)
            )
        
        return data
    
    def _create_main_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建主标签（4分类）
        
        这是最重要的标签！
        - 3: 次日涨停（优秀）
        - 2: 次日大涨（>5%）
        - 1: 次日上涨（2-5%）
        - 0: 次日震荡或下跌
        """
        
        data['main_label'] = pd.cut(
            data['return_1d'],
            bins=[-np.inf, 0.02, 0.05, 0.095, np.inf],
            labels=[0, 1, 2, 3],
            include_lowest=True
        )
        
        # 转换为整数
        data['main_label'] = data['main_label'].astype('int')
        
        return data
    
    def _create_sustainability_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建持续性标签
        
        评估首板后的持续上涨能力
        - 3: 超级强势（8天+持续上涨）
        - 2: 中线强势（4-7天）
        - 1: 短线强势（2-3天）
        - 0: 一日游（次日即跌）
        """
        
        def calculate_sustainability(row):
            # 检查是否有数据
            if pd.isna(row['return_1d']):
                return 0
            
            # 统计连续上涨天数
            up_days = 0
            if row['return_1d'] > 0.02:
                up_days += 1
            if not pd.isna(row['return_2d']) and row['return_2d'] > row['return_1d']:
                up_days += 1
            if not pd.isna(row['return_3d']) and row['return_3d'] > row['return_2d']:
                up_days += 1
            
            # 判断持续性
            if up_days >= 3 and row.get('return_5d', 0) > 0.2:
                return 3  # 超级强势
            elif up_days >= 2 and row.get('return_5d', 0) > 0.1:
                return 2  # 中线强势
            elif row['return_1d'] > 0.02 and row.get('return_2d', 0) > 0:
                return 1  # 短线强势
            else:
                return 0  # 一日游
        
        data['sustainability'] = data.apply(calculate_sustainability, axis=1)
        
        return data
    
    def _create_success_probability_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建成功概率标签
        
        基于历史同类案例的成功率
        """
        
        # 按相似特征分组计算成功率
        feature_cols = ['sector', 'seal_strength_category', 'theme_hotness_category']
        
        # 创建分类特征
        data['seal_strength_category'] = pd.cut(
            data['seal_strength'],
            bins=[0, 60, 80, 90, 100],
            labels=['weak', 'medium', 'strong', 'very_strong']
        )
        
        data['theme_hotness_category'] = pd.cut(
            data.get('theme_hotness', pd.Series([0]*len(data))),
            bins=[-1, 2, 5, 10, 100],
            labels=['cold', 'warm', 'hot', 'super_hot']
        )
        
        # 计算每组的成功率（次日涨幅>5%为成功）
        success_rate = (
            data.groupby(feature_cols, observed=True)['return_1d']
            .apply(lambda x: (x > 0.05).sum() / len(x) if len(x) > 0 else 0.5)
            .to_dict()
        )
        
        # 映射成功概率
        data['success_probability'] = data.apply(
            lambda row: success_rate.get(
                (row['sector'], row['seal_strength_category'], row['theme_hotness_category']),
                0.5  # 默认50%
            ),
            axis=1
        )
        
        # 分类为高/中/低
        data['success_probability_category'] = pd.cut(
            data['success_probability'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        return data
    
    def _create_maxmin_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建最大收益和最大回撤标签"""
        
        # 5日内最高涨幅
        data['max_return_5d'] = data[
            ['return_1d', 'return_2d', 'return_3d', 'return_5d']
        ].max(axis=1)
        
        # 5日内最大回撤
        data['max_drawdown_5d'] = data[
            ['return_1d', 'return_2d', 'return_3d', 'return_5d']
        ].min(axis=1)
        
        # 盈亏比
        data['profit_loss_ratio'] = np.where(
            data['max_drawdown_5d'] < 0,
            data['max_return_5d'] / abs(data['max_drawdown_5d']),
            data['max_return_5d'] * 10  # 无回撤则设为高值
        )
        
        return data
    
    def _create_consecutive_board_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建连板成功标签
        
        判断首板是否能成功连板
        """
        
        # 次日是否涨停
        data['next_day_limitup'] = (data['return_1d'] >= 0.095).astype(int)
        
        # 2连板标签
        data['is_2board_success'] = data['next_day_limitup']
        
        # 连板预期（基于次日涨幅判断）
        data['board_continuation_score'] = np.where(
            data['return_1d'] >= 0.095, 10,  # 涨停=满分
            np.where(
                data['return_1d'] >= 0.05, 7,  # 大涨=7分
                np.where(
                    data['return_1d'] >= 0.02, 4,  # 小涨=4分
                    0  # 其他=0分
                )
            )
        )
        
        return data
    
    def get_label_statistics(self, data: pd.DataFrame) -> Dict:
        """获取标签统计信息"""
        
        stats = {
            'total_samples': len(data),
            'main_label_dist': data['main_label'].value_counts().to_dict(),
            'sustainability_dist': data['sustainability'].value_counts().to_dict(),
            'success_prob_dist': data['success_probability_category'].value_counts().to_dict(),
            'next_day_limitup_rate': data['next_day_limitup'].mean(),
            'avg_return_1d': data['return_1d'].mean(),
            'median_return_1d': data['return_1d'].median(),
        }
        
        # 成功率统计
        stats['success_rates'] = {
            'excellent_rate': (data['main_label'] == 3).mean(),  # 涨停率
            'great_rate': (data['main_label'] == 2).mean(),      # 大涨率
            'good_rate': (data['main_label'] == 1).mean(),       # 上涨率
            'combined_success_rate': (data['main_label'] >= 2).mean()  # 大涨+涨停
        }
        
        return stats
    
    def create_weighted_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建加权标签（用于训练）
        
        对成功案例（尤其是涨停）赋予更高权重
        """
        
        data = data.copy()
        
        # 样本权重
        weights = {
            0: 0.5,   # 震荡/下跌 - 低权重
            1: 1.0,   # 小涨 - 正常权重
            2: 2.0,   # 大涨 - 2倍权重
            3: 3.0    # 涨停 - 3倍权重（最重要！）
        }
        
        data['sample_weight'] = data['main_label'].map(weights)
        
        # 额外加权：持续性好的样本
        data['sample_weight'] = data['sample_weight'] * (1 + data['sustainability'] * 0.2)
        
        return data
    
    def filter_quality_samples(
        self, 
        data: pd.DataFrame, 
        min_success_rate: float = 0.3
    ) -> pd.DataFrame:
        """
        过滤高质量样本
        
        移除明显错误或低质量的数据
        """
        
        filtered = data.copy()
        
        # 移除缺失标签
        filtered = filtered[filtered['main_label'].notna()]
        
        # 移除异常收益率（可能是数据错误）
        filtered = filtered[
            (filtered['return_1d'] >= -0.15) & 
            (filtered['return_1d'] <= 0.2)
        ]
        
        # 移除封板强度异常
        filtered = filtered[
            (filtered['seal_strength'] >= 0) & 
            (filtered['seal_strength'] <= 100)
        ]
        
        return filtered


def demo_usage():
    """演示用法"""
    
    # 创建示例数据
    data = pd.DataFrame({
        'code': ['000001'] * 10,
        'date': pd.date_range('2024-01-01', periods=10),
        'close': [10, 11, 11.5, 11.8, 10.9, 11.2, 12, 13.2, 13, 13.5],
        'sector': ['科技'] * 10,
        'seal_strength': [85, 90, 70, 95, 60, 80, 88, 92, 75, 85],
        'theme_hotness': [6, 7, 5, 8, 4, 6, 7, 9, 5, 6],
    })
    
    # 创建标签
    labeler = EnhancedLabeling()
    labeled_data = labeler.create_enhanced_labels(data)
    
    print("标签统计:")
    print(labeler.get_label_statistics(labeled_data))
    
    # 创建加权标签
    weighted_data = labeler.create_weighted_labels(labeled_data)
    
    print("\n加权后前5行:")
    print(weighted_data[['date', 'main_label', 'sustainability', 'sample_weight']].head())


if __name__ == '__main__':
    demo_usage()
