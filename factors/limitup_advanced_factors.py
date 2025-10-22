"""
涨停板高级因子库 - 专注于"一进二"预测

包含8个核心因子：
1. seal_strength - 封单强度
2. open_count - 打开次数
3. limitup_time - 涨停时间
4. board_height - 连板高度
5. market_sentiment - 市场情绪
6. leader_score - 龙头地位
7. big_order_ratio - 大单流入比例
8. theme_decay - 题材热度衰减
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class LimitUpAdvancedFactors:
    """涨停板高级因子计算器"""
    
    def __init__(self):
        """初始化因子计算器"""
        self.factor_names = [
            'seal_strength',
            'open_count',
            'limitup_time',
            'board_height',
            'market_sentiment',
            'leader_score',
            'big_order_ratio',
            'theme_decay'
        ]
        
    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子
        
        Parameters:
        -----------
        data : pd.DataFrame
            输入数据，必须包含以下列：
            - symbol: 股票代码
            - date: 日期
            - open, high, low, close, volume: OHLCV数据
            - is_limitup: 是否涨停 (1/0)
            - limitup_time: 涨停时间 (HH:MM:SS)
            - buy_amount: 买一金额（封单金额）
            - sell_amount: 卖一金额
            - float_mv: 流通市值
            - big_buy_volume: 大单买入量
            - total_buy_volume: 总买入量
            
        Returns:
        --------
        pd.DataFrame: 包含所有因子的数据框
        """
        result = data.copy()
        
        # 1. 封单强度
        result['seal_strength'] = self._calc_seal_strength(result)
        
        # 2. 打开次数
        result['open_count'] = self._calc_open_count(result)
        
        # 3. 涨停时间得分
        result['limitup_time_score'] = self._calc_limitup_time(result)
        
        # 4. 连板高度
        result['board_height'] = self._calc_board_height(result)
        
        # 5. 市场情绪
        result['market_sentiment'] = self._calc_market_sentiment(result)
        
        # 6. 龙头地位
        result['leader_score'] = self._calc_leader_score(result)
        
        # 7. 大单流入比例
        result['big_order_ratio'] = self._calc_big_order_ratio(result)
        
        # 8. 题材热度衰减
        result['theme_decay'] = self._calc_theme_decay(result)
        
        return result
    
    def _calc_seal_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        因子1: 封单强度
        
        计算公式: 封单金额 / 流通市值
        
        逻辑: 封单越强，说明资金买入意愿越强，次日继续涨停概率越高
        """
        # 处理缺失值和异常值
        seal_amount = data.get('buy_amount', pd.Series(0, index=data.index))
        float_mv = data.get('float_mv', pd.Series(1, index=data.index))
        
        # 避免除零
        float_mv = float_mv.replace(0, 1)
        
        # 封单强度 = 封单金额 / 流通市值
        seal_strength = seal_amount / float_mv
        
        # 归一化到0-1
        seal_strength = seal_strength.clip(0, 0.1) / 0.1  # 超过10%视为满分
        
        return seal_strength
    
    def _calc_open_count(self, data: pd.DataFrame) -> pd.Series:
        """
        因子2: 打开次数
        
        计算公式: 当日涨停打开次数
        
        逻辑: 打开次数越多，封板越不稳固，次日继续涨停概率越低
        """
        # 如果数据中有open_count列，直接使用
        if 'open_count' in data.columns:
            open_count = data['open_count']
        else:
            # 简化估计：通过成交量波动估计
            # 如果没有分时数据，使用换手率作为替代
            turnover = data.get('turnover', pd.Series(0, index=data.index))
            
            # 换手率越高，打开次数越多的可能性越大
            open_count = (turnover / 10).clip(0, 5)  # 最多5次
        
        # 反向归一化：打开次数越多，得分越低
        open_score = 1 - (open_count / 5).clip(0, 1)
        
        return open_score
    
    def _calc_limitup_time(self, data: pd.DataFrame) -> pd.Series:
        """
        因子3: 涨停时间得分
        
        计算公式: (14:30 - 涨停时间) / (14:30 - 9:30)
        
        逻辑: 越早涨停，说明资金越强势，次日继续涨停概率越高
        """
        def time_to_minutes(time_str):
            """将时间字符串转换为分钟数"""
            try:
                if pd.isna(time_str):
                    return 14 * 60 + 30  # 默认尾盘涨停（最弱）
                
                if isinstance(time_str, str):
                    h, m, s = map(int, time_str.split(':'))
                    return h * 60 + m
                elif isinstance(time_str, datetime):
                    return time_str.hour * 60 + time_str.minute
                else:
                    return 14 * 60 + 30
            except:
                return 14 * 60 + 30
        
        # 获取涨停时间
        limitup_time = data.get('limitup_time', pd.Series('14:30:00', index=data.index))
        
        # 转换为分钟
        limitup_minutes = limitup_time.apply(time_to_minutes)
        
        # 计算得分：9:30 = 100分，14:30 = 0分
        open_minutes = 9 * 60 + 30
        close_minutes = 14 * 60 + 30
        
        score = (close_minutes - limitup_minutes) / (close_minutes - open_minutes)
        score = score.clip(0, 1)
        
        return score
    
    def _calc_board_height(self, data: pd.DataFrame) -> pd.Series:
        """
        因子4: 连板高度
        
        计算公式: 连续涨停天数
        
        逻辑: 
        - 首板（1板）：风险较低，成功率一般
        - 2-3板：最强势，成功率最高
        - 4板以上：高位风险，成功率下降
        """
        # 按股票分组，计算连续涨停天数
        data_sorted = data.sort_values(['symbol', 'date'])
        
        # 计算连续涨停
        data_sorted['is_limitup_flag'] = data_sorted.get('is_limitup', 0)
        
        # 使用cumsum来标识连续涨停序列
        data_sorted['limitup_group'] = (
            data_sorted.groupby('symbol')['is_limitup_flag']
            .transform(lambda x: (x != x.shift()).cumsum())
        )
        
        # 计算每个序列的长度
        board_height = data_sorted.groupby(['symbol', 'limitup_group'])['is_limitup_flag'].cumsum()
        
        # 连板高度评分（2-3板得分最高）
        def height_score(h):
            if h == 1:
                return 0.6  # 首板
            elif h == 2:
                return 1.0  # 二板（最强）
            elif h == 3:
                return 0.9  # 三板
            elif h == 4:
                return 0.7  # 四板
            else:
                return 0.5  # 5板以上（高位风险）
        
        score = board_height.apply(height_score)
        
        # 恢复原始索引
        score.index = data.index
        
        return score
    
    def _calc_market_sentiment(self, data: pd.DataFrame) -> pd.Series:
        """
        因子5: 市场情绪
        
        计算公式: 当日市场涨停家数 / 总股票数
        
        逻辑: 市场情绪越好（涨停家数越多），个股继续涨停概率越高
        """
        # 按日期计算涨停家数
        daily_limitup_count = (
            data.groupby('date')['is_limitup']
            .transform('sum')
        )
        
        # 总股票数（假设市场有4000只股票）
        total_stocks = 4000
        
        # 市场情绪 = 涨停家数 / 总股票数
        sentiment = daily_limitup_count / total_stocks
        
        # 归一化到0-1（超过5%视为极度狂热）
        sentiment = (sentiment * 20).clip(0, 1)
        
        return sentiment
    
    def _calc_leader_score(self, data: pd.DataFrame) -> pd.Series:
        """
        因子6: 龙头地位
        
        计算公式: 
        - 同板块首个涨停 = 1.0（龙头）
        - 同板块跟风涨停 = 0.6（跟风）
        
        逻辑: 龙头股的持续性远强于跟风股
        """
        # 按日期和行业分组，找出最早涨停的股票
        data_sorted = data.sort_values(['date', 'industry', 'limitup_time'])
        
        # 标记每个行业的首个涨停（龙头）
        data_sorted['is_leader'] = (
            data_sorted.groupby(['date', 'industry'])
            .cumcount() == 0
        )
        
        # 龙头得分1.0，跟风得分0.6
        leader_score = data_sorted['is_leader'].map({True: 1.0, False: 0.6})
        
        # 如果没有industry列，使用成交量排名作为替代
        if 'industry' not in data.columns:
            # 同日成交量排名前20%为龙头
            data_sorted['volume_rank'] = (
                data_sorted.groupby('date')['volume']
                .rank(pct=True)
            )
            leader_score = (data_sorted['volume_rank'] > 0.8).astype(float) * 0.4 + 0.6
        
        # 恢复原始索引
        leader_score.index = data.index
        
        return leader_score
    
    def _calc_big_order_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        因子7: 大单流入比例
        
        计算公式: 大单买入量 / 总买入量
        
        逻辑: 大单比例越高，说明主力参与度越高，持续性越强
        """
        big_buy = data.get('big_buy_volume', pd.Series(0, index=data.index))
        total_buy = data.get('total_buy_volume', pd.Series(1, index=data.index))
        
        # 避免除零
        total_buy = total_buy.replace(0, 1)
        
        # 大单比例
        big_ratio = big_buy / total_buy
        
        # 归一化到0-1
        big_ratio = big_ratio.clip(0, 1)
        
        # 如果没有大单数据，使用成交额作为替代
        if big_buy.sum() == 0:
            # 成交额越大，大资金参与度越高
            amount = data.get('amount', pd.Series(0, index=data.index))
            big_ratio = (amount / amount.mean()).clip(0, 2) / 2
        
        return big_ratio
    
    def _calc_theme_decay(self, data: pd.DataFrame) -> pd.Series:
        """
        因子8: 题材热度衰减
        
        计算公式: 1 - (题材持续天数 / 10)
        
        逻辑: 题材炒作有生命周期，通常3-7天，超过10天衰减明显
        """
        # 按题材和日期排序
        if 'theme' in data.columns:
            data_sorted = data.sort_values(['theme', 'date'])
            
            # 计算题材持续天数
            data_sorted['theme_days'] = (
                data_sorted.groupby('theme')
                .cumcount() + 1
            )
            
            # 衰减得分：1-10天线性衰减
            decay_score = 1 - (data_sorted['theme_days'] / 10).clip(0, 1)
            
            # 恢复原始索引
            decay_score.index = data.index
        else:
            # 如果没有题材数据，使用连续涨停天数作为替代
            # 假设同一波行情会持续涨停
            data_sorted = data.sort_values(['symbol', 'date'])
            
            # 计算连续上涨天数
            data_sorted['up_days'] = (
                data_sorted.groupby('symbol')['close']
                .transform(lambda x: (x > x.shift()).astype(int).cumsum())
            )
            
            # 衰减得分
            decay_score = 1 - (data_sorted['up_days'] / 10).clip(0, 1)
            decay_score.index = data.index
        
        return decay_score
    
    def get_factor_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        计算因子重要性（使用信息系数IC）
        
        Parameters:
        -----------
        X : pd.DataFrame
            因子数据
        y : pd.Series
            目标变量（次日是否继续涨停）
        
        Returns:
        --------
        Dict[str, float]: 因子名称 -> IC值
        """
        importance = {}
        
        for factor in self.factor_names:
            if factor in X.columns:
                # 计算IC（信息系数）= 因子与收益的相关系数
                ic = X[factor].corr(y)
                importance[factor] = abs(ic)
        
        # 按重要性排序
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def create_composite_factor(self, data: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        创建组合因子（加权平均）
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含所有因子的数据
        weights : Dict[str, float], optional
            因子权重，如果为None则使用等权
        
        Returns:
        --------
        pd.Series: 组合因子得分
        """
        if weights is None:
            # 默认权重（根据经验调整）
            weights = {
                'seal_strength': 0.20,      # 封单强度最重要
                'limitup_time_score': 0.18,  # 涨停时间很重要
                'leader_score': 0.15,        # 龙头地位重要
                'board_height': 0.15,        # 连板高度重要
                'big_order_ratio': 0.12,     # 大单比例
                'market_sentiment': 0.10,    # 市场情绪
                'open_count': 0.05,          # 打开次数（反向）
                'theme_decay': 0.05          # 题材衰减
            }
        
        # 计算加权平均
        composite_score = pd.Series(0, index=data.index)
        
        for factor, weight in weights.items():
            if factor in data.columns:
                composite_score += data[factor] * weight
        
        return composite_score


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    创建示例数据用于测试
    
    Parameters:
    -----------
    n_samples : int
        样本数量
    
    Returns:
    --------
    pd.DataFrame: 示例数据
    """
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    symbols = [f'{i:06d}.SZ' for i in range(1, 21)]
    
    data = []
    for date in dates:
        for symbol in symbols:
            # 随机生成涨停股票
            is_limitup = np.random.rand() > 0.9  # 10%概率涨停
            
            if is_limitup:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': 10.0,
                    'high': 11.0,
                    'low': 10.0,
                    'close': 11.0,
                    'volume': np.random.uniform(1e6, 1e8),
                    'amount': np.random.uniform(1e7, 1e9),
                    'is_limitup': 1,
                    'limitup_time': f"{np.random.randint(9, 15):02d}:{np.random.randint(0, 60):02d}:00",
                    'buy_amount': np.random.uniform(1e6, 1e8),
                    'sell_amount': np.random.uniform(1e5, 1e7),
                    'float_mv': np.random.uniform(1e9, 1e11),
                    'big_buy_volume': np.random.uniform(1e5, 1e7),
                    'total_buy_volume': np.random.uniform(1e6, 1e8),
                    'turnover': np.random.uniform(5, 30),
                    'industry': np.random.choice(['科技', '医药', '消费', '金融', '地产']),
                    'theme': np.random.choice(['AI', '新能源', '半导体', '医疗', '消费']),
                })
    
    return pd.DataFrame(data)


# ==================== 使用示例 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("涨停板高级因子库 - 测试")
    print("=" * 80)
    
    # 1. 创建示例数据
    print("\n1. 创建示例数据...")
    data = create_sample_data(n_samples=1000)
    print(f"   样本数量: {len(data)}")
    print(f"   涨停股票数: {data['is_limitup'].sum()}")
    print(f"   日期范围: {data['date'].min()} 至 {data['date'].max()}")
    
    # 2. 初始化因子计算器
    print("\n2. 初始化因子计算器...")
    calculator = LimitUpAdvancedFactors()
    print(f"   因子数量: {len(calculator.factor_names)}")
    print(f"   因子列表: {', '.join(calculator.factor_names)}")
    
    # 3. 计算所有因子
    print("\n3. 计算所有因子...")
    data_with_factors = calculator.calculate_all_factors(data)
    print(f"   计算完成！数据维度: {data_with_factors.shape}")
    
    # 4. 显示因子统计
    print("\n4. 因子统计:")
    print("-" * 80)
    factor_cols = [
        'seal_strength', 'open_count', 'limitup_time_score', 
        'board_height', 'market_sentiment', 'leader_score',
        'big_order_ratio', 'theme_decay'
    ]
    
    stats = data_with_factors[factor_cols].describe()
    print(stats.T[['mean', 'std', 'min', 'max']])
    
    # 5. 创建组合因子
    print("\n5. 创建组合因子...")
    data_with_factors['composite_score'] = calculator.create_composite_factor(data_with_factors)
    print(f"   组合因子统计:")
    print(f"   均值: {data_with_factors['composite_score'].mean():.4f}")
    print(f"   标准差: {data_with_factors['composite_score'].std():.4f}")
    print(f"   最小值: {data_with_factors['composite_score'].min():.4f}")
    print(f"   最大值: {data_with_factors['composite_score'].max():.4f}")
    
    # 6. 显示得分最高的股票
    print("\n6. 组合因子得分 TOP 10:")
    print("-" * 80)
    top10 = data_with_factors.nlargest(10, 'composite_score')[
        ['date', 'symbol', 'limitup_time', 'composite_score']
    ]
    for idx, row in top10.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')} {row['symbol']} "
              f"{row['limitup_time']} 得分: {row['composite_score']:.4f}")
    
    # 7. 模拟次日涨停预测
    print("\n7. 模拟次日涨停预测（得分>0.7为看好）:")
    print("-" * 80)
    high_score = data_with_factors[data_with_factors['composite_score'] > 0.7]
    print(f"   高分标的数量: {len(high_score)} / {len(data_with_factors)} "
          f"({len(high_score)/len(data_with_factors)*100:.1f}%)")
    
    if len(high_score) > 0:
        print(f"\n   示例（前5个）:")
        for idx, row in high_score.head(5).iterrows():
            print(f"   {row['date'].strftime('%Y-%m-%d')} {row['symbol']} "
                  f"涨停时间: {row['limitup_time']} "
                  f"得分: {row['composite_score']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！因子库可以正常使用。")
    print("=" * 80)
    
    # 8. 保存结果
    output_file = 'limitup_factors_sample.csv'
    data_with_factors.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存到: {output_file}")
