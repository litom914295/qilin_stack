"""
高频数据分析模块 - 涨停板分时特征

使用1分钟/5分钟级别数据分析涨停板的盘中特征：
1. 涨停前量能爆发
2. 涨停后封单稳定性
3. 大单流入节奏
4. 尾盘封单强度（关键！）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class HighFreqLimitUpAnalyzer:
    """高频数据涨停板分析器"""
    
    def __init__(self, freq: str = '1min'):
        """
        初始化分析器
        
        Parameters:
        -----------
        freq : str
            频率 ('1min', '5min', '15min')
        """
        self.freq = freq
        self.freq_minutes = self._parse_freq(freq)
    
    def _parse_freq(self, freq: str) -> int:
        """解析频率字符串为分钟数"""
        if freq == '1min':
            return 1
        elif freq == '5min':
            return 5
        elif freq == '15min':
            return 15
        else:
            return 1
    
    def analyze_intraday_pattern(
        self,
        data: pd.DataFrame,
        limitup_time: str
    ) -> Dict[str, float]:
        """
        分析涨停当日的分时特征
        
        Parameters:
        -----------
        data : pd.DataFrame
            高频数据，必须包含：
            - time: 时间 (HH:MM:SS)
            - open, high, low, close: OHLC
            - volume: 成交量
            - amount: 成交额
            - buy_volume: 买入量
            - sell_volume: 卖出量
        limitup_time : str
            涨停时间 (HH:MM:SS)
        
        Returns:
        --------
        Dict: 高频特征
        """
        # 转换时间格式
        data = data.copy()
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.time
        
        limitup_time_obj = datetime.strptime(limitup_time, '%H:%M:%S').time()
        
        # 1. 涨停前30分钟量能爆发
        volume_burst_before = self._calc_volume_burst_before(
            data, limitup_time_obj
        )
        
        # 2. 涨停后封单稳定性
        seal_stability = self._calc_seal_stability_after(
            data, limitup_time_obj
        )
        
        # 3. 大单流入节奏
        big_order_rhythm = self._calc_big_order_rhythm(data)
        
        # 4. 尾盘封单强度（最关键！）
        close_seal_strength = self._calc_close_seal_strength(data)
        
        # 5. 涨停打开次数
        open_count = self._calc_open_count(data, limitup_time_obj)
        
        # 6. 涨停后成交量萎缩度
        volume_shrink = self._calc_volume_shrink_after(
            data, limitup_time_obj
        )
        
        return {
            'volume_burst_before_limit': volume_burst_before,
            'seal_stability': seal_stability,
            'big_order_rhythm': big_order_rhythm,
            'close_seal_strength': close_seal_strength,
            'intraday_open_count': open_count,
            'volume_shrink_after_limit': volume_shrink
        }
    
    def _calc_volume_burst_before(
        self,
        data: pd.DataFrame,
        limitup_time: datetime.time
    ) -> float:
        """
        计算涨停前30分钟的量能爆发指标
        
        逻辑：涨停前30分钟平均成交量 / 开盘后全天平均成交量
        """
        # 筛选涨停前30分钟的数据
        limitup_minutes = limitup_time.hour * 60 + limitup_time.minute
        before_30_start = limitup_minutes - 30
        
        before_30_data = data[
            (data['time'].apply(lambda t: t.hour * 60 + t.minute) >= before_30_start) &
            (data['time'].apply(lambda t: t.hour * 60 + t.minute) < limitup_minutes)
        ]
        
        if len(before_30_data) == 0:
            return 0.0
        
        # 计算涨停前30分钟的平均量
        before_30_avg_volume = before_30_data['volume'].mean()
        
        # 计算全天平均量
        all_day_avg_volume = data['volume'].mean()
        
        if all_day_avg_volume == 0:
            return 0.0
        
        # 量能爆发倍数
        volume_burst = before_30_avg_volume / all_day_avg_volume
        
        # 归一化到0-1（超过3倍视为满分）
        return min(1.0, volume_burst / 3.0)
    
    def _calc_seal_stability_after(
        self,
        data: pd.DataFrame,
        limitup_time: datetime.time
    ) -> float:
        """
        计算涨停后封单稳定性
        
        逻辑：涨停后价格波动的标准差（越小越稳定）
        """
        # 筛选涨停后的数据
        limitup_minutes = limitup_time.hour * 60 + limitup_time.minute
        
        after_data = data[
            data['time'].apply(lambda t: t.hour * 60 + t.minute) >= limitup_minutes
        ]
        
        if len(after_data) == 0:
            return 0.5
        
        # 计算涨停后价格波动
        if len(after_data) > 1:
            price_std = after_data['close'].std()
            price_mean = after_data['close'].mean()
            
            if price_mean > 0:
                cv = price_std / price_mean  # 变异系数
                # 变异系数越小，稳定性越高
                stability = 1.0 - min(1.0, cv * 100)
            else:
                stability = 0.5
        else:
            stability = 1.0  # 只有一个数据点，视为完全稳定
        
        return max(0.0, stability)
    
    def _calc_big_order_rhythm(self, data: pd.DataFrame) -> float:
        """
        计算大单流入节奏
        
        逻辑：大单持续流入的时间比例
        """
        if 'buy_volume' not in data.columns or 'sell_volume' not in data.columns:
            # 如果没有买卖盘数据，使用成交量替代
            return 0.5
        
        # 计算每个时间点的净买入
        data['net_buy'] = data['buy_volume'] - data['sell_volume']
        
        # 计算持续净买入的时间比例
        positive_count = (data['net_buy'] > 0).sum()
        total_count = len(data)
        
        if total_count == 0:
            return 0.0
        
        rhythm_score = positive_count / total_count
        
        return rhythm_score
    
    def _calc_close_seal_strength(self, data: pd.DataFrame) -> float:
        """
        计算尾盘封单强度（最关键！）
        
        逻辑：14:00-15:00的平均成交量 vs 全天平均
        """
        # 筛选尾盘数据（14:00-15:00）
        close_data = data[
            (data['time'].apply(lambda t: t.hour * 60 + t.minute) >= 14 * 60) &
            (data['time'].apply(lambda t: t.hour * 60 + t.minute) < 15 * 60)
        ]
        
        if len(close_data) == 0:
            return 0.0
        
        # 尾盘平均量
        close_avg_volume = close_data['volume'].mean()
        
        # 全天平均量
        all_day_avg_volume = data['volume'].mean()
        
        if all_day_avg_volume == 0:
            return 0.0
        
        # 尾盘量能比
        close_strength = close_avg_volume / all_day_avg_volume
        
        # 归一化：尾盘量小于全天平均表示封得稳
        # 量越小，封单越强
        if close_strength < 0.5:
            strength_score = 1.0  # 尾盘量很小，封得很好
        elif close_strength < 1.0:
            strength_score = 0.7  # 尾盘量适中
        else:
            strength_score = 0.3  # 尾盘量大，可能不稳
        
        return strength_score
    
    def _calc_open_count(
        self,
        data: pd.DataFrame,
        limitup_time: datetime.time
    ) -> int:
        """
        计算涨停打开次数
        
        逻辑：涨停后价格低于涨停价的次数
        """
        # 筛选涨停后的数据
        limitup_minutes = limitup_time.hour * 60 + limitup_time.minute
        
        after_data = data[
            data['time'].apply(lambda t: t.hour * 60 + t.minute) >= limitup_minutes
        ]
        
        if len(after_data) == 0:
            return 0
        
        # 假设涨停价是涨停后的最高价
        limitup_price = after_data['high'].max()
        
        # 计算打开次数（close < limitup_price * 0.99）
        open_count = (after_data['close'] < limitup_price * 0.99).sum()
        
        return open_count
    
    def _calc_volume_shrink_after(
        self,
        data: pd.DataFrame,
        limitup_time: datetime.time
    ) -> float:
        """
        计算涨停后成交量萎缩度
        
        逻辑：涨停后平均量 / 涨停前平均量（越小越好）
        """
        limitup_minutes = limitup_time.hour * 60 + limitup_time.minute
        
        # 涨停前数据
        before_data = data[
            data['time'].apply(lambda t: t.hour * 60 + t.minute) < limitup_minutes
        ]
        
        # 涨停后数据
        after_data = data[
            data['time'].apply(lambda t: t.hour * 60 + t.minute) >= limitup_minutes
        ]
        
        if len(before_data) == 0 or len(after_data) == 0:
            return 0.5
        
        before_avg = before_data['volume'].mean()
        after_avg = after_data['volume'].mean()
        
        if before_avg == 0:
            return 0.5
        
        shrink_ratio = after_avg / before_avg
        
        # 萎缩度：ratio越小，萎缩越明显，封单越强
        # 转换为得分：萎缩明显=高分
        shrink_score = 1.0 - min(1.0, shrink_ratio)
        
        return shrink_score
    
    def batch_analyze(
        self,
        stocks_data: Dict[str, Tuple[pd.DataFrame, str]]
    ) -> pd.DataFrame:
        """
        批量分析多只股票
        
        Parameters:
        -----------
        stocks_data : Dict[str, Tuple[pd.DataFrame, str]]
            {股票代码: (高频数据, 涨停时间)}
        
        Returns:
        --------
        pd.DataFrame: 所有股票的高频特征
        """
        results = []
        
        for symbol, (data, limitup_time) in stocks_data.items():
            try:
                features = self.analyze_intraday_pattern(data, limitup_time)
                features['symbol'] = symbol
                results.append(features)
            except Exception as e:
                print(f"⚠️  分析 {symbol} 失败: {e}")
        
        return pd.DataFrame(results)


def create_sample_high_freq_data(symbol: str = '000001.SZ') -> pd.DataFrame:
    """
    创建模拟高频数据用于测试
    
    Parameters:
    -----------
    symbol : str
        股票代码
    
    Returns:
    --------
    pd.DataFrame: 模拟的1分钟数据
    """
    np.random.seed(42)
    
    # 生成交易时间（9:30-15:00）
    times = []
    
    # 上午: 9:30-11:30
    for h in range(9, 12):
        for m in range(60):
            if h == 9 and m < 30:
                continue
            if h == 11 and m >= 30:
                break
            times.append(f"{h:02d}:{m:02d}:00")
    
    # 下午: 13:00-15:00
    for h in range(13, 15):
        for m in range(60):
            times.append(f"{h:02d}:{m:02d}:00")
    
    n = len(times)
    
    # 模拟价格（涨停过程）
    base_price = 10.0
    limitup_price = base_price * 1.10
    
    # 涨停时间设定为10:30
    limitup_index = times.index("10:30:00")
    
    prices = []
    for i in range(n):
        if i < limitup_index:
            # 涨停前：逐步上涨
            progress = i / limitup_index
            price = base_price + (limitup_price - base_price) * progress
        else:
            # 涨停后：在涨停价附近波动
            price = limitup_price * (1 + np.random.uniform(-0.001, 0.001))
        
        prices.append(price)
    
    # 模拟成交量
    volumes = []
    for i in range(n):
        if i < limitup_index - 30:
            # 涨停前30分钟之前：正常量
            vol = np.random.uniform(1000, 5000)
        elif i < limitup_index:
            # 涨停前30分钟：量能爆发
            vol = np.random.uniform(10000, 30000)
        else:
            # 涨停后：量萎缩
            vol = np.random.uniform(500, 2000)
        
        volumes.append(vol)
    
    data = pd.DataFrame({
        'time': times,
        'open': [p * (1 + np.random.uniform(-0.002, 0.002)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'close': prices,
        'volume': volumes,
        'amount': [v * p for v, p in zip(volumes, prices)],
        'buy_volume': [v * np.random.uniform(0.5, 0.7) for v in volumes],
        'sell_volume': [v * np.random.uniform(0.3, 0.5) for v in volumes]
    })
    
    return data


# ==================== 使用示例 ====================

def main():
    """示例：分析涨停板高频数据"""
    print("=" * 80)
    print("高频数据涨停板分析 - 测试")
    print("=" * 80)
    
    # 1. 创建模拟数据
    print("\n📊 生成模拟高频数据...")
    data = create_sample_high_freq_data('000001.SZ')
    print(f"   数据点数: {len(data)}")
    print(f"   时间范围: {data['time'].iloc[0]} 至 {data['time'].iloc[-1]}")
    
    # 2. 初始化分析器
    print("\n🔬 初始化高频分析器...")
    analyzer = HighFreqLimitUpAnalyzer(freq='1min')
    
    # 3. 分析涨停板特征
    print("\n📈 分析涨停板分时特征...")
    features = analyzer.analyze_intraday_pattern(
        data=data,
        limitup_time='10:30:00'
    )
    
    # 4. 显示结果
    print("\n" + "=" * 80)
    print("📊 分析结果")
    print("=" * 80)
    
    print("\n高频特征:")
    for key, value in features.items():
        desc = {
            'volume_burst_before_limit': '涨停前量能爆发',
            'seal_stability': '涨停后封单稳定性',
            'big_order_rhythm': '大单流入节奏',
            'close_seal_strength': '尾盘封单强度',
            'intraday_open_count': '涨停打开次数',
            'volume_shrink_after_limit': '涨停后量萎缩度'
        }
        
        if isinstance(value, float):
            print(f"  {desc.get(key, key)}: {value:.4f}")
        else:
            print(f"  {desc.get(key, key)}: {value}")
    
    # 5. 综合评分
    print("\n" + "=" * 80)
    print("🎯 综合评分")
    print("=" * 80)
    
    # 计算综合得分
    weights = {
        'volume_burst_before_limit': 0.15,
        'seal_stability': 0.25,
        'big_order_rhythm': 0.15,
        'close_seal_strength': 0.30,  # 最重要
        'volume_shrink_after_limit': 0.15
    }
    
    score = 0.0
    for key, weight in weights.items():
        if key in features and isinstance(features[key], (int, float)):
            score += features[key] * weight
    
    print(f"\n综合得分: {score:.2%}")
    
    if score >= 0.80:
        print("✅ 评级: 强势涨停，次日继续涨停概率高")
    elif score >= 0.60:
        print("⚠️  评级: 一般涨停，次日走势不确定")
    else:
        print("❌ 评级: 弱势涨停，次日继续涨停概率低")
    
    # 6. 批量分析示例
    print("\n" + "=" * 80)
    print("📊 批量分析示例")
    print("=" * 80)
    
    stocks_data = {
        '000001.SZ': (create_sample_high_freq_data('000001.SZ'), '10:30:00'),
        '000002.SZ': (create_sample_high_freq_data('000002.SZ'), '11:00:00'),
        '600000.SH': (create_sample_high_freq_data('600000.SH'), '09:45:00')
    }
    
    batch_results = analyzer.batch_analyze(stocks_data)
    
    print("\n批量分析结果:")
    print(batch_results.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
