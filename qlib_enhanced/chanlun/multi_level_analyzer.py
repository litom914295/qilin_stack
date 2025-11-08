"""
多级别联合分析器 - Phase 5.1

功能:
- 多周期联合分析 (日线/60分/30分/15分)
- 级别共振检测
- 趋势一致性判断
- 多级别买卖点确认

双模式复用:
- Qlib系统: 多周期因子生成
- 独立系统: 实时多级别信号

作者: Warp AI Assistant
日期: 2025-01
版本: v1.7
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeLevel(Enum):
    """时间级别枚举"""
    DAY = "日线"
    M60 = "60分"
    M30 = "30分"
    M15 = "15分"
    M5 = "5分"


class TrendDirection(Enum):
    """趋势方向"""
    UP = "上涨"
    DOWN = "下跌"
    SIDEWAYS = "震荡"
    UNKNOWN = "未知"


@dataclass
class LevelAnalysis:
    """单个级别分析结果"""
    level: TimeLevel
    trend: TrendDirection
    bi_direction: str  # 当前笔方向
    in_zs: bool  # 是否在中枢内
    buy_point: Optional[str] = None  # 买点类型
    sell_point: Optional[str] = None  # 卖点类型
    strength: float = 0.0  # 趋势强度 [0-1]
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'trend': self.trend.value,
            'bi_direction': self.bi_direction,
            'in_zs': self.in_zs,
            'buy_point': self.buy_point,
            'sell_point': self.sell_point,
            'strength': self.strength
        }


@dataclass
class ResonanceSignal:
    """共振信号"""
    signal_type: str  # 'buy' | 'sell'
    resonance_level: int  # 共振级别数 (2-4)
    levels: List[TimeLevel]  # 共振的级别
    strength: float  # 信号强度 [0-1]
    reason: str  # 共振原因
    timestamp: pd.Timestamp = None
    
    def to_dict(self) -> Dict:
        return {
            'signal_type': self.signal_type,
            'resonance_level': self.resonance_level,
            'levels': [l.value for l in self.levels],
            'strength': self.strength,
            'reason': self.reason,
            'timestamp': str(self.timestamp) if self.timestamp else None
        }


class MultiLevelAnalyzer:
    """
    多级别联合分析器
    
    核心功能:
    1. 多周期数据分析
    2. 级别共振检测
    3. 趋势一致性判断
    4. 信号强度评分
    
    Examples:
        >>> analyzer = MultiLevelAnalyzer(
        ...     levels=[TimeLevel.DAY, TimeLevel.M60, TimeLevel.M30],
        ...     enable_resonance=True
        ... )
        >>> 
        >>> # 分析多级别数据
        >>> data = {
        ...     TimeLevel.DAY: day_df,
        ...     TimeLevel.M60: m60_df
        ... }
        >>> result = analyzer.analyze(data)
        >>> 
        >>> # 检查共振信号
        >>> if result.has_buy_resonance():
        ...     print(f"买入共振: {result.buy_signal.resonance_level}级")
    """
    
    def __init__(
        self,
        levels: List[TimeLevel] = None,
        enable_resonance: bool = True,
        min_resonance_level: int = 2,
        trend_consistency_threshold: float = 0.7
    ):
        """
        初始化多级别分析器
        
        Args:
            levels: 要分析的级别列表
            enable_resonance: 是否启用共振检测
            min_resonance_level: 最小共振级别数
            trend_consistency_threshold: 趋势一致性阈值
        """
        self.levels = levels or [TimeLevel.DAY, TimeLevel.M60, TimeLevel.M30]
        self.enable_resonance = enable_resonance
        self.min_resonance_level = min_resonance_level
        self.trend_consistency_threshold = trend_consistency_threshold
        
        logger.info(
            f"多级别分析器初始化: {len(self.levels)}个级别, "
            f"共振检测={'启用' if enable_resonance else '禁用'}"
        )
    
    def analyze(
        self,
        data: Dict[TimeLevel, pd.DataFrame],
        symbol: str = "unknown"
    ) -> 'MultiLevelResult':
        """
        执行多级别分析
        
        Args:
            data: 各级别的缠论特征数据 {TimeLevel: DataFrame}
            symbol: 股票代码
        
        Returns:
            MultiLevelResult: 多级别分析结果
        """
        # 1. 分析各个级别
        level_results = {}
        for level in self.levels:
            if level not in data or data[level] is None or len(data[level]) == 0:
                logger.warning(f"{level.value} 数据缺失,跳过")
                continue
            
            try:
                analysis = self._analyze_single_level(level, data[level])
                level_results[level] = analysis
            except Exception as e:
                logger.error(f"{level.value} 分析失败: {e}")
                continue
        
        if not level_results:
            logger.warning("所有级别分析失败")
            return MultiLevelResult(symbol=symbol, levels={})
        
        # 2. 检测共振
        buy_signal = None
        sell_signal = None
        
        if self.enable_resonance and len(level_results) >= self.min_resonance_level:
            buy_signal = self._detect_buy_resonance(level_results)
            sell_signal = self._detect_sell_resonance(level_results)
        
        # 3. 计算趋势一致性
        trend_consistency = self._calculate_trend_consistency(level_results)
        
        return MultiLevelResult(
            symbol=symbol,
            levels=level_results,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
            trend_consistency=trend_consistency
        )
    
    def _analyze_single_level(
        self,
        level: TimeLevel,
        df: pd.DataFrame
    ) -> LevelAnalysis:
        """分析单个级别"""
        if len(df) == 0:
            raise ValueError(f"{level.value} 数据为空")
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        # 1. 判断趋势
        trend = self._determine_trend(df)
        
        # 2. 当前笔方向
        bi_direction = latest.get('bi_direction', 'unknown')
        if bi_direction == 1:
            bi_direction = 'up'
        elif bi_direction == -1:
            bi_direction = 'down'
        else:
            bi_direction = 'unknown'
        
        # 3. 是否在中枢
        in_zs = bool(latest.get('in_zs', False))
        
        # 4. 买卖点
        buy_point = None
        sell_point = None
        
        if 'buy_point' in latest and latest['buy_point']:
            buy_point = str(latest['buy_point'])
        if 'sell_point' in latest and latest['sell_point']:
            sell_point = str(latest['sell_point'])
        
        # 5. 趋势强度
        strength = self._calculate_trend_strength(df)
        
        return LevelAnalysis(
            level=level,
            trend=trend,
            bi_direction=bi_direction,
            in_zs=in_zs,
            buy_point=buy_point,
            sell_point=sell_point,
            strength=strength
        )
    
    def _determine_trend(self, df: pd.DataFrame) -> TrendDirection:
        """判断趋势方向"""
        if len(df) < 20:
            return TrendDirection.UNKNOWN
        
        # 使用最近20根K线判断
        recent = df.tail(20)
        
        # 计算收盘价趋势
        close = recent['close'] if 'close' in recent.columns else recent.get('close_price', pd.Series([]))
        if len(close) == 0:
            return TrendDirection.UNKNOWN
        
        # 线性回归斜率
        x = np.arange(len(close))
        slope = np.polyfit(x, close.values, 1)[0]
        
        # 标准差 (震荡度)
        std_ratio = close.std() / close.mean() if close.mean() != 0 else 0
        
        # 判断
        if slope > close.mean() * 0.002 and std_ratio < 0.05:
            return TrendDirection.UP
        elif slope < -close.mean() * 0.002 and std_ratio < 0.05:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度 [0-1]"""
        if len(df) < 20:
            return 0.0
        
        recent = df.tail(20)
        close = recent['close'] if 'close' in recent.columns else recent.get('close_price', pd.Series([]))
        
        if len(close) == 0:
            return 0.0
        
        # 方法1: 线性回归R²
        x = np.arange(len(close))
        slope, intercept = np.polyfit(x, close.values, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((close.values - y_pred) ** 2)
        ss_tot = np.sum((close.values - close.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 方法2: 笔方向一致性
        bi_consistency = 0.0
        if 'bi_direction' in recent.columns:
            bi_dir = recent['bi_direction'].dropna()
            if len(bi_dir) > 0:
                bi_consistency = abs(bi_dir.mean())
        
        # 综合强度
        strength = (r2 * 0.6 + bi_consistency * 0.4)
        return max(0.0, min(1.0, strength))
    
    def _detect_buy_resonance(
        self,
        level_results: Dict[TimeLevel, LevelAnalysis]
    ) -> Optional[ResonanceSignal]:
        """检测买入共振"""
        buy_levels = []
        reasons = []
        
        for level, analysis in level_results.items():
            has_buy = False
            
            # 条件1: 出现买点
            if analysis.buy_point:
                has_buy = True
                reasons.append(f"{level.value}出现{analysis.buy_point}")
            
            # 条件2: 上涨趋势 + 笔向上
            elif (analysis.trend == TrendDirection.UP and 
                  analysis.bi_direction == 'up' and 
                  analysis.strength > 0.5):
                has_buy = True
                reasons.append(f"{level.value}上涨趋势确认")
            
            if has_buy:
                buy_levels.append(level)
        
        # 检查是否达到共振级别
        if len(buy_levels) >= self.min_resonance_level:
            strength = self._calculate_resonance_strength(
                [level_results[l] for l in buy_levels]
            )
            
            return ResonanceSignal(
                signal_type='buy',
                resonance_level=len(buy_levels),
                levels=buy_levels,
                strength=strength,
                reason='; '.join(reasons)
            )
        
        return None
    
    def _detect_sell_resonance(
        self,
        level_results: Dict[TimeLevel, LevelAnalysis]
    ) -> Optional[ResonanceSignal]:
        """检测卖出共振"""
        sell_levels = []
        reasons = []
        
        for level, analysis in level_results.items():
            has_sell = False
            
            # 条件1: 出现卖点
            if analysis.sell_point:
                has_sell = True
                reasons.append(f"{level.value}出现{analysis.sell_point}")
            
            # 条件2: 下跌趋势 + 笔向下
            elif (analysis.trend == TrendDirection.DOWN and 
                  analysis.bi_direction == 'down' and 
                  analysis.strength > 0.5):
                has_sell = True
                reasons.append(f"{level.value}下跌趋势确认")
            
            if has_sell:
                sell_levels.append(level)
        
        # 检查是否达到共振级别
        if len(sell_levels) >= self.min_resonance_level:
            strength = self._calculate_resonance_strength(
                [level_results[l] for l in sell_levels]
            )
            
            return ResonanceSignal(
                signal_type='sell',
                resonance_level=len(sell_levels),
                levels=sell_levels,
                strength=strength,
                reason='; '.join(reasons)
            )
        
        return None
    
    def _calculate_resonance_strength(
        self,
        analyses: List[LevelAnalysis]
    ) -> float:
        """计算共振强度"""
        if not analyses:
            return 0.0
        
        # 平均趋势强度
        avg_strength = np.mean([a.strength for a in analyses])
        
        # 级别数加成 (2级=1.0, 3级=1.1, 4级=1.2)
        level_bonus = 1.0 + (len(analyses) - 2) * 0.1
        
        # 综合强度
        strength = avg_strength * level_bonus
        return max(0.0, min(1.0, strength))
    
    def _calculate_trend_consistency(
        self,
        level_results: Dict[TimeLevel, LevelAnalysis]
    ) -> float:
        """计算趋势一致性 [0-1]"""
        if len(level_results) < 2:
            return 0.0
        
        # 统计各方向数量
        up_count = sum(1 for a in level_results.values() if a.trend == TrendDirection.UP)
        down_count = sum(1 for a in level_results.values() if a.trend == TrendDirection.DOWN)
        
        total = len(level_results)
        
        # 一致性 = 主导方向占比
        consistency = max(up_count, down_count) / total
        
        return consistency


@dataclass
class MultiLevelResult:
    """多级别分析结果"""
    symbol: str
    levels: Dict[TimeLevel, LevelAnalysis] = field(default_factory=dict)
    buy_signal: Optional[ResonanceSignal] = None
    sell_signal: Optional[ResonanceSignal] = None
    trend_consistency: float = 0.0
    
    def has_buy_resonance(self) -> bool:
        """是否有买入共振"""
        return self.buy_signal is not None
    
    def has_sell_resonance(self) -> bool:
        """是否有卖出共振"""
        return self.sell_signal is not None
    
    def get_main_trend(self) -> TrendDirection:
        """获取主趋势"""
        if not self.levels:
            return TrendDirection.UNKNOWN
        
        # 统计各方向
        trends = [a.trend for a in self.levels.values()]
        up = trends.count(TrendDirection.UP)
        down = trends.count(TrendDirection.DOWN)
        
        if up > down:
            return TrendDirection.UP
        elif down > up:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    def to_dict(self) -> Dict:
        """转为字典"""
        return {
            'symbol': self.symbol,
            'levels': {l.value: a.to_dict() for l, a in self.levels.items()},
            'buy_signal': self.buy_signal.to_dict() if self.buy_signal else None,
            'sell_signal': self.sell_signal.to_dict() if self.sell_signal else None,
            'trend_consistency': self.trend_consistency,
            'main_trend': self.get_main_trend().value
        }


# ========== 测试代码 ==========

if __name__ == '__main__':
    import random
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    def create_mock_data(n=100, trend='up'):
        """创建模拟缠论特征数据"""
        base_price = 100
        prices = []
        
        for i in range(n):
            if trend == 'up':
                price = base_price + i * 0.2 + random.uniform(-1, 1)
            elif trend == 'down':
                price = base_price - i * 0.2 + random.uniform(-1, 1)
            else:
                price = base_price + random.uniform(-2, 2)
            prices.append(price)
        
        df = pd.DataFrame({
            'close': prices,
            'bi_direction': [1 if trend == 'up' else -1 if trend == 'down' else 0] * n,
            'in_zs': [False] * n,
            'buy_point': [None] * (n-1) + ['I类买点'] if trend == 'up' else [None] * n,
            'sell_point': [None] * (n-1) + ['I类卖点'] if trend == 'down' else [None] * n,
        })
        return df
    
    print("\n=== 测试多级别分析器 ===\n")
    
    # 测试1: 多级别上涨共振
    print("--- 测试1: 多级别上涨共振 ---")
    analyzer = MultiLevelAnalyzer(
        levels=[TimeLevel.DAY, TimeLevel.M60, TimeLevel.M30],
        enable_resonance=True,
        min_resonance_level=2
    )
    
    data = {
        TimeLevel.DAY: create_mock_data(100, 'up'),
        TimeLevel.M60: create_mock_data(100, 'up'),
        TimeLevel.M30: create_mock_data(100, 'up')
    }
    
    result = analyzer.analyze(data, symbol='000001')
    
    print(f"股票: {result.symbol}")
    print(f"主趋势: {result.get_main_trend().value}")
    print(f"趋势一致性: {result.trend_consistency:.2%}")
    
    for level, analysis in result.levels.items():
        print(f"\n{level.value}:")
        print(f"  趋势: {analysis.trend.value}")
        print(f"  笔方向: {analysis.bi_direction}")
        print(f"  趋势强度: {analysis.strength:.2f}")
        if analysis.buy_point:
            print(f"  买点: {analysis.buy_point}")
    
    if result.has_buy_resonance():
        print(f"\n✅ 买入共振信号:")
        print(f"  共振级别: {result.buy_signal.resonance_level}")
        print(f"  共振周期: {[l.value for l in result.buy_signal.levels]}")
        print(f"  信号强度: {result.buy_signal.strength:.2f}")
        print(f"  原因: {result.buy_signal.reason}")
    
    # 测试2: 趋势分歧
    print("\n\n--- 测试2: 趋势分歧 ---")
    data2 = {
        TimeLevel.DAY: create_mock_data(100, 'up'),
        TimeLevel.M60: create_mock_data(100, 'down'),
        TimeLevel.M30: create_mock_data(100, 'sideways')
    }
    
    result2 = analyzer.analyze(data2, symbol='000002')
    print(f"股票: {result2.symbol}")
    print(f"主趋势: {result2.get_main_trend().value}")
    print(f"趋势一致性: {result2.trend_consistency:.2%}")
    print(f"买入共振: {result2.has_buy_resonance()}")
    print(f"卖出共振: {result2.has_sell_resonance()}")
    
    print("\n=== 测试完成 ===")
