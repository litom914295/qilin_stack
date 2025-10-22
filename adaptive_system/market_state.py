"""
自适应市场状态检测系统
识别牛市/熊市/震荡市，动态调整策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 市场状态定义
# ============================================================================

class MarketRegime(Enum):
    """市场状态"""
    BULL = "bull"  # 牛市
    BEAR = "bear"  # 熊市
    SIDEWAYS = "sideways"  # 震荡市
    VOLATILE = "volatile"  # 高波动
    UNKNOWN = "unknown"  # 未知


@dataclass
class MarketState:
    """市场状态"""
    regime: MarketRegime
    confidence: float  # 置信度
    trend_strength: float  # 趋势强度 -1到1
    volatility: float  # 波动率
    indicators: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'trend_strength': self.trend_strength,
            'volatility': self.volatility,
            'indicators': self.indicators,
            'timestamp': self.timestamp.isoformat()
        }


# ============================================================================
# 市场状态检测器
# ============================================================================

class MarketStateDetector:
    """市场状态检测器"""
    
    def __init__(self,
                 lookback_days: int = 60,
                 ma_short: int = 5,
                 ma_long: int = 20):
        """
        初始化检测器
        
        Args:
            lookback_days: 回溯天数
            ma_short: 短期均线
            ma_long: 长期均线
        """
        self.lookback_days = lookback_days
        self.ma_short = ma_short
        self.ma_long = ma_long
        
        # 历史状态
        self.state_history: List[MarketState] = []
        
        logger.info("✅ 市场状态检测器初始化完成")
    
    def detect_state(self, market_data: pd.DataFrame) -> MarketState:
        """
        检测市场状态
        
        Args:
            market_data: 市场数据，包含close, volume等
        
        Returns:
            当前市场状态
        """
        # 计算技术指标
        indicators = self._calculate_indicators(market_data)
        
        # 分析趋势
        trend_strength = self._analyze_trend(market_data, indicators)
        
        # 分析波动率
        volatility = self._calculate_volatility(market_data)
        
        # 确定市场状态
        regime, confidence = self._determine_regime(trend_strength, volatility, indicators)
        
        state = MarketState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility=volatility,
            indicators=indicators,
            timestamp=datetime.now()
        )
        
        # 保存到历史
        self.state_history.append(state)
        
        logger.info(f"检测到市场状态: {regime.value} (置信度:{confidence:.2%})")
        
        return state
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标"""
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones_like(close)
        
        # 短期均线
        ma_short = pd.Series(close).rolling(self.ma_short).mean().iloc[-1]
        
        # 长期均线
        ma_long = pd.Series(close).rolling(self.ma_long).mean().iloc[-1]
        
        # RSI
        rsi = self._calculate_rsi(pd.Series(close))
        
        # MACD
        macd, signal = self._calculate_macd(pd.Series(close))
        
        # 成交量比
        volume_ratio = volume[-1] / (volume[-20:].mean() + 1e-8)
        
        return {
            'ma_short': ma_short,
            'ma_long': ma_long,
            'ma_ratio': ma_short / (ma_long + 1e-8),
            'rsi': rsi,
            'macd': macd,
            'macd_signal': signal,
            'volume_ratio': volume_ratio,
            'price': close[-1],
            'price_change_20d': (close[-1] - close[-20]) / (close[-20] + 1e-8)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series,
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """计算MACD"""
        exp_fast = prices.ewm(span=fast, adjust=False).mean()
        exp_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1], macd_signal.iloc[-1]
    
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict[str, float]) -> float:
        """分析趋势强度"""
        close = data['close'].values
        
        # 基于均线
        ma_score = 0.0
        if indicators['ma_ratio'] > 1.05:
            ma_score = 0.5
        elif indicators['ma_ratio'] < 0.95:
            ma_score = -0.5
        
        # 基于MACD
        macd_score = 0.0
        if indicators['macd'] > indicators['macd_signal']:
            macd_score = 0.3
        elif indicators['macd'] < indicators['macd_signal']:
            macd_score = -0.3
        
        # 基于价格变化
        price_score = indicators['price_change_20d'] * 2  # 归一化
        price_score = max(-0.5, min(0.5, price_score))
        
        # 综合得分
        trend_strength = ma_score + macd_score + price_score
        trend_strength = max(-1.0, min(1.0, trend_strength))
        
        return trend_strength
    
    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """计算波动率"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(period).std().iloc[-1]
        return volatility * np.sqrt(252)  # 年化
    
    def _determine_regime(self,
                         trend_strength: float,
                         volatility: float,
                         indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """确定市场状态"""
        # 高波动检测
        if volatility > 0.4:
            return MarketRegime.VOLATILE, 0.8
        
        # 牛市/熊市判断
        if trend_strength > 0.5:
            confidence = min(0.9, 0.6 + trend_strength * 0.2)
            return MarketRegime.BULL, confidence
        elif trend_strength < -0.5:
            confidence = min(0.9, 0.6 + abs(trend_strength) * 0.2)
            return MarketRegime.BEAR, confidence
        
        # 震荡市
        if abs(trend_strength) < 0.3:
            return MarketRegime.SIDEWAYS, 0.7
        
        # 默认
        return MarketRegime.UNKNOWN, 0.5
    
    def get_state_history(self, lookback: int = 10) -> List[MarketState]:
        """获取历史状态"""
        return self.state_history[-lookback:]


# ============================================================================
# 自适应策略调整器
# ============================================================================

class AdaptiveStrategyAdjuster:
    """自适应策略调整器"""
    
    def __init__(self):
        self.detector = MarketStateDetector()
        self.current_parameters = self._default_parameters()
    
    def _default_parameters(self) -> Dict[str, Any]:
        """默认参数"""
        return {
            'position_size': 0.5,  # 仓位大小
            'stop_loss': -0.05,  # 止损
            'take_profit': 0.10,  # 止盈
            'holding_period': 5,  # 持仓周期（天）
            'rebalance_threshold': 0.1  # 调仓阈值
        }
    
    def adjust_strategy(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        根据市场状态调整策略参数
        
        Returns:
            调整后的策略参数
        """
        # 检测市场状态
        state = self.detector.detect_state(market_data)
        
        # 根据状态调整参数
        adjusted = self._adjust_by_regime(state)
        
        # 记录调整
        logger.info(f"策略参数已调整: {state.regime.value}")
        
        self.current_parameters = adjusted
        return adjusted
    
    def _adjust_by_regime(self, state: MarketState) -> Dict[str, Any]:
        """根据市场状态调整"""
        params = self._default_parameters().copy()
        
        if state.regime == MarketRegime.BULL:
            # 牛市：增加仓位，放宽止损
            params['position_size'] = 0.7
            params['stop_loss'] = -0.08
            params['take_profit'] = 0.15
            params['holding_period'] = 10
            
        elif state.regime == MarketRegime.BEAR:
            # 熊市：降低仓位，收紧止损
            params['position_size'] = 0.3
            params['stop_loss'] = -0.03
            params['take_profit'] = 0.05
            params['holding_period'] = 3
            
        elif state.regime == MarketRegime.SIDEWAYS:
            # 震荡市：中等仓位，频繁交易
            params['position_size'] = 0.4
            params['stop_loss'] = -0.04
            params['take_profit'] = 0.06
            params['holding_period'] = 5
            params['rebalance_threshold'] = 0.05
            
        elif state.regime == MarketRegime.VOLATILE:
            # 高波动：大幅降低仓位
            params['position_size'] = 0.2
            params['stop_loss'] = -0.02
            params['take_profit'] = 0.04
            params['holding_period'] = 2
        
        # 根据置信度调整
        params['position_size'] *= state.confidence
        
        return params
    
    def get_current_state(self) -> Optional[MarketState]:
        """获取当前市场状态"""
        if self.detector.state_history:
            return self.detector.state_history[-1]
        return None


# ============================================================================
# 测试
# ============================================================================

def test_adaptive_system():
    """测试自适应系统"""
    print("=== 自适应系统测试 ===\n")
    
    # 生成模拟市场数据
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    
    # 模拟牛市数据
    np.random.seed(42)
    prices = 100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod()
    prices = prices * (1 + np.linspace(0, 0.2, len(dates)))  # 上涨趋势
    
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # 测试市场状态检测
    print("1️⃣ 测试市场状态检测:")
    detector = MarketStateDetector()
    state = detector.detect_state(market_data)
    
    print(f"\n市场状态: {state.regime.value}")
    print(f"置信度: {state.confidence:.2%}")
    print(f"趋势强度: {state.trend_strength:.2f}")
    print(f"波动率: {state.volatility:.2%}")
    print(f"\n技术指标:")
    for key, value in state.indicators.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试策略调整
    print("\n2️⃣ 测试策略自适应调整:")
    adjuster = AdaptiveStrategyAdjuster()
    adjusted_params = adjuster.adjust_strategy(market_data)
    
    print(f"\n调整后的策略参数:")
    for key, value in adjusted_params.items():
        print(f"  {key}: {value}")
    
    # 测试不同市场状态
    print("\n3️⃣ 测试不同市场状态:")
    
    # 熊市数据
    bear_prices = 100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod()
    bear_prices = bear_prices * (1 - np.linspace(0, 0.15, len(dates)))
    bear_data = market_data.copy()
    bear_data['close'] = bear_prices
    
    bear_state = detector.detect_state(bear_data)
    print(f"\n熊市检测: {bear_state.regime.value} (置信度:{bear_state.confidence:.2%})")
    
    bear_params = adjuster.adjust_strategy(bear_data)
    print(f"  仓位: {bear_params['position_size']:.2f}")
    print(f"  止损: {bear_params['stop_loss']:.2%}")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_adaptive_system()
