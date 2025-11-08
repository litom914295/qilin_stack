#!/usr/bin/env python
"""
TA-Lib 技术指标包装器
提供150+技术指标和K线形态识别功能，支持缠论形态学分析
"""
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Union


class TALibIndicators:
    """TA-Lib技术指标计算器"""
    
    # ==================== 趋势指标 ====================
    
    @staticmethod
    def SMA(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """简单移动平均线"""
        return talib.SMA(close, timeperiod=timeperiod)
    
    @staticmethod
    def EMA(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """指数移动平均线"""
        return talib.EMA(close, timeperiod=timeperiod)
    
    @staticmethod
    def WMA(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """加权移动平均线"""
        return talib.WMA(close, timeperiod=timeperiod)
    
    @staticmethod
    def DEMA(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """双重指数移动平均线"""
        return talib.DEMA(close, timeperiod=timeperiod)
    
    @staticmethod
    def TEMA(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """三重指数移动平均线"""
        return talib.TEMA(close, timeperiod=timeperiod)
    
    @staticmethod
    def MACD(close: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, 
             signalperiod: int = 9) -> tuple:
        """
        MACD指标
        Returns: (macd, signal, hist)
        """
        return talib.MACD(close, fastperiod=fastperiod, 
                         slowperiod=slowperiod, signalperiod=signalperiod)
    
    @staticmethod
    def ADX(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            timeperiod: int = 14) -> np.ndarray:
        """平均趋向指数"""
        return talib.ADX(high, low, close, timeperiod=timeperiod)
    
    @staticmethod
    def AROON(high: np.ndarray, low: np.ndarray, timeperiod: int = 14) -> tuple:
        """
        阿隆指标
        Returns: (aroondown, aroonup)
        """
        return talib.AROON(high, low, timeperiod=timeperiod)
    
    @staticmethod
    def SAR(high: np.ndarray, low: np.ndarray, 
            acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
        """抛物线指标"""
        return talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
    
    # ==================== 动量指标 ====================
    
    @staticmethod
    def RSI(close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """相对强弱指标"""
        return talib.RSI(close, timeperiod=timeperiod)
    
    @staticmethod
    def STOCH(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              fastk_period: int = 5, slowk_period: int = 3, 
              slowd_period: int = 3) -> tuple:
        """
        随机指标
        Returns: (slowk, slowd)
        """
        return talib.STOCH(high, low, close, 
                          fastk_period=fastk_period,
                          slowk_period=slowk_period, 
                          slowd_period=slowd_period)
    
    @staticmethod
    def CCI(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            timeperiod: int = 14) -> np.ndarray:
        """顺势指标"""
        return talib.CCI(high, low, close, timeperiod=timeperiod)
    
    @staticmethod
    def MOM(close: np.ndarray, timeperiod: int = 10) -> np.ndarray:
        """动量指标"""
        return talib.MOM(close, timeperiod=timeperiod)
    
    @staticmethod
    def ROC(close: np.ndarray, timeperiod: int = 10) -> np.ndarray:
        """变动率指标"""
        return talib.ROC(close, timeperiod=timeperiod)
    
    @staticmethod
    def WILLR(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              timeperiod: int = 14) -> np.ndarray:
        """威廉指标"""
        return talib.WILLR(high, low, close, timeperiod=timeperiod)
    
    # ==================== 波动率指标 ====================
    
    @staticmethod
    def ATR(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            timeperiod: int = 14) -> np.ndarray:
        """平均真实波幅"""
        return talib.ATR(high, low, close, timeperiod=timeperiod)
    
    @staticmethod
    def NATR(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             timeperiod: int = 14) -> np.ndarray:
        """归一化平均真实波幅"""
        return talib.NATR(high, low, close, timeperiod=timeperiod)
    
    @staticmethod
    def BBANDS(close: np.ndarray, timeperiod: int = 20, 
               nbdevup: float = 2.0, nbdevdn: float = 2.0) -> tuple:
        """
        布林带
        Returns: (upper, middle, lower)
        """
        return talib.BBANDS(close, timeperiod=timeperiod,
                           nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
    
    @staticmethod
    def STDDEV(close: np.ndarray, timeperiod: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """标准差"""
        return talib.STDDEV(close, timeperiod=timeperiod, nbdev=nbdev)
    
    # ==================== 成交量指标 ====================
    
    @staticmethod
    def OBV(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """能量潮"""
        return talib.OBV(close, volume)
    
    @staticmethod
    def AD(high: np.ndarray, low: np.ndarray, close: np.ndarray,
           volume: np.ndarray) -> np.ndarray:
        """累积/派发线"""
        return talib.AD(high, low, close, volume)
    
    @staticmethod
    def ADOSC(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              volume: np.ndarray, fastperiod: int = 3, slowperiod: int = 10) -> np.ndarray:
        """累积/派发震荡指标"""
        return talib.ADOSC(high, low, close, volume, 
                          fastperiod=fastperiod, slowperiod=slowperiod)
    
    @staticmethod
    def MFI(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            volume: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """资金流量指标"""
        return talib.MFI(high, low, close, volume, timeperiod=timeperiod)


class TALibPatterns:
    """TA-Lib K线形态识别（支持缠论形态学）"""
    
    # ==================== 反转形态 ====================
    
    @staticmethod
    def CDLDOJI(open_: np.ndarray, high: np.ndarray, 
                low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """十字星"""
        return talib.CDLDOJI(open_, high, low, close)
    
    @staticmethod
    def CDLHAMMER(open_: np.ndarray, high: np.ndarray,
                  low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """锤子线"""
        return talib.CDLHAMMER(open_, high, low, close)
    
    @staticmethod
    def CDLINVERTEDHAMMER(open_: np.ndarray, high: np.ndarray,
                          low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """倒锤子线"""
        return talib.CDLINVERTEDHAMMER(open_, high, low, close)
    
    @staticmethod
    def CDLHANGINGMAN(open_: np.ndarray, high: np.ndarray,
                      low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """吊颈线"""
        return talib.CDLHANGINGMAN(open_, high, low, close)
    
    @staticmethod
    def CDLSHOOTINGSTAR(open_: np.ndarray, high: np.ndarray,
                        low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """射击之星"""
        return talib.CDLSHOOTINGSTAR(open_, high, low, close)
    
    @staticmethod
    def CDLENGULFING(open_: np.ndarray, high: np.ndarray,
                     low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """吞没形态"""
        return talib.CDLENGULFING(open_, high, low, close)
    
    @staticmethod
    def CDLHARAMI(open_: np.ndarray, high: np.ndarray,
                  low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """孕线形态"""
        return talib.CDLHARAMI(open_, high, low, close)
    
    @staticmethod
    def CDLMORNINGSTAR(open_: np.ndarray, high: np.ndarray,
                       low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """早晨之星"""
        return talib.CDLMORNINGSTAR(open_, high, low, close, penetration=0.3)
    
    @staticmethod
    def CDLEVENINGSTAR(open_: np.ndarray, high: np.ndarray,
                       low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """黄昏之星"""
        return talib.CDLEVENINGSTAR(open_, high, low, close, penetration=0.3)
    
    # ==================== 持续形态 ====================
    
    @staticmethod
    def CDL3WHITESOLDIERS(open_: np.ndarray, high: np.ndarray,
                          low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """三白兵"""
        return talib.CDL3WHITESOLDIERS(open_, high, low, close)
    
    @staticmethod
    def CDL3BLACKCROWS(open_: np.ndarray, high: np.ndarray,
                       low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """三只乌鸦"""
        return talib.CDL3BLACKCROWS(open_, high, low, close)
    
    @staticmethod
    def CDLRISEFALL3METHODS(open_: np.ndarray, high: np.ndarray,
                            low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """上升/下降三法"""
        return talib.CDLRISEFALL3METHODS(open_, high, low, close)
    
    # ==================== 缠论核心形态 ====================
    
    @staticmethod
    def detect_bi_pattern(open_: np.ndarray, high: np.ndarray,
                         low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        缠论笔的形态识别（基于K线形态组合）
        
        Returns:
            dict: {
                'top_reversal': 顶部反转信号,
                'bottom_reversal': 底部反转信号,
                'continuation': 延续信号
            }
        """
        # 顶部反转形态组合
        shooting_star = talib.CDLSHOOTINGSTAR(open_, high, low, close)
        evening_star = talib.CDLEVENINGSTAR(open_, high, low, close)
        hanging_man = talib.CDLHANGINGMAN(open_, high, low, close)
        
        # 底部反转形态组合
        hammer = talib.CDLHAMMER(open_, high, low, close)
        morning_star = talib.CDLMORNINGSTAR(open_, high, low, close)
        inverted_hammer = talib.CDLINVERTEDHAMMER(open_, high, low, close)
        
        # 延续形态
        three_white = talib.CDL3WHITESOLDIERS(open_, high, low, close)
        three_black = talib.CDL3BLACKCROWS(open_, high, low, close)
        
        return {
            'top_reversal': np.maximum.reduce([shooting_star, evening_star, hanging_man]),
            'bottom_reversal': np.maximum.reduce([hammer, morning_star, inverted_hammer]),
            'continuation_up': three_white,
            'continuation_down': three_black
        }
    
    @staticmethod
    def detect_duan_pattern(open_: np.ndarray, high: np.ndarray,
                           low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        缠论段的形态识别（更高级别的结构）
        
        Returns:
            dict: {
                'strong_reversal': 强反转信号,
                'weak_reversal': 弱反转信号,
                'consolidation': 盘整信号
            }
        """
        # 强反转形态
        engulfing = talib.CDLENGULFING(open_, high, low, close)
        
        # 弱反转形态
        harami = talib.CDLHARAMI(open_, high, low, close)
        doji = talib.CDLDOJI(open_, high, low, close)
        
        return {
            'strong_reversal': engulfing,
            'weak_reversal': np.maximum(harami, doji),
        }
    
    @staticmethod
    def get_all_patterns(open_: np.ndarray, high: np.ndarray,
                        low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """
        获取所有K线形态识别结果
        
        Returns:
            DataFrame: 包含所有形态的DataFrame
        """
        patterns = {
            # 反转形态
            'doji': talib.CDLDOJI(open_, high, low, close),
            'hammer': talib.CDLHAMMER(open_, high, low, close),
            'inverted_hammer': talib.CDLINVERTEDHAMMER(open_, high, low, close),
            'hanging_man': talib.CDLHANGINGMAN(open_, high, low, close),
            'shooting_star': talib.CDLSHOOTINGSTAR(open_, high, low, close),
            'engulfing': talib.CDLENGULFING(open_, high, low, close),
            'harami': talib.CDLHARAMI(open_, high, low, close),
            'morning_star': talib.CDLMORNINGSTAR(open_, high, low, close),
            'evening_star': talib.CDLEVENINGSTAR(open_, high, low, close),
            
            # 持续形态
            'three_white_soldiers': talib.CDL3WHITESOLDIERS(open_, high, low, close),
            'three_black_crows': talib.CDL3BLACKCROWS(open_, high, low, close),
        }
        
        return pd.DataFrame(patterns)


class TALibFeatureGenerator:
    """TA-Lib特征生成器（用于机器学习）"""
    
    def __init__(self, include_patterns: bool = True):
        """
        Args:
            include_patterns: 是否包含K线形态特征
        """
        self.include_patterns = include_patterns
        self.indicators = TALibIndicators()
        self.patterns = TALibPatterns()
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成完整的TA-Lib特征集
        
        Args:
            df: 包含 open, high, low, close, volume 的DataFrame
            
        Returns:
            DataFrame: 包含所有TA-Lib特征
        """
        features = pd.DataFrame(index=df.index)
        
        # 提取OHLCV数据
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # ==================== 趋势指标 ====================
        features['sma_5'] = self.indicators.SMA(close, 5)
        features['sma_10'] = self.indicators.SMA(close, 10)
        features['sma_20'] = self.indicators.SMA(close, 20)
        features['sma_60'] = self.indicators.SMA(close, 60)
        
        features['ema_5'] = self.indicators.EMA(close, 5)
        features['ema_10'] = self.indicators.EMA(close, 10)
        features['ema_20'] = self.indicators.EMA(close, 20)
        
        macd, signal, hist = self.indicators.MACD(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        features['adx_14'] = self.indicators.ADX(high, low, close, 14)
        
        # ==================== 动量指标 ====================
        features['rsi_6'] = self.indicators.RSI(close, 6)
        features['rsi_14'] = self.indicators.RSI(close, 14)
        features['rsi_24'] = self.indicators.RSI(close, 24)
        
        slowk, slowd = self.indicators.STOCH(high, low, close)
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        
        features['cci_14'] = self.indicators.CCI(high, low, close, 14)
        features['mom_10'] = self.indicators.MOM(close, 10)
        features['roc_10'] = self.indicators.ROC(close, 10)
        features['willr_14'] = self.indicators.WILLR(high, low, close, 14)
        
        # ==================== 波动率指标 ====================
        features['atr_14'] = self.indicators.ATR(high, low, close, 14)
        features['natr_14'] = self.indicators.NATR(high, low, close, 14)
        
        upper, middle, lower = self.indicators.BBANDS(close, 20)
        features['bbands_upper'] = upper
        features['bbands_middle'] = middle
        features['bbands_lower'] = lower
        features['bbands_width'] = (upper - lower) / middle
        
        # ==================== 成交量指标 ====================
        features['obv'] = self.indicators.OBV(close, volume)
        features['ad'] = self.indicators.AD(high, low, close, volume)
        features['mfi_14'] = self.indicators.MFI(high, low, close, volume, 14)
        
        # ==================== K线形态 ====================
        if self.include_patterns:
            pattern_df = self.patterns.get_all_patterns(open_, high, low, close)
            features = pd.concat([features, pattern_df], axis=1)
            
            # 缠论形态
            bi_patterns = self.patterns.detect_bi_pattern(open_, high, low, close)
            for name, values in bi_patterns.items():
                features[f'bi_{name}'] = values
            
            duan_patterns = self.patterns.detect_duan_pattern(open_, high, low, close)
            for name, values in duan_patterns.items():
                features[f'duan_{name}'] = values
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        # 趋势指标
        trend_features = [
            'sma_5', 'sma_10', 'sma_20', 'sma_60',
            'ema_5', 'ema_10', 'ema_20',
            'macd', 'macd_signal', 'macd_hist',
            'adx_14'
        ]
        
        # 动量指标
        momentum_features = [
            'rsi_6', 'rsi_14', 'rsi_24',
            'stoch_k', 'stoch_d',
            'cci_14', 'mom_10', 'roc_10', 'willr_14'
        ]
        
        # 波动率指标
        volatility_features = [
            'atr_14', 'natr_14',
            'bbands_upper', 'bbands_middle', 'bbands_lower', 'bbands_width'
        ]
        
        # 成交量指标
        volume_features = [
            'obv', 'ad', 'mfi_14'
        ]
        
        all_features = trend_features + momentum_features + volatility_features + volume_features
        
        # K线形态
        if self.include_patterns:
            pattern_features = [
                'doji', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
                'engulfing', 'harami', 'morning_star', 'evening_star',
                'three_white_soldiers', 'three_black_crows',
                'bi_top_reversal', 'bi_bottom_reversal', 'bi_continuation_up', 'bi_continuation_down',
                'duan_strong_reversal', 'duan_weak_reversal'
            ]
            all_features += pattern_features
        
        return all_features


# ==================== 工具函数 ====================

def calculate_indicator(df: pd.DataFrame, indicator_name: str, **kwargs) -> Union[np.ndarray, tuple]:
    """
    计算单个TA-Lib指标
    
    Args:
        df: 包含OHLCV数据的DataFrame
        indicator_name: 指标名称（如'RSI', 'MACD'等）
        **kwargs: 指标参数
        
    Returns:
        指标值（ndarray或tuple）
        
    Example:
        >>> rsi = calculate_indicator(df, 'RSI', timeperiod=14)
        >>> macd, signal, hist = calculate_indicator(df, 'MACD')
    """
    indicators = TALibIndicators()
    
    if not hasattr(indicators, indicator_name):
        raise ValueError(f"Unknown indicator: {indicator_name}")
    
    func = getattr(indicators, indicator_name)
    
    # 准备参数
    close = df['close'].values
    params = {'close': close}
    
    # 某些指标需要OHLCV
    if indicator_name in ['ATR', 'ADX', 'NATR', 'CCI', 'STOCH', 'WILLR']:
        params['high'] = df['high'].values
        params['low'] = df['low'].values
        if indicator_name in ['ADX', 'NATR']:
            params['close'] = close
    
    if indicator_name in ['OBV', 'MFI', 'AD', 'ADOSC']:
        params['volume'] = df['volume'].values
        if indicator_name in ['MFI', 'AD', 'ADOSC']:
            params['high'] = df['high'].values
            params['low'] = df['low'].values
            params['close'] = close
    
    # 合并用户参数
    params.update(kwargs)
    
    return func(**params)


def detect_pattern(df: pd.DataFrame, pattern_name: str) -> np.ndarray:
    """
    检测K线形态
    
    Args:
        df: 包含OHLC数据的DataFrame
        pattern_name: 形态名称（如'CDLDOJI', 'CDLHAMMER'等）
        
    Returns:
        形态信号数组（100=看涨, -100=看跌, 0=无信号）
        
    Example:
        >>> signal = detect_pattern(df, 'CDLDOJI')
    """
    patterns = TALibPatterns()
    
    if not hasattr(patterns, pattern_name):
        raise ValueError(f"Unknown pattern: {pattern_name}")
    
    func = getattr(patterns, pattern_name)
    
    return func(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )


if __name__ == '__main__':
    # 测试代码
    print("TA-Lib 技术指标包装器")
    print("=" * 60)
    print("✅ 趋势指标: SMA, EMA, MACD, ADX, AROON, SAR等")
    print("✅ 动量指标: RSI, STOCH, CCI, MOM, ROC, WILLR等")
    print("✅ 波动率指标: ATR, BBANDS, STDDEV等")
    print("✅ 成交量指标: OBV, AD, MFI等")
    print("✅ K线形态: 100+形态识别函数")
    print("✅ 缠论形态: 笔、段级别形态识别")
    print("=" * 60)
    
    # 测试特征生成器
    print("\n可用特征列表:")
    generator = TALibFeatureGenerator(include_patterns=True)
    features = generator.get_feature_names()
    print(f"总计 {len(features)} 个特征")
    for i, feat in enumerate(features, 1):
        print(f"{i:2d}. {feat}")
