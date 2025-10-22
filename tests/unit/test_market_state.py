"""
市场状态检测单元测试
"""
import pytest
import pandas as pd
import numpy as np
from adaptive_system.market_state import (
    MarketRegime,
    MarketState,
    MarketStateDetector,
    AdaptiveStrategyAdjuster
)


class TestMarketRegime:
    """测试市场状态枚举"""
    
    def test_regime_types(self):
        """测试市场状态类型"""
        assert MarketRegime.BULL
        assert MarketRegime.BEAR
        assert MarketRegime.SIDEWAYS
        assert MarketRegime.VOLATILE
        assert MarketRegime.UNKNOWN


class TestMarketState:
    """测试市场状态数据类"""
    
    def test_state_creation(self):
        """测试状态创建"""
        state = MarketState(
            regime=MarketRegime.BULL,
            confidence=0.85,
            indicators={'rsi': 65, 'trend': 'up'}
        )
        assert state.regime == MarketRegime.BULL
        assert state.confidence == 0.85
        assert state.indicators['rsi'] == 65


class TestMarketStateDetector:
    """测试市场状态检测器"""
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = MarketStateDetector()
        assert detector.ma_short == 20
        assert detector.ma_long == 60
    
    def test_calculate_ma(self, sample_market_data):
        """测试移动平均线计算"""
        detector = MarketStateDetector()
        ma = detector._calculate_ma(sample_market_data['close'], window=20)
        
        assert len(ma) == len(sample_market_data)
        assert not ma.iloc[-1] != ma.iloc[-1]  # 最后一个值不是NaN
    
    def test_calculate_rsi(self, sample_market_data):
        """测试RSI计算"""
        detector = MarketStateDetector()
        rsi = detector._calculate_rsi(sample_market_data['close'])
        
        assert len(rsi) == len(sample_market_data)
        # RSI应在0-100之间
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_calculate_macd(self, sample_market_data):
        """测试MACD计算"""
        detector = MarketStateDetector()
        macd, signal = detector._calculate_macd(sample_market_data['close'])
        
        assert len(macd) == len(sample_market_data)
        assert len(signal) == len(sample_market_data)
    
    def test_detect_bull_market(self):
        """测试牛市检测"""
        detector = MarketStateDetector()
        
        # 创建牛市数据：上升趋势 + 高RSI
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 100 + np.arange(100) * 0.5  # 持续上涨
        volume = np.random.randint(1000000, 2000000, 100)
        
        data = pd.DataFrame({
            'date': dates,
            'close': close,
            'volume': volume
        })
        
        state = detector.detect_state(data)
        
        assert state.regime == MarketRegime.BULL
        assert state.confidence > 0.5
        assert 'ma_short' in state.indicators
        assert 'rsi' in state.indicators
    
    def test_detect_bear_market(self):
        """测试熊市检测"""
        detector = MarketStateDetector()
        
        # 创建熊市数据：下降趋势 + 低RSI
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 100 - np.arange(100) * 0.5  # 持续下跌
        volume = np.random.randint(1000000, 2000000, 100)
        
        data = pd.DataFrame({
            'date': dates,
            'close': close,
            'volume': volume
        })
        
        state = detector.detect_state(data)
        
        assert state.regime == MarketRegime.BEAR
        assert state.confidence > 0.5
    
    def test_detect_sideways_market(self):
        """测试震荡市检测"""
        detector = MarketStateDetector()
        
        # 创建震荡数据：小幅波动
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 100 + np.sin(np.arange(100) * 0.1) * 2  # 正弦波动
        volume = np.random.randint(1000000, 2000000, 100)
        
        data = pd.DataFrame({
            'date': dates,
            'close': close,
            'volume': volume
        })
        
        state = detector.detect_state(data)
        
        # 震荡市或未知
        assert state.regime in [MarketRegime.SIDEWAYS, MarketRegime.UNKNOWN, MarketRegime.VOLATILE]
    
    def test_detect_volatile_market(self):
        """测试高波动市场检测"""
        detector = MarketStateDetector()
        
        # 创建高波动数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 100 + np.random.randn(100) * 10  # 大幅随机波动
        volume = np.random.randint(1000000, 5000000, 100)
        
        data = pd.DataFrame({
            'date': dates,
            'close': close,
            'volume': volume
        })
        
        state = detector.detect_state(data)
        
        # 应检测到高波动或熊市（因为不稳定）
        assert state.regime in [MarketRegime.VOLATILE, MarketRegime.BEAR, MarketRegime.UNKNOWN]
    
    def test_detect_insufficient_data(self):
        """测试数据不足的情况"""
        detector = MarketStateDetector()
        
        # 只有少量数据
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(1000000, 2000000, 10)
        })
        
        state = detector.detect_state(data)
        
        # 数据不足应返回UNKNOWN
        assert state.regime == MarketRegime.UNKNOWN
        assert state.confidence < 0.5


class TestAdaptiveStrategyAdjuster:
    """测试自适应策略调整器"""
    
    def test_adjuster_initialization(self):
        """测试调整器初始化"""
        adjuster = AdaptiveStrategyAdjuster()
        assert adjuster.detector is not None
        assert adjuster.default_params is not None
    
    def test_adjust_for_bull_market(self):
        """测试牛市策略调整"""
        adjuster = AdaptiveStrategyAdjuster()
        
        # 创建牛市数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.arange(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        params = adjuster.adjust_strategy(data)
        
        # 牛市应增加仓位、放宽止损
        assert params['position_size'] >= adjuster.default_params['position_size']
        assert abs(params['stop_loss']) >= abs(adjuster.default_params['stop_loss'])
    
    def test_adjust_for_bear_market(self):
        """测试熊市策略调整"""
        adjuster = AdaptiveStrategyAdjuster()
        
        # 创建熊市数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 - np.arange(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        params = adjuster.adjust_strategy(data)
        
        # 熊市应减少仓位、收紧止损
        assert params['position_size'] <= adjuster.default_params['position_size']
        assert abs(params['stop_loss']) <= abs(adjuster.default_params['stop_loss'])
    
    def test_adjust_for_volatile_market(self):
        """测试高波动市场策略调整"""
        adjuster = AdaptiveStrategyAdjuster()
        
        # 创建高波动数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.random.randn(100) * 10,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        params = adjuster.adjust_strategy(data)
        
        # 高波动应大幅减少仓位
        assert params['position_size'] < adjuster.default_params['position_size']
    
    def test_strategy_params_validation(self):
        """测试策略参数验证"""
        adjuster = AdaptiveStrategyAdjuster()
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.arange(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        params = adjuster.adjust_strategy(data)
        
        # 验证参数范围
        assert 0 < params['position_size'] <= 1
        assert -1 < params['stop_loss'] < 0
        assert 0 < params['take_profit'] <= 1
        assert params['holding_period'] > 0
    
    def test_get_current_state(self):
        """测试获取当前市场状态"""
        adjuster = AdaptiveStrategyAdjuster()
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.arange(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        # 先调整策略
        adjuster.adjust_strategy(data)
        
        # 获取当前状态
        state = adjuster.get_current_state()
        
        assert state is not None
        assert isinstance(state, MarketState)
        assert state.regime in MarketRegime


@pytest.mark.integration
class TestMarketStateIntegration:
    """市场状态系统集成测试"""
    
    def test_full_adaptive_cycle(self):
        """测试完整自适应周期"""
        adjuster = AdaptiveStrategyAdjuster()
        
        # 模拟不同市场条件
        scenarios = [
            # 牛市
            pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100, freq='D'),
                'close': 100 + np.arange(100) * 0.5,
                'volume': np.random.randint(1000000, 2000000, 100)
            }),
            # 熊市
            pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100, freq='D'),
                'close': 100 - np.arange(100) * 0.5,
                'volume': np.random.randint(1000000, 2000000, 100)
            }),
            # 震荡
            pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100, freq='D'),
                'close': 100 + np.sin(np.arange(100) * 0.1) * 2,
                'volume': np.random.randint(1000000, 2000000, 100)
            })
        ]
        
        results = []
        for data in scenarios:
            params = adjuster.adjust_strategy(data)
            state = adjuster.get_current_state()
            results.append({
                'regime': state.regime,
                'confidence': state.confidence,
                'position_size': params['position_size']
            })
        
        # 验证每个场景都有结果
        assert len(results) == 3
        
        # 牛市和熊市的仓位应该有明显差异
        bull_pos = results[0]['position_size']
        bear_pos = results[1]['position_size']
        assert bull_pos > bear_pos
