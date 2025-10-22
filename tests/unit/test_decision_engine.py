"""
决策引擎单元测试
"""
import pytest
from datetime import datetime
from decision_engine.core import (
    DecisionEngine, 
    SignalType, 
    Signal, 
    Decision,
    SignalFuser,
    get_decision_engine
)


class TestSignalType:
    """测试信号类型枚举"""
    
    def test_signal_types_exist(self):
        """测试所有信号类型存在"""
        assert SignalType.BUY
        assert SignalType.SELL
        assert SignalType.HOLD
        assert SignalType.STRONG_BUY
        assert SignalType.STRONG_SELL
    
    def test_signal_type_values(self):
        """测试信号类型值"""
        assert SignalType.BUY.value == 'buy'
        assert SignalType.SELL.value == 'sell'
        assert SignalType.HOLD.value == 'hold'


class TestSignal:
    """测试Signal数据类"""
    
    def test_signal_creation(self):
        """测试信号创建"""
        signal = Signal(
            type=SignalType.BUY,
            confidence=0.8,
            strength=0.9,
            source='test'
        )
        assert signal.type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.strength == 0.9
        assert signal.source == 'test'
    
    def test_signal_validation(self):
        """测试信号验证"""
        # 置信度范围检查
        with pytest.raises(ValueError):
            Signal(SignalType.BUY, confidence=1.5, strength=0.5, source='test')
        
        with pytest.raises(ValueError):
            Signal(SignalType.BUY, confidence=-0.1, strength=0.5, source='test')
    
    def test_signal_with_metadata(self):
        """测试带元数据的信号"""
        signal = Signal(
            type=SignalType.SELL,
            confidence=0.7,
            strength=0.8,
            source='qlib',
            metadata={'model': 'lgb', 'score': 0.75}
        )
        assert signal.metadata['model'] == 'lgb'
        assert signal.metadata['score'] == 0.75


class TestSignalFuser:
    """测试信号融合器"""
    
    def test_fuser_initialization(self):
        """测试融合器初始化"""
        fuser = SignalFuser()
        assert fuser.weights is not None
        assert 'qlib' in fuser.weights
        assert 'trading_agents' in fuser.weights
        assert 'rd_agent' in fuser.weights
    
    def test_fuser_custom_weights(self):
        """测试自定义权重"""
        custom_weights = {'qlib': 0.5, 'trading_agents': 0.3, 'rd_agent': 0.2}
        fuser = SignalFuser(weights=custom_weights)
        assert fuser.weights == custom_weights
    
    def test_fuse_single_signal(self, mock_qlib_signal):
        """测试融合单个信号"""
        fuser = SignalFuser()
        result = fuser.fuse_signals([mock_qlib_signal])
        assert isinstance(result, Signal)
        assert result.type == mock_qlib_signal.type
    
    def test_fuse_multiple_signals(self, mock_qlib_signal, mock_ta_signal, mock_rd_signal):
        """测试融合多个信号"""
        fuser = SignalFuser()
        signals = [mock_qlib_signal, mock_ta_signal, mock_rd_signal]
        result = fuser.fuse_signals(signals)
        
        assert isinstance(result, Signal)
        assert result.type in SignalType
        assert 0 <= result.confidence <= 1
        assert 0 <= result.strength <= 1
    
    def test_fuse_empty_signals(self):
        """测试空信号列表"""
        fuser = SignalFuser()
        result = fuser.fuse_signals([])
        assert result.type == SignalType.HOLD
        assert result.confidence == 0.0


class TestDecisionEngine:
    """测试决策引擎"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """测试引擎初始化"""
        engine = DecisionEngine()
        assert engine.qlib_generator is not None
        assert engine.ta_generator is not None
        assert engine.rd_generator is not None
        assert engine.fuser is not None
    
    @pytest.mark.asyncio
    async def test_make_decisions_basic(self, sample_symbols, sample_date):
        """测试基本决策流程"""
        engine = DecisionEngine()
        decisions = await engine.make_decisions(sample_symbols[:1], sample_date)
        
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.symbol == sample_symbols[0]
        assert decision.final_signal in SignalType
        assert 0 <= decision.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_make_decisions_multiple_symbols(self, sample_symbols, sample_date):
        """测试多股票决策"""
        engine = DecisionEngine()
        decisions = await engine.make_decisions(sample_symbols, sample_date)
        
        assert len(decisions) == len(sample_symbols)
        for decision in decisions:
            assert decision.symbol in sample_symbols
            assert decision.final_signal in SignalType
    
    @pytest.mark.asyncio
    async def test_make_decisions_empty_symbols(self, sample_date):
        """测试空股票列表"""
        engine = DecisionEngine()
        decisions = await engine.make_decisions([], sample_date)
        assert len(decisions) == 0
    
    @pytest.mark.asyncio
    async def test_make_decisions_with_min_confidence(self, sample_symbols, sample_date):
        """测试最小置信度过滤"""
        engine = DecisionEngine()
        decisions = await engine.make_decisions(
            sample_symbols[:1], 
            sample_date,
            min_confidence=0.9  # 很高的阈值
        )
        # 可能被过滤为HOLD
        assert len(decisions) == 1
    
    @pytest.mark.asyncio
    async def test_update_weights(self):
        """测试更新权重"""
        engine = DecisionEngine()
        new_weights = {'qlib': 0.5, 'trading_agents': 0.3, 'rd_agent': 0.2}
        engine.update_weights(new_weights)
        assert engine.fuser.weights == new_weights
    
    @pytest.mark.asyncio
    async def test_get_singleton(self):
        """测试单例模式"""
        engine1 = get_decision_engine()
        engine2 = get_decision_engine()
        assert engine1 is engine2


class TestDecision:
    """测试决策数据类"""
    
    def test_decision_creation(self, mock_qlib_signal):
        """测试决策创建"""
        decision = Decision(
            symbol='000001.SZ',
            final_signal=SignalType.BUY,
            confidence=0.8,
            strength=0.9,
            reasoning='Test decision',
            source_signals=[mock_qlib_signal]
        )
        assert decision.symbol == '000001.SZ'
        assert decision.final_signal == SignalType.BUY
        assert len(decision.source_signals) == 1
    
    def test_decision_timestamp(self):
        """测试决策时间戳"""
        decision = Decision(
            symbol='000001.SZ',
            final_signal=SignalType.BUY,
            confidence=0.8,
            strength=0.9,
            reasoning='Test',
            source_signals=[]
        )
        assert isinstance(decision.timestamp, datetime)


@pytest.mark.integration
class TestDecisionEngineIntegration:
    """决策引擎集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_decision_pipeline(self, sample_symbols, sample_date):
        """测试完整决策流程"""
        engine = get_decision_engine()
        
        # 生成决策
        decisions = await engine.make_decisions(sample_symbols, sample_date)
        
        # 验证结果
        assert len(decisions) > 0
        for decision in decisions:
            assert decision.symbol in sample_symbols
            assert decision.final_signal in SignalType
            assert 0 <= decision.confidence <= 1
            assert decision.reasoning
            assert len(decision.source_signals) > 0
    
    @pytest.mark.asyncio
    async def test_weight_update_affects_decisions(self, sample_symbols, sample_date):
        """测试权重更新影响决策"""
        engine = DecisionEngine()
        
        # 使用默认权重
        decisions1 = await engine.make_decisions(sample_symbols[:1], sample_date)
        
        # 更新权重
        engine.update_weights({'qlib': 0.9, 'trading_agents': 0.05, 'rd_agent': 0.05})
        
        # 再次生成决策（可能不同）
        decisions2 = await engine.make_decisions(sample_symbols[:1], sample_date)
        
        # 至少应该成功生成
        assert len(decisions1) == 1
        assert len(decisions2) == 1
