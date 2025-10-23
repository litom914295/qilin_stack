"""
智能决策引擎核心模块
融合Qlib、TradingAgents、RD-Agent三个系统的信号
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Optional as _Optional

# 监控
try:
    from monitoring.metrics import get_monitor as _get_monitor
except Exception:  # noqa: BLE001
    _get_monitor = lambda: None  # type: ignore

# 配置
try:
    import yaml as _yaml  # type: ignore
except Exception:  # noqa: BLE001
    _yaml = None  # type: ignore

logger = logging.getLogger(__name__)

# 导入性能优化模块
try:
    from performance.concurrency import get_optimizer, ConcurrencyOptimizer
    from performance.cache import cached, get_cache
    PERFORMANCE_ENABLED = True
except ImportError:
    PERFORMANCE_ENABLED = False
    logger.warning("⚠️ 性能优化模块未找到，使用标准模式")

# Qlib 一次性初始化
_QLIB_READY = False

def _ensure_qlib() -> bool:
    """确保 Qlib 已初始化（幂等）。返回是否可用。"""
    global _QLIB_READY
    if _QLIB_READY:
        return True
    try:
        import qlib  # type: ignore
        from qlib.config import REG_CN  # type: ignore
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
        _QLIB_READY = True
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Qlib init失败: {e}")
        return False


# ============================================================================
# 信号类型定义
# ============================================================================

class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalSource(Enum):
    """信号来源"""
    QLIB = "qlib"
    TRADING_AGENTS = "trading_agents"
    RD_AGENT = "rd_agent"


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    source: SignalSource
    confidence: float  # 0-1
    strength: float  # 信号强度 -1到1
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'source': self.source.value,
            'confidence': self.confidence,
            'strength': self.strength,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class FusedSignal:
    """融合后的信号"""
    symbol: str
    final_signal: SignalType
    confidence: float
    strength: float
    component_signals: List[Signal]
    weights: Dict[SignalSource, float]
    reasoning: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            'symbol': self.symbol,
            'final_signal': self.final_signal.value,
            'confidence': self.confidence,
            'strength': self.strength,
            'component_count': len(self.component_signals),
            'weights': {k.value: v for k, v in self.weights.items()},
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


# ============================================================================
# 信号生成器抽象接口
# ============================================================================

class SignalGenerator(ABC):
    """信号生成器基类"""
    
    def __init__(self, source: SignalSource):
        self.source = source
    
    @abstractmethod
    async def generate_signals(self, symbols: List[str], date: str) -> List[Signal]:
        """生成信号"""
        pass
    
    def _create_signal(self,
                      symbol: str,
                      signal_type: SignalType,
                      confidence: float,
                      strength: float,
                      reasoning: str,
                      metadata: Optional[Dict] = None) -> Signal:
        """创建信号"""
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            source=self.source,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )


# ============================================================================
# Qlib信号生成器
# ============================================================================

class QlibSignalGenerator(SignalGenerator):
    """Qlib信号生成器"""
    
    def __init__(self):
        super().__init__(SignalSource.QLIB)
        self.model = None
        self._artifacts_dir = Path("layer2_qlib/artifacts")
        # 轻量缓存（symbol,date -> 预测dict）
        self._pred_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    
    async def generate_signals(self, symbols: List[str], date: str) -> List[Signal]:
        """
        从Qlib模型生成信号
        
        基于：
        - 预测收益率
        - 因子得分
        - 回测表现
        """
        signals = []
        monitor = _get_monitor() if callable(_get_monitor) else None
        
        for symbol in symbols:
            try:
                # 模拟Qlib预测
                prediction = await self._get_qlib_prediction(symbol, date)
                
                # 转换为信号
                signal = self._prediction_to_signal(symbol, prediction)
                signals.append(signal)
                if monitor:
                    monitor.record_signal(self.source.value, signal.signal_type.value, signal.confidence)
                
            except Exception as e:
                logger.error(f"Qlib信号生成失败 {symbol}: {e}")
        
        return signals
    
    async def _get_qlib_prediction(self, symbol: str, date: str) -> Dict[str, float]:
        """获取Qlib预测（带简单缓存与监控）。"""
        cache_key = (symbol, date)
        if cache_key in self._pred_cache:
            monitor = _get_monitor() if callable(_get_monitor) else None
            if monitor:
                monitor.collector.increment_counter("qlib_cache_hit_total")
            return self._pred_cache[cache_key]

        # 1) 离线预测产物（layer2_qlib/scripts/predict_online_qlib.py）
        try:
            day = date.replace("-", "")
            p = self._artifacts_dir / f"preds_{day}.csv"
            if p.exists():
                df = pd.read_csv(p)
                sym_variants = {symbol, symbol.replace(".SZ", "").replace(".SH", ""), str(symbol).upper()}
                match_row = None
                for _, r in df.iterrows():
                    sval = str(r.get("symbol", ""))
                    if sval in sym_variants or sval.upper() in sym_variants:
                        match_row = r
                        break
                if match_row is not None:
                    pred = {"return": float(match_row.get("score", 0.0)), "confidence": 0.8, "sharpe": 1.5}
                    self._pred_cache[cache_key] = pred
                    return pred
        except Exception as e:
            logger.debug(f"Qlib artifacts读取失败: {e}")

        # 2) Qlib简化估计（基于动量）
        try:
            if _ensure_qlib():
                from qlib.data import D  # type: ignore
                import time as _time
                t0 = _time.time()
                start = (pd.Timestamp(date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
                data = D.features([symbol], ['$close', '$volume'], start_time=start, end_time=date, freq='day')
                latency = _time.time() - t0
                monitor = _get_monitor() if callable(_get_monitor) else None
                if monitor:
                    monitor.collector.observe_histogram("qlib_pred_latency_seconds", latency, labels={"symbol": symbol})
                if not data.empty:
                    closes = data['$close'].droplevel(0) if isinstance(data.index, pd.MultiIndex) else data['$close']
                    ret5 = closes.pct_change(5).iloc[-1]
                    ret1 = closes.pct_change(1).iloc[-1]
                    predv = float(0.7 * ret5 + 0.3 * ret1)
                    conf = float(min(0.95, 0.6 + abs(predv) * 5))
                    pred = {"return": predv, "confidence": conf, "sharpe": 1.2}
                    self._pred_cache[cache_key] = pred
                    return pred
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Qlib简化估计失败: {e}")

        # 3) 兜底
        pred = {"return": 0.0, "confidence": 0.5, "sharpe": 1.0}
        self._pred_cache[cache_key] = pred
        return pred
    
    def _prediction_to_signal(self, symbol: str, prediction: Dict[str, float]) -> Signal:
        """预测转信号"""
        pred_return = prediction['return']
        confidence = prediction['confidence']
        
        # 信号类型判断
        if pred_return > 0.03:
            signal_type = SignalType.STRONG_BUY
        elif pred_return > 0.01:
            signal_type = SignalType.BUY
        elif pred_return < -0.03:
            signal_type = SignalType.STRONG_SELL
        elif pred_return < -0.01:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        reasoning = f"Qlib预测收益率: {pred_return:.2%}, Sharpe: {prediction['sharpe']:.2f}"
        
        return self._create_signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=pred_return * 10,  # 归一化到-1到1
            reasoning=reasoning,
            metadata=prediction
        )


# ============================================================================
# TradingAgents信号生成器
# ============================================================================

class TradingAgentsSignalGenerator(SignalGenerator):
    """TradingAgents信号生成器"""
    
    def __init__(self):
        super().__init__(SignalSource.TRADING_AGENTS)
        from tradingagents_integration.real_integration import create_integration
        from data_pipeline.unified_data import UnifiedDataPipeline
        self.integration = create_integration()
        self.pipeline = UnifiedDataPipeline()
    
    async def generate_signals(self, symbols: List[str], date: str) -> List[Signal]:
        """
        从TradingAgents多智能体生成信号
        
        基于：
        - AnalystAgent分析
        - RiskAgent评估
        - ExecutionAgent决策
        """
        signals = []
        
        for symbol in symbols:
            try:
                # 模拟多智能体决策
                decision = await self._get_agent_decision(symbol, date)
                
                # 转换为信号
                signal = self._decision_to_signal(symbol, decision)
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"TradingAgents信号生成失败 {symbol}: {e}")
        
        return signals
    
    async def _get_agent_decision(self, symbol: str, date: str) -> Dict[str, Any]:
        """通过TradingAgents生成真实决策"""
        try:
            from data_pipeline.unified_data import DataFrequency
            # 拉取近40日数据，构造技术指标
            start = (pd.Timestamp(date) - pd.Timedelta(days=40)).strftime('%Y-%m-%d')
            df = self.pipeline.get_bars([symbol], start, date, DataFrequency.DAY)
            price = float(df['close'].iloc[-1]) if not df.empty else float('nan')
            change_pct = float(df['close'].pct_change().iloc[-1]) if not df.empty else 0.0
            volume = float(df['volume'].iloc[-1]) if not df.empty else 0.0
            closes = df['close'].droplevel(0) if isinstance(df.index, pd.MultiIndex) else df['close']
            # 计算RSI与MACD
            rsi = 50.0
            macd = 0.0
            macd_signal = 0.0
            if not df.empty:
                delta = closes.diff()
                up = delta.clip(lower=0).rolling(14).mean()
                down = (-delta.clip(upper=0)).rolling(14).mean()
                rs = up / (down + 1e-9)
                rsi = float(100 - 100/(1+rs).iloc[-1])
                ema12 = closes.ewm(span=12, adjust=False).mean()
                ema26 = closes.ewm(span=26, adjust=False).mean()
                macd_series = ema12 - ema26
                macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()
                macd = float(macd_series.iloc[-1])
                macd_signal = float(macd_signal_series.iloc[-1])
            market_data = {
                'price': price,
                'change_pct': change_pct,
                'volume': volume,
                'technical_indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                }
            }
            # 注入一进二上下文（题材热度/龙虎榜净买入，如可获取）
            injected = False
            try:
                from rd_agent.limit_up_data import LimitUpDataInterface  # type: ignore
                data_if = LimitUpDataInterface(data_source='qlib')
                feats = data_if.get_limit_up_features([symbol], date)
                if feats is not None and not feats.empty:
                    row = feats.iloc[0]
                    ch = row.get('concept_heat'); lhb = row.get('lhb_netbuy')
                    if ch is not None:
                        market_data['concept_heat'] = float(ch)
                    if lhb is not None:
                        market_data['lhb_netbuy'] = float(lhb)
                    injected = True
            except Exception as _e:
                logger.debug(f"注入题材/龙虎榜失败: {_e}")
            # 适配层：如未能从 RD-Agent 得到，则用 adapters 拉取
            if not injected:
                try:
                    from layer3_online.adapters.selector import get_concept_heat as _gch, get_lhb_netbuy as _glhb
                    ch = _gch(symbol, date)
                    lhb = _glhb(symbol, date)
                    if ch: market_data['concept_heat'] = float(ch)
                    if lhb: market_data['lhb_netbuy'] = float(lhb)
                except Exception as _e2:
                    logger.debug(f"适配层注入失败: {_e2}")
        except Exception as e:
            logger.debug(f"TradingAgents 市场数据构造失败: {e}")
            market_data = {'technical_indicators': {}}
        # 调用多智能体系统
        try:
            ctx = {"date": date}
            result = await self.integration.analyze_stock(symbol, market_data, context=ctx)
            consensus = result.get('consensus', {})
            sig = str(consensus.get('signal', 'HOLD')).upper()
            action_map = {'BUY': 'buy', 'SELL': 'sell', 'HOLD': 'hold', 'STRONG_BUY': 'buy', 'STRONG_SELL': 'sell'}
            action = action_map.get(sig, 'hold')
            confidence = float(consensus.get('confidence', 0.6))
            reasoning = consensus.get('reasoning', 'TradingAgents综合分析')
            risk_score = max(0.0, 1.0 - confidence)
            return {'action': action, 'confidence': confidence, 'risk_score': risk_score, 'reasoning': reasoning}
        except Exception as e:
            logger.debug(f"TradingAgents 调用失败: {e}")
            return {'action': 'hold', 'confidence': 0.6, 'risk_score': 0.4, 'reasoning': 'fallback'}
    
    def _decision_to_signal(self, symbol: str, decision: Dict[str, Any]) -> Signal:
        """决策转信号"""
        action = decision['action']
        confidence = decision['confidence']
        risk_score = decision['risk_score']
        
        # 映射动作到信号类型
        signal_map = {
            'buy': SignalType.BUY,
            'sell': SignalType.SELL,
            'hold': SignalType.HOLD
        }
        signal_type = signal_map.get(action, SignalType.HOLD)
        
        # 强度受风险影响
        strength = (1.0 if action == 'buy' else -1.0 if action == 'sell' else 0.0)
        strength *= (1.0 - risk_score)  # 风险越高，强度越低
        
        reasoning = f"TradingAgents: {decision['reasoning']}, 风险: {risk_score:.2f}"
        
        return self._create_signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            reasoning=reasoning,
            metadata=decision
        )


# ============================================================================
# RD-Agent信号生成器
# ============================================================================

class RDAgentSignalGenerator(SignalGenerator):
    """RD-Agent信号生成器"""
    
    def __init__(self):
        super().__init__(SignalSource.RD_AGENT)
        from rd_agent.limit_up_data import LimitUpDataInterface, LimitUpFactorLibrary
        self.data_interface = LimitUpDataInterface(data_source="qlib")
        self.factor_lib = LimitUpFactorLibrary()
        self.factors = None
    
    async def generate_signals(self, symbols: List[str], date: str) -> List[Signal]:
        """
        从RD-Agent因子研究生成信号
        
        基于：
        - 新发现的因子
        - 因子得分
        - 历史表现
        """
        signals = []
        
        for symbol in symbols:
            try:
                # 模拟因子评分
                factor_scores = await self._get_factor_scores(symbol, date)
                
                # 转换为信号
                signal = self._factors_to_signal(symbol, factor_scores)
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"RD-Agent信号生成失败 {symbol}: {e}")
        
        return signals
    
    async def _get_factor_scores(self, symbol: str, date: str) -> Dict[str, float]:
        """获取因子得分（基于涨停板特征与因子库的组合评分）"""
        try:
            # “一进二”前置条件：前一交易日应为涨停（近似判断）
            feats = self.data_interface.get_limit_up_features([symbol], date)
            if not feats.empty:
                # 字段别名兼容（不同来源口径）
                row = feats.iloc[0]
                rowd = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.to_dict().items()}
                # seal_strength(0~1) → seal_quality(1~10)
                if 'seal_quality' not in rowd and 'seal_strength' in rowd:
                    try:
                        val = float(rowd['seal_strength'])
                        rowd['seal_quality'] = max(1.0, min(10.0, val * 10.0))
                    except Exception:
                        rowd['seal_quality'] = 5.0
                # board_height → continuous_board（粗略映射）
                if 'continuous_board' not in rowd and 'board_height' in rowd:
                    try:
                        rowd['continuous_board'] = float(rowd['board_height'])
                    except Exception:
                        rowd['continuous_board'] = 1.0

                def norm(v, lo, hi):
                    try:
                        return float(max(0.0, min(1.0, (float(v) - lo) / (hi - lo))))
                    except Exception:
                        return 0.5
                comp = (
                    0.40 * norm(rowd.get('limit_up_strength', rowd.get('seal_quality', 0)), 60, 100) +
                    0.20 * norm(rowd.get('seal_quality', 0), 1.0, 10.0) +
                    0.25 * norm(rowd.get('volume_surge', 1.0), 1.5, 8.0) +
                    0.10 * norm(rowd.get('concept_heat', 1), 1, 20) -
                    0.15 * norm(rowd.get('continuous_board', 1), 2, 6)  # 一进二偏好低连板数
                )
                comp = max(-1.0, min(1.0, (comp - 0.5) * 2.0))
                # 若上一日并非强势涨停（强度<95 或 seal_quality<6），降低评分
                lu_strength = float(rowd.get('limit_up_strength', 0.0) or 0.0)
                seal_q = float(rowd.get('seal_quality', 0.0) or 0.0)
                if lu_strength < 95.0 and seal_q < 6.0:
                    return {'momentum': 0.0, 'value': 0.0, 'quality': 0.0, 'composite_score': 0.0, 'confidence': 0.5}
                confidence = min(0.95, 0.65 + 0.30 * max(0.0, comp))
                return {
                    'momentum': 0.0,
                    'value': 0.0,
                    'quality': 0.0,
                    'composite_score': comp,
                    'confidence': confidence,
                }
        except Exception as e:  # noqa: BLE001
            logger.debug(f"RD-Agent 因子评分失败: {e}")
        return {
            'momentum': 0.0,
            'value': 0.0,
            'quality': 0.0,
            'composite_score': 0.0,
            'confidence': 0.5,
        }
    
    def _factors_to_signal(self, symbol: str, factor_scores: Dict[str, float]) -> Signal:
        """因子转信号"""
        composite_score = factor_scores['composite_score']
        confidence = factor_scores['confidence']
        
        # 信号类型判断
        if composite_score > 0.5:
            signal_type = SignalType.STRONG_BUY
        elif composite_score > 0.2:
            signal_type = SignalType.BUY
        elif composite_score < -0.5:
            signal_type = SignalType.STRONG_SELL
        elif composite_score < -0.2:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        reasoning = (
            f"RD-Agent因子: 动量={factor_scores['momentum']:.2f}, "
            f"价值={factor_scores['value']:.2f}, "
            f"质量={factor_scores['quality']:.2f}"
        )
        
        return self._create_signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=composite_score,
            reasoning=reasoning,
            metadata=factor_scores
        )


# ============================================================================
# 信号融合器
# ============================================================================

class SignalFuser:
    """信号融合器"""
    
    def __init__(self,
                 weight_qlib: float = 0.4,
                 weight_ta: float = 0.35,
                 weight_rd: float = 0.25):
        """
        初始化融合器
        
        Args:
            weight_qlib: Qlib权重
            weight_ta: TradingAgents权重
            weight_rd: RD-Agent权重
        """
        self.weights = {
            SignalSource.QLIB: weight_qlib,
            SignalSource.TRADING_AGENTS: weight_ta,
            SignalSource.RD_AGENT: weight_rd
        }
        
        # 权重归一化
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def fuse_signals(self, signals: List[Signal], symbol: str) -> FusedSignal:
        """
        融合多个信号
        
        策略：
        1. 加权平均强度
        2. 置信度加权
        3. 冲突检测和解决
        """
        # 过滤该股票的信号
        symbol_signals = [s for s in signals if s.symbol == symbol]
        
        if not symbol_signals:
            # 无信号，返回HOLD
            return self._create_hold_signal(symbol)
        
        # 加权融合
        weighted_strength = 0.0
        weighted_confidence = 0.0
        
        for signal in symbol_signals:
            weight = self.weights.get(signal.source, 0.0)
            weighted_strength += signal.strength * weight * signal.confidence
            weighted_confidence += signal.confidence * weight
        
        # 归一化
        if weighted_confidence > 0:
            weighted_strength /= weighted_confidence
        
        # 确定最终信号类型
        final_signal = self._strength_to_signal_type(weighted_strength)
        
        # 生成推理说明
        reasoning = self._generate_reasoning(symbol_signals, weighted_strength)
        
        return FusedSignal(
            symbol=symbol,
            final_signal=final_signal,
            confidence=weighted_confidence,
            strength=weighted_strength,
            component_signals=symbol_signals,
            weights=self.weights,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _strength_to_signal_type(self, strength: float) -> SignalType:
        """强度转信号类型"""
        if strength > 0.6:
            return SignalType.STRONG_BUY
        elif strength > 0.2:
            return SignalType.BUY
        elif strength < -0.6:
            return SignalType.STRONG_SELL
        elif strength < -0.2:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _generate_reasoning(self, signals: List[Signal], final_strength: float) -> str:
        """生成推理说明"""
        reasons = []
        
        for signal in signals:
            source_name = signal.source.value
            signal_name = signal.signal_type.value
            conf = signal.confidence
            reasons.append(f"{source_name}: {signal_name}(置信度:{conf:.2f})")
        
        reasoning = " | ".join(reasons)
        reasoning += f" → 融合强度: {final_strength:.2f}"
        
        return reasoning
    
    def _create_hold_signal(self, symbol: str) -> FusedSignal:
        """创建HOLD信号"""
        return FusedSignal(
            symbol=symbol,
            final_signal=SignalType.HOLD,
            confidence=0.5,
            strength=0.0,
            component_signals=[],
            weights=self.weights,
            reasoning="无可用信号",
            timestamp=datetime.now()
        )
    
    def update_weights(self, new_weights: Dict[SignalSource, float]):
        """动态更新权重"""
        total = sum(new_weights.values())
        self.weights = {k: v/total for k, v in new_weights.items()}
        logger.info(f"权重已更新: {self.weights}")


# ============================================================================
# 决策引擎
# ============================================================================

class DecisionEngine:
    """智能决策引擎（集成并发优化和缓存）"""
    
    def __init__(self, enable_performance: bool = True):
        # 信号生成器
        self.qlib_generator = QlibSignalGenerator()
        self.ta_generator = TradingAgentsSignalGenerator()
        self.rd_generator = RDAgentSignalGenerator()
        
        # 信号融合器
        self.fuser = SignalFuser()
        
        # 性能优化
        self.enable_performance = enable_performance and PERFORMANCE_ENABLED
        if self.enable_performance:
            self.optimizer = get_optimizer()
            self.cache = get_cache()
            logger.info("✅ 性能优化已启用（并发+缓存）")
        else:
            self.optimizer = None
            self.cache = None
            logger.info("⚠️ 性能优化未启用")
        
        # 历史记录
        self.signal_history: List[Signal] = []
        self.fused_history: List[FusedSignal] = []

        # Gates 配置（可选）
        self.gate_cfg = self._load_gate_config()
        
        logger.info("✅ 智能决策引擎初始化完成")
    
    async def make_decisions(self, symbols: List[str], date: str) -> List[FusedSignal]:
        """
        生成交易决策（支持并行优化）
        
        流程：
        1. 从三个系统并行获取信号
        2. 融合信号
        3. 风险过滤
        4. 返回最终决策
        """
        logger.info(f"开始生成决策: {len(symbols)}只股票")
        
        # 1. 收集所有信号（并行或串行）
        all_signals = []
        
        if self.enable_performance and self.optimizer:
            # 并行模式
            all_signals = await self._generate_signals_parallel(symbols, date)
        else:
            # 串行模式
            all_signals = await self._generate_signals_sequential(symbols, date)
        
        # 保存到历史
        self.signal_history.extend(all_signals)
        
        # 2. 融合信号
        fused_signals = []
        for symbol in symbols:
            fused = self.fuser.fuse_signals(all_signals, symbol)
            fused_signals.append(fused)
        
        # 3. 风险过滤 + Gates
        filtered_signals = self._apply_risk_filters(fused_signals, date)
        
        # 保存到历史
        self.fused_history.extend(filtered_signals)
        
        logger.info(f"✅ 生成 {len(filtered_signals)} 个决策")
        
        return filtered_signals
    
    async def _generate_signals_parallel(self, symbols: List[str], date: str) -> List[Signal]:
        """并行生成信号"""
        logger.info("🚀 并行模式")
        
        # 并行调用三个生成器
        tasks = [
            self.qlib_generator.generate_signals(symbols, date),
            self.ta_generator.generate_signals(symbols, date),
            self.rd_generator.generate_signals(symbols, date)
        ]
        
        results = await self.optimizer.gather_parallel(*tasks)
        
        # 收集所有信号
        all_signals = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"信号生成失败: {result}")
            else:
                all_signals.extend(result)
                source_names = ['Qlib', 'TradingAgents', 'RD-Agent']
                logger.info(f"{source_names[i]}生成 {len(result)} 个信号")
        
        return all_signals
    
    async def _generate_signals_sequential(self, symbols: List[str], date: str) -> List[Signal]:
        """串行生成信号"""
        logger.info("🐌 串行模式")
        
        all_signals = []
        
        # Qlib信号
        try:
            qlib_signals = await self.qlib_generator.generate_signals(symbols, date)
            all_signals.extend(qlib_signals)
            logger.info(f"Qlib生成 {len(qlib_signals)} 个信号")
        except Exception as e:
            logger.error(f"Qlib信号生成失败: {e}")
        
        # TradingAgents信号
        try:
            ta_signals = await self.ta_generator.generate_signals(symbols, date)
            all_signals.extend(ta_signals)
            logger.info(f"TradingAgents生成 {len(ta_signals)} 个信号")
        except Exception as e:
            logger.error(f"TradingAgents信号生成失败: {e}")
        
        # RD-Agent信号
        try:
            rd_signals = await self.rd_generator.generate_signals(symbols, date)
            all_signals.extend(rd_signals)
            logger.info(f"RD-Agent生成 {len(rd_signals)} 个信号")
        except Exception as e:
            logger.error(f"RD-Agent信号生成失败: {e}")
        
        return all_signals
    
    def _apply_risk_filters(self, signals: List[FusedSignal], date: str) -> List[FusedSignal]:
        """应用风险过滤与 Gates 规则。"""
        filtered: List[FusedSignal] = []
        monitor = _get_monitor() if callable(_get_monitor) else None

        # 预取候选的一进二特征（可选）
        feats_map: Dict[str, Dict[str, Any]] = {}
        try:
            from rd_agent.limit_up_data import LimitUpDataInterface  # type: ignore
            data_if = LimitUpDataInterface(data_source='qlib')
            syms = [s.symbol for s in signals]
            feats_df = data_if.get_limit_up_features(list(set(syms)), date)
            if feats_df is not None and not feats_df.empty:
                for idx, row in feats_df.iterrows():
                    feats_map[str(idx)] = {k: row[k] for k in row.index if k}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Gates 特征获取失败（跳过）：{e}")

        for fused in signals:
            # 置信度过滤
            if fused.confidence < 0.5:
                logger.debug(f"过滤低置信度信号: {fused.symbol} ({fused.confidence:.2f})")
                if monitor:
                    monitor.collector.increment_counter("gate_reject_total", labels={"reason": "low_confidence"})
                continue
            
            # 强度过弱 → HOLD
            if abs(fused.strength) < 0.1:
                fused.final_signal = SignalType.HOLD

            # Gates 应用：仅对买入方向进行硬门槛
            if self.gate_cfg and fused.final_signal in (SignalType.BUY, SignalType.STRONG_BUY):
                reasons = self._check_gates(fused.symbol, feats_map.get(fused.symbol, {}))
                if reasons:
                    # 命中任何拒绝原因：降为 HOLD 并记录
                    fused.reasoning = (fused.reasoning + f" | gates: reject={','.join(reasons)}").strip()
                    fused.final_signal = SignalType.HOLD
                    if monitor:
                        for r in reasons:
                            monitor.collector.increment_counter("gate_reject_total", labels={"reason": r})
            
            filtered.append(fused)
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_signals': len(self.signal_history),
            'total_decisions': len(self.fused_history),
            'current_weights': self.fuser.weights,
            'signal_distribution': self._get_signal_distribution()
        }
    
    def _get_signal_distribution(self) -> Dict[str, int]:
        """获取信号分布"""
        distribution: Dict[str, int] = {}
        for signal in self.fused_history[-100:]:  # 最近100个
            sig_type = signal.final_signal.value
            distribution[sig_type] = distribution.get(sig_type, 0) + 1
        return distribution

    # ---------------- Gates 支持 ----------------
    def _load_gate_config(self) -> Dict[str, Any]:
        """从 config/tradingagents.yaml 读取 gates 配置（可选）。"""
        cfg_path = Path("config/tradingagents.yaml")
        defaults = {
            "gates": {
                "first_touch_minutes_max": 30,
                "open_count_max": 2,
                "volume_surge_min": 2.0,
                "seal_quality_min": 6.5,
                "price_min": 3.0,
                "price_max": 40.0,
                "mcap_min_e8_cny": 200,
                "mcap_max_e8_cny": 8000,
                "turnover_min": 0.02,
                "turnover_max": 0.35,
                "concept_heat_min": 3,
            }
        }
        try:
            if _yaml and cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as f:
                    data = _yaml.safe_load(f) or {}
                    # 允许嵌套 tradingagents.gates 或顶层 gates
                    ta = data.get("tradingagents", {})
                    gates = (ta.get("gates") or data.get("gates") or {})
                    defaults["gates"].update({k: v for k, v in gates.items() if v is not None})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"加载 Gates 配置失败，使用默认：{e}")
        return defaults

    def _check_gates(self, symbol: str, feats: Dict[str, Any]) -> List[str]:
        """返回不满足 Gates 的原因列表。"""
        reasons: List[str] = []
        g = self.gate_cfg.get("gates", {}) if isinstance(self.gate_cfg, dict) else {}
        if not g:
            return reasons

        def val(name: str, alt: str = "") -> _Optional[float]:
            v = feats.get(name)
            if v is None and alt:
                v = feats.get(alt)
            try:
                return float(v)
            except Exception:
                return None

        # 逐项检查（缺项则跳过该规则）
        v = val("first_touch_minutes")
        if v is not None and v > g.get("first_touch_minutes_max", 1e9):
            reasons.append("first_touch")
        v = val("open_count", alt="intraday_open_count")
        if v is not None and v > g.get("open_count_max", 1e9):
            reasons.append("open_count")
        v = val("volume_surge")
        if v is not None and v < g.get("volume_surge_min", -1e9):
            reasons.append("volume_surge")
        v = val("seal_quality", alt="seal_strength")
        if v is not None:
            q = v if "seal_quality" in feats else (v * 10.0)
            if q < g.get("seal_quality_min", -1e9):
                reasons.append("seal_quality")
        price = val("price", alt="close")
        if price is not None:
            if price < g.get("price_min", -1e9) or price > g.get("price_max", 1e9):
                reasons.append("price")
        v = val("mcap_e8_cny", alt="market_cap")
        if v is not None:
            if v < g.get("mcap_min_e8_cny", -1e9) or v > g.get("mcap_max_e8_cny", 1e9):
                reasons.append("mcap")
        v = val("turnover")
        if v is not None:
            if v < g.get("turnover_min", -1e9) or v > g.get("turnover_max", 1e9):
                reasons.append("turnover")
        v = val("concept_heat")
        if v is not None and v < g.get("concept_heat_min", -1e9):
            reasons.append("concept_heat")

        return reasons


# ============================================================================
# 工厂函数
# ============================================================================

_engine_instance = None

def get_decision_engine() -> DecisionEngine:
    """获取决策引擎单例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DecisionEngine()
    return _engine_instance


# ============================================================================
# 测试
# ============================================================================

async def test_decision_engine():
    """测试决策引擎"""
    print("=== 智能决策引擎测试 ===\n")
    
    engine = get_decision_engine()
    
    # 测试决策生成
    symbols = ['000001.SZ', '600000.SH', '000002.SZ']
    date = '2024-06-30'
    
    print(f"1️⃣ 测试决策生成: {symbols}")
    decisions = await engine.make_decisions(symbols, date)
    
    print(f"\n生成 {len(decisions)} 个决策:\n")
    
    for decision in decisions:
        print(f"股票: {decision.symbol}")
        print(f"  信号: {decision.final_signal.value}")
        print(f"  置信度: {decision.confidence:.2%}")
        print(f"  强度: {decision.strength:.2f}")
        print(f"  推理: {decision.reasoning}")
        print(f"  组件数: {len(decision.component_signals)}")
        print()
    
    # 统计信息
    print("2️⃣ 统计信息:")
    stats = engine.get_statistics()
    print(f"  总信号数: {stats['total_signals']}")
    print(f"  总决策数: {stats['total_decisions']}")
    print(f"  当前权重: {stats['current_weights']}")
    print(f"  信号分布: {stats['signal_distribution']}")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_decision_engine())
