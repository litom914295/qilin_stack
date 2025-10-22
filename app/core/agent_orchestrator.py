"""
Agent协同机制模块
实现Agent动态权重调整、冲突解决、协同决策等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """信号强度等级"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class AgentSignal:
    """Agent信号"""
    agent_id: str
    agent_type: str
    symbol: str
    signal: SignalStrength
    confidence: float  # 0-1之间
    reasoning: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'symbol': self.symbol,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ConsensusSignal:
    """共识信号"""
    symbol: str
    final_signal: SignalStrength
    consensus_score: float  # -1到1之间
    confidence: float
    supporting_agents: List[str]
    dissenting_agents: List[str]
    signal_distribution: Dict[SignalStrength, int]
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class AgentPerformanceTracker:
    """Agent性能跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.accuracy_scores: Dict[str, float] = {}
        self.reliability_scores: Dict[str, float] = {}
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def record_prediction(self, agent_id: str, predicted: SignalStrength, actual: SignalStrength, confidence: float):
        """记录预测结果"""
        # 计算预测准确度
        if predicted == actual:
            accuracy = 1.0
        elif abs(predicted.value - actual.value) == 1:
            accuracy = 0.5  # 方向正确但强度有偏差
        else:
            accuracy = 0.0
        
        # 考虑置信度的加权准确度
        weighted_accuracy = accuracy * confidence
        self.performance_history[agent_id].append(weighted_accuracy)
        
        # 更新准确度分数
        if agent_id in self.performance_history:
            self.accuracy_scores[agent_id] = np.mean(self.performance_history[agent_id])
    
    def record_response_time(self, agent_id: str, response_time: float):
        """记录响应时间"""
        self.response_times[agent_id].append(response_time)
    
    def get_agent_weight(self, agent_id: str) -> float:
        """获取Agent权重"""
        # 基于历史准确度计算权重
        if agent_id not in self.accuracy_scores:
            return 0.5  # 默认权重
        
        accuracy = self.accuracy_scores[agent_id]
        
        # 响应时间因子（响应越快权重越高）
        if agent_id in self.response_times and len(self.response_times[agent_id]) > 0:
            avg_response_time = np.mean(self.response_times[agent_id])
            time_factor = 1.0 / (1.0 + avg_response_time)  # 时间越短，因子越接近1
        else:
            time_factor = 1.0
        
        # 可靠性因子（准确度的稳定性）
        if agent_id in self.performance_history and len(self.performance_history[agent_id]) > 1:
            stability = 1.0 - np.std(self.performance_history[agent_id])
            reliability_factor = max(0.5, stability)
        else:
            reliability_factor = 0.5
        
        # 综合权重 = 准确度 * 时间因子 * 可靠性因子
        weight = accuracy * time_factor * reliability_factor
        
        return min(1.0, max(0.1, weight))  # 限制权重在0.1-1.0之间


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.resolution_history = []
        
    def resolve_conflicts(self, signals: List[AgentSignal], weights: Dict[str, float]) -> ConsensusSignal:
        """解决信号冲突"""
        if not signals:
            raise ValueError("没有信号可供处理")
        
        symbol = signals[0].symbol
        
        # 计算加权信号
        weighted_sum = 0
        total_weight = 0
        signal_counts = defaultdict(int)
        
        for signal in signals:
            agent_weight = weights.get(signal.agent_id, 0.5)
            signal_weight = agent_weight * signal.confidence
            weighted_sum += signal.signal.value * signal_weight
            total_weight += signal_weight
            signal_counts[signal.signal] += 1
        
        # 计算共识分数
        consensus_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # 确定最终信号
        if consensus_score >= 1.5:
            final_signal = SignalStrength.STRONG_BUY
        elif consensus_score >= 0.5:
            final_signal = SignalStrength.BUY
        elif consensus_score <= -1.5:
            final_signal = SignalStrength.STRONG_SELL
        elif consensus_score <= -0.5:
            final_signal = SignalStrength.SELL
        else:
            final_signal = SignalStrength.NEUTRAL
        
        # 识别支持和反对的agents
        supporting_agents = []
        dissenting_agents = []
        
        for signal in signals:
            if signal.signal.value * final_signal.value > 0:  # 同向
                supporting_agents.append(signal.agent_id)
            elif signal.signal.value * final_signal.value < 0:  # 反向
                dissenting_agents.append(signal.agent_id)
        
        # 计算整体置信度
        confidence_scores = [s.confidence for s in signals]
        avg_confidence = np.mean(confidence_scores)
        
        # 如果分歧较大，降低置信度
        signal_variance = np.var([s.signal.value for s in signals])
        confidence_penalty = 1.0 / (1.0 + signal_variance)
        final_confidence = avg_confidence * confidence_penalty
        
        consensus = ConsensusSignal(
            symbol=symbol,
            final_signal=final_signal,
            consensus_score=consensus_score,
            confidence=final_confidence,
            supporting_agents=supporting_agents,
            dissenting_agents=dissenting_agents,
            signal_distribution=dict(signal_counts),
            timestamp=datetime.now(),
            metadata={
                'signal_count': len(signals),
                'signal_variance': float(signal_variance),
                'weighted_sum': weighted_sum,
                'total_weight': total_weight
            }
        
        # 记录解决历史
        self.resolution_history.append({
            'timestamp': consensus.timestamp,
            'symbol': symbol,
            'input_signals': len(signals),
            'consensus_score': consensus_score,
            'final_signal': final_signal.value
        })
        
        return consensus


class AgentOrchestrator:
    """Agent协调器主类"""
    
    def __init__(self, 
                 max_workers: int = 10,
                 timeout: float = 5.0,
                 enable_async: bool = True):
        """
        初始化协调器
        
        Args:
            max_workers: 最大并发工作线程数
            timeout: Agent响应超时时间（秒）
            enable_async: 是否启用异步执行
        """
        self.agents: Dict[str, Any] = {}
        self.agent_types: Dict[str, str] = {}
        self.performance_tracker = AgentPerformanceTracker()
        self.conflict_resolver = ConflictResolver()
        
        self.max_workers = max_workers
        self.timeout = timeout
        self.enable_async = enable_async
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 动态权重配置
        self.base_weights: Dict[str, float] = {}
        self.dynamic_weights: Dict[str, float] = {}
        self.weight_update_frequency = 10  # 每10次决策更新一次权重
        self.decision_count = 0
        
        # 协同规则
        self.collaboration_rules = self._initialize_collaboration_rules()
        
        # 决策历史
        self.decision_history = deque(maxlen=1000)
        
        logger.info(f"Agent协调器初始化完成，最大工作线程: {max_workers}")
    
    def register_agent(self, agent_id: str, agent: Any, agent_type: str, base_weight: float = 0.5):
        """注册Agent"""
        self.agents[agent_id] = agent
        self.agent_types[agent_id] = agent_type
        self.base_weights[agent_id] = base_weight
        self.dynamic_weights[agent_id] = base_weight
        logger.info(f"注册Agent: {agent_id}, 类型: {agent_type}, 基础权重: {base_weight}")
    
    def _initialize_collaboration_rules(self) -> Dict:
        """初始化协同规则"""
        return {
            'require_minimum_agents': 3,  # 最少需要3个agent参与决策
            'confidence_threshold': 0.6,  # 置信度阈值
            'conflict_resolution_method': 'weighted_voting',  # 冲突解决方法
            'enable_veto': True,  # 是否允许否决权
            'veto_agents': ['risk_controller'],  # 拥有否决权的agent
            'consensus_threshold': 0.7,  # 共识阈值
        }
    
    def _call_agent(self, agent_id: str, context: Dict) -> Optional[AgentSignal]:
        """调用单个Agent"""
        try:
            start_time = datetime.now()
            agent = self.agents[agent_id]
            
            # 调用Agent的analyze方法
            result = agent.analyze(context)
            
            # 记录响应时间
            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_tracker.record_response_time(agent_id, response_time)
            
            # 构建信号
            signal = AgentSignal(
                agent_id=agent_id,
                agent_type=self.agent_types[agent_id],
                symbol=context.get('symbol', 'UNKNOWN'),
                signal=SignalStrength(result.get('signal', 0)),
                confidence=result.get('confidence', 0.5),
                reasoning=result.get('reasoning', ''),
                timestamp=datetime.now(),
                metadata=result.get('metadata', {})
            
            return signal
            
        except Exception as e:
            logger.error(f"Agent {agent_id} 执行失败: {str(e)}")
            return None
    
    def collect_signals(self, context: Dict, agent_ids: Optional[List[str]] = None) -> List[AgentSignal]:
        """收集所有Agent的信号"""
        if agent_ids is None:
            agent_ids = list(self.agents.keys())
        
        signals = []
        
        if self.enable_async:
            # 异步执行
            futures = {
                self.executor.submit(self._call_agent, agent_id, context): agent_id 
                for agent_id in agent_ids
            }
            
            for future in as_completed(futures, timeout=self.timeout):
                try:
                    signal = future.result()
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    agent_id = futures[future]
                    logger.error(f"Agent {agent_id} 超时或失败: {str(e)}")
        else:
            # 同步执行
            for agent_id in agent_ids:
                signal = self._call_agent(agent_id, context)
                if signal:
                    signals.append(signal)
        
        logger.info(f"收集到 {len(signals)}/{len(agent_ids)} 个Agent信号")
        return signals
    
    def make_collaborative_decision(self, context: Dict) -> ConsensusSignal:
        """做出协同决策"""
        # 收集信号
        signals = self.collect_signals(context)
        
        # 检查最小agent数量要求
        if len(signals) < self.collaboration_rules['require_minimum_agents']:
            raise ValueError(f"Agent数量不足，需要至少{self.collaboration_rules['require_minimum_agents']}个")
        
        # 获取动态权重
        weights = self._get_dynamic_weights()
        
        # 处理否决权
        if self.collaboration_rules['enable_veto']:
            veto_signal = self._check_veto(signals)
            if veto_signal:
                logger.warning(f"Agent {veto_signal.agent_id} 行使否决权")
                # 如果有否决，返回保守信号
                return ConsensusSignal(
                    symbol=context.get('symbol', 'UNKNOWN'),
                    final_signal=SignalStrength.NEUTRAL,
                    consensus_score=0,
                    confidence=0.1,
                    supporting_agents=[],
                    dissenting_agents=[s.agent_id for s in signals],
                    signal_distribution={SignalStrength.NEUTRAL: len(signals)},
                    timestamp=datetime.now(),
                    metadata={'veto': True, 'veto_agent': veto_signal.agent_id}
        
        # 解决冲突，生成共识
        consensus = self.conflict_resolver.resolve_conflicts(signals, weights)
        
        # 更新决策计数
        self.decision_count += 1
        
        # 定期更新权重
        if self.decision_count % self.weight_update_frequency == 0:
            self._update_dynamic_weights()
        
        # 记录决策历史
        self.decision_history.append({
            'timestamp': consensus.timestamp,
            'symbol': consensus.symbol,
            'consensus': consensus.to_dict() if hasattr(consensus, 'to_dict') else str(consensus),
            'signals': [s.to_dict() for s in signals]
        })
        
        return consensus
    
    def _check_veto(self, signals: List[AgentSignal]) -> Optional[AgentSignal]:
        """检查是否有Agent行使否决权"""
        veto_agents = self.collaboration_rules.get('veto_agents', [])
        
        for signal in signals:
            if signal.agent_id in veto_agents:
                # 如果拥有否决权的agent发出强烈卖出信号且置信度高
                if signal.signal == SignalStrength.STRONG_SELL and signal.confidence > 0.8:
                    return signal
        
        return None
    
    def _get_dynamic_weights(self) -> Dict[str, float]:
        """获取动态权重"""
        weights = {}
        
        for agent_id in self.agents.keys():
            # 基础权重
            base = self.base_weights.get(agent_id, 0.5)
            
            # 性能调整
            performance_weight = self.performance_tracker.get_agent_weight(agent_id)
            
            # 综合权重 = 基础权重 * 0.3 + 性能权重 * 0.7
            weights[agent_id] = base * 0.3 + performance_weight * 0.7
        
        return weights
    
    def _update_dynamic_weights(self):
        """更新动态权重"""
        logger.info("更新Agent动态权重")
        
        for agent_id in self.agents.keys():
            self.dynamic_weights[agent_id] = self._get_dynamic_weights().get(agent_id, 0.5)
        
        # 输出权重信息
        weight_info = {aid: f"{w:.3f}" for aid, w in self.dynamic_weights.items()}
        logger.info(f"当前权重分布: {weight_info}")
    
    def evaluate_agent_performance(self, agent_id: str, actual_results: Dict):
        """评估Agent性能"""
        # 从决策历史中找到agent的预测
        for decision in reversed(self.decision_history):
            for signal_dict in decision.get('signals', []):
                if signal_dict['agent_id'] == agent_id:
                    predicted = SignalStrength(signal_dict['signal'])
                    actual = SignalStrength(actual_results.get('actual_signal', 0))
                    confidence = signal_dict['confidence']
                    
                    # 记录性能
                    self.performance_tracker.record_prediction(
                        agent_id, predicted, actual, confidence
                    break
    
    def get_orchestration_report(self) -> Dict:
        """获取协调报告"""
        report = {
            'registered_agents': list(self.agents.keys()),
            'agent_weights': {
                aid: {
                    'base': self.base_weights.get(aid, 0),
                    'dynamic': self.dynamic_weights.get(aid, 0),
                    'performance': self.performance_tracker.get_agent_weight(aid)
                }
                for aid in self.agents.keys()
            },
            'performance_scores': self.performance_tracker.accuracy_scores,
            'total_decisions': self.decision_count,
            'recent_decisions': list(self.decision_history)[-10:] if self.decision_history else [],
            'collaboration_rules': self.collaboration_rules
        }
        
        return report
    
    def export_decision_history(self, filepath: str):
        """导出决策历史"""
        history_data = {
            'decisions': list(self.decision_history),
            'agent_performance': {
                aid: {
                    'accuracy': self.performance_tracker.accuracy_scores.get(aid, 0),
                    'avg_response_time': np.mean(self.performance_tracker.response_times[aid]) 
                                        if aid in self.performance_tracker.response_times else 0
                }
                for aid in self.agents.keys()
            },
            'conflict_resolutions': self.conflict_resolver.resolution_history[-100:],
            'metadata': {
                'total_decisions': self.decision_count,
                'export_time': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"决策历史已导出至: {filepath}")


class AgentEnsemble:
    """Agent集成学习"""
    
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.ensemble_strategies = {
            'voting': self._voting_ensemble,
            'stacking': self._stacking_ensemble,
            'boosting': self._boosting_ensemble,
            'blending': self._blending_ensemble
        }
    
    def _voting_ensemble(self, signals: List[AgentSignal]) -> SignalStrength:
        """投票集成"""
        votes = defaultdict(int)
        for signal in signals:
            votes[signal.signal] += 1
        
        # 返回得票最多的信号
        return max(votes, key=votes.get)
    
    def _stacking_ensemble(self, signals: List[AgentSignal]) -> SignalStrength:
        """堆叠集成"""
        # 使用元学习器组合基础预测
        features = []
        for signal in signals:
            features.append([
                signal.signal.value,
                signal.confidence,
                len(signal.reasoning)  # 推理长度作为特征
            ])
        
        # 这里应该有一个训练好的元模型
        # 暂时使用加权平均代替
        weighted_sum = sum(f[0] * f[1] for f in features)
        total_weight = sum(f[1] for f in features)
        
        avg_signal = weighted_sum / total_weight if total_weight > 0 else 0
        
        if avg_signal >= 1.5:
            return SignalStrength.STRONG_BUY
        elif avg_signal >= 0.5:
            return SignalStrength.BUY
        elif avg_signal <= -1.5:
            return SignalStrength.STRONG_SELL
        elif avg_signal <= -0.5:
            return SignalStrength.SELL
        else:
            return SignalStrength.NEUTRAL
    
    def _boosting_ensemble(self, signals: List[AgentSignal]) -> SignalStrength:
        """增强集成"""
        # 给表现好的Agent更高权重
        weights = {}
        for signal in signals:
            weight = self.orchestrator.performance_tracker.get_agent_weight(signal.agent_id)
            weights[signal.agent_id] = weight
        
        # 加权投票
        weighted_votes = defaultdict(float)
        for signal in signals:
            weight = weights.get(signal.agent_id, 0.5)
            weighted_votes[signal.signal] += weight * signal.confidence
        
        # 返回加权得分最高的信号
        return max(weighted_votes, key=weighted_votes.get)
    
    def _blending_ensemble(self, signals: List[AgentSignal]) -> SignalStrength:
        """混合集成"""
        # 结合多种集成策略
        voting_result = self._voting_ensemble(signals)
        stacking_result = self._stacking_ensemble(signals)
        boosting_result = self._boosting_ensemble(signals)
        
        # 综合三种方法的结果
        final_value = (voting_result.value + stacking_result.value + boosting_result.value) / 3
        
        if final_value >= 1.5:
            return SignalStrength.STRONG_BUY
        elif final_value >= 0.5:
            return SignalStrength.BUY
        elif final_value <= -1.5:
            return SignalStrength.STRONG_SELL
        elif final_value <= -0.5:
            return SignalStrength.SELL
        else:
            return SignalStrength.NEUTRAL
    
    def ensemble_predict(self, signals: List[AgentSignal], strategy: str = 'blending') -> SignalStrength:
        """使用集成策略预测"""
        if strategy not in self.ensemble_strategies:
            raise ValueError(f"未知的集成策略: {strategy}")
        
        return self.ensemble_strategies[strategy](signals)


# 示例Agent基类
class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    @abstractmethod
    def analyze(self, context: Dict) -> Dict:
        """分析并返回信号"""
        pass


# 示例Agent实现
class TrendFollowingAgent(BaseAgent):
    """趋势跟踪Agent示例"""
    
    def analyze(self, context: Dict) -> Dict:
        """分析趋势并生成信号"""
        # 模拟分析逻辑
        price_data = context.get('price_data', [])
        
        if len(price_data) < 20:
            return {
                'signal': 0,
                'confidence': 0.3,
                'reasoning': '数据不足',
                'metadata': {}
            }
        
        # 计算简单移动平均
        short_ma = np.mean(price_data[-10:])
        long_ma = np.mean(price_data[-20:])
        
        if short_ma > long_ma * 1.02:
            return {
                'signal': 1,  # BUY
                'confidence': 0.7,
                'reasoning': '短期均线上穿长期均线，趋势向上',
                'metadata': {'short_ma': short_ma, 'long_ma': long_ma}
            }
        elif short_ma < long_ma * 0.98:
            return {
                'signal': -1,  # SELL
                'confidence': 0.7,
                'reasoning': '短期均线下穿长期均线，趋势向下',
                'metadata': {'short_ma': short_ma, 'long_ma': long_ma}
            }
        else:
            return {
                'signal': 0,  # NEUTRAL
                'confidence': 0.5,
                'reasoning': '趋势不明确',
                'metadata': {'short_ma': short_ma, 'long_ma': long_ma}
            }


if __name__ == "__main__":
    # 示例用法
    orchestrator = AgentOrchestrator(max_workers=5, timeout=3.0)
    
    # 注册示例Agents
    for i in range(3):
        agent = TrendFollowingAgent(f"trend_agent_{i}")
        orchestrator.register_agent(f"trend_agent_{i}", agent, "trend_following", 0.5 + i * 0.1)
    
    # 模拟上下文
    context = {
        'symbol': 'TEST_STOCK',
        'price_data': [100 + i + np.random.randn() for i in range(30)]
    }
    
    # 做出协同决策
    # consensus = orchestrator.make_collaborative_decision(context)
    # print(f"共识信号: {consensus.final_signal}, 置信度: {consensus.confidence:.2f}")
    
    # 获取报告
    # report = orchestrator.get_orchestration_report()
    # print(json.dumps(report, indent=2, ensure_ascii=False))