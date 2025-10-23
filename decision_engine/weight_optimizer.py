"""
动态权重优化模块
基于历史表现自动调整三系统权重
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from .core import SignalSource, Signal

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """纯指标（与测试用例兼容的结构）。"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    win_rate: float
    avg_return: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemPerformance:
    """某一系统在一次评估下的表现快照。"""
    system_name: str
    metrics: PerformanceMetrics
    sample_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class WeightOptimizer:
    """权重优化器"""
    
    def __init__(self,
                 lookback_days: int = 20,
                 update_frequency: str = 'daily',
                 min_weight: float = 0.1,
                 max_weight: float = 0.6):
        """初始化权重优化器"""
        self.lookback_days = lookback_days
        self.update_frequency = update_frequency
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 历史性能记录（按系统聚合的快照）
        self.performance_history: List[SystemPerformance] = []

        # 系统清单/默认权重（与测试保持一致）
        self.systems: List[str] = ['qlib', 'trading_agents', 'rd_agent']
        self.default_weights: Dict[str, float] = {
            'qlib': 0.4,
            'trading_agents': 0.35,
            'rd_agent': 0.25,
        }

        # 当前权重（字符串键，便于指标上报/序列化）
        self.current_weights: Dict[str, float] = dict(self.default_weights)

        logger.info("✅ 权重优化器初始化完成")
    
    def evaluate_performance(self, *args, **kwargs) -> Union[Dict[SignalSource, PerformanceMetrics], SystemPerformance]:
        """评估性能（双接口兼容）。
        - evaluate_performance(signals: List[Signal], actual_returns: Dict[str,float])
        - evaluate_performance(system_name: str, predictions: np.ndarray, actuals: np.ndarray, returns: np.ndarray)
        """
        # 信号级评估
        if args and isinstance(args[0], list):
            signals: List[Signal] = args[0]
            actual_returns: Dict[str, float] = args[1] if len(args) > 1 else {}
            metrics_map: Dict[SignalSource, PerformanceMetrics] = {}
            for source in SignalSource:
                source_signals = [s for s in signals if s.source == source]
                if not source_signals:
                    continue
                # 以“方向是否判断正确”构造 predictions/actuals
                preds: List[int] = []
                acts: List[int] = []
                rets: List[float] = []
                for s in source_signals:
                    if s.symbol not in actual_returns:
                        continue
                    actual_r = float(actual_returns[s.symbol])
                    preds.append(1 if s.strength > 0 else 0)
                    acts.append(1 if actual_r > 0 else 0)
                    rets.append(actual_r if (preds[-1] == acts[-1]) else -abs(actual_r))
                if not preds:
                    continue
                pm = self._calculate_metrics(np.array(preds), np.array(acts), np.array(rets))
                metrics_map[source] = pm
                self.performance_history.append(SystemPerformance(system_name=source.value, metrics=pm, sample_size=len(preds)))
            logger.info(f"性能评估完成: {len(metrics_map)} 个系统")
            return metrics_map

        # 简单数组接口（兼容单测）
        system_name: str = kwargs.get('system_name') or (args[0] if args else 'unknown')
        predictions = kwargs.get('predictions') or (args[1] if len(args) > 1 else None)
        actuals = kwargs.get('actuals') or (args[2] if len(args) > 2 else None)
        returns = kwargs.get('returns') or (args[3] if len(args) > 3 else None)
        pm = self._calculate_metrics(np.asarray(predictions), np.asarray(actuals), np.asarray(returns) if returns is not None else None)
        sp = SystemPerformance(system_name=system_name, metrics=pm, sample_size=len(predictions) if predictions is not None else 0)
        self.performance_history.append(sp)
        return sp
    
    def _calculate_metrics(self,
                          predictions: np.ndarray,
                          actuals: np.ndarray,
                          returns: Optional[np.ndarray] = None) -> PerformanceMetrics:
        """计算性能指标（与单测兼容的签名）。"""
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)
        if returns is None:
            # 若未提供收益，按命中给 +1，错给 -1 的虚拟收益
            returns = np.where(predictions == actuals, 1.0, -1.0)
        returns = np.asarray(returns)

        if predictions.size == 0:
            return PerformanceMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1_score=0.5, sharpe_ratio=0.0, win_rate=0.5, avg_return=0.0)

        accuracy = float((predictions == actuals).mean())
        tp = int(((predictions == 1) & (actuals == 1)).sum())
        fp = int(((predictions == 1) & (actuals == 0)).sum())
        fn = int(((predictions == 0) & (actuals == 1)).sum())
        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        sharpe = float(self._calculate_sharpe_ratio(returns))
        win_rate = float((returns > 0).mean())
        avg_return = float(returns.mean())
        return PerformanceMetrics(accuracy, precision, recall, f1, sharpe, win_rate, avg_return)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算夏普比率（与单测兼容）。"""
        returns = np.asarray(returns)
        if returns.size == 0:
            return 0.0
        mu = float(returns.mean())
        sigma = float(returns.std())
        if sigma == 0:
            return 0.0
        return mu / sigma * np.sqrt(252)
    
    def optimize_weights(self, performance: Optional[Union[Dict[SignalSource, PerformanceMetrics], Dict[str, PerformanceMetrics]]] = None) -> Dict[str, float]:
        """优化权重；若不给 performance，就用最近一次/窗口聚合。返回 {'qlib': w1, ...}。
        评分: 0.3*Acc + 0.3*F1 + 0.2*Sharpe(截断/归一) + 0.2*WinRate
        """
        # 聚合待评分
        agg: Dict[str, PerformanceMetrics] = {}
        if performance:
            for k, v in performance.items():
                key = k.value if isinstance(k, SignalSource) else str(k)
                agg[key] = v
        else:
            # 使用历史窗口的最后一次各系统表现
            for sys in self.systems:
                for snap in reversed(self.performance_history):
                    if snap.system_name == sys:
                        agg[sys] = snap.metrics
                        break
            if not agg:
                return dict(self.current_weights)

        # 计算综合评分
        scores: Dict[str, float] = {}
        for sys, m in agg.items():
            s = 0.3 * m.accuracy + 0.3 * m.f1_score + 0.2 * (m.sharpe_ratio / 3.0 if m.sharpe_ratio > 0 else 0) + 0.2 * m.win_rate
            scores[sys] = max(0.1, min(1.0, float(s)))
        total = sum(scores.values()) or 1.0
        raw = {sys: sc / total for sys, sc in scores.items()}

        # 约束与归一
        constrained = self._apply_constraints_str(raw)
        self._log_weight_changes_str(constrained)
        self.current_weights = constrained
        return dict(constrained)
    
    def _apply_constraints_str(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重上下限并归一化（字符串键）。"""
        bounded = {k: max(self.min_weight, min(self.max_weight, float(v))) for k, v in weights.items()}
        total = sum(bounded.values()) or 1.0
        return {k: v / total for k, v in bounded.items()}
    
    def _log_weight_changes_str(self, new_weights: Dict[str, float]):
        """记录权重变化（字符串键）。"""
        logger.info("权重优化结果:")
        for sys in self.systems:
            old_w = float(self.current_weights.get(sys, 0.0))
            new_w = float(new_weights.get(sys, old_w))
            change = new_w - old_w
            logger.info(f"  {sys}: {old_w:.3f} → {new_w:.3f} ({change:+.3f})")
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """返回最近一次各系统的摘要（与单测的断言形态一致）。"""
        summary: Dict[str, Dict[str, float]] = {}
        for sys in self.systems:
            for snap in reversed(self.performance_history):
                if snap.system_name == sys:
                    m = snap.metrics
                    summary[sys] = {
                        'accuracy': float(m.accuracy),
                        'f1_score': float(m.f1_score),
                        'sharpe': float(m.sharpe_ratio),
                        'win_rate': float(m.win_rate),
                        'avg_return': float(m.avg_return),
                    }
                    break
        return summary
    
    def should_update(self, last_update: datetime) -> bool:
        """判断是否应该更新权重。"""
        now = datetime.now()
        delta = now - last_update
        if self.update_frequency == 'daily':
            return delta.days >= 1
        if self.update_frequency == 'weekly':
            return delta.days >= 7
        if self.update_frequency == 'monthly':
            return delta.days >= 30
        return False


# ============================================================================
# 自适应权重策略
# ============================================================================

class AdaptiveWeightStrategy:
    """自适应权重策略"""

    def __init__(self):
        self.optimizer = WeightOptimizer()
        self.last_update = datetime.now()

    async def update_if_needed(self,
                              signals: List[Signal],
                              actual_returns: Dict[str, float]) -> Optional[Dict[str, float]]:
        """如果需要则更新权重。"""
        if not self.optimizer.should_update(self.last_update):
            return None
        perf = self.optimizer.evaluate_performance(signals, actual_returns)
        new_weights = self.optimizer.optimize_weights(perf)
        self.last_update = datetime.now()
        return new_weights
    
    def get_current_weights(self) -> Dict[SignalSource, float]:
        """获取当前权重"""
        return self.optimizer.current_weights


# ============================================================================
# 测试
# ============================================================================

def test_weight_optimizer():
    """测试权重优化器"""
    print("=== 权重优化器测试 ===\n")
    
    optimizer = WeightOptimizer()
    
    # 模拟信号和收益
    from .core import SignalType
    
    signals = []
    actual_returns = {}
    
    for i, symbol in enumerate(['000001.SZ', '000002.SZ', '600000.SH']):
        for source in SignalSource:
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                source=source,
                confidence=0.8,
                strength=0.5 if i % 2 == 0 else -0.5,
                reasoning="test",
                timestamp=datetime.now(),
                metadata={}
            )
            signals.append(signal)
        
        # 模拟实际收益
        actual_returns[symbol] = np.random.uniform(-0.05, 0.05)
    
    # 评估性能
    print("1️⃣ 评估性能:")
    performance = optimizer.evaluate_performance(signals, actual_returns)
    
    for source, metrics in performance.items():
        print(f"\n{source.value}:")
        print(f"  准确率: {metrics.accuracy:.2%}")
        print(f"  F1分数: {metrics.f1_score:.2f}")
        print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
        print(f"  胜率: {metrics.win_rate:.2%}")
    
    # 优化权重
    print("\n2️⃣ 优化权重:")
    new_weights = optimizer.optimize_weights(performance)
    
    print("\n新权重:")
    for source, weight in new_weights.items():
        print(f"  {source.value}: {weight:.3f}")
    
    # 性能摘要
    print("\n3️⃣ 性能摘要:")
    summary = optimizer.get_performance_summary()
    if not summary.empty:
        print(summary.to_string())
    else:
        print("  暂无数据")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_weight_optimizer()
