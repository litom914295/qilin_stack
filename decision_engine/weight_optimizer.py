"""
动态权重优化模块
基于历史表现自动调整三系统权重
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .core import SignalSource, Signal, FusedSignal

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    source: SignalSource
    accuracy: float  # 准确率
    precision: float  # 精确率
    recall: float  # 召回率
    f1_score: float  # F1分数
    sharpe_ratio: float  # 夏普比率
    win_rate: float  # 胜率
    avg_return: float  # 平均收益
    timestamp: datetime


class WeightOptimizer:
    """权重优化器"""
    
    def __init__(self,
                 lookback_days: int = 20,
                 update_frequency: str = 'daily',
                 min_weight: float = 0.1,
                 max_weight: float = 0.6):
        """
        初始化权重优化器
        
        Args:
            lookback_days: 回溯天数
            update_frequency: 更新频率（daily/weekly/monthly）
            min_weight: 最小权重
            max_weight: 最大权重
        """
        self.lookback_days = lookback_days
        self.update_frequency = update_frequency
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 历史性能记录
        self.performance_history: List[PerformanceMetrics] = []
        
        # 当前权重
        self.current_weights = {
            SignalSource.QLIB: 0.4,
            SignalSource.TRADING_AGENTS: 0.35,
            SignalSource.RD_AGENT: 0.25
        }
        
        logger.info("✅ 权重优化器初始化完成")
    
    def evaluate_performance(self,
                            signals: List[Signal],
                            actual_returns: Dict[str, float]) -> Dict[SignalSource, PerformanceMetrics]:
        """
        评估各系统性能
        
        Args:
            signals: 历史信号
            actual_returns: 实际收益 {symbol: return}
        
        Returns:
            各系统性能指标
        """
        metrics = {}
        
        for source in SignalSource:
            source_signals = [s for s in signals if s.source == source]
            
            if not source_signals:
                continue
            
            # 计算性能指标
            perf = self._calculate_metrics(source_signals, actual_returns)
            metrics[source] = perf
            
            # 保存到历史
            self.performance_history.append(perf)
        
        logger.info(f"性能评估完成: {len(metrics)} 个系统")
        return metrics
    
    def _calculate_metrics(self,
                          signals: List[Signal],
                          actual_returns: Dict[str, float]) -> PerformanceMetrics:
        """计算性能指标"""
        source = signals[0].source
        
        # 预测和实际
        predictions = []
        actuals = []
        returns = []
        
        for signal in signals:
            symbol = signal.symbol
            if symbol not in actual_returns:
                continue
            
            actual_return = actual_returns[symbol]
            
            # 预测方向（1=up, 0=down）
            pred = 1 if signal.strength > 0 else 0
            actual = 1 if actual_return > 0 else 0
            
            predictions.append(pred)
            actuals.append(actual)
            returns.append(actual_return if pred == actual else -abs(actual_return))
        
        if not predictions:
            return self._empty_metrics(source)
        
        # 计算指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        returns = np.array(returns)
        
        accuracy = (predictions == actuals).mean()
        
        # 精确率和召回率
        true_positive = ((predictions == 1) & (actuals == 1)).sum()
        false_positive = ((predictions == 1) & (actuals == 0)).sum()
        false_negative = ((predictions == 0) & (actuals == 1)).sum()
        
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # 夏普比率
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 平均收益
        avg_return = returns.mean()
        
        return PerformanceMetrics(
            source=source,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            avg_return=avg_return,
            timestamp=datetime.now()
        )
    
    def _empty_metrics(self, source: SignalSource) -> PerformanceMetrics:
        """空指标"""
        return PerformanceMetrics(
            source=source,
            accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            sharpe_ratio=0.0,
            win_rate=0.5,
            avg_return=0.0,
            timestamp=datetime.now()
        )
    
    def optimize_weights(self, performance: Dict[SignalSource, PerformanceMetrics]) -> Dict[SignalSource, float]:
        """
        优化权重
        
        策略：
        1. 基于综合评分排序
        2. 表现好的增加权重
        3. 表现差的降低权重
        4. 保证权重在合理范围
        """
        # 计算综合评分
        scores = {}
        for source, metrics in performance.items():
            # 综合评分 = 0.3*准确率 + 0.3*F1 + 0.2*夏普 + 0.2*胜率
            score = (
                0.3 * metrics.accuracy +
                0.3 * metrics.f1_score +
                0.2 * (metrics.sharpe_ratio / 3.0 if metrics.sharpe_ratio > 0 else 0) +
                0.2 * metrics.win_rate
            )
            scores[source] = max(0.1, min(1.0, score))  # 限制在0.1-1.0
        
        # 归一化为权重
        total_score = sum(scores.values())
        new_weights = {source: score / total_score for source, score in scores.items()}
        
        # 应用约束
        new_weights = self._apply_constraints(new_weights)
        
        # 记录权重变化
        self._log_weight_changes(new_weights)
        
        self.current_weights = new_weights
        return new_weights
    
    def _apply_constraints(self, weights: Dict[SignalSource, float]) -> Dict[SignalSource, float]:
        """应用权重约束"""
        constrained = {}
        
        for source, weight in weights.items():
            # 限制在min_weight和max_weight之间
            constrained[source] = max(self.min_weight, min(self.max_weight, weight))
        
        # 重新归一化
        total = sum(constrained.values())
        constrained = {k: v / total for k, v in constrained.items()}
        
        return constrained
    
    def _log_weight_changes(self, new_weights: Dict[SignalSource, float]):
        """记录权重变化"""
        logger.info("权重优化结果:")
        for source in SignalSource:
            old_w = self.current_weights.get(source, 0.0)
            new_w = new_weights.get(source, 0.0)
            change = new_w - old_w
            logger.info(f"  {source.value}: {old_w:.3f} → {new_w:.3f} ({change:+.3f})")
    
    def get_performance_summary(self, lookback: int = 10) -> pd.DataFrame:
        """获取性能摘要"""
        recent = self.performance_history[-lookback:]
        
        if not recent:
            return pd.DataFrame()
        
        data = []
        for perf in recent:
            data.append({
                'source': perf.source.value,
                'accuracy': perf.accuracy,
                'f1_score': perf.f1_score,
                'sharpe': perf.sharpe_ratio,
                'win_rate': perf.win_rate,
                'avg_return': perf.avg_return,
                'timestamp': perf.timestamp
            })
        
        return pd.DataFrame(data)
    
    def should_update(self, last_update: datetime) -> bool:
        """判断是否应该更新权重"""
        now = datetime.now()
        delta = now - last_update
        
        if self.update_frequency == 'daily':
            return delta.days >= 1
        elif self.update_frequency == 'weekly':
            return delta.days >= 7
        elif self.update_frequency == 'monthly':
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
                              actual_returns: Dict[str, float]) -> Optional[Dict[SignalSource, float]]:
        """如果需要则更新权重"""
        if not self.optimizer.should_update(self.last_update):
            return None
        
        # 评估性能
        performance = self.optimizer.evaluate_performance(signals, actual_returns)
        
        # 优化权重
        new_weights = self.optimizer.optimize_weights(performance)
        
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
