"""
策略对比工具
提供多策略性能对比、指标分析、最佳策略推荐等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class StrategyMetrics:
    """策略指标"""
    name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    
    def to_dict(self):
        return {
            '策略名称': self.name,
            '总收益率': f"{self.total_return:.2%}",
            '年化收益': f"{self.annual_return:.2%}",
            '夏普比率': f"{self.sharpe_ratio:.2f}",
            '最大回撤': f"{self.max_drawdown:.2%}",
            '胜率': f"{self.win_rate:.2%}",
            '交易次数': self.trade_count
        }


class StrategyComparator:
    """策略对比器"""
    
    def __init__(self):
        """初始化"""
        self.strategies: Dict[str, StrategyMetrics] = {}
    
    def add_strategy(self, metrics: StrategyMetrics):
        """添加策略"""
        self.strategies[metrics.name] = metrics
    
    def compare_metrics(self) -> pd.DataFrame:
        """对比指标"""
        if not self.strategies:
            return pd.DataFrame()
        
        return pd.DataFrame([
            s.to_dict() for s in self.strategies.values()
        ])
    
    def rank_strategies(self, by: str = 'sharpe_ratio') -> List[str]:
        """按指标排名策略"""
        sorted_strategies = sorted(
            self.strategies.values(),
            key=lambda s: getattr(s, by),
            reverse=True
        )
        return [s.name for s in sorted_strategies]
    
    def recommend_best(self, weights: Optional[Dict[str, float]] = None) -> str:
        """推荐最佳策略（综合评分）"""
        if not self.strategies:
            return ""
        
        # 默认权重
        if weights is None:
            weights = {
                'annual_return': 0.3,
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.2,  # 负向指标，回撤越小越好
                'win_rate': 0.2
            }
        
        scores = {}
        for name, strategy in self.strategies.items():
            score = (
                strategy.annual_return * weights['annual_return'] +
                strategy.sharpe_ratio * 0.1 * weights['sharpe_ratio'] +  # 归一化
                (1 + strategy.max_drawdown) * weights['max_drawdown'] +  # 回撤是负值
                strategy.win_rate * weights['win_rate']
            )
            scores[name] = score
        
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        return best_strategy
    
    def get_comparison_summary(self) -> Dict:
        """获取对比摘要"""
        if not self.strategies:
            return {}
        
        return {
            'best_return': max(self.strategies.values(), key=lambda s: s.total_return).name,
            'best_sharpe': max(self.strategies.values(), key=lambda s: s.sharpe_ratio).name,
            'min_drawdown': min(self.strategies.values(), key=lambda s: s.max_drawdown).name,
            'best_winrate': max(self.strategies.values(), key=lambda s: s.win_rate).name,
            'strategy_count': len(self.strategies)
        }


__all__ = ['StrategyMetrics', 'StrategyComparator']
