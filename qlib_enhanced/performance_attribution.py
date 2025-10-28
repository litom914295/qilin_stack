"""
绩效归因分析系统 (Performance Attribution)
实现Brinson归因模型、因子收益分解、交易成本分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """归因分析结果"""
    allocation_effect: float  # 资产配置效应
    selection_effect: float   # 证券选择效应
    interaction_effect: float  # 交互效应
    total_active_return: float  # 总超额收益
    factor_contributions: Dict[str, float]  # 因子贡献


class BrinsonAttribution:
    """Brinson归因模型"""
    
    def __init__(self, portfolio_weights: pd.DataFrame, 
                 portfolio_returns: pd.DataFrame,
                 benchmark_weights: pd.DataFrame,
                 benchmark_returns: pd.DataFrame):
        """
        初始化Brinson归因
        
        Args:
            portfolio_weights: 组合权重 (资产×时间)
            portfolio_returns: 组合收益 (资产×时间)
            benchmark_weights: 基准权重 (资产×时间)
            benchmark_returns: 基准收益 (资产×时间)
        """
        self.pw = portfolio_weights
        self.pr = portfolio_returns
        self.bw = benchmark_weights
        self.br = benchmark_returns
        
        logger.info(f"Brinson归因初始化: {portfolio_weights.shape[0]}个资产")
    
    def analyze(self) -> AttributionResult:
        """执行Brinson归因分析"""
        # 计算各项效应
        allocation = self._allocation_effect()
        selection = self._selection_effect()
        interaction = self._interaction_effect()
        
        total_return = allocation + selection + interaction
        
        return AttributionResult(
            allocation_effect=allocation,
            selection_effect=selection,
            interaction_effect=interaction,
            total_active_return=total_return,
            factor_contributions={}
        )
    
    def _allocation_effect(self) -> float:
        """资产配置效应 = Σ(Wp - Wb) * Rb"""
        weight_diff = self.pw - self.bw
        return (weight_diff * self.br).sum().sum()
    
    def _selection_effect(self) -> float:
        """证券选择效应 = Σ Wb * (Rp - Rb)"""
        return_diff = self.pr - self.br
        return (self.bw * return_diff).sum().sum()
    
    def _interaction_effect(self) -> float:
        """交互效应 = Σ(Wp - Wb) * (Rp - Rb)"""
        weight_diff = self.pw - self.bw
        return_diff = self.pr - self.br
        return (weight_diff * return_diff).sum().sum()


class FactorAttribution:
    """因子归因分析"""
    
    def __init__(self, returns: pd.Series, factors: pd.DataFrame):
        """
        初始化因子归因
        
        Args:
            returns: 组合收益率时间序列
            factors: 因子暴露矩阵 (时间×因子)
        """
        self.returns = returns
        self.factors = factors
        
    def analyze(self) -> Dict[str, float]:
        """分解收益到各因子"""
        # 简化版: 使用线性回归分解
        from scipy import stats
        
        contributions = {}
        total_explained = 0.0
        
        for factor_name in self.factors.columns:
            factor_values = self.factors[factor_name].values
            returns_values = self.returns.values
            
            # 回归分析
            if len(factor_values) > 1 and len(returns_values) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    factor_values, returns_values
                )
                contribution = slope * factor_values.mean()
                contributions[factor_name] = contribution
                total_explained += contribution
        
        # 残差(特异性收益)
        contributions['Residual'] = self.returns.mean() - total_explained
        
        return contributions


class TransactionCostAnalysis:
    """交易成本分析"""
    
    def __init__(self, trades: pd.DataFrame):
        """
        初始化交易成本分析
        
        Args:
            trades: 交易记录 (columns: symbol, quantity, price, timestamp)
        """
        self.trades = trades
        
    def analyze(self, commission_rate: float = 0.001,
                slippage_bps: float = 5.0) -> Dict[str, float]:
        """
        分析交易成本
        
        Args:
            commission_rate: 佣金率
            slippage_bps: 滑点(基点)
        
        Returns:
            成本分解
        """
        total_value = (self.trades['quantity'] * self.trades['price']).sum()
        
        # 佣金成本
        commission_cost = total_value * commission_rate
        
        # 滑点成本
        slippage_cost = total_value * (slippage_bps / 10000)
        
        # 市场冲击成本(简化估计)
        market_impact = total_value * 0.0001  # 0.01%
        
        total_cost = commission_cost + slippage_cost + market_impact
        
        return {
            'total_cost': total_cost,
            'commission_cost': commission_cost,
            'slippage_cost': slippage_cost,
            'market_impact_cost': market_impact,
            'cost_as_pct_of_value': total_cost / total_value if total_value > 0 else 0
        }


def create_sample_attribution_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """创建示例归因数据"""
    np.random.seed(42)
    
    assets = ['Asset_A', 'Asset_B', 'Asset_C']
    periods = 12
    
    # 组合权重
    pw = pd.DataFrame({
        'Asset_A': np.random.uniform(0.2, 0.4, periods),
        'Asset_B': np.random.uniform(0.3, 0.5, periods),
        'Asset_C': np.random.uniform(0.2, 0.3, periods)
    })
    pw = pw.div(pw.sum(axis=1), axis=0)
    
    # 基准权重
    bw = pd.DataFrame({
        'Asset_A': [0.33] * periods,
        'Asset_B': [0.33] * periods,
        'Asset_C': [0.34] * periods
    })
    
    # 收益率
    pr = pd.DataFrame({
        'Asset_A': np.random.normal(0.008, 0.02, periods),
        'Asset_B': np.random.normal(0.010, 0.025, periods),
        'Asset_C': np.random.normal(0.006, 0.015, periods)
    })
    
    br = pd.DataFrame({
        'Asset_A': np.random.normal(0.007, 0.018, periods),
        'Asset_B': np.random.normal(0.008, 0.020, periods),
        'Asset_C': np.random.normal(0.007, 0.018, periods)
    })
    
    return pw, pr, bw, br


def main():
    """示例: 绩效归因分析"""
    print("=" * 80)
    print("绩效归因分析 - 示例")
    print("=" * 80)
    
    # 1. Brinson归因
    print("\n📊 Brinson归因分析...")
    pw, pr, bw, br = create_sample_attribution_data()
    
    brinson = BrinsonAttribution(pw, pr, bw, br)
    result = brinson.analyze()
    
    print(f"\n资产配置效应: {result.allocation_effect:.4f}")
    print(f"证券选择效应: {result.selection_effect:.4f}")
    print(f"交互效应: {result.interaction_effect:.4f}")
    print(f"总超额收益: {result.total_active_return:.4f}")
    
    # 2. 因子归因
    print("\n📈 因子归因分析...")
    returns = pd.Series(np.random.normal(0.01, 0.02, 12))
    factors = pd.DataFrame({
        'Market': np.random.normal(0.008, 0.015, 12),
        'Size': np.random.normal(0.002, 0.01, 12),
        'Value': np.random.normal(0.003, 0.01, 12)
    })
    
    factor_attr = FactorAttribution(returns, factors)
    contributions = factor_attr.analyze()
    
    print("\n因子贡献:")
    for factor, contrib in contributions.items():
        print(f"  {factor}: {contrib:.4f}")
    
    # 3. 交易成本分析
    print("\n💰 交易成本分析...")
    trades = pd.DataFrame({
        'symbol': ['A', 'B', 'C'] * 10,
        'quantity': np.random.randint(100, 1000, 30),
        'price': np.random.uniform(10, 100, 30),
        'timestamp': pd.date_range('2024-01-01', periods=30, freq='D')
    })
    
    cost_analysis = TransactionCostAnalysis(trades)
    costs = cost_analysis.analyze()
    
    print(f"\n总交易成本: ¥{costs['total_cost']:,.2f}")
    print(f"佣金成本: ¥{costs['commission_cost']:,.2f}")
    print(f"滑点成本: ¥{costs['slippage_cost']:,.2f}")
    print(f"市场冲击成本: ¥{costs['market_impact_cost']:,.2f}")
    print(f"成本占比: {costs['cost_as_pct_of_value']:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
