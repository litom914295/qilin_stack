"""
投资组合优化器模块
实现多种资产配置优化算法：均值方差、Black-Litterman、风险平价等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from scipy.optimize import minimize
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class OptimizationResult:
    """优化结果"""
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    method: str
    constraints: Dict
    metadata: Dict


# ============================================================================
# 均值方差优化器 (Markowitz)
# ============================================================================

class MeanVarianceOptimizer:
    """
    均值方差优化器 (Markowitz模型)
    最大化夏普比率或最小化风险
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.03):
        """
        初始化优化器
        
        Args:
            returns: 资产收益率数据 (N x M: N个时间点, M个资产)
            risk_free_rate: 无风险利率
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = returns.shape[1]
        
        # 计算期望收益和协方差矩阵
        self.expected_returns = returns.mean() * 252  # 年化
        self.cov_matrix = returns.cov() * 252  # 年化
        
        logger.info(f"均值方差优化器初始化: {self.n_assets}个资产")
    
    def optimize_sharpe(self,
                       target_return: Optional[float] = None,
                       allow_short: bool = False) -> OptimizationResult:
        """
        最大化夏普比率
        
        Args:
            target_return: 目标收益率 (可选)
            allow_short: 是否允许做空
        
        Returns:
            优化结果
        """
        logger.info("优化目标: 最大化夏普比率")
        
        # 目标函数: 负夏普比率
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # 权重和为1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, self.expected_returns) - target_return
            })
        
        # 边界
        if allow_short:
            bounds = [(-1, 1) for _ in range(self.n_assets)]
        else:
            bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # 初始权重 (等权)
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # 优化
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning(f"优化未收敛: {result.message}")
        
        # 计算性能指标
        weights = result.x
        expected_return = np.dot(weights, self.expected_returns)
        expected_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            method='max_sharpe',
            constraints={'target_return': target_return, 'allow_short': allow_short},
            metadata={'success': result.success, 'message': result.message}
        )
    
    def optimize_min_volatility(self, allow_short: bool = False) -> OptimizationResult:
        """
        最小化波动率
        
        Args:
            allow_short: 是否允许做空
        
        Returns:
            优化结果
        """
        logger.info("优化目标: 最小化波动率")
        
        # 目标函数: 波动率
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # 边界
        if allow_short:
            bounds = [(-1, 1) for _ in range(self.n_assets)]
        else:
            bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # 初始权重
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # 优化
        result = minimize(
            portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # 计算性能指标
        weights = result.x
        expected_return = np.dot(weights, self.expected_returns)
        expected_risk = portfolio_volatility(weights)
        sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            method='min_volatility',
            constraints={'allow_short': allow_short},
            metadata={'success': result.success}
        )
    
    def efficient_frontier(self, n_points: int = 50) -> List[OptimizationResult]:
        """
        计算有效前沿
        
        Args:
            n_points: 前沿点数
        
        Returns:
            有效前沿上的优化结果列表
        """
        logger.info(f"计算有效前沿: {n_points}个点")
        
        # 找到最小和最大收益
        min_vol_result = self.optimize_min_volatility()
        max_return = self.expected_returns.max()
        
        target_returns = np.linspace(
            min_vol_result.expected_return,
            max_return * 0.95,
            n_points
        )
        
        frontier = []
        for target_return in target_returns:
            try:
                result = self.optimize_sharpe(target_return=target_return)
                frontier.append(result)
            except Exception as e:
                logger.warning(f"计算前沿点失败 (target={target_return:.4f}): {e}")
        
        return frontier


# ============================================================================
# Black-Litterman模型
# ============================================================================

class BlackLittermanOptimizer:
    """
    Black-Litterman资产配置模型
    结合市场均衡和投资者观点
    """
    
    def __init__(self,
                 returns: pd.DataFrame,
                 market_caps: Optional[np.ndarray] = None,
                 risk_free_rate: float = 0.03,
                 tau: float = 0.05):
        """
        初始化Black-Litterman优化器
        
        Args:
            returns: 资产收益率数据
            market_caps: 市场市值 (用于计算市场权重)
            risk_free_rate: 无风险利率
            tau: 先验不确定性参数
        """
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        
        # 协方差矩阵
        self.cov_matrix = returns.cov() * 252
        
        # 市场权重 (如果没有提供市值，使用等权)
        if market_caps is not None:
            self.market_weights = market_caps / market_caps.sum()
        else:
            self.market_weights = np.ones(self.n_assets) / self.n_assets
        
        # 计算隐含收益 (市场均衡收益)
        self.implied_returns = self._calculate_implied_returns()
        
        logger.info(f"Black-Litterman优化器初始化: {self.n_assets}个资产")
    
    def _calculate_implied_returns(self, delta: float = 2.5) -> np.ndarray:
        """
        计算隐含收益率
        
        Args:
            delta: 风险厌恶系数
        
        Returns:
            隐含收益率向量
        """
        return delta * np.dot(self.cov_matrix, self.market_weights)
    
    def optimize_with_views(self,
                           views: Dict[int, float],
                           view_confidence: float = 0.5) -> OptimizationResult:
        """
        基于投资者观点进行优化
        
        Args:
            views: 投资者观点 {资产索引: 预期收益率}
            view_confidence: 观点置信度 (0-1)
        
        Returns:
            优化结果
        """
        logger.info(f"基于观点优化: {len(views)}个观点")
        
        # 构建观点矩阵P和观点向量Q
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        for i, (asset_idx, expected_return) in enumerate(views.items()):
            P[i, asset_idx] = 1
            Q[i] = expected_return
        
        # 观点不确定性矩阵Ω
        omega = np.diag(np.diag(np.dot(P, np.dot(self.tau * self.cov_matrix, P.T)))) / view_confidence
        
        # Black-Litterman公式
        # 后验协方差
        M_inverse = np.linalg.inv(np.linalg.inv(self.tau * self.cov_matrix) + 
                                  np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        
        # 后验期望收益
        posterior_returns = np.dot(M_inverse,
                                  np.dot(np.linalg.inv(self.tau * self.cov_matrix), 
                                        self.implied_returns) +
                                  np.dot(P.T, np.dot(np.linalg.inv(omega), Q)))
        
        # 使用后验收益优化权重
        def neg_utility(weights):
            return -(np.dot(weights, posterior_returns) - 
                    0.5 * np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(self.n_assets)]
        init_weights = self.market_weights
        
        result = minimize(neg_utility, init_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        weights = result.x
        expected_return = np.dot(weights, posterior_returns)
        expected_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            method='black_litterman',
            constraints={'views': views, 'confidence': view_confidence},
            metadata={'posterior_returns': posterior_returns}
        )


# ============================================================================
# 风险平价 (Risk Parity)
# ============================================================================

class RiskParityOptimizer:
    """
    风险平价优化器
    使每个资产对组合风险的贡献相等
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        初始化风险平价优化器
        
        Args:
            returns: 资产收益率数据
        """
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.cov_matrix = returns.cov() * 252
        
        logger.info(f"风险平价优化器初始化: {self.n_assets}个资产")
    
    def optimize(self) -> OptimizationResult:
        """
        风险平价优化
        
        Returns:
            优化结果
        """
        logger.info("执行风险平价优化")
        
        def risk_parity_objective(weights):
            """目标函数: 最小化风险贡献的方差"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # 每个资产的边际风险贡献
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            
            # 风险贡献
            risk_contrib = weights * marginal_contrib
            
            # 目标收益贡献均等
            target_risk = portfolio_vol / self.n_assets
            
            # 最小化与目标的偏差
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # 约束和边界
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # 初始权重
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        # 优化
        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        expected_return = np.dot(weights, self.returns.mean() * 252)
        expected_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = expected_return / expected_risk
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            method='risk_parity',
            constraints={},
            metadata={'success': result.success}
        )


# ============================================================================
# 使用示例
# ============================================================================

def create_sample_returns(n_assets: int = 5, n_days: int = 252) -> pd.DataFrame:
    """创建模拟收益率数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # 生成相关的收益率
    mean_returns = np.random.uniform(0.0001, 0.001, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)
    
    # 相关系数矩阵
    corr_matrix = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # 对称化
    
    # 协方差矩阵
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    # 生成收益率
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    columns = [f'Asset_{i+1}' for i in range(n_assets)]
    return pd.DataFrame(returns, index=dates, columns=columns)


def main():
    """示例：投资组合优化"""
    print("=" * 80)
    print("投资组合优化 - 示例")
    print("=" * 80)
    
    # 1. 创建数据
    print("\n📊 生成模拟数据...")
    returns = create_sample_returns(n_assets=5, n_days=252)
    print(f"数据维度: {returns.shape}")
    
    # 2. 均值方差优化
    print("\n🎯 均值方差优化 (最大化夏普比率)...")
    mv_optimizer = MeanVarianceOptimizer(returns, risk_free_rate=0.03)
    result_sharpe = mv_optimizer.optimize_sharpe()
    
    print(f"权重: {result_sharpe.weights}")
    print(f"预期收益: {result_sharpe.expected_return:.2%}")
    print(f"预期风险: {result_sharpe.expected_risk:.2%}")
    print(f"夏普比率: {result_sharpe.sharpe_ratio:.2f}")
    
    # 3. 最小波动率
    print("\n📉 最小波动率优化...")
    result_minvol = mv_optimizer.optimize_min_volatility()
    
    print(f"权重: {result_minvol.weights}")
    print(f"预期收益: {result_minvol.expected_return:.2%}")
    print(f"预期风险: {result_minvol.expected_risk:.2%}")
    
    # 4. Black-Litterman
    print("\n🔮 Black-Litterman优化...")
    bl_optimizer = BlackLittermanOptimizer(returns)
    
    # 投资者观点: Asset_1预期收益15%, Asset_3预期收益10%
    views = {0: 0.15, 2: 0.10}
    result_bl = bl_optimizer.optimize_with_views(views, view_confidence=0.7)
    
    print(f"权重: {result_bl.weights}")
    print(f"预期收益: {result_bl.expected_return:.2%}")
    print(f"预期风险: {result_bl.expected_risk:.2%}")
    print(f"夏普比率: {result_bl.sharpe_ratio:.2f}")
    
    # 5. 风险平价
    print("\n⚖️  风险平价优化...")
    rp_optimizer = RiskParityOptimizer(returns)
    result_rp = rp_optimizer.optimize()
    
    print(f"权重: {result_rp.weights}")
    print(f"预期收益: {result_rp.expected_return:.2%}")
    print(f"预期风险: {result_rp.expected_risk:.2%}")
    
    # 6. 有效前沿
    print("\n📈 计算有效前沿...")
    frontier = mv_optimizer.efficient_frontier(n_points=20)
    print(f"有效前沿点数: {len(frontier)}")
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
