#!/usr/bin/env python
"""
一进二涨停板因子组合优化器
用于因子权重优化、IC计算、因子筛选
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FactorOptimizer:
    """
    因子组合优化器
    
    功能：
    1. 计算因子IC（信息系数）
    2. 因子权重优化
    3. 因子正交化处理
    4. 因子组合评分
    """
    
    def __init__(self, cache_dir: str = "./workspace/factor_optimizer_cache"):
        """初始化优化器"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化历史
        self.optimization_history: List[Dict] = []
        
        logger.info("✅ 因子组合优化器初始化成功")
    
    def calculate_ic(
        self,
        factor_values: pd.Series,
        target_returns: pd.Series
    ) -> Dict[str, float]:
        """
        计算因子IC（信息系数）
        
        Args:
            factor_values: 因子值序列
            target_returns: 目标收益率序列
            
        Returns:
            IC统计指标
        """
        # 删除缺失值
        valid_mask = factor_values.notna() & target_returns.notna()
        factor_clean = factor_values[valid_mask]
        returns_clean = target_returns[valid_mask]
        
        if len(factor_clean) < 10:
            logger.warning(f"有效样本过少: {len(factor_clean)}")
            return {'ic': 0, 'rank_ic': 0, 'ir': 0}
        
        # 计算Pearson IC
        ic = factor_clean.corr(returns_clean)
        
        # 计算Spearman Rank IC
        rank_ic = factor_clean.rank().corr(returns_clean.rank())
        
        # 计算IC的标准差（用于IR）
        # 这里简化处理，实际应该用时间序列IC的标准差
        ic_std = 0.05  # 假设值
        
        # 计算IR (Information Ratio)
        ir = ic / ic_std if ic_std > 0 else 0
        
        return {
            'ic': float(ic) if not np.isnan(ic) else 0,
            'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0,
            'ir': float(ir) if not np.isnan(ir) else 0,
            'n_samples': len(factor_clean)
        }
    
    def optimize_factor_weights(
        self,
        factors: List[Dict[str, Any]],
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        method: str = 'ic_weighted'
    ) -> Dict[str, float]:
        """
        优化因子权重
        
        Args:
            factors: 因子列表
            factor_matrix: 因子值矩阵 (样本 x 因子)
            target_returns: 目标收益率
            method: 优化方法 ['ic_weighted', 'equal', 'max_ic', 'ridge']
            
        Returns:
            因子权重字典
        """
        logger.info(f"开始因子权重优化，方法: {method}")
        
        weights = {}
        
        if method == 'equal':
            # 等权重
            n_factors = len(factors)
            for factor in factors:
                weights[factor['name']] = 1.0 / n_factors
        
        elif method == 'ic_weighted':
            # IC加权
            ic_scores = {}
            total_ic = 0
            
            for factor in factors:
                factor_name = factor['name']
                if factor_name in factor_matrix.columns:
                    ic_result = self.calculate_ic(
                        factor_matrix[factor_name],
                        target_returns
                    )
                    ic_scores[factor_name] = abs(ic_result['ic'])
                    total_ic += abs(ic_result['ic'])
            
            # 归一化
            if total_ic > 0:
                for factor_name, ic in ic_scores.items():
                    weights[factor_name] = ic / total_ic
            else:
                # 退化到等权
                for factor in factors:
                    weights[factor['name']] = 1.0 / len(factors)
        
        elif method == 'max_ic':
            # 只选择IC最高的因子
            best_factor = None
            max_ic = -1
            
            for factor in factors:
                factor_name = factor['name']
                if factor_name in factor_matrix.columns:
                    ic_result = self.calculate_ic(
                        factor_matrix[factor_name],
                        target_returns
                    )
                    if abs(ic_result['ic']) > max_ic:
                        max_ic = abs(ic_result['ic'])
                        best_factor = factor_name
            
            if best_factor:
                weights = {best_factor: 1.0}
                for factor in factors:
                    if factor['name'] != best_factor:
                        weights[factor['name']] = 0.0
        
        elif method == 'ridge':
            # 岭回归优化（简化版本）
            from sklearn.linear_model import Ridge
            
            X = factor_matrix.fillna(0)
            y = target_returns.fillna(0)
            
            model = Ridge(alpha=0.1)
            model.fit(X, y)
            
            # 归一化系数为权重
            coef_abs = np.abs(model.coef_)
            coef_sum = coef_abs.sum()
            
            if coef_sum > 0:
                for i, factor in enumerate(factors):
                    weights[factor['name']] = coef_abs[i] / coef_sum
            else:
                for factor in factors:
                    weights[factor['name']] = 1.0 / len(factors)
        
        logger.info(f"权重优化完成: {weights}")
        return weights
    
    def select_best_factors(
        self,
        factors: List[Dict[str, Any]],
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        n_select: int = 10,
        min_ic: float = 0.05,
        max_corr: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        筛选最优因子组合
        
        Args:
            factors: 因子列表
            factor_matrix: 因子值矩阵
            target_returns: 目标收益率
            n_select: 选择因子数量
            min_ic: 最小IC阈值
            max_corr: 最大相关系数阈值（用于去相关）
            
        Returns:
            筛选后的因子列表
        """
        logger.info(f"开始因子筛选，目标选择 {n_select} 个因子")
        
        # 1. 计算每个因子的IC
        factor_scores = []
        
        for factor in factors:
            factor_name = factor['name']
            if factor_name in factor_matrix.columns:
                ic_result = self.calculate_ic(
                    factor_matrix[factor_name],
                    target_returns
                )
                
                factor_scores.append({
                    'factor': factor,
                    'ic': abs(ic_result['ic']),
                    'rank_ic': abs(ic_result['rank_ic']),
                    'ir': ic_result['ir']
                })
        
        # 2. 按IC排序
        factor_scores.sort(key=lambda x: x['ic'], reverse=True)
        
        # 3. 去掉IC过低的因子
        factor_scores = [f for f in factor_scores if f['ic'] >= min_ic]
        
        if not factor_scores:
            logger.warning("没有满足IC阈值的因子")
            return []
        
        # 4. 逐步添加因子，避免高相关
        selected_factors = []
        selected_names = []
        
        for score_info in factor_scores:
            if len(selected_factors) >= n_select:
                break
            
            factor_name = score_info['factor']['name']
            
            # 检查与已选因子的相关性
            if selected_names:
                correlations = []
                for selected_name in selected_names:
                    corr = factor_matrix[factor_name].corr(
                        factor_matrix[selected_name]
                    )
                    correlations.append(abs(corr))
                
                # 如果与任何已选因子相关性过高，跳过
                if max(correlations) > max_corr:
                    logger.info(f"跳过因子 {factor_name}，相关性过高: {max(correlations):.3f}")
                    continue
            
            # 添加因子
            factor_info = score_info['factor'].copy()
            factor_info['actual_ic'] = score_info['ic']
            factor_info['actual_rank_ic'] = score_info['rank_ic']
            factor_info['ir'] = score_info['ir']
            
            selected_factors.append(factor_info)
            selected_names.append(factor_name)
            
            logger.info(f"选择因子 {factor_name}, IC={score_info['ic']:.4f}")
        
        logger.info(f"最终选择 {len(selected_factors)} 个因子")
        return selected_factors
    
    def create_composite_score(
        self,
        factor_matrix: pd.DataFrame,
        weights: Dict[str, float],
        standardize: bool = True
    ) -> pd.Series:
        """
        创建因子组合评分
        
        Args:
            factor_matrix: 因子值矩阵
            weights: 因子权重
            standardize: 是否标准化
            
        Returns:
            综合评分序列
        """
        scores = pd.Series(0.0, index=factor_matrix.index)
        
        for factor_name, weight in weights.items():
            if factor_name in factor_matrix.columns and weight > 0:
                factor_values = factor_matrix[factor_name]
                
                if standardize:
                    # 标准化到 [0, 1]
                    factor_min = factor_values.min()
                    factor_max = factor_values.max()
                    if factor_max > factor_min:
                        factor_std = (factor_values - factor_min) / (factor_max - factor_min)
                    else:
                        factor_std = factor_values
                else:
                    factor_std = factor_values
                
                scores += weight * factor_std.fillna(0)
        
        return scores
    
    def backtest_factors(
        self,
        factors: List[Dict[str, Any]],
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        回测因子组合
        
        Args:
            factors: 因子列表
            factor_matrix: 因子值矩阵
            target_returns: 目标收益率
            weights: 因子权重（可选）
            
        Returns:
            回测结果
        """
        logger.info("开始因子回测")
        
        # 如果没有提供权重，使用IC加权
        if weights is None:
            weights = self.optimize_factor_weights(
                factors, factor_matrix, target_returns, method='ic_weighted'
            )
        
        # 创建综合评分
        composite_scores = self.create_composite_score(factor_matrix, weights)
        
        # 按评分分组（五分位）
        composite_scores_clean = composite_scores.dropna()
        target_returns_clean = target_returns[composite_scores_clean.index]
        
        quintiles = pd.qcut(composite_scores_clean, 5, labels=False, duplicates='drop')
        
        # 计算各分组收益
        group_returns = {}
        for q in range(5):
            mask = (quintiles == q)
            if mask.sum() > 0:
                group_returns[f'Q{q+1}'] = target_returns_clean[mask].mean()
        
        # 多空收益（最高分组 - 最低分组）
        long_short_return = group_returns.get('Q5', 0) - group_returns.get('Q1', 0)
        
        # 单调性检验
        monotonicity = all(
            group_returns.get(f'Q{i}', 0) <= group_returns.get(f'Q{i+1}', 0)
            for i in range(1, 5)
        )
        
        results = {
            'group_returns': group_returns,
            'long_short_return': long_short_return,
            'monotonicity': monotonicity,
            'weights': weights,
            'n_samples': len(composite_scores_clean)
        }
        
        logger.info(f"回测完成，多空收益: {long_short_return:.4f}")
        return results
    
    def save_optimization_result(
        self,
        result: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """保存优化结果"""
        if filename is None:
            filename = f"optimization_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"优化结果已保存: {filepath}")
        return str(filepath)


# 演示使用
async def demo():
    """演示因子优化"""
    print("=" * 70)
    print("因子组合优化演示")
    print("=" * 70)
    
    # 创建优化器
    optimizer = FactorOptimizer()
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 100
    
    # 假设有5个因子
    factors = [
        {'name': '封板强度', 'expected_ic': 0.08},
        {'name': '连板高度', 'expected_ic': 0.12},
        {'name': '题材共振', 'expected_ic': 0.10},
        {'name': '早盘涨停', 'expected_ic': 0.15},
        {'name': '量能爆发', 'expected_ic': 0.09}
    ]
    
    # 生成因子值矩阵
    factor_matrix = pd.DataFrame({
        '封板强度': np.random.randn(n_samples),
        '连板高度': np.random.randn(n_samples),
        '题材共振': np.random.randn(n_samples),
        '早盘涨停': np.random.randn(n_samples),
        '量能爆发': np.random.randn(n_samples)
    })
    
    # 生成目标收益（与因子有相关性）
    target_returns = (
        0.08 * factor_matrix['封板强度'] +
        0.12 * factor_matrix['连板高度'] +
        0.10 * factor_matrix['题材共振'] +
        0.15 * factor_matrix['早盘涨停'] +
        0.09 * factor_matrix['量能爆发'] +
        np.random.randn(n_samples) * 0.5
    )
    
    # 1. 计算IC
    print("\n📊 步骤1: 计算各因子IC")
    for factor in factors:
        ic_result = optimizer.calculate_ic(
            factor_matrix[factor['name']],
            target_returns
        )
        print(f"  {factor['name']}: IC={ic_result['ic']:.4f}, Rank IC={ic_result['rank_ic']:.4f}")
    
    # 2. 优化权重
    print("\n⚖️  步骤2: 优化因子权重")
    weights = optimizer.optimize_factor_weights(
        factors, factor_matrix, target_returns, method='ic_weighted'
    )
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    # 3. 选择最优因子
    print("\n🔍 步骤3: 筛选最优因子")
    selected = optimizer.select_best_factors(
        factors, factor_matrix, target_returns,
        n_select=3, min_ic=0.05, max_corr=0.7
    )
    print(f"  选择了 {len(selected)} 个因子:")
    for factor in selected:
        print(f"    - {factor['name']}: IC={factor.get('actual_ic', 0):.4f}")
    
    # 4. 回测
    print("\n📈 步骤4: 回测因子组合")
    backtest_result = optimizer.backtest_factors(
        factors, factor_matrix, target_returns
    )
    print(f"  分组收益: {backtest_result['group_returns']}")
    print(f"  多空收益: {backtest_result['long_short_return']:.4f}")
    print(f"  单调性: {backtest_result['monotonicity']}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo())
