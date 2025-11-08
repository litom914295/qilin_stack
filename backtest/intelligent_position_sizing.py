"""
智能仓位管理系统
基于Kelly准则、波动率调整和风险限制的动态仓位管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSignal:
    """仓位信号"""
    symbol: str
    recommended_position: float  # 推荐仓位比例 (0-1)
    kelly_position: float        # Kelly准则仓位
    risk_adjusted_position: float # 风险调整后仓位
    confidence: float            # 信心度 (0-1)
    expected_return: float       # 预期收益率
    win_probability: float       # 胜率
    risk_metrics: Dict[str, float]  # 风险指标
    


@dataclass 
class PortfolioAllocation:
    """组合配置"""
    timestamp: datetime
    allocations: Dict[str, float]  # symbol -> position size
    total_exposure: float          # 总暴露度
    risk_budget_used: float        # 已使用风险预算
    expected_portfolio_return: float  # 组合预期收益
    portfolio_volatility: float    # 组合波动率
    sharpe_ratio: float            # 夏普比率
    max_drawdown_risk: float      # 最大回撤风险
    concentration_risk: float      # 集中度风险
    recommendations: List[str]     # 建议


class IntelligentPositionSizer:
    """智能仓位管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化仓位管理器
        
        Args:
            config: 配置参数
        """
        config = config or {}
        
        # Kelly准则参数
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Kelly分数 (保守化)
        self.max_kelly_position = config.get('max_kelly_position', 0.3)  # 单只最大Kelly仓位
        
        # 风险限制
        self.max_single_position = config.get('max_single_position', 0.15)  # 单只最大仓位
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)   # 板块最大暴露
        self.max_total_exposure = config.get('max_total_exposure', 0.95)    # 最大总仓位
        self.max_portfolio_volatility = config.get('max_portfolio_volatility', 0.2)  # 最大组合波动率
        
        # 风险预算
        self.total_risk_budget = config.get('total_risk_budget', 0.15)  # 总风险预算 (VaR)
        self.confidence_level = config.get('confidence_level', 0.95)     # 置信水平
        
        # 动态调整参数
        self.volatility_lookback = config.get('volatility_lookback', 20)  # 波动率回看期
        self.correlation_lookback = config.get('correlation_lookback', 60) # 相关性回看期
        self.min_confidence_threshold = config.get('min_confidence', 0.6)  # 最小信心阈值
        
        # 历史数据缓存
        self.price_history: Dict[str, pd.Series] = {}
        self.signal_history: List[PositionSignal] = []
        self.allocation_history: List[PortfolioAllocation] = []
        
    def calculate_position_sizes(self,
                                signals: Dict[str, Dict],
                                market_data: pd.DataFrame,
                                current_portfolio: Optional[Dict[str, float]] = None) -> PortfolioAllocation:
        """
        计算智能仓位配置
        
        Args:
            signals: 交易信号 {symbol: {win_prob, expected_return, confidence, ...}}
            market_data: 市场数据
            current_portfolio: 当前持仓
            
        Returns:
            PortfolioAllocation: 仓位配置方案
        """
        timestamp = datetime.now()
        current_portfolio = current_portfolio or {}
        
        # 1. 计算单个标的Kelly仓位
        position_signals = []
        for symbol, signal in signals.items():
            if signal.get('confidence', 0) < self.min_confidence_threshold:
                continue
                
            position_signal = self._calculate_kelly_position(symbol, signal, market_data)
            position_signals.append(position_signal)
        
        # 2. 风险调整
        risk_adjusted_signals = self._apply_risk_adjustments(position_signals, market_data)
        
        # 3. 组合优化
        optimized_allocations = self._optimize_portfolio(risk_adjusted_signals, market_data)
        
        # 4. 应用约束条件
        final_allocations = self._apply_constraints(optimized_allocations, current_portfolio)
        
        # 5. 计算组合指标
        portfolio_metrics = self._calculate_portfolio_metrics(final_allocations, market_data)
        
        # 6. 生成建议
        recommendations = self._generate_recommendations(
            final_allocations, portfolio_metrics, current_portfolio
        )
        
        allocation = PortfolioAllocation(
            timestamp=timestamp,
            allocations=final_allocations,
            total_exposure=sum(final_allocations.values()),
            risk_budget_used=portfolio_metrics['risk_budget_used'],
            expected_portfolio_return=portfolio_metrics['expected_return'],
            portfolio_volatility=portfolio_metrics['volatility'],
            sharpe_ratio=portfolio_metrics['sharpe_ratio'],
            max_drawdown_risk=portfolio_metrics['max_drawdown_risk'],
            concentration_risk=portfolio_metrics['concentration_risk'],
            recommendations=recommendations
        )
        
        # 保存历史
        self.allocation_history.append(allocation)
        if len(self.allocation_history) > 1000:
            self.allocation_history.pop(0)
            
        return allocation
    
    def _calculate_kelly_position(self, symbol: str, signal: Dict, 
                                 market_data: pd.DataFrame) -> PositionSignal:
        """
        计算Kelly准则仓位
        
        Kelly公式: f* = (p*b - q) / b
        其中:
        - f* = 最优仓位比例
        - p = 获胜概率
        - q = 1 - p = 失败概率  
        - b = 赔率 (获胜时的收益/失败时的损失)
        """
        win_prob = signal.get('win_probability', 0.5)
        expected_return = signal.get('expected_return', 0)
        confidence = signal.get('confidence', 0.5)
        stop_loss = signal.get('stop_loss', 0.05)  # 默认5%止损
        
        # 计算赔率
        if stop_loss > 0:
            odds_ratio = abs(expected_return / stop_loss)
        else:
            odds_ratio = 2.0  # 默认赔率
        
        # Kelly公式
        if odds_ratio > 0:
            kelly_position = (win_prob * odds_ratio - (1 - win_prob)) / odds_ratio
        else:
            kelly_position = 0
        
        # 应用Kelly分数（保守化）
        kelly_position *= self.kelly_fraction
        
        # 限制最大Kelly仓位
        kelly_position = min(kelly_position, self.max_kelly_position)
        kelly_position = max(kelly_position, 0)  # 不允许负仓位
        
        # 根据信心度调整
        adjusted_position = kelly_position * confidence
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(symbol, market_data)
        
        return PositionSignal(
            symbol=symbol,
            recommended_position=adjusted_position,
            kelly_position=kelly_position,
            risk_adjusted_position=adjusted_position,  # 后续会进一步调整
            confidence=confidence,
            expected_return=expected_return,
            win_probability=win_prob,
            risk_metrics=risk_metrics
        )
    
    def _calculate_risk_metrics(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """计算风险指标"""
        risk_metrics = {}
        
        if symbol in market_data.index:
            row = market_data.loc[symbol]
            
            # 波动率
            if 'volatility' in row:
                risk_metrics['volatility'] = row['volatility']
            else:
                # 使用历史数据估算
                if symbol in self.price_history:
                    returns = self.price_history[symbol].pct_change()
                    risk_metrics['volatility'] = returns.std() * np.sqrt(252)
                else:
                    risk_metrics['volatility'] = 0.3  # 默认30%年化波动率
            
            # 流动性风险
            volume = row.get('volume', 0)
            risk_metrics['liquidity_score'] = min(1.0, volume / 1e6)  # 百万成交量为基准
            
            # 最大回撤
            if symbol in self.price_history and len(self.price_history[symbol]) > 20:
                prices = self.price_history[symbol]
                rolling_max = prices.expanding().max()
                drawdown = (prices - rolling_max) / rolling_max
                risk_metrics['max_drawdown'] = drawdown.min()
            else:
                risk_metrics['max_drawdown'] = -0.1  # 默认-10%
            
            # Beta (简化计算)
            risk_metrics['beta'] = row.get('beta', 1.0)
            
        else:
            # 默认风险指标
            risk_metrics = {
                'volatility': 0.3,
                'liquidity_score': 0.5,
                'max_drawdown': -0.1,
                'beta': 1.0
            }
        
        return risk_metrics
    
    def _apply_risk_adjustments(self, position_signals: List[PositionSignal],
                               market_data: pd.DataFrame) -> List[PositionSignal]:
        """应用风险调整"""
        adjusted_signals = []
        
        for signal in position_signals:
            # 波动率调整
            volatility = signal.risk_metrics.get('volatility', 0.3)
            vol_adjustment = min(1.0, 0.2 / volatility)  # 目标20%波动率
            
            # 流动性调整
            liquidity_adjustment = signal.risk_metrics.get('liquidity_score', 0.5)
            
            # 回撤风险调整
            max_drawdown = signal.risk_metrics.get('max_drawdown', -0.1)
            drawdown_adjustment = min(1.0, 0.1 / abs(max_drawdown))  # 目标最大回撤10%
            
            # 综合调整
            total_adjustment = vol_adjustment * liquidity_adjustment * drawdown_adjustment
            
            # 调整仓位
            signal.risk_adjusted_position = signal.recommended_position * total_adjustment
            signal.risk_adjusted_position = min(signal.risk_adjusted_position, self.max_single_position)
            
            adjusted_signals.append(signal)
        
        return adjusted_signals
    
    def _optimize_portfolio(self, position_signals: List[PositionSignal],
                          market_data: pd.DataFrame) -> Dict[str, float]:
        """
        组合优化
        使用简化的均值-方差优化
        """
        if not position_signals:
            return {}
        
        # 提取数据
        symbols = [s.symbol for s in position_signals]
        expected_returns = np.array([s.expected_return for s in position_signals])
        initial_weights = np.array([s.risk_adjusted_position for s in position_signals])
        
        # 归一化初始权重
        if initial_weights.sum() > 0:
            initial_weights = initial_weights / initial_weights.sum()
        else:
            initial_weights = np.ones(len(symbols)) / len(symbols)
        
        # 计算协方差矩阵（简化版）
        cov_matrix = self._estimate_covariance_matrix(symbols, market_data)
        
        # 风险预算优化
        optimized_weights = self._risk_budgeting_optimization(
            expected_returns, cov_matrix, initial_weights
        )
        
        # 构建配置字典
        allocations = {}
        for i, symbol in enumerate(symbols):
            if optimized_weights[i] > 0.001:  # 最小仓位阈值
                allocations[symbol] = optimized_weights[i]
        
        return allocations
    
    def _estimate_covariance_matrix(self, symbols: List[str], 
                                   market_data: pd.DataFrame) -> np.ndarray:
        """估算协方差矩阵"""
        n = len(symbols)
        
        # 简化：使用恒定相关系数和个体波动率
        correlation = 0.3  # 假设平均相关系数
        cov_matrix = np.full((n, n), correlation)
        np.fill_diagonal(cov_matrix, 1.0)
        
        # 使用个体波动率调整
        volatilities = []
        for symbol in symbols:
            if symbol in market_data.index:
                vol = market_data.loc[symbol].get('volatility', 0.3)
            else:
                vol = 0.3
            volatilities.append(vol)
        
        vol_array = np.array(volatilities)
        cov_matrix = cov_matrix * np.outer(vol_array, vol_array)
        
        return cov_matrix
    
    def _risk_budgeting_optimization(self, expected_returns: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    initial_weights: np.ndarray) -> np.ndarray:
        """
        风险预算优化
        简化版：基于风险平价原理
        """
        n = len(expected_returns)
        
        # 计算风险贡献
        portfolio_vol = np.sqrt(initial_weights @ cov_matrix @ initial_weights)
        marginal_risk = cov_matrix @ initial_weights / portfolio_vol
        risk_contribution = initial_weights * marginal_risk
        
        # 目标：均衡风险贡献
        target_risk = self.total_risk_budget / n
        
        # 迭代调整权重
        weights = initial_weights.copy()
        for _ in range(10):  # 简单迭代
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol == 0:
                break
                
            marginal_risk = cov_matrix @ weights / portfolio_vol
            risk_contribution = weights * marginal_risk
            
            # 调整权重
            adjustment = target_risk / (risk_contribution + 1e-6)
            weights = weights * np.power(adjustment, 0.2)  # 缓慢调整
            
            # 归一化
            weights = weights / weights.sum() * min(self.max_total_exposure, weights.sum())
            
            # 应用单个限制
            weights = np.minimum(weights, self.max_single_position)
        
        return weights
    
    def _apply_constraints(self, allocations: Dict[str, float],
                          current_portfolio: Dict[str, float]) -> Dict[str, float]:
        """应用约束条件"""
        constrained = {}
        
        # 单只股票限制
        for symbol, weight in allocations.items():
            constrained[symbol] = min(weight, self.max_single_position)
        
        # 总仓位限制
        total = sum(constrained.values())
        if total > self.max_total_exposure:
            scale = self.max_total_exposure / total
            for symbol in constrained:
                constrained[symbol] *= scale
        
        # 平滑调整（避免频繁调仓）
        smoothed = {}
        smoothing_factor = 0.3  # 调整速度
        
        for symbol, target_weight in constrained.items():
            current_weight = current_portfolio.get(symbol, 0)
            new_weight = current_weight + smoothing_factor * (target_weight - current_weight)
            
            # 最小调整阈值
            if abs(new_weight - current_weight) > 0.01:  # 1%以上才调整
                smoothed[symbol] = new_weight
            else:
                smoothed[symbol] = current_weight
        
        # 处理需要清仓的持仓
        for symbol, current_weight in current_portfolio.items():
            if symbol not in smoothed and current_weight > 0.01:
                smoothed[symbol] = 0  # 清仓信号
        
        return smoothed
    
    def _calculate_portfolio_metrics(self, allocations: Dict[str, float],
                                    market_data: pd.DataFrame) -> Dict[str, float]:
        """计算组合指标"""
        metrics = {}
        
        if not allocations:
            return {
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown_risk': 0,
                'concentration_risk': 0,
                'risk_budget_used': 0
            }
        
        weights = np.array(list(allocations.values()))
        symbols = list(allocations.keys())
        
        # 预期收益
        expected_returns = []
        for symbol in symbols:
            if symbol in market_data.index:
                exp_ret = market_data.loc[symbol].get('expected_return', 0)
            else:
                exp_ret = 0
            expected_returns.append(exp_ret)
        
        expected_returns = np.array(expected_returns)
        metrics['expected_return'] = weights @ expected_returns
        
        # 组合波动率
        cov_matrix = self._estimate_covariance_matrix(symbols, market_data)
        metrics['volatility'] = np.sqrt(weights @ cov_matrix @ weights)
        
        # 夏普比率
        risk_free_rate = 0.03  # 3%无风险利率
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['expected_return'] - risk_free_rate) / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # 最大回撤风险（简化估算）
        metrics['max_drawdown_risk'] = -metrics['volatility'] * 2.0  # 2倍标准差
        
        # 集中度风险 (HHI)
        metrics['concentration_risk'] = np.sum(weights ** 2)
        
        # 风险预算使用
        var_95 = metrics['volatility'] * 1.645  # 95% VaR
        metrics['risk_budget_used'] = min(1.0, var_95 / self.total_risk_budget)
        
        return metrics
    
    def _generate_recommendations(self, allocations: Dict[str, float],
                                 portfolio_metrics: Dict[str, float],
                                 current_portfolio: Dict[str, float]) -> List[str]:
        """生成仓位建议"""
        recommendations = []
        
        # 风险预算检查
        if portfolio_metrics['risk_budget_used'] > 0.9:
            recommendations.append("⚠️ 风险预算接近上限，建议降低总仓位")
        
        # 集中度检查
        if portfolio_metrics['concentration_risk'] > 0.2:
            recommendations.append("📊 组合集中度过高，建议分散投资")
        
        # 波动率检查
        if portfolio_metrics['volatility'] > self.max_portfolio_volatility:
            recommendations.append(f"📈 组合波动率({portfolio_metrics['volatility']:.1%})超过限制，建议降低高波动资产")
        
        # 夏普比率检查
        if portfolio_metrics['sharpe_ratio'] < 0.5:
            recommendations.append("📉 夏普比率偏低，建议优化风险收益比")
        
        # 调仓建议
        major_changes = []
        for symbol, new_weight in allocations.items():
            old_weight = current_portfolio.get(symbol, 0)
            change = new_weight - old_weight
            if abs(change) > 0.05:  # 5%以上的调整
                if change > 0:
                    major_changes.append(f"加仓 {symbol}: {old_weight:.1%} → {new_weight:.1%}")
                else:
                    major_changes.append(f"减仓 {symbol}: {old_weight:.1%} → {new_weight:.1%}")
        
        if major_changes:
            recommendations.append("建议调仓:")
            recommendations.extend(major_changes[:5])  # 最多显示5个
        
        # 新建仓建议
        new_positions = [s for s in allocations if s not in current_portfolio and allocations[s] > 0.01]
        if new_positions:
            recommendations.append(f"建议新建仓: {', '.join(new_positions[:3])}")
        
        # 清仓建议
        close_positions = [s for s in allocations if allocations[s] == 0 and current_portfolio.get(s, 0) > 0]
        if close_positions:
            recommendations.append(f"建议清仓: {', '.join(close_positions[:3])}")
        
        return recommendations
    
    def calculate_kelly_fraction(self, historical_returns: pd.Series) -> float:
        """
        基于历史数据计算最优Kelly分数
        
        Args:
            historical_returns: 历史收益率序列
            
        Returns:
            float: 最优Kelly分数
        """
        if len(historical_returns) < 30:
            return self.kelly_fraction  # 数据不足，使用默认值
        
        # 计算历史统计
        mean_return = historical_returns.mean()
        std_return = historical_returns.std()
        win_rate = (historical_returns > 0).mean()
        
        # 计算平均赢亏比
        wins = historical_returns[historical_returns > 0]
        losses = historical_returns[historical_returns < 0]
        
        if len(losses) > 0:
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean())
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        else:
            win_loss_ratio = 2
        
        # Kelly公式
        if win_loss_ratio > 0:
            full_kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        else:
            full_kelly = 0
        
        # 考虑参数不确定性，使用保守的Kelly分数
        # 经验法则：使用1/4到1/3的完整Kelly
        conservative_factor = 0.25
        
        # 根据夏普比率调整
        sharpe = mean_return / std_return if std_return > 0 else 0
        if sharpe > 1:
            conservative_factor = 0.33
        elif sharpe < 0.5:
            conservative_factor = 0.15
        
        optimal_kelly = max(0, min(full_kelly * conservative_factor, 0.25))
        
        return optimal_kelly
    
    def generate_allocation_report(self, allocation: PortfolioAllocation) -> str:
        """生成仓位配置报告"""
        report = []
        report.append("=" * 60)
        report.append("📊 智能仓位配置报告")
        report.append(f"时间: {allocation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # 组合概览
        report.append("\n📈 组合概览")
        report.append(f"  • 总仓位: {allocation.total_exposure:.1%}")
        report.append(f"  • 持仓数量: {len(allocation.allocations)}")
        report.append(f"  • 风险预算使用: {allocation.risk_budget_used:.1%}")
        
        # 风险收益指标
        report.append("\n💰 风险收益指标")
        report.append(f"  • 预期收益率: {allocation.expected_portfolio_return:.2%}")
        report.append(f"  • 组合波动率: {allocation.portfolio_volatility:.2%}")
        report.append(f"  • 夏普比率: {allocation.sharpe_ratio:.2f}")
        report.append(f"  • 最大回撤风险: {allocation.max_drawdown_risk:.2%}")
        report.append(f"  • 集中度风险: {allocation.concentration_risk:.3f}")
        
        # 仓位配置
        report.append("\n📊 仓位配置 (前10)")
        sorted_positions = sorted(allocation.allocations.items(), key=lambda x: x[1], reverse=True)
        for symbol, weight in sorted_positions[:10]:
            report.append(f"  • {symbol}: {weight:.2%}")
        
        # 操作建议
        if allocation.recommendations:
            report.append("\n💡 操作建议")
            for rec in allocation.recommendations:
                report.append(f"  • {rec}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# 测试代码
if __name__ == "__main__":
    # 创建仓位管理器
    position_sizer = IntelligentPositionSizer()
    
    # 生成测试信号
    test_signals = {
        "STOCK_001": {
            "win_probability": 0.65,
            "expected_return": 0.15,
            "confidence": 0.8,
            "stop_loss": 0.05
        },
        "STOCK_002": {
            "win_probability": 0.60,
            "expected_return": 0.12,
            "confidence": 0.7,
            "stop_loss": 0.04
        },
        "STOCK_003": {
            "win_probability": 0.55,
            "expected_return": 0.20,
            "confidence": 0.6,
            "stop_loss": 0.08
        },
        "STOCK_004": {
            "win_probability": 0.70,
            "expected_return": 0.10,
            "confidence": 0.9,
            "stop_loss": 0.03
        },
        "STOCK_005": {
            "win_probability": 0.58,
            "expected_return": 0.18,
            "confidence": 0.65,
            "stop_loss": 0.06
        }
    }
    
    # 生成市场数据
    np.random.seed(42)
    market_data = pd.DataFrame({
        'volatility': np.random.uniform(0.2, 0.4, 5),
        'volume': np.random.uniform(5e5, 5e6, 5),
        'expected_return': [0.15, 0.12, 0.20, 0.10, 0.18],
        'beta': np.random.uniform(0.8, 1.2, 5)
    }, index=["STOCK_001", "STOCK_002", "STOCK_003", "STOCK_004", "STOCK_005"])
    
    # 当前持仓
    current_portfolio = {
        "STOCK_001": 0.10,
        "STOCK_002": 0.15,
        "STOCK_006": 0.05  # 需要清仓的持仓
    }
    
    print("🚀 开始测试智能仓位管理系统...\n")
    
    # 计算仓位配置
    allocation = position_sizer.calculate_position_sizes(
        test_signals, market_data, current_portfolio
    )
    
    # 生成报告
    report = position_sizer.generate_allocation_report(allocation)
    print(report)
    
    # 测试Kelly分数计算
    print("\n📊 Kelly分数优化测试")
    print("-" * 40)
    
    # 生成历史收益数据
    historical_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    optimal_kelly = position_sizer.calculate_kelly_fraction(historical_returns)
    
    print(f"历史收益统计:")
    print(f"  • 平均收益: {historical_returns.mean():.4f}")
    print(f"  • 收益波动: {historical_returns.std():.4f}")
    print(f"  • 胜率: {(historical_returns > 0).mean():.2%}")
    print(f"  • 最优Kelly分数: {optimal_kelly:.4f}")
    
    print("\n✅ 测试完成！")