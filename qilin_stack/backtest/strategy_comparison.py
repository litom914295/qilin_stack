"""
策略对比分析工具 (Strategy Comparison Tool)
多维度横向对比多个策略的表现

核心功能：
1. 多策略并行回测
2. 多维度指标对比
3. 相对表现分析
4. 统计显著性检验
5. 综合评分排名
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class StrategyMetrics:
    """策略指标"""
    name: str
    
    # 收益指标
    total_return: float
    annual_return: float
    cumulative_return: float
    
    # 风险指标
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # 交易指标
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_return: float
    
    # 时序数据
    equity_curve: pd.Series
    returns: pd.Series
    drawdowns: pd.Series


class ComparisonDimension(Enum):
    """对比维度"""
    RETURN = "收益性"
    RISK = "风险控制"
    CONSISTENCY = "稳定性"
    EFFICIENCY = "交易效率"
    ROBUSTNESS = "鲁棒性"


@dataclass
class ComparisonResult:
    """对比结果"""
    strategies: List[str]
    winner: Dict[ComparisonDimension, str]  # 各维度最优策略
    scores: pd.DataFrame                    # 综合评分表
    metrics_table: pd.DataFrame             # 指标对比表
    correlation_matrix: pd.DataFrame        # 策略收益相关性
    ranking: List[str]                      # 综合排名


class StrategyComparator:
    """策略对比分析器"""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.benchmark: Optional[StrategyMetrics] = None
    
    def add_strategy(self, metrics: StrategyMetrics, is_benchmark: bool = False):
        """
        添加策略
        
        Args:
            metrics: 策略指标
            is_benchmark: 是否为基准策略
        """
        self.strategies[metrics.name] = metrics
        
        if is_benchmark:
            self.benchmark = metrics
        
        print(f"✅ 添加策略: {metrics.name}" + (" (基准)" if is_benchmark else ""))
    
    def compare(self) -> ComparisonResult:
        """
        执行策略对比
        
        Returns:
            ComparisonResult: 对比结果
        """
        if len(self.strategies) < 2:
            raise ValueError("至少需要2个策略才能进行对比")
        
        print(f"\n🔍 开始对比{len(self.strategies)}个策略...\n")
        
        # 构建指标对比表
        metrics_table = self._build_metrics_table()
        
        # 计算各维度得分
        scores = self._calculate_dimension_scores()
        
        # 确定各维度最优策略
        winners = self._determine_winners(scores)
        
        # 计算策略收益相关性
        correlation = self._calculate_correlation()
        
        # 综合排名
        ranking = self._calculate_ranking(scores)
        
        result = ComparisonResult(
            strategies=list(self.strategies.keys()),
            winner=winners,
            scores=scores,
            metrics_table=metrics_table,
            correlation_matrix=correlation,
            ranking=ranking
        )
        
        return result
    
    def _build_metrics_table(self) -> pd.DataFrame:
        """构建指标对比表"""
        data = []
        
        for name, metrics in self.strategies.items():
            row = {
                '策略名称': name,
                '总收益率': f"{metrics.total_return:.2%}",
                '年化收益': f"{metrics.annual_return:.2%}",
                '波动率': f"{metrics.volatility:.2%}",
                '夏普比率': f"{metrics.sharpe_ratio:.2f}",
                '索提诺比率': f"{metrics.sortino_ratio:.2f}",
                '卡玛比率': f"{metrics.calmar_ratio:.2f}",
                '最大回撤': f"{metrics.max_drawdown:.2%}",
                '回撤持续期': f"{metrics.max_drawdown_duration}天",
                '总交易次数': metrics.total_trades,
                '胜率': f"{metrics.win_rate:.2%}",
                '盈亏比': f"{metrics.profit_factor:.2f}",
                '平均盈利': f"{metrics.avg_win:.2%}",
                '平均亏损': f"{metrics.avg_loss:.2%}",
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def _calculate_dimension_scores(self) -> pd.DataFrame:
        """计算各维度得分（0-100分）"""
        scores_data = []
        
        for name, metrics in self.strategies.items():
            # 收益性得分（权重：年化收益40% + 累计收益30% + 夏普比率30%）
            return_score = (
                self._normalize_metric(metrics.annual_return, 'higher_better') * 0.4 +
                self._normalize_metric(metrics.cumulative_return, 'higher_better') * 0.3 +
                self._normalize_metric(metrics.sharpe_ratio, 'higher_better', abs_scale=True) * 0.3
            ) * 100
            
            # 风险控制得分（权重：最大回撤50% + 波动率30% + 索提诺比率20%）
            risk_score = (
                self._normalize_metric(metrics.max_drawdown, 'lower_better') * 0.5 +
                self._normalize_metric(metrics.volatility, 'lower_better') * 0.3 +
                self._normalize_metric(metrics.sortino_ratio, 'higher_better', abs_scale=True) * 0.2
            ) * 100
            
            # 稳定性得分（权重：卡玛比率40% + 回撤持续期30% + 胜率30%）
            consistency_score = (
                self._normalize_metric(metrics.calmar_ratio, 'higher_better', abs_scale=True) * 0.4 +
                self._normalize_metric(metrics.max_drawdown_duration, 'lower_better') * 0.3 +
                self._normalize_metric(metrics.win_rate, 'higher_better') * 0.3
            ) * 100
            
            # 交易效率得分（权重：盈亏比50% + 平均交易收益30% + 交易频率20%）
            # 交易频率：假设每年250个交易日
            trade_frequency = metrics.total_trades / 250 if metrics.total_trades > 0 else 0
            efficiency_score = (
                self._normalize_metric(metrics.profit_factor, 'higher_better') * 0.5 +
                self._normalize_metric(metrics.avg_trade_return, 'higher_better') * 0.3 +
                self._normalize_metric(trade_frequency, 'moderate_better') * 0.2  # 适中最好
            ) * 100
            
            # 鲁棒性得分（收益曲线平滑度 + 回撤恢复能力）
            returns_std = metrics.returns.std()
            robustness_score = (
                self._normalize_metric(returns_std, 'lower_better') * 0.6 +
                self._normalize_metric(metrics.max_drawdown_duration, 'lower_better') * 0.4
            ) * 100
            
            scores_data.append({
                '策略': name,
                '收益性': round(return_score, 2),
                '风险控制': round(risk_score, 2),
                '稳定性': round(consistency_score, 2),
                '交易效率': round(efficiency_score, 2),
                '鲁棒性': round(robustness_score, 2),
                '综合得分': round((return_score * 0.3 + risk_score * 0.25 + 
                                  consistency_score * 0.2 + efficiency_score * 0.15 + 
                                  robustness_score * 0.1), 2)
            })
        
        df = pd.DataFrame(scores_data)
        return df
    
    def _normalize_metric(self, value: float, direction: str = 'higher_better', 
                          abs_scale: bool = False) -> float:
        """
        归一化指标到0-1区间
        
        Args:
            value: 指标值
            direction: 'higher_better' 或 'lower_better' 或 'moderate_better'
            abs_scale: 是否为绝对值量表（如夏普比率可能为负）
        """
        # 收集所有策略的该指标值
        # 简化实现：假设value已经在合理范围内
        
        if abs_scale:
            # 对于可能为负的指标（如夏普比率），先平移到正值
            if value < 0:
                return 0.0
            # 夏普比率：0以下为0分，2以上为1分
            return min(value / 2.0, 1.0)
        
        if direction == 'higher_better':
            # 数值越大越好：简单映射
            return min(max(value, 0), 1.0)
        
        elif direction == 'lower_better':
            # 数值越小越好
            if value <= 0:
                return 1.0
            # 例如：回撤从0%-50%映射到1-0
            return max(1 - min(abs(value), 0.5) * 2, 0)
        
        elif direction == 'moderate_better':
            # 适中最好（例如交易频率）
            # 假设最优在0.5附近
            optimal = 0.5
            deviation = abs(value - optimal)
            return max(1 - deviation * 2, 0)
        
        return 0.5  # 默认中等
    
    def _determine_winners(self, scores: pd.DataFrame) -> Dict[ComparisonDimension, str]:
        """确定各维度最优策略"""
        winners = {}
        
        dimension_cols = {
            ComparisonDimension.RETURN: '收益性',
            ComparisonDimension.RISK: '风险控制',
            ComparisonDimension.CONSISTENCY: '稳定性',
            ComparisonDimension.EFFICIENCY: '交易效率',
            ComparisonDimension.ROBUSTNESS: '鲁棒性'
        }
        
        for dimension, col in dimension_cols.items():
            idx = scores[col].idxmax()
            winner_name = scores.loc[idx, '策略']
            winners[dimension] = winner_name
        
        return winners
    
    def _calculate_correlation(self) -> pd.DataFrame:
        """计算策略收益相关性矩阵"""
        returns_dict = {}
        
        for name, metrics in self.strategies.items():
            returns_dict[name] = metrics.returns
        
        returns_df = pd.DataFrame(returns_dict)
        correlation = returns_df.corr()
        
        return correlation
    
    def _calculate_ranking(self, scores: pd.DataFrame) -> List[str]:
        """计算综合排名"""
        sorted_df = scores.sort_values('综合得分', ascending=False)
        return sorted_df['策略'].tolist()
    
    def print_comparison(self, result: ComparisonResult):
        """打印对比报告"""
        print("\n" + "="*80)
        print("📊 策略对比分析报告")
        print("="*80)
        
        # 1. 基本信息
        print(f"\n对比策略数量: {len(result.strategies)}")
        print(f"策略列表: {', '.join(result.strategies)}")
        
        if self.benchmark:
            print(f"基准策略: {self.benchmark.name}")
        
        # 2. 指标对比表
        print("\n" + "-"*80)
        print("📈 指标对比表")
        print("-"*80)
        print(result.metrics_table.to_string(index=False))
        
        # 3. 维度得分
        print("\n" + "-"*80)
        print("🎯 维度得分（0-100分）")
        print("-"*80)
        print(result.scores.to_string(index=False))
        
        # 4. 各维度冠军
        print("\n" + "-"*80)
        print("🏆 各维度最优策略")
        print("-"*80)
        for dimension, winner in result.winner.items():
            score = result.scores[result.scores['策略'] == winner][dimension.value].values[0]
            print(f"  {dimension.value:8s}: {winner:15s} (得分: {score:.2f})")
        
        # 5. 综合排名
        print("\n" + "-"*80)
        print("🥇 综合排名")
        print("-"*80)
        for i, strategy in enumerate(result.ranking, 1):
            score = result.scores[result.scores['策略'] == strategy]['综合得分'].values[0]
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"  {medal} {strategy:15s} - 综合得分: {score:.2f}")
        
        # 6. 策略相关性
        print("\n" + "-"*80)
        print("🔗 策略收益相关性矩阵")
        print("-"*80)
        print(result.correlation_matrix.round(3).to_string())
        
        # 7. 统计显著性检验（如果有基准）
        if self.benchmark:
            print("\n" + "-"*80)
            print("📊 vs 基准统计检验")
            print("-"*80)
            self._print_significance_tests()
        
        print("\n" + "="*80 + "\n")
    
    def _print_significance_tests(self):
        """打印统计显著性检验结果"""
        if not self.benchmark:
            return
        
        benchmark_returns = self.benchmark.returns.dropna()
        
        for name, metrics in self.strategies.items():
            if name == self.benchmark.name:
                continue
            
            strategy_returns = metrics.returns.dropna()
            
            # 对齐日期
            common_dates = benchmark_returns.index.intersection(strategy_returns.index)
            if len(common_dates) < 30:
                print(f"  {name}: 数据点不足，无法检验")
                continue
            
            bench_aligned = benchmark_returns.loc[common_dates]
            strat_aligned = strategy_returns.loc[common_dates]
            
            # t检验：检验平均收益是否显著不同
            t_stat, p_value = stats.ttest_ind(strat_aligned, bench_aligned)
            
            # 解读结果
            if p_value < 0.01:
                significance = "极显著"
                symbol = "***"
            elif p_value < 0.05:
                significance = "显著"
                symbol = "**"
            elif p_value < 0.1:
                significance = "边际显著"
                symbol = "*"
            else:
                significance = "不显著"
                symbol = ""
            
            # 判断方向
            mean_diff = strat_aligned.mean() - bench_aligned.mean()
            direction = "优于" if mean_diff > 0 else "劣于"
            
            print(f"  {name:15s} {direction} 基准 ({significance} {symbol}, p={p_value:.4f})")
    
    def generate_summary(self, result: ComparisonResult) -> Dict:
        """生成可序列化的摘要"""
        summary = {
            'strategies': result.strategies,
            'winners': {dim.value: winner for dim, winner in result.winner.items()},
            'ranking': result.ranking,
            'top_strategy': result.ranking[0],
            'top_score': result.scores[result.scores['策略'] == result.ranking[0]]['综合得分'].values[0],
            'metrics_comparison': result.metrics_table.to_dict('records'),
            'dimension_scores': result.scores.to_dict('records'),
            'correlation': result.correlation_matrix.to_dict()
        }
        
        return summary


# 使用示例
if __name__ == "__main__":
    # 模拟3个策略的数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # 策略1：保守型（低波动，中等收益）
    returns_1 = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    equity_1 = (1 + returns_1).cumprod()
    
    # 策略2：激进型（高波动，高收益）
    returns_2 = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    equity_2 = (1 + returns_2).cumprod()
    
    # 策略3：稳健型（低波动，低收益）
    returns_3 = pd.Series(np.random.normal(0.0003, 0.005, len(dates)), index=dates)
    equity_3 = (1 + returns_3).cumprod()
    
    # 计算回撤
    def calculate_drawdown(equity: pd.Series) -> pd.Series:
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown
    
    drawdowns_1 = calculate_drawdown(equity_1)
    drawdowns_2 = calculate_drawdown(equity_2)
    drawdowns_3 = calculate_drawdown(equity_3)
    
    # 创建策略指标
    strategy1 = StrategyMetrics(
        name="保守策略",
        total_return=equity_1.iloc[-1] - 1,
        annual_return=(equity_1.iloc[-1] - 1),
        cumulative_return=equity_1.iloc[-1] - 1,
        volatility=returns_1.std() * np.sqrt(252),
        sharpe_ratio=returns_1.mean() / returns_1.std() * np.sqrt(252),
        sortino_ratio=returns_1.mean() / returns_1[returns_1 < 0].std() * np.sqrt(252) if len(returns_1[returns_1 < 0]) > 0 else 0,
        calmar_ratio=(equity_1.iloc[-1] - 1) / abs(drawdowns_1.min()) if drawdowns_1.min() < 0 else 0,
        max_drawdown=drawdowns_1.min(),
        max_drawdown_duration=30,
        total_trades=150,
        win_rate=0.58,
        profit_factor=1.45,
        avg_win=0.012,
        avg_loss=-0.008,
        avg_trade_return=0.0003,
        equity_curve=equity_1,
        returns=returns_1,
        drawdowns=drawdowns_1
    )
    
    strategy2 = StrategyMetrics(
        name="激进策略",
        total_return=equity_2.iloc[-1] - 1,
        annual_return=(equity_2.iloc[-1] - 1),
        cumulative_return=equity_2.iloc[-1] - 1,
        volatility=returns_2.std() * np.sqrt(252),
        sharpe_ratio=returns_2.mean() / returns_2.std() * np.sqrt(252),
        sortino_ratio=returns_2.mean() / returns_2[returns_2 < 0].std() * np.sqrt(252) if len(returns_2[returns_2 < 0]) > 0 else 0,
        calmar_ratio=(equity_2.iloc[-1] - 1) / abs(drawdowns_2.min()) if drawdowns_2.min() < 0 else 0,
        max_drawdown=drawdowns_2.min(),
        max_drawdown_duration=45,
        total_trades=300,
        win_rate=0.52,
        profit_factor=1.35,
        avg_win=0.025,
        avg_loss=-0.018,
        avg_trade_return=0.0004,
        equity_curve=equity_2,
        returns=returns_2,
        drawdowns=drawdowns_2
    )
    
    strategy3 = StrategyMetrics(
        name="稳健策略",
        total_return=equity_3.iloc[-1] - 1,
        annual_return=(equity_3.iloc[-1] - 1),
        cumulative_return=equity_3.iloc[-1] - 1,
        volatility=returns_3.std() * np.sqrt(252),
        sharpe_ratio=returns_3.mean() / returns_3.std() * np.sqrt(252),
        sortino_ratio=returns_3.mean() / returns_3[returns_3 < 0].std() * np.sqrt(252) if len(returns_3[returns_3 < 0]) > 0 else 0,
        calmar_ratio=(equity_3.iloc[-1] - 1) / abs(drawdowns_3.min()) if drawdowns_3.min() < 0 else 0,
        max_drawdown=drawdowns_3.min(),
        max_drawdown_duration=20,
        total_trades=100,
        win_rate=0.62,
        profit_factor=1.65,
        avg_win=0.008,
        avg_loss=-0.005,
        avg_trade_return=0.0002,
        equity_curve=equity_3,
        returns=returns_3,
        drawdowns=drawdowns_3
    )
    
    # 创建对比器
    comparator = StrategyComparator()
    
    # 添加策略
    comparator.add_strategy(strategy1)
    comparator.add_strategy(strategy2)
    comparator.add_strategy(strategy3, is_benchmark=True)  # 稳健策略作为基准
    
    # 执行对比
    result = comparator.compare()
    
    # 打印报告
    comparator.print_comparison(result)
    
    # 生成摘要
    summary = comparator.generate_summary(result)
    print(f"最优策略: {summary['top_strategy']}")
    print(f"综合得分: {summary['top_score']:.2f}")
    
    print("\n✅ 完成")
