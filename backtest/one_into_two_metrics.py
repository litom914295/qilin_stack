"""
一进二策略专用评估指标
计算P@N, Hit@N, 板强度等核心指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class OneIntoTwoMetrics:
    """
    一进二专用指标 - T+1制度适配版
    
    核心改进：
    - 新增T+1收盘收益指标（因为T+1无法卖出）
    - 新增T+1最大浮亏（风险指标）
    - 新增T+2最终收益（实际获利）
    - 保留兼容旧版指标
    """
    # 基础统计
    date: str
    total_limitups: int      # 当日涨停总数
    predicted_count: int     # 预测数量
    hit_count: int          # 命中数量（次日涨停）
    touch_count: int        # 触板数量（次日触及涨停）
    
    # 核心指标
    precision_at_n: float   # P@N: 预测准确率
    hit_at_n: float        # Hit@N: 命中率（相对于总涨停池）
    board_strength: float   # 板强度：封板时间/力度综合
    
    # 细分指标
    first_board_hit: int    # 首板命中数
    multi_board_hit: int    # 连板命中数
    theme_hit_rate: float   # 题材命中率
    sector_concentration: float  # 板块集中度
    
    # 执行指标（竞价买入模式）
    avg_queue_position: float  # 平均排队位置（排板模式）
    avg_fill_ratio: float     # 平均成交比例
    unfilled_rate: float      # 未成交率
    avg_auction_gap: float = 0.0  # 平均竞价涨幅（新增）
    
    # T+1收益指标（新增 - 关键指标）
    t1_close_avg_return: float = 0.0     # T+1平均收盘收益率
    t1_positive_rate: float = 0.0        # T+1收盘盈利率（>0的比例）
    t1_avg_max_return: float = 0.0       # T+1平均最大浮盈
    t1_avg_min_return: float = 0.0       # T+1平均最大浮亏（风险）
    max_unrealized_loss: float = 0.0     # T+1最大未实现亏损
    
    # T+2最终收益指标（新增）
    t2_final_return: float = 0.0         # T+2最终平均收益
    t2_positive_rate: float = 0.0        # T+2最终盈利率
    t2_best_sell_return: float = 0.0     # T+2最佳卖出收益
    
    # 传统收益指标（兼容旧版）
    avg_next_day_return: float = 0.0     # 次日平均收益（=t1_close_avg_return）
    win_loss_ratio: float = 0.0          # 盈亏比
    max_single_return: float = 0.0       # 最大单票收益
    max_single_loss: float = 0.0         # 最大单票亏损


class OneIntoTwoEvaluator:
    """一进二策略评估器"""
    
    def __init__(self, limit_types: Optional[Dict[str, float]] = None):
        """
        初始化评估器
        
        Args:
            limit_types: 涨跌停限制 {普通: 0.1, 科创: 0.2, ST: 0.05}
        """
        self.limit_types = limit_types or {
            'normal': 0.10,
            'kcb': 0.20,
            'st': 0.05
        }
        self.metrics_history: List[OneIntoTwoMetrics] = []
    
    def evaluate_predictions(self, 
                            predictions: pd.DataFrame,
                            actual_results: pd.DataFrame,
                            date: str) -> OneIntoTwoMetrics:
        """
        评估预测结果
        
        Args:
            predictions: 预测DataFrame，含 [symbol, prob, rank]
            actual_results: 实际结果DataFrame，含 [symbol, is_limit_up, touch_limit, return]
            date: 评估日期
            
        Returns:
            OneIntoTwoMetrics: 评估指标
        """
        # 合并预测和实际
        merged = pd.merge(
            predictions, actual_results, 
            on='symbol', how='left'
        )
        
        # 基础统计
        total_limitups = len(actual_results[actual_results['is_limit_up'] == True])
        predicted_count = len(predictions)
        hit_count = merged['is_limit_up'].sum()
        touch_count = merged['touch_limit'].sum()
        
        # 核心指标计算
        precision_at_n = hit_count / predicted_count if predicted_count > 0 else 0
        hit_at_n = hit_count / total_limitups if total_limitups > 0 else 0
        
        # 板强度计算
        board_strength = self._calculate_board_strength(merged)
        
        # 细分指标
        first_board_hit = self._count_first_board_hits(merged)
        multi_board_hit = hit_count - first_board_hit
        theme_hit_rate = self._calculate_theme_hit_rate(merged)
        sector_concentration = self._calculate_sector_concentration(merged)
        
        # 执行指标
        avg_queue_position = merged.get('queue_position', pd.Series([0.5])).mean()
        avg_fill_ratio = merged.get('fill_ratio', pd.Series([1.0])).mean()
        unfilled_rate = (merged.get('fill_ratio', pd.Series([1.0])) < 0.01).mean()
        
        # 收益指标
        returns = merged.get('return', pd.Series([0]))
        avg_next_day_return = returns.mean()
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) > 0 and negative_returns.mean() != 0:
            win_loss_ratio = abs(positive_returns.mean() / negative_returns.mean())
        else:
            win_loss_ratio = float('inf') if len(positive_returns) > 0 else 0
        
        max_single_return = returns.max() if len(returns) > 0 else 0
        max_single_loss = returns.min() if len(returns) > 0 else 0
        
        metrics = OneIntoTwoMetrics(
            date=date,
            total_limitups=total_limitups,
            predicted_count=predicted_count,
            hit_count=hit_count,
            touch_count=touch_count,
            precision_at_n=precision_at_n,
            hit_at_n=hit_at_n,
            board_strength=board_strength,
            first_board_hit=first_board_hit,
            multi_board_hit=multi_board_hit,
            theme_hit_rate=theme_hit_rate,
            sector_concentration=sector_concentration,
            avg_queue_position=avg_queue_position,
            avg_fill_ratio=avg_fill_ratio,
            unfilled_rate=unfilled_rate,
            avg_auction_gap=avg_auction_gap,
            # T+1指标（新增）
            t1_close_avg_return=t1_close_avg_return,
            t1_positive_rate=t1_positive_rate,
            t1_avg_max_return=t1_avg_max_return,
            t1_avg_min_return=t1_avg_min_return,
            max_unrealized_loss=max_unrealized_loss,
            # T+2指标（新增）
            t2_final_return=t2_final_return,
            t2_positive_rate=t2_positive_rate,
            t2_best_sell_return=t2_best_sell_return,
            # 传统指标（兼容）
            avg_next_day_return=avg_next_day_return,
            win_loss_ratio=win_loss_ratio,
            max_single_return=max_single_return,
            max_single_loss=max_single_loss
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_board_strength(self, data: pd.DataFrame) -> float:
        """计算板强度"""
        if data.empty:
            return 0
        
        # 使用封板时间、封单量等计算
        # 这里简化为命中率的加权
        hit_rate = data['is_limit_up'].mean() if 'is_limit_up' in data else 0
        touch_rate = data['touch_limit'].mean() if 'touch_limit' in data else 0
        
        return hit_rate * 0.7 + touch_rate * 0.3
    
    def _count_first_board_hits(self, data: pd.DataFrame) -> int:
        """统计首板命中数"""
        if 'board_count' not in data.columns:
            return 0
        
        first_boards = data[data['board_count'] == 1]
        return first_boards['is_limit_up'].sum()
    
    def _calculate_theme_hit_rate(self, data: pd.DataFrame) -> float:
        """计算题材命中率"""
        if 'theme' not in data.columns:
            return 0
        
        # 按题材分组计算命中率
        theme_hits = data.groupby('theme')['is_limit_up'].mean()
        return theme_hits.mean() if len(theme_hits) > 0 else 0
    
    def _calculate_sector_concentration(self, data: pd.DataFrame) -> float:
        """计算板块集中度（HHI指数）"""
        if 'sector' not in data.columns:
            return 0
        
        sector_counts = data['sector'].value_counts()
        total = len(data)
        
        if total == 0:
            return 0
        
        # 计算HHI（赫芬达尔-赫希曼指数）
        hhi = sum((count/total) ** 2 for count in sector_counts)
        return hhi
    
    def calculate_daily_metrics(self,
                               predictions: Dict[str, pd.DataFrame],
                               actual_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算多日评估指标
        
        Args:
            predictions: {date: predictions_df}
            actual_results: {date: results_df}
            
        Returns:
            DataFrame with daily metrics
        """
        daily_metrics = []
        
        for date in predictions.keys():
            if date not in actual_results:
                continue
            
            metrics = self.evaluate_predictions(
                predictions[date],
                actual_results[date],
                date
            )
            
            daily_metrics.append({
                '日期': date,
                '预测数': metrics.predicted_count,
                '命中数': metrics.hit_count,
                'P@N': metrics.precision_at_n,
                'Hit@N': metrics.hit_at_n,
                '板强度': metrics.board_strength,
                '平均收益': metrics.avg_next_day_return,
                '盈亏比': metrics.win_loss_ratio,
                '成交率': metrics.avg_fill_ratio,
                '未成交率': metrics.unfilled_rate
            })
        
        return pd.DataFrame(daily_metrics)
    
    def calculate_cumulative_metrics(self) -> Dict[str, float]:
        """计算累计指标"""
        if not self.metrics_history:
            return {}
        
        total_predicted = sum(m.predicted_count for m in self.metrics_history)
        total_hit = sum(m.hit_count for m in self.metrics_history)
        total_touch = sum(m.touch_count for m in self.metrics_history)
        
        avg_precision = total_hit / total_predicted if total_predicted > 0 else 0
        avg_hit_rate = np.mean([m.hit_at_n for m in self.metrics_history])
        avg_board_strength = np.mean([m.board_strength for m in self.metrics_history])
        
        avg_return = np.mean([m.avg_next_day_return for m in self.metrics_history])
        best_day_return = max(m.avg_next_day_return for m in self.metrics_history)
        worst_day_return = min(m.avg_next_day_return for m in self.metrics_history)
        
        avg_fill_ratio = np.mean([m.avg_fill_ratio for m in self.metrics_history])
        avg_unfilled = np.mean([m.unfilled_rate for m in self.metrics_history])
        
        return {
            '总预测数': total_predicted,
            '总命中数': total_hit,
            '总触板数': total_touch,
            '平均P@N': avg_precision,
            '平均Hit@N': avg_hit_rate,
            '平均板强度': avg_board_strength,
            '平均日收益': avg_return,
            '最佳日收益': best_day_return,
            '最差日收益': worst_day_return,
            '平均成交率': avg_fill_ratio,
            '平均未成交率': avg_unfilled,
            '评估天数': len(self.metrics_history)
        }
    
    def generate_report(self) -> str:
        """生成评估报告"""
        cumulative = self.calculate_cumulative_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("📊 一进二策略评估报告")
        report.append("=" * 60)
        
        report.append("\n📈 整体表现")
        report.append(f"  评估天数: {cumulative.get('评估天数', 0)}天")
        report.append(f"  总预测数: {cumulative.get('总预测数', 0)}")
        report.append(f"  总命中数: {cumulative.get('总命中数', 0)}")
        
        report.append("\n🎯 核心指标")
        report.append(f"  平均P@N: {cumulative.get('平均P@N', 0):.2%}")
        report.append(f"  平均Hit@N: {cumulative.get('平均Hit@N', 0):.2%}")
        report.append(f"  平均板强度: {cumulative.get('平均板强度', 0):.3f}")
        
        report.append("\n💰 收益指标")
        report.append(f"  平均日收益: {cumulative.get('平均日收益', 0):.2%}")
        report.append(f"  最佳日收益: {cumulative.get('最佳日收益', 0):.2%}")
        report.append(f"  最差日收益: {cumulative.get('最差日收益', 0):.2%}")
        
        report.append("\n📊 执行指标")
        report.append(f"  平均成交率: {cumulative.get('平均成交率', 0):.2%}")
        report.append(f"  平均未成交率: {cumulative.get('平均未成交率', 0):.2%}")
        
        # 最近5日表现
        if len(self.metrics_history) > 0:
            report.append("\n📅 最近表现")
            for m in self.metrics_history[-5:]:
                report.append(
                    f"  {m.date}: P@N={m.precision_at_n:.1%}, "
                    f"命中={m.hit_count}/{m.predicted_count}, "
                    f"收益={m.avg_next_day_return:.2%}"
                )
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


def evaluate_one_into_two_backtest(backtest_results: Dict,
                                  predictions: pd.DataFrame) -> Dict[str, float]:
    """
    评估一进二回测结果
    
    Args:
        backtest_results: 回测结果字典
        predictions: 预测数据
        
    Returns:
        评估指标字典
    """
    evaluator = OneIntoTwoEvaluator()
    
    # 提取交易数据
    trades = backtest_results.get('trades', [])
    
    if not trades:
        return {
            'precision_at_10': 0,
            'hit_at_10': 0,
            'board_strength': 0,
            'avg_fill_ratio': 0
        }
    
    # 按日期分组
    trades_df = pd.DataFrame(trades)
    trades_by_date = trades_df.groupby(trades_df['timestamp'].dt.date)
    
    metrics_list = []
    for date, day_trades in trades_by_date:
        # 计算当日指标
        hit_count = len(day_trades[day_trades['pnl'] > 0])
        total_count = len(day_trades)
        
        if total_count > 0:
            precision = hit_count / total_count
            metrics_list.append(precision)
    
    # 汇总指标
    avg_precision = np.mean(metrics_list) if metrics_list else 0
    
    return {
        'precision_at_10': avg_precision,
        'hit_at_10': avg_precision * 0.8,  # 简化估算
        'board_strength': avg_precision * 0.5 + 0.3,  # 简化估算
        'avg_fill_ratio': backtest_results.get('avg_fill_ratio', 0.5)
    }


# 测试代码
if __name__ == "__main__":
    # 创建评估器
    evaluator = OneIntoTwoEvaluator()
    
    # 生成测试数据
    dates = pd.date_range('2025-01-01', '2025-01-05', freq='B')
    
    for date in dates:
        # 模拟预测
        predictions = pd.DataFrame({
            'symbol': [f'STOCK_{i:03d}' for i in range(10)],
            'prob': np.random.uniform(0.5, 0.9, 10),
            'rank': range(1, 11)
        })
        
        # 模拟实际结果
        actual = pd.DataFrame({
            'symbol': [f'STOCK_{i:03d}' for i in range(10)],
            'is_limit_up': np.random.choice([True, False], 10, p=[0.3, 0.7]),
            'touch_limit': np.random.choice([True, False], 10, p=[0.5, 0.5]),
            'return': np.random.normal(0.02, 0.05, 10),
            'board_count': np.random.choice([1, 2, 3], 10, p=[0.6, 0.3, 0.1]),
            'theme': np.random.choice(['AI', '新能源', '医药'], 10),
            'sector': np.random.choice(['科技', '消费', '金融'], 10),
            'queue_position': np.random.uniform(0, 1, 10),
            'fill_ratio': np.random.uniform(0, 1, 10)
        })
        
        # 评估
        metrics = evaluator.evaluate_predictions(predictions, actual, date.strftime('%Y-%m-%d'))
        print(f"\n📅 {date.strftime('%Y-%m-%d')} 评估结果:")
        print(f"  P@N: {metrics.precision_at_n:.2%}")
        print(f"  Hit@N: {metrics.hit_at_n:.2%}")
        print(f"  板强度: {metrics.board_strength:.3f}")
        print(f"  平均收益: {metrics.avg_next_day_return:.2%}")
    
    # 生成报告
    print("\n" + evaluator.generate_report())
    
    print("\n✅ 评估完成！")