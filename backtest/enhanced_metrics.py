"""
增强的回测评估指标
提供更详细的策略评估和可执行性分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeDetail:
    """单笔交易详情"""
    symbol: str
    date: str
    side: str  # 'buy' or 'sell'
    planned_qty: int
    filled_qty: int
    fill_ratio: float
    price: float
    commission: float
    slippage: float
    pnl: Optional[float] = None
    stock_type: str = 'main'  # main/chinext/st
    limit_strength: str = 'medium'  # strong/medium/weak


class EnhancedMetricsCalculator:
    """增强指标计算器"""
    
    def __init__(self):
        self.trades: List[TradeDetail] = []
        self.daily_stats = []
        
    def add_trade(self, trade: TradeDetail):
        """添加交易记录"""
        self.trades.append(trade)
        
    def calculate_all_metrics(self) -> Dict:
        """计算所有增强指标"""
        if not self.trades:
            return self._empty_metrics()
            
        basic_metrics = self._calculate_basic_metrics()
        fill_metrics = self._calculate_fill_metrics()
        symbol_metrics = self._calculate_symbol_metrics()
        execution_score = self._calculate_execution_score()
        
        return {
            **basic_metrics,
            **fill_metrics,
            **symbol_metrics,
            'execution_score': execution_score,
            'trade_details': self._get_trade_summary()
        }
    
    def _calculate_basic_metrics(self) -> Dict:
        """计算基础指标"""
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        
        # 盈亏统计
        profitable_trades = df[df['pnl'] > 0] if 'pnl' in df else pd.DataFrame()
        losing_trades = df[df['pnl'] < 0] if 'pnl' in df else pd.DataFrame()
        
        total_pnl = df['pnl'].sum() if 'pnl' in df else 0
        win_rate = len(profitable_trades) / len(df) if len(df) > 0 else 0
        
        avg_win = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        return {
            'total_trades': len(df),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': profitable_trades['pnl'].max() if len(profitable_trades) > 0 else 0,
            'max_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
        }
    
    def _calculate_fill_metrics(self) -> Dict:
        """计算成交相关指标"""
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        buy_trades = df[df['side'] == 'buy']
        
        if len(buy_trades) == 0:
            return {
                'avg_fill_ratio': 1.0,
                'unfilled_rate': 0.0,
                'partial_fill_rate': 0.0,
                'full_fill_rate': 1.0
            }
        
        # 成交比例分布
        fill_ratios = buy_trades['fill_ratio'].values
        
        unfilled = (fill_ratios < 0.01).sum()
        partial = ((fill_ratios >= 0.01) & (fill_ratios < 0.99)).sum()
        full = (fill_ratios >= 0.99).sum()
        
        # 按股票类型分组统计
        type_stats = {}
        for stype in ['main', 'chinext', 'st']:
            type_trades = buy_trades[buy_trades['stock_type'] == stype]
            if len(type_trades) > 0:
                type_stats[f'{stype}_avg_fill'] = type_trades['fill_ratio'].mean()
                type_stats[f'{stype}_trades'] = len(type_trades)
        
        # 按封板强度分组统计
        strength_stats = {}
        for strength in ['strong', 'medium', 'weak']:
            strength_trades = buy_trades[buy_trades['limit_strength'] == strength]
            if len(strength_trades) > 0:
                strength_stats[f'{strength}_avg_fill'] = strength_trades['fill_ratio'].mean()
                strength_stats[f'{strength}_trades'] = len(strength_trades)
        
        return {
            'avg_fill_ratio': fill_ratios.mean(),
            'unfilled_rate': unfilled / len(buy_trades),
            'partial_fill_rate': partial / len(buy_trades),
            'full_fill_rate': full / len(buy_trades),
            'fill_ratio_std': fill_ratios.std(),
            'fill_ratio_25pct': np.percentile(fill_ratios, 25),
            'fill_ratio_median': np.median(fill_ratios),
            'fill_ratio_75pct': np.percentile(fill_ratios, 75),
            **type_stats,
            **strength_stats
        }
    
    def _calculate_symbol_metrics(self) -> Dict:
        """计算个股维度指标"""
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        
        # 按股票分组
        symbol_groups = df.groupby('symbol')
        
        symbol_stats = []
        for symbol, group in symbol_groups:
            buy_trades = group[group['side'] == 'buy']
            sell_trades = group[group['side'] == 'sell']
            
            # 计算该股票的统计
            stats = {
                'symbol': symbol,
                'total_trades': len(group),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'avg_fill_ratio': buy_trades['fill_ratio'].mean() if len(buy_trades) > 0 else 1.0,
                'total_pnl': group['pnl'].sum() if 'pnl' in group.columns else 0,
                'avg_holding_days': self._calc_holding_days(symbol, df),
                'win_rate': self._calc_symbol_win_rate(symbol, df)
            }
            symbol_stats.append(stats)
        
        # 转换为DataFrame便于分析
        symbol_df = pd.DataFrame(symbol_stats)
        
        if len(symbol_df) > 0:
            # 找出最佳和最差的股票
            best_symbol = symbol_df.nlargest(1, 'total_pnl')['symbol'].values[0] if len(symbol_df) > 0 else None
            worst_symbol = symbol_df.nsmallest(1, 'total_pnl')['symbol'].values[0] if len(symbol_df) > 0 else None
            
            return {
                'unique_symbols': len(symbol_df),
                'best_symbol': best_symbol,
                'worst_symbol': worst_symbol,
                'symbol_concentration': symbol_df['total_trades'].max() / symbol_df['total_trades'].sum(),
                'symbol_stats': symbol_stats
            }
        else:
            return {
                'unique_symbols': 0,
                'best_symbol': None,
                'worst_symbol': None,
                'symbol_concentration': 0,
                'symbol_stats': []
            }
    
    def _calculate_execution_score(self) -> float:
        """
        计算策略可执行性评分 (0-100)
        
        评分维度：
        1. 成交率 (40分)
        2. 成交稳定性 (20分)
        3. 成本控制 (20分)
        4. 风险控制 (20分)
        """
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        buy_trades = df[df['side'] == 'buy']
        
        if len(buy_trades) == 0:
            return 50.0  # 无交易时给中性分
        
        score = 0.0
        
        # 1. 成交率评分 (40分)
        avg_fill_ratio = buy_trades['fill_ratio'].mean()
        fill_score = avg_fill_ratio * 40
        score += fill_score
        
        # 2. 成交稳定性 (20分)
        fill_std = buy_trades['fill_ratio'].std()
        stability_score = max(0, 20 * (1 - fill_std))  # 标准差越小越稳定
        score += stability_score
        
        # 3. 成本控制 (20分)
        # 计算平均成本率（手续费+滑点）
        if 'commission' in df.columns and 'slippage' in df.columns:
            total_value = (df['filled_qty'] * df['price']).sum()
            total_cost = df['commission'].sum() + (df['slippage'] * df['filled_qty'] * df['price']).sum()
            cost_rate = total_cost / total_value if total_value > 0 else 0
            cost_score = max(0, 20 * (1 - cost_rate * 100))  # 成本率越低越好
        else:
            cost_score = 15  # 默认中等分数
        score += cost_score
        
        # 4. 风险控制 (20分)
        # 基于盈亏分布评估
        if 'pnl' in df.columns:
            win_rate = (df['pnl'] > 0).sum() / len(df)
            risk_score = win_rate * 20
        else:
            risk_score = 10  # 默认中等分数
        score += risk_score
        
        # 额外扣分项
        # 如果未成交率过高，扣分
        unfilled_rate = (buy_trades['fill_ratio'] < 0.01).sum() / len(buy_trades)
        if unfilled_rate > 0.3:
            score -= 10
        
        # 限制在0-100之间
        return max(0, min(100, score))
    
    def _calc_holding_days(self, symbol: str, df: pd.DataFrame) -> float:
        """计算平均持仓天数"""
        # 简化实现：假设买卖配对
        symbol_df = df[df['symbol'] == symbol].sort_values('date')
        
        holding_days = []
        buy_date = None
        
        for _, row in symbol_df.iterrows():
            if row['side'] == 'buy':
                buy_date = pd.to_datetime(row['date'])
            elif row['side'] == 'sell' and buy_date is not None:
                sell_date = pd.to_datetime(row['date'])
                days = (sell_date - buy_date).days
                holding_days.append(days)
                buy_date = None
        
        return np.mean(holding_days) if holding_days else 0
    
    def _calc_symbol_win_rate(self, symbol: str, df: pd.DataFrame) -> float:
        """计算个股胜率"""
        symbol_df = df[df['symbol'] == symbol]
        if 'pnl' not in symbol_df.columns:
            return 0.5
        
        trades_with_pnl = symbol_df.dropna(subset=['pnl'])
        if len(trades_with_pnl) == 0:
            return 0.5
            
        return (trades_with_pnl['pnl'] > 0).sum() / len(trades_with_pnl)
    
    def _get_trade_summary(self) -> Dict:
        """获取交易汇总"""
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        
        # 按日期汇总
        df['date'] = pd.to_datetime(df['date'])
        daily_summary = df.groupby(df['date'].dt.date).agg({
            'symbol': 'count',  # 交易次数
            'fill_ratio': 'mean',  # 平均成交比例
            'pnl': 'sum' if 'pnl' in df else lambda x: 0  # 当日盈亏
        }).rename(columns={'symbol': 'trades'})
        
        return {
            'daily_summary': daily_summary.to_dict(),
            'total_days': len(daily_summary),
            'avg_daily_trades': daily_summary['trades'].mean(),
            'best_day': daily_summary.idxmax()['pnl'] if 'pnl' in daily_summary else None,
            'worst_day': daily_summary.idxmin()['pnl'] if 'pnl' in daily_summary else None
        }
    
    def _empty_metrics(self) -> Dict:
        """返回空指标"""
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_fill_ratio': 1.0,
            'unfilled_rate': 0,
            'execution_score': 50.0,
            'symbol_stats': []
        }
    
    def generate_execution_report(self) -> str:
        """生成可执行性报告"""
        metrics = self.calculate_all_metrics()
        
        report = []
        report.append("="*60)
        report.append("策略可执行性评估报告")
        report.append("="*60)
        
        # 执行评分
        score = metrics['execution_score']
        grade = self._get_grade(score)
        report.append(f"\n【执行评分】 {score:.1f}/100 - {grade}")
        
        # 成交统计
        report.append("\n【成交统计】")
        report.append(f"  平均成交比例: {metrics['avg_fill_ratio']:.1%}")
        report.append(f"  未成交率: {metrics['unfilled_rate']:.1%}")
        report.append(f"  部分成交率: {metrics.get('partial_fill_rate', 0):.1%}")
        report.append(f"  完全成交率: {metrics.get('full_fill_rate', 0):.1%}")
        
        # 分类统计
        report.append("\n【分类分析】")
        
        # 按股票类型
        for stype in ['main', 'chinext', 'st']:
            key_fill = f'{stype}_avg_fill'
            key_trades = f'{stype}_trades'
            if key_fill in metrics:
                report.append(f"  {stype.upper()}板块: 平均成交{metrics[key_fill]:.1%}, 交易{metrics.get(key_trades, 0)}次")
        
        # 按封板强度
        report.append("\n【封板强度分析】")
        for strength in ['strong', 'medium', 'weak']:
            key_fill = f'{strength}_avg_fill'
            key_trades = f'{strength}_trades'
            if key_fill in metrics:
                strength_cn = {'strong': '强势', 'medium': '中等', 'weak': '弱势'}[strength]
                report.append(f"  {strength_cn}封板: 平均成交{metrics[key_fill]:.1%}, 交易{metrics.get(key_trades, 0)}次")
        
        # 建议
        report.append("\n【优化建议】")
        suggestions = self._get_suggestions(metrics)
        for i, suggestion in enumerate(suggestions, 1):
            report.append(f"  {i}. {suggestion}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def _get_grade(self, score: float) -> str:
        """根据评分返回等级"""
        if score >= 90:
            return "优秀 ⭐⭐⭐⭐⭐"
        elif score >= 80:
            return "良好 ⭐⭐⭐⭐"
        elif score >= 70:
            return "中等 ⭐⭐⭐"
        elif score >= 60:
            return "及格 ⭐⭐"
        else:
            return "较差 ⭐"
    
    def _get_suggestions(self, metrics: Dict) -> List[str]:
        """根据指标生成优化建议"""
        suggestions = []
        
        # 成交率建议
        if metrics['avg_fill_ratio'] < 0.5:
            suggestions.append("成交率偏低，建议降低封板强度要求或调整下单时机")
        if metrics['unfilled_rate'] > 0.2:
            suggestions.append("未成交率过高，建议优化选股条件或使用分批建仓")
        
        # 稳定性建议
        if metrics.get('fill_ratio_std', 0) > 0.3:
            suggestions.append("成交波动较大，建议增加对市场流动性的考虑")
        
        # 集中度建议
        if metrics.get('symbol_concentration', 0) > 0.3:
            suggestions.append("交易过于集中在少数股票，建议分散投资")
        
        # 类型建议
        if metrics.get('chinext_avg_fill', 1) < 0.3:
            suggestions.append("创业板成交困难，建议减少20%涨停板的参与")
        
        if not suggestions:
            suggestions.append("策略执行良好，继续保持当前配置")
        
        return suggestions


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    calculator = EnhancedMetricsCalculator()
    
    # 添加一些测试交易
    import random
    from datetime import datetime, timedelta
    
    symbols = ['000001', '000002', '300001', '688001', 'ST0001']
    start_date = datetime(2024, 1, 1)
    
    for i in range(50):
        date = start_date + timedelta(days=i)
        symbol = random.choice(symbols)
        
        # 判断股票类型
        if 'ST' in symbol:
            stock_type = 'st'
        elif symbol.startswith('3') or symbol.startswith('688'):
            stock_type = 'chinext'
        else:
            stock_type = 'main'
        
        # 买入交易
        if random.random() > 0.3:  # 70%概率买入
            trade = TradeDetail(
                symbol=symbol,
                date=date.strftime('%Y-%m-%d'),
                side='buy',
                planned_qty=1000,
                filled_qty=int(1000 * random.uniform(0, 1)),
                fill_ratio=random.uniform(0, 1),
                price=10 + random.uniform(-2, 2),
                commission=5,
                slippage=0.001,
                stock_type=stock_type,
                limit_strength=random.choice(['strong', 'medium', 'weak'])
            )
            calculator.add_trade(trade)
        
        # 卖出交易
        if i > 5 and random.random() > 0.5:  # 50%概率卖出
            trade = TradeDetail(
                symbol=symbol,
                date=date.strftime('%Y-%m-%d'),
                side='sell',
                planned_qty=1000,
                filled_qty=1000,
                fill_ratio=1.0,
                price=10 + random.uniform(-2, 3),
                commission=5,
                slippage=0.001,
                pnl=random.uniform(-100, 200),
                stock_type=stock_type,
                limit_strength='medium'
            )
            calculator.add_trade(trade)
    
    # 计算指标
    metrics = calculator.calculate_all_metrics()
    
    # 打印结果
    print("\n基础指标:")
    print(f"  总交易数: {metrics['total_trades']}")
    print(f"  胜率: {metrics['win_rate']:.2%}")
    print(f"  盈亏比: {metrics['profit_factor']:.2f}")
    
    print("\n成交指标:")
    print(f"  平均成交比例: {metrics['avg_fill_ratio']:.2%}")
    print(f"  未成交率: {metrics['unfilled_rate']:.2%}")
    
    print("\n执行评分:")
    print(f"  可执行性得分: {metrics['execution_score']:.1f}/100")
    
    # 生成报告
    report = calculator.generate_execution_report()
    print(report)