"""
回测对比模块
对比竞价买入策略 vs 传统排板买入策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    total_return: float
    avg_return: float
    max_return: float
    min_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_hold_days: float
    total_profit: float
    total_loss: float


class AuctionVsLimitUpBacktest:
    """
    竞价买入 vs 排板买入回测对比
    
    核心对比维度：
    1. 竞价买入：T日涨停，T+1集合竞价买入，T+2卖出
    2. 排板买入：T日涨停排板买入，T+1卖出
    """
    
    def __init__(self, initial_capital: float = 1000000):
        """
        初始化回测
        
        Parameters:
        -----------
        initial_capital: float
            初始资金
        """
        self.initial_capital = initial_capital
        self.results = {}
    
    def run_backtest(self,
                     data: pd.DataFrame,
                     start_date: str,
                     end_date: str) -> Dict[str, BacktestResult]:
        """
        运行回测对比
        
        Parameters:
        -----------
        data: DataFrame
            历史数据，包含涨停信息、竞价数据、价格数据
        start_date: str
            开始日期
        end_date: str
            结束日期
            
        Returns:
        --------
        Dict[str, BacktestResult]: 两种策略的回测结果
        """
        print(f"\n{'='*60}")
        print(f"回测对比: 竞价买入 vs 排板买入")
        print(f"{'='*60}")
        print(f"回测期间: {start_date} ~ {end_date}")
        print(f"初始资金: ¥{self.initial_capital:,.0f}")
        print(f"{'='*60}\n")
        
        # 策略1: 竞价买入策略
        print("正在回测策略1: 竞价买入策略...")
        auction_result = self._run_auction_strategy(data, start_date, end_date)
        
        # 策略2: 排板买入策略
        print("正在回测策略2: 排板买入策略...")
        limitup_result = self._run_limitup_strategy(data, start_date, end_date)
        
        self.results = {
            'auction': auction_result,
            'limitup': limitup_result
        }
        
        # 打印对比结果
        self._print_comparison()
        
        return self.results
    
    def _run_auction_strategy(self,
                              data: pd.DataFrame,
                              start_date: str,
                              end_date: str) -> BacktestResult:
        """
        运行竞价买入策略回测
        
        策略逻辑：
        T日: 涨停股票筛选
        T+1日: 集合竞价买入（9:25开盘价）
        T+2日: 开盘卖出
        """
        trades = []
        
        # 筛选回测期间的涨停数据
        mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        backtest_data = data[mask].copy()
        
        # 按日期分组
        for date, group in backtest_data.groupby('date'):
            # T日涨停股票
            limitup_stocks = group[group['is_limitup'] == True]
            
            if len(limitup_stocks) == 0:
                continue
            
            # 模拟T+1竞价买入
            for idx, stock in limitup_stocks.iterrows():
                symbol = stock['symbol']
                
                # T+1竞价买入价（开盘价）
                t1_open_price = stock.get('t1_open_price', stock['close'] * 1.03)
                
                # 竞价强度筛选
                auction_strength = stock.get('auction_strength', np.random.uniform(0, 1))
                if auction_strength < 0.6:  # 竞价强度阈值
                    continue
                
                # T+1收盘价
                t1_close_price = stock.get('t1_close_price', t1_open_price * np.random.uniform(0.95, 1.10))
                
                # T+2开盘卖出价
                t2_open_price = stock.get('t2_open_price', t1_close_price * np.random.uniform(0.97, 1.08))
                
                # 计算收益
                buy_price = t1_open_price
                sell_price = t2_open_price
                return_rate = (sell_price / buy_price - 1) * 100
                
                # 记录交易
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': return_rate,
                    'hold_days': 2,
                    'profit': (sell_price - buy_price) * 1000  # 假设1000股
                })
        
        # 计算回测指标
        return self._calculate_metrics(trades, "竞价买入策略")
    
    def _run_limitup_strategy(self,
                             data: pd.DataFrame,
                             start_date: str,
                             end_date: str) -> BacktestResult:
        """
        运行排板买入策略回测
        
        策略逻辑：
        T日: 涨停排板买入（涨停价）
        T+1日: 开盘卖出
        """
        trades = []
        
        # 筛选回测期间的涨停数据
        mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        backtest_data = data[mask].copy()
        
        # 按日期分组
        for date, group in backtest_data.groupby('date'):
            # T日涨停股票
            limitup_stocks = group[group['is_limitup'] == True]
            
            if len(limitup_stocks) == 0:
                continue
            
            # 模拟T日排板买入
            for idx, stock in limitup_stocks.iterrows():
                symbol = stock['symbol']
                
                # T日涨停价买入
                limitup_price = stock['close']
                
                # 排板成功率（模拟）
                success_rate = stock.get('seal_strength', 5) / 10  # 封单强度影响成功率
                if np.random.random() > success_rate:
                    continue  # 未能排板成功
                
                # T+1开盘卖出价
                t1_open_price = stock.get('t1_open_price', limitup_price * np.random.uniform(0.92, 1.08))
                
                # 计算收益
                buy_price = limitup_price
                sell_price = t1_open_price
                return_rate = (sell_price / buy_price - 1) * 100
                
                # 记录交易
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': return_rate,
                    'hold_days': 1,
                    'profit': (sell_price - buy_price) * 1000
                })
        
        # 计算回测指标
        return self._calculate_metrics(trades, "排板买入策略")
    
    def _calculate_metrics(self,
                          trades: List[Dict],
                          strategy_name: str) -> BacktestResult:
        """计算回测指标"""
        if len(trades) == 0:
            return BacktestResult(
                strategy_name=strategy_name,
                total_trades=0,
                win_trades=0,
                loss_trades=0,
                win_rate=0,
                total_return=0,
                avg_return=0,
                max_return=0,
                min_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                profit_factor=0,
                avg_hold_days=0,
                total_profit=0,
                total_loss=0
            )
        
        df = pd.DataFrame(trades)
        
        # 基础统计
        total_trades = len(df)
        win_trades = len(df[df['return'] > 0])
        loss_trades = len(df[df['return'] <= 0])
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        
        # 收益统计
        returns = df['return'].values
        total_return = returns.sum()
        avg_return = returns.mean()
        max_return = returns.max()
        min_return = returns.min()
        
        # 夏普比率
        sharpe_ratio = (avg_return / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + df['return'] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # 盈亏比
        total_profit = df[df['profit'] > 0]['profit'].sum()
        total_loss = abs(df[df['profit'] <= 0]['profit'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 持仓天数
        avg_hold_days = df['hold_days'].mean()
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_trades=total_trades,
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=win_rate,
            total_return=total_return,
            avg_return=avg_return,
            max_return=max_return,
            min_return=min_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_hold_days=avg_hold_days,
            total_profit=total_profit,
            total_loss=total_loss
        )
    
    def _print_comparison(self):
        """打印对比结果"""
        auction = self.results['auction']
        limitup = self.results['limitup']
        
        print(f"\n{'='*80}")
        print(f"{'回测对比结果':^76}")
        print(f"{'='*80}")
        
        # 表头
        print(f"{'指标':<20} {'竞价买入':>20} {'排板买入':>20} {'优势':>15}")
        print(f"{'-'*80}")
        
        # 对比数据
        comparisons = [
            ("交易次数", auction.total_trades, limitup.total_trades, "次"),
            ("胜率", f"{auction.win_rate:.2f}%", f"{limitup.win_rate:.2f}%", ""),
            ("平均收益率", f"{auction.avg_return:+.2f}%", f"{limitup.avg_return:+.2f}%", ""),
            ("累计收益率", f"{auction.total_return:+.2f}%", f"{limitup.total_return:+.2f}%", ""),
            ("最大单笔收益", f"{auction.max_return:+.2f}%", f"{limitup.max_return:+.2f}%", ""),
            ("最大单笔亏损", f"{auction.min_return:+.2f}%", f"{limitup.min_return:+.2f}%", ""),
            ("夏普比率", f"{auction.sharpe_ratio:.2f}", f"{limitup.sharpe_ratio:.2f}", ""),
            ("最大回撤", f"{auction.max_drawdown:.2f}%", f"{limitup.max_drawdown:.2f}%", ""),
            ("盈亏比", f"{auction.profit_factor:.2f}", f"{limitup.profit_factor:.2f}", ""),
            ("平均持仓天数", f"{auction.avg_hold_days:.1f}", f"{limitup.avg_hold_days:.1f}", "天"),
            ("总盈利", f"¥{auction.total_profit:,.0f}", f"¥{limitup.total_profit:,.0f}", ""),
            ("总亏损", f"¥{auction.total_loss:,.0f}", f"¥{limitup.total_loss:,.0f}", ""),
        ]
        
        for metric, val1, val2, unit in comparisons:
            # 判断优势
            if isinstance(val1, str) and '%' in val1:
                num1 = float(val1.replace('%', '').replace('+', ''))
                num2 = float(val2.replace('%', '').replace('+', ''))
                winner = "竞价 ✓" if num1 > num2 else "排板 ✓" if num2 > num1 else "平"
            elif isinstance(val1, str) and '¥' in val1:
                num1 = float(val1.replace('¥', '').replace(',', ''))
                num2 = float(val2.replace('¥', '').replace(',', ''))
                if "亏损" in metric:
                    winner = "竞价 ✓" if num1 < num2 else "排板 ✓" if num2 < num1 else "平"
                else:
                    winner = "竞价 ✓" if num1 > num2 else "排板 ✓" if num2 > num1 else "平"
            else:
                try:
                    num1 = float(str(val1).replace('次', '').replace('天', ''))
                    num2 = float(str(val2).replace('次', '').replace('天', ''))
                    if metric in ["最大回撤", "平均持仓天数"]:
                        winner = "竞价 ✓" if num1 < num2 else "排板 ✓" if num2 < num1 else "平"
                    else:
                        winner = "竞价 ✓" if num1 > num2 else "排板 ✓" if num2 > num1 else "平"
                except:
                    winner = "-"
            
            print(f"{metric:<20} {str(val1):>20} {str(val2):>20} {winner:>15}")
        
        print(f"{'='*80}")
        
        # 综合评价
        print(f"\n{'综合评价':^76}")
        print(f"{'-'*80}")
        
        # 计算综合得分
        auction_score = 0
        limitup_score = 0
        
        if auction.win_rate > limitup.win_rate:
            auction_score += 1
        elif limitup.win_rate > auction.win_rate:
            limitup_score += 1
        
        if auction.avg_return > limitup.avg_return:
            auction_score += 2
        elif limitup.avg_return > auction.avg_return:
            limitup_score += 2
        
        if auction.sharpe_ratio > limitup.sharpe_ratio:
            auction_score += 1
        elif limitup.sharpe_ratio > auction.sharpe_ratio:
            limitup_score += 1
        
        if auction.max_drawdown > limitup.max_drawdown:  # 回撤小更好
            limitup_score += 1
        elif limitup.max_drawdown > auction.max_drawdown:
            auction_score += 1
        
        print(f"竞价买入策略得分: {auction_score}")
        print(f"排板买入策略得分: {limitup_score}")
        
        if auction_score > limitup_score:
            print(f"\n✅ 推荐策略: 竞价买入策略")
            print(f"   优势: 更高的胜率和平均收益，风险可控")
        elif limitup_score > auction_score:
            print(f"\n✅ 推荐策略: 排板买入策略")
            print(f"   优势: 更短的持仓周期，资金周转效率高")
        else:
            print(f"\n⚖️  两种策略表现相当，可根据市场环境选择")
        
        print(f"{'='*80}\n")
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """绘制对比图表"""
        if not self.results:
            print("请先运行回测！")
            return
        
        auction = self.results['auction']
        limitup = self.results['limitup']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('竞价买入 vs 排板买入策略对比', fontsize=16, fontweight='bold')
        
        # 1. 胜率对比
        ax1 = axes[0, 0]
        strategies = ['竞价买入', '排板买入']
        win_rates = [auction.win_rate, limitup.win_rate]
        bars1 = ax1.bar(strategies, win_rates, color=['#2E86AB', '#A23B72'])
        ax1.set_ylabel('胜率 (%)')
        ax1.set_title('胜率对比')
        ax1.set_ylim(0, 100)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. 平均收益率对比
        ax2 = axes[0, 1]
        avg_returns = [auction.avg_return, limitup.avg_return]
        bars2 = ax2.bar(strategies, avg_returns, color=['#2E86AB', '#A23B72'])
        ax2.set_ylabel('平均收益率 (%)')
        ax2.set_title('平均收益率对比')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.2f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. 夏普比率对比
        ax3 = axes[0, 2]
        sharpe_ratios = [auction.sharpe_ratio, limitup.sharpe_ratio]
        bars3 = ax3.bar(strategies, sharpe_ratios, color=['#2E86AB', '#A23B72'])
        ax3.set_ylabel('夏普比率')
        ax3.set_title('夏普比率对比')
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 4. 最大回撤对比
        ax4 = axes[1, 0]
        max_drawdowns = [auction.max_drawdown, limitup.max_drawdown]
        bars4 = ax4.bar(strategies, max_drawdowns, color=['#2E86AB', '#A23B72'])
        ax4.set_ylabel('最大回撤 (%)')
        ax4.set_title('最大回撤对比')
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='top')
        
        # 5. 盈亏比对比
        ax5 = axes[1, 1]
        profit_factors = [auction.profit_factor, limitup.profit_factor]
        bars5 = ax5.bar(strategies, profit_factors, color=['#2E86AB', '#A23B72'])
        ax5.set_ylabel('盈亏比')
        ax5.set_title('盈亏比对比')
        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 6. 持仓天数对比
        ax6 = axes[1, 2]
        hold_days = [auction.avg_hold_days, limitup.avg_hold_days]
        bars6 = ax6.bar(strategies, hold_days, color=['#2E86AB', '#A23B72'])
        ax6.set_ylabel('平均持仓天数')
        ax6.set_title('持仓周期对比')
        for bar in bars6:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}天', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def export_results(self, output_path: str):
        """导出回测结果到CSV"""
        if not self.results:
            print("请先运行回测！")
            return
        
        auction = self.results['auction']
        limitup = self.results['limitup']
        
        comparison_df = pd.DataFrame({
            '指标': [
                '交易次数', '盈利次数', '亏损次数', '胜率(%)',
                '累计收益率(%)', '平均收益率(%)', '最大单笔收益(%)', '最大单笔亏损(%)',
                '夏普比率', '最大回撤(%)', '盈亏比', '平均持仓天数',
                '总盈利(¥)', '总亏损(¥)'
            ],
            '竞价买入': [
                auction.total_trades, auction.win_trades, auction.loss_trades, f"{auction.win_rate:.2f}",
                f"{auction.total_return:.2f}", f"{auction.avg_return:.2f}", f"{auction.max_return:.2f}", f"{auction.min_return:.2f}",
                f"{auction.sharpe_ratio:.2f}", f"{auction.max_drawdown:.2f}", f"{auction.profit_factor:.2f}", f"{auction.avg_hold_days:.1f}",
                f"{auction.total_profit:.0f}", f"{auction.total_loss:.0f}"
            ],
            '排板买入': [
                limitup.total_trades, limitup.win_trades, limitup.loss_trades, f"{limitup.win_rate:.2f}",
                f"{limitup.total_return:.2f}", f"{limitup.avg_return:.2f}", f"{limitup.max_return:.2f}", f"{limitup.min_return:.2f}",
                f"{limitup.sharpe_ratio:.2f}", f"{limitup.max_drawdown:.2f}", f"{limitup.profit_factor:.2f}", f"{limitup.avg_hold_days:.1f}",
                f"{limitup.total_profit:.0f}", f"{limitup.total_loss:.0f}"
            ]
        })
        
        comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"回测结果已导出到: {output_path}")


def create_sample_backtest_data(n_days: int = 100, n_stocks_per_day: int = 30) -> pd.DataFrame:
    """
    生成模拟回测数据
    
    Parameters:
    -----------
    n_days: int
        回测天数
    n_stocks_per_day: int
        每天涨停股票数
        
    Returns:
    --------
    DataFrame: 模拟的历史数据
    """
    np.random.seed(42)
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        
        for stock_id in range(n_stocks_per_day):
            symbol = f"{stock_id:06d}.SZ"
            
            # T日数据
            close_price = np.random.uniform(10, 100)
            seal_strength = np.random.uniform(1, 10)
            auction_strength = np.random.uniform(0.3, 0.95)
            
            # T+1数据
            t1_open_price = close_price * np.random.uniform(0.98, 1.08)
            t1_close_price = t1_open_price * np.random.uniform(0.95, 1.10)
            
            # T+2数据
            t2_open_price = t1_close_price * np.random.uniform(0.97, 1.08)
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'close': close_price,
                'is_limitup': True,
                'seal_strength': seal_strength,
                'auction_strength': auction_strength,
                't1_open_price': t1_open_price,
                't1_close_price': t1_close_price,
                't2_open_price': t2_open_price,
            })
    
    return pd.DataFrame(data)


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    print("生成模拟回测数据...")
    data = create_sample_backtest_data(n_days=100, n_stocks_per_day=30)
    
    # 创建回测实例
    backtest = AuctionVsLimitUpBacktest(initial_capital=1000000)
    
    # 运行回测
    results = backtest.run_backtest(
        data=data,
        start_date='2024-01-01',
        end_date='2024-04-10'
    )
    
    # 绘制对比图表
    print("\n绘制对比图表...")
    backtest.plot_comparison(save_path='backtest_comparison.png')
    
    # 导出结果
    backtest.export_results('backtest_results.csv')
    
    print("\n✅ 回测完成！")
