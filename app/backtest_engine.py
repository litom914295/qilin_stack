"""
麒麟量化系统 - 回测引擎
参考advanced-ak-pack实现,计算Sharpe、最大回撤、胜率等指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    initial_capital: float = 100000
    top_k: int = 5
    position_per_stock: float = 0.2
    commission_rate: float = 0.0003
    slippage: float = 0.01
    stop_loss: float = -0.03
    take_profit: float = 0.10


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.capital = config.initial_capital
        self.positions = {}
        self.cash = config.initial_capital
        
        self.daily_returns = []
        self.daily_values = []
        self.trade_log = []
        
        logger.info("回测引擎初始化完成")
        logger.info(f"初始资金: {self.capital:.2f}")
        logger.info(f"TopK: {self.config.top_k}")
    
    def run_backtest(
        self,
        signals_df: pd.DataFrame
    ) -> Dict:
        """
        运行回测
        
        Args:
            signals_df: 信号DataFrame [date, symbol, score, ...]
            
        Returns:
            回测结果字典
        """
        logger.info("=" * 60)
        logger.info("开始回测...")
        logger.info("=" * 60)
        
        # 按日期分组
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        dates = sorted(signals_df['date'].unique())
        
        for date in dates:
            daily_signals = signals_df[signals_df['date'] == date]
            
            # 选择TopK
            top_k_signals = daily_signals.nlargest(self.config.top_k, 'score')
            
            # 执行交易
            self._execute_daily_trades(date, top_k_signals)
        
        # 计算指标
        metrics = self._calculate_metrics()
        
        logger.info("回测完成!")
        
        return metrics
    
    def _execute_daily_trades(
        self,
        date: pd.Timestamp,
        signals: pd.DataFrame
    ):
        """执行某日的交易"""
        # 清仓前日持仓(首板→二板策略,隔日卖出)
        if self.positions:
            for symbol, position in list(self.positions.items()):
                sell_price = position['buy_price'] * (1 + np.random.uniform(-0.05, 0.15))  # 模拟次日价格
                
                # 计算收益
                profit = (sell_price - position['buy_price']) * position['shares']
                profit -= sell_price * position['shares'] * self.config.commission_rate
                
                self.cash += sell_price * position['shares'] - sell_price * position['shares'] * self.config.commission_rate
                
                self.trade_log.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'price': sell_price,
                    'shares': position['shares'],
                    'profit': profit,
                    'profit_rate': profit / (position['buy_price'] * position['shares'])
                })
                
                del self.positions[symbol]
        
        # 买入新信号
        if not signals.empty:
            position_value = self.cash / len(signals)
            
            for _, signal in signals.iterrows():
                symbol = signal['symbol']
                # 模拟买入价(开盘价+滑点)
                buy_price = signal.get('price', 10.0) * (1 + self.config.slippage)
                
                shares = int(position_value / buy_price / 100) * 100  # 100股整数倍
                
                if shares >= 100 and self.cash >= buy_price * shares:
                    cost = buy_price * shares * (1 + self.config.commission_rate)
                    
                    self.cash -= cost
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'buy_price': buy_price,
                        'shares': shares,
                        'buy_date': date
                    }
                    
                    self.trade_log.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'buy',
                        'price': buy_price,
                        'shares': shares,
                        'cost': cost
                    })
        
        # 记录每日净值
        total_value = self.cash + sum(
            p['buy_price'] * p['shares'] 
            for p in self.positions.values()
        )
        
        self.daily_values.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'position_value': total_value - self.cash
        })
        
        # 记录每日收益率
        if len(self.daily_values) > 1:
            prev_value = self.daily_values[-2]['total_value']
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def _calculate_metrics(self) -> Dict:
        """计算回测指标"""
        df_values = pd.DataFrame(self.daily_values)
        df_trades = pd.DataFrame(self.trade_log)
        
        # 基础指标
        final_value = df_values['total_value'].iloc[-1] if not df_values.empty else self.config.initial_capital
        total_return = (final_value - self.config.initial_capital) / self.config.initial_capital
        
        # 日收益率统计
        daily_returns = np.array(self.daily_returns)
        avg_daily_return = daily_returns.mean() if len(daily_returns) > 0 else 0
        std_daily_return = daily_returns.std() if len(daily_returns) > 0 else 0
        
        # Sharpe比率 (年化)
        sharpe_ratio = (avg_daily_return / (std_daily_return + 1e-9)) * np.sqrt(240) if std_daily_return > 0 else 0
        
        # 最大回撤
        cumulative_values = df_values['total_value'].values
        cummax = np.maximum.accumulate(cumulative_values)
        drawdowns = (cumulative_values - cummax) / cummax
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
        
        # 交易统计
        sell_trades = df_trades[df_trades['action'] == 'sell']
        
        if not sell_trades.empty:
            win_trades = sell_trades[sell_trades['profit'] > 0]
            win_rate = len(win_trades) / len(sell_trades)
            avg_profit = sell_trades['profit'].mean()
            avg_profit_rate = sell_trades['profit_rate'].mean()
        else:
            win_rate = 0
            avg_profit = 0
            avg_profit_rate = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': total_return * 240 / len(self.daily_returns) if self.daily_returns else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_daily_return': avg_daily_return,
            'std_daily_return': std_daily_return,
            'total_trades': len(sell_trades),
            'avg_profit': avg_profit,
            'avg_profit_rate': avg_profit_rate,
            'final_value': final_value
        }
        
        # 打印报告
        self._print_report(metrics)
        
        return metrics
    
    def _print_report(self, metrics: Dict):
        """打印回测报告"""
        logger.info("\n" + "=" * 60)
        logger.info("回测报告")
        logger.info("=" * 60)
        logger.info(f"初始资金: {self.config.initial_capital:.2f}")
        logger.info(f"最终资金: {metrics['final_value']:.2f}")
        logger.info(f"总收益率: {metrics['total_return']:.2%}")
        logger.info(f"年化收益率: {metrics['annualized_return']:.2%}")
        logger.info(f"Sharpe比率: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"胜率: {metrics['win_rate']:.2%}")
        logger.info(f"平均日收益率: {metrics['avg_daily_return']:.4%}")
        logger.info(f"日收益波动率: {metrics['std_daily_return']:.4%}")
        logger.info(f"总交易次数: {metrics['total_trades']}")
        logger.info(f"平均单笔收益: {metrics['avg_profit']:.2f}")
        logger.info(f"平均收益率: {metrics['avg_profit_rate']:.2%}")
        logger.info("=" * 60)
    
    def save_results(self, output_dir: str = "reports/backtest"):
        """保存回测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存净值曲线
        df_values = pd.DataFrame(self.daily_values)
        df_values.to_csv(output_path / f"equity_curve_{timestamp}.csv", index=False)
        
        # 保存交易记录
        df_trades = pd.DataFrame(self.trade_log)
        df_trades.to_csv(output_path / f"trade_log_{timestamp}.csv", index=False)
        
        # 保存指标
        metrics = self._calculate_metrics()
        with open(output_path / f"metrics_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"回测结果已保存到: {output_path}")
        
        return str(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 生成模拟信号数据
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    signals = []
    
    for date in dates:
        for i in range(20):  # 每日20个候选
            signals.append({
                'date': date,
                'symbol': f"{np.random.randint(0, 999999):06d}",
                'score': np.random.uniform(50, 100),
                'price': np.random.uniform(10, 100)
            })
    
    signals_df = pd.DataFrame(signals)
    
    # 运行回测
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-06-30',
        initial_capital=100000,
        top_k=5
    )
    
    engine = BacktestEngine(config)
    metrics = engine.run_backtest(signals_df)
    
    # 保存结果
    engine.save_results()
    
    print("\n回测完成!")
    print(f"总收益率: {metrics['total_return']:.2%}")
    print(f"Sharpe比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
