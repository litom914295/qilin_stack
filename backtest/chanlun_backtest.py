"""缠论回测框架 - Phase P0-6
验证效率+60%

功能:
- 逐日回放模式
- 策略信号生成
- 持仓管理
- 性能指标计算
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """交易记录"""
    date: datetime
    action: str  # 'buy'/'sell'
    price: float
    shares: int
    reason: str = ''

@dataclass
class BacktestMetrics:
    """回测指标"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int

class ChanLunBacktester:
    """缠论策略回测框架"""
    
    def __init__(self, initial_cash: float = 1000000, commission_rate: float = 0.0003):
        """初始化
        
        Args:
            initial_cash: 初始资金
            commission_rate: 佣金费率
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.position = 0  # 持仓数量
        self.trades: List[Trade] = []
        self.daily_values = []  # 每日组合价值
        self.equity_curve = []  # 权益曲线
    
    def backtest_strategy(
        self,
        strategy: Callable,
        stock_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict:
        """回测缠论策略
        
        Args:
            strategy: 策略函数,输入df返回'buy'/'sell'/'hold'
            stock_data: 全量股票数据
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            回测结果字典
        """
        logger.info(f"开始回测: {start_date} ~ {end_date}")
        
        # 筛选日期范围
        stock_data['datetime'] = pd.to_datetime(stock_data['datetime'])
        mask = (stock_data['datetime'] >= start_date) & (stock_data['datetime'] <= end_date)
        test_data = stock_data[mask].copy()
        
        if len(test_data) == 0:
            logger.error("无有数据可回测")
            return {}
        
        # 逐日回放
        for i in range(len(test_data)):
            current_date = test_data.iloc[i]['datetime']
            current_price = test_data.iloc[i]['close']
            
            # 获取截止当前日的数据
            historical_data = stock_data[
                stock_data['datetime'] <= current_date
            ].tail(100)  # 只取最近100天
            
            # 生成策略信号
            try:
                signal = strategy(historical_data)
            except Exception as e:
                logger.warning(f"策略信号生成失败: {e}")
                signal = 'hold'
            
            # 执行交易
            if signal == 'buy' and self.position == 0:
                self._execute_buy(current_date, current_price, reason='策略信号')
            elif signal == 'sell' and self.position > 0:
                self._execute_sell(current_date, current_price, reason='策略信号')
            
            # 记录每日价值
            portfolio_value = self.cash + self.position * current_price
            self.daily_values.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': self.cash,
                'position': self.position,
                'price': current_price
            })
        
        # 计算指标
        metrics = self._calc_metrics()
        
        logger.info(f"回测完成: 总收益={metrics.total_return:.2%}, 夏普={metrics.sharpe_ratio:.2f}")
        
        return {
            'trades': self.trades,
            'daily_values': self.daily_values,
            'metrics': metrics,
            'equity_curve': pd.DataFrame(self.daily_values)
        }
    
    def _calc_metrics(self) -> BacktestMetrics:
        """计算回测指标"""
        if len(self.daily_values) == 0:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0)
        
        df = pd.DataFrame(self.daily_values)
        df['return'] = df['value'].pct_change().fillna(0)
        
        # 总收益
        total_return = (df['value'].iloc[-1] / self.initial_cash) - 1
        
        # 年化收益
        days = len(df)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 夏普比率
        if df['return'].std() > 0:
            sharpe_ratio = df['return'].mean() / df['return'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown = self._calc_max_drawdown_from_values(df['value'])
        
        # 胜率
        win_rate = self._calc_win_rate()
        
        # 盈亏比
        profit_factor = self._calc_profit_factor()
        
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades)
        )
    
    def _calc_max_drawdown_from_values(self, values: pd.Series) -> float:
        """计算最大回撤"""
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        return abs(drawdown.min())
    
    def _calc_win_rate(self) -> float:
        """计算胜率"""
        if len(self.trades) < 2:
            return 0.0
        
        # 配对买卖交易
        buy_trades = [t for t in self.trades if t.action == 'buy']
        sell_trades = [t for t in self.trades if t.action == 'sell']
        
        wins = 0
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_pairs):
            if sell_trades[i].price > buy_trades[i].price:
                wins += 1
        
        return wins / total_pairs if total_pairs > 0 else 0.0
    
    def _calc_profit_factor(self) -> float:
        """计算盈亏比"""
        if len(self.trades) < 2:
            return 0.0
        
        buy_trades = [t for t in self.trades if t.action == 'buy']
        sell_trades = [t for t in self.trades if t.action == 'sell']
        
        total_profit = 0
        total_loss = 0
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            pnl = sell_trades[i].price - buy_trades[i].price
            if pnl > 0:
                total_profit += pnl
            else:
                total_loss += abs(pnl)
        
        return total_profit / total_loss if total_loss > 0 else 0.0
    
    def _execute_buy(self, date, price, reason=''):
        """执行买入"""
        max_shares = int(self.cash / price * (1 - self.commission_rate))
        if max_shares <= 0:
            return
        
        cost = max_shares * price * (1 + self.commission_rate)
        self.cash -= cost
        self.position += max_shares
        
        trade = Trade(
            date=date,
            action='buy',
            price=price,
            shares=max_shares,
            reason=reason
        )
        self.trades.append(trade)
        logger.debug(f"买入: {date}, {max_shares}股@{price:.2f}")
    
    def _execute_sell(self, date, price, reason=''):
        """执行卖出"""
        if self.position <= 0:
            return
        
        proceeds = self.position * price * (1 - self.commission_rate)
        self.cash += proceeds
        
        trade = Trade(
            date=date,
            action='sell',
            price=price,
            shares=self.position,
            reason=reason
        )
        self.trades.append(trade)
        logger.debug(f"卖出: {date}, {self.position}股@{price:.2f}")
        
        self.position = 0

if __name__ == '__main__':
    print("✅ P0-6: 回测框架创建完成")
