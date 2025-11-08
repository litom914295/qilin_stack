# -*- coding: utf-8 -*-
"""
写实回测引擎 - 模拟涨停板真实成交环境

功能：
1. 涨停排队模拟器
2. 成交概率计算
3. 真实滑点模型
4. 交易成本核算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderExecution:
    """订单执行详情"""
    symbol: str
    order_time: str
    execute_time: str
    order_price: float
    execute_price: float
    order_volume: int
    execute_volume: int
    is_executed: bool
    queue_position: int  # 排队位置
    seal_amount: float  # 封单金额（万元）
    execution_probability: float  # 成交概率
    slippage: float  # 滑点
    commission: float  # 手续费
    stamp_tax: float  # 印花税
    total_cost: float  # 总成本


class LimitUpQueueSimulator:
    """涨停板排队模拟器"""
    
    def __init__(self):
        """初始化模拟器参数"""
        # 排队成交概率模型参数
        self.queue_params = {
            'seal_threshold': 10000,  # 封单阈值（万元）
            'time_decay': 0.8,  # 时间衰减因子
            'volatility_impact': 0.6,  # 波动率影响
            'retail_ratio': 0.3  # 散户参与比例
        }
        
        # 成本参数
        self.cost_params = {
            'commission_rate': 0.0003,  # 佣金率（万三）
            'stamp_tax_rate': 0.001,  # 印花税（千一，卖出时）
            'min_commission': 5,  # 最低佣金
        }
        
    def simulate_limit_up_queue(
        self,
        symbol: str,
        order_time: str,
        seal_amount: float,
        open_times: int,
        limitup_time: str,
        order_volume: int,
        order_price: float
    ) -> OrderExecution:
        """
        模拟涨停板排队成交
        
        Args:
            symbol: 股票代码
            order_time: 下单时间
            seal_amount: 封单金额（万元）
            open_times: 开板次数
            limitup_time: 涨停时间
            order_volume: 下单量（股）
            order_price: 下单价格
            
        Returns:
            订单执行详情
        """
        
        # 计算排队位置
        queue_position = self._calculate_queue_position(
            order_time, limitup_time, seal_amount
        )
        
        # 计算成交概率
        execution_prob = self._calculate_execution_probability(
            queue_position, seal_amount, open_times, order_time, limitup_time
        )
        
        # 决定是否成交
        is_executed = np.random.random() < execution_prob
        
        if is_executed:
            # 计算实际成交量（可能部分成交）
            execute_volume = self._calculate_executed_volume(
                order_volume, seal_amount, queue_position
            )
            
            # 计算滑点
            slippage = self._calculate_slippage(
                order_price, seal_amount, open_times
            )
            
            execute_price = order_price * (1 + slippage)
            execute_time = self._estimate_execution_time(
                order_time, queue_position, seal_amount
            )
            
        else:
            execute_volume = 0
            execute_price = 0
            execute_time = ""
            slippage = 0
            
        # 计算交易成本
        commission = self._calculate_commission(execute_price * execute_volume)
        stamp_tax = 0  # 买入无印花税
        total_cost = commission
        
        return OrderExecution(
            symbol=symbol,
            order_time=order_time,
            execute_time=execute_time,
            order_price=order_price,
            execute_price=execute_price,
            order_volume=order_volume,
            execute_volume=execute_volume,
            is_executed=is_executed,
            queue_position=queue_position,
            seal_amount=seal_amount,
            execution_probability=execution_prob,
            slippage=slippage,
            commission=commission,
            stamp_tax=stamp_tax,
            total_cost=total_cost
        )
    
    def _calculate_queue_position(
        self,
        order_time: str,
        limitup_time: str,
        seal_amount: float
    ) -> int:
        """
        计算排队位置
        
        早下单、封单大的排队靠前
        """
        # 时间差（分钟）
        order_dt = pd.to_datetime(order_time)
        limitup_dt = pd.to_datetime(limitup_time)
        time_diff = (order_dt - limitup_dt).total_seconds() / 60
        
        if time_diff <= 0:
            # 涨停前下单，排队最前
            base_position = np.random.randint(1, 100)
        elif time_diff <= 5:
            # 涨停后5分钟内
            base_position = np.random.randint(100, 1000)
        elif time_diff <= 30:
            # 涨停后30分钟内
            base_position = np.random.randint(1000, 10000)
        else:
            # 涨停后30分钟以上
            base_position = np.random.randint(10000, 100000)
            
        # 根据封单金额调整
        if seal_amount > 50000:  # 5亿以上
            position_factor = 0.5
        elif seal_amount > 10000:  # 1亿以上
            position_factor = 0.7
        elif seal_amount > 5000:  # 5000万以上
            position_factor = 1.0
        else:
            position_factor = 1.5
            
        return int(base_position * position_factor)
    
    def _calculate_execution_probability(
        self,
        queue_position: int,
        seal_amount: float,
        open_times: int,
        order_time: str,
        limitup_time: str
    ) -> float:
        """
        计算成交概率
        
        考虑因素：
        1. 排队位置
        2. 封单金额
        3. 开板次数
        4. 下单时机
        """
        
        # 基础概率（根据排队位置）
        if queue_position <= 100:
            base_prob = 0.95
        elif queue_position <= 1000:
            base_prob = 0.80
        elif queue_position <= 5000:
            base_prob = 0.60
        elif queue_position <= 10000:
            base_prob = 0.40
        elif queue_position <= 50000:
            base_prob = 0.20
        else:
            base_prob = 0.05
            
        # 封单金额影响
        if seal_amount > 50000:  # 5亿以上
            seal_factor = 1.2
        elif seal_amount > 10000:  # 1亿以上
            seal_factor = 1.1
        elif seal_amount > 5000:  # 5000万以上
            seal_factor = 1.0
        elif seal_amount > 1000:  # 1000万以上
            seal_factor = 0.8
        else:
            seal_factor = 0.5
            
        # 开板次数影响
        if open_times == 0:
            open_factor = 1.2  # 一字板
        elif open_times == 1:
            open_factor = 1.0
        elif open_times == 2:
            open_factor = 0.7
        else:
            open_factor = 0.3  # 烂板
            
        # 时间影响（尾盘涨停难成交）
        order_hour = pd.to_datetime(order_time).hour
        if order_hour < 10:
            time_factor = 1.2
        elif order_hour < 14:
            time_factor = 1.0
        else:
            time_factor = 0.6  # 14点后
            
        # 综合概率
        final_prob = base_prob * seal_factor * open_factor * time_factor
        
        return min(max(final_prob, 0.01), 0.99)
    
    def _calculate_executed_volume(
        self,
        order_volume: int,
        seal_amount: float,
        queue_position: int
    ) -> int:
        """计算实际成交量（可能部分成交）"""
        
        if queue_position <= 1000:
            # 排队靠前，大概率全部成交
            execution_ratio = np.random.uniform(0.9, 1.0)
        elif queue_position <= 10000:
            # 中等位置，可能部分成交
            execution_ratio = np.random.uniform(0.5, 0.9)
        else:
            # 排队靠后，小部分成交
            execution_ratio = np.random.uniform(0.1, 0.5)
            
        # 封单小的话，成交量也会受限
        if seal_amount < 1000:  # 封单小于1000万
            execution_ratio *= 0.5
            
        executed = int(order_volume * execution_ratio)
        
        # 确保是100的整数倍
        executed = (executed // 100) * 100
        
        return max(executed, 100)  # 至少成交100股
    
    def _calculate_slippage(
        self,
        order_price: float,
        seal_amount: float,
        open_times: int
    ) -> float:
        """
        计算滑点
        
        涨停板买入通常没有负滑点，但可能有正滑点（买不到）
        """
        
        if open_times == 0:
            # 一字板，基本无滑点
            slippage = 0
        elif open_times <= 2:
            # 开板1-2次，小滑点
            slippage = np.random.uniform(0, 0.001)
        else:
            # 多次开板，可能有较大滑点
            slippage = np.random.uniform(0, 0.003)
            
        # 封单小时滑点增大
        if seal_amount < 1000:
            slippage *= 2
            
        return slippage
    
    def _estimate_execution_time(
        self,
        order_time: str,
        queue_position: int,
        seal_amount: float
    ) -> str:
        """估算成交时间"""
        
        order_dt = pd.to_datetime(order_time)
        
        # 根据排队位置估算延迟
        if queue_position <= 100:
            delay_minutes = np.random.randint(0, 1)
        elif queue_position <= 1000:
            delay_minutes = np.random.randint(1, 5)
        elif queue_position <= 10000:
            delay_minutes = np.random.randint(5, 30)
        else:
            delay_minutes = np.random.randint(30, 180)
            
        execute_dt = order_dt + timedelta(minutes=delay_minutes)
        
        # 确保不超过收盘时间
        close_time = order_dt.replace(hour=15, minute=0)
        if execute_dt > close_time:
            execute_dt = close_time
            
        return execute_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def _calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        commission = amount * self.cost_params['commission_rate']
        return max(commission, self.cost_params['min_commission'])


class RealisticBacktester:
    """写实回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # 当前持仓
        self.trades = []  # 成交记录
        self.daily_stats = []  # 每日统计
        
        self.queue_simulator = LimitUpQueueSimulator()
        
    def run_backtest(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        运行写实回测
        
        Args:
            signals: 交易信号（含涨停相关数据）
            market_data: 市场数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        
        logger.info(f"开始写实回测: {start_date} 至 {end_date}")
        
        trading_days = pd.bdate_range(start_date, end_date)
        
        for date in trading_days:
            date_str = date.strftime('%Y-%m-%d')
            
            # 获取当日信号
            daily_signals = signals[signals['date'] == date_str]
            
            if not daily_signals.empty:
                # 处理买入信号
                self._process_buy_signals(daily_signals, date_str)
            
            # 处理卖出（次日卖出）
            self._process_sell_signals(date_str, market_data)
            
            # 更新每日统计
            self._update_daily_stats(date_str, market_data)
            
        # 计算回测指标
        results = self._calculate_metrics()
        
        return results
    
    def _process_buy_signals(self, signals: pd.DataFrame, date: str):
        """处理买入信号（涨停板排队买入）"""
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            
            # 检查资金是否充足
            if self.capital < 10000:  # 最少1万元一笔
                logger.warning(f"资金不足，跳过 {symbol}")
                continue
            
            # 计算买入金额（等权重或根据信号强度）
            position_size = min(
                self.capital * 0.2,  # 单股最多20%仓位
                100000  # 单笔最多10万
            )
            
            order_price = signal['limit_price']  # 涨停价
            order_volume = int(position_size / order_price / 100) * 100
            
            # 模拟涨停板排队成交
            execution = self.queue_simulator.simulate_limit_up_queue(
                symbol=symbol,
                order_time=f"{date} 09:30:00",  # 假设开盘下单
                seal_amount=signal.get('seal_amount', 5000),
                open_times=signal.get('open_times', 1),
                limitup_time=signal.get('limitup_time', f"{date} 09:35:00"),
                order_volume=order_volume,
                order_price=order_price
            )
            
            if execution.is_executed:
                # 成交，更新持仓和资金
                cost = execution.execute_price * execution.execute_volume
                total_cost = cost + execution.total_cost
                
                if total_cost <= self.capital:
                    self.capital -= total_cost
                    
                    # 更新持仓
                    if symbol not in self.positions:
                        self.positions[symbol] = {
                            'volume': 0,
                            'cost': 0,
                            'trades': []
                        }
                    
                    self.positions[symbol]['volume'] += execution.execute_volume
                    self.positions[symbol]['cost'] += total_cost
                    self.positions[symbol]['trades'].append(execution)
                    
                    # 记录交易
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': execution.execute_price,
                        'volume': execution.execute_volume,
                        'amount': cost,
                        'commission': execution.commission,
                        'execution_prob': execution.execution_probability,
                        'queue_position': execution.queue_position
                    })
                    
                    logger.info(
                        f"买入成交: {symbol} "
                        f"价格:{execution.execute_price:.2f} "
                        f"数量:{execution.execute_volume} "
                        f"排队:{execution.queue_position} "
                        f"概率:{execution.execution_probability:.2%}"
                    )
            else:
                logger.info(f"买入失败: {symbol} 排队位置:{execution.queue_position}")
    
    def _process_sell_signals(self, date: str, market_data: pd.DataFrame):
        """处理卖出信号（次日卖出）"""
        
        for symbol in list(self.positions.keys()):
            if self.positions[symbol]['volume'] > 0:
                # 获取卖出价格（次日开盘价或涨停价）
                symbol_data = market_data[
                    (market_data['symbol'] == symbol) &
                    (market_data['date'] == date)
                ]
                
                if not symbol_data.empty:
                    # 简化：使用开盘价卖出
                    sell_price = symbol_data.iloc[0]['open']
                    sell_volume = self.positions[symbol]['volume']
                    
                    # 计算卖出收入
                    amount = sell_price * sell_volume
                    
                    # 计算交易成本
                    commission = max(
                        amount * 0.0003,  # 万三佣金
                        5  # 最低5元
                    )
                    stamp_tax = amount * 0.001  # 印花税千一
                    
                    total_cost = commission + stamp_tax
                    net_amount = amount - total_cost
                    
                    # 更新资金
                    self.capital += net_amount
                    
                    # 计算盈亏
                    cost_basis = self.positions[symbol]['cost']
                    profit = net_amount - cost_basis
                    profit_rate = profit / cost_basis
                    
                    # 记录交易
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': sell_price,
                        'volume': sell_volume,
                        'amount': amount,
                        'commission': commission,
                        'stamp_tax': stamp_tax,
                        'profit': profit,
                        'profit_rate': profit_rate
                    })
                    
                    # 清空持仓
                    del self.positions[symbol]
                    
                    logger.info(
                        f"卖出: {symbol} "
                        f"价格:{sell_price:.2f} "
                        f"盈亏:{profit:.2f}({profit_rate:.2%})"
                    )
    
    def _update_daily_stats(self, date: str, market_data: pd.DataFrame):
        """更新每日统计"""
        
        # 计算持仓市值
        positions_value = 0
        for symbol, position in self.positions.items():
            if position['volume'] > 0:
                symbol_data = market_data[
                    (market_data['symbol'] == symbol) &
                    (market_data['date'] == date)
                ]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['close']
                    positions_value += current_price * position['volume']
        
        # 总资产
        total_value = self.capital + positions_value
        
        # 收益率
        returns = (total_value - self.initial_capital) / self.initial_capital
        
        self.daily_stats.append({
            'date': date,
            'capital': self.capital,
            'positions_value': positions_value,
            'total_value': total_value,
            'returns': returns
        })
    
    def _calculate_metrics(self) -> Dict:
        """计算回测指标"""
        
        if not self.daily_stats:
            return {}
        
        df_stats = pd.DataFrame(self.daily_stats)
        df_trades = pd.DataFrame(self.trades)
        
        # 基础指标
        total_returns = df_stats['returns'].iloc[-1]
        
        # 计算日收益率
        df_stats['daily_returns'] = df_stats['total_value'].pct_change()
        
        # 最大回撤
        cummax = (1 + df_stats['returns']).cummax()
        drawdown = ((1 + df_stats['returns']) / cummax - 1)
        max_drawdown = drawdown.min()
        
        # 夏普比率（假设无风险利率3%）
        risk_free = 0.03
        excess_returns = df_stats['daily_returns'].mean() * 252 - risk_free
        volatility = df_stats['daily_returns'].std() * np.sqrt(252)
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # 交易统计
        if not df_trades.empty:
            winning_trades = df_trades[df_trades['profit'] > 0]
            losing_trades = df_trades[df_trades['profit'] < 0]
            
            win_rate = len(winning_trades) / len(df_trades) if len(df_trades) > 0 else 0
            
            avg_win = winning_trades['profit_rate'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['profit_rate'].mean() if not losing_trades.empty else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # 涨停板成交统计
            buy_trades = df_trades[df_trades['action'] == 'BUY']
            avg_execution_prob = buy_trades['execution_prob'].mean() if not buy_trades.empty else 0
            avg_queue_position = buy_trades['queue_position'].mean() if not buy_trades.empty else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_execution_prob = 0
            avg_queue_position = 0
        
        return {
            'total_returns': total_returns,
            'annual_returns': (1 + total_returns) ** (252 / len(df_stats)) - 1,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(df_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_execution_prob': avg_execution_prob,
            'avg_queue_position': avg_queue_position,
            'daily_stats': df_stats,
            'trades': df_trades
        }


def run_realistic_backtest_example():
    """运行写实回测示例"""
    
    # 创建模拟数据
    dates = pd.bdate_range('2024-01-01', '2024-01-31')
    
    # 模拟交易信号
    signals = []
    for i in range(10):
        signals.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'symbol': f'00000{i%5}',
            'limit_price': 10.0 + i * 0.1,
            'seal_amount': np.random.uniform(1000, 50000),
            'open_times': np.random.randint(0, 3),
            'limitup_time': f"{dates[i].strftime('%Y-%m-%d')} 09:{30+i%30:02d}:00"
        })
    signals_df = pd.DataFrame(signals)
    
    # 模拟市场数据
    market_data = []
    for date in dates:
        for i in range(5):
            market_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': f'00000{i}',
                'open': 10.0 + np.random.uniform(-0.5, 0.5),
                'close': 10.0 + np.random.uniform(-0.5, 0.5),
                'high': 11.0,
                'low': 9.0
            })
    market_df = pd.DataFrame(market_data)
    
    # 运行回测
    backtester = RealisticBacktester(initial_capital=1000000)
    results = backtester.run_backtest(
        signals=signals_df,
        market_data=market_df,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    # 输出结果
    print("\n=== 写实回测结果 ===")
    print(f"总收益率: {results['total_returns']:.2%}")
    print(f"年化收益: {results['annual_returns']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"盈亏比: {results['profit_factor']:.2f}")
    print(f"总交易数: {results['total_trades']}")
    print(f"平均成交概率: {results['avg_execution_prob']:.2%}")
    print(f"平均排队位置: {results['avg_queue_position']:.0f}")
    
    return results


if __name__ == "__main__":
    results = run_realistic_backtest_example()