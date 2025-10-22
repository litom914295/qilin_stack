"""
回测系统引擎
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from decision_engine.core import get_decision_engine, SignalType
from persistence.returns_store import get_returns_store

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000.0  # 初始资金
    commission: float = 0.0003  # 手续费率
    slippage: float = 0.001  # 滑点
    max_position_size: float = 0.2  # 最大单次仓位
    stop_loss: float = -0.05  # 止损
    take_profit: float = 0.10  # 止盈


@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    symbol: str
    action: str  # buy, sell
    price: float
    quantity: int
    commission: float
    pnl: Optional[float] = None


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float
    pnl: float
    pnl_pct: float


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.decision_engine = get_decision_engine()
        
        # 状态
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [self.capital]
        self.dates: List[datetime] = []
        
    async def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_source: pd.DataFrame
    ) -> Dict:
        """运行回测"""
        logger.info(f"开始回测: {start_date} 至 {end_date}")
        logger.info(f"股票池: {symbols}")
        logger.info(f"初始资金: {self.capital:,.2f}")
        
        # 生成交易日列表
        dates = pd.date_range(start_date, end_date, freq='B')  # B = 工作日
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            self.dates.append(date)
            
            # 更新持仓价格
            self._update_positions(data_source, date)
            
            # 生成决策
            decisions = await self.decision_engine.make_decisions(symbols, date_str)
            
            # 执行交易
            for decision in decisions:
                await self._execute_decision(decision, data_source, date)
            
            # 记录权益
            total_equity = self._calculate_total_equity(data_source, date)
            self.equity_curve.append(total_equity)
            
            # 进度
            if len(self.dates) % 20 == 0:
                logger.info(
                    f"进度: {date_str}, 权益: {total_equity:,.2f}, 收益率: {(total_equity/self.capital-1)*100:.2f}%"
                )
        
        # 计算回测结果
        results = self._calculate_metrics()
        return results
    
    def _update_positions(self, data: pd.DataFrame, date: datetime):
        """更新持仓信息"""
        for symbol, position in self.positions.items():
            # 获取当前价格
            try:
                current_price = self._get_price(data, symbol, date)
                position.current_price = current_price
                position.pnl = (current_price - position.entry_price) * position.quantity
                position.pnl_pct = (current_price / position.entry_price - 1)
                
                # 止损止盈检查
                if position.pnl_pct <= self.config.stop_loss:
                    logger.warning(f"止损: {symbol}, 亏损: {position.pnl_pct:.2%}")
                    self._close_position(symbol, current_price, date, "stop_loss")
                elif position.pnl_pct >= self.config.take_profit:
                    logger.info(f"止盈: {symbol}, 盈利: {position.pnl_pct:.2%}")
                    self._close_position(symbol, current_price, date, "take_profit")
            except:
                pass  # 数据缺失，跳过
    
    async def _execute_decision(self, decision, data: pd.DataFrame, date: datetime):
        """执行决策"""
        symbol = decision.symbol
        signal = decision.final_signal
        
        try:
            current_price = self._get_price(data, symbol, date)
        except:
            return  # 无数据，跳过
        
        # 买入信号
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            if symbol not in self.positions:
                # 计算可买数量
                position_value = self.capital * self.config.max_position_size
                quantity = int(position_value / current_price / 100) * 100  # 整百股
                
                if quantity > 0 and self.capital >= quantity * current_price:
                    self._open_position(symbol, current_price, quantity, date)
        
        # 卖出信号
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            if symbol in self.positions:
                self._close_position(symbol, current_price, date, "signal")
    
    def _open_position(self, symbol: str, price: float, quantity: int, date: datetime):
        """开仓"""
        cost = price * quantity
        commission = cost * self.config.commission
        total_cost = cost + commission
        
        if self.capital >= total_cost:
            self.capital -= total_cost
            
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=date,
                current_price=price,
                pnl=0.0,
                pnl_pct=0.0
            )
            self.positions[symbol] = position
            
            trade = Trade(
                timestamp=date,
                symbol=symbol,
                action='buy',
                price=price,
                quantity=quantity,
                commission=commission
            )
            self.trades.append(trade)
            
            logger.info(f"📈 买入: {symbol}, 价格: {price:.2f}, 数量: {quantity}, 成本: {total_cost:,.2f}")
    
    def _close_position(self, symbol: str, price: float, date: datetime, reason: str):
        """平仓"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        proceeds = price * position.quantity
        commission = proceeds * self.config.commission
        net_proceeds = proceeds - commission
        
        self.capital += net_proceeds
        
        pnl = net_proceeds - (position.entry_price * position.quantity)
        
        trade = Trade(
            timestamp=date,
            symbol=symbol,
            action='sell',
            price=price,
            quantity=position.quantity,
            commission=commission,
            pnl=pnl
        )
        self.trades.append(trade)
        
        # 记录已实现收益至回放存储（用于自适应权重）
        try:
            realized_return = position.pnl_pct
            get_returns_store().record(symbol=symbol, realized_return=realized_return,
                                       date=date.strftime('%Y-%m-%d'))
        except Exception:
            pass
        
        logger.info(f"📉 卖出: {symbol}, 价格: {price:.2f}, 盈亏: {pnl:,.2f} ({position.pnl_pct:.2%}), 原因: {reason}")
        
        del self.positions[symbol]
    
    def _get_price(self, data: pd.DataFrame, symbol: str, date: datetime) -> float:
        """获取价格"""
        # 简化实现：从数据中获取收盘价
        try:
            price_data = data[(data['symbol'] == symbol) & (data['date'] == date)]
            if len(price_data) > 0:
                return float(price_data.iloc[0]['close'])
        except:
            pass
        raise ValueError(f"无价格数据: {symbol} @ {date}")
    
    def _calculate_total_equity(self, data: pd.DataFrame, date: datetime) -> float:
        """计算总权益"""
        total = self.capital
        for symbol, position in self.positions.items():
            try:
                current_price = self._get_price(data, symbol, date)
                total += current_price * position.quantity
            except:
                total += position.entry_price * position.quantity  # 使用成本价
        return total
    
    def _calculate_metrics(self) -> Dict:
        """计算回测指标"""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # 基本指标
        total_return = (equity[-1] / equity[0] - 1)
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 风险指标
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 交易统计
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / len([t for t in self.trades if t.pnl]) if self.trades else 0
        
        metrics = {
            'initial_capital': self.config.initial_capital,
            'final_equity': equity[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
        
        return metrics
    
    def print_summary(self, metrics: Dict):
        """打印回测摘要"""
        lines = [
            "="*60,
            "回测结果摘要",
            "="*60,
            f"初始资金: {metrics['initial_capital']:,.2f}",
            f"最终权益: {metrics['final_equity']:,.2f}",
            f"总收益率: {metrics['total_return']:.2%}",
            f"年化收益率: {metrics['annual_return']:.2%}",
            "风险指标:",
            f"波动率: {metrics['volatility']:.2%}",
            f"夏普比率: {metrics['sharpe_ratio']:.2f}",
            f"最大回撤: {metrics['max_drawdown']:.2%}",
            "交易统计:",
            f"总交易次数: {metrics['total_trades']}",
            f"胜率: {metrics['win_rate']:.2%}",
            f"盈利交易: {metrics['winning_trades']}",
            f"亏损交易: {metrics['losing_trades']}",
            "="*60,
        ]
        logger.info("\n".join(lines))


async def run_simple_backtest():
    """简单回测示例"""
    # 创建模拟数据
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
    symbols = ['000001.SZ', '600000.SH']
    
    data_list = []
    for symbol in symbols:
        for date in dates:
            # 生成随机价格
            base_price = 10 if symbol == '000001.SZ' else 8
            price = base_price + np.random.randn() * 0.5
            data_list.append({
                'symbol': symbol,
                'date': date,
                'close': price,
                'open': price * 0.99,
                'high': price * 1.01,
                'low': price * 0.98,
                'volume': np.random.randint(1000000, 10000000)
            })
    
    data = pd.DataFrame(data_list)
    
    # 运行回测
    config = BacktestConfig(
        initial_capital=1000000.0,
        max_position_size=0.3,
        stop_loss=-0.05,
        take_profit=0.10
    )
    
    engine = BacktestEngine(config)
    metrics = await engine.run_backtest(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-06-30',
        data_source=data
    )
    
    engine.print_summary(metrics)
    return metrics


if __name__ == '__main__':
    import asyncio
    asyncio.run(run_simple_backtest())
