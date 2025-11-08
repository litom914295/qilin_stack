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
    fill_model: str = "deterministic"  # 成交模型：deterministic（基于前日特征的确定性比例）


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
        
        # 初始化涨停排队模拟器（可选）
        self.queue_simulator = None
        if self.config.fill_model == 'queue':
            from qilin_stack.backtest.limit_up_queue_simulator import LimitUpQueueSimulator
            self.queue_simulator = LimitUpQueueSimulator()
        
        # 状态
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [self.capital]
        self.dates: List[datetime] = []
        
        # 统计（撮合/成交）
        self.stats = {
            'orders_attempted': 0,
            'orders_unfilled': 0,
            'shares_planned': 0,
            'shares_filled': 0,
            'fill_ratios': [],  # 记录每次成交比例
        }
        
        
    async def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_source: pd.DataFrame,
        *,
        trade_at: str = 'next_open',  # 'next_open' | 'same_day_close'
        avoid_limit_up_unfillable: bool = True,
    ) -> Dict:
        """运行回测。
        trade_at:
          - same_day_close: 维持原有（理想化）口径
          - next_open: 更贴近实务的T+1在开盘成交口径；若开盘即涨停且无法成交，则跳过下单
        """
        logger.info(f"开始回测: {start_date} 至 {end_date}")
        logger.info(f"股票池: {symbols}")
        logger.info(f"初始资金: {self.capital:,.2f}")
        
        # 生成交易日列表
        dates = pd.date_range(start_date, end_date, freq='B')  # B = 工作日
        n_dates = len(dates)
        
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            self.dates.append(date)
            
            # 更新持仓价格（以当日收盘估值）
            self._update_positions(data_source, date)
            
            # 生成当日决策
            decisions = await self.decision_engine.make_decisions(symbols, date_str)
            
            # 确定执行日与价格字段
            exec_date = date
            price_field = 'close'
            if trade_at == 'next_open':
                if i + 1 >= n_dates:
                    # 无下一交易日，跳过执行
                    exec_date = None
                else:
                    exec_date = dates[i + 1]
                    price_field = 'open'
            
            # 执行交易
            if exec_date is not None:
                for decision in decisions:
                    await self._execute_decision(
                        decision, data_source, exec_date,
                        price_field=price_field,
                        avoid_limit_up_unfillable=avoid_limit_up_unfillable,
                        prev_date=(date if trade_at == 'next_open' else None),
                    )
            
            # 记录权益（以当日收盘口径）
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
    
    async def _execute_decision(self, decision, data: pd.DataFrame, date: datetime,
                               *, price_field: str = 'close',
                               avoid_limit_up_unfillable: bool = True,
                               prev_date: Optional[datetime] = None):
        """执行决策。price_field: 'close' or 'open'。"""
        symbol = decision.symbol
        signal = decision.final_signal
        
        # T+1 在开盘成交：如果开盘一字/涨停，视为无法成交（统计不返回）
        unfillable_open = False
        if price_field == 'open' and avoid_limit_up_unfillable:
            try:
                if prev_date is not None and self._approx_is_limit_up_open(data, symbol, date, prev_date):
                    unfillable_open = True
            except Exception:
                pass
        
        try:
            current_price = self._get_price(data, symbol, date, field=price_field)
        except Exception:
            return  # 无数据，跳过
        
        # 买入信号
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            if symbol not in self.positions:
                # 计划买入数量（上限）
                position_value = self.capital * self.config.max_position_size
                plan_qty = int(position_value / current_price / 100) * 100  # 整百股
                # 统计尝试/计划股数
                self.stats['orders_attempted'] += 1
                self.stats['shares_planned'] += plan_qty
                # 开盘一字不可成交
                if unfillable_open:
                    self.stats['orders_unfilled'] += 1
                    logger.info(f"未成交(开盘涨停): {symbol} @ {date.strftime('%Y-%m-%d')}")
                    return

                # 成交比例（T+1 开盘时基于前日特征/概率）
                fill_ratio = 1.0
                if price_field == 'open':
                    if self.config.fill_model == 'deterministic':
                        fill_ratio = self._compute_fill_ratio(symbol, date, prev_date)
                    elif self.config.fill_model == 'prob':
                        fill_ratio = self._compute_fill_ratio_prob(symbol, date, prev_date)
                    elif self.config.fill_model == 'queue' and self.queue_simulator:
                        # 使用涨停排队模拟器
                        fill_ratio = self._compute_fill_ratio_queue(symbol, date, prev_date)

                quantity = int((plan_qty * fill_ratio) / 100) * 100
                # 记录成交比例
                self.stats['fill_ratios'].append(fill_ratio)
                # 若有成交概率但整百后为0，则尝试最小100股
                if quantity == 0 and fill_ratio > 0 and plan_qty >= 100 and self.capital >= 100 * current_price:
                    quantity = 100
                
                if quantity > 0 and self.capital >= quantity * current_price:
                    # 应用滑点（买入价上移）
                    eff_price = current_price * (1.0 + abs(self.config.slippage))
                    self._open_position(symbol, eff_price, quantity, date)
                    self.stats['shares_filled'] += quantity
                    # 监控指标记录 (可选)
                    try:
                        from monitoring.metrics import get_monitor
                        mon = get_monitor()
                        mon.collector.increment_counter("orders_attempted_total")
                        mon.collector.increment_counter("orders_filled_total")
                    except Exception:
                        pass  # 监控模块不可用时跳过
                else:
                    self.stats['orders_unfilled'] += 1
                    # 监控指标记录 (可选)
                    try:
                        from monitoring.metrics import get_monitor
                        mon = get_monitor()
                        mon.collector.increment_counter("orders_attempted_total")
                        mon.collector.increment_counter("orders_unfilled_total")
                    except Exception:
                        pass  # 监控模块不可用时跳过
        
        # 卖出信号
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            if symbol in self.positions:
                # 应用滑点（卖出价下移）
                eff_price = current_price * (1.0 - abs(self.config.slippage))
                self._close_position(symbol, eff_price, date, "signal")
    
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
    
    def _get_price(self, data: pd.DataFrame, symbol: str, date: datetime, *, field: str = 'close') -> float:
        """获取价格，field 支持 'close' 或 'open'。"""
        try:
            price_data = data[(data['symbol'] == symbol) & (data['date'] == date)]
            if len(price_data) > 0:
                fld = field if field in price_data.columns else 'close'
                return float(price_data.iloc[0][fld])
        except Exception:
            pass
        raise ValueError(f"无价格数据: {symbol} @ {date}")
    
    def _get_stock_type(self, symbol: str) -> 'StockType':
        """根据股票代码判断股票类型"""
        from qilin_stack.backtest.limit_up_queue_simulator import StockType
        
        # ST股票（名称中包含ST）
        if 'ST' in symbol.upper():
            return StockType.ST
        
        # 创业板（3开头）和科创板（688开头） - 20%涨停
        if symbol.startswith('3') or symbol.startswith('688'):
            return StockType.CHINEXT
        
        # 其他为主板 - 10%涨停
        return StockType.MAIN_BOARD
    
    def _get_limit_up_ratio(self, symbol: str) -> float:
        """获取涨停板比例"""
        stock_type = self._get_stock_type(symbol)
        from qilin_stack.backtest.limit_up_queue_simulator import StockType
        
        if stock_type == StockType.ST:
            return 0.05  # 5%
        elif stock_type == StockType.CHINEXT:
            return 0.20  # 20%
        else:
            return 0.10  # 10%
    
    def _approx_is_limit_up_open(self, data: pd.DataFrame, symbol: str, date: datetime, prev_date: datetime) -> bool:
        """近似判断是否开盘涨停（一字/无法成交）。
        支持不同涨停板类型：10%、 20%、ST 5%。
        """
        try:
            prev = data[(data['symbol'] == symbol) & (data['date'] == prev_date)]
            today = data[(data['symbol'] == symbol) & (data['date'] == date)]
            if len(prev) > 0 and len(today) > 0:
                prev_close = float(prev.iloc[0]['close'])
                open_price = float(today.iloc[0]['open'])
                
                # 根据股票类型获取涨停板比例
                limit_ratio = self._get_limit_up_ratio(symbol)
                limit_threshold = 1.0 + limit_ratio - 0.002  # 留一点缓冲
                
                if prev_close > 0 and (open_price / prev_close) >= limit_threshold:
                    return True
        except Exception:
            return False
        return False

    def _compute_fill_ratio_queue(self, symbol: str, exec_date: datetime, prev_date: Optional[datetime]) -> float:
        """使用涨停排队模拟器计算成交比例"""
        try:
            if prev_date is None or not self.queue_simulator:
                return 1.0
            
            # 计算计划买入金额
            position_value = self.capital * self.config.max_position_size
            
            # 获取前一日封板强度
            from rd_agent.limit_up_data import LimitUpDataInterface
            data_if = LimitUpDataInterface(data_source='qlib')
            feats = data_if.get_limit_up_features([symbol], prev_date.strftime('%Y-%m-%d'))
            
            if feats is None or feats.empty:
                return 0.5  # 无数据时默认一半概率
            
            row = feats.iloc[0]
            seal_quality = float(row.get('seal_quality', row.get('seal_strength', 0.6) * 10.0))
            cont_board = int(row.get('continuous_board', row.get('board_height', 1.0)))
            
            # 根据封板质量判断强度
            from qilin_stack.backtest.limit_up_queue_simulator import LimitUpStrength
            if seal_quality > 8:
                strength = LimitUpStrength.STRONG
            elif seal_quality > 5:
                strength = LimitUpStrength.MEDIUM
            else:
                strength = LimitUpStrength.WEAK
            
            # 获取股票类型
            stock_type = self._get_stock_type(symbol)
            
            # 模拟排队
            can_buy, reason = self.queue_simulator.can_buy(
                limit_up_strength=strength,
                my_capital=position_value,  # 使用计划买入金额
                total_seal_amount=seal_quality * 1e8,  # 粗略估算封单金额
                stock_type=stock_type  # 传入股票类型
            )
            
            if can_buy:
                # 可以买入，但根据排队位置决定成交比例
                queue_position = self.queue_simulator.estimate_queue_position(
                    my_capital=position_value,
                    total_seal_amount=seal_quality * 1e8
                )
                # 根据排队位置计算成交比例
                if queue_position < 0.2:  # 排在前20%
                    return 1.0
                elif queue_position < 0.5:  # 排在20%-50%
                    return 0.7
                elif queue_position < 0.8:  # 排在50%-80%
                    return 0.3
                else:
                    return 0.1
            else:
                return 0.0  # 不能买入
            
        except Exception as e:
            logger.warning(f"队列模拟器计算失败: {e}")
            return 0.5  # 出错时默认一半概率
    
    def _compute_fill_ratio(self, symbol: str, exec_date: datetime, prev_date: Optional[datetime]) -> float:
        """计算订单成交比例的基础实现(确定性)
        
        根据配置的fill_model选择不同的成交模型:
        - deterministic: 使用确定性基础比例(基于前一日特征)
        - probability: 使用概率性成交比例
        - queue: 使用涨停排队模拟器
        """
        if self.config.fill_model == 'queue':
            return self._compute_fill_ratio_queue(symbol, exec_date, prev_date)
        elif self.config.fill_model == 'probability':
            return self._compute_fill_ratio_prob_original(symbol, exec_date, prev_date)
        else:
            # deterministic 模式 - 返回确定性比例
            return self._compute_fill_ratio_prob_original(symbol, exec_date, prev_date)
    
    def _compute_fill_ratio_prob(self, symbol: str, exec_date: datetime, prev_date: Optional[datetime]) -> float:
        """概率性成交比例 - 基于随机概率"""
        # 先获取确定性基础比例
        base_ratio = self._compute_fill_ratio_prob_original(symbol, exec_date, prev_date)
        
        # 加入随机性
        if base_ratio > 0:
            # 使用Beta分布生成随机成交比例
            # alpha和beta参数根据基础比例调整
            alpha = base_ratio * 4  # 控制分布形状
            beta = (1 - base_ratio) * 4
            random_ratio = np.random.beta(alpha, beta)
            return random_ratio
        return 0.0
    
    def _compute_fill_ratio_prob_original(self, symbol: str, exec_date: datetime, prev_date: Optional[datetime]) -> float:
        """确定性成交比例（0~1）。基于前一交易日的“一进二”相关特征。
        规则：高位连板（>2）降低；封板质量/量能/题材热度提高；范围压缩在[0.3, 1.0]。
        取值完全确定，不引入随机数，保证回测稳定。
        """
        try:
            if prev_date is None:
                return 1.0
            # 拉取前一日特征
            from rd_agent.limit_up_data import LimitUpDataInterface  # type: ignore
            data_if = LimitUpDataInterface(data_source='qlib')
            feats = data_if.get_limit_up_features([symbol], prev_date.strftime('%Y-%m-%d'))
            if feats is None or feats.empty:
                return 1.0
            row = feats.iloc[0]
            def get(name, default=0.0):
                try:
                    return float(row.get(name, default))
                except Exception:
                    return float(default)

            seal_quality = get('seal_quality', get('seal_strength', 0.6) * 10.0)
            volume_surge = get('volume_surge', 2.0)
            concept_heat = get('concept_heat', 3.0)
            cont_board = get('continuous_board', get('board_height', 1.0))

            def clamp(x, lo=0.0, hi=1.0):
                return max(lo, min(hi, float(x)))

            ratio = 1.0
            # 连板越高越难在次日开盘吃到合理成交量
            ratio *= clamp(1.2 - 0.15 * max(0.0, cont_board - 1.0), 0.3, 1.0)
            # 封板质量提升成交把握
            ratio *= clamp(0.5 + 0.05 * seal_quality, 0.4, 1.0)
            # 量能突增有利于流动性
            ratio *= clamp(0.6 + 0.05 * (volume_surge - 2.0), 0.4, 1.0)
            # 题材热度适度提升
            ratio *= clamp(0.7 + 0.02 * concept_heat, 0.5, 1.0)

            return clamp(ratio, 0.0, 1.0)
        except Exception:
            return 1.0

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
        
        # 计算未成交率和平均成交比例
        unfilled_rate = 0.0
        avg_fill_ratio = 1.0
        if self.stats['fill_ratios']:
            avg_fill_ratio = np.mean(self.stats['fill_ratios'])
            # 未成交率：成交比例 < 1% 的订单比例
            unfilled_rate = len([r for r in self.stats['fill_ratios'] if r < 0.01]) / len(self.stats['fill_ratios'])
        
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
            'losing_trades': len(losing_trades),
            # 执行/撮合统计
            'orders_attempted': self.stats.get('orders_attempted', 0),
            'orders_unfilled': self.stats.get('orders_unfilled', 0),
            'unfilled_rate': unfilled_rate,  # 未成交率
            'avg_fill_ratio': avg_fill_ratio,  # 平均成交比例
            'fill_ratio_realized': (
                self.stats['shares_filled'] / self.stats['shares_planned']
                if self.stats.get('shares_planned', 0) > 0 else 0.0
            ),
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
            "成交统计:",
            f"未成交率: {metrics.get('unfilled_rate', 0):.2%}",
            f"平均成交比例: {metrics.get('avg_fill_ratio', 1):.2%}",
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
        take_profit=0.10,
        fill_model='queue'  # 使用队列模拟模式
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
    from app.core.logging_setup import setup_logging
    setup_logging()
    asyncio.run(run_simple_backtest())
