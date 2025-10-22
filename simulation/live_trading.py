"""
实盘模拟交易系统
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from backtest.engine import BacktestConfig, Trade, Position
from decision_engine.core import get_decision_engine, SignalType
from monitoring.metrics import get_monitor
from persistence.database import get_db, DecisionRecord
from persistence.returns_store import get_returns_store

logger = logging.getLogger(__name__)


@dataclass
class LiveTradingConfig:
    """实盘配置"""
    initial_capital: float = 1000000.0
    max_position_size: float = 0.2
    stop_loss: float = -0.05
    take_profit: float = 0.10
    check_interval: int = 60  # 检查间隔（秒）
    trading_hours: tuple = (9, 15)  # 交易时间 9:00-15:00


class LiveTradingSimulator:
    """实盘模拟器"""
    
    def __init__(self, config: Optional[LiveTradingConfig] = None):
        self.config = config or LiveTradingConfig()
        self.decision_engine = get_decision_engine()
        self.monitor = get_monitor()
        self.db = get_db()
        
        # 状态
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.is_running = False
    
    async def start(self, symbols: List[str]):
        """启动实盘模拟"""
        logger.info("🚀 启动实盘模拟")
        logger.info(f"初始资金: {self.capital:,.2f}")
        logger.info(f"股票池: {symbols}")
        logger.info(f"检查间隔: {self.config.check_interval}秒")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # 检查是否在交易时间
                if self._is_trading_time():
                    await self._trading_cycle(symbols)
                else:
                    logger.info("⏸️ 非交易时间，休眠...")
                
                # 等待下一个周期
                await asyncio.sleep(self.config.check_interval)
        
        except KeyboardInterrupt:
            logger.info("⏹️ 用户停止")
        finally:
            self.stop()
    
    def stop(self):
        """停止交易"""
        self.is_running = False
        logger.info("📊 交易总结")
        logger.info(f"最终资金: {self.capital:,.2f}")
        logger.info(f"总收益: {self.capital - self.config.initial_capital:,.2f}")
        logger.info(f"收益率: {(self.capital/self.config.initial_capital-1)*100:.2f}%")
        logger.info(f"总交易: {len(self.trades)}")
        logger.info(f"当前持仓: {len(self.positions)}")
    
    async def _trading_cycle(self, symbols: List[str]):
        """交易周期"""
        logger.info(f"⏰ {datetime.now().strftime('%H:%M:%S')} - 交易周期")
        
        # 1. 更新持仓
        await self._update_positions()
        
        # 2. 生成决策
        decisions = await self.decision_engine.make_decisions(
            symbols, 
            datetime.now().strftime('%Y-%m-%d')
        )
        
        # 3. 执行交易
        for decision in decisions:
            await self._execute_decision(decision)
        
        # 4. 记录监控
        self._record_metrics(decisions)
        
        # 5. 持久化
        self._save_to_db(decisions)
        
        # 6. 打印状态
        self._print_status()
    
    async def _update_positions(self):
        """更新持仓"""
        for symbol, position in list(self.positions.items()):
            # 获取最新价格
            current_price = await self._get_latest_price(symbol)
            
            if current_price:
                position.current_price = current_price
                position.pnl = (current_price - position.entry_price) * position.quantity
                position.pnl_pct = (current_price / position.entry_price - 1)
                
                # 止损止盈检查
                if position.pnl_pct <= self.config.stop_loss:
                    logger.warning(f"止损: {symbol}, {position.pnl_pct:.2%}")
                    await self._close_position(symbol, current_price, "stop_loss")
                elif position.pnl_pct >= self.config.take_profit:
                    logger.info(f"止盈: {symbol}, {position.pnl_pct:.2%}")
                    await self._close_position(symbol, current_price, "take_profit")
    
    async def _execute_decision(self, decision):
        """执行决策"""
        symbol = decision.symbol
        signal = decision.final_signal
        
        # 获取当前价格
        current_price = await self._get_latest_price(symbol)
        if not current_price:
            return
        
        # 买入信号
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            if symbol not in self.positions and decision.confidence > 0.6:
                await self._open_position(symbol, current_price)
        
        # 卖出信号
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            if symbol in self.positions:
                await self._close_position(symbol, current_price, "signal")
    
    async def _open_position(self, symbol: str, price: float):
        """开仓"""
        position_value = self.capital * self.config.max_position_size
        quantity = int(position_value / price / 100) * 100  # 整百股
        
        if quantity > 0:
            cost = price * quantity * 1.0003  # 含手续费
            
            if self.capital >= cost:
                self.capital -= cost
                
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price,
                    pnl=0.0,
                    pnl_pct=0.0
                )
                self.positions[symbol] = position
                
                logger.info(f"📈 买入: {symbol}, 价格: {price:.2f}, 数量: {quantity}")
    
    async def _close_position(self, symbol: str, price: float, reason: str):
        """平仓"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        proceeds = price * position.quantity * 0.9997  # 扣手续费
        
        self.capital += proceeds
        pnl = proceeds - (position.entry_price * position.quantity)
        
        # 记录已实现收益至回放存储（用于自适应权重）
        try:
            realized_return = position.pnl_pct  # 比例收益
            get_returns_store().record(symbol=symbol, realized_return=realized_return,
                                       date=datetime.now().strftime('%Y-%m-%d'))
        except Exception:
            pass
        
        logger.info(f"📉 卖出: {symbol}, 价格: {price:.2f}, 盈亏: {pnl:,.2f} ({position.pnl_pct:.2%})")
        
        del self.positions[symbol]
    
    async def _get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格（模拟）"""
        # 实际应该从AKShare或其他数据源获取
        import random
        await asyncio.sleep(0.01)
        return 10.0 + random.random() * 2
    
    def _is_trading_time(self) -> bool:
        """检查是否交易时间"""
        now = datetime.now()
        hour = now.hour
        
        # 周末不交易
        if now.weekday() >= 5:
            return False
        
        # 检查交易时段
        return self.config.trading_hours[0] <= hour < self.config.trading_hours[1]
    
    def _record_metrics(self, decisions):
        """记录监控指标"""
        for decision in decisions:
            self.monitor.record_decision(
                symbol=decision.symbol,
                decision=decision.final_signal.value,
                latency=0.05,
                confidence=decision.confidence
            )
    
    def _save_to_db(self, decisions):
        """保存到数据库"""
        for decision in decisions:
            record = DecisionRecord(
                timestamp=datetime.now(),
                symbol=decision.symbol,
                signal=decision.final_signal.value,
                confidence=decision.confidence,
                strength=decision.strength,
                reasoning=decision.reasoning
            )
            self.db.save_decision(record)
    
    def _print_status(self):
        """打印状态"""
        total_equity = self.capital
        for position in self.positions.values():
            total_equity += position.current_price * position.quantity
        
        pnl = total_equity - self.config.initial_capital
        pnl_pct = (total_equity / self.config.initial_capital - 1) * 100
        
        print(f"💰 资金: {self.capital:,.2f}")
        print(f"📊 总权益: {total_equity:,.2f}")
        print(f"📈 盈亏: {pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"📦 持仓: {len(self.positions)}")


async def run_live_simulation():
    """运行实盘模拟示例"""
    config = LiveTradingConfig(
        initial_capital=1000000.0,
        max_position_size=0.2,
        check_interval=10,  # 10秒检查一次（实际应该更长）
        trading_hours=(0, 24)  # 测试用，全天交易
    )
    
    simulator = LiveTradingSimulator(config)
    symbols = ['000001.SZ', '600000.SH']
    
    # 运行5分钟后自动停止
    await asyncio.wait_for(
        simulator.start(symbols),
        timeout=300
    )


if __name__ == '__main__':
    asyncio.run(run_live_simulation())
