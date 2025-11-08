"""
生产级实盘交易系统
P2-2任务: 实盘交易完整接口 (80h estimated, ROI 200%)

功能:
1. 订单管理系统 (OMS)
2. 风险控制器 (RiskManager)
3. 券商API适配器 (BrokerAdapter)
4. 仓位监控 (PositionMonitor)
5. 熔断机制 (CircuitBreaker)

作者: Qilin Stack Team
日期: 2025-11-07
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time
import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"     # 市价单
    LIMIT = "limit"       # 限价单
    STOP = "stop"         # 止损单


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"       # 待提交
    SUBMITTED = "submitted"   # 已提交
    PARTIAL_FILLED = "partial_filled"  # 部分成交
    FILLED = "filled"         # 完全成交
    CANCELLED = "cancelled"   # 已撤销
    REJECTED = "rejected"     # 被拒绝
    FAILED = "failed"         # 失败


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str                  # 股票代码
    side: OrderSide             # 买卖方向
    size: float                 # 数量
    price: Optional[float] = None  # 价格 (None表示市价)
    signal_time: datetime = field(default_factory=datetime.now)
    strategy_id: str = "default"
    confidence: float = 1.0     # 信号置信度 (0-1)
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    filled_size: float = 0.0
    filled_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    broker_order_id: Optional[str] = None  # 券商订单ID
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status)


@dataclass
class Position:
    """持仓"""
    symbol: str
    size: float                 # 持仓量
    avg_cost: float            # 平均成本
    market_value: float = 0.0  # 市值
    unrealized_pnl: float = 0.0  # 浮动盈亏
    realized_pnl: float = 0.0    # 已实现盈亏
    update_time: datetime = field(default_factory=datetime.now)


@dataclass
class OrderResult:
    """订单执行结果"""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    error_code: Optional[str] = None


@dataclass
class RiskCheckResult:
    """风险检查结果"""
    passed: bool
    reason: str = ""
    risk_score: float = 0.0  # 0-1, 0=安全, 1=高风险


@dataclass
class RiskConfig:
    """风险控制配置"""
    max_order_amount: float = 100000.0      # 单笔限额 (元)
    max_daily_trades: int = 100             # 日内最大交易次数
    max_position_pct: float = 0.2           # 单票最大仓位比例 (20%)
    max_drawdown_threshold: float = 0.1     # 最大回撤阈值 (10%)
    max_daily_loss: float = 50000.0         # 日内最大亏损 (元)
    trading_hours: List[tuple] = field(default_factory=lambda: [
        (time(9, 30), time(11, 30)),   # 上午盘
        (time(13, 0), time(15, 0))      # 下午盘
    ])


# ==================== 券商API适配器 (抽象基类) ====================

class BrokerAdapter(ABC):
    """
    券商API适配器抽象基类
    
    支持的券商:
    - 同花顺
    - 东方财富
    - 雪球
    - 模拟盘
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化券商适配器
        
        Args:
            config: 券商配置
                {
                    'broker_name': str,
                    'api_key': str,
                    'api_secret': str,
                    'account_id': str,
                    ...
                }
        """
        self.config = config
        self.broker_name = config.get('broker_name', 'unknown')
        self.connected = False
        
        logger.info(f"初始化{self.broker_name}券商适配器")
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接券商API"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> OrderResult:
        """提交订单"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """查询订单状态"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """查询持仓"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """查询账户信息"""
        pass


class MockBrokerAdapter(BrokerAdapter):
    """模拟券商适配器 (用于测试)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.account_balance = config.get('initial_cash', 1000000.0)
    
    async def connect(self) -> bool:
        """连接模拟券商"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        self.connected = True
        logger.info("✅ 模拟券商连接成功")
        return True
    
    async def disconnect(self) -> bool:
        """断开连接"""
        self.connected = False
        logger.info("断开模拟券商连接")
        return True
    
    async def submit_order(self, order: Order) -> OrderResult:
        """提交订单 (模拟)"""
        if not self.connected:
            return OrderResult(False, error_code="NOT_CONNECTED", message="未连接券商")
        
        await asyncio.sleep(0.05)  # 模拟网络延迟
        
        # 生成券商订单ID
        broker_order_id = f"MOCK_{order.order_id}"
        order.broker_order_id = broker_order_id
        order.status = OrderStatus.SUBMITTED
        order.update_time = datetime.now()
        
        # 保存订单
        self.orders[order.order_id] = order
        
        # 模拟立即成交 (90%概率)
        import random
        if random.random() < 0.9:
            await asyncio.sleep(0.1)  # 模拟成交延迟
            order.filled_size = order.size
            order.filled_price = order.price or 10.0  # 默认价格
            order.status = OrderStatus.FILLED
            order.update_time = datetime.now()
            
            # 更新持仓
            self._update_position(order)
            
            logger.info(
                f"✅ 订单成交: {order.symbol} {order.side.value} "
                f"{order.filled_size}股 @{order.filled_price}"
            )
        
        return OrderResult(
            success=True,
            order_id=order.order_id,
            message="订单已提交"
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                order.update_time = datetime.now()
                logger.info(f"订单已撤销: {order_id}")
                return True
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """查询订单状态"""
        return self.orders.get(order_id)
    
    async def get_positions(self) -> List[Position]:
        """查询持仓"""
        return list(self.positions.values())
    
    async def get_account_info(self) -> Dict:
        """查询账户信息"""
        total_market_value = sum(p.market_value for p in self.positions.values())
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in self.positions.values())
        
        return {
            'account_id': self.config.get('account_id', 'mock_account'),
            'balance': self.account_balance,
            'market_value': total_market_value,
            'total_asset': self.account_balance + total_market_value,
            'total_pnl': total_pnl,
            'positions_count': len(self.positions)
        }
    
    def _update_position(self, order: Order):
        """更新持仓 (内部方法)"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                size=0.0,
                avg_cost=0.0
            )
        
        pos = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # 买入
            total_cost = pos.size * pos.avg_cost + order.filled_size * order.filled_price
            pos.size += order.filled_size
            pos.avg_cost = total_cost / pos.size if pos.size > 0 else 0
            self.account_balance -= order.filled_size * order.filled_price
        else:
            # 卖出
            pos.size -= order.filled_size
            pos.realized_pnl += (order.filled_price - pos.avg_cost) * order.filled_size
            self.account_balance += order.filled_size * order.filled_price
            
            if pos.size <= 0:
                # 清仓
                del self.positions[symbol]
                return
        
        pos.update_time = datetime.now()


# ==================== 风险控制器 ====================

class RiskManager:
    """
    风险控制管理器
    
    检查项:
    1. 单笔订单限额
    2. 日内交易次数限制
    3. 仓位比例限制
    4. 最大回撤熔断
    5. 单票集中度控制
    6. 交易时段检查
    """
    
    def __init__(self, config: RiskConfig):
        """
        初始化风险控制器
        
        Args:
            config: 风险控制配置
        """
        self.config = config
        self.daily_trades = 0
        self.daily_turnover = 0.0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        logger.info("风险控制器初始化完成")
    
    def check_order(self, signal: TradingSignal, account_info: Dict) -> RiskCheckResult:
        """
        订单级风险检查
        
        Args:
            signal: 交易信号
            account_info: 账户信息
            
        Returns:
            result: 风险检查结果
        """
        # 重置日内计数器
        self._reset_daily_stats_if_needed()
        
        # 1. 交易时段检查
        if not self._is_trading_time():
            return RiskCheckResult(False, "非交易时段")
        
        # 2. 单笔限额检查
        order_value = signal.size * (signal.price or 10.0)
        if order_value > self.config.max_order_amount:
            return RiskCheckResult(
                False,
                f"超过单笔限额: {order_value:.2f} > {self.config.max_order_amount:.2f}"
            )
        
        # 3. 日内交易次数检查
        if self.daily_trades >= self.config.max_daily_trades:
            return RiskCheckResult(False, "超过日内交易次数限制")
        
        # 4. 日内亏损检查
        if self.daily_pnl < -self.config.max_daily_loss:
            return RiskCheckResult(
                False,
                f"超过日内最大亏损: {self.daily_pnl:.2f} < {-self.config.max_daily_loss:.2f}"
            )
        
        # 5. 账户余额检查 (买入时)
        if signal.side == OrderSide.BUY:
            available_balance = account_info.get('balance', 0)
            if order_value > available_balance:
                return RiskCheckResult(
                    False,
                    f"账户余额不足: 需要{order_value:.2f}, 可用{available_balance:.2f}"
                )
        
        # 全部通过
        return RiskCheckResult(True, "通过", risk_score=0.0)
    
    def check_risk_limit(self, positions: List[Position], account_info: Dict) -> bool:
        """
        组合级风险检查
        
        Args:
            positions: 持仓列表
            account_info: 账户信息
            
        Returns:
            triggered: 是否触发熔断 (True=触发)
        """
        # 1. 单票集中度检查
        if positions:
            total_value = account_info.get('total_asset', 1)
            for pos in positions:
                position_pct = pos.market_value / total_value
                if position_pct > self.config.max_position_pct:
                    logger.warning(
                        f"⚠️ 单票仓位过高: {pos.symbol} {position_pct:.2%} "
                        f"> {self.config.max_position_pct:.2%}"
                    )
                    return True
        
        # 2. 最大回撤检查
        current_drawdown = self._calculate_drawdown(account_info)
        if current_drawdown > self.config.max_drawdown_threshold:
            logger.critical(
                f"⚠️ 最大回撤超限: {current_drawdown:.2%} "
                f"> {self.config.max_drawdown_threshold:.2%}"
            )
            return True
        
        return False
    
    def _is_trading_time(self) -> bool:
        """检查是否在交易时段"""
        now = datetime.now().time()
        for start, end in self.config.trading_hours:
            if start <= now <= end:
                return True
        return False
    
    def _calculate_drawdown(self, account_info: Dict) -> float:
        """计算回撤"""
        # 简化计算: 基于日内盈亏
        total_pnl = account_info.get('total_pnl', 0)
        total_asset = account_info.get('total_asset', 1)
        drawdown = abs(min(total_pnl, 0)) / total_asset
        return drawdown
    
    def _reset_daily_stats_if_needed(self):
        """重置日内统计 (跨日)"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.daily_turnover = 0.0
            self.daily_pnl = 0.0
            self.last_reset_date = today
            logger.info("日内统计已重置")
    
    def record_trade(self, order: Order):
        """记录交易 (更新统计)"""
        self.daily_trades += 1
        self.daily_turnover += order.filled_size * (order.filled_price or 0)
        logger.debug(f"日内交易数: {self.daily_trades}, 成交额: {self.daily_turnover:.2f}")


# ==================== 订单管理系统 (OMS) ====================

class OrderManagementSystem:
    """
    订单管理系统 (OMS)
    
    功能:
    1. 订单生命周期管理
    2. 订单状态跟踪
    3. 订单持久化
    4. 订单查询和统计
    """
    
    def __init__(self):
        """初始化OMS"""
        self.orders: Dict[str, Order] = {}
        self.order_sequence = 0
        
        logger.info("订单管理系统(OMS)初始化完成")
    
    def create_order(self, signal: TradingSignal) -> Order:
        """
        创建订单
        
        Args:
            signal: 交易信号
            
        Returns:
            order: 订单对象
        """
        # 生成订单ID
        self.order_sequence += 1
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d')}_{self.order_sequence:06d}"
        
        # 确定订单类型
        order_type = OrderType.LIMIT if signal.price else OrderType.MARKET
        
        # 创建订单
        order = Order(
            order_id=order_id,
            symbol=signal.symbol,
            side=signal.side,
            order_type=order_type,
            size=signal.size,
            price=signal.price
        )
        
        # 保存订单
        self.orders[order_id] = order
        
        logger.info(f"创建订单: {order_id} {signal.symbol} {signal.side.value} {signal.size}股")
        
        return order
    
    def track_order(self, order: Order, result: OrderResult):
        """
        跟踪订单
        
        Args:
            order: 订单
            result: 执行结果
        """
        if result.success:
            logger.info(f"订单跟踪: {order.order_id} -> {order.status.value}")
        else:
            order.status = OrderStatus.FAILED
            logger.error(f"订单失败: {order.order_id} - {result.message}")
        
        order.update_time = datetime.now()
    
    def update_order_status(self, order_id: str, new_status: OrderStatus):
        """更新订单状态"""
        if order_id in self.orders:
            order = self.orders[order_id]
            old_status = order.status
            order.status = new_status
            order.update_time = datetime.now()
            logger.info(f"订单状态更新: {order_id} {old_status.value} -> {new_status.value}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """获取活跃订单 (未完成的订单)"""
        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILLED
        }
        return [
            order for order in self.orders.values()
            if order.status in active_statuses
        ]
    
    def get_statistics(self) -> Dict:
        """获取订单统计"""
        total_orders = len(self.orders)
        filled_orders = sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED)
        failed_orders = sum(1 for o in self.orders.values() if o.status == OrderStatus.FAILED)
        
        success_rate = filled_orders / total_orders if total_orders > 0 else 0
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'failed_orders': failed_orders,
            'success_rate': success_rate,
            'active_orders': len(self.get_active_orders())
        }


# ==================== 仓位监控器 ====================

class PositionMonitor:
    """
    仓位监控器
    
    功能:
    1. 实时仓位查询
    2. 盈亏监控
    3. 市值更新
    """
    
    def __init__(self, broker: BrokerAdapter):
        """
        初始化仓位监控器
        
        Args:
            broker: 券商适配器
        """
        self.broker = broker
        self.positions_cache: Dict[str, Position] = {}
        self.last_update_time = None
        
        logger.info("仓位监控器初始化完成")
    
    async def get_positions(self, force_refresh: bool = False) -> List[Position]:
        """
        获取持仓列表
        
        Args:
            force_refresh: 是否强制刷新 (默认使用缓存)
            
        Returns:
            positions: 持仓列表
        """
        # 使用缓存 (5秒内)
        if not force_refresh and self.last_update_time:
            elapsed = (datetime.now() - self.last_update_time).total_seconds()
            if elapsed < 5:
                return list(self.positions_cache.values())
        
        # 从券商获取最新持仓
        positions = await self.broker.get_positions()
        
        # 更新缓存
        self.positions_cache = {p.symbol: p for p in positions}
        self.last_update_time = datetime.now()
        
        return positions
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定股票的持仓"""
        positions = await self.get_positions()
        return self.positions_cache.get(symbol)


# ==================== 工厂函数 ====================

def create_live_trading_system(broker_config: Dict[str, Any]) -> 'LiveTradingSystem':
    """
    创建实盘交易系统 (工厂函数)
    
    Args:
        broker_config: 券商配置
            {
                'broker_name': str,  # 'mock', 'ptrade', 'qmt'
                ...
            }
    
    Returns:
        system: 实盘交易系统实例
    """
    # 创建券商适配器
    broker_name = broker_config.get('broker_name', 'mock')
    
    if broker_name == 'mock':
        broker = MockBrokerAdapter(broker_config)
    else:
        # 导入其他券商适配器
        try:
            from trading.broker_adapters import create_broker_adapter
            broker = create_broker_adapter(broker_name, broker_config)
        except ImportError:
            logger.warning(f"未找到{broker_name}适配器,使用Mock适配器")
            broker = MockBrokerAdapter(broker_config)
    
    # 创建风险控制器
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    
    # 创建仓位监控器
    position_monitor = PositionMonitor(broker)
    
    # 创建交易系统
    system = LiveTradingSystem(broker, risk_manager, position_monitor)
    
    return system


# ==================== 实盘交易系统 (主类) ====================

class LiveTradingSystem:
    """
    实盘交易系统
    
    三级架构:
    1. 信号接收层
    2. 风险控制层
    3. 订单执行层
    
    核心流程:
    信号 -> 风险检查 -> 订单生成 -> 券商提交 -> 监控跟踪
    """
    
    def __init__(
        self,
        broker_adapter: BrokerAdapter,
        risk_manager: RiskManager,
        position_monitor: PositionMonitor
    ):
        """
        初始化实盘交易系统
        
        Args:
            broker_adapter: 券商适配器
            risk_manager: 风险控制器
            position_monitor: 仓位监控器
        """
        self.broker = broker_adapter
        self.risk_mgr = risk_manager
        self.position_mon = position_monitor
        
        # 订单管理系统
        self.oms = OrderManagementSystem()
        
        # 运行状态
        self.is_running = False
        self.circuit_breaker_triggered = False
        
        logger.info("✅ 实盘交易系统初始化完成")
    
    async def start(self):
        """启动实盘交易系统"""
        logger.info("🚀 实盘交易系统启动...")
        
        # 连接券商
        connected = await self.broker.connect()
        if not connected:
            logger.error("❌ 券商连接失败,系统启动中止")
            return False
        
        self.is_running = True
        self.circuit_breaker_triggered = False
        
        # 启动监控循环
        asyncio.create_task(self._monitor_loop())
        
        logger.info("✅ 实盘交易系统运行中")
        return True
    
    async def stop(self):
        """停止实盘交易系统"""
        logger.info("⏹️ 实盘交易系统停止...")
        
        self.is_running = False
        
        # 断开券商连接
        await self.broker.disconnect()
        
        logger.info("✅ 实盘交易系统已停止")
    
    async def process_signal(self, signal: TradingSignal) -> OrderResult:
        """
        处理交易信号
        
        流程:
        1. 信号验证
        2. 风险检查
        3. 生成订单
        4. 提交执行
        5. 监控成交
        
        Args:
            signal: 交易信号
            
        Returns:
            result: 执行结果
        """
        if not self.is_running:
            return OrderResult(False, message="系统未运行")
        
        if self.circuit_breaker_triggered:
            return OrderResult(False, message="熔断触发,交易暂停")
        
        # 1. 信号验证
        if not self._validate_signal(signal):
            return OrderResult(False, message="信号验证失败")
        
        # 2. 风险检查
        account_info = await self.broker.get_account_info()
        risk_check = self.risk_mgr.check_order(signal, account_info)
        
        if not risk_check.passed:
            logger.warning(f"⚠️ 风险检查未通过: {risk_check.reason}")
            return OrderResult(False, message=risk_check.reason)
        
        # 3. 生成订单
        order = self.oms.create_order(signal)
        
        # 4. 提交到券商
        try:
            result = await self.broker.submit_order(order)
            
            # 5. 记录和监控
            self.oms.track_order(order, result)
            
            if result.success:
                self.risk_mgr.record_trade(order)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 订单提交失败: {e}")
            return OrderResult(False, message=str(e))
    
    async def _monitor_loop(self):
        """监控循环"""
        logger.info("监控循环启动")
        
        while self.is_running:
            try:
                # 1. 检查仓位
                positions = await self.position_mon.get_positions(force_refresh=True)
                
                # 2. 获取账户信息
                account_info = await self.broker.get_account_info()
                
                # 3. 风险检查
                if self.risk_mgr.check_risk_limit(positions, account_info):
                    self._trigger_circuit_breaker()
                
                # 4. 更新订单状态 (活跃订单)
                await self._update_active_orders()
                
                # 等待5秒
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(10)
    
    async def _update_active_orders(self):
        """更新活跃订单状态"""
        active_orders = self.oms.get_active_orders()
        
        for order in active_orders:
            try:
                # 从券商查询订单状态
                updated_order = await self.broker.get_order_status(order.order_id)
                if updated_order and updated_order.status != order.status:
                    self.oms.update_order_status(order.order_id, updated_order.status)
            except Exception as e:
                logger.error(f"更新订单状态失败 {order.order_id}: {e}")
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """验证交易信号"""
        # 基本验证
        if not signal.symbol or signal.size <= 0:
            return False
        
        # 置信度检查
        if signal.confidence < 0.5:
            logger.warning(f"信号置信度过低: {signal.confidence}")
            return False
        
        return True
    
    def _trigger_circuit_breaker(self):
        """触发熔断"""
        if not self.circuit_breaker_triggered:
            self.circuit_breaker_triggered = True
            logger.critical("🔴 熔断触发!交易暂停!")
            # TODO: 发送告警通知 (邮件/短信/微信)
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        oms_stats = self.oms.get_statistics()
        
        return {
            'is_running': self.is_running,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'broker_connected': self.broker.connected,
            'orders_stats': oms_stats,
            'daily_trades': self.risk_mgr.daily_trades
        }


# ==================== 便捷创建函数 ====================

def create_live_trading_system(
    broker_config: Optional[Dict] = None,
    risk_config: Optional[RiskConfig] = None
) -> LiveTradingSystem:
    """
    创建实盘交易系统的便捷函数
    
    Args:
        broker_config: 券商配置 (None=使用模拟券商)
        risk_config: 风险控制配置
        
    Returns:
        system: 实盘交易系统
    """
    # 默认使用模拟券商
    if broker_config is None:
        broker_config = {
            'broker_name': '模拟券商',
            'initial_cash': 1000000.0
        }
    
    broker = MockBrokerAdapter(broker_config)
    
    # 默认风险配置
    if risk_config is None:
        risk_config = RiskConfig()
    
    risk_mgr = RiskManager(risk_config)
    position_mon = PositionMonitor(broker)
    
    return LiveTradingSystem(broker, risk_mgr, position_mon)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_live_trading():
        """测试实盘交易系统"""
        print("=" * 60)
        print("测试: 实盘交易系统")
        print("=" * 60)
        
        # 1. 创建系统
        system = create_live_trading_system()
        
        # 2. 启动系统
        await system.start()
        
        # 3. 发送测试信号
        signals = [
            TradingSignal("000001.SZ", OrderSide.BUY, 1000, 10.0),
            TradingSignal("000002.SZ", OrderSide.BUY, 2000, 15.0),
            TradingSignal("000001.SZ", OrderSide.SELL, 500, 10.5),
        ]
        
        print("\n发送交易信号:")
        for i, signal in enumerate(signals, 1):
            print(f"  信号{i}: {signal.symbol} {signal.side.value} {signal.size}股 @{signal.price}")
            result = await system.process_signal(signal)
            print(f"  结果: {'✅成功' if result.success else '❌失败'} - {result.message}")
            await asyncio.sleep(0.5)
        
        # 4. 等待订单处理
        await asyncio.sleep(2)
        
        # 5. 查询系统状态
        status = system.get_status()
        print("\n系统状态:")
        print(f"  运行中: {status['is_running']}")
        print(f"  券商连接: {status['broker_connected']}")
        print(f"  订单统计: {status['orders_stats']}")
        print(f"  日内交易: {status['daily_trades']}笔")
        
        # 6. 查询持仓
        positions = await system.position_mon.get_positions()
        print(f"\n持仓情况 ({len(positions)}个):")
        for pos in positions:
            print(f"  {pos.symbol}: {pos.size}股 @{pos.avg_cost:.2f}")
        
        # 7. 查询账户
        account = await system.broker.get_account_info()
        print("\n账户信息:")
        print(f"  可用资金: {account['balance']:.2f}")
        print(f"  持仓市值: {account['market_value']:.2f}")
        print(f"  总资产: {account['total_asset']:.2f}")
        
        # 8. 停止系统
        await system.stop()
        
        print("\n✅ 测试完成!")
    
    # 运行测试
    asyncio.run(test_live_trading())
