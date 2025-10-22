"""
交易执行模块
处理订单生成、执行、状态跟踪，支持多种交易接口
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP = "stop"  # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 跟踪止损单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"  # 待提交
    SUBMITTED = "submitted"  # 已提交
    PARTIAL = "partial"  # 部分成交
    FILLED = "filled"  # 完全成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 已拒绝
    EXPIRED = "expired"  # 已过期


class TimeInForce(Enum):
    """订单有效期"""
    GTC = "gtc"  # Good Till Cancel
    GTD = "gtd"  # Good Till Date
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    DAY = "day"  # 当日有效


@dataclass
class Order:
    """订单数据结构"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    avg_fill_price: float = 0
    commission: float = 0
    slippage: float = 0
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expire_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'slippage': self.slippage,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'metadata': self.metadata
        }


@dataclass
class Execution:
    """成交记录"""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: datetime
    metadata: Dict = field(default_factory=dict)


class BrokerInterface:
    """券商接口基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connected = False
        
    async def connect(self):
        """连接券商"""
        raise NotImplementedError
    
    async def disconnect(self):
        """断开连接"""
        raise NotImplementedError
    
    async def submit_order(self, order: Order) -> str:
        """提交订单"""
        raise NotImplementedError
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        raise NotImplementedError
    
    async def get_order_status(self, order_id: str) -> Order:
        """获取订单状态"""
        raise NotImplementedError
    
    async def get_positions(self) -> List[Position]:
        """获取持仓"""
        raise NotImplementedError
    
    async def get_account_info(self) -> Dict:
        """获取账户信息"""
        raise NotImplementedError


class SimulatedBroker(BrokerInterface):
    """模拟券商接口"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.executions: List[Execution] = []
        self.account = {
            'cash': config.get('initial_capital', 1000000),
            'buying_power': config.get('initial_capital', 1000000),
            'total_value': config.get('initial_capital', 1000000)
        }
        self.market_data: Dict[str, Dict] = {}  # 模拟市场数据
        
    async def connect(self):
        """连接模拟券商"""
        self.connected = True
        logger.info("模拟券商已连接")
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
        logger.info("模拟券商已断开")
    
    async def submit_order(self, order: Order) -> str:
        """提交订单"""
        if not self.connected:
            raise ConnectionError("未连接到券商")
        
        # 生成订单ID
        order.order_id = f"SIM_{uuid.uuid4().hex[:8]}"
        order.client_order_id = order.client_order_id or f"CLIENT_{uuid.uuid4().hex[:8]}"
        order.created_at = datetime.now()
        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        
        # 保存订单
        self.orders[order.order_id] = order
        
        # 模拟执行
        await self._simulate_execution(order)
        
        return order.order_id
    
    async def _simulate_execution(self, order: Order):
        """模拟订单执行"""
        # 获取市场价格（模拟）
        market_price = self._get_market_price(order.symbol)
        
        # 检查是否可以执行
        can_execute = False
        execution_price = market_price
        
        if order.order_type == OrderType.MARKET:
            can_execute = True
            # 添加滑点
            if order.side == OrderSide.BUY:
                execution_price *= (1 + self.config.get('slippage', 0.001))
            else:
                execution_price *= (1 - self.config.get('slippage', 0.001))
                
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and market_price <= order.price:
                can_execute = True
                execution_price = order.price
            elif order.side == OrderSide.SELL and market_price >= order.price:
                can_execute = True
                execution_price = order.price
        
        if can_execute:
            # 计算手续费
            commission = order.quantity * execution_price * self.config.get('commission_rate', 0.0003)
            commission = max(commission, self.config.get('min_commission', 5))
            
            # 更新订单
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.avg_fill_price = execution_price
            order.commission = commission
            order.slippage = abs(execution_price - market_price) * order.quantity
            order.filled_at = datetime.now()
            
            # 创建成交记录
            execution = Execution(
                execution_id=f"EXEC_{uuid.uuid4().hex[:8]}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                commission=commission,
                timestamp=datetime.now()
            self.executions.append(execution)
            
            # 更新持仓
            await self._update_position(order, execution)
            
            # 更新账户
            await self._update_account(order, execution)
            
            logger.info(f"订单执行: {order.symbol} {order.side.value} {order.quantity}@{execution_price:.2f}")
    
    def _get_market_price(self, symbol: str) -> float:
        """获取市场价格（模拟）"""
        # 从市场数据中获取或生成随机价格
        if symbol in self.market_data:
            return self.market_data[symbol].get('last', 100)
        return 100 + np.random.randn() * 5  # 模拟价格
    
    async def _update_position(self, order: Order, execution: Execution):
        """更新持仓"""
        symbol = order.symbol
        
        if order.side == OrderSide.BUY:
            if symbol in self.positions:
                position = self.positions[symbol]
                total_cost = position.avg_cost * position.quantity + execution.price * execution.quantity
                position.quantity += execution.quantity
                position.avg_cost = total_cost / position.quantity
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=execution.quantity,
                    avg_cost=execution.price,
                    current_price=execution.price,
                    market_value=execution.quantity * execution.price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    last_update=datetime.now()
        else:  # SELL
            if symbol in self.positions:
                position = self.positions[symbol]
                position.quantity -= execution.quantity
                
                # 计算已实现盈亏
                realized_pnl = (execution.price - position.avg_cost) * execution.quantity
                position.realized_pnl += realized_pnl
                
                if position.quantity <= 0:
                    del self.positions[symbol]
    
    async def _update_account(self, order: Order, execution: Execution):
        """更新账户"""
        total_value = execution.price * execution.quantity + execution.commission
        
        if order.side == OrderSide.BUY:
            self.account['cash'] -= total_value
        else:
            self.account['cash'] += (execution.price * execution.quantity - execution.commission)
        
        # 更新总值
        positions_value = sum(pos.market_value for pos in self.positions.values())
        self.account['total_value'] = self.account['cash'] + positions_value
        self.account['buying_power'] = self.account['cash']  # 简化处理
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now()
                return True
        return False
    
    async def get_order_status(self, order_id: str) -> Order:
        """获取订单状态"""
        return self.orders.get(order_id)
    
    async def get_positions(self) -> List[Position]:
        """获取持仓"""
        return list(self.positions.values())
    
    async def get_account_info(self) -> Dict:
        """获取账户信息"""
        return self.account.copy()


class OrderManager:
    """订单管理器"""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.execution_history: List[Execution] = []
        self.order_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    async def submit_order(self, order: Order, callback: Optional[Callable] = None) -> str:
        """提交订单"""
        try:
            # 提交到券商
            order_id = await self.broker.submit_order(order)
            order.order_id = order_id
            
            # 保存订单
            self.active_orders[order_id] = order
            
            # 注册回调
            if callback:
                self.order_callbacks[order_id].append(callback)
            
            logger.info(f"订单已提交: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"订单提交失败: {str(e)}")
            order.status = OrderStatus.REJECTED
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            success = await self.broker.cancel_order(order_id)
            if success and order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                self._move_to_history(order_id)
            return success
            
        except Exception as e:
            logger.error(f"订单取消失败: {str(e)}")
            return False
    
    async def update_order_status(self, order_id: str):
        """更新订单状态"""
        try:
            updated_order = await self.broker.get_order_status(order_id)
            
            if updated_order and order_id in self.active_orders:
                self.active_orders[order_id] = updated_order
                
                # 检查是否完成
                if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    self._move_to_history(order_id)
                    
                    # 触发回调
                    await self._trigger_callbacks(order_id, updated_order)
                    
        except Exception as e:
            logger.error(f"更新订单状态失败: {str(e)}")
    
    def _move_to_history(self, order_id: str):
        """将订单移至历史"""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            self.order_history.append(order)
    
    async def _trigger_callbacks(self, order_id: str, order: Order):
        """触发订单回调"""
        if order_id in self.order_callbacks:
            for callback in self.order_callbacks[order_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order)
                    else:
                        callback(order)
                except Exception as e:
                    logger.error(f"回调执行失败: {str(e)}")
            
            # 清理回调
            del self.order_callbacks[order_id]
    
    def get_active_orders(self) -> List[Order]:
        """获取活跃订单"""
        return list(self.active_orders.values())
    
    def get_order_history(self) -> List[Order]:
        """获取历史订单"""
        return self.order_history.copy()


class ExecutionEngine:
    """交易执行引擎"""
    
    def __init__(self, broker_type: str = "simulated", config: Dict = None):
        """
        初始化执行引擎
        
        Args:
            broker_type: 券商类型
            config: 配置参数
        """
        self.config = config or {}
        
        # 创建券商接口
        if broker_type == "simulated":
            self.broker = SimulatedBroker(self.config)
        else:
            raise ValueError(f"不支持的券商类型: {broker_type}")
        
        # 创建订单管理器
        self.order_manager = OrderManager(self.broker)
        
        # 执行策略
        self.execution_strategies: Dict[str, Callable] = {}
        self.register_default_strategies()
        
        # 风险检查
        self.risk_checks: List[Callable] = []
        self.register_default_risk_checks()
        
        # 性能监控
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_commission': 0,
            'total_slippage': 0
        }
        
        # 执行状态
        self.running = False
        self.execution_thread = None
        
    async def connect(self):
        """连接执行引擎"""
        await self.broker.connect()
        self.running = True
        logger.info("执行引擎已启动")
    
    async def disconnect(self):
        """断开执行引擎"""
        self.running = False
        await self.broker.disconnect()
        logger.info("执行引擎已停止")
    
    def register_execution_strategy(self, name: str, strategy: Callable):
        """注册执行策略"""
        self.execution_strategies[name] = strategy
    
    def register_risk_check(self, check: Callable):
        """注册风险检查"""
        self.risk_checks.append(check)
    
    def register_default_strategies(self):
        """注册默认执行策略"""
        # VWAP策略
        self.execution_strategies['vwap'] = self._vwap_strategy
        # TWAP策略
        self.execution_strategies['twap'] = self._twap_strategy
        # 冰山策略
        self.execution_strategies['iceberg'] = self._iceberg_strategy
        # 智能路由
        self.execution_strategies['smart'] = self._smart_routing_strategy
    
    def register_default_risk_checks(self):
        """注册默认风险检查"""
        self.risk_checks.append(self._check_position_limit)
        self.risk_checks.append(self._check_order_size)
        self.risk_checks.append(self._check_buying_power)
    
    async def execute_order(self, 
                           symbol: str,
                           side: OrderSide,
                           quantity: float,
                           order_type: OrderType = OrderType.MARKET,
                           price: Optional[float] = None,
                           strategy: str = "direct",
                           **kwargs) -> str:
        """
        执行订单
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            order_type: 订单类型
            price: 价格（限价单）
            strategy: 执行策略
            **kwargs: 额外参数
        
        Returns:
            订单ID
        """
        # 创建订单
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            metadata=kwargs
        
        # 风险检查
        for check in self.risk_checks:
            if not await check(order):
                logger.warning(f"订单未通过风险检查: {order.symbol}")
                order.status = OrderStatus.REJECTED
                return None
        
        # 选择执行策略
        if strategy == "direct":
            # 直接执行
            order_id = await self.order_manager.submit_order(order)
        elif strategy in self.execution_strategies:
            # 使用指定策略
            order_id = await self.execution_strategies[strategy](order)
        else:
            raise ValueError(f"未知的执行策略: {strategy}")
        
        # 更新统计
        self.execution_metrics['total_orders'] += 1
        
        return order_id
    
    async def _vwap_strategy(self, order: Order) -> str:
        """VWAP执行策略"""
        # 将大单分割成多个小单，按照成交量加权平均价格执行
        total_quantity = order.quantity
        slice_size = min(total_quantity / 10, 1000)  # 每单最多1000股
        
        order_ids = []
        remaining = total_quantity
        
        while remaining > 0:
            current_size = min(slice_size, remaining)
            
            # 创建子订单
            sub_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT if order.price else OrderType.MARKET,
                quantity=current_size,
                price=order.price,
                metadata={'parent_order': order.metadata.get('parent_id')}
            
            # 提交子订单
            order_id = await self.order_manager.submit_order(sub_order)
            order_ids.append(order_id)
            
            remaining -= current_size
            
            # 等待一段时间
            await asyncio.sleep(1)
        
        return ','.join(order_ids)
    
    async def _twap_strategy(self, order: Order) -> str:
        """TWAP执行策略"""
        # 时间加权平均价格策略
        total_quantity = order.quantity
        duration = order.metadata.get('duration', 300)  # 默认5分钟
        slices = order.metadata.get('slices', 10)  # 默认分10次
        
        slice_size = total_quantity / slices
        interval = duration / slices
        
        order_ids = []
        
        for i in range(slices):
            # 创建子订单
            sub_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=slice_size,
                price=order.price,
                metadata={'parent_order': order.metadata.get('parent_id'), 'slice': i+1}
            
            # 提交子订单
            order_id = await self.order_manager.submit_order(sub_order)
            order_ids.append(order_id)
            
            # 等待
            if i < slices - 1:
                await asyncio.sleep(interval)
        
        return ','.join(order_ids)
    
    async def _iceberg_strategy(self, order: Order) -> str:
        """冰山订单策略"""
        # 只显示部分数量，隐藏大部分订单量
        total_quantity = order.quantity
        visible_quantity = order.metadata.get('visible_quantity', total_quantity * 0.1)
        
        order_ids = []
        remaining = total_quantity
        
        while remaining > 0:
            current_size = min(visible_quantity, remaining)
            
            # 创建可见订单
            sub_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=current_size,
                price=order.price or self._get_limit_price(order.symbol, order.side),
                metadata={'iceberg': True, 'total': total_quantity}
            
            # 提交订单
            order_id = await self.order_manager.submit_order(sub_order)
            order_ids.append(order_id)
            
            # 等待成交
            await self._wait_for_fill(order_id)
            
            remaining -= current_size
        
        return ','.join(order_ids)
    
    async def _smart_routing_strategy(self, order: Order) -> str:
        """智能路由策略"""
        # 根据市场情况选择最佳执行方式
        quantity = order.quantity
        
        # 小单直接执行
        if quantity < 1000:
            return await self.order_manager.submit_order(order)
        
        # 中等订单使用VWAP
        elif quantity < 10000:
            return await self._vwap_strategy(order)
        
        # 大单使用TWAP或冰山
        else:
            if order.metadata.get('aggressive', False):
                return await self._twap_strategy(order)
            else:
                return await self._iceberg_strategy(order)
    
    def _get_limit_price(self, symbol: str, side: OrderSide) -> float:
        """获取限价（模拟）"""
        # 实际应该从市场数据获取
        base_price = 100
        if side == OrderSide.BUY:
            return base_price * 0.99  # 买单略低于市价
        else:
            return base_price * 1.01  # 卖单略高于市价
    
    async def _wait_for_fill(self, order_id: str, timeout: float = 30):
        """等待订单成交"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            await self.order_manager.update_order_status(order_id)
            
            if order_id not in self.order_manager.active_orders:
                # 订单已完成
                return
            
            await asyncio.sleep(1)
        
        # 超时取消
        await self.order_manager.cancel_order(order_id)
    
    async def _check_position_limit(self, order: Order) -> bool:
        """检查持仓限制"""
        max_position = self.config.get('max_position_size', 100000)
        positions = await self.broker.get_positions()
        
        for position in positions:
            if position.symbol == order.symbol:
                if order.side == OrderSide.BUY:
                    new_quantity = position.quantity + order.quantity
                else:
                    new_quantity = position.quantity - order.quantity
                
                if abs(new_quantity) > max_position:
                    logger.warning(f"订单超过持仓限制: {order.symbol}")
                    return False
        
        return True
    
    async def _check_order_size(self, order: Order) -> bool:
        """检查订单大小"""
        max_order_size = self.config.get('max_order_size', 10000)
        min_order_size = self.config.get('min_order_size', 100)
        
        if order.quantity > max_order_size:
            logger.warning(f"订单过大: {order.quantity} > {max_order_size}")
            return False
        
        if order.quantity < min_order_size:
            logger.warning(f"订单过小: {order.quantity} < {min_order_size}")
            return False
        
        return True
    
    async def _check_buying_power(self, order: Order) -> bool:
        """检查购买力"""
        if order.side == OrderSide.BUY:
            account = await self.broker.get_account_info()
            required = order.quantity * (order.price or 100)  # 估算所需资金
            
            if required > account.get('buying_power', 0):
                logger.warning(f"购买力不足: 需要{required}, 可用{account.get('buying_power', 0)}")
                return False
        
        return True
    
    def get_execution_report(self) -> Dict:
        """获取执行报告"""
        active_orders = self.order_manager.get_active_orders()
        history = self.order_manager.get_order_history()
        
        # 计算统计
        filled_orders = [o for o in history if o.status == OrderStatus.FILLED]
        cancelled_orders = [o for o in history if o.status == OrderStatus.CANCELLED]
        rejected_orders = [o for o in history if o.status == OrderStatus.REJECTED]
        
        total_commission = sum(o.commission for o in filled_orders)
        total_slippage = sum(o.slippage for o in filled_orders)
        
        return {
            'summary': {
                'total_orders': len(history) + len(active_orders),
                'active_orders': len(active_orders),
                'filled_orders': len(filled_orders),
                'cancelled_orders': len(cancelled_orders),
                'rejected_orders': len(rejected_orders),
                'fill_rate': len(filled_orders) / (len(history) + len(active_orders)) if history or active_orders else 0,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'avg_commission': total_commission / len(filled_orders) if filled_orders else 0,
                'avg_slippage': total_slippage / len(filled_orders) if filled_orders else 0
            },
            'active_orders': [o.to_dict() for o in active_orders],
            'recent_orders': [o.to_dict() for o in history[-10:]],  # 最近10个订单
            'metrics': self.execution_metrics
        }


if __name__ == "__main__":
    async def main():
        # 创建执行引擎
        config = {
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'slippage': 0.001,
            'min_commission': 5,
            'max_position_size': 10000,
            'max_order_size': 5000,
            'min_order_size': 100
        }
        
        engine = ExecutionEngine(broker_type="simulated", config=config)
        
        # 连接引擎
        await engine.connect()
        
        try:
            # 执行一些订单
            # 市价单
            order_id1 = await engine.execute_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=1000,
                order_type=OrderType.MARKET
            print(f"市价单ID: {order_id1}")
            
            # 限价单
            order_id2 = await engine.execute_order(
                symbol="GOOGL",
                side=OrderSide.BUY,
                quantity=500,
                order_type=OrderType.LIMIT,
                price=2500
            print(f"限价单ID: {order_id2}")
            
            # 使用VWAP策略的大单
            order_id3 = await engine.execute_order(
                symbol="MSFT",
                side=OrderSide.BUY,
                quantity=3000,
                strategy="vwap"
            print(f"VWAP订单ID: {order_id3}")
            
            # 等待一会儿
            await asyncio.sleep(5)
            
            # 获取执行报告
            report = engine.get_execution_report()
            print("\n执行报告:")
            print(json.dumps(report['summary'], indent=2, ensure_ascii=False))
            
            # 获取账户信息
            account = await engine.broker.get_account_info()
            print("\n账户信息:")
            print(json.dumps(account, indent=2, ensure_ascii=False))
            
            # 获取持仓
            positions = await engine.broker.get_positions()
            print("\n持仓信息:")
            for position in positions:
                print(f"  {position.symbol}: {position.quantity}股 @ {position.avg_cost:.2f}")
            
        finally:
            # 断开连接
            await engine.disconnect()
    
    # 运行示例
    # asyncio.run(main())