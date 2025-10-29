"""
麒麟量化系统 - 交易执行接口
支持自动买入和卖出操作
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "市价单"
    LIMIT = "限价单"
    OPEN_LIMIT = "开盘价限价"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "待成交"
    FILLED = "已成交"
    PARTIAL = "部分成交"
    CANCELLED = "已撤销"
    FAILED = "失败"


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    name: str
    direction: str           # "buy" / "sell"
    order_type: OrderType
    price: Optional[float]   # 限价单价格
    volume: int              # 股数
    status: OrderStatus
    filled_volume: int       # 已成交数量
    filled_price: float      # 成交价格
    commission: float        # 手续费
    timestamp: str
    reason: str              # 买入/卖出理由


class TradingExecutor:
    """交易执行器"""
    
    def __init__(
        self, 
        broker: str = "同花顺",
        account_balance: float = 100000,
        max_position_per_stock: float = 0.2,
        max_total_position: float = 0.95,
        enable_real_trading: bool = False
    ):
        """
        初始化交易执行器
        
        Args:
            broker: 券商名称
            account_balance: 账户余额
            max_position_per_stock: 单只股票最大仓位比例
            max_total_position: 总仓位上限
            enable_real_trading: 是否启用真实交易 (False为模拟)
        """
        self.broker = broker
        self.account_balance = account_balance
        self.available_cash = account_balance
        self.max_position_per_stock = max_position_per_stock
        self.max_total_position = max_total_position
        self.enable_real_trading = enable_real_trading
        
        # 持仓
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # 订单历史
        self.orders: List[Order] = []
        self.order_counter = 0
        
        # 交易记录
        self.trade_history = []
        
        logger.info(f"交易执行器初始化完成 (模式: {'真实' if enable_real_trading else '模拟'})")
        logger.info(f"账户余额: {account_balance:.2f}, 单股仓位上限: {max_position_per_stock:.1%}")
    
    def calculate_position_size(
        self, 
        symbol: str, 
        price: float,
        rl_score: float
    ) -> int:
        """
        计算仓位大小
        
        Args:
            symbol: 股票代码
            price: 当前价格
            rl_score: RL得分 (0-100)
            
        Returns:
            股数 (100的倍数)
        """
        # 基于RL得分动态调整仓位
        # 得分越高,仓位越大
        score_ratio = min(rl_score / 100, 1.0)
        
        # 单股最大金额
        max_amount = self.account_balance * self.max_position_per_stock * score_ratio
        
        # 可用资金限制
        max_amount = min(max_amount, self.available_cash)
        
        # 计算股数 (100的倍数)
        volume = int(max_amount / price / 100) * 100
        
        return max(volume, 100)  # 最少1手
    
    def buy(
        self, 
        symbol: str, 
        name: str,
        price: float,
        rl_score: float,
        reason: str,
        order_type: OrderType = OrderType.OPEN_LIMIT
    ) -> Optional[Order]:
        """
        买入股票
        
        Args:
            symbol: 股票代码
            name: 股票名称
            price: 买入价格
            rl_score: RL得分
            reason: 买入理由
            order_type: 订单类型
            
        Returns:
            订单对象
        """
        # 检查是否已持仓
        if symbol in self.positions:
            logger.warning(f"{symbol} 已持仓,跳过买入")
            return None
        
        # 计算仓位
        volume = self.calculate_position_size(symbol, price, rl_score)
        
        if volume < 100:
            logger.warning(f"{symbol} 可用资金不足,无法买入")
            return None
        
        # 计算成本
        cost = price * volume
        commission = cost * 0.0003  # 万三佣金
        total_cost = cost + commission
        
        if total_cost > self.available_cash:
            logger.warning(f"{symbol} 资金不足: 需要 {total_cost:.2f}, 可用 {self.available_cash:.2f}")
            return None
        
        # 创建订单
        self.order_counter += 1
        order = Order(
            order_id=f"BUY_{self.order_counter:06d}",
            symbol=symbol,
            name=name,
            direction="buy",
            order_type=order_type,
            price=price,
            volume=volume,
            status=OrderStatus.PENDING,
            filled_volume=0,
            filled_price=0,
            commission=0,
            timestamp=datetime.now().isoformat(),
            reason=reason
        )
        
        # 执行订单
        if self.enable_real_trading:
            # TODO: 对接真实券商API
            # success = self._execute_real_order(order)
            logger.warning("真实交易未实现,自动切换到模拟模式")
            success = self._execute_simulated_order(order)
        else:
            success = self._execute_simulated_order(order)
        
        if success:
            # 更新持仓
            self.positions[symbol] = {
                "name": name,
                "volume": order.filled_volume,
                "cost_price": order.filled_price,
                "current_price": order.filled_price,
                "cost": order.filled_price * order.filled_volume,
                "market_value": order.filled_price * order.filled_volume,
                "profit": 0,
                "profit_rate": 0,
                "buy_date": datetime.now().strftime("%Y-%m-%d"),
                "buy_reason": reason,
                "rl_score": rl_score
            }
            
            # 更新可用资金
            self.available_cash -= (order.filled_price * order.filled_volume + order.commission)
            
            logger.info(
                f"✅ 买入成功: {symbol} {name}, "
                f"价格 {order.filled_price:.2f}, "
                f"数量 {order.filled_volume}, "
                f"成本 {order.filled_price * order.filled_volume:.2f}, "
                f"剩余资金 {self.available_cash:.2f}"
            )
        else:
            logger.error(f"❌ 买入失败: {symbol} {name}")
        
        self.orders.append(order)
        return order
    
    def sell(
        self, 
        symbol: str,
        reason: str,
        price: Optional[float] = None,
        volume: Optional[int] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> Optional[Order]:
        """
        卖出股票
        
        Args:
            symbol: 股票代码
            reason: 卖出理由
            price: 卖出价格 (None为市价)
            volume: 卖出数量 (None为全部)
            order_type: 订单类型
            
        Returns:
            订单对象
        """
        # 检查持仓
        if symbol not in self.positions:
            logger.warning(f"{symbol} 未持仓,无法卖出")
            return None
        
        position = self.positions[symbol]
        sell_volume = volume or position["volume"]
        
        if sell_volume > position["volume"]:
            logger.warning(f"{symbol} 卖出数量超过持仓")
            sell_volume = position["volume"]
        
        # 创建订单
        self.order_counter += 1
        order = Order(
            order_id=f"SELL_{self.order_counter:06d}",
            symbol=symbol,
            name=position["name"],
            direction="sell",
            order_type=order_type,
            price=price or position["current_price"],
            volume=sell_volume,
            status=OrderStatus.PENDING,
            filled_volume=0,
            filled_price=0,
            commission=0,
            timestamp=datetime.now().isoformat(),
            reason=reason
        )
        
        # 执行订单
        if self.enable_real_trading:
            # TODO: 对接真实券商API
            success = self._execute_simulated_order(order)
        else:
            success = self._execute_simulated_order(order)
        
        if success:
            # 计算盈亏
            profit = (order.filled_price - position["cost_price"]) * order.filled_volume - order.commission
            profit_rate = profit / (position["cost_price"] * order.filled_volume)
            
            # 更新资金
            self.available_cash += order.filled_price * order.filled_volume - order.commission
            
            # 更新持仓
            position["volume"] -= order.filled_volume
            if position["volume"] == 0:
                del self.positions[symbol]
            
            # 记录交易
            self.trade_history.append({
                "symbol": symbol,
                "name": position["name"],
                "buy_price": position["cost_price"],
                "sell_price": order.filled_price,
                "volume": order.filled_volume,
                "profit": profit,
                "profit_rate": profit_rate,
                "buy_date": position["buy_date"],
                "sell_date": datetime.now().strftime("%Y-%m-%d"),
                "buy_reason": position["buy_reason"],
                "sell_reason": reason
            })
            
            logger.info(
                f"✅ 卖出成功: {symbol} {position['name']}, "
                f"价格 {order.filled_price:.2f}, "
                f"数量 {order.filled_volume}, "
                f"盈亏 {profit:.2f} ({profit_rate:.2%}), "
                f"可用资金 {self.available_cash:.2f}"
            )
        else:
            logger.error(f"❌ 卖出失败: {symbol}")
        
        self.orders.append(order)
        return order
    
    def _execute_simulated_order(self, order: Order) -> bool:
        """
        模拟执行订单
        
        Args:
            order: 订单
            
        Returns:
            是否成功
        """
        # 模拟成交 (简化版)
        # 实际应考虑滑点、成交率等
        
        if order.order_type == OrderType.OPEN_LIMIT:
            # 开盘价限价单: 假设以开盘价+1%成交
            filled_price = order.price * 1.01
        elif order.order_type == OrderType.LIMIT:
            # 限价单: 假设以限价成交
            filled_price = order.price
        else:
            # 市价单: 假设以市价成交
            filled_price = order.price
        
        order.filled_volume = order.volume
        order.filled_price = filled_price
        order.commission = filled_price * order.volume * 0.0003
        order.status = OrderStatus.FILLED
        
        return True
    
    def batch_buy(self, selected_stocks: List[Dict[str, Any]]) -> List[Order]:
        """
        批量买入
        
        Args:
            selected_stocks: 选中的股票列表 (来自RL决策)
            
        Returns:
            订单列表
        """
        logger.info("=" * 60)
        logger.info("💰 开始批量买入...")
        logger.info("=" * 60)
        
        orders = []
        
        for stock in selected_stocks:
            # 检查总仓位
            current_position_ratio = (
                self.account_balance - self.available_cash
            ) / self.account_balance
            
            if current_position_ratio >= self.max_total_position:
                logger.warning(f"总仓位已达上限 {self.max_total_position:.1%}, 停止买入")
                break
            
            # 使用竞价终盘价作为买入价
            price = stock["auction_info"]["final_price"]
            
            order = self.buy(
                symbol=stock["symbol"],
                name=stock["name"],
                price=price,
                rl_score=stock["rl_score"],
                reason=f"RL得分 {stock['rl_score']:.2f}, 竞价强度 {stock['auction_info']['strength']:.1f}",
                order_type=OrderType.OPEN_LIMIT
            )
            
            if order:
                orders.append(order)
        
        logger.info(f"批量买入完成,共 {len(orders)} 笔订单")
        return orders
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取持仓状态"""
        total_market_value = sum(p["market_value"] for p in self.positions.values())
        total_cost = sum(p["cost"] for p in self.positions.values())
        total_profit = total_market_value - total_cost
        
        return {
            "account_balance": self.account_balance,
            "available_cash": self.available_cash,
            "position_count": len(self.positions),
            "total_market_value": total_market_value,
            "total_cost": total_cost,
            "total_profit": total_profit,
            "profit_rate": total_profit / total_cost if total_cost > 0 else 0,
            "position_ratio": (self.account_balance - self.available_cash) / self.account_balance,
            "positions": self.positions
        }
    
    def update_positions_price(self, market_data: Dict[str, float]):
        """
        更新持仓价格
        
        Args:
            market_data: {symbol: current_price}
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position["current_price"] = market_data[symbol]
                position["market_value"] = position["volume"] * position["current_price"]
                position["profit"] = position["market_value"] - position["cost"]
                position["profit_rate"] = position["profit"] / position["cost"]
    
    def save_report(self, filepath: str):
        """保存交易报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": self.get_portfolio_status(),
            "orders": [
                {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "name": o.name,
                    "direction": o.direction,
                    "price": o.filled_price,
                    "volume": o.filled_volume,
                    "status": o.status.value,
                    "timestamp": o.timestamp,
                    "reason": o.reason
                }
                for o in self.orders
            ],
            "trades": self.trade_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"交易报告已保存: {filepath}")


if __name__ == "__main__":
    # 测试用例
    executor = TradingExecutor(
        account_balance=100000,
        max_position_per_stock=0.3,
        enable_real_trading=False
    )
    
    # 模拟选中股票
    selected = [
        {
            "symbol": "000001",
            "name": "平安银行",
            "rl_score": 85.5,
            "auction_info": {
                "final_price": 10.5,
                "strength": 82.0
            }
        },
        {
            "symbol": "300750",
            "name": "宁德时代",
            "rl_score": 92.3,
            "auction_info": {
                "final_price": 200.5,
                "strength": 88.0
            }
        }
    ]
    
    # 批量买入
    orders = executor.batch_buy(selected)
    
    # 查看持仓
    portfolio = executor.get_portfolio_status()
    print(f"\n持仓状态:")
    print(f"  可用资金: {portfolio['available_cash']:.2f}")
    print(f"  持仓数量: {portfolio['position_count']}")
    print(f"  仓位比例: {portfolio['position_ratio']:.1%}")
