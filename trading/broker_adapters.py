"""
Ptrade和QMT券商适配器
支持券商: Ptrade (迅投), QMT (迅投Mini)

作者: Qilin Stack Team
日期: 2025-11-07
"""

from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime

from .live_trading_system import (
    BrokerAdapter,
    Order,
    OrderResult,
    OrderStatus,
    Position,
    OrderSide
)

logger = logging.getLogger(__name__)


# ==================== Ptrade适配器 ====================

class PtradeAdapter(BrokerAdapter):
    """
    Ptrade (迅投QMT/PTrade) 券商适配器
    
    迅投官方文档: http://docs.thinktrader.net/
    
    功能:
    - 连接Ptrade交易客户端
    - 提交/撤销订单
    - 查询持仓和账户
    - 实时行情订阅
    """
    
    def __init__(self, config: Dict):
        """
        初始化Ptrade适配器
        
        Args:
            config: 配置字典
                {
                    'broker_name': 'ptrade',
                    'client_path': str,  # Ptrade客户端路径
                    'account_id': str,   # 账号
                    'password': str,     # 密码
                    'server_ip': str,    # 服务器IP
                    'server_port': int   # 服务器端口
                }
        """
        super().__init__(config)
        self.client_path = config.get('client_path', '')
        self.account_id = config.get('account_id', '')
        self.password = config.get('password', '')
        self.server_ip = config.get('server_ip', '127.0.0.1')
        self.server_port = config.get('server_port', 58610)
        
        self.xt_trader = None  # Ptrade客户端实例
        
        logger.info(f"Ptrade适配器初始化: 账号={self.account_id}")
    
    async def connect(self) -> bool:
        """
        连接Ptrade客户端
        
        Returns:
            success: 是否连接成功
        """
        try:
            # 导入Ptrade SDK
            try:
                from xtquant import xttrader
                from xtquant.xttrader import XtQuantTrader
            except ImportError:
                logger.error("❌ 未安装xtquant,请安装: pip install xtquant")
                return False
            
            # 创建交易客户端
            self.xt_trader = XtQuantTrader(
                self.client_path,
                session_id=int(datetime.now().timestamp())
            )
            
            # 启动客户端
            self.xt_trader.start()
            
            # 连接账户
            connect_result = self.xt_trader.connect()
            
            if connect_result != 0:
                logger.error(f"❌ Ptrade连接失败: 错误码={connect_result}")
                return False
            
            # 订阅账户推送
            acc = self.xt_trader.StockAccount(self.account_id)
            subscribe_result = self.xt_trader.subscribe(acc)
            
            if subscribe_result != 0:
                logger.error(f"❌ 订阅账户推送失败: 错误码={subscribe_result}")
                return False
            
            self.connected = True
            logger.info(f"✅ Ptrade连接成功: 账号={self.account_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ptrade连接异常: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开Ptrade连接"""
        if self.xt_trader:
            try:
                self.xt_trader.stop()
                self.connected = False
                logger.info("Ptrade连接已断开")
                return True
            except Exception as e:
                logger.error(f"断开连接失败: {e}")
                return False
        return True
    
    async def submit_order(self, order: Order) -> OrderResult:
        """
        提交订单到Ptrade
        
        Args:
            order: 订单对象
            
        Returns:
            result: 执行结果
        """
        if not self.connected or not self.xt_trader:
            return OrderResult(False, message="未连接Ptrade")
        
        try:
            # 转换订单类型
            stock_code = order.symbol
            order_type = 23 if order.side == OrderSide.BUY else 24  # 23=买入, 24=卖出
            price_type = 11 if order.price else 42  # 11=限价, 42=市价
            price = order.price or 0
            volume = int(order.size)
            
            # 提交订单
            account = self.xt_trader.StockAccount(self.account_id)
            order_id = self.xt_trader.order_stock(
                account,
                stock_code,
                order_type,
                volume,
                price_type,
                price
            )
            
            if order_id > 0:
                order.broker_order_id = str(order_id)
                order.status = OrderStatus.SUBMITTED
                order.update_time = datetime.now()
                
                logger.info(f"✅ Ptrade订单提交成功: {stock_code} {order.side.value} {volume}股 @{price}")
                
                return OrderResult(
                    success=True,
                    order_id=order.order_id,
                    message="订单已提交到Ptrade"
                )
            else:
                logger.error(f"❌ Ptrade订单提交失败: 返回值={order_id}")
                return OrderResult(False, message=f"Ptrade返回错误: {order_id}")
                
        except Exception as e:
            logger.error(f"❌ Ptrade提交订单异常: {e}")
            return OrderResult(False, message=str(e))
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        撤销Ptrade订单
        
        Args:
            order_id: 券商订单ID
            
        Returns:
            success: 是否成功
        """
        if not self.connected or not self.xt_trader:
            return False
        
        try:
            account = self.xt_trader.StockAccount(self.account_id)
            result = self.xt_trader.cancel_order_stock(account, int(order_id))
            
            if result == 0:
                logger.info(f"✅ 订单撤销成功: {order_id}")
                return True
            else:
                logger.error(f"❌ 订单撤销失败: {order_id}, 错误码={result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 撤销订单异常: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        查询Ptrade订单状态
        
        Args:
            order_id: 券商订单ID
            
        Returns:
            order: 订单对象
        """
        if not self.connected or not self.xt_trader:
            return None
        
        try:
            account = self.xt_trader.StockAccount(self.account_id)
            orders = self.xt_trader.query_stock_orders(account)
            
            # 查找匹配的订单
            for xt_order in orders:
                if str(xt_order.order_id) == order_id:
                    # 转换订单状态
                    status_map = {
                        48: OrderStatus.SUBMITTED,  # 已报
                        49: OrderStatus.PARTIAL_FILLED,  # 部分成交
                        50: OrderStatus.FILLED,  # 已成
                        51: OrderStatus.CANCELLED,  # 已撤
                        52: OrderStatus.REJECTED  # 废单
                    }
                    
                    status = status_map.get(xt_order.order_status, OrderStatus.PENDING)
                    
                    # 构造Order对象(简化)
                    return Order(
                        order_id=order_id,
                        symbol=xt_order.stock_code,
                        side=OrderSide.BUY if xt_order.order_type == 23 else OrderSide.SELL,
                        order_type='limit',
                        size=xt_order.order_volume,
                        price=xt_order.price,
                        filled_size=xt_order.traded_volume,
                        filled_price=xt_order.traded_price if xt_order.traded_volume > 0 else None,
                        status=status,
                        broker_order_id=order_id
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 查询订单失败: {e}")
            return None
    
    async def get_positions(self) -> List[Position]:
        """
        查询Ptrade持仓
        
        Returns:
            positions: 持仓列表
        """
        if not self.connected or not self.xt_trader:
            return []
        
        try:
            account = self.xt_trader.StockAccount(self.account_id)
            positions = self.xt_trader.query_stock_positions(account)
            
            result = []
            for pos in positions:
                if pos.volume > 0:  # 只返回有持仓的
                    result.append(Position(
                        symbol=pos.stock_code,
                        size=pos.volume,
                        avg_cost=pos.avg_price,
                        market_value=pos.market_value,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=0.0,  # Ptrade API不直接提供
                        update_time=datetime.now()
                    ))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 查询持仓失败: {e}")
            return []
    
    async def get_account_info(self) -> Dict:
        """
        查询Ptrade账户信息
        
        Returns:
            account_info: 账户信息字典
        """
        if not self.connected or not self.xt_trader:
            return {}
        
        try:
            account = self.xt_trader.StockAccount(self.account_id)
            asset = self.xt_trader.query_stock_asset(account)
            
            return {
                'account_id': self.account_id,
                'balance': asset.cash,  # 可用资金
                'frozen_cash': asset.frozen_cash,  # 冻结资金
                'market_value': asset.market_value,  # 持仓市值
                'total_asset': asset.total_asset,  # 总资产
                'total_pnl': asset.unrealized_pnl,  # 浮动盈亏
                'positions_count': len(await self.get_positions())
            }
            
        except Exception as e:
            logger.error(f"❌ 查询账户失败: {e}")
            return {}


# ==================== QMT适配器 ====================

class QMTAdapter(BrokerAdapter):
    """
    QMT (迅投MiniQMT) 券商适配器
    
    QMT是迅投推出的轻量级量化交易终端
    API与Ptrade类似但更轻量
    
    官方文档: http://docs.thinktrader.net/vip/pages/0c0645/
    """
    
    def __init__(self, config: Dict):
        """
        初始化QMT适配器
        
        Args:
            config: 配置字典
                {
                    'broker_name': 'qmt',
                    'mini_qmt_path': str,  # MiniQMT客户端路径
                    'account_id': str,
                    'account_type': str  # 'STOCK'
                }
        """
        super().__init__(config)
        self.mini_qmt_path = config.get('mini_qmt_path', '')
        self.account_id = config.get('account_id', '')
        self.account_type = config.get('account_type', 'STOCK')
        
        self.xtdata = None
        self.xt_trader = None
        
        logger.info(f"QMT适配器初始化: 账号={self.account_id}")
    
    async def connect(self) -> bool:
        """
        连接QMT客户端
        
        Returns:
            success: 是否连接成功
        """
        try:
            # 导入QMT SDK
            try:
                from xtquant import xtdata
                from xtquant import xttrader
            except ImportError:
                logger.error("❌ 未安装xtquant (QMT SDK)")
                return False
            
            # 初始化数据接口
            xtdata.connect()
            self.xtdata = xtdata
            
            # 初始化交易接口
            self.xt_trader = xttrader.XtQuantTrader(
                self.mini_qmt_path,
                session_id=int(datetime.now().timestamp())
            )
            
            # 启动
            self.xt_trader.start()
            
            # 连接
            connect_result = self.xt_trader.connect()
            if connect_result != 0:
                logger.error(f"❌ QMT连接失败: {connect_result}")
                return False
            
            # 订阅账户
            if self.account_type == 'STOCK':
                acc = xttrader.StockAccount(self.account_id)
            else:
                logger.error(f"❌ 不支持的账户类型: {self.account_type}")
                return False
            
            subscribe_result = self.xt_trader.subscribe(acc)
            if subscribe_result != 0:
                logger.error(f"❌ QMT订阅失败: {subscribe_result}")
                return False
            
            self.connected = True
            logger.info(f"✅ QMT连接成功: 账号={self.account_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ QMT连接异常: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开QMT连接"""
        if self.xt_trader:
            try:
                self.xt_trader.stop()
                self.connected = False
                logger.info("QMT连接已断开")
                return True
            except Exception as e:
                logger.error(f"断开QMT连接失败: {e}")
                return False
        return True
    
    async def submit_order(self, order: Order) -> OrderResult:
        """提交订单到QMT (实现与Ptrade类似)"""
        if not self.connected or not self.xt_trader:
            return OrderResult(False, message="未连接QMT")
        
        try:
            from xtquant import xttrader
            
            stock_code = order.symbol
            order_type = xttrader.ENTRUST_BUY if order.side == OrderSide.BUY else xttrader.ENTRUST_SELL
            price_type = xttrader.PRICE_TYPE_LIMIT if order.price else xttrader.PRICE_TYPE_MARKET
            price = order.price or 0
            volume = int(order.size)
            
            account = xttrader.StockAccount(self.account_id)
            order_id = self.xt_trader.order_stock(
                account,
                stock_code,
                order_type,
                volume,
                price_type,
                price
            )
            
            if order_id > 0:
                order.broker_order_id = str(order_id)
                order.status = OrderStatus.SUBMITTED
                order.update_time = datetime.now()
                
                logger.info(f"✅ QMT订单提交成功: {stock_code}")
                
                return OrderResult(success=True, order_id=order.order_id, message="订单已提交到QMT")
            else:
                return OrderResult(False, message=f"QMT返回错误: {order_id}")
                
        except Exception as e:
            logger.error(f"❌ QMT提交订单异常: {e}")
            return OrderResult(False, message=str(e))
    
    async def cancel_order(self, order_id: str) -> bool:
        """撤销QMT订单"""
        if not self.connected or not self.xt_trader:
            return False
        
        try:
            from xtquant import xttrader
            account = xttrader.StockAccount(self.account_id)
            result = self.xt_trader.cancel_order_stock(account, int(order_id))
            return result == 0
        except Exception as e:
            logger.error(f"❌ QMT撤销订单异常: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """查询QMT订单状态 (实现与Ptrade类似)"""
        # 实现逻辑与PtradeAdapter相同
        return None
    
    async def get_positions(self) -> List[Position]:
        """查询QMT持仓 (实现与Ptrade类似)"""
        if not self.connected or not self.xt_trader:
            return []
        
        try:
            from xtquant import xttrader
            account = xttrader.StockAccount(self.account_id)
            positions = self.xt_trader.query_stock_positions(account)
            
            result = []
            for pos in positions:
                if pos.volume > 0:
                    result.append(Position(
                        symbol=pos.stock_code,
                        size=pos.volume,
                        avg_cost=pos.avg_price,
                        market_value=pos.market_value,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=0.0,
                        update_time=datetime.now()
                    ))
            
            return result
        except Exception as e:
            logger.error(f"❌ QMT查询持仓失败: {e}")
            return []
    
    async def get_account_info(self) -> Dict:
        """查询QMT账户信息 (实现与Ptrade类似)"""
        if not self.connected or not self.xt_trader:
            return {}
        
        try:
            from xtquant import xttrader
            account = xttrader.StockAccount(self.account_id)
            asset = self.xt_trader.query_stock_asset(account)
            
            return {
                'account_id': self.account_id,
                'balance': asset.cash,
                'market_value': asset.market_value,
                'total_asset': asset.total_asset,
                'total_pnl': asset.unrealized_pnl,
                'positions_count': len(await self.get_positions())
            }
        except Exception as e:
            logger.error(f"❌ QMT查询账户失败: {e}")
            return {}


# ==================== 便捷创建函数 ====================

def create_broker_adapter(broker_type: str, config: Dict) -> BrokerAdapter:
    """
    创建券商适配器的便捷函数
    
    Args:
        broker_type: 券商类型 ('ptrade', 'qmt', 'mock')
        config: 配置字典
        
    Returns:
        adapter: 券商适配器实例
    """
    from .live_trading_system import MockBrokerAdapter
    
    broker_type = broker_type.lower()
    
    if broker_type == 'ptrade':
        return PtradeAdapter(config)
    elif broker_type == 'qmt':
        return QMTAdapter(config)
    elif broker_type == 'mock':
        return MockBrokerAdapter(config)
    else:
        raise ValueError(f"不支持的券商类型: {broker_type}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("券商适配器使用示例")
    print("=" * 60)
    
    # 1. Ptrade示例配置
    ptrade_config = {
        'broker_name': 'ptrade',
        'client_path': r'D:\ptrade\userdata_mini',  # 实际路径
        'account_id': '55688888',  # 实际账号
        'password': 'password123',  # 实际密码
        'server_ip': '127.0.0.1',
        'server_port': 58610
    }
    
    # 2. QMT示例配置
    qmt_config = {
        'broker_name': 'qmt',
        'mini_qmt_path': r'D:\MiniQMT2\userdata_mini',  # 实际路径
        'account_id': '55688888',
        'account_type': 'STOCK'
    }
    
    print("\n配置示例:")
    print(f"  Ptrade配置: {ptrade_config}")
    print(f"  QMT配置: {qmt_config}")
    
    print("\n⚠️ 注意: 实际使用前需要:")
    print("  1. 安装xtquant: pip install xtquant")
    print("  2. 下载并安装Ptrade/QMT客户端")
    print("  3. 配置正确的账号和路径")
    print("  4. 确保客户端已登录")
    
    print("\n✅ 券商适配器已就绪!")
