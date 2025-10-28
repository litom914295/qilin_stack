"""
涨停板回测适配器
将涨停排队模拟器集成到回测引擎中，实现真实的涨停板交易约束
"""

from datetime import datetime
from typing import Optional, Dict
import logging

from .limit_up_queue_simulator import (
    LimitUpQueueSimulator,
    LimitUpStrength,
    QueueExecution
)

logger = logging.getLogger(__name__)


class LimitUpBacktestAdapter:
    """涨停板回测适配器"""
    
    def __init__(self, 
                 simulator: Optional[LimitUpQueueSimulator] = None,
                 enable_one_word_block: bool = True,
                 strict_mode: bool = True):
        """
        初始化适配器
        
        Args:
            simulator: 涨停排队模拟器
            enable_one_word_block: 是否启用一字板无法成交规则
            strict_mode: 严格模式（一字板完全无法成交）
        """
        self.simulator = simulator or LimitUpQueueSimulator()
        self.enable_one_word_block = enable_one_word_block
        self.strict_mode = strict_mode
        
        # 缓存今日涨停板信息
        self.limit_up_cache: Dict[str, Dict] = {}
    
    def is_limit_up(self, 
                   symbol: str,
                   current_price: float,
                   prev_close: float,
                   limit_up_ratio: float = 0.10) -> bool:
        """
        判断是否涨停
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前收盘价
            limit_up_ratio: 涨停幅度（默认10%，科创板/创业板为20%）
            
        Returns:
            是否涨停
        """
        if prev_close <= 0:
            return False
        
        # 计算理论涨停价
        limit_price = prev_close * (1 + limit_up_ratio)
        limit_price = round(limit_price, 2)
        
        # 允许1分钱误差
        return abs(current_price - limit_price) <= 0.01
    
    def calculate_limit_price(self,
                             prev_close: float,
                             limit_up_ratio: float = 0.10) -> float:
        """
        计算涨停价
        
        Args:
            prev_close: 前收盘价
            limit_up_ratio: 涨停幅度
            
        Returns:
            涨停价
        """
        limit_price = prev_close * (1 + limit_up_ratio)
        return round(limit_price, 2)
    
    def can_buy_at_limit_up(self,
                           symbol: str,
                           order_time: datetime,
                           target_shares: int,
                           limit_price: float,
                           seal_amount: float,
                           seal_time: datetime,
                           open_times: int = 0,
                           market_data: Optional[Dict] = None) -> tuple[bool, Optional[QueueExecution]]:
        """
        判断能否在涨停价买入
        
        Args:
            symbol: 股票代码
            order_time: 下单时间
            target_shares: 目标股数
            limit_price: 涨停价
            seal_amount: 封单金额
            seal_time: 封板时间
            open_times: 开板次数
            market_data: 市场数据（可选）
            
        Returns:
            (是否可以成交, 成交详情)
        """
        # 评估排队状态
        queue_status = self.simulator.evaluate_queue_status(
            symbol=symbol,
            limit_price=limit_price,
            seal_amount=seal_amount,
            seal_time=seal_time,
            current_time=order_time,
            target_shares=target_shares,
            open_times=open_times
        )
        
        # 严格模式：一字板绝对无法成交
        if self.enable_one_word_block and self.strict_mode:
            if queue_status.strength == LimitUpStrength.ONE_WORD:
                logger.warning(
                    f"⛔ 一字板无法成交: {symbol} "
                    f"封单={seal_amount:,.0f}元, "
                    f"强度评分={queue_status.strength_score:.1f}/100"
                )
                
                # 返回未成交结果
                execution = QueueExecution(
                    symbol=symbol,
                    order_time=order_time,
                    target_shares=target_shares,
                    limit_price=limit_price,
                    filled=False,
                    filled_shares=0,
                    fill_time=None,
                    avg_fill_price=0,
                    initial_queue_position=queue_status.queue_position,
                    final_queue_position=queue_status.queue_position,
                    execution_reason="一字板封单过强，无法成交",
                    seal_broke=False,
                    warnings=["一字板规则：严格模式下禁止成交"]
                )
                return False, execution
        
        # 模拟排队成交
        execution = self.simulator.simulate_queue_execution(
            symbol=symbol,
            order_time=order_time,
            target_shares=target_shares,
            limit_price=limit_price,
            queue_status=queue_status,
            seal_broke=open_times > 0  # 有开板记录则认为可能成交
        )
        
        # 记录日志
        if execution.filled:
            logger.info(
                f"✅ 涨停板成交: {symbol} "
                f"目标={target_shares}股, "
                f"实际={execution.filled_shares}股 "
                f"({execution.filled_shares/target_shares:.0%}), "
                f"强度={queue_status.strength.value}, "
                f"概率={queue_status.fill_probability:.1%}"
            )
        else:
            logger.warning(
                f"❌ 涨停板未成交: {symbol} "
                f"目标={target_shares}股, "
                f"强度={queue_status.strength.value}, "
                f"概率={queue_status.fill_probability:.1%}, "
                f"原因: {execution.execution_reason}"
            )
        
        return execution.filled, execution
    
    def adjust_execution_for_limit_up(self,
                                     original_quantity: int,
                                     execution: QueueExecution) -> tuple[int, float]:
        """
        根据涨停排队结果调整成交数量和价格
        
        Args:
            original_quantity: 原始订单数量
            execution: 排队成交结果
            
        Returns:
            (调整后数量, 调整后价格)
        """
        if not execution.filled:
            return 0, 0.0
        
        # 返回实际成交数量和价格
        return execution.filled_shares, execution.avg_fill_price
    
    def get_limit_up_ratio(self, symbol: str) -> float:
        """
        根据股票代码判断涨停幅度
        
        Args:
            symbol: 股票代码
            
        Returns:
            涨停幅度（小数形式）
        """
        # 科创板 (688开头)
        if symbol.startswith('SH688') or symbol.startswith('688'):
            return 0.20  # 20%
        
        # 创业板 (300开头, 新股除外)
        if symbol.startswith('SZ300') or symbol.startswith('300'):
            return 0.20  # 20%
        
        # 北交所 (8/4开头)
        if symbol.startswith('BJ8') or symbol.startswith('BJ4'):
            return 0.30  # 30%
        
        # ST股票
        if 'ST' in symbol.upper():
            return 0.05  # 5%
        
        # 主板默认
        return 0.10  # 10%
    
    def validate_limit_up_order(self,
                               symbol: str,
                               order_price: float,
                               prev_close: float,
                               current_price: float) -> tuple[bool, str]:
        """
        验证涨停价订单
        
        Args:
            symbol: 股票代码
            order_price: 订单价格
            prev_close: 前收盘价
            current_price: 当前价格
            
        Returns:
            (是否有效, 原因)
        """
        limit_up_ratio = self.get_limit_up_ratio(symbol)
        limit_price = self.calculate_limit_price(prev_close, limit_up_ratio)
        
        # 检查是否涨停
        if not self.is_limit_up(symbol, current_price, prev_close, limit_up_ratio):
            return True, "非涨停状态，正常交易"
        
        # 检查订单价格是否为涨停价
        if abs(order_price - limit_price) > 0.01:
            return False, f"涨停价订单价格错误: 期望{limit_price:.2f}, 实际{order_price:.2f}"
        
        # 涨停板买入订单有效，但能否成交需进一步判断
        return True, "涨停板买入订单，排队等待成交"


# 使用示例
if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    
    adapter = LimitUpBacktestAdapter(
        enable_one_word_block=True,
        strict_mode=True
    )
    
    print("=== 测试1：一字板无法成交（严格模式） ===")
    can_fill, execution = adapter.can_buy_at_limit_up(
        symbol="SZ000001",
        order_time=datetime(2024, 1, 15, 9, 40),
        target_shares=10000,
        limit_price=11.00,
        seal_amount=100_000_000,  # 1亿封单
        seal_time=datetime(2024, 1, 15, 9, 30),  # 开盘即封
        open_times=0
    )
    
    print(f"能否成交: {can_fill}")
    print(f"成交股数: {execution.filled_shares}")
    print(f"原因: {execution.execution_reason}")
    if execution.warnings:
        print("警告:")
        for w in execution.warnings:
            print(f"  - {w}")
    
    print("\n=== 测试2：盘中封板可能成交 ===")
    can_fill2, execution2 = adapter.can_buy_at_limit_up(
        symbol="SZ000001",
        order_time=datetime(2024, 1, 15, 10, 40),
        target_shares=10000,
        limit_price=11.00,
        seal_amount=30_000_000,  # 3000万封单
        seal_time=datetime(2024, 1, 15, 10, 30),  # 盘中封板
        open_times=0
    )
    
    print(f"能否成交: {can_fill2}")
    print(f"成交股数: {execution2.filled_shares}")
    print(f"原因: {execution2.execution_reason}")
    
    print("\n=== 测试3：涨停幅度判断 ===")
    test_symbols = [
        ("SH688001", "科创板"),
        ("SZ300001", "创业板"),
        ("SH600000", "主板"),
        ("SZST0001", "ST股票")
    ]
    
    for symbol, name in test_symbols:
        ratio = adapter.get_limit_up_ratio(symbol)
        print(f"{name} ({symbol}): {ratio:.0%}")
