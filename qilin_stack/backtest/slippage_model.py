"""
滑点与冲击成本模型 (Slippage & Market Impact Model)
模拟真实交易中的价格偏离和市场冲击

核心功能：
1. 基于流动性的滑点计算
2. 订单规模冲击成本模型
3. 分笔成交模拟
4. 实际成交价格计算
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class OrderSide(Enum):
    """订单方向"""
    BUY = "买入"
    SELL = "卖出"


class SlippageModel(Enum):
    """滑点模型类型"""
    FIXED = "固定滑点"              # 固定百分比
    LINEAR = "线性滑点"             # 与订单量线性相关
    SQRT = "平方根滑点"             # 与订单量平方根相关（更符合实际）
    LIQUIDITY_BASED = "流动性滑点"  # 基于流动性评估


@dataclass
class OrderExecution:
    """订单执行结果"""
    symbol: str
    side: OrderSide
    
    # 订单信息
    target_shares: int              # 目标股数
    target_price: float             # 目标价格（信号价）
    
    # 执行结果
    executed_shares: int            # 实际成交股数
    avg_execution_price: float      # 平均成交价
    
    # 成本分析
    slippage: float                 # 滑点（元/股）
    slippage_pct: float             # 滑点比例
    market_impact: float            # 市场冲击（元/股）
    total_cost: float               # 总成交金额
    
    # 分笔成交
    fills: List[Tuple[int, float]]  # [(股数, 价格), ...]
    
    # 元数据
    execution_reason: str           # 执行说明
    warnings: List[str]             # 警告信息


@dataclass
class MarketDepth:
    """市场深度信息"""
    bid_prices: List[float]         # 买盘价格（从高到低）
    bid_volumes: List[int]          # 买盘量
    ask_prices: List[float]         # 卖盘价格（从低到高）
    ask_volumes: List[int]          # 卖盘量
    
    mid_price: float                # 中间价
    spread: float                   # 买卖价差
    
    # 流动性指标
    total_bid_volume: int           # 总买盘量
    total_ask_volume: int           # 总卖盘量
    liquidity_score: float          # 流动性评分（0-100）


class SlippageEngine:
    """滑点引擎"""
    
    def __init__(self,
                 model: SlippageModel = SlippageModel.LIQUIDITY_BASED,
                 fixed_slippage_bps: float = 5,              # 固定滑点（基点，5=0.05%）
                 impact_coefficient: float = 0.1,            # 冲击系数
                 max_participation_rate: float = 0.05):      # 最大参与率（5%）
        """
        初始化滑点引擎
        
        Args:
            model: 滑点模型类型
            fixed_slippage_bps: 固定滑点（基点）
            impact_coefficient: 市场冲击系数
            max_participation_rate: 单次最大参与率（订单量/市场量）
        """
        self.model = model
        self.fixed_slippage_bps = fixed_slippage_bps
        self.impact_coefficient = impact_coefficient
        self.max_participation_rate = max_participation_rate
    
    def execute_order(self,
                     symbol: str,
                     side: OrderSide,
                     target_shares: int,
                     target_price: float,
                     market_depth: Optional[MarketDepth] = None,
                     avg_daily_volume: Optional[float] = None,
                     liquidity_score: Optional[float] = None) -> OrderExecution:
        """
        模拟订单执行
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            target_shares: 目标股数
            target_price: 目标价格（信号价）
            market_depth: 市场深度数据
            avg_daily_volume: 日均成交量
            liquidity_score: 流动性评分
            
        Returns:
            OrderExecution: 执行结果
        """
        warnings = []
        
        # 1. 根据模型计算滑点
        if self.model == SlippageModel.FIXED:
            slippage = self._fixed_slippage(target_price)
            execution_reason = f"固定滑点{self.fixed_slippage_bps}bps"
            
        elif self.model == SlippageModel.LINEAR:
            slippage = self._linear_slippage(
                target_price, target_shares, avg_daily_volume
            )
            execution_reason = "线性滑点模型"
            
        elif self.model == SlippageModel.SQRT:
            slippage = self._sqrt_slippage(
                target_price, target_shares, avg_daily_volume
            )
            execution_reason = "平方根滑点模型"
            
        else:  # LIQUIDITY_BASED
            if market_depth and liquidity_score is not None:
                slippage, fills = self._liquidity_based_execution(
                    side, target_shares, target_price, market_depth, liquidity_score
                )
                execution_reason = "基于流动性的真实成交模拟"
            else:
                # 降级到平方根模型
                slippage = self._sqrt_slippage(
                    target_price, target_shares, avg_daily_volume
                )
                fills = [(target_shares, target_price + slippage)]
                execution_reason = "流动性数据缺失，降级到平方根模型"
                warnings.append("缺少流动性数据，使用简化模型")
        
        # 2. 计算市场冲击
        market_impact = self._calculate_market_impact(
            target_shares, target_price, avg_daily_volume, liquidity_score
        )
        
        # 3. 计算实际成交价
        if side == OrderSide.BUY:
            avg_execution_price = target_price + slippage + market_impact
        else:  # SELL
            avg_execution_price = target_price - slippage - market_impact
        
        # 4. 检查成交可行性
        executed_shares = target_shares
        if avg_daily_volume and target_shares > avg_daily_volume * self.max_participation_rate:
            warnings.append(
                f"订单量({target_shares:,})超过日均量{self.max_participation_rate:.0%}"
            )
            # 部分成交
            executed_shares = int(avg_daily_volume * self.max_participation_rate)
        
        # 5. 生成分笔成交记录（如果没有）
        if 'fills' not in locals():
            fills = [(executed_shares, avg_execution_price)]
        
        # 6. 计算总成本
        total_cost = executed_shares * avg_execution_price
        
        # 7. 计算滑点比例
        slippage_pct = (avg_execution_price - target_price) / target_price if side == OrderSide.BUY else \
                       (target_price - avg_execution_price) / target_price
        
        return OrderExecution(
            symbol=symbol,
            side=side,
            target_shares=target_shares,
            target_price=target_price,
            executed_shares=executed_shares,
            avg_execution_price=avg_execution_price,
            slippage=slippage,
            slippage_pct=slippage_pct,
            market_impact=market_impact,
            total_cost=total_cost,
            fills=fills,
            execution_reason=execution_reason,
            warnings=warnings
        )
    
    def _fixed_slippage(self, price: float) -> float:
        """固定滑点"""
        return price * (self.fixed_slippage_bps / 10000)
    
    def _linear_slippage(self, 
                        price: float,
                        shares: int,
                        avg_daily_volume: Optional[float]) -> float:
        """线性滑点模型"""
        if avg_daily_volume is None or avg_daily_volume == 0:
            return self._fixed_slippage(price)
        
        participation_rate = shares / avg_daily_volume
        # 参与率每增加1%，滑点增加0.01%
        slippage_pct = participation_rate * 0.01
        return price * slippage_pct
    
    def _sqrt_slippage(self,
                      price: float,
                      shares: int,
                      avg_daily_volume: Optional[float]) -> float:
        """
        平方根滑点模型（更符合实际）
        基于文献：Almgren-Chriss模型
        """
        if avg_daily_volume is None or avg_daily_volume == 0:
            return self._fixed_slippage(price)
        
        participation_rate = shares / avg_daily_volume
        # 平方根关系：冲击随订单量的平方根增长
        slippage_pct = self.impact_coefficient * np.sqrt(participation_rate)
        return price * slippage_pct
    
    def _liquidity_based_execution(self,
                                   side: OrderSide,
                                   target_shares: int,
                                   target_price: float,
                                   market_depth: MarketDepth,
                                   liquidity_score: float) -> Tuple[float, List[Tuple[int, float]]]:
        """
        基于真实盘口的执行模拟（最真实）
        
        模拟逐档吃单过程，返回平均成交价和分笔成交
        """
        fills = []
        remaining_shares = target_shares
        total_value = 0
        
        if side == OrderSide.BUY:
            # 买入：从最低卖价开始吃单
            prices = market_depth.ask_prices
            volumes = market_depth.ask_volumes
        else:
            # 卖出：从最高买价开始吃单
            prices = market_depth.bid_prices
            volumes = market_depth.bid_volumes
        
        # 逐档成交
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            if remaining_shares <= 0:
                break
            
            # 本档可成交量
            fill_shares = min(remaining_shares, volume)
            fills.append((fill_shares, price))
            total_value += fill_shares * price
            remaining_shares -= fill_shares
        
        # 如果还有剩余（盘口深度不够）
        if remaining_shares > 0:
            # 使用最后价格 + 额外滑点
            extra_slippage = target_price * 0.005  # 额外0.5%滑点
            last_price = prices[-1] if prices else target_price
            fill_price = last_price + extra_slippage if side == OrderSide.BUY else last_price - extra_slippage
            fills.append((remaining_shares, fill_price))
            total_value += remaining_shares * fill_price
        
        # 计算平均成交价
        avg_price = total_value / target_shares
        slippage = abs(avg_price - target_price)
        
        return slippage, fills
    
    def _calculate_market_impact(self,
                                 shares: int,
                                 price: float,
                                 avg_daily_volume: Optional[float],
                                 liquidity_score: Optional[float]) -> float:
        """
        计算市场冲击（订单对价格的推动作用）
        
        市场冲击 ∝ 订单规模 / 流动性
        """
        if avg_daily_volume is None or avg_daily_volume == 0:
            return 0
        
        # 基础冲击：参与率的平方根
        participation_rate = shares / avg_daily_volume
        base_impact = price * self.impact_coefficient * np.sqrt(participation_rate)
        
        # 流动性调整：流动性越差，冲击越大
        if liquidity_score is not None:
            liquidity_factor = (100 - liquidity_score) / 100  # 0-1之间
            impact = base_impact * (1 + liquidity_factor)
        else:
            impact = base_impact
        
        return impact
    
    def calculate_total_slippage(self, 
                                execution: OrderExecution) -> Dict[str, float]:
        """
        计算总滑点成本分析
        
        Returns:
            {
                'slippage_cost': 滑点成本（元）,
                'impact_cost': 冲击成本（元）,
                'total_cost': 总交易成本（元）,
                'cost_bps': 成本基点
            }
        """
        slippage_cost = execution.slippage * execution.executed_shares
        impact_cost = execution.market_impact * execution.executed_shares
        total_cost = slippage_cost + impact_cost
        
        # 转换为基点（1bp = 0.01%）
        cost_bps = (total_cost / (execution.target_price * execution.executed_shares)) * 10000
        
        return {
            'slippage_cost': slippage_cost,
            'impact_cost': impact_cost,
            'total_cost': total_cost,
            'cost_bps': cost_bps
        }


# 使用示例
if __name__ == "__main__":
    # 创建滑点引擎
    engine = SlippageEngine(
        model=SlippageModel.LIQUIDITY_BASED,
        impact_coefficient=0.1
    )
    
    print("=== 场景1：基于流动性的真实成交模拟 ===")
    
    # 模拟市场深度
    market_depth = MarketDepth(
        bid_prices=[10.49, 10.48, 10.47, 10.46, 10.45],
        bid_volumes=[50000, 45000, 40000, 35000, 30000],
        ask_prices=[10.50, 10.51, 10.52, 10.53, 10.54],
        ask_volumes=[48000, 42000, 38000, 33000, 28000],
        mid_price=10.495,
        spread=0.01,
        total_bid_volume=200000,
        total_ask_volume=189000,
        liquidity_score=75.0
    )
    
    # 执行买入订单
    execution = engine.execute_order(
        symbol="000001.SZ",
        side=OrderSide.BUY,
        target_shares=100_000,  # 买入10万股
        target_price=10.50,
        market_depth=market_depth,
        avg_daily_volume=3_000_000,
        liquidity_score=75.0
    )
    
    print(f"股票: {execution.symbol}")
    print(f"方向: {execution.side.value}")
    print(f"目标: {execution.target_shares:,}股 @ {execution.target_price:.2f}元")
    print(f"实际成交: {execution.executed_shares:,}股 @ {execution.avg_execution_price:.3f}元")
    print(f"滑点: {execution.slippage:.4f}元/股 ({execution.slippage_pct:.2%})")
    print(f"市场冲击: {execution.market_impact:.4f}元/股")
    print(f"总成交金额: {execution.total_cost:,.2f}元")
    print(f"执行说明: {execution.execution_reason}")
    
    print("\n分笔成交明细:")
    for i, (shares, price) in enumerate(execution.fills, 1):
        print(f"  第{i}笔: {shares:,}股 @ {price:.3f}元")
    
    # 成本分析
    cost_analysis = engine.calculate_total_slippage(execution)
    print("\n成本分析:")
    print(f"  滑点成本: {cost_analysis['slippage_cost']:,.2f}元")
    print(f"  冲击成本: {cost_analysis['impact_cost']:,.2f}元")
    print(f"  总交易成本: {cost_analysis['total_cost']:,.2f}元")
    print(f"  成本基点: {cost_analysis['cost_bps']:.2f}bps")
    
    if execution.warnings:
        print("\n⚠️ 警告:")
        for warning in execution.warnings:
            print(f"  - {warning}")
    
    # 场景2：不同滑点模型对比
    print("\n\n=== 场景2：不同滑点模型对比 ===")
    models = [
        SlippageModel.FIXED,
        SlippageModel.LINEAR,
        SlippageModel.SQRT,
        SlippageModel.LIQUIDITY_BASED
    ]
    
    for model in models:
        engine_test = SlippageEngine(model=model)
        exec_test = engine_test.execute_order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            target_shares=100_000,
            target_price=10.50,
            market_depth=market_depth,
            avg_daily_volume=3_000_000,
            liquidity_score=75.0
        )
        
        cost = engine_test.calculate_total_slippage(exec_test)
        print(f"\n{model.value}:")
        print(f"  平均成交价: {exec_test.avg_execution_price:.3f}元")
        print(f"  总成本: {cost['total_cost']:,.2f}元 ({cost['cost_bps']:.2f}bps)")
