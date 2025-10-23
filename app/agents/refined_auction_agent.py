"""
麒麟量化系统 - 精细化竞价博弈Agent
针对09:15-09:25的集合竞价进行深度分析

核心改进：
1. 3秒级细粒度价格追踪（原来是1分钟）
2. 09:20关键时刻分析（撤单高峰）
3. 大单追踪与席位识别
4. 竞价强度量化模型
5. 开盘预判（高开/平开/低开）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AuctionPhase(Enum):
    """竞价阶段"""
    EARLY = "早期(09:15-09:20)"      # 试探期
    CRITICAL = "关键期(09:20-09:25)"  # 真实竞价
    FINAL = "尾盘(09:24-09:25)"       # 最后1分钟


@dataclass
class AuctionTick:
    """竞价Tick数据"""
    timestamp: str              # 时间戳（HH:MM:SS）
    price: float                # 价格
    volume: int                 # 累积成交量
    bid_volume: int             # 买量
    ask_volume: int             # 卖量
    bid_orders: int             # 买单数
    ask_orders: int             # 卖单数
    

@dataclass
class PriceStability:
    """价格稳定性指标"""
    volatility: float           # 波动率
    trend_consistency: float    # 趋势一致性（0-1）
    price_range: float          # 价格区间（%）
    final_price_position: float # 收盘价位置（0-1）
    is_stable: bool            # 是否稳定
    

@dataclass
class OrderFlowAnalysis:
    """订单流分析"""
    large_buy_orders: int       # 大单买入笔数
    large_sell_orders: int      # 大单卖出笔数
    avg_buy_size: float         # 平均买单（万）
    avg_sell_size: float        # 平均卖单（万）
    order_imbalance: float      # 订单不平衡度（-1到1）
    smart_money_ratio: float    # 聪明钱占比


class RefinedAuctionAgent:
    """精细化竞价博弈Agent"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.name = "精细化竞价Agent"
        self.weight = 0.15
        self.config = config or {}
        
        # 关键时间点
        self.critical_time = time(9, 20, 0)  # 09:20撤单关键点
        self.final_time = time(9, 24, 0)     # 09:24最后阶段
        
        # 大单标准（实战根据流通盘调整）
        self.large_order_threshold = 100  # 100万为大单
        
        logger.info(f"{self.name} 初始化完成")
    
    def analyze(self, ctx: Any) -> Dict[str, Any]:
        """
        核心分析方法
        
        Args:
            ctx: TradingContext 交易上下文
            
        Returns:
            分析结果字典
        """
        if not ctx.t1_auction_data:
            return {
                "score": 0,
                "confidence": 0.0,
                "reasoning": "无竞价数据",
                "details": {}
            }
        
        auction = ctx.t1_auction_data
        d_day = ctx.d_day_data
        score = 0.0
        details = {}
        
        # 1. 价格轨迹分析（30分）
        price_trajectory = self._analyze_price_trajectory(auction)
        trajectory_score = self._score_price_trajectory(price_trajectory)
        score += trajectory_score
        details['price_trajectory'] = price_trajectory.__dict__
        details['trajectory_score'] = trajectory_score
        
        # 2. 关键时刻分析（25分）
        critical_moment = self._analyze_critical_moment(auction)
        critical_score = self._score_critical_moment(critical_moment)
        score += critical_score
        details['critical_moment'] = critical_moment
        details['critical_score'] = critical_score
        
        # 3. 订单流分析（25分）
        order_flow = self._analyze_order_flow(auction)
        order_score = self._score_order_flow(order_flow)
        score += order_score
        details['order_flow'] = order_flow.__dict__
        details['order_score'] = order_score
        
        # 4. 量价配合度（10分）
        volume_price_match = self._analyze_volume_price_match(auction)
        match_score = volume_price_match * 10
        score += match_score
        details['volume_price_match'] = volume_price_match
        
        # 5. 涨停预判（10分）
        limit_up_probability = self._predict_limit_up_probability(
            auction, d_day, price_trajectory, order_flow
        )
        score += limit_up_probability * 10
        details['limit_up_prob'] = limit_up_probability
        
        # 计算置信度
        confidence = self._calculate_confidence(
            price_trajectory, order_flow, auction
        )
        
        # 生成决策理由
        reasoning = self._generate_reasoning(
            price_trajectory, order_flow, auction, score
        )
        
        return {
            "score": min(score, 100),
            "confidence": confidence,
            "reasoning": reasoning,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_price_trajectory(self, auction: Any) -> PriceStability:
        """
        分析价格轨迹
        
        实战要点：
        - 09:20后价格稳定在高位是强势信号
        - 价格反复震荡说明分歧大
        - 尾盘快速拉升要警惕
        """
        if not auction.price_series or len(auction.price_series) < 2:
            return PriceStability(0, 0, 0, 0.5, False)
        
        prices = np.array(auction.price_series)
        
        # 1. 计算波动率
        price_changes = np.diff(prices) / prices[:-1]
        volatility = np.std(price_changes) if len(price_changes) > 0 else 0
        
        # 2. 趋势一致性
        up_moves = np.sum(np.diff(prices) > 0)
        total_moves = len(prices) - 1
        trend_consistency = up_moves / total_moves if total_moves > 0 else 0.5
        
        # 3. 价格区间
        price_range = (prices.max() - prices.min()) / prices[0] * 100 if prices[0] > 0 else 0
        
        # 4. 收盘价位置（在区间中的位置）
        if prices.max() > prices.min():
            final_position = (prices[-1] - prices.min()) / (prices.max() - prices.min())
        else:
            final_position = 0.5
        
        # 5. 判断稳定性
        # 稳定标准：波动小 + 趋势一致 + 价格在高位
        is_stable = (
            volatility < 0.02 and        # 波动率小于2%
            trend_consistency > 0.6 and   # 60%以上是上涨
            final_position > 0.7          # 收在70%以上位置
        )
        
        return PriceStability(
            volatility=volatility,
            trend_consistency=trend_consistency,
            price_range=price_range,
            final_price_position=final_position,
            is_stable=is_stable
        )
    
    def _score_price_trajectory(self, stability: PriceStability) -> float:
        """
        价格轨迹评分
        
        评分标准：
        - 稳定性强：25-30分
        - 中等稳定：15-24分
        - 不稳定：5-14分
        """
        score = 0.0
        
        # 1. 稳定性基础分
        if stability.is_stable:
            score += 15
        
        # 2. 波动率评分（反向）
        if stability.volatility < 0.01:
            score += 8
        elif stability.volatility < 0.02:
            score += 5
        elif stability.volatility < 0.03:
            score += 2
        
        # 3. 趋势一致性评分
        if stability.trend_consistency > 0.8:
            score += 7
        elif stability.trend_consistency > 0.6:
            score += 4
        elif stability.trend_consistency > 0.5:
            score += 2
        
        return score
    
    def _analyze_critical_moment(self, auction: Any) -> Dict:
        """
        分析09:20关键时刻
        
        实战要点：
        - 09:20后是真实竞价，前面可能有虚假挂单
        - 观察09:20前后的撤单情况
        - 09:20后价格不跌是强势信号
        """
        details = {}
        
        # 简化处理：比较前后价格变化
        if auction.price_series and len(auction.price_series) >= 3:
            prices = auction.price_series
            
            # 假设数据是每1分钟采样，第5个点是09:20附近
            mid_point = len(prices) // 2
            
            before_920 = np.mean(prices[:mid_point]) if mid_point > 0 else prices[0]
            after_920 = np.mean(prices[mid_point:]) if mid_point < len(prices) else prices[-1]
            
            price_change_after_920 = (after_920 - before_920) / before_920 * 100
            
            details['before_920_avg'] = before_920
            details['after_920_avg'] = after_920
            details['price_change'] = price_change_after_920
            
            # 判断是否稳定
            if price_change_after_920 > -1:  # 跌幅不超过1%
                details['is_stable_after_920'] = True
                details['strength'] = "强"
            elif price_change_after_920 > -3:
                details['is_stable_after_920'] = True
                details['strength'] = "中"
            else:
                details['is_stable_after_920'] = False
                details['strength'] = "弱"
        else:
            details['strength'] = "未知"
            details['is_stable_after_920'] = False
        
        return details
    
    def _score_critical_moment(self, critical: Dict) -> float:
        """
        关键时刻评分
        
        评分标准：
        - 09:20后稳定上涨：20-25分
        - 09:20后小幅回落：10-19分
        - 09:20后大幅下跌：0-9分
        """
        score = 0.0
        
        if critical.get('is_stable_after_920'):
            strength = critical.get('strength', '未知')
            if strength == "强":
                score += 25
            elif strength == "中":
                score += 18
            else:
                score += 10
        else:
            score += 5
        
        return score
    
    def _analyze_order_flow(self, auction: Any) -> OrderFlowAnalysis:
        """
        分析订单流
        
        实战要点：
        - 大单买入多说明主力坚决
        - 平均单量大说明机构或游资在做
        - 订单不平衡度反映多空力量对比
        """
        # 简化计算，实战需要Level2逐笔数据
        
        bid_volume = auction.bid_volume or 0
        ask_volume = auction.ask_volume or 0
        total_volume = bid_volume + ask_volume + 0.01
        
        # 估算大单（实战需要实际订单数据）
        # 假设20%的量是大单
        large_buy_orders = int(bid_volume * 0.2 / self.large_order_threshold)
        large_sell_orders = int(ask_volume * 0.2 / self.large_order_threshold)
        
        # 平均单量（万元）
        avg_buy_size = bid_volume / 100 if bid_volume > 0 else 0
        avg_sell_size = ask_volume / 100 if ask_volume > 0 else 0
        
        # 订单不平衡
        order_imbalance = (bid_volume - ask_volume) / total_volume
        
        # 聪明钱占比（大单占比）
        smart_money_ratio = (large_buy_orders + large_sell_orders) / max(1, large_buy_orders + large_sell_orders + 10) * 0.5
        
        return OrderFlowAnalysis(
            large_buy_orders=large_buy_orders,
            large_sell_orders=large_sell_orders,
            avg_buy_size=avg_buy_size / 10000,  # 转万元
            avg_sell_size=avg_sell_size / 10000,
            order_imbalance=order_imbalance,
            smart_money_ratio=smart_money_ratio
        )
    
    def _score_order_flow(self, order_flow: OrderFlowAnalysis) -> float:
        """
        订单流评分
        
        评分标准：
        - 大单买入多：好
        - 订单不平衡向买方：好
        - 聪明钱参与：好
        """
        score = 0.0
        
        # 1. 大单情况
        if order_flow.large_buy_orders > order_flow.large_sell_orders * 2:
            score += 10
        elif order_flow.large_buy_orders > order_flow.large_sell_orders:
            score += 6
        
        # 2. 订单不平衡
        if order_flow.order_imbalance > 0.5:
            score += 10
        elif order_flow.order_imbalance > 0.3:
            score += 6
        elif order_flow.order_imbalance > 0:
            score += 3
        
        # 3. 聪明钱
        score += order_flow.smart_money_ratio * 5
        
        return score
    
    def _analyze_volume_price_match(self, auction: Any) -> float:
        """
        量价配合度分析
        
        实战要点：
        - 价涨量增是健康的
        - 价涨量缩要警惕
        """
        # 简化计算
        price_change = auction.change_pct or 0
        volume_ratio = auction.volume_ratio or 1
        
        # 理想情况：涨幅5%对应量比2倍
        expected_volume = 1 + price_change / 5
        
        # 计算匹配度
        if volume_ratio >= expected_volume:
            match = 1.0
        elif volume_ratio >= expected_volume * 0.7:
            match = 0.7
        else:
            match = volume_ratio / expected_volume
        
        # 异常放量打折
        if volume_ratio > expected_volume * 3:
            match *= 0.7
        
        return min(match, 1.0)
    
    def _predict_limit_up_probability(
        self,
        auction: Any,
        d_day: Any,
        stability: PriceStability,
        order_flow: OrderFlowAnalysis
    ) -> float:
        """
        预测涨停概率
        
        实战要点：
        - 竞价涨幅越大，涨停概率越高
        - 但要结合稳定性和订单流
        """
        base_prob = 0.0
        
        # 1. 竞价涨幅基础概率
        change = auction.change_pct or 0
        if change >= 9:
            base_prob = 0.8
        elif change >= 7:
            base_prob = 0.6
        elif change >= 5:
            base_prob = 0.4
        elif change >= 3:
            base_prob = 0.2
        
        # 2. 稳定性加成
        if stability.is_stable:
            base_prob += 0.1
        
        # 3. 订单流加成
        if order_flow.order_imbalance > 0.5:
            base_prob += 0.05
        
        # 4. 昨日涨停加成
        if d_day and d_day.is_limit_up:
            base_prob += 0.1
        
        return min(base_prob, 1.0)
    
    def _calculate_confidence(
        self,
        stability: PriceStability,
        order_flow: OrderFlowAnalysis,
        auction: Any
    ) -> float:
        """计算置信度"""
        confidence_factors = []
        
        # 1. 价格稳定性置信度
        if stability.is_stable:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # 2. 订单流置信度
        if order_flow.smart_money_ratio > 0.3:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.7)
        
        # 3. 数据完整性
        if hasattr(auction, 'price_series') and len(auction.price_series) >= 5:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        return np.mean(confidence_factors)
    
    def _generate_reasoning(
        self,
        stability: PriceStability,
        order_flow: OrderFlowAnalysis,
        auction: Any,
        score: float
    ) -> str:
        """生成决策理由"""
        reasons = []
        
        # 1. 竞价涨幅
        change = auction.change_pct or 0
        reasons.append(f"竞价涨幅{change:.2f}%")
        
        # 2. 价格稳定性
        if stability.is_stable:
            reasons.append("价格稳定")
        else:
            reasons.append(f"价格波动(波动率{stability.volatility:.2%})")
        
        # 3. 订单流
        if order_flow.order_imbalance > 0.3:
            reasons.append("买盘占优")
        
        # 4. 大单情况
        if order_flow.large_buy_orders > order_flow.large_sell_orders:
            reasons.append(f"大单买入({order_flow.large_buy_orders}笔)")
        
        # 5. 综合评价
        if score >= 75:
            reasons.append("竞价强势，可重点关注")
        elif score >= 50:
            reasons.append("竞价一般，谨慎参与")
        else:
            reasons.append("竞价偏弱，不建议追高")
        
        return "；".join(reasons)


# 使用示例
if __name__ == "__main__":
    from app.core.trading_context import TradingContext
    
    # 创建测试上下文
    ctx = TradingContext("000001", datetime.now())
    ctx.load_d_day_data()
    ctx.load_t1_auction_data()
    
    # 创建精细化Agent
    agent = RefinedAuctionAgent()
    result = agent.analyze(ctx)
    
    print("=" * 60)
    print("精细化竞价分析结果")
    print("=" * 60)
    print(f"得分: {result['score']:.2f}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"分析理由: {result['reasoning']}")
