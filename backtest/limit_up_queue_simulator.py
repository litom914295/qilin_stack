"""
涨停板排队模拟器
模拟真实市场涨停板排队买入情况，考虑：
1. 封板强度（强/中/弱）
2. 排队位置（前/中/后）
3. 不同涨停类型（10%、20%、ST 5%）
4. 部分成交概率
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LimitUpStrength(Enum):
    """涨停板强度分类"""
    STRONG = "strong"   # 强势封板（封单量大、很少开板）
    MEDIUM = "medium"   # 中等封板（封单适中、偶尔开板）
    WEAK = "weak"       # 弱势封板（封单小、频繁开板）


class StockType(Enum):
    """股票类型"""
    MAIN_BOARD = "main"      # 主板（10%涨停）
    CHINEXT = "chinext"      # 创业板/科创板（20%涨停）
    ST = "st"                # ST股票（5%涨停）


class LimitUpQueueSimulator:
    """涨停板排队模拟器"""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        初始化模拟器
        
        Args:
            random_seed: 随机种子，保证回测可重复性
        """
        if random_seed:
            np.random.seed(random_seed)
        
        # 定义不同强度下的成交概率
        self.fill_probabilities = {
            LimitUpStrength.STRONG: {
                'can_buy_prob': 0.3,    # 能排上队的概率
                'full_fill_prob': 0.1,  # 全额成交概率
                'partial_fill_range': (0.0, 0.3),  # 部分成交比例范围
            },
            LimitUpStrength.MEDIUM: {
                'can_buy_prob': 0.6,
                'full_fill_prob': 0.3,
                'partial_fill_range': (0.2, 0.7),
            },
            LimitUpStrength.WEAK: {
                'can_buy_prob': 0.9,
                'full_fill_prob': 0.6,
                'partial_fill_range': (0.5, 1.0),
            }
        }
        
        # 不同股票类型的调整系数
        self.type_adjustments = {
            StockType.MAIN_BOARD: 1.0,   # 基准
            StockType.CHINEXT: 0.8,      # 20%涨停更难买
            StockType.ST: 1.2,            # ST相对容易
        }
    
    def can_buy(
        self, 
        limit_up_strength: LimitUpStrength,
        my_capital: float,
        total_seal_amount: float,
        stock_type: StockType = StockType.MAIN_BOARD
    ) -> Tuple[bool, str]:
        """
        判断是否能买入（是否能排上队）
        
        Args:
            limit_up_strength: 涨停强度
            my_capital: 我的买入金额
            total_seal_amount: 总封单金额
            stock_type: 股票类型
        
        Returns:
            (是否能买入, 原因说明)
        """
        config = self.fill_probabilities[limit_up_strength]
        type_adj = self.type_adjustments[stock_type]
        
        # 根据资金量调整概率
        capital_factor = min(1.0, my_capital / max(total_seal_amount, 1e6))
        
        # 计算最终能买入的概率
        final_prob = config['can_buy_prob'] * type_adj * (1 + capital_factor * 0.2)
        final_prob = min(1.0, final_prob)
        
        # 随机决定是否能买入
        can_buy = np.random.random() < final_prob
        
        if can_buy:
            reason = f"成功排队（概率: {final_prob:.1%}）"
        else:
            if limit_up_strength == LimitUpStrength.STRONG:
                reason = "封单太强，无法排队"
            elif limit_up_strength == LimitUpStrength.MEDIUM:
                reason = "排队人数过多，未能排上"
            else:
                reason = "虽然封板弱，但仍未排上"
        
        logger.debug(f"排队判断: 强度={limit_up_strength.value}, 能买={can_buy}, {reason}")
        
        return can_buy, reason
    
    def calculate_fill_ratio(
        self,
        limit_up_strength: LimitUpStrength,
        queue_position: float,  # 0-1之间，0表示最前，1表示最后
        stock_type: StockType = StockType.MAIN_BOARD,
        market_sentiment: float = 0.5  # 0-1之间，市场情绪
    ) -> float:
        """
        计算成交比例
        
        Args:
            limit_up_strength: 涨停强度
            queue_position: 排队位置（0-1）
            stock_type: 股票类型
            market_sentiment: 市场情绪（0-1）
        
        Returns:
            成交比例（0-1）
        """
        config = self.fill_probabilities[limit_up_strength]
        type_adj = self.type_adjustments[stock_type]
        
        # 判断是否全额成交
        full_fill_prob_adj = config['full_fill_prob'] * type_adj * (1 + market_sentiment * 0.3)
        if np.random.random() < full_fill_prob_adj:
            return 1.0
        
        # 计算部分成交比例
        min_ratio, max_ratio = config['partial_fill_range']
        
        # 根据排队位置调整成交比例
        # 越靠前成交越多
        position_factor = 1.0 - queue_position
        
        # 根据市场情绪调整
        sentiment_factor = 0.8 + market_sentiment * 0.4
        
        # 计算最终成交比例
        base_ratio = min_ratio + (max_ratio - min_ratio) * position_factor
        final_ratio = base_ratio * sentiment_factor * type_adj
        
        # 确保在0-1范围内
        final_ratio = max(0.0, min(1.0, final_ratio))
        
        # 加入一些随机性
        noise = np.random.normal(0, 0.05)
        final_ratio += noise
        final_ratio = max(0.0, min(1.0, final_ratio))
        
        logger.debug(f"成交比例计算: 强度={limit_up_strength.value}, 位置={queue_position:.2f}, 比例={final_ratio:.2%}")
        
        return final_ratio
    
    def estimate_queue_position(
        self,
        my_capital: float,
        total_seal_amount: float,
        submit_time_rank: float = 0.5  # 提交时间排名（0-1）
    ) -> float:
        """
        估算排队位置
        
        Args:
            my_capital: 我的买入金额
            total_seal_amount: 总封单金额
            submit_time_rank: 提交时间排名（0最早，1最晚）
        
        Returns:
            排队位置（0-1）
        """
        # 资金量占比
        capital_ratio = min(1.0, my_capital / max(total_seal_amount, 1e6))
        
        # 时间因素权重更大（先到先得）
        time_weight = 0.7
        capital_weight = 0.3
        
        # 计算综合排队位置
        queue_position = (submit_time_rank * time_weight + 
                         (1 - capital_ratio) * capital_weight)
        
        # 加入随机扰动
        noise = np.random.normal(0, 0.05)
        queue_position += noise
        
        # 确保在0-1范围
        queue_position = max(0.0, min(1.0, queue_position))
        
        return queue_position
    
    def simulate_limit_up_trading(
        self,
        limit_up_strength: LimitUpStrength,
        my_capital: float,
        total_seal_amount: float,
        stock_type: StockType = StockType.MAIN_BOARD,
        submit_time_rank: float = 0.5,
        market_sentiment: float = 0.5
    ) -> dict:
        """
        完整的涨停板交易模拟
        
        Returns:
            {
                'can_buy': bool,
                'reason': str,
                'queue_position': float,
                'fill_ratio': float,
                'filled_amount': float,
                'unfilled_amount': float
            }
        """
        # 判断能否排队
        can_buy, reason = self.can_buy(
            limit_up_strength, 
            my_capital,
            total_seal_amount,
            stock_type
        )
        
        if not can_buy:
            return {
                'can_buy': False,
                'reason': reason,
                'queue_position': 1.0,
                'fill_ratio': 0.0,
                'filled_amount': 0.0,
                'unfilled_amount': my_capital
            }
        
        # 估算排队位置
        queue_position = self.estimate_queue_position(
            my_capital,
            total_seal_amount,
            submit_time_rank
        )
        
        # 计算成交比例
        fill_ratio = self.calculate_fill_ratio(
            limit_up_strength,
            queue_position,
            stock_type,
            market_sentiment
        )
        
        # 计算成交金额
        filled_amount = my_capital * fill_ratio
        unfilled_amount = my_capital * (1 - fill_ratio)
        
        return {
            'can_buy': True,
            'reason': reason,
            'queue_position': queue_position,
            'fill_ratio': fill_ratio,
            'filled_amount': filled_amount,
            'unfilled_amount': unfilled_amount
        }


# 测试代码
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    simulator = LimitUpQueueSimulator(random_seed=42)
    
    # 测试不同场景
    test_cases = [
        ("强势封板", LimitUpStrength.STRONG, 100000, 1e8),
        ("中等封板", LimitUpStrength.MEDIUM, 100000, 5e7),
        ("弱势封板", LimitUpStrength.WEAK, 100000, 1e7),
    ]
    
    for name, strength, capital, seal in test_cases:
        print(f"\n{name}:")
        result = simulator.simulate_limit_up_trading(
            strength, capital, seal,
            submit_time_rank=0.3,  # 较早提交
            market_sentiment=0.6    # 市场情绪偏好
        )
        
        print(f"  能否买入: {result['can_buy']}")
        print(f"  原因: {result['reason']}")
        if result['can_buy']:
            print(f"  排队位置: {result['queue_position']:.1%}")
            print(f"  成交比例: {result['fill_ratio']:.1%}")
            print(f"  成交金额: {result['filled_amount']:,.0f}")
            print(f"  未成交金额: {result['unfilled_amount']:,.0f}")