"""
麒麟量化系统 - 增强版Agent评分逻辑
包含非线性关系、条件触发和更精细的评分规则
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from core.trading_context import TradingContext

logger = logging.getLogger(__name__)


class EnhancedAuctionGameAgent:
    """增强版竞价博弈Agent - 更精细的竞价分析"""
    
    def __init__(self):
        self.name = "竞价博弈Agent"
        self.weight = 0.15
        
    def analyze(self, ctx: TradingContext) -> Dict[str, Any]:
        """
        分析集合竞价的博弈情况
        
        关键逻辑：
        1. 价格稳定性：09:20后价格是否稳定在均价线上方
        2. 封板强度：区分一字板、秒板、烂板回封
        3. 量价配合：竞价量能是否匹配价格涨幅
        4. 资金态度：大单主导还是散单推动
        """
        score = 0.0
        factors = {}
        
        if not ctx.t1_auction_data:
            return {"score": 0, "factors": {}, "reason": "无竞价数据"}
        
        auction = ctx.t1_auction_data
        d_day = ctx.d_day_data
        
        # 1. 价格稳定性分析（非线性）
        price_stability = self._analyze_price_stability(auction)
        factors['price_stability'] = price_stability
        
        # 价格稳定性的非线性影响
        if price_stability > 0.8:
            score += 30  # 非常稳定，给高分
        elif price_stability > 0.6:
            score += 20 * price_stability  # 中等稳定，线性给分
        else:
            score += 5  # 不稳定，给基础分
        
        # 2. 封板强度分析（区分板型）
        if d_day and d_day.is_limit_up:
            seal_strength = self._analyze_seal_strength(d_day, auction)
            factors['seal_strength'] = seal_strength
            
            # 根据板型给分
            if d_day.limit_type == "一字板":
                # 一字板竞价高开是强势延续
                if auction.change_pct > 5:
                    score += 35
                else:
                    score += 20  # 一字板后竞价不强，可能要开板
                    
            elif d_day.limit_type == "秒板":
                # 秒板看封单和竞价强度
                if seal_strength > 0.7 and auction.change_pct > 7:
                    score += 30
                else:
                    score += 15
                    
            elif d_day.limit_type == "烂板回封":
                # 烂板回封风险较大，需要竞价特别强
                if auction.change_pct > 8 and seal_strength > 0.8:
                    score += 25
                else:
                    score += 10
        
        # 3. 量价配合度（条件触发）
        volume_price_match = self._analyze_volume_price_match(auction, d_day)
        factors['volume_price_match'] = volume_price_match
        
        # 条件触发：只有在价格上涨时，量能配合才有意义
        if auction.change_pct > 3:
            if volume_price_match > 0.8:
                score += 20
            elif volume_price_match > 0.5:
                score += 10
        else:
            # 价格不涨，量能再大也没用
            score += 0
        
        # 4. 资金态度分析
        money_attitude = self._analyze_money_attitude(auction)
        factors['money_attitude'] = money_attitude
        
        # 市场情绪的条件影响
        if ctx.d_day_market and ctx.d_day_market.sentiment_score > 70:
            # 情绪好时，资金态度更重要
            score += money_attitude * 15
        else:
            # 情绪差时，资金态度影响减弱
            score += money_attitude * 8
        
        # 综合评估
        reason = self._generate_reason(factors, ctx)
        
        return {
            "score": min(score, 100),  # 最高100分
            "factors": factors,
            "reason": reason,
            "details": {
                "price_stability": price_stability,
                "volume_price_match": volume_price_match,
                "money_attitude": money_attitude,
                "auction_change": auction.change_pct
            }
        }
    
    def _analyze_price_stability(self, auction: Any) -> float:
        """
        分析价格稳定性
        通过分析09:20-09:25的价格序列判断稳定性
        """
        if not auction.price_series or len(auction.price_series) < 2:
            return 0.5
        
        prices = auction.price_series
        
        # 计算价格波动率
        price_std = np.std(prices)
        price_mean = np.mean(prices)
        volatility = price_std / price_mean if price_mean > 0 else 0
        
        # 计算趋势一致性
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        positive_changes = sum(1 for c in price_changes if c > 0)
        trend_consistency = positive_changes / len(price_changes) if price_changes else 0
        
        # 判断是否在均价上方
        above_mean = sum(1 for p in prices[-3:] if p >= price_mean) / min(3, len(prices))
        
        # 综合稳定性评分
        stability = (1 - min(volatility * 10, 1)) * 0.4 + \
                   trend_consistency * 0.3 + \
                   above_mean * 0.3
        
        return stability
    
    def _analyze_seal_strength(self, d_day: Any, auction: Any) -> float:
        """
        分析封板强度
        考虑封单金额、封单比例、开板次数等
        """
        # 封单比例
        seal_ratio_score = min(d_day.seal_ratio / 0.1, 1) if d_day.seal_ratio else 0
        
        # 开板次数的负面影响（非线性）
        if d_day.open_times == 0:
            open_penalty = 0
        elif d_day.open_times == 1:
            open_penalty = 0.2
        elif d_day.open_times == 2:
            open_penalty = 0.5
        else:
            open_penalty = 0.8
        
        # 涨停时间的影响
        if d_day.limit_up_time:
            if d_day.limit_up_time < "10:00":
                time_score = 1.0
            elif d_day.limit_up_time < "11:00":
                time_score = 0.7
            elif d_day.limit_up_time < "14:00":
                time_score = 0.4
            else:
                time_score = 0.2
        else:
            time_score = 0.5
        
        # 综合封板强度
        strength = seal_ratio_score * 0.5 + \
                  (1 - open_penalty) * 0.3 + \
                  time_score * 0.2
        
        return strength
    
    def _analyze_volume_price_match(self, auction: Any, d_day: Optional[Any]) -> float:
        """
        分析量价配合度
        竞价量能是否匹配价格涨幅
        """
        if not d_day:
            return 0.5
        
        # 量比分析
        volume_ratio = auction.volume_ratio if auction.volume_ratio else 1
        
        # 价格涨幅
        price_change = auction.change_pct
        
        # 理想的量价配合：涨幅7%应该有2倍以上量比
        expected_volume_ratio = 1 + (price_change / 10) * 2
        
        # 计算匹配度
        if volume_ratio >= expected_volume_ratio:
            match_score = 1.0
        elif volume_ratio >= expected_volume_ratio * 0.7:
            match_score = 0.7
        else:
            match_score = volume_ratio / expected_volume_ratio
        
        # 异常检测：量能过大可能是出货
        if volume_ratio > expected_volume_ratio * 3:
            match_score *= 0.7  # 过度放量要打折
        
        return match_score
    
    def _analyze_money_attitude(self, auction: Any) -> float:
        """
        分析资金态度
        通过委买委卖比、大单占比等判断
        """
        # 委买委卖比
        bid_ask = auction.bid_ask_ratio if auction.bid_ask_ratio else 1
        
        # 委比得分（非线性）
        if bid_ask > 3:
            bid_score = 1.0
        elif bid_ask > 2:
            bid_score = 0.8
        elif bid_ask > 1.5:
            bid_score = 0.6
        elif bid_ask > 1:
            bid_score = 0.4
        else:
            bid_score = 0.2
        
        # 竞价强度
        strength = auction.strength / 100 if auction.strength else 0.5
        
        # 综合资金态度
        attitude = bid_score * 0.6 + strength * 0.4
        
        return attitude
    
    def _generate_reason(self, factors: Dict, ctx: TradingContext) -> str:
        """生成分析理由"""
        reasons = []
        
        if factors.get('price_stability', 0) > 0.8:
            reasons.append("竞价价格稳定")
        elif factors.get('price_stability', 0) < 0.4:
            reasons.append("竞价波动较大")
        
        if factors.get('volume_price_match', 0) > 0.7:
            reasons.append("量价配合良好")
        
        if factors.get('money_attitude', 0) > 0.7:
            reasons.append("资金态度积极")
        
        if ctx.d_day_data and ctx.d_day_data.limit_type:
            reasons.append(f"昨日{ctx.d_day_data.limit_type}")
        
        return "，".join(reasons) if reasons else "竞价表现一般"


class EnhancedMarketEcologyAgent:
    """增强版市场生态Agent - 考虑市场环境的非线性影响"""
    
    def __init__(self):
        self.name = "市场生态Agent"
        self.weight = 0.12
        
    def analyze(self, ctx: TradingContext) -> Dict[str, Any]:
        """
        分析市场生态环境
        
        核心逻辑：
        1. 情绪周期：识别情绪所处阶段（启动/发酵/高潮/退潮）
        2. 板块轮动：当前板块是否处于轮动风口
        3. 高标效应：高位股对市场的带动作用
        4. 资金偏好：当前市场资金的风格偏好
        """
        score = 0.0
        factors = {}
        
        if not ctx.d_day_market:
            return {"score": 0, "factors": {}, "reason": "无市场数据"}
        
        market = ctx.d_day_market
        
        # 1. 情绪周期分析（非线性）
        emotion_cycle = self._analyze_emotion_cycle(market, ctx)
        factors['emotion_cycle'] = emotion_cycle
        
        # 情绪周期的非线性影响
        if emotion_cycle['phase'] == "发酵期":
            score += 35  # 发酵期最好做
            factors['emotion_bonus'] = "最佳做多窗口"
        elif emotion_cycle['phase'] == "启动期":
            score += 25
            factors['emotion_bonus'] = "趋势形成中"
        elif emotion_cycle['phase'] == "高潮期":
            score += 15  # 高潮期要谨慎
            factors['emotion_bonus'] = "注意风险"
        else:  # 退潮期
            score += 5
            factors['emotion_bonus'] = "防守为主"
        
        # 2. 板块轮动分析
        sector_rotation = self._analyze_sector_rotation(ctx)
        factors['sector_rotation'] = sector_rotation
        
        # 条件触发：只有在情绪好的时候，板块轮动才有效
        if market.sentiment_score > 60:
            score += sector_rotation * 25
        else:
            score += sector_rotation * 10
        
        # 3. 高标效应分析
        high_board_effect = self._analyze_high_board_effect(market)
        factors['high_board_effect'] = high_board_effect
        
        # 高标的非线性影响
        if high_board_effect > 0.8:
            # 有强势高标带动，市场空间打开
            score += 25
            factors['space'] = "空间已打开"
        elif high_board_effect > 0.5:
            score += 15
        else:
            score += 5
        
        # 4. 资金偏好分析
        money_preference = self._analyze_money_preference(market, ctx)
        factors['money_preference'] = money_preference
        
        # 根据个股是否符合当前资金偏好加分
        if self._match_money_preference(ctx, money_preference):
            score += 15
            factors['preference_match'] = True
        else:
            factors['preference_match'] = False
        
        reason = self._generate_reason(factors, ctx)
        
        return {
            "score": min(score, 100),
            "factors": factors,
            "reason": reason,
            "details": emotion_cycle
        }
    
    def _analyze_emotion_cycle(self, market: Any, ctx: TradingContext) -> Dict:
        """
        分析情绪周期
        通过涨停数量、连板梯队、赚钱效应等判断
        """
        # 涨停数量分析
        limit_up_count = market.limit_up_count
        natural_limit = market.natural_limit_up
        
        # 连板梯队分析
        second_board = market.second_board_count
        third_board = market.third_board_count
        high_board = market.high_board_count
        
        # 赚钱效应
        money_effect = market.money_effect
        
        # 判断情绪阶段
        if limit_up_count > 150 and high_board > 5 and money_effect > 0.7:
            phase = "高潮期"
            strength = 0.9
        elif limit_up_count > 100 and third_board > 10 and money_effect > 0.6:
            phase = "发酵期"
            strength = 0.8
        elif limit_up_count > 60 and second_board > 20 and money_effect > 0.5:
            phase = "启动期"
            strength = 0.6
        else:
            phase = "退潮期"
            strength = 0.3
        
        # 计算情绪持续性
        sustainability = self._calculate_sustainability(market)
        
        return {
            'phase': phase,
            'strength': strength,
            'sustainability': sustainability,
            'limit_up_count': limit_up_count,
            'money_effect': money_effect
        }
    
    def _calculate_sustainability(self, market: Any) -> float:
        """计算情绪可持续性"""
        # 简化计算，实际需要历史数据对比
        if market.high_board_count > 0:
            # 有高位股，情绪有支撑
            base = 0.6
        else:
            base = 0.3
        
        # 根据梯队结构调整
        if market.third_board_count > market.second_board_count * 0.4:
            # 梯队结构合理
            base += 0.2
        
        return min(base, 1.0)
    
    def _analyze_sector_rotation(self, ctx: TradingContext) -> float:
        """分析板块轮动"""
        if not ctx.d_day_data:
            return 0.5
        
        # 检查个股所在板块是否是热门板块
        stock_sector = ctx.d_day_data.sector
        
        if ctx.d_day_market and ctx.d_day_market.hot_sectors:
            for hot_sector in ctx.d_day_market.hot_sectors[:3]:  # 前三热门
                if stock_sector == hot_sector['name']:
                    # 在热门板块中
                    if ctx.d_day_data.sector_rank <= 3:
                        return 0.9  # 板块龙头
                    elif ctx.d_day_data.sector_rank <= 5:
                        return 0.7  # 板块前排
                    else:
                        return 0.5  # 板块跟风
        
        return 0.3  # 不在热门板块
    
    def _analyze_high_board_effect(self, market: Any) -> float:
        """分析高标效应"""
        if market.high_board_count >= 3:
            # 有多只高位股，空间已打开
            return 0.9
        elif market.high_board_count >= 1:
            # 有高位股带动
            return 0.6
        elif market.third_board_count >= 5:
            # 虽无高位，但有中位股支撑
            return 0.4
        else:
            return 0.2
    
    def _analyze_money_preference(self, market: Any, ctx: TradingContext) -> str:
        """分析当前市场资金偏好"""
        # 简化判断，实际需要更复杂的逻辑
        if market.hot_sectors and market.hot_sectors[0]['change'] > 5:
            return "题材股"
        elif market.sentiment_score < 40:
            return "防御型"
        else:
            return "均衡型"
    
    def _match_money_preference(self, ctx: TradingContext, preference: str) -> bool:
        """判断个股是否符合资金偏好"""
        if not ctx.d_day_data:
            return False
        
        if preference == "题材股":
            # 看是否在热门板块
            return ctx.d_day_data.sector_rank <= 10
        elif preference == "防御型":
            # 看是否是大盘股、白马股
            return ctx.d_day_data.turnover_rate < 5
        else:
            return True
    
    def _generate_reason(self, factors: Dict, ctx: TradingContext) -> str:
        """生成分析理由"""
        reasons = []
        
        if 'emotion_bonus' in factors:
            reasons.append(factors['emotion_bonus'])
        
        if factors.get('sector_rotation', 0) > 0.7:
            reasons.append("板块强势")
        
        if factors.get('preference_match'):
            reasons.append("符合资金偏好")
        
        if 'space' in factors:
            reasons.append(factors['space'])
        
        return "，".join(reasons) if reasons else "市场生态一般"


# 使用示例
if __name__ == "__main__":
    from datetime import datetime
    from core.trading_context import TradingContext, ContextManager
    
    # 创建上下文
    current_time = datetime(2024, 12, 20, 8, 55, 0)
    manager = ContextManager(current_time)
    
    # 加载数据
    ctx = manager.create_context('000001')
    ctx.load_d_day_data()
    ctx.load_t1_auction_data()
    
    # 测试增强版Agent
    auction_agent = EnhancedAuctionGameAgent()
    result = auction_agent.analyze(ctx)
    print(f"竞价博弈得分: {result['score']}")
    print(f"分析理由: {result['reason']}")
    print(f"详细因子: {result['factors']}")
    
    ecology_agent = EnhancedMarketEcologyAgent()
    result = ecology_agent.analyze(ctx)
    print(f"\n市场生态得分: {result['score']}")
    print(f"分析理由: {result['reason']}")
    print(f"情绪周期: {result['details']}")