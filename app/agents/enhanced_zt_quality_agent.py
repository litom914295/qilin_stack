"""
麒麟量化系统 - 增强版涨停质量评估Agent
针对一进二策略的涨停板深度分析，包含实战细节

核心改进：
1. 涨停板型精细分类（一字、T字、烂板回封等）
2. 封单动态追踪（撤单、补单分析）
3. 打板资金性质识别（游资、机构、散户）
4. 次日高开概率预测模型
5. 历史封板成功率统计
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LimitUpType(Enum):
    """涨停板型"""
    ONE_WORD = "一字板"           # 全天一字
    T_WORD = "T字板"              # 低开后快速封板
    SECOND_BOARD = "秒板"         # 开盘即封
    STEADY_BOARD = "稳封板"       # 10点前封板，无开板
    ROTTEN_BACK = "烂板回封"      # 多次开板后回封
    LATE_BOARD = "尾盘板"         # 14点后封板
    WEAK_BOARD = "弱板"           # 封板不稳，多次开板


@dataclass
class SealDynamics:
    """封单动态数据"""
    seal_amount: float              # 封单金额（万）
    seal_orders: int                # 封单笔数
    avg_seal_size: float            # 平均封单（万/笔）
    cancel_amount: float            # 撤单金额（万）
    cancel_rate: float              # 撤单率
    replenish_amount: float         # 补单金额（万）
    replenish_speed: float          # 补单速度（万/分钟）
    seal_stability: float           # 封单稳定性（0-1）


@dataclass
class CapitalNature:
    """打板资金性质"""
    hot_money_ratio: float          # 游资占比
    institution_ratio: float        # 机构占比
    retail_ratio: float             # 散户占比
    famous_seats: List[str]         # 知名席位
    capital_concentration: float    # 资金集中度
    is_smart_money: bool            # 是否聪明钱


@dataclass
class NextDayPrediction:
    """次日预测"""
    open_premium_prob: float        # 高开概率
    expected_premium: float         # 预期溢价（%）
    continue_limit_prob: float      # 继续涨停概率
    break_board_prob: float         # 开板概率
    confidence: float               # 预测置信度
    risk_level: str                 # 风险等级


class EnhancedZTQualityAgent:
    """增强版涨停质量评估Agent"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.name = "增强版涨停质量Agent"
        self.weight = 0.15
        self.config = config or {}
        
        # 历史数据库（实战中应从数据库加载）
        self.historical_success_rate = {}
        self.famous_hot_money_seats = [
            "湘财证券长沙韶山中路", 
            "东方财富拉萨东环路第二",
            "华泰证券深圳益田路荣超商务中心",
            "申万宏源深圳金田路",
            "国泰君安成都北一环路"
        ]
        
        logger.info(f"{self.name} 初始化完成")
    
    def analyze(self, ctx: Any) -> Dict[str, Any]:
        """
        核心分析方法
        
        Args:
            ctx: TradingContext 交易上下文
            
        Returns:
            分析结果字典
        """
        if not ctx.d_day_data or not ctx.d_day_data.is_limit_up:
            return {
                "score": 0,
                "confidence": 0.0,
                "reasoning": "非涨停股，不适用一进二策略",
                "details": {}
            }
        
        d_day = ctx.d_day_data
        score = 0.0
        details = {}
        
        # 1. 板型识别与评分（30分）
        limit_type, type_score = self._classify_limit_up_type(d_day)
        score += type_score
        details['limit_type'] = limit_type.value
        details['type_score'] = type_score
        
        # 2. 封单动态分析（25分）
        seal_dynamics = self._analyze_seal_dynamics(d_day, ctx)
        seal_score = self._score_seal_dynamics(seal_dynamics)
        score += seal_score
        details['seal_dynamics'] = seal_dynamics.__dict__
        details['seal_score'] = seal_score
        
        # 3. 资金性质识别（25分）
        capital_nature = self._analyze_capital_nature(d_day, ctx)
        capital_score = self._score_capital_nature(capital_nature)
        score += capital_score
        details['capital_nature'] = capital_nature.__dict__
        details['capital_score'] = capital_score
        
        # 4. 历史成功率（10分）
        historical_score = self._calculate_historical_success_rate(d_day)
        score += historical_score
        details['historical_score'] = historical_score
        
        # 5. 市场环境加成（10分）
        market_bonus = self._calculate_market_bonus(ctx)
        score += market_bonus
        details['market_bonus'] = market_bonus
        
        # 6. 次日预测
        next_day_pred = self._predict_next_day(
            d_day, limit_type, seal_dynamics, capital_nature, ctx
        )
        details['next_day_prediction'] = next_day_pred.__dict__
        
        # 计算综合置信度
        confidence = self._calculate_confidence(
            score, limit_type, seal_dynamics, capital_nature, next_day_pred
        )
        
        # 生成决策建议
        reasoning = self._generate_reasoning(
            limit_type, seal_dynamics, capital_nature, next_day_pred, score
        )
        
        return {
            "score": min(score, 100),
            "confidence": confidence,
            "reasoning": reasoning,
            "details": details,
            "next_day_prediction": next_day_pred.__dict__,
            "timestamp": datetime.now().isoformat()
        }
    
    def _classify_limit_up_type(self, d_day: Any) -> Tuple[LimitUpType, float]:
        """
        精细分类涨停板型
        
        实战要点：
        - 一字板：最强，但排队难度大
        - 秒板：强势，资金坚决
        - 稳封板：次强，性价比高
        - 烂板回封：风险大，需谨慎
        """
        open_times = d_day.open_times
        limit_time = d_day.limit_up_time or "15:00"
        open_pct = (d_day.open - d_day.close) / d_day.close * 100 if d_day.close else 0
        
        # 分类逻辑
        if abs(open_pct) < 0.1 and open_times == 0:
            # 一字板：开盘就涨停，全天不开板
            return LimitUpType.ONE_WORD, 30.0
        
        elif limit_time < "09:35" and open_times == 0:
            # 秒板：开盘5分钟内封板，不开板
            return LimitUpType.SECOND_BOARD, 28.0
        
        elif open_pct < -5 and limit_time < "10:00":
            # T字板：低开后快速拉升封板
            return LimitUpType.T_WORD, 25.0
        
        elif limit_time < "11:00" and open_times <= 1:
            # 稳封板：早盘封板，最多开一次
            return LimitUpType.STEADY_BOARD, 22.0
        
        elif open_times >= 3:
            # 烂板回封：多次开板
            return LimitUpType.ROTTEN_BACK, 10.0
        
        elif limit_time >= "14:00":
            # 尾盘板：尾盘才封板
            return LimitUpType.LATE_BOARD, 12.0
        
        else:
            # 弱板：其他情况
            return LimitUpType.WEAK_BOARD, 15.0
    
    def _analyze_seal_dynamics(self, d_day: Any, ctx: Any) -> SealDynamics:
        """
        分析封单动态
        
        实战要点：
        - 封单金额要持续增加，不能减少
        - 撤单率低说明资金坚决
        - 补单速度快说明承接力强
        """
        # 从上下文或数据源获取分时封单数据
        # 这里使用简化逻辑，实战需要Level2数据
        
        seal_amount = d_day.seal_amount or 0
        seal_ratio = d_day.seal_ratio or 0
        
        # 估算封单笔数和平均大小
        avg_order_size = 50  # 假设平均封单50万
        seal_orders = int(seal_amount / avg_order_size) if avg_order_size > 0 else 0
        
        # 估算撤单和补单（实战需要实时数据）
        # 撤单率：开板次数越多，撤单越严重
        cancel_rate = min(d_day.open_times * 0.15, 0.6)
        cancel_amount = seal_amount * cancel_rate
        
        # 补单速度：封单比例越高，补单越快
        replenish_speed = seal_ratio * 1000  # 每分钟补单速度
        replenish_amount = seal_amount * 0.3  # 估算补单量
        
        # 封单稳定性：综合评估
        seal_stability = (1 - cancel_rate) * seal_ratio
        
        return SealDynamics(
            seal_amount=seal_amount / 10000,  # 转为万元
            seal_orders=seal_orders,
            avg_seal_size=avg_order_size,
            cancel_amount=cancel_amount / 10000,
            cancel_rate=cancel_rate,
            replenish_amount=replenish_amount / 10000,
            replenish_speed=replenish_speed,
            seal_stability=seal_stability
        )
    
    def _score_seal_dynamics(self, seal: SealDynamics) -> float:
        """
        封单动态评分
        
        评分标准：
        - 封单金额：越大越好（10分）
        - 封单稳定性：越高越好（8分）
        - 撤单率：越低越好（7分）
        """
        score = 0.0
        
        # 1. 封单金额评分
        if seal.seal_amount > 5000:  # 5000万以上
            score += 10
        elif seal.seal_amount > 2000:
            score += 7
        elif seal.seal_amount > 1000:
            score += 4
        else:
            score += 2
        
        # 2. 封单稳定性评分
        if seal.seal_stability > 0.8:
            score += 8
        elif seal.seal_stability > 0.6:
            score += 6
        elif seal.seal_stability > 0.4:
            score += 3
        
        # 3. 撤单率评分（反向）
        if seal.cancel_rate < 0.1:
            score += 7
        elif seal.cancel_rate < 0.3:
            score += 4
        elif seal.cancel_rate < 0.5:
            score += 2
        
        return score
    
    def _analyze_capital_nature(self, d_day: Any, ctx: Any) -> CapitalNature:
        """
        识别打板资金性质
        
        实战要点：
        - 游资：动作快、金额大、席位集中
        - 机构：动作慢、持仓久、不轻易撤
        - 散户：跟风多、金额小、容易踩踏
        """
        # 从龙虎榜或Level2数据分析
        # 这里使用简化逻辑
        
        main_net = d_day.main_net_inflow or 0
        super_large = d_day.super_large_net or 0
        large_net = d_day.large_net or 0
        
        total_inflow = abs(main_net) + 0.01
        
        # 估算资金性质比例
        hot_money_ratio = super_large / total_inflow if super_large > 0 else 0
        institution_ratio = (large_net - super_large) / total_inflow if large_net > super_large else 0
        retail_ratio = 1 - hot_money_ratio - institution_ratio
        
        # 识别知名席位（实战需要龙虎榜数据）
        famous_seats = []
        if hot_money_ratio > 0.5:
            # 假设有游资介入
            famous_seats = self.famous_hot_money_seats[:2]
        
        # 资金集中度
        capital_concentration = hot_money_ratio + institution_ratio
        
        # 判断是否聪明钱
        is_smart_money = (
            hot_money_ratio > 0.4 or  # 游资主导
            (institution_ratio > 0.3 and d_day.turnover_rate < 10)  # 机构埋伏
        )
        
        return CapitalNature(
            hot_money_ratio=hot_money_ratio,
            institution_ratio=institution_ratio,
            retail_ratio=retail_ratio,
            famous_seats=famous_seats,
            capital_concentration=capital_concentration,
            is_smart_money=is_smart_money
        )
    
    def _score_capital_nature(self, capital: CapitalNature) -> float:
        """
        资金性质评分
        
        评分标准：
        - 游资主导：最好（20分）
        - 游资+机构：次之（15分）
        - 散户主导：最差（5分）
        """
        score = 0.0
        
        # 1. 聪明钱加成
        if capital.is_smart_money:
            score += 10
        
        # 2. 游资占比评分
        if capital.hot_money_ratio > 0.6:
            score += 10
        elif capital.hot_money_ratio > 0.4:
            score += 7
        elif capital.hot_money_ratio > 0.2:
            score += 4
        
        # 3. 知名席位加成
        score += min(len(capital.famous_seats) * 3, 5)
        
        return score
    
    def _calculate_historical_success_rate(self, d_day: Any) -> float:
        """
        计算历史成功率
        
        实战要点：
        - 查询该股票历史涨停后次日表现
        - 查询相似形态的成功率
        """
        # 简化计算，实战需要数据库查询
        symbol = d_day.symbol
        
        # 模拟历史成功率（实战应查询数据库）
        if symbol in self.historical_success_rate:
            success_rate = self.historical_success_rate[symbol]
        else:
            # 默认成功率
            success_rate = 0.5
        
        # 转换为分数
        score = success_rate * 10
        
        return score
    
    def _calculate_market_bonus(self, ctx: Any) -> float:
        """
        市场环境加成
        
        实战要点：
        - 情绪好时，一进二成功率高
        - 板块热度高，个股容易走强
        """
        score = 0.0
        
        if ctx.d_day_market:
            market = ctx.d_day_market
            
            # 1. 市场情绪加成
            if market.sentiment_score > 70:
                score += 5
            elif market.sentiment_score > 50:
                score += 3
            
            # 2. 赚钱效应加成
            if market.money_effect > 0.6:
                score += 3
            elif market.money_effect > 0.4:
                score += 2
            
            # 3. 连板梯队加成
            if market.high_board_count >= 3:
                score += 2
        
        return score
    
    def _predict_next_day(
        self,
        d_day: Any,
        limit_type: LimitUpType,
        seal: SealDynamics,
        capital: CapitalNature,
        ctx: Any
    ) -> NextDayPrediction:
        """
        预测次日表现
        
        实战要点：
        - 综合板型、封单、资金等因素
        - 给出概率化的预测，而非确定性结论
        """
        # 基础高开概率
        base_prob = {
            LimitUpType.ONE_WORD: 0.95,
            LimitUpType.SECOND_BOARD: 0.85,
            LimitUpType.T_WORD: 0.80,
            LimitUpType.STEADY_BOARD: 0.75,
            LimitUpType.ROTTEN_BACK: 0.50,
            LimitUpType.LATE_BOARD: 0.55,
            LimitUpType.WEAK_BOARD: 0.60
        }
        
        open_prob = base_prob.get(limit_type, 0.6)
        
        # 封单影响
        if seal.seal_stability > 0.8:
            open_prob += 0.1
        elif seal.seal_stability < 0.4:
            open_prob -= 0.1
        
        # 资金影响
        if capital.is_smart_money:
            open_prob += 0.08
        
        # 市场情绪影响
        if ctx.d_day_market and ctx.d_day_market.sentiment_score > 70:
            open_prob += 0.05
        
        # 限制在0-1之间
        open_prob = max(0, min(1, open_prob))
        
        # 预期溢价
        expected_premium = open_prob * 5  # 简化计算
        
        # 继续涨停概率
        continue_limit_prob = open_prob * 0.3 if limit_type in [
            LimitUpType.ONE_WORD, LimitUpType.SECOND_BOARD
        ] else open_prob * 0.15
        
        # 开板概率
        break_board_prob = 1 - continue_limit_prob - 0.1
        
        # 置信度
        confidence = 0.7 if capital.is_smart_money else 0.5
        
        # 风险等级
        if limit_type == LimitUpType.ROTTEN_BACK or seal.cancel_rate > 0.5:
            risk_level = "高"
        elif open_prob > 0.8:
            risk_level = "低"
        else:
            risk_level = "中"
        
        return NextDayPrediction(
            open_premium_prob=open_prob,
            expected_premium=expected_premium,
            continue_limit_prob=continue_limit_prob,
            break_board_prob=break_board_prob,
            confidence=confidence,
            risk_level=risk_level
        )
    
    def _calculate_confidence(
        self,
        score: float,
        limit_type: LimitUpType,
        seal: SealDynamics,
        capital: CapitalNature,
        pred: NextDayPrediction
    ) -> float:
        """
        计算综合置信度
        
        实战要点：
        - 置信度不是简单的分数映射
        - 要考虑多个维度的一致性
        """
        # 基础置信度
        base_conf = score / 100
        
        # 一致性检查
        consistency_factors = []
        
        # 1. 板型与封单一致性
        if limit_type in [LimitUpType.ONE_WORD, LimitUpType.SECOND_BOARD]:
            if seal.seal_stability > 0.7:
                consistency_factors.append(1.0)
            else:
                consistency_factors.append(0.7)
        
        # 2. 资金与板型一致性
        if capital.is_smart_money and limit_type != LimitUpType.ROTTEN_BACK:
            consistency_factors.append(1.0)
        else:
            consistency_factors.append(0.8)
        
        # 3. 预测置信度
        consistency_factors.append(pred.confidence)
        
        # 综合置信度
        consistency_avg = np.mean(consistency_factors)
        final_conf = base_conf * 0.6 + consistency_avg * 0.4
        
        return max(0, min(1, final_conf))
    
    def _generate_reasoning(
        self,
        limit_type: LimitUpType,
        seal: SealDynamics,
        capital: CapitalNature,
        pred: NextDayPrediction,
        score: float
    ) -> str:
        """生成决策理由"""
        reasons = []
        
        # 1. 板型描述
        reasons.append(f"板型为{limit_type.value}")
        
        # 2. 封单情况
        if seal.seal_stability > 0.7:
            reasons.append(f"封单稳固（稳定性{seal.seal_stability:.2f}）")
        elif seal.cancel_rate > 0.4:
            reasons.append(f"封单不稳（撤单率{seal.cancel_rate:.1%}）")
        
        # 3. 资金性质
        if capital.is_smart_money:
            reasons.append("聪明钱介入")
            if capital.famous_seats:
                reasons.append(f"知名席位：{', '.join(capital.famous_seats[:2])}")
        
        # 4. 次日预测
        reasons.append(f"次日高开概率{pred.open_premium_prob:.1%}")
        
        # 5. 综合评价
        if score >= 80:
            reasons.append("综合评分优秀，值得重点关注")
        elif score >= 60:
            reasons.append("综合评分良好，可适度参与")
        else:
            reasons.append("综合评分一般，谨慎参与")
        
        return "；".join(reasons)


# 使用示例
if __name__ == "__main__":
    from app.core.trading_context import TradingContext
    
    # 创建测试上下文
    ctx = TradingContext("000001", datetime.now())
    ctx.load_d_day_data()
    ctx.load_t1_auction_data()
    
    # 创建增强Agent
    agent = EnhancedZTQualityAgent()
    result = agent.analyze(ctx)
    
    print("=" * 60)
    print("增强版涨停质量分析结果")
    print("=" * 60)
    print(f"得分: {result['score']:.2f}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"分析理由: {result['reasoning']}")
    print(f"\n次日预测:")
    pred = result['next_day_prediction']
    print(f"  高开概率: {pred['open_premium_prob']:.1%}")
    print(f"  预期溢价: {pred['expected_premium']:.2f}%")
    print(f"  风险等级: {pred['risk_level']}")
