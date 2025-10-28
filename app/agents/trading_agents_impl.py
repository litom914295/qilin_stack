"""
麒麟量化系统 - 10个专业交易Agent的完整实现
包含具体的打分逻辑和决策规则
"""

import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import logging
from dataclasses import dataclass
import asyncio

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class MarketContext:
    """市场上下文数据"""
    ohlcv: pd.DataFrame  # OHLCV数据
    news_titles: List[str]  # 新闻标题
    lhb_netbuy: float  # 龙虎榜净买入（亿元）
    market_mood_score: float  # 市场情绪分 (0-100)
    sector_heat: Dict[str, float]  # 板块热度
    money_flow: Dict[str, float]  # 资金流向
    technical_indicators: Dict[str, float]  # 技术指标
    fundamental_data: Dict[str, Any]  # 基本面数据
    


class ZTQualityAgent:
    """涨停质量评估Agent - 评估涨停板的质量和持续性"""
    
    def __init__(self):
        self.name = "涨停质量评估Agent"
        self.weight = 0.15
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        分析涨停板质量
        
        评分规则：
        1. 封单金额占比 (30分): 封单金额/流通市值
        2. 涨停时间 (20分): 越早涨停分数越高
        3. 开板次数 (20分): 开板次数越少越好
        4. 换手率 (15分): 适中最佳(5-15%)
        5. 量能变化 (15分): 放量但不过度
        """
        score = 0.0
        details = {}
        
        try:
            df = ctx.ohlcv
            if df.empty:
                return {"score": 0, "details": {"error": "无数据"}}
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # 1. 封单金额占比 (假设从level2数据获取)
            seal_ratio = ctx.technical_indicators.get('seal_ratio', 0.05)
            if seal_ratio > 0.1:
                score += 30
            elif seal_ratio > 0.05:
                score += 20
            elif seal_ratio > 0.02:
                score += 10
            details['seal_ratio'] = seal_ratio
            
            # 2. 涨停时间 (从分时数据推断)
            zt_time = ctx.technical_indicators.get('zt_time', '14:30')
            if zt_time < '10:00':
                score += 20
            elif zt_time < '11:00':
                score += 15
            elif zt_time < '14:00':
                score += 10
            else:
                score += 5
            details['zt_time'] = zt_time
            
            # 3. 开板次数
            open_times = ctx.technical_indicators.get('open_times', 0)
            if open_times == 0:
                score += 20
            elif open_times == 1:
                score += 10
            elif open_times == 2:
                score += 5
            details['open_times'] = open_times
            
            # 4. 换手率
            turnover = latest.get('turnover_rate', 0)
            if 5 <= turnover <= 15:
                score += 15
            elif 3 <= turnover < 5 or 15 < turnover <= 20:
                score += 10
            elif turnover < 3:
                score += 5
            details['turnover_rate'] = turnover
            
            # 5. 量能变化
            volume_ratio = latest['volume'] / prev['volume'] if prev['volume'] > 0 else 1
            if 1.5 <= volume_ratio <= 3:
                score += 15
            elif 1.2 <= volume_ratio < 1.5 or 3 < volume_ratio <= 5:
                score += 10
            elif volume_ratio < 1.2:
                score += 5
            details['volume_ratio'] = volume_ratio
            
            self.logger.info(f"{symbol} 涨停质量得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.9, # 硬数据，置信度高
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class LeaderAgent:
    """龙头识别Agent - 识别板块龙头股"""
    
    def __init__(self):
        self.name = "龙头识别Agent"
        self.weight = 0.15
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        识别是否为龙头股
        
        评分规则：
        1. 板块内涨幅排名 (25分): 第一名满分
        2. 连板数量 (25分): 连板越多越好
        3. 市场关注度 (20分): 新闻、评论数量
        4. 资金流入强度 (15分): 主力资金流入
        5. 历史龙头次数 (15分): 曾经当过龙头
        """
        score = 0.0
        details = {}
        
        try:
            # 1. 板块内涨幅排名
            sector_rank = ctx.sector_heat.get(f'{symbol}_rank', 5)
            if sector_rank == 1:
                score += 25
            elif sector_rank == 2:
                score += 18
            elif sector_rank == 3:
                score += 12
            elif sector_rank <= 5:
                score += 6
            details['sector_rank'] = sector_rank
            
            # 2. 连板数量
            consecutive_limit = ctx.technical_indicators.get('consecutive_limit', 0)
            if consecutive_limit >= 5:
                score += 25
            elif consecutive_limit >= 3:
                score += 20
            elif consecutive_limit >= 2:
                score += 15
            elif consecutive_limit == 1:
                score += 8
            details['consecutive_limit'] = consecutive_limit
            
            # 3. 市场关注度 (新闻数量)
            news_count = len([n for n in ctx.news_titles if symbol in n])
            if news_count >= 10:
                score += 20
            elif news_count >= 5:
                score += 15
            elif news_count >= 2:
                score += 10
            elif news_count >= 1:
                score += 5
            details['news_count'] = news_count
            
            # 4. 资金流入强度
            money_flow = ctx.money_flow.get(symbol, 0)
            if money_flow > 5:  # 亿元
                score += 15
            elif money_flow > 2:
                score += 10
            elif money_flow > 0.5:
                score += 5
            details['money_flow'] = money_flow
            
            # 5. 历史龙头次数
            history_leader = ctx.fundamental_data.get('history_leader_times', 0)
            if history_leader >= 3:
                score += 15
            elif history_leader >= 2:
                score += 10
            elif history_leader >= 1:
                score += 5
            details['history_leader'] = history_leader
            
            self.logger.info(f"{symbol} 龙头识别得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.8, # 龙头判断有一定主观性，置信度中等偏上
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class AuctionAgent:
    """集合竞价Agent - 分析竞价阶段的资金博弈"""
    
    def __init__(self):
        self.name = "集合竞价Agent"
        self.weight = 0.12
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        分析集合竞价情况
        
        评分规则：
        1. 竞价涨幅 (30分): 竞价涨幅强度
        2. 竞价量能 (25分): 竞价成交量占比
        3. 竞价一致性 (20分): 买卖挂单比例
        4. 竞价稳定性 (15分): 竞价过程中的波动
        5. 竞价资金 (10分): 大单参与度
        """
        score = 0.0
        details = {}
        
        try:
            # 1. 竞价涨幅
            auction_change = ctx.technical_indicators.get('auction_change', 0)
            if auction_change >= 5:
                score += 30
            elif auction_change >= 3:
                score += 20
            elif auction_change >= 1:
                score += 10
            elif auction_change >= 0:
                score += 5
            details['auction_change'] = auction_change
            
            # 2. 竞价量能
            auction_volume_ratio = ctx.technical_indicators.get('auction_volume_ratio', 0.05)
            if auction_volume_ratio > 0.15:
                score += 25
            elif auction_volume_ratio > 0.10:
                score += 18
            elif auction_volume_ratio > 0.05:
                score += 10
            details['auction_volume_ratio'] = auction_volume_ratio
            
            # 3. 竞价一致性 (买卖比)
            bid_ask_ratio = ctx.technical_indicators.get('bid_ask_ratio', 1.0)
            if bid_ask_ratio > 3:
                score += 20
            elif bid_ask_ratio > 2:
                score += 15
            elif bid_ask_ratio > 1.5:
                score += 10
            elif bid_ask_ratio > 1:
                score += 5
            details['bid_ask_ratio'] = bid_ask_ratio
            
            # 4. 竞价稳定性
            auction_volatility = ctx.technical_indicators.get('auction_volatility', 0.5)
            if auction_volatility < 0.2:
                score += 15
            elif auction_volatility < 0.5:
                score += 10
            elif auction_volatility < 1:
                score += 5
            details['auction_volatility'] = auction_volatility
            
            # 5. 大单参与度
            large_order_ratio = ctx.technical_indicators.get('large_order_ratio', 0.1)
            if large_order_ratio > 0.3:
                score += 10
            elif large_order_ratio > 0.2:
                score += 7
            elif large_order_ratio > 0.1:
                score += 3
            details['large_order_ratio'] = large_order_ratio
            
            self.logger.info(f"{symbol} 集合竞价得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.85, # 竞价数据相对客观，但有博弈成分
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class MoneyFlowAgent:
    """资金流向Agent - 分析资金流向和主力行为"""
    
    def __init__(self):
        self.name = "资金流向Agent"
        self.weight = 0.12
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        分析资金流向
        
        评分规则：
        1. 主力净流入 (30分): 主力资金净流入金额
        2. 超大单比例 (20分): 超大单占比
        3. 连续流入天数 (20分): 连续净流入
        4. 板块资金流向 (15分): 所属板块资金情况
        5. 北向资金 (15分): 北向资金持仓变化
        """
        score = 0.0
        details = {}
        
        try:
            # 1. 主力净流入
            main_netflow = ctx.money_flow.get(f'{symbol}_main', 0)
            if main_netflow > 3:  # 亿元
                score += 30
            elif main_netflow > 1:
                score += 20
            elif main_netflow > 0.3:
                score += 10
            elif main_netflow > 0:
                score += 5
            details['main_netflow'] = main_netflow
            
            # 2. 超大单比例
            super_large_ratio = ctx.money_flow.get(f'{symbol}_super_ratio', 0)
            if super_large_ratio > 0.25:
                score += 20
            elif super_large_ratio > 0.15:
                score += 15
            elif super_large_ratio > 0.08:
                score += 8
            details['super_large_ratio'] = super_large_ratio
            
            # 3. 连续流入天数
            consecutive_inflow = ctx.money_flow.get(f'{symbol}_consecutive', 0)
            if consecutive_inflow >= 5:
                score += 20
            elif consecutive_inflow >= 3:
                score += 15
            elif consecutive_inflow >= 2:
                score += 10
            elif consecutive_inflow == 1:
                score += 5
            details['consecutive_inflow'] = consecutive_inflow
            
            # 4. 板块资金流向
            sector_flow = ctx.sector_heat.get('sector_money_flow', 0)
            if sector_flow > 10:  # 亿元
                score += 15
            elif sector_flow > 5:
                score += 10
            elif sector_flow > 1:
                score += 5
            details['sector_flow'] = sector_flow
            
            # 5. 北向资金
            northbound = ctx.money_flow.get(f'{symbol}_northbound', 0)
            if northbound > 0.5:  # 亿元
                score += 15
            elif northbound > 0.1:
                score += 10
            elif northbound > 0:
                score += 5
            details['northbound'] = northbound
            
            self.logger.info(f"{symbol} 资金流向得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.8, # 资金流数据质量中等
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class EmotionAgent:
    """市场情绪Agent - 评估市场情绪和热度"""
    
    def __init__(self):
        self.name = "市场情绪Agent"
        self.weight = 0.10
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        分析市场情绪
        
        评分规则：
        1. 整体市场情绪 (25分): 市场情绪指数
        2. 个股热度 (25分): 搜索指数、讨论度
        3. 媒体情绪 (20分): 新闻正负面
        4. 散户情绪 (15分): 散户关注度
        5. 机构情绪 (15分): 研报评级
        """
        score = 0.0
        details = {}
        
        try:
            # 1. 整体市场情绪
            market_mood = ctx.market_mood_score
            if market_mood > 80:
                score += 25
            elif market_mood > 60:
                score += 18
            elif market_mood > 40:
                score += 10
            elif market_mood > 20:
                score += 5
            details['market_mood'] = market_mood
            
            # 2. 个股热度
            stock_heat = ctx.sector_heat.get(f'{symbol}_heat', 50)
            if stock_heat > 90:
                score += 25
            elif stock_heat > 70:
                score += 18
            elif stock_heat > 50:
                score += 10
            elif stock_heat > 30:
                score += 5
            details['stock_heat'] = stock_heat
            
            # 3. 媒体情绪
            positive_news = sum(1 for title in ctx.news_titles if any(
                word in title for word in ['强势', '突破', '新高', '利好', '增长']
            ))
            negative_news = sum(1 for title in ctx.news_titles if any(
                word in title for word in ['下跌', '利空', '风险', '调整', '减持']
            ))
            news_sentiment = positive_news - negative_news
            if news_sentiment > 5:
                score += 20
            elif news_sentiment > 2:
                score += 15
            elif news_sentiment > 0:
                score += 8
            details['news_sentiment'] = news_sentiment
            
            # 4. 散户情绪
            retail_sentiment = ctx.fundamental_data.get('retail_sentiment', 50)
            if retail_sentiment > 80:
                score += 15
            elif retail_sentiment > 60:
                score += 10
            elif retail_sentiment > 40:
                score += 5
            details['retail_sentiment'] = retail_sentiment
            
            # 5. 机构情绪
            institution_rating = ctx.fundamental_data.get('institution_rating', 3)
            if institution_rating >= 4.5:
                score += 15
            elif institution_rating >= 4:
                score += 10
            elif institution_rating >= 3.5:
                score += 5
            details['institution_rating'] = institution_rating
            
            self.logger.info(f"{symbol} 市场情绪得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.65, # 情绪数据主观性最强，置信度偏低
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class TechnicalAgent:
    """技术分析Agent - 技术指标综合分析"""
    
    def __init__(self):
        self.name = "技术分析Agent"
        self.weight = 0.08
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        技术指标分析
        
        评分规则：
        1. 趋势强度 (25分): MA系统、趋势线
        2. 动量指标 (25分): RSI、MACD
        3. 成交量指标 (20分): OBV、量价配合
        4. 支撑压力 (15分): 关键价位
        5. 形态识别 (15分): K线形态
        """
        score = 0.0
        details = {}
        
        try:
            df = ctx.ohlcv
            
            # 1. 趋势强度 (简化计算)
            ma5 = df['close'].tail(5).mean()
            ma20 = df['close'].tail(20).mean() if len(df) >= 20 else ma5
            current_price = df.iloc[-1]['close']
            
            if current_price > ma5 > ma20:
                score += 25
            elif current_price > ma5:
                score += 15
            elif current_price > ma20:
                score += 8
            details['trend'] = 'strong' if current_price > ma5 > ma20 else 'weak'
            
            # 2. 动量指标
            rsi = ctx.technical_indicators.get('rsi', 50)
            if 60 <= rsi <= 80:
                score += 25
            elif 50 <= rsi < 60:
                score += 15
            elif 40 <= rsi < 50:
                score += 8
            details['rsi'] = rsi
            
            # 3. 成交量指标
            volume_ma = df['volume'].tail(5).mean()
            current_volume = df.iloc[-1]['volume']
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
            
            if 1.5 <= volume_ratio <= 3:
                score += 20
            elif 1.2 <= volume_ratio < 1.5:
                score += 12
            elif 1 <= volume_ratio < 1.2:
                score += 6
            details['volume_ratio'] = volume_ratio
            
            # 4. 支撑压力
            resistance_distance = ctx.technical_indicators.get('resistance_distance', 0.05)
            if resistance_distance > 0.05:  # 离压力位还有5%以上空间
                score += 15
            elif resistance_distance > 0.03:
                score += 10
            elif resistance_distance > 0.01:
                score += 5
            details['resistance_distance'] = resistance_distance
            
            # 5. 形态识别
            pattern = ctx.technical_indicators.get('pattern', 'none')
            bullish_patterns = ['cup_handle', 'ascending_triangle', 'flag', 'pennant']
            if pattern in bullish_patterns:
                score += 15
            elif pattern == 'consolidation':
                score += 8
            details['pattern'] = pattern
            
            self.logger.info(f"{symbol} 技术分析得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.9, # 技术指标是客观计算的
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class PositionAgent:
    """仓位控制Agent - 动态仓位管理"""
    
    def __init__(self):
        self.name = "仓位控制Agent"
        self.weight = 0.08
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        仓位控制分析
        
        评分规则：
        1. 市场强度 (25分): 市场整体强弱
        2. 个股强度 (25分): 个股相对强度
        3. 风险评估 (20分): VaR、波动率
        4. 相关性 (15分): 与持仓股相关性
        5. 时机评分 (15分): 入场时机
        """
        score = 0.0
        details = {}
        
        try:
            # 1. 市场强度
            market_strength = ctx.market_mood_score / 100
            if market_strength > 0.7:
                score += 25
            elif market_strength > 0.5:
                score += 18
            elif market_strength > 0.3:
                score += 10
            else:
                score += 5
            details['market_strength'] = market_strength
            
            # 2. 个股强度
            stock_rs = ctx.technical_indicators.get('relative_strength', 50)
            if stock_rs > 80:
                score += 25
            elif stock_rs > 60:
                score += 18
            elif stock_rs > 40:
                score += 10
            else:
                score += 5
            details['stock_rs'] = stock_rs
            
            # 3. 风险评估
            volatility = ctx.technical_indicators.get('volatility', 0.02)
            if volatility < 0.02:
                score += 20
            elif volatility < 0.03:
                score += 15
            elif volatility < 0.05:
                score += 8
            else:
                score += 3
            details['volatility'] = volatility
            
            # 4. 相关性 (假设与现有持仓的相关性)
            correlation = ctx.technical_indicators.get('portfolio_correlation', 0.5)
            if correlation < 0.3:  # 低相关性好
                score += 15
            elif correlation < 0.5:
                score += 10
            elif correlation < 0.7:
                score += 5
            details['correlation'] = correlation
            
            # 5. 时机评分
            timing_score = ctx.technical_indicators.get('timing_score', 50)
            if timing_score > 80:
                score += 15
            elif timing_score > 60:
                score += 10
            elif timing_score > 40:
                score += 5
            details['timing_score'] = timing_score
            
            # 建议仓位
            suggested_position = min(score / 100 * 0.3, 0.25)  # 最大25%仓位
            details['suggested_position'] = f"{suggested_position:.1%}"
            
            self.logger.info(f"{symbol} 仓位控制得分: {score}, 建议仓位: {details['suggested_position']}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.8, # 仓位建议带有预测成分
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class RiskAgent:
    """风险评估Agent - 全面风险评估"""
    
    def __init__(self):
        self.name = "风险评估Agent"
        self.weight = 0.10
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        风险评估分析
        
        评分规则（反向计分，风险越低分数越高）：
        1. 系统性风险 (25分): 市场系统风险
        2. 个股风险 (25分): 个股特定风险
        3. 流动性风险 (20分): 成交量、换手率
        4. 监管风险 (15分): 监管政策影响
        5. 财务风险 (15分): 财务健康度
        """
        score = 100.0  # 从100分开始扣分
        risks = []
        details = {}
        
        try:
            # 1. 系统性风险
            vix = ctx.market_mood_score
            if vix < 20:  # 低风险
                pass  # 不扣分
            elif vix < 40:
                score -= 10
                risks.append("中等系统性风险")
            elif vix < 60:
                score -= 18
                risks.append("较高系统性风险")
            else:
                score -= 25
                risks.append("高系统性风险")
            details['system_risk'] = vix
            
            # 2. 个股风险
            stock_volatility = ctx.technical_indicators.get('volatility', 0.03)
            if stock_volatility < 0.02:
                pass
            elif stock_volatility < 0.04:
                score -= 10
                risks.append("中等波动风险")
            elif stock_volatility < 0.06:
                score -= 18
                risks.append("较高波动风险")
            else:
                score -= 25
                risks.append("高波动风险")
            details['stock_volatility'] = stock_volatility
            
            # 3. 流动性风险
            turnover = ctx.ohlcv.iloc[-1].get('turnover_rate', 0)
            if turnover > 10:
                pass
            elif turnover > 5:
                score -= 8
                risks.append("一般流动性")
            elif turnover > 2:
                score -= 15
                risks.append("流动性较差")
            else:
                score -= 20
                risks.append("流动性风险高")
            details['liquidity'] = turnover
            
            # 4. 监管风险
            regulatory_risk = ctx.fundamental_data.get('regulatory_risk', 'low')
            if regulatory_risk == 'low':
                pass
            elif regulatory_risk == 'medium':
                score -= 8
                risks.append("中等监管风险")
            elif regulatory_risk == 'high':
                score -= 15
                risks.append("高监管风险")
            details['regulatory'] = regulatory_risk
            
            # 5. 财务风险
            financial_health = ctx.fundamental_data.get('financial_score', 80)
            if financial_health > 80:
                pass
            elif financial_health > 60:
                score -= 8
                risks.append("财务状况一般")
            elif financial_health > 40:
                score -= 12
                risks.append("财务风险较高")
            else:
                score -= 15
                risks.append("财务风险高")
            details['financial_health'] = financial_health
            
            details['risk_list'] = risks
            details['risk_level'] = 'low' if score > 80 else 'medium' if score > 60 else 'high'
            
            self.logger.info(f"{symbol} 风险评估得分: {score}, 风险: {risks}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": max(score, 0),  # 确保不为负数
            "confidence": 0.95, # 风险评估基于事实数据，置信度高
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class NewsAgent:
    """消息面Agent - 新闻和公告分析"""
    
    def __init__(self):
        self.name = "消息面Agent"
        self.weight = 0.08
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        消息面分析
        
        评分规则：
        1. 重大利好 (30分): 重组、大订单等
        2. 一般利好 (20分): 业绩预增、合作等
        3. 中性消息 (10分): 常规公告
        4. 热度加成 (20分): 消息传播度
        5. 时效性 (20分): 消息新鲜度
        """
        score = 0.0
        details = {}
        news_impact = []
        
        try:
            # 分析新闻标题
            major_positive = ['重组', '收购', '大单', '中标', '突破', '新高']
            moderate_positive = ['增长', '合作', '签约', '利好', '上调']
            negative = ['亏损', '下调', '处罚', '调查', '减持']
            
            major_count = 0
            moderate_count = 0
            negative_count = 0
            
            for title in ctx.news_titles:
                if any(word in title for word in major_positive):
                    major_count += 1
                    news_impact.append(f"重大利好: {title[:30]}...")
                elif any(word in title for word in moderate_positive):
                    moderate_count += 1
                    news_impact.append(f"一般利好: {title[:30]}...")
                elif any(word in title for word in negative):
                    negative_count += 1
                    news_impact.append(f"利空: {title[:30]}...")
            
            # 1. 重大利好
            if major_count > 0:
                score += min(major_count * 15, 30)
            details['major_positive'] = major_count
            
            # 2. 一般利好
            if moderate_count > 0:
                score += min(moderate_count * 10, 20)
            details['moderate_positive'] = moderate_count
            
            # 3. 利空扣分
            if negative_count > 0:
                score -= min(negative_count * 10, 20)
            details['negative'] = negative_count
            
            # 4. 热度加成
            total_news = len(ctx.news_titles)
            if total_news > 10:
                score += 20
            elif total_news > 5:
                score += 15
            elif total_news > 2:
                score += 10
            elif total_news > 0:
                score += 5
            details['news_count'] = total_news
            
            # 5. 时效性 (假设有时间戳)
            news_freshness = ctx.fundamental_data.get('news_freshness', 0.5)
            score += news_freshness * 20
            details['freshness'] = news_freshness
            
            details['news_impact'] = news_impact[:5]  # 只保留前5条
            
            self.logger.info(f"{symbol} 消息面得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": max(score, 0),
            "confidence": 0.7, # 新闻解读有歧义，置信度中等
            "details": details,
            "timestamp": datetime.now().isoformat()
        }


class SectorAgent:
    """板块协同Agent - 板块联动分析"""
    
    def __init__(self):
        self.name = "板块协同Agent"
        self.weight = 0.07
        self.logger = logging.getLogger(self.name)
        
    async def analyze(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        板块协同分析
        
        评分规则：
        1. 板块强度 (30分): 板块整体涨幅
        2. 板块地位 (25分): 在板块中的地位
        3. 板块轮动 (20分): 是否处于轮动风口
        4. 板块资金 (15分): 板块资金流入
        5. 板块持续性 (10分): 板块热度持续天数
        """
        score = 0.0
        details = {}
        
        try:
            # 1. 板块强度
            sector_strength = ctx.sector_heat.get('sector_change', 0)
            if sector_strength > 5:
                score += 30
            elif sector_strength > 3:
                score += 20
            elif sector_strength > 1:
                score += 10
            elif sector_strength > 0:
                score += 5
            details['sector_strength'] = sector_strength
            
            # 2. 板块地位
            sector_position = ctx.sector_heat.get(f'{symbol}_sector_rank', 10)
            if sector_position <= 3:
                score += 25
            elif sector_position <= 5:
                score += 18
            elif sector_position <= 10:
                score += 10
            elif sector_position <= 20:
                score += 5
            details['sector_position'] = sector_position
            
            # 3. 板块轮动
            rotation_score = ctx.sector_heat.get('rotation_score', 50)
            if rotation_score > 80:
                score += 20
            elif rotation_score > 60:
                score += 15
            elif rotation_score > 40:
                score += 8
            details['rotation_score'] = rotation_score
            
            # 4. 板块资金
            sector_money = ctx.sector_heat.get('sector_money_flow', 0)
            if sector_money > 20:  # 亿元
                score += 15
            elif sector_money > 10:
                score += 10
            elif sector_money > 5:
                score += 5
            details['sector_money'] = sector_money
            
            # 5. 板块持续性
            sector_days = ctx.sector_heat.get('hot_days', 0)
            if sector_days > 5:
                score += 10
            elif sector_days > 3:
                score += 7
            elif sector_days > 1:
                score += 3
            details['sector_hot_days'] = sector_days
            
            self.logger.info(f"{symbol} 板块协同得分: {score}, 详情: {details}")
            
        except Exception as e:
            self.logger.error(f"{symbol} 分析失败: {e}")
            score = 0
            details = {"error": str(e)}
            
        return {
            "score": score,
            "confidence": 0.75, # 板块分析介于宏观和微观之间
            "details": details,
            "timestamp": datetime.now().isoformat()
        }





class MarketRegimeMarshal:
    """
    市场风格元帅 - v2.1 新增
    系统的最高指挥官，根据市场整体状态，动态调整系统作战模式。
    """
    def __init__(self, config: Dict = None):
        self.name = "市场风格元帅"
        self.logger = logging.getLogger(self.name)
        # TODO: 应从config加载更多配置
        self.config = config if config is not None else {}

    async def get_operational_command(self) -> Dict[str, Any]:
        """
        分析当前市场，发布作战指令。
        
        作战指令包含：
        - market_regime: 当前市场风格 (e.g., '主升浪', '混沌期', '退潮期')
        - agent_weights: 针对当前风格，动态调整各Agent的权重
        - risk_parameters: 动态调整风控参数 (e.g., 总仓位, 止损线)

        NOTE: 这是一个简化的实现。在实际应用中，这里需要接入大盘数据
              （如成交量、涨跌家数、情绪指标等）来做更精准的判断。
        """
        self.logger.info("元帅正在分析市场，制定今日作战指令...")
        
        # 简化逻辑：此处暂时返回一个默认的“游资混战期”指令
        # 实际应基于对市场数据的分析来选择
        market_regime = "HOT_MONEY_CHAOS (游资混战期)"
        
        # 默认权重，后续可根据不同风格调整
        agent_weights = {
            'zt_quality': 0.15,
            'leader': 0.15,
            'auction': 0.12,
            'money_flow': 0.12,
            'emotion': 0.10,
            'risk': -0.10,  # 风险Agent为负权重或在决策时特殊处理
            'technical': 0.08,
            'position': 0.08,
            'news': 0.08,
            'sector': 0.07
        }
        
        # 默认风控参数
        risk_parameters = {
            'max_total_position': 0.8,
            'default_stop_loss': 0.05,
            'default_take_profit': 0.10,
        }
        
        command = {
            "market_regime": market_regime,
            "agent_weights": agent_weights,
            "risk_parameters": risk_parameters
        }
        
        self.logger.info(f"今日作战指令已生成: {command}")
        
        return command


class IntegratedDecisionAgent:
    """综合决策Agent - 汇总所有Agent分析结果做最终决策"""
    
    def __init__(self):
        self.name = "综合决策Agent"
        self.logger = logging.getLogger(self.name)
        
        # 初始化所有子Agent
        self.agents = {
            'zt_quality': ZTQualityAgent(),
            'leader': LeaderAgent(),
            'auction': AuctionAgent(),
            'money_flow': MoneyFlowAgent(),
            'emotion': EmotionAgent(),
            'technical': TechnicalAgent(),
            'position': PositionAgent(),
            'risk': RiskAgent(),
            'news': NewsAgent(),
            'sector': SectorAgent()
        }
        
    async def analyze_parallel(self, symbol: str, ctx: MarketContext) -> Dict[str, Any]:
        """
        并行执行所有Agent分析
        
        Returns:
            综合分析结果和交易决策
        """
        start_time = datetime.now()
        self.logger.info(f"开始并行分析 {symbol}")
        
        # 创建所有分析任务
        tasks = []
        for name, agent in self.agents.items():
            task = asyncio.create_task(agent.analyze(symbol, ctx))
            tasks.append((name, task))
        
        # 等待所有任务完成
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                self.logger.error(f"{name} 分析失败: {e}")
                results[name] = {"score": 0, "details": {"error": str(e)}}
        
        # 计算综合得分和综合置信度
        total_score = 0
        total_confidence = 0
        total_weight = 0
        weighted_score = 0
        details = {}
        
        for name, result in results.items():
            agent = self.agents[name]
            score = result.get('score', 0)
            confidence = result.get('confidence', 0.5) # 默认置信度0.5
            weight = agent.weight
            
            weighted_score += score * weight
            total_confidence += confidence * weight # 用权重加权计算综合置信度
            total_weight += weight
            
            total_score += score
            details[name] = {
                'score': score,
                'confidence': confidence,
                'weight': weight,
                'weighted': score * weight,
                'details': result.get('details', {})
            }
        
        if total_weight > 0:
            total_confidence /= total_weight
        
        # 生成交易建议
        decision = self._make_decision(weighted_score, total_confidence, results)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"{symbol} 分析完成，耗时: {elapsed:.2f}秒，综合得分: {weighted_score:.2f}")
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'total_score': total_score,
            'weighted_score': weighted_score,
            'total_confidence': total_confidence, # v2.1 新增：综合置信度
            'decision': decision,
            'details': details,
            'analysis_time': elapsed
        }
    
    def _make_decision(self, score: float, confidence: float, results: Dict) -> Dict[str, Any]:
        """ v2.1 升级：基于“分数+置信度”双重门槛生成交易决策 """

        # 从Agent结果中提取关键决策参数
        risk_details = results.get('risk', {}).get('details', {})
        risk_level = risk_details.get('risk_level', 'high')  # 默认高风险

        position_details = results.get('position', {}).get('details', {})
        position_suggestion = position_details.get('suggested_position', '0%') # 默认0仓位

        # TODO: 这些阈值后续应从 config.yaml 文件动态加载
        strong_buy_score = 85
        strong_buy_confidence = 0.75
        buy_score = 70
        buy_confidence = 0.65
        watch_score = 40

        if score >= strong_buy_score and confidence >= strong_buy_confidence and risk_level != 'high':
            action = 'strong_buy'
            position = position_suggestion
            reason = "多项指标强势，且确定性高，风险可控"
        elif score >= buy_score and confidence >= buy_confidence and risk_level in ['low', 'medium']:
            action = 'buy'
            position = position_suggestion
            reason = "指标良好，确定性较高，可适度参与"
        elif score >= watch_score:
            action = 'watch'
            position = '0%'
            reason = "指标中性或确定性不足，建议观望"
        else:
            action = 'avoid'
            position = '0%'
            reason = "指标偏弱或不确定性高，建议回避"

        return {
            'action': action,
            'confidence': confidence, # 返回的是综合置信度
            'position': position,
            'reason': reason,
            'risk_level': risk_level,
            # v2.1 升级：增加决策归因追溯
            'decision_trace': self._generate_decision_trace(results),
            'entry_price': None,  # 需要根据具体策略设定
            'stop_loss': None,    # 需要根据具体策略设定
            'take_profit': None   # 需要根据具体策略设定
        }

    def _generate_decision_trace(self, results: Dict) -> Dict:
        """ v2.1 新增：生成决策归因路径 """
        trace = {
            "contributions": [],
            "warnings": []
        }
        # 排序，找到贡献最大的因子
        sorted_agents = sorted(results.items(), key=lambda item: item[1].get('score', 0) * self.agents[item[0]].weight, reverse=True)

        for name, result in sorted_agents:
            agent = self.agents[name]
            trace["contributions"].append({
                "agent": name,
                "score": result.get('score', 0),
                "confidence": result.get('confidence', 0.5),
                "weight": agent.weight,
                "contribution": result.get('score', 0) * agent.weight
            })
        
        # 记录风险项
        risk_details = results.get('risk', {}).get('details', {})
        if risk_details.get('risk_list'):
            trace["warnings"] = risk_details['risk_list']

        return trace


# 导出便捷函数
async def analyze_stock(symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析单只股票
    
    Args:
        symbol: 股票代码
        market_data: 市场数据
        
    Returns:
        分析结果和交易决策
    """
    # 构建市场上下文
    ctx = MarketContext(
        ohlcv=market_data.get('ohlcv', pd.DataFrame()),
        news_titles=market_data.get('news_titles', []),
        lhb_netbuy=market_data.get('lhb_netbuy', 0),
        market_mood_score=market_data.get('market_mood_score', 50),
        sector_heat=market_data.get('sector_heat', {}),
        money_flow=market_data.get('money_flow', {}),
        technical_indicators=market_data.get('technical_indicators', {}),
        fundamental_data=market_data.get('fundamental_data', {})
    )
    
    # 创建综合决策Agent并分析
    agent = IntegratedDecisionAgent()
    result = await agent.analyze_parallel(symbol, ctx)
    
    return result


async def batch_analyze(symbols: List[str], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    批量分析多只股票
    
    Args:
        symbols: 股票代码列表
        market_data: 市场数据
        
    Returns:
        所有股票的分析结果
    """
    tasks = []
    for symbol in symbols:
        task = asyncio.create_task(analyze_stock(symbol, market_data))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # 按得分排序
    results.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    return results


# 测试代码
if __name__ == "__main__":
    async def test():
        # 模拟市场数据
        test_data = {
            'ohlcv': pd.DataFrame({
                'close': [10, 10.5, 11, 11.5, 12],
                'volume': [1000, 1200, 1500, 1800, 2000],
                'turnover_rate': [5, 6, 8, 10, 12]
            }),
            'news_titles': [
                '公司获得大订单',
                '行业景气度提升',
                '机构调研密集'
            ],
            'lhb_netbuy': 2.5,
            'market_mood_score': 65,
            'sector_heat': {
                'sector_change': 3.5,
                '000001_rank': 2,
                '000001_heat': 75
            },
            'money_flow': {
                '000001_main': 1.8,
                '000001_super_ratio': 0.18
            },
            'technical_indicators': {
                'rsi': 68,
                'volatility': 0.025,
                'seal_ratio': 0.08,
                'zt_time': '10:30'
            },
            'fundamental_data': {
                'financial_score': 75,
                'regulatory_risk': 'low'
            }
        }
        
        # 分析单只股票
        result = await analyze_stock('000001', test_data)
        
        print(f"股票: {result['symbol']}")
        print(f"综合得分: {result['weighted_score']:.2f}")
        print(f"决策: {result['decision']}")
        print(f"耗时: {result['analysis_time']:.2f}秒")
        
        # 批量分析
        symbols = ['000001', '000002', '000003']
        batch_results = await batch_analyze(symbols, test_data)
        
        print("\n批量分析结果:")
        for r in batch_results:
            print(f"{r['symbol']}: {r['weighted_score']:.2f} - {r['decision']['action']}")
    
    asyncio.run(test())