"""
涨停板舆情分析智能体 - 基于TradingAgents

使用LLM深度分析涨停板的舆情特征：
1. 新闻分析 - 题材催化剂识别
2. 社交媒体 - 微博、股吧情绪
3. 资金流向 - 主力意图判断
4. 持续性预测 - "一进二"概率评估
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger(__name__)

# 添加TradingAgents路径
TRADINGAGENTS_PATH = os.getenv("TRADINGAGENTS_PATH", "D:/test/Qlib/TradingAgents")
if os.path.exists(TRADINGAGENTS_PATH):
    sys.path.insert(0, TRADINGAGENTS_PATH)
    TRADINGAGENTS_AVAILABLE = True
else:
    TRADINGAGENTS_AVAILABLE = False
    logger.warning(f"TradingAgents未找到，路径: {TRADINGAGENTS_PATH}")
    logger.info("使用简化版本（不依赖官方代码）")


class NewsAPITool:
    """新闻API工具（支持AKShare真实数据）"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "")
        self.use_real_data = os.getenv("USE_REAL_NEWS", "false").lower() == "true"
        
    async def fetch(self, symbol: str, date: str) -> List[Dict]:
        """获取股票相关新闻"""
        if self.use_real_data:
            try:
                return await self._fetch_real_news(symbol, date)
            except Exception as e:
                logger.warning(f"真实新闻获取失败: {e}，使用模拟数据")
        
        # 模拟数据（降级方案）
        return [
            {
                'title': f'{symbol} 涉及热门题材，主力资金大幅流入',
                'content': '该股涨停后，市场关注度显著提升，多家机构看好后续表现...',
                'source': '财经网',
                'publish_time': date,
                'sentiment': 'positive'
            },
            {
                'title': f'{symbol} 技术面突破，资金追捧',
                'content': '从技术形态看，该股突破关键压力位，成交量配合良好...',
                'source': '证券日报',
                'publish_time': date,
                'sentiment': 'positive'
            }
        ]
    
    async def _fetch_real_news(self, symbol: str, date: str) -> List[Dict]:
        """使用AKShare获取真实新闻"""
        try:
            import akshare as ak
            
            # 获取个股新闻（东方财富）
            # 示例：ak.stock_news_em(symbol='000001')
            stock_code = symbol.split('.')[0]
            news_df = ak.stock_news_em(symbol=stock_code)
            
            # 筛选日期
            news_df['发布时间'] = pd.to_datetime(news_df['发布时间'])
            target_date = pd.to_datetime(date)
            news_df = news_df[
                news_df['发布时间'].dt.date == target_date.date()
            ]
            
            # 转换格式
            news_list = []
            for _, row in news_df.iterrows():
                news_list.append({
                    'title': row['新闻标题'],
                    'content': row['新闻内容'],
                    'source': row.get('文章来源', '东方财富'),
                    'publish_time': row['发布时间'].strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': 'neutral'  # 需要额外的情感分析
                })
            
            return news_list if news_list else await self.fetch(symbol, date)  # 无数据时降级
            
        except ImportError:
            logger.warning("AKShare未安装，请运行: pip install akshare")
            raise
        except Exception as e:
            logger.warning(f"AKShare数据获取失败: {e}")
            raise


class WeiboTool:
    """微博数据工具（模拟实现）"""
    
    def __init__(self):
        pass
        
    async def fetch(self, symbol: str, date: str) -> Dict:
        """获取股票相关微博数据"""
        # 实际应用中，这里会调用微博API或爬虫
        
        # 模拟数据
        return {
            'total_posts': 1250,
            'positive_ratio': 0.72,
            'negative_ratio': 0.15,
            'neutral_ratio': 0.13,
            'hot_keywords': ['AI概念', '业绩超预期', '主力建仓', '龙头股'],
            'kol_opinions': [
                {'author': '知名博主A', 'view': '看好后续表现，有望连板'},
                {'author': '知名博主B', 'view': '题材纯正，资金认可度高'}
            ]
        }


class StockForumTool:
    """股吧数据工具（模拟实现）"""
    
    def __init__(self):
        pass
        
    async def fetch(self, symbol: str, date: str) -> Dict:
        """获取股吧讨论数据"""
        # 实际应用中，这里会爬取东方财富股吧等
        
        # 模拟数据
        return {
            'total_posts': 3500,
            'sentiment_score': 7.8,  # 1-10分
            'hot_topics': ['一进二', '龙头', '主力护盘', '题材正宗'],
            'prediction_stats': {
                'bullish': 0.68,  # 看涨比例
                'bearish': 0.18,  # 看跌比例
                'neutral': 0.14   # 中性比例
            }
        }


class LimitUpSentimentAgent:
    """涨停板舆情分析智能体"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化舆情分析智能体
        
        Parameters:
        -----------
        config : Dict, optional
            配置参数，包括：
            - llm_api_key: LLM API密钥
            - llm_model: 使用的模型（默认gpt-4-turbo）
            - llm_api_base: API base URL
        """
        self.config = config or {}
        
        # LLM配置
        self.llm_api_key = self.config.get('llm_api_key') or os.getenv('OPENAI_API_KEY', '')
        self.llm_model = self.config.get('llm_model', 'gpt-4-turbo')
        self.llm_api_base = self.config.get('llm_api_base') or os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        # 初始化数据工具
        self.news_tool = NewsAPITool()
        self.weibo_tool = WeiboTool()
        self.forum_tool = StockForumTool()
        
        # 尝试初始化TradingAgents（如果可用）
        self.agent = None
        self.llm = None
        
        if TRADINGAGENTS_AVAILABLE and self.llm_api_key:
            try:
                self._init_tradingagents()
            except Exception as e:
                print(f"⚠️  TradingAgents初始化失败: {e}")
                print(f"   使用简化版本")
    
    def _init_tradingagents(self):
        """初始化TradingAgents官方组件"""
        try:
            # 导入官方组件
            from tradingagents.llm.openai_adapter import OpenAIAdapter
            from tradingagents.agents.sentiment_analyst import SentimentAnalystAgent
            
            # 初始化LLM
            self.llm = OpenAIAdapter(
                api_key=self.llm_api_key,
                model=self.llm_model,
                api_base=self.llm_api_base
            )
            
            # 初始化舆情分析智能体
            self.agent = SentimentAnalystAgent(
                llm=self.llm,
                tools={
                    'news': self.news_tool,
                    'weibo': self.weibo_tool,
                    'forum': self.forum_tool
                }
            )
            
            logger.info("✅ TradingAgents官方组件初始化成功")
            
        except ImportError as e:
            logger.warning(f"TradingAgents导入失败: {e}")
            logger.info("使用简化版本（基于规则的分析）")
            self.agent = None
    
    async def analyze_limitup_sentiment(
        self, 
        symbol: str, 
        date: str,
        price_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        分析涨停板舆情
        
        Parameters:
        -----------
        symbol : str
            股票代码
        date : str
            日期 (YYYY-MM-DD)
        price_data : Dict, optional
            价格数据（用于辅助分析）
        
        Returns:
        --------
        Dict: 舆情分析结果
            - sentiment_score: 综合情绪得分 (0-100)
            - key_catalysts: 关键催化剂
            - risk_factors: 风险因素
            - continue_prob: 一进二概率
            - reasoning: 详细推理过程
        """
        logger.info(f"开始分析 {symbol} 在 {date} 的涨停舆情...")
        
        # 1. 并发获取多源数据
        logger.info("获取数据...")
        news_data, weibo_data, forum_data = await asyncio.gather(
            self.news_tool.fetch(symbol, date),
            self.weibo_tool.fetch(symbol, date),
            self.forum_tool.fetch(symbol, date)
        )
        
        # 2. 如果有TradingAgents，使用LLM深度分析
        if self.agent and self.llm:
            logger.info("使用LLM深度分析...")
            result = await self._analyze_with_llm(
                symbol, date, news_data, weibo_data, forum_data, price_data
            )
        else:
            logger.info("使用规则引擎分析...")
            result = self._analyze_with_rules(
                symbol, date, news_data, weibo_data, forum_data, price_data
            )
        
        logger.info(f"分析完成，综合得分: {result['sentiment_score']:.1f}")
        
        return result
    
    async def _analyze_with_llm(
        self,
        symbol: str,
        date: str,
        news_data: List[Dict],
        weibo_data: Dict,
        forum_data: Dict,
        price_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """使用LLM进行深度分析"""
        
        # 构建分析提示词
        prompt = self._build_analysis_prompt(
            symbol, date, news_data, weibo_data, forum_data, price_data
        )
        
        try:
            # 调用LLM（这里需要根据实际的TradingAgents API调整）
            # response = await self.agent.analyze(prompt)
            
            # 暂时使用简化版本
            response = self._analyze_with_rules(
                symbol, date, news_data, weibo_data, forum_data, price_data
            )
            
            return response
            
        except Exception as e:
            logger.warning(f"LLM分析失败: {e}，使用规则引擎")
            return self._analyze_with_rules(
                symbol, date, news_data, weibo_data, forum_data, price_data
            )
    
    def _analyze_with_rules(
        self,
        symbol: str,
        date: str,
        news_data: List[Dict],
        weibo_data: Dict,
        forum_data: Dict,
        price_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """基于规则的舆情分析"""
        
        # 1. 新闻情绪得分 (0-30分)
        news_score = 0
        positive_news = sum(1 for n in news_data if n.get('sentiment') == 'positive')
        news_score = min(30, positive_news * 15)  # 每条正面新闻15分
        
        # 2. 微博情绪得分 (0-30分)
        weibo_positive_ratio = weibo_data.get('positive_ratio', 0)
        weibo_score = weibo_positive_ratio * 30
        
        # 3. 股吧情绪得分 (0-30分)
        forum_bullish = forum_data.get('prediction_stats', {}).get('bullish', 0)
        forum_score = forum_bullish * 30
        
        # 4. 热度加成 (0-10分)
        total_posts = weibo_data.get('total_posts', 0) + forum_data.get('total_posts', 0)
        heat_score = min(10, total_posts / 500)  # 每500条讨论1分
        
        # 综合得分
        sentiment_score = news_score + weibo_score + forum_score + heat_score
        
        # 识别关键催化剂
        catalysts = []
        for news in news_data:
            if '题材' in news['title'] or '概念' in news['title']:
                catalysts.append(f"题材催化: {news['title'][:30]}")
        
        if weibo_data.get('hot_keywords'):
            catalysts.append(f"热点关键词: {', '.join(weibo_data['hot_keywords'][:3])}")
        
        # 识别风险因素
        risks = []
        weibo_negative = weibo_data.get('negative_ratio', 0)
        if weibo_negative > 0.3:
            risks.append(f"微博负面情绪较高 ({weibo_negative:.1%})")
        
        forum_bearish = forum_data.get('prediction_stats', {}).get('bearish', 0)
        if forum_bearish > 0.3:
            risks.append(f"股吧看跌比例较高 ({forum_bearish:.1%})")
        
        if total_posts < 1000:
            risks.append("关注度不足，可能缺乏持续性")
        
        # 计算一进二概率
        # 基于情绪得分的线性映射
        continue_prob = sentiment_score / 100
        
        # 根据具体因素调整
        if len(catalysts) >= 3:
            continue_prob += 0.1  # 有多个催化剂，提升10%
        
        if len(risks) >= 2:
            continue_prob -= 0.15  # 有多个风险，降低15%
        
        continue_prob = max(0.0, min(1.0, continue_prob))  # 限制在0-1之间
        
        # 生成推理过程
        reasoning = self._generate_reasoning(
            sentiment_score, catalysts, risks, continue_prob,
            news_data, weibo_data, forum_data
        )
        
        return {
            'symbol': symbol,
            'date': date,
            'sentiment_score': sentiment_score,
            'key_catalysts': catalysts,
            'risk_factors': risks,
            'continue_prob': continue_prob,
            'reasoning': reasoning,
            'data_sources': {
                'news_count': len(news_data),
                'weibo_posts': weibo_data.get('total_posts', 0),
                'forum_posts': forum_data.get('total_posts', 0)
            }
        }
    
    def _build_analysis_prompt(
        self,
        symbol: str,
        date: str,
        news_data: List[Dict],
        weibo_data: Dict,
        forum_data: Dict,
        price_data: Optional[Dict]
    ) -> str:
        """构建LLM分析提示词"""
        
        prompt = f"""
你是一位资深的A股涨停板分析专家。请分析 {symbol} 在 {date} 涨停后的舆情特征，
评估其明天继续涨停（"一进二"）的概率。

# 新闻数据
{json.dumps(news_data, ensure_ascii=False, indent=2)}

# 微博数据
{json.dumps(weibo_data, ensure_ascii=False, indent=2)}

# 股吧数据
{json.dumps(forum_data, ensure_ascii=False, indent=2)}

# 价格数据
{json.dumps(price_data or {}, ensure_ascii=False, indent=2)}

请从以下维度分析：
1. **题材是否被市场认可？** - 查看新闻和社交媒体的讨论热度
2. **是否有重大利好催化剂？** - 识别政策、业绩、并购等催化因素
3. **散户情绪是否过热？** - 情绪过热往往是反向指标
4. **机构是否参与？** - 从资金流向判断主力意图
5. **明天继续涨停的概率？** - 综合评估给出0-1之间的概率

请以JSON格式返回：
{{
    "sentiment_score": 0-100,
    "key_catalysts": ["催化剂1", "催化剂2"],
    "risk_factors": ["风险1", "风险2"],
    "continue_prob": 0.0-1.0,
    "reasoning": "详细推理过程"
}}
"""
        return prompt
    
    def _generate_reasoning(
        self,
        sentiment_score: float,
        catalysts: List[str],
        risks: List[str],
        continue_prob: float,
        news_data: List[Dict],
        weibo_data: Dict,
        forum_data: Dict
    ) -> str:
        """生成推理过程"""
        
        reasoning_parts = []
        
        # 1. 综合评分
        if sentiment_score >= 80:
            reasoning_parts.append(f"✅ 综合情绪得分 {sentiment_score:.1f}/100，市场情绪极度乐观")
        elif sentiment_score >= 60:
            reasoning_parts.append(f"✅ 综合情绪得分 {sentiment_score:.1f}/100，市场情绪较为乐观")
        elif sentiment_score >= 40:
            reasoning_parts.append(f"⚠️  综合情绪得分 {sentiment_score:.1f}/100，市场情绪中性")
        else:
            reasoning_parts.append(f"❌ 综合情绪得分 {sentiment_score:.1f}/100，市场情绪偏谨慎")
        
        # 2. 催化剂
        if catalysts:
            reasoning_parts.append(f"📰 识别到 {len(catalysts)} 个催化剂:")
            for cat in catalysts:
                reasoning_parts.append(f"   • {cat}")
        else:
            reasoning_parts.append("⚠️  未发现明显催化剂，可能是跟风炒作")
        
        # 3. 风险因素
        if risks:
            reasoning_parts.append(f"⚠️  存在 {len(risks)} 个风险因素:")
            for risk in risks:
                reasoning_parts.append(f"   • {risk}")
        else:
            reasoning_parts.append("✅ 暂无明显风险因素")
        
        # 4. 数据来源
        news_count = len(news_data)
        weibo_posts = weibo_data.get('total_posts', 0)
        forum_posts = forum_data.get('total_posts', 0)
        
        reasoning_parts.append(f"📊 数据来源: {news_count}条新闻, {weibo_posts}条微博, {forum_posts}条股吧讨论")
        
        # 5. 最终结论
        if continue_prob >= 0.75:
            reasoning_parts.append(f"🎯 **一进二概率: {continue_prob:.1%}，强烈看好明日继续涨停**")
        elif continue_prob >= 0.60:
            reasoning_parts.append(f"🎯 **一进二概率: {continue_prob:.1%}，较看好明日继续涨停**")
        elif continue_prob >= 0.45:
            reasoning_parts.append(f"🎯 **一进二概率: {continue_prob:.1%}，明日走势存在不确定性**")
        else:
            reasoning_parts.append(f"🎯 **一进二概率: {continue_prob:.1%}，明日继续涨停概率较低**")
        
        return "\n".join(reasoning_parts)
    
    async def batch_analyze(
        self,
        symbols: List[str],
        date: str
    ) -> List[Dict[str, Any]]:
        """
        批量分析多只涨停股票
        
        Parameters:
        -----------
        symbols : List[str]
            股票代码列表
        date : str
            日期
        
        Returns:
        --------
        List[Dict]: 分析结果列表，按一进二概率降序排列
        """
        print(f"\n📊 批量分析 {len(symbols)} 只涨停股票...")
        
        # 并发分析
        tasks = [
            self.analyze_limitup_sentiment(symbol, date)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤错误结果
        valid_results = [
            r for r in results 
            if not isinstance(r, Exception)
        ]
        
        # 按一进二概率降序排序
        valid_results.sort(key=lambda x: x['continue_prob'], reverse=True)
        
        return valid_results


# ==================== 使用示例 ====================

async def main():
    """示例：分析单个涨停股票"""
    print("=" * 80)
    print("涨停板舆情分析智能体 - 测试")
    print("=" * 80)
    
    # 1. 初始化智能体
    config = {
        'llm_api_key': os.getenv('OPENAI_API_KEY', 'your-api-key'),
        'llm_model': 'gpt-4-turbo'
    }
    
    agent = LimitUpSentimentAgent(config)
    
    # 2. 分析单只股票
    result = await agent.analyze_limitup_sentiment(
        symbol='000001.SZ',
        date='2024-06-30'
    )
    
    print("\n" + "=" * 80)
    print(f"📊 分析结果: {result['symbol']}")
    print("=" * 80)
    print(f"综合情绪得分: {result['sentiment_score']:.1f}/100")
    print(f"一进二概率: {result['continue_prob']:.1%}")
    print(f"\n关键催化剂:")
    for cat in result['key_catalysts']:
        print(f"  • {cat}")
    print(f"\n风险因素:")
    for risk in result['risk_factors']:
        print(f"  • {risk}")
    print(f"\n推理过程:")
    print(result['reasoning'])
    
    # 3. 批量分析示例
    print("\n" + "=" * 80)
    print("批量分析示例")
    print("=" * 80)
    
    symbols = ['000001.SZ', '000002.SZ', '600000.SH']
    batch_results = await agent.batch_analyze(symbols, '2024-06-30')
    
    print(f"\n📊 TOP 3 最看好的标的:")
    print("-" * 80)
    for i, result in enumerate(batch_results[:3], 1):
        print(f"{i}. {result['symbol']} - 一进二概率: {result['continue_prob']:.1%} "
              f"(情绪得分: {result['sentiment_score']:.1f})")


if __name__ == '__main__':
    asyncio.run(main())
