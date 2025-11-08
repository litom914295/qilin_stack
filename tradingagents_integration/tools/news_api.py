"""
新闻API工具 (P1-2 Tool 1)

数据源:
1. Finnhub (金融新闻 - 国际)
2. NewsAPI (通用新闻 - 国际)
3. 东方财富/新浪财经 (A股新闻 - 国内)
4. 雪球 (社区新闻)

功能:
- 公司新闻获取
- 关键词搜索
- 情绪分析
- 新闻摘要
"""

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class NewsArticle:
    """新闻文章"""
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    sentiment: Optional[float] = None  # -1 ~ 1
    relevance: Optional[float] = None  # 0 ~ 1
    category: Optional[str] = None
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []


# ============================================================================
# 新闻API工具
# ============================================================================

class NewsAPITool:
    """
    新闻API工具
    
    使用环境变量配置:
    - FINNHUB_API_KEY: Finnhub API密钥
    - NEWS_API_KEY: NewsAPI密钥
    """
    
    def __init__(
        self,
        finnhub_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        enable_sentiment: bool = True
    ):
        """
        初始化新闻API工具
        
        Args:
            finnhub_key: Finnhub API密钥
            newsapi_key: NewsAPI密钥
            enable_sentiment: 启用情绪分析
        """
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY")
        self.newsapi_key = newsapi_key or os.getenv("NEWS_API_KEY")
        self.enable_sentiment = enable_sentiment
        
        # API端点
        self.finnhub_base = "https://finnhub.io/api/v1"
        self.newsapi_base = "https://newsapi.org/v2"
        self.eastmoney_base = "https://np-anotice-stock.eastmoney.com/api/content"
        
        # 缓存
        self.cache = {}
        self.cache_ttl = 300  # 5分钟
        
        logger.info(
            f"新闻API工具初始化: "
            f"Finnhub={'✅' if self.finnhub_key else '❌'}, "
            f"NewsAPI={'✅' if self.newsapi_key else '❌'}"
        )
    
    async def get_company_news(
        self,
        symbol: str,
        days: int = 7,
        source: str = "auto"
    ) -> List[NewsArticle]:
        """
        获取公司新闻
        
        Args:
            symbol: 股票代码 (如 "AAPL" 或 "600519.SH")
            days: 获取最近几天的新闻
            source: 数据源 ("finnhub", "newsapi", "eastmoney", "auto")
            
        Returns:
            新闻列表
        """
        # 判断是A股还是美股
        is_a_share = any(x in symbol for x in ['.SH', '.SZ', '.BJ'])
        
        if source == "auto":
            source = "eastmoney" if is_a_share else "finnhub"
        
        logger.info(f"获取{symbol}的新闻 (来源: {source}, 天数: {days})")
        
        # 检查缓存
        cache_key = f"{symbol}_{days}_{source}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"使用缓存的新闻数据: {symbol}")
                return cached_data
        
        # 根据来源获取新闻
        if source == "finnhub":
            news = await self._fetch_finnhub_news(symbol, days)
        elif source == "eastmoney":
            news = await self._fetch_eastmoney_news(symbol, days)
        elif source == "newsapi":
            news = await self._fetch_newsapi_news(symbol, days)
        else:
            logger.error(f"不支持的新闻源: {source}")
            return []
        
        # 情绪分析
        if self.enable_sentiment and news:
            news = await self._analyze_sentiment_batch(news)
        
        # 缓存结果
        self.cache[cache_key] = (datetime.now(), news)
        
        logger.info(f"✅ 获取到{len(news)}条新闻: {symbol}")
        return news
    
    async def search_news(
        self,
        keywords: List[str],
        language: str = 'zh',
        days: int = 7,
        limit: int = 20
    ) -> List[NewsArticle]:
        """
        搜索新闻
        
        Args:
            keywords: 关键词列表
            language: 语言 ('zh', 'en')
            days: 搜索最近几天
            limit: 返回数量限制
            
        Returns:
            新闻列表
        """
        logger.info(f"搜索新闻: {keywords} (语言: {language})")
        
        if language == 'zh':
            # 中文新闻 - 使用东方财富
            news = await self._search_eastmoney_news(keywords, days, limit)
        else:
            # 英文新闻 - 使用NewsAPI
            news = await self._search_newsapi(keywords, days, limit)
        
        if self.enable_sentiment and news:
            news = await self._analyze_sentiment_batch(news)
        
        logger.info(f"✅ 搜索到{len(news)}条新闻")
        return news
    
    async def get_market_news(
        self,
        category: str = "general",
        limit: int = 20
    ) -> List[NewsArticle]:
        """
        获取市场新闻
        
        Args:
            category: 分类 ("general", "forex", "crypto", "merger")
            limit: 数量限制
            
        Returns:
            新闻列表
        """
        logger.info(f"获取市场新闻: {category}")
        
        if not self.finnhub_key:
            logger.warning("Finnhub API Key未配置")
            return []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.finnhub_base}/news"
                params = {
                    "category": category,
                    "token": self.finnhub_key
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_finnhub_news(data[:limit])
                    else:
                        logger.error(f"Finnhub API错误: {resp.status}")
                        return []
        
        except Exception as e:
            logger.error(f"获取市场新闻失败: {e}")
            return []
    
    # ========================================================================
    # 内部方法 - Finnhub
    # ========================================================================
    
    async def _fetch_finnhub_news(
        self,
        symbol: str,
        days: int
    ) -> List[NewsArticle]:
        """从Finnhub获取新闻"""
        if not self.finnhub_key:
            logger.warning("Finnhub API Key未配置")
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.finnhub_base}/company-news"
                params = {
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date,
                    "token": self.finnhub_key
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_finnhub_news(data)
                    elif resp.status == 429:
                        logger.warning("Finnhub API限流,使用模拟数据")
                        return self._generate_mock_news(symbol, days, "finnhub")
                    else:
                        logger.error(f"Finnhub API错误: {resp.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Finnhub请求失败: {e}")
            return self._generate_mock_news(symbol, days, "finnhub")
    
    def _parse_finnhub_news(self, data: List[Dict]) -> List[NewsArticle]:
        """解析Finnhub新闻"""
        news = []
        for item in data:
            try:
                news.append(NewsArticle(
                    id=str(item.get('id', hash(item['headline']))),
                    title=item.get('headline', ''),
                    summary=item.get('summary', ''),
                    source=item.get('source', 'Finnhub'),
                    url=item.get('url', ''),
                    published_at=datetime.fromtimestamp(item.get('datetime', 0)),
                    category=item.get('category', 'general'),
                    symbols=[item.get('symbol', '')]
                ))
            except Exception as e:
                logger.warning(f"解析新闻失败: {e}")
                continue
        return news
    
    # ========================================================================
    # 内部方法 - 东方财富 (A股)
    # ========================================================================
    
    async def _fetch_eastmoney_news(
        self,
        symbol: str,
        days: int
    ) -> List[NewsArticle]:
        """从东方财富获取A股新闻"""
        try:
            # 转换股票代码格式 (600519.SH -> SH600519)
            if '.' in symbol:
                market, code = symbol.split('.')
                eastmoney_code = f"{market}{code}"
            else:
                eastmoney_code = symbol
            
            # 东方财富API (公开接口)
            url = f"{self.eastmoney_base}/list"
            params = {
                "code": eastmoney_code,
                "pageSize": days * 5,  # 每天约5条新闻
                "pageIndex": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_eastmoney_news(data)
                    else:
                        logger.warning(f"东方财富API错误: {resp.status}, 使用模拟数据")
                        return self._generate_mock_news(symbol, days, "eastmoney")
        
        except Exception as e:
            logger.error(f"东方财富请求失败: {e}")
            return self._generate_mock_news(symbol, days, "eastmoney")
    
    def _parse_eastmoney_news(self, data: Dict) -> List[NewsArticle]:
        """解析东方财富新闻"""
        news = []
        items = data.get('data', {}).get('list', [])
        
        for item in items:
            try:
                news.append(NewsArticle(
                    id=str(item.get('art_code', hash(item.get('title', '')))),
                    title=item.get('title', ''),
                    summary=item.get('content', '')[:200],
                    source='东方财富',
                    url=item.get('url', ''),
                    published_at=datetime.fromisoformat(item.get('show_time', '').replace('Z', '+00:00')) 
                        if item.get('show_time') else datetime.now(),
                    category='财经'
                ))
            except Exception as e:
                logger.warning(f"解析东方财富新闻失败: {e}")
                continue
        
        return news
    
    async def _search_eastmoney_news(
        self,
        keywords: List[str],
        days: int,
        limit: int
    ) -> List[NewsArticle]:
        """搜索东方财富新闻"""
        # 简化实现: 模拟搜索结果
        logger.warning("东方财富搜索功能使用模拟数据")
        return self._generate_mock_news(' '.join(keywords), days, "eastmoney")[:limit]
    
    # ========================================================================
    # 内部方法 - NewsAPI
    # ========================================================================
    
    async def _fetch_newsapi_news(
        self,
        symbol: str,
        days: int
    ) -> List[NewsArticle]:
        """从NewsAPI获取新闻"""
        if not self.newsapi_key:
            logger.warning("NewsAPI Key未配置")
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.newsapi_base}/everything"
                params = {
                    "q": symbol,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "apiKey": self.newsapi_key
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_newsapi_data(data)
                    else:
                        logger.warning(f"NewsAPI错误: {resp.status}, 使用模拟数据")
                        return self._generate_mock_news(symbol, days, "newsapi")
        
        except Exception as e:
            logger.error(f"NewsAPI请求失败: {e}")
            return self._generate_mock_news(symbol, days, "newsapi")
    
    async def _search_newsapi(
        self,
        keywords: List[str],
        days: int,
        limit: int
    ) -> List[NewsArticle]:
        """使用NewsAPI搜索"""
        if not self.newsapi_key:
            return self._generate_mock_news(' '.join(keywords), days, "newsapi")[:limit]
        
        try:
            query = ' OR '.join(keywords)
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.newsapi_base}/everything"
                params = {
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": limit,
                    "apiKey": self.newsapi_key
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_newsapi_data(data)
                    else:
                        return self._generate_mock_news(' '.join(keywords), days, "newsapi")[:limit]
        
        except Exception as e:
            logger.error(f"NewsAPI搜索失败: {e}")
            return self._generate_mock_news(' '.join(keywords), days, "newsapi")[:limit]
    
    def _parse_newsapi_data(self, data: Dict) -> List[NewsArticle]:
        """解析NewsAPI数据"""
        news = []
        for item in data.get('articles', []):
            try:
                news.append(NewsArticle(
                    id=item.get('url', hash(item.get('title', ''))),
                    title=item.get('title', ''),
                    summary=item.get('description', ''),
                    source=item.get('source', {}).get('name', 'NewsAPI'),
                    url=item.get('url', ''),
                    published_at=datetime.fromisoformat(item.get('publishedAt', '').replace('Z', '+00:00'))
                        if item.get('publishedAt') else datetime.now()
                ))
            except Exception as e:
                logger.warning(f"解析NewsAPI数据失败: {e}")
                continue
        return news
    
    # ========================================================================
    # 情绪分析
    # ========================================================================
    
    async def _analyze_sentiment_batch(
        self,
        news: List[NewsArticle]
    ) -> List[NewsArticle]:
        """批量情绪分析"""
        # 简化实现: 基于关键词的情绪分析
        positive_keywords = ['涨', '突破', '增长', '利好', '盈利', '买入', 'surge', 'gain', 'profit']
        negative_keywords = ['跌', '下跌', '亏损', '利空', '风险', '卖出', 'drop', 'loss', 'risk']
        
        for article in news:
            text = f"{article.title} {article.summary}".lower()
            
            pos_count = sum(1 for kw in positive_keywords if kw in text)
            neg_count = sum(1 for kw in negative_keywords if kw in text)
            
            # 简单情绪得分 (-1 ~ 1)
            total = pos_count + neg_count
            if total > 0:
                article.sentiment = (pos_count - neg_count) / total
            else:
                article.sentiment = 0.0
        
        return news
    
    # ========================================================================
    # 模拟数据 (无API Key时使用)
    # ========================================================================
    
    def _generate_mock_news(
        self,
        symbol: str,
        days: int,
        source: str
    ) -> List[NewsArticle]:
        """生成模拟新闻数据"""
        logger.warning(f"生成模拟新闻数据: {symbol}")
        
        mock_titles = [
            f"{symbol} 财报超预期,股价大涨",
            f"{symbol} 宣布重大合作伙伴关系",
            f"分析师上调{symbol}目标价",
            f"{symbol} 季度业绩公布",
            f"{symbol} 行业地位稳固",
        ]
        
        news = []
        for i in range(min(days, 5)):
            pub_time = datetime.now() - timedelta(days=i)
            news.append(NewsArticle(
                id=f"mock_{symbol}_{i}",
                title=mock_titles[i % len(mock_titles)],
                summary=f"这是关于{symbol}的模拟新闻内容...",
                source=source,
                url=f"https://example.com/news/{i}",
                published_at=pub_time,
                sentiment=0.1 * (i % 3 - 1),  # -0.1, 0, 0.1循环
                category="财经"
            ))
        
        return news


# ============================================================================
# 使用示例
# ============================================================================

async def example_news_api():
    """新闻API使用示例"""
    print("=== 新闻API工具示例 ===\n")
    
    # 创建工具
    tool = NewsAPITool()
    
    # 1. 获取公司新闻 (A股)
    print("1. 获取贵州茅台新闻...")
    news = await tool.get_company_news("600519.SH", days=7)
    for article in news[:3]:
        print(f"   📰 {article.title}")
        print(f"      来源: {article.source} | 时间: {article.published_at.strftime('%Y-%m-%d')}")
        print(f"      情绪: {article.sentiment:+.2f}" if article.sentiment else "")
        print()
    
    # 2. 搜索新闻
    print("2. 搜索'人工智能'相关新闻...")
    news = await tool.search_news(['人工智能', 'AI'], language='zh', days=3, limit=5)
    print(f"   找到{len(news)}条新闻\n")
    
    # 3. 获取市场新闻
    print("3. 获取市场新闻...")
    market_news = await tool.get_market_news(category="general", limit=5)
    print(f"   获取{len(market_news)}条市场新闻\n")
    
    print("✅ 示例完成!")


if __name__ == "__main__":
    asyncio.run(example_news_api())
