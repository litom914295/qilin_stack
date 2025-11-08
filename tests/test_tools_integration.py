"""
P1-2测试: TradingAgents工具库

测试范围:
1. NewsAPITool - 新闻API
2. MarketDataTool - 行情数据
3. ToolManager - 工具管理器
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents_integration.tools.news_api import NewsAPITool, NewsArticle
from tradingagents_integration.tools.market_data import MarketDataTool


async def test_news_api():
    """测试新闻API"""
    print("=== 测试新闻API ===")
    
    tool = NewsAPITool()
    
    # 测试获取公司新闻
    print("1. 获取公司新闻...")
    news = await tool.get_company_news("600519.SH", days=3)
    print(f"   ✅ 获取到{len(news)}条新闻")
    
    if news:
        article = news[0]
        print(f"   标题: {article.title}")
        print(f"   来源: {article.source}")
        print(f"   情绪: {article.sentiment:+.2f}" if article.sentiment else "")
    
    # 测试搜索新闻
    print("\n2. 搜索新闻...")
    news = await tool.search_news(['AI', '人工智能'], language='zh', limit=5)
    print(f"   ✅ 搜索到{len(news)}条新闻")
    
    print()


async def test_market_data():
    """测试行情数据"""
    print("=== 测试行情数据 ===")
    
    tool = MarketDataTool()
    
    # 测试实时行情
    print("1. 获取实时行情...")
    quote = await tool.get_realtime_quote("600519.SH")
    print(f"   ✅ 价格: {quote['price']:.2f}")
    print(f"      涨跌: {quote['change']:+.2f}%")
    print(f"      成交量: {quote['volume']:,}")
    
    # 测试盘口数据
    print("\n2. 获取盘口数据...")
    order_book = await tool.get_order_book("600519.SH", depth=5)
    print(f"   ✅ 买盘档位: {len(order_book['bids'])}")
    print(f"      卖盘档位: {len(order_book['asks'])}")
    
    print()


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("P1-2工具库集成测试")
    print("=" * 60)
    print()
    
    # 测试新闻API
    await test_news_api()
    
    # 测试行情数据
    await test_market_data()
    
    print("=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
