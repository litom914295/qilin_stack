"""
TradingAgents工具库 (P1-2)

包含6类工具API:
1. NewsAPITool - 新闻API
2. MarketDataTool - 实时行情
3. FinancialReportTool - 财报解析 (占位)
4. SentimentTool - 情绪分析 (占位)
5. EconomicCalendarTool - 经济日历 (占位)
6. TechnicalIndicatorTool - 技术指标 (占位)
"""

from .news_api import NewsAPITool, NewsArticle
from .market_data import MarketDataTool
from .tool_manager import ToolManager, ToolCallResult, ToolMetadata

__all__ = [
    'NewsAPITool',
    'NewsArticle',
    'MarketDataTool',
    'ToolManager',
    'ToolCallResult',
    'ToolMetadata',
]

__version__ = "1.0.0"
