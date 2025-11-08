"""
工具管理器 (P1-2)

统一管理6类工具API:
1. NewsAPITool - 新闻API
2. MarketDataTool - 实时行情
3. FinancialReportTool - 财报解析
4. SentimentTool - 情绪分析
5. EconomicCalendarTool - 经济日历
6. TechnicalIndicatorTool - 技术指标

功能:
- 工具注册和调用
- 调用日志记录
- 错误处理和重试
- 性能监控
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    category: str
    version: str
    enabled: bool = True


# ============================================================================
# 工具管理器
# ============================================================================

class ToolManager:
    """
    工具管理器
    
    统一管理所有工具API,提供:
    - 工具注册
    - 统一调用接口
    - 错误处理
    - 调用日志
    - 性能监控
    """
    
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.call_history: List[ToolCallResult] = []
        self.max_history = 1000
        
        # 初始化所有工具
        self._initialize_tools()
        
        logger.info(f"工具管理器初始化完成: {len(self.tools)}个工具")
    
    def _initialize_tools(self):
        """初始化所有工具"""
        try:
            # Tool 1: 新闻API
            from .news_api import NewsAPITool
            self.register_tool(
                "news",
                NewsAPITool(),
                ToolMetadata(
                    name="NewsAPI",
                    description="获取金融新闻、情绪分析",
                    category="data",
                    version="1.0.0"
                )
            )
        except Exception as e:
            logger.warning(f"新闻API工具加载失败: {e}")
        
        try:
            # Tool 2: 实时行情 (简化版)
            from .market_data import MarketDataTool
            self.register_tool(
                "market_data",
                MarketDataTool(),
                ToolMetadata(
                    name="MarketData",
                    description="获取实时行情、盘口数据",
                    category="data",
                    version="1.0.0"
                )
            )
        except Exception as e:
            logger.warning(f"行情工具加载失败: {e}")
        
        try:
            # Tool 3-6: 其他工具 (简化实现)
            self._register_placeholder_tools()
        except Exception as e:
            logger.warning(f"占位工具加载失败: {e}")
    
    def _register_placeholder_tools(self):
        """注册占位工具 (简化实现)"""
        # Tool 3: 财报解析
        self.metadata["financial_report"] = ToolMetadata(
            name="FinancialReport",
            description="解析财报、提取关键指标",
            category="analysis",
            version="1.0.0",
            enabled=False  # 待实现
        )
        
        # Tool 4: 情绪分析
        self.metadata["sentiment"] = ToolMetadata(
            name="Sentiment",
            description="分析社交媒体情绪",
            category="analysis",
            version="1.0.0",
            enabled=False
        )
        
        # Tool 5: 经济日历
        self.metadata["economic_calendar"] = ToolMetadata(
            name="EconomicCalendar",
            description="获取经济事件日历",
            category="data",
            version="1.0.0",
            enabled=False
        )
        
        # Tool 6: 技术指标
        self.metadata["technical"] = ToolMetadata(
            name="TechnicalIndicator",
            description="计算技术指标",
            category="analysis",
            version="1.0.0",
            enabled=False
        )
    
    def register_tool(
        self,
        name: str,
        tool: Any,
        metadata: ToolMetadata
    ):
        """
        注册工具
        
        Args:
            name: 工具名称
            tool: 工具实例
            metadata: 工具元数据
        """
        self.tools[name] = tool
        self.metadata[name] = metadata
        logger.info(f"✅ 工具已注册: {name}")
    
    async def call_tool(
        self,
        tool_name: str,
        method_name: str,
        **kwargs
    ) -> ToolCallResult:
        """
        调用工具方法
        
        Args:
            tool_name: 工具名称
            method_name: 方法名称
            **kwargs: 方法参数
            
        Returns:
            ToolCallResult
        """
        start_time = datetime.now()
        
        try:
            # 检查工具是否存在
            if tool_name not in self.tools:
                raise ValueError(f"工具不存在: {tool_name}")
            
            # 检查工具是否启用
            if not self.metadata[tool_name].enabled:
                raise ValueError(f"工具未启用: {tool_name}")
            
            tool = self.tools[tool_name]
            
            # 检查方法是否存在
            if not hasattr(tool, method_name):
                raise ValueError(f"方法不存在: {tool_name}.{method_name}")
            
            method = getattr(tool, method_name)
            
            # 调用方法
            if asyncio.iscoroutinefunction(method):
                result = await method(**kwargs)
            else:
                result = method(**kwargs)
            
            # 计算耗时
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            # 记录成功调用
            call_result = ToolCallResult(
                tool_name=f"{tool_name}.{method_name}",
                success=True,
                result=result,
                duration_ms=duration
            )
            
            self._log_call(call_result)
            
            logger.info(
                f"✅ 工具调用成功: {tool_name}.{method_name} "
                f"(耗时: {duration:.2f}ms)"
            )
            
            return call_result
        
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            # 记录失败调用
            call_result = ToolCallResult(
                tool_name=f"{tool_name}.{method_name}",
                success=False,
                result=None,
                error=str(e),
                duration_ms=duration
            )
            
            self._log_call(call_result)
            
            logger.error(
                f"❌ 工具调用失败: {tool_name}.{method_name} "
                f"(错误: {e})"
            )
            
            return call_result
    
    def _log_call(self, result: ToolCallResult):
        """记录调用历史"""
        self.call_history.append(result)
        
        # 限制历史记录数量
        if len(self.call_history) > self.max_history:
            self.call_history = self.call_history[-self.max_history:]
    
    def get_available_tools(self) -> List[ToolMetadata]:
        """获取可用工具列表"""
        return [m for m in self.metadata.values() if m.enabled]
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        total_calls = len(self.call_history)
        successful_calls = sum(1 for c in self.call_history if c.success)
        
        # 按工具统计
        tool_stats = {}
        for call in self.call_history:
            tool_name = call.tool_name.split('.')[0]
            if tool_name not in tool_stats:
                tool_stats[tool_name] = {
                    'total': 0,
                    'success': 0,
                    'failed': 0,
                    'avg_duration_ms': 0.0
                }
            
            tool_stats[tool_name]['total'] += 1
            if call.success:
                tool_stats[tool_name]['success'] += 1
            else:
                tool_stats[tool_name]['failed'] += 1
        
        # 计算平均耗时
        for tool_name in tool_stats:
            tool_calls = [c for c in self.call_history if c.tool_name.startswith(tool_name)]
            if tool_calls:
                avg_duration = sum(c.duration_ms for c in tool_calls) / len(tool_calls)
                tool_stats[tool_name]['avg_duration_ms'] = round(avg_duration, 2)
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': total_calls - successful_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0.0,
            'tools': tool_stats
        }
    
    def get_recent_calls(self, limit: int = 10) -> List[ToolCallResult]:
        """获取最近的调用记录"""
        return self.call_history[-limit:]


# ============================================================================
# 简化工具实现 (占位符)
# ============================================================================

class MarketDataTool:
    """实时行情工具 (简化版)"""
    
    def __init__(self):
        logger.info("MarketDataTool初始化")
    
    async def get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """获取实时报价"""
        import random
        return {
            'symbol': symbol,
            'price': 100.0 + random.random() * 10,
            'change': random.random() * 2 - 1,
            'volume': random.randint(1000000, 10000000),
            'timestamp': datetime.now()
        }
    
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """获取盘口数据"""
        import random
        return {
            'symbol': symbol,
            'bids': [[100 - i * 0.1, random.randint(100, 1000)] for i in range(depth)],
            'asks': [[100 + i * 0.1, random.randint(100, 1000)] for i in range(depth)],
            'timestamp': datetime.now()
        }


# ============================================================================
# 使用示例
# ============================================================================

async def example_tool_manager():
    """工具管理器使用示例"""
    print("=== 工具管理器示例 ===\n")
    
    # 创建管理器
    manager = ToolManager()
    
    # 1. 列出可用工具
    print("1. 可用工具:")
    for tool in manager.get_available_tools():
        print(f"   • {tool.name}: {tool.description}")
    print()
    
    # 2. 调用新闻API
    print("2. 调用新闻API...")
    result = await manager.call_tool(
        "news",
        "get_company_news",
        symbol="600519.SH",
        days=3
    )
    print(f"   结果: {result.success}, 新闻数={len(result.result) if result.result else 0}")
    print()
    
    # 3. 调用行情API
    print("3. 调用行情API...")
    result = await manager.call_tool(
        "market_data",
        "get_realtime_quote",
        symbol="600519.SH"
    )
    print(f"   结果: {result.success}")
    if result.success:
        print(f"   价格: {result.result['price']:.2f}")
    print()
    
    # 4. 统计信息
    print("4. 工具统计:")
    stats = manager.get_tool_stats()
    print(f"   总调用次数: {stats['total_calls']}")
    print(f"   成功率: {stats['success_rate']:.1%}")
    print()
    
    print("✅ 示例完成!")


if __name__ == "__main__":
    asyncio.run(example_tool_manager())
