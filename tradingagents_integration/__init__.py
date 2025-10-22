"""
TradingAgents项目整合模块
将tradingagents项目的功能整合到麒麟堆栈中
"""

import sys
from pathlib import Path

# 添加tradingagents项目路径
TRADINGAGENTS_PATH = Path("D:/test/Qlib/tradingagents")
if str(TRADINGAGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(TRADINGAGENTS_PATH))

# 导入tradingagents核心组件
try:
    from tradingagents.agents import BaseAgent
    from tradingagents.llm.base import BaseLLM
    from tradingagents.tools.base import BaseTool
    from tradingagents.utils.logging_utils import get_logger
    TRADINGAGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import tradingagents: {e}")
    TRADINGAGENTS_AVAILABLE = False

# 导入我们的10个智能体
from .qilin_agents_enhanced import (
    QilinMultiAgentSystem,
    MarketEcologyAgent,
    AuctionGameAgent,
    PositionControlAgent,
    VolumeAnalysisAgent,
    TechnicalIndicatorAgent,
    SentimentAnalysisAgent,
    RiskManagementAgent,
    PatternRecognitionAgent,
    MacroeconomicAgent,
    ArbitrageAgent
)

# 导入整合适配器
from .integration_adapter import TradingAgentsAdapter

__all__ = [
    'QilinMultiAgentSystem',
    'TradingAgentsAdapter',
    'MarketEcologyAgent',
    'AuctionGameAgent',
    'PositionControlAgent',
    'VolumeAnalysisAgent',
    'TechnicalIndicatorAgent',
    'SentimentAnalysisAgent',
    'RiskManagementAgent',
    'PatternRecognitionAgent',
    'MacroeconomicAgent',
    'ArbitrageAgent',
    'TRADINGAGENTS_AVAILABLE'
]