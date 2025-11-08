"""
TradingAgents项目整合模块
将tradingagents项目的功能整合到麒麟堆栈中
"""

import sys
from pathlib import Path
from .config import load_config

# 添加tradingagents项目路径（优先读取配置/环境变量）
_config = load_config()
TRADINGAGENTS_PATH = Path(_config.tradingagents_path)
if TRADINGAGENTS_PATH.exists() and str(TRADINGAGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(TRADINGAGENTS_PATH))

# 导入tradingagents核心组件 (TradingAgents-CN-Plus uses LangGraph architecture)
try:
    from tradingagents.agents import (
        create_trader,
        create_research_manager,
        create_risk_manager,
        AgentState,
        Toolkit
    )
    from tradingagents.utils.logging_init import get_logger
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