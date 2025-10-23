"""
TradingAgents完整智能体集成
集成10个专业A股交易智能体，提供最全面的市场分析
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, field

# 导入配置
from .config import TradingAgentsConfig, load_config

logger = logging.getLogger(__name__)


# ============================================================================
# 检查并导入TradingAgents的10个智能体
# ============================================================================

def setup_tradingagents_path(config: TradingAgentsConfig):
    """设置TradingAgents路径"""
    ta_path = Path(config.tradingagents_path)
    
    if not ta_path.exists():
        raise FileNotFoundError(f"TradingAgents路径不存在: {ta_path}")
    
    if str(ta_path) not in sys.path:
        sys.path.insert(0, str(ta_path))
    
    logger.info(f"TradingAgents路径已添加: {ta_path}")


# 导入10个专业智能体
try:
    from tradingagents.agents.qilin_agents import (
        QilinMultiAgentCoordinator,
        MarketEcologyAgent,
        AuctionGameAgent,
        PositionControlAgent,
        VolumeAnalysisAgent,
        TechnicalIndicatorAgent,
        SentimentAnalysisAgent,
        RiskManagementAgent,
        PatternRecognitionAgent,
        MacroeconomicAgent,
        ArbitrageAgent,
        SignalType,
        TradingSignal
    )
    
    FULL_AGENTS_AVAILABLE = True
    logger.info("✅ 成功导入TradingAgents的10个专业智能体")
    
except ImportError as e:
    logger.error(f"❌ 无法导入完整智能体: {e}")
    FULL_AGENTS_AVAILABLE = False
    raise


# ============================================================================
# 智能体配置
# ============================================================================

@dataclass
class AgentWeightConfig:
    """智能体权重配置"""
    
    # 10个专业智能体的默认权重
    market_ecology: float = 0.12      # 市场生态
    auction_game: float = 0.08        # 竞价博弈
    volume: float = 0.10              # 成交量分析
    technical: float = 0.12           # 技术指标
    sentiment: float = 0.10           # 市场情绪
    pattern: float = 0.10             # K线形态
    macroeconomic: float = 0.08       # 宏观经济
    arbitrage: float = 0.05           # 套利机会
    position_control: float = 0.15    # 仓位控制（重要）
    risk: float = 0.10                # 风险管理
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "market_ecology": self.market_ecology,
            "auction_game": self.auction_game,
            "volume": self.volume,
            "technical": self.technical,
            "sentiment": self.sentiment,
            "pattern": self.pattern,
            "macroeconomic": self.macroeconomic,
            "arbitrage": self.arbitrage,
            "position_control": self.position_control,
            "risk": self.risk
        }
    
    def validate(self) -> bool:
        """验证权重和为1"""
        total = sum(self.to_dict().values())
        return abs(total - 1.0) < 0.01


@dataclass
class AnalysisResult:
    """完整分析结果"""
    symbol: str
    final_signal: Any  # TradingSignal
    
    # 10个智能体的个别分析
    market_ecology_signal: Optional[Any] = None
    auction_game_signal: Optional[Any] = None
    volume_signal: Optional[Any] = None
    technical_signal: Optional[Any] = None
    sentiment_signal: Optional[Any] = None
    pattern_signal: Optional[Any] = None
    macroeconomic_signal: Optional[Any] = None
    arbitrage_signal: Optional[Any] = None
    
    # 特殊分析
    position_advice: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    reasoning: str = ""


# ============================================================================
# 完整智能体集成
# ============================================================================

class FullAgentsIntegration:
    """
    TradingAgents完整智能体集成
    集成10个专业A股交易智能体
    """
    
    def __init__(self, config: Optional[TradingAgentsConfig] = None):
        """
        初始化完整集成
        
        Args:
            config: 配置对象
        """
        if not FULL_AGENTS_AVAILABLE:
            raise ImportError(
                "TradingAgents完整智能体不可用。\n"
                "请确保:\n"
                "1. TradingAgents项目已克隆到正确路径\n"
                "2. qilin_agents.py文件存在\n"
                "3. 路径配置正确"
            )
        
        self.config = config or load_config()
        
        # 设置路径
        setup_tradingagents_path(self.config)
        
        # 智能体权重配置
        self.weights = AgentWeightConfig()
        
        # 初始化10个智能体
        self._init_agents()
        
        # 初始化协调器
        self._init_coordinator()
        
        logger.info("✅ TradingAgents完整集成初始化成功（10个智能体）")
    
    def _init_agents(self):
        """初始化10个专业智能体"""
        logger.info("初始化10个专业智能体...")
        
        # 获取LLM（如果配置）
        llm = self._get_llm()
        
        self.agents = {
            # 1. 市场生态智能体
            "market_ecology": MarketEcologyAgent(llm),
            
            # 2. 竞价博弈智能体
            "auction_game": AuctionGameAgent(llm),
            
            # 3. 仓位控制智能体
            "position_control": PositionControlAgent(llm),
            
            # 4. 成交量分析智能体
            "volume": VolumeAnalysisAgent(llm),
            
            # 5. 技术指标智能体
            "technical": TechnicalIndicatorAgent(llm),
            
            # 6. 市场情绪智能体
            "sentiment": SentimentAnalysisAgent(llm),
            
            # 7. 风险管理智能体
            "risk": RiskManagementAgent(llm),
            
            # 8. K线形态识别智能体
            "pattern": PatternRecognitionAgent(llm),
            
            # 9. 宏观经济智能体
            "macroeconomic": MacroeconomicAgent(llm),
            
            # 10. 套利机会智能体
            "arbitrage": ArbitrageAgent(llm)
        }
        
        logger.info(f"✅ 成功初始化{len(self.agents)}个智能体")
    
    def _get_llm(self):
        """获取LLM（如果配置）"""
        # 简化实现，返回None
        # 实际应该根据config创建LLM实例
        return None
    
    def _init_coordinator(self):
        """初始化多智能体协调器"""
        self.coordinator = QilinMultiAgentCoordinator(
            agents=self.agents,
            agent_weights=self.weights.to_dict()
        )
        logger.info("✅ 协调器初始化完成")
    
    async def analyze_comprehensive(self, 
                                   symbol: str,
                                   market_data: Dict[str, Any]) -> AnalysisResult:
        """
        全面分析股票（使用所有10个智能体）
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            
        Returns:
            完整分析结果
        """
        logger.info(f"🔬 开始全面分析: {symbol}")
        
        # 使用协调器运行所有智能体
        coordinator_result = await self.coordinator.analyze(market_data)
        
        # 提取个别智能体的信号
        individual_signals = coordinator_result.get("individual_signals", {})
        
        # 构建完整结果
        result = AnalysisResult(
            symbol=symbol,
            final_signal=coordinator_result.get("final_signal"),
            market_ecology_signal=individual_signals.get("market_ecology"),
            auction_game_signal=individual_signals.get("auction_game"),
            volume_signal=individual_signals.get("volume"),
            technical_signal=individual_signals.get("technical"),
            sentiment_signal=individual_signals.get("sentiment"),
            pattern_signal=individual_signals.get("pattern"),
            macroeconomic_signal=individual_signals.get("macroeconomic"),
            arbitrage_signal=individual_signals.get("arbitrage"),
            position_advice=coordinator_result.get("position_advice"),
            risk_assessment=coordinator_result.get("risk_assessment"),
            confidence=coordinator_result.get("final_signal").confidence if coordinator_result.get("final_signal") else 0,
            reasoning=coordinator_result.get("final_signal").reason if coordinator_result.get("final_signal") else ""
        )
        
        logger.info(f"✅ 分析完成: {result.final_signal.signal_type.value}")
        return result
    
    async def batch_analyze(self, 
                          symbols: List[str],
                          market_data_provider) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        Args:
            symbols: 股票代码列表
            market_data_provider: 市场数据提供者函数
            
        Returns:
            分析结果列表
        """
        logger.info(f"📊 开始批量分析{len(symbols)}只股票...")
        
        results = []
        for symbol in symbols:
            try:
                market_data = await market_data_provider(symbol)
                result = await self.analyze_comprehensive(symbol, market_data)
                results.append(result)
            except Exception as e:
                logger.error(f"分析{symbol}失败: {e}")
        
        return results
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新智能体权重
        
        Args:
            new_weights: 新的权重配置
        """
        # 验证权重和
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"权重和必须为1.0，当前为{total}")
        
        # 更新权重
        for agent_name, weight in new_weights.items():
            if hasattr(self.weights, agent_name):
                setattr(self.weights, agent_name, weight)
        
        # 重新初始化协调器
        self.coordinator.agent_weights = new_weights
        logger.info("✅ 智能体权重已更新")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "full_agents_available": FULL_AGENTS_AVAILABLE,
            "mode": "full_10_agents",
            "agents_count": len(self.agents),
            "agents_list": list(self.agents.keys()),
            "weights": self.weights.to_dict(),
            "weights_valid": self.weights.validate(),
            "config": self.config.to_dict()
        }


# ============================================================================
# 智能体分组管理
# ============================================================================

class AgentGroupManager:
    """智能体分组管理器"""
    
    def __init__(self, integration: FullAgentsIntegration):
        self.integration = integration
        
        # 定义智能体分组
        self.groups = {
            "core": ["market_ecology", "technical", "risk"],
            "timing": ["auction_game", "volume", "pattern"],
            "sentiment": ["sentiment", "macroeconomic"],
            "advanced": ["arbitrage", "position_control"]
        }
    
    async def analyze_by_group(self, 
                              group_name: str,
                              symbol: str,
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        按分组分析
        
        Args:
            group_name: 分组名称
            symbol: 股票代码
            market_data: 市场数据
            
        Returns:
            分组分析结果
        """
        if group_name not in self.groups:
            raise ValueError(f"未知分组: {group_name}")
        
        agent_names = self.groups[group_name]
        results = {}
        
        for agent_name in agent_names:
            agent = self.integration.agents.get(agent_name)
            if agent:
                try:
                    signal = await agent.analyze(market_data)
                    results[agent_name] = signal
                except Exception as e:
                    logger.error(f"智能体{agent_name}分析失败: {e}")
        
        return {
            "group": group_name,
            "symbol": symbol,
            "signals": results,
            "count": len(results)
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_full_integration(config_file: Optional[str] = None) -> FullAgentsIntegration:
    """
    创建完整的TradingAgents集成实例
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        完整集成实例（10个智能体）
        
    Raises:
        ImportError: 如果TradingAgents不可用
    """
    config = load_config(config_file)
    return FullAgentsIntegration(config)


# ============================================================================
# 测试
# ============================================================================

async def test_full_integration():
    """测试完整集成"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("TradingAgents完整集成测试（10个智能体）")
    
    try:
        # 创建集成
        integration = create_full_integration()
        
        # 检查状态
        status = integration.get_status()
        logger.info("系统状态:")
        logger.info(f"  模式: {status['mode']}")
        logger.info(f"  智能体数量: {status['agents_count']}")
        logger.info(f"  智能体列表: {status['agents_list']}")
        logger.info(f"  权重验证: {status['weights_valid']}")
        
        # 模拟市场数据
        market_data = {
            "symbol": "000001.SZ",
            "price": 15.5,
            "prev_close": 15.0,
            "volume": 1000000,
            "avg_volume": 800000,
            "advances": 2500,
            "declines": 1500,
            "money_inflow": 1000000000,
            "money_outflow": 800000000,
            "technical_indicators": {
                "rsi": 65,
                "macd": 0.5,
                "kdj_k": 75
            },
            "returns": [-0.01, 0.02, -0.005, 0.03, 0.01],
            "prices": [15.0, 15.1, 15.2, 15.3, 15.5]
        }
        
        # 测试全面分析
        logger.info("测试全面分析...")
        result = await integration.analyze_comprehensive("000001.SZ", market_data)
        
        logger.info("分析完成:")
        logger.info(f"  最终信号: {result.final_signal.signal_type.value}")
        logger.info(f"  置信度: {result.confidence:.2%}")
        logger.info(f"  理由: {result.reasoning}")
        
        if result.position_advice:
            logger.info("仓位建议:")
            logger.info(f"  推荐仓位: {result.position_advice.get('recommended_position', 0):.2%}")
        
        if result.risk_assessment:
            logger.info("风险评估:")
            logger.info(f"  风险等级: {result.risk_assessment.get('risk_level', 'N/A')}")
        
        # 测试分组分析
        logger.info("测试分组分析...")
        group_manager = AgentGroupManager(integration)
        core_result = await group_manager.analyze_by_group("core", "000001.SZ", market_data)
        logger.info(f"  核心分组分析: {core_result['count']}个智能体")
        
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.info("请确保TradingAgents已正确安装和配置")
    except Exception as e:
        import traceback
        logger.exception(f"测试失败: {e}")


if __name__ == "__main__":
    from app.core.logging_setup import setup_logging
    setup_logging()
    asyncio.run(test_full_integration())
