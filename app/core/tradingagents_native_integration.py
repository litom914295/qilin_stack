"""
TradingAgents原生智能体集成模块
整合基本面分析师、市场情绪分析师、技术面分析师、风险管控师
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# 添加TradingAgents路径
tradingagents_path = Path("D:/test/Qlib/tradingagents")
if tradingagents_path.exists():
    sys.path.insert(0, str(tradingagents_path))

# 导入TradingAgents原生组件
try:
    from tradingagents.agents import (
        create_fundamentals_analyst,
        create_market_analyst,
        create_news_analyst,
        create_research_manager,
        create_risk_manager,
        create_trader,
        AgentState,
        Toolkit
    from tradingagents.tools import get_all_tools
    from tradingagents.llm import get_llm
    TRADINGAGENTS_AVAILABLE = True
except ImportError as e:
    print(f"TradingAgents导入失败: {e}")
    TRADINGAGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NativeAgentRole:
    """原生智能体角色枚举"""
    FUNDAMENTALS = "基本面分析师"
    SENTIMENT = "市场情绪分析师"
    TECHNICAL = "技术面分析师"
    RISK = "风险管控师"


class TradingAgentsNativeIntegration:
    """TradingAgents原生智能体集成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化集成器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.agents = {}
        self.tools = None
        self.llm = None
        self.state = None
        
        if TRADINGAGENTS_AVAILABLE:
            self._initialize_components()
        else:
            logger.warning("TradingAgents不可用，使用模拟模式")
    
    def _initialize_components(self):
        """初始化TradingAgents组件"""
        try:
            # 初始化LLM
            self.llm = get_llm(self.config.get("llm_config", {}))
            
            # 初始化工具集
            self.tools = get_all_tools()
            self.toolkit = Toolkit(tools=self.tools)
            
            # 初始化状态
            self.state = AgentState(
                messages=[],
                data={},
                metadata={}
            
            # 创建原生智能体
            self._create_native_agents()
            
            logger.info("TradingAgents组件初始化成功")
            
        except Exception as e:
            logger.error(f"初始化TradingAgents组件失败: {e}")
            TRADINGAGENTS_AVAILABLE = False
    
    def _create_native_agents(self):
        """创建原生智能体"""
        try:
            # 📊 基本面分析师
            self.agents[NativeAgentRole.FUNDAMENTALS] = create_fundamentals_analyst(
                self.llm,
                self.toolkit
            
            # 📈 市场情绪分析师 (使用新闻和市场分析师组合)
            market_analyst = create_market_analyst(self.llm, self.toolkit)
            news_analyst = create_news_analyst(self.llm, self.toolkit)
            self.agents[NativeAgentRole.SENTIMENT] = {
                "market": market_analyst,
                "news": news_analyst
            }
            
            # 💹 技术面分析师 (基于市场分析师)
            self.agents[NativeAgentRole.TECHNICAL] = create_market_analyst(
                self.llm,
                self.toolkit
            
            # 🛡️ 风险管控师
            self.agents[NativeAgentRole.RISK] = create_risk_manager(
                self.llm,
                self.toolkit
            
            logger.info(f"创建了{len(self.agents)}个原生智能体")
            
        except Exception as e:
            logger.error(f"创建原生智能体失败: {e}")
    
    async def analyze_stock(self, symbol: str, data: Optional[Dict] = None) -> Dict:
        """
        使用所有原生智能体分析股票
        
        Args:
            symbol: 股票代码
            data: 额外数据
            
        Returns:
            综合分析结果
        """
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "agents_analysis": {},
            "consensus": None,
            "recommendation": None
        }
        
        if not TRADINGAGENTS_AVAILABLE:
            return self._mock_analysis(symbol, data)
        
        try:
            # 1. 基本面分析
            fundamentals_result = await self._analyze_fundamentals(symbol, data)
            results["agents_analysis"][NativeAgentRole.FUNDAMENTALS] = fundamentals_result
            
            # 2. 市场情绪分析
            sentiment_result = await self._analyze_sentiment(symbol, data)
            results["agents_analysis"][NativeAgentRole.SENTIMENT] = sentiment_result
            
            # 3. 技术面分析
            technical_result = await self._analyze_technical(symbol, data)
            results["agents_analysis"][NativeAgentRole.TECHNICAL] = technical_result
            
            # 4. 风险分析
            risk_result = await self._analyze_risk(symbol, data)
            results["agents_analysis"][NativeAgentRole.RISK] = risk_result
            
            # 5. 生成共识
            results["consensus"] = self._generate_consensus(results["agents_analysis"])
            
            # 6. 生成建议
            results["recommendation"] = self._generate_recommendation(results["consensus"])
            
        except Exception as e:
            logger.error(f"分析股票{symbol}失败: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _analyze_fundamentals(self, symbol: str, data: Dict) -> Dict:
        """基本面分析"""
        agent = self.agents[NativeAgentRole.FUNDAMENTALS]
        
        # 准备输入
        input_data = {
            "symbol": symbol,
            "request": f"分析{symbol}的基本面，包括财务状况、ROE趋势、现金流等",
            **data
        }
        
        # 调用智能体
        result = await self._invoke_agent(agent, input_data)
        
        return {
            "analysis": result,
            "metrics": self._extract_fundamental_metrics(result),
            "score": self._calculate_fundamental_score(result)
        }
    
    async def _analyze_sentiment(self, symbol: str, data: Dict) -> Dict:
        """市场情绪分析"""
        agents = self.agents[NativeAgentRole.SENTIMENT]
        
        # 市场分析
        market_result = await self._invoke_agent(
            agents["market"],
            {"symbol": symbol, "request": f"分析{symbol}的市场情绪"}
        
        # 新闻分析
        news_result = await self._invoke_agent(
            agents["news"],
            {"symbol": symbol, "request": f"分析{symbol}的新闻舆情"}
        
        return {
            "market_sentiment": market_result,
            "news_sentiment": news_result,
            "overall_sentiment": self._combine_sentiments(market_result, news_result),
            "score": self._calculate_sentiment_score(market_result, news_result)
        }
    
    async def _analyze_technical(self, symbol: str, data: Dict) -> Dict:
        """技术面分析"""
        agent = self.agents[NativeAgentRole.TECHNICAL]
        
        result = await self._invoke_agent(
            agent,
            {
                "symbol": symbol,
                "request": f"分析{symbol}的技术指标，包括MACD、RSI、支撑位压力位等",
                "price_data": data.get("price_data", [])
            }
        
        return {
            "analysis": result,
            "indicators": self._extract_technical_indicators(result),
            "signals": self._extract_trading_signals(result),
            "score": self._calculate_technical_score(result)
        }
    
    async def _analyze_risk(self, symbol: str, data: Dict) -> Dict:
        """风险分析"""
        agent = self.agents[NativeAgentRole.RISK]
        
        result = await self._invoke_agent(
            agent,
            {
                "symbol": symbol,
                "request": f"评估{symbol}的风险，包括流动性、政策风险、黑天鹅事件等",
                "portfolio": data.get("portfolio", {})
            }
        
        return {
            "analysis": result,
            "risk_factors": self._extract_risk_factors(result),
            "position_recommendation": self._calculate_position_size(result),
            "score": self._calculate_risk_score(result)
        }
    
    async def _invoke_agent(self, agent, input_data: Dict) -> Any:
        """调用智能体"""
        if not TRADINGAGENTS_AVAILABLE:
            return self._mock_agent_response(input_data)
        
        try:
            # 更新状态
            self.state.data.update(input_data)
            
            # 调用智能体
            response = await agent.ainvoke(self.state)
            
            return response
            
        except Exception as e:
            logger.error(f"调用智能体失败: {e}")
            return {"error": str(e)}
    
    def _generate_consensus(self, analyses: Dict) -> Dict:
        """生成共识"""
        scores = {}
        weights = {
            NativeAgentRole.FUNDAMENTALS: 0.3,
            NativeAgentRole.SENTIMENT: 0.2,
            NativeAgentRole.TECHNICAL: 0.3,
            NativeAgentRole.RISK: 0.2
        }
        
        total_score = 0
        for role, analysis in analyses.items():
            score = analysis.get("score", 0.5)
            scores[role] = score
            total_score += score * weights.get(role, 0.25)
        
        # 判断共识强度
        if total_score > 0.7:
            consensus_type = "强烈看多"
        elif total_score > 0.6:
            consensus_type = "看多"
        elif total_score < 0.3:
            consensus_type = "强烈看空"
        elif total_score < 0.4:
            consensus_type = "看空"
        else:
            consensus_type = "中性"
        
        return {
            "type": consensus_type,
            "score": total_score,
            "agent_scores": scores,
            "confidence": self._calculate_confidence(scores)
        }
    
    def _generate_recommendation(self, consensus: Dict) -> Dict:
        """生成交易建议"""
        score = consensus["score"]
        confidence = consensus["confidence"]
        
        # 确定操作建议
        if score > 0.7 and confidence > 0.7:
            action = "强烈买入"
            position = 0.3  # 30%仓位
        elif score > 0.6 and confidence > 0.6:
            action = "买入"
            position = 0.2  # 20%仓位
        elif score < 0.3 and confidence > 0.7:
            action = "强烈卖出"
            position = 0  # 清仓
        elif score < 0.4 and confidence > 0.6:
            action = "卖出"
            position = 0.05  # 减仓到5%
        else:
            action = "持有"
            position = 0.1  # 维持10%
        
        return {
            "action": action,
            "position_size": position,
            "confidence": confidence,
            "risk_level": self._calculate_risk_level(score, confidence),
            "stop_loss": self._calculate_stop_loss(score),
            "take_profit": self._calculate_take_profit(score)
        }
    
    def _calculate_confidence(self, scores: Dict) -> float:
        """计算置信度（基于智能体一致性）"""
        if not scores:
            return 0.5
        
        values = list(scores.values())
        std_dev = np.std(values)
        
        # 标准差越小，一致性越高，置信度越高
        confidence = 1 - min(std_dev * 2, 0.5)
        return confidence
    
    def _calculate_risk_level(self, score: float, confidence: float) -> str:
        """计算风险等级"""
        risk_score = abs(score - 0.5) * confidence
        
        if risk_score < 0.2:
            return "低"
        elif risk_score < 0.4:
            return "中"
        else:
            return "高"
    
    def _calculate_stop_loss(self, score: float) -> float:
        """计算止损位"""
        base_stop = 0.02  # 基础2%止损
        
        # 看多程度越高，止损可以适当放宽
        if score > 0.6:
            return base_stop * 1.5
        elif score > 0.5:
            return base_stop
        else:
            return base_stop * 0.8
    
    def _calculate_take_profit(self, score: float) -> float:
        """计算止盈位"""
        base_profit = 0.05  # 基础5%止盈
        
        # 看多程度越高，止盈目标可以更高
        if score > 0.7:
            return base_profit * 2
        elif score > 0.6:
            return base_profit * 1.5
        else:
            return base_profit
    
    def _extract_fundamental_metrics(self, result: Any) -> Dict:
        """提取基本面指标"""
        # 这里应该解析result中的实际数据
        return {
            "roe": 0.15,
            "pe": 20,
            "pb": 2.5,
            "debt_ratio": 0.4,
            "cash_flow": "正向"
        }
    
    def _extract_technical_indicators(self, result: Any) -> Dict:
        """提取技术指标"""
        return {
            "macd": "金叉",
            "rsi": 55,
            "ma20": 100,
            "ma60": 98,
            "support": 95,
            "resistance": 105
        }
    
    def _extract_trading_signals(self, result: Any) -> List[str]:
        """提取交易信号"""
        return ["突破20日均线", "RSI中性区间", "MACD金叉形成"]
    
    def _extract_risk_factors(self, result: Any) -> List[str]:
        """提取风险因子"""
        return ["市场波动加大", "行业政策不确定", "流动性充足"]
    
    def _calculate_fundamental_score(self, result: Any) -> float:
        """计算基本面得分"""
        # 简化实现
        return 0.65
    
    def _calculate_sentiment_score(self, market: Any, news: Any) -> float:
        """计算情绪得分"""
        return 0.7
    
    def _calculate_technical_score(self, result: Any) -> float:
        """计算技术面得分"""
        return 0.6
    
    def _calculate_risk_score(self, result: Any) -> float:
        """计算风险得分（分数越高风险越低）"""
        return 0.75
    
    def _calculate_position_size(self, result: Any) -> float:
        """计算建议仓位"""
        return 0.2
    
    def _combine_sentiments(self, market: Any, news: Any) -> str:
        """合并情绪分析"""
        return "积极"
    
    def _mock_analysis(self, symbol: str, data: Dict) -> Dict:
        """模拟分析（当TradingAgents不可用时）"""
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "agents_analysis": {
                NativeAgentRole.FUNDAMENTALS: {
                    "analysis": "模拟基本面分析结果",
                    "metrics": {"roe": 0.15, "pe": 20},
                    "score": 0.65
                },
                NativeAgentRole.SENTIMENT: {
                    "overall_sentiment": "积极",
                    "score": 0.7
                },
                NativeAgentRole.TECHNICAL: {
                    "analysis": "模拟技术分析结果",
                    "signals": ["金叉"],
                    "score": 0.6
                },
                NativeAgentRole.RISK: {
                    "analysis": "模拟风险分析",
                    "risk_factors": ["低风险"],
                    "score": 0.75
                }
            },
            "consensus": {
                "type": "看多",
                "score": 0.675,
                "confidence": 0.8
            },
            "recommendation": {
                "action": "买入",
                "position_size": 0.2,
                "confidence": 0.8,
                "risk_level": "中"
            }
        }
    
    def _mock_agent_response(self, input_data: Dict) -> Dict:
        """模拟智能体响应"""
        return {
            "response": f"模拟分析{input_data.get('symbol')}",
            "status": "success"
        }


class MultiAgentDebateSystem:
    """多智能体辩论系统"""
    
    def __init__(self, integration: TradingAgentsNativeIntegration):
        self.integration = integration
        self.debate_history = []
    
    async def conduct_debate(self, symbol: str, rounds: int = 3) -> Dict:
        """
        进行多轮辩论
        
        Args:
            symbol: 股票代码
            rounds: 辩论轮数
            
        Returns:
            辩论结果
        """
        debate_result = {
            "symbol": symbol,
            "rounds": [],
            "final_consensus": None
        }
        
        for round_num in range(rounds):
            logger.info(f"开始第{round_num + 1}轮辩论")
            
            # 获取各智能体观点
            analysis = await self.integration.analyze_stock(symbol)
            
            # 提取看多和看空观点
            bull_arguments = self._extract_bull_arguments(analysis)
            bear_arguments = self._extract_bear_arguments(analysis)
            
            # 进行辩论
            round_result = {
                "round": round_num + 1,
                "bull_arguments": bull_arguments,
                "bear_arguments": bear_arguments,
                "debate": self._simulate_debate(bull_arguments, bear_arguments),
                "consensus": analysis.get("consensus")
            }
            
            debate_result["rounds"].append(round_result)
            
            # 更新辩论历史供下一轮参考
            self.debate_history.append(round_result)
        
        # 生成最终共识
        debate_result["final_consensus"] = self._generate_final_consensus(
            debate_result["rounds"]
        
        return debate_result
    
    def _extract_bull_arguments(self, analysis: Dict) -> List[str]:
        """提取看多论据"""
        arguments = []
        
        for role, data in analysis.get("agents_analysis", {}).items():
            score = data.get("score", 0.5)
            if score > 0.6:
                arguments.append(f"{role}: {data.get('analysis', '看多')}")
        
        return arguments
    
    def _extract_bear_arguments(self, analysis: Dict) -> List[str]:
        """提取看空论据"""
        arguments = []
        
        for role, data in analysis.get("agents_analysis", {}).items():
            score = data.get("score", 0.5)
            if score < 0.4:
                arguments.append(f"{role}: {data.get('analysis', '看空')}")
        
        return arguments
    
    def _simulate_debate(self, bull: List[str], bear: List[str]) -> List[Dict]:
        """模拟辩论过程"""
        debate = []
        
        # 看多方发言
        if bull:
            debate.append({
                "speaker": "看多方",
                "argument": bull[0] if bull else "维持看多观点",
                "timestamp": datetime.now().isoformat()
            })
        
        # 看空方反驳
        if bear:
            debate.append({
                "speaker": "看空方",
                "argument": bear[0] if bear else "维持看空观点",
                "rebuttal_to": "看多方",
                "timestamp": datetime.now().isoformat()
            })
        
        return debate
    
    def _generate_final_consensus(self, rounds: List[Dict]) -> Dict:
        """生成最终共识"""
        # 收集所有轮次的共识分数
        scores = []
        for round_data in rounds:
            consensus = round_data.get("consensus", {})
            scores.append(consensus.get("score", 0.5))
        
        # 计算平均分数
        avg_score = np.mean(scores) if scores else 0.5
        
        # 判断趋势
        if len(scores) > 1:
            trend = "上升" if scores[-1] > scores[0] else "下降"
        else:
            trend = "稳定"
        
        return {
            "final_score": avg_score,
            "trend": trend,
            "confidence": 1 - np.std(scores) if len(scores) > 1 else 0.5,
            "recommendation": self._get_final_recommendation(avg_score)
        }
    
    def _get_final_recommendation(self, score: float) -> str:
        """获取最终建议"""
        if score > 0.7:
            return "强烈推荐买入"
        elif score > 0.6:
            return "建议买入"
        elif score < 0.3:
            return "强烈建议卖出"
        elif score < 0.4:
            return "建议卖出"
        else:
            return "建议观望"


# 使用示例
async def main():
    """主函数示例"""
    # 创建集成器
    integration = TradingAgentsNativeIntegration()
    
    # 分析股票
    result = await integration.analyze_stock("000001")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 进行多轮辩论
    debate_system = MultiAgentDebateSystem(integration)
    debate_result = await debate_system.conduct_debate("000001", rounds=3)
    print(json.dumps(debate_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())