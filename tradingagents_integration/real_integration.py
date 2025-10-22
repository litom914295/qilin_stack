"""
TradingAgents真实集成实现
引入TradingAgents的核心组件，实现完整的多智能体系统
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field

# 导入配置
from .config import TradingAgentsConfig, load_config

logger = logging.getLogger(__name__)


# ============================================================================
# 核心组件检查和导入
# ============================================================================

def check_tradingagents_available(config: TradingAgentsConfig) -> tuple[bool, str]:
    """
    检查TradingAgents是否可用
    
    Returns:
        (is_available, message)
    """
    ta_path = Path(config.tradingagents_path)
    
    if not ta_path.exists():
        return False, f"TradingAgents路径不存在: {ta_path}"
    
    # 添加到系统路径
    if str(ta_path) not in sys.path:
        sys.path.insert(0, str(ta_path))
    
    # 尝试导入核心模块
    try:
        # 这些是TradingAgents项目的核心模块
        import tradingagents
        return True, "TradingAgents已成功加载"
    except ImportError as e:
        return False, f"无法导入TradingAgents: {e}"


class TradingAgentsNotAvailableError(Exception):
    """TradingAgents不可用异常"""
    pass


# ============================================================================
# LLM适配器
# ============================================================================

class LLMAdapter:
    """
    统一的LLM适配器
    支持多个LLM提供商
    """
    
    def __init__(self, config: TradingAgentsConfig):
        self.config = config
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化LLM客户端"""
        try:
            if self.config.llm_provider == "openai":
                import openai
                self.client = openai.OpenAI(
                    api_key=self.config.llm_api_key,
                    base_url=self.config.llm_api_base
                )
                logger.info(f"OpenAI客户端已初始化: {self.config.llm_model}")
            
            elif self.config.llm_provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=self.config.llm_api_key
                )
                logger.info(f"Anthropic客户端已初始化")
            
            else:
                logger.warning(f"不支持的LLM提供商: {self.config.llm_provider}")
        
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            self.client = None
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      **kwargs) -> str:
        """
        生成响应
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        if not self.client:
            return "LLM未配置，无法生成响应"
        
        try:
            if self.config.llm_provider == "openai":
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=kwargs.get('temperature', self.config.llm_temperature),
                    max_tokens=kwargs.get('max_tokens', self.config.llm_max_tokens)
                )
                return response.choices[0].message.content
            
            elif self.config.llm_provider == "anthropic":
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.config.llm_model,
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', self.config.llm_max_tokens)
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"生成失败: {e}"


# ============================================================================
# 基础智能体
# ============================================================================

@dataclass
class AgentResponse:
    """智能体响应"""
    agent_name: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    reasoning: str
    analysis: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class BaseAgent:
    """基础智能体类"""
    
    def __init__(self, name: str, llm: LLMAdapter, config: TradingAgentsConfig):
        self.name = name
        self.llm = llm
        self.config = config
        self.weight = config.agent_weights.get(name, 0.2)
    
    async def analyze(self, 
                     symbol: str, 
                     market_data: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        分析股票
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            context: 上下文信息
            
        Returns:
            智能体响应
        """
        raise NotImplementedError("子类必须实现analyze方法")
    
    def _format_market_data(self, market_data: Dict[str, Any]) -> str:
        """格式化市场数据为文本"""
        lines = []
        
        if 'price' in market_data:
            lines.append(f"当前价格: {market_data['price']}")
        
        if 'change_pct' in market_data:
            lines.append(f"涨跌幅: {market_data['change_pct']:.2%}")
        
        if 'volume' in market_data:
            lines.append(f"成交量: {market_data['volume']:,.0f}")
        
        if 'technical_indicators' in market_data:
            ti = market_data['technical_indicators']
            lines.append(f"技术指标: RSI={ti.get('rsi', 'N/A')}, MACD={ti.get('macd', 'N/A')}")
        
        return "\n".join(lines)


class MarketAnalystAgent(BaseAgent):
    """市场分析师智能体"""
    
    def __init__(self, llm: LLMAdapter, config: TradingAgentsConfig):
        super().__init__("market_analyst", llm, config)
    
    async def analyze(self, 
                     symbol: str, 
                     market_data: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """市场整体分析"""
        
        # 构建提示词
        market_info = self._format_market_data(market_data)
        
        prompt = f"""
作为市场分析师，请分析股票 {symbol} 的市场状况：

市场数据：
{market_info}

请提供：
1. 市场趋势分析
2. 买卖建议 (BUY/SELL/HOLD)
3. 置信度 (0-1)
4. 详细理由

请以JSON格式回复：
{{
    "signal": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "详细理由",
    "market_trend": "上涨/下跌/震荡",
    "key_factors": ["因素1", "因素2"]
}}
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = await self.llm.generate(messages)
            
            # 解析响应（简化处理）
            import json
            import re
            
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                # 如果没有JSON，使用默认值
                response_data = {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "reasoning": response_text[:200],
                    "market_trend": "uncertain"
                }
            
            return AgentResponse(
                agent_name=self.name,
                signal=response_data.get("signal", "HOLD"),
                confidence=float(response_data.get("confidence", 0.5)),
                reasoning=response_data.get("reasoning", "无详细理由"),
                analysis={
                    "market_trend": response_data.get("market_trend", "uncertain"),
                    "key_factors": response_data.get("key_factors", [])
                }
            )
        
        except Exception as e:
            logger.error(f"市场分析师分析失败: {e}")
            return AgentResponse(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning=f"分析失败: {e}",
                analysis={}
            )


class FundamentalAnalystAgent(BaseAgent):
    """基本面分析师智能体"""
    
    def __init__(self, llm: LLMAdapter, config: TradingAgentsConfig):
        super().__init__("fundamental_analyst", llm, config)
    
    async def analyze(self, 
                     symbol: str, 
                     market_data: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """基本面分析"""
        
        fundamental_data = market_data.get('fundamental_data', {})
        
        prompt = f"""
作为基本面分析师，请分析股票 {symbol} 的基本面：

财务数据：
- PE比率: {fundamental_data.get('pe_ratio', 'N/A')}
- PB比率: {fundamental_data.get('pb_ratio', 'N/A')}
- ROE: {fundamental_data.get('roe', 'N/A')}
- 营收增长: {fundamental_data.get('revenue_growth', 'N/A')}
- 净利润增长: {fundamental_data.get('profit_growth', 'N/A')}

请提供投资建议和置信度。

JSON格式：
{{
    "signal": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "详细理由",
    "valuation": "低估/合理/高估"
}}
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = await self.llm.generate(messages)
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "reasoning": response_text[:200]
                }
            
            return AgentResponse(
                agent_name=self.name,
                signal=response_data.get("signal", "HOLD"),
                confidence=float(response_data.get("confidence", 0.5)),
                reasoning=response_data.get("reasoning", "无详细理由"),
                analysis={
                    "valuation": response_data.get("valuation", "unknown")
                }
            )
        
        except Exception as e:
            logger.error(f"基本面分析师分析失败: {e}")
            return AgentResponse(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning=f"分析失败: {e}",
                analysis={}
            )


class TechnicalAnalystAgent(BaseAgent):
    """技术分析师智能体"""
    
    def __init__(self, llm: LLMAdapter, config: TradingAgentsConfig):
        super().__init__("technical_analyst", llm, config)
    
    async def analyze(self, 
                     symbol: str, 
                     market_data: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """技术分析"""
        
        ti = market_data.get('technical_indicators', {})
        
        # 基于技术指标的简单规则
        signal = "HOLD"
        confidence = 0.6
        reasoning = []
        
        # RSI分析
        rsi = ti.get('rsi', 50)
        if rsi < 30:
            signal = "BUY"
            confidence = 0.7
            reasoning.append(f"RSI({rsi:.1f})超卖，建议买入")
        elif rsi > 70:
            signal = "SELL"
            confidence = 0.7
            reasoning.append(f"RSI({rsi:.1f})超买，建议卖出")
        else:
            reasoning.append(f"RSI({rsi:.1f})中性")
        
        # MACD分析
        macd = ti.get('macd', 0)
        macd_signal = ti.get('macd_signal', 0)
        if macd > macd_signal:
            reasoning.append("MACD金叉，看涨信号")
            if signal == "HOLD":
                signal = "BUY"
                confidence = 0.6
        elif macd < macd_signal:
            reasoning.append("MACD死叉，看跌信号")
            if signal == "HOLD":
                signal = "SELL"
                confidence = 0.6
        
        return AgentResponse(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning="; ".join(reasoning),
            analysis={
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal
            }
        )


class SentimentAnalystAgent(BaseAgent):
    """情绪分析师智能体"""
    
    def __init__(self, llm: LLMAdapter, config: TradingAgentsConfig):
        super().__init__("sentiment_analyst", llm, config)
    
    async def analyze(self, 
                     symbol: str, 
                     market_data: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """市场情绪分析"""
        
        sentiment_data = market_data.get('sentiment', {})
        sentiment_score = sentiment_data.get('score', 0.5)
        
        # 基于情绪分数的简单规则
        if sentiment_score > 0.7:
            signal = "BUY"
            confidence = 0.65
            reasoning = f"市场情绪积极(得分: {sentiment_score:.2f})"
        elif sentiment_score < 0.3:
            signal = "SELL"
            confidence = 0.65
            reasoning = f"市场情绪消极(得分: {sentiment_score:.2f})"
        else:
            signal = "HOLD"
            confidence = 0.5
            reasoning = f"市场情绪中性(得分: {sentiment_score:.2f})"
        
        return AgentResponse(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            analysis={
                "sentiment_score": sentiment_score,
                "sentiment_category": "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"
            }
        )


# ============================================================================
# 智能体协调器
# ============================================================================

class AgentOrchestrator:
    """智能体协调器 - 协调多个智能体的分析"""
    
    def __init__(self, 
                 agents: List[BaseAgent],
                 config: TradingAgentsConfig):
        self.agents = agents
        self.config = config
        logger.info(f"协调器已初始化，包含 {len(agents)} 个智能体")
    
    async def coordinate(self, 
                        symbol: str,
                        market_data: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        协调所有智能体进行分析
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            context: 上下文信息
            
        Returns:
            综合分析结果
        """
        logger.info(f"开始协调分析: {symbol}")
        
        # 并行执行所有智能体的分析
        tasks = [
            agent.analyze(symbol, market_data, context)
            for agent in self.agents
        ]
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"智能体分析失败: {e}")
            responses = []
        
        # 过滤失败的响应
        valid_responses = [
            r for r in responses
            if isinstance(r, AgentResponse)
        ]
        
        if not valid_responses:
            return {
                "symbol": symbol,
                "consensus": {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "所有智能体分析失败"
                },
                "individual_results": [],
                "timestamp": datetime.now()
            }
        
        # 生成共识
        consensus = self._generate_consensus(valid_responses)
        
        return {
            "symbol": symbol,
            "consensus": consensus,
            "individual_results": [
                {
                    "agent": r.agent_name,
                    "signal": r.signal,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "analysis": r.analysis
                }
                for r in valid_responses
            ],
            "agent_count": len(valid_responses),
            "timestamp": datetime.now()
        }
    
    def _generate_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """生成共识决策"""
        
        if self.config.consensus_method == "weighted_vote":
            return self._weighted_vote_consensus(responses)
        elif self.config.consensus_method == "confidence_based":
            return self._confidence_based_consensus(responses)
        else:
            return self._simple_vote_consensus(responses)
    
from monitoring.metrics import get_monitor

def _weighted_vote_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """加权投票共识（带一进二硬门槛，优先提高精度）"""
        # 1) 首板校验
        for r in responses:
            if r.agent_name == 'limitup_validator' and isinstance(r.analysis, dict):
                if not r.analysis.get('validated', False):
                    get_monitor().collector.increment_counter("gate_reject_total", labels={"reason": "validator"})
                    return {"signal": "HOLD", "confidence": 0.6, "reasoning": f"首板校验未通过 | {r.reasoning}", "method": "gated"}
        # 2) 封板质量
        for r in responses:
            if r.agent_name == 'seal_quality' and isinstance(r.analysis, dict):
                val = r.analysis.get('seal_quality')
                if val is not None and val < self.config.min_seal_quality:
                    get_monitor().collector.increment_counter("gate_reject_total", labels={"reason": "seal_quality"})
                    return {"signal": "HOLD", "confidence": 0.6, "reasoning": f"封板质量不足({val:.2f}<{self.config.min_seal_quality})", "method": "gated"}
        # 3) 量能突增
        for r in responses:
            if r.agent_name == 'volume_surge' and isinstance(r.analysis, dict):
                val = r.analysis.get('volume_surge')
                if val is not None and val < self.config.min_volume_surge:
                    get_monitor().collector.increment_counter("gate_reject_total", labels={"reason": "volume_surge"})
                    return {"signal": "HOLD", "confidence": 0.6, "reasoning": f"量能不足({val:.2f}<{self.config.min_volume_surge})", "method": "gated"}
        # 4) 首触涨停时间
        for r in responses:
            if r.agent_name == 'limitup_validator' and isinstance(r.analysis, dict):
                v = r.analysis.get('limit_up_minutes')
                if v is not None and v > self.config.max_limit_up_minutes:
                    get_monitor().collector.increment_counter("gate_reject_total", labels={"reason": "late_limit_up"})
                    return {"signal": "HOLD", "confidence": 0.6, "reasoning": f"首触涨停过晚({v}>{self.config.max_limit_up_minutes}min)", "method": "gated"}
        # 5) 开板次数
        for r in responses:
            if r.agent_name == 'limitup_validator' and isinstance(r.analysis, dict):
                v = r.analysis.get('open_count')
                if v is not None and v > self.config.max_open_count:
                    get_monitor().collector.increment_counter("gate_reject_total", labels={"reason": "too_many_open"})
                    return {"signal": "HOLD", "confidence": 0.6, "reasoning": f"开板次数过多({v}>{self.config.max_open_count})", "method": "gated"}
        # 6) 价格区间
        for r in responses:
            if r.agent_name == 'limitup_validator' and isinstance(r.analysis, dict):
                p = r.analysis.get('close')
                if p is not None and (p < self.config.min_price or p > self.config.max_price):
                    get_monitor().collector.increment_counter("gate_reject_total", labels={"reason": "price_range"})
                    return {"signal": "HOLD", "confidence": 0.6, "reasoning": f"价格不在区间({p:.2f}∉[{self.config.min_price},{self.config.max_price}])", "method": "gated"}
        
        signal_weights = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0
        reasonings = []
        
        for response in responses:
            weight = self.config.agent_weights.get(response.agent_name, 0.2)
            weight *= response.confidence  # 置信度加权
            
            signal_weights[response.signal] += weight
            total_weight += weight
            reasonings.append(f"{response.agent_name}: {response.reasoning}")
        
        # 归一化
        if total_weight > 0:
            signal_weights = {k: v/total_weight for k, v in signal_weights.items()}
        
        # 选择权重最高的信号
        final_signal = max(signal_weights, key=signal_weights.get)
        final_confidence = signal_weights[final_signal]
        
        return {
            "signal": final_signal,
            "confidence": final_confidence,
            "reasoning": " | ".join(reasonings),
            "signal_distribution": signal_weights,
            "method": "weighted_vote"
        }
    
    def _confidence_based_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """基于置信度的共识"""
        
        # 选择置信度最高的响应
        best_response = max(responses, key=lambda r: r.confidence)
        
        return {
            "signal": best_response.signal,
            "confidence": best_response.confidence,
            "reasoning": f"采用{best_response.agent_name}的建议(置信度最高): {best_response.reasoning}",
            "best_agent": best_response.agent_name,
            "method": "confidence_based"
        }
    
    def _simple_vote_consensus(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """简单投票共识"""
        
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for response in responses:
            signal_counts[response.signal] += 1
        
        final_signal = max(signal_counts, key=signal_counts.get)
        total_votes = sum(signal_counts.values())
        final_confidence = signal_counts[final_signal] / total_votes if total_votes > 0 else 0.5
        
        return {
            "signal": final_signal,
            "confidence": final_confidence,
            "reasoning": f"投票结果: BUY={signal_counts['BUY']}, SELL={signal_counts['SELL']}, HOLD={signal_counts['HOLD']}",
            "vote_counts": signal_counts,
            "method": "simple_vote"
        }


# ============================================================================
# 主集成类
# ============================================================================

# =============================
# 一进二专用智能体（6个）
# =============================

class LimitUpValidatorAgent(BaseAgent):
    """首板校验：昨日是否涨停、是否满足入选条件（价格/新股/ST等）。"""
    async def analyze(self, symbol: str, market_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        from rd_agent.limit_up_data import LimitUpDataInterface
        import pandas as pd
        date = (context or {}).get('date') or datetime.now().strftime('%Y-%m-%d')
        data_if = LimitUpDataInterface('qlib')
        feats = data_if.get_limit_up_features([symbol], date)
        ok = False
        score = 0.0
        reason = ""
        analysis = {}
        if not feats.empty:
            row = feats.loc[symbol]
            strength = float(row.get('limit_up_strength', 0.0))
            cont = int(row.get('continuous_board', 0))
            limit_up_minutes = row.get('limit_up_minutes') if 'limit_up_minutes' in row else None
            open_count = row.get('open_count') if 'open_count' in row else None
            close_price = row.get('close') if 'close' in row else None
            ok = strength >= 95.0 and cont <= 2
            score = (strength - 90.0) / 10.0
            reason = f"昨日涨停强度={strength:.1f}, 连板={cont}, 首触涨停分钟={limit_up_minutes}, 开板次数={open_count}"
            analysis = {
                "validated": ok,
                "limit_up_minutes": float(limit_up_minutes) if limit_up_minutes is not None and pd.notna(limit_up_minutes) else None,
                "open_count": int(open_count) if open_count is not None and pd.notna(open_count) else None,
                "close": float(close_price) if close_price is not None and pd.notna(close_price) else None,
            }
        sig = "BUY" if ok else "HOLD"
        conf = max(0.5, min(0.9, 0.6 + score*0.3)) if ok else 0.5
        return AgentResponse(self.name, sig, conf, reason, analysis)

class SealQualityAgent(BaseAgent):
    """封板质量：收盘贴近最高、下影线短。"""
    async def analyze(self, symbol: str, market_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        from rd_agent.limit_up_data import LimitUpDataInterface
        date = (context or {}).get('date') or datetime.now().strftime('%Y-%m-%d')
        data_if = LimitUpDataInterface('qlib')
        feats = data_if.get_limit_up_features([symbol], date)
        seal = 0.0
        if not feats.empty:
            seal = float(feats.loc[symbol].get('seal_quality', 0.0))
        sig = "BUY" if seal >= 6.0 else "HOLD"
        conf = min(0.9, 0.55 + seal/20.0)
        return AgentResponse(self.name, sig, conf, f"封板质量={seal:.2f}", {"seal_quality": seal})

class VolumeSurgeAgent(BaseAgent):
    """量能突增：当日/20日均量。"""
    async def analyze(self, symbol: str, market_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        from rd_agent.limit_up_data import LimitUpDataInterface
        date = (context or {}).get('date') or datetime.now().strftime('%Y-%m-%d')
        data_if = LimitUpDataInterface('qlib')
        feats = data_if.get_limit_up_features([symbol], date)
        vs = 1.0
        if not feats.empty:
            vs = float(feats.loc[symbol].get('volume_surge', 1.0))
        sig = "BUY" if vs >= 2.0 else "HOLD"
        conf = min(0.9, 0.5 + (vs-1.0)/5.0)
        return AgentResponse(self.name, sig, conf, f"量能突增={vs:.2f}", {"volume_surge": vs})

class BoardContinuityAgent(BaseAgent):
    """连板约束：一进二偏好低连板（<=2）。"""
    async def analyze(self, symbol: str, market_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        from rd_agent.limit_up_data import LimitUpDataInterface
        date = (context or {}).get('date') or datetime.now().strftime('%Y-%m-%d')
        data_if = LimitUpDataInterface('qlib')
        feats = data_if.get_limit_up_features([symbol], date)
        cont = 0
        if not feats.empty:
            cont = int(feats.loc[symbol].get('continuous_board', 0))
        sig = "BUY" if cont <= 2 else "SELL"
        conf = 0.7 if cont <= 2 else 0.6
        return AgentResponse(self.name, sig, conf, f"连板数={cont}", {"continuous_board": cont})

class QlibMomentumAgent(BaseAgent):
    """Qlib动量：5日/1日动量加权。"""
    async def analyze(self, symbol: str, market_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        try:
            from qlib.data import D
            from qlib.config import REG_CN
            import qlib
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            end = (context or {}).get('date') or datetime.now().strftime('%Y-%m-%d')
            start = (pd.Timestamp(end) - pd.Timedelta(days=40)).strftime('%Y-%m-%d')
            df = D.features([symbol], ['$close'], start_time=start, end_time=end, freq='day')
            if df is None or df.empty:
                return AgentResponse(self.name, "HOLD", 0.5, "无数据", {})
            sdf = df.xs(symbol, level=0) if isinstance(df.index, pd.MultiIndex) else df
            closes = sdf['$close']
            ret5 = float(closes.pct_change(5).iloc[-1])
            ret1 = float(closes.pct_change(1).iloc[-1])
            score = 0.7*ret5 + 0.3*ret1
            sig = "BUY" if score > 0 else ("SELL" if score < 0 else "HOLD")
            conf = min(0.9, 0.6 + abs(score)*5)
            return AgentResponse(self.name, sig, conf, f"动量={score:.4f}", {"momentum": score})
        except Exception as e:
            return AgentResponse(self.name, "HOLD", 0.5, f"Qlib失败: {e}", {})

class RDCompositeAgent(BaseAgent):
    """RD组合评分：综合强度/封板/量能/连板（与决策引擎一致）。"""
    async def analyze(self, symbol: str, market_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        from rd_agent.limit_up_data import LimitUpDataInterface
        date = (context or {}).get('date') or datetime.now().strftime('%Y-%m-%d')
        data_if = LimitUpDataInterface('qlib')
        feats = data_if.get_limit_up_features([symbol], date)
        if feats.empty:
            return AgentResponse(self.name, "HOLD", 0.5, "无特征", {})
        row = feats.loc[symbol]
        strength = float(row.get('limit_up_strength', 0))
        seal = float(row.get('seal_quality', 0))
        vs = float(row.get('volume_surge', 1.0))
        cont = float(row.get('continuous_board', 0))
        comp = 0.40*((strength-60)/(100-60)) + 0.20*(seal/10.0) + 0.25*((vs-1.5)/(8.0-1.5)) - 0.15*((cont-2)/(6-2))
        comp = max(-1.0, min(1.0, comp))
        sig = "BUY" if comp>0 else ("SELL" if comp<-0 else "HOLD")
        conf = min(0.95, 0.65 + 0.30*max(0.0, comp))
        return AgentResponse(self.name, sig, conf, f"复合={comp:.3f}", {"composite": comp})


class RealTradingAgentsIntegration(
    """
    真实的TradingAgents集成
    完整实现多智能体系统
    """
    
    def __init__(self, config: Optional[TradingAgentsConfig] = None):
        """
        初始化集成
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or load_config()
        self.is_available = False
        self.llm = None
        self.agents = {}
        self.orchestrator = None
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化系统"""
        logger.info("初始化TradingAgents集成...")
        
        # 检查TradingAgents可用性
        is_available, message = check_tradingagents_available(self.config)
        # 强制官方模式：若要求官方且不可用，直接报错提示
        if self.config.force_official and not is_available:
            raise TradingAgentsNotAvailableError(
                f"已开启FORCE_TA_OFFICIAL，但未检测到官方 TradingAgents 于 {self.config.tradingagents_path}"
            )
        self.is_available = is_available
        logger.info(message)
        
        # 初始化LLM
        self.llm = LLMAdapter(self.config)
        
        # 初始化智能体
        self._init_agents()
        
        # 初始化协调器
        agent_list = list(self.agents.values())
        if agent_list:
            self.orchestrator = AgentOrchestrator(agent_list, self.config)
            logger.info(f"系统初始化完成，共{len(agent_list)}个智能体")
        else:
            logger.warning("没有启用的智能体")
    
    def _init_agents(self):
        """初始化智能体"""
        enabled_agents = self.config.get_enabled_agents()
        
        for agent_name in enabled_agents:
            try:
                if agent_name == "market_analyst":
                    self.agents[agent_name] = MarketAnalystAgent(self.llm, self.config)
                elif agent_name == "fundamental_analyst":
                    self.agents[agent_name] = FundamentalAnalystAgent(self.llm, self.config)
                elif agent_name == "technical_analyst":
                    self.agents[agent_name] = TechnicalAnalystAgent(self.llm, self.config)
                elif agent_name == "sentiment_analyst":
                    self.agents[agent_name] = SentimentAnalystAgent(self.llm, self.config)
                elif agent_name == "limitup_validator":
                    self.agents[agent_name] = LimitUpValidatorAgent(self.llm, self.config)
                elif agent_name == "seal_quality":
                    self.agents[agent_name] = SealQualityAgent(self.llm, self.config)
                elif agent_name == "volume_surge":
                    self.agents[agent_name] = VolumeSurgeAgent(self.llm, self.config)
                elif agent_name == "board_continuity":
                    self.agents[agent_name] = BoardContinuityAgent(self.llm, self.config)
                elif agent_name == "qlib_momentum":
                    self.agents[agent_name] = QlibMomentumAgent(self.llm, self.config)
                elif agent_name == "rd_composite":
                    self.agents[agent_name] = RDCompositeAgent(self.llm, self.config)
                
                logger.info(f"智能体已初始化: {agent_name}")
            
            except Exception as e:
                logger.error(f"初始化智能体失败 {agent_name}: {e}")
    
    async def analyze_stock(self, 
                           symbol: str,
                           market_data: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析股票
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            context: 上下文信息
            
        Returns:
            综合分析结果
        """
        if not self.orchestrator:
            return {
                "error": "系统未正确初始化",
                "symbol": symbol
            }
        
        try:
            result = await self.orchestrator.coordinate(symbol, market_data, context)
            return result
        
        except Exception as e:
            logger.error(f"分析股票失败: {e}")
            return {
                "error": str(e),
                "symbol": symbol
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "is_available": self.is_available,
            "llm_configured": self.llm.client is not None,
            "agents_count": len(self.agents),
            "enabled_agents": list(self.agents.keys()),
            "config": self.config.to_dict()
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_integration(config_file: Optional[str] = None) -> RealTradingAgentsIntegration:
    """
    创建TradingAgents集成实例
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        集成实例
    """
    config = load_config(config_file)
    return RealTradingAgentsIntegration(config)


# ============================================================================
# 测试和示例
# ============================================================================

async def test_integration():
    """测试集成"""
    print("=== TradingAgents集成测试 ===\n")
    
    # 创建集成
    integration = create_integration()
    
    # 检查状态
    status = integration.get_status()
    print("系统状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()
    
    # 测试分析
    symbol = "000001"
    market_data = {
        "price": 15.5,
        "change_pct": 0.025,
        "volume": 1000000,
        "technical_indicators": {
            "rsi": 65,
            "macd": 0.5,
            "macd_signal": 0.3
        },
        "fundamental_data": {
            "pe_ratio": 15.5,
            "pb_ratio": 2.1,
            "roe": 0.15
        },
        "sentiment": {
            "score": 0.65
        }
    }
    
    print(f"分析股票: {symbol}")
    result = await integration.analyze_stock(symbol, market_data)
    
    print("\n分析结果:")
    print(f"  最终决策: {result['consensus']['signal']}")
    print(f"  置信度: {result['consensus']['confidence']:.2%}")
    print(f"  理由: {result['consensus']['reasoning']}")
    print(f"\n参与智能体数量: {result['agent_count']}")
    
    for agent_result in result['individual_results']:
        print(f"\n  {agent_result['agent']}:")
        print(f"    信号: {agent_result['signal']}")
        print(f"    置信度: {agent_result['confidence']:.2%}")
        print(f"    理由: {agent_result['reasoning'][:100]}...")


if __name__ == "__main__":
    asyncio.run(test_integration())
