"""
智能问答与对话系统
支持自然语言交互、策略咨询、市场分析等功能
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import re
from collections import deque

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 导入其他模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from agents.trading_agents import MultiAgentManager
from rd_agent.research_agent import RDAgent
from qlib_integration.qlib_engine import QlibIntegrationEngine
from knowledge.knowledge_graph import KnowledgeGraphManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型"""
    USER_QUERY = "user_query"
    SYSTEM_RESPONSE = "system_response"
    AGENT_RESPONSE = "agent_response"
    ERROR_MESSAGE = "error_message"
    INFO_MESSAGE = "info_message"


class IntentType(Enum):
    """意图类型"""
    STRATEGY_QUERY = "strategy_query"          # 策略咨询
    MARKET_ANALYSIS = "market_analysis"        # 市场分析
    STOCK_INFO = "stock_info"                  # 股票信息
    PORTFOLIO_STATUS = "portfolio_status"      # 组合状态
    TRADING_COMMAND = "trading_command"        # 交易指令
    RISK_ASSESSMENT = "risk_assessment"        # 风险评估
    FACTOR_RESEARCH = "factor_research"        # 因子研究
    BACKTEST_REQUEST = "backtest_request"      # 回测请求
    GENERAL_CHAT = "general_chat"              # 一般对话
    SYSTEM_HELP = "system_help"                # 系统帮助


@dataclass
class Message:
    """对话消息"""
    content: str
    type: MessageType
    timestamp: datetime
    sender: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueContext:
    """对话上下文"""
    session_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    intent: Optional[IntentType] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


class ChatSystem:
    """智能对话系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话系统
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 核心组件
        self.nlu_engine = NLUEngine(config)
        self.dialogue_manager = DialogueManager(config)
        self.response_generator = ResponseGenerator(config)
        self.knowledge_base = KnowledgeBase(config)
        
        # 集成其他系统
        self.agent_manager = MultiAgentManager()
        self.rd_agent = RDAgent(config)
        self.qlib_engine = QlibIntegrationEngine(config)
        
        # 会话管理
        self.sessions = {}
        
        # 初始化大语言模型
        self._init_llm()
    
    def _init_llm(self):
        """初始化大语言模型"""
        if self.config.get("use_openai", False):
            openai.api_key = self.config.get("openai_api_key")
            self.llm_model = "gpt-4"
        else:
            # 使用本地模型
            model_name = self.config.get("local_model", "THUDM/chatglm-6b")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).half().cuda()
    
    async def process_message(self, 
                             user_id: str,
                             message: str,
                             session_id: Optional[str] = None) -> str:
        """
        处理用户消息
        
        Args:
            user_id: 用户ID
            message: 消息内容
            session_id: 会话ID
            
        Returns:
            响应内容
        """
        # 获取或创建会话
        if session_id is None:
            session_id = self._create_session_id()
        
        context = self._get_or_create_context(session_id, user_id)
        
        # 添加用户消息
        user_msg = Message(
            content=message,
            type=MessageType.USER_QUERY,
            timestamp=datetime.now(),
            sender=user_id
        )
        context.messages.append(user_msg)
        
        try:
            # NLU理解
            intent, entities = await self.nlu_engine.understand(message, context)
            context.intent = intent
            context.entities = entities
            
            # 对话管理
            action = await self.dialogue_manager.decide_action(context)
            
            # 执行动作
            response_data = await self._execute_action(action, context)
            
            # 生成响应
            response = await self.response_generator.generate(
                response_data,
                context
            )
            
            # 添加系统响应
            sys_msg = Message(
                content=response,
                type=MessageType.SYSTEM_RESPONSE,
                timestamp=datetime.now(),
                sender="system",
                metadata={"action": action, "data": response_data}
            )
            context.messages.append(sys_msg)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "抱歉，我在处理您的请求时遇到了问题。请稍后再试。"
    
    async def _execute_action(self, 
                             action: str,
                             context: DialogueContext) -> Dict[str, Any]:
        """执行动作"""
        if action == "analyze_strategy":
            return await self._analyze_strategy(context)
        
        elif action == "analyze_market":
            return await self._analyze_market(context)
        
        elif action == "get_stock_info":
            return await self._get_stock_info(context)
        
        elif action == "check_portfolio":
            return await self._check_portfolio(context)
        
        elif action == "execute_trade":
            return await self._execute_trade(context)
        
        elif action == "assess_risk":
            return await self._assess_risk(context)
        
        elif action == "research_factor":
            return await self._research_factor(context)
        
        elif action == "run_backtest":
            return await self._run_backtest(context)
        
        elif action == "provide_help":
            return await self._provide_help(context)
        
        else:
            return await self._general_chat(context)
    
    async def _analyze_strategy(self, context: DialogueContext) -> Dict[str, Any]:
        """分析策略"""
        entities = context.entities
        strategy_name = entities.get("strategy")
        
        # 调用多智能体系统分析
        analysis = await self.agent_manager.analyze_strategy(strategy_name)
        
        return {
            "strategy": strategy_name,
            "analysis": analysis,
            "recommendations": self._generate_recommendations(analysis)
        }
    
    async def _analyze_market(self, context: DialogueContext) -> Dict[str, Any]:
        """市场分析"""
        entities = context.entities
        market = entities.get("market", "A股")
        timeframe = entities.get("timeframe", "日线")
        
        # 获取市场数据
        market_data = await self._get_market_data(market, timeframe)
        
        # 调用智能体分析
        analysis = await self.agent_manager.analyze_market(market_data)
        
        return {
            "market": market,
            "timeframe": timeframe,
            "analysis": analysis,
            "indicators": self._calculate_indicators(market_data)
        }
    
    async def _get_stock_info(self, context: DialogueContext) -> Dict[str, Any]:
        """获取股票信息"""
        entities = context.entities
        symbol = entities.get("symbol")
        
        if not symbol:
            return {"error": "请提供股票代码"}
        
        # 获取股票数据
        stock_data = await self.qlib_engine.get_stock_data(symbol)
        
        # 获取基本面信息
        fundamental = await self._get_fundamental_data(symbol)
        
        # 技术分析
        technical = await self._technical_analysis(stock_data)
        
        return {
            "symbol": symbol,
            "price": stock_data["close"].iloc[-1],
            "change": stock_data["pct_change"].iloc[-1],
            "volume": stock_data["volume"].iloc[-1],
            "fundamental": fundamental,
            "technical": technical
        }
    
    async def _check_portfolio(self, context: DialogueContext) -> Dict[str, Any]:
        """检查组合状态"""
        user_id = context.user_id
        
        # 获取用户组合
        portfolio = await self._get_user_portfolio(user_id)
        
        # 计算绩效指标
        performance = await self._calculate_performance(portfolio)
        
        # 风险分析
        risk_metrics = await self._analyze_portfolio_risk(portfolio)
        
        return {
            "portfolio": portfolio,
            "performance": performance,
            "risk_metrics": risk_metrics
        }
    
    async def _execute_trade(self, context: DialogueContext) -> Dict[str, Any]:
        """执行交易"""
        entities = context.entities
        
        # 提取交易参数
        symbol = entities.get("symbol")
        action = entities.get("action")  # buy/sell
        quantity = entities.get("quantity")
        
        # 验证交易参数
        if not all([symbol, action, quantity]):
            return {"error": "缺少必要的交易参数"}
        
        # 风险检查
        risk_check = await self._pre_trade_risk_check({
            "symbol": symbol,
            "action": action,
            "quantity": quantity
        })
        
        if not risk_check["passed"]:
            return {
                "error": "风险检查未通过",
                "reason": risk_check["reason"]
            }
        
        # 执行交易（模拟）
        trade_result = {
            "order_id": f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "status": "submitted"
        }
        
        return trade_result
    
    async def _assess_risk(self, context: DialogueContext) -> Dict[str, Any]:
        """风险评估"""
        entities = context.entities
        target = entities.get("target", "portfolio")  # portfolio/stock/strategy
        
        if target == "portfolio":
            risk_data = await self._assess_portfolio_risk(context.user_id)
        elif target == "stock":
            symbol = entities.get("symbol")
            risk_data = await self._assess_stock_risk(symbol)
        else:
            strategy = entities.get("strategy")
            risk_data = await self._assess_strategy_risk(strategy)
        
        return {
            "target": target,
            "risk_data": risk_data,
            "recommendations": self._generate_risk_recommendations(risk_data)
        }
    
    async def _research_factor(self, context: DialogueContext) -> Dict[str, Any]:
        """因子研究"""
        entities = context.entities
        factor_name = entities.get("factor")
        
        # 调用RD-Agent进行因子研究
        research_result = await self.rd_agent.research_factor(factor_name)
        
        return {
            "factor": factor_name,
            "research": research_result,
            "performance": research_result.get("performance"),
            "significance": research_result.get("significance")
        }
    
    async def _run_backtest(self, context: DialogueContext) -> Dict[str, Any]:
        """运行回测"""
        entities = context.entities
        
        strategy = entities.get("strategy")
        start_date = entities.get("start_date", "2023-01-01")
        end_date = entities.get("end_date", "2024-01-01")
        
        # 调用Qlib引擎运行回测
        backtest_result = await self.qlib_engine.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "strategy": strategy,
            "period": f"{start_date} 至 {end_date}",
            "returns": backtest_result["total_return"],
            "sharpe": backtest_result["sharpe_ratio"],
            "max_drawdown": backtest_result["max_drawdown"],
            "trades": backtest_result["num_trades"]
        }
    
    async def _provide_help(self, context: DialogueContext) -> Dict[str, Any]:
        """提供帮助"""
        help_topics = [
            "策略咨询：询问关于交易策略的建议",
            "市场分析：获取市场趋势和分析",
            "股票查询：查询个股信息和行情",
            "组合管理：查看和管理投资组合",
            "风险评估：评估投资风险",
            "因子研究：研究和开发量化因子",
            "策略回测：测试交易策略的历史表现",
            "交易执行：下达和管理交易订单"
        ]
        
        return {
            "topics": help_topics,
            "examples": self._get_example_queries()
        }
    
    async def _general_chat(self, context: DialogueContext) -> Dict[str, Any]:
        """一般对话"""
        # 使用LLM生成响应
        response = await self._generate_llm_response(context)
        
        return {
            "response": response
        }
    
    async def _generate_llm_response(self, context: DialogueContext) -> str:
        """使用LLM生成响应"""
        # 构建提示
        prompt = self._build_prompt(context)
        
        if self.config.get("use_openai", False):
            # 使用OpenAI API
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的量化交易助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        
        else:
            # 使用本地模型
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_length=500)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _build_prompt(self, context: DialogueContext) -> str:
        """构建提示"""
        # 获取最近的对话历史
        recent_messages = context.messages[-5:]
        
        prompt = "对话历史：\n"
        for msg in recent_messages:
            prompt += f"{msg.sender}: {msg.content}\n"
        
        prompt += "\n请根据对话历史，提供专业的量化交易建议。"
        
        return prompt
    
    def _get_or_create_context(self, session_id: str, user_id: str) -> DialogueContext:
        """获取或创建对话上下文"""
        if session_id not in self.sessions:
            self.sessions[session_id] = DialogueContext(
                session_id=session_id,
                user_id=user_id
            )
        return self.sessions[session_id]
    
    def _create_session_id(self) -> str:
        """创建会话ID"""
        return f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    async def _get_market_data(self, market: str, timeframe: str) -> pd.DataFrame:
        """获取市场数据"""
        # 简化实现
        return pd.DataFrame()
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标"""
        return {}
    
    async def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """获取基本面数据"""
        return {}
    
    async def _technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """技术分析"""
        return {}
    
    async def _get_user_portfolio(self, user_id: str) -> Dict[str, Any]:
        """获取用户组合"""
        return {}
    
    async def _calculate_performance(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        """计算绩效"""
        return {}
    
    async def _analyze_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        """分析组合风险"""
        return {}
    
    async def _pre_trade_risk_check(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """交易前风险检查"""
        return {"passed": True}
    
    async def _assess_portfolio_risk(self, user_id: str) -> Dict[str, Any]:
        """评估组合风险"""
        return {}
    
    async def _assess_stock_risk(self, symbol: str) -> Dict[str, Any]:
        """评估股票风险"""
        return {}
    
    async def _assess_strategy_risk(self, strategy: str) -> Dict[str, Any]:
        """评估策略风险"""
        return {}
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成建议"""
        return []
    
    def _generate_risk_recommendations(self, risk_data: Dict[str, Any]) -> List[str]:
        """生成风险建议"""
        return []
    
    def _get_example_queries(self) -> List[str]:
        """获取示例查询"""
        return [
            "分析一下今天的市场走势",
            "600000的最新行情如何？",
            "我的投资组合表现怎么样？",
            "帮我研究一下动量因子",
            "测试一下均值回归策略的效果"
        ]


class NLUEngine:
    """自然语言理解引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._init_intent_patterns()
        self._init_entity_extractors()
    
    def _init_intent_patterns(self):
        """初始化意图模式"""
        self.intent_patterns = {
            IntentType.STRATEGY_QUERY: [
                r"策略", r"交易系统", r"量化模型", r"算法交易"
            ],
            IntentType.MARKET_ANALYSIS: [
                r"市场", r"行情", r"趋势", r"大盘", r"指数"
            ],
            IntentType.STOCK_INFO: [
                r"股票", r"个股", r"[0-9]{6}", r"代码"
            ],
            IntentType.PORTFOLIO_STATUS: [
                r"组合", r"持仓", r"收益", r"仓位"
            ],
            IntentType.TRADING_COMMAND: [
                r"买入", r"卖出", r"下单", r"交易", r"平仓"
            ],
            IntentType.RISK_ASSESSMENT: [
                r"风险", r"风控", r"止损", r"VaR", r"波动"
            ],
            IntentType.FACTOR_RESEARCH: [
                r"因子", r"特征", r"指标", r"研究", r"挖掘"
            ],
            IntentType.BACKTEST_REQUEST: [
                r"回测", r"测试", r"历史", r"模拟", r"验证"
            ],
            IntentType.SYSTEM_HELP: [
                r"帮助", r"怎么", r"如何", r"教程", r"说明"
            ]
        }
    
    def _init_entity_extractors(self):
        """初始化实体提取器"""
        self.entity_patterns = {
            "symbol": r"\b[036][0-9]{5}\b",  # 股票代码
            "date": r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",  # 日期
            "quantity": r"\d+[手股]",  # 数量
            "price": r"\d+\.?\d*[元]?",  # 价格
            "percent": r"\d+\.?\d*%",  # 百分比
            "timeframe": r"(日线|周线|月线|分钟|小时)",  # 时间框架
            "action": r"(买入|卖出|做多|做空|平仓)",  # 交易动作
        }
    
    async def understand(self, 
                        text: str,
                        context: DialogueContext) -> Tuple[IntentType, Dict[str, Any]]:
        """
        理解用户输入
        
        Returns:
            (意图类型, 实体字典)
        """
        # 意图识别
        intent = self._classify_intent(text)
        
        # 实体提取
        entities = self._extract_entities(text)
        
        # 上下文补充
        entities = self._enrich_with_context(entities, context)
        
        return intent, entities
    
    def _classify_intent(self, text: str) -> IntentType:
        """分类意图"""
        scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            scores[intent_type] = score
        
        # 返回得分最高的意图
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return IntentType.GENERAL_CHAT
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """提取实体"""
        entities = {}
        
        for entity_name, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_name] = matches[0] if len(matches) == 1 else matches
        
        return entities
    
    def _enrich_with_context(self, 
                            entities: Dict[str, Any],
                            context: DialogueContext) -> Dict[str, Any]:
        """使用上下文信息丰富实体"""
        # 从上下文中补充缺失的实体
        if "symbol" not in entities and "symbol" in context.entities:
            entities["symbol"] = context.entities["symbol"]
        
        return entities


class DialogueManager:
    """对话管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.action_rules = self._init_action_rules()
    
    def _init_action_rules(self) -> Dict[IntentType, str]:
        """初始化动作规则"""
        return {
            IntentType.STRATEGY_QUERY: "analyze_strategy",
            IntentType.MARKET_ANALYSIS: "analyze_market",
            IntentType.STOCK_INFO: "get_stock_info",
            IntentType.PORTFOLIO_STATUS: "check_portfolio",
            IntentType.TRADING_COMMAND: "execute_trade",
            IntentType.RISK_ASSESSMENT: "assess_risk",
            IntentType.FACTOR_RESEARCH: "research_factor",
            IntentType.BACKTEST_REQUEST: "run_backtest",
            IntentType.SYSTEM_HELP: "provide_help",
            IntentType.GENERAL_CHAT: "general_chat"
        }
    
    async def decide_action(self, context: DialogueContext) -> str:
        """决定执行的动作"""
        intent = context.intent
        
        # 根据意图返回对应的动作
        action = self.action_rules.get(intent, "general_chat")
        
        # 可以根据上下文进行动作调整
        if self._should_clarify(context):
            action = "request_clarification"
        
        return action
    
    def _should_clarify(self, context: DialogueContext) -> bool:
        """判断是否需要澄清"""
        # 如果关键实体缺失，需要澄清
        if context.intent == IntentType.STOCK_INFO and "symbol" not in context.entities:
            return True
        
        if context.intent == IntentType.TRADING_COMMAND:
            required = ["symbol", "action", "quantity"]
            if not all(k in context.entities for k in required):
                return True
        
        return False


class ResponseGenerator:
    """响应生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = self._init_templates()
    
    def _init_templates(self) -> Dict[str, str]:
        """初始化响应模板"""
        return {
            "stock_info": "【{symbol}】当前价格：{price:.2f}元，涨跌幅：{change:.2%}，成交量：{volume}",
            "market_analysis": "市场{market}在{timeframe}级别上{analysis}",
            "portfolio_status": "您的投资组合总价值：{total_value:.2f}元，今日收益：{daily_return:.2%}",
            "trade_confirmation": "订单已提交 - {action}{quantity}股{symbol}，订单号：{order_id}",
            "risk_warning": "⚠️ 风险提示：{risk_message}",
            "backtest_result": "策略回测结果 - 收益率：{returns:.2%}，夏普比率：{sharpe:.2f}，最大回撤：{max_drawdown:.2%}"
        }
    
    async def generate(self, 
                      data: Dict[str, Any],
                      context: DialogueContext) -> str:
        """生成响应"""
        intent = context.intent
        
        # 根据意图选择模板
        if intent == IntentType.STOCK_INFO:
            template = self.templates.get("stock_info")
            return template.format(**data) if not data.get("error") else data["error"]
        
        elif intent == IntentType.MARKET_ANALYSIS:
            return self._generate_market_analysis(data)
        
        elif intent == IntentType.PORTFOLIO_STATUS:
            return self._generate_portfolio_report(data)
        
        elif intent == IntentType.TRADING_COMMAND:
            if data.get("error"):
                return f"❌ 交易失败：{data['error']}"
            template = self.templates.get("trade_confirmation")
            return template.format(**data)
        
        elif intent == IntentType.BACKTEST_REQUEST:
            template = self.templates.get("backtest_result")
            return template.format(**data)
        
        elif intent == IntentType.SYSTEM_HELP:
            return self._generate_help_message(data)
        
        else:
            return data.get("response", "我理解了您的问题，让我来帮您分析...")
    
    def _generate_market_analysis(self, data: Dict[str, Any]) -> str:
        """生成市场分析报告"""
        analysis = data.get("analysis", {})
        
        report = f"📊 {data['market']}市场分析（{data['timeframe']}）\n\n"
        
        if "trend" in analysis:
            report += f"趋势：{analysis['trend']}\n"
        
        if "support" in analysis:
            report += f"支撑位：{analysis['support']}\n"
        
        if "resistance" in analysis:
            report += f"阻力位：{analysis['resistance']}\n"
        
        if "recommendation" in analysis:
            report += f"\n💡 建议：{analysis['recommendation']}"
        
        return report
    
    def _generate_portfolio_report(self, data: Dict[str, Any]) -> str:
        """生成组合报告"""
        portfolio = data.get("portfolio", {})
        performance = data.get("performance", {})
        risk = data.get("risk_metrics", {})
        
        report = "📈 投资组合报告\n\n"
        
        # 持仓信息
        if portfolio.get("positions"):
            report += "持仓明细：\n"
            for pos in portfolio["positions"][:5]:  # 显示前5个
                report += f"  {pos['symbol']}: {pos['quantity']}股, 盈亏{pos['pnl']:.2%}\n"
        
        # 绩效指标
        if performance:
            report += f"\n绩效指标：\n"
            report += f"  总收益率：{performance.get('total_return', 0):.2%}\n"
            report += f"  夏普比率：{performance.get('sharpe_ratio', 0):.2f}\n"
        
        # 风险指标
        if risk:
            report += f"\n风险指标：\n"
            report += f"  最大回撤：{risk.get('max_drawdown', 0):.2%}\n"
            report += f"  波动率：{risk.get('volatility', 0):.2%}\n"
        
        return report
    
    def _generate_help_message(self, data: Dict[str, Any]) -> str:
        """生成帮助信息"""
        topics = data.get("topics", [])
        examples = data.get("examples", [])
        
        help_msg = "💬 我可以帮助您：\n\n"
        
        for topic in topics:
            help_msg += f"• {topic}\n"
        
        if examples:
            help_msg += "\n示例问题：\n"
            for example in examples[:3]:
                help_msg += f"  \"{example}\"\n"
        
        return help_msg


class KnowledgeBase:
    """知识库"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents = []
        self.vector_store = None
        self._init_knowledge_base()
    
    def _init_knowledge_base(self):
        """初始化知识库"""
        # 加载文档
        self._load_documents()
        
        # 创建向量存储
        if self.config.get("use_embeddings", True):
            self._create_vector_store()
    
    def _load_documents(self):
        """加载文档"""
        # 加载交易知识、策略文档等
        doc_paths = self.config.get("document_paths", [])
        
        for path in doc_paths:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents.append({
                    "path": path,
                    "content": content
                })
    
    def _create_vector_store(self):
        """创建向量存储"""
        if not self.documents:
            return
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = []
        for doc in self.documents:
            chunks = text_splitter.split_text(doc["content"])
            texts.extend(chunks)
        
        # 创建嵌入
        embeddings = OpenAIEmbeddings() if self.config.get("use_openai") else None
        
        if embeddings:
            # 创建FAISS向量存储
            self.vector_store = FAISS.from_texts(texts, embeddings)
    
    async def search(self, query: str, k: int = 3) -> List[str]:
        """搜索相关知识"""
        if self.vector_store:
            results = self.vector_store.similarity_search(query, k=k)
            return [r.page_content for r in results]
        
        # 简单的关键词搜索
        results = []
        for doc in self.documents:
            if query.lower() in doc["content"].lower():
                # 提取相关段落
                lines = doc["content"].split('\n')
                for line in lines:
                    if query.lower() in line.lower():
                        results.append(line)
                        if len(results) >= k:
                            break
        
        return results


# 多轮对话管理
class ConversationManager:
    """多轮对话管理器"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations = {}
    
    def get_conversation(self, session_id: str) -> List[Message]:
        """获取会话历史"""
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_history)
        return list(self.conversations[session_id])
    
    def add_message(self, session_id: str, message: Message):
        """添加消息"""
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_history)
        self.conversations[session_id].append(message)
    
    def clear_conversation(self, session_id: str):
        """清除会话"""
        if session_id in self.conversations:
            del self.conversations[session_id]


if __name__ == "__main__":
    # 测试代码
    async def test():
        config = {
            "use_openai": False,
            "local_model": "THUDM/chatglm-6b",
            "use_embeddings": False
        }
        
        chat_system = ChatSystem(config)
        
        # 测试对话
        queries = [
            "帮我分析一下000001的行情",
            "今天市场怎么样？",
            "我想买入1000股600000",
            "帮我回测一下均值回归策略"
        ]
        
        user_id = "test_user"
        session_id = "test_session"
        
        for query in queries:
            print(f"用户: {query}")
            response = await chat_system.process_message(user_id, query, session_id)
            print(f"系统: {response}\n")
    
    # 运行测试
    # asyncio.run(test())