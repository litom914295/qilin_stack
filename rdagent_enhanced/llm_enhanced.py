"""
RD-Agent完整LLM增强
集成LLMManager、优化Prompt工程、实现因子发现和策略优化的LLM增强
支持OpenAI、Azure、Claude等多个LLM提供商
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# LLM提供商枚举
# ============================================================================

class LLMProvider(Enum):
    """LLM提供商类型"""
    OPENAI = "openai"
    AZURE = "azure"
    CLAUDE = "claude"
    QWEN = "qwen"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model: str
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str


# ============================================================================
# LLM管理器
# ============================================================================

class LLMManager:
    """统一LLM管理器"""
    
    def __init__(self, config: LLMConfig):
        """
        初始化LLM管理器
        
        Args:
            config: LLM配置
        """
        self.config = config
        self.provider = config.provider
        self.client = None
        
        self._init_client()
        logger.info(f"LLM管理器初始化: {config.provider.value}/{config.model}")
    
    def _init_client(self):
        """初始化LLM客户端"""
        if self.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config.api_key or os.getenv('OPENAI_API_KEY'),
                base_url=self.config.api_base or None,
                timeout=self.config.timeout
            )
        
        elif self.provider == LLMProvider.AZURE:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.config.api_key or os.getenv('AZURE_OPENAI_KEY'),
                api_version="2024-02-01",
                azure_endpoint=self.config.api_base
            )
        
        elif self.provider == LLMProvider.CLAUDE:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
            )
        
        elif self.provider == LLMProvider.QWEN:
            # 通义千问
            import dashscope
            dashscope.api_key = self.config.api_key or os.getenv('DASHSCOPE_API_KEY')
            self.client = dashscope
        
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
    
    async def complete(self, 
                      prompt: str,
                      system_prompt: str = "",
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """
        生成LLM补全
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度
            max_tokens: 最大token数
            
        Returns:
            LLM响应
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        if self.provider in [LLMProvider.OPENAI, LLMProvider.AZURE]:
            return await self._openai_complete(prompt, system_prompt, temp, max_tok)
        
        elif self.provider == LLMProvider.CLAUDE:
            return await self._claude_complete(prompt, system_prompt, temp, max_tok)
        
        elif self.provider == LLMProvider.QWEN:
            return await self._qwen_complete(prompt, system_prompt, temp, max_tok)
        
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
    
    async def _openai_complete(self, prompt: str, system_prompt: str, 
                              temperature: float, max_tokens: int) -> LLMResponse:
        """OpenAI补全"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            model=response.model,
            finish_reason=response.choices[0].finish_reason
        )
    
    async def _claude_complete(self, prompt: str, system_prompt: str,
                              temperature: float, max_tokens: int) -> LLMResponse:
        """Claude补全"""
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else None,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            model=response.model,
            finish_reason=response.stop_reason
        )
    
    async def _qwen_complete(self, prompt: str, system_prompt: str,
                            temperature: float, max_tokens: int) -> LLMResponse:
        """通义千问补全"""
        from dashscope import Generation
        
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        response = Generation.call(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            result_format='message'
        )
        
        return LLMResponse(
            content=response.output.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            },
            model=self.config.model,
            finish_reason=response.output.choices[0].finish_reason
        )


# ============================================================================
# Prompt模板
# ============================================================================

class PromptTemplates:
    """Prompt模板库"""
    
    # 因子发现系统提示
    FACTOR_DISCOVERY_SYSTEM = """你是一个专业的量化因子研究专家。
你的任务是基于市场数据、技术指标和基本面信息，发现和设计新的alpha因子。
请确保因子：
1. 具有经济学直觉和可解释性
2. 与现有因子具有较低相关性
3. 在历史回测中表现稳定
4. 计算效率高，适合实盘使用"""
    
    # 因子发现用户提示
    FACTOR_DISCOVERY_USER = """请基于以下信息设计一个新的alpha因子：

现有因子表现：
{existing_factors}

市场特征：
{market_features}

要求：
1. 提供因子的数学表达式（使用Qlib语法）
2. 解释因子的经济学含义
3. 说明为什么该因子可能有效
4. 给出预期的IC和IR范围

请以JSON格式返回：
{{
    "factor_name": "因子名称",
    "expression": "Qlib表达式",
    "description": "因子描述",
    "rationale": "有效性原因",
    "expected_ic": [最小IC, 最大IC],
    "expected_ir": [最小IR, 最大IR]
}}"""
    
    # 策略优化系统提示
    STRATEGY_OPTIMIZATION_SYSTEM = """你是一个专业的量化策略优化专家。
你的任务是分析策略回测结果，识别问题，并提出优化建议。
请确保优化建议：
1. 针对性强，能解决实际问题
2. 可操作性强，易于实现
3. 风险可控，不会引入新的问题
4. 有理论支撑或实证依据"""
    
    # 策略优化用户提示
    STRATEGY_OPTIMIZATION_USER = """请分析以下策略回测结果并提出优化建议：

策略概况：
{strategy_summary}

回测结果：
{backtest_results}

风险指标：
{risk_metrics}

问题：
{issues}

请以JSON格式返回：
{{
    "analysis": "问题分析",
    "optimizations": [
        {{
            "type": "优化类型（参数/逻辑/风控等）",
            "description": "优化描述",
            "rationale": "优化理由",
            "implementation": "实现方式",
            "expected_improvement": "预期改善"
        }}
    ],
    "priority": "优先级（高/中/低）"
}}"""
    
    # 模型解释系统提示
    MODEL_EXPLANATION_SYSTEM = """你是一个专业的机器学习模型解释专家。
你的任务是分析模型特征重要性、预测结果，提供可解释的分析。
请确保解释：
1. 准确反映模型行为
2. 易于非技术人员理解
3. 突出关键因素和驱动力
4. 提供可操作的洞察"""
    
    # 模型解释用户提示
    MODEL_EXPLANATION_USER = """请解释以下模型的预测结果：

模型类型：{model_type}

特征重要性：
{feature_importance}

预测股票：{symbol}
预测收益率：{predicted_return}

历史数据：
{historical_data}

请以JSON格式返回：
{{
    "summary": "预测摘要",
    "key_factors": ["关键因素1", "关键因素2", ...],
    "explanation": "详细解释",
    "confidence": "置信度（高/中/低）",
    "risks": ["风险1", "风险2", ...]
}}"""


# ============================================================================
# LLM增强的因子发现
# ============================================================================

class LLMFactorDiscovery:
    """LLM增强的因子发现"""
    
    def __init__(self, llm_manager: LLMManager):
        """
        初始化因子发现
        
        Args:
            llm_manager: LLM管理器
        """
        self.llm = llm_manager
        self.templates = PromptTemplates()
    
    async def discover_factors(self,
                              existing_factors: Dict[str, Any],
                              market_features: Dict[str, Any],
                              n_factors: int = 5) -> List[Dict[str, Any]]:
        """
        发现新因子
        
        Args:
            existing_factors: 现有因子信息
            market_features: 市场特征
            n_factors: 生成因子数量
            
        Returns:
            新因子列表
        """
        new_factors = []
        
        for i in range(n_factors):
            try:
                # 构建提示
                user_prompt = self.templates.FACTOR_DISCOVERY_USER.format(
                    existing_factors=json.dumps(existing_factors, ensure_ascii=False, indent=2),
                    market_features=json.dumps(market_features, ensure_ascii=False, indent=2)
                )
                
                # 调用LLM
                response = await self.llm.complete(
                    prompt=user_prompt,
                    system_prompt=self.templates.FACTOR_DISCOVERY_SYSTEM,
                    temperature=0.8  # 较高温度以增加创造性
                )
                
                # 解析响应
                factor_data = self._parse_factor_response(response.content)
                if factor_data:
                    new_factors.append(factor_data)
                    logger.info(f"发现新因子: {factor_data['factor_name']}")
                
            except Exception as e:
                logger.error(f"因子发现失败 ({i+1}/{n_factors}): {e}")
        
        return new_factors
    
    def _parse_factor_response(self, content: str) -> Optional[Dict[str, Any]]:
        """解析LLM响应为因子数据"""
        try:
            # 提取JSON部分
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            factor_data = json.loads(content.strip())
            return factor_data
        except Exception as e:
            logger.error(f"解析因子响应失败: {e}")
            return None


# ============================================================================
# LLM增强的策略优化
# ============================================================================

class LLMStrategyOptimizer:
    """LLM增强的策略优化"""
    
    def __init__(self, llm_manager: LLMManager):
        """
        初始化策略优化器
        
        Args:
            llm_manager: LLM管理器
        """
        self.llm = llm_manager
        self.templates = PromptTemplates()
    
    async def optimize_strategy(self,
                               strategy_summary: str,
                               backtest_results: Dict[str, Any],
                               risk_metrics: Dict[str, Any],
                               issues: List[str]) -> Dict[str, Any]:
        """
        优化策略
        
        Args:
            strategy_summary: 策略概况
            backtest_results: 回测结果
            risk_metrics: 风险指标
            issues: 已知问题
            
        Returns:
            优化建议
        """
        try:
            # 构建提示
            user_prompt = self.templates.STRATEGY_OPTIMIZATION_USER.format(
                strategy_summary=strategy_summary,
                backtest_results=json.dumps(backtest_results, ensure_ascii=False, indent=2),
                risk_metrics=json.dumps(risk_metrics, ensure_ascii=False, indent=2),
                issues=json.dumps(issues, ensure_ascii=False)
            )
            
            # 调用LLM
            response = await self.llm.complete(
                prompt=user_prompt,
                system_prompt=self.templates.STRATEGY_OPTIMIZATION_SYSTEM,
                temperature=0.5  # 中等温度，平衡创造性和准确性
            )
            
            # 解析响应
            optimization = self._parse_optimization_response(response.content)
            return optimization
            
        except Exception as e:
            logger.error(f"策略优化失败: {e}")
            return {}
    
    def _parse_optimization_response(self, content: str) -> Dict[str, Any]:
        """解析优化响应"""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"解析优化响应失败: {e}")
            return {}


# ============================================================================
# LLM增强的模型解释
# ============================================================================

class LLMModelExplainer:
    """LLM增强的模型解释"""
    
    def __init__(self, llm_manager: LLMManager):
        """
        初始化模型解释器
        
        Args:
            llm_manager: LLM管理器
        """
        self.llm = llm_manager
        self.templates = PromptTemplates()
    
    async def explain_prediction(self,
                                model_type: str,
                                feature_importance: Dict[str, float],
                                symbol: str,
                                predicted_return: float,
                                historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解释预测结果
        
        Args:
            model_type: 模型类型
            feature_importance: 特征重要性
            symbol: 股票代码
            predicted_return: 预测收益率
            historical_data: 历史数据
            
        Returns:
            解释结果
        """
        try:
            # 构建提示
            user_prompt = self.templates.MODEL_EXPLANATION_USER.format(
                model_type=model_type,
                feature_importance=json.dumps(feature_importance, ensure_ascii=False, indent=2),
                symbol=symbol,
                predicted_return=f"{predicted_return:.2%}",
                historical_data=json.dumps(historical_data, ensure_ascii=False, indent=2)
            )
            
            # 调用LLM
            response = await self.llm.complete(
                prompt=user_prompt,
                system_prompt=self.templates.MODEL_EXPLANATION_SYSTEM,
                temperature=0.3  # 较低温度以确保准确性
            )
            
            # 解析响应
            explanation = self._parse_explanation_response(response.content)
            return explanation
            
        except Exception as e:
            logger.error(f"模型解释失败: {e}")
            return {}
    
    def _parse_explanation_response(self, content: str) -> Dict[str, Any]:
        """解析解释响应"""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"解析解释响应失败: {e}")
            return {}


# ============================================================================
# 使用示例
# ============================================================================

async def example_llm_enhanced():
    """LLM增强示例"""
    print("=== RD-Agent LLM增强示例 ===\n")
    
    # 配置LLM
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        temperature=0.7,
        max_tokens=2000
    )
    
    llm_manager = LLMManager(config)
    
    # 1. 因子发现
    print("1. LLM因子发现")
    factor_discovery = LLMFactorDiscovery(llm_manager)
    
    existing_factors = {
        "momentum": {"ic": 0.05, "ir": 1.2},
        "volatility": {"ic": 0.03, "ir": 0.8}
    }
    market_features = {
        "trend": "上涨",
        "volatility": "中等",
        "sector_rotation": "科技板块领涨"
    }
    
    new_factors = await factor_discovery.discover_factors(
        existing_factors=existing_factors,
        market_features=market_features,
        n_factors=2
    )
    
    print(f"发现 {len(new_factors)} 个新因子")
    for factor in new_factors:
        print(f"  - {factor.get('factor_name', 'Unknown')}")
    
    # 2. 策略优化
    print("\n2. LLM策略优化")
    optimizer = LLMStrategyOptimizer(llm_manager)
    
    optimization = await optimizer.optimize_strategy(
        strategy_summary="动量因子多头策略",
        backtest_results={"annual_return": 0.15, "sharpe": 1.5, "max_drawdown": -0.12},
        risk_metrics={"var_95": -0.03, "cvar_95": -0.045},
        issues=["回撤较大", "换手率过高"]
    )
    
    print(f"优化建议: {optimization.get('analysis', 'N/A')}")
    
    # 3. 模型解释
    print("\n3. LLM模型解释")
    explainer = LLMModelExplainer(llm_manager)
    
    explanation = await explainer.explain_prediction(
        model_type="LightGBM",
        feature_importance={"momentum_20": 0.35, "volatility_60": 0.25, "volume_ratio": 0.20},
        symbol="600519.SH",
        predicted_return=0.08,
        historical_data={"recent_trend": "上涨", "volume": "放量"}
    )
    
    print(f"预测解释: {explanation.get('summary', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(example_llm_enhanced())
