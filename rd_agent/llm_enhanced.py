"""
RD-Agent完整LLM增强模块
实现LLM管理器、Prompt工程优化、因子发现和策略优化
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ============================================================================
# LLM管理器
# ============================================================================

class LLMManager:
    """LLM管理器 - 统一管理多个LLM提供商"""
    
    SUPPORTED_PROVIDERS = ['openai', 'anthropic', 'azure', 'local']
    
    def __init__(self,
                 provider: str = 'openai',
                 model: str = 'gpt-4-turbo',
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 4000):
        """
        初始化LLM管理器
        
        Args:
            provider: LLM提供商 ('openai', 'anthropic', 'azure', 'local')
            model: 模型名称
            api_key: API密钥
            base_url: API基础URL(可选)
            temperature: 温度参数
            max_tokens: 最大token数
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"不支持的provider: {provider}, 支持: {self.SUPPORTED_PROVIDERS}")
        
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = None
        self._init_client()
        
        logger.info(f"LLM管理器初始化: provider={provider}, model={model}")
    
    def _init_client(self):
        """初始化LLM客户端"""
        try:
            if self.provider == 'openai':
                import openai
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info("✅ OpenAI客户端初始化成功")
                
            elif self.provider == 'anthropic':
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("✅ Anthropic客户端初始化成功")
                
            elif self.provider == 'azure':
                import openai
                self.client = openai.AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.base_url
                )
                logger.info("✅ Azure OpenAI客户端初始化成功")
                
            elif self.provider == 'local':
                # 本地模型(如Ollama)
                import openai
                self.client = openai.OpenAI(
                    base_url=self.base_url or "http://localhost:11434/v1",
                    api_key="dummy"  # 本地模型不需要真实key
                )
                logger.info("✅ 本地模型客户端初始化成功")
                
        except ImportError as e:
            logger.error(f"❌ 无法导入{self.provider}库: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 客户端初始化失败: {e}")
            raise
    
    async def generate(self,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度(可选,覆盖默认值)
            max_tokens: 最大token(可选,覆盖默认值)
            
        Returns:
            生成的文本
        """
        if self.client is None:
            raise RuntimeError("LLM客户端未初始化")
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            if self.provider in ['openai', 'azure', 'local']:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok
                )
                
                return response.choices[0].message.content
                
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tok,
                    temperature=temp,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """获取LLM管理器状态"""
        return {
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'client_initialized': self.client is not None
        }


# ============================================================================
# Prompt工程师
# ============================================================================

class PromptEngineer:
    """Prompt工程优化 - 构建高质量的提示词"""
    
    @staticmethod
    def build_factor_discovery_prompt(
        data_stats: Dict[str, Any],
        objectives: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建因子发现的最优prompt
        
        Args:
            data_stats: 数据统计信息
            objectives: 目标要求
            constraints: 约束条件
            
        Returns:
            优化的prompt
        """
        prompt = f"""你是一位资深量化研究员,擅长发现alpha因子。

【数据概况】
- 股票池规模: {data_stats.get('num_stocks', 'N/A')}只股票
- 时间跨度: {data_stats.get('date_range', 'N/A')}
- 可用特征: {', '.join(data_stats.get('features', [])[:10])}等
- 数据频率: {data_stats.get('frequency', '日线')}

【目标要求】
- IC目标: {objectives.get('target_ic', '> 0.05')}
- 最大回撤: {objectives.get('max_drawdown', '< 20%')}
- 夏普比率: {objectives.get('sharpe', '> 2.0')}

【约束条件】"""
        
        if constraints:
            for key, value in constraints.items():
                prompt += f"\n- {key}: {value}"
        else:
            prompt += "\n- 无特殊约束"
        
        prompt += """

【任务】
请提出3-5个创新的alpha因子假设,每个因子需包含:

1. **因子名称**: 简洁且有意义
2. **因子公式**: 使用数学表达式或伪代码
3. **经济学解释**: 为什么这个因子有效?背后的逻辑是什么?
4. **预期IC**: 基于你的分析,预期信息系数范围
5. **风险提示**: 这个因子可能失效的情况

请按以下格式回复:

### 因子1: [因子名称]
- **公式**: [公式]
- **解释**: [经济学解释]
- **预期IC**: [IC范围]
- **风险**: [风险说明]

### 因子2: ...

开始你的分析:"""
        
        return prompt
    
    @staticmethod
    def build_strategy_optimization_prompt(
        performance: Dict[str, Any],
        current_params: Dict[str, Any],
        market_condition: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建策略优化的prompt
        
        Args:
            performance: 当前性能指标
            current_params: 当前参数配置
            market_condition: 市场状态(可选)
            
        Returns:
            优化的prompt
        """
        prompt = f"""你是一位量化策略优化专家,擅长根据回测结果改进策略。

【当前性能】
- 年化收益率: {performance.get('annual_return', 'N/A'):.2%}
- 夏普比率: {performance.get('sharpe', 'N/A'):.2f}
- 最大回撤: {performance.get('max_drawdown', 'N/A'):.2%}
- 信息系数(IC): {performance.get('ic', 'N/A'):.4f}
- 胜率: {performance.get('win_rate', 'N/A'):.2%}

【当前参数】"""
        
        for key, value in current_params.items():
            prompt += f"\n- {key}: {value}"
        
        if market_condition:
            prompt += f"""

【市场状态】
- 趋势: {market_condition.get('trend', '震荡')}
- 波动率: {market_condition.get('volatility', '中等')}
- 成交量: {market_condition.get('volume', '正常')}"""
        
        prompt += """

【优化目标】
1. 提升夏普比率至 > 2.0
2. 降低最大回撤至 < 15%
3. 保持年化收益率 > 20%

【任务】
请分析当前策略的问题,并提出具体的优化建议:

1. **问题诊断**: 指出当前策略的主要问题
2. **参数优化**: 建议调整哪些参数?如何调整?
3. **策略改进**: 是否需要添加新的逻辑或规则?
4. **风险控制**: 如何进一步降低风险?
5. **实施步骤**: 分步骤说明如何实施这些改进

请详细说明你的分析和建议:"""
        
        return prompt
    
    @staticmethod
    def build_model_interpretation_prompt(
        model_type: str,
        feature_importance: Dict[str, float],
        predictions: List[float],
        actuals: List[float]
    ) -> str:
        """
        构建模型解释的prompt
        
        Args:
            model_type: 模型类型
            feature_importance: 特征重要性
            predictions: 预测值
            actuals: 实际值
            
        Returns:
            prompt
        """
        # 计算简单的评估指标
        import numpy as np
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 0 else 0
        
        # 前10个重要特征
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        prompt = f"""你是一位机器学习模型解释专家。

【模型信息】
- 模型类型: {model_type}
- 预测-实际相关系数: {correlation:.4f}
- 预测样本数: {len(predictions)}

【Top 10 重要特征】"""
        
        for i, (feature, importance) in enumerate(top_features, 1):
            prompt += f"\n{i}. {feature}: {importance:.4f}"
        
        prompt += """

【任务】
请解释这个模型:

1. **模型行为**: 从特征重要性看,模型主要依赖哪些信息?
2. **预测逻辑**: 模型可能的决策逻辑是什么?
3. **优势与局限**: 当前模型配置的优势和潜在问题?
4. **改进建议**: 如何进一步提升模型性能?

请提供你的分析:"""
        
        return prompt
    
    @staticmethod
    def build_risk_assessment_prompt(
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        historical_volatility: float
    ) -> str:
        """
        构建风险评估的prompt
        
        Args:
            portfolio: 投资组合信息
            market_data: 市场数据
            historical_volatility: 历史波动率
            
        Returns:
            prompt
        """
        prompt = f"""你是一位风险管理专家。

【投资组合】
- 持仓数量: {portfolio.get('num_positions', 0)}只
- 总市值: {portfolio.get('total_value', 0):,.0f}元
- 集中度(Top 5): {portfolio.get('concentration_top5', 0):.2%}
- 行业分布: {', '.join([f"{k}:{v:.1%}" for k, v in portfolio.get('sector_weights', {}).items()][:3])}

【市场环境】
- 市场情绪: {market_data.get('sentiment', '中性')}
- 当前波动率: {market_data.get('current_volatility', historical_volatility):.2%}
- 历史波动率: {historical_volatility:.2%}
- 成交量比率: {market_data.get('volume_ratio', 1.0):.2f}

【任务】
请评估当前投资组合的风险:

1. **主要风险**: 识别当前最大的风险因素
2. **风险量化**: 估计可能的最大损失(VaR)
3. **对冲建议**: 如何对冲主要风险?
4. **预警指标**: 需要关注哪些预警信号?
5. **应急方案**: 如果风险爆发,应如何应对?

请提供详细的风险评估报告:"""
        
        return prompt


# ============================================================================
# 完整LLM集成
# ============================================================================

class FullLLMIntegration:
    """完整LLM集成 - 结合LLM管理器和Prompt工程"""
    
    def __init__(self,
                 provider: str = 'openai',
                 model: str = 'gpt-4-turbo',
                 api_key: Optional[str] = None):
        """
        初始化完整LLM集成
        
        Args:
            provider: LLM提供商
            model: 模型名称
            api_key: API密钥
        """
        self.llm_manager = LLMManager(
            provider=provider,
            model=model,
            api_key=api_key
        )
        
        self.prompt_engineer = PromptEngineer()
        
        # 历史记录
        self.generation_history = []
        
        logger.info("完整LLM集成初始化成功")
    
    async def generate_factor_hypothesis(self,
                                        data_stats: Dict[str, Any],
                                        objectives: Dict[str, Any],
                                        constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成因子假设
        
        Args:
            data_stats: 数据统计
            objectives: 目标
            constraints: 约束
            
        Returns:
            因子假设结果
        """
        logger.info("生成因子假设...")
        
        # 构建prompt
        prompt = self.prompt_engineer.build_factor_discovery_prompt(
            data_stats, objectives, constraints
        )
        
        # 系统prompt
        system_prompt = "你是一位世界顶级的量化研究员,精通alpha因子挖掘和金融市场分析。"
        
        # 生成
        response = await self.llm_manager.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.8  # 较高温度以获得更有创意的建议
        )
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'task': 'factor_hypothesis',
            'prompt_length': len(prompt),
            'response': response,
            'response_length': len(response)
        }
        
        self.generation_history.append(result)
        
        return result
    
    async def optimize_strategy(self,
                               performance: Dict[str, Any],
                               current_params: Dict[str, Any],
                               market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        优化策略
        
        Args:
            performance: 性能指标
            current_params: 当前参数
            market_condition: 市场状态
            
        Returns:
            优化建议
        """
        logger.info("生成策略优化建议...")
        
        prompt = self.prompt_engineer.build_strategy_optimization_prompt(
            performance, current_params, market_condition
        )
        
        system_prompt = "你是一位经验丰富的量化策略优化专家,擅长分析策略性能并提出改进方案。"
        
        response = await self.llm_manager.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'task': 'strategy_optimization',
            'response': response
        }
        
        self.generation_history.append(result)
        
        return result
    
    async def interpret_model(self,
                            model_type: str,
                            feature_importance: Dict[str, float],
                            predictions: List[float],
                            actuals: List[float]) -> Dict[str, Any]:
        """
        解释模型
        
        Args:
            model_type: 模型类型
            feature_importance: 特征重要性
            predictions: 预测值
            actuals: 实际值
            
        Returns:
            模型解释
        """
        logger.info("生成模型解释...")
        
        prompt = self.prompt_engineer.build_model_interpretation_prompt(
            model_type, feature_importance, predictions, actuals
        )
        
        system_prompt = "你是一位机器学习模型解释专家,擅长解读模型行为并提供洞察。"
        
        response = await self.llm_manager.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6
        )
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'task': 'model_interpretation',
            'response': response
        }
        
        self.generation_history.append(result)
        
        return result
    
    async def assess_risk(self,
                         portfolio: Dict[str, Any],
                         market_data: Dict[str, Any],
                         historical_volatility: float) -> Dict[str, Any]:
        """
        评估风险
        
        Args:
            portfolio: 投资组合
            market_data: 市场数据
            historical_volatility: 历史波动率
            
        Returns:
            风险评估报告
        """
        logger.info("生成风险评估报告...")
        
        prompt = self.prompt_engineer.build_risk_assessment_prompt(
            portfolio, market_data, historical_volatility
        )
        
        system_prompt = "你是一位资深风险管理专家,精通市场风险分析和对冲策略。"
        
        response = await self.llm_manager.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5  # 较低温度以获得更保守的评估
        )
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'task': 'risk_assessment',
            'response': response
        }
        
        self.generation_history.append(result)
        
        return result
    
    def export_history(self, output_path: str):
        """导出生成历史"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.generation_history, f, indent=2, ensure_ascii=False)
        logger.info(f"历史记录已导出到 {output_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'llm_manager': self.llm_manager.get_status(),
            'generation_count': len(self.generation_history),
            'tasks_completed': [h['task'] for h in self.generation_history]
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_llm_integration(config_file: Optional[str] = None) -> FullLLMIntegration:
    """
    创建LLM集成实例
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        FullLLMIntegration实例
    """
    config = {}
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return FullLLMIntegration(
        provider=config.get('provider', 'openai'),
        model=config.get('model', 'gpt-4-turbo'),
        api_key=config.get('api_key')
    )


# ============================================================================
# 示例
# ============================================================================

async def example_usage():
    """使用示例"""
    
    # 创建LLM集成
    llm = FullLLMIntegration(
        provider='openai',
        model='gpt-4-turbo',
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # 1. 生成因子假设
    print("=" * 70)
    print("任务1: 生成因子假设")
    print("=" * 70)
    
    result1 = await llm.generate_factor_hypothesis(
        data_stats={
            'num_stocks': 3000,
            'date_range': '2020-01-01 to 2024-12-31',
            'features': ['close', 'volume', 'turnover', 'pe', 'pb'],
            'frequency': '日线'
        },
        objectives={
            'target_ic': '> 0.05',
            'max_drawdown': '< 20%',
            'sharpe': '> 2.0'
        }
    )
    
    print(result1['response'])
    
    # 2. 优化策略
    print("\n" + "=" * 70)
    print("任务2: 策略优化")
    print("=" * 70)
    
    result2 = await llm.optimize_strategy(
        performance={
            'annual_return': 0.15,
            'sharpe': 1.5,
            'max_drawdown': 0.25,
            'ic': 0.04,
            'win_rate': 0.55
        },
        current_params={
            'lookback_period': 20,
            'rebalance_frequency': 'weekly',
            'max_position_size': 0.05
        }
    )
    
    print(result2['response'])
    
    # 3. 查看状态
    print("\n" + "=" * 70)
    print("LLM集成状态")
    print("=" * 70)
    
    status = llm.get_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(example_usage())
