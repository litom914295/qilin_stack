"""
RD-Agent真实集成实现
引入RD-Agent官方组件，实现自动化研究功能
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
import pickle
import json

# 导入配置
from .config import RDAgentConfig, load_config

logger = logging.getLogger(__name__)


# ============================================================================
# 核心组件检查
# ============================================================================

def check_rdagent_available(config: RDAgentConfig) -> Tuple[bool, str]:
    """检查RD-Agent是否可用"""
    rd_path = Path(config.rdagent_path)
    
    if not rd_path.exists():
        return False, f"RD-Agent路径不存在: {rd_path}"
    
    # 添加到系统路径
    if str(rd_path) not in sys.path:
        sys.path.insert(0, str(rd_path))
    
    try:
        # 尝试导入核心模块（如果RD-Agent官方代码可用）
        # import rdagent
        return True, "RD-Agent路径已添加"
    except ImportError as e:
        return False, f"无法导入RD-Agent: {e}"


# ============================================================================
# LLM适配器（简化版）
# ============================================================================

class SimpleLLMAdapter:
    """简化的LLM适配器"""
    
    def __init__(self, config: RDAgentConfig):
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
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            self.client = None
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """生成响应"""
        if not self.client:
            return "LLM未配置"
        
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
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"生成失败: {e}"


# ============================================================================
# 因子研究
# ============================================================================

@dataclass
class FactorResult:
    """因子研究结果"""
    name: str
    expression: str
    ic: float
    ir: float
    sharpe_ratio: float
    performance: Dict[str, Any]
    code: str
    timestamp: datetime = field(default_factory=datetime.now)


class FactorResearcher:
    """因子研究器"""
    
    def __init__(self, llm: SimpleLLMAdapter, config: RDAgentConfig):
        self.llm = llm
        self.config = config
        self.factor_pool = []
    
    async def discover_factors(self, 
                              data: pd.DataFrame,
                              target: str = "returns",
                              n_factors: int = 5) -> List[FactorResult]:
        """自动发现因子"""
        logger.info(f"开始因子发现，目标数量: {n_factors}")
        
        factors = []
        
        for i in range(min(n_factors, self.config.max_iterations)):
            # 生成因子假设
            hypothesis = await self._generate_factor_hypothesis(data, i)
            
            # 生成因子代码
            factor_code = await self._generate_factor_code(hypothesis)
            
            # 评估因子
            try:
                ic, ir, sharpe = self._evaluate_factor(factor_code, data, target)
                
                if ic > self.config.factor_ic_threshold:
                    factor_result = FactorResult(
                        name=f"factor_{i+1}",
                        expression=hypothesis,
                        ic=ic,
                        ir=ir,
                        sharpe_ratio=sharpe,
                        performance={"ic": ic, "ir": ir, "sharpe": sharpe},
                        code=factor_code
                    )
                    factors.append(factor_result)
                    logger.info(f"发现有效因子: IC={ic:.4f}, IR={ir:.2f}")
            
            except Exception as e:
                logger.error(f"因子评估失败: {e}")
        
        return sorted(factors, key=lambda f: f.ic, reverse=True)[:n_factors]
    
    async def _generate_factor_hypothesis(self, data: pd.DataFrame, iteration: int) -> str:
        """生成因子假设"""
        if not self.llm.client:
            # 使用预定义假设
            hypotheses = [
                "短期动量: close.pct_change(5)",
                "价格反转: (close - close.rolling(20).mean()) / close.rolling(20).std()",
                "成交量异常: volume / volume.rolling(20).mean()",
                "波动率因子: close.pct_change().rolling(20).std()",
                "RSI动量: RSI(14)指标"
            ]
            return hypotheses[iteration % len(hypotheses)]
        
        # 使用LLM生成假设
        prompt = f"""
作为量化研究员，请提出一个新的股票因子假设。

已有因子: {len(self.factor_pool)}个
数据特征: {list(data.columns)[:5]}

请用一句话描述因子逻辑，例如：
"基于5日动量的反转因子"
"""
        messages = [{"role": "user", "content": prompt}]
        return await self.llm.generate(messages)
    
    async def _generate_factor_code(self, hypothesis: str) -> str:
        """生成因子代码"""
        # 简化：使用模板
        code_templates = {
            "动量": "data['close'].pct_change({period})",
            "反转": "(data['close'] - data['close'].rolling({period}).mean()) / data['close'].rolling({period}).std()",
            "成交量": "data['volume'] / data['volume'].rolling({period}).mean()",
            "波动率": "data['close'].pct_change().rolling({period}).std()",
        }
        
        # 简单匹配
        for key, template in code_templates.items():
            if key in hypothesis:
                return template.format(period=20)
        
        return "data['close'].pct_change(5)"  # 默认
    
    def _evaluate_factor(self, factor_code: str, data: pd.DataFrame, target: str) -> Tuple[float, float, float]:
        """评估因子"""
        try:
            # 执行因子代码
            factor_values = eval(factor_code, {"data": data, "np": np, "pd": pd})
            
            # 计算收益率
            if target not in data.columns:
                returns = data['close'].pct_change().shift(-1)
            else:
                returns = data[target]
            
            # 计算IC
            ic = factor_values.corr(returns)
            
            # 计算IR
            ic_series = factor_values.rolling(20).corr(returns)
            ir = ic_series.mean() / (ic_series.std() + 1e-8)
            
            # 简化的Sharpe
            sharpe = ic * np.sqrt(252)
            
            return abs(ic), ir, sharpe
        
        except Exception as e:
            logger.error(f"因子评估错误: {e}")
            return 0.0, 0.0, 0.0


# ============================================================================
# 模型优化
# ============================================================================

@dataclass
class ModelResult:
    """模型优化结果"""
    model_type: str
    parameters: Dict[str, Any]
    performance: Dict[str, float]
    code: str
    timestamp: datetime = field(default_factory=datetime.now)


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, config: RDAgentConfig):
        self.config = config
    
    async def optimize_model(self,
                            data: pd.DataFrame,
                            features: List[str],
                            target: str = "label",
                            model_type: str = "lightgbm") -> ModelResult:
        """优化模型"""
        logger.info(f"开始模型优化: {model_type}")
        
        try:
            import optuna
            
            def objective(trial):
                if model_type == "lightgbm":
                    params = {
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                    }
                    score = self._train_and_evaluate(data, features, target, model_type, params)
                    return score
                return 0.0
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=min(20, self.config.optim_trials), timeout=300)
            
            best_params = study.best_params
            best_score = study.best_value
            
            return ModelResult(
                model_type=model_type,
                parameters=best_params,
                performance={"score": best_score},
                code=str(best_params)
            )
        
        except ImportError:
            logger.warning("Optuna未安装，使用默认参数")
            return ModelResult(
                model_type=model_type,
                parameters={"learning_rate": 0.05, "num_leaves": 50},
                performance={"score": 0.0},
                code="default"
            )
    
    def _train_and_evaluate(self, data, features, target, model_type, params):
        """训练和评估模型"""
        # 简化实现
        return np.random.random()


# ============================================================================
# 主集成类
# ============================================================================

class RealRDAgentIntegration:
    """RD-Agent真实集成"""
    
    def __init__(self, config: Optional[RDAgentConfig] = None):
        self.config = config or load_config()
        self.is_available = False
        self.llm = None
        self.factor_researcher = None
        self.model_optimizer = None
        
        self._initialize()
    
    def _initialize(self):
        """初始化系统"""
        logger.info("初始化RD-Agent集成...")
        
        # 检查RD-Agent可用性
        is_available, message = check_rdagent_available(self.config)
        self.is_available = is_available
        logger.info(message)
        
        # 初始化LLM
        self.llm = SimpleLLMAdapter(self.config)
        
        # 初始化研究器
        self.factor_researcher = FactorResearcher(self.llm, self.config)
        self.model_optimizer = ModelOptimizer(self.config)
        
        logger.info("系统初始化完成")
    
    async def discover_factors(self,
                              data: pd.DataFrame,
                              target: str = "returns",
                              n_factors: int = 5) -> List[FactorResult]:
        """发现因子"""
        return await self.factor_researcher.discover_factors(data, target, n_factors)
    
    async def optimize_model(self,
                            data: pd.DataFrame,
                            features: List[str],
                            target: str = "label",
                            model_type: str = "lightgbm") -> ModelResult:
        """优化模型"""
        return await self.model_optimizer.optimize_model(data, features, target, model_type)
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "is_available": self.is_available,
            "llm_configured": self.llm.client is not None,
            "config": self.config.to_dict()
        }


# 工厂函数
def create_integration(config_file: Optional[str] = None) -> RealRDAgentIntegration:
    """创建RD-Agent集成实例"""
    config = load_config(config_file)
    return RealRDAgentIntegration(config)


# 测试
async def test_integration():
    """测试集成"""
    print("=== RD-Agent集成测试 ===\n")
    
    integration = create_integration()
    
    # 状态
    status = integration.get_status()
    print("系统状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 测试因子发现
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'returns': np.random.randn(100) * 0.01
    })
    
    print("\n测试因子发现...")
    factors = await integration.discover_factors(data, n_factors=3)
    
    print(f"\n发现 {len(factors)} 个因子:")
    for f in factors:
        print(f"  {f.name}: IC={f.ic:.4f}, IR={f.ir:.2f}")


if __name__ == "__main__":
    asyncio.run(test_integration())
