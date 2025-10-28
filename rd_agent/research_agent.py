"""
RD-Agent智能研究系统集成
实现自动化因子挖掘、策略研究、模型优化等功能
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import logging
from pathlib import Path
import ast
import inspect
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib

# 机器学习相关
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler

# 大语言模型相关
import openai
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResearchHypothesis:
    """研究假设"""
    id: str
    title: str
    description: str
    category: str  # factor/strategy/model
    confidence: float
    created_at: datetime
    status: str = "pending"  # pending/testing/validated/rejected
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorDefinition:
    """因子定义"""
    name: str
    expression: str
    description: str
    category: str
    parameters: Dict[str, Any]
    performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyTemplate:
    """策略模板"""
    name: str
    description: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, Any]
    parameters: Dict[str, Any]
    backtest_results: Dict[str, Any] = field(default_factory=dict)


class RDAgent:
    """RD-Agent主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RD-Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.hypothesis_generator = HypothesisGenerator(config)
        self.code_generator = CodeGenerator(config)
        self.execution_engine = ExecutionEngine(config)
        self.feedback_evaluator = FeedbackEvaluator(config)
        self.knowledge_base = KnowledgeBase(config)
        
        # 执行器
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # 研究历史
        self.research_history = []
        
    async def research_pipeline(self, 
                               research_topic: str,
                               data: pd.DataFrame,
                               max_iterations: int = 10) -> Dict[str, Any]:
        """
        完整的研究流程
        
        Args:
            research_topic: 研究主题
            data: 历史数据
            max_iterations: 最大迭代次数
            
        Returns:
            研究结果
        """
        results = {
            "topic": research_topic,
            "hypotheses": [],
            "factors": [],
            "strategies": [],
            "models": [],
            "best_solution": None
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Research iteration {iteration + 1}/{max_iterations}")
            
            # 1. 生成假设
            hypotheses = await self.hypothesis_generator.generate(
                research_topic, data, self.knowledge_base
            )
            results["hypotheses"].extend(hypotheses)
            
            # 2. 对每个假设进行研究
            for hypothesis in hypotheses:
                if hypothesis.status != "pending":
                    continue
                
                # 生成代码
                code = await self.code_generator.generate_code(hypothesis)
                
                # 执行测试
                test_results = await self.execution_engine.execute(
                    code, data, hypothesis
                )
                
                # 评估反馈
                evaluation = await self.feedback_evaluator.evaluate(
                    hypothesis, test_results, data
                )
                
                # 更新假设状态
                hypothesis.status = evaluation["status"]
                hypothesis.results = evaluation
                
                # 保存有效的因子/策略
                if hypothesis.status == "validated":
                    if hypothesis.category == "factor":
                        results["factors"].append(evaluation["factor"])
                    elif hypothesis.category == "strategy":
                        results["strategies"].append(evaluation["strategy"])
                    elif hypothesis.category == "model":
                        results["models"].append(evaluation["model"])
                
                # 更新知识库
                self.knowledge_base.update(hypothesis, evaluation)
            
            # 3. 检查是否找到满意的解决方案
            if self._check_convergence(results):
                logger.info(f"Research converged at iteration {iteration + 1}")
                break
        
        # 4. 选择最佳解决方案
        results["best_solution"] = self._select_best_solution(results)
        
        # 保存研究历史
        self.research_history.append(results)
        
        return results
    
    async def discover_factors(self, 
                              data: pd.DataFrame,
                              target: str = "returns",
                              n_factors: int = 10) -> List[FactorDefinition]:
        """
        自动发现因子
        
        Args:
            data: 历史数据
            target: 目标变量
            n_factors: 要发现的因子数量
            
        Returns:
            因子列表
        """
        factors = []
        
        # 生成因子假设
        factor_hypotheses = await self.hypothesis_generator.generate_factor_hypotheses(
            data, target, n_factors * 2  # 生成更多以供筛选
        )
        # 测试每个因子
        for hypothesis in factor_hypotheses:
            # 生成因子计算代码
            factor_code = await self.code_generator.generate_factor_code(hypothesis)
            
            # 计算因子值
            factor_values = await self.execution_engine.calculate_factor(
                factor_code, data
            )
            
            # 评估因子有效性
            evaluation = await self.feedback_evaluator.evaluate_factor(
                factor_values, data[target]
            )
                
            if evaluation["is_valid"]:
                factor = FactorDefinition(
                    name=hypothesis.title,
                    expression=factor_code,
                    description=hypothesis.description,
                    category="technical",  # 可以更细分
                    parameters={},
                    performance=evaluation["metrics"]
                )
                factors.append(factor)
                
                if len(factors) >= n_factors:
                    break
        
        return factors
    
    async def optimize_strategy(self,
                               strategy: StrategyTemplate,
                               data: pd.DataFrame,
                               n_trials: int = 100) -> StrategyTemplate:
        """
        优化策略参数
        
        Args:
            strategy: 策略模板
            data: 历史数据
            n_trials: 优化试验次数
            
        Returns:
            优化后的策略
        """
        # 使用Optuna进行超参数优化
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        def objective(trial):
            # 采样参数
            params = {}
            for param_name, param_config in strategy.parameters.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["min"],
                        param_config["max"]
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["min"],
                        param_config["max"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
            
            # 回测策略
            backtest_results = self._backtest_strategy(strategy, data, params)
            
            # 返回优化目标（如夏普比率）
            return backtest_results.get("sharpe_ratio", 0)
        
        # 运行优化
        study.optimize(objective, n_trials=n_trials)
        
        # 更新策略参数为最优值
        best_params = study.best_params
        strategy.parameters.update(best_params)
        strategy.backtest_results = self._backtest_strategy(strategy, data, best_params)
        
        return strategy
    
    def _backtest_strategy(self,
                          strategy: StrategyTemplate,
                          data: pd.DataFrame,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """回测策略"""
        # 简化的回测逻辑
        returns = []
        positions = []
        
        for i in range(len(data)):
            # 检查入场条件
            if self._check_conditions(strategy.entry_conditions, data.iloc[i], params):
                positions.append(1)
            # 检查出场条件
            elif self._check_conditions(strategy.exit_conditions, data.iloc[i], params):
                positions.append(0)
            else:
                positions.append(positions[-1] if positions else 0)
            
            # 计算收益
            if i > 0 and positions[i-1] > 0:
                returns.append(data.iloc[i]["returns"])
            else:
                returns.append(0)
        
        # 计算评估指标
        returns_series = pd.Series(returns)
        total_return = (1 + returns_series).prod() - 1
        sharpe_ratio = returns_series.mean() / (returns_series.std() + 1e-8) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(returns_series)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": (returns_series > 0).sum() / (returns_series != 0).sum()
        }
    
    def _check_conditions(self, conditions: List[str], data: pd.Series, params: Dict) -> bool:
        """检查条件是否满足"""
        for condition in conditions:
            # 简单的条件解析（实际应该更复杂）
            if not eval(condition, {"data": data, "params": params}):
                return False
        return True
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _check_convergence(self, results: Dict[str, Any]) -> bool:
        """检查是否收敛"""
        # 简单的收敛判断
        if len(results["factors"]) >= 5 or len(results["strategies"]) >= 3:
            return True
        return False
    
    def _select_best_solution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """选择最佳解决方案"""
        best_solution = {
            "type": None,
            "solution": None,
            "performance": {}
        }
        
        # 选择最佳因子
        if results["factors"]:
            best_factor = max(results["factors"], 
                            key=lambda f: f.performance.get("ic", 0))
            best_solution = {
                "type": "factor",
                "solution": best_factor,
                "performance": best_factor.performance
            }
        
        # 选择最佳策略
        if results["strategies"]:
            best_strategy = max(results["strategies"],
                              key=lambda s: s.backtest_results.get("sharpe_ratio", 0))
            if best_strategy.backtest_results.get("sharpe_ratio", 0) > \
               best_solution["performance"].get("sharpe_ratio", 0):
                best_solution = {
                    "type": "strategy",
                    "solution": best_strategy,
                    "performance": best_strategy.backtest_results
                }
        
        return best_solution


class HypothesisGenerator:
    """假设生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._init_llm()
        
    def _init_llm(self):
        """初始化大语言模型"""
        # 这里可以配置不同的LLM
        return None  # 示例中简化处理
    
    async def generate(self,
                       topic: str,
                       data: pd.DataFrame,
                       knowledge_base: 'KnowledgeBase') -> List[ResearchHypothesis]:
        """生成研究假设"""
        hypotheses = []
        
        # 基于市场特征生成假设
        market_features = self._analyze_market_features(data)
        
        # 生成技术分析假设
        tech_hypotheses = self._generate_technical_hypotheses(market_features)
        hypotheses.extend(tech_hypotheses)
        
        # 生成基本面假设
        fundamental_hypotheses = self._generate_fundamental_hypotheses(market_features)
        hypotheses.extend(fundamental_hypotheses)
        
        # 基于知识库生成假设
        kb_hypotheses = self._generate_from_knowledge_base(knowledge_base, market_features)
        hypotheses.extend(kb_hypotheses)
        
        return hypotheses
    
    async def generate_factor_hypotheses(self,
                                        data: pd.DataFrame,
                                        target: str,
                                        n_hypotheses: int) -> List[ResearchHypothesis]:
        """生成因子假设"""
        hypotheses = []
        
        # 动量类因子
        hypotheses.append(ResearchHypothesis(
            id=f"factor_{len(hypotheses)}",
            title="短期动量因子",
            description="基于过去5日收益率的动量因子",
            category="factor",
            confidence=0.7,
            created_at=datetime.now()
        ))
        
        # 反转类因子
        hypotheses.append(ResearchHypothesis(
            id=f"factor_{len(hypotheses)}",
            title="均值回归因子",
            description="价格偏离20日均线的程度",
            category="factor",
            confidence=0.6,
            created_at=datetime.now()
        ))
        
        # 波动率因子
        hypotheses.append(ResearchHypothesis(
            id=f"factor_{len(hypotheses)}",
            title="波动率因子",
            description="20日滚动波动率",
            category="factor",
            confidence=0.8,
            created_at=datetime.now()
        ))
        
        # 成交量因子
        hypotheses.append(ResearchHypothesis(
            id=f"factor_{len(hypotheses)}",
            title="量价背离因子",
            description="价格与成交量的背离程度",
            category="factor",
            confidence=0.65,
            created_at=datetime.now()
        ))
        
        # 技术指标组合因子
        hypotheses.append(ResearchHypothesis(
            id=f"factor_{len(hypotheses)}",
            title="RSI-MACD组合因子",
            description="RSI和MACD的组合信号",
            category="factor",
            confidence=0.75,
            created_at=datetime.now()
        ))
        
        return hypotheses[:n_hypotheses]
    
    def _analyze_market_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析市场特征"""
        features = {}
        
        if "close" in data.columns:
            features["volatility"] = data["close"].pct_change().std()
            features["trend"] = (data["close"].iloc[-1] - data["close"].iloc[0]) / data["close"].iloc[0]
            features["mean_volume"] = data.get("volume", pd.Series()).mean()
        
        return features
    
    def _generate_technical_hypotheses(self, features: Dict[str, Any]) -> List[ResearchHypothesis]:
        """生成技术分析假设"""
        hypotheses = []
        
        # 根据波动率生成假设
        if features.get("volatility", 0) > 0.02:
            hypotheses.append(ResearchHypothesis(
                id=f"tech_{len(hypotheses)}",
                title="高波动率交易策略",
                description="在高波动环境下使用均值回归策略",
                category="strategy",
                confidence=0.7,
                created_at=datetime.now()
            ))
        
        # 根据趋势生成假设
        if abs(features.get("trend", 0)) > 0.1:
            hypotheses.append(ResearchHypothesis(
                id=f"tech_{len(hypotheses)}",
                title="趋势跟踪策略",
                description="使用移动平均线交叉信号进行趋势跟踪",
                category="strategy",
                confidence=0.75,
                created_at=datetime.now()
            ))
        
        return hypotheses
    
    def _generate_fundamental_hypotheses(self, features: Dict[str, Any]) -> List[ResearchHypothesis]:
        """生成基本面假设"""
        hypotheses = []
        
        # 这里可以加入更多基本面逻辑
        hypotheses.append(ResearchHypothesis(
            id=f"fund_0",
            title="价值投资因子",
            description="基于市盈率和市净率的价值因子",
            category="factor",
            confidence=0.8,
            created_at=datetime.now()
        ))
        
        return hypotheses
    
    def _generate_from_knowledge_base(self,
                                     knowledge_base: 'KnowledgeBase',
                                     features: Dict[str, Any]) -> List[ResearchHypothesis]:
        """基于知识库生成假设"""
        hypotheses = []
        
        # 查询相似的成功案例
        similar_cases = knowledge_base.find_similar_cases(features)
        
        for case in similar_cases[:3]:  # 只取前3个
            hypothesis = ResearchHypothesis(
                id=f"kb_{len(hypotheses)}",
                title=f"基于历史案例的{case['type']}",
                description=f"参考历史成功案例：{case['description']}",
                category=case['category'],
                confidence=case['success_rate'],
                created_at=datetime.now()
            )
            hypotheses.append(hypothesis)
        
        return hypotheses


class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """加载代码模板"""
        return {
            "factor": """
def calculate_factor(data):
    '''
    {description}
    '''
    {implementation}
    return factor_values
""",
            "strategy": """
class Strategy:
    def __init__(self, params):
        self.params = params
    
    def generate_signals(self, data):
        '''
        {description}
        '''
        signals = []
        {implementation}
        return signals
""",
            "model": """
class Model:
    def __init__(self, params):
        self.params = params
        
    def fit(self, X, y):
        {fit_implementation}
        
    def predict(self, X):
        {predict_implementation}
        return predictions
"""
        }
    
    async def generate_code(self, hypothesis: ResearchHypothesis) -> str:
        """生成代码"""
        if hypothesis.category == "factor":
            return await self.generate_factor_code(hypothesis)
        elif hypothesis.category == "strategy":
            return await self.generate_strategy_code(hypothesis)
        elif hypothesis.category == "model":
            return await self.generate_model_code(hypothesis)
        else:
            raise ValueError(f"Unknown category: {hypothesis.category}")
    
    async def generate_factor_code(self, hypothesis: ResearchHypothesis) -> str:
        """生成因子代码"""
        implementations = {
            "短期动量因子": "factor_values = data['close'].pct_change(5)",
            "均值回归因子": "ma20 = data['close'].rolling(20).mean()\n    factor_values = (data['close'] - ma20) / ma20",
            "波动率因子": "factor_values = data['close'].pct_change().rolling(20).std()",
            "量价背离因子": "price_change = data['close'].pct_change()\n    volume_change = data['volume'].pct_change()\n    factor_values = price_change - volume_change",
            "RSI-MACD组合因子": """
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    
    # 组合
    factor_values = (rsi - 50) / 50 + (macd - signal) / data['close']"""
        }
        
        implementation = implementations.get(
            hypothesis.title,
            "factor_values = data['close'].pct_change()"  # 默认实现
        )
        
        code = self.templates["factor"].format(
            description=hypothesis.description,
            implementation=implementation
        )
        
        return code
    
    async def generate_strategy_code(self, hypothesis: ResearchHypothesis) -> str:
        """生成策略代码"""
        implementations = {
            "高波动率交易策略": """
        for i in range(len(data)):
            if i < 20:
                signals.append(0)
                continue
            
            # 计算波动率
            volatility = data.iloc[i-20:i]['close'].pct_change().std()
            
            # 计算偏离度
            ma20 = data.iloc[i-20:i]['close'].mean()
            deviation = (data.iloc[i]['close'] - ma20) / ma20
            
            # 高波动率下的均值回归
            if volatility > self.params.get('vol_threshold', 0.02):
                if deviation < -self.params.get('entry_threshold', 0.03):
                    signals.append(1)  # 买入
                elif deviation > self.params.get('exit_threshold', 0.03):
                    signals.append(-1)  # 卖出
                else:
                    signals.append(0)
            else:
                signals.append(0)""",
                
            "趋势跟踪策略": """
        for i in range(len(data)):
            if i < 50:
                signals.append(0)
                continue
            
            # 计算移动平均线
            ma20 = data.iloc[i-20:i]['close'].mean()
            ma50 = data.iloc[i-50:i]['close'].mean()
            
            # 交叉信号
            prev_ma20 = data.iloc[i-21:i-1]['close'].mean()
            prev_ma50 = data.iloc[i-51:i-1]['close'].mean()
            
            if ma20 > ma50 and prev_ma20 <= prev_ma50:
                signals.append(1)  # 金叉买入
            elif ma20 < ma50 and prev_ma20 >= prev_ma50:
                signals.append(-1)  # 死叉卖出
            else:
                signals.append(0)"""
        }
        
        implementation = implementations.get(
            hypothesis.title,
            "signals = [0] * len(data)  # 默认无信号"
        )
        
        code = self.templates["strategy"].format(
            description=hypothesis.description,
            implementation=implementation
        )
        
        return code
    
    async def generate_model_code(self, hypothesis: ResearchHypothesis) -> str:
        """生成模型代码"""
        # 简化的模型代码生成
        code = self.templates["model"].format(
            fit_implementation="# Model fitting logic here",
            predict_implementation="# Prediction logic here"
        )
        return code


class ExecutionEngine:
    """执行引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sandbox = self._create_sandbox()
    
    def _create_sandbox(self) -> Dict[str, Any]:
        """创建安全的执行沙箱"""
        import numpy as np
        import pandas as pd
        
        sandbox = {
            'np': np,
            'pd': pd,
            'datetime': datetime,
            '__builtins__': {
                'len': len,
                'range': range,
                'min': min,
                'max': max,
                'abs': abs,
                'sum': sum,
                'print': print
            }
        }
        return sandbox
    
    async def execute(self,
                     code: str,
                     data: pd.DataFrame,
                     hypothesis: ResearchHypothesis) -> Dict[str, Any]:
        """执行代码"""
        try:
            # 准备执行环境
            exec_globals = self.sandbox.copy()
            exec_globals['data'] = data
            
            # 执行代码
            exec(code, exec_globals)
            
            # 获取结果
            if hypothesis.category == "factor":
                result = exec_globals.get('calculate_factor')(data)
                return {"factor_values": result, "success": True}
            elif hypothesis.category == "strategy":
                # 创建策略实例
                strategy_class = exec_globals.get('Strategy')
                if strategy_class:
                    strategy = strategy_class({'vol_threshold': 0.02})
                    signals = strategy.generate_signals(data)
                    return {"signals": signals, "success": True}
            elif hypothesis.category == "model":
                model_class = exec_globals.get('Model')
                if model_class:
                    return {"model": model_class, "success": True}
            
            return {"success": False, "error": "No result found"}
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def calculate_factor(self, factor_code: str, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        try:
            exec_globals = self.sandbox.copy()
            exec_globals['data'] = data
            
            exec(factor_code, exec_globals)
            
            calculate_func = exec_globals.get('calculate_factor')
            if calculate_func:
                return calculate_func(data)
            
            return pd.Series()
            
        except Exception as e:
            logger.error(f"Factor calculation error: {e}")
            return pd.Series()


class FeedbackEvaluator:
    """反馈评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_calculator = MetricsCalculator()
    
    async def evaluate(self,
                      hypothesis: ResearchHypothesis,
                      test_results: Dict[str, Any],
                      data: pd.DataFrame) -> Dict[str, Any]:
        """评估测试结果"""
        evaluation = {
            "status": "rejected",
            "metrics": {},
            "feedback": ""
        }
        
        if not test_results.get("success"):
            evaluation["feedback"] = f"Execution failed: {test_results.get('error')}"
            return evaluation
        
        if hypothesis.category == "factor":
            factor_eval = await self.evaluate_factor(
                test_results.get("factor_values"),
                data.get("returns", data["close"].pct_change())
            )
            evaluation.update(factor_eval)
            
        elif hypothesis.category == "strategy":
            strategy_eval = await self.evaluate_strategy(
                test_results.get("signals"),
                data
            )
            evaluation.update(strategy_eval)
            
        elif hypothesis.category == "model":
            model_eval = await self.evaluate_model(
                test_results.get("model"),
                data
            )
            evaluation.update(model_eval)
        
        return evaluation
    
    async def evaluate_factor(self,
                             factor_values: pd.Series,
                             returns: pd.Series) -> Dict[str, Any]:
        """评估因子"""
        if factor_values is None or factor_values.empty:
            return {"status": "rejected", "feedback": "Empty factor values"}
        
        # 计算IC（信息系数）
        ic = factor_values.corr(returns.shift(-1))
        
        # 计算IR（信息比率）
        ic_series = factor_values.rolling(20).corr(returns.shift(-1))
        ir = ic_series.mean() / (ic_series.std() + 1e-8)
        
        # 计算因子收益
        factor_returns = self._calculate_factor_returns(factor_values, returns)
        
        metrics = {
            "ic": ic,
            "ir": ir,
            "factor_returns": factor_returns.mean(),
            "factor_sharpe": factor_returns.mean() / (factor_returns.std() + 1e-8) * np.sqrt(252)
        }
        
        # 判断是否有效
        is_valid = abs(ic) > 0.03 and abs(ir) > 0.5
        
        return {
            "status": "validated" if is_valid else "rejected",
            "metrics": metrics,
            "factor": FactorDefinition(
                name=f"factor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                expression=str(factor_values),
                description="Auto-generated factor",
                category="technical",
                parameters={},
                performance=metrics
            ) if is_valid else None,
            "feedback": f"IC={ic:.4f}, IR={ir:.4f}"
        }
    
    async def evaluate_strategy(self,
                               signals: List[int],
                               data: pd.DataFrame) -> Dict[str, Any]:
        """评估策略"""
        if not signals:
            return {"status": "rejected", "feedback": "No signals generated"}
        
        # 计算策略收益
        returns = data["close"].pct_change()
        strategy_returns = pd.Series([
            returns.iloc[i] if i > 0 and signals[i-1] == 1 else 0
            for i in range(len(returns))
        ])
        
        # 计算评估指标
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(strategy_returns)
        win_rate = (strategy_returns > 0).sum() / (strategy_returns != 0).sum()
        
        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }
        
        # 判断是否有效
        is_valid = sharpe_ratio > 0.5 and max_drawdown > -0.2
        
        return {
            "status": "validated" if is_valid else "rejected",
            "metrics": metrics,
            "strategy": StrategyTemplate(
                name=f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="Auto-generated strategy",
                entry_conditions=[],
                exit_conditions=[],
                risk_management={},
                parameters={},
                backtest_results=metrics
            ) if is_valid else None,
            "feedback": f"Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2%}"
        }
    
    async def evaluate_model(self,
                           model_class: type,
                           data: pd.DataFrame) -> Dict[str, Any]:
        """评估模型"""
        # 简化的模型评估
        return {
            "status": "validated",
            "metrics": {},
            "model": model_class,
            "feedback": "Model validated"
        }
    
    def _calculate_factor_returns(self,
                                 factor_values: pd.Series,
                                 returns: pd.Series) -> pd.Series:
        """计算因子收益"""
        # 根据因子值分组
        quantiles = pd.qcut(factor_values, q=5, labels=False)
        
        # 计算多空组合收益
        top_returns = returns[quantiles == 4]
        bottom_returns = returns[quantiles == 0]
        
        factor_returns = top_returns.mean() - bottom_returns.mean()
        
        return pd.Series([factor_returns] * len(returns))
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class MetricsCalculator:
    """指标计算器"""
    
    def calculate_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算信息系数"""
        return factor.corr(returns.shift(-1))
    
    def calculate_ir(self, factor: pd.Series, returns: pd.Series, window: int = 20) -> float:
        """计算信息比率"""
        ic_series = factor.rolling(window).corr(returns.shift(-1))
        return ic_series.mean() / (ic_series.std() + 1e-8)
    
    def calculate_sharpe(self, returns: pd.Series, risk_free: float = 0) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free
        return excess_returns.mean() / (excess_returns.std() + 1e-8) * np.sqrt(252)


class KnowledgeBase:
    """知识库"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get("storage_path", "./knowledge_base"))
        self.storage_path.mkdir(exist_ok=True)
        self.cases = self._load_cases()
    
    def _load_cases(self) -> List[Dict[str, Any]]:
        """加载历史案例"""
        cases_file = self.storage_path / "cases.json"
        if cases_file.exists():
            with open(cases_file, 'r') as f:
                return json.load(f)
        return []
    
    def update(self, hypothesis: ResearchHypothesis, evaluation: Dict[str, Any]):
        """更新知识库"""
        case = {
            "id": hypothesis.id,
            "title": hypothesis.title,
            "description": hypothesis.description,
            "category": hypothesis.category,
            "type": hypothesis.category,
            "status": hypothesis.status,
            "metrics": evaluation.get("metrics", {}),
            "success_rate": evaluation.get("metrics", {}).get("sharpe_ratio", 0),
            "created_at": hypothesis.created_at.isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # 更新或添加案例
        existing_index = next((i for i, c in enumerate(self.cases) 
                              if c["id"] == hypothesis.id), None)
        if existing_index is not None:
            self.cases[existing_index] = case
        else:
            self.cases.append(case)
        
        # 保存到文件
        self._save_cases()
    
    def find_similar_cases(self, features: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
        """查找相似案例"""
        # 简化的相似度计算
        valid_cases = [c for c in self.cases if c["status"] == "validated"]
        
        # 按成功率排序
        valid_cases.sort(key=lambda c: c.get("success_rate", 0), reverse=True)
        
        return valid_cases[:n]
    
    def _save_cases(self):
        """保存案例"""
        cases_file = self.storage_path / "cases.json"
        with open(cases_file, 'w') as f:
            json.dump(self.cases, f, indent=2, default=str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_cases = len(self.cases)
        validated_cases = sum(1 for c in self.cases if c["status"] == "validated")
        
        stats = {
            "total_cases": total_cases,
            "validated_cases": validated_cases,
            "success_rate": validated_cases / total_cases if total_cases > 0 else 0,
            "by_category": {},
            "best_performers": []
        }
        
        # 按类别统计
        for category in ["factor", "strategy", "model"]:
            category_cases = [c for c in self.cases if c["category"] == category]
            stats["by_category"][category] = {
                "total": len(category_cases),
                "validated": sum(1 for c in category_cases if c["status"] == "validated")
            }
        
        # 最佳表现者
        validated = [c for c in self.cases if c["status"] == "validated"]
        validated.sort(key=lambda c: c.get("success_rate", 0), reverse=True)
        stats["best_performers"] = validated[:5]
        
        return stats


if __name__ == "__main__":
    # 测试代码
    async def test():
        config = {
            "storage_path": "./rd_agent_knowledge"
        }
        
        rd_agent = RDAgent(config)
        
        # 生成测试数据
        dates = pd.date_range('2022-01-01', '2023-12-31')
        data = pd.DataFrame({
            'date': dates,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'returns': np.random.randn(len(dates)) * 0.02
        })
        
        # 发现因子
        factors = await rd_agent.discover_factors(data, target="returns", n_factors=5)
        print(f"Discovered {len(factors)} factors")
        
        for factor in factors:
            print(f"- {factor.name}: IC={factor.performance.get('ic', 0):.4f}")
    
    # 运行测试
    # asyncio.run(test())