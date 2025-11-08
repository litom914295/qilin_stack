"""
RD-Agent兼容层包装器

任务: P0-1.5 + P0-1.6
功能: 保持原有RDAgent API,内部调用官方RD-Agent组件
集成: official_integration.py + research_agent.py (数据类型)
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

# 导入官方集成管理器
from .official_integration import (
    OfficialRDAgentManager,
    create_official_manager,
    OfficialIntegrationError,
    ConfigValidationError
)

# 导入自研数据类型(保持兼容)
from .research_agent import (
    ResearchHypothesis,
    FactorDefinition,
    StrategyTemplate
)

logger = logging.getLogger(__name__)


class ResultConversionError(Exception):
    """结果转换错误"""
    pass


class DataNotFoundError(Exception):
    """数据未找到错误"""
    pass


class _ConfigAdapter:
    """
    配置适配器: Dict → 官方配置
    
    职责: 将自研的Dict配置转换为官方RD-Agent需要的格式
    """
    
    @staticmethod
    def to_official_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换为官方配置格式
        
        Args:
            config: 自研配置字典
            
        Returns:
            官方配置字典
        """
        official_config = {}
        
        # LLM配置映射
        if "llm_model" in config:
            official_config["llm_model"] = config["llm_model"]
        elif "model" in config:
            official_config["llm_model"] = config["model"]
        
        if "llm_api_key" in config:
            official_config["llm_api_key"] = config["llm_api_key"]
        elif "api_key" in config:
            official_config["llm_api_key"] = config["api_key"]
        
        if "llm_provider" in config:
            official_config["llm_provider"] = config["llm_provider"]
        else:
            # 根据model推断provider
            model = official_config.get("llm_model", "")
            if "gpt" in model.lower():
                official_config["llm_provider"] = "openai"
            elif "claude" in model.lower():
                official_config["llm_provider"] = "anthropic"
        
        if "llm_base_url" in config:
            official_config["llm_base_url"] = config["llm_base_url"]
        
        if "llm_temperature" in config:
            official_config["llm_temperature"] = config["llm_temperature"]
        elif "temperature" in config:
            official_config["llm_temperature"] = config["temperature"]
        
        # 执行配置映射
        if "max_iterations" in config:
            official_config["max_iterations"] = config["max_iterations"]
        
        # 路径配置映射
        if "qlib_data_path" in config:
            official_config["qlib_data_path"] = config["qlib_data_path"]
        
        if "storage_path" in config:
            official_config["storage_path"] = config["storage_path"]
        
        return official_config
    
    @staticmethod
    def apply_to_environment(config: Dict[str, Any]):
        """
        应用配置到环境变量(官方RD-Agent从环境变量读取)
        
        Args:
            config: 配置字典
        """
        # LLM配置
        if "llm_provider" in config:
            os.environ["LLM_PROVIDER"] = config["llm_provider"]
        
        if "llm_model" in config:
            os.environ["LLM_MODEL"] = config["llm_model"]
        
        if "llm_api_key" in config:
            provider = config.get("llm_provider", "openai").lower()
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = config["llm_api_key"]
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = config["llm_api_key"]
        
        if "llm_base_url" in config:
            os.environ["OPENAI_BASE_URL"] = config["llm_base_url"]
        
        if "llm_temperature" in config:
            os.environ["LLM_TEMPERATURE"] = str(config["llm_temperature"])


class _ResultAdapter:
    """
    结果适配器: 官方格式 → 自研格式
    
    职责: 将官方RD-Agent的Trace/Experiment转换为自研的Dict/FactorDefinition
    """
    
    @staticmethod
    def trace_to_results_dict(trace, topic: str) -> Dict[str, Any]:
        """
        Trace → research_pipeline返回的Dict
        
        Args:
            trace: 官方Trace对象
            topic: 研究主题
            
        Returns:
            自研格式的结果字典
        """
        results = {
            "topic": topic,
            "hypotheses": [],
            "factors": [],
            "strategies": [],
            "models": [],
            "best_solution": None
        }
        
        try:
            # 转换所有历史实验
            for exp, feedback in trace.hist:
                # 转换为ResearchHypothesis
                hypo = ResearchHypothesis(
                    id=f"hypo_{id(exp)}",
                    title=str(exp.hypothesis.hypothesis)[:50] if hasattr(exp.hypothesis, 'hypothesis') else "Unknown",
                    description=str(exp.hypothesis.hypothesis) if hasattr(exp.hypothesis, 'hypothesis') else "",
                    category="factor",
                    confidence=0.8 if feedback.decision else 0.3,
                    created_at=datetime.now(),
                    status="validated" if feedback.decision else "rejected",
                    results={"decision": feedback.decision}
                )
                results["hypotheses"].append(hypo)
                
                # 如果实验成功,转换为Factor
                if feedback.decision and hasattr(exp, 'result') and exp.result is not None:
                    try:
                        factor = _ResultAdapter.exp_to_factor(exp)
                        results["factors"].append(factor)
                    except Exception as e:
                        logger.warning(f"Failed to convert experiment to factor: {e}")
            
            # 选择最佳解决方案
            if results["factors"]:
                best_factor = results["factors"][-1]  # SOTA是最后一个
                results["best_solution"] = {
                    "type": "factor",
                    "solution": best_factor,
                    "performance": best_factor.performance
                }
            
        except Exception as e:
            raise ResultConversionError(
                f"Failed to convert Trace to results dict: {e}"
            ) from e
        
        return results
    
    @staticmethod
    def exp_to_factor(exp) -> FactorDefinition:
        """
        Experiment → FactorDefinition (增强鲁棒性版本)
        
        ✅ P0-4 修复:
        - 鲁棒地获取 workspace (多路径尝试)
        - 多文件名候选 (factor.py/code.py/main.py/implementation.py)
        - 多指标键名尝试 (IC/ic/information_coefficient)
        - 完整的错误日志
        
        Args:
            exp: 官方Experiment对象
            
        Returns:
            自研FactorDefinition对象
        
        Raises:
            ResultConversionError: 无法提取必需信息时
        """
        try:
            # ========== 1. 鲁棒地获取 workspace ==========
            workspace = None
            code_file_name = None
            
            # 尝试路径 1: sub_workspace_list[0]
            if hasattr(exp, 'sub_workspace_list') and exp.sub_workspace_list:
                workspace = exp.sub_workspace_list[0]
                logger.debug("Workspace found via sub_workspace_list[0]")
            # 尝试路径 2: workspace
            elif hasattr(exp, 'workspace') and exp.workspace is not None:
                workspace = exp.workspace
                logger.debug("Workspace found via workspace")
            # 尝试路径 3: sub_workspace (单数形式)
            elif hasattr(exp, 'sub_workspace') and exp.sub_workspace is not None:
                workspace = exp.sub_workspace
                logger.debug("Workspace found via sub_workspace")
            else:
                raise ResultConversionError(
                    "No workspace found in experiment. "
                    f"Available attributes: {dir(exp)}"
                )
            
            # ========== 2. 多文件名候选提取代码 ==========
            factor_code = ""
            file_candidates = ['factor.py', 'code.py', 'main.py', 'implementation.py', 'factor_code.py']
            
            file_dict = {}
            if hasattr(workspace, 'file_dict'):
                file_dict = workspace.file_dict
            elif hasattr(workspace, 'files'):
                file_dict = workspace.files
            elif isinstance(workspace, dict):
                file_dict = workspace.get('file_dict', workspace.get('files', {}))
            
            # 尝试每个候选文件名
            for filename in file_candidates:
                if filename in file_dict:
                    factor_code = file_dict[filename]
                    code_file_name = filename
                    logger.debug(f"Factor code found in: {filename}")
                    break
            
            # 如果都没找到,尝试获取第一个.py文件
            if not factor_code:
                py_files = {k: v for k, v in file_dict.items() if k.endswith('.py')}
                if py_files:
                    code_file_name = list(py_files.keys())[0]
                    factor_code = py_files[code_file_name]
                    logger.warning(
                        f"Standard factor files not found. Using first .py file: {code_file_name}"
                    )
                else:
                    logger.error(
                        f"No factor code found. Available files: {list(file_dict.keys())}"
                    )
                    # 不抛出异常,使用空代码
                    factor_code = "# Factor code not available"
                    code_file_name = "unknown.py"
            
            # ========== 3. 多指标键名尝试提取性能 ==========
            performance = {}
            
            if hasattr(exp, 'result') and exp.result is not None:
                result_data = exp.result
                
                # 处理 DataFrame 格式
                if isinstance(result_data, pd.DataFrame):
                    # 提取 IC (多键名候选)
                    ic_keys = ['IC', 'ic', 'information_coefficient', 'IC_mean', 'ic_mean']
                    for key in ic_keys:
                        if key in result_data.index:
                            try:
                                performance["ic"] = float(result_data.loc[key].iloc[0])
                                logger.debug(f"IC found via key: {key}")
                                break
                            except (IndexError, ValueError, TypeError) as e:
                                logger.warning(f"Failed to extract IC from key '{key}': {e}")
                                continue
                    
                    # 提取 IR (新增)
                    ir_keys = ['IR', 'ir', 'information_ratio', 'IC_IR', 'ic_ir']
                    for key in ir_keys:
                        if key in result_data.index:
                            try:
                                performance["ir"] = float(result_data.loc[key].iloc[0])
                                logger.debug(f"IR found via key: {key}")
                                break
                            except (IndexError, ValueError, TypeError) as e:
                                logger.warning(f"Failed to extract IR from key '{key}': {e}")
                                continue
                    
                    # 提取年化收益 (多键名候选)
                    annual_return_keys = [
                        "1day.excess_return_with_cost.annualized_return",
                        "annualized_return",
                        "annual_return",
                        "excess_return_with_cost.annualized_return"
                    ]
                    for key in annual_return_keys:
                        if key in result_data.index:
                            try:
                                performance["annual_return"] = float(result_data.loc[key].iloc[0])
                                logger.debug(f"Annual return found via key: {key}")
                                break
                            except (IndexError, ValueError, TypeError) as e:
                                logger.warning(f"Failed to extract annual_return from key '{key}': {e}")
                                continue
                    
                    # 提取最大回撤 (多键名候选)
                    max_dd_keys = [
                        "1day.excess_return_with_cost.max_drawdown",
                        "max_drawdown",
                        "maximum_drawdown",
                        "excess_return_with_cost.max_drawdown"
                    ]
                    for key in max_dd_keys:
                        if key in result_data.index:
                            try:
                                performance["max_drawdown"] = float(result_data.loc[key].iloc[0])
                                logger.debug(f"Max drawdown found via key: {key}")
                                break
                            except (IndexError, ValueError, TypeError) as e:
                                logger.warning(f"Failed to extract max_drawdown from key '{key}': {e}")
                                continue
                    
                    # 日志未找到的指标
                    if not performance.get('ic'):
                        logger.warning(
                            f"IC not found. Available metrics: {list(result_data.index)}"
                        )
                
                # 处理 dict 格式
                elif isinstance(result_data, dict):
                    # 直接尝试从 dict 提取
                    ic_keys = ['IC', 'ic', 'information_coefficient']
                    for key in ic_keys:
                        if key in result_data:
                            try:
                                performance["ic"] = float(result_data[key])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    ir_keys = ['IR', 'ir', 'information_ratio']
                    for key in ir_keys:
                        if key in result_data:
                            try:
                                performance["ir"] = float(result_data[key])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    if not performance.get('ic'):
                        logger.warning(
                            f"IC not found in dict. Available keys: {list(result_data.keys())}"
                        )
            
            # ========== 4. 提取元数据 ==========
            factor_name = f"factor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = ""
            
            # 多路径尝试提取 hypothesis
            if hasattr(exp, 'hypothesis'):
                hypo = exp.hypothesis
                if hasattr(hypo, 'hypothesis'):
                    description = str(hypo.hypothesis)
                elif hasattr(hypo, 'description'):
                    description = str(hypo.description)
                elif isinstance(hypo, str):
                    description = hypo
            
            # 提取版本信息 (新增)
            version = "unknown"
            if hasattr(workspace, 'version'):
                version = str(workspace.version)
            elif isinstance(workspace, dict) and 'version' in workspace:
                version = str(workspace['version'])
            
            # ========== 5. 创建 FactorDefinition ==========
            factor = FactorDefinition(
                name=factor_name,
                expression=factor_code,
                description=description,
                category="auto_generated",
                parameters={
                    'code_file': code_file_name,
                    'version': version
                },
                performance=performance
            )
            
            logger.info(
                f"Successfully converted experiment to factor: {factor_name} "
                f"(IC={performance.get('ic', 'N/A')}, file={code_file_name})"
            )
            
            return factor
            
        except ResultConversionError:
            # 直接重新抛出已知错误
            raise
        except Exception as e:
            raise ResultConversionError(
                f"Failed to convert Experiment to FactorDefinition: {e}. "
                f"Experiment attributes: {dir(exp)}"
            ) from e
    
    @staticmethod
    def experiments_to_factors(trace, n_factors: int = 10) -> List[FactorDefinition]:
        """
        提取前N个有效因子
        
        Args:
            trace: 官方Trace对象
            n_factors: 要提取的因子数量
            
        Returns:
            FactorDefinition列表
        """
        factors = []
        
        try:
            for exp, feedback in trace.hist:
                if feedback.decision and hasattr(exp, 'result') and exp.result is not None:
                    try:
                        factor = _ResultAdapter.exp_to_factor(exp)
                        factors.append(factor)
                        if len(factors) >= n_factors:
                            break
                    except Exception as e:
                        logger.warning(f"Failed to convert experiment to factor: {e}")
                        continue
        except Exception as e:
            raise ResultConversionError(
                f"Failed to extract factors from experiments: {e}"
            ) from e
        
        return factors


class RDAgentWrapper:
    """
    RD-Agent兼容层包装器
    
    职责:
    1. 保持原有RDAgent的API不变
    2. 内部调用官方RD-Agent组件
    3. 转换配置和结果格式
    4. 提供与自研版本相同的行为
    
    使用示例:
        # 创建Wrapper (与原RDAgent相同的API)
        config = {
            "llm_model": "gpt-4-turbo",
            "llm_api_key": "sk-xxx",
            "max_iterations": 10
        }
        agent = RDAgentWrapper(config)
        
        # 使用原有API
        results = await agent.research_pipeline(
            research_topic="A股动量因子研究",
            data=df,
            max_iterations=5
        )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RD-Agent包装器
        
        Args:
            config: 配置字典 (与原RDAgent相同的格式)
        """
        self.config = config
        
        # 转换配置
        official_config = _ConfigAdapter.to_official_config(config)
        
        # 应用环境变量
        _ConfigAdapter.apply_to_environment(official_config)
        
        # 创建官方管理器
        try:
            self._official_manager = create_official_manager(official_config)
            logger.info("RDAgentWrapper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RDAgentWrapper: {e}")
            raise
        
        # Phase 1.1: 初始化 FileStorage 日志 (新增)
        workspace_path = config.get('workspace_path', './logs/rdagent')
        try:
            from rd_agent.logging_integration import QilinRDAgentLogger
            self.qilin_logger = QilinRDAgentLogger(workspace_path)
            logger.info(f"✅ FileStorage logging enabled at {workspace_path}")
        except Exception as e:
            logger.warning(f"⚠️ FileStorage logging unavailable: {e}")
            self.qilin_logger = None
        
        # 保存研究历史 (兼容原API)
        self.research_history = []
    
    async def research_pipeline(self,
                               research_topic: str,
                               data: pd.DataFrame,
                               max_iterations: int = 10) -> Dict[str, Any]:
        """
        完整的研究流程 (保持原有API签名)
        
        Args:
            research_topic: 研究主题
            data: 历史数据 (暂未使用,因为官方RD-Agent使用Qlib数据)
            max_iterations: 最大迭代次数
            
        Returns:
            研究结果字典 (与原RDAgent相同的格式)
        """
        logger.info(f"Starting research pipeline: {research_topic}")
        
        try:
            # 1. 获取官方FactorLoop
            factor_loop = self._official_manager.get_factor_loop()
            
            # 2. 运行官方循环
            logger.info(f"Running FactorRDLoop for {max_iterations} iterations...")
            await factor_loop.run(loop_n=max_iterations)
            
            # 3. 转换结果格式
            results = _ResultAdapter.trace_to_results_dict(
                factor_loop.trace,
                topic=research_topic
            )
            
            # Phase 1.1: 记录实验到 FileStorage (新增)
            if self.qilin_logger:
                try:
                    for exp, feedback in factor_loop.trace.hist:
                        if feedback.decision:  # 只记录被采纳的实验
                            self.qilin_logger.log_experiment(exp, tag='limitup.factor')
                    
                    # 记录汇总指标
                    summary_metrics = {
                        'topic': research_topic,
                        'total_experiments': len(factor_loop.trace.hist),
                        'successful_factors': len(results['factors']),
                        'max_iterations': max_iterations
                    }
                    self.qilin_logger.log_metrics(summary_metrics, tag='limitup.summary')
                    logger.info("✅ Logged experiments to FileStorage")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log to FileStorage: {e}")
            
            # 保存到历史
            self.research_history.append(results)
            
            logger.info(f"Research pipeline completed. Found {len(results['factors'])} factors.")
            return results
            
        except Exception as e:
            logger.error(f"Research pipeline failed: {e}")
            # 返回空结果而不是抛出异常(保持兼容)
            return {
                "topic": research_topic,
                "hypotheses": [],
                "factors": [],
                "strategies": [],
                "models": [],
                "best_solution": None,
                "error": str(e)
            }
    
    async def discover_factors(self,
                              data: pd.DataFrame,
                              target: str = "returns",
                              n_factors: int = 10) -> List[FactorDefinition]:
        """
        自动发现因子 (保持原有API签名)
        
        Args:
            data: 历史数据 (暂未使用)
            target: 目标变量
            n_factors: 要发现的因子数量
            
        Returns:
            FactorDefinition列表
        """
        logger.info(f"Discovering {n_factors} factors...")
        
        try:
            # 1. 获取官方FactorLoop
            factor_loop = self._official_manager.get_factor_loop()
            
            # 2. 运行1-2轮发现新因子
            await factor_loop.run(loop_n=2)
            
            # 3. 提取因子
            factors = _ResultAdapter.experiments_to_factors(
                factor_loop.trace,
                n_factors=n_factors
            )
            
            logger.info(f"Discovered {len(factors)} factors")
            return factors
            
        except Exception as e:
            logger.error(f"Factor discovery failed: {e}")
            return []
    
    async def optimize_strategy(self,
                               strategy: StrategyTemplate,
                               data: pd.DataFrame,
                               n_trials: int = 100) -> StrategyTemplate:
        """
        优化策略参数 (保持原有API签名)
        
        注意: 暂时保留自研实现,因为官方ModelLoop主要用于模型优化
        
        Args:
            strategy: 策略模板
            data: 历史数据
            n_trials: 优化试验次数
            
        Returns:
            优化后的策略
        """
        logger.warning(
            "optimize_strategy is not yet migrated to official RD-Agent. "
            "Consider using ModelRDLoop for model optimization."
        )
        
        # TODO: 研究官方是否支持策略优化
        # 暂时返回原策略
        return strategy
    
    def get_trace(self):
        """
        获取官方Trace对象 (新增API,用于高级用法)
        
        Returns:
            Trace对象
        """
        return self._official_manager.get_trace()
    
    # Phase 1.2: 离线读取功能 (新增)
    def load_historical_factors(self, workspace_path: str = None, n_factors: int = 10) -> List[FactorDefinition]:
        """
        从历史实验日志加载因子 (离线模式)
        
        Args:
            workspace_path: 工作目录路径 (如果不提供,使用初始化时的路径)
            n_factors: 要加载的因子数量
            
        Returns:
            FactorDefinition列表
            
        Example:
            # 加载历史因子
            factors = agent.load_historical_factors('./logs/rdagent', n_factors=10)
            for factor in factors:
                print(f'{factor.name}: IC={factor.performance["ic"]}')
        """
        if workspace_path is None:
            workspace_path = self.config.get('workspace_path', './logs/rdagent')
        
        logger.info(f"📂 Loading historical factors from {workspace_path}...")
        
        try:
            from rd_agent.logging_integration import QilinRDAgentLogger
            
            # 创建 logger
            hist_logger = QilinRDAgentLogger(workspace_path)
            factors = []
            
            # 读取历史实验
            for exp in hist_logger.iter_experiments(tag='limitup.factor'):
                try:
                    factor = _ResultAdapter.exp_to_factor(exp)
                    factors.append(factor)
                    if len(factors) >= n_factors:
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to convert experiment: {e}")
                    continue
            
            logger.info(f"✅ Loaded {len(factors)} factors from FileStorage")
            return factors
            
        except Exception as e:
            logger.error(f"❌ Failed to load from FileStorage: {e}")
            return []
    
    def load_historical_metrics(self, workspace_path: str = None) -> List[Dict[str, Any]]:
        """
        从历史日志加载指标 (离线模式)
        
        Args:
            workspace_path: 工作目录路径
            
        Returns:
            指标列表
        """
        if workspace_path is None:
            workspace_path = self.config.get('workspace_path', './logs/rdagent')
        
        try:
            from rd_agent.logging_integration import QilinRDAgentLogger
            
            hist_logger = QilinRDAgentLogger(workspace_path)
            metrics_list = list(hist_logger.iter_metrics(tag='limitup.summary'))
            
            logger.info(f"✅ Loaded {len(metrics_list)} metrics from FileStorage")
            return metrics_list
            
        except Exception as e:
            logger.error(f"❌ Failed to load metrics: {e}")
            return []
    
    def load_factors_with_fallback(self, workspace_path: str = None, n_factors: int = 10) -> List[FactorDefinition]:
        """
        多级兜底的因子加载
        
        兜底策略:
        1. FileStorage (pkl) - 最优
        2. 运行时 trace - 备用
        3. trace.json - 兜底
        4. 错误诊断 - 失败处理
        
        Args:
            workspace_path: 工作目录路径
            n_factors: 要加载的因子数量
            
        Returns:
            FactorDefinition列表
            
        Raises:
            DataNotFoundError: 所有数据源都不可用时
            
        Example:
            # 自动尝试多种数据源
            try:
                factors = agent.load_factors_with_fallback()
            except DataNotFoundError as e:
                print(f'无法加载因子: {e}')
        """
        if workspace_path is None:
            workspace_path = self.config.get('workspace_path', './logs/rdagent')
        
        logger.info(f"🔄 Loading factors with fallback strategy...")
        
        # 1. 尝试从 FileStorage 读取 (最优)
        try:
            factors = self.load_historical_factors(workspace_path, n_factors)
            if factors:
                logger.info(f"✅ Level 1: Loaded {len(factors)} factors from FileStorage")
                return factors
        except Exception as e:
            logger.warning(f"⚠️ Level 1 (FileStorage) unavailable: {e}")
        
        # 2. 尝试从运行时 trace 读取 (备用)
        try:
            trace = self._official_manager.get_trace()
            if trace and hasattr(trace, 'hist'):
                factors = _ResultAdapter.experiments_to_factors(trace, n_factors)
                if factors:
                    logger.info(f"✅ Level 2: Loaded {len(factors)} factors from runtime trace")
                    return factors
        except Exception as e:
            logger.warning(f"⚠️ Level 2 (Runtime trace) unavailable: {e}")
        
        # 3. 尝试从 trace.json 读取 (兜底)
        try:
            from pathlib import Path
            import json
            
            trace_file = Path(workspace_path) / 'trace.json'
            if trace_file.exists():
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                
                # TODO: 解析 trace.json 格式
                logger.warning("⚠️ trace.json parsing not yet implemented")
        except Exception as e:
            logger.warning(f"⚠️ Level 3 (trace.json) unavailable: {e}")
        
        # 4. 失败处理 + 诊断建议
        from pathlib import Path
        diagnostics = []
        diagnostics.append(f"Cannot load factors from {workspace_path}")
        diagnostics.append("\nDiagnostics:")
        
        # 检查 FileStorage
        pkl_files = list(Path(workspace_path).glob('**/*.pkl'))
        diagnostics.append(f"- FileStorage: {len(pkl_files)} pkl files found")
        
        # 检查运行时 trace
        trace = self._official_manager.get_trace()
        if trace and hasattr(trace, 'hist'):
            diagnostics.append(f"- Runtime trace: {len(trace.hist)} experiments found")
        else:
            diagnostics.append("- Runtime trace: Not available")
        
        # 检查 trace.json
        trace_file = Path(workspace_path) / 'trace.json'
        diagnostics.append(f"- trace.json: {'Found' if trace_file.exists() else 'Not found'}")
        
        diagnostics.append("\nSuggestions:")
        diagnostics.append("1. Run a factor discovery experiment first")
        diagnostics.append("2. Check workspace_path is correct")
        diagnostics.append("3. Ensure experiments were logged to FileStorage")
        
        error_msg = "\n".join(diagnostics)
        logger.error(error_msg)
        
        raise DataNotFoundError(error_msg)
    
    def reset(self):
        """重置所有状态"""
        self._official_manager.reset()
        self.research_history = []
        logger.info("RDAgentWrapper reset")


# 为了向后兼容,提供别名
RDAgent = RDAgentWrapper


# 测试代码
if __name__ == "__main__":
    """
    测试兼容层包装器
    
    运行方式:
        python rd_agent/compat_wrapper.py
    """
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("测试RD-Agent兼容层包装器")
    print("=" * 60)
    
    # 测试配置 (使用原RDAgent的配置格式)
    test_config = {
        "llm_model": "gpt-4-turbo",
        # "llm_api_key": "sk-xxx",  # 从环境变量读取
        "max_iterations": 2,
    }
    
    try:
        # 1. 创建Wrapper
        print("\n1. 创建RDAgentWrapper...")
        agent = RDAgentWrapper(test_config)
        print("   ✅ 成功")
        
        # 2. 测试配置转换
        print("\n2. 测试配置转换...")
        official_config = _ConfigAdapter.to_official_config(test_config)
        print(f"   原配置: {test_config}")
        print(f"   官方配置: {official_config}")
        print("   ✅ 成功")
        
        # 3. 测试获取Trace
        print("\n3. 测试获取Trace...")
        trace = agent.get_trace()
        print(f"   Trace: {trace}")
        print("   ✅ 成功")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        print("\n注意: 完整测试需要在异步环境中运行research_pipeline()")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
