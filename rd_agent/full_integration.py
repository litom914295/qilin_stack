"""
RD-Agent完整集成（无降级版本）
直接使用RD-Agent官方组件，提供完整的自动化研发能力
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
from dataclasses import dataclass, field

# 导入配置
from .config import RDAgentConfig, load_config

logger = logging.getLogger(__name__)


# ============================================================================
# 检查并导入RD-Agent官方组件
# ============================================================================

def setup_rdagent_path(config: RDAgentConfig):
    """设置RD-Agent路径"""
    rd_path = Path(config.rdagent_path)
    
    if not rd_path.exists():
        raise FileNotFoundError(f"RD-Agent路径不存在: {rd_path}")
    
    # 添加到系统路径
    if str(rd_path) not in sys.path:
        sys.path.insert(0, str(rd_path))
    
    logger.info(f"RD-Agent路径已添加: {rd_path}")


# 导入RD-Agent核心组件
try:
    from rdagent.scenarios.qlib.experiment.factor_experiment import (
        QlibFactorExperiment,
        QlibFactorScenario
    )
    from rdagent.scenarios.qlib.experiment.model_experiment import (
        QlibModelExperiment,
        QlibModelScenario
    )
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
    from rdagent.app.qlib_rd_loop.model import ModelRDLoop
    from rdagent.app.qlib_rd_loop.conf import (
        FACTOR_PROP_SETTING,
        MODEL_PROP_SETTING,
        FactorBasePropSetting,
        ModelBasePropSetting
    )
    from rdagent.components.workflow.rd_loop import RDLoop
    from rdagent.core.exception import FactorEmptyError, ModelEmptyError
    from rdagent.log import rdagent_logger
    
    RDAGENT_AVAILABLE = True
    logger.info("✅ RD-Agent官方组件导入成功")
    
except ImportError as e:
    logger.error(f"❌ RD-Agent导入失败: {e}")
    logger.error("请确保RD-Agent项目已正确安装")
    RDAGENT_AVAILABLE = False
    raise


# ============================================================================
# 因子研究循环
# ============================================================================

@dataclass
class FactorResearchResult:
    """因子研究结果"""
    factors: List[Dict[str, Any]]
    best_factor: Dict[str, Any]
    performance_metrics: Dict[str, float]
    research_log: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class FactorResearchLoop:
    """因子研究循环封装"""
    
    def __init__(self, config: RDAgentConfig):
        self.config = config
        self.rd_loop: Optional[FactorRDLoop] = None
        self._initialize()
    
    def _initialize(self):
        """初始化因子研究循环"""
        try:
            # 使用自定义配置或默认配置
            if hasattr(self.config, 'factor_prop_setting'):
                prop_setting = self.config.factor_prop_setting
            else:
                prop_setting = FACTOR_PROP_SETTING
            
            # 创建因子研发循环
            self.rd_loop = FactorRDLoop(prop_setting)
            logger.info("✅ 因子研究循环初始化成功")
            
        except Exception as e:
            logger.error(f"因子研究循环初始化失败: {e}")
            raise
    
    async def run_research(self, 
                          step_n: int = 10,
                          loop_n: int = 5,
                          all_duration: Optional[str] = None) -> FactorResearchResult:
        """
        运行因子研究
        
        Args:
            step_n: 每轮步骤数
            loop_n: 循环轮数
            all_duration: 总时长限制
            
        Returns:
            因子研究结果
        """
        logger.info(f"🔬 开始因子研究: step_n={step_n}, loop_n={loop_n}")
        
        try:
            # 运行RD-Agent研发循环
            result = await self.rd_loop.run(
                step_n=step_n,
                loop_n=loop_n,
                all_duration=all_duration
            )
            
            # 解析结果
            factors = self._extract_factors(result)
            best_factor = self._select_best_factor(factors)
            metrics = self._calculate_metrics(result)
            log = self._extract_log(result)
            
            research_result = FactorResearchResult(
                factors=factors,
                best_factor=best_factor,
                performance_metrics=metrics,
                research_log=log
            )
            
            logger.info(f"✅ 因子研究完成，发现{len(factors)}个因子")
            return research_result
            
        except FactorEmptyError as e:
            logger.error(f"因子提取失败: {e}")
            raise
        except Exception as e:
            logger.error(f"因子研究失败: {e}")
            raise
    
    def _extract_factors(self, result: Any) -> List[Dict[str, Any]]:
        """从结果中提取因子"""
        # 根据RD-Agent的实际结果结构提取因子
        factors = []
        
        # TODO: 根据实际result结构实现
        if hasattr(result, 'experiments'):
            for exp in result.experiments:
                if hasattr(exp, 'factor_code'):
                    factors.append({
                        'code': exp.factor_code,
                        'performance': getattr(exp, 'performance', {}),
                        'name': getattr(exp, 'name', 'unknown')
                    })
        
        return factors
    
    def _select_best_factor(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """选择最佳因子"""
        if not factors:
            return {}
        
        # 根据IC选择最佳因子
        return max(
            factors,
            key=lambda f: f.get('performance', {}).get('ic', 0)
        )
    
    def _calculate_metrics(self, result: Any) -> Dict[str, float]:
        """计算性能指标"""
        return {
            'total_experiments': getattr(result, 'total_experiments', 0),
            'success_rate': getattr(result, 'success_rate', 0),
            'avg_ic': getattr(result, 'avg_ic', 0),
        }
    
    def _extract_log(self, result: Any) -> List[str]:
        """提取研究日志"""
        return getattr(result, 'log', [])


# ============================================================================
# 模型研究循环
# ============================================================================

@dataclass
class ModelResearchResult:
    """模型研究结果"""
    model_code: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    research_log: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ModelResearchLoop:
    """模型研究循环封装"""
    
    def __init__(self, config: RDAgentConfig):
        self.config = config
        self.rd_loop: Optional[ModelRDLoop] = None
        self._initialize()
    
    def _initialize(self):
        """初始化模型研究循环"""
        try:
            # 使用自定义配置或默认配置
            if hasattr(self.config, 'model_prop_setting'):
                prop_setting = self.config.model_prop_setting
            else:
                prop_setting = MODEL_PROP_SETTING
            
            # 创建模型研发循环
            self.rd_loop = ModelRDLoop(prop_setting)
            logger.info("✅ 模型研究循环初始化成功")
            
        except Exception as e:
            logger.error(f"模型研究循环初始化失败: {e}")
            raise
    
    async def run_research(self,
                          step_n: int = 10,
                          loop_n: int = 5,
                          all_duration: Optional[str] = None) -> ModelResearchResult:
        """
        运行模型研究
        
        Args:
            step_n: 每轮步骤数
            loop_n: 循环轮数
            all_duration: 总时长限制
            
        Returns:
            模型研究结果
        """
        logger.info(f"🔬 开始模型研究: step_n={step_n}, loop_n={loop_n}")
        
        try:
            # 运行RD-Agent研发循环
            result = await self.rd_loop.run(
                step_n=step_n,
                loop_n=loop_n,
                all_duration=all_duration
            )
            
            # 解析结果
            model_code = self._extract_model_code(result)
            parameters = self._extract_parameters(result)
            metrics = self._calculate_metrics(result)
            log = self._extract_log(result)
            
            research_result = ModelResearchResult(
                model_code=model_code,
                parameters=parameters,
                performance_metrics=metrics,
                research_log=log
            )
            
            logger.info(f"✅ 模型研究完成")
            return research_result
            
        except ModelEmptyError as e:
            logger.error(f"模型提取失败: {e}")
            raise
        except Exception as e:
            logger.error(f"模型研究失败: {e}")
            raise
    
    def _extract_model_code(self, result: Any) -> str:
        """提取模型代码"""
        return getattr(result, 'model_code', '')
    
    def _extract_parameters(self, result: Any) -> Dict[str, Any]:
        """提取模型参数"""
        return getattr(result, 'parameters', {})
    
    def _calculate_metrics(self, result: Any) -> Dict[str, float]:
        """计算性能指标"""
        return {
            'sharpe_ratio': getattr(result, 'sharpe_ratio', 0),
            'max_drawdown': getattr(result, 'max_drawdown', 0),
            'annual_return': getattr(result, 'annual_return', 0),
        }
    
    def _extract_log(self, result: Any) -> List[str]:
        """提取研究日志"""
        return getattr(result, 'log', [])


# ============================================================================
# 主集成类
# ============================================================================

class FullRDAgentIntegration:
    """
    RD-Agent完整集成（无降级）
    直接使用官方组件提供完整功能
    """
    
    def __init__(self, config: Optional[RDAgentConfig] = None):
        """
        初始化完整集成
        
        Args:
            config: 配置对象
        """
        if not RDAGENT_AVAILABLE:
            raise ImportError(
                "RD-Agent官方组件不可用。\n"
                "请确保:\n"
                "1. RD-Agent项目已克隆到正确路径\n"
                "2. 依赖已安装: pip install -r requirements.txt\n"
                "3. 路径配置正确"
            )
        
        self.config = config or load_config()
        
        # 设置路径
        setup_rdagent_path(self.config)
        
        # 初始化研究循环
        self.factor_research = FactorResearchLoop(self.config)
        self.model_research = ModelResearchLoop(self.config)
        
        logger.info("✅ RD-Agent完整集成初始化成功")
    
    async def discover_factors(self,
                              step_n: int = 10,
                              loop_n: int = 5) -> FactorResearchResult:
        """
        自动发现因子
        
        Args:
            step_n: 每轮步骤数
            loop_n: 循环轮数
            
        Returns:
            因子研究结果
        """
        return await self.factor_research.run_research(step_n, loop_n)
    
    async def optimize_model(self,
                            step_n: int = 10,
                            loop_n: int = 5) -> ModelResearchResult:
        """
        优化模型
        
        Args:
            step_n: 每轮步骤数
            loop_n: 循环轮数
            
        Returns:
            模型研究结果
        """
        return await self.model_research.run_research(step_n, loop_n)
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'rdagent_available': RDAGENT_AVAILABLE,
            'mode': 'full_integration',
            'factor_loop_ready': self.factor_research.rd_loop is not None,
            'model_loop_ready': self.model_research.rd_loop is not None,
            'config': self.config.to_dict()
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_full_integration(config_file: Optional[str] = None) -> FullRDAgentIntegration:
    """
    创建完整的RD-Agent集成实例
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        完整集成实例
        
    Raises:
        ImportError: 如果RD-Agent不可用
    """
    config = load_config(config_file)
    return FullRDAgentIntegration(config)


# ============================================================================
# 测试
# ============================================================================

import logging
logger = logging.getLogger(__name__)

async def test_full_integration():
    """测试完整集成"""
    logger.info("RD-Agent完整集成测试")
    
    try:
        # 创建集成
        integration = create_full_integration()
        
        # 检查状态
        status = integration.get_status()
        logger.info("系统状态:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        
        # 测试因子发现
        logger.info("测试因子发现...")
        factor_result = await integration.discover_factors(step_n=2, loop_n=1)
        
        logger.info(f"发现 {len(factor_result.factors)} 个因子")
        logger.info(f"最佳因子: {factor_result.best_factor.get('name', 'N/A')}")
        logger.info(f"性能指标: {factor_result.performance_metrics}")
        
        # 测试模型优化
        logger.info("测试模型优化...")
        model_result = await integration.optimize_model(step_n=2, loop_n=1)
        
        logger.info("模型优化完成")
        logger.info(f"性能指标: {model_result.performance_metrics}")
        
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.info("请确保RD-Agent已正确安装和配置")
    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    from app.core.logging_setup import setup_logging
    setup_logging()
    asyncio.run(test_full_integration())
