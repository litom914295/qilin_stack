"""
RD-Agent Integration Adapter for 麒麟量化系统
完整集成RD-Agent项目的因子研究、模型开发和自动化研发能力
"""

import sys
import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from datetime import datetime

# 添加RD-Agent到Python路径
RDAGENT_PATH = "D:/test/Qlib/RD-Agent"
if RDAGENT_PATH not in sys.path:
    sys.path.insert(0, RDAGENT_PATH)

# 导入RD-Agent核心模块
try:
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
    from rdagent.app.qlib_rd_loop.model import ModelRDLoop
    from rdagent.app.qlib_rd_loop.quant import QuantRDLoop
    from rdagent.app.qlib_rd_loop.conf import (
        FACTOR_PROP_SETTING,
        MODEL_PROP_SETTING,
        QUANT_PROP_SETTING
    )
    from rdagent.components.workflow.rd_loop import RDLoop
    from rdagent.core.proposal import (
        Hypothesis,
        Experiment,
        HypothesisFeedback
    )
    from rdagent.core.exception import FactorEmptyError, ModelEmptyError
    from rdagent.log import rdagent_logger
    RDAGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RD-Agent导入失败: {e}")
    RDAGENT_AVAILABLE = False
    # 创建模拟类，使得即使RD-Agent未安装也能运行
    class FactorRDLoop:
        pass
    class ModelRDLoop:
        pass
    class QuantRDLoop:
        pass
    class RDLoop:
        pass


@dataclass
class RDAgentConfig:
    """RD-Agent配置"""
    
    # 研发循环配置
    max_loops: int = 10
    max_steps: int = 50
    parallel_jobs: int = 4
    
    # 因子研究配置
    factor_loop_enabled: bool = True
    factor_min_ic: float = 0.02
    factor_max_corr: float = 0.95
    
    # 模型研究配置
    model_loop_enabled: bool = True
    model_min_sharpe: float = 1.0
    model_max_drawdown: float = 0.2
    
    # 存储配置
    workspace_dir: str = "D:/test/Qlib/qilin_stack_with_ta/workspace/rdagent"
    log_dir: str = "logs/rdagent"
    checkpoint_dir: str = "checkpoints/rdagent"


class RDAgentIntegration:
    """RD-Agent完整集成接口"""
    
    def __init__(self, config: Optional[RDAgentConfig] = None):
        """
        初始化RD-Agent集成
        
        Args:
            config: RD-Agent配置
        """
        self.config = config or RDAgentConfig()
        self.logger = self._setup_logger()
        
        if not RDAGENT_AVAILABLE:
            raise ImportError("RD-Agent模块不可用，请确保RD-Agent项目正确安装")
        
        # 初始化研发循环
        self.factor_loop: Optional[FactorRDLoop] = None
        self.model_loop: Optional[ModelRDLoop] = None
        self.quant_loop: Optional[QuantRDLoop] = None
        
        # 状态追踪
        self.active_loops: Dict[str, RDLoop] = {}
        self.research_history: List[Dict] = []
        
        # 初始化工作空间
        self._setup_workspace()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("RDAgentIntegration")
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 文件处理器
        fh = logging.FileHandler(
            log_path / f"rdagent_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _setup_workspace(self):
        """设置工作空间目录"""
        for dir_path in [
            self.config.workspace_dir,
            self.config.checkpoint_dir,
            f"{self.config.workspace_dir}/factors",
            f"{self.config.workspace_dir}/models",
            f"{self.config.workspace_dir}/experiments"
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    async def start_factor_research(
        self,
        hypothesis: Optional[str] = None,
        data_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        启动因子研究循环
        
        Args:
            hypothesis: 初始假设
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            研究结果
        """
        if not self.config.factor_loop_enabled:
            return {"error": "因子研究循环未启用"}
            
        try:
            self.logger.info(f"启动因子研究循环: {hypothesis}")
            
            # 创建因子研究循环
            self.factor_loop = FactorRDLoop(FACTOR_PROP_SETTING)
            
            # 运行研究循环
            result = await self.factor_loop.run(
                step_n=kwargs.get('step_n', 10),
                loop_n=kwargs.get('loop_n', 5)
            )
            
            # 保存结果
            self._save_research_result("factor", result)
            
            return {
                "status": "success",
                "type": "factor_research",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"因子研究失败: {e}")
            return {"error": str(e)}
            
    async def start_model_research(
        self,
        hypothesis: Optional[str] = None,
        base_model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        启动模型研究循环
        
        Args:
            hypothesis: 初始假设
            base_model: 基础模型
            **kwargs: 其他参数
            
        Returns:
            研究结果
        """
        if not self.config.model_loop_enabled:
            return {"error": "模型研究循环未启用"}
            
        try:
            self.logger.info(f"启动模型研究循环: {hypothesis}")
            
            # 创建模型研究循环
            self.model_loop = ModelRDLoop(MODEL_PROP_SETTING)
            
            # 运行研究循环
            result = await self.model_loop.run(
                step_n=kwargs.get('step_n', 10),
                loop_n=kwargs.get('loop_n', 5)
            )
            
            # 保存结果
            self._save_research_result("model", result)
            
            return {
                "status": "success",
                "type": "model_research",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"模型研究失败: {e}")
            return {"error": str(e)}
            
    async def start_quant_research(
        self,
        research_type: str = "both",
        **kwargs
    ) -> Dict[str, Any]:
        """
        启动综合量化研究（因子+模型）
        
        Args:
            research_type: 研究类型 (factor/model/both)
            **kwargs: 其他参数
            
        Returns:
            研究结果
        """
        try:
            self.logger.info(f"启动综合量化研究: {research_type}")
            
            # 创建综合研究循环
            self.quant_loop = QuantRDLoop(QUANT_PROP_SETTING)
            
            # 运行研究循环
            result = await self.quant_loop.run(
                step_n=kwargs.get('step_n', 20),
                loop_n=kwargs.get('loop_n', 10)
            )
            
            # 保存结果
            self._save_research_result("quant", result)
            
            return {
                "status": "success",
                "type": "quant_research",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"综合量化研究失败: {e}")
            return {"error": str(e)}
            
    def generate_hypothesis(
        self,
        context: Dict[str, Any],
        research_type: str = "factor"
    ) -> str:
        """
        生成研究假设
        
        Args:
            context: 上下文信息
            research_type: 研究类型
            
        Returns:
            生成的假设
        """
        if research_type == "factor":
            # 基于市场观察生成因子假设
            return self._generate_factor_hypothesis(context)
        elif research_type == "model":
            # 基于性能要求生成模型假设
            return self._generate_model_hypothesis(context)
        else:
            # 综合假设
            return self._generate_quant_hypothesis(context)
            
    def _generate_factor_hypothesis(self, context: Dict) -> str:
        """生成因子假设"""
        market_regime = context.get('market_regime', 'normal')
        target_return = context.get('target_return', 0.1)
        
        hypothesis = f"""
        在当前{market_regime}市场环境下，通过分析以下维度可以发现超额收益:
        1. 价量关系的非线性特征
        2. 资金流向的结构性变化
        3. 市场情绪的极值反转
        目标年化收益率: {target_return:.1%}
        """
        return hypothesis
        
    def _generate_model_hypothesis(self, context: Dict) -> str:
        """生成模型假设"""
        model_type = context.get('model_type', 'ensemble')
        optimization_target = context.get('optimization_target', 'sharpe')
        
        hypothesis = f"""
        采用{model_type}模型架构，通过以下优化可以提升表现:
        1. 特征工程的自动化筛选
        2. 模型参数的动态调整
        3. 风险控制的自适应机制
        优化目标: {optimization_target}
        """
        return hypothesis
        
    def _generate_quant_hypothesis(self, context: Dict) -> str:
        """生成综合量化假设"""
        strategy_type = context.get('strategy_type', 'multi-factor')
        risk_tolerance = context.get('risk_tolerance', 'moderate')
        
        hypothesis = f"""
        构建{strategy_type}策略，在{risk_tolerance}风险偏好下:
        1. 因子层面: 挖掘低相关性高IC因子
        2. 模型层面: 集成多种预测模型
        3. 组合层面: 动态权重优化
        预期夏普比率 > 2.0
        """
        return hypothesis
        
    def evaluate_research_result(
        self,
        result: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        评估研究结果
        
        Args:
            result: 研究结果
            metrics: 评估指标阈值
            
        Returns:
            评估报告
        """
        metrics = metrics or {
            'min_ic': self.config.factor_min_ic,
            'max_corr': self.config.factor_max_corr,
            'min_sharpe': self.config.model_min_sharpe,
            'max_drawdown': self.config.model_max_drawdown
        }
        
        evaluation = {
            'passed': True,
            'scores': {},
            'warnings': [],
            'recommendations': []
        }
        
        # 评估因子质量
        if 'factors' in result:
            factor_eval = self._evaluate_factors(result['factors'], metrics)
            evaluation['scores']['factors'] = factor_eval
            if not factor_eval['passed']:
                evaluation['passed'] = False
                evaluation['warnings'].append("因子质量未达标")
                
        # 评估模型性能
        if 'model' in result:
            model_eval = self._evaluate_model(result['model'], metrics)
            evaluation['scores']['model'] = model_eval
            if not model_eval['passed']:
                evaluation['passed'] = False
                evaluation['warnings'].append("模型性能未达标")
                
        # 生成改进建议
        evaluation['recommendations'] = self._generate_recommendations(evaluation)
        
        return evaluation
        
    def _evaluate_factors(self, factors: List[Dict], metrics: Dict) -> Dict:
        """评估因子"""
        passed_factors = []
        
        for factor in factors:
            ic = factor.get('ic', 0)
            if ic >= metrics['min_ic']:
                passed_factors.append(factor)
                
        return {
            'passed': len(passed_factors) > 0,
            'total': len(factors),
            'qualified': len(passed_factors),
            'best_ic': max([f.get('ic', 0) for f in factors]) if factors else 0
        }
        
    def _evaluate_model(self, model: Dict, metrics: Dict) -> Dict:
        """评估模型"""
        sharpe = model.get('sharpe_ratio', 0)
        drawdown = model.get('max_drawdown', 1.0)
        
        return {
            'passed': sharpe >= metrics['min_sharpe'] and drawdown <= metrics['max_drawdown'],
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
            'annual_return': model.get('annual_return', 0)
        }
        
    def _generate_recommendations(self, evaluation: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if 'factors' in evaluation['scores']:
            factor_score = evaluation['scores']['factors']
            if factor_score['qualified'] < factor_score['total'] * 0.3:
                recommendations.append("建议增加因子多样性或改进因子构建方法")
                
        if 'model' in evaluation['scores']:
            model_score = evaluation['scores']['model']
            if model_score['sharpe_ratio'] < 1.5:
                recommendations.append("建议优化模型参数或尝试集成学习")
                
        return recommendations
        
    def _save_research_result(self, research_type: str, result: Any):
        """保存研究结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{research_type}_{timestamp}.json"
        filepath = Path(self.config.workspace_dir) / "experiments" / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "type": research_type,
                        "timestamp": timestamp,
                        "result": str(result)  # 简单字符串化，实际应该序列化
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            self.logger.info(f"研究结果已保存: {filepath}")
        except Exception as e:
            self.logger.error(f"保存研究结果失败: {e}")
            
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载检查点继续研究
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            是否成功加载
        """
        try:
            # 从检查点恢复研发循环
            if "factor" in checkpoint_path:
                self.factor_loop = FactorRDLoop.load(checkpoint_path)
            elif "model" in checkpoint_path:
                self.model_loop = ModelRDLoop.load(checkpoint_path)
            elif "quant" in checkpoint_path:
                self.quant_loop = QuantRDLoop.load(checkpoint_path)
                
            self.logger.info(f"成功加载检查点: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False
            
    def get_research_status(self) -> Dict[str, Any]:
        """获取当前研究状态"""
        status = {
            "active_loops": {},
            "research_history": len(self.research_history),
            "workspace": self.config.workspace_dir
        }
        
        if self.factor_loop:
            status["active_loops"]["factor"] = "running"
        if self.model_loop:
            status["active_loops"]["model"] = "running"
        if self.quant_loop:
            status["active_loops"]["quant"] = "running"
            
        return status


class RDAgentAPIClient:
    """RD-Agent API客户端，用于与麒麟系统其他模块交互"""
    
    def __init__(self, integration: RDAgentIntegration):
        """
        初始化API客户端
        
        Args:
            integration: RD-Agent集成实例
        """
        self.integration = integration
        self.logger = logging.getLogger("RDAgentAPIClient")
        
    async def process_factor_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理因子研究请求
        
        Args:
            request: 请求数据
            
        Returns:
            响应数据
        """
        hypothesis = request.get('hypothesis')
        data_path = request.get('data_path')
        parameters = request.get('parameters', {})
        
        # 启动因子研究
        result = await self.integration.start_factor_research(
            hypothesis=hypothesis,
            data_path=data_path,
            **parameters
        )
        
        # 评估结果
        if 'result' in result:
            evaluation = self.integration.evaluate_research_result(result['result'])
            result['evaluation'] = evaluation
            
        return result
        
    async def process_model_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理模型研究请求
        
        Args:
            request: 请求数据
            
        Returns:
            响应数据
        """
        hypothesis = request.get('hypothesis')
        base_model = request.get('base_model')
        parameters = request.get('parameters', {})
        
        # 启动模型研究
        result = await self.integration.start_model_research(
            hypothesis=hypothesis,
            base_model=base_model,
            **parameters
        )
        
        # 评估结果
        if 'result' in result:
            evaluation = self.integration.evaluate_research_result(result['result'])
            result['evaluation'] = evaluation
            
        return result
        
    async def process_quant_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理综合量化研究请求
        
        Args:
            request: 请求数据
            
        Returns:
            响应数据
        """
        research_type = request.get('research_type', 'both')
        parameters = request.get('parameters', {})
        
        # 启动综合研究
        result = await self.integration.start_quant_research(
            research_type=research_type,
            **parameters
        )
        
        # 评估结果
        if 'result' in result:
            evaluation = self.integration.evaluate_research_result(result['result'])
            result['evaluation'] = evaluation
            
        return result
        
    def generate_hypothesis_from_context(
        self,
        market_data: Dict[str, Any],
        strategy_config: Dict[str, Any]
    ) -> str:
        """
        基于市场数据和策略配置生成研究假设
        
        Args:
            market_data: 市场数据
            strategy_config: 策略配置
            
        Returns:
            生成的假设
        """
        context = {
            **market_data,
            **strategy_config
        }
        
        research_type = strategy_config.get('research_type', 'factor')
        return self.integration.generate_hypothesis(context, research_type)


# 导出便捷函数
async def create_rdagent_integration(
    config: Optional[RDAgentConfig] = None
) -> RDAgentIntegration:
    """
    创建RD-Agent集成实例
    
    Args:
        config: 配置对象
        
    Returns:
        RD-Agent集成实例
    """
    integration = RDAgentIntegration(config)
    return integration


async def run_factor_research(
    hypothesis: str,
    **kwargs
) -> Dict[str, Any]:
    """
    运行因子研究的快捷函数
    
    Args:
        hypothesis: 研究假设
        **kwargs: 其他参数
        
    Returns:
        研究结果
    """
    integration = await create_rdagent_integration()
    return await integration.start_factor_research(hypothesis, **kwargs)


async def run_model_research(
    hypothesis: str,
    **kwargs
) -> Dict[str, Any]:
    """
    运行模型研究的快捷函数
    
    Args:
        hypothesis: 研究假设
        **kwargs: 其他参数
        
    Returns:
        研究结果
    """
    integration = await create_rdagent_integration()
    return await integration.start_model_research(hypothesis, **kwargs)


async def run_quant_research(**kwargs) -> Dict[str, Any]:
    """
    运行综合量化研究的快捷函数
    
    Args:
        **kwargs: 参数
        
    Returns:
        研究结果
    """
    integration = await create_rdagent_integration()
    return await integration.start_quant_research(**kwargs)


# 示例使用
if __name__ == "__main__":
    async def main():
        # 创建集成实例
        config = RDAgentConfig(
            max_loops=5,
            factor_min_ic=0.03,
            model_min_sharpe=1.5
        )
        integration = await create_rdagent_integration(config)
        
        # 生成假设
        hypothesis = integration.generate_hypothesis(
            {
                'market_regime': 'volatile',
                'target_return': 0.15,
                'research_type': 'factor'
            },
            research_type='factor'
        )
        
        print(f"生成的假设: {hypothesis}")
        
        # 运行因子研究
        result = await integration.start_factor_research(
            hypothesis=hypothesis,
            step_n=3,
            loop_n=2
        )
        
        print(f"研究结果: {result}")
        
        # 评估结果
        if 'result' in result:
            evaluation = integration.evaluate_research_result(result['result'])
            print(f"评估报告: {evaluation}")
            
    # 运行示例
    asyncio.run(main())