"""
RD-Agent官方组件集成模块

任务: P0-1.4
功能: 管理官方RD-Agent组件的初始化、配置和生命周期
集成: P0-2 LLM集成 + P0-3 路径配置
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# 集成P0-3路径配置
# 暂时使用简化的路径配置(P0-3的paths.py还未创建到config目录)
import sys
from pathlib import Path

# 简化的路径配置类(临时)
class PathConfig:
    @staticmethod
    def get_rdagent_path():
        # 检查常见位置
        possible_paths = [
            Path("G:/test/RD-Agent"),
            Path("../RD-Agent"),
            Path(os.environ.get("RDAGENT_PATH", "")),
        ]
        for p in possible_paths:
            if p.exists() and (p / "rdagent").exists():
                return p
        return None
    
    @staticmethod
    def get_qlib_data_path():
        qlib_path = os.environ.get("QLIB_DATA_PATH")
        if qlib_path:
            return Path(qlib_path)
        # 默认路径
        default_path = Path.home() / ".qlib" / "qlib_data" / "cn_data"
        if default_path.exists():
            return default_path
        return None

def init_paths(verbose=False):
    """简化的路径初始化"""
    return True

logger = logging.getLogger(__name__)


class OfficialIntegrationError(Exception):
    """官方组件集成错误"""
    pass


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class OfficialRDAgentManager:
    """
    官方RD-Agent组件管理器
    
    职责:
    1. 添加官方RD-Agent到sys.path
    2. 配置LLM (集成P0-2)
    3. 配置路径 (集成P0-3)
    4. 初始化FactorRDLoop和ModelRDLoop
    5. 管理组件生命周期
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: Optional[str] = None):
        """
        初始化官方组件管理器
        
        Args:
            config: 配置字典,包含LLM、路径等配置
            checkpoint_path: 检查点路径（用于会话恢复）
        """
        self.config = config
        self.checkpoint_path = checkpoint_path or config.get('checkpoint_path')  # ✅ P0-1
        self._factor_loop = None
        self._model_loop = None
        self._rdagent_added = False
        
        # 第1步: 初始化路径配置 (P0-3)
        self._setup_paths()
        
        # 第2步: 添加官方RD-Agent到sys.path
        self._add_rdagent_to_path()
        
        # 第3步: 配置环境变量 (LLM等)
        self._setup_environment()
        
        logger.info("OfficialRDAgentManager initialized successfully")
    
    def _setup_paths(self):
        """初始化路径配置 (集成P0-3)"""
        try:
            # 使用P0-3的路径初始化
            success = init_paths(verbose=False)
            if not success:
                logger.warning("Path initialization had warnings, but continuing...")
            
            logger.info("Path configuration completed")
        except Exception as e:
            raise OfficialIntegrationError(
                f"Failed to setup paths: {e}"
            ) from e
    
    def _add_rdagent_to_path(self):
        """添加官方RD-Agent到sys.path"""
        if self._rdagent_added:
            return
        
        try:
            # 获取RD-Agent路径 (使用P0-3的PathConfig)
            rdagent_path = PathConfig.get_rdagent_path()
            
            if rdagent_path is None:
                raise OfficialIntegrationError(
                    "RD-Agent path not found. Please ensure RD-Agent is installed "
                    "or set RDAGENT_PATH environment variable."
                )
            
            rdagent_str = str(rdagent_path)
            
            # 添加到sys.path (如果还没添加)
            if rdagent_str not in sys.path:
                sys.path.insert(0, rdagent_str)
                logger.info(f"Added RD-Agent to sys.path: {rdagent_str}")
            
            self._rdagent_added = True
            
        except Exception as e:
            raise OfficialIntegrationError(
                f"Failed to add RD-Agent to sys.path: {e}"
            ) from e
    
    def _setup_environment(self):
        """配置环境变量 (LLM、Qlib等)"""
        try:
            # LLM配置
            if "llm_provider" in self.config:
                os.environ["LLM_PROVIDER"] = self.config["llm_provider"]
            
            if "llm_model" in self.config:
                os.environ["LLM_MODEL"] = self.config["llm_model"]
            
            if "llm_api_key" in self.config:
                # 根据provider设置对应的API key
                provider = self.config.get("llm_provider", "openai").lower()
                if provider == "openai":
                    os.environ["OPENAI_API_KEY"] = self.config["llm_api_key"]
                elif provider == "anthropic":
                    os.environ["ANTHROPIC_API_KEY"] = self.config["llm_api_key"]
            
            if "llm_base_url" in self.config:
                os.environ["OPENAI_BASE_URL"] = self.config["llm_base_url"]
            
            if "llm_temperature" in self.config:
                os.environ["LLM_TEMPERATURE"] = str(self.config["llm_temperature"])
            
            # Qlib数据路径
            qlib_data_path = PathConfig.get_qlib_data_path()
            if qlib_data_path:
                os.environ["QLIB_DATA_PATH"] = str(qlib_data_path)
            
            logger.info("Environment variables configured")
            
        except Exception as e:
            raise OfficialIntegrationError(
                f"Failed to setup environment: {e}"
            ) from e
    
    def get_factor_loop(self, resume: bool = False):
        """
        获取官方FactorRDLoop实例 (懒加载)
        
        Args:
            resume: 是否从checkpoint恢复 (如果checkpoint_path配置存在)
        
        Returns:
            FactorRDLoop实例
        """
        if self._factor_loop is not None:
            return self._factor_loop
        
        try:
            # 动态导入官方RD-Agent组件
            from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
            from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
            
            # 如果需要恢复且checkpoint路径存在
            if resume and self.checkpoint_path:
                checkpoint_file = Path(self.checkpoint_path)
                if checkpoint_file.exists():
                    logger.info(f"Resuming FactorRDLoop from checkpoint: {self.checkpoint_path}")
                    self._factor_loop = FactorRDLoop.load(str(checkpoint_file), checkout=True)
                    logger.info(f"✅ Resumed from checkpoint: {self.checkpoint_path}")
                    return self._factor_loop
                else:
                    logger.warning(f"Checkpoint file not found: {self.checkpoint_path}, creating new loop")
            
            # 创建配置
            factor_setting = FactorBasePropSetting(
                # 使用默认场景
                scen="rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario",
                
                # 使用默认组件
                hypothesis_gen="rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen",
                hypothesis2experiment="rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment",
                coder="rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER",
                runner="rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner",
                summarizer="rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback",
                
                # 执行配置
                evolving_n=self.config.get("max_iterations", 10),
            )
            
            # 创建FactorRDLoop
            self._factor_loop = FactorRDLoop(factor_setting)
            
            logger.info("FactorRDLoop created successfully")
            return self._factor_loop
            
        except ImportError as e:
            raise OfficialIntegrationError(
                f"Failed to import official RD-Agent components. "
                f"Please ensure RD-Agent is installed: {e}"
            ) from e
        except Exception as e:
            raise OfficialIntegrationError(
                f"Failed to create FactorRDLoop: {e}"
            ) from e
    
    def get_model_loop(self, resume: bool = False):
        """
        获取官方ModelRDLoop实例 (懒加载)
        
        Args:
            resume: 是否从checkpoint恢复 (如果checkpoint_path配置存在)
        
        Returns:
            ModelRDLoop实例
        """
        if self._model_loop is not None:
            return self._model_loop
        
        try:
            # 动态导入官方RD-Agent组件
            from rdagent.app.qlib_rd_loop.model import ModelRDLoop
            from rdagent.app.qlib_rd_loop.conf import ModelBasePropSetting
            
            # 如果需要恢复且checkpoint路径存在
            if resume and self.checkpoint_path:
                checkpoint_file = Path(self.checkpoint_path)
                if checkpoint_file.exists():
                    logger.info(f"Resuming ModelRDLoop from checkpoint: {self.checkpoint_path}")
                    self._model_loop = ModelRDLoop.load(str(checkpoint_file), checkout=True)
                    logger.info(f"✅ Resumed from checkpoint: {self.checkpoint_path}")
                    return self._model_loop
                else:
                    logger.warning(f"Checkpoint file not found: {self.checkpoint_path}, creating new loop")
            
            # 创建配置
            model_setting = ModelBasePropSetting(
                # 使用默认场景
                scen="rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario",
                
                # 使用默认组件
                hypothesis_gen="rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen",
                hypothesis2experiment="rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment",
                coder="rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER",
                runner="rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner",
                summarizer="rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback",
                
                # 执行配置
                evolving_n=self.config.get("max_iterations", 10),
            )
            
            # 创建ModelRDLoop
            self._model_loop = ModelRDLoop(model_setting)
            
            logger.info("ModelRDLoop created successfully")
            return self._model_loop
            
        except ImportError as e:
            raise OfficialIntegrationError(
                f"Failed to import official RD-Agent components. "
                f"Please ensure RD-Agent is installed: {e}"
            ) from e
        except Exception as e:
            raise OfficialIntegrationError(
                f"Failed to create ModelRDLoop: {e}"
            ) from e
    
    @staticmethod
    def validate_config(config: Dict[str, Any]):
        """
        验证配置完整性
        
        Args:
            config: 配置字典
            
        Raises:
            ConfigValidationError: 配置不完整或无效
        """
        # 检查必需的配置项
        required_keys = []
        
        # LLM配置是可选的(可以从环境变量读取)
        # 但如果提供了api_key,必须提供provider
        if "llm_api_key" in config and "llm_provider" not in config:
            raise ConfigValidationError(
                "llm_provider is required when llm_api_key is provided"
            )
        
        # 检查缺失的必需配置
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ConfigValidationError(
                f"Missing required config keys: {missing}"
            )
        
        logger.info("Configuration validated successfully")
    
    def resume_from_checkpoint(self, checkpoint_path: str = None, mode: str = "factor"):
        """
        从checkpoint恢复Loop状态
        
        Args:
            checkpoint_path: Checkpoint文件路径 (如果不提供,使用配置中的checkpoint_path)
            mode: 恢复模式 - "factor" 或 "model"
        
        Returns:
            恢复后的Loop实例
        
        Raises:
            OfficialIntegrationError: 恢复失败
            ValueError: 不支持的mode
        """
        # 如果提供了checkpoint_path,更新配置
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        
        if not self.checkpoint_path:
            raise OfficialIntegrationError(
                "No checkpoint_path configured. Please provide checkpoint_path in config or as parameter."
            )
        
        try:
            if mode == "factor":
                logger.info(f"Resuming factor loop from: {self.checkpoint_path}")
                return self.get_factor_loop(resume=True)
            elif mode == "model":
                logger.info(f"Resuming model loop from: {self.checkpoint_path}")
                return self.get_model_loop(resume=True)
            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'factor' or 'model'.")
        except Exception as e:
            raise OfficialIntegrationError(
                f"Failed to resume from checkpoint: {e}"
            ) from e
    
    def get_trace(self):
        """
        获取当前Trace (从FactorLoop或ModelLoop)
        
        Returns:
            Trace对象,如果没有运行过则返回None
        """
        if self._factor_loop is not None:
            return self._factor_loop.trace
        elif self._model_loop is not None:
            return self._model_loop.trace
        return None
    
    def reset(self):
        """重置所有Loop状态"""
        self._factor_loop = None
        self._model_loop = None
        logger.info("All loops reset")


def create_official_manager(config: Dict[str, Any]) -> OfficialRDAgentManager:
    """
    工厂函数: 创建OfficialRDAgentManager
    
    Args:
        config: 配置字典
        
    Returns:
        OfficialRDAgentManager实例
        
    Raises:
        ConfigValidationError: 配置无效
        OfficialIntegrationError: 初始化失败
    """
    # 验证配置
    OfficialRDAgentManager.validate_config(config)
    
    # 创建管理器
    manager = OfficialRDAgentManager(config)
    
    return manager


# 测试代码
if __name__ == "__main__":
    """
    测试官方集成
    
    运行方式:
        python rd_agent/official_integration.py
    """
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("测试官方RD-Agent集成")
    print("=" * 60)
    
    # 测试配置
    test_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4-turbo",
        # "llm_api_key": "sk-xxx",  # 从环境变量读取
        "max_iterations": 5,
    }
    
    try:
        # 创建管理器
        print("\n1. 创建OfficialRDAgentManager...")
        manager = create_official_manager(test_config)
        print("   ✅ 成功")
        
        # 测试路径配置
        print("\n2. 测试路径配置...")
        rdagent_path = PathConfig.get_rdagent_path()
        qlib_data_path = PathConfig.get_qlib_data_path()
        print(f"   RD-Agent路径: {rdagent_path}")
        print(f"   Qlib数据路径: {qlib_data_path}")
        print("   ✅ 成功")
        
        # 测试FactorLoop创建 (懒加载)
        print("\n3. 测试FactorLoop创建 (懒加载)...")
        factor_loop = manager.get_factor_loop()
        print(f"   FactorLoop类型: {type(factor_loop).__name__}")
        print("   ✅ 成功")
        
        # 测试ModelLoop创建 (懒加载)
        print("\n4. 测试ModelLoop创建 (懒加载)...")
        model_loop = manager.get_model_loop()
        print(f"   ModelLoop类型: {type(model_loop).__name__}")
        print("   ✅ 成功")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        
    except ConfigValidationError as e:
        print(f"\n❌ 配置验证失败: {e}")
        sys.exit(1)
    except OfficialIntegrationError as e:
        print(f"\n❌ 集成失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
