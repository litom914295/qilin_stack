"""
rd_agent.official_integration 模块单元测试

测试 OfficialRDAgentManager 类的所有功能:
- 初始化和配置
- sys.path 注入
- P0-1: 会话恢复 (checkpoint)
- 环境变量配置
- Loop 创建和管理
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rd_agent.official_integration import (
    OfficialRDAgentManager,
    OfficialIntegrationError,
    ConfigValidationError,
    PathConfig
)


class TestPathConfig:
    """PathConfig 辅助类测试"""
    
    def test_get_rdagent_path_from_env(self):
        """测试从环境变量获取 RD-Agent 路径"""
        os.environ["RDAGENT_PATH"] = "G:/test/RD-Agent"
        
        # 注意: 实际路径可能不存在,所以此测试仅检查逻辑
        path = PathConfig.get_rdagent_path()
        # 可能返回 None (如果路径不存在) 或者有效路径
        assert path is None or isinstance(path, Path)
        
        # 清理
        if "RDAGENT_PATH" in os.environ:
            del os.environ["RDAGENT_PATH"]
    
    def test_get_qlib_data_path(self):
        """测试获取 Qlib 数据路径"""
        path = PathConfig.get_qlib_data_path()
        
        # 可能返回 None (如果未配置) 或者有效路径
        assert path is None or isinstance(path, Path)


class TestOfficialRDAgentManager:
    """OfficialRDAgentManager 核心功能测试"""
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_initialization(self, mock_init_paths, mock_get_rdagent_path):
        """测试初始化"""
        # Mock 路径配置
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        config = {
            "llm_model": "gpt-4",
            "max_iterations": 5
        }
        
        manager = OfficialRDAgentManager(config)
        
        assert manager.config == config
        assert manager._factor_loop is None
        assert manager._model_loop is None
        assert manager._rdagent_added == True
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_sys_path_injection(self, mock_init_paths, mock_get_rdagent_path):
        """测试 sys.path 注入"""
        # Mock 返回一个测试路径
        test_rdagent_path = Path("/tmp/test_rdagent")
        mock_get_rdagent_path.return_value = test_rdagent_path
        mock_init_paths.return_value = True
        
        config = {}
        
        # 记录原始 sys.path
        original_sys_path = sys.path.copy()
        
        try:
            manager = OfficialRDAgentManager(config)
            
            # 验证 RD-Agent 路径是否添加到 sys.path
            assert str(test_rdagent_path) in sys.path
        finally:
            # 恢复 sys.path
            sys.path[:] = original_sys_path
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_environment_setup(self, mock_init_paths, mock_get_rdagent_path):
        """测试环境变量配置"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "llm_api_key": "test_api_key_12345",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_temperature": 0.8
        }
        
        # 保存原始环境变量
        original_env = {k: os.environ.get(k) for k in [
            "LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", 
            "OPENAI_BASE_URL", "LLM_TEMPERATURE"
        ]}
        
        try:
            manager = OfficialRDAgentManager(config)
            
            # 验证环境变量是否设置
            assert os.environ.get("LLM_PROVIDER") == "openai"
            assert os.environ.get("LLM_MODEL") == "gpt-4"
            assert os.environ.get("OPENAI_API_KEY") == "test_api_key_12345"
            assert os.environ.get("OPENAI_BASE_URL") == "https://api.openai.com/v1"
            assert os.environ.get("LLM_TEMPERATURE") == "0.8"
        finally:
            # 恢复环境变量
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_rdagent_path_not_found(self, mock_init_paths, mock_get_rdagent_path):
        """测试 RD-Agent 路径未找到时的错误处理"""
        mock_get_rdagent_path.return_value = None  # 路径不存在
        mock_init_paths.return_value = True
        
        config = {}
        
        with pytest.raises(OfficialIntegrationError, match="RD-Agent path not found"):
            OfficialRDAgentManager(config)


# P0-1: 会话恢复测试
class TestSessionResume:
    """P0-1: 测试会话恢复功能"""
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_checkpoint_path_from_config(self, mock_init_paths, mock_get_rdagent_path):
        """测试从配置读取 checkpoint 路径"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        checkpoint_path = "./checkpoints/test_factor.pkl"
        config = {
            "checkpoint_path": checkpoint_path,
            "max_iterations": 10
        }
        
        manager = OfficialRDAgentManager(config)
        
        assert manager.checkpoint_path == checkpoint_path
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_checkpoint_path_from_parameter(self, mock_init_paths, mock_get_rdagent_path):
        """测试从参数传入 checkpoint 路径"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        checkpoint_path = "./checkpoints/test_factor2.pkl"
        config = {}
        
        manager = OfficialRDAgentManager(config, checkpoint_path=checkpoint_path)
        
        assert manager.checkpoint_path == checkpoint_path
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    @patch('rdagent.app.qlib_rd_loop.factor.FactorRDLoop')
    def test_get_factor_loop_resume_from_checkpoint(
        self, 
        mock_factor_loop_class, 
        mock_init_paths, 
        mock_get_rdagent_path,
        tmp_path
    ):
        """测试从 checkpoint 恢复 FactorRDLoop"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        # 创建临时 checkpoint 文件
        checkpoint_file = tmp_path / "test_checkpoint.pkl"
        checkpoint_file.write_text("mock checkpoint data")
        
        # Mock FactorRDLoop.load
        mock_loaded_loop = Mock()
        mock_factor_loop_class.load.return_value = mock_loaded_loop
        
        config = {
            "checkpoint_path": str(checkpoint_file),
            "max_iterations": 10
        }
        
        # 需要 mock 导入才能避免真实导入官方 RD-Agent
        with patch.dict('sys.modules', {
            'rdagent.app.qlib_rd_loop.factor': Mock(FactorRDLoop=mock_factor_loop_class),
            'rdagent.app.qlib_rd_loop.conf': Mock()
        }):
            manager = OfficialRDAgentManager(config)
            
            # 尝试恢复
            factor_loop = manager.get_factor_loop(resume=True)
            
            # 验证是否调用了 load
            mock_factor_loop_class.load.assert_called_once_with(
                str(checkpoint_file), 
                checkout=True
            )
            
            # 验证返回的是加载的实例
            assert factor_loop == mock_loaded_loop
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    @patch('rdagent.app.qlib_rd_loop.factor.FactorRDLoop')
    @patch('rdagent.app.qlib_rd_loop.conf.FactorBasePropSetting')
    def test_get_factor_loop_checkpoint_not_exist(
        self,
        mock_setting_class,
        mock_factor_loop_class,
        mock_init_paths,
        mock_get_rdagent_path,
        tmp_path
    ):
        """测试 checkpoint 文件不存在时创建新 Loop"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        # checkpoint 文件不存在
        checkpoint_file = tmp_path / "non_existent.pkl"
        
        # Mock FactorRDLoop 构造函数
        mock_new_loop = Mock()
        mock_factor_loop_class.return_value = mock_new_loop
        mock_setting = Mock()
        mock_setting_class.return_value = mock_setting
        
        config = {
            "checkpoint_path": str(checkpoint_file),
            "max_iterations": 10
        }
        
        with patch.dict('sys.modules', {
            'rdagent.app.qlib_rd_loop.factor': Mock(FactorRDLoop=mock_factor_loop_class),
            'rdagent.app.qlib_rd_loop.conf': Mock(FactorBasePropSetting=mock_setting_class)
        }):
            manager = OfficialRDAgentManager(config)
            
            # 即使 resume=True,但文件不存在,应该创建新实例
            factor_loop = manager.get_factor_loop(resume=True)
            
            # 验证创建了新实例而不是调用 load
            assert factor_loop == mock_new_loop
            mock_factor_loop_class.assert_called_once()
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_resume_from_checkpoint_method_factor(self, mock_init_paths, mock_get_rdagent_path, tmp_path):
        """测试 resume_from_checkpoint 方法 (factor 模式)"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        checkpoint_file = tmp_path / "factor_checkpoint.pkl"
        checkpoint_file.write_text("mock data")
        
        config = {}
        
        with patch.dict('sys.modules', {
            'rdagent.app.qlib_rd_loop.factor': Mock(
                FactorRDLoop=Mock(load=Mock(return_value=Mock()))
            ),
            'rdagent.app.qlib_rd_loop.conf': Mock()
        }):
            manager = OfficialRDAgentManager(config)
            
            # 通过 resume_from_checkpoint 恢复
            loop = manager.resume_from_checkpoint(
                checkpoint_path=str(checkpoint_file),
                mode="factor"
            )
            
            assert loop is not None
            assert manager.checkpoint_path == str(checkpoint_file)
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_resume_from_checkpoint_method_model(self, mock_init_paths, mock_get_rdagent_path, tmp_path):
        """测试 resume_from_checkpoint 方法 (model 模式)"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        checkpoint_file = tmp_path / "model_checkpoint.pkl"
        checkpoint_file.write_text("mock data")
        
        config = {}
        
        with patch.dict('sys.modules', {
            'rdagent.app.qlib_rd_loop.model': Mock(
                ModelRDLoop=Mock(load=Mock(return_value=Mock()))
            ),
            'rdagent.app.qlib_rd_loop.conf': Mock()
        }):
            manager = OfficialRDAgentManager(config)
            
            loop = manager.resume_from_checkpoint(
                checkpoint_path=str(checkpoint_file),
                mode="model"
            )
            
            assert loop is not None
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_resume_from_checkpoint_invalid_mode(self, mock_init_paths, mock_get_rdagent_path):
        """测试 resume_from_checkpoint 无效模式"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        config = {"checkpoint_path": "./test.pkl"}
        manager = OfficialRDAgentManager(config)
        
        # 实际代码会将 ValueError 包装为 OfficialIntegrationError
        with pytest.raises(OfficialIntegrationError, match="Failed to resume from checkpoint"):
            manager.resume_from_checkpoint(mode="invalid_mode")
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_resume_from_checkpoint_no_path(self, mock_init_paths, mock_get_rdagent_path):
        """测试 resume_from_checkpoint 没有路径"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        config = {}  # 没有 checkpoint_path
        manager = OfficialRDAgentManager(config)
        
        with pytest.raises(OfficialIntegrationError, match="No checkpoint_path configured"):
            manager.resume_from_checkpoint()


class TestLoopCreation:
    """测试 Loop 创建和管理"""
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    @patch('rdagent.app.qlib_rd_loop.factor.FactorRDLoop')
    @patch('rdagent.app.qlib_rd_loop.conf.FactorBasePropSetting')
    def test_get_factor_loop_create_new(
        self,
        mock_setting_class,
        mock_factor_loop_class,
        mock_init_paths,
        mock_get_rdagent_path
    ):
        """测试创建新的 FactorRDLoop"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        mock_new_loop = Mock()
        mock_factor_loop_class.return_value = mock_new_loop
        mock_setting = Mock()
        mock_setting_class.return_value = mock_setting
        
        config = {"max_iterations": 15}
        
        with patch.dict('sys.modules', {
            'rdagent.app.qlib_rd_loop.factor': Mock(FactorRDLoop=mock_factor_loop_class),
            'rdagent.app.qlib_rd_loop.conf': Mock(FactorBasePropSetting=mock_setting_class)
        }):
            manager = OfficialRDAgentManager(config)
            
            # 第一次调用应该创建新实例
            loop1 = manager.get_factor_loop()
            
            assert loop1 is not None
            assert loop1 == mock_new_loop
            
            # 第二次调用应该返回相同实例 (缓存)
            loop2 = manager.get_factor_loop()
            
            assert loop1 is loop2
            # 只应该调用一次构造函数
            mock_factor_loop_class.assert_called_once()
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    @patch('rdagent.app.qlib_rd_loop.model.ModelRDLoop')
    @patch('rdagent.app.qlib_rd_loop.conf.ModelBasePropSetting')
    def test_get_model_loop_create_new(
        self,
        mock_setting_class,
        mock_model_loop_class,
        mock_init_paths,
        mock_get_rdagent_path
    ):
        """测试创建新的 ModelRDLoop"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        mock_new_loop = Mock()
        mock_model_loop_class.return_value = mock_new_loop
        mock_setting = Mock()
        mock_setting_class.return_value = mock_setting
        
        config = {"max_iterations": 20}
        
        with patch.dict('sys.modules', {
            'rdagent.app.qlib_rd_loop.model': Mock(ModelRDLoop=mock_model_loop_class),
            'rdagent.app.qlib_rd_loop.conf': Mock(ModelBasePropSetting=mock_setting_class)
        }):
            manager = OfficialRDAgentManager(config)
            
            loop1 = manager.get_model_loop()
            
            assert loop1 is not None
            assert loop1 == mock_new_loop
            
            # 第二次调用应该返回相同实例
            loop2 = manager.get_model_loop()
            
            assert loop1 is loop2


class TestConfigValidation:
    """测试配置验证"""
    
    def test_validate_config_success(self):
        """测试配置验证成功"""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "max_iterations": 10
        }
        
        # 应该不抛出异常
        OfficialRDAgentManager.validate_config(config)
    
    def test_validate_config_api_key_without_provider(self):
        """测试提供 api_key 但没有 provider"""
        config = {
            "llm_api_key": "test_key"
            # 缺少 llm_provider
        }
        
        with pytest.raises(ConfigValidationError, match="llm_provider is required"):
            OfficialRDAgentManager.validate_config(config)
    
    def test_validate_config_empty(self):
        """测试空配置"""
        config = {}
        
        # 空配置应该通过验证 (所有字段都是可选的)
        OfficialRDAgentManager.validate_config(config)


# 集成测试
class TestOfficialRDAgentManagerIntegration:
    """OfficialRDAgentManager 集成测试"""
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_full_workflow_without_checkpoint(self, mock_init_paths, mock_get_rdagent_path):
        """测试完整工作流 (不使用 checkpoint)"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "max_iterations": 10
        }
        
        # 验证配置
        OfficialRDAgentManager.validate_config(config)
        
        # 创建管理器
        manager = OfficialRDAgentManager(config)
        
        assert manager is not None
        assert manager.checkpoint_path is None
    
    @patch('rd_agent.official_integration.PathConfig.get_rdagent_path')
    @patch('rd_agent.official_integration.init_paths')
    def test_full_workflow_with_checkpoint(self, mock_init_paths, mock_get_rdagent_path, tmp_path):
        """测试完整工作流 (使用 checkpoint)"""
        mock_get_rdagent_path.return_value = Path("G:/test/RD-Agent")
        mock_init_paths.return_value = True
        
        checkpoint_file = tmp_path / "integration_test.pkl"
        
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "checkpoint_path": str(checkpoint_file),
            "enable_auto_checkpoint": True,
            "max_iterations": 10
        }
        
        # 验证配置
        OfficialRDAgentManager.validate_config(config)
        
        # 创建管理器
        manager = OfficialRDAgentManager(config)
        
        assert manager is not None
        assert manager.checkpoint_path == str(checkpoint_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
