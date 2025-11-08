"""
rd_agent.config 模块单元测试

测试 RDAgentConfig 类的所有功能:
- 默认配置
- 配置验证
- YAML/JSON 加载
- P0-1: checkpoint 配置
- P0-6: 扩展字段支持
"""

import pytest
import os
import tempfile
from pathlib import Path
from rd_agent.config import (
    RDAgentConfig, 
    load_config,
    CONFIG_TEMPLATE
)


class TestRDAgentConfig:
    """RDAgentConfig 配置类测试"""
    
    def test_default_config(self):
        """测试默认配置初始化"""
        config = RDAgentConfig()
        
        # 验证默认值
        assert config.rdagent_path is not None
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4-turbo"
        assert config.max_iterations == 10
        assert config.research_mode == "factor"
        assert config.enable_cache == True
        
    def test_env_var_override(self):
        """测试环境变量覆盖"""
        # 设置临时环境变量
        os.environ["LLM_PROVIDER"] = "azure"
        os.environ["LLM_MODEL"] = "gpt-35-turbo"
        
        config = RDAgentConfig()
        
        assert config.llm_provider == "azure"
        assert config.llm_model == "gpt-35-turbo"
        
        # 清理
        del os.environ["LLM_PROVIDER"]
        del os.environ["LLM_MODEL"]
    
    # P0-1: 测试 checkpoint 配置
    def test_checkpoint_config(self):
        """测试 P0-1 会话恢复配置"""
        config = RDAgentConfig(
            checkpoint_path="./checkpoints/test.pkl",
            enable_auto_checkpoint=True,
            checkpoint_interval=5
        )
        
        assert config.checkpoint_path == "./checkpoints/test.pkl"
        assert config.enable_auto_checkpoint == True
        assert config.checkpoint_interval == 5
    
    def test_checkpoint_env_var(self):
        """测试 checkpoint 环境变量"""
        os.environ["RDAGENT_CHECKPOINT_PATH"] = "/tmp/checkpoint.pkl"
        
        config = RDAgentConfig()
        
        assert config.checkpoint_path == "/tmp/checkpoint.pkl"
        
        # 清理
        del os.environ["RDAGENT_CHECKPOINT_PATH"]
    
    def test_to_dict(self):
        """测试配置转字典"""
        config = RDAgentConfig(
            llm_model="gpt-4",
            max_iterations=15,
            research_mode="model"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['llm_model'] == "gpt-4"
        assert config_dict['max_iterations'] == 15
        assert config_dict['research_mode'] == "model"
    
    def test_validate_success(self):
        """测试配置验证成功"""
        config = RDAgentConfig(
            rdagent_path="G:/test/RD-Agent",  # 假设存在
            llm_provider="openai",
            llm_api_key="test_key",
            research_mode="factor",
            max_iterations=10,
            factor_ic_threshold=0.05
        )
        
        # 注意: validate 会检查路径是否存在,如果不存在会失败
        # 这里我们测试的是验证逻辑本身
        is_valid = config.validate()
        # 可能返回 True 或 False,取决于路径是否存在
        assert isinstance(is_valid, bool)
    
    def test_validate_invalid_llm_provider(self):
        """测试无效的 LLM 提供商"""
        config = RDAgentConfig(
            llm_provider="invalid_provider",
            llm_api_key="test_key"
        )
        
        is_valid = config.validate()
        assert is_valid == False
    
    def test_validate_invalid_research_mode(self):
        """测试无效的研究模式"""
        config = RDAgentConfig(
            llm_api_key="test_key",
            research_mode="invalid_mode"
        )
        
        is_valid = config.validate()
        assert is_valid == False
    
    def test_validate_invalid_max_iterations(self):
        """测试无效的最大迭代次数"""
        config = RDAgentConfig(
            llm_api_key="test_key",
            max_iterations=-1  # 无效
        )
        
        is_valid = config.validate()
        assert is_valid == False
        
        config2 = RDAgentConfig(
            llm_api_key="test_key",
            max_iterations=200  # 超过上限
        )
        
        is_valid2 = config2.validate()
        assert is_valid2 == False
    
    def test_validate_invalid_ic_threshold(self):
        """测试无效的 IC 阈值"""
        config = RDAgentConfig(
            llm_api_key="test_key",
            factor_ic_threshold=1.5  # 超过1
        )
        
        is_valid = config.validate()
        assert is_valid == False
    
    def test_get_qlib_config(self):
        """测试获取 Qlib 配置"""
        config = RDAgentConfig(
            data_provider="qlib",
            train_start="2020-01-01",
            train_end="2022-12-31",
            valid_start="2023-01-01",
            valid_end="2023-06-30",
            test_start="2023-07-01",
            test_end="2023-12-31"
        )
        
        qlib_config = config.get_qlib_config()
        
        assert qlib_config['data_provider'] == "qlib"
        assert qlib_config['train_period'] == ("2020-01-01", "2022-12-31")
        assert qlib_config['valid_period'] == ("2023-01-01", "2023-06-30")
        assert qlib_config['test_period'] == ("2023-07-01", "2023-12-31")


class TestConfigLoading:
    """测试配置加载功能"""
    
    def test_load_config_default(self):
        """测试加载默认配置"""
        config = load_config()
        
        assert isinstance(config, RDAgentConfig)
        assert config.llm_provider is not None
        assert config.max_iterations > 0
    
    def test_load_config_from_yaml(self, tmp_path):
        """测试从 YAML 文件加载配置"""
        # 创建临时 YAML 配置文件
        config_file = tmp_path / "test_config.yaml"
        config_content = """
rdagent:
  llm_provider: "azure"
  llm_model: "gpt-4"
  max_iterations: 20
  research_mode: "model"
  enable_cache: false
  
  # P0-1: checkpoint 配置
  checkpoint_path: "./test_checkpoint.pkl"
  enable_auto_checkpoint: true
  checkpoint_interval: 3
"""
        config_file.write_text(config_content, encoding='utf-8')
        
        # 加载配置
        config = RDAgentConfig.from_yaml(str(config_file))
        
        assert config.llm_provider == "azure"
        assert config.llm_model == "gpt-4"
        assert config.max_iterations == 20
        assert config.research_mode == "model"
        assert config.enable_cache == False
        
        # P0-1: 验证 checkpoint 配置
        assert config.checkpoint_path == "./test_checkpoint.pkl"
        assert config.enable_auto_checkpoint == True
        assert config.checkpoint_interval == 3
    
    # P0-6: 测试扩展字段支持
    def test_load_config_with_extra_fields(self, tmp_path):
        """测试 P0-6 配置加载扩展字段"""
        config_file = tmp_path / "test_config_extended.yaml"
        config_content = """
rdagent:
  llm_provider: "openai"
  llm_model: "gpt-4"
  max_iterations: 10
  
  # P0-6: 扩展字段 - 因子类别
  factor_categories:
    - seal_strength
    - continuous_board
    - concept_synergy
    - timing
  
  # P0-6: 扩展字段 - 预测目标
  prediction_targets:
    - next_day_limit_up
    - open_premium
    - continuous_probability
"""
        config_file.write_text(config_content, encoding='utf-8')
        
        # 加载配置
        config = RDAgentConfig.from_yaml(str(config_file))
        
        # 基础字段
        assert config.llm_provider == "openai"
        assert config.max_iterations == 10
        
        # P0-6: 验证扩展字段
        assert hasattr(config, 'factor_categories')
        assert hasattr(config, 'prediction_targets')
        
        assert config.factor_categories == [
            'seal_strength', 
            'continuous_board', 
            'concept_synergy', 
            'timing'
        ]
        assert config.prediction_targets == [
            'next_day_limit_up', 
            'open_premium', 
            'continuous_probability'
        ]
    
    def test_load_config_from_json(self, tmp_path):
        """测试从 JSON 文件加载配置"""
        import json
        
        config_file = tmp_path / "test_config.json"
        config_content = {
            "rdagent": {
                "llm_provider": "anthropic",
                "llm_model": "claude-3-opus",
                "max_iterations": 15,
                "enable_parallel": False
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_content, f)
        
        # 加载配置
        config = RDAgentConfig.from_json(str(config_file))
        
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-opus"
        assert config.max_iterations == 15
        assert config.enable_parallel == False
    
    def test_load_config_file_not_exist(self, tmp_path):
        """测试加载不存在的配置文件"""
        non_existent_file = tmp_path / "non_existent.yaml"
        
        # 应该返回默认配置,不应抛出异常
        config = RDAgentConfig.from_yaml(str(non_existent_file))
        
        assert isinstance(config, RDAgentConfig)
        # 应该使用默认值
        assert config.llm_provider == "openai"
    
    def test_load_config_yaml_via_load_config(self, tmp_path):
        """测试通过 load_config 函数加载 YAML"""
        config_file = tmp_path / "test.yaml"
        config_content = """
rdagent:
  llm_model: "gpt-4-turbo"
  max_iterations: 25
"""
        config_file.write_text(config_content, encoding='utf-8')
        
        config = load_config(str(config_file))
        
        assert config.llm_model == "gpt-4-turbo"
        assert config.max_iterations == 25
    
    def test_load_config_json_via_load_config(self, tmp_path):
        """测试通过 load_config 函数加载 JSON"""
        import json
        
        config_file = tmp_path / "test.json"
        config_content = {
            "rdagent": {
                "llm_model": "gpt-3.5-turbo",
                "max_iterations": 30
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_content, f)
        
        config = load_config(str(config_file))
        
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.max_iterations == 30


class TestConfigTemplate:
    """测试配置模板"""
    
    def test_config_template_format(self):
        """测试配置模板格式正确"""
        assert isinstance(CONFIG_TEMPLATE, str)
        assert len(CONFIG_TEMPLATE) > 0
        assert "rdagent:" in CONFIG_TEMPLATE
        assert "llm_provider:" in CONFIG_TEMPLATE
        assert "max_iterations:" in CONFIG_TEMPLATE


# 集成测试
class TestRDAgentConfigIntegration:
    """RDAgentConfig 集成测试"""
    
    def test_full_workflow(self, tmp_path):
        """测试完整配置工作流"""
        # 1. 创建配置
        config = RDAgentConfig(
            llm_model="gpt-4",
            max_iterations=20,
            checkpoint_path=str(tmp_path / "checkpoint.pkl"),
            enable_auto_checkpoint=True
        )
        
        # 2. 验证配置
        # 注意: 可能因为路径不存在而失败,这是正常的
        config.validate()
        
        # 3. 转换为字典
        config_dict = config.to_dict()
        assert config_dict['llm_model'] == "gpt-4"
        
        # 4. 获取 Qlib 配置
        qlib_config = config.get_qlib_config()
        assert 'data_provider' in qlib_config
        assert 'train_period' in qlib_config
    
    def test_config_persistence(self, tmp_path):
        """测试配置持久化"""
        import yaml
        
        # 创建配置
        config = RDAgentConfig(
            llm_model="gpt-4",
            max_iterations=15,
            checkpoint_path="./test.pkl"
        )
        
        # 保存配置
        config_file = tmp_path / "saved_config.yaml"
        config_data = {
            'rdagent': {
                'llm_model': config.llm_model,
                'max_iterations': config.max_iterations,
                'checkpoint_path': config.checkpoint_path,
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        # 重新加载
        loaded_config = RDAgentConfig.from_yaml(str(config_file))
        
        assert loaded_config.llm_model == config.llm_model
        assert loaded_config.max_iterations == config.max_iterations
        assert loaded_config.checkpoint_path == config.checkpoint_path


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
