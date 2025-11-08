"""
RD-Agent迁移测试套件

任务: P0-1.7 + P0-1.8 + P0-1.9
覆盖: 单元测试、集成测试、性能对比
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime

# 导入兼容层
import sys
sys.path.insert(0, "G:/test/qilin_stack")

from rd_agent.compat_wrapper import RDAgent, _ConfigAdapter, _ResultAdapter
from rd_agent.research_agent import ResearchHypothesis, FactorDefinition


class TestConfigAdapter:
    """P0-1.7.1: 配置转换测试"""
    
    def test_basic_config_conversion(self):
        """测试基本配置转换"""
        old_config = {
            "llm_model": "gpt-4-turbo",
            "max_iterations": 20
        }
        
        new_config = _ConfigAdapter.to_official_config(old_config)
        
        assert new_config["llm_model"] == "gpt-4-turbo"
        assert new_config["max_iterations"] == 20
    
    def test_provider_inference(self):
        """测试provider自动推断"""
        config = {"llm_model": "gpt-4"}
        new_config = _ConfigAdapter.to_official_config(config)
        assert new_config["llm_provider"] == "openai"
        
        config = {"llm_model": "claude-3"}
        new_config = _ConfigAdapter.to_official_config(config)
        assert new_config["llm_provider"] == "anthropic"


class TestRDAgentWrapper:
    """P0-1.7.2 + P0-1.8: RDAgentWrapper测试"""
    
    def test_initialization(self):
        """测试初始化"""
        config = {"llm_model": "gpt-4-turbo", "max_iterations": 5}
        agent = RDAgent(config)
        
        assert agent is not None
        assert agent.config == config
        assert agent.research_history == []
    
    def test_config_stored(self):
        """测试配置存储"""
        config = {"llm_model": "gpt-4", "max_iterations": 10}
        agent = RDAgent(config)
        
        assert agent.config["llm_model"] == "gpt-4"
        assert agent.config["max_iterations"] == 10


class TestIntegration:
    """P0-1.9: 集成测试"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_research_pipeline_basic(self):
        """测试基本研究流程"""
        config = {
            "llm_model": "gpt-4-turbo",
            "max_iterations": 1  # 快速测试
        }
        
        agent = RDAgent(config)
        df = pd.DataFrame()  # 空DataFrame,使用Qlib数据
        
        # 注意: 需要有效的API key才能真正运行
        # results = await agent.research_pipeline(
        #     research_topic="测试因子研究",
        #     data=df,
        #     max_iterations=1
        # )
        # 
        # assert isinstance(results, dict)
        # assert "factors" in results
        # assert "hypotheses" in results
        
        # 暂时只测试API签名
        assert callable(agent.research_pipeline)


@pytest.mark.skipif(
    True,  # 默认跳过,因为需要真实API key
    reason="Requires real LLM API key"
)
class TestPerformanceComparison:
    """P0-1.9: 性能对比测试"""
    
    @pytest.mark.asyncio
    async def test_factor_quality_comparison(self):
        """对比因子质量"""
        config = {"llm_model": "gpt-4-turbo", "max_iterations": 3}
        df = pd.DataFrame()
        
        # 官方版本
        agent = RDAgent(config)
        results = await agent.research_pipeline("测试", df, 3)
        
        # 验证结果格式
        assert "factors" in results
        assert "best_solution" in results
        
        print(f"✅ 发现{len(results['factors'])}个因子")
        if results['best_solution']:
            ic = results['best_solution']['performance'].get('ic', 0)
            print(f"✅ 最佳IC: {ic:.4f}")


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
