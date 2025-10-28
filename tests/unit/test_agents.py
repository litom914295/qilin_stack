"""
单元测试 - 交易智能体
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.agents.trading_agents_impl import (
    MarketContext,
    ZTQualityAgent,
    LeaderAgent,
    IntegratedDecisionAgent
)

class TestZTQualityAgent:
    """涨停质量Agent测试"""
    
    @pytest.fixture
    def agent(self):
        return ZTQualityAgent()
    
    @pytest.fixture
    def mock_context(self):
        """模拟市场上下文"""
        return MarketContext(
            ohlcv=pd.DataFrame({
                'close': [10, 10.5, 11, 11.5, 12],
                'volume': [1000, 1200, 1500, 1800, 2000],
                'turnover_rate': [5, 6, 8, 10, 12]
            }),
            news_titles=['测试新闻1', '测试新闻2'],
            lhb_netbuy=2.5,
            market_mood_score=65,
            sector_heat={'sector_change': 3.5},
            money_flow={'000001_main': 1.8},
            technical_indicators={
                'seal_ratio': 0.08,
                'zt_time': '10:30',
                'open_times': 1
            },
            fundamental_data={}
        )
    
    @pytest.mark.asyncio
    async def test_analyze(self, agent, mock_context):
        """测试分析功能"""
        result = await agent.analyze('000001', mock_context)
        
        assert 'score' in result
        assert 'details' in result
        assert 'timestamp' in result
        assert 0 <= result['score'] <= 100
        
    @pytest.mark.asyncio
    async def test_high_quality_zt(self, agent, mock_context):
        """测试高质量涨停"""
        # 设置高质量涨停参数
        mock_context.technical_indicators['seal_ratio'] = 0.15
        mock_context.technical_indicators['zt_time'] = '09:35'
        mock_context.technical_indicators['open_times'] = 0
        
        result = await agent.analyze('000001', mock_context)
        
        # 高质量涨停应该得高分
        assert result['score'] > 70


class TestLeaderAgent:
    """龙头识别Agent测试"""
    
    @pytest.fixture
    def agent(self):
        return LeaderAgent()
    
    @pytest.mark.asyncio
    async def test_leader_identification(self, agent):
        """测试龙头识别"""
        context = MarketContext(
            ohlcv=pd.DataFrame({'close': [10, 11], 'volume': [1000, 1200]}),
            news_titles=['000001强势涨停', '000001引领板块'],
            lhb_netbuy=5.0,
            market_mood_score=70,
            sector_heat={'000001_rank': 1},
            money_flow={'000001': 8.5},
            technical_indicators={'consecutive_limit': 3},
            fundamental_data={'history_leader_times': 2}
        )
        
        result = await agent.analyze('000001', context)
        
        # 龙头特征明显应该得高分
        assert result['score'] > 60


class TestIntegratedDecisionAgent:
    """综合决策Agent测试"""
    
    @pytest.fixture
    def agent(self):
        return IntegratedDecisionAgent()
    
    @pytest.mark.asyncio
    async def test_parallel_analysis(self, agent):
        """测试并行分析"""
        context = MarketContext(
            ohlcv=pd.DataFrame({
                'close': [10, 11, 12],
                'volume': [1000, 1500, 2000],
                'turnover_rate': [5, 8, 10]
            }),
            news_titles=['利好消息'],
            lhb_netbuy=1.0,
            market_mood_score=60,
            sector_heat={},
            money_flow={},
            technical_indicators={},
            fundamental_data={}
        )
        
        result = await agent.analyze_parallel('000001', context)
        
        assert 'symbol' in result
        assert 'weighted_score' in result
        assert 'decision' in result
        assert 'details' in result
        assert result['symbol'] == '000001'
        
    @pytest.mark.asyncio 
    async def test_decision_making(self, agent):
        """测试决策生成"""
        # 创建高分场景
        context = MarketContext(
            ohlcv=pd.DataFrame({
                'close': [10, 11, 12, 13, 14],
                'volume': [1000, 2000, 3000, 4000, 5000],
                'turnover_rate': [10, 12, 15, 18, 20]
            }),
            news_titles=['重大利好'] * 5,
            lhb_netbuy=10.0,
            market_mood_score=85,
            sector_heat={'sector_change': 5.0, '000001_rank': 1},
            money_flow={'000001_main': 5.0},
            technical_indicators={
                'rsi': 70,
                'seal_ratio': 0.12,
                'zt_time': '09:30'
            },
            fundamental_data={'financial_score': 85}
        )
        
        result = await agent.analyze_parallel('000001', context)
        decision = result['decision']
        
        # 高分应该给出买入建议
        assert decision['action'] in ['buy', 'strong_buy']
        assert decision['confidence'] > 0.5


class TestAgentIntegration:
    """Agent集成测试"""
    
    @pytest.mark.asyncio
    async def test_multiple_agents(self):
        """测试多个Agent协同"""
        agents = [
            ZTQualityAgent(),
            LeaderAgent()
        ]
        
        context = MarketContext(
            ohlcv=pd.DataFrame({'close': [10, 11], 'volume': [1000, 1200]}),
            news_titles=[],
            lhb_netbuy=0,
            market_mood_score=50,
            sector_heat={},
            money_flow={},
            technical_indicators={},
            fundamental_data={}
        )
        
        results = []
        for agent in agents:
            result = await agent.analyze('000001', context)
            results.append(result)
        
        assert len(results) == 2
        assert all('score' in r for r in results)


@pytest.mark.asyncio
async def test_performance():
    """性能测试"""
    agent = IntegratedDecisionAgent()
    
    # 创建大量股票进行测试
    symbols = [f'00000{i}' for i in range(10)]
    
    context = MarketContext(
        ohlcv=pd.DataFrame({
            'close': np.random.uniform(10, 20, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'turnover_rate': np.random.uniform(1, 20, 100)
        }),
        news_titles=[],
        lhb_netbuy=0,
        market_mood_score=50,
        sector_heat={},
        money_flow={},
        technical_indicators={},
        fundamental_data={}
    )
    
    start_time = datetime.now()
    
    # 并行分析所有股票
    tasks = [agent.analyze_parallel(symbol, context) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    assert len(results) == 10
    assert elapsed < 5  # 10只股票应该在5秒内完成
    
    print(f"分析{len(symbols)}只股票耗时: {elapsed:.2f}秒")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])