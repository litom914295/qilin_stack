"""
TradingAgents项目整合适配器
实现麒麟堆栈与TradingAgents项目的双向通信和协作
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# 添加tradingagents项目路径
sys.path.insert(0, str(Path("D:/test/Qlib/tradingagents")))

# 导入tradingagents组件
try:
    from tradingagents.agents import BaseAgent
    from tradingagents.llm.base import BaseLLM
    from tradingagents.tools import (
        SearchTool,
        CalculatorTool,
        ChartTool,
        DataAnalysisTool
    from tradingagents.dataflows import DataFlow
    from tradingagents.utils.logging_utils import get_logger
    TRADINGAGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import tradingagents components: {e}")
    TRADINGAGENTS_AVAILABLE = False
    BaseAgent = object  # 占位符

# 导入麒麟堆栈组件
from ..agents.trading_agents import MultiAgentManager, TradingSignal
from ..data_layer.data_access_layer import DataAccessLayer
from ..qlib_integration.qlib_engine import QlibIntegrationEngine
from ..trading.realtime_trading_system import RealtimeTradingSystem

logger = logging.getLogger(__name__)


class TradingAgentsAdapter:
    """
    TradingAgents项目整合适配器
    实现麒麟堆栈与TradingAgents的无缝集成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化适配器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self._init_components()
        
        # 注册的智能体
        self.registered_agents = {}
        
        # 数据流管理
        self.data_flows = {}
        
    def _init_components(self):
        """初始化各组件"""
        # 麒麟堆栈组件
        self.qilin_agent_manager = MultiAgentManager()
        self.data_access = DataAccessLayer(self.config.get("data", {}))
        self.qlib_engine = QlibIntegrationEngine(self.config)
        
        # TradingAgents工具
        if TRADINGAGENTS_AVAILABLE:
            self.ta_tools = {
                'search': SearchTool(),
                'calculator': CalculatorTool(),
                'chart': ChartTool(),
                'data_analysis': DataAnalysisTool()
            }
        else:
            self.ta_tools = {}
    
    def register_tradingagent(self, agent: 'BaseAgent', name: str):
        """
        注册TradingAgents项目的智能体
        
        Args:
            agent: TradingAgents的智能体实例
            name: 智能体名称
        """
        if not TRADINGAGENTS_AVAILABLE:
            self.logger.warning("TradingAgents not available, cannot register agent")
            return
            
        self.registered_agents[name] = agent
        self.logger.info(f"Registered TradingAgent: {name}")
    
    def register_qilin_agent(self, agent, name: str):
        """
        注册麒麟堆栈的智能体
        
        Args:
            agent: 麒麟堆栈的智能体实例
            name: 智能体名称
        """
        self.qilin_agent_manager.register_agent(name, agent)
        self.logger.info(f"Registered Qilin agent: {name}")
    
    async def process_with_tradingagents(self, 
                                        task: str,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用TradingAgents处理任务
        
        Args:
            task: 任务描述
            context: 上下文信息
            
        Returns:
            处理结果
        """
        if not TRADINGAGENTS_AVAILABLE:
            return {"error": "TradingAgents not available"}
        
        results = {}
        
        # 调用所有注册的TradingAgents智能体
        for name, agent in self.registered_agents.items():
            try:
                # TradingAgents的标准调用接口
                result = await agent.process(task, context)
                results[name] = result
                self.logger.info(f"TradingAgent {name} processed task successfully")
            except Exception as e:
                self.logger.error(f"Error in TradingAgent {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    async def process_with_qilin(self,
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用麒麟堆栈处理市场数据
        
        Args:
            market_data: 市场数据
            
        Returns:
            分析结果
        """
        # 调用麒麟堆栈的多智能体分析
        analysis = await self.qilin_agent_manager.analyze(market_data)
        
        return analysis
    
    async def hybrid_analysis(self,
                             market_data: Dict[str, Any],
                             task: str = "analyze market") -> Dict[str, Any]:
        """
        混合分析：同时使用TradingAgents和麒麟堆栈
        
        Args:
            market_data: 市场数据
            task: 任务描述
            
        Returns:
            综合分析结果
        """
        results = {
            'timestamp': datetime.now(),
            'qilin_analysis': {},
            'tradingagents_analysis': {},
            'consensus': None
        }
        
        # 并行执行两个系统的分析
        tasks = []
        
        # 麒麟堆栈分析
        tasks.append(self.process_with_qilin(market_data))
        
        # TradingAgents分析
        if TRADINGAGENTS_AVAILABLE:
            context = self._prepare_ta_context(market_data)
            tasks.append(self.process_with_tradingagents(task, context))
        
        # 等待所有分析完成
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        if len(results_list) > 0 and not isinstance(results_list[0], Exception):
            results['qilin_analysis'] = results_list[0]
        
        if len(results_list) > 1 and not isinstance(results_list[1], Exception):
            results['tradingagents_analysis'] = results_list[1]
        
        # 生成共识
        results['consensus'] = self._generate_consensus(results)
        
        return results
    
    def _prepare_ta_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        为TradingAgents准备上下文
        
        Args:
            market_data: 市场数据
            
        Returns:
            TradingAgents格式的上下文
        """
        context = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'tools_available': list(self.ta_tools.keys()) if TRADINGAGENTS_AVAILABLE else [],
            'data_sources': ['qlib', 'tushare', 'yahoo_finance'],
            'trading_rules': {
                'max_position': 0.3,
                'stop_loss': 0.05,
                'take_profit': 0.1
            }
        }
        
        return context
    
    def _generate_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成共识结果
        
        Args:
            results: 各系统的分析结果
            
        Returns:
            共识结果
        """
        consensus = {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reasoning': []
        }
        
        # 从麒麟分析中提取信号
        qilin_signal = None
        if 'qilin_analysis' in results and results['qilin_analysis']:
            if 'final_signal' in results['qilin_analysis']:
                qilin_signal = results['qilin_analysis']['final_signal']
                consensus['reasoning'].append(f"Qilin: {qilin_signal}")
        
        # 从TradingAgents分析中提取信号
        ta_signals = []
        if 'tradingagents_analysis' in results and results['tradingagents_analysis']:
            for agent_name, agent_result in results['tradingagents_analysis'].items():
                if isinstance(agent_result, dict) and 'signal' in agent_result:
                    ta_signals.append(agent_result['signal'])
                    consensus['reasoning'].append(f"TA-{agent_name}: {agent_result['signal']}")
        
        # 综合判断
        all_signals = []
        if qilin_signal:
            all_signals.append(qilin_signal)
        all_signals.extend(ta_signals)
        
        if all_signals:
            # 简单投票机制
            buy_count = sum(1 for s in all_signals if 'BUY' in str(s).upper())
            sell_count = sum(1 for s in all_signals if 'SELL' in str(s).upper())
            
            if buy_count > sell_count:
                consensus['signal'] = 'BUY'
                consensus['confidence'] = buy_count / len(all_signals)
            elif sell_count > buy_count:
                consensus['signal'] = 'SELL'
                consensus['confidence'] = sell_count / len(all_signals)
            else:
                consensus['signal'] = 'HOLD'
                consensus['confidence'] = 0.5
        
        return consensus
    
    async def execute_trade_with_ta_tools(self, 
                                         signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用TradingAgents的工具执行交易
        
        Args:
            signal: 交易信号
            
        Returns:
            执行结果
        """
        if not TRADINGAGENTS_AVAILABLE:
            return {"error": "TradingAgents not available"}
        
        result = {
            'status': 'pending',
            'signal': signal,
            'execution_details': {}
        }
        
        try:
            # 使用数据分析工具验证信号
            if 'data_analysis' in self.ta_tools:
                validation = await self.ta_tools['data_analysis'].analyze({
                    'signal': signal,
                    'market_conditions': await self._get_current_market_conditions()
                })
                result['validation'] = validation
            
            # 使用计算器工具计算仓位
            if 'calculator' in self.ta_tools:
                position_calc = await self.ta_tools['calculator'].calculate({
                    'type': 'position_sizing',
                    'signal_strength': signal.get('confidence', 0.5),
                    'risk_parameters': self.config.get('risk', {})
                })
                result['position'] = position_calc
            
            # 生成图表
            if 'chart' in self.ta_tools:
                chart = await self.ta_tools['chart'].generate({
                    'type': 'signal_visualization',
                    'data': signal
                })
                result['visualization'] = chart
            
            result['status'] = 'completed'
            
        except Exception as e:
            self.logger.error(f"Error executing trade with TA tools: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    async def _get_current_market_conditions(self) -> Dict[str, Any]:
        """获取当前市场状况"""
        # 从数据层获取市场数据
        market_data = await self.data_access.get_market_overview()
        
        return {
            'timestamp': datetime.now(),
            'volatility': market_data.get('volatility', 0.02),
            'volume': market_data.get('volume', 0),
            'trend': market_data.get('trend', 'neutral'),
            'sentiment': market_data.get('sentiment', 'neutral')
        }
    
    def create_unified_dashboard_data(self) -> Dict[str, Any]:
        """
        创建统一的仪表板数据
        整合两个系统的数据供Web界面使用
        
        Returns:
            仪表板数据
        """
        dashboard_data = {
            'timestamp': datetime.now(),
            'systems': {
                'qilin': {
                    'status': 'active',
                    'agents_count': len(self.qilin_agent_manager.agents),
                    'agents': list(self.qilin_agent_manager.agents.keys())
                },
                'tradingagents': {
                    'status': 'active' if TRADINGAGENTS_AVAILABLE else 'unavailable',
                    'agents_count': len(self.registered_agents),
                    'agents': list(self.registered_agents.keys()),
                    'tools': list(self.ta_tools.keys()) if TRADINGAGENTS_AVAILABLE else []
                }
            },
            'performance': self._get_combined_performance(),
            'active_signals': self._get_active_signals(),
            'risk_metrics': self._get_risk_metrics()
        }
        
        return dashboard_data
    
    def _get_combined_performance(self) -> Dict[str, Any]:
        """获取综合绩效"""
        # 这里应该从两个系统获取真实的绩效数据
        return {
            'total_return': 0.125,
            'sharpe_ratio': 1.85,
            'max_drawdown': 0.082,
            'win_rate': 0.625
        }
    
    def _get_active_signals(self) -> List[Dict[str, Any]]:
        """获取活跃信号"""
        signals = []
        
        # 从麒麟系统获取信号
        # qilin_signals = self.qilin_agent_manager.get_recent_signals()
        # signals.extend(qilin_signals)
        
        # 从TradingAgents获取信号
        # if TRADINGAGENTS_AVAILABLE:
        #     ta_signals = self._get_ta_signals()
        #     signals.extend(ta_signals)
        
        return signals
    
    def _get_risk_metrics(self) -> Dict[str, float]:
        """获取风险指标"""
        return {
            'portfolio_var': 52380.50,
            'portfolio_volatility': 0.0823,
            'correlation_risk': 0.45,
            'concentration_risk': 0.32
        }


class UnifiedTradingSystem:
    """
    统一交易系统
    整合麒麟堆栈和TradingAgents的所有功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化统一交易系统
        
        Args:
            config: 系统配置
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化适配器
        self.adapter = TradingAgentsAdapter(config)
        
        # 初始化实时交易系统
        self.trading_system = RealtimeTradingSystem(config)
        
        # 系统状态
        self.running = False
        
    async def start(self):
        """启动统一交易系统"""
        self.running = True
        self.logger.info("Starting Unified Trading System...")
        
        # 启动各子系统
        tasks = [
            self.trading_system.start(),
            self._monitor_loop(),
            self._signal_aggregation_loop()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """停止系统"""
        self.running = False
        await self.trading_system.stop()
        self.logger.info("Unified Trading System stopped")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 获取系统状态
                status = self.adapter.create_unified_dashboard_data()
                
                # 记录或推送状态
                self.logger.info(f"System status: {status}")
                
                await asyncio.sleep(60)  # 每分钟更新一次
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)
    
    async def _signal_aggregation_loop(self):
        """信号聚合循环"""
        while self.running:
            try:
                # 获取市场数据
                market_data = await self._get_market_data()
                
                # 混合分析
                analysis = await self.adapter.hybrid_analysis(market_data)
                
                # 处理共识信号
                if analysis['consensus']['signal'] != 'HOLD':
                    await self._process_consensus_signal(analysis['consensus'])
                
                await asyncio.sleep(30)  # 每30秒分析一次
                
            except Exception as e:
                self.logger.error(f"Signal aggregation error: {e}")
                await asyncio.sleep(30)
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """获取市场数据"""
        # 从数据层获取数据
        return await self.adapter.data_access.get_realtime_data(
            symbols=self.config.get('symbols', ['000001', '000002', '600000'])
    
    async def _process_consensus_signal(self, consensus: Dict[str, Any]):
        """处理共识信号"""
        self.logger.info(f"Processing consensus signal: {consensus}")
        
        # 这里可以执行实际的交易逻辑
        # await self.trading_system.execute_signal(consensus)


# 导出
__all__ = [
    'TradingAgentsAdapter',
    'UnifiedTradingSystem'
]