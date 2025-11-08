"""
TradingAgents-CN-Plus 完整集成适配器
真正调用原项目的完整智能体系统进行深度分析
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import logging

# 加载环境变量
try:
    from dotenv import load_dotenv
    # 加载项目根目录的.env文件
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logging.info(f"✅ 已加载环境变量: {env_file}")
except ImportError:
    logging.warning("⚠️ python-dotenv未安装，无法自动加载.env文件")

logger = logging.getLogger(__name__)


def _check_module_available(module_name: str) -> bool:
    """检查模块是否可用"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


class TradingAgentsCNPlusAdapter:
    """TradingAgents-CN-Plus完整系统适配器"""
    
    def __init__(self, 
                 tradingagents_path: str = "G:/test/tradingagents-cn-plus",
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化适配器
        
        Args:
            tradingagents_path: TradingAgents-CN-Plus项目路径
            config: 配置字典
        """
        self.tradingagents_path = Path(tradingagents_path)
        self.config = config or {}
        self.graph = None
        self.initialization_error = None
        
        # 检查路径是否存在
        if not self.tradingagents_path.exists():
            error_msg = (
                f"TradingAgents-CN-Plus项目路径不存在: {self.tradingagents_path}\n"
                f"请执行以下命令克隆项目:\n"
                f"git clone https://github.com/your-repo/tradingagents-cn-plus.git {self.tradingagents_path}"
            )
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            return
        
        # 添加到Python路径
        if str(self.tradingagents_path) not in sys.path:
            sys.path.insert(0, str(self.tradingagents_path))
        
        logger.info(f"✅ TradingAgents-CN-Plus路径已添加: {self.tradingagents_path}")
        
        # 初始化图
        try:
            self._initialize_graph()
        except Exception as e:
            self.initialization_error = str(e)
            logger.warning(f"⚠️ 初始化失败，适配器将以降级模式运行")
    
    def _initialize_graph(self):
        """初始化TradingAgentsGraph"""
        try:
            # 首先检查关键依赖
            missing_deps = self._check_dependencies()
            if missing_deps:
                error_msg = f"缺少以下依赖包: {', '.join(missing_deps)}\n\n"
                error_msg += "请执行以下命令安装:\n"
                error_msg += f"cd {self.tradingagents_path}\n"
                error_msg += "pip install -e .\n\n"
                error_msg += "或者安装必需的依赖:\n"
                error_msg += f"pip install {' '.join(missing_deps)}"
                logger.error(f"❌ 依赖检查失败:\n{error_msg}")
                raise ImportError(error_msg)
            
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents.default_config import DEFAULT_CONFIG
            
            # 合并配置
            graph_config = DEFAULT_CONFIG.copy()
            graph_config.update(self.config)
            
            # 强制覆盖配置（优先使用环境变量）
            graph_config["llm_provider"] = os.getenv("LLM_PROVIDER", "google")
            graph_config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "gemini-2.0-flash")
            graph_config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "gemini-2.0-flash")
            graph_config["max_debate_rounds"] = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))
            graph_config["online_tools"] = True
            
            # 如果使用Google，确保API基地址正确
            if graph_config["llm_provider"] == "google":
                # Google不需要backend_url，使用官方API
                graph_config.pop("backend_url", None)
            
            # 创建图实例
            self.graph = TradingAgentsGraph(
                selected_analysts=["market", "fundamentals", "news", "social"],
                debug=True,
                config=graph_config
            )
            
            logger.info("✅ TradingAgentsGraph初始化成功")
            logger.info(f"   - LLM Provider: {graph_config['llm_provider']}")
            logger.info(f"   - 深度思考模型: {graph_config['deep_think_llm']}")
            logger.info(f"   - 快速思考模型: {graph_config['quick_think_llm']}")
            
        except Exception as e:
            logger.error(f"❌ TradingAgentsGraph初始化失败: {e}")
            raise
    
    async def analyze_stock_full(self, 
                                 symbol: str,
                                 date: Optional[str] = None) -> Dict[str, Any]:
        """
        完整分析股票（调用原项目的完整流程）
        
        Args:
            symbol: 股票代码（支持中国A股代码，如 000001）
            date: 分析日期（格式：YYYY-MM-DD），默认为今天
            
        Returns:
            包含完整分析结果的字典
        """
        if not self.graph:
            raise RuntimeError("TradingAgentsGraph未初始化")
        
        # 转换日期格式
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # 转换股票代码格式（如果需要）
        symbol_converted = self._convert_symbol(symbol)
        
        logger.info(f"🔬 开始完整分析: {symbol} ({symbol_converted}) at {date}")
        
        try:
            # 在线程池中运行同步的propagate方法
            loop = asyncio.get_event_loop()
            state, decision = await loop.run_in_executor(
                None, 
                self.graph.propagate,
                symbol_converted,
                date
            )
            
            logger.info(f"✅ 分析完成: {symbol}")
            
            # 转换结果格式以兼容现有接口
            result = self._format_result(state, decision, symbol)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 分析失败: {symbol} - {e}")
            raise
    
    def _convert_symbol(self, symbol: str) -> str:
        """
        转换股票代码格式
        
        原项目可能需要特定格式，如：
        - A股: 000001.SZ, 600000.SH
        - 美股: AAPL, NVDA
        """
        symbol = symbol.strip().upper()
        
        # 如果是6位纯数字（A股）
        if symbol.isdigit() and len(symbol) == 6:
            # 判断上海还是深圳
            if symbol.startswith('6'):
                return f"{symbol}.SH"
            else:
                return f"{symbol}.SZ"
        
        # 如果已经包含市场后缀
        if '.SH' in symbol or '.SZ' in symbol:
            return symbol
        
        # 美股或其他，保持原样
        return symbol
    
    def _format_result(self, 
                      state: Dict[str, Any],
                      decision: Dict[str, Any],
                      original_symbol: str) -> Dict[str, Any]:
        """
        将原项目的结果格式转换为统一格式
        
        Args:
            state: 原项目的state对象
            decision: 原项目的decision对象
            original_symbol: 原始股票代码
            
        Returns:
            统一格式的结果字典
        """
        
        # 提取决策信息
        action = decision.get('action', 'HOLD')
        confidence = decision.get('confidence', 0.5)
        target_price = decision.get('target_price', 'N/A')
        reasoning = decision.get('reasoning', '')
        risk_score = decision.get('risk_score', 0.5)
        
        # 提取各智能体的分析结果
        individual_results = []
        
        # 1. 市场技术分析师
        if 'market_report' in state and state['market_report']:
            individual_results.append({
                'agent': '市场技术分析师',
                'signal': self._extract_signal_from_report(state['market_report'], action),
                'confidence': confidence,
                'reasoning': state['market_report']
            })
        
        # 2. 基本面分析师
        if 'fundamentals_report' in state and state['fundamentals_report']:
            individual_results.append({
                'agent': '基本面分析师',
                'signal': self._extract_signal_from_report(state['fundamentals_report'], action),
                'confidence': confidence,
                'reasoning': state['fundamentals_report']
            })
        
        # 3. 新闻分析师
        if 'news_report' in state and state['news_report']:
            individual_results.append({
                'agent': '新闻分析师',
                'signal': self._extract_signal_from_report(state['news_report'], action),
                'confidence': confidence,
                'reasoning': state['news_report']
            })
        
        # 4. 社交媒体分析师
        if 'sentiment_report' in state and state['sentiment_report']:
            individual_results.append({
                'agent': '社交媒体情绪分析师',
                'signal': self._extract_signal_from_report(state['sentiment_report'], action),
                'confidence': confidence,
                'reasoning': state['sentiment_report']
            })
        
        # 5. 多头研究员
        if 'investment_debate_state' in state and state['investment_debate_state']:
            debate_state = state['investment_debate_state']
            if debate_state.get('bull_history'):
                individual_results.append({
                    'agent': '多头研究员',
                    'signal': 'BUY',
                    'confidence': 0.8,
                    'reasoning': debate_state['bull_history']
                })
        
        # 6. 空头研究员
        if 'investment_debate_state' in state and state['investment_debate_state']:
            debate_state = state['investment_debate_state']
            if debate_state.get('bear_history'):
                individual_results.append({
                    'agent': '空头研究员',
                    'signal': 'SELL',
                    'confidence': 0.8,
                    'reasoning': debate_state['bear_history']
                })
        
        # 7. 研究经理
        if 'investment_debate_state' in state and state['investment_debate_state']:
            debate_state = state['investment_debate_state']
            if debate_state.get('judge_decision'):
                individual_results.append({
                    'agent': '研究经理',
                    'signal': action,
                    'confidence': confidence,
                    'reasoning': debate_state['judge_decision']
                })
        
        # 8. 风险管理团队
        if 'risk_assessment' in state and state['risk_assessment']:
            individual_results.append({
                'agent': '风险管理团队',
                'signal': 'HOLD' if risk_score > 0.6 else action,
                'confidence': 1 - risk_score,
                'reasoning': state['risk_assessment']
            })
        
        # 构建统一格式的返回结果
        result = {
            'consensus': {
                'signal': action,
                'confidence': confidence,
                'reasoning': reasoning
            },
            'individual_results': individual_results,
            'symbol': original_symbol,
            'timestamp': datetime.now().isoformat(),
            
            # 保留原始详细数据
            'detailed_analysis': {
                'target_price': target_price,
                'risk_score': risk_score,
                'market_report': state.get('market_report', ''),
                'fundamentals_report': state.get('fundamentals_report', ''),
                'news_report': state.get('news_report', ''),
                'sentiment_report': state.get('sentiment_report', ''),
                'risk_assessment': state.get('risk_assessment', ''),
                'investment_plan': state.get('investment_plan', ''),
                'investment_debate_state': state.get('investment_debate_state', {}),
                'risk_debate_state': state.get('risk_debate_state', {}),
                'trader_investment_plan': state.get('trader_investment_plan', ''),
                'final_trade_decision': state.get('final_trade_decision', '')
            },
            
            # 元数据
            'metadata': {
                'is_full_analysis': True,
                'analysis_mode': 'TradingAgents-CN-Plus完整流程',
                'analysts_count': len(individual_results)
            }
        }
        
        return result
    
    def _extract_signal_from_report(self, report: str, default_signal: str) -> str:
        """从报告文本中提取交易信号"""
        report_lower = report.lower()
        
        # 简单的关键词匹配
        if 'buy' in report_lower or '买入' in report_lower or '看涨' in report_lower:
            return 'BUY'
        elif 'sell' in report_lower or '卖出' in report_lower or '看跌' in report_lower:
            return 'SELL'
        elif 'hold' in report_lower or '持有' in report_lower or '观望' in report_lower:
            return 'HOLD'
        else:
            return default_signal
    
    def _check_dependencies(self) -> List[str]:
        """检查必需的依赖包"""
        required_deps = [
            'langgraph',
            'langchain_anthropic',
            'langchain_openai',
            'akshare',
            'yfinance',
            'pandas'
        ]
        
        missing = []
        for dep in required_deps:
            if not _check_module_available(dep):
                missing.append(dep)
        
        return missing
    
    def get_status(self) -> Dict[str, Any]:
        """获取适配器状态"""
        status = {
            'available': self.graph is not None,
            'mode': 'tradingagents_cn_plus_full',
            'project_path': str(self.tradingagents_path),
            'config': self.config
        }
        
        if self.initialization_error:
            status['error'] = self.initialization_error
            status['available'] = False
        
        return status


def create_tradingagents_cn_plus_adapter(
    tradingagents_path: str = "G:/test/tradingagents-cn-plus",
    config: Optional[Dict[str, Any]] = None
) -> TradingAgentsCNPlusAdapter:
    """
    创建TradingAgents-CN-Plus适配器的工厂函数
    
    Args:
        tradingagents_path: 项目路径
        config: 配置字典
        
    Returns:
        适配器实例
    """
    return TradingAgentsCNPlusAdapter(tradingagents_path, config)
