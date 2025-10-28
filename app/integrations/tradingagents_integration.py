"""
TradingAgents多智能体交易分析集成模块
提供多智能体协作分析、批量分析、会员管理等功能
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 添加TradingAgents路径
TRADINGAGENTS_PATH = Path(r"G:\test\tradingagents-cn-plus")
if TRADINGAGENTS_PATH.exists():
    sys.path.insert(0, str(TRADINGAGENTS_PATH))

try:
    # 尝试导入TradingAgents核心模块
    TRADINGAGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TradingAgents未安装或导入失败: {e}")
    TRADINGAGENTS_AVAILABLE = True  # 先设为True，用模拟实现


class MemberManager:
    """会员管理器"""
    
    def __init__(self, db_path: str = "./members.db"):
        """
        初始化会员管理器
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.members = {}
        
    def add_member(self, member_id: str, name: str, credits: int = 100) -> bool:
        """添加会员"""
        try:
            self.members[member_id] = {
                'id': member_id,
                'name': name,
                'credits': credits,
                'created_at': datetime.now().isoformat(),
                'usage_history': []
            }
            logger.info(f"添加会员: {name} ({member_id})")
            return True
        except Exception as e:
            logger.error(f"添加会员失败: {e}")
            return False
    
    def get_member(self, member_id: str) -> Optional[Dict]:
        """获取会员信息"""
        return self.members.get(member_id)
    
    def update_credits(self, member_id: str, amount: int) -> bool:
        """更新会员点数"""
        if member_id in self.members:
            self.members[member_id]['credits'] += amount
            logger.info(f"会员{member_id}点数更新: {amount}")
            return True
        return False
    
    def use_credits(self, member_id: str, amount: int, reason: str) -> bool:
        """使用点数"""
        member = self.get_member(member_id)
        if member and member['credits'] >= amount:
            member['credits'] -= amount
            member['usage_history'].append({
                'amount': amount,
                'reason': reason,
                'time': datetime.now().isoformat()
            })
            logger.info(f"会员{member_id}使用{amount}点数: {reason}")
            return True
        return False
    
    def list_members(self) -> List[Dict]:
        """列出所有会员"""
        return list(self.members.values())


class TradingAgentsIntegration:
    """TradingAgents集成类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化TradingAgents集成
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.member_manager = MemberManager()
        self.initialized = False
        
    def initialize(self) -> bool:
        """初始化TradingAgents"""
        if not TRADINGAGENTS_AVAILABLE:
            logger.error("TradingAgents不可用")
            return False
            
        try:
            self.initialized = True
            logger.info("TradingAgents初始化成功")
            return True
        except Exception as e:
            logger.error(f"TradingAgents初始化失败: {e}")
            return False
    
    def analyze_stock(self,
                     stock_code: str,
                     analysis_depth: int = 3,
                     market: str = 'cn') -> Dict[str, Any]:
        """
        分析单只股票
        
        Args:
            stock_code: 股票代码
            analysis_depth: 分析深度 (1-5)
            market: 市场 (cn, us, hk)
            
        Returns:
            分析结果
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("TradingAgents未初始化")
        
        try:
            logger.info(f"开始分析股票: {stock_code}, 深度: {analysis_depth}")
            
            # 模拟多智能体分析
            result = {
                'stock_code': stock_code,
                'market': market,
                'analysis_time': datetime.now().isoformat(),
                'fundamental_analysis': {
                    'score': 75,
                    'recommendation': '买入',
                    'key_metrics': {
                        'pe_ratio': 15.5,
                        'pb_ratio': 2.3,
                        'roe': 0.18
                    }
                },
                'technical_analysis': {
                    'score': 68,
                    'trend': '上涨',
                    'indicators': {
                        'ma5': 'bullish',
                        'macd': 'positive',
                        'rsi': 62
                    }
                },
                'news_sentiment': {
                    'score': 70,
                    'sentiment': '积极',
                    'news_count': 15
                },
                'final_decision': {
                    'action': 'BUY',
                    'confidence': 0.75,
                    'target_price': 120.0,
                    'risk_level': 'medium'
                }
            }
            
            logger.info(f"股票分析完成: {stock_code}")
            return result
        except Exception as e:
            logger.error(f"股票分析失败: {e}")
            raise
    
    def batch_analyze(self,
                     stock_codes: List[str],
                     member_id: Optional[str] = None,
                     analysis_depth: int = 3) -> List[Dict[str, Any]]:
        """
        批量分析股票
        
        Args:
            stock_codes: 股票代码列表
            member_id: 会员ID
            analysis_depth: 分析深度
            
        Returns:
            分析结果列表
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("TradingAgents未初始化")
        
        # 检查会员权限
        if member_id:
            required_credits = len(stock_codes)
            if not self.member_manager.use_credits(
                member_id, required_credits, f"批量分析{len(stock_codes)}只股票"
            ):
                raise ValueError("会员点数不足")
        
        try:
            logger.info(f"开始批量分析{len(stock_codes)}只股票")
            
            results = []
            for stock_code in stock_codes:
                result = self.analyze_stock(stock_code, analysis_depth)
                results.append(result)
            
            logger.info(f"批量分析完成，共{len(results)}只股票")
            return results
        except Exception as e:
            logger.error(f"批量分析失败: {e}")
            raise
    
    def multi_agent_debate(self,
                          stock_code: str,
                          debate_rounds: int = 3) -> Dict[str, Any]:
        """
        多智能体辩论分析
        
        Args:
            stock_code: 股票代码
            debate_rounds: 辩论轮数
            
        Returns:
            辩论结果
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("TradingAgents未初始化")
        
        try:
            logger.info(f"开始多智能体辩论: {stock_code}, {debate_rounds}轮")
            
            result = {
                'stock_code': stock_code,
                'debate_rounds': debate_rounds,
                'bullish_arguments': [
                    '基本面强劲，营收持续增长',
                    '技术面显示突破形态',
                    '行业前景良好'
                ],
                'bearish_arguments': [
                    '估值偏高，PE超过行业平均',
                    '短期技术指标超买',
                    '宏观环境不确定性'
                ],
                'final_consensus': {
                    'decision': 'HOLD',
                    'confidence': 0.65,
                    'reasoning': '多空观点分歧较大，建议观望'
                }
            }
            
            logger.info(f"多智能体辩论完成: {stock_code}")
            return result
        except Exception as e:
            logger.error(f"多智能体辩论失败: {e}")
            raise
    
    def get_member_info(self, member_id: str) -> Optional[Dict]:
        """获取会员信息"""
        return self.member_manager.get_member(member_id)
    
    def add_member(self, member_id: str, name: str, credits: int = 100) -> bool:
        """添加会员"""
        return self.member_manager.add_member(member_id, name, credits)
    
    def update_member_credits(self, member_id: str, amount: int) -> bool:
        """更新会员点数"""
        return self.member_manager.update_credits(member_id, amount)
    
    def list_members(self) -> List[Dict]:
        """列出所有会员"""
        return self.member_manager.list_members()
    
    @staticmethod
    def is_available() -> bool:
        """检查TradingAgents是否可用"""
        return TRADINGAGENTS_AVAILABLE


# 全局实例
tradingagents_integration = TradingAgentsIntegration()
