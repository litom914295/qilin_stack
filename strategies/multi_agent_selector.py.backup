"""
"""多智能体股票筛选系统 - 简化版

整合核心智能体进行综合选股评分:
1. 缠论智能体 (ChanLunAgent) - 70% 权重
2. 基本面智能体 (FundamentalAgent) - 30% 权重

注意:
- TechnicalAgent/VolumeAgent/SentimentAgent 已移除
- 这些功能应通过 Qlib Alpha191 因子实现
- 推荐使用 ChanLunEnhancedStrategy 替代本类

作者: Warp AI Assistant
日期: 2025年1月
项目: 麒麟量化系统 - Phase 2 重构后
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# 导入缠论智能体
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.chanlun_agent import ChanLunScoringAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentScore:
    """智能体评分结果"""
    agent_name: str
    score: float  # 0-100分
    confidence: float  # 0-1置信度
    signals: Dict[str, Any]  # 详细信号
    explanation: str


# TechnicalAgent/VolumeAgent/SentimentAgent 已删除
# 这些功能应通过 Qlib Alpha191 因子实现
# 推荐使用 ChanLunEnhancedStrategy 策略类

# 保留核心智能体:
            confidence = min(current_vol / avg_vol20, 2.0) / 2.0
            
            explanation = ", ".join([f"{k}:{v}" for k, v in signals.items()])
            if not explanation:
                explanation = f"量比={current_vol/avg_vol20:.2f}"
            
            return AgentScore(
                agent_name="Volume",
                score=max(0, min(100, score)),
                confidence=confidence,
                signals=signals,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Volume agent error for {code}: {e}")
            return AgentScore("Volume", 50.0, 0.0, {}, f"评分失败: {e}")


class FundamentalAgent:
    """基本面智能体
    
    基于基本面指标评分 (简化版):
    - PE估值
    - 市净率
    - ROE
    - 增长率
    """
    
    def score(self, df: pd.DataFrame, code: str = None,
             fundamentals: Dict[str, float] = None) -> AgentScore:
        """基本面评分
        
        参数:
            df: 价格数据
            code: 股票代码
            fundamentals: 基本面数据字典 {'pe': 15.5, 'pb': 2.0, 'roe': 0.15, ...}
        """
        if fundamentals is None or not fundamentals:
            return AgentScore("Fundamental", 50.0, 0.3, {}, "无基本面数据")
        
        try:
            scores = {}
            signals = {}
            
            # 1. PE估值
            if 'pe' in fundamentals:
                pe = fundamentals['pe']
                if 0 < pe < 15:
                    scores['pe'] = 80
                    signals['pe'] = f"低估值({pe:.1f})"
                elif 15 <= pe < 30:
                    scores['pe'] = 60
                    signals['pe'] = f"合理估值({pe:.1f})"
                elif 30 <= pe < 50:
                    scores['pe'] = 40
                    signals['pe'] = f"偏高估值({pe:.1f})"
                else:
                    scores['pe'] = 20
                    signals['pe'] = f"高估值({pe:.1f})"
            
            # 2. 市净率
            if 'pb' in fundamentals:
                pb = fundamentals['pb']
                if 0 < pb < 1.5:
                    scores['pb'] = 80
                    signals['pb'] = f"低PB({pb:.2f})"
                elif 1.5 <= pb < 3:
                    scores['pb'] = 60
                    signals['pb'] = f"合理PB({pb:.2f})"
                else:
                    scores['pb'] = 40
                    signals['pb'] = f"高PB({pb:.2f})"
            
            # 3. ROE
            if 'roe' in fundamentals:
                roe = fundamentals['roe'] * 100
                if roe > 20:
                    scores['roe'] = 90
                    signals['roe'] = f"高ROE({roe:.1f}%)"
                elif roe > 15:
                    scores['roe'] = 70
                    signals['roe'] = f"良好ROE({roe:.1f}%)"
                elif roe > 10:
                    scores['roe'] = 50
                    signals['roe'] = f"一般ROE({roe:.1f}%)"
                else:
                    scores['roe'] = 30
                    signals['roe'] = f"低ROE({roe:.1f}%)"
            
            # 加权平均
            if scores:
                total_score = np.mean(list(scores.values()))
                confidence = len(scores) / 3.0  # 最多3个指标
            else:
                total_score = 50.0
                confidence = 0.1
            
            explanation = ", ".join([f"{k}:{v}" for k, v in signals.items()])
            
            return AgentScore(
                agent_name="Fundamental",
                score=total_score,
                confidence=confidence,
                signals=signals,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Fundamental agent error for {code}: {e}")
            return AgentScore("Fundamental", 50.0, 0.0, {}, f"评分失败: {e}")


class SentimentAgent:
    """市场情绪智能体
    
    基于市场情绪指标评分:
    - 涨跌幅
    - 换手率
    - 振幅
    - 资金流向
    """
    
    def score(self, df: pd.DataFrame, code: str = None) -> AgentScore:
        """情绪评分"""
        try:
            if len(df) < 5:
                return AgentScore("Sentiment", 50.0, 0.0, {}, "数据不足")
            
            signals = {}
            score = 50.0
            
            # 1. 短期涨跌幅
            returns_5d = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
            
            if returns_5d > 20:
                score += 20
                signals['momentum'] = f"强势({returns_5d:.1f}%)"
            elif returns_5d > 10:
                score += 10
                signals['momentum'] = f"上涨({returns_5d:.1f}%)"
            elif returns_5d < -10:
                score -= 15
                signals['momentum'] = f"弱势({returns_5d:.1f}%)"
            
            # 2. 振幅 (波动性)
            if 'high' in df.columns and 'low' in df.columns:
                amplitude = (df['high'].iloc[-1] / df['low'].iloc[-1] - 1) * 100
                if amplitude > 8:
                    score += 5
                    signals['amplitude'] = f"高波动({amplitude:.1f}%)"
                elif amplitude < 2:
                    score -= 5
                    signals['amplitude'] = f"低波动({amplitude:.1f}%)"
            
            # 3. 连涨/连跌
            if len(df) >= 3:
                recent_changes = df['close'].diff().iloc[-3:]
                if all(recent_changes > 0):
                    score += 15
                    signals['streak'] = "三连阳"
                elif all(recent_changes < 0):
                    score -= 15
                    signals['streak'] = "三连阴"
            
            score = max(0, min(100, score))
            confidence = min(abs(returns_5d) / 20, 1.0)
            
            explanation = ", ".join([f"{k}:{v}" for k, v in signals.items()])
            if not explanation:
                explanation = f"5日涨幅={returns_5d:.1f}%"
            
            return AgentScore(
                agent_name="Sentiment",
                score=score,
                confidence=confidence,
                signals=signals,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Sentiment agent error for {code}: {e}")
            return AgentScore("Sentiment", 50.0, 0.0, {}, f"评分失败: {e}")


class MultiAgentStockSelector:
    """多智能体股票筛选系统
    
    整合5个智能体进行综合评分和选股
    """
    
    def __init__(self,
                 chanlun_weight: float = 0.35,
                 technical_weight: float = 0.25,
                 volume_weight: float = 0.15,
                 fundamental_weight: float = 0.15,
                 sentiment_weight: float = 0.10,
                 enable_chanlun: bool = True,
                 enable_technical: bool = True,
                 enable_volume: bool = True,
                 enable_fundamental: bool = True,
                 enable_sentiment: bool = True):
        """
        参数:
            chanlun_weight: 缠论智能体权重
            technical_weight: 技术指标智能体权重
            volume_weight: 成交量智能体权重
            fundamental_weight: 基本面智能体权重
            sentiment_weight: 市场情绪智能体权重
            enable_*: 各智能体开关
        """
        self.weights = {
            'chanlun': chanlun_weight if enable_chanlun else 0,
            'technical': technical_weight if enable_technical else 0,
            'volume': volume_weight if enable_volume else 0,
            'fundamental': fundamental_weight if enable_fundamental else 0,
            'sentiment': sentiment_weight if enable_sentiment else 0
        }
        
        # 权重归一化
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # 初始化各智能体
        self.agents = {}
        if enable_chanlun:
            self.agents['chanlun'] = ChanLunScoringAgent()
        if enable_technical:
            self.agents['technical'] = TechnicalAgent()
        if enable_volume:
            self.agents['volume'] = VolumeAgent()
        if enable_fundamental:
            self.agents['fundamental'] = FundamentalAgent()
        if enable_sentiment:
            self.agents['sentiment'] = SentimentAgent()
        
        logger.info(f"MultiAgentStockSelector initialized with weights: {self.weights}")
    
    def score(self, df: pd.DataFrame, code: str = None,
             fundamentals: Dict[str, float] = None,
             return_details: bool = False) -> Any:
        """单股票综合评分
        
        参数:
            df: 股票价格数据
            code: 股票代码
            fundamentals: 基本面数据
            return_details: 是否返回详细评分
            
        返回:
            float或Tuple[float, Dict]: 总分或(总分, 详细信息)
        """
        agent_scores = {}
        
        # 1. 缠论评分
        if 'chanlun' in self.agents:
            agent_scores['chanlun'] = self.agents['chanlun'].score(df, code)
        
        # 2. 技术指标评分
        if 'technical' in self.agents:
            agent_scores['technical'] = self.agents['technical'].score(df, code)
        
        # 3. 成交量评分
        if 'volume' in self.agents:
            agent_scores['volume'] = self.agents['volume'].score(df, code)
        
        # 4. 基本面评分
        if 'fundamental' in self.agents:
            agent_scores['fundamental'] = self.agents['fundamental'].score(
                df, code, fundamentals
            )
        
        # 5. 市场情绪评分
        if 'sentiment' in self.agents:
            agent_scores['sentiment'] = self.agents['sentiment'].score(df, code)
        
        # 加权融合
        total_score = 0.0
        total_confidence = 0.0
        
        for name, weight in self.weights.items():
            if name in agent_scores:
                if isinstance(agent_scores[name], AgentScore):
                    score = agent_scores[name].score
                    conf = agent_scores[name].confidence
                else:
                    score = agent_scores[name]
                    conf = 1.0
                
                total_score += score * weight
                total_confidence += conf * weight
        
        if return_details:
            details = {
                'total_score': total_score,
                'confidence': total_confidence,
                'agent_scores': agent_scores,
                'weights': self.weights,
                'grade': self._get_grade(total_score)
            }
            return total_score, details
        
        return total_score
    
    def batch_score(self, stock_data: Dict[str, pd.DataFrame],
                   fundamentals_data: Dict[str, Dict] = None,
                   top_n: int = None) -> pd.DataFrame:
        """批量评分
        
        参数:
            stock_data: {code: DataFrame}
            fundamentals_data: {code: {pe: ..., pb: ...}}
            top_n: 返回前N名
            
        返回:
            DataFrame with columns: code, score, grade, chanlun, technical, ...
        """
        results = []
        
        for code, df in stock_data.items():
            fundamentals = fundamentals_data.get(code) if fundamentals_data else None
            
            try:
                total_score, details = self.score(df, code, fundamentals, return_details=True)
                
                row = {
                    'code': code,
                    'score': total_score,
                    'confidence': details['confidence'],
                    'grade': details['grade']
                }
                
                # 添加各智能体评分
                for name, agent_score in details['agent_scores'].items():
                    if isinstance(agent_score, AgentScore):
                        row[f'{name}_score'] = agent_score.score
                        row[f'{name}_signal'] = agent_score.explanation
                    else:
                        row[f'{name}_score'] = agent_score
                
                results.append(row)
                
            except Exception as e:
                logger.error(f"Batch scoring error for {code}: {e}")
        
        df_result = pd.DataFrame(results)
        
        if len(df_result) > 0:
            df_result = df_result.sort_values('score', ascending=False)
            if top_n:
                df_result = df_result.head(top_n)
        
        return df_result
    
    def _get_grade(self, score: float) -> str:
        """评分等级"""
        if score >= 85:
            return "强烈推荐"
        elif score >= 70:
            return "推荐"
        elif score >= 55:
            return "中性偏多"
        elif score >= 40:
            return "中性"
        elif score >= 25:
            return "观望"
        else:
            return "规避"


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 生成测试数据
    dates = pd.date_range('2023-01-01', periods=100)
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'datetime': dates,
        'open': 10 + np.random.randn(100).cumsum() * 0.1,
        'close': 10 + np.random.randn(100).cumsum() * 0.1,
        'high': 10.5 + np.random.randn(100).cumsum() * 0.1,
        'low': 9.5 + np.random.randn(100).cumsum() * 0.1,
        'volume': 1000000 + np.random.randint(-100000, 100000, 100)
    })
    
    # 添加技术指标
    test_data['macd'] = np.random.randn(100) * 0.1
    test_data['macd_signal'] = test_data['macd'].rolling(9).mean()
    test_data['rsi'] = 50 + np.random.randn(100) * 10
    
    # 测试单股票评分
    selector = MultiAgentStockSelector()
    score, details = selector.score(test_data, '000001.SZ', return_details=True)
    
    print(f"\n{'='*60}")
    print(f"股票: 000001.SZ")
    print(f"综合评分: {score:.2f}")
    print(f"置信度: {details['confidence']:.2f}")
    print(f"评级: {details['grade']}")
    print(f"{'='*60}\n")
    
    print("各智能体评分:")
    for name, agent_score in details['agent_scores'].items():
        if isinstance(agent_score, AgentScore):
            print(f"  {name.upper():12s}: {agent_score.score:5.1f} (置信度: {agent_score.confidence:.2f})")
            print(f"               {agent_score.explanation}")
        else:
            print(f"  {name.upper():12s}: {agent_score:5.1f}")
    
    print(f"\n权重配置:")
    for name, weight in details['weights'].items():
        print(f"  {name:12s}: {weight:.2%}")
    
    print("\n✅ 多智能体系统测试完成!")
