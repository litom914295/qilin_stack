"""
一进二涨停策略缠论智能体

专门针对"一进二"场景优化的缠论智能体:
- 一进: 首次涨停板
- 二进: 次日继续上涨

该智能体在缠论评分基础上，增加涨停板特征分析。

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - 涨停板策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.chanlun_agent import ChanLunScoringAgent

logger = logging.getLogger(__name__)


class LimitUpChanLunAgent(ChanLunScoringAgent):
    """一进二涨停策略缠论智能体
    
    在标准缠论评分基础上，针对涨停板场景进行优化:
    1. 涨停质量评估 (封单强度/开板次数)
    2. 涨停后缠论形态 (是否一买二买)
    3. 板块效应 (同板块涨停数量)
    4. 次日竞价强度预判
    """
    
    def __init__(self, 
                 morphology_weight: float = 0.30,
                 bsp_weight: float = 0.30,
                 limitup_weight: float = 0.30,
                 divergence_weight: float = 0.10,
                 **kwargs):
        """
        参数:
            morphology_weight: 形态权重
            bsp_weight: 买卖点权重
            limitup_weight: 涨停质量权重 (新增)
            divergence_weight: 背驰权重
        """
        super().__init__(
            morphology_weight=morphology_weight,
            bsp_weight=bsp_weight,
            divergence_weight=divergence_weight,
            **kwargs
        )
        self.limitup_weight = limitup_weight
        
        # 重新归一化权重
        total = (self.morphology_weight + self.bsp_weight + 
                self.divergence_weight + self.limitup_weight)
        
        if total > 0:
            self.morphology_weight /= total
            self.bsp_weight /= total
            self.divergence_weight /= total
            self.limitup_weight /= total
        
        logger.info(
            f"一进二智能体初始化: "
            f"形态{self.morphology_weight:.1%} "
            f"买卖点{self.bsp_weight:.1%} "
            f"涨停{self.limitup_weight:.1%} "
            f"背驰{self.divergence_weight:.1%}"
        )
    
    def score(self, df: pd.DataFrame, code: str = None, 
             return_details: bool = False,
             sector_limitup_count: int = 0) -> Any:
        """涨停策略评分
        
        参数:
            df: 价格数据 (必须包含最近涨停日数据)
            code: 股票代码
            return_details: 是否返回详细信息
            sector_limitup_count: 同板块涨停数量
            
        返回:
            float或Tuple[float, Dict]
        """
        if len(df) < 20:
            if return_details:
                return 0.0, {'error': '数据不足'}
            return 0.0
        
        # 1. 基础缠论评分
        base_score, base_details = super().score(df, code, return_details=True)
        
        # 2. 涨停质量评分
        limitup_score, limitup_signals = self._score_limitup_quality(df)
        
        # 3. 板块效应评分
        sector_score = self._score_sector_effect(sector_limitup_count)
        
        # 4. 综合涨停分
        combined_limitup_score = (limitup_score * 0.7 + sector_score * 0.3)
        
        # 5. 加权总分
        total_score = (
            base_details['morphology_score'] * self.morphology_weight +
            base_details['bsp_score'] * self.bsp_weight +
            base_details['divergence_score'] * self.divergence_weight +
            combined_limitup_score * self.limitup_weight
        )
        
        if return_details:
            details = {
                'total_score': total_score,
                'base_score': base_score,
                'limitup_score': limitup_score,
                'sector_score': sector_score,
                'combined_limitup_score': combined_limitup_score,
                'limitup_signals': limitup_signals,
                'morphology_score': base_details['morphology_score'],
                'bsp_score': base_details['bsp_score'],
                'divergence_score': base_details['divergence_score'],
                'grade': self._get_limitup_grade(total_score),
                'explanation': self._generate_limitup_explanation(
                    base_details, limitup_signals, sector_limitup_count
                )
            }
            return total_score, details
        
        return total_score
    
    def _score_limitup_quality(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """涨停质量评分
        
        返回:
            (score, signals)
        """
        signals = {}
        score = 50.0
        
        try:
            # 假设最后一天是涨停日
            limitup_day = df.iloc[-1]
            
            # 1. 涨停幅度检测
            pct_change = (limitup_day['close'] - limitup_day['open']) / limitup_day['open']
            
            if pct_change < 0.095:  # 未达到涨停
                signals['not_limitup'] = "非涨停板"
                return 0.0, signals
            
            signals['is_limitup'] = f"涨停({pct_change:.2%})"
            
            # 2. 封单强度 (用成交量变化衡量)
            avg_volume_20 = df['volume'].iloc[-20:-1].mean()
            limitup_volume = limitup_day['volume']
            volume_ratio = limitup_volume / avg_volume_20
            
            if volume_ratio > 3.0:
                score += 20
                signals['volume_strength'] = f"巨量封板(量比{volume_ratio:.1f})"
            elif volume_ratio > 2.0:
                score += 15
                signals['volume_strength'] = f"放量封板(量比{volume_ratio:.1f})"
            elif volume_ratio > 1.5:
                score += 10
                signals['volume_strength'] = f"温和放量(量比{volume_ratio:.1f})"
            else:
                score -= 10
                signals['volume_strength'] = f"缩量涨停(量比{volume_ratio:.1f})"
            
            # 3. 开板检测 (用振幅判断)
            amplitude = (limitup_day['high'] - limitup_day['low']) / limitup_day['low']
            
            if amplitude > 0.12:  # 振幅>12%，说明开板过
                score -= 15
                signals['open_board'] = f"曾开板(振幅{amplitude:.1%})"
            elif amplitude > 0.105:
                score -= 5
                signals['open_board'] = f"接近开板(振幅{amplitude:.1%})"
            else:
                score += 10
                signals['seal_quality'] = f"封板牢固(振幅{amplitude:.1%})"
            
            # 4. 位置判断 (相对20日最高价)
            high_20 = df['high'].iloc[-20:].max()
            close_position = limitup_day['close'] / high_20
            
            if close_position > 0.99:
                score += 15
                signals['position'] = "创新高涨停"
            elif close_position > 0.95:
                score += 10
                signals['position'] = "接近新高"
            elif close_position < 0.80:
                score -= 10
                signals['position'] = "低位涨停"
            
            # 5. 涨停前走势 (前5日涨幅)
            pre_limitup_return = (df['close'].iloc[-2] / df['close'].iloc[-7] - 1) * 100
            
            if -5 < pre_limitup_return < 5:
                score += 10
                signals['pre_trend'] = f"横盘后突破(前期{pre_limitup_return:+.1f}%)"
            elif pre_limitup_return > 20:
                score -= 15
                signals['pre_trend'] = f"连续上涨(前期{pre_limitup_return:+.1f}%)"
            elif pre_limitup_return < -10:
                score += 5
                signals['pre_trend'] = f"超跌反弹(前期{pre_limitup_return:+.1f}%)"
            
            score = max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"涨停质量评分错误 {code}: {e}")
            signals['error'] = str(e)
        
        return score, signals
    
    def _score_sector_effect(self, sector_limitup_count: int) -> float:
        """板块效应评分
        
        参数:
            sector_limitup_count: 同板块涨停数量
        """
        if sector_limitup_count >= 5:
            return 90.0  # 板块大面积涨停
        elif sector_limitup_count >= 3:
            return 75.0  # 板块有联动
        elif sector_limitup_count >= 2:
            return 60.0  # 板块小幅联动
        else:
            return 40.0  # 孤立涨停
    
    def _get_limitup_grade(self, score: float) -> str:
        """涨停策略评级"""
        if score >= 85:
            return "强烈追高"
        elif score >= 75:
            return "可以追高"
        elif score >= 60:
            return "谨慎追高"
        elif score >= 45:
            return "观望等回调"
        else:
            return "不建议追高"
    
    def _generate_limitup_explanation(self, base_details: Dict, 
                                     limitup_signals: Dict,
                                     sector_count: int) -> str:
        """生成说明"""
        parts = []
        
        # 涨停质量
        if 'is_limitup' in limitup_signals:
            parts.append(limitup_signals['is_limitup'])
        
        if 'volume_strength' in limitup_signals:
            parts.append(limitup_signals['volume_strength'])
        
        if 'seal_quality' in limitup_signals:
            parts.append(limitup_signals['seal_quality'])
        elif 'open_board' in limitup_signals:
            parts.append(limitup_signals['open_board'])
        
        # 板块效应
        if sector_count > 0:
            parts.append(f"板块{sector_count}只涨停")
        
        # 缠论信号
        if base_details.get('bsp_score', 0) > 70:
            parts.append("缠论买点")
        
        return ", ".join(parts) if parts else "无明显信号"


class LimitUpSignalGenerator:
    """一进二信号生成器
    
    基于涨停板智能体，生成次日操作信号。
    """
    
    def __init__(self, agent: LimitUpChanLunAgent = None):
        """
        参数:
            agent: 涨停板智能体实例
        """
        self.agent = agent if agent else LimitUpChanLunAgent()
    
    def generate_signals(self, 
                        stock_data: Dict[str, pd.DataFrame],
                        sector_info: Dict[str, int] = None,
                        min_score: float = 70.0) -> pd.DataFrame:
        """生成一进二信号
        
        参数:
            stock_data: {code: DataFrame} 包含涨停板当日数据
            sector_info: {code: sector_limitup_count}
            min_score: 最低评分阈值
            
        返回:
            DataFrame with columns: code, score, grade, signal, explanation
        """
        results = []
        
        for code, df in stock_data.items():
            try:
                # 检测是否涨停
                if not self._is_limitup(df):
                    continue
                
                # 评分
                sector_count = sector_info.get(code, 0) if sector_info else 0
                score, details = self.agent.score(
                    df, code, 
                    return_details=True,
                    sector_limitup_count=sector_count
                )
                
                if score < min_score:
                    continue
                
                # 生成信号
                signal = self._generate_signal(score, details)
                
                results.append({
                    'code': code,
                    'score': score,
                    'grade': details['grade'],
                    'signal': signal,
                    'base_score': details['base_score'],
                    'limitup_score': details['limitup_score'],
                    'sector_score': details['sector_score'],
                    'explanation': details['explanation']
                })
                
            except Exception as e:
                logger.error(f"Signal generation error for {code}: {e}")
        
        df_result = pd.DataFrame(results)
        
        if len(df_result) > 0:
            df_result = df_result.sort_values('score', ascending=False)
        
        return df_result
    
    def _is_limitup(self, df: pd.DataFrame) -> bool:
        """检测是否涨停"""
        if len(df) == 0:
            return False
        
        last_day = df.iloc[-1]
        pct_change = (last_day['close'] - last_day['open']) / last_day['open']
        
        return pct_change >= 0.095
    
    def _generate_signal(self, score: float, details: Dict) -> str:
        """生成操作信号"""
        if score >= 85:
            return "强烈买入"
        elif score >= 75:
            return "买入"
        elif score >= 60:
            return "小仓试探"
        else:
            return "观望"
    
    def get_top_candidates(self, 
                          stock_data: Dict[str, pd.DataFrame],
                          sector_info: Dict[str, int] = None,
                          top_n: int = 10) -> pd.DataFrame:
        """获取Top N候选股票
        
        参数:
            stock_data: 股票数据
            sector_info: 板块信息
            top_n: 返回前N名
            
        返回:
            Top N DataFrame
        """
        signals = self.generate_signals(stock_data, sector_info, min_score=0)
        
        if len(signals) > 0:
            return signals.head(top_n)
        
        return signals


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 生成涨停板测试数据
    dates = pd.date_range('2023-01-01', periods=100)
    np.random.seed(42)
    
    # 模拟涨停板
    test_data = pd.DataFrame({
        'datetime': dates,
        'open': np.concatenate([
            10 + np.random.randn(99).cumsum() * 0.05,
            [10.5]  # 最后一天开盘
        ]),
        'close': np.concatenate([
            10 + np.random.randn(99).cumsum() * 0.05,
            [11.55]  # 最后一天涨停 (+10%)
        ]),
        'high': np.concatenate([
            10.2 + np.random.randn(99).cumsum() * 0.05,
            [11.6]  # 最后一天最高
        ]),
        'low': np.concatenate([
            9.8 + np.random.randn(99).cumsum() * 0.05,
            [10.45]  # 最后一天最低
        ]),
        'volume': np.concatenate([
            np.random.randint(900000, 1100000, 99),
            [2500000]  # 最后一天放量
        ])
    })
    
    # 测试涨停智能体
    agent = LimitUpChanLunAgent()
    score, details = agent.score(test_data, '000001.SZ', return_details=True, 
                                 sector_limitup_count=3)
    
    print(f"\n{'='*60}")
    print(f"股票: 000001.SZ (涨停板)")
    print(f"综合评分: {score:.2f}")
    print(f"评级: {details['grade']}")
    print(f"{'='*60}\n")
    
    print("分项评分:")
    print(f"  基础缠论评分: {details['base_score']:.1f}")
    print(f"  涨停质量评分: {details['limitup_score']:.1f}")
    print(f"  板块效应评分: {details['sector_score']:.1f}")
    print(f"  形态评分: {details['morphology_score']:.1f}")
    print(f"  买卖点评分: {details['bsp_score']:.1f}")
    print(f"  背驰评分: {details['divergence_score']:.1f}")
    
    print(f"\n涨停信号:")
    for key, value in details['limitup_signals'].items():
        print(f"  {key}: {value}")
    
    print(f"\n说明: {details['explanation']}")
    
    # 测试信号生成器
    print(f"\n{'='*60}")
    print("测试信号生成器")
    print(f"{'='*60}\n")
    
    generator = LimitUpSignalGenerator(agent)
    
    # 准备多只股票数据
    stock_data = {
        '000001.SZ': test_data,
        '000002.SZ': test_data.copy(),
        '600000.SH': test_data.copy()
    }
    
    sector_info = {
        '000001.SZ': 3,
        '000002.SZ': 1,
        '600000.SH': 5
    }
    
    signals = generator.generate_signals(stock_data, sector_info, min_score=50)
    
    if len(signals) > 0:
        print("一进二信号:")
        print(signals[['code', 'score', 'grade', 'signal']].to_string(index=False))
    else:
        print("无符合条件的信号")
    
    print("\n✅ 一进二智能体测试完成!")
