"""缠论评分智能体 - 完整版

基于CZSC和Chan.py特征的0-100分评分系统
支持形态、买卖点、背驰、多级别共振、区间套、深度学习六个评分维度
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qlib_enhanced.chanlun.interval_trap import IntervalTrapStrategy
    INTERVAL_TRAP_AVAILABLE = True
except ImportError:
    INTERVAL_TRAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChanLunScoringAgent:
    """
    缠论评分智能体
    
    评分维度 (0-100分):
    1. 形态评分 (25%): 分型/笔/中枢质量
    2. 买卖点评分 (25%): 买卖点类型和有效性
    3. 背驰评分 (10%): MACD背驰风险
    4. 多级别共振 (10%): 跨周期一致性
    5. 区间套策略 (20%): 大小级别共振确认
    6. 深度学习模型 (10%): CNN买卖点识别
    
    评分等级:
    - 90-100: 强烈推荐 (Strong Buy)
    - 75-89: 推荐 (Buy)
    - 60-74: 中性偏多 (Slight Buy)
    - 40-59: 中性 (Neutral)
    - 25-39: 观望 (Wait)
    - 0-24: 规避 (Avoid)
    """
    
    def __init__(self,
                 morphology_weight=0.25,
                 bsp_weight=0.25,
                 divergence_weight=0.10,
                 multi_level_weight=0.10,
                 interval_trap_weight=0.20,
                 dl_model_weight=0.10,
                 enable_bsp=True,
                 enable_divergence=True,
                 use_multi_level=False,
                 enable_interval_trap=True,
                 enable_dl_model=False,
                 interval_trap_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        初始化缠论智能体
        
        Args:
            morphology_weight: 形态权重
            bsp_weight: 买卖点权重
            divergence_weight: 背驰权重
            multi_level_weight: 多级别权重
            interval_trap_weight: 区间套权重
            dl_model_weight: 深度学习模型权重
            enable_bsp: 是否启用买卖点评分
            enable_divergence: 是否启用背驰评分
            use_multi_level: 是否启用多级别共振
            enable_interval_trap: 是否启用区间套策略
            enable_dl_model: 是否启用深度学习模型
            interval_trap_data: 区间套多级别数据字典 {'day': df, '60m': df}
        """
        # 权重配置
        self.morphology_weight = morphology_weight
        self.bsp_weight = bsp_weight if enable_bsp else 0
        self.divergence_weight = divergence_weight if enable_divergence else 0
        self.multi_level_weight = multi_level_weight if use_multi_level else 0
        self.interval_trap_weight = interval_trap_weight if (enable_interval_trap and INTERVAL_TRAP_AVAILABLE) else 0
        self.dl_model_weight = dl_model_weight if enable_dl_model else 0
        
        # 归一化权重
        total_weight = (self.morphology_weight + self.bsp_weight + 
                       self.divergence_weight + self.multi_level_weight +
                       self.interval_trap_weight + self.dl_model_weight)
        if total_weight > 0:
            self.morphology_weight /= total_weight
            self.bsp_weight /= total_weight
            self.divergence_weight /= total_weight
            self.multi_level_weight /= total_weight
            self.interval_trap_weight /= total_weight
            self.dl_model_weight /= total_weight
        
        self.enable_bsp = enable_bsp
        self.enable_divergence = enable_divergence
        self.use_multi_level = use_multi_level
        self.enable_interval_trap = enable_interval_trap and INTERVAL_TRAP_AVAILABLE
        self.enable_dl_model = enable_dl_model
        
        # 区间套策略初始化
        self.interval_trap_strategy = None
        self.interval_trap_data = interval_trap_data or {}
        if self.enable_interval_trap:
            self.interval_trap_strategy = IntervalTrapStrategy(
                major_level='day',
                minor_level='60m',
                max_time_diff_days=3
            )
        
        logger.info(f"缠论智能体初始化: 形态{self.morphology_weight:.1%} "
                   f"买卖点{self.bsp_weight:.1%} 背驰{self.divergence_weight:.1%} "
                   f"多级别{self.multi_level_weight:.1%} "
                   f"区间套{self.interval_trap_weight:.1%} "
                   f"深度学习{self.dl_model_weight:.1%}")
    
    def score(self, df: pd.DataFrame, code: str, return_details=False) -> float:
        """
        对单只股票评分
        
        Args:
            df: 包含缠论特征的DataFrame
            code: 股票代码
            return_details: 是否返回详细评分
        
        Returns:
            总分 (0-100) 或 (总分, 详情字典)
        """
        if len(df) < 20:
            logger.warning(f"{code}: 数据不足20条")
            return (0, {}) if return_details else 0
        
        # 1. 形态评分
        morphology_score = self._score_morphology(df)
        
        # 2. 买卖点评分
        bsp_score = self._score_buy_sell_point(df) if self.enable_bsp else 50
        
        # 3. 背驰评分
        divergence_score = self._score_divergence(df) if self.enable_divergence else 50
        
        # 4. 多级别共振评分
        multi_level_score = self._score_multi_level(df) if self.use_multi_level else 50
        
        # 5. 区间套策略评分
        interval_trap_score = self._score_interval_trap(df, code) if self.enable_interval_trap else 50
        
        # 6. 深度学习模型评分
        dl_score = self._score_deep_learning(df, code) if self.enable_dl_model else 50
        
        # 7. 加权总分
        total_score = (
            morphology_score * self.morphology_weight +
            bsp_score * self.bsp_weight +
            divergence_score * self.divergence_weight +
            multi_level_score * self.multi_level_weight +
            interval_trap_score * self.interval_trap_weight +
            dl_score * self.dl_model_weight
        )
        
        total_score = np.clip(total_score, 0, 100)
        
        if return_details:
            details = {
                'total_score': total_score,
                'morphology_score': morphology_score,
                'bsp_score': bsp_score,
                'divergence_score': divergence_score,
                'multi_level_score': multi_level_score,
                'interval_trap_score': interval_trap_score,
                'dl_score': dl_score,
                'grade': self._get_grade(total_score),
                'explanation': self._generate_explanation(
                    morphology_score, bsp_score, divergence_score, multi_level_score,
                    interval_trap_score, dl_score
                )
            }
            return total_score, details
        
        return total_score
    
    def _score_morphology(self, df: pd.DataFrame) -> float:
        """
        形态评分 (0-100)
        
        评估:
        - 分型质量 (30%): 顶底分型分布
        - 笔的有效性 (40%): 笔的幅度和方向
        - 中枢状态 (30%): 是否形成中枢
        """
        score = 50  # 基础分
        
        # 1. 分型评分 (30分)
        if 'fx_mark' in df.columns:
            fx_marks = df['fx_mark'].iloc[-20:]  # 最近20根K线
            top_fx = (fx_marks == 1).sum()
            bottom_fx = (fx_marks == -1).sum()
            
            if top_fx > 0 or bottom_fx > 0:
                # 底分型更好 (看多视角)
                fx_score = min(30, bottom_fx * 10 - top_fx * 5)
                score += fx_score
        
        # 2. 笔评分 (40分)
        if 'bi_direction' in df.columns and 'bi_power' in df.columns:
            recent_bi = df[['bi_direction', 'bi_power']].iloc[-10:]
            
            # 笔方向一致性
            if recent_bi['bi_direction'].iloc[-1] == 1:  # 最近上涨笔
                bi_score = 20
            elif recent_bi['bi_direction'].iloc[-1] == -1:  # 下跌笔
                bi_score = -10
            else:
                bi_score = 0
            
            # 笔幅度加分
            avg_power = recent_bi['bi_power'].mean()
            if avg_power > 0.05:  # 5%以上幅度
                bi_score += 20
            elif avg_power > 0.03:
                bi_score += 10
            
            score += bi_score
        
        # 3. 中枢评分 (30分)
        if 'in_zs' in df.columns or 'in_chanpy_zs' in df.columns:
            # 使用Chan.py中枢（更精确）
            in_zs = df.get('in_chanpy_zs', df.get('in_zs', 0))
            
            if in_zs.iloc[-1] == 0:  # 不在中枢内（可能突破）
                score += 20
            else:  # 在中枢内（震荡）
                score += 5
        
        return np.clip(score, 0, 100)
    
    def _score_buy_sell_point(self, df: pd.DataFrame) -> float:
        """
        买卖点评分 (0-100)
        
        买点类型权重:
        - 一买: 60分 (底部首次买点)
        - 二买: 80分 (回调买点，推荐)
        - 三买: 75分 (突破买点)
        
        卖点权重:
        - 一卖: -40分 (顶部首次卖点)
        - 二卖: -60分 (反弹卖点，警示)
        - 三卖: -50分 (跌破卖点)
        """
        score = 50  # 基础分
        
        if 'is_buy_point' not in df.columns:
            return score
        
        # 最近5根K线的买卖点
        recent = df[['is_buy_point', 'is_sell_point', 'bsp_type', 'bsp_is_buy']].iloc[-5:]
        
        # 检查是否有买点
        buy_points = recent[recent['is_buy_point'] == 1]
        if len(buy_points) > 0:
            # 取最近的买点
            latest_buy = buy_points.iloc[-1]
            bsp_type = latest_buy['bsp_type']
            
            if bsp_type == 1:  # 一买
                score = 60
            elif bsp_type == 2:  # 二买（最佳）
                score = 85
            elif bsp_type == 3:  # 三买
                score = 75
            else:
                score = 65
            
            # 买点越近越好
            days_ago = len(recent) - buy_points.index[-1] - 1
            if days_ago == 0:
                score += 10  # 当天买点
            elif days_ago == 1:
                score += 5   # 昨天买点
        
        # 检查是否有卖点
        sell_points = recent[recent['is_sell_point'] == 1]
        if len(sell_points) > 0:
            latest_sell = sell_points.iloc[-1]
            bsp_type = latest_sell['bsp_type']
            
            if bsp_type == 1:  # 一卖
                score = 40
            elif bsp_type == 2:  # 二卖（风险）
                score = 30
            elif bsp_type == 3:  # 三卖
                score = 35
        
        return np.clip(score, 0, 100)
    
    def _score_divergence(self, df: pd.DataFrame) -> float:
        """
        背驰评分 (0-100)
        
        评估:
        - MACD背驰检测
        - 成交量背驰
        - 价格背驰
        
        无背驰: 80分
        轻微背驰: 60分
        严重背驰: 30分
        """
        score = 70  # 默认无明显背驰
        
        if len(df) < 30:
            return score
        
        # 简化背驰判断: 使用价格和成交量
        recent = df.iloc[-20:]
        
        # 1. 价格背驰
        if 'close' in df.columns:
            prices = recent['close']
            if len(prices) >= 10:
                first_half_high = prices.iloc[:10].max()
                second_half_high = prices.iloc[10:].max()
                
                # 价格创新高但力度减弱
                if second_half_high > first_half_high * 1.02:
                    # 检查动量
                    first_momentum = prices.iloc[5:10].mean() - prices.iloc[:5].mean()
                    second_momentum = prices.iloc[15:].mean() - prices.iloc[10:15].mean()
                    
                    if second_momentum < first_momentum * 0.7:
                        score -= 30  # 顶背驰
        
        # 2. 成交量背驰
        if 'volume' in df.columns:
            volumes = recent['volume']
            if len(volumes) >= 10:
                first_vol = volumes.iloc[:10].mean()
                second_vol = volumes.iloc[10:].mean()
                
                if second_vol < first_vol * 0.6:
                    score -= 20  # 量能背驰
        
        return np.clip(score, 0, 100)
    
    def _score_multi_level(self, df: pd.DataFrame) -> float:
        """
        多级别共振评分 (0-100)
        
        评估:
        - 日线/60分/30分三级别趋势一致性
        - 目前简化实现，未来可扩展
        """
        # 简化: 基于当前级别的趋势强度
        score = 50
        
        if 'bi_direction' in df.columns:
            recent_bi = df['bi_direction'].iloc[-10:]
            up_count = (recent_bi == 1).sum()
            down_count = (recent_bi == -1).sum()
            
            # 趋势一致性
            if up_count > down_count * 2:
                score = 70  # 强势上涨
            elif down_count > up_count * 2:
                score = 30  # 强势下跌
        
        return score
    
    def _score_interval_trap(self, df: pd.DataFrame, code: str) -> float:
        """
        区间套策略评分 (0-100)
        
        评估:
        - 日线+60分钟多级别共振
        - 买卖点时间窗口确认
        - 信号强度和置信度
        
        信号强度>=80: 90分
        信号强度>=70: 75分
        信号强度>=60: 65分
        无信号: 50分
        """
        if not self.enable_interval_trap or not self.interval_trap_strategy:
            return 50
        
        # 检查是否有多级别数据
        if 'day' not in self.interval_trap_data or '60m' not in self.interval_trap_data:
            # 使用简化评分: 基于当前数据的买卖点信号
            return self._score_buy_sell_point(df)
        
        try:
            # 调用区间套策略
            major_data = self.interval_trap_data.get('day')
            minor_data = self.interval_trap_data.get('60m')
            
            if major_data is None or minor_data is None:
                return 50
            
            # 检测买点信号
            buy_signals = self.interval_trap_strategy.find_interval_trap_signals(
                major_data=major_data,
                minor_data=minor_data,
                code=code,
                signal_type='buy'
            )
            
            if len(buy_signals) == 0:
                return 50  # 无买点信号
            
            # 取最强信号
            best_signal = max(buy_signals, key=lambda s: s.signal_strength)
            
            # 根据信号强度评分
            if best_signal.signal_strength >= 80:
                score = 90
            elif best_signal.signal_strength >= 70:
                score = 80
            elif best_signal.signal_strength >= 60:
                score = 70
            else:
                score = 60
            
            # 置信度加成
            if best_signal.confidence >= 0.8:
                score += 10
            elif best_signal.confidence >= 0.7:
                score += 5
            
            return np.clip(score, 0, 100)
        
        except Exception as e:
            logger.warning(f"{code} 区间套评分失败: {e}")
            return 50
    
    def _score_deep_learning(self, df: pd.DataFrame, code: str) -> float:
        """
        深度学习模型评分 (0-100)
        
        评估:
        - CNN识别买卖点概率
        - 一买/二买/三买预测
        
        预测为二买(prob>0.7): 90分
        预测为三买(prob>0.6): 80分
        预测为一买(prob>0.6): 70分
        无明确信号: 50分
        """
        if not self.enable_dl_model:
            return 50
        
        # TODO: 加载训练好的模型并预测
        # 目前返回默认值，待模型训练完成后集成
        logger.debug(f"{code} 深度学习模型暂未启用")
        return 50
    
    def _get_grade(self, score: float) -> str:
        """评分等级"""
        if score >= 90:
            return "强烈推荐"
        elif score >= 75:
            return "推荐"
        elif score >= 60:
            return "中性偏多"
        elif score >= 40:
            return "中性"
        elif score >= 25:
            return "观望"
        else:
            return "规避"
    
    def _generate_explanation(self, morphology, bsp, divergence, multi_level, 
                             interval_trap=50, dl_score=50) -> str:
        """生成评分说明"""
        explanations = []
        
        if morphology >= 70:
            explanations.append("形态优秀")
        elif morphology <= 40:
            explanations.append("形态较弱")
        
        if self.enable_bsp:
            if bsp >= 75:
                explanations.append("出现买点信号")
            elif bsp <= 40:
                explanations.append("出现卖点信号")
        
        if self.enable_divergence:
            if divergence <= 50:
                explanations.append("存在背驰风险")
        
        if self.enable_interval_trap:
            if interval_trap >= 80:
                explanations.append("区间套强烈确认")
            elif interval_trap >= 70:
                explanations.append("区间套确认")
        
        if self.enable_dl_model:
            if dl_score >= 80:
                explanations.append("DL模型推荐")
        
        if not explanations:
            explanations.append("综合表现平稳")
        
        return " | ".join(explanations)
    
    def batch_score(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        批量评分
        
        Args:
            stock_data: {code: dataframe} 字典
        
        Returns:
            DataFrame with columns [code, score, grade, explanation]
        """
        results = []
        
        for code, df in stock_data.items():
            try:
                score, details = self.score(df, code, return_details=True)
                results.append({
                    'code': code,
                    'score': score,
                    'grade': details['grade'],
                    'morphology': details['morphology_score'],
                    'bsp': details['bsp_score'],
                    'divergence': details['divergence_score'],
                    'explanation': details['explanation']
                })
            except Exception as e:
                logger.error(f"{code} 评分失败: {e}")
                continue
        
        return pd.DataFrame(results)
