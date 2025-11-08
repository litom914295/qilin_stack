"""背驰检测器 - Phase P0-2

功能:
- 检测盘整背驰和趋势背驰
- 量化背驰强度评分
- 集成为Alpha因子

作者: Warp AI Assistant
日期: 2025-01
版本: v1.8
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DivergenceType(Enum):
    """背驰类型"""
    TOP_DIVERGENCE = "top_divergence"          # 顶背驰
    BOTTOM_DIVERGENCE = "bottom_divergence"    # 底背驰
    CONSOLIDATION = "consolidation"            # 盘整背驰
    TREND = "trend"                            # 趋势背驰
    NONE = "none"                              # 无背驰


@dataclass
class DivergenceSignal:
    """背驰信号"""
    type: str  # 'top_divergence' / 'bottom_divergence'
    score: float  # 背驰强度 [0-1]
    reason: str  # 原因说明
    macd_ratio: float = 0.0  # MACD比率
    price_diff: float = 0.0  # 价格差异
    divergence_category: str = "unknown"  # 背驰分类


class DivergenceDetector:
    """背驰检测器
    
    基于缠论背驰理论:
    - 价格新高/新低,但MACD力度减弱
    - 区分盘整背驰和趋势背驰
    - 量化背驰强度
    
    Examples:
        >>> detector = DivergenceDetector()
        >>> signal = detector.detect_divergence(current_seg, prev_seg)
        >>> if signal and signal.type == 'top_divergence':
        ...     print(f"顶背驰:{signal.score:.1%}")
    """
    
    def __init__(self,
                 macd_threshold: float = 0.9,
                 macd_algo: str = 'area'):
        """
        初始化背驰检测器
        
        Args:
            macd_threshold: MACD力度阈值(90%)
            macd_algo: MACD计算算法('area'/'peak'/'slope')
        """
        self.macd_threshold = macd_threshold
        self.macd_algo = macd_algo
    
    def detect_divergence(self,
                         current_item,
                         prev_item,
                         macd_algo: str = None) -> Optional[DivergenceSignal]:
        """
        检测背驰
        
        Args:
            current_item: 当前线段/笔(chan.py的Seg或Bi对象)
            prev_item: 前一个线段/笔
            macd_algo: MACD算法(可选,默认用初始化值)
            
        Returns:
            DivergenceSignal or None
        """
        if not current_item or not prev_item:
            return None
        
        macd_algo = macd_algo or self.macd_algo
        
        try:
            # 1. 计算MACD指标
            current_macd = current_item.cal_macd_metric(macd_algo, is_reverse=True)
            prev_macd = prev_item.cal_macd_metric(macd_algo, is_reverse=False)
            
            if current_macd <= 0 or prev_macd <= 0:
                return None
            
            # 2. 价格对比
            current_price = current_item.get_end_val()
            prev_price = prev_item.get_end_val()
            
            # 3. 判断背驰
            if current_item.is_up():
                # 上涨段 - 检测顶背驰
                price_higher = current_price > prev_price
                macd_lower = current_macd < prev_macd * self.macd_threshold
                
                if price_higher and macd_lower:
                    divergence_score = 1.0 - (current_macd / prev_macd)
                    macd_ratio = current_macd / prev_macd
                    price_diff = (current_price - prev_price) / prev_price
                    
                    return DivergenceSignal(
                        type='top_divergence',
                        score=divergence_score,
                        reason=f"价格新高+{price_diff:.1%},但MACD减弱{(1-macd_ratio):.1%}",
                        macd_ratio=macd_ratio,
                        price_diff=price_diff
                    )
            else:
                # 下跌段 - 检测底背驰
                price_lower = current_price < prev_price
                macd_lower = current_macd < prev_macd * self.macd_threshold
                
                if price_lower and macd_lower:
                    divergence_score = 1.0 - (current_macd / prev_macd)
                    macd_ratio = current_macd / prev_macd
                    price_diff = (prev_price - current_price) / prev_price
                    
                    return DivergenceSignal(
                        type='bottom_divergence',
                        score=divergence_score,
                        reason=f"价格新低-{price_diff:.1%},但MACD减弱{(1-macd_ratio):.1%}",
                        macd_ratio=macd_ratio,
                        price_diff=price_diff
                    )
        
        except Exception as e:
            logger.error(f"背驰检测错误: {e}")
            return None
        
        return None
    
    def classify_divergence_type(self, item, zs_list) -> str:
        """
        分类背驰类型(盘整/趋势)
        
        Args:
            item: 线段/笔对象
            zs_list: 中枢列表
            
        Returns:
            'consolidation_divergence' / 'trend_divergence'
        """
        if not zs_list:
            return 'trend_divergence'
        
        last_zs = zs_list[-1]
        
        # 判断是否在中枢内
        if hasattr(last_zs, 'in_range') and last_zs.in_range(item):
            return 'consolidation_divergence'  # 盘整背驰
        else:
            return 'trend_divergence'  # 趋势背驰
    
    def detect_batch_divergence(self, seg_list, min_segs: int = 2) -> list:
        """
        批量检测背驰信号
        
        Args:
            seg_list: 线段列表
            min_segs: 最少线段数
            
        Returns:
            list: 背驰信号列表
        """
        if not seg_list or len(seg_list) < min_segs + 1:
            return []
        
        signals = []
        
        # 遍历相邻线段对
        for i in range(len(seg_list) - 1):
            prev_seg = seg_list[i]
            current_seg = seg_list[i + 1]
            
            # 只比较同方向的线段
            if prev_seg.is_up() == current_seg.is_up():
                signal = self.detect_divergence(current_seg, prev_seg)
                if signal:
                    signals.append({
                        'index': i + 1,
                        'signal': signal
                    })
        
        return signals
    
    def calculate_divergence_risk_score(self, df: pd.DataFrame, seg_list) -> pd.Series:
        """
        计算背驰风险评分(用于Alpha因子)
        
        Args:
            df: DataFrame
            seg_list: 线段列表
            
        Returns:
            pd.Series: 背驰风险评分 [-1, 1]
        """
        scores = pd.Series(0.0, index=df.index)
        
        if not seg_list or len(seg_list) < 2:
            return scores
        
        # 检测所有背驰信号
        divergence_signals = self.detect_batch_divergence(seg_list)
        
        # 映射到DataFrame索引
        for item in divergence_signals:
            seg_idx = item['index']
            signal = item['signal']
            
            if seg_idx < len(seg_list):
                seg = seg_list[seg_idx]
                # 获取线段结束时间对应的df索引
                end_klu = seg.get_end_klu()
                
                # 在df中查找对应行(简化处理,假设按idx对齐)
                try:
                    if hasattr(end_klu, 'idx') and end_klu.idx < len(df):
                        if signal.type == 'top_divergence':
                            scores.iloc[end_klu.idx] = -signal.score  # 负值=卖出风险
                        elif signal.type == 'bottom_divergence':
                            scores.iloc[end_klu.idx] = signal.score  # 正值=买入机会
                except Exception as e:
                    logger.debug(f"映射背驰信号到df失败: {e}")
                    continue
        
        return scores


# ========== Alpha因子集成 ==========

def calculate_divergence_alpha(df: pd.DataFrame, seg_list=None) -> pd.Series:
    """
    Alpha因子: 背驰风险因子
    
    Args:
        df: DataFrame with price data
        seg_list: 线段列表(可选,如果没有则返回0)
        
    Returns:
        pd.Series: 背驰Alpha因子 [-1, 1]
    """
    if seg_list is None or len(seg_list) < 2:
        return pd.Series(0.0, index=df.index)
    
    detector = DivergenceDetector()
    return detector.calculate_divergence_risk_score(df, seg_list)


if __name__ == '__main__':
    """测试代码"""
    logging.basicConfig(level=logging.INFO)
    
    print("=== 背驰检测器测试 ===\n")
    
    # 创建检测器
    detector = DivergenceDetector(macd_threshold=0.9)
    
    print("✅ DivergenceDetector初始化成功")
    print(f"   MACD阈值: {detector.macd_threshold:.0%}")
    print(f"   MACD算法: {detector.macd_algo}")
    
    print("\n✅ 背驰检测器创建完成!")
    print("📝 使用方法:")
    print("   from qlib_enhanced.chanlun.divergence_detector import DivergenceDetector")
    print("   detector = DivergenceDetector()")
    print("   signal = detector.detect_divergence(current_seg, prev_seg)")
    print("\n📝 Alpha因子使用:")
    print("   from qlib_enhanced.chanlun.divergence_detector import calculate_divergence_alpha")
    print("   alpha = calculate_divergence_alpha(df, seg_list)")
