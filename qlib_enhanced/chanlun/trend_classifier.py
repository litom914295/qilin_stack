"""走势类型分类器 - Phase P0-1

功能:
- 识别上涨趋势/下跌趋势/盘整
- 基于中枢位置变化和线段方向
- 用于过滤逆势信号,提升胜率

作者: Warp AI Assistant
日期: 2025-01
版本: v1.8
"""

import numpy as np
from enum import Enum
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TrendType(Enum):
    """走势类型枚举"""
    UPTREND = "uptrend"      # 上涨趋势
    DOWNTREND = "downtrend"  # 下跌趋势
    SIDEWAYS = "sideways"    # 盘整震荡
    UNKNOWN = "unknown"      # 未知/数据不足


class TrendClassifier:
    """走势类型分类器
    
    基于缠论理论:
    - 趋势 = 中枢抬高/降低 + 连续同向线段
    - 盘整 = 震荡在中枢范围内
    
    Examples:
        >>> classifier = TrendClassifier()
        >>> trend = classifier.classify_trend(seg_list, zs_list)
        >>> if trend == TrendType.UPTREND:
        ...     print("当前上涨趋势")
    """
    
    def __init__(self, 
                 zs_threshold: float = 0.02,
                 seg_consistency_threshold: float = 0.6):
        """
        初始化走势分类器
        
        Args:
            zs_threshold: 中枢移动判断阈值(2%)
            seg_consistency_threshold: 线段方向一致性阈值(60%)
        """
        self.zs_threshold = zs_threshold
        self.seg_consistency_threshold = seg_consistency_threshold
    
    def classify_trend(self, seg_list, zs_list) -> TrendType:
        """
        分类走势类型
        
        Args:
            seg_list: 线段列表(chan.py的seg_list)
            zs_list: 中枢列表(从seg中提取的zs_lst)
            
        Returns:
            TrendType: 走势类型
        """
        if not seg_list or len(seg_list) < 3:
            logger.debug("线段数据不足,无法判断走势")
            return TrendType.UNKNOWN
        
        # 方法1: 基于中枢位置变化(优先)
        if zs_list and len(zs_list) >= 2:
            zs_trend = self._analyze_zs_trend(zs_list)
            if zs_trend == 'rising':
                logger.debug("中枢抬高,判断为上涨趋势")
                return TrendType.UPTREND
            elif zs_trend == 'falling':
                logger.debug("中枢降低,判断为下跌趋势")
                return TrendType.DOWNTREND
            # 如果中枢横向震荡,继续用线段方向判断
        
        # 方法2: 基于线段方向一致性
        seg_trend = self._analyze_seg_direction(seg_list)
        return seg_trend
    
    def _analyze_zs_trend(self, zs_list) -> str:
        """
        分析中枢趋势
        
        Args:
            zs_list: 中枢列表
            
        Returns:
            'rising' / 'falling' / 'sideways'
        """
        if len(zs_list) < 2:
            return 'unknown'
        
        # 取最近2个中枢对比
        last_zs = zs_list[-1]
        prev_zs = zs_list[-2]
        
        # 中枢中点对比
        last_mid = last_zs.mid
        prev_mid = prev_zs.mid
        
        # 计算变化率
        change_rate = (last_mid - prev_mid) / prev_mid
        
        if change_rate > self.zs_threshold:
            return 'rising'
        elif change_rate < -self.zs_threshold:
            return 'falling'
        else:
            return 'sideways'
    
    def _analyze_seg_direction(self, seg_list) -> TrendType:
        """
        分析线段方向一致性
        
        Args:
            seg_list: 线段列表
            
        Returns:
            TrendType
        """
        # 取最近3-5个线段
        recent_segs = seg_list[-5:] if len(seg_list) >= 5 else seg_list[-3:]
        
        # 统计上涨线段数量
        up_count = sum(1 for seg in recent_segs if seg.is_up())
        total_count = len(recent_segs)
        
        up_ratio = up_count / total_count
        
        # 判断方向一致性
        if up_ratio >= self.seg_consistency_threshold:
            logger.debug(f"线段上涨比例{up_ratio:.1%},判断为上涨趋势")
            return TrendType.UPTREND
        elif up_ratio <= (1 - self.seg_consistency_threshold):
            logger.debug(f"线段下跌比例{1-up_ratio:.1%},判断为下跌趋势")
            return TrendType.DOWNTREND
        else:
            logger.debug(f"线段方向混合{up_ratio:.1%},判断为盘整")
            return TrendType.SIDEWAYS
    
    def get_trend_strength(self, seg_list, zs_list) -> float:
        """
        计算趋势强度
        
        Args:
            seg_list: 线段列表
            zs_list: 中枢列表
            
        Returns:
            float: 趋势强度 [0-1]
        """
        if not seg_list or len(seg_list) < 3:
            return 0.0
        
        strength_scores = []
        
        # 1. 中枢移动强度
        if zs_list and len(zs_list) >= 2:
            last_zs = zs_list[-1]
            prev_zs = zs_list[-2]
            zs_move_strength = abs((last_zs.mid - prev_zs.mid) / prev_zs.mid)
            strength_scores.append(min(1.0, zs_move_strength / 0.1))  # 归一化到[0-1]
        
        # 2. 线段方向一致性
        recent_segs = seg_list[-5:] if len(seg_list) >= 5 else seg_list[-3:]
        up_count = sum(1 for seg in recent_segs if seg.is_up())
        direction_consistency = abs(up_count / len(recent_segs) - 0.5) * 2  # 转为[0-1]
        strength_scores.append(direction_consistency)
        
        # 3. 线段幅度
        if len(recent_segs) > 0:
            avg_amp = np.mean([seg.amp() for seg in recent_segs])
            # 假设5%幅度为标准
            amp_strength = min(1.0, avg_amp / (recent_segs[0].get_begin_val() * 0.05))
            strength_scores.append(amp_strength)
        
        # 综合强度
        if strength_scores:
            return np.mean(strength_scores)
        return 0.0
    
    def classify_with_details(self, seg_list, zs_list) -> dict:
        """
        分类走势并返回详细信息
        
        Args:
            seg_list: 线段列表
            zs_list: 中枢列表
            
        Returns:
            dict: {
                'trend_type': TrendType,
                'strength': float,
                'reason': str,
                'details': dict
            }
        """
        trend_type = self.classify_trend(seg_list, zs_list)
        strength = self.get_trend_strength(seg_list, zs_list)
        
        # 生成原因说明
        reason_parts = []
        
        if zs_list and len(zs_list) >= 2:
            zs_trend = self._analyze_zs_trend(zs_list)
            if zs_trend == 'rising':
                reason_parts.append("中枢抬高")
            elif zs_trend == 'falling':
                reason_parts.append("中枢降低")
            else:
                reason_parts.append("中枢横向")
        
        if seg_list and len(seg_list) >= 3:
            recent_segs = seg_list[-5:] if len(seg_list) >= 5 else seg_list[-3:]
            up_count = sum(1 for seg in recent_segs if seg.is_up())
            up_ratio = up_count / len(recent_segs)
            reason_parts.append(f"线段上涨比例{up_ratio:.0%}")
        
        reason = ", ".join(reason_parts) if reason_parts else "数据不足"
        
        return {
            'trend_type': trend_type,
            'strength': strength,
            'reason': reason,
            'details': {
                'seg_count': len(seg_list) if seg_list else 0,
                'zs_count': len(zs_list) if zs_list else 0,
                'recent_segs': len(seg_list[-5:]) if seg_list and len(seg_list) >= 5 else len(seg_list) if seg_list else 0
            }
        }


# ========== 工具函数 ==========

def extract_zs_from_segs(seg_list) -> list:
    """
    从线段列表中提取所有中枢
    
    Args:
        seg_list: chan.py的线段列表
        
    Returns:
        list: 所有中枢的列表
    """
    all_zs = []
    for seg in seg_list:
        if hasattr(seg, 'zs_lst') and seg.zs_lst:
            all_zs.extend(seg.zs_lst)
    return all_zs


if __name__ == '__main__':
    """测试代码"""
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== 走势类型分类器测试 ===\n")
    
    # 创建分类器
    classifier = TrendClassifier()
    
    print("✅ TrendClassifier初始化成功")
    print(f"   中枢移动阈值: {classifier.zs_threshold:.1%}")
    print(f"   线段一致性阈值: {classifier.seg_consistency_threshold:.1%}")
    
    print("\n✅ 走势类型分类器创建完成!")
    print("📝 使用方法:")
    print("   from qlib_enhanced.chanlun.trend_classifier import TrendClassifier, TrendType")
    print("   classifier = TrendClassifier()")
    print("   trend = classifier.classify_trend(seg_list, zs_list)")
