"""中枢扩展与升级分析器 - P1-1

功能:
- 中枢扩展检测 (第三类买卖点未突破回到中枢)
- 中枢升级检测 (小级别→大级别)
- 中枢移动分析 (rising/falling/sideways)

作者: Warp AI Assistant
日期: 2025-01
"""
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from Common.CEnum import FX_TYPE


@dataclass
class ZSExtension:
    """中枢扩展结果"""
    original_zs: object  # 原始中枢
    extended_by: object  # 扩展笔/线段
    new_low: float
    new_high: float
    extension_type: str  # 'upward' / 'downward'
    reason: str


@dataclass
class ZSUpgrade:
    """中枢升级结果"""
    sub_zs_list: List  # 小级别中枢列表
    overlap_low: float  # 重叠区间下沿
    overlap_high: float  # 重叠区间上沿
    upgraded_level: str  # 升级后级别
    strength: float  # 升级强度 [0-1]


class ZSAnalyzer:
    """中枢分析器"""
    
    def __init__(self, extension_threshold: float = 0.02):
        """初始化
        
        Args:
            extension_threshold: 中枢扩展判定阈值 (2%)
        """
        self.extension_threshold = extension_threshold
    
    def detect_zs_extension(self, zs, new_bi) -> Optional[ZSExtension]:
        """检测中枢扩展
        
        理论: 第三类买卖点未能突破,价格回到中枢内部,导致中枢区间扩大
        
        Args:
            zs: 中枢对象
            new_bi: 新的笔对象
        
        Returns:
            ZSExtension or None
        """
        if not zs or not new_bi:
            return None
        
        try:
            # 1. 检查是否是第三类买卖点尝试突破
            if not hasattr(zs, 'is_third_buy_sell_point'):
                # 简化判断: 检查笔是否尝试突破中枢
                bi_high = new_bi._high() if hasattr(new_bi, '_high') else new_bi.high
                bi_low = new_bi._low() if hasattr(new_bi, '_low') else new_bi.low
                
                # 向上突破失败
                if bi_high > zs.high * (1 + self.extension_threshold):
                    if bi_low < zs.high:  # 回落到中枢内
                        return ZSExtension(
                            original_zs=zs,
                            extended_by=new_bi,
                            new_low=zs.low,
                            new_high=max(zs.high, bi_high * 0.98),
                            extension_type='upward',
                            reason='向上突破失败,中枢上沿扩展'
                        )
                
                # 向下突破失败
                if bi_low < zs.low * (1 - self.extension_threshold):
                    if bi_high > zs.low:  # 回升到中枢内
                        return ZSExtension(
                            original_zs=zs,
                            extended_by=new_bi,
                            new_low=min(zs.low, bi_low * 1.02),
                            new_high=zs.high,
                            extension_type='downward',
                            reason='向下突破失败,中枢下沿扩展'
                        )
            
            return None
            
        except Exception as e:
            return None
    
    def detect_zs_upgrade(self, seg_list: List) -> Optional[ZSUpgrade]:
        """检测中枢升级
        
        理论: 连续3个或更多小级别中枢有重叠区间,形成大级别中枢
        
        Args:
            seg_list: 线段列表
        
        Returns:
            ZSUpgrade or None
        """
        if not seg_list or len(seg_list) < 3:
            return None
        
        try:
            # 1. 提取所有中枢
            zs_list = []
            for seg in seg_list[-10:]:  # 只看最近10个线段
                if hasattr(seg, 'zs_lst') and seg.zs_lst:
                    zs_list.extend(seg.zs_lst)
            
            if len(zs_list) < 3:
                return None
            
            # 2. 检查最近3个中枢是否有重叠
            recent_3_zs = zs_list[-3:]
            overlap = self._calculate_zs_overlap(recent_3_zs)
            
            if overlap:
                overlap_low, overlap_high = overlap
                overlap_size = overlap_high - overlap_low
                
                # 3. 判断是否足够强(重叠区间足够大)
                avg_zs_size = np.mean([zs.high - zs.low for zs in recent_3_zs])
                strength = overlap_size / avg_zs_size if avg_zs_size > 0 else 0
                
                if strength > 0.3:  # 重叠超过30%
                    return ZSUpgrade(
                        sub_zs_list=recent_3_zs,
                        overlap_low=overlap_low,
                        overlap_high=overlap_high,
                        upgraded_level='higher',
                        strength=strength
                    )
            
            return None
            
        except Exception as e:
            return None
    
    def analyze_zs_movement(self, zs_list: List) -> Dict[str, any]:
        """分析中枢移动方向
        
        Args:
            zs_list: 中枢列表
        
        Returns:
            {
                'direction': 'rising'/'falling'/'sideways',
                'slope': float,
                'confidence': float
            }
        """
        if not zs_list or len(zs_list) < 3:
            return {
                'direction': 'unknown',
                'slope': 0,
                'confidence': 0
            }
        
        try:
            # 1. 提取中枢中点
            last_n = min(5, len(zs_list))
            recent_zs = zs_list[-last_n:]
            
            mid_points = []
            for zs in recent_zs:
                mid = (zs.high + zs.low) / 2
                mid_points.append(mid)
            
            # 2. 线性回归计算斜率
            x = np.arange(last_n)
            y = np.array(mid_points)
            
            if len(x) < 2:
                return {'direction': 'unknown', 'slope': 0, 'confidence': 0}
            
            # 多项式拟合 (1次 = 线性)
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            # 3. 计算R²判断置信度
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 4. 判断方向
            avg_price = np.mean(mid_points)
            threshold = avg_price * 0.01  # 1%阈值
            
            if slope > threshold:
                direction = 'rising'
            elif slope < -threshold:
                direction = 'falling'
            else:
                direction = 'sideways'
            
            return {
                'direction': direction,
                'slope': slope,
                'confidence': max(0, min(1, r_squared)),
                'avg_zs_mid': avg_price
            }
            
        except Exception as e:
            return {
                'direction': 'error',
                'slope': 0,
                'confidence': 0,
                'error': str(e)
            }
    
    def _calculate_zs_overlap(self, zs_list: List) -> Optional[tuple]:
        """计算多个中枢的重叠区间
        
        Returns:
            (overlap_low, overlap_high) or None
        """
        if len(zs_list) < 2:
            return None
        
        # 找到所有中枢的公共区间
        max_low = max(zs.low for zs in zs_list)
        min_high = min(zs.high for zs in zs_list)
        
        if max_low < min_high:
            return (max_low, min_high)
        else:
            return None
    
    def analyze_zs_strength(self, zs) -> float:
        """分析单个中枢强度
        
        Returns:
            强度评分 [0-1]
        """
        try:
            # 1. 中枢宽度 (宽度越大越强)
            zs_range = zs.high - zs.low
            
            # 2. 中枢K线数量 (越多越强)
            if hasattr(zs, 'begin') and hasattr(zs, 'end'):
                duration = abs(zs.end.idx - zs.begin.idx)
            else:
                duration = 5  # 默认值
            
            # 3. 综合评分
            width_score = min(1.0, zs_range / (zs.low * 0.1))  # 10%为满分
            duration_score = min(1.0, duration / 20)  # 20根K线为满分
            
            strength = (width_score * 0.6 + duration_score * 0.4)
            return strength
            
        except:
            return 0.5  # 默认中等强度


if __name__ == '__main__':
    print("="*60)
    print("P1-1: 中枢扩展与升级分析器测试")
    print("="*60)
    
    # 模拟中枢
    class MockZS:
        def __init__(self, low, high):
            self.low = low
            self.high = high
    
    class MockBi:
        def __init__(self, low, high):
            self.low = low
            self.high = high
        def _low(self):
            return self.low
        def _high(self):
            return self.high
    
    analyzer = ZSAnalyzer()
    
    # 测试1: 中枢扩展
    print("\n测试1: 中枢扩展检测")
    zs = MockZS(10, 11)
    new_bi = MockBi(9.5, 10.8)  # 向下突破失败
    
    extension = analyzer.detect_zs_extension(zs, new_bi)
    if extension:
        print(f"✅ 检测到中枢扩展: {extension.reason}")
        print(f"   新区间: [{extension.new_low:.2f}, {extension.new_high:.2f}]")
    else:
        print("❌ 未检测到扩展")
    
    # 测试2: 中枢升级
    print("\n测试2: 中枢升级检测")
    zs_list = [MockZS(10, 11), MockZS(10.2, 11.2), MockZS(10.5, 11.5)]
    
    class MockSeg:
        def __init__(self, zs_list):
            self.zs_lst = zs_list
    
    seg_list = [MockSeg([zs]) for zs in zs_list]
    
    upgrade = analyzer.detect_zs_upgrade(seg_list)
    if upgrade:
        print(f"✅ 检测到中枢升级")
        print(f"   重叠区间: [{upgrade.overlap_low:.2f}, {upgrade.overlap_high:.2f}]")
        print(f"   强度: {upgrade.strength:.2%}")
    else:
        print("❌ 未检测到升级")
    
    # 测试3: 中枢移动
    print("\n测试3: 中枢移动分析")
    rising_zs = [MockZS(10+i*0.5, 11+i*0.5) for i in range(5)]
    
    result = analyzer.analyze_zs_movement(rising_zs)
    print(f"✅ 移动方向: {result['direction']}")
    print(f"   斜率: {result['slope']:.4f}")
    print(f"   置信度: {result['confidence']:.2%}")
    
    print("\n✅ P1-1测试完成!")
