"""动态止损止盈管理器 - P1-2

功能:
- 基于中枢/线段/固定比例的动态止损
- 分批止盈策略
- 风险收益比计算

作者: Warp AI Assistant
日期: 2025-01
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StopLossLevel:
    """止损位"""
    method: str  # 'zs_support' / 'seg_support' / 'fixed_ratio'
    price: float
    reason: str
    priority: int  # 优先级 (数字越大越优先)


@dataclass
class TakeProfitLevel:
    """止盈位"""
    method: str
    price: float
    percentage: float  # 分批比例
    reason: str


class ChanLunStopLossManager:
    """缠论动态止损管理器"""
    
    def __init__(
        self,
        zs_buffer: float = 0.02,
        seg_buffer: float = 0.02,
        fixed_stop_loss: float = 0.08,
        fixed_take_profit: float = 0.15
    ):
        """初始化
        
        Args:
            zs_buffer: 中枢止损缓冲 (2%)
            seg_buffer: 线段止损缓冲 (2%)
            fixed_stop_loss: 固定止损比例 (-8%)
            fixed_take_profit: 固定止盈比例 (+15%)
        """
        self.zs_buffer = zs_buffer
        self.seg_buffer = seg_buffer
        self.fixed_stop_loss = fixed_stop_loss
        self.fixed_take_profit = fixed_take_profit
    
    def calculate_stop_loss(
        self,
        entry_point: float,
        current_seg=None,
        zs_list: List = None,
        strategy: str = 'conservative'
    ) -> StopLossLevel:
        """计算止损位
        
        Args:
            entry_point: 入场价格
            current_seg: 当前线段对象
            zs_list: 中枢列表
            strategy: 'conservative'(保守) / 'aggressive'(激进)
        
        Returns:
            StopLossLevel
        """
        stop_losses = []
        
        # 方法1: 中枢止损
        if zs_list and len(zs_list) > 0:
            last_zs = zs_list[-1]
            zs_low = last_zs.low if hasattr(last_zs, 'low') else entry_point * 0.95
            zs_stop = zs_low * (1 - self.zs_buffer)
            
            stop_losses.append(StopLossLevel(
                method='zs_support',
                price=zs_stop,
                reason=f'中枢下沿-{self.zs_buffer:.1%}',
                priority=3  # 最高优先级
            ))
        
        # 方法2: 线段止损
        if current_seg is not None:
            try:
                if hasattr(current_seg, 'is_up') and current_seg.is_up():
                    seg_start = current_seg.start_bi.get_begin_val() if hasattr(current_seg, 'start_bi') else entry_point * 0.96
                    seg_stop = seg_start * (1 - self.seg_buffer)
                    
                    stop_losses.append(StopLossLevel(
                        method='seg_support',
                        price=seg_stop,
                        reason=f'线段起点-{self.seg_buffer:.1%}',
                        priority=2
                    ))
            except:
                pass
        
        # 方法3: 固定比例止损 (保底)
        fixed_stop = entry_point * (1 - self.fixed_stop_loss)
        stop_losses.append(StopLossLevel(
            method='fixed_ratio',
            price=fixed_stop,
            reason=f'固定止损-{self.fixed_stop_loss:.1%}',
            priority=1
        ))
        
        # 选择策略
        if strategy == 'conservative':
            # 保守: 选择最高的止损位(离入场价最近)
            selected = max(stop_losses, key=lambda x: x.price)
        else:
            # 激进: 选择最低的止损位(给更多空间)
            selected = min(stop_losses, key=lambda x: x.price)
        
        logger.debug(f"止损位计算: {selected.method} = {selected.price:.2f} ({selected.reason})")
        return selected
    
    def calculate_take_profit(
        self,
        entry_point: float,
        target_seg=None,
        zs_list: List = None
    ) -> List[TakeProfitLevel]:
        """计算止盈位 (分批)
        
        Args:
            entry_point: 入场价格
            target_seg: 目标线段
            zs_list: 中枢列表
        
        Returns:
            止盈位列表 (按价格从低到高)
        """
        take_profits = []
        
        # 目标1: 固定比例止盈 (第一批出货)
        fixed_target = entry_point * (1 + self.fixed_take_profit)
        take_profits.append(TakeProfitLevel(
            method='fixed_ratio',
            price=fixed_target,
            percentage=0.3,  # 30%仓位
            reason=f'固定止盈+{self.fixed_take_profit:.1%}'
        ))
        
        # 目标2: 中枢阻力位 (第二批)
        if zs_list and len(zs_list) > 0:
            last_zs = zs_list[-1]
            zs_high = last_zs.high if hasattr(last_zs, 'high') else entry_point * 1.1
            zs_resistance = zs_high * 1.02  # 中枢上沿+2%
            
            if zs_resistance > fixed_target:
                take_profits.append(TakeProfitLevel(
                    method='zs_resistance',
                    price=zs_resistance,
                    percentage=0.4,  # 40%仓位
                    reason='中枢上沿+2%'
                ))
        
        # 目标3: 线段目标位 (第三批)
        if target_seg is not None:
            try:
                seg_target = target_seg.get_end_val() if hasattr(target_seg, 'get_end_val') else entry_point * 1.2
                
                if seg_target > fixed_target:
                    take_profits.append(TakeProfitLevel(
                        method='seg_target',
                        price=seg_target,
                        percentage=0.3,  # 30%仓位
                        reason='线段目标位'
                    ))
            except:
                pass
        
        # 按价格排序
        take_profits.sort(key=lambda x: x.price)
        
        # 归一化分批比例
        total_pct = sum(tp.percentage for tp in take_profits)
        if total_pct > 0:
            for tp in take_profits:
                tp.percentage = tp.percentage / total_pct
        
        return take_profits
    
    def calculate_risk_reward_ratio(
        self,
        entry_point: float,
        stop_loss: StopLossLevel,
        take_profit: TakeProfitLevel
    ) -> float:
        """计算风险收益比
        
        Returns:
            风险收益比 (>2为良好)
        """
        risk = abs(entry_point - stop_loss.price)
        reward = abs(take_profit.price - entry_point)
        
        if risk > 0:
            return reward / risk
        else:
            return 0
    
    def should_adjust_stop_loss(
        self,
        entry_point: float,
        current_price: float,
        original_stop: float,
        profit_percentage: float = 0.05
    ) -> Optional[float]:
        """判断是否应该调整止损 (移动止损)
        
        Args:
            entry_point: 入场价
            current_price: 当前价
            original_stop: 原始止损位
            profit_percentage: 盈利多少后开始移动止损 (5%)
        
        Returns:
            新的止损位 or None (不调整)
        """
        # 只有盈利时才考虑调整
        if current_price <= entry_point:
            return None
        
        profit = (current_price - entry_point) / entry_point
        
        # 盈利超过阈值后,移动止损到保本位
        if profit >= profit_percentage:
            # 移动到保本位上方2%
            new_stop = entry_point * 1.02
            
            # 确保新止损比原止损高
            if new_stop > original_stop:
                logger.info(f"移动止损: {original_stop:.2f} → {new_stop:.2f} (盈利{profit:.1%})")
                return new_stop
        
        return None
    
    def generate_exit_plan(
        self,
        entry_point: float,
        position_size: int,
        stop_loss: StopLossLevel,
        take_profits: List[TakeProfitLevel]
    ) -> Dict:
        """生成完整退出计划
        
        Returns:
            {
                'stop_loss': {...},
                'take_profits': [...],
                'position_distribution': [...]
            }
        """
        plan = {
            'entry_price': entry_point,
            'position_size': position_size,
            'stop_loss': {
                'price': stop_loss.price,
                'method': stop_loss.method,
                'shares': position_size,
                'loss': (stop_loss.price - entry_point) * position_size
            },
            'take_profits': [],
            'total_expected_profit': 0
        }
        
        # 分批止盈计划
        remaining_shares = position_size
        for i, tp in enumerate(take_profits):
            shares = int(position_size * tp.percentage)
            if i == len(take_profits) - 1:
                shares = remaining_shares  # 最后一批清仓
            
            profit = (tp.price - entry_point) * shares
            
            plan['take_profits'].append({
                'level': i + 1,
                'price': tp.price,
                'method': tp.method,
                'shares': shares,
                'percentage': tp.percentage,
                'profit': profit,
                'reason': tp.reason
            })
            
            plan['total_expected_profit'] += profit
            remaining_shares -= shares
        
        return plan


if __name__ == '__main__':
    print("="*60)
    print("P1-2: 动态止损止盈管理器测试")
    print("="*60)
    
    # 模拟数据
    class MockZS:
        def __init__(self, low, high):
            self.low = low
            self.high = high
    
    class MockSeg:
        def is_up(self):
            return True
        class StartBi:
            def get_begin_val(self):
                return 9.5
        start_bi = StartBi()
    
    manager = ChanLunStopLossManager()
    
    # 测试1: 计算止损位
    print("\n测试1: 止损位计算")
    entry = 10.0
    zs_list = [MockZS(9.0, 10.5)]
    seg = MockSeg()
    
    stop = manager.calculate_stop_loss(entry, seg, zs_list, strategy='conservative')
    print(f"✅ 止损位: {stop.price:.2f} ({stop.method})")
    print(f"   理由: {stop.reason}")
    print(f"   风险: {(entry - stop.price) / entry:.2%}")
    
    # 测试2: 计算止盈位
    print("\n测试2: 止盈位计算 (分批)")
    take_profits = manager.calculate_take_profit(entry, None, zs_list)
    
    print(f"✅ 分{len(take_profits)}批止盈:")
    for i, tp in enumerate(take_profits, 1):
        print(f"   第{i}批: {tp.price:.2f} ({tp.percentage:.1%}仓位) - {tp.reason}")
    
    # 测试3: 风险收益比
    print("\n测试3: 风险收益比")
    if take_profits:
        rrr = manager.calculate_risk_reward_ratio(entry, stop, take_profits[0])
        print(f"✅ 风险收益比: {rrr:.2f} {'✅良好' if rrr >= 2 else '⚠️偏低'}")
    
    # 测试4: 完整退出计划
    print("\n测试4: 完整退出计划")
    plan = manager.generate_exit_plan(entry, 1000, stop, take_profits)
    
    print(f"✅ 入场: {plan['entry_price']:.2f}, 仓位: {plan['position_size']}股")
    print(f"   止损: {plan['stop_loss']['price']:.2f}, 最大亏损: {plan['stop_loss']['loss']:.2f}元")
    print(f"   止盈计划:")
    for tp in plan['take_profits']:
        print(f"     第{tp['level']}批: {tp['price']:.2f}, {tp['shares']}股, 盈利{tp['profit']:.2f}元")
    print(f"   预期总盈利: {plan['total_expected_profit']:.2f}元")
    
    print("\n✅ P1-2测试完成!")
