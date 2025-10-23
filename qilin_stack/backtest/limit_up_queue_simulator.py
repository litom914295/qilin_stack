"""
涨停排队模拟器 (Limit Up Queue Simulator)
模拟涨停板封单排队和成交过程

核心功能：
1. 涨停封单规模追踪
2. 排队位置计算
3. 成交概率估算
4. 真实成交时间模拟
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class LimitUpStrength(Enum):
    """涨停强度"""
    ONE_WORD = "一字板"         # 开盘即封死
    EARLY_SEAL = "早盘封板"     # 9:45前封板
    MID_SEAL = "盘中封板"       # 10:00-14:00封板
    LATE_SEAL = "尾盘封板"      # 14:00后封板
    WEAK_SEAL = "弱封"          # 多次开板


@dataclass
class LimitUpQueueStatus:
    """涨停排队状态"""
    symbol: str
    timestamp: datetime
    
    # 封单信息
    limit_price: float              # 涨停价
    seal_amount: float              # 封单金额（元）
    seal_shares: int                # 封单股数
    seal_orders: int                # 封单笔数
    
    # 排队信息
    queue_position: int             # 排队位置（前面有多少股）
    queue_ahead_amount: float       # 前面排队金额
    estimated_wait_time: float      # 预计等待时间（分钟）
    
    # 成交概率
    fill_probability: float         # 成交概率（0-1）
    expected_fill_time: Optional[datetime]  # 预计成交时间
    
    # 强度评估
    strength: LimitUpStrength
    strength_score: float           # 强度评分（0-100）
    
    # 元数据
    warnings: List[str]


@dataclass
class QueueExecution:
    """排队成交结果"""
    symbol: str
    order_time: datetime            # 下单时间
    
    # 订单信息
    target_shares: int              # 目标股数
    limit_price: float              # 涨停价
    
    # 成交结果
    filled: bool                    # 是否成交
    filled_shares: int              # 实际成交股数
    fill_time: Optional[datetime]   # 成交时间
    avg_fill_price: float           # 平均成交价
    
    # 排队信息
    initial_queue_position: int     # 初始排队位置
    final_queue_position: int       # 最终排队位置
    
    # 分析
    execution_reason: str           # 成交/未成交原因
    seal_broke: bool                # 是否开板
    warnings: List[str]


class LimitUpQueueSimulator:
    """涨停排队模拟器"""
    
    def __init__(self,
                 one_word_fill_prob: float = 0.05,      # 一字板成交概率5%
                 early_seal_fill_prob: float = 0.20,    # 早盘封板成交概率20%
                 mid_seal_fill_prob: float = 0.50,      # 盘中封板成交概率50%
                 late_seal_fill_prob: float = 0.80,     # 尾盘封板成交概率80%
                 weak_seal_fill_prob: float = 0.95):    # 弱封成交概率95%
        """
        初始化涨停排队模拟器
        
        Args:
            one_word_fill_prob: 一字板成交概率
            early_seal_fill_prob: 早盘封板成交概率
            mid_seal_fill_prob: 盘中封板成交概率
            late_seal_fill_prob: 尾盘封板成交概率
            weak_seal_fill_prob: 弱封成交概率
        """
        self.fill_probs = {
            LimitUpStrength.ONE_WORD: one_word_fill_prob,
            LimitUpStrength.EARLY_SEAL: early_seal_fill_prob,
            LimitUpStrength.MID_SEAL: mid_seal_fill_prob,
            LimitUpStrength.LATE_SEAL: late_seal_fill_prob,
            LimitUpStrength.WEAK_SEAL: weak_seal_fill_prob
        }
    
    def evaluate_queue_status(self,
                             symbol: str,
                             limit_price: float,
                             seal_amount: float,
                             seal_time: datetime,
                             current_time: datetime,
                             target_shares: int,
                             total_queue_shares: Optional[int] = None,
                             open_times: int = 0) -> LimitUpQueueStatus:
        """
        评估涨停排队状态
        
        Args:
            symbol: 股票代码
            limit_price: 涨停价
            seal_amount: 封单金额（元）
            seal_time: 封板时间
            current_time: 当前时间
            target_shares: 目标建仓股数
            total_queue_shares: 总排队股数（如已知）
            open_times: 开板次数
            
        Returns:
            LimitUpQueueStatus
        """
        warnings = []
        
        # 1. 判断涨停强度
        strength = self._determine_strength(seal_time, open_times)
        strength_score = self._calculate_strength_score(
            seal_amount, seal_time, open_times
        )
        
        # 2. 计算封单股数和笔数
        seal_shares = int(seal_amount / limit_price)
        seal_orders = max(seal_shares // 10000, 1)  # 假设平均每单1万股
        
        # 3. 估算排队位置
        if total_queue_shares is not None:
            queue_position = total_queue_shares
        else:
            # 根据封单规模估算总排队
            queue_position = self._estimate_queue_position(
                seal_shares, strength, seal_time
            )
        
        queue_ahead_amount = queue_position * limit_price
        
        if queue_position > seal_shares * 10:  # 排队超过封单10倍
            warnings.append(f"排队过长：{queue_position:,}股")
        
        # 4. 计算成交概率
        fill_probability = self._calculate_fill_probability(
            strength, queue_position, seal_shares, target_shares
        )
        
        # 5. 估算等待时间和成交时间
        estimated_wait_time = self._estimate_wait_time(
            strength, queue_position, seal_shares
        )
        
        if fill_probability > 0.5:
            expected_fill_time = current_time + timedelta(minutes=estimated_wait_time)
            # 确保不超过收盘时间
            market_close = current_time.replace(hour=15, minute=0, second=0)
            if expected_fill_time > market_close:
                expected_fill_time = market_close
        else:
            expected_fill_time = None
            warnings.append(f"成交概率低: {fill_probability:.1%}")
        
        return LimitUpQueueStatus(
            symbol=symbol,
            timestamp=current_time,
            limit_price=limit_price,
            seal_amount=seal_amount,
            seal_shares=seal_shares,
            seal_orders=seal_orders,
            queue_position=queue_position,
            queue_ahead_amount=queue_ahead_amount,
            estimated_wait_time=estimated_wait_time,
            fill_probability=fill_probability,
            expected_fill_time=expected_fill_time,
            strength=strength,
            strength_score=strength_score,
            warnings=warnings
        )
    
    def simulate_queue_execution(self,
                                symbol: str,
                                order_time: datetime,
                                target_shares: int,
                                limit_price: float,
                                queue_status: LimitUpQueueStatus,
                                seal_broke: bool = False) -> QueueExecution:
        """
        模拟排队成交过程
        
        Args:
            symbol: 股票代码
            order_time: 下单时间
            target_shares: 目标股数
            limit_price: 涨停价
            queue_status: 排队状态
            seal_broke: 是否开板
            
        Returns:
            QueueExecution: 成交结果
        """
        warnings = []
        
        # 1. 判断是否成交
        if seal_broke:
            # 开板了，大概率成交
            filled = True
            fill_probability = 0.95
            execution_reason = "涨停板打开，顺利成交"
        else:
            # 根据成交概率随机决定
            fill_probability = queue_status.fill_probability
            filled = np.random.random() < fill_probability
            
            if filled:
                execution_reason = f"排队成功成交（概率{fill_probability:.1%}）"
            else:
                execution_reason = f"排队未成交（概率{fill_probability:.1%}）"
                warnings.append("封板过强，排队未能成交")
        
        # 2. 计算成交股数和时间
        if filled:
            # 部分成交场景
            if queue_status.strength == LimitUpStrength.ONE_WORD:
                # 一字板只能成交一部分
                fill_ratio = np.random.uniform(0.1, 0.3)
                filled_shares = int(target_shares * fill_ratio)
                warnings.append(f"一字板仅部分成交: {fill_ratio:.0%}")
            else:
                filled_shares = target_shares
            
            # 成交时间
            if queue_status.expected_fill_time:
                # 在预期时间附近随机
                fill_time = queue_status.expected_fill_time + timedelta(
                    minutes=np.random.uniform(-5, 5)
                )
            else:
                # 尾盘成交
                fill_time = order_time.replace(hour=14, minute=55)
            
            avg_fill_price = limit_price
            final_queue_position = 0
            
        else:
            filled_shares = 0
            fill_time = None
            avg_fill_price = 0
            final_queue_position = queue_status.queue_position
            
        return QueueExecution(
            symbol=symbol,
            order_time=order_time,
            target_shares=target_shares,
            limit_price=limit_price,
            filled=filled,
            filled_shares=filled_shares,
            fill_time=fill_time,
            avg_fill_price=avg_fill_price,
            initial_queue_position=queue_status.queue_position,
            final_queue_position=final_queue_position,
            execution_reason=execution_reason,
            seal_broke=seal_broke,
            warnings=warnings
        )
    
    def _determine_strength(self, 
                           seal_time: datetime,
                           open_times: int) -> LimitUpStrength:
        """判断涨停强度"""
        if open_times >= 3:
            return LimitUpStrength.WEAK_SEAL
        
        hour = seal_time.hour
        minute = seal_time.minute
        
        if hour == 9 and minute <= 30:  # 开盘即涨停
            return LimitUpStrength.ONE_WORD
        elif hour == 9 or (hour == 10 and minute < 45):
            return LimitUpStrength.EARLY_SEAL
        elif hour < 14:
            return LimitUpStrength.MID_SEAL
        else:
            return LimitUpStrength.LATE_SEAL
    
    def _calculate_strength_score(self,
                                  seal_amount: float,
                                  seal_time: datetime,
                                  open_times: int) -> float:
        """
        计算强度评分（0-100）
        
        评分维度：
        1. 封单规模（40分）
        2. 封板时间（30分）
        3. 开板次数（30分）
        """
        score = 0.0
        
        # 1. 封单规模评分（40分）
        # 假设5000万封单为满分
        seal_score = min(seal_amount / 50_000_000, 1.0) * 40
        score += seal_score
        
        # 2. 封板时间评分（30分）- 越早越高
        hour = seal_time.hour
        minute = seal_time.minute
        time_minutes = (hour - 9) * 60 + minute - 30  # 从9:30开始计算
        
        if time_minutes <= 0:  # 开盘即封
            time_score = 30
        elif time_minutes <= 60:  # 1小时内
            time_score = 30 * (1 - time_minutes / 60)
        else:
            time_score = 0
        score += time_score
        
        # 3. 开板次数评分（30分）- 次数越少越高
        open_score = max(30 - open_times * 10, 0)
        score += open_score
        
        return round(score, 2)
    
    def _estimate_queue_position(self,
                                seal_shares: int,
                                strength: LimitUpStrength,
                                seal_time: datetime) -> int:
        """估算排队位置"""
        # 基础排队 = 封单量 * 系数
        base_queue = seal_shares
        
        # 根据强度调整系数
        strength_multipliers = {
            LimitUpStrength.ONE_WORD: 5.0,      # 一字板排队是封单的5倍
            LimitUpStrength.EARLY_SEAL: 3.0,
            LimitUpStrength.MID_SEAL: 2.0,
            LimitUpStrength.LATE_SEAL: 1.5,
            LimitUpStrength.WEAK_SEAL: 1.2
        }
        
        multiplier = strength_multipliers.get(strength, 2.0)
        estimated_queue = int(base_queue * multiplier)
        
        return estimated_queue
    
    def _calculate_fill_probability(self,
                                    strength: LimitUpStrength,
                                    queue_position: int,
                                    seal_shares: int,
                                    target_shares: int) -> float:
        """计算成交概率"""
        # 基础概率（根据强度）
        base_prob = self.fill_probs.get(strength, 0.5)
        
        # 排队位置调整
        queue_ratio = queue_position / seal_shares if seal_shares > 0 else 10
        
        if queue_ratio < 1:  # 排队少于封单
            position_factor = 1.0
        elif queue_ratio < 3:  # 排队1-3倍封单
            position_factor = 0.8
        elif queue_ratio < 5:  # 排队3-5倍
            position_factor = 0.5
        else:  # 排队超过5倍
            position_factor = 0.2
        
        # 订单规模调整（小单更容易成交）
        if target_shares < 10000:  # 1万股以下
            size_factor = 1.1
        elif target_shares < 50000:  # 1-5万股
            size_factor = 1.0
        else:  # 5万股以上
            size_factor = 0.9
        
        # 综合概率
        final_prob = base_prob * position_factor * size_factor
        
        return min(final_prob, 0.99)  # 最高99%
    
    def _estimate_wait_time(self,
                           strength: LimitUpStrength,
                           queue_position: int,
                           seal_shares: int) -> float:
        """估算等待时间（分钟）"""
        # 基础等待时间
        base_wait = {
            LimitUpStrength.ONE_WORD: 180,      # 一字板平均等3小时
            LimitUpStrength.EARLY_SEAL: 120,
            LimitUpStrength.MID_SEAL: 60,
            LimitUpStrength.LATE_SEAL: 30,
            LimitUpStrength.WEAK_SEAL: 10
        }
        
        wait = base_wait.get(strength, 60)
        
        # 排队位置调整
        queue_ratio = queue_position / seal_shares if seal_shares > 0 else 1
        wait *= queue_ratio
        
        return min(wait, 240)  # 最多4小时


# 使用示例
if __name__ == "__main__":
    simulator = LimitUpQueueSimulator()
    
    print("=== 场景1：早盘强势涨停排队 ===")
    
    # 评估排队状态
    queue_status = simulator.evaluate_queue_status(
        symbol="000001.SZ",
        limit_price=11.00,
        seal_amount=50_000_000,  # 5000万封单
        seal_time=datetime(2024, 1, 15, 9, 35),  # 9:35封板
        current_time=datetime(2024, 1, 15, 9, 40),
        target_shares=20_000,
        open_times=0
    )
    
    print(f"股票: {queue_status.symbol}")
    print(f"涨停价: {queue_status.limit_price:.2f}元")
    print(f"封单金额: {queue_status.seal_amount:,.0f}元 ({queue_status.seal_shares:,}股)")
    print(f"涨停强度: {queue_status.strength.value} (评分: {queue_status.strength_score:.1f}/100)")
    print(f"排队位置: {queue_status.queue_position:,}股")
    print(f"成交概率: {queue_status.fill_probability:.1%}")
    print(f"预计等待: {queue_status.estimated_wait_time:.0f}分钟")
    
    if queue_status.expected_fill_time:
        print(f"预计成交时间: {queue_status.expected_fill_time.strftime('%H:%M')}")
    
    if queue_status.warnings:
        print("\n⚠️ 警告:")
        for warning in queue_status.warnings:
            print(f"  - {warning}")
    
    # 模拟排队成交
    print("\n=== 模拟排队成交（10次） ===")
    success_count = 0
    for i in range(10):
        execution = simulator.simulate_queue_execution(
            symbol="000001.SZ",
            order_time=datetime(2024, 1, 15, 9, 40),
            target_shares=20_000,
            limit_price=11.00,
            queue_status=queue_status,
            seal_broke=False
        )
        
        if execution.filled:
            success_count += 1
    
    print(f"10次模拟，成交{success_count}次，成交率: {success_count/10:.0%}")
    print(f"理论成交概率: {queue_status.fill_probability:.1%}")
    
    # 场景2：不同强度对比
    print("\n\n=== 场景2：不同涨停强度对比 ===")
    scenarios = [
        ("一字板", datetime(2024, 1, 15, 9, 30), 0),
        ("早盘封板", datetime(2024, 1, 15, 9, 35), 0),
        ("盘中封板", datetime(2024, 1, 15, 10, 30), 0),
        ("尾盘封板", datetime(2024, 1, 15, 14, 30), 0),
        ("弱封", datetime(2024, 1, 15, 10, 0), 3)
    ]
    
    for name, seal_time, open_times in scenarios:
        status = simulator.evaluate_queue_status(
            symbol="TEST",
            limit_price=10.00,
            seal_amount=30_000_000,
            seal_time=seal_time,
            current_time=seal_time,
            target_shares=10_000,
            open_times=open_times
        )
        
        print(f"\n{name}:")
        print(f"  强度评分: {status.strength_score:.1f}/100")
        print(f"  成交概率: {status.fill_probability:.1%}")
        print(f"  预计等待: {status.estimated_wait_time:.0f}分钟")
