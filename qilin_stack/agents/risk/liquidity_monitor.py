"""
流动性监控模块 (Liquidity Monitor)
评估股票流动性风险，防止在流动性不足时建仓

核心功能：
1. 多维度流动性评估
2. 实时流动性监控
3. 流动性预警机制
4. 建仓/平仓流动性检查
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class LiquidityLevel(Enum):
    """流动性等级"""
    EXCELLENT = "优秀"      # 流动性极好
    GOOD = "良好"           # 流动性良好
    MODERATE = "中等"       # 流动性中等
    POOR = "较差"           # 流动性较差
    VERY_POOR = "极差"      # 流动性极差（禁止交易）


@dataclass
class LiquidityMetrics:
    """流动性指标"""
    symbol: str
    timestamp: datetime
    
    # 成交量相关
    avg_volume_5d: float        # 5日均量
    avg_volume_20d: float       # 20日均量
    volume_ratio: float         # 量比（当日/5日均）
    
    # 流动性深度
    bid_ask_spread: float       # 买卖价差（元）
    spread_ratio: float         # 价差比率（价差/中间价）
    market_depth: float         # 盘口深度（前5档总量）
    
    # 换手率
    turnover_rate: float        # 当日换手率
    avg_turnover_5d: float      # 5日平均换手率
    
    # 流动性评分
    liquidity_score: float      # 综合流动性评分（0-100）
    liquidity_level: LiquidityLevel
    
    # 交易能力评估
    can_buy: bool               # 是否可建仓
    can_sell: bool              # 是否可平仓
    max_position_size: float    # 最大建仓规模（股）
    
    # 警告信息
    warnings: List[str]


class LiquidityMonitor:
    """流动性监控器"""
    
    def __init__(self, 
                 min_avg_volume: float = 1_000_000,      # 最小平均成交量（股）
                 max_spread_ratio: float = 0.002,        # 最大价差比率（0.2%）
                 min_turnover_rate: float = 0.01,        # 最小换手率（1%）
                 min_liquidity_score: float = 60):       # 最小流动性评分
        """
        初始化流动性监控器
        
        Args:
            min_avg_volume: 最小平均成交量（股）
            max_spread_ratio: 最大可接受价差比率
            min_turnover_rate: 最小换手率
            min_liquidity_score: 最小流动性评分
        """
        self.min_avg_volume = min_avg_volume
        self.max_spread_ratio = max_spread_ratio
        self.min_turnover_rate = min_turnover_rate
        self.min_liquidity_score = min_liquidity_score
        
        # 流动性历史记录
        self.liquidity_history: Dict[str, List[LiquidityMetrics]] = {}
    
    def evaluate_liquidity(self, 
                          symbol: str,
                          current_price: float,
                          volume_data: pd.DataFrame,
                          order_book: Optional[Dict] = None) -> LiquidityMetrics:
        """
        评估股票流动性
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            volume_data: 成交量历史数据（包含volume, turnover_rate列）
            order_book: 盘口数据（可选）
            
        Returns:
            LiquidityMetrics: 流动性指标
        """
        timestamp = datetime.now()
        warnings = []
        
        # 1. 计算成交量指标
        volumes = volume_data['volume'].values
        avg_volume_5d = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes.mean()
        avg_volume_20d = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes.mean()
        
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        volume_ratio = current_volume / avg_volume_5d if avg_volume_5d > 0 else 0
        
        if avg_volume_5d < self.min_avg_volume:
            warnings.append(f"平均成交量过低: {avg_volume_5d:,.0f}股")
        
        # 2. 计算换手率指标
        turnover_rates = volume_data['turnover_rate'].values if 'turnover_rate' in volume_data.columns else np.zeros(len(volumes))
        turnover_rate = turnover_rates[-1] if len(turnover_rates) > 0 else 0
        avg_turnover_5d = np.mean(turnover_rates[-5:]) if len(turnover_rates) >= 5 else turnover_rate
        
        if turnover_rate < self.min_turnover_rate:
            warnings.append(f"换手率过低: {turnover_rate:.2%}")
        
        # 3. 计算盘口指标（如果有盘口数据）
        if order_book:
            bid_prices = order_book.get('bid_prices', [])
            ask_prices = order_book.get('ask_prices', [])
            bid_volumes = order_book.get('bid_volumes', [])
            ask_volumes = order_book.get('ask_volumes', [])
            
            if bid_prices and ask_prices:
                bid_ask_spread = ask_prices[0] - bid_prices[0]
                mid_price = (ask_prices[0] + bid_prices[0]) / 2
                spread_ratio = bid_ask_spread / mid_price if mid_price > 0 else 0
                
                if spread_ratio > self.max_spread_ratio:
                    warnings.append(f"买卖价差过大: {spread_ratio:.2%}")
            else:
                bid_ask_spread = 0
                spread_ratio = 0
                warnings.append("缺少盘口数据")
            
            # 计算市场深度（前5档总量）
            market_depth = sum(bid_volumes[:5]) + sum(ask_volumes[:5])
        else:
            bid_ask_spread = 0
            spread_ratio = 0
            market_depth = 0
            warnings.append("未提供盘口数据，使用默认值")
        
        # 4. 计算综合流动性评分（0-100）
        liquidity_score = self._calculate_liquidity_score(
            avg_volume_5d=avg_volume_5d,
            turnover_rate=turnover_rate,
            spread_ratio=spread_ratio,
            volume_ratio=volume_ratio
        )
        
        # 5. 确定流动性等级
        liquidity_level = self._determine_liquidity_level(liquidity_score)
        
        # 6. 评估交易能力
        can_buy = self._can_open_position(liquidity_score, spread_ratio, turnover_rate)
        can_sell = self._can_close_position(liquidity_score, avg_volume_5d)
        
        # 7. 计算最大建仓规模（不超过日均成交量的5%）
        max_position_size = avg_volume_5d * 0.05
        
        metrics = LiquidityMetrics(
            symbol=symbol,
            timestamp=timestamp,
            avg_volume_5d=avg_volume_5d,
            avg_volume_20d=avg_volume_20d,
            volume_ratio=volume_ratio,
            bid_ask_spread=bid_ask_spread,
            spread_ratio=spread_ratio,
            market_depth=market_depth,
            turnover_rate=turnover_rate,
            avg_turnover_5d=avg_turnover_5d,
            liquidity_score=liquidity_score,
            liquidity_level=liquidity_level,
            can_buy=can_buy,
            can_sell=can_sell,
            max_position_size=max_position_size,
            warnings=warnings
        )
        
        # 记录历史
        if symbol not in self.liquidity_history:
            self.liquidity_history[symbol] = []
        self.liquidity_history[symbol].append(metrics)
        
        return metrics
    
    def _calculate_liquidity_score(self,
                                   avg_volume_5d: float,
                                   turnover_rate: float,
                                   spread_ratio: float,
                                   volume_ratio: float) -> float:
        """
        计算综合流动性评分（0-100）
        
        评分维度：
        1. 成交量充足度（40分）
        2. 换手率活跃度（30分）
        3. 价差合理性（20分）
        4. 量比健康度（10分）
        """
        score = 0.0
        
        # 1. 成交量评分（40分）
        volume_score = min(avg_volume_5d / 5_000_000, 1.0) * 40  # 500万股为满分
        score += volume_score
        
        # 2. 换手率评分（30分）
        turnover_score = min(turnover_rate / 0.05, 1.0) * 30  # 5%为满分
        score += turnover_score
        
        # 3. 价差评分（20分）- 价差越小越好
        if spread_ratio <= 0.001:  # 0.1%以下满分
            spread_score = 20
        elif spread_ratio <= 0.003:  # 0.1%-0.3%之间线性递减
            spread_score = 20 * (1 - (spread_ratio - 0.001) / 0.002)
        else:  # 超过0.3%不得分
            spread_score = 0
        score += spread_score
        
        # 4. 量比评分（10分）- 0.8-1.5之间为正常
        if 0.8 <= volume_ratio <= 1.5:
            ratio_score = 10
        elif volume_ratio < 0.8:  # 量比过低
            ratio_score = 10 * (volume_ratio / 0.8)
        else:  # 量比过高（可能异常放量）
            ratio_score = 10 * (2.0 - volume_ratio) if volume_ratio < 2.0 else 0
        score += ratio_score
        
        return round(score, 2)
    
    def _determine_liquidity_level(self, score: float) -> LiquidityLevel:
        """根据评分确定流动性等级"""
        if score >= 85:
            return LiquidityLevel.EXCELLENT
        elif score >= 70:
            return LiquidityLevel.GOOD
        elif score >= 60:
            return LiquidityLevel.MODERATE
        elif score >= 40:
            return LiquidityLevel.POOR
        else:
            return LiquidityLevel.VERY_POOR
    
    def _can_open_position(self, 
                          liquidity_score: float,
                          spread_ratio: float,
                          turnover_rate: float) -> bool:
        """判断是否可以建仓"""
        # 流动性评分必须达标
        if liquidity_score < self.min_liquidity_score:
            return False
        
        # 价差不能过大
        if spread_ratio > self.max_spread_ratio:
            return False
        
        # 换手率不能过低
        if turnover_rate < self.min_turnover_rate:
            return False
        
        return True
    
    def _can_close_position(self, 
                           liquidity_score: float,
                           avg_volume_5d: float) -> bool:
        """判断是否可以平仓（标准放宽）"""
        # 平仓时流动性要求略低（允许40分以上）
        if liquidity_score < 40:
            return False
        
        # 成交量必须有基本保证
        if avg_volume_5d < self.min_avg_volume * 0.5:
            return False
        
        return True
    
    def check_position_size(self,
                           symbol: str,
                           target_shares: int,
                           metrics: Optional[LiquidityMetrics] = None) -> Tuple[bool, str, int]:
        """
        检查建仓规模是否合理
        
        Args:
            symbol: 股票代码
            target_shares: 目标建仓股数
            metrics: 流动性指标（如未提供则使用最新记录）
            
        Returns:
            (是否允许, 原因说明, 建议股数)
        """
        if metrics is None:
            if symbol not in self.liquidity_history or not self.liquidity_history[symbol]:
                return False, "无流动性数据", 0
            metrics = self.liquidity_history[symbol][-1]
        
        # 不超过最大建仓规模
        if target_shares > metrics.max_position_size:
            return False, f"超过最大建仓规模（{metrics.max_position_size:,.0f}股）", int(metrics.max_position_size)
        
        # 检查是否可建仓
        if not metrics.can_buy:
            return False, f"流动性不足（评分:{metrics.liquidity_score:.1f}）", 0
        
        # 根据流动性等级调整建议规模
        if metrics.liquidity_level == LiquidityLevel.EXCELLENT:
            recommended = target_shares
        elif metrics.liquidity_level == LiquidityLevel.GOOD:
            recommended = int(target_shares * 0.8)
        elif metrics.liquidity_level == LiquidityLevel.MODERATE:
            recommended = int(target_shares * 0.5)
        else:
            recommended = int(target_shares * 0.3)
        
        recommended = min(recommended, int(metrics.max_position_size))
        
        return True, "流动性检查通过", recommended
    
    def get_liquidity_report(self, symbol: str, days: int = 5) -> Dict:
        """
        生成流动性报告
        
        Args:
            symbol: 股票代码
            days: 统计天数
            
        Returns:
            流动性报告字典
        """
        if symbol not in self.liquidity_history:
            return {"error": "无历史数据"}
        
        history = self.liquidity_history[symbol][-days:]
        if not history:
            return {"error": "数据不足"}
        
        latest = history[-1]
        
        # 统计流动性趋势
        scores = [m.liquidity_score for m in history]
        avg_score = np.mean(scores)
        score_trend = "上升" if scores[-1] > scores[0] else "下降" if scores[-1] < scores[0] else "平稳"
        
        # 统计可交易天数
        tradable_days = sum(1 for m in history if m.can_buy)
        
        report = {
            "symbol": symbol,
            "report_time": latest.timestamp,
            "current_metrics": {
                "流动性评分": f"{latest.liquidity_score:.1f}/100",
                "流动性等级": latest.liquidity_level.value,
                "平均成交量(5日)": f"{latest.avg_volume_5d:,.0f}股",
                "换手率": f"{latest.turnover_rate:.2%}",
                "买卖价差比率": f"{latest.spread_ratio:.3%}",
                "可建仓": "是" if latest.can_buy else "否",
                "最大建仓规模": f"{latest.max_position_size:,.0f}股"
            },
            "trend_analysis": {
                f"近{days}日平均评分": f"{avg_score:.1f}",
                "评分趋势": score_trend,
                f"可交易天数": f"{tradable_days}/{days}天"
            },
            "warnings": latest.warnings,
            "recommendation": self._generate_recommendation(latest, avg_score)
        }
        
        return report
    
    def _generate_recommendation(self, 
                                metrics: LiquidityMetrics,
                                avg_score: float) -> str:
        """生成交易建议"""
        if metrics.liquidity_level == LiquidityLevel.EXCELLENT:
            return "流动性优秀，可正常交易"
        elif metrics.liquidity_level == LiquidityLevel.GOOD:
            return "流动性良好，可适度交易"
        elif metrics.liquidity_level == LiquidityLevel.MODERATE:
            return "流动性中等，建议减小仓位或分批交易"
        elif metrics.liquidity_level == LiquidityLevel.POOR:
            return "流动性较差，谨慎交易，严格控制仓位"
        else:
            return "流动性极差，不建议交易"


# 使用示例
if __name__ == "__main__":
    # 创建流动性监控器
    monitor = LiquidityMonitor(
        min_avg_volume=1_000_000,
        max_spread_ratio=0.002,
        min_turnover_rate=0.01,
        min_liquidity_score=60
    )
    
    # 模拟成交量数据
    volume_data = pd.DataFrame({
        'volume': [2_500_000, 2_300_000, 2_800_000, 2_600_000, 2_700_000,
                   2_900_000, 3_200_000, 2_800_000, 3_000_000, 3_500_000],
        'turnover_rate': [0.025, 0.023, 0.028, 0.026, 0.027,
                         0.029, 0.032, 0.028, 0.030, 0.035]
    })
    
    # 模拟盘口数据
    order_book = {
        'bid_prices': [10.50, 10.49, 10.48, 10.47, 10.46],
        'ask_prices': [10.51, 10.52, 10.53, 10.54, 10.55],
        'bid_volumes': [50000, 45000, 40000, 35000, 30000],
        'ask_volumes': [48000, 42000, 38000, 33000, 28000]
    }
    
    # 评估流动性
    metrics = monitor.evaluate_liquidity(
        symbol="000001.SZ",
        current_price=10.50,
        volume_data=volume_data,
        order_book=order_book
    )
    
    print("=== 流动性评估结果 ===")
    print(f"股票代码: {metrics.symbol}")
    print(f"流动性评分: {metrics.liquidity_score:.1f}/100")
    print(f"流动性等级: {metrics.liquidity_level.value}")
    print(f"可建仓: {'是' if metrics.can_buy else '否'}")
    print(f"可平仓: {'是' if metrics.can_sell else '否'}")
    print(f"最大建仓规模: {metrics.max_position_size:,.0f}股")
    
    if metrics.warnings:
        print("\n⚠️ 警告:")
        for warning in metrics.warnings:
            print(f"  - {warning}")
    
    # 检查建仓规模
    print("\n=== 建仓规模检查 ===")
    target_shares = 150_000
    allowed, reason, recommended = monitor.check_position_size(
        symbol="000001.SZ",
        target_shares=target_shares
    )
    print(f"目标建仓: {target_shares:,}股")
    print(f"检查结果: {reason}")
    print(f"建议规模: {recommended:,}股")
    
    # 生成报告
    print("\n=== 流动性报告 ===")
    report = monitor.get_liquidity_report("000001.SZ", days=1)
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")
