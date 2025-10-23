"""
动态头寸管理器 (Position Manager)
根据市场状况、账户风险、个股特征动态调整仓位

核心功能：
1. 动态仓位计算（Kelly公式、风险平价）
2. 风险预算分配
3. 相关性控制（避免同板块过度集中）
4. 加减仓决策辅助
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class PositionSizeMethod(Enum):
    """仓位计算方法"""
    FIXED = "固定仓位"              # 固定比例
    KELLY = "凯利公式"              # Kelly Criterion
    RISK_PARITY = "风险平价"        # Risk Parity
    VOLATILITY_ADJUSTED = "波动率调整"  # 根据波动率调整


class RiskLevel(Enum):
    """风险等级"""
    CONSERVATIVE = "保守"   # 保守型（低风险）
    MODERATE = "稳健"       # 稳健型（中等风险）
    AGGRESSIVE = "激进"     # 激进型（高风险）


@dataclass
class PositionLimits:
    """仓位限制"""
    max_single_position: float      # 单个股票最大仓位比例
    max_total_position: float        # 总仓位上限
    max_sector_exposure: float       # 单一板块最大暴露
    max_correlation_exposure: float  # 相关性暴露上限
    
    min_position_size: float         # 最小建仓规模（元）
    max_loss_per_trade: float        # 单笔最大损失比例


@dataclass
class PositionRecommendation:
    """仓位建议"""
    symbol: str
    recommended_shares: int          # 建议股数
    recommended_value: float         # 建议金额
    position_ratio: float            # 占总资金比例
    
    method: PositionSizeMethod       # 计算方法
    rationale: str                   # 推荐理由
    
    # 风险指标
    estimated_risk: float            # 预估风险（标准差）
    max_drawdown_risk: float         # 最大回撤风险
    
    # 调整建议
    adjustment_suggestion: str       # 调整建议
    warnings: List[str]              # 警告信息


@dataclass
class PortfolioMetrics:
    """组合指标"""
    total_value: float               # 总资金
    cash_available: float            # 可用现金
    total_position_value: float      # 持仓市值
    position_ratio: float            # 仓位比例
    
    # 风险指标
    portfolio_volatility: float      # 组合波动率
    max_drawdown: float              # 最大回撤
    beta: float                      # 市场Beta
    
    # 持仓分布
    num_positions: int               # 持仓数量
    top_holdings: List[Tuple[str, float]]  # 前N大持仓
    sector_exposure: Dict[str, float]      # 板块暴露
    
    # 风险预算
    risk_budget_used: float          # 已使用风险预算
    risk_budget_available: float     # 可用风险预算


class PositionManager:
    """头寸管理器"""
    
    def __init__(self,
                 total_capital: float,
                 risk_level: RiskLevel = RiskLevel.MODERATE,
                 position_limits: Optional[PositionLimits] = None):
        """
        初始化头寸管理器
        
        Args:
            total_capital: 总资金
            risk_level: 风险等级
            position_limits: 仓位限制（如不提供则使用默认）
        """
        self.total_capital = total_capital
        self.risk_level = risk_level
        
        # 设置仓位限制
        if position_limits is None:
            self.limits = self._get_default_limits(risk_level)
        else:
            self.limits = position_limits
        
        # 当前持仓
        self.positions: Dict[str, Dict] = {}  # {symbol: {shares, avg_cost, sector, ...}}
        
        # 板块映射（简化版）
        self.sector_map: Dict[str, str] = {}
    
    def _get_default_limits(self, risk_level: RiskLevel) -> PositionLimits:
        """根据风险等级获取默认仓位限制"""
        if risk_level == RiskLevel.CONSERVATIVE:
            return PositionLimits(
                max_single_position=0.10,      # 单股最多10%
                max_total_position=0.60,       # 总仓位60%
                max_sector_exposure=0.25,      # 单板块25%
                max_correlation_exposure=0.35,
                min_position_size=5000,
                max_loss_per_trade=0.02        # 单笔最大损失2%
            )
        elif risk_level == RiskLevel.MODERATE:
            return PositionLimits(
                max_single_position=0.15,
                max_total_position=0.80,
                max_sector_exposure=0.35,
                max_correlation_exposure=0.50,
                min_position_size=3000,
                max_loss_per_trade=0.03
            )
        else:  # AGGRESSIVE
            return PositionLimits(
                max_single_position=0.20,
                max_total_position=0.95,
                max_sector_exposure=0.50,
                max_correlation_exposure=0.70,
                min_position_size=2000,
                max_loss_per_trade=0.05
            )
    
    def calculate_position_size(self,
                               symbol: str,
                               current_price: float,
                               stop_loss_price: float,
                               win_rate: Optional[float] = None,
                               avg_return: Optional[float] = None,
                               volatility: Optional[float] = None,
                               method: PositionSizeMethod = PositionSizeMethod.VOLATILITY_ADJUSTED,
                               sector: Optional[str] = None) -> PositionRecommendation:
        """
        计算建仓规模
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            stop_loss_price: 止损价格
            win_rate: 胜率（Kelly公式需要）
            avg_return: 平均收益率（Kelly公式需要）
            volatility: 波动率（波动率调整需要）
            method: 计算方法
            sector: 所属板块
            
        Returns:
            PositionRecommendation
        """
        warnings = []
        
        # 1. 计算风险（止损距离）
        risk_per_share = abs(current_price - stop_loss_price)
        risk_ratio = risk_per_share / current_price
        
        if risk_ratio > 0.15:  # 止损超过15%
            warnings.append(f"止损距离过大: {risk_ratio:.1%}")
        
        # 2. 根据方法计算仓位
        if method == PositionSizeMethod.FIXED:
            position_ratio = self.limits.max_single_position * 0.5  # 使用50%的上限
            rationale = "固定仓位策略"
            
        elif method == PositionSizeMethod.KELLY and win_rate and avg_return:
            kelly_fraction = self._kelly_criterion(win_rate, avg_return, risk_ratio)
            position_ratio = min(kelly_fraction * 0.5, self.limits.max_single_position)  # 半凯利
            rationale = f"凯利公式（胜率{win_rate:.1%}，平均收益{avg_return:.1%}）"
            
        elif method == PositionSizeMethod.VOLATILITY_ADJUSTED and volatility:
            base_ratio = 0.10  # 基准仓位10%
            volatility_adjustment = 0.02 / volatility if volatility > 0 else 1.0  # 目标波动率2%
            position_ratio = min(base_ratio * volatility_adjustment, self.limits.max_single_position)
            rationale = f"波动率调整（波动率{volatility:.2%}）"
            
        else:  # 默认：基于风险的仓位
            max_loss_amount = self.total_capital * self.limits.max_loss_per_trade
            max_shares = max_loss_amount / risk_per_share
            position_ratio = (max_shares * current_price) / self.total_capital
            position_ratio = min(position_ratio, self.limits.max_single_position)
            rationale = "基于风险预算的仓位计算"
        
        # 3. 检查仓位限制
        position_ratio, adjustment = self._apply_position_limits(
            symbol, position_ratio, sector
        )
        
        # 4. 计算建议股数和金额
        recommended_value = self.total_capital * position_ratio
        recommended_shares = int(recommended_value / current_price / 100) * 100  # 取整到百股
        recommended_value = recommended_shares * current_price
        
        if recommended_value < self.limits.min_position_size:
            warnings.append(f"建仓金额过小: {recommended_value:.0f}元")
            recommended_shares = 0
            recommended_value = 0
        
        # 5. 估算风险
        estimated_risk = recommended_value * risk_ratio
        max_drawdown_risk = recommended_value * 0.20  # 假设最大回撤20%
        
        return PositionRecommendation(
            symbol=symbol,
            recommended_shares=recommended_shares,
            recommended_value=recommended_value,
            position_ratio=position_ratio,
            method=method,
            rationale=rationale,
            estimated_risk=estimated_risk,
            max_drawdown_risk=max_drawdown_risk,
            adjustment_suggestion=adjustment,
            warnings=warnings
        )
    
    def _kelly_criterion(self, win_rate: float, avg_return: float, avg_loss: float) -> float:
        """
        凯利公式计算最优仓位
        f* = (p * b - q) / b
        其中 p=胜率, q=1-p, b=盈亏比
        """
        if avg_loss <= 0:
            return 0
        
        p = win_rate
        q = 1 - p
        b = avg_return / avg_loss  # 盈亏比
        
        kelly = (p * b - q) / b
        return max(0, min(kelly, 0.25))  # 限制在0-25%之间
    
    def _apply_position_limits(self,
                               symbol: str,
                               position_ratio: float,
                               sector: Optional[str]) -> Tuple[float, str]:
        """应用仓位限制"""
        adjustment = ""
        
        # 1. 单股上限
        if position_ratio > self.limits.max_single_position:
            position_ratio = self.limits.max_single_position
            adjustment = f"触及单股上限{self.limits.max_single_position:.0%}"
        
        # 2. 总仓位上限
        current_position_ratio = self._get_current_position_ratio()
        if current_position_ratio + position_ratio > self.limits.max_total_position:
            position_ratio = self.limits.max_total_position - current_position_ratio
            adjustment = f"触及总仓位上限{self.limits.max_total_position:.0%}"
        
        # 3. 板块暴露上限
        if sector:
            sector_exposure = self._get_sector_exposure(sector)
            if sector_exposure + position_ratio > self.limits.max_sector_exposure:
                position_ratio = self.limits.max_sector_exposure - sector_exposure
                adjustment = f"触及板块暴露上限{self.limits.max_sector_exposure:.0%}"
        
        return max(0, position_ratio), adjustment or "无调整"
    
    def _get_current_position_ratio(self) -> float:
        """计算当前总仓位比例"""
        total_position_value = sum(
            pos['shares'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        return total_position_value / self.total_capital
    
    def _get_sector_exposure(self, sector: str) -> float:
        """计算板块暴露"""
        sector_value = sum(
            pos['shares'] * pos['current_price']
            for sym, pos in self.positions.items()
            if self.sector_map.get(sym) == sector
        )
        return sector_value / self.total_capital
    
    def add_position(self,
                    symbol: str,
                    shares: int,
                    price: float,
                    sector: Optional[str] = None):
        """添加持仓"""
        if symbol in self.positions:
            # 加仓
            old_shares = self.positions[symbol]['shares']
            old_cost = self.positions[symbol]['avg_cost']
            new_shares = old_shares + shares
            new_cost = (old_shares * old_cost + shares * price) / new_shares
            
            self.positions[symbol]['shares'] = new_shares
            self.positions[symbol]['avg_cost'] = new_cost
        else:
            # 新建仓
            self.positions[symbol] = {
                'shares': shares,
                'avg_cost': price,
                'current_price': price,
                'sector': sector
            }
        
        if sector:
            self.sector_map[symbol] = sector
    
    def remove_position(self, symbol: str, shares: Optional[int] = None):
        """减少或清空持仓"""
        if symbol not in self.positions:
            return
        
        if shares is None or shares >= self.positions[symbol]['shares']:
            # 清仓
            del self.positions[symbol]
        else:
            # 减仓
            self.positions[symbol]['shares'] -= shares
    
    def update_prices(self, price_updates: Dict[str, float]):
        """更新持仓价格"""
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                self.positions[symbol]['current_price'] = price
    
    def get_portfolio_metrics(self, market_volatility: float = 0.02) -> PortfolioMetrics:
        """获取组合指标"""
        # 计算持仓市值
        total_position_value = sum(
            pos['shares'] * pos['current_price']
            for pos in self.positions.values()
        )
        
        cash_available = self.total_capital - total_position_value
        position_ratio = total_position_value / self.total_capital
        
        # 统计前N大持仓
        holdings = [
            (sym, pos['shares'] * pos['current_price'] / self.total_capital)
            for sym, pos in self.positions.items()
        ]
        top_holdings = sorted(holdings, key=lambda x: x[1], reverse=True)[:5]
        
        # 统计板块暴露
        sector_exposure = {}
        for sym, pos in self.positions.items():
            sector = self.sector_map.get(sym, "未知")
            value = pos['shares'] * pos['current_price']
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value / self.total_capital
        
        # 估算风险指标（简化版）
        portfolio_volatility = market_volatility * np.sqrt(position_ratio)
        max_drawdown = -0.15 * position_ratio  # 假设最大回撤15%
        
        # 风险预算
        risk_budget_used = position_ratio * 0.5  # 简化估算
        risk_budget_available = 1.0 - risk_budget_used
        
        return PortfolioMetrics(
            total_value=self.total_capital,
            cash_available=cash_available,
            total_position_value=total_position_value,
            position_ratio=position_ratio,
            portfolio_volatility=portfolio_volatility,
            max_drawdown=max_drawdown,
            beta=1.0,  # 简化为1
            num_positions=len(self.positions),
            top_holdings=top_holdings,
            sector_exposure=sector_exposure,
            risk_budget_used=risk_budget_used,
            risk_budget_available=risk_budget_available
        )
    
    def should_reduce_position(self,
                              symbol: str,
                              current_price: float,
                              stop_loss_price: float) -> Tuple[bool, str, int]:
        """
        判断是否应该减仓
        
        Returns:
            (是否减仓, 原因, 建议减持股数)
        """
        if symbol not in self.positions:
            return False, "无持仓", 0
        
        pos = self.positions[symbol]
        
        # 1. 检查是否触及止损
        if current_price <= stop_loss_price:
            return True, "触及止损", pos['shares']
        
        # 2. 检查浮盈是否过大（适度止盈）
        profit_ratio = (current_price - pos['avg_cost']) / pos['avg_cost']
        if profit_ratio > 0.30:  # 盈利超过30%
            reduce_shares = int(pos['shares'] * 0.3 / 100) * 100  # 减仓30%
            return True, f"浮盈过大({profit_ratio:.1%})，建议止盈", reduce_shares
        
        # 3. 检查是否超过仓位上限（市值上涨导致）
        position_value = pos['shares'] * current_price
        position_ratio = position_value / self.total_capital
        
        if position_ratio > self.limits.max_single_position * 1.2:  # 超过上限20%
            excess_value = position_value - self.total_capital * self.limits.max_single_position
            reduce_shares = int(excess_value / current_price / 100) * 100
            return True, "超过单股仓位上限", reduce_shares
        
        return False, "正常持仓", 0


# 使用示例
if __name__ == "__main__":
    # 创建头寸管理器
    manager = PositionManager(
        total_capital=1_000_000,  # 100万资金
        risk_level=RiskLevel.MODERATE
    )
    
    print("=== 仓位计算示例 ===")
    
    # 计算建仓规模
    recommendation = manager.calculate_position_size(
        symbol="000001.SZ",
        current_price=10.50,
        stop_loss_price=9.50,
        win_rate=0.60,
        avg_return=0.08,
        volatility=0.025,
        method=PositionSizeMethod.KELLY,
        sector="金融"
    )
    
    print(f"股票: {recommendation.symbol}")
    print(f"建议股数: {recommendation.recommended_shares:,}股")
    print(f"建议金额: {recommendation.recommended_value:,.0f}元")
    print(f"仓位比例: {recommendation.position_ratio:.1%}")
    print(f"计算方法: {recommendation.method.value}")
    print(f"理由: {recommendation.rationale}")
    print(f"预估风险: {recommendation.estimated_risk:,.0f}元")
    print(f"调整说明: {recommendation.adjustment_suggestion}")
    
    if recommendation.warnings:
        print("\n警告:")
        for warning in recommendation.warnings:
            print(f"  - {warning}")
    
    # 模拟建仓
    print("\n=== 建仓操作 ===")
    manager.add_position(
        symbol="000001.SZ",
        shares=recommendation.recommended_shares,
        price=10.50,
        sector="金融"
    )
    
    # 更新价格
    manager.update_prices({"000001.SZ": 11.20})
    
    # 获取组合指标
    print("\n=== 组合指标 ===")
    metrics = manager.get_portfolio_metrics()
    
    print(f"总资金: {metrics.total_value:,.0f}元")
    print(f"持仓市值: {metrics.total_position_value:,.0f}元")
    print(f"可用现金: {metrics.cash_available:,.0f}元")
    print(f"仓位比例: {metrics.position_ratio:.1%}")
    print(f"组合波动率: {metrics.portfolio_volatility:.2%}")
    print(f"持仓数量: {metrics.num_positions}")
    
    print("\n前5大持仓:")
    for symbol, ratio in metrics.top_holdings:
        print(f"  {symbol}: {ratio:.1%}")
    
    print("\n板块暴露:")
    for sector, ratio in metrics.sector_exposure.items():
        print(f"  {sector}: {ratio:.1%}")
    
    # 检查是否需要减仓
    print("\n=== 减仓检查 ===")
    should_reduce, reason, reduce_shares = manager.should_reduce_position(
        symbol="000001.SZ",
        current_price=11.20,
        stop_loss_price=9.50
    )
    
    print(f"是否减仓: {'是' if should_reduce else '否'}")
    print(f"原因: {reason}")
    if should_reduce:
        print(f"建议减持: {reduce_shares:,}股")
