"""
麒麟量化系统 - 风险管理模块
实现仓位管理、止损止盈、风险度量等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    cvar_95: float  # 95% Conditional VaR
    cvar_99: float  # 99% Conditional VaR
    max_drawdown: float  # 最大回撤
    current_drawdown: float  # 当前回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    beta: float  # 贝塔系数
    correlation: float  # 相关性
    volatility: float  # 波动率
    downside_volatility: float  # 下行波动率
    risk_level: RiskLevel  # 风险等级


class PositionSizer:
    """仓位管理器"""
    
    def __init__(
        self,
        max_position_size: float = 0.2,
        max_positions: int = 10,
        risk_per_trade: float = 0.02,
        use_kelly: bool = False
    ):
        """
        初始化仓位管理器
        
        Args:
            max_position_size: 单个持仓最大占比
            max_positions: 最大持仓数量
            risk_per_trade: 每笔交易风险
            use_kelly: 是否使用凯利公式
        """
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.use_kelly = use_kelly
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        stop_loss: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> int:
        """
        计算仓位大小
        
        Args:
            capital: 可用资金
            price: 当前价格
            stop_loss: 止损价格
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            volatility: 波动率
            
        Returns:
            建议仓位数量
        """
        if self.use_kelly and win_rate and avg_win and avg_loss:
            # 凯利公式
            kelly_fraction = self._calculate_kelly_fraction(
                win_rate, avg_win, avg_loss
            position_value = capital * min(kelly_fraction, self.max_position_size)
        
        elif stop_loss:
            # 基于止损的仓位计算
            risk_amount = capital * self.risk_per_trade
            risk_per_share = abs(price - stop_loss)
            
            if risk_per_share > 0:
                shares = risk_amount / risk_per_share
            else:
                shares = 0
            
            # 限制最大仓位
            max_shares = (capital * self.max_position_size) / price
            position_value = min(shares, max_shares) * price
        
        elif volatility:
            # 基于波动率的仓位计算
            target_volatility = 0.15  # 目标年化波动率
            position_fraction = target_volatility / (volatility * np.sqrt(252))
            position_fraction = min(position_fraction, self.max_position_size)
            position_value = capital * position_fraction
        
        else:
            # 等权重分配
            position_value = capital / self.max_positions
            position_value = min(position_value, capital * self.max_position_size)
        
        # 计算股数（向下取整到100股）
        shares = int(position_value / price / 100) * 100
        
        return max(shares, 0)
    
    def _calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        计算凯利比例
        
        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            
        Returns:
            凯利比例
        """
        if avg_loss == 0:
            return 0
        
        # f = (p * b - q) / b
        # p: 胜率, q: 败率, b: 盈亏比
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly = (win_rate * b - q) / b
        
        # 使用保守的凯利比例（25%）
        kelly = kelly * 0.25
        
        return max(0, min(kelly, 1))
    
    def rebalance_positions(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        capital: float,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        重新平衡仓位
        
        Args:
            current_positions: 当前持仓
            target_weights: 目标权重
            capital: 总资金
            prices: 当前价格
            
        Returns:
            调整后的仓位
        """
        adjustments = {}
        
        for symbol, target_weight in target_weights.items():
            target_value = capital * min(target_weight, self.max_position_size)
            target_shares = int(target_value / prices[symbol] / 100) * 100
            
            current_shares = current_positions.get(symbol, 0)
            adjustment = target_shares - current_shares
            
            if abs(adjustment) > 100:  # 最小调整单位
                adjustments[symbol] = adjustment
        
        return adjustments


class StopLossManager:
    """止损止盈管理器"""
    
    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20,
        trailing_stop_pct: float = 0.05,
        use_atr: bool = True
    ):
        """
        初始化止损管理器
        
        Args:
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            trailing_stop_pct: 移动止损百分比
            use_atr: 是否使用ATR
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.use_atr = use_atr
        
        # 记录每个持仓的止损止盈位
        self.stop_levels = {}
        self.highest_prices = {}
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: Optional[float] = None,
        support_level: Optional[float] = None
    ) -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            atr: 平均真实波幅
            support_level: 支撑位
            
        Returns:
            止损价格
        """
        if self.use_atr and atr:
            # 基于ATR的止损
            stop_loss = entry_price - 2 * atr
        elif support_level:
            # 基于支撑位的止损
            stop_loss = support_level * 0.98
        else:
            # 固定百分比止损
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        atr: Optional[float] = None,
        resistance_level: Optional[float] = None,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        计算止盈价格
        
        Args:
            entry_price: 入场价格
            atr: 平均真实波幅
            resistance_level: 阻力位
            risk_reward_ratio: 风险收益比
            
        Returns:
            止盈价格
        """
        if self.use_atr and atr:
            # 基于ATR的止盈
            take_profit = entry_price + risk_reward_ratio * 2 * atr
        elif resistance_level:
            # 基于阻力位的止盈
            take_profit = resistance_level * 0.98
        else:
            # 固定百分比止盈
            take_profit = entry_price * (1 + self.take_profit_pct)
        
        return take_profit
    
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        entry_price: float
    ) -> Optional[float]:
        """
        更新移动止损
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            entry_price: 入场价格
            
        Returns:
            新的止损价格
        """
        # 更新最高价
        if symbol not in self.highest_prices:
            self.highest_prices[symbol] = current_price
        else:
            self.highest_prices[symbol] = max(
                self.highest_prices[symbol],
                current_price
        
        # 计算移动止损
        highest = self.highest_prices[symbol]
        trailing_stop = highest * (1 - self.trailing_stop_pct)
        
        # 止损只能上移，不能下移
        current_stop = self.stop_levels.get(symbol, entry_price * (1 - self.stop_loss_pct))
        new_stop = max(current_stop, trailing_stop)
        
        self.stop_levels[symbol] = new_stop
        
        return new_stop
    
    def check_exit_signals(
        self,
        symbol: str,
        current_price: float,
        entry_price: float
    ) -> Tuple[bool, str]:
        """
        检查退出信号
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            entry_price: 入场价格
            
        Returns:
            (是否退出, 退出原因)
        """
        # 检查止损
        stop_loss = self.stop_levels.get(
            symbol,
            entry_price * (1 - self.stop_loss_pct)
        
        if current_price <= stop_loss:
            return True, "stop_loss"
        
        # 检查止盈
        take_profit = entry_price * (1 + self.take_profit_pct)
        if current_price >= take_profit:
            return True, "take_profit"
        
        # 更新移动止损
        self.update_trailing_stop(symbol, current_price, entry_price)
        
        return False, ""


class RiskCalculator:
    """风险度量计算器"""
    
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        计算VaR (Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            method: 计算方法 ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR值
        """
        if len(returns) < 20:
            return 0
        
        if method == 'historical':
            # 历史模拟法
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # 参数法（假设正态分布）
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean - z_score * std
        
        elif method == 'monte_carlo':
            # 蒙特卡洛模拟
            mean = returns.mean()
            std = returns.std()
            simulations = np.random.normal(mean, std, 10000)
            var = np.percentile(simulations, (1 - confidence_level) * 100)
        
        else:
            var = 0
        
        return abs(var)
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        计算CVaR (Conditional Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            CVaR值
        """
        var = RiskCalculator.calculate_var(returns, confidence_level)
        
        # 计算超过VaR的平均损失
        losses_beyond_var = returns[returns <= -var]
        
        if len(losses_beyond_var) > 0:
            cvar = abs(losses_beyond_var.mean())
        else:
            cvar = var
        
        return cvar
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        计算最大回撤
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            (最大回撤, 开始日期, 结束日期)
        """
        if len(equity_curve) < 2:
            return 0, None, None
        
        # 计算累积最高点
        cummax = equity_curve.expanding().max()
        
        # 计算回撤
        drawdown = (equity_curve - cummax) / cummax
        
        # 找到最大回撤
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # 找到回撤开始点
        start_idx = equity_curve[:max_dd_idx][cummax[:max_dd_idx] == cummax[max_dd_idx]].index[-1]
        
        return abs(max_dd), start_idx, max_dd_idx
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.03
    ) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            夏普比率
        """
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        
        if excess_returns.std() > 0:
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.03,
        target_return: float = 0
    ) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            target_return: 目标收益率
            
        Returns:
            索提诺比率
        """
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        
        # 计算下行偏差
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino = excess_returns.mean() / downside_std * np.sqrt(252)
            else:
                sortino = 0
        else:
            sortino = float('inf')  # 没有下行风险
        
        return sortino
    
    @staticmethod
    def calculate_beta(
        returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        计算贝塔系数
        
        Args:
            returns: 资产收益率
            market_returns: 市场收益率
            
        Returns:
            贝塔系数
        """
        if len(returns) < 20 or len(market_returns) < 20:
            return 1.0
        
        # 对齐数据
        aligned = pd.DataFrame({
            'asset': returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < 20:
            return 1.0
        
        # 计算协方差和市场方差
        covariance = aligned['asset'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        
        if market_variance > 0:
            beta = covariance / market_variance
        else:
            beta = 1.0
        
        return beta


class RiskManager:
    """综合风险管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化风险管理器
        
        Args:
            config: 配置参数
        """
        config = config or {}
        
        # 初始化各组件
        self.position_sizer = PositionSizer(
            max_position_size=config.get('max_position_size', 0.2),
            max_positions=config.get('max_positions', 10),
            risk_per_trade=config.get('risk_per_trade', 0.02)
        
        self.stop_loss_manager = StopLossManager(
            stop_loss_pct=config.get('stop_loss', 0.08),
            take_profit_pct=config.get('take_profit', 0.20),
            trailing_stop_pct=config.get('trailing_stop', 0.05)
        
        self.risk_calculator = RiskCalculator()
        
        # 风险限制
        self.max_drawdown_limit = config.get('max_drawdown', 0.15)
        self.max_var_limit = config.get('max_var', 0.05)
        self.min_sharpe_ratio = config.get('min_sharpe', 0.5)
        
    def assess_portfolio_risk(
        self,
        returns: pd.Series,
        positions: Dict[str, float],
        prices: Dict[str, float],
        capital: float
    ) -> RiskMetrics:
        """
        评估组合风险
        
        Args:
            returns: 历史收益率
            positions: 当前持仓
            prices: 当前价格
            capital: 总资金
            
        Returns:
            风险指标
        """
        # 计算各项风险指标
        var_95 = self.risk_calculator.calculate_var(returns, 0.95)
        var_99 = self.risk_calculator.calculate_var(returns, 0.99)
        cvar_95 = self.risk_calculator.calculate_cvar(returns, 0.95)
        cvar_99 = self.risk_calculator.calculate_cvar(returns, 0.99)
        
        # 计算回撤
        equity_curve = (1 + returns).cumprod() * capital
        max_drawdown, _, _ = self.risk_calculator.calculate_max_drawdown(equity_curve)
        current_drawdown = (equity_curve.iloc[-1] - equity_curve.max()) / equity_curve.max()
        
        # 计算风险调整收益
        sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(returns)
        sortino_ratio = self.risk_calculator.calculate_sortino_ratio(returns)
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 判断风险等级
        risk_level = self._determine_risk_level(
            var_95, max_drawdown, volatility, sharpe_ratio
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            beta=1.0,  # 需要市场数据计算
            correlation=0,  # 需要市场数据计算
            volatility=volatility,
            downside_volatility=downside_volatility,
            risk_level=risk_level
    
    def _determine_risk_level(
        self,
        var: float,
        max_drawdown: float,
        volatility: float,
        sharpe_ratio: float
    ) -> RiskLevel:
        """
        判断风险等级
        
        Args:
            var: VaR值
            max_drawdown: 最大回撤
            volatility: 波动率
            sharpe_ratio: 夏普比率
            
        Returns:
            风险等级
        """
        risk_score = 0
        
        # VaR评分
        if var > 0.1:
            risk_score += 3
        elif var > 0.05:
            risk_score += 2
        elif var > 0.02:
            risk_score += 1
        
        # 回撤评分
        if max_drawdown > 0.2:
            risk_score += 3
        elif max_drawdown > 0.1:
            risk_score += 2
        elif max_drawdown > 0.05:
            risk_score += 1
        
        # 波动率评分
        if volatility > 0.3:
            risk_score += 3
        elif volatility > 0.2:
            risk_score += 2
        elif volatility > 0.1:
            risk_score += 1
        
        # 夏普比率评分（反向）
        if sharpe_ratio < 0:
            risk_score += 3
        elif sharpe_ratio < 0.5:
            risk_score += 2
        elif sharpe_ratio < 1:
            risk_score += 1
        
        # 综合评级
        if risk_score >= 9:
            return RiskLevel.EXTREME
        elif risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def should_reduce_risk(self, metrics: RiskMetrics) -> bool:
        """
        判断是否需要降低风险
        
        Args:
            metrics: 风险指标
            
        Returns:
            是否需要降低风险
        """
        if metrics.risk_level == RiskLevel.EXTREME:
            return True
        
        if metrics.max_drawdown > self.max_drawdown_limit:
            return True
        
        if metrics.var_95 > self.max_var_limit:
            return True
        
        if metrics.sharpe_ratio < self.min_sharpe_ratio:
            return True
        
        return False
    
    def adjust_position_for_risk(
        self,
        symbol: str,
        suggested_size: int,
        current_risk: RiskMetrics
    ) -> int:
        """
        根据风险调整仓位
        
        Args:
            symbol: 股票代码
            suggested_size: 建议仓位
            current_risk: 当前风险
            
        Returns:
            调整后的仓位
        """
        adjustment_factor = 1.0
        
        # 根据风险等级调整
        if current_risk.risk_level == RiskLevel.EXTREME:
            adjustment_factor = 0.25
        elif current_risk.risk_level == RiskLevel.HIGH:
            adjustment_factor = 0.5
        elif current_risk.risk_level == RiskLevel.MEDIUM:
            adjustment_factor = 0.75
        
        # 根据回撤调整
        if current_risk.current_drawdown < -0.1:
            adjustment_factor *= 0.5
        
        # 根据VaR调整
        if current_risk.var_95 > 0.05:
            adjustment_factor *= 0.75
        
        adjusted_size = int(suggested_size * adjustment_factor / 100) * 100
        
        return max(0, adjusted_size)


# 使用示例
if __name__ == "__main__":
    import yfinance as yf
    from scipy import stats
    
    # 创建风险管理器
    risk_manager = RiskManager({
        'max_position_size': 0.2,
        'max_positions': 10,
        'stop_loss': 0.08,
        'take_profit': 0.20
    })
    
    # 生成模拟数据
    returns = pd.Series(np.random.randn(252) * 0.02)
    
    # 评估风险
    metrics = risk_manager.assess_portfolio_risk(
        returns=returns,
        positions={'AAPL': 100, 'GOOGL': 50},
        prices={'AAPL': 150, 'GOOGL': 2800},
        capital=100000
    
    print("=== 风险评估结果 ===")
    print(f"VaR (95%): {metrics.var_95:.2%}")
    print(f"CVaR (95%): {metrics.cvar_95:.2%}")
    print(f"最大回撤: {metrics.max_drawdown:.2%}")
    print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"风险等级: {metrics.risk_level.value}")
    print(f"是否需要降低风险: {risk_manager.should_reduce_risk(metrics)}")
    
    # 计算建议仓位
    position_size = risk_manager.position_sizer.calculate_position_size(
        capital=100000,
        price=150,
        stop_loss=138,
        volatility=0.25
    print(f"\n建议仓位: {position_size}股")
    
    # 根据风险调整仓位
    adjusted_size = risk_manager.adjust_position_for_risk(
        'AAPL',
        position_size,
        metrics
    print(f"风险调整后仓位: {adjusted_size}股")