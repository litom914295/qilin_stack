"""
实时风险对冲系统
动态风险敞口监控、自动对冲策略、Delta中性对冲
支持期权、期货等衍生品对冲
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# 对冲工具类型
# ============================================================================

class HedgingInstrument(Enum):
    """对冲工具"""
    INDEX_FUTURE = "index_future"    # 指数期货
    ETF = "etf"                      # ETF
    OPTION = "option"                # 期权
    STOCK = "stock"                  # 股票


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    price: float
    value: float
    beta: float = 1.0


@dataclass
class RiskMetrics:
    """风险指标"""
    total_value: float
    market_exposure: float      # 市场敞口
    beta_exposure: float        # Beta敞口
    delta: float               # Delta
    gamma: float = 0.0         # Gamma
    vega: float = 0.0          # Vega
    theta: float = 0.0         # Theta


# ============================================================================
# 风险敞口监控器
# ============================================================================

class RiskExposureMonitor:
    """风险敞口监控器"""
    
    def __init__(self, 
                 portfolio: Dict[str, Position],
                 benchmark: str = "000300.SH"):
        """
        初始化风险监控器
        
        Args:
            portfolio: 投资组合
            benchmark: 基准指数
        """
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.risk_limits = {}
        
        logger.info(f"风险监控初始化: {len(portfolio)} 个持仓")
    
    def calculate_exposure(self) -> RiskMetrics:
        """
        计算风险敞口
        
        Returns:
            风险指标
        """
        total_value = sum(pos.value for pos in self.portfolio.values())
        
        # 市场敞口 = 总持仓价值
        market_exposure = total_value
        
        # Beta敞口 = Σ(持仓价值 × Beta)
        beta_exposure = sum(pos.value * pos.beta for pos in self.portfolio.values())
        
        # Delta = Beta敞口 / 总价值
        delta = beta_exposure / total_value if total_value > 0 else 0
        
        metrics = RiskMetrics(
            total_value=total_value,
            market_exposure=market_exposure,
            beta_exposure=beta_exposure,
            delta=delta
        )
        
        logger.info(f"风险敞口: 总价值={total_value:.2f}, Beta敞口={beta_exposure:.2f}, Delta={delta:.2f}")
        return metrics
    
    def set_risk_limit(self, metric_name: str, limit: float):
        """
        设置风险限制
        
        Args:
            metric_name: 风险指标名称
            limit: 限制值
        """
        self.risk_limits[metric_name] = limit
        logger.info(f"设置风险限制: {metric_name} = {limit}")
    
    def check_risk_limits(self, metrics: RiskMetrics) -> List[str]:
        """
        检查是否违反风险限制
        
        Args:
            metrics: 当前风险指标
            
        Returns:
            违规列表
        """
        violations = []
        
        if 'max_exposure' in self.risk_limits:
            if abs(metrics.market_exposure) > self.risk_limits['max_exposure']:
                violations.append(f"市场敞口超限: {metrics.market_exposure:.2f}")
        
        if 'max_delta' in self.risk_limits:
            if abs(metrics.delta) > self.risk_limits['max_delta']:
                violations.append(f"Delta超限: {metrics.delta:.2f}")
        
        if violations:
            logger.warning(f"风险限制违规: {violations}")
        
        return violations


# ============================================================================
# Delta中性对冲
# ============================================================================

class DeltaNeutralHedger:
    """Delta中性对冲器"""
    
    def __init__(self, target_delta: float = 0.0):
        """
        初始化对冲器
        
        Args:
            target_delta: 目标Delta（0为完全中性）
        """
        self.target_delta = target_delta
        logger.info(f"Delta中性对冲初始化: 目标Delta={target_delta}")
    
    def calculate_hedge_size(self, 
                            current_delta: float,
                            hedge_beta: float = 1.0) -> float:
        """
        计算对冲头寸大小
        
        Args:
            current_delta: 当前Delta
            hedge_beta: 对冲工具Beta
            
        Returns:
            需要的对冲头寸大小（负数表示卖出）
        """
        # 需要对冲的Delta = 当前Delta - 目标Delta
        delta_to_hedge = current_delta - self.target_delta
        
        # 对冲头寸 = -需要对冲的Delta / 对冲工具Beta
        hedge_size = -delta_to_hedge / hedge_beta if hedge_beta != 0 else 0
        
        logger.info(f"计算对冲: 当前Delta={current_delta:.2f}, "
                   f"需对冲={delta_to_hedge:.2f}, 对冲头寸={hedge_size:.2f}")
        
        return hedge_size
    
    def generate_hedge_orders(self,
                             current_metrics: RiskMetrics,
                             hedge_instrument: str = "510300.SH",
                             hedge_price: float = 4.0) -> List[Dict[str, Any]]:
        """
        生成对冲订单
        
        Args:
            current_metrics: 当前风险指标
            hedge_instrument: 对冲工具代码
            hedge_price: 对冲工具价格
            
        Returns:
            订单列表
        """
        # 计算需要的对冲数量
        hedge_size = self.calculate_hedge_size(current_metrics.delta)
        
        # 转换为股数
        hedge_shares = int(hedge_size * current_metrics.total_value / hedge_price)
        
        if hedge_shares == 0:
            logger.info("无需对冲")
            return []
        
        # 生成订单
        order = {
            'symbol': hedge_instrument,
            'direction': 'sell' if hedge_shares < 0 else 'buy',
            'quantity': abs(hedge_shares),
            'price': hedge_price,
            'order_type': 'market',
            'purpose': 'delta_hedge'
        }
        
        logger.info(f"生成对冲订单: {order}")
        return [order]


# ============================================================================
# 动态对冲策略
# ============================================================================

class DynamicHedgingStrategy:
    """动态对冲策略"""
    
    def __init__(self,
                 rebalance_threshold: float = 0.1,
                 rebalance_interval: int = 5):
        """
        初始化动态对冲策略
        
        Args:
            rebalance_threshold: 再平衡阈值（Delta偏离度）
            rebalance_interval: 再平衡间隔（分钟）
        """
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_interval = rebalance_interval
        self.last_rebalance = None
        
        logger.info(f"动态对冲策略初始化: 阈值={rebalance_threshold}, 间隔={rebalance_interval}分钟")
    
    def should_rebalance(self, current_delta: float, target_delta: float = 0.0) -> bool:
        """
        判断是否需要再平衡
        
        Args:
            current_delta: 当前Delta
            target_delta: 目标Delta
            
        Returns:
            是否需要再平衡
        """
        # 检查Delta偏离
        delta_deviation = abs(current_delta - target_delta)
        
        if delta_deviation > self.rebalance_threshold:
            logger.info(f"Delta偏离超过阈值: {delta_deviation:.2f} > {self.rebalance_threshold}")
            return True
        
        # 检查时间间隔
        if self.last_rebalance is not None:
            elapsed = (datetime.now() - self.last_rebalance).total_seconds() / 60
            if elapsed >= self.rebalance_interval:
                logger.info(f"达到再平衡时间间隔: {elapsed:.0f}分钟")
                return True
        
        return False
    
    def execute_rebalance(self, hedger: DeltaNeutralHedger, 
                         current_metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """
        执行再平衡
        
        Args:
            hedger: Delta对冲器
            current_metrics: 当前风险指标
            
        Returns:
            对冲订单列表
        """
        orders = hedger.generate_hedge_orders(current_metrics)
        
        if orders:
            self.last_rebalance = datetime.now()
            logger.info(f"执行再平衡: 生成{len(orders)}个订单")
        
        return orders


# ============================================================================
# 期权对冲
# ============================================================================

class OptionHedger:
    """期权对冲器"""
    
    def __init__(self):
        """初始化期权对冲器"""
        logger.info("期权对冲器初始化")
    
    def calculate_option_greeks(self,
                               spot: float,
                               strike: float,
                               volatility: float,
                               time_to_expiry: float,
                               risk_free_rate: float = 0.03) -> Dict[str, float]:
        """
        计算期权希腊字母（简化版Black-Scholes）
        
        Args:
            spot: 标的价格
            strike: 行权价
            volatility: 波动率
            time_to_expiry: 到期时间（年）
            risk_free_rate: 无风险利率
            
        Returns:
            希腊字母字典
        """
        from scipy.stats import norm
        
        # d1和d2
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Delta
        delta = norm.cdf(d1)
        
        # Gamma
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
        
        # Vega
        vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry)
        
        # Theta
        theta = -(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) - \
                risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,  # 转换为1%波动率变化的影响
            'theta': theta / 365  # 转换为每日Theta
        }
        
        logger.info(f"期权Greeks: Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}")
        return greeks
    
    def design_protective_put(self,
                             portfolio_value: float,
                             protection_level: float = 0.95) -> Dict[str, Any]:
        """
        设计保护性看跌期权策略
        
        Args:
            portfolio_value: 组合价值
            protection_level: 保护水平（0.95表示保护95%的价值）
            
        Returns:
            期权策略
        """
        # 计算需要的看跌期权参数
        strike = portfolio_value * protection_level
        
        strategy = {
            'type': 'protective_put',
            'portfolio_value': portfolio_value,
            'put_strike': strike,
            'protection_level': protection_level,
            'description': f'购买行权价{strike:.2f}的看跌期权保护组合'
        }
        
        logger.info(f"设计保护性看跌: {strategy}")
        return strategy


# ============================================================================
# 综合对冲管理器
# ============================================================================

class HedgingManager:
    """综合对冲管理器"""
    
    def __init__(self, portfolio: Dict[str, Position]):
        """
        初始化对冲管理器
        
        Args:
            portfolio: 投资组合
        """
        self.portfolio = portfolio
        self.monitor = RiskExposureMonitor(portfolio)
        self.delta_hedger = DeltaNeutralHedger(target_delta=0.0)
        self.dynamic_strategy = DynamicHedgingStrategy()
        self.option_hedger = OptionHedger()
        
        # 对冲历史
        self.hedge_history = []
        
        logger.info("综合对冲管理器初始化完成")
    
    def update_portfolio(self, portfolio: Dict[str, Position]):
        """更新投资组合"""
        self.portfolio = portfolio
        self.monitor.portfolio = portfolio
    
    def run_hedging_cycle(self) -> List[Dict[str, Any]]:
        """
        运行对冲周期
        
        Returns:
            对冲订单列表
        """
        # 1. 计算当前风险敞口
        current_metrics = self.monitor.calculate_exposure()
        
        # 2. 检查风险限制
        violations = self.monitor.check_risk_limits(current_metrics)
        
        if violations:
            logger.warning(f"风险限制违规: {violations}")
        
        # 3. 判断是否需要对冲
        if self.dynamic_strategy.should_rebalance(current_metrics.delta):
            # 4. 生成对冲订单
            orders = self.dynamic_strategy.execute_rebalance(
                self.delta_hedger,
                current_metrics
            )
            
            # 5. 记录历史
            self.hedge_history.append({
                'timestamp': datetime.now(),
                'metrics': current_metrics,
                'orders': orders
            })
            
            return orders
        
        logger.info("当前无需对冲")
        return []
    
    def get_hedging_report(self) -> pd.DataFrame:
        """获取对冲报告"""
        if not self.hedge_history:
            return pd.DataFrame()
        
        records = []
        for record in self.hedge_history:
            records.append({
                'timestamp': record['timestamp'],
                'total_value': record['metrics'].total_value,
                'delta': record['metrics'].delta,
                'n_orders': len(record['orders'])
            })
        
        return pd.DataFrame(records)


# ============================================================================
# 使用示例
# ============================================================================

def example_risk_hedging():
    """风险对冲示例"""
    print("=== 实时风险对冲系统示例 ===\n")
    
    # 1. 创建投资组合
    print("1. 投资组合")
    portfolio = {
        '600519.SH': Position('600519.SH', 1000, 1800, 1800000, beta=0.8),
        '000001.SZ': Position('000001.SZ', 5000, 15, 75000, beta=1.2),
        '601318.SH': Position('601318.SH', 2000, 50, 100000, beta=0.9)
    }
    
    for symbol, pos in portfolio.items():
        print(f"  {symbol}: {pos.quantity}股 @ {pos.price}元, Beta={pos.beta}")
    
    # 2. 风险敞口监控
    print("\n2. 风险敞口监控")
    monitor = RiskExposureMonitor(portfolio)
    metrics = monitor.calculate_exposure()
    print(f"  总价值: {metrics.total_value:.2f}")
    print(f"  Beta敞口: {metrics.beta_exposure:.2f}")
    print(f"  Delta: {metrics.delta:.2f}")
    
    # 设置风险限制
    monitor.set_risk_limit('max_delta', 0.2)
    violations = monitor.check_risk_limits(metrics)
    if violations:
        print(f"  ⚠️ 风险违规: {violations}")
    
    # 3. Delta中性对冲
    print("\n3. Delta中性对冲")
    hedger = DeltaNeutralHedger(target_delta=0.0)
    orders = hedger.generate_hedge_orders(metrics, hedge_instrument='510300.SH', hedge_price=4.0)
    
    if orders:
        print(f"  对冲订单:")
        for order in orders:
            print(f"    {order['direction']} {order['quantity']}股 {order['symbol']}")
    
    # 4. 综合对冲管理
    print("\n4. 综合对冲管理")
    manager = HedgingManager(portfolio)
    manager.monitor.set_risk_limit('max_delta', 0.2)
    
    hedge_orders = manager.run_hedging_cycle()
    print(f"  生成对冲订单: {len(hedge_orders)}个")
    
    print("\n风险对冲系统演示完成!")


if __name__ == "__main__":
    example_risk_hedging()
