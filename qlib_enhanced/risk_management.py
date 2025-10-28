"""
实时风险管理系统模块
实现VaR/CVaR计算、压力测试、风险监控和预警系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)


# ============================================================================
# 风险等级定义
# ============================================================================

class RiskLevel(Enum):
    """风险等级"""
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "极高风险"


@dataclass
class RiskAlert:
    """风险预警"""
    timestamp: datetime
    risk_type: str
    level: RiskLevel
    message: str
    affected_assets: List[str]
    recommended_action: str


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float


# ============================================================================
# VaR/CVaR计算器
# ============================================================================

class ValueAtRiskCalculator:
    """
    风险价值 (VaR) 和条件风险价值 (CVaR) 计算器
    """
    
    def __init__(self, returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]):
        """
        初始化VaR计算器
        
        Args:
            returns: 收益率序列
            confidence_levels: 置信水平列表
        """
        self.returns = returns
        self.confidence_levels = confidence_levels
        
        logger.info(f"VaR计算器初始化: {len(returns)}个数据点")
    
    def calculate_historical_var(self, confidence: float = 0.95) -> float:
        """
        历史模拟法计算VaR
        
        Args:
            confidence: 置信水平
        
        Returns:
            VaR值 (负数表示损失)
        """
        alpha = 1 - confidence
        var = np.percentile(self.returns, alpha * 100)
        
        logger.debug(f"历史VaR ({confidence:.0%}): {var:.4f}")
        return var
    
    def calculate_parametric_var(self, confidence: float = 0.95) -> float:
        """
        参数法计算VaR (假设正态分布)
        
        Args:
            confidence: 置信水平
        
        Returns:
            VaR值
        """
        mean = self.returns.mean()
        std = self.returns.std()
        
        z_score = stats.norm.ppf(1 - confidence)
        var = mean + z_score * std
        
        logger.debug(f"参数VaR ({confidence:.0%}): {var:.4f}")
        return var
    
    def calculate_cvar(self, confidence: float = 0.95, method: str = 'historical') -> float:
        """
        计算条件风险价值 (CVaR / Expected Shortfall)
        
        Args:
            confidence: 置信水平
            method: 计算方法 ('historical' or 'parametric')
        
        Returns:
            CVaR值
        """
        if method == 'historical':
            var = self.calculate_historical_var(confidence)
            # CVaR = 超过VaR的损失的平均值
            cvar = self.returns[self.returns <= var].mean()
        else:
            # 参数法
            mean = self.returns.mean()
            std = self.returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            
            # CVaR公式 (正态分布)
            cvar = mean - std * stats.norm.pdf(z_score) / (1 - confidence)
        
        logger.debug(f"CVaR ({confidence:.0%}, {method}): {cvar:.4f}")
        return cvar
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """计算所有VaR/CVaR指标"""
        metrics = {}
        
        for conf in self.confidence_levels:
            metrics[f'var_{int(conf*100)}'] = self.calculate_historical_var(conf)
            metrics[f'cvar_{int(conf*100)}'] = self.calculate_cvar(conf)
        
        return metrics


# ============================================================================
# 压力测试
# ============================================================================

class StressTest:
    """
    压力测试模块
    模拟极端市场情景对组合的影响
    """
    
    def __init__(self, portfolio_returns: pd.Series, market_returns: pd.Series):
        """
        初始化压力测试
        
        Args:
            portfolio_returns: 组合收益率
            market_returns: 市场收益率
        """
        self.portfolio_returns = portfolio_returns
        self.market_returns = market_returns
        
        # 计算Beta
        self.beta = self._calculate_beta()
        
        logger.info("压力测试模块初始化")
    
    def _calculate_beta(self) -> float:
        """计算组合Beta"""
        # 对齐数据
        aligned = pd.DataFrame({
            'portfolio': self.portfolio_returns,
            'market': self.market_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 1.0
        
        cov = aligned['portfolio'].cov(aligned['market'])
        var = aligned['market'].var()
        
        beta = cov / var if var > 0 else 1.0
        return beta
    
    def scenario_market_crash(self, crash_magnitude: float = -0.20) -> Dict[str, float]:
        """
        场景1: 市场暴跌
        
        Args:
            crash_magnitude: 暴跌幅度 (负数)
        
        Returns:
            压力测试结果
        """
        logger.info(f"压力测试: 市场暴跌 {crash_magnitude:.0%}")
        
        # 使用Beta估计组合损失
        portfolio_loss = self.beta * crash_magnitude
        
        # 估计VaR在压力下的表现
        current_var_95 = np.percentile(self.portfolio_returns, 5)
        stressed_var = current_var_95 * (1 + abs(crash_magnitude) / 0.20)
        
        return {
            'scenario': 'market_crash',
            'market_move': crash_magnitude,
            'estimated_portfolio_loss': portfolio_loss,
            'stressed_var_95': stressed_var,
            'loss_probability': 0.05  # 5%概率
        }
    
    def scenario_volatility_spike(self, vol_multiplier: float = 2.0) -> Dict[str, float]:
        """
        场景2: 波动率飙升
        
        Args:
            vol_multiplier: 波动率倍数
        
        Returns:
            压力测试结果
        """
        logger.info(f"压力测试: 波动率飙升 {vol_multiplier}x")
        
        current_vol = self.portfolio_returns.std()
        stressed_vol = current_vol * vol_multiplier
        
        # 估计VaR
        mean_return = self.portfolio_returns.mean()
        stressed_var_95 = mean_return - 1.65 * stressed_vol
        
        return {
            'scenario': 'volatility_spike',
            'current_volatility': current_vol,
            'stressed_volatility': stressed_vol,
            'stressed_var_95': stressed_var_95,
            'loss_probability': 0.10  # 10%概率
        }
    
    def scenario_liquidity_crisis(self, liquidity_impact: float = 0.05) -> Dict[str, float]:
        """
        场景3: 流动性危机
        
        Args:
            liquidity_impact: 流动性冲击 (额外滑点)
        
        Returns:
            压力测试结果
        """
        logger.info(f"压力测试: 流动性危机 (冲击{liquidity_impact:.1%})")
        
        # 估计清算成本
        portfolio_value = 1000000  # 假设组合价值
        liquidation_cost = portfolio_value * liquidity_impact
        
        # 估计损失
        market_impact = -0.05  # 假设市场下跌5%
        total_loss = self.beta * market_impact + liquidity_impact
        
        return {
            'scenario': 'liquidity_crisis',
            'estimated_liquidation_cost': liquidation_cost,
            'estimated_total_loss': total_loss,
            'loss_probability': 0.03  # 3%概率
        }
    
    def run_all_scenarios(self) -> List[Dict[str, float]]:
        """运行所有压力测试场景"""
        scenarios = [
            self.scenario_market_crash(-0.10),
            self.scenario_market_crash(-0.20),
            self.scenario_market_crash(-0.30),
            self.scenario_volatility_spike(2.0),
            self.scenario_volatility_spike(3.0),
            self.scenario_liquidity_crisis(0.05)
        ]
        
        return scenarios


# ============================================================================
# 风险监控系统
# ============================================================================

class RiskMonitor:
    """
    实时风险监控系统
    持续监控各项风险指标并发出预警
    """
    
    def __init__(self,
                 var_threshold_95: float = -0.05,
                 var_threshold_99: float = -0.08,
                 drawdown_threshold: float = 0.15,
                 volatility_threshold: float = 0.30):
        """
        初始化风险监控
        
        Args:
            var_threshold_95: VaR 95%阈值
            var_threshold_99: VaR 99%阈值
            drawdown_threshold: 最大回撤阈值
            volatility_threshold: 波动率阈值
        """
        self.var_threshold_95 = var_threshold_95
        self.var_threshold_99 = var_threshold_99
        self.drawdown_threshold = drawdown_threshold
        self.volatility_threshold = volatility_threshold
        
        self.alerts = []
        
        logger.info("风险监控系统初始化")
    
    def calculate_metrics(self, returns: pd.Series, prices: pd.Series) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            returns: 收益率序列
            prices: 价格序列
        
        Returns:
            风险指标
        """
        # VaR/CVaR
        var_calc = ValueAtRiskCalculator(returns)
        var_metrics = var_calc.calculate_all_metrics()
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        mean_return = returns.mean() * 252
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Beta (假设市场收益率与组合收益率相关性为1)
        beta = 1.0  # 简化处理
        
        return RiskMetrics(
            var_95=var_metrics['var_95'],
            var_99=var_metrics['var_99'],
            cvar_95=var_metrics['cvar_95'],
            cvar_99=var_metrics['cvar_99'],
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            sharpe_ratio=sharpe
        )
    
    def check_risk_levels(self, metrics: RiskMetrics, symbols: List[str]) -> List[RiskAlert]:
        """
        检查风险水平并生成预警
        
        Args:
            metrics: 风险指标
            symbols: 涉及的资产列表
        
        Returns:
            风险预警列表
        """
        alerts = []
        
        # 检查VaR
        if metrics.var_95 < self.var_threshold_95:
            level = RiskLevel.HIGH if metrics.var_95 < self.var_threshold_95 * 1.5 else RiskLevel.MEDIUM
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type='VaR超限',
                level=level,
                message=f"VaR 95% ({metrics.var_95:.2%}) 超过阈值 ({self.var_threshold_95:.2%})",
                affected_assets=symbols,
                recommended_action="考虑降低仓位或增加对冲"
            ))
        
        # 检查最大回撤
        if abs(metrics.max_drawdown) > self.drawdown_threshold:
            level = RiskLevel.CRITICAL if abs(metrics.max_drawdown) > self.drawdown_threshold * 1.5 else RiskLevel.HIGH
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type='回撤超限',
                level=level,
                message=f"最大回撤 ({metrics.max_drawdown:.2%}) 超过阈值 ({self.drawdown_threshold:.2%})",
                affected_assets=symbols,
                recommended_action="立即止损或重新平衡组合"
            ))
        
        # 检查波动率
        if metrics.volatility > self.volatility_threshold:
            level = RiskLevel.MEDIUM
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type='波动率过高',
                level=level,
                message=f"波动率 ({metrics.volatility:.2%}) 超过阈值 ({self.volatility_threshold:.2%})",
                affected_assets=symbols,
                recommended_action="考虑降低杠杆或增加防御性资产"
            ))
        
        # 检查夏普比率
        if metrics.sharpe_ratio < 0.5:
            level = RiskLevel.LOW
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                risk_type='收益风险比低',
                level=level,
                message=f"夏普比率 ({metrics.sharpe_ratio:.2f}) 低于预期",
                affected_assets=symbols,
                recommended_action="优化资产配置以提高风险调整后收益"
            ))
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_risk_summary(self, metrics: RiskMetrics) -> Dict[str, any]:
        """
        生成风险摘要
        
        Args:
            metrics: 风险指标
        
        Returns:
            风险摘要
        """
        # 计算综合风险评分 (0-100, 100最危险)
        var_score = min(100, abs(metrics.var_95) / abs(self.var_threshold_95) * 50)
        dd_score = min(100, abs(metrics.max_drawdown) / self.drawdown_threshold * 30)
        vol_score = min(100, metrics.volatility / self.volatility_threshold * 20)
        
        total_score = var_score + dd_score + vol_score
        
        # 风险等级
        if total_score < 30:
            overall_risk = RiskLevel.LOW
        elif total_score < 60:
            overall_risk = RiskLevel.MEDIUM
        elif total_score < 80:
            overall_risk = RiskLevel.HIGH
        else:
            overall_risk = RiskLevel.CRITICAL
        
        return {
            'overall_risk_level': overall_risk,
            'risk_score': total_score,
            'var_contribution': var_score,
            'drawdown_contribution': dd_score,
            'volatility_contribution': vol_score,
            'active_alerts': len([a for a in self.alerts if 
                                 (datetime.now() - a.timestamp).seconds < 3600])
        }


# ============================================================================
# 使用示例
# ============================================================================

def create_sample_data(days: int = 252) -> Tuple[pd.Series, pd.Series]:
    """创建模拟数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # 生成收益率 (带趋势和波动)
    returns = np.random.normal(0.0005, 0.02, days)
    
    # 添加几个极端事件
    returns[50] = -0.08  # 极端下跌
    returns[150] = -0.06
    returns[200] = 0.05  # 极端上涨
    
    returns_series = pd.Series(returns, index=dates)
    
    # 生成价格
    prices = 100 * (1 + returns_series).cumprod()
    
    return returns_series, prices


def main():
    """示例：风险管理系统"""
    print("=" * 80)
    print("实时风险管理系统 - 示例")
    print("=" * 80)
    
    # 1. 创建数据
    print("\n📊 生成模拟数据...")
    returns, prices = create_sample_data(days=252)
    print(f"数据点数: {len(returns)}")
    
    # 2. VaR/CVaR计算
    print("\n💰 计算VaR/CVaR...")
    var_calc = ValueAtRiskCalculator(returns)
    
    var_95 = var_calc.calculate_historical_var(0.95)
    var_99 = var_calc.calculate_historical_var(0.99)
    cvar_95 = var_calc.calculate_cvar(0.95)
    cvar_99 = var_calc.calculate_cvar(0.99)
    
    print(f"VaR 95%: {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"VaR 99%: {var_99:.4f} ({var_99*100:.2f}%)")
    print(f"CVaR 95%: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"CVaR 99%: {cvar_99:.4f} ({cvar_99*100:.2f}%)")
    
    # 3. 压力测试
    print("\n🔥 运行压力测试...")
    market_returns = returns * 0.8  # 模拟市场收益
    stress_test = StressTest(returns, market_returns)
    
    scenarios = stress_test.run_all_scenarios()
    print(f"压力测试场景数: {len(scenarios)}")
    
    for scenario in scenarios[:3]:  # 显示前3个
        print(f"\n场景: {scenario['scenario']}")
        for key, value in scenario.items():
            if key != 'scenario':
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
    
    # 4. 风险监控
    print("\n🔍 风险监控...")
    monitor = RiskMonitor()
    
    metrics = monitor.calculate_metrics(returns, prices)
    print(f"\n风险指标:")
    print(f"  VaR 95%: {metrics.var_95:.4f}")
    print(f"  CVaR 95%: {metrics.cvar_95:.4f}")
    print(f"  最大回撤: {metrics.max_drawdown:.4f}")
    print(f"  波动率: {metrics.volatility:.4f}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
    
    # 5. 风险预警
    print("\n⚠️  检查风险预警...")
    alerts = monitor.check_risk_levels(metrics, ['ASSET_1', 'ASSET_2'])
    
    if alerts:
        print(f"发现 {len(alerts)} 个风险预警:")
        for alert in alerts:
            print(f"\n  [{alert.level.value}] {alert.risk_type}")
            print(f"  消息: {alert.message}")
            print(f"  建议: {alert.recommended_action}")
    else:
        print("✅ 未发现风险预警")
    
    # 6. 风险摘要
    print("\n📋 风险摘要...")
    summary = monitor.get_risk_summary(metrics)
    print(f"综合风险等级: {summary['overall_risk_level'].value}")
    print(f"风险评分: {summary['risk_score']:.1f}/100")
    print(f"活跃预警数: {summary['active_alerts']}")
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
