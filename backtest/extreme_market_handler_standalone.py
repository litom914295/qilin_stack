"""
极端行情综合处理系统（独立版）
整合流动性监控、极端市场保护和动态仓位管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态"""
    BULL = "牛市"           # 持续上涨
    BEAR = "熊市"           # 持续下跌
    VOLATILE = "震荡市"     # 高波动震荡
    RANGING = "盘整"        # 窄幅震荡
    CRASH = "崩盘"          # 市场崩溃
    RECOVERY = "恢复期"     # 崩盘后恢复


class ProtectionLevel(Enum):
    """保护级别"""
    NONE = "无保护"         # 正常交易
    LOW = "低级保护"        # 轻度限制
    MEDIUM = "中级保护"     # 中度限制
    HIGH = "高级保护"       # 严格限制
    CRITICAL = "紧急保护"   # 紧急措施


class LiquidityLevel(Enum):
    """流动性级别"""
    HIGH = "高流动性"
    MEDIUM = "中流动性"
    LOW = "低流动性"
    ILLIQUID = "无流动性"


class MarketCondition(Enum):
    """市场状态"""
    NORMAL = "正常"
    CAUTION = "警戒"
    ALERT = "警报"
    CRISIS = "危机"


@dataclass
class RiskAssessment:
    """综合风险评估"""
    timestamp: datetime
    
    # 市场状态
    market_regime: MarketRegime
    market_condition: MarketCondition
    
    # 风险等级
    liquidity_risk: float      # 流动性风险（0-100）
    extreme_risk: float        # 极端事件风险（0-100）
    systemic_risk: float       # 系统性风险（0-100）
    overall_risk: float        # 综合风险（0-100）
    
    # 保护措施
    protection_level: ProtectionLevel
    position_adjustment: float  # 建议仓位调整（-1到1，负数减仓）
    max_position_allowed: float # 最大允许仓位
    
    # 具体建议
    actions: List[str]
    warnings: List[str]
    
    # 禁止列表
    blacklist_symbols: List[str]  # 禁止交易的股票
    restricted_symbols: List[str]  # 限制交易的股票


class ExtremeMarketHandler:
    """极端行情处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化极端行情处理器
        
        Args:
            config: 配置参数
        """
        config = config or {}
        
        # 风险阈值
        self.risk_thresholds = {
            'low': 30,      # 低风险
            'medium': 50,   # 中风险
            'high': 70,     # 高风险
            'critical': 85  # 危机
        }
        
        # 配置参数
        self.max_intraday_drop = config.get('max_intraday_drop', 0.07)
        self.max_intraday_rise = config.get('max_intraday_rise', 0.15)
        self.crash_threshold = config.get('crash_threshold', 0.05)
        self.min_avg_volume = config.get('min_avg_volume', 1_000_000)
        self.max_spread_ratio = config.get('max_spread_ratio', 0.002)
        
        # 历史记录
        self.risk_history: List[RiskAssessment] = []
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        
        # 黑名单和限制名单
        self.blacklist: set = set()
        self.restricted: set = set()
        
        # 市场状态缓存
        self.current_regime = MarketRegime.RANGING
        self.regime_change_time = datetime.now()
    
    def assess_market_risk(self, 
                          market_data: pd.DataFrame,
                          portfolio: Dict[str, float]) -> RiskAssessment:
        """
        综合评估市场风险
        
        Args:
            market_data: 市场数据
            portfolio: 当前持仓
            
        Returns:
            RiskAssessment: 综合风险评估
        """
        timestamp = datetime.now()
        warnings = []
        actions = []
        
        # 1. 识别市场状态
        market_regime = self._identify_market_regime(market_data)
        
        # 2. 评估市场条件
        market_condition = self._evaluate_market_condition(market_data)
        
        # 3. 计算各维度风险
        
        # 流动性风险
        liquidity_risk = self._calculate_liquidity_risk(market_data, portfolio)
        if liquidity_risk > self.risk_thresholds['high']:
            warnings.append(f"流动性风险偏高: {liquidity_risk:.1f}")
            actions.append("减少低流动性股票仓位")
        
        # 极端事件风险
        extreme_risk = self._calculate_extreme_risk(market_data)
        if extreme_risk > self.risk_thresholds['high']:
            warnings.append(f"极端事件风险高: {extreme_risk:.1f}")
            actions.append("启动极端行情保护措施")
        
        # 系统性风险
        systemic_risk = self._calculate_systemic_risk(market_data, market_regime)
        if systemic_risk > self.risk_thresholds['high']:
            warnings.append(f"系统性风险高: {systemic_risk:.1f}")
            actions.append("降低整体仓位")
        
        # 4. 计算综合风险
        overall_risk = self._calculate_overall_risk(
            liquidity_risk, extreme_risk, systemic_risk
        )
        
        # 5. 确定保护等级
        protection_level = self._determine_protection_level(overall_risk)
        
        # 6. 计算仓位调整建议
        position_adjustment = self._calculate_position_adjustment(
            overall_risk, market_regime, protection_level
        )
        
        max_position_allowed = self._calculate_max_position(
            overall_risk, market_regime
        )
        
        # 7. 更新黑名单和限制名单
        self._update_restricted_lists(market_data, liquidity_risk, extreme_risk)
        
        # 8. 生成具体操作建议
        if protection_level == ProtectionLevel.CRITICAL:
            actions.insert(0, "⚠️ 立即执行紧急风控措施")
            actions.append("清仓所有高风险持仓")
            actions.append("暂停所有新开仓操作")
        elif protection_level == ProtectionLevel.HIGH:
            actions.append("将仓位降至50%以下")
            actions.append("只保留核心持仓")
        elif protection_level == ProtectionLevel.MEDIUM:
            actions.append("适度减仓，保持谨慎")
            actions.append("避免追高，严格止损")
        
        assessment = RiskAssessment(
            timestamp=timestamp,
            market_regime=market_regime,
            market_condition=market_condition,
            liquidity_risk=liquidity_risk,
            extreme_risk=extreme_risk,
            systemic_risk=systemic_risk,
            overall_risk=overall_risk,
            protection_level=protection_level,
            position_adjustment=position_adjustment,
            max_position_allowed=max_position_allowed,
            actions=actions,
            warnings=warnings,
            blacklist_symbols=list(self.blacklist),
            restricted_symbols=list(self.restricted)
        )
        
        # 保存历史记录
        self.risk_history.append(assessment)
        if len(self.risk_history) > 1000:  # 只保留最近1000条
            self.risk_history.pop(0)
        
        return assessment
    
    def _identify_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """识别市场状态"""
        if market_data.empty:
            return MarketRegime.RANGING
        
        # 计算市场指标
        returns = market_data['close'].pct_change()
        volatility = returns.std()
        trend = returns.mean()
        
        # 计算涨跌股比例
        up_ratio = (returns > 0).mean()
        
        # 判断市场状态
        if trend > 0.02 and up_ratio > 0.7:
            return MarketRegime.BULL
        elif trend < -0.02 and up_ratio < 0.3:
            return MarketRegime.BEAR
        elif volatility > 0.03:
            return MarketRegime.VOLATILE
        elif returns.min() < -self.crash_threshold:
            return MarketRegime.CRASH
        elif self.current_regime == MarketRegime.CRASH and trend > 0:
            return MarketRegime.RECOVERY
        else:
            return MarketRegime.RANGING
    
    def _evaluate_market_condition(self, market_data: pd.DataFrame) -> MarketCondition:
        """评估市场条件"""
        if market_data.empty:
            return MarketCondition.NORMAL
        
        # 计算市场指标
        returns = market_data['close'].pct_change()
        max_drop = returns.min()
        volatility = returns.std()
        
        # 判断市场条件
        if max_drop < -self.crash_threshold or volatility > 0.05:
            return MarketCondition.CRISIS
        elif max_drop < -0.03 or volatility > 0.03:
            return MarketCondition.ALERT
        elif max_drop < -0.02 or volatility > 0.02:
            return MarketCondition.CAUTION
        else:
            return MarketCondition.NORMAL
    
    def _calculate_liquidity_risk(self, market_data: pd.DataFrame, 
                                 portfolio: Dict[str, float]) -> float:
        """计算流动性风险"""
        if market_data.empty:
            return 50.0
        
        risks = []
        
        for symbol in portfolio:
            if symbol in market_data.index:
                row = market_data.loc[symbol]
                
                # 成交量风险
                volume = row.get('volume', 0)
                if volume < self.min_avg_volume:
                    volume_risk = min(100, (1 - volume / self.min_avg_volume) * 100)
                else:
                    volume_risk = 0
                
                # 价差风险
                spread = row.get('spread', 0)
                if spread > self.max_spread_ratio:
                    spread_risk = min(100, (spread / self.max_spread_ratio - 1) * 50)
                else:
                    spread_risk = 0
                
                # 换手率风险
                turnover = row.get('turnover_rate', 0)
                if turnover < 0.01:
                    turnover_risk = min(100, (1 - turnover / 0.01) * 100)
                else:
                    turnover_risk = 0
                
                # 综合流动性风险
                liquidity_risk = (volume_risk + spread_risk + turnover_risk) / 3
                risks.append(liquidity_risk * portfolio[symbol])
        
        return min(100, sum(risks) / max(sum(portfolio.values()), 0.01))
    
    def _calculate_extreme_risk(self, market_data: pd.DataFrame) -> float:
        """计算极端事件风险"""
        if market_data.empty:
            return 30.0
        
        # 计算各项极端风险指标
        returns = market_data['close'].pct_change()
        
        # 暴跌风险
        crash_risk = 0
        if returns.min() < -self.crash_threshold:
            crash_risk = min(100, abs(returns.min()) / self.crash_threshold * 50)
        
        # 暴涨风险（可能的泡沫）
        bubble_risk = 0
        if returns.max() > self.max_intraday_rise:
            bubble_risk = min(100, returns.max() / self.max_intraday_rise * 30)
        
        # 波动率风险
        volatility = returns.std()
        vol_risk = min(100, volatility / 0.05 * 100)
        
        # 尾部风险
        tail_risk = self._calculate_tail_risk(returns)
        
        # 综合极端风险
        extreme_risk = (crash_risk * 0.4 + bubble_risk * 0.2 + 
                       vol_risk * 0.2 + tail_risk * 0.2)
        
        return min(100, extreme_risk)
    
    def _calculate_tail_risk(self, returns: pd.Series) -> float:
        """计算尾部风险（VaR和CVaR）"""
        if len(returns) < 20:
            return 50.0
        
        # 计算95% VaR
        var_95 = returns.quantile(0.05)
        
        # 计算CVaR (Expected Shortfall)
        cvar = returns[returns <= var_95].mean()
        
        # 转换为风险分数
        tail_risk = min(100, abs(cvar) / 0.05 * 100)
        
        return tail_risk
    
    def _calculate_systemic_risk(self, market_data: pd.DataFrame, 
                                market_regime: MarketRegime) -> float:
        """计算系统性风险"""
        base_risk = 30.0
        
        # 根据市场状态调整基础风险
        regime_risk_map = {
            MarketRegime.CRASH: 90,
            MarketRegime.BEAR: 70,
            MarketRegime.VOLATILE: 60,
            MarketRegime.RECOVERY: 50,
            MarketRegime.RANGING: 30,
            MarketRegime.BULL: 40  # 牛市也有泡沫风险
        }
        
        regime_risk = regime_risk_map.get(market_regime, 30)
        
        # 计算相关性风险
        if not market_data.empty and 'close' in market_data.columns:
            returns = market_data['close'].pct_change()
            # 简化：使用标准差作为系统性风险的代理
            corr_risk = min(100, returns.std() * 1000)
        else:
            corr_risk = 50
        
        # 综合系统性风险
        systemic_risk = regime_risk * 0.6 + corr_risk * 0.4
        
        return min(100, systemic_risk)
    
    def _calculate_overall_risk(self, liquidity_risk: float, 
                               extreme_risk: float, 
                               systemic_risk: float) -> float:
        """计算综合风险"""
        # 加权平均，极端风险权重最高
        weights = {
            'liquidity': 0.25,
            'extreme': 0.45,
            'systemic': 0.30
        }
        
        overall = (liquidity_risk * weights['liquidity'] + 
                  extreme_risk * weights['extreme'] + 
                  systemic_risk * weights['systemic'])
        
        # 如果有任一风险超过临界值，提升整体风险
        if max(liquidity_risk, extreme_risk, systemic_risk) > self.risk_thresholds['critical']:
            overall = max(overall, self.risk_thresholds['high'])
        
        return min(100, overall)
    
    def _determine_protection_level(self, overall_risk: float) -> ProtectionLevel:
        """确定保护级别"""
        if overall_risk >= self.risk_thresholds['critical']:
            return ProtectionLevel.CRITICAL
        elif overall_risk >= self.risk_thresholds['high']:
            return ProtectionLevel.HIGH
        elif overall_risk >= self.risk_thresholds['medium']:
            return ProtectionLevel.MEDIUM
        elif overall_risk >= self.risk_thresholds['low']:
            return ProtectionLevel.LOW
        else:
            return ProtectionLevel.NONE
    
    def _calculate_position_adjustment(self, overall_risk: float,
                                      market_regime: MarketRegime,
                                      protection_level: ProtectionLevel) -> float:
        """计算仓位调整建议"""
        # 基于风险的调整
        risk_adjustment = 0
        if overall_risk > self.risk_thresholds['critical']:
            risk_adjustment = -0.8  # 减仓80%
        elif overall_risk > self.risk_thresholds['high']:
            risk_adjustment = -0.5  # 减仓50%
        elif overall_risk > self.risk_thresholds['medium']:
            risk_adjustment = -0.3  # 减仓30%
        elif overall_risk > self.risk_thresholds['low']:
            risk_adjustment = -0.1  # 减仓10%
        
        # 基于市场状态的调整
        regime_adjustment = {
            MarketRegime.CRASH: -0.5,
            MarketRegime.BEAR: -0.3,
            MarketRegime.VOLATILE: -0.2,
            MarketRegime.RECOVERY: 0,
            MarketRegime.RANGING: 0,
            MarketRegime.BULL: 0.1
        }.get(market_regime, 0)
        
        # 综合调整（取更保守的值）
        adjustment = min(risk_adjustment, regime_adjustment)
        
        return max(-1, min(1, adjustment))
    
    def _calculate_max_position(self, overall_risk: float,
                               market_regime: MarketRegime) -> float:
        """计算最大允许仓位"""
        # 基础最大仓位
        base_max = 1.0
        
        # 根据风险调整
        if overall_risk > self.risk_thresholds['critical']:
            base_max = 0.1  # 最多10%仓位
        elif overall_risk > self.risk_thresholds['high']:
            base_max = 0.3  # 最多30%仓位
        elif overall_risk > self.risk_thresholds['medium']:
            base_max = 0.5  # 最多50%仓位
        elif overall_risk > self.risk_thresholds['low']:
            base_max = 0.7  # 最多70%仓位
        
        # 根据市场状态进一步调整
        regime_factor = {
            MarketRegime.CRASH: 0.2,
            MarketRegime.BEAR: 0.5,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.RECOVERY: 0.8,
            MarketRegime.RANGING: 0.9,
            MarketRegime.BULL: 1.0
        }.get(market_regime, 0.5)
        
        return base_max * regime_factor
    
    def _update_restricted_lists(self, market_data: pd.DataFrame,
                                liquidity_risk: float,
                                extreme_risk: float):
        """更新黑名单和限制名单"""
        # 清理旧的名单
        self.blacklist.clear()
        self.restricted.clear()
        
        if market_data.empty:
            return
        
        for symbol in market_data.index:
            row = market_data.loc[symbol]
            
            # 检查是否应加入黑名单
            if self._should_blacklist(row, liquidity_risk, extreme_risk):
                self.blacklist.add(symbol)
            # 检查是否应加入限制名单
            elif self._should_restrict(row):
                self.restricted.add(symbol)
    
    def _should_blacklist(self, stock_data: pd.Series,
                         liquidity_risk: float,
                         extreme_risk: float) -> bool:
        """判断是否应加入黑名单"""
        # 流动性太差
        if stock_data.get('volume', 0) < self.min_avg_volume * 0.1:
            return True
        
        # 价格异常波动
        if abs(stock_data.get('return', 0)) > 0.2:  # 单日涨跌超20%
            return True
        
        # 综合风险过高
        if liquidity_risk > 90 or extreme_risk > 90:
            return True
        
        return False
    
    def _should_restrict(self, stock_data: pd.Series) -> bool:
        """判断是否应加入限制名单"""
        # 流动性较差
        if stock_data.get('volume', 0) < self.min_avg_volume * 0.5:
            return True
        
        # 波动较大
        if abs(stock_data.get('return', 0)) > 0.1:  # 单日涨跌超10%
            return True
        
        return False
    
    def generate_risk_report(self, assessment: RiskAssessment) -> str:
        """生成风险报告"""
        report = []
        report.append("=" * 60)
        report.append(f"📊 极端行情风险评估报告")
        report.append(f"时间: {assessment.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # 市场状态
        report.append(f"\n📈 市场状态")
        report.append(f"  • 市场制度: {assessment.market_regime.value}")
        report.append(f"  • 市场条件: {assessment.market_condition.value}")
        
        # 风险评分
        report.append(f"\n⚠️ 风险评分")
        report.append(f"  • 流动性风险: {assessment.liquidity_risk:.1f}/100")
        report.append(f"  • 极端事件风险: {assessment.extreme_risk:.1f}/100")
        report.append(f"  • 系统性风险: {assessment.systemic_risk:.1f}/100")
        report.append(f"  • 综合风险: {assessment.overall_risk:.1f}/100")
        
        # 保护措施
        report.append(f"\n🛡️ 保护措施")
        report.append(f"  • 保护级别: {assessment.protection_level.value}")
        report.append(f"  • 仓位调整: {assessment.position_adjustment:+.1%}")
        report.append(f"  • 最大仓位: {assessment.max_position_allowed:.1%}")
        
        # 警告信息
        if assessment.warnings:
            report.append(f"\n⚠️ 警告信息")
            for warning in assessment.warnings:
                report.append(f"  • {warning}")
        
        # 操作建议
        if assessment.actions:
            report.append(f"\n💡 操作建议")
            for action in assessment.actions:
                report.append(f"  • {action}")
        
        # 限制名单
        if assessment.blacklist_symbols:
            report.append(f"\n🚫 黑名单股票: {', '.join(assessment.blacklist_symbols[:5])}")
        if assessment.restricted_symbols:
            report.append(f"\n⚠️ 限制股票: {', '.join(assessment.restricted_symbols[:5])}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# 测试代码
if __name__ == "__main__":
    # 创建处理器
    handler = ExtremeMarketHandler()
    
    # 生成测试数据
    np.random.seed(42)
    
    # 测试不同的市场情况
    test_scenarios = [
        {
            "name": "正常市场",
            "data": pd.DataFrame({
                'close': 100 + np.random.normal(0, 2, 100),
                'volume': np.random.uniform(1e6, 5e6, 100),
                'spread': np.random.uniform(0.001, 0.002, 100),
                'turnover_rate': np.random.uniform(0.02, 0.05, 100),
                'return': np.random.normal(0.001, 0.01, 100)
            }, index=[f"STOCK_{i:03d}" for i in range(100)])
        },
        {
            "name": "极端下跌",
            "data": pd.DataFrame({
                'close': 100 * np.exp(np.cumsum(np.random.normal(-0.02, 0.05, 100))),
                'volume': np.random.uniform(5e5, 2e6, 100),
                'spread': np.random.uniform(0.003, 0.01, 100),
                'turnover_rate': np.random.uniform(0.001, 0.01, 100),
                'return': np.random.normal(-0.03, 0.05, 100)
            }, index=[f"STOCK_{i:03d}" for i in range(100)])
        },
        {
            "name": "流动性危机",
            "data": pd.DataFrame({
                'close': 100 + np.random.normal(0, 5, 100),
                'volume': np.random.uniform(1e4, 1e5, 100),  # 极低成交量
                'spread': np.random.uniform(0.01, 0.05, 100),  # 高价差
                'turnover_rate': np.random.uniform(0.0001, 0.001, 100),  # 极低换手
                'return': np.random.normal(0, 0.03, 100)
            }, index=[f"STOCK_{i:03d}" for i in range(100)])
        }
    ]
    
    # 模拟持仓
    portfolio = {
        f"STOCK_{i:03d}": 0.01 for i in range(10)  # 持有10只股票，每只1%
    }
    
    print("🚀 开始测试极端行情处理系统...\n")
    
    for scenario in test_scenarios:
        print(f"\n📋 测试场景: {scenario['name']}")
        print("-" * 40)
        
        # 评估风险
        assessment = handler.assess_market_risk(scenario['data'], portfolio)
        
        # 生成报告
        report = handler.generate_risk_report(assessment)
        print(report)
        
        print("\n")
    
    print("✅ 测试完成！")