"""
极端行情综合处理系统
整合流动性监控、极端市场保护和动态仓位管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

# 导入风控组件
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qilin_stack.agents.risk.liquidity_monitor import LiquidityMonitor, LiquidityLevel
from qilin_stack.agents.risk.extreme_market_guard import ExtremeMarketGuard, MarketCondition
from qilin_stack.agents.risk.position_manager import PositionManager as RiskPositionManager

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
        
        # 初始化子组件
        self.liquidity_monitor = LiquidityMonitor(
            min_avg_volume=config.get('min_avg_volume', 1_000_000),
            max_spread_ratio=config.get('max_spread_ratio', 0.002),
            min_turnover_rate=config.get('min_turnover_rate', 0.01)
        )
        
        self.market_guard = ExtremeMarketGuard(
            max_intraday_drop=config.get('max_intraday_drop', 0.07),
            max_intraday_rise=config.get('max_intraday_rise', 0.15),
            crash_threshold=config.get('crash_threshold', 0.05)
        )
        
        self.position_manager = RiskPositionManager(
            max_position_size=config.get('max_position_size', 0.2),
            max_portfolio_risk=config.get('max_portfolio_risk', 0.15),
            max_correlation=config.get('max_correlation', 0.7)
        )
        
        # 风险阈值
        self.risk_thresholds = {
            'low': 30,      # 低风险
            'medium': 50,   # 中风险
            'high': 70,     # 高风险
            'critical': 85  # 危机
        }
        
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
        
        # 2. 评估市场健康度
        market_health = self.market_guard.evaluate_market_health(market_data)
        market_condition = market_health.market_condition
        
        # 3. 计算各维度风险
        
        # 流动性风险
        liquidity_risk = self._calculate_liquidity_risk(market_data, portfolio)
        if liquidity_risk > self.risk_thresholds['high']:
            warnings.append(f"流动性风险偏高: {liquidity_risk:.1f}")
            actions.append("减少低流动性股票仓位")
        
        # 极端事件风险
        extreme_risk = self._calculate_extreme_risk(market_health)
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
        
        # 记录历史
        self.risk_history.append(assessment)
        
        # 记录市场状态变化
        if market_regime != self.current_regime:
            self.regime_history.append((timestamp, market_regime))
            self.current_regime = market_regime
            self.regime_change_time = timestamp
            logger.info(f"市场状态切换: {self.current_regime.value} -> {market_regime.value}")
        
        return assessment
    
    def _identify_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """识别市场状态"""
        if 'index_return' not in market_data.columns:
            return MarketRegime.RANGING
        
        # 计算市场指数收益率统计
        returns = market_data['index_return'].values
        
        # 短期和长期收益
        short_return = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        long_return = np.mean(returns[-20:]) if len(returns) >= 20 else 0
        
        # 波动率
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        # 判断市场状态
        if long_return < -0.20:
            return MarketRegime.CRASH
        elif long_return < -0.10:
            return MarketRegime.BEAR
        elif long_return > 0.15:
            return MarketRegime.BULL
        elif volatility > 0.03:
            return MarketRegime.VOLATILE
        elif abs(long_return) < 0.05:
            return MarketRegime.RANGING
        else:
            return MarketRegime.RECOVERY
    
    def _calculate_liquidity_risk(self, market_data: pd.DataFrame, 
                                 portfolio: Dict[str, float]) -> float:
        """计算流动性风险（0-100）"""
        if not portfolio:
            return 0
        
        total_risk = 0
        total_weight = sum(portfolio.values())
        
        for symbol, weight in portfolio.items():
            # 获取该股票的流动性数据
            symbol_data = market_data[market_data['symbol'] == symbol] if 'symbol' in market_data else pd.DataFrame()
            
            if len(symbol_data) > 0:
                # 简化的流动性风险计算
                volume = symbol_data['volume'].values[-1] if 'volume' in symbol_data else 0
                turnover = symbol_data['turnover_rate'].values[-1] if 'turnover_rate' in symbol_data else 0
                
                # 低成交量和低换手率增加风险
                if volume < 1_000_000:
                    stock_risk = 80
                elif volume < 5_000_000:
                    stock_risk = 50
                else:
                    stock_risk = 20
                
                if turnover < 0.01:
                    stock_risk += 20
                
                total_risk += stock_risk * (weight / total_weight)
            else:
                # 无数据时假设高风险
                total_risk += 70 * (weight / total_weight)
        
        return min(100, total_risk)
    
    def _calculate_extreme_risk(self, market_health) -> float:
        """计算极端事件风险（0-100）"""
        # 基于市场健康度指标计算
        risk = 0
        
        # 恐慌指数贡献
        risk += market_health.panic_index * 0.5
        
        # 跌停股数量贡献
        if market_health.stocks_limit_down > 100:
            risk += 30
        elif market_health.stocks_limit_down > 50:
            risk += 20
        elif market_health.stocks_limit_down > 20:
            risk += 10
        
        # 市场宽度贡献
        if market_health.advance_decline_ratio < 0.3:
            risk += 20
        elif market_health.advance_decline_ratio < 0.5:
            risk += 10
        
        # 根据市场状况调整
        condition_multiplier = {
            MarketCondition.CRASH: 1.5,
            MarketCondition.PANIC: 1.3,
            MarketCondition.VOLATILE: 1.1,
            MarketCondition.NORMAL: 1.0,
            MarketCondition.EUPHORIA: 0.9
        }
        
        risk *= condition_multiplier.get(market_health.market_condition, 1.0)
        
        return min(100, risk)
    
    def _calculate_systemic_risk(self, market_data: pd.DataFrame, 
                                market_regime: MarketRegime) -> float:
        """计算系统性风险（0-100）"""
        base_risk = {
            MarketRegime.CRASH: 90,
            MarketRegime.BEAR: 70,
            MarketRegime.VOLATILE: 60,
            MarketRegime.RANGING: 30,
            MarketRegime.RECOVERY: 40,
            MarketRegime.BULL: 20
        }
        
        risk = base_risk.get(market_regime, 50)
        
        # 根据市场下跌幅度调整
        if 'index_return' in market_data.columns:
            recent_return = market_data['index_return'].iloc[-5:].sum()
            if recent_return < -0.1:
                risk += 20
            elif recent_return < -0.05:
                risk += 10
        
        return min(100, risk)
    
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
        
        overall = (
            liquidity_risk * weights['liquidity'] +
            extreme_risk * weights['extreme'] +
            systemic_risk * weights['systemic']
        )
        
        return min(100, overall)
    
    def _determine_protection_level(self, overall_risk: float) -> ProtectionLevel:
        """确定保护等级"""
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
        """计算仓位调整建议（-1到1）"""
        # 基础调整
        base_adjustment = 0
        
        # 根据风险水平调整
        if overall_risk > 80:
            base_adjustment = -0.8  # 大幅减仓
        elif overall_risk > 60:
            base_adjustment = -0.5  # 中度减仓
        elif overall_risk > 40:
            base_adjustment = -0.2  # 轻微减仓
        elif overall_risk < 20:
            base_adjustment = 0.2   # 可以加仓
        
        # 根据市场状态微调
        regime_adjustment = {
            MarketRegime.CRASH: -0.3,
            MarketRegime.BEAR: -0.2,
            MarketRegime.VOLATILE: -0.1,
            MarketRegime.RANGING: 0,
            MarketRegime.RECOVERY: 0.1,
            MarketRegime.BULL: 0.2
        }
        
        final_adjustment = base_adjustment + regime_adjustment.get(market_regime, 0)
        
        # 限制在-1到1之间
        return max(-1, min(1, final_adjustment))
    
    def _calculate_max_position(self, overall_risk: float,
                               market_regime: MarketRegime) -> float:
        """计算最大允许仓位"""
        # 基础最大仓位
        if overall_risk > 80:
            base_max = 0.2  # 20%
        elif overall_risk > 60:
            base_max = 0.4  # 40%
        elif overall_risk > 40:
            base_max = 0.6  # 60%
        elif overall_risk > 20:
            base_max = 0.8  # 80%
        else:
            base_max = 1.0  # 100%
        
        # 根据市场状态调整
        regime_multiplier = {
            MarketRegime.CRASH: 0.3,
            MarketRegime.BEAR: 0.6,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.RANGING: 1.0,
            MarketRegime.RECOVERY: 0.8,
            MarketRegime.BULL: 1.0
        }
        
        return base_max * regime_multiplier.get(market_regime, 0.8)
    
    def _update_restricted_lists(self, market_data: pd.DataFrame,
                                liquidity_risk: float,
                                extreme_risk: float):
        """更新黑名单和限制名单"""
        # 清空旧名单
        self.blacklist.clear()
        self.restricted.clear()
        
        if 'symbol' not in market_data.columns:
            return
        
        symbols = market_data['symbol'].unique()
        
        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol]
            
            # 检查是否应该加入黑名单
            # 1. 连续跌停
            if self._is_continuous_limit_down(symbol_data):
                self.blacklist.add(symbol)
                continue
            
            # 2. 流动性枯竭
            if 'volume' in symbol_data.columns:
                recent_volume = symbol_data['volume'].iloc[-5:].mean()
                if recent_volume < 100_000:
                    self.blacklist.add(symbol)
                    continue
            
            # 检查是否应该限制交易
            # 1. 高波动
            if 'close' in symbol_data.columns:
                returns = symbol_data['close'].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0.05:
                    self.restricted.add(symbol)
            
            # 2. ST股票
            if 'ST' in symbol.upper():
                self.restricted.add(symbol)
    
    def _is_continuous_limit_down(self, symbol_data: pd.DataFrame) -> bool:
        """检查是否连续跌停"""
        if 'close' not in symbol_data.columns or 'low' not in symbol_data.columns:
            return False
        
        # 检查最近3天是否都是跌停
        recent_data = symbol_data.iloc[-3:]
        if len(recent_data) < 3:
            return False
        
        for i in range(len(recent_data)):
            close = recent_data.iloc[i]['close']
            low = recent_data.iloc[i]['low']
            # 简单判断：收盘价等于最低价且跌幅超过9.5%
            if i > 0:
                prev_close = recent_data.iloc[i-1]['close']
                if close != low or (close - prev_close) / prev_close > -0.095:
                    return False
        
        return True
    
    def generate_risk_report(self) -> str:
        """生成风险报告"""
        if not self.risk_history:
            return "暂无风险评估记录"
        
        latest = self.risk_history[-1]
        
        report = []
        report.append("="*60)
        report.append("极端行情风险评估报告")
        report.append("="*60)
        report.append(f"时间: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"市场状态: {latest.market_regime.value}")
        report.append(f"市场状况: {latest.market_condition.value}")
        report.append("")
        
        report.append("【风险指标】")
        report.append(f"  流动性风险: {latest.liquidity_risk:.1f}/100")
        report.append(f"  极端事件风险: {latest.extreme_risk:.1f}/100")
        report.append(f"  系统性风险: {latest.systemic_risk:.1f}/100")
        report.append(f"  综合风险: {latest.overall_risk:.1f}/100")
        report.append("")
        
        report.append("【保护措施】")
        report.append(f"  保护等级: {latest.protection_level.name}")
        report.append(f"  仓位调整建议: {latest.position_adjustment:+.1%}")
        report.append(f"  最大允许仓位: {latest.max_position_allowed:.1%}")
        report.append("")
        
        if latest.warnings:
            report.append("【风险警告】")
            for warning in latest.warnings:
                report.append(f"  ⚠️ {warning}")
            report.append("")
        
        if latest.actions:
            report.append("【建议操作】")
            for i, action in enumerate(latest.actions, 1):
                report.append(f"  {i}. {action}")
            report.append("")
        
        if latest.blacklist_symbols:
            report.append("【交易黑名单】")
            report.append(f"  禁止交易: {', '.join(latest.blacklist_symbols)}")
            report.append("")
        
        if latest.restricted_symbols:
            report.append("【限制交易】")
            report.append(f"  谨慎交易: {', '.join(latest.restricted_symbols)}")
        
        report.append("="*60)
        
        return "\n".join(report)


# 测试代码
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建处理器
    handler = ExtremeMarketHandler()
    
    # 模拟市场数据
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    # 模拟崩盘场景
    market_data = pd.DataFrame({
        'date': dates,
        'index_return': np.concatenate([
            np.random.normal(0.001, 0.01, 10),  # 正常
            np.random.normal(-0.03, 0.02, 10),  # 暴跌
            np.random.normal(-0.01, 0.015, 10)  # 恢复
        ]),
        'volume': np.random.uniform(1e6, 1e7, 30),
        'turnover_rate': np.random.uniform(0.01, 0.05, 30)
    })
    
    # 模拟持仓
    portfolio = {
        '000001': 0.3,
        '000002': 0.2,
        'ST0001': 0.1,
        '600000': 0.4
    }
    
    # 评估风险
    assessment = handler.assess_market_risk(market_data, portfolio)
    
    # 生成报告
    report = handler.generate_risk_report()
    print(report)
    
    # 测试不同市场环境
    print("\n测试不同市场环境:")
    scenarios = [
        ("正常市场", np.random.normal(0.001, 0.01, 20)),
        ("牛市", np.random.normal(0.02, 0.01, 20)),
        ("熊市", np.random.normal(-0.015, 0.01, 20)),
        ("崩盘", np.random.normal(-0.05, 0.03, 20))
    ]
    
    for name, returns in scenarios:
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20, freq='D'),
            'index_return': returns,
            'volume': np.random.uniform(1e6, 1e7, 20),
            'turnover_rate': np.random.uniform(0.01, 0.05, 20)
        })
        
        assessment = handler.assess_market_risk(test_data, portfolio)
        print(f"\n{name}:")
        print(f"  市场状态: {assessment.market_regime.value}")
        print(f"  综合风险: {assessment.overall_risk:.1f}")
        print(f"  保护等级: {assessment.protection_level.name}")
        print(f"  仓位调整: {assessment.position_adjustment:+.1%}")