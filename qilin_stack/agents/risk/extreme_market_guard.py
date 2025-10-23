"""
极端行情保护模块 (Extreme Market Guard)
识别并应对暴涨暴跌、闪崩、千股跌停等极端市况

核心功能：
1. 极端行情识别（暴涨暴跌、闪崩、连续涨跌停）
2. 市场情绪监控（千股跌停、恐慌指数）
3. 保护性措施触发（紧急止损、暂停交易）
4. 极端行情应对策略
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class MarketCondition(Enum):
    """市场状况"""
    NORMAL = "正常"                 # 正常市况
    VOLATILE = "波动加剧"           # 波动性增加
    PANIC = "恐慌性下跌"            # 恐慌抛售
    EUPHORIA = "狂热上涨"           # 非理性繁荣
    CRASH = "闪崩"                  # 快速暴跌
    CIRCUIT_BREAKER = "熔断"        # 触发熔断机制


class ProtectionLevel(Enum):
    """保护等级"""
    NONE = 0        # 无需保护
    LOW = 1         # 低级保护（提高警惕）
    MEDIUM = 2      # 中级保护（减少仓位）
    HIGH = 3        # 高级保护（暂停开仓）
    CRITICAL = 4    # 危机保护（紧急平仓）


@dataclass
class ExtremeEvent:
    """极端事件"""
    symbol: str
    timestamp: datetime
    event_type: str                 # 事件类型
    severity: float                 # 严重程度（0-10）
    description: str                # 事件描述
    
    # 触发指标
    price_change: float             # 价格变化幅度
    volume_surge: float             # 成交量激增倍数
    volatility_spike: float         # 波动率飙升倍数
    
    # 保护措施
    protection_level: ProtectionLevel
    recommended_action: str         # 建议操作
    auto_triggered: bool            # 是否自动触发


@dataclass
class MarketHealthMetrics:
    """市场健康度指标"""
    timestamp: datetime
    
    # 整体市场指标
    market_condition: MarketCondition
    panic_index: float              # 恐慌指数（0-100）
    
    # 涨跌统计
    stocks_limit_up: int            # 涨停股数量
    stocks_limit_down: int          # 跌停股数量
    stocks_rising: int              # 上涨股数量
    stocks_falling: int             # 下跌股数量
    
    # 市场宽度
    advance_decline_ratio: float    # 涨跌比
    
    # 流动性指标
    total_turnover: float           # 总成交额
    avg_turnover_ratio: float       # 平均换手率
    
    # 警告信息
    warnings: List[str]
    protection_level: ProtectionLevel


class ExtremeMarketGuard:
    """极端行情保护器"""
    
    def __init__(self,
                 max_intraday_drop: float = 0.07,           # 最大日内跌幅（7%触发警报）
                 max_intraday_rise: float = 0.15,           # 最大日内涨幅（15%触发警报）
                 crash_threshold: float = 0.05,             # 闪崩阈值（5分钟跌5%）
                 volatility_spike_threshold: float = 3.0,   # 波动率飙升倍数
                 panic_index_threshold: float = 70):        # 恐慌指数阈值
        """
        初始化极端行情保护器
        
        Args:
            max_intraday_drop: 最大可容忍日内跌幅
            max_intraday_rise: 最大可容忍日内涨幅
            crash_threshold: 闪崩识别阈值
            volatility_spike_threshold: 波动率异常倍数
            panic_index_threshold: 恐慌指数阈值
        """
        self.max_intraday_drop = max_intraday_drop
        self.max_intraday_rise = max_intraday_rise
        self.crash_threshold = crash_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        self.panic_index_threshold = panic_index_threshold
        
        # 极端事件历史
        self.extreme_events: Dict[str, List[ExtremeEvent]] = {}
        
        # 市场健康度历史
        self.market_health_history: List[MarketHealthMetrics] = []
    
    def detect_extreme_event(self,
                           symbol: str,
                           price_data: pd.DataFrame,
                           volume_data: pd.DataFrame,
                           timeframe: str = "1min") -> Optional[ExtremeEvent]:
        """
        检测个股极端事件
        
        Args:
            symbol: 股票代码
            price_data: 价格数据（包含open, high, low, close）
            volume_data: 成交量数据
            timeframe: 时间周期
            
        Returns:
            ExtremeEvent或None
        """
        if len(price_data) < 2:
            return None
        
        timestamp = datetime.now()
        
        # 1. 计算价格变化
        current_price = price_data['close'].iloc[-1]
        prev_price = price_data['close'].iloc[-2]
        open_price = price_data['open'].iloc[0]
        
        price_change = (current_price - prev_price) / prev_price
        intraday_change = (current_price - open_price) / open_price
        
        # 2. 计算成交量变化
        current_volume = volume_data['volume'].iloc[-1]
        avg_volume = volume_data['volume'].iloc[:-1].mean()
        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 3. 计算波动率
        returns = price_data['close'].pct_change().dropna()
        current_volatility = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
        historical_volatility = returns.std()
        volatility_spike = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # 4. 检测极端事件
        event = None
        
        # 闪崩检测（短时间内大幅下跌）
        if timeframe == "1min" and price_change < -self.crash_threshold:
            event = ExtremeEvent(
                symbol=symbol,
                timestamp=timestamp,
                event_type="闪崩",
                severity=min(abs(price_change) * 100, 10),
                description=f"1分钟内暴跌{price_change:.2%}，疑似闪崩",
                price_change=price_change,
                volume_surge=volume_surge,
                volatility_spike=volatility_spike,
                protection_level=ProtectionLevel.CRITICAL,
                recommended_action="立即平仓，停止交易",
                auto_triggered=True
            )
        
        # 日内暴跌
        elif intraday_change < -self.max_intraday_drop:
            severity = min(abs(intraday_change) / self.max_intraday_drop * 7, 10)
            event = ExtremeEvent(
                symbol=symbol,
                timestamp=timestamp,
                event_type="暴跌",
                severity=severity,
                description=f"日内暴跌{intraday_change:.2%}",
                price_change=intraday_change,
                volume_surge=volume_surge,
                volatility_spike=volatility_spike,
                protection_level=ProtectionLevel.HIGH if severity > 7 else ProtectionLevel.MEDIUM,
                recommended_action="考虑止损，暂停开仓" if severity > 7 else "提高警惕，减少仓位",
                auto_triggered=severity > 7
            )
        
        # 日内暴涨（可能是操纵或非理性繁荣）
        elif intraday_change > self.max_intraday_rise:
            severity = min(intraday_change / self.max_intraday_rise * 5, 10)
            event = ExtremeEvent(
                symbol=symbol,
                timestamp=timestamp,
                event_type="暴涨",
                severity=severity,
                description=f"日内暴涨{intraday_change:.2%}，警惕回调风险",
                price_change=intraday_change,
                volume_surge=volume_surge,
                volatility_spike=volatility_spike,
                protection_level=ProtectionLevel.MEDIUM if severity > 7 else ProtectionLevel.LOW,
                recommended_action="考虑止盈，避免追高" if severity > 7 else "保持警惕",
                auto_triggered=False
            )
        
        # 波动率异常飙升
        elif volatility_spike > self.volatility_spike_threshold:
            event = ExtremeEvent(
                symbol=symbol,
                timestamp=timestamp,
                event_type="波动率飙升",
                severity=min(volatility_spike / self.volatility_spike_threshold * 5, 10),
                description=f"波动率异常飙升{volatility_spike:.1f}倍",
                price_change=price_change,
                volume_surge=volume_surge,
                volatility_spike=volatility_spike,
                protection_level=ProtectionLevel.MEDIUM,
                recommended_action="降低仓位，收紧止损",
                auto_triggered=False
            )
        
        # 记录事件
        if event:
            if symbol not in self.extreme_events:
                self.extreme_events[symbol] = []
            self.extreme_events[symbol].append(event)
        
        return event
    
    def evaluate_market_health(self,
                              market_data: Dict[str, pd.DataFrame]) -> MarketHealthMetrics:
        """
        评估整体市场健康度
        
        Args:
            market_data: 市场数据字典 {symbol: price_df}
            
        Returns:
            MarketHealthMetrics
        """
        timestamp = datetime.now()
        warnings = []
        
        if not market_data:
            return MarketHealthMetrics(
                timestamp=timestamp,
                market_condition=MarketCondition.NORMAL,
                panic_index=0,
                stocks_limit_up=0,
                stocks_limit_down=0,
                stocks_rising=0,
                stocks_falling=0,
                advance_decline_ratio=1.0,
                total_turnover=0,
                avg_turnover_ratio=0,
                warnings=["无市场数据"],
                protection_level=ProtectionLevel.NONE
            )
        
        # 统计涨跌情况
        stocks_limit_up = 0
        stocks_limit_down = 0
        stocks_rising = 0
        stocks_falling = 0
        total_turnover = 0
        turnover_ratios = []
        
        for symbol, df in market_data.items():
            if len(df) < 2:
                continue
            
            change = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]
            
            # 涨跌停统计（A股涨跌停约±10%，ST股约±5%）
            if change >= 0.095:  # 接近涨停
                stocks_limit_up += 1
            elif change <= -0.095:  # 接近跌停
                stocks_limit_down += 1
            elif change > 0:
                stocks_rising += 1
            else:
                stocks_falling += 1
            
            # 成交额和换手率
            if 'turnover' in df.columns:
                total_turnover += df['turnover'].iloc[-1]
            if 'turnover_rate' in df.columns:
                turnover_ratios.append(df['turnover_rate'].iloc[-1])
        
        total_stocks = len(market_data)
        advance_decline_ratio = stocks_rising / stocks_falling if stocks_falling > 0 else 999
        avg_turnover_ratio = np.mean(turnover_ratios) if turnover_ratios else 0
        
        # 计算恐慌指数（0-100）
        panic_index = self._calculate_panic_index(
            stocks_limit_down=stocks_limit_down,
            stocks_falling=stocks_falling,
            total_stocks=total_stocks,
            advance_decline_ratio=advance_decline_ratio
        )
        
        # 判断市场状况
        market_condition = self._determine_market_condition(
            panic_index=panic_index,
            stocks_limit_up=stocks_limit_up,
            stocks_limit_down=stocks_limit_down,
            total_stocks=total_stocks
        )
        
        # 生成警告
        if stocks_limit_down > total_stocks * 0.1:  # 超过10%的股票跌停
            warnings.append(f"千股跌停：{stocks_limit_down}只股票跌停（{stocks_limit_down/total_stocks:.1%}）")
        
        if panic_index > self.panic_index_threshold:
            warnings.append(f"恐慌指数过高：{panic_index:.1f}/100")
        
        if advance_decline_ratio < 0.3:
            warnings.append(f"市场严重分化：涨跌比{advance_decline_ratio:.2f}")
        
        # 确定保护等级
        protection_level = self._determine_protection_level(
            market_condition=market_condition,
            panic_index=panic_index,
            stocks_limit_down=stocks_limit_down,
            total_stocks=total_stocks
        )
        
        metrics = MarketHealthMetrics(
            timestamp=timestamp,
            market_condition=market_condition,
            panic_index=panic_index,
            stocks_limit_up=stocks_limit_up,
            stocks_limit_down=stocks_limit_down,
            stocks_rising=stocks_rising,
            stocks_falling=stocks_falling,
            advance_decline_ratio=advance_decline_ratio,
            total_turnover=total_turnover,
            avg_turnover_ratio=avg_turnover_ratio,
            warnings=warnings,
            protection_level=protection_level
        )
        
        self.market_health_history.append(metrics)
        
        return metrics
    
    def _calculate_panic_index(self,
                              stocks_limit_down: int,
                              stocks_falling: int,
                              total_stocks: int,
                              advance_decline_ratio: float) -> float:
        """
        计算恐慌指数（0-100）
        
        指标：
        1. 跌停股比例（40分）
        2. 下跌股比例（30分）
        3. 涨跌比（30分）
        """
        if total_stocks == 0:
            return 0
        
        score = 0.0
        
        # 1. 跌停股比例（跌停越多，恐慌越大）
        limit_down_ratio = stocks_limit_down / total_stocks
        score += min(limit_down_ratio * 400, 40)  # 10%跌停为满分
        
        # 2. 下跌股比例
        falling_ratio = stocks_falling / total_stocks
        score += min(falling_ratio * 50, 30)  # 60%下跌为满分
        
        # 3. 涨跌比（越低越恐慌）
        if advance_decline_ratio < 1.0:
            score += 30 * (1 - advance_decline_ratio)
        
        return round(min(score, 100), 2)
    
    def _determine_market_condition(self,
                                   panic_index: float,
                                   stocks_limit_up: int,
                                   stocks_limit_down: int,
                                   total_stocks: int) -> MarketCondition:
        """判断市场状况"""
        limit_down_ratio = stocks_limit_down / total_stocks if total_stocks > 0 else 0
        limit_up_ratio = stocks_limit_up / total_stocks if total_stocks > 0 else 0
        
        # 熔断级别（超过20%跌停）
        if limit_down_ratio > 0.2:
            return MarketCondition.CIRCUIT_BREAKER
        
        # 闪崩（超过15%跌停且恐慌指数极高）
        if limit_down_ratio > 0.15 and panic_index > 85:
            return MarketCondition.CRASH
        
        # 恐慌性下跌
        if panic_index > self.panic_index_threshold:
            return MarketCondition.PANIC
        
        # 狂热上涨（超过10%涨停）
        if limit_up_ratio > 0.1:
            return MarketCondition.EUPHORIA
        
        # 波动加剧
        if panic_index > 50 or limit_down_ratio > 0.05:
            return MarketCondition.VOLATILE
        
        return MarketCondition.NORMAL
    
    def _determine_protection_level(self,
                                   market_condition: MarketCondition,
                                   panic_index: float,
                                   stocks_limit_down: int,
                                   total_stocks: int) -> ProtectionLevel:
        """确定保护等级"""
        if market_condition == MarketCondition.CIRCUIT_BREAKER:
            return ProtectionLevel.CRITICAL
        
        if market_condition == MarketCondition.CRASH:
            return ProtectionLevel.CRITICAL
        
        if market_condition == MarketCondition.PANIC:
            return ProtectionLevel.HIGH
        
        if market_condition == MarketCondition.VOLATILE:
            return ProtectionLevel.MEDIUM
        
        if market_condition == MarketCondition.EUPHORIA:
            return ProtectionLevel.MEDIUM  # 狂热时也要警惕
        
        return ProtectionLevel.LOW if panic_index > 30 else ProtectionLevel.NONE
    
    def should_halt_trading(self, 
                           symbol: Optional[str] = None) -> Tuple[bool, str]:
        """
        判断是否应暂停交易
        
        Args:
            symbol: 股票代码（None表示检查整体市场）
            
        Returns:
            (是否暂停, 原因)
        """
        # 检查个股极端事件
        if symbol and symbol in self.extreme_events:
            recent_events = [e for e in self.extreme_events[symbol]
                           if (datetime.now() - e.timestamp).seconds < 600]  # 10分钟内
            
            for event in recent_events:
                if event.protection_level == ProtectionLevel.CRITICAL:
                    return True, f"检测到{event.event_type}，{event.recommended_action}"
        
        # 检查市场健康度
        if self.market_health_history:
            latest_health = self.market_health_history[-1]
            
            if latest_health.protection_level == ProtectionLevel.CRITICAL:
                return True, f"市场{latest_health.market_condition.value}，暂停所有交易"
            
            if latest_health.protection_level == ProtectionLevel.HIGH:
                return True, f"市场{latest_health.market_condition.value}，暂停开仓"
        
        return False, "正常交易"
    
    def get_protection_report(self) -> Dict:
        """生成保护报告"""
        if not self.market_health_history:
            return {"error": "无市场数据"}
        
        latest = self.market_health_history[-1]
        
        # 统计近期极端事件
        recent_events = []
        for symbol, events in self.extreme_events.items():
            for event in events:
                if (datetime.now() - event.timestamp).seconds < 3600:  # 1小时内
                    recent_events.append({
                        "symbol": symbol,
                        "type": event.event_type,
                        "severity": event.severity,
                        "description": event.description
                    })
        
        report = {
            "report_time": latest.timestamp,
            "market_status": {
                "市场状况": latest.market_condition.value,
                "恐慌指数": f"{latest.panic_index:.1f}/100",
                "保护等级": latest.protection_level.name
            },
            "market_breadth": {
                "涨停数": latest.stocks_limit_up,
                "跌停数": latest.stocks_limit_down,
                "上涨数": latest.stocks_rising,
                "下跌数": latest.stocks_falling,
                "涨跌比": f"{latest.advance_decline_ratio:.2f}"
            },
            "recent_extreme_events": recent_events[:10],  # 最近10条
            "warnings": latest.warnings,
            "trading_recommendation": self._get_trading_recommendation(latest)
        }
        
        return report
    
    def _get_trading_recommendation(self, metrics: MarketHealthMetrics) -> str:
        """生成交易建议"""
        if metrics.protection_level == ProtectionLevel.CRITICAL:
            return "🚨 危机保护：立即停止所有交易，优先保护本金"
        elif metrics.protection_level == ProtectionLevel.HIGH:
            return "⚠️ 高级保护：暂停开仓，考虑减仓，收紧止损"
        elif metrics.protection_level == ProtectionLevel.MEDIUM:
            return "⚡ 中级保护：降低仓位，提高警惕，严格止损"
        elif metrics.protection_level == ProtectionLevel.LOW:
            return "👀 低级保护：保持观察，适度谨慎"
        else:
            return "✅ 正常交易：可按策略执行"


# 使用示例
if __name__ == "__main__":
    guard = ExtremeMarketGuard()
    
    # 模拟个股价格数据
    print("=== 个股极端事件检测 ===")
    price_data = pd.DataFrame({
        'open': [10.0] * 100,
        'high': [10.2] * 50 + [9.8] * 50,
        'low': [9.8] * 50 + [9.0] * 50,
        'close': [10.0] * 50 + [9.3, 9.2, 9.0, 8.8, 8.5] + [8.5] * 45
    })
    
    volume_data = pd.DataFrame({
        'volume': [1_000_000] * 50 + [5_000_000] * 50
    })
    
    event = guard.detect_extreme_event(
        symbol="000001.SZ",
        price_data=price_data,
        volume_data=volume_data
    )
    
    if event:
        print(f"检测到极端事件: {event.event_type}")
        print(f"严重程度: {event.severity:.1f}/10")
        print(f"描述: {event.description}")
        print(f"保护等级: {event.protection_level.name}")
        print(f"建议操作: {event.recommended_action}")
    
    # 模拟市场数据
    print("\n=== 市场健康度评估 ===")
    market_data = {}
    for i in range(1000):
        symbol = f"{i:06d}.SZ"
        # 模拟千股跌停场景
        if i < 150:  # 15%跌停
            close_price = 9.0
        elif i < 700:  # 55%下跌
            close_price = 9.5
        else:  # 30%上涨
            close_price = 10.5
        
        market_data[symbol] = pd.DataFrame({
            'open': [10.0],
            'close': [close_price],
            'turnover': [10_000_000],
            'turnover_rate': [0.02]
        })
    
    health = guard.evaluate_market_health(market_data)
    
    print(f"市场状况: {health.market_condition.value}")
    print(f"恐慌指数: {health.panic_index:.1f}/100")
    print(f"保护等级: {health.protection_level.name}")
    print(f"涨停/跌停: {health.stocks_limit_up}/{health.stocks_limit_down}")
    print(f"涨跌比: {health.advance_decline_ratio:.2f}")
    
    if health.warnings:
        print("\n警告:")
        for warning in health.warnings:
            print(f"  - {warning}")
    
    # 检查是否暂停交易
    print("\n=== 交易暂停检查 ===")
    should_halt, reason = guard.should_halt_trading()
    print(f"暂停交易: {'是' if should_halt else '否'}")
    print(f"原因: {reason}")
    
    # 生成保护报告
    print("\n=== 保护报告 ===")
    report = guard.get_protection_report()
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list) and value:
            print(f"\n{key}:")
            for item in value[:3]:  # 只打印前3条
                print(f"  - {item}")
        elif not isinstance(value, list):
            print(f"{key}: {value}")
