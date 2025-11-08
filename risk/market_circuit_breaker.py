"""
市场环境熔断机制
在市场极端情况下停止交易或降低仓位
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


class MarketCondition(Enum):
    """市场环境状态"""
    NORMAL = "正常"
    CAUTION = "谨慎"
    WARNING = "警告"
    DANGER = "危险"
    HALT = "熔断"


class CircuitBreakerLevel(Enum):
    """熔断级别"""
    LEVEL_0 = 0  # 无熔断，正常交易
    LEVEL_1 = 1  # 1级熔断，降低30%仓位
    LEVEL_2 = 2  # 2级熔断，降低60%仓位
    LEVEL_3 = 3  # 3级熔断，停止新开仓，保留现有持仓
    LEVEL_4 = 4  # 4级熔断，全部平仓，停止交易


@dataclass
class MarketSignal:
    """市场信号"""
    timestamp: datetime
    condition: MarketCondition
    breaker_level: CircuitBreakerLevel
    position_adjust_ratio: float  # 仓位调整比例
    allow_new_positions: bool  # 是否允许新开仓
    force_close_all: bool  # 是否强制平仓
    reason: str  # 触发原因
    metrics: Dict  # 相关指标


class MarketCircuitBreaker:
    """
    市场环境熔断机制
    
    监控维度：
    1. 大盘指标：上证、深证、创业板涨跌幅
    2. 市场情绪：涨停数、跌停数、换手率
    3. 北向资金：流入流出
    4. 板块轮动：热点板块数量、持续性
    5. 自身表现：当日盈亏、连续亏损天数
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化熔断机制
        
        Parameters:
        -----------
        config: Dict
            配置参数
        """
        self.config = config or self._default_config()
        self.history = []  # 历史信号
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            # 大盘指标阈值
            'index_thresholds': {
                'danger_drop': -3.0,     # 大盘暴跌3%
                'warning_drop': -2.0,    # 大盘大跌2%
                'caution_drop': -1.0,    # 大盘下跌1%
            },
            
            # 市场情绪阈值
            'sentiment_thresholds': {
                'limit_down_ratio': 0.05,   # 跌停比例>5%
                'limit_up_ratio': 0.01,     # 涨停比例<1%
                'low_turnover': 0.5,        # 换手率<0.5%
            },
            
            # 北向资金阈值（亿元）
            'northbound_thresholds': {
                'large_outflow': -100,  # 大额流出>100亿
                'outflow': -50,         # 流出>50亿
            },
            
            # 自身表现阈值
            'self_performance_thresholds': {
                'daily_loss_ratio': -0.05,       # 当日亏损>5%
                'continuous_loss_days': 3,       # 连续亏损3天
                'max_drawdown': -0.15,           # 最大回撤>15%
            },
            
            # 熔断级别配置
            'breaker_levels': {
                CircuitBreakerLevel.LEVEL_1: {
                    'position_ratio': 0.7,
                    'allow_new': True,
                    'force_close': False
                },
                CircuitBreakerLevel.LEVEL_2: {
                    'position_ratio': 0.4,
                    'allow_new': False,
                    'force_close': False
                },
                CircuitBreakerLevel.LEVEL_3: {
                    'position_ratio': 1.0,
                    'allow_new': False,
                    'force_close': False
                },
                CircuitBreakerLevel.LEVEL_4: {
                    'position_ratio': 0.0,
                    'allow_new': False,
                    'force_close': True
                }
            }
        }
    
    def check_market_condition(self,
                               market_data: Dict) -> MarketSignal:
        """
        检查市场环境并生成信号
        
        Parameters:
        -----------
        market_data: Dict
            市场数据，包含：
            - index_changes: 大盘涨跌幅 {'sh': -1.5, 'sz': -2.0, 'cyb': -2.5}
            - limit_up_count: 涨停数
            - limit_down_count: 跌停数
            - total_stocks: 总股票数
            - avg_turnover: 平均换手率
            - northbound_flow: 北向资金流入（亿元）
            - daily_pnl_ratio: 当日盈亏比例
            - continuous_loss_days: 连续亏损天数
            - max_drawdown: 最大回撤
            
        Returns:
        --------
        MarketSignal: 市场信号
        """
        timestamp = datetime.now()
        
        # 评分系统（0-10分，分数越高风险越大）
        risk_score = 0
        reasons = []
        metrics = {}
        
        # 1. 大盘指标检查
        index_risk, index_reasons, index_metrics = self._check_index(market_data)
        risk_score += index_risk
        reasons.extend(index_reasons)
        metrics.update(index_metrics)
        
        # 2. 市场情绪检查
        sentiment_risk, sentiment_reasons, sentiment_metrics = self._check_sentiment(market_data)
        risk_score += sentiment_risk
        reasons.extend(sentiment_reasons)
        metrics.update(sentiment_metrics)
        
        # 3. 北向资金检查
        northbound_risk, northbound_reasons, northbound_metrics = self._check_northbound(market_data)
        risk_score += northbound_risk
        reasons.extend(northbound_reasons)
        metrics.update(northbound_metrics)
        
        # 4. 自身表现检查
        self_risk, self_reasons, self_metrics = self._check_self_performance(market_data)
        risk_score += self_risk
        reasons.extend(self_reasons)
        metrics.update(self_metrics)
        
        # 根据风险评分确定市场状态和熔断级别
        condition, breaker_level = self._determine_level(risk_score)
        
        # 获取熔断配置
        breaker_config = self.config['breaker_levels'].get(
            breaker_level,
            {'position_ratio': 1.0, 'allow_new': True, 'force_close': False}
        )
        
        # 生成信号
        signal = MarketSignal(
            timestamp=timestamp,
            condition=condition,
            breaker_level=breaker_level,
            position_adjust_ratio=breaker_config['position_ratio'],
            allow_new_positions=breaker_config['allow_new'],
            force_close_all=breaker_config['force_close'],
            reason="; ".join(reasons) if reasons else "市场正常",
            metrics=metrics
        )
        
        # 记录历史
        self.history.append(signal)
        
        # 打印信号
        self._print_signal(signal, risk_score)
        
        return signal
    
    def _check_index(self, market_data: Dict) -> Tuple[float, List[str], Dict]:
        """检查大盘指标"""
        risk_score = 0
        reasons = []
        metrics = {}
        
        index_changes = market_data.get('index_changes', {})
        
        # 上证指数
        sh_change = index_changes.get('sh', 0)
        sz_change = index_changes.get('sz', 0)
        cyb_change = index_changes.get('cyb', 0)
        
        # 计算平均跌幅
        avg_change = (sh_change + sz_change + cyb_change) / 3
        
        metrics['上证涨跌幅'] = sh_change
        metrics['深证涨跌幅'] = sz_change
        metrics['创业板涨跌幅'] = cyb_change
        metrics['平均涨跌幅'] = avg_change
        
        # 评分
        if avg_change <= self.config['index_thresholds']['danger_drop']:
            risk_score += 4
            reasons.append(f"大盘暴跌({avg_change:.2f}%)")
        elif avg_change <= self.config['index_thresholds']['warning_drop']:
            risk_score += 2
            reasons.append(f"大盘大跌({avg_change:.2f}%)")
        elif avg_change <= self.config['index_thresholds']['caution_drop']:
            risk_score += 1
            reasons.append(f"大盘下跌({avg_change:.2f}%)")
        
        return risk_score, reasons, metrics
    
    def _check_sentiment(self, market_data: Dict) -> Tuple[float, List[str], Dict]:
        """检查市场情绪"""
        risk_score = 0
        reasons = []
        metrics = {}
        
        limit_up_count = market_data.get('limit_up_count', 0)
        limit_down_count = market_data.get('limit_down_count', 0)
        total_stocks = market_data.get('total_stocks', 4800)
        avg_turnover = market_data.get('avg_turnover', 2.0)
        
        # 涨跌停比例
        limit_up_ratio = limit_up_count / total_stocks if total_stocks > 0 else 0
        limit_down_ratio = limit_down_count / total_stocks if total_stocks > 0 else 0
        
        metrics['涨停数'] = limit_up_count
        metrics['跌停数'] = limit_down_count
        metrics['涨停比例'] = limit_up_ratio * 100
        metrics['跌停比例'] = limit_down_ratio * 100
        metrics['平均换手率'] = avg_turnover
        
        # 评分
        if limit_down_ratio >= self.config['sentiment_thresholds']['limit_down_ratio']:
            risk_score += 3
            reasons.append(f"跌停股过多({limit_down_count}只, {limit_down_ratio*100:.1f}%)")
        
        if limit_up_ratio <= self.config['sentiment_thresholds']['limit_up_ratio']:
            risk_score += 2
            reasons.append(f"涨停股过少({limit_up_count}只)")
        
        if avg_turnover <= self.config['sentiment_thresholds']['low_turnover']:
            risk_score += 1
            reasons.append(f"换手率过低({avg_turnover:.2f}%)")
        
        return risk_score, reasons, metrics
    
    def _check_northbound(self, market_data: Dict) -> Tuple[float, List[str], Dict]:
        """检查北向资金"""
        risk_score = 0
        reasons = []
        metrics = {}
        
        northbound_flow = market_data.get('northbound_flow', 0)
        
        metrics['北向资金流入'] = northbound_flow
        
        # 评分
        if northbound_flow <= self.config['northbound_thresholds']['large_outflow']:
            risk_score += 2
            reasons.append(f"北向资金大额流出({northbound_flow:.1f}亿)")
        elif northbound_flow <= self.config['northbound_thresholds']['outflow']:
            risk_score += 1
            reasons.append(f"北向资金流出({northbound_flow:.1f}亿)")
        
        return risk_score, reasons, metrics
    
    def _check_self_performance(self, market_data: Dict) -> Tuple[float, List[str], Dict]:
        """检查自身表现"""
        risk_score = 0
        reasons = []
        metrics = {}
        
        daily_pnl_ratio = market_data.get('daily_pnl_ratio', 0)
        continuous_loss_days = market_data.get('continuous_loss_days', 0)
        max_drawdown = market_data.get('max_drawdown', 0)
        
        metrics['当日盈亏比例'] = daily_pnl_ratio * 100
        metrics['连续亏损天数'] = continuous_loss_days
        metrics['最大回撤'] = max_drawdown * 100
        
        # 评分
        if daily_pnl_ratio <= self.config['self_performance_thresholds']['daily_loss_ratio']:
            risk_score += 2
            reasons.append(f"当日亏损严重({daily_pnl_ratio*100:.2f}%)")
        
        if continuous_loss_days >= self.config['self_performance_thresholds']['continuous_loss_days']:
            risk_score += 2
            reasons.append(f"连续亏损{continuous_loss_days}天")
        
        if max_drawdown <= self.config['self_performance_thresholds']['max_drawdown']:
            risk_score += 3
            reasons.append(f"最大回撤严重({max_drawdown*100:.2f}%)")
        
        return risk_score, reasons, metrics
    
    def _determine_level(self, risk_score: float) -> Tuple[MarketCondition, CircuitBreakerLevel]:
        """根据风险评分确定市场状态和熔断级别"""
        if risk_score >= 9:
            return MarketCondition.HALT, CircuitBreakerLevel.LEVEL_4
        elif risk_score >= 7:
            return MarketCondition.DANGER, CircuitBreakerLevel.LEVEL_3
        elif risk_score >= 5:
            return MarketCondition.WARNING, CircuitBreakerLevel.LEVEL_2
        elif risk_score >= 3:
            return MarketCondition.CAUTION, CircuitBreakerLevel.LEVEL_1
        else:
            return MarketCondition.NORMAL, CircuitBreakerLevel.LEVEL_0
    
    def _print_signal(self, signal: MarketSignal, risk_score: float):
        """打印市场信号"""
        print(f"\n{'='*100}")
        print(f"市场环境熔断检查 - {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")
        
        # 市场状态
        condition_colors = {
            MarketCondition.NORMAL: "🟢",
            MarketCondition.CAUTION: "🟡",
            MarketCondition.WARNING: "🟠",
            MarketCondition.DANGER: "🔴",
            MarketCondition.HALT: "⛔"
        }
        
        color = condition_colors.get(signal.condition, "⚪")
        print(f"\n{color} 市场状态: {signal.condition.value}")
        print(f"   风险评分: {risk_score:.1f}/10")
        print(f"   熔断级别: {signal.breaker_level.name}")
        
        # 指标详情
        print(f"\n指标详情:")
        for key, value in signal.metrics.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.2f}")
            else:
                print(f"  - {key}: {value}")
        
        # 触发原因
        print(f"\n触发原因:")
        if signal.reason:
            for i, reason in enumerate(signal.reason.split("; "), 1):
                print(f"  {i}. {reason}")
        else:
            print(f"  无异常")
        
        # 操作建议
        print(f"\n操作建议:")
        print(f"  - 仓位调整: {signal.position_adjust_ratio * 100:.0f}%")
        print(f"  - 允许新开仓: {'是' if signal.allow_new_positions else '否'}")
        print(f"  - 强制平仓: {'是' if signal.force_close_all else '否'}")
        
        print(f"{'='*100}\n")
    
    def get_recent_signals(self, n: int = 10) -> List[MarketSignal]:
        """获取最近N个信号"""
        return self.history[-n:] if self.history else []
    
    def export_history(self, output_path: str):
        """导出历史信号到CSV"""
        if not self.history:
            print("无历史信号")
            return
        
        df = pd.DataFrame([{
            '时间': s.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            '市场状态': s.condition.value,
            '熔断级别': s.breaker_level.name,
            '仓位调整比例': f"{s.position_adjust_ratio * 100:.0f}%",
            '允许新开仓': '是' if s.allow_new_positions else '否',
            '强制平仓': '是' if s.force_close_all else '否',
            '触发原因': s.reason
        } for s in self.history])
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"历史信号已导出到: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 创建熔断机制
    breaker = MarketCircuitBreaker()
    
    # 模拟不同市场环境
    scenarios = [
        {
            'name': '正常市场',
            'data': {
                'index_changes': {'sh': 0.5, 'sz': 0.3, 'cyb': 0.8},
                'limit_up_count': 80,
                'limit_down_count': 30,
                'total_stocks': 4800,
                'avg_turnover': 2.5,
                'northbound_flow': 30,
                'daily_pnl_ratio': 0.02,
                'continuous_loss_days': 0,
                'max_drawdown': -0.05
            }
        },
        {
            'name': '谨慎市场',
            'data': {
                'index_changes': {'sh': -1.2, 'sz': -1.5, 'cyb': -1.8},
                'limit_up_count': 30,
                'limit_down_count': 100,
                'total_stocks': 4800,
                'avg_turnover': 1.8,
                'northbound_flow': -30,
                'daily_pnl_ratio': -0.01,
                'continuous_loss_days': 1,
                'max_drawdown': -0.08
            }
        },
        {
            'name': '极端市场',
            'data': {
                'index_changes': {'sh': -3.5, 'sz': -4.0, 'cyb': -4.5},
                'limit_up_count': 10,
                'limit_down_count': 300,
                'total_stocks': 4800,
                'avg_turnover': 0.8,
                'northbound_flow': -120,
                'daily_pnl_ratio': -0.08,
                'continuous_loss_days': 4,
                'max_drawdown': -0.18
            }
        }
    ]
    
    # 测试各种场景
    for scenario in scenarios:
        print(f"\n\n{'#'*100}")
        print(f"场景测试: {scenario['name']}")
        print(f"{'#'*100}")
        
        signal = breaker.check_market_condition(scenario['data'])
    
    # 导出历史
    breaker.export_history('circuit_breaker_history.csv')
    
    print("\n✅ 市场环境熔断机制测试完成！")
