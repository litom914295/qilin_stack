"""
业务金指标监控系统（P0-11）
核心业务指标：推荐命中率、收益率、信号质量、用户行为
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client import CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)


class RecommendationAction(str, Enum):
    """推荐动作"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalQuality(str, Enum):
    """信号质量"""
    HIGH = "high"      # 置信度 >= 0.8
    MEDIUM = "medium"  # 置信度 0.6-0.8
    LOW = "low"        # 置信度 < 0.6


@dataclass
class RecommendationResult:
    """推荐结果"""
    recommendation_id: str
    stock_code: str
    action: RecommendationAction
    confidence: float
    target_price: Optional[float]
    timestamp: datetime
    signal_quality: SignalQuality
    
    # T+1验证字段
    actual_return: Optional[float] = None  # 实际收益率
    is_correct: Optional[bool] = None      # 方向是否正确
    validated_at: Optional[datetime] = None


@dataclass
class BusinessMetrics:
    """业务指标快照"""
    timestamp: datetime
    
    # 推荐指标
    total_recommendations: int
    high_quality_recommendations: int
    medium_quality_recommendations: int
    low_quality_recommendations: int
    
    # 命中率指标（T+1验证）
    validated_recommendations: int
    correct_recommendations: int
    hit_rate: float  # 命中率
    
    # 收益指标
    avg_return: float       # 平均收益率
    max_return: float       # 最大收益率
    min_return: float       # 最小收益率
    positive_returns: int   # 正收益数量
    
    # 信号质量
    avg_confidence: float
    signal_coverage: float  # 信号覆盖率
    
    # 用户行为（如果有）
    total_users: int = 0
    active_users: int = 0
    recommendation_views: int = 0
    recommendation_follows: int = 0


class BusinessMetricsCollector:
    """业务指标采集器"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._init_prometheus_metrics()
        
        # 内存存储（生产环境应使用数据库）
        self.recommendations: Dict[str, RecommendationResult] = {}
        self.daily_metrics: List[BusinessMetrics] = []
    
    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        
        # 推荐总数
        self.recommendations_total = Counter(
            'qilin_recommendations_total',
            'Total number of recommendations generated',
            ['action', 'quality'],
            registry=self.registry
        
        # 推荐命中率
        self.recommendation_hit_rate = Gauge(
            'qilin_recommendation_hit_rate',
            'Recommendation hit rate (T+1 validation)',
            ['timeframe'],  # 1d, 7d, 30d
            registry=self.registry
        
        # 推荐收益率
        self.recommendation_return = Histogram(
            'qilin_recommendation_return_percent',
            'Recommendation return percentage',
            buckets=[-10, -5, -2, -1, 0, 1, 2, 5, 10, 20],
            registry=self.registry
        
        # 平均收益率
        self.avg_return = Gauge(
            'qilin_avg_return_percent',
            'Average return percentage',
            ['timeframe'],
            registry=self.registry
        
        # 推荐置信度
        self.recommendation_confidence = Histogram(
            'qilin_recommendation_confidence',
            'Recommendation confidence score',
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            registry=self.registry
        
        # 信号覆盖率
        self.signal_coverage = Gauge(
            'qilin_signal_coverage',
            'Signal coverage ratio',
            registry=self.registry
        
        # 当日推荐数量
        self.daily_recommendations = Gauge(
            'qilin_daily_recommendations',
            'Number of recommendations generated today',
            ['action'],
            registry=self.registry
        
        # 用户行为指标
        self.active_users = Gauge(
            'qilin_active_users',
            'Number of active users',
            ['period'],  # daily, weekly, monthly
            registry=self.registry
        
        self.recommendation_follows = Counter(
            'qilin_recommendation_follows_total',
            'Total recommendation follows by users',
            ['action'],
            registry=self.registry
    
    def record_recommendation(
        self,
        recommendation_id: str,
        stock_code: str,
        action: RecommendationAction,
        confidence: float,
        target_price: Optional[float] = None
    ) -> RecommendationResult:
        """
        记录推荐
        
        Args:
            recommendation_id: 推荐ID
            stock_code: 股票代码
            action: 推荐动作
            confidence: 置信度
            target_price: 目标价
            
        Returns:
            推荐结果对象
        """
        # 判断信号质量
        if confidence >= 0.8:
            quality = SignalQuality.HIGH
        elif confidence >= 0.6:
            quality = SignalQuality.MEDIUM
        else:
            quality = SignalQuality.LOW
        
        result = RecommendationResult(
            recommendation_id=recommendation_id,
            stock_code=stock_code,
            action=action,
            confidence=confidence,
            target_price=target_price,
            timestamp=datetime.now(),
            signal_quality=quality
        
        # 存储
        self.recommendations[recommendation_id] = result
        
        # 更新Prometheus指标
        self.recommendations_total.labels(
            action=action.value,
            quality=quality.value
        ).inc()
        
        self.recommendation_confidence.observe(confidence)
        
        logger.info(
            f"Recorded recommendation {recommendation_id}: "
            f"{stock_code} {action.value} (confidence={confidence:.2f}, quality={quality.value})"
        
        return result
    
    def validate_recommendation(
        self,
        recommendation_id: str,
        actual_return: float
    ) -> bool:
        """
        T+1验证推荐结果
        
        Args:
            recommendation_id: 推荐ID
            actual_return: 实际收益率
            
        Returns:
            是否验证成功
        """
        if recommendation_id not in self.recommendations:
            logger.warning(f"Recommendation {recommendation_id} not found")
            return False
        
        rec = self.recommendations[recommendation_id]
        
        # 判断方向是否正确
        if rec.action == RecommendationAction.BUY:
            is_correct = actual_return > 0
        elif rec.action == RecommendationAction.SELL:
            is_correct = actual_return < 0
        else:  # HOLD
            is_correct = abs(actual_return) < 0.02  # ±2%视为持有正确
        
        # 更新验证结果
        rec.actual_return = actual_return
        rec.is_correct = is_correct
        rec.validated_at = datetime.now()
        
        # 更新Prometheus指标
        self.recommendation_return.observe(actual_return * 100)  # 转为百分比
        
        logger.info(
            f"Validated recommendation {recommendation_id}: "
            f"return={actual_return:.2%}, correct={is_correct}"
        
        return True
    
    def calculate_hit_rate(self, days: int = 1) -> float:
        """
        计算命中率
        
        Args:
            days: 统计天数
            
        Returns:
            命中率（0-1）
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        validated = [
            rec for rec in self.recommendations.values()
            if rec.validated_at and rec.validated_at >= cutoff_time
        ]
        
        if not validated:
            return 0.0
        
        correct = sum(1 for rec in validated if rec.is_correct)
        hit_rate = correct / len(validated)
        
        # 更新Prometheus指标
        timeframe = f"{days}d"
        self.recommendation_hit_rate.labels(timeframe=timeframe).set(hit_rate)
        
        return hit_rate
    
    def calculate_avg_return(self, days: int = 1) -> float:
        """
        计算平均收益率
        
        Args:
            days: 统计天数
            
        Returns:
            平均收益率
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        returns = [
            rec.actual_return for rec in self.recommendations.values()
            if rec.actual_return is not None 
            and rec.validated_at 
            and rec.validated_at >= cutoff_time
        ]
        
        if not returns:
            return 0.0
        
        avg = sum(returns) / len(returns)
        
        # 更新Prometheus指标
        timeframe = f"{days}d"
        self.avg_return.labels(timeframe=timeframe).set(avg * 100)
        
        return avg
    
    def calculate_signal_coverage(self, total_stocks: int) -> float:
        """
        计算信号覆盖率
        
        Args:
            total_stocks: 股票池总数
            
        Returns:
            覆盖率（0-1）
        """
        today = datetime.now().date()
        today_recs = [
            rec for rec in self.recommendations.values()
            if rec.timestamp.date() == today
        ]
        
        unique_stocks = len(set(rec.stock_code for rec in today_recs))
        coverage = unique_stocks / total_stocks if total_stocks > 0 else 0
        
        self.signal_coverage.set(coverage)
        
        return coverage
    
    def get_daily_summary(self) -> BusinessMetrics:
        """
        获取当日业务指标汇总
        
        Returns:
            业务指标对象
        """
        today = datetime.now().date()
        
        # 今日推荐
        today_recs = [
            rec for rec in self.recommendations.values()
            if rec.timestamp.date() == today
        ]
        
        # 已验证推荐
        validated_recs = [rec for rec in today_recs if rec.validated_at]
        correct_recs = [rec for rec in validated_recs if rec.is_correct]
        
        # 收益统计
        returns = [rec.actual_return for rec in validated_recs if rec.actual_return is not None]
        
        metrics = BusinessMetrics(
            timestamp=datetime.now(),
            total_recommendations=len(today_recs),
            high_quality_recommendations=sum(
                1 for rec in today_recs if rec.signal_quality == SignalQuality.HIGH
            ),
            medium_quality_recommendations=sum(
                1 for rec in today_recs if rec.signal_quality == SignalQuality.MEDIUM
            ),
            low_quality_recommendations=sum(
                1 for rec in today_recs if rec.signal_quality == SignalQuality.LOW
            ),
            validated_recommendations=len(validated_recs),
            correct_recommendations=len(correct_recs),
            hit_rate=len(correct_recs) / len(validated_recs) if validated_recs else 0,
            avg_return=sum(returns) / len(returns) if returns else 0,
            max_return=max(returns) if returns else 0,
            min_return=min(returns) if returns else 0,
            positive_returns=sum(1 for r in returns if r > 0),
            avg_confidence=sum(rec.confidence for rec in today_recs) / len(today_recs) if today_recs else 0,
            signal_coverage=self.calculate_signal_coverage(3000)  # 假设3000只股票
        
        self.daily_metrics.append(metrics)
        
        # 更新每日推荐数量
        for action in RecommendationAction:
            count = sum(1 for rec in today_recs if rec.action == action)
            self.daily_recommendations.labels(action=action.value).set(count)
        
        return metrics
    
    def export_metrics(self) -> str:
        """
        导出Prometheus指标
        
        Returns:
            Prometheus文本格式
        """
        # 先更新各时间段的指标
        self.calculate_hit_rate(days=1)
        self.calculate_hit_rate(days=7)
        self.calculate_hit_rate(days=30)
        
        self.calculate_avg_return(days=1)
        self.calculate_avg_return(days=7)
        self.calculate_avg_return(days=30)
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_report(self, days: int = 7) -> Dict[str, Any]:
        """
        生成业务指标报告
        
        Args:
            days: 统计天数
            
        Returns:
            指标报告字典
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        period_recs = [
            rec for rec in self.recommendations.values()
            if rec.timestamp >= cutoff_time
        ]
        
        validated_recs = [rec for rec in period_recs if rec.validated_at]
        correct_recs = [rec for rec in validated_recs if rec.is_correct]
        
        returns = [rec.actual_return for rec in validated_recs if rec.actual_return is not None]
        
        report = {
            'period': f'last_{days}d',
            'start_date': cutoff_time.date().isoformat(),
            'end_date': datetime.now().date().isoformat(),
            'total_recommendations': len(period_recs),
            'validated_recommendations': len(validated_recs),
            'validation_rate': len(validated_recs) / len(period_recs) if period_recs else 0,
            'hit_rate': len(correct_recs) / len(validated_recs) if validated_recs else 0,
            'avg_return': sum(returns) / len(returns) if returns else 0,
            'max_return': max(returns) if returns else 0,
            'min_return': min(returns) if returns else 0,
            'positive_return_ratio': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
            'by_action': {},
            'by_quality': {}
        }
        
        # 按动作统计
        for action in RecommendationAction:
            action_recs = [rec for rec in validated_recs if rec.action == action]
            if action_recs:
                action_correct = sum(1 for rec in action_recs if rec.is_correct)
                report['by_action'][action.value] = {
                    'count': len(action_recs),
                    'hit_rate': action_correct / len(action_recs)
                }
        
        # 按质量统计
        for quality in SignalQuality:
            quality_recs = [rec for rec in validated_recs if rec.signal_quality == quality]
            if quality_recs:
                quality_correct = sum(1 for rec in quality_recs if rec.is_correct)
                report['by_quality'][quality.value] = {
                    'count': len(quality_recs),
                    'hit_rate': quality_correct / len(quality_recs)
                }
        
        return report


def main():
    """示例用法"""
    from app.core.logging_setup import setup_logging
    setup_logging()
    
    collector = BusinessMetricsCollector()
    
    # 模拟记录推荐
    rec1 = collector.record_recommendation(
        recommendation_id="rec_001",
        stock_code="000001",
        action=RecommendationAction.BUY,
        confidence=0.85,
        target_price=15.0
    )
    
    rec2 = collector.record_recommendation(
        recommendation_id="rec_002",
        stock_code="000002",
        action=RecommendationAction.SELL,
        confidence=0.72,
        target_price=10.0
    )
    
    # 模拟T+1验证
    collector.validate_recommendation("rec_001", actual_return=0.05)  # +5%
    collector.validate_recommendation("rec_002", actual_return=-0.03)  # -3%
    
    # 计算指标
    hit_rate = collector.calculate_hit_rate(days=1)
    avg_return = collector.calculate_avg_return(days=1)
    
    logger.info(f"Hit Rate: {hit_rate:.1%}")
    logger.info(f"Avg Return: {avg_return:.2%}")
    
    # 获取汇总
    summary = collector.get_daily_summary()
    logger.info("Daily Summary:")
    logger.info(f"  Total Recommendations: {summary.total_recommendations}")
    logger.info(f"  Hit Rate: {summary.hit_rate:.1%}")
    logger.info(f"  Avg Return: {summary.avg_return:.2%}")
    
    # 导出Prometheus指标
    logger.info("\n" + collector.export_metrics())


if __name__ == "__main__":
    main()
