"""
业务指标HTTP端点
提供Prometheus metrics endpoint和业务报告API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from typing import Optional
import logging

from monitoring.business_metrics import (
    BusinessMetricsCollector,
    RecommendationAction
)
from monitoring.pool_metrics import start_pool_metrics_collector
from monitoring.slo_metrics import get_slo_metrics

logger = logging.getLogger(__name__)

# 全局业务指标采集器
metrics_collector = BusinessMetricsCollector()


def create_metrics_app() -> FastAPI:
    """创建metrics API应用"""
    app = FastAPI(
        title="Qilin Business Metrics API",
        description="业务指标采集与导出",
        version="1.0.0"
    )

    # 启动/停止钩子：启动连接池指标采集，复用业务指标的registry
    @app.on_event("startup")
    async def _on_startup():
        try:
            # 连接池指标采集
            start_pool_metrics_collector(registry=metrics_collector.registry, interval_seconds=10.0)
            logger.info("Pool metrics collector started (shared registry)")
            # 初始化 SLO 指标到同一 registry
            _ = get_slo_metrics(registry=metrics_collector.registry)
            logger.info("SLO metrics initialized (shared registry)")
        except Exception as e:
            logger.warning(f"Failed to init metrics collectors: {e}")

    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        """
        Prometheus metrics endpoint
        
        供Prometheus抓取业务指标
        """
        try:
            return metrics_collector.export_metrics()
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to export metrics")
    
    @app.post("/api/recommendations")
    async def record_recommendation(
        recommendation_id: str,
        stock_code: str,
        action: str,
        confidence: float,
        target_price: Optional[float] = None
    ):
        """
        记录推荐
        
        Args:
            recommendation_id: 推荐ID
            stock_code: 股票代码
            action: 动作 (buy/sell/hold)
            confidence: 置信度
            target_price: 目标价
        """
        try:
            action_enum = RecommendationAction(action.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {action}. Must be buy/sell/hold"
            )
        
        if not 0 <= confidence <= 1:
            raise HTTPException(
                status_code=400,
                detail="Confidence must be between 0 and 1"
            )
        
        result = metrics_collector.record_recommendation(
            recommendation_id=recommendation_id,
            stock_code=stock_code,
            action=action_enum,
            confidence=confidence,
            target_price=target_price
        )
        
        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "signal_quality": result.signal_quality.value
        }
    
    @app.post("/api/recommendations/{recommendation_id}/validate")
    async def validate_recommendation(
        recommendation_id: str,
        actual_return: float
    ):
        """
        T+1验证推荐
        
        Args:
            recommendation_id: 推荐ID
            actual_return: 实际收益率
        """
        success = metrics_collector.validate_recommendation(
            recommendation_id=recommendation_id,
            actual_return=actual_return
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Recommendation {recommendation_id} not found"
            )
        
        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "actual_return": actual_return
        }
    
    @app.get("/api/metrics/hit-rate")
    async def get_hit_rate(days: int = 1):
        """
        获取命中率
        
        Args:
            days: 统计天数
        """
        hit_rate = metrics_collector.calculate_hit_rate(days=days)
        return {
            "hit_rate": hit_rate,
            "timeframe": f"{days}d"
        }
    
    @app.get("/api/metrics/avg-return")
    async def get_avg_return(days: int = 1):
        """
        获取平均收益率
        
        Args:
            days: 统计天数
        """
        avg_return = metrics_collector.calculate_avg_return(days=days)
        return {
            "avg_return": avg_return,
            "avg_return_percent": avg_return * 100,
            "timeframe": f"{days}d"
        }
    
    @app.get("/api/metrics/summary")
    async def get_summary():
        """获取当日业务指标汇总"""
        summary = metrics_collector.get_daily_summary()
        return {
            "timestamp": summary.timestamp.isoformat(),
            "total_recommendations": summary.total_recommendations,
            "high_quality_recommendations": summary.high_quality_recommendations,
            "medium_quality_recommendations": summary.medium_quality_recommendations,
            "low_quality_recommendations": summary.low_quality_recommendations,
            "validated_recommendations": summary.validated_recommendations,
            "correct_recommendations": summary.correct_recommendations,
            "hit_rate": summary.hit_rate,
            "avg_return": summary.avg_return,
            "max_return": summary.max_return,
            "min_return": summary.min_return,
            "positive_returns": summary.positive_returns,
            "avg_confidence": summary.avg_confidence,
            "signal_coverage": summary.signal_coverage
        }
    
    @app.get("/api/metrics/report")
    async def get_report(days: int = 7):
        """
        获取业务指标报告
        
        Args:
            days: 统计天数
        """
        report = metrics_collector.get_metrics_report(days=days)
        return report
    
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "healthy"}
    
    return app


# 创建应用实例
app = create_metrics_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        log_level="info"
    )
