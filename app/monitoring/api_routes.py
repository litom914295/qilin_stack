"""
监控API路由
"""

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from .health import health_checker
from .metrics import metrics
from prometheus_client import CONTENT_TYPE_LATEST

router = APIRouter()


@router.get("/health/live")
async def liveness():
    """存活检查端点"""
    result = await health_checker.check_liveness()
    status_code = 200 if result['status'] == 'healthy' else 503
    return JSONResponse(content=result, status_code=status_code)


@router.get("/health/ready")
async def readiness():
    """就绪检查端点"""
    result = await health_checker.check_readiness()
    status_code = 200 if result['status'] in ['healthy', 'degraded'] else 503
    return JSONResponse(content=result, status_code=status_code)


@router.get("/health/startup")
async def startup():
    """启动检查端点"""
    result = await health_checker.check_startup()
    status_code = 200 if result['status'] == 'healthy' else 503
    return JSONResponse(content=result, status_code=status_code)


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics端点"""
    return Response(
        content=metrics.get_metrics(),
        media_type=CONTENT_TYPE_LATEST
