"""
健康检查API端点
提供依赖健康状态查询和整体健康检查
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from typing import Optional
import logging
import yaml
import os

from monitoring.dependency_probes import (
    DependencyHealthProbe,
    DependencyConfig,
    DependencyType,
    HealthStatus
)

logger = logging.getLogger(__name__)

# 全局探针实例
health_probe: Optional[DependencyHealthProbe] = None


def load_dependencies_config(config_path: str = "monitoring/dependencies.yaml"):
    """加载依赖配置"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return []
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dependencies = []
    for dep in config.get('dependencies', []):
        dependencies.append(DependencyConfig(
            name=dep['name'],
            type=DependencyType(dep['type']),
            endpoint=dep['endpoint'],
            timeout_seconds=dep.get('timeout_seconds', 5.0),
            critical=dep.get('critical', True)
        ))
    
    return dependencies


def create_health_app() -> FastAPI:
    """创建健康检查API"""
    app = FastAPI(
        title="Qilin Health Check API",
        description="依赖健康状态监控",
        version="1.0.0"

   )
    
    global health_probe
    health_probe = DependencyHealthProbe()
    
    # 加载依赖配置
    dependencies = load_dependencies_config()
    for dep in dependencies:
        health_probe.register_dependency(dep)
    
    logger.info(f"Registered {len(dependencies)} dependencies")
    
    @app.get("/health")
    async def health_check():
        """
        基础健康检查
        
        返回服务自身是否健康
        """
        return {"status": "healthy"}
    
    @app.get("/ready")
    async def readiness_check():
        """
        就绪检查
        
        检查所有关键依赖是否就绪
        """
        results = await health_probe.probe_all()
        overall_status = health_probe.get_overall_health()
        
        if overall_status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "Critical dependencies unavailable"
                }
            )
        
        return {"status": "ready"}
    
    @app.get("/health/dependencies")
    async def get_dependencies_health():
        """
        获取所有依赖健康状态
        
        Returns:
            依赖健康摘要
        """
        await health_probe.probe_all()
        return health_probe.get_health_summary()
    
    @app.get("/health/dependencies/{dependency_name}")
    async def get_dependency_health(dependency_name: str):
        """
        获取单个依赖健康状态
        
        Args:
            dependency_name: 依赖名称
        """
        try:
            result = await health_probe.probe_dependency(dependency_name)
            return {
                "name": result.dependency_name,
                "type": result.dependency_type.value,
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "timestamp": result.timestamp.isoformat(),
                "error": result.error_message,
                "details": result.details
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        """
        Prometheus metrics endpoint
        
        导出依赖健康指标
        """
        return health_probe.export_metrics()
    
    @app.on_event("startup")
    async def startup_event():
        """启动时执行初始探测"""
        logger.info("Starting initial health probe...")
        await health_probe.probe_all()
        logger.info("Initial health probe completed")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """关闭时清理资源"""
        logger.info("Cleaning up health probe resources...")
        await health_probe.cleanup()
    
    return app


# 创建应用实例
app = create_health_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
