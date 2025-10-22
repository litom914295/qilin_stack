"""
审计中间件
自动记录API请求审计日志
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging

from security.audit_enhanced import AuditLogger, AuditEventType

logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """API审计中间件"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_logger = AuditLogger(enable_console=False)
    
    async def dispatch(self, request: Request, call_next):
        """处理请求并记录审计日志"""
        
        # 提取请求信息
        user_id = request.headers.get("X-User-ID", "anonymous")
        user_role = request.headers.get("X-User-Role", None)
        client_ip = request.client.host if request.client else "unknown"
        
        start_time = time.time()
        
        # 处理请求
        try:
            response: Response = await call_next(request)
            
            # 判断结果
            result = "success" if response.status_code < 400 else "failure"
            
            # 记录审计日志
            await self.audit_logger.log_event(
                event_type=AuditEventType.API_CALL,
                user_id=user_id,
                action=request.method,
                resource=str(request.url.path),
                result=result,
                ip_address=client_ip,
                user_role=user_role,
                metadata={
                    "method": request.method,
                    "path": str(request.url.path),
                    "query_params": dict(request.query_params),
                    "status_code": response.status_code,
                    "duration_ms": (time.time() - start_time) * 1000
                }
            )
            
            return response
            
        except Exception as e:
            # 记录失败
            await self.audit_logger.log_event(
                event_type=AuditEventType.API_CALL,
                user_id=user_id,
                action=request.method,
                resource=str(request.url.path),
                result="error",
                ip_address=client_ip,
                user_role=user_role,
                metadata={
                    "method": request.method,
                    "path": str(request.url.path),
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
            )
            raise


def setup_audit_middleware(app):
    """
    设置审计中间件
    
    使用方法:
    ```python
    from fastapi import FastAPI
    from api.audit_middleware import setup_audit_middleware
    
    app = FastAPI()
    setup_audit_middleware(app)
    ```
    """
    app.add_middleware(AuditMiddleware)
    logger.info("Audit middleware enabled")
