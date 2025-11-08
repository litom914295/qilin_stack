"""
Jaeger分布式追踪集成
为麒麟量化系统提供全链路追踪能力
"""

import os
from jaeger_client import Config
from jaeger_client.reporter import LogReporter
from opentracing import tags, Format
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class JaegerTracer:
    """Jaeger追踪器封装"""
    
    def __init__(self, service_name: str = "qilin-stack", config: dict = None):
        """
        初始化Jaeger追踪器
        
        Parameters:
        -----------
        service_name: str
            服务名称
        config: dict
            Jaeger配置
        """
        self.service_name = service_name
        
        if config is None:
            config = {
                'sampler': {
                    'type': 'const',
                    'param': 1,  # 采样率：1表示100%采样
                },
                'local_agent': {
                    'reporting_host': os.getenv('JAEGER_AGENT_HOST', 'localhost'),
                    'reporting_port': int(os.getenv('JAEGER_AGENT_PORT', 6831)),
                },
                'logging': True,
            }
        
        self.jaeger_config = Config(
            config=config,
            service_name=service_name,
            validate=True
        )
        
        self.tracer = self.jaeger_config.initialize_tracer()
        logger.info(f"Jaeger tracer initialized for service: {service_name}")
    
    def trace(self, operation_name: str = None, tags_dict: dict = None):
        """
        函数装饰器，用于追踪函数调用
        
        Parameters:
        -----------
        operation_name: str
            操作名称，默认使用函数名
        tags_dict: dict
            追踪标签
        
        Usage:
        ------
        @tracer.trace(operation_name="predict", tags_dict={"model": "lgb"})
        def predict(data):
            return model.predict(data)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                
                with self.tracer.start_active_span(op_name) as scope:
                    span = scope.span
                    
                    # 设置基本标签
                    span.set_tag(tags.COMPONENT, self.service_name)
                    span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_SERVER)
                    
                    # 设置自定义标签
                    if tags_dict:
                        for key, value in tags_dict.items():
                            span.set_tag(key, value)
                    
                    # 记录输入参数
                    span.log_kv({
                        'event': 'function_call',
                        'args': str(args)[:100],  # 限制长度
                        'kwargs': str(kwargs)[:100]
                    })
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # 记录成功
                        span.set_tag(tags.ERROR, False)
                        span.log_kv({'event': 'success'})
                        
                        return result
                    
                    except Exception as e:
                        # 记录错误
                        span.set_tag(tags.ERROR, True)
                        span.log_kv({
                            'event': 'error',
                            'error.kind': type(e).__name__,
                            'error.message': str(e),
                        })
                        raise
            
            return wrapper
        return decorator
    
    def start_span(self, operation_name: str, child_of=None, tags_dict: dict = None):
        """
        手动启动一个span
        
        Parameters:
        -----------
        operation_name: str
            操作名称
        child_of: Span
            父span
        tags_dict: dict
            标签字典
            
        Returns:
        --------
        Span: OpenTracing span对象
        """
        span = self.tracer.start_span(
            operation_name=operation_name,
            child_of=child_of
        )
        
        if tags_dict:
            for key, value in tags_dict.items():
                span.set_tag(key, value)
        
        return span
    
    def inject_span_context(self, span, carrier: dict):
        """
        将span上下文注入到carrier中（用于跨服务传递）
        
        Parameters:
        -----------
        span: Span
            当前span
        carrier: dict
            用于传递的字典
        """
        self.tracer.inject(
            span_context=span.context,
            format=Format.HTTP_HEADERS,
            carrier=carrier
        )
    
    def extract_span_context(self, carrier: dict):
        """
        从carrier中提取span上下文
        
        Parameters:
        -----------
        carrier: dict
            包含span上下文的字典
            
        Returns:
        --------
        SpanContext: 提取的span上下文
        """
        return self.tracer.extract(
            format=Format.HTTP_HEADERS,
            carrier=carrier
        )
    
    def close(self):
        """关闭追踪器"""
        if self.tracer:
            self.tracer.close()
            logger.info(f"Jaeger tracer closed for service: {self.service_name}")


# 全局追踪器实例
_global_tracer = None


def init_tracer(service_name: str = "qilin-stack", config: dict = None):
    """
    初始化全局追踪器
    
    Parameters:
    -----------
    service_name: str
        服务名称
    config: dict
        配置字典
    """
    global _global_tracer
    _global_tracer = JaegerTracer(service_name=service_name, config=config)
    return _global_tracer


def get_tracer() -> JaegerTracer:
    """获取全局追踪器"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = JaegerTracer()
    return _global_tracer


# 使用示例
if __name__ == "__main__":
    # 初始化追踪器
    tracer = init_tracer(service_name="qilin-auction-engine")
    
    # 方式1: 使用装饰器
    @tracer.trace(operation_name="load_candidates", tags_dict={"source": "database"})
    def load_t_day_candidates():
        """加载T日候选股票"""
        print("Loading T-day candidates...")
        # 模拟加载
        import time
        time.sleep(0.1)
        return ["000001.SZ", "600519.SH"]
    
    @tracer.trace(operation_name="extract_features")
    def extract_auction_features(symbols):
        """提取竞价特征"""
        print(f"Extracting features for {len(symbols)} symbols...")
        import time
        time.sleep(0.2)
        return {"features": [0.5, 0.3, 0.8]}
    
    @tracer.trace(operation_name="predict")
    def predict_auction_strength(features):
        """预测竞价强度"""
        print("Predicting auction strength...")
        import time
        time.sleep(0.15)
        return 0.85
    
    # 执行追踪
    print("Starting traced execution...")
    
    candidates = load_t_day_candidates()
    features = extract_auction_features(candidates)
    prediction = predict_auction_strength(features)
    
    print(f"Prediction result: {prediction}")
    
    # 方式2: 手动创建span
    with tracer.tracer.start_active_span('manual_operation') as scope:
        span = scope.span
        span.set_tag('custom_tag', 'custom_value')
        span.log_kv({'event': 'manual_log', 'info': 'This is a manual span'})
        
        print("Manual span created")
        import time
        time.sleep(0.1)
    
    # 关闭追踪器
    print("\nClosing tracer...")
    tracer.close()
    
    print("\n访问 Jaeger UI: http://localhost:16686")
