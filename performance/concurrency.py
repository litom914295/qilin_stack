"""
并发优化模块 - 提升系统性能
"""
import asyncio
import concurrent.futures
from typing import List, Callable, Any, Dict
from functools import wraps
import time


class ConcurrencyOptimizer:
    """并发优化器"""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers//2)
    
    async def gather_parallel(self, *tasks, return_exceptions: bool = True):
        """并行执行多个异步任务"""
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    
    async def run_in_thread(self, func: Callable, *args, **kwargs):
        """在线程池中运行同步函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, lambda: func(*args, **kwargs))
    
    async def run_in_process(self, func: Callable, *args, **kwargs):
        """在进程池中运行CPU密集型函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, lambda: func(*args, **kwargs))
    
    async def batch_process(self, items: List[Any], func: Callable, batch_size: int = 10):
        """批量并行处理"""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [func(item) for item in batch]
            batch_results = await self.gather_parallel(*batch_tasks)
            results.extend(batch_results)
        return results
    
    def cleanup(self):
        """清理资源"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


# 全局优化器实例
_optimizer = None

def get_optimizer() -> ConcurrencyOptimizer:
    """获取全局优化器实例"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ConcurrencyOptimizer()
    return _optimizer


def parallel_task(func):
    """装饰器：将函数标记为可并行执行"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            optimizer = get_optimizer()
            return await optimizer.run_in_thread(func, *args, **kwargs)
    return wrapper


async def parallel_decision_generation(symbols: List[str], date: str):
    """并行生成多个股票的决策（演示）。"""
    from decision_engine.core import get_decision_engine

    engine = get_decision_engine()
    optimizer = get_optimizer()

    # 并行调用三个信号生成器（批量）
    tasks = [
        engine.qlib_generator.generate_signals(symbols, date),
        engine.ta_generator.generate_signals(symbols, date),
        engine.rd_generator.generate_signals(symbols, date),
    ]
    results = await optimizer.gather_parallel(*tasks)

    # 扁平化并过滤异常
    flat: list = []
    for res in results:
        if isinstance(res, Exception):
            continue
        flat.extend(res)
    return flat


# 使用示例
async def demo_concurrency():
    """并发优化演示"""
    optimizer = get_optimizer()
    
    # 模拟任务
    async def task(n):
        await asyncio.sleep(0.1)
        return f"Task {n} completed"
    
    # 并行执行10个任务
    tasks = [task(i) for i in range(10)]
    results = await optimizer.gather_parallel(*tasks)
    
    print(f"完成{len(results)}个任务")
    optimizer.cleanup()
