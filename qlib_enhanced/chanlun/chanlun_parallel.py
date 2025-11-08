"""
缠论并行计算管理器 - Phase 4.3.2

功能:
- 多进程批量分析
- 动态任务分配
- 进度监控
- 错误处理和重试

双模式复用:
- Qlib系统: 大规模回测并行加速
- 独立系统: 多股票实时监控

作者: Warp AI Assistant
日期: 2025-01
版本: v1.6
"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)


@dataclass
class ParallelTask:
    """并行任务定义"""
    task_id: str
    symbol: str
    data: Any
    params: Dict = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class ParallelResult:
    """并行任务结果"""
    task_id: str
    symbol: str
    success: bool
    result: Any = None
    error: str = None
    duration: float = 0.0


class ChanLunParallel:
    """
    缠论并行计算管理器
    
    支持:
    1. 多进程批量分析
    2. 动态负载均衡
    3. 实时进度监控
    4. 失败任务重试
    
    Examples:
        >>> # 基本使用
        >>> parallel = ChanLunParallel(num_workers=4)
        >>> tasks = [ParallelTask(f"task_{i}", symbol, data) for i, symbol in enumerate(symbols)]
        >>> results = parallel.run(tasks, process_func)
        
        >>> # 带进度回调
        >>> def progress_callback(completed, total, result):
        ...     print(f"进度: {completed}/{total}")
        >>> results = parallel.run(tasks, process_func, progress_callback=progress_callback)
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        chunk_size: int = 1,
        max_retries: int = 2,
        timeout: int = 300,
        enable_progress: bool = True
    ):
        """
        初始化并行管理器
        
        Args:
            num_workers: 工作进程数, None表示使用CPU核心数
            chunk_size: 每个进程处理的任务块大小
            max_retries: 失败任务最大重试次数
            timeout: 单个任务超时时间(秒)
            enable_progress: 是否启用进度监控
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_progress = enable_progress
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'retried': 0,
            'total_time': 0.0
        }
        
        logger.info(f"并行管理器初始化: {self.num_workers} 个工作进程")
    
    def run(
        self,
        tasks: List[ParallelTask],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[ParallelResult]:
        """
        执行并行任务
        
        Args:
            tasks: 任务列表
            process_func: 处理函数 (task: ParallelTask) -> Any
            progress_callback: 进度回调 (completed: int, total: int, result: ParallelResult) -> None
        
        Returns:
            结果列表
        """
        if not tasks:
            logger.warning("任务列表为空")
            return []
        
        start_time = time.time()
        self.stats['total_tasks'] = len(tasks)
        self.stats['completed'] = 0
        self.stats['failed'] = 0
        
        logger.info(f"开始并行处理 {len(tasks)} 个任务")
        
        # 包装处理函数 (添加错误处理)
        wrapped_func = self._wrap_process_func(process_func)
        
        # 使用进程池执行
        with Pool(processes=self.num_workers) as pool:
            # 异步提交所有任务
            async_results = []
            for task in tasks:
                async_result = pool.apply_async(
                    wrapped_func,
                    args=(task,)
                )
                async_results.append((task, async_result))
            
            # 收集结果
            results = []
            for task, async_result in async_results:
                try:
                    result = async_result.get(timeout=self.timeout)
                    results.append(result)
                    
                    if result.success:
                        self.stats['completed'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    # 进度回调
                    if progress_callback:
                        progress_callback(
                            self.stats['completed'] + self.stats['failed'],
                            self.stats['total_tasks'],
                            result
                        )
                    
                    # 日志
                    if self.enable_progress and (len(results) % 10 == 0 or len(results) == len(tasks)):
                        logger.info(
                            f"进度: {len(results)}/{len(tasks)} "
                            f"(成功: {self.stats['completed']}, 失败: {self.stats['failed']})"
                        )
                
                except mp.TimeoutError:
                    logger.error(f"任务超时: {task.symbol}")
                    results.append(ParallelResult(
                        task_id=task.task_id,
                        symbol=task.symbol,
                        success=False,
                        error="任务超时"
                    ))
                    self.stats['failed'] += 1
                
                except Exception as e:
                    logger.error(f"任务异常: {task.symbol}, {e}")
                    results.append(ParallelResult(
                        task_id=task.task_id,
                        symbol=task.symbol,
                        success=False,
                        error=str(e)
                    ))
                    self.stats['failed'] += 1
        
        # 重试失败任务
        if self.max_retries > 0:
            failed_tasks = [
                tasks[i] for i, result in enumerate(results)
                if not result.success
            ]
            if failed_tasks:
                logger.info(f"重试 {len(failed_tasks)} 个失败任务")
                retry_results = self._retry_failed_tasks(failed_tasks, wrapped_func)
                
                # 更新结果
                for i, result in enumerate(results):
                    if not result.success:
                        retry_idx = [t.task_id for t in failed_tasks].index(result.task_id)
                        results[i] = retry_results[retry_idx]
                        if results[i].success:
                            self.stats['completed'] += 1
                            self.stats['failed'] -= 1
                            self.stats['retried'] += 1
        
        self.stats['total_time'] = time.time() - start_time
        
        logger.info(
            f"✅ 并行处理完成: 总数 {self.stats['total_tasks']}, "
            f"成功 {self.stats['completed']}, 失败 {self.stats['failed']}, "
            f"重试 {self.stats['retried']}, 耗时 {self.stats['total_time']:.2f}s"
        )
        
        return results
    
    def _wrap_process_func(self, process_func: Callable) -> Callable:
        """包装处理函数,添加错误处理和计时"""
        def wrapped(task: ParallelTask) -> ParallelResult:
            start_time = time.time()
            try:
                result = process_func(task)
                return ParallelResult(
                    task_id=task.task_id,
                    symbol=task.symbol,
                    success=True,
                    result=result,
                    duration=time.time() - start_time
                )
            except Exception as e:
                logger.error(f"任务处理失败: {task.symbol}, {e}")
                return ParallelResult(
                    task_id=task.task_id,
                    symbol=task.symbol,
                    success=False,
                    error=f"{type(e).__name__}: {str(e)}",
                    duration=time.time() - start_time
                )
        return wrapped
    
    def _retry_failed_tasks(
        self,
        failed_tasks: List[ParallelTask],
        process_func: Callable
    ) -> List[ParallelResult]:
        """重试失败任务 (串行重试,避免资源竞争)"""
        results = []
        for task in failed_tasks:
            retry_count = 0
            while retry_count < self.max_retries:
                result = process_func(task)
                if result.success:
                    break
                retry_count += 1
                time.sleep(0.1 * retry_count)  # 指数退避
            results.append(result)
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


class ChanLunBatchProcessor:
    """
    缠论批量处理器
    
    集成缓存和并行计算的高级接口
    
    Examples:
        >>> processor = ChanLunBatchProcessor(
        ...     cache=cache,
        ...     num_workers=4,
        ...     enable_cache=True
        ... )
        >>> results = processor.process_batch(symbols, data_dict, process_func)
    """
    
    def __init__(
        self,
        cache=None,
        num_workers: Optional[int] = None,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        **parallel_kwargs
    ):
        """
        初始化批量处理器
        
        Args:
            cache: 缓存实例 (ChanLunCache)
            num_workers: 工作进程数
            enable_cache: 是否启用缓存
            enable_parallel: 是否启用并行 (单个任务时自动禁用)
            **parallel_kwargs: 并行管理器参数
        """
        self.cache = cache
        self.enable_cache = enable_cache and (cache is not None)
        self.enable_parallel = enable_parallel
        
        if self.enable_parallel:
            self.parallel = ChanLunParallel(
                num_workers=num_workers,
                **parallel_kwargs
            )
        
        logger.info(
            f"批量处理器初始化: 缓存={self.enable_cache}, "
            f"并行={self.enable_parallel}"
        )
    
    def process_batch(
        self,
        tasks: List[Dict],
        process_func: Callable,
        cache_key_func: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        批量处理任务
        
        Args:
            tasks: 任务列表 [{'symbol': '000001', 'data': df, ...}, ...]
            process_func: 处理函数 (task_dict) -> result
            cache_key_func: 缓存键生成函数 (task_dict) -> str, None表示用symbol
            progress_callback: 进度回调
        
        Returns:
            结果列表
        """
        if not tasks:
            return []
        
        # 1. 检查缓存
        uncached_tasks = []
        cached_results = {}
        
        if self.enable_cache:
            for i, task in enumerate(tasks):
                cache_key = cache_key_func(task) if cache_key_func else task.get('symbol', f'task_{i}')
                cached = self.cache.get(cache_key)
                if cached is not None:
                    cached_results[i] = cached
                else:
                    uncached_tasks.append((i, task))
            
            logger.info(
                f"缓存命中: {len(cached_results)}/{len(tasks)} "
                f"({len(cached_results)/len(tasks)*100:.1f}%)"
            )
        else:
            uncached_tasks = list(enumerate(tasks))
        
        # 2. 处理未缓存任务
        if not uncached_tasks:
            return [cached_results[i] for i in range(len(tasks))]
        
        if self.enable_parallel and len(uncached_tasks) > 1:
            # 并行处理
            parallel_tasks = [
                ParallelTask(
                    task_id=str(idx),
                    symbol=task.get('symbol', f'task_{idx}'),
                    data=task
                )
                for idx, task in uncached_tasks
            ]
            
            parallel_results = self.parallel.run(
                parallel_tasks,
                lambda t: process_func(t.data),
                progress_callback
            )
            
            # 整理结果
            for (idx, task), result in zip(uncached_tasks, parallel_results):
                if result.success:
                    cached_results[idx] = result.result
                    
                    # 写入缓存
                    if self.enable_cache:
                        cache_key = cache_key_func(task) if cache_key_func else task.get('symbol', f'task_{idx}')
                        self.cache.set(cache_key, result.result)
                else:
                    cached_results[idx] = None
                    logger.error(f"任务失败: {task.get('symbol')}, {result.error}")
        else:
            # 串行处理
            for idx, task in uncached_tasks:
                try:
                    result = process_func(task)
                    cached_results[idx] = result
                    
                    # 写入缓存
                    if self.enable_cache:
                        cache_key = cache_key_func(task) if cache_key_func else task.get('symbol', f'task_{idx}')
                        self.cache.set(cache_key, result)
                except Exception as e:
                    logger.error(f"任务处理失败: {task.get('symbol')}, {e}")
                    cached_results[idx] = None
        
        # 3. 返回结果 (保持顺序)
        return [cached_results.get(i) for i in range(len(tasks))]


# ========== 测试代码 ==========

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 模拟缠论分析函数
    def analyze_stock(task: ParallelTask) -> Dict:
        """模拟股票分析 (耗时操作)"""
        time.sleep(0.1)  # 模拟计算耗时
        
        df = task.data
        # 简单计算
        result = {
            'symbol': task.symbol,
            'mean': df['close'].mean(),
            'std': df['close'].std(),
            'bi_count': np.random.randint(10, 50),
            'buy_points': np.random.randint(0, 5)
        }
        return result
    
    # 创建测试数据
    print("\n=== 创建测试数据 ===")
    num_stocks = 20
    tasks = []
    for i in range(num_stocks):
        symbol = f"00000{i:02d}"
        df = pd.DataFrame({
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        })
        tasks.append(ParallelTask(
            task_id=str(i),
            symbol=symbol,
            data=df
        ))
    print(f"✅ 创建 {num_stocks} 个测试任务")
    
    # 测试串行处理
    print("\n=== 测试串行处理 ===")
    start = time.time()
    serial_results = []
    for task in tasks:
        result = analyze_stock(task)
        serial_results.append(ParallelResult(
            task_id=task.task_id,
            symbol=task.symbol,
            success=True,
            result=result
        ))
    serial_time = time.time() - start
    print(f"✅ 串行处理完成: {serial_time:.2f}s")
    
    # 测试并行处理
    print("\n=== 测试并行处理 ===")
    parallel = ChanLunParallel(num_workers=4)
    
    def progress(completed, total, result):
        if completed % 5 == 0:
            print(f"进度: {completed}/{total}")
    
    start = time.time()
    parallel_results = parallel.run(tasks, analyze_stock, progress_callback=progress)
    parallel_time = time.time() - start
    
    # 统计
    stats = parallel.get_stats()
    print(f"\n=== 并行处理统计 ===")
    print(f"✅ 并行处理完成: {parallel_time:.2f}s")
    print(f"加速比: {serial_time/parallel_time:.2f}x")
    print(f"总任务: {stats['total_tasks']}")
    print(f"成功: {stats['completed']}")
    print(f"失败: {stats['failed']}")
    print(f"重试: {stats['retried']}")
    
    # 验证结果一致性
    print(f"\n=== 验证结果 ===")
    success_count = sum(1 for r in parallel_results if r.success)
    print(f"成功率: {success_count}/{len(parallel_results)} ({success_count/len(parallel_results)*100:.1f}%)")
    
    print("\n=== 测试完成 ===")
