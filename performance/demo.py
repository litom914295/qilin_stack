"""
性能优化演示 - 展示并发和缓存的效果
"""
import asyncio
import time
import sys
import os
import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from performance.concurrency import get_optimizer, parallel_task
from performance.cache import get_cache, cached


# ============================================================================
# 演示1: 并发优化
# ============================================================================

async def demo_concurrency():
    """演示并发优化"""
    logger.info("=" * 70)
    logger.info("🚀 演示1: 并发优化")
    logger.info("=" * 70)
    
    optimizer = get_optimizer()
    
    # 模拟耗时任务
    async def slow_task(n: int):
        await asyncio.sleep(0.1)  # 模拟IO
        return f"任务{n}完成"
    
    # 测试数据
    num_tasks = 10
    
    # 1. 串行执行
    logger.info(f"📊 串行执行 {num_tasks} 个任务:")
    start = time.time()
    results_seq = []
    for i in range(num_tasks):
        result = await slow_task(i)
        results_seq.append(result)
    time_seq = time.time() - start
    logger.info(f"  耗时: {time_seq:.3f}秒")
    
    # 2. 并行执行
    logger.info(f"📊 并行执行 {num_tasks} 个任务:")
    start = time.time()
    tasks = [slow_task(i) for i in range(num_tasks)]
    results_par = await optimizer.gather_parallel(*tasks)
    time_par = time.time() - start
    logger.info(f"  耗时: {time_par:.3f}秒")
    
    # 对比
    speedup = time_seq / time_par
    logger.info(f"⚡ 加速比: {speedup:.2f}x")
    logger.info(f"⏱️  节省时间: {(time_seq - time_par):.3f}秒")
    
    # 清理
    optimizer.cleanup()


# ============================================================================
# 演示2: 缓存优化
# ============================================================================

async def demo_cache():
    """演示缓存优化"""
    logger.info("=" * 70)
    logger.info("💾 演示2: 缓存优化")
    logger.info("=" * 70)
    
    cache = get_cache()
    
    # 模拟昂贵计算
    call_count = {'count': 0}
    
    async def expensive_calculation(x: int) -> int:
        """模拟昂贵计算"""
        call_count['count'] += 1
        await asyncio.sleep(0.2)  # 模拟计算
        return x * x
    
    # 测试相同输入
    test_value = 42
    
    # 1. 无缓存 - 多次调用
    logger.info("📊 无缓存 - 调用3次:")
    call_count['count'] = 0
    start = time.time()
    for _ in range(3):
        result = await expensive_calculation(test_value)
    time_no_cache = time.time() - start
    logger.info(f"  耗时: {time_no_cache:.3f}秒")
    logger.info(f"  实际计算次数: {call_count['count']}")
    
    # 2. 有缓存 - 多次调用
    logger.info("📊 有缓存 - 调用3次:")
    
    # 使用缓存装饰器
    @cached(ttl=300, key_prefix="expensive")
    async def expensive_calculation_cached(x: int) -> int:
        call_count['count'] += 1
        await asyncio.sleep(0.2)
        return x * x
    
    call_count['count'] = 0
    start = time.time()
    for _ in range(3):
        result = await expensive_calculation_cached(test_value)
    time_with_cache = time.time() - start
    logger.info(f"  耗时: {time_with_cache:.3f}秒")
    logger.info(f"  实际计算次数: {call_count['count']}")
    
    # 对比
    speedup = time_no_cache / time_with_cache
    logger.info(f"⚡ 加速比: {speedup:.2f}x")
    logger.info(f"⏱️  节省时间: {(time_no_cache - time_with_cache):.3f}秒")
    
    # 缓存命中率
    hit_rate = (3 - call_count['count']) / 3 * 100
    logger.info(f"📊 缓存命中率: {hit_rate:.0f}%")


# ============================================================================
# 演示3: 综合优化
# ============================================================================

async def demo_combined():
    """演示并发+缓存组合优化"""
    logger.info("=" * 70)
    logger.info("🎯 演示3: 并发 + 缓存组合优化")
    logger.info("=" * 70)
    
    optimizer = get_optimizer()
    
    # 模拟数据获取
    @cached(ttl=300, key_prefix="data")
    async def fetch_data(symbol: str) -> dict:
        """获取数据（带缓存）"""
        await asyncio.sleep(0.1)  # 模拟网络IO
        return {
            'symbol': symbol,
            'price': 100.0,
            'volume': 10000
        }
    
    symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000001.SZ']
    
    # 1. 串行 + 无缓存
    logger.info("📊 串行模式（无缓存）：")
    # 清空缓存
    from performance.cache import get_cache
    get_cache().clear_all()
    
    start = time.time()
    results = []
    for symbol in symbols:
        result = await fetch_data(symbol)
        results.append(result)
    time_seq_no_cache = time.time() - start
    logger.info(f"  耗时: {time_seq_no_cache:.3f}秒")
    
    # 2. 并行 + 缓存（第二次运行，有缓存）
    logger.info("📊 并行模式（有缓存）：")
    start = time.time()
    tasks = [fetch_data(symbol) for symbol in symbols]
    results = await optimizer.gather_parallel(*tasks)
    time_par_cache = time.time() - start
    logger.info(f"  耗时: {time_par_cache:.3f}秒")
    
    # 对比
    speedup = time_seq_no_cache / time_par_cache
    logger.info(f"⚡ 总加速比: {speedup:.2f}x")
    logger.info(f"⏱️  总节省时间: {(time_seq_no_cache - time_par_cache):.3f}秒")
    
    # 清理
    optimizer.cleanup()


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """运行所有演示"""
    logger.info("🎬 " * 17)
    logger.info("Qilin Stack 性能优化演示")
    logger.info("🎬 " * 17)
    
    await demo_concurrency()
    await demo_cache()
    await demo_combined()
    
    logger.info("=" * 70)
    logger.info("✅ 所有演示完成")
    logger.info("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
