"""
性能基准测试 - 对比并发优化效果
"""
import asyncio
import time
from typing import List, Dict
from dataclasses import dataclass
import statistics
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from decision_engine.core import DecisionEngine


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    mode: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # 决策数/秒


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, num_runs: int = 10):
        self.num_runs = num_runs
        self.test_symbols = [
            '000001.SZ', '000002.SZ', '600000.SH', '600036.SH',
            '000651.SZ', '600519.SH', '601318.SH', '000858.SZ',
            '002594.SZ', '601398.SH'
        ]
        self.test_date = '2024-06-30'
    
    async def run_benchmark(self) -> Dict[str, BenchmarkResult]:
        """运行完整基准测试"""
        print("=" * 70)
        print("🚀 Qilin Stack 性能基准测试")
        print("=" * 70)
        print(f"\n配置:")
        print(f"  - 测试股票数: {len(self.test_symbols)}")
        print(f"  - 测试轮次: {self.num_runs}")
        print(f"  - 测试日期: {self.test_date}\n")
        
        results = {}
        
        # 1. 串行模式测试
        print("📊 测试1: 串行模式（无优化）")
        print("-" * 70)
        results['sequential'] = await self._benchmark_mode(enable_performance=False)
        self._print_result(results['sequential'])
        
        # 2. 并行模式测试
        print("\n📊 测试2: 并行模式（并发优化）")
        print("-" * 70)
        results['parallel'] = await self._benchmark_mode(enable_performance=True)
        self._print_result(results['parallel'])
        
        # 3. 对比分析
        print("\n📈 性能对比分析")
        print("=" * 70)
        self._compare_results(results['sequential'], results['parallel'])
        
        return results
    
    async def _benchmark_mode(self, enable_performance: bool) -> BenchmarkResult:
        """测试特定模式"""
        mode = "并行" if enable_performance else "串行"
        times = []
        
        for i in range(self.num_runs):
            # 创建新引擎实例
            engine = DecisionEngine(enable_performance=enable_performance)
            
            # 测量时间
            start_time = time.time()
            decisions = await engine.make_decisions(self.test_symbols, self.test_date)
            elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            
            # 显示进度
            print(f"  轮次 {i+1}/{self.num_runs}: {elapsed_time:.3f}秒 ({len(decisions)}个决策)")
        
        # 计算统计信息
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = len(self.test_symbols) / avg_time
        
        return BenchmarkResult(
            mode=mode,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=throughput
        )
    
    def _print_result(self, result: BenchmarkResult):
        """打印单个结果"""
        print(f"\n  模式: {result.mode}")
        print(f"  总时间: {result.total_time:.3f}秒")
        print(f"  平均时间: {result.avg_time:.3f}秒")
        print(f"  最小时间: {result.min_time:.3f}秒")
        print(f"  最大时间: {result.max_time:.3f}秒")
        print(f"  标准差: {result.std_dev:.3f}秒")
        print(f"  吞吐量: {result.throughput:.2f} 决策/秒")
    
    def _compare_results(self, seq: BenchmarkResult, par: BenchmarkResult):
        """对比结果"""
        speedup = seq.avg_time / par.avg_time
        time_saved = seq.avg_time - par.avg_time
        improvement = (1 - par.avg_time / seq.avg_time) * 100
        
        print(f"\n串行模式:")
        print(f"  平均时间: {seq.avg_time:.3f}秒")
        print(f"  吞吐量: {seq.throughput:.2f} 决策/秒")
        
        print(f"\n并行模式:")
        print(f"  平均时间: {par.avg_time:.3f}秒")
        print(f"  吞吐量: {par.throughput:.2f} 决策/秒")
        
        print(f"\n性能提升:")
        print(f"  ⚡ 加速比: {speedup:.2f}x")
        print(f"  ⏱️  节省时间: {time_saved:.3f}秒 ({improvement:.1f}%)")
        print(f"  📊 吞吐量提升: {(par.throughput/seq.throughput-1)*100:.1f}%")
        
        # 判断
        if speedup >= 2.5:
            emoji = "🏆"
            comment = "卓越！并发优化效果显著"
        elif speedup >= 1.5:
            emoji = "✅"
            comment = "良好，并发优化有明显效果"
        elif speedup >= 1.2:
            emoji = "👍"
            comment = "不错，有一定优化效果"
        else:
            emoji = "⚠️"
            comment = "优化效果有限，可能受IO限制"
        
        print(f"\n{emoji} 评价: {comment}")


async def quick_test():
    """快速测试"""
    print("🔬 快速性能测试（3轮）\n")
    
    benchmark = PerformanceBenchmark(num_runs=3)
    await benchmark.run_benchmark()


async def full_test():
    """完整测试"""
    print("🔬 完整性能测试（10轮）\n")
    
    benchmark = PerformanceBenchmark(num_runs=10)
    await benchmark.run_benchmark()


async def stress_test():
    """压力测试"""
    print("💪 压力测试（大量股票）\n")
    
    # 生成100只股票
    symbols = [f"{i:06d}.SZ" for i in range(1, 101)]
    
    print(f"测试{len(symbols)}只股票...\n")
    
    # 串行模式
    engine_seq = DecisionEngine(enable_performance=False)
    start = time.time()
    decisions_seq = await engine_seq.make_decisions(symbols, '2024-06-30')
    time_seq = time.time() - start
    print(f"串行模式: {time_seq:.2f}秒")
    
    # 并行模式
    engine_par = DecisionEngine(enable_performance=True)
    start = time.time()
    decisions_par = await engine_par.make_decisions(symbols, '2024-06-30')
    time_par = time.time() - start
    print(f"并行模式: {time_par:.2f}秒")
    
    # 对比
    speedup = time_seq / time_par
    print(f"\n加速比: {speedup:.2f}x")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'quick':
            asyncio.run(quick_test())
        elif mode == 'full':
            asyncio.run(full_test())
        elif mode == 'stress':
            asyncio.run(stress_test())
        else:
            print("用法: python benchmark.py [quick|full|stress]")
    else:
        # 默认快速测试
        asyncio.run(quick_test())
