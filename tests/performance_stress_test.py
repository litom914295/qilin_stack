"""
性能压力测试模块
Performance Stress Testing Module

功能:
1. 100并发订单测试
2. 长时间稳定性测试
3. 内存泄漏检查
4. 异常恢复测试

Author: Qilin Stack Team
Date: 2025-11-07
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import traceback
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from trading.live_trading_system import LiveTradingSystem, TradingSignal
    from trading.broker_adapters import MockBrokerAdapter
    from qlib_enhanced.performance_optimization import FastFactorCalculator, FastBacktester
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 模块导入失败: {e}")
    MODULES_AVAILABLE = False


class PerformanceStressTestSuite:
    """性能压力测试套件"""
    
    def __init__(self):
        """初始化测试套件"""
        self.test_results = []
        self.system = None
        self.process = psutil.Process()
        
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息 (MB)"""
        mem_info = self.process.memory_info()
        return {
            'rss': mem_info.rss / 1024 / 1024,  # 常驻内存
            'vms': mem_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': self.process.memory_percent()  # 内存占用百分比
        }
    
    def record_test_result(self, test_name: str, success: bool, 
                          details: Dict[str, Any], duration: float):
        """记录测试结果"""
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
    
    async def test_concurrent_orders(self, num_orders: int = 100) -> bool:
        """测试1: 并发订单压力测试"""
        test_name = f"并发订单压力测试 ({num_orders}个订单)"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {
            'num_orders': num_orders,
            'results': [],
            'memory_start': self.get_memory_info(),
        }
        
        try:
            # 创建交易系统
            self.system = LiveTradingSystem(broker_config={'broker_name': 'mock'})
            await self.system.start()
            print(f"✅ 交易系统启动成功")
            
            # 生成测试订单
            symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
            tasks = []
            
            print(f"\n📤 发送 {num_orders} 个并发订单...")
            
            for i in range(num_orders):
                signal = TradingSignal(
                    symbol=symbols[i % len(symbols)],
                    action='buy' if i % 2 == 0 else 'sell',
                    quantity=100 * (i % 10 + 1),
                    price=10.0 + (i % 20) * 0.1,
                    signal_id=f'stress_test_{i}'
                )
                tasks.append(self.system.process_signal(signal))
            
            # 并发执行
            concurrent_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            concurrent_duration = time.time() - concurrent_start
            
            # 统计结果
            success_count = 0
            error_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_count += 1
                    details['results'].append({
                        'order_id': i,
                        'success': False,
                        'error': str(result)
                    })
                elif result.get('success'):
                    success_count += 1
                else:
                    error_count += 1
            
            details['success_count'] = success_count
            details['error_count'] = error_count
            details['success_rate'] = success_count / num_orders
            details['concurrent_duration'] = concurrent_duration
            details['throughput'] = num_orders / concurrent_duration  # 订单/秒
            details['avg_latency'] = (concurrent_duration / num_orders) * 1000  # ms
            details['memory_end'] = self.get_memory_info()
            details['memory_increase'] = details['memory_end']['rss'] - details['memory_start']['rss']
            
            print(f"\n✅ 并发测试完成:")
            print(f"  成功订单: {success_count}/{num_orders}")
            print(f"  成功率: {details['success_rate']*100:.1f}%")
            print(f"  总耗时: {concurrent_duration:.2f}秒")
            print(f"  吞吐量: {details['throughput']:.1f} 订单/秒")
            print(f"  平均延迟: {details['avg_latency']:.2f}ms")
            print(f"  内存增长: {details['memory_increase']:.2f}MB")
            
            # 判断成功: 成功率>95% 且平均延迟<500ms
            success = (details['success_rate'] >= 0.95 and 
                      details['avg_latency'] < 500)
            
            # 清理
            await self.system.stop()
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
            if self.system:
                await self.system.stop()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"总耗时: {duration:.2f}秒")
        
        return success
    
    async def test_long_running_stability(self, duration_minutes: int = 5) -> bool:
        """测试2: 长时间稳定性测试"""
        test_name = f"长时间稳定性测试 ({duration_minutes}分钟)"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {
            'duration_minutes': duration_minutes,
            'samples': [],
            'memory_start': self.get_memory_info(),
        }
        
        try:
            # 创建交易系统
            self.system = LiveTradingSystem(broker_config={'broker_name': 'mock'})
            await self.system.start()
            print(f"✅ 交易系统启动成功")
            
            end_time = start_time + duration_minutes * 60
            sample_interval = 10  # 每10秒采样一次
            order_count = 0
            error_count = 0
            
            print(f"\n🕐 开始 {duration_minutes} 分钟稳定性测试...")
            print(f"采样间隔: {sample_interval}秒")
            
            symbols = ['000001.SZ', '000002.SZ', '600000.SH']
            
            while time.time() < end_time:
                sample_start = time.time()
                
                # 发送一批订单
                for i in range(5):  # 每次发送5个订单
                    signal = TradingSignal(
                        symbol=symbols[i % len(symbols)],
                        action='buy' if order_count % 2 == 0 else 'sell',
                        quantity=100,
                        price=10.0,
                        signal_id=f'stability_test_{order_count}'
                    )
                    
                    try:
                        result = await self.system.process_signal(signal)
                        order_count += 1
                        if not result.get('success'):
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                
                # 采样
                sample = {
                    'timestamp': time.time() - start_time,
                    'order_count': order_count,
                    'error_count': error_count,
                    'memory': self.get_memory_info(),
                    'cpu_percent': self.process.cpu_percent()
                }
                details['samples'].append(sample)
                
                # 打印进度
                elapsed = time.time() - start_time
                progress = (elapsed / (duration_minutes * 60)) * 100
                print(f"⏱️  进度: {progress:.1f}% | "
                      f"订单: {order_count} | "
                      f"错误: {error_count} | "
                      f"内存: {sample['memory']['rss']:.1f}MB | "
                      f"CPU: {sample['cpu_percent']:.1f}%")
                
                # 等待到下一个采样点
                sleep_time = sample_interval - (time.time() - sample_start)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            details['total_orders'] = order_count
            details['total_errors'] = error_count
            details['error_rate'] = error_count / order_count if order_count > 0 else 0
            details['memory_end'] = self.get_memory_info()
            details['memory_increase'] = details['memory_end']['rss'] - details['memory_start']['rss']
            
            # 检查内存增长趋势 (线性回归)
            if len(details['samples']) > 2:
                times = np.array([s['timestamp'] for s in details['samples']])
                mems = np.array([s['memory']['rss'] for s in details['samples']])
                
                # 简单线性回归
                coef = np.polyfit(times, mems, 1)
                details['memory_growth_rate'] = coef[0]  # MB/秒
            
            print(f"\n✅ 稳定性测试完成:")
            print(f"  总订单数: {order_count}")
            print(f"  错误数: {error_count}")
            print(f"  错误率: {details['error_rate']*100:.2f}%")
            print(f"  内存增长: {details['memory_increase']:.2f}MB")
            if 'memory_growth_rate' in details:
                print(f"  内存增长率: {details['memory_growth_rate']:.4f}MB/秒")
            
            # 判断成功: 错误率<5% 且内存增长率<0.1MB/秒 (可能有内存泄漏)
            success = (details['error_rate'] < 0.05 and 
                      details.get('memory_growth_rate', 0) < 0.1)
            
            # 清理
            await self.system.stop()
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
            if self.system:
                await self.system.stop()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"总耗时: {duration:.2f}秒")
        
        return success
    
    async def test_memory_leak_detection(self) -> bool:
        """测试3: 内存泄漏检测"""
        test_name = "内存泄漏检测"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {'iterations': []}
        
        try:
            num_iterations = 10
            orders_per_iteration = 50
            
            print(f"📊 执行 {num_iterations} 轮迭代, 每轮 {orders_per_iteration} 个订单")
            
            for iteration in range(num_iterations):
                print(f"\n🔄 迭代 {iteration + 1}/{num_iterations}")
                
                # 记录迭代前内存
                gc.collect()  # 强制垃圾回收
                mem_before = self.get_memory_info()
                
                # 创建交易系统
                system = LiveTradingSystem(broker_config={'broker_name': 'mock'})
                await system.start()
                
                # 发送订单
                for i in range(orders_per_iteration):
                    signal = TradingSignal(
                        symbol='000001.SZ',
                        action='buy' if i % 2 == 0 else 'sell',
                        quantity=100,
                        price=10.0,
                        signal_id=f'leak_test_{iteration}_{i}'
                    )
                    await system.process_signal(signal)
                
                # 停止系统
                await system.stop()
                del system
                
                # 强制垃圾回收
                gc.collect()
                await asyncio.sleep(0.5)  # 等待清理
                
                # 记录迭代后内存
                mem_after = self.get_memory_info()
                
                iteration_data = {
                    'iteration': iteration,
                    'mem_before': mem_before['rss'],
                    'mem_after': mem_after['rss'],
                    'mem_increase': mem_after['rss'] - mem_before['rss']
                }
                details['iterations'].append(iteration_data)
                
                print(f"  内存: {mem_before['rss']:.1f}MB → {mem_after['rss']:.1f}MB "
                      f"(+{iteration_data['mem_increase']:.2f}MB)")
            
            # 分析内存泄漏趋势
            mem_increases = [it['mem_increase'] for it in details['iterations']]
            details['avg_increase'] = np.mean(mem_increases)
            details['std_increase'] = np.std(mem_increases)
            details['max_increase'] = np.max(mem_increases)
            details['total_increase'] = sum(mem_increases)
            
            print(f"\n📊 内存泄漏分析:")
            print(f"  平均增长: {details['avg_increase']:.2f}MB/轮")
            print(f"  标准差: {details['std_increase']:.2f}MB")
            print(f"  最大增长: {details['max_increase']:.2f}MB")
            print(f"  总增长: {details['total_increase']:.2f}MB")
            
            # 判断成功: 平均增长<5MB/轮 (相对宽松的标准)
            success = details['avg_increase'] < 5.0
            
            if success:
                print(f"✅ 未检测到显著内存泄漏")
            else:
                print(f"⚠️ 检测到可能的内存泄漏")
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"总耗时: {duration:.2f}秒")
        
        return success
    
    async def test_exception_recovery(self) -> bool:
        """测试4: 异常恢复测试"""
        test_name = "异常恢复测试"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {'scenarios': []}
        
        try:
            # 场景1: 系统重启后恢复
            print(f"\n📋 场景1: 系统重启恢复")
            
            system = LiveTradingSystem(broker_config={'broker_name': 'mock'})
            await system.start()
            
            # 发送订单
            signal = TradingSignal('000001.SZ', 'buy', 100, 10.0, 'recovery_test_1')
            result1 = await system.process_signal(signal)
            
            # 停止并重启
            await system.stop()
            await asyncio.sleep(0.5)
            await system.start()
            
            # 再次发送订单
            signal = TradingSignal('000001.SZ', 'sell', 100, 10.2, 'recovery_test_2')
            result2 = await system.process_signal(signal)
            
            await system.stop()
            
            scenario1 = {
                'name': '系统重启恢复',
                'success': result1['success'] and result2['success']
            }
            details['scenarios'].append(scenario1)
            print(f"  {'✅' if scenario1['success'] else '❌'} {scenario1['name']}")
            
            # 场景2: 异常订单处理
            print(f"\n📋 场景2: 异常订单恢复")
            
            system = LiveTradingSystem(broker_config={'broker_name': 'mock'})
            await system.start()
            
            # 发送异常订单
            signal = TradingSignal('000001.SZ', 'buy', 1000000, 10.0, 'recovery_test_3')
            result3 = await system.process_signal(signal)  # 应该被拒绝
            
            # 发送正常订单 (验证系统仍然正常)
            signal = TradingSignal('000001.SZ', 'buy', 100, 10.0, 'recovery_test_4')
            result4 = await system.process_signal(signal)
            
            await system.stop()
            
            scenario2 = {
                'name': '异常订单恢复',
                'success': not result3['success'] and result4['success']  # 异常订单被拒绝,正常订单成功
            }
            details['scenarios'].append(scenario2)
            print(f"  {'✅' if scenario2['success'] else '❌'} {scenario2['name']}")
            
            # 场景3: 并发异常恢复
            print(f"\n📋 场景3: 并发异常恢复")
            
            system = LiveTradingSystem(broker_config={'broker_name': 'mock'})
            await system.start()
            
            # 混合正常和异常订单
            signals = []
            for i in range(20):
                if i % 5 == 0:
                    # 异常订单 (超大数量)
                    signals.append(TradingSignal('000001.SZ', 'buy', 1000000, 10.0, f'mixed_{i}'))
                else:
                    # 正常订单
                    signals.append(TradingSignal('000001.SZ', 'buy', 100, 10.0, f'mixed_{i}'))
            
            tasks = [system.process_signal(s) for s in signals]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计: 正常订单应该成功,异常订单应该被拒绝
            normal_success = sum(1 for i, r in enumerate(results) 
                               if i % 5 != 0 and not isinstance(r, Exception) and r.get('success'))
            abnormal_rejected = sum(1 for i, r in enumerate(results) 
                                  if i % 5 == 0 and not isinstance(r, Exception) and not r.get('success'))
            
            await system.stop()
            
            scenario3 = {
                'name': '并发异常恢复',
                'normal_success': normal_success,
                'abnormal_rejected': abnormal_rejected,
                'success': normal_success == 16 and abnormal_rejected == 4  # 16正常+4异常
            }
            details['scenarios'].append(scenario3)
            print(f"  {'✅' if scenario3['success'] else '❌'} {scenario3['name']}")
            print(f"    正常订单成功: {normal_success}/16")
            print(f"    异常订单拒绝: {abnormal_rejected}/4")
            
            # 判断整体成功
            success = all(s['success'] for s in details['scenarios'])
            
            print(f"\n📊 异常恢复测试总结:")
            for s in details['scenarios']:
                print(f"  {'✅' if s['success'] else '❌'} {s['name']}")
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"总耗时: {duration:.2f}秒")
        
        return success
    
    def test_factor_calculation_performance(self) -> bool:
        """测试5: 因子计算性能测试"""
        test_name = "因子计算性能测试"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {}
        
        try:
            calculator = FastFactorCalculator()
            
            # 生成测试数据
            data_sizes = [1000, 5000, 10000, 50000]
            
            for size in data_sizes:
                print(f"\n📊 测试数据量: {size}条")
                
                prices = np.random.randn(size).cumsum() + 100
                
                # MA计算
                ma_start = time.time()
                ma = calculator.calculate_ma(prices, 20)
                ma_duration = time.time() - ma_start
                ma_throughput = size / ma_duration
                
                # RSI计算
                rsi_start = time.time()
                rsi = calculator.calculate_rsi(prices)
                rsi_duration = time.time() - rsi_start
                rsi_throughput = size / rsi_duration
                
                # MACD计算
                macd_start = time.time()
                macd = calculator.calculate_macd(prices)
                macd_duration = time.time() - macd_start
                macd_throughput = size / macd_duration
                
                print(f"  MA20:  {ma_duration*1000:.2f}ms ({ma_throughput:.0f} 样本/秒)")
                print(f"  RSI:   {rsi_duration*1000:.2f}ms ({rsi_throughput:.0f} 样本/秒)")
                print(f"  MACD:  {macd_duration*1000:.2f}ms ({macd_throughput:.0f} 样本/秒)")
                
                details[f'size_{size}'] = {
                    'ma_throughput': ma_throughput,
                    'rsi_throughput': rsi_throughput,
                    'macd_throughput': macd_throughput
                }
            
            # 判断成功: 10K数据的MA吞吐量>1000样本/秒
            success = details['size_10000']['ma_throughput'] > 1000
            
            print(f"\n✅ 因子计算性能测试完成")
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"总耗时: {duration:.2f}秒")
        
        return success
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print(f"\n{'='*60}")
        print(f"🧪 Qilin Stack 性能压力测试套件")
        print(f"{'='*60}")
        
        # 运行测试
        await self.test_concurrent_orders(num_orders=100)
        await self.test_long_running_stability(duration_minutes=2)  # 缩短到2分钟以便快速测试
        await self.test_memory_leak_detection()
        await self.test_exception_recovery()
        self.test_factor_calculation_performance()
        
        # 生成测试摘要
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_duration': sum(r['duration'] for r in self.test_results),
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


async def main():
    """主函数"""
    test_suite = PerformanceStressTestSuite()
    summary = await test_suite.run_all_tests()
    
    # 打印摘要
    print("\n" + "="*60)
    print("📊 性能压力测试摘要")
    print("="*60)
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过: {summary['passed_tests']} ✅")
    print(f"失败: {summary['failed_tests']} ❌")
    print(f"成功率: {summary['success_rate']*100:.1f}%")
    print(f"总耗时: {summary['total_duration']:.2f}秒")
    print("="*60)
    
    # 保存结果
    output_file = f"performance_stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存到: {output_file}")
    
    return summary


if __name__ == '__main__':
    if not MODULES_AVAILABLE:
        print("❌ 模块不可用,请先安装依赖")
        sys.exit(1)
    
    asyncio.run(main())
