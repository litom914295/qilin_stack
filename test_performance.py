"""
快速验证性能优化集成
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


async def test_basic():
    """基础功能测试"""
    print("🧪 测试1: 基础功能验证\n")
    
    # 测试并发优化
    print("📊 测试并发优化模块...")
    try:
        from performance.concurrency import get_optimizer
        optimizer = get_optimizer()
        
        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2
        
        tasks = [task(i) for i in range(5)]
        results = await optimizer.gather_parallel(*tasks)
        
        print(f"  ✅ 并发优化正常: {results}")
        optimizer.cleanup()
    except Exception as e:
        print(f"  ❌ 并发优化失败: {e}")
        return False
    
    # 测试缓存
    print("\n📊 测试缓存模块...")
    try:
        from performance.cache import get_cache, cached
        cache = get_cache()
        
        cache.set('test_key', 'test_value', ttl=10)
        value = cache.get('test_key')
        
        if value == 'test_value':
            print(f"  ✅ 缓存正常: {value}")
        else:
            print(f"  ⚠️ 缓存值不匹配: {value}")
    except Exception as e:
        print(f"  ❌ 缓存失败: {e}")
        return False
    
    return True


async def test_decision_engine():
    """测试决策引擎集成"""
    print("\n🧪 测试2: 决策引擎集成\n")
    
    try:
        from decision_engine.core import DecisionEngine
        
        # 测试串行模式
        print("📊 测试串行模式...")
        engine_seq = DecisionEngine(enable_performance=False)
        symbols = ['000001.SZ', '000002.SZ']
        
        import time
        start = time.time()
        decisions = await engine_seq.make_decisions(symbols, '2024-06-30')
        time_seq = time.time() - start
        
        print(f"  ✅ 串行模式: {len(decisions)}个决策, 耗时{time_seq:.3f}秒")
        
        # 测试并行模式
        print("\n📊 测试并行模式...")
        engine_par = DecisionEngine(enable_performance=True)
        
        start = time.time()
        decisions = await engine_par.make_decisions(symbols, '2024-06-30')
        time_par = time.time() - start
        
        print(f"  ✅ 并行模式: {len(decisions)}个决策, 耗时{time_par:.3f}秒")
        
        # 对比
        if time_seq > 0:
            speedup = time_seq / time_par
            print(f"\n⚡ 加速比: {speedup:.2f}x")
            
            if speedup >= 1.5:
                print("  🏆 性能优化效果显著！")
            elif speedup >= 1.2:
                print("  ✅ 性能有所提升")
            else:
                print("  ⚠️ 性能提升有限（可能因为任务太少）")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 决策引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("=" * 70)
    print("🚀 Qilin Stack 性能优化集成测试")
    print("=" * 70)
    print()
    
    # 测试基础功能
    basic_ok = await test_basic()
    
    if not basic_ok:
        print("\n❌ 基础功能测试失败，请检查模块安装")
        return
    
    # 测试决策引擎集成
    engine_ok = await test_decision_engine()
    
    print("\n" + "=" * 70)
    if basic_ok and engine_ok:
        print("✅ 所有测试通过！性能优化已成功集成")
        print("\n下一步:")
        print("  1. 运行完整基准测试: python performance/benchmark.py quick")
        print("  2. 查看演示: python performance/demo.py")
        print("  3. 运行压力测试: python performance/benchmark.py stress")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
