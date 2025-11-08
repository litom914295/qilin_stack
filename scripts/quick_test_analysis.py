"""
快速测试分析功能
直接调用适配器进行一次简单分析
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("🧪 快速分析测试")
print("=" * 70)

# 导入适配器
print("\n📦 步骤1: 导入适配器...")
try:
    from tradingagents_integration.tradingagents_cn_plus_adapter import create_tradingagents_cn_plus_adapter
    print("✅ 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 创建适配器
print("\n🔧 步骤2: 创建适配器...")
try:
    adapter = create_tradingagents_cn_plus_adapter()
    status = adapter.get_status()
    
    if not status.get('available'):
        print(f"❌ 适配器不可用")
        if status.get('error'):
            print(f"   错误: {status['error']}")
        sys.exit(1)
    
    print("✅ 适配器创建成功")
    print(f"   模式: {status.get('mode')}")
except Exception as e:
    print(f"❌ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 运行分析
print("\n🔬 步骤3: 运行分析 (000001)...")
print("⏳ 这可能需要1-3分钟，请耐心等待...")
print()

async def run_analysis():
    try:
        result = await adapter.analyze_stock_full(
            symbol="000001",
            date=None
        )
        
        print("\n" + "=" * 70)
        print("✅ 分析完成！")
        print("=" * 70)
        
        consensus = result.get('consensus', {})
        print(f"\n📊 分析结果:")
        print(f"   最终建议: {consensus.get('signal', 'N/A')}")
        print(f"   置信度: {consensus.get('confidence', 0)*100:.1f}%")
        print(f"   参与智能体: {len(result.get('individual_results', []))}个")
        
        # 显示前5个智能体的观点
        print(f"\n👥 智能体观点 (前5个):")
        for idx, agent in enumerate(result.get('individual_results', [])[:5], 1):
            print(f"   {idx}. {agent.get('agent', 'Agent')}: {agent.get('signal', 'HOLD')} ({agent.get('confidence', 0)*100:.1f}%)")
        
        if len(result.get('individual_results', [])) > 5:
            print(f"   ... 还有 {len(result.get('individual_results', [])) - 5} 个智能体")
        
        # 保存报告
        from web.tabs.tradingagents.enhanced_report_generator import create_enhanced_report
        
        print("\n📝 生成增强报告...")
        report = create_enhanced_report("000001", result, "完整")
        
        output_file = Path(__file__).parent.parent / f"test_report_000001.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 报告已保存到: {output_file}")
        
        print("\n" + "=" * 70)
        print("🎉 测试成功!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 运行
try:
    success = asyncio.run(run_analysis())
    sys.exit(0 if success else 1)
except KeyboardInterrupt:
    print("\n\n⏹️  用户中断")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 运行失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
