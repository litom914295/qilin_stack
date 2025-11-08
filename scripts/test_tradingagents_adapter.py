"""
TradingAgents-CN-Plus 适配器快速测试
验证适配器是否能正常初始化和运行
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_adapter_import():
    """测试适配器导入"""
    print("=" * 70)
    print("📦 测试1: 导入适配器")
    print("=" * 70)
    
    try:
        from tradingagents_integration.tradingagents_cn_plus_adapter import create_tradingagents_cn_plus_adapter
        print("✅ 适配器导入成功")
        return True, create_tradingagents_cn_plus_adapter
    except Exception as e:
        print(f"❌ 适配器导入失败: {e}")
        return False, None


def test_adapter_creation(create_func):
    """测试适配器创建"""
    print("\n" + "=" * 70)
    print("🔧 测试2: 创建适配器实例")
    print("=" * 70)
    
    try:
        adapter = create_func()
        print("✅ 适配器实例创建成功")
        return True, adapter
    except Exception as e:
        print(f"❌ 适配器实例创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_adapter_status(adapter):
    """测试适配器状态"""
    print("\n" + "=" * 70)
    print("📊 测试3: 检查适配器状态")
    print("=" * 70)
    
    try:
        status = adapter.get_status()
        print(f"状态信息:")
        print(f"  - 可用: {status.get('available')}")
        print(f"  - 模式: {status.get('mode')}")
        print(f"  - 路径: {status.get('project_path')}")
        
        if 'error' in status:
            print(f"  - 错误: {status['error']}")
            return False
        
        if status.get('available'):
            print("✅ 适配器状态正常")
            return True
        else:
            print("❌ 适配器不可用")
            return False
    except Exception as e:
        print(f"❌ 状态检查失败: {e}")
        return False


async def test_simple_analysis(adapter):
    """测试简单分析"""
    print("\n" + "=" * 70)
    print("🔬 测试4: 运行简单分析 (000001)")
    print("=" * 70)
    print("⏳ 这可能需要30秒-2分钟，请耐心等待...")
    
    try:
        result = await adapter.analyze_stock_full(
            symbol="000001",
            date=None
        )
        
        print("\n分析结果:")
        consensus = result.get('consensus', {})
        print(f"  - 最终建议: {consensus.get('signal', 'N/A')}")
        print(f"  - 置信度: {consensus.get('confidence', 0)*100:.1f}%")
        print(f"  - 参与智能体: {len(result.get('individual_results', []))}个")
        
        # 显示各智能体观点
        print("\n  智能体观点:")
        for idx, agent in enumerate(result.get('individual_results', [])[:5], 1):
            agent_name = agent.get('agent', 'Agent')
            signal = agent.get('signal', 'HOLD')
            conf = agent.get('confidence', 0)
            print(f"    {idx}. {agent_name}: {signal} ({conf*100:.1f}%)")
        
        if len(result.get('individual_results', [])) > 5:
            print(f"    ... 还有 {len(result.get('individual_results', [])) - 5} 个智能体")
        
        print("\n✅ 分析测试成功！")
        return True
    except Exception as e:
        print(f"❌ 分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("\n" + "🚀" * 35)
    print("TradingAgents-CN-Plus 适配器测试")
    print("🚀" * 35 + "\n")
    
    # 测试1: 导入
    success, create_func = test_adapter_import()
    if not success:
        print("\n❌ 测试失败: 无法导入适配器")
        return
    
    # 测试2: 创建实例
    success, adapter = test_adapter_creation(create_func)
    if not success:
        print("\n❌ 测试失败: 无法创建适配器实例")
        return
    
    # 测试3: 状态检查
    success = test_adapter_status(adapter)
    if not success:
        print("\n⚠️  适配器状态异常，跳过分析测试")
        print("\n💡 提示:")
        print("   1. 检查依赖是否完整安装")
        print("   2. 检查API密钥是否配置")
        print("   3. 运行: python scripts/check_env.py")
        return
    
    # 测试4: 简单分析
    print("\n是否运行实际分析测试？这会调用LLM API (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            success = asyncio.run(test_simple_analysis(adapter))
            if not success:
                print("\n❌ 分析测试失败")
                return
        else:
            print("⏭️  跳过分析测试")
    except KeyboardInterrupt:
        print("\n⏭️  用户取消")
    
    # 总结
    print("\n" + "=" * 70)
    print("📋 测试总结")
    print("=" * 70)
    print("\n✅ 所有测试通过!")
    print("\n🎉 TradingAgents-CN-Plus 适配器已就绪")
    print("\n📝 下一步:")
    print("   1. 启动 Streamlit 应用: streamlit run web/main.py")
    print("   2. 进入 TradingAgents → 决策分析 tab")
    print("   3. 选择分析深度 '完整'")
    print("   4. 开始深度分析")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
