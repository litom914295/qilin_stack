"""
集成测试脚本
测试三个项目的集成是否正常
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("🧪 测试模块导入...")
    print("=" * 60)
    
    try:
        from app.integrations import (
            qlib_integration,
            rdagent_integration,
            tradingagents_integration,
            data_bridge
        )
        print("✅ 所有集成模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_qlib_integration():
    """测试Qlib集成"""
    print("\n" + "=" * 60)
    print("📊 测试Qlib集成...")
    print("=" * 60)
    
    try:
        from app.integrations import qlib_integration
        
        # 检查可用性
        is_available = qlib_integration.is_available()
        print(f"Qlib可用性: {'✅' if is_available else '❌'}")
        
        if is_available:
            print("提示: 可以使用Qlib功能，但需要先下载数据")
        
        return True
    except Exception as e:
        print(f"❌ Qlib集成测试失败: {e}")
        return False


def test_rdagent_integration():
    """测试RD-Agent集成"""
    print("\n" + "=" * 60)
    print("🤖 测试RD-Agent集成...")
    print("=" * 60)
    
    try:
        from app.integrations import rdagent_integration
        
        is_available = rdagent_integration.is_available()
        print(f"RD-Agent可用性: {'✅' if is_available else '❌'}")
        
        if is_available:
            # 测试因子生成
            print("测试自动因子生成...")
            factors = rdagent_integration.auto_generate_factors(
                market_data=None,
                num_factors=3,
                iterations=1
            )
            print(f"✅ 成功生成{len(factors)}个因子")
            
            for factor in factors:
                print(f"  - {factor['name']}: IC={factor['ic']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ RD-Agent集成测试失败: {e}")
        return False


def test_tradingagents_integration():
    """测试TradingAgents集成"""
    print("\n" + "=" * 60)
    print("👥 测试TradingAgents集成...")
    print("=" * 60)
    
    try:
        from app.integrations import tradingagents_integration
        
        is_available = tradingagents_integration.is_available()
        print(f"TradingAgents可用性: {'✅' if is_available else '❌'}")
        
        if is_available:
            # 测试会员管理
            print("测试会员管理...")
            success = tradingagents_integration.add_member(
                'test_001', '测试用户', 100
            )
            print(f"{'✅' if success else '❌'} 添加会员")
            
            # 测试单股分析
            print("测试单股分析...")
            result = tradingagents_integration.analyze_stock(
                stock_code='000001',
                analysis_depth=3,
                market='cn'
            )
            print(f"✅ 分析完成: {result['final_decision']['action']}")
        
        return True
    except Exception as e:
        print(f"❌ TradingAgents集成测试失败: {e}")
        return False


def test_data_bridge():
    """测试数据共享桥接"""
    print("\n" + "=" * 60)
    print("🌉 测试数据共享桥接...")
    print("=" * 60)
    
    try:
        from app.integrations.data_bridge import data_bridge
        
        # 测试因子保存
        print("测试因子保存...")
        success = data_bridge.save_factor(
            factor_name='test_factor',
            factor_data={'formula': '(close - mean(close, 5))'},
            source='test'
        )
        print(f"{'✅' if success else '❌'} 保存因子")
        
        # 测试因子加载
        print("测试因子加载...")
        factor = data_bridge.load_factor('test_factor')
        if factor:
            print(f"✅ 加载因子: {factor['name']}")
        else:
            print("❌ 加载因子失败")
        
        # 列出因子
        factors = data_bridge.list_factors()
        print(f"✅ 共有{len(factors)}个因子")
        
        return True
    except Exception as e:
        print(f"❌ 数据桥接测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n")
    print("🦄" * 20)
    print("麒麟量化统一平台 - 集成测试")
    print("🦄" * 20)
    
    results = []
    
    # 运行所有测试
    results.append(("模块导入", test_imports()))
    results.append(("Qlib集成", test_qlib_integration()))
    results.append(("RD-Agent集成", test_rdagent_integration()))
    results.append(("TradingAgents集成", test_tradingagents_integration()))
    results.append(("数据桥接", test_data_bridge()))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} {status}")
    
    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        print("\n下一步:")
        print("  1. 运行 'python run_unified_dashboard.py' 启动Web界面")
        print("  2. 浏览器访问 http://localhost:8501")
    else:
        print("⚠️ 部分测试失败，请检查配置")
        print("\n提示:")
        print("  - 确保三个项目已正确安装")
        print("  - 检查路径配置是否正确")
        print("  - 查看详细错误信息")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
