"""
运行测试脚本
正确设置路径并运行测试
"""

import sys
import os
from pathlib import Path

# 添加项目路径到系统路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'app'))

# 打印路径信息
print("="*60)
print("测试环境信息")
print("="*60)
print(f"项目目录: {current_dir}")
print(f"Python路径: {sys.executable}")
print(f"系统路径: {sys.path[:3]}")
print("="*60)

# 检查模块是否存在
modules_to_check = [
    'app/core/advanced_indicators.py',
    'app/core/risk_management.py',
    'app/core/backtest_engine.py',
    'app/core/agent_orchestrator.py',
    'app/core/performance_monitor.py',
    'app/core/trade_executor.py',
    'app/core/trading_context.py'
]

print("\n模块检查:")
all_exist = True
for module in modules_to_check:
    module_path = current_dir / module
    exists = module_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {module}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n⚠️ 警告: 某些模块不存在，测试可能会失败")
else:
    print("\n✓ 所有模块都存在")

print("\n" + "="*60)
print("开始运行测试")
print("="*60 + "\n")

# 尝试运行测试
try:
    # 导入测试模块
    from tests.test_suite import run_all_tests
    
    # 运行测试
    success = run_all_tests()
    
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败")
        
    sys.exit(0 if success else 1)
    
except ImportError as e:
    print(f"\n导入错误: {e}")
    print("\n尝试运行简单测试...")
    
    # 运行简单的导入测试
    try:
        print("\n测试导入核心模块...")
        
        # 测试导入技术指标
        try:
            from app.core.advanced_indicators import TechnicalIndicators
            print("  ✓ advanced_indicators 导入成功")
        except Exception as e:
            print(f"  ✗ advanced_indicators 导入失败: {e}")
        
        # 测试导入风险管理
        try:
            from app.core.risk_management import RiskManager
            print("  ✓ risk_management 导入成功")
        except Exception as e:
            print(f"  ✗ risk_management 导入失败: {e}")
        
        # 测试导入回测引擎
        try:
            from app.core.backtest_engine import BacktestEngine
            print("  ✓ backtest_engine 导入成功")
        except Exception as e:
            print(f"  ✗ backtest_engine 导入失败: {e}")
        
        # 测试导入Agent协调器
        try:
            from app.core.agent_orchestrator import AgentOrchestrator
            print("  ✓ agent_orchestrator 导入成功")
        except Exception as e:
            print(f"  ✗ agent_orchestrator 导入失败: {e}")
        
        # 测试导入性能监控
        try:
            from app.core.performance_monitor import PerformanceMonitor
            print("  ✓ performance_monitor 导入成功")
        except Exception as e:
            print(f"  ✗ performance_monitor 导入失败: {e}")
        
        # 测试导入交易执行
        try:
            from app.core.trade_executor import ExecutionEngine
            print("  ✓ trade_executor 导入成功")
        except Exception as e:
            print(f"  ✗ trade_executor 导入失败: {e}")
        
        # 测试导入交易上下文
        try:
            from app.core.trading_context import TradingContext
            print("  ✓ trading_context 导入成功")
        except Exception as e:
            print(f"  ✗ trading_context 导入失败: {e}")
        
        print("\n基本导入测试完成")
        
    except Exception as e:
        print(f"\n基本测试失败: {e}")
        sys.exit(1)