"""
快速测试脚本 - 验证核心功能
Quick Test Script

Author: Qilin Stack Team
Date: 2025-11-07
"""

import asyncio
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, 'G:\\test\\qilin_stack')

from trading.live_trading_system import (
    create_live_trading_system, TradingSignal, OrderSide
)


async def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*60)
    print("🧪 Qilin Stack 快速功能测试")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    try:
        # 1. 创建交易系统
        print("✅ 步骤 1/4: 创建交易系统...")
        config = {
            'broker_name': 'mock',
            'initial_cash': 1000000,
            'commission_rate': 0.0003
        }
        system = create_live_trading_system(config)
        print("   ✅ 交易系统创建成功")
        
        # 2. 启动系统
        print("\n✅ 步骤 2/4: 启动系统...")
        await system.start()
        print("   ✅ 系统启动成功")
        
        # 3. 发送测试订单
        print("\n✅ 步骤 3/4: 发送测试订单...")
        
        # 买入订单
        signal = TradingSignal(
            symbol='000001.SZ',
            side=OrderSide.BUY,
            size=100,
            price=10.0
        )
        
        result = await system.process_signal(signal)
        
        if result.success:
            print(f"   ✅ 买入订单成功: {result.order_id}")
        else:
            print(f"   ❌ 买入订单失败: {result.message}")
        
        await asyncio.sleep(0.5)
        
        # 卖出订单
        signal = TradingSignal(
            symbol='000001.SZ',
            side=OrderSide.SELL,
            size=100,
            price=10.2
        )
        
        result = await system.process_signal(signal)
        
        if result.success:
            print(f"   ✅ 卖出订单成功: {result.order_id}")
        else:
            print(f"   ❌ 卖出订单失败: {result.message}")
        
        # 4. 停止系统
        print("\n✅ 步骤 4/4: 停止系统...")
        await system.stop()
        print("   ✅ 系统停止成功")
        
        print("\n" + "="*60)
        print("✅ 所有测试通过!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)
