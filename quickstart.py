#!/usr/bin/env python
"""
麒麟量化系统 - 快速开始示例
演示系统的基本功能
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from app.core.trading_context import ContextManager
from app.agents.trading_agents_impl import IntegratedDecisionAgent


async def quick_demo():
    """快速演示系统功能"""
    
    print("=" * 80)
    print("麒麟量化系统 - 快速演示")
    print("=" * 80)
    
    # 1. 创建上下文管理器
    print("\n1. 初始化系统...")
    current_time = datetime.now()
    manager = ContextManager(current_time)
    
    # 2. 设置股票池
    symbols = ['000001', '000002', '300750']
    print(f"2. 股票池: {symbols}")
    
    # 3. 加载数据
    print("3. 加载市场数据...")
    contexts = manager.load_all_data(symbols)
    
    # 4. 创建决策Agent
    print("4. 初始化智能决策系统...")
    agent = IntegratedDecisionAgent()
    
    # 5. 分析股票
    print("5. 开始分析...\n")
    print("-" * 80)
    
    results = []
    for symbol in symbols:
        ctx = contexts[symbol]
        
        # 构建市场上下文（简化版）
        from app.agents.trading_agents_impl import MarketContext
        
        market_ctx = MarketContext(
            ohlcv=pd.DataFrame({
                'close': [10, 10.5, 11, 11.5, 12],
                'volume': [1000000, 1200000, 1500000, 1800000, 2000000],
                'turnover_rate': [5, 6, 8, 10, 12]
            }),
            news_titles=[f"{symbol}获得大订单", f"机构看好{symbol}"],
            lhb_netbuy=2.5,
            market_mood_score=65,
            sector_heat={'sector_change': 3.5, f'{symbol}_rank': 5},
            money_flow={f'{symbol}_main': 1.8, f'{symbol}_super_ratio': 0.15},
            technical_indicators={
                'rsi': 65,
                'volatility': 0.025,
                'seal_ratio': 0.08,
                'zt_time': '10:30',
                'open_times': 1,
                'consecutive_limit': 2
            },
            fundamental_data={
                'financial_score': 75,
                'regulatory_risk': 'low'
            }
        )
        
        # 运行分析
        result = await agent.analyze_parallel(symbol, market_ctx)
        results.append(result)
        
        # 显示结果
        print(f"\n股票: {symbol}")
        print(f"综合得分: {result['weighted_score']:.2f}")
        print(f"决策建议: {result['decision']['action']}")
        print(f"置信度: {result['decision']['confidence']:.2%}")
        print(f"建议仓位: {result['decision']['position']}")
        print(f"理由: {result['decision']['reason']}")
        print(f"风险等级: {result['decision']['risk_level']}")
        
        # 显示各Agent得分
        print(f"\n各Agent评分:")
        for agent_name, agent_data in result['details'].items():
            print(f"  - {agent_name}: {agent_data['score']:.1f}分")
    
    print("-" * 80)
    
    # 6. 排序和推荐
    print("\n6. 投资建议:")
    results.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    for i, result in enumerate(results[:3], 1):
        if result['decision']['action'] in ['buy', 'strong_buy']:
            print(f"  {i}. {result['symbol']}: {result['decision']['action']} "
                  f"(得分:{result['weighted_score']:.2f})")
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


async def test_parallel_performance():
    """测试并行性能"""
    
    print("\n" + "=" * 80)
    print("性能测试 - 并行分析")
    print("=" * 80)
    
    # 创建50只股票
    symbols = [f'{i:06d}' for i in range(1, 51)]
    print(f"测试股票数量: {len(symbols)}")
    
    # 创建模拟数据
    from app.agents.trading_agents_impl import MarketContext
    
    context = MarketContext(
        ohlcv=pd.DataFrame({
            'close': [10, 11, 12],
            'volume': [1000000, 1500000, 2000000],
            'turnover_rate': [5, 8, 10]
        }),
        news_titles=[],
        lhb_netbuy=0,
        market_mood_score=50,
        sector_heat={},
        money_flow={},
        technical_indicators={},
        fundamental_data={}
    )
    
    agent = IntegratedDecisionAgent()
    
    # 开始计时
    start_time = datetime.now()
    
    # 并行分析
    tasks = [agent.analyze_parallel(symbol, context) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    
    # 计算耗时
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n分析完成:")
    print(f"  - 股票数量: {len(results)}")
    print(f"  - 总耗时: {elapsed:.2f}秒")
    print(f"  - 平均每股: {elapsed/len(results)*1000:.0f}毫秒")
    print(f"  - 预计1000股耗时: {elapsed/len(results)*1000:.1f}秒")
    
    print("=" * 80)


def show_menu():
    """显示菜单"""
    print("\n麒麟量化系统 - 功能演示")
    print("=" * 40)
    print("1. 快速演示")
    print("2. 性能测试") 
    print("3. 退出")
    print("=" * 40)
    return input("请选择 (1-3): ")


async def main():
    """主函数"""
    
    print("""
    ╔═══════════════════════════════════════════╗
    ║        麒麟量化系统 - 快速开始            ║
    ║        Qilin Trading System               ║
    ╚═══════════════════════════════════════════╝
    """)
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            await quick_demo()
        elif choice == '2':
            await test_parallel_performance()
        elif choice == '3':
            print("\n感谢使用麒麟量化系统！")
            break
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    asyncio.run(main())