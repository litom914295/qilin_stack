"""
涨停板"一进二"策略完整示例
使用RD-Agent自动研究涨停板因子和模型
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rd_agent.limitup_integration import create_limitup_integration
from rd_agent.limit_up_data import LimitUpDataInterface, LimitUpFactorLibrary
import logging

logger = logging.getLogger(__name__)


async def example_factor_discovery():
    """示例1: 涨停板因子发现"""
    logger.info("=" * 60)
    logger.info("示例1: 涨停板因子发现")
    logger.info("=" * 60)
    
    integration = create_limitup_integration()
    
    # 发现因子
    factors = await integration.discover_limit_up_factors(
        start_date="2024-01-01",
        end_date="2024-06-30",
        n_factors=10
    )
    
    logger.info(f"✅ 发现 {len(factors)} 个高质量涨停板因子:")
    
    for i, factor in enumerate(factors, 1):
        logger.info(f"{i}. {factor['name']}")
        logger.info(f"   类别: {factor['category']}")
        logger.info(f"   描述: {factor['description']}")
        logger.info(f"   表达式: {factor['expression']}")
        
        if 'performance' in factor:
            perf = factor['performance']
            logger.info(f"   性能: IC={perf['ic']:.4f}, IR={perf['ir']:.2f}, "
                        f"次日涨停率={perf['next_day_limit_up_rate']:.2%}")


async def example_model_optimization():
    """示例2: 模型优化"""
    logger.info("=" * 60)
    logger.info("示例2: 涨停板预测模型优化")
    logger.info("=" * 60)
    
    integration = create_limitup_integration()
    
    # 先发现因子
    factors = await integration.discover_limit_up_factors(
        start_date="2024-01-01",
        end_date="2024-06-30",
        n_factors=10
    )
    
    # 优化模型
    model_result = await integration.optimize_limit_up_model(
        factors=factors,
        start_date="2024-01-01",
        end_date="2024-06-30"
    )
    
    logger.info("✅ 最优模型配置:")
    logger.info(f"模型类型: {model_result['model_type']}")
    logger.info("参数:")
    for key, value in model_result['parameters'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("性能指标:")
    for key, value in model_result['performance'].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2%}")


async def example_data_interface():
    """示例3: 数据接口使用"""
    logger.info("=" * 60)
    logger.info("示例3: 涨停板数据接口")
    logger.info("=" * 60)
    
    data_interface = LimitUpDataInterface(data_source="qlib")
    
    # 模拟获取涨停股票
    date = "2024-06-15"
    logger.info(f"获取 {date} 涨停股票...")
    
    limit_ups = data_interface.get_limit_up_stocks(
        date=date,
        min_price=2.0,
        max_price=300.0,
        exclude_st=True,
        exclude_new=True
    )
    
    logger.info(f"找到 {len(limit_ups)} 只涨停股票")
    
    # 模拟特征数据
    symbols = [f"000{i:03d}.SZ" for i in range(1, 11)]
    
    logger.info("获取特征数据...")
    features = data_interface.get_limit_up_features(symbols, date)
    
    logger.info("涨停板特征:")
    logger.info("\n" + str(features.head()))
    
    logger.info("获取次日结果...")
    results = data_interface.get_next_day_result(symbols, date)
    
    logger.info("次日表现:")
    logger.info("\n" + str(results.head()))
    
    # 统计
    limit_up_rate = results['next_limit_up'].mean()
    avg_return = results['next_return'].mean()
    
    logger.info("统计:")
    logger.info(f"  次日涨停率: {limit_up_rate:.2%}")
    logger.info(f"  平均收益率: {avg_return:.2%}")


async def example_factor_library():
    """示例4: 因子库使用"""
    logger.info("=" * 60)
    logger.info("示例4: 涨停板因子库")
    logger.info("=" * 60)
    
    # 模拟数据
    data = pd.DataFrame({
        'seal_amount': np.random.uniform(1000, 10000, 10),
        'market_cap': np.random.uniform(50, 500, 10),
        'continuous_board': np.random.randint(1, 5, 10),
        'volume_ratio': np.random.uniform(2, 8, 10),
        'concept_heat': np.random.randint(5, 20, 10),
        'limit_up_strength': np.random.uniform(70, 100, 10),
        'volume': np.random.uniform(10000, 100000, 10),
        'volume_ma20': np.random.uniform(8000, 50000, 10),
        'large_buy': np.random.uniform(5000, 50000, 10),
        'large_sell': np.random.uniform(3000, 45000, 10),
        'amount': np.random.uniform(50000, 500000, 10),
    }, index=[f"00000{i}" for i in range(10)])
    
    logger.info("原始数据:")
    logger.info("\n" + str(data[['seal_amount', 'market_cap', 'continuous_board']].head()))
    
    # 应用因子
    factor_lib = LimitUpFactorLibrary()
    
    logger.info("应用因子:")
    
    seal_strength = factor_lib.factor_seal_strength(data)
    logger.info("1. 封板强度因子:")
    logger.info("\n" + str(seal_strength.head()))
    
    continuous_momentum = factor_lib.factor_continuous_momentum(data)
    logger.info("2. 连板动量因子:")
    logger.info("\n" + str(continuous_momentum.head()))
    
    concept_synergy = factor_lib.factor_concept_synergy(data)
    logger.info("3. 题材共振因子:")
    logger.info("\n" + str(concept_synergy.head()))
    
    # 所有因子
    all_factors = factor_lib.get_all_factors()
    logger.info(f"预定义因子数量: {len(all_factors)}")
    logger.info(f"因子列表: {list(all_factors.keys())}")


async def example_end_to_end():
    """示例5: 端到端流程"""
    logger.info("=" * 60)
    logger.info("示例5: 涨停板研究端到端流程")
    logger.info("=" * 60)
    
    integration = create_limitup_integration()
    
    # 1. 查看系统状态
    logger.info("1️⃣ 系统状态:")
    status = integration.get_status()
    for key, value in status.items():
        logger.info(f"  {key}: {value}")
    
    # 2. 因子发现
    logger.info("2️⃣ 因子发现:")
    factors = await integration.discover_limit_up_factors(
        start_date="2024-01-01",
        end_date="2024-06-30",
        n_factors=5
    )
    logger.info(f"  发现 {len(factors)} 个因子")
    
    # 3. 模型优化
    logger.info("3️⃣ 模型优化:")
    model = await integration.optimize_limit_up_model(
        factors=factors,
        start_date="2024-01-01",
        end_date="2024-06-30"
    )
    logger.info(f"  模型类型: {model['model_type']}")
    logger.info(f"  准确率: {model['performance']['accuracy']:.2%}")
    
    # 4. 策略配置
    logger.info("4️⃣ 策略配置:")
    logger.info("  打板时机: 09:35:00")
    logger.info("  最大仓位: 30%")
    logger.info("  止损: -5%")
    logger.info("  目标利润: +20%")
    
    logger.info("✅ 端到端流程完成!")


async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("RD-Agent 涨停板场景完整示例")
    logger.info("=" * 60)
    
    examples = [
        ("因子发现", example_factor_discovery),
        ("模型优化", example_model_optimization),
        ("数据接口", example_data_interface),
        ("因子库", example_factor_library),
        ("端到端流程", example_end_to_end),
    ]
    
    logger.info("可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        logger.info(f"  {i}. {name}")
    
    logger.info("运行所有示例...")
    
    for name, func in examples:
        try:
            await func()
            await asyncio.sleep(0.5)  # 短暂暂停
        except Exception as e:
            logger.exception(f"❌ {name} 失败: {e}")
    
    logger.info("=" * 60)
    logger.info("所有示例运行完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    from app.core.logging_setup import setup_logging
    setup_logging()
    asyncio.run(main())
