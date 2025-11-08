"""
RD-Agent 集成示例代码

本文件提供麒麟项目中 RD-Agent 集成的完整使用示例,包括:
- 配置加载
- 因子发现
- 会话恢复 (P0-1)
- 数据接口使用 (P0-5)
- 代码沙盒 (安全执行)

版本: v1.0
维护者: 麒麟量化团队
"""

import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path


# ============================================================================
# 示例 1: 基础配置加载
# ============================================================================

def example_basic_config():
    """示例 1: 加载和验证配置"""
    from rd_agent.config import RDAgentConfig
    
    print("=" * 60)
    print("示例 1: 基础配置加载")
    print("=" * 60)
    
    # 方式 1: 从 YAML 文件加载
    config = RDAgentConfig.from_yaml("config/rdagent_limitup.yaml")
    
    # 验证配置
    if config.validate():
        print("✅ 配置验证通过")
        print(f"  - LLM 模型: {config.llm_model}")
        print(f"  - 研究模式: {config.research_mode}")
        print(f"  - 最大迭代次数: {config.max_iterations}")
    else:
        print("❌ 配置验证失败")
        return
    
    # 方式 2: 直接创建配置对象
    custom_config = RDAgentConfig(
        llm_model="gpt-4-turbo",
        max_iterations=20,
        checkpoint_path="./checkpoints/factor.pkl"
    )
    
    print(f"\n自定义配置:")
    print(f"  - LLM 模型: {custom_config.llm_model}")
    print(f"  - Checkpoint 路径: {custom_config.checkpoint_path}")
    
    # 转换为字典
    config_dict = custom_config.to_dict()
    print(f"\n配置字段数量: {len(config_dict)}")


# ============================================================================
# 示例 2: P0-1 会话恢复
# ============================================================================

def example_session_recovery():
    """示例 2: P0-1 会话恢复功能"""
    from rd_agent.official_integration import OfficialRDAgentManager
    
    print("\n" + "=" * 60)
    print("示例 2: P0-1 会话恢复")
    print("=" * 60)
    
    # 配置 checkpoint
    config = {
        "llm_model": "gpt-4-turbo",
        "max_iterations": 20,
        "checkpoint_path": "./checkpoints/factor_loop.pkl",
        "enable_auto_checkpoint": True,
        "checkpoint_interval": 5
    }
    
    manager = OfficialRDAgentManager(config)
    print("✅ OfficialRDAgentManager 初始化完成")
    
    # 场景 1: 首次创建 Loop
    print("\n场景 1: 首次创建因子研发循环")
    try:
        factor_loop = manager.get_factor_loop(resume=False)
        print("✅ 因子研发循环创建成功")
        print(f"  - 类型: {type(factor_loop).__name__}")
    except Exception as e:
        print(f"⚠️ 创建失败 (可能缺少 rdagent 依赖): {e}")
    
    # 场景 2: 从 checkpoint 恢复
    print("\n场景 2: 从 checkpoint 恢复")
    checkpoint_path = Path("./checkpoints/factor_loop.pkl")
    
    if checkpoint_path.exists():
        try:
            factor_loop = manager.resume_from_checkpoint(mode="factor")
            print("✅ 从 checkpoint 恢复成功")
            print(f"  - Checkpoint 路径: {checkpoint_path}")
        except Exception as e:
            print(f"❌ 恢复失败: {e}")
    else:
        print(f"⚠️ Checkpoint 文件不存在: {checkpoint_path}")
        print("  请先运行一次因子发现任务生成 checkpoint")
    
    # 场景 3: 指定 checkpoint 路径恢复
    print("\n场景 3: 指定 checkpoint 路径恢复")
    custom_checkpoint = "./checkpoints/iter_10.pkl"
    print(f"  - 自定义路径: {custom_checkpoint}")
    print("  (演示用,实际需要文件存在)")


# ============================================================================
# 示例 3: 涨停板因子发现
# ============================================================================

async def example_factor_discovery():
    """示例 3: 涨停板因子发现 (P0-3)"""
    from rd_agent.limitup_integration import LimitUpRDAgentIntegration
    
    print("\n" + "=" * 60)
    print("示例 3: 涨停板因子发现 (P0-3)")
    print("=" * 60)
    
    # 初始化集成
    integration = LimitUpRDAgentIntegration()
    print("✅ LimitUpRDAgentIntegration 初始化完成")
    
    # 因子发现 (演示用,实际需要 LLM API)
    print("\n开始因子发现...")
    print("  - 日期范围: 2024-01-01 ~ 2024-01-31")
    print("  - 因子数量: 10")
    print("  ⚠️ 注意: 实际运行需要配置 LLM API 密钥")
    
    try:
        factors = await integration.discover_limit_up_factors(
            start_date="2024-01-01",
            end_date="2024-01-31",
            n_factors=10
        )
        
        print(f"\n✅ 发现 {len(factors)} 个因子\n")
        
        # 显示因子信息
        for i, factor in enumerate(factors, 1):
            print(f"因子 {i}: {factor['name']}")
            print(f"  - 类别: {factor['category']}")
            
            perf = factor['performance']
            print(f"  - IC: {perf['ic']:.4f}")
            print(f"  - IR: {perf['ir']:.4f}")
            print(f"  - 次日涨停率: {perf.get('next_day_limit_up_rate', 0):.2%}")
            print(f"  - 样本数: {perf['sample_count']}")
            print()
    
    except Exception as e:
        print(f"⚠️ 因子发现失败 (可能缺少 LLM API 或数据): {e}")
        print("  这是演示代码,实际使用需要:")
        print("  1. 配置 LLM API 密钥 (RDAGENT_LLM_API_KEY)")
        print("  2. 准备 Qlib 数据")
        print("  3. 配置涨停板数据源")


# ============================================================================
# 示例 4: P0-5 数据接口使用
# ============================================================================

def example_data_interface():
    """示例 4: P0-5 数据接口使用"""
    from rd_agent.limit_up_data import LimitUpDataInterface
    
    print("\n" + "=" * 60)
    print("示例 4: P0-5 数据接口使用")
    print("=" * 60)
    
    # 初始化数据接口
    data_interface = LimitUpDataInterface(data_source="qlib")
    print("✅ LimitUpDataInterface 初始化完成")
    
    # 测试股票和日期
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    date = "2024-01-15"
    
    print(f"\n测试股票: {', '.join(symbols)}")
    print(f"测试日期: {date}")
    print("⚠️ 注意: 演示代码,实际运行需要 Qlib 数据")
    
    # P0-5 新增字段演示
    print("\n--- P0-5 新增字段 ---")
    
    try:
        # 1. 封单金额
        print("\n1. 封单金额 (get_seal_amount)")
        for symbol in symbols[:1]:  # 演示第一个
            seal_amount = data_interface.get_seal_amount(
                symbol=symbol,
                date=date,
                prev_close=10.0
            )
            print(f"  {symbol}: {seal_amount:.2f} 万元")
    
    except Exception as e:
        print(f"  ⚠️ 计算失败 (缺少数据): {e}")
    
    try:
        # 2. 连续涨停天数
        print("\n2. 连续涨停天数 (get_continuous_board)")
        for symbol in symbols[:1]:
            continuous_days = data_interface.get_continuous_board(
                symbol=symbol,
                date=date
            )
            
            if continuous_days == 1:
                board_label = "首板"
            elif continuous_days == 2:
                board_label = "二板"
            elif continuous_days >= 3:
                board_label = f"{continuous_days}连板"
            else:
                board_label = "未涨停"
            
            print(f"  {symbol}: {board_label}")
    
    except Exception as e:
        print(f"  ⚠️ 计算失败 (缺少数据): {e}")
    
    try:
        # 3. 题材热度
        print("\n3. 题材热度 (get_concept_heat)")
        for symbol in symbols[:1]:
            concept_heat = data_interface.get_concept_heat(
                symbol=symbol,
                date=date
            )
            print(f"  {symbol}: 热度 {concept_heat:.0f} (同题材涨停数)")
    
    except Exception as e:
        print(f"  ⚠️ 计算失败 (缺少数据): {e}")
    
    try:
        # 4. 获取完整特征
        print("\n4. 完整特征矩阵 (get_limit_up_features)")
        features = data_interface.get_limit_up_features(
            symbols=symbols,
            date=date,
            lookback_days=20
        )
        
        print(f"  特征矩阵形状: {features.shape}")
        print(f"  特征列: {list(features.columns)}")
        
        # 显示 P0-5 新增字段
        p0_5_fields = ['seal_amount', 'continuous_board', 'concept_heat']
        if all(f in features.columns for f in p0_5_fields):
            print(f"\n  P0-5 新增字段示例:")
            print(features[p0_5_fields].head())
    
    except Exception as e:
        print(f"  ⚠️ 获取失败 (缺少数据): {e}")


# ============================================================================
# 示例 5: 代码沙盒安全执行
# ============================================================================

def example_code_sandbox():
    """示例 5: 代码沙盒安全执行"""
    from rd_agent.code_sandbox import execute_safe, CodeSandbox, SecurityLevel
    
    print("\n" + "=" * 60)
    print("示例 5: 代码沙盒安全执行")
    print("=" * 60)
    
    # 准备测试数据
    test_df = pd.DataFrame({
        'close': [10.0, 11.0, 12.0, 11.5, 13.0],
        'volume': [1000, 1100, 1200, 1150, 1300],
        'open': [9.8, 10.5, 11.2, 11.8, 12.0],
    })
    
    print("测试数据:")
    print(test_df)
    
    # 场景 1: 安全代码执行
    print("\n场景 1: 安全代码 (因子计算)")
    safe_code = """
# 计算简单因子
factor = df['close'] / df['volume']
result = factor.mean()
"""
    
    result = execute_safe(
        code=safe_code,
        context={'df': test_df},
        timeout=5
    )
    
    if result.success:
        print("✅ 执行成功")
        print(f"  - 因子均值: {result.locals['result']:.6f}")
        print(f"  - 因子值: {result.locals['factor'].tolist()}")
    else:
        print(f"❌ 执行失败: {result.error}")
    
    # 场景 2: 危险代码检测 (导入 os)
    print("\n场景 2: 危险代码检测 (import os)")
    dangerous_code_1 = """
import os
os.system('ls')
"""
    
    result = execute_safe(
        code=dangerous_code_1,
        context={},
        timeout=5
    )
    
    if result.success:
        print("⚠️ 危险代码未被拦截!")
    else:
        print(f"✅ 安全拦截: {result.error}")
    
    # 场景 3: 危险函数检测 (exec)
    print("\n场景 3: 危险函数检测 (exec)")
    dangerous_code_2 = """
exec('print("dangerous")')
"""
    
    result = execute_safe(
        code=dangerous_code_2,
        context={},
        timeout=5
    )
    
    if result.success:
        print("⚠️ 危险代码未被拦截!")
    else:
        print(f"✅ 安全拦截: {result.error}")
    
    # 场景 4: 不同安全级别
    print("\n场景 4: 不同安全级别")
    
    code = """
import numpy as np
result = np.mean([1, 2, 3])
"""
    
    # STRICT 模式
    sandbox_strict = CodeSandbox(
        security_level=SecurityLevel.STRICT,
        timeout=5
    )
    result = sandbox_strict.execute(code, context={})
    print(f"  STRICT 模式: {'✅ 通过' if result.success else f'❌ 拦截: {result.error}'}")
    
    # MODERATE 模式
    sandbox_moderate = CodeSandbox(
        security_level=SecurityLevel.MODERATE,
        timeout=5
    )
    result = sandbox_moderate.execute(code, context={}, allowed_modules=['numpy'])
    print(f"  MODERATE 模式 (允许 numpy): {'✅ 通过' if result.success else f'❌ 拦截: {result.error}'}")
    
    # 场景 5: 超时控制 (仅 Unix/Linux/macOS)
    print("\n场景 5: 超时控制")
    import platform
    
    if platform.system() != 'Windows':
        timeout_code = """
import time
time.sleep(10)  # 模拟长时间运行
"""
        
        result = execute_safe(
            code=timeout_code,
            context={},
            timeout=2  # 2秒超时
        )
        
        if result.success:
            print("⚠️ 超时未生效")
        else:
            print(f"✅ 超时拦截: {result.error}")
    else:
        print("  ⚠️ Windows 不支持超时控制 (需要替代方案)")


# ============================================================================
# 示例 6: P0-6 扩展字段配置
# ============================================================================

def example_extended_fields():
    """示例 6: P0-6 扩展字段配置"""
    from rd_agent.config import RDAgentConfig
    
    print("\n" + "=" * 60)
    print("示例 6: P0-6 扩展字段配置")
    print("=" * 60)
    
    # 加载配置
    config = RDAgentConfig.from_yaml("config/rdagent_limitup.yaml")
    
    # 显示 P0-6 扩展字段
    print("\nP0-6 因子类别 (factor_categories):")
    for i, category in enumerate(config.factor_categories, 1):
        print(f"  {i}. {category}")
    
    print("\nP0-6 预测目标 (prediction_targets):")
    for i, target in enumerate(config.prediction_targets, 1):
        print(f"  {i}. {target}")
    
    # 自定义扩展字段
    print("\n自定义扩展字段示例:")
    custom_config = RDAgentConfig(
        factor_categories=["limit_up", "sentiment", "momentum"],
        prediction_targets=["next_day_limit_up"]
    )
    
    print(f"  因子类别: {custom_config.factor_categories}")
    print(f"  预测目标: {custom_config.prediction_targets}")
    
    print("\n说明:")
    print("  - factor_categories 指导 LLM 生成特定类别的因子")
    print("  - prediction_targets 定义因子评估时计算的指标")
    print("  - 列表顺序决定优先级 (越靠前优先级越高)")


# ============================================================================
# 示例 7: 完整因子发现流程
# ============================================================================

async def example_complete_workflow():
    """示例 7: 完整因子发现流程"""
    from rd_agent.config import RDAgentConfig
    from rd_agent.limitup_integration import LimitUpRDAgentIntegration
    
    print("\n" + "=" * 60)
    print("示例 7: 完整因子发现流程")
    print("=" * 60)
    
    # Step 1: 加载配置
    print("\nStep 1: 加载配置")
    config = RDAgentConfig.from_yaml("config/rdagent_limitup.yaml")
    
    if not config.validate():
        print("❌ 配置验证失败")
        return
    
    print("✅ 配置验证通过")
    print(f"  - LLM 模型: {config.llm_model}")
    print(f"  - Checkpoint: {config.checkpoint_path}")
    print(f"  - 因子类别: {', '.join(config.factor_categories)}")
    
    # Step 2: 初始化集成
    print("\nStep 2: 初始化 RD-Agent 集成")
    integration = LimitUpRDAgentIntegration()
    print("✅ 集成初始化完成")
    
    # Step 3: 因子发现 (演示)
    print("\nStep 3: 因子发现")
    print("  ⚠️ 演示代码,实际运行需要:")
    print("     - LLM API 密钥")
    print("     - Qlib 数据")
    print("     - 涨停板数据源")
    
    try:
        factors = await integration.discover_limit_up_factors(
            start_date="2024-01-01",
            end_date="2024-01-31",
            n_factors=5
        )
        
        # Step 4: 结果分析
        print("\nStep 4: 结果分析")
        print(f"✅ 发现 {len(factors)} 个因子\n")
        
        # 按 IC 排序
        sorted_factors = sorted(
            factors,
            key=lambda f: f['performance']['ic'],
            reverse=True
        )
        
        for i, factor in enumerate(sorted_factors, 1):
            perf = factor['performance']
            print(f"Top {i}: {factor['name']}")
            print(f"  - IC: {perf['ic']:.4f}")
            print(f"  - IR: {perf['ir']:.4f}")
            print(f"  - 次日涨停率: {perf.get('next_day_limit_up_rate', 0):.2%}")
            print()
    
    except Exception as e:
        print(f"⚠️ 因子发现失败: {e}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print(" " * 15 + "RD-Agent 集成示例")
    print("=" * 60)
    
    # 示例 1: 基础配置加载
    example_basic_config()
    
    # 示例 2: P0-1 会话恢复
    example_session_recovery()
    
    # 示例 3: 涨停板因子发现 (异步)
    print("\n运行异步示例 3...")
    asyncio.run(example_factor_discovery())
    
    # 示例 4: P0-5 数据接口使用
    example_data_interface()
    
    # 示例 5: 代码沙盒安全执行
    example_code_sandbox()
    
    # 示例 6: P0-6 扩展字段配置
    example_extended_fields()
    
    # 示例 7: 完整流程 (异步)
    print("\n运行异步示例 7...")
    asyncio.run(example_complete_workflow())
    
    print("\n" + "=" * 60)
    print(" " * 20 + "示例运行完成")
    print("=" * 60)
    print("\n说明:")
    print("  - 部分示例需要配置 LLM API 密钥和 Qlib 数据才能完整运行")
    print("  - 演示代码展示了 API 的正确用法,实际使用请根据环境调整")
    print("  - 详细文档请参考: docs/API_REFERENCE.md")


if __name__ == "__main__":
    main()
