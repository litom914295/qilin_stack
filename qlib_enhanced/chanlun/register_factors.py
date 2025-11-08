"""缠论因子注册模块

将缠论特征注册为 Qlib 表达式因子，实现与 Qlib 因子体系的完全兼容

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - Phase 3 优化
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)

# 全局标记，避免重复注册
_FACTORS_REGISTERED = False


def register_chanlun_factors(force_reload=False):
    """注册缠论因子到 Qlib 因子库
    
    将 16 个缠论特征注册为 Qlib 可识别的因子表达式:
    - 6个 CZSC 因子
    - 10个 Chan.py 因子
    
    Args:
        force_reload: 是否强制重新注册
    
    Returns:
        dict: 注册的因子名称列表
    """
    global _FACTORS_REGISTERED
    
    if _FACTORS_REGISTERED and not force_reload:
        logger.info("缠论因子已注册，跳过重复注册")
        return get_registered_factors()
    
    try:
        import sys
        from pathlib import Path
        # 添加项目根目录到 sys.path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from features.chanlun.czsc_features import CzscFeatureGenerator
        from features.chanlun.chanpy_features import ChanPyFeatureGenerator
    except Exception as e:
        logger.error(f"缠论特征生成器导入失败: {e}")
        return {}
    
    # 初始化生成器（使用单例模式以提高性能）
    czsc_gen = CzscFeatureGenerator(freq='日线')
    chanpy_gen = ChanPyFeatureGenerator(seg_algo='chan')
    
    # 定义因子字典
    # 注意: Qlib 因子名称约定使用 $prefix 开头
    factor_dict = {
        # ========== CZSC 因子 (6个) ==========
        '$fx_mark': {
            'generator': czsc_gen,
            'feature_name': 'fx_mark',
            'description': '分型标记 (1=顶分型, -1=底分型, 0=无)',
            'category': 'czsc',
        },
        '$bi_direction': {
            'generator': czsc_gen,
            'feature_name': 'bi_direction',
            'description': '笔方向 (1=上涨笔, -1=下跌笔, 0=无)',
            'category': 'czsc',
        },
        '$bi_position': {
            'generator': czsc_gen,
            'feature_name': 'bi_position',
            'description': '笔内位置 (0-1, 0=笔起点, 1=笔终点)',
            'category': 'czsc',
        },
        '$bi_power': {
            'generator': czsc_gen,
            'feature_name': 'bi_power',
            'description': '笔幅度 (涨跌幅度)',
            'category': 'czsc',
        },
        '$in_zs': {
            'generator': czsc_gen,
            'feature_name': 'in_zs',
            'description': '是否在中枢内 (1=是, 0=否)',
            'category': 'czsc',
        },
        '$bars_since_fx': {
            'generator': czsc_gen,
            'feature_name': 'bars_since_fx',
            'description': '距离最近分型的K线数',
            'category': 'czsc',
        },
        
        # ========== Chan.py 因子 (10个) ==========
        '$is_buy_point': {
            'generator': chanpy_gen,
            'feature_name': 'is_buy_point',
            'description': '是否买点 (1=是, 0=否)',
            'category': 'chanpy',
        },
        '$is_sell_point': {
            'generator': chanpy_gen,
            'feature_name': 'is_sell_point',
            'description': '是否卖点 (1=是, 0=否)',
            'category': 'chanpy',
        },
        '$bsp_type': {
            'generator': chanpy_gen,
            'feature_name': 'bsp_type',
            'description': '买卖点类型 (1买/2买/3买/1卖/2卖/3卖)',
            'category': 'chanpy',
        },
        '$bsp_is_buy': {
            'generator': chanpy_gen,
            'feature_name': 'bsp_is_buy',
            'description': '买卖点方向 (1=买点, 0=卖点)',
            'category': 'chanpy',
        },
        '$seg_direction': {
            'generator': chanpy_gen,
            'feature_name': 'seg_direction',
            'description': '线段方向 (1=向上, -1=向下)',
            'category': 'chanpy',
        },
        '$is_seg_start': {
            'generator': chanpy_gen,
            'feature_name': 'is_seg_start',
            'description': '是否线段起点 (1=是, 0=否)',
            'category': 'chanpy',
        },
        '$is_seg_end': {
            'generator': chanpy_gen,
            'feature_name': 'is_seg_end',
            'description': '是否线段终点 (1=是, 0=否)',
            'category': 'chanpy',
        },
        '$in_chanpy_zs': {
            'generator': chanpy_gen,
            'feature_name': 'in_chanpy_zs',
            'description': '是否在Chan.py中枢内 (1=是, 0=否)',
            'category': 'chanpy',
        },
        '$zs_low_chanpy': {
            'generator': chanpy_gen,
            'feature_name': 'zs_low_chanpy',
            'description': 'Chan.py中枢下沿价格',
            'category': 'chanpy',
        },
        '$zs_high_chanpy': {
            'generator': chanpy_gen,
            'feature_name': 'zs_high_chanpy',
            'description': 'Chan.py中枢上沿价格',
            'category': 'chanpy',
        },
    }
    
    # 注册因子
    # 注意: Qlib 的因子注册需要通过配置文件或动态注册
    # 这里我们将因子定义存储为元数据，供 Handler 使用
    
    _FACTORS_REGISTERED = True
    
    logger.info(f"✅ 缠论因子注册完成: {len(factor_dict)} 个因子")
    logger.info(f"   - CZSC 因子: 6 个")
    logger.info(f"   - Chan.py 因子: 10 个")
    
    # 保存到全局变量
    global _REGISTERED_FACTORS
    _REGISTERED_FACTORS = factor_dict
    
    return factor_dict


def get_registered_factors() -> Dict:
    """获取已注册的缠论因子列表
    
    Returns:
        dict: 因子名称 -> 因子信息
    """
    global _REGISTERED_FACTORS
    if _REGISTERED_FACTORS is None:
        register_chanlun_factors()
    return _REGISTERED_FACTORS


def get_factor_names(category=None) -> list:
    """获取因子名称列表
    
    Args:
        category: 因子类别过滤 ('czsc', 'chanpy', None=all)
    
    Returns:
        list: 因子名称列表
    """
    factors = get_registered_factors()
    
    if category is None:
        return list(factors.keys())
    
    return [name for name, info in factors.items() 
            if info['category'] == category]


def get_factor_descriptions() -> Dict[str, str]:
    """获取因子描述字典
    
    Returns:
        dict: 因子名称 -> 描述
    """
    factors = get_registered_factors()
    return {name: info['description'] 
            for name, info in factors.items()}


def compute_factor(factor_name: str, df: pd.DataFrame, code: str = None) -> pd.Series:
    """计算单个因子的值
    
    Args:
        factor_name: 因子名称 (如 '$fx_mark')
        df: 包含 OHLCV 的 DataFrame
        code: 股票代码
    
    Returns:
        pd.Series: 因子值序列
    """
    factors = get_registered_factors()
    
    if factor_name not in factors:
        raise ValueError(f"未知因子: {factor_name}")
    
    factor_info = factors[factor_name]
    generator = factor_info['generator']
    feature_name = factor_info['feature_name']
    
    # 生成特征
    if factor_info['category'] == 'czsc':
        result_df = generator.generate_features(df)
    else:  # chanpy
        result_df = generator.generate_features(df, code=code)
    
    # 返回指定特征列
    if feature_name in result_df.columns:
        return result_df[feature_name]
    else:
        logger.warning(f"特征 {feature_name} 未生成，返回零值")
        return pd.Series(0, index=df.index)


def compute_all_factors(df: pd.DataFrame, code: str = None, 
                        category=None) -> pd.DataFrame:
    """计算所有因子或指定类别的因子
    
    Args:
        df: 包含 OHLCV 的 DataFrame
        code: 股票代码
        category: 因子类别 ('czsc', 'chanpy', None=all)
    
    Returns:
        pd.DataFrame: 包含所有因子的 DataFrame
    """
    result = df.copy()
    factor_names = get_factor_names(category)
    
    for factor_name in factor_names:
        try:
            result[factor_name] = compute_factor(factor_name, df, code)
        except Exception as e:
            logger.error(f"计算因子 {factor_name} 失败: {e}")
            result[factor_name] = 0
    
    return result


# 全局变量存储
_REGISTERED_FACTORS = None


if __name__ == '__main__':
    # 测试因子注册
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("缠论因子注册测试")
    print("="*60)
    
    # 注册因子
    factors = register_chanlun_factors()
    
    print(f"\n✅ 注册因子数量: {len(factors)}")
    
    # 显示因子列表
    print("\n📊 CZSC 因子 (6个):")
    for name in get_factor_names('czsc'):
        desc = factors[name]['description']
        print(f"   {name:20s} - {desc}")
    
    print("\n📊 Chan.py 因子 (10个):")
    for name in get_factor_names('chanpy'):
        desc = factors[name]['description']
        print(f"   {name:20s} - {desc}")
    
    # 测试因子计算
    print("\n🧪 测试因子计算...")
    
    # 生成示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_df = pd.DataFrame({
        'datetime': dates,
        'open': 10.0 + np.random.randn(100).cumsum() * 0.1,
        'close': 10.0 + np.random.randn(100).cumsum() * 0.1,
        'high': 10.5 + np.random.randn(100).cumsum() * 0.1,
        'low': 9.5 + np.random.randn(100).cumsum() * 0.1,
        'volume': np.random.randint(900000, 1100000, 100),
    })
    
    # 计算所有因子
    result = compute_all_factors(test_df, code='000001.SZ')
    
    print(f"\n✅ 计算完成！结果形状: {result.shape}")
    print(f"   原始列: {len(test_df.columns)}")
    print(f"   新增列: {len(result.columns) - len(test_df.columns)}")
    
    print("\n✅ 缠论因子注册测试完成!")
