"""缠论Alpha因子生成器

基于16个基础缠论特征，构造10个高级Alpha因子
用于Qlib融合系统和独立缠论系统

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - Phase 4.1
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from .divergence_detector import DivergenceDetector, calculate_divergence_alpha

logger = logging.getLogger(__name__)


class ChanLunAlphaFactors:
    """缠论Alpha因子库
    
    基于16个基础缠论因子，构造10个复合Alpha因子：
    1. alpha_buy_strength - 买点强度
    2. alpha_sell_risk - 卖点风险
    3. alpha_trend_consistency - 趋势一致性
    4. alpha_pattern_breakthrough - 形态突破
    5. alpha_zs_oscillation - 中枢震荡度
    6. alpha_buy_persistence - 买点持续性
    7. alpha_pattern_momentum - 形态转折动量
    8. alpha_bi_ma_resonance - 笔段共振
    9. alpha_bsp_ratio - 买卖点比率
    10. alpha_chanlun_momentum - 缠论动量
    
    复用性:
    - Qlib系统: Handler自动加载，输入ML模型
    - 独立系统: MultiAgent直接调用，增强评分
    """
    
    @staticmethod
    def generate_alpha_factors(df: pd.DataFrame, code: str = None) -> pd.DataFrame:
        """生成所有Alpha因子
        
        Args:
            df: 包含基础缠论因子的DataFrame
                必需列: $fx_mark, $bi_direction, $bi_power, $bi_position,
                       $is_buy_point, $is_sell_point, $seg_direction,
                       $in_chanpy_zs, $zs_low_chanpy, $zs_high_chanpy,
                       close
            code: 股票代码（可选，用于日志）
            
        Returns:
            包含所有Alpha因子的DataFrame
        """
        result = df.copy()
        
        try:
            # Alpha1: 买点强度 (买点×笔力度)
            result['alpha_buy_strength'] = ChanLunAlphaFactors._calc_buy_strength(df)
            
            # Alpha2: 卖点风险 (卖点×笔力度，负值表示风险)
            result['alpha_sell_risk'] = ChanLunAlphaFactors._calc_sell_risk(df)
            
            # Alpha3: 趋势一致性 (笔方向×线段方向)
            result['alpha_trend_consistency'] = ChanLunAlphaFactors._calc_trend_consistency(df)
            
            # Alpha4: 形态突破 (分型×笔位置)
            result['alpha_pattern_breakthrough'] = ChanLunAlphaFactors._calc_pattern_breakthrough(df)
            
            # Alpha5: 中枢震荡度
            result['alpha_zs_oscillation'] = ChanLunAlphaFactors._calc_zs_oscillation(df)
            
            # Alpha6: 买点持续性 (近5日买点出现频率)
            result['alpha_buy_persistence'] = ChanLunAlphaFactors._calc_buy_persistence(df)
            
            # Alpha7: 形态转折动量
            result['alpha_pattern_momentum'] = ChanLunAlphaFactors._calc_pattern_momentum(df)
            
            # Alpha8: 笔段共振 (笔方向×均线方向)
            result['alpha_bi_ma_resonance'] = ChanLunAlphaFactors._calc_bi_ma_resonance(df)
            
            # Alpha9: 买卖点比率 (近20日)
            result['alpha_bsp_ratio'] = ChanLunAlphaFactors._calc_bsp_ratio(df)
            
            # Alpha10: 缠论动量 (笔力度×方向的移动平均)
            result['alpha_chanlun_momentum'] = ChanLunAlphaFactors._calc_chanlun_momentum(df)
            
            # P0-2: Alpha11: 背驰风险因子
            result['alpha_divergence_risk'] = ChanLunAlphaFactors._calc_divergence_risk(df)

            # P2-1: AlphaZ1: 中枢移动强度（方向×置信度）
            result['alpha_zs_movement'] = ChanLunAlphaFactors._calc_alpha_zs_movement(df)

            # P2-1: AlphaZ2: 中枢升级强度（是否升级×强度）
            result['alpha_zs_upgrade'] = ChanLunAlphaFactors._calc_alpha_zs_upgrade(df)

            # P2-1: AlphaZ3: 多周期共振强度（tanh归一化）
            result['alpha_confluence'] = ChanLunAlphaFactors._calc_alpha_confluence(df)
            
            logger.debug(f"Alpha因子生成完成: {code or 'unknown'}")
            
        except Exception as e:
            logger.error(f"Alpha因子生成失败 ({code}): {e}")
            # 失败时填充0
            for col in ChanLunAlphaFactors.get_alpha_feature_names():
                if col not in result.columns:
                    result[col] = 0
        
        return result
    
    @staticmethod
    def _calc_buy_strength(df: pd.DataFrame) -> pd.Series:
        """Alpha1: 买点强度
        
        公式: is_buy_point × bi_power
        含义: 买点出现时的笔力度，力度越大信号越强
        """
        if '$is_buy_point' not in df.columns or '$bi_power' not in df.columns:
            return pd.Series(0, index=df.index)
        
        return df['$is_buy_point'] * df['$bi_power']
    
    @staticmethod
    def _calc_sell_risk(df: pd.DataFrame) -> pd.Series:
        """Alpha2: 卖点风险
        
        公式: -is_sell_point × bi_power
        含义: 卖点出现时的风险，负值表示应该卖出
        """
        if '$is_sell_point' not in df.columns or '$bi_power' not in df.columns:
            return pd.Series(0, index=df.index)
        
        return -df['$is_sell_point'] * df['$bi_power']
    
    @staticmethod
    def _calc_trend_consistency(df: pd.DataFrame) -> pd.Series:
        """Alpha3: 趋势一致性
        
        公式: bi_direction × seg_direction
        含义: 笔方向与线段方向一致性，1表示完全一致
        """
        if '$bi_direction' not in df.columns or '$seg_direction' not in df.columns:
            return pd.Series(0, index=df.index)
        
        return df['$bi_direction'] * df['$seg_direction']
    
    @staticmethod
    def _calc_pattern_breakthrough(df: pd.DataFrame) -> pd.Series:
        """Alpha4: 形态突破
        
        公式: fx_mark × bi_position
        含义: 分型出现在笔的不同位置，位置越高/低信号越强
        """
        if '$fx_mark' not in df.columns or '$bi_position' not in df.columns:
            return pd.Series(0, index=df.index)
        
        return df['$fx_mark'] * df['$bi_position']
    
    @staticmethod
    def _calc_zs_oscillation(df: pd.DataFrame) -> pd.Series:
        """Alpha5: 中枢震荡度
        
        公式: in_zs × (1 - |close - zs_mid| / zs_range)
        含义: 在中枢内且接近边界时，震荡度高
        """
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        if '$in_chanpy_zs' not in df.columns:
            return pd.Series(0, index=df.index)
        
        if '$zs_high_chanpy' not in df.columns or '$zs_low_chanpy' not in df.columns:
            return pd.Series(0, index=df.index)
        
        zs_high = df['$zs_high_chanpy']
        zs_low = df['$zs_low_chanpy']
        zs_mid = (zs_high + zs_low) / 2
        zs_range = zs_high - zs_low
        close = df['close']
        
        # 避免除零
        zs_range = zs_range.replace(0, np.nan)
        
        oscillation = df['$in_chanpy_zs'] * (
            1 - np.abs(close - zs_mid) / zs_range
        )
        
        return oscillation.fillna(0)
    
    @staticmethod
    def _calc_buy_persistence(df: pd.DataFrame) -> pd.Series:
        """Alpha6: 买点持续性
        
        公式: Sum(is_buy_point, 5) / 5
        含义: 近5日买点出现频率，频率越高信号越持续
        """
        if '$is_buy_point' not in df.columns:
            return pd.Series(0, index=df.index)
        
        return df['$is_buy_point'].rolling(5, min_periods=1).sum() / 5
    
    @staticmethod
    def _calc_pattern_momentum(df: pd.DataFrame) -> pd.Series:
        """Alpha7: 形态转折动量
        
        公式: Delta(fx_mark, 1)
        含义: 分型变化，从无到有或变化方向
        """
        if '$fx_mark' not in df.columns:
            return pd.Series(0, index=df.index)
        
        return df['$fx_mark'].diff().fillna(0)
    
    @staticmethod
    def _calc_bi_ma_resonance(df: pd.DataFrame) -> pd.Series:
        """Alpha8: 笔段共振
        
        公式: bi_direction × Sign(MA5 - MA10)
        含义: 笔方向与均线方向一致时，共振信号强
        """
        if 'close' not in df.columns or '$bi_direction' not in df.columns:
            return pd.Series(0, index=df.index)
        
        ma5 = df['close'].rolling(5, min_periods=1).mean()
        ma10 = df['close'].rolling(10, min_periods=1).mean()
        ma_direction = np.sign(ma5 - ma10)
        
        return df['$bi_direction'] * ma_direction
    
    @staticmethod
    def _calc_bsp_ratio(df: pd.DataFrame) -> pd.Series:
        """Alpha9: 买卖点比率
        
        公式: Sum(is_buy_point, 20) / (Sum(is_sell_point, 20) + 1)
        含义: 近20日买点/卖点比率，>1表示买点更多
        """
        if '$is_buy_point' not in df.columns or '$is_sell_point' not in df.columns:
            return pd.Series(1, index=df.index)
        
        buy_count = df['$is_buy_point'].rolling(20, min_periods=1).sum()
        sell_count = df['$is_sell_point'].rolling(20, min_periods=1).sum()
        
        return buy_count / (sell_count + 1)
    
    @staticmethod
    def _calc_chanlun_momentum(df: pd.DataFrame) -> pd.Series:
        """Alpha10: 缠论动量
        
        公式: Mean(bi_power × bi_direction, 5)
        含义: 笔力度×方向的移动平均，表示近期动量
        """
        if '$bi_power' not in df.columns or '$bi_direction' not in df.columns:
            return pd.Series(0, index=df.index)
        
        momentum = df['$bi_power'] * df['$bi_direction']
        return momentum.rolling(5, min_periods=1).mean()
    
    @staticmethod
    def _calc_divergence_risk(df: pd.DataFrame) -> pd.Series:
        """P0-2: Alpha11: 背驰风险因子
        
        使用DivergenceDetector检测背驰,返回风险评分
        负值表示顶背驰(卖出风险),正值表示底背驰(买入机会)
        """
        try:
            # 使用P0-2的背驰检测器
            return calculate_divergence_alpha(df)
        except Exception as e:
            logger.warning(f"背驰因子计算失败: {e}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def get_alpha_feature_names() -> List[str]:
        """获取所有Alpha因子名称"""
        return [
            'alpha_buy_strength',
            'alpha_sell_risk',
            'alpha_trend_consistency',
            'alpha_pattern_breakthrough',
            'alpha_zs_oscillation',
            'alpha_buy_persistence',
            'alpha_pattern_momentum',
            'alpha_bi_ma_resonance',
            'alpha_bsp_ratio',
            'alpha_chanlun_momentum',
'alpha_divergence_risk',  # P0-2
            # P2-1
            'alpha_zs_movement',
            'alpha_zs_upgrade',
            'alpha_confluence',
        ]
    
    @staticmethod
    def get_alpha_descriptions() -> dict:
        """获取Alpha因子描述"""
        return {
            'alpha_buy_strength': '买点强度 (买点×笔力度)',
            'alpha_sell_risk': '卖点风险 (负值表示风险)',
            'alpha_trend_consistency': '趋势一致性 (笔×线段方向)',
            'alpha_pattern_breakthrough': '形态突破 (分型×笔位置)',
            'alpha_zs_oscillation': '中枢震荡度',
            'alpha_buy_persistence': '买点持续性 (近5日频率)',
            'alpha_pattern_momentum': '形态转折动量',
            'alpha_bi_ma_resonance': '笔段共振 (笔×均线)',
            'alpha_bsp_ratio': '买卖点比率 (近20日)',
            'alpha_chanlun_momentum': '缠论动量 (笔力度×方向MA5)',
'alpha_divergence_risk': 'P0-2背驰风险 (负=顶背驰,正=底背驰)',  # P0-2
            # P2-1
            'alpha_zs_movement': '中枢移动强度 (方向×置信度)',
            'alpha_zs_upgrade': '中枢升级强度 (是否升级×强度)',
            'alpha_confluence': '多周期共振强度 (tanh归一化)',
        }
    
    @staticmethod
    def select_important_features(top_n: int = 5) -> List[str]:
        """选择重要的Alpha因子
        
        Args:
            top_n: 选择前N个重要因子
            
        Returns:
            因子名称列表
        """
        # 根据经验排序
        importance_ranking = [
            'alpha_buy_strength',      # 最重要
            'alpha_chanlun_momentum',   
            'alpha_trend_consistency',
            'alpha_bi_ma_resonance',
            'alpha_buy_persistence',
            'alpha_sell_risk',
            'alpha_bsp_ratio',
            'alpha_pattern_breakthrough',
            'alpha_zs_oscillation',
            'alpha_pattern_momentum',   # 最不重要
        ]
        
        return importance_ranking[:top_n]

    # ===== P2-1: 新增Alpha计算 =====
    @staticmethod
    def _get_first_available(df: pd.DataFrame, names: List[str], default: float = 0.0) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series(default, index=df.index)

    @staticmethod
    def _calc_alpha_zs_movement(df: pd.DataFrame) -> pd.Series:
        dir_s = ChanLunAlphaFactors._get_first_available(df, ['$zs_movement_direction','zs_movement_direction'], 0)
        conf_s = ChanLunAlphaFactors._get_first_available(df, ['$zs_movement_confidence','zs_movement_confidence'], 0.0)
        try:
            return (dir_s.astype(float) * conf_s.astype(float)).fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=df.index)

    @staticmethod
    def _calc_alpha_zs_upgrade(df: pd.DataFrame) -> pd.Series:
        flag_s = ChanLunAlphaFactors._get_first_available(df, ['$zs_upgrade_flag','zs_upgrade_flag'], 0)
        strength_s = ChanLunAlphaFactors._get_first_available(df, ['$zs_upgrade_strength','zs_upgrade_strength'], 0.0)
        try:
            return (flag_s.astype(float) * strength_s.astype(float)).fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=df.index)

    @staticmethod
    def _calc_alpha_confluence(df: pd.DataFrame) -> pd.Series:
        score_s = ChanLunAlphaFactors._get_first_available(df, ['$confluence_score','confluence_score'], 0.0)
        try:
            # tanh归一化到[-1,1]
            return np.tanh(score_s.astype(float)).fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=df.index)


if __name__ == '__main__':
    """测试Alpha因子生成"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("缠论Alpha因子生成器测试")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    
    test_df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='D'),
        'close': 10 + np.random.randn(n).cumsum() * 0.1,
        # 基础缠论因子（模拟）
        '$fx_mark': np.random.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1]),
        '$bi_direction': np.random.choice([-1, 1], n),
        '$bi_power': np.abs(np.random.randn(n) * 0.05),
        '$bi_position': np.random.rand(n),
        '$is_buy_point': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        '$is_sell_point': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        '$seg_direction': np.random.choice([-1, 1], n),
        '$in_chanpy_zs': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        '$zs_low_chanpy': 9.5 + np.random.rand(n) * 0.3,
        '$zs_high_chanpy': 10.2 + np.random.rand(n) * 0.3,
    })
    
    # 生成Alpha因子
    print("\n生成Alpha因子...")
    result = ChanLunAlphaFactors.generate_alpha_factors(test_df, code='TEST001')
    
    # 显示结果
    print(f"\n✅ 生成完成！")
    print(f"   原始列数: {len(test_df.columns)}")
    print(f"   新增列数: {len(result.columns) - len(test_df.columns)}")
    print(f"   总列数: {len(result.columns)}")
    
    # 显示Alpha因子统计
    print("\n📊 Alpha因子统计:")
    alpha_features = ChanLunAlphaFactors.get_alpha_feature_names()
    for feat in alpha_features:
        if feat in result.columns:
            mean_val = result[feat].mean()
            std_val = result[feat].std()
            print(f"   {feat:30s}: mean={mean_val:7.4f}, std={std_val:7.4f}")
    
    # 显示因子描述
    print("\n📝 Alpha因子描述:")
    descriptions = ChanLunAlphaFactors.get_alpha_descriptions()
    for name, desc in descriptions.items():
        print(f"   {name:30s}: {desc}")
    
    # 显示重要因子
    print("\n⭐ Top5 重要因子:")
    important = ChanLunAlphaFactors.select_important_features(5)
    for i, feat in enumerate(important, 1):
        print(f"   {i}. {feat} - {descriptions[feat]}")
    
    print("\n✅ 缠论Alpha因子生成器测试完成!")
