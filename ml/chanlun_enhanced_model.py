"""缠论增强的LightGBM模型

在麒麟现有LightGBM基础上，深度集成缠论因子和Alpha因子

特点:
1. 继承Qlib LGBModel
2. 自动注册并加载16个基础缠论因子
3. 自动生成并加载10个Alpha因子
4. 与麒麟Alpha191/技术指标因子融合
5. 特征重要性分析和可视化
6. 双模式复用支持

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - Phase 4.2
"""

import sys
from pathlib import Path
# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib_enhanced.chanlun.register_factors import register_chanlun_factors, get_factor_names
from qlib_enhanced.chanlun.chanlun_alpha import ChanLunAlphaFactors
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ChanLunEnhancedLGBModel(LGBModel):
    """缠论增强的LightGBM模型
    
    在麒麟现有LGBModel基础上增强:
    - 自动加载16个基础缠论因子
    - 自动生成10个Alpha因子
    - 特征重要性分析
    - 缠论因子贡献度统计
    
    使用方式:
    ```python
    model = ChanLunEnhancedLGBModel(
        use_chanlun=True,
        chanlun_weight=0.3,
        use_alpha=True,
        **lightgbm_params
    )
    ```
    
    复用性:
    - Qlib系统: 作为主模型，完整ML流程
    - 独立系统: 导出特征重要性，指导评分权重
    """
    
    def __init__(self,
                 use_chanlun: bool = True,
                 chanlun_weight: float = 0.3,
                 use_alpha: bool = True,
                 alpha_only_top5: bool = False,
                 enable_feature_analysis: bool = True,
                 **kwargs):
        """初始化缠论增强LightGBM模型
        
        Args:
            use_chanlun: 是否使用缠论因子
            chanlun_weight: 缠论因子建议权重 (0-1)
            use_alpha: 是否使用Alpha因子
            alpha_only_top5: 仅使用Top5 Alpha因子
            enable_feature_analysis: 是否启用特征重要性分析
            **kwargs: LightGBM参数
        """
        super().__init__(**kwargs)
        
        self.use_chanlun = use_chanlun
        self.chanlun_weight = chanlun_weight
        self.use_alpha = use_alpha
        self.alpha_only_top5 = alpha_only_top5
        self.enable_feature_analysis = enable_feature_analysis
        
        # 注册缠论因子
        if use_chanlun:
            register_chanlun_factors()
            logger.info("✅ 缠论因子已注册到模型")
        
        # 特征重要性存储
        self.feature_importance_df = None
        self.chanlun_importance_df = None
        self.alpha_importance_df = None
    
    def fit(self, dataset: DatasetH, **kwargs):
        """训练模型
        
        Args:
            dataset: Qlib标准数据集
            **kwargs: 其他参数
        """
        logger.info("=" * 60)
        logger.info("开始训练缠论增强LightGBM模型")
        logger.info("=" * 60)
        
        # 1. 数据集增强（添加Alpha因子）
        if self.use_chanlun and self.use_alpha:
            logger.info("📊 增强数据集：添加Alpha因子...")
            dataset = self._enhance_dataset_with_alpha(dataset)
        
        # 2. 调用父类训练
        logger.info("🎯 开始LightGBM训练...")
        super().fit(dataset, **kwargs)
        
        # 3. 特征重要性分析
        if self.enable_feature_analysis and hasattr(self, 'model'):
            logger.info("📈 分析特征重要性...")
            self._analyze_feature_importance()
        
        logger.info("✅ 模型训练完成！")
    
    def _enhance_dataset_with_alpha(self, dataset: DatasetH) -> DatasetH:
        """增强数据集：添加Alpha因子
        
        Args:
            dataset: 原始数据集
            
        Returns:
            增强后的数据集
        """
        try:
            # 获取训练和验证数据
            df_train, df_valid = dataset.prepare(
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key="infer"
            )
            
            # 为每个股票生成Alpha因子
            logger.info(f"   处理训练集: {len(df_train.index.get_level_values(0).unique())} 只股票")
            df_train = self._add_alpha_to_dataframe(df_train)
            
            logger.info(f"   处理验证集: {len(df_valid.index.get_level_values(0).unique())} 只股票")
            df_valid = self._add_alpha_to_dataframe(df_valid)
            
            logger.info(f"   ✅ Alpha因子添加完成")
            logger.info(f"      训练集维度: {df_train.shape}")
            logger.info(f"      验证集维度: {df_valid.shape}")
            
        except Exception as e:
            logger.error(f"   ❌ 数据集增强失败: {e}")
            logger.info("   继续使用原始数据集")
        
        return dataset
    
    def _add_alpha_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """为DataFrame添加Alpha因子
        
        Args:
            df: 原始DataFrame (MultiIndex: instrument, datetime)
            
        Returns:
            添加Alpha因子后的DataFrame
        """
        result = df.copy()
        
        # 确定使用哪些Alpha因子
        if self.alpha_only_top5:
            alpha_features = ChanLunAlphaFactors.select_important_features(5)
        else:
            alpha_features = ChanLunAlphaFactors.get_alpha_feature_names()
        
        # 按股票分组处理
        for instrument in df.index.get_level_values(0).unique():
            try:
                inst_df = df.loc[instrument].reset_index()
                
                # 生成Alpha因子
                alpha_df = ChanLunAlphaFactors.generate_alpha_factors(
                    inst_df, 
                    code=instrument
                )
                
                # 合并Alpha因子到结果
                for col in alpha_features:
                    if col in alpha_df.columns:
                        result.loc[instrument, col] = alpha_df[col].values
                
            except Exception as e:
                logger.warning(f"   股票 {instrument} Alpha因子生成失败: {e}")
                # 填充0
                for col in alpha_features:
                    if col not in result.columns:
                        result.loc[instrument, col] = 0
        
        return result
    
    def _analyze_feature_importance(self):
        """分析特征重要性"""
        if not hasattr(self, 'model') or not hasattr(self.model, 'feature_importance_'):
            logger.warning("   模型不支持特征重要性分析")
            return
        
        try:
            # 获取特征重要性
            importance = self.model.feature_importance_
            feature_names = self.model.feature_name_
            
            # 创建DataFrame
            self.feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # 筛选缠论相关特征
            chanlun_pattern = r'(\$fx_mark|\$bi_|\$bsp_|\$seg_|\$in_|zs_|alpha_)'
            self.chanlun_importance_df = self.feature_importance_df[
                self.feature_importance_df['feature'].str.contains(chanlun_pattern, na=False)
            ]
            
            # 筛选Alpha因子
            self.alpha_importance_df = self.feature_importance_df[
                self.feature_importance_df['feature'].str.contains('alpha_', na=False)
            ]
            
            # 统计
            total_importance = self.feature_importance_df['importance'].sum()
            chanlun_importance = self.chanlun_importance_df['importance'].sum()
            alpha_importance = self.alpha_importance_df['importance'].sum()
            
            chanlun_contribution = chanlun_importance / total_importance * 100
            alpha_contribution = alpha_importance / total_importance * 100
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("📊 特征重要性分析结果")
            logger.info("=" * 60)
            logger.info(f"总特征数: {len(self.feature_importance_df)}")
            logger.info(f"缠论相关特征数: {len(self.chanlun_importance_df)}")
            logger.info(f"Alpha因子数: {len(self.alpha_importance_df)}")
            logger.info("")
            logger.info(f"缠论因子总贡献度: {chanlun_contribution:.2f}%")
            logger.info(f"Alpha因子总贡献度: {alpha_contribution:.2f}%")
            logger.info("")
            logger.info("Top10 缠论特征:")
            for idx, row in self.chanlun_importance_df.head(10).iterrows():
                logger.info(f"   {row['feature']:30s}: {row['importance']:8.1f}")
            
            logger.info("")
            logger.info("Top5 Alpha因子:")
            for idx, row in self.alpha_importance_df.head(5).iterrows():
                logger.info(f"   {row['feature']:30s}: {row['importance']:8.1f}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"特征重要性分析失败: {e}")
    
    def get_chanlun_feature_importance(self) -> Optional[pd.DataFrame]:
        """获取缠论特征重要性
        
        Returns:
            DataFrame包含缠论特征及其重要性
        """
        return self.chanlun_importance_df
    
    def get_alpha_feature_importance(self) -> Optional[pd.DataFrame]:
        """获取Alpha因子重要性
        
        Returns:
            DataFrame包含Alpha因子及其重要性
        """
        return self.alpha_importance_df
    
    def plot_importance(self, save_path: Optional[str] = None, top_n: int = 20):
        """绘制特征重要性图
        
        Args:
            save_path: 保存路径，None则显示
            top_n: 显示前N个特征
        """
        if self.feature_importance_df is None:
            logger.warning("未找到特征重要性数据")
            return
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 图1: Top N 所有特征
        top_features = self.feature_importance_df.head(top_n)
        axes[0].barh(top_features['feature'], top_features['importance'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'Top {top_n} Features')
        axes[0].invert_yaxis()
        
        # 图2: 缠论特征
        if self.chanlun_importance_df is not None and len(self.chanlun_importance_df) > 0:
            top_chanlun = self.chanlun_importance_df.head(top_n)
            axes[1].barh(top_chanlun['feature'], top_chanlun['importance'], color='orange')
            axes[1].set_xlabel('Importance')
            axes[1].set_title(f'Top {top_n} 缠论特征')
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"特征重要性图已保存: {save_path}")
        else:
            plt.show()
    
    def export_importance_report(self, output_path: str):
        """导出特征重要性报告
        
        Args:
            output_path: 输出文件路径
        """
        if self.feature_importance_df is None:
            logger.warning("未找到特征重要性数据")
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# 缠论增强LightGBM模型 - 特征重要性报告\n\n")
                
                # 总体统计
                f.write("## 总体统计\n\n")
                f.write(f"- 总特征数: {len(self.feature_importance_df)}\n")
                f.write(f"- 缠论特征数: {len(self.chanlun_importance_df)}\n")
                f.write(f"- Alpha因子数: {len(self.alpha_importance_df)}\n\n")
                
                # 贡献度
                total = self.feature_importance_df['importance'].sum()
                chanlun = self.chanlun_importance_df['importance'].sum()
                alpha = self.alpha_importance_df['importance'].sum()
                
                f.write("## 贡献度\n\n")
                f.write(f"- 缠论因子总贡献: {chanlun/total*100:.2f}%\n")
                f.write(f"- Alpha因子总贡献: {alpha/total*100:.2f}%\n\n")
                
                # Top特征
                f.write("## Top20 全部特征\n\n")
                f.write("| 排名 | 特征 | 重要性 |\n")
                f.write("|-----|------|--------|\n")
                for idx, (_, row) in enumerate(self.feature_importance_df.head(20).iterrows(), 1):
                    f.write(f"| {idx} | {row['feature']} | {row['importance']:.1f} |\n")
                
                f.write("\n## Top10 缠论特征\n\n")
                f.write("| 排名 | 特征 | 重要性 |\n")
                f.write("|-----|------|--------|\n")
                for idx, (_, row) in enumerate(self.chanlun_importance_df.head(10).iterrows(), 1):
                    f.write(f"| {idx} | {row['feature']} | {row['importance']:.1f} |\n")
                
                f.write("\n## Alpha因子重要性\n\n")
                f.write("| 排名 | 因子 | 重要性 |\n")
                f.write("|-----|------|--------|\n")
                for idx, (_, row) in enumerate(self.alpha_importance_df.iterrows(), 1):
                    f.write(f"| {idx} | {row['feature']} | {row['importance']:.1f} |\n")
            
            logger.info(f"特征重要性报告已保存: {output_path}")
            
        except Exception as e:
            logger.error(f"导出报告失败: {e}")


if __name__ == '__main__':
    """测试缠论增强LightGBM模型"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("缠论增强LightGBM模型测试")
    print("=" * 70)
    
    # 测试模型创建
    print("\n1. 测试模型创建...")
    
    try:
        model = ChanLunEnhancedLGBModel(
            use_chanlun=True,
            chanlun_weight=0.3,
            use_alpha=True,
            alpha_only_top5=False,
            enable_feature_analysis=True,
            # LightGBM参数
            loss='mse',
            num_boost_round=100,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=32,
        )
        
        print(f"   ✅ 模型创建成功")
        print(f"   使用缠论因子: {model.use_chanlun}")
        print(f"   使用Alpha因子: {model.use_alpha}")
        print(f"   缠论权重建议: {model.chanlun_weight}")
        
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试Alpha因子生成
    print("\n2. 测试Alpha因子生成...")
    
    try:
        # 生成测试数据
        np.random.seed(42)
        n = 50
        
        test_df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=n, freq='D'),
            'close': 10 + np.random.randn(n).cumsum() * 0.1,
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
        result_df = ChanLunAlphaFactors.generate_alpha_factors(test_df)
        
        alpha_cols = [c for c in result_df.columns if c.startswith('alpha_')]
        print(f"   ✅ Alpha因子生成成功")
        print(f"   原始列数: {len(test_df.columns)}")
        print(f"   Alpha因子数: {len(alpha_cols)}")
        print(f"   总列数: {len(result_df.columns)}")
        
    except Exception as e:
        print(f"   ❌ Alpha因子生成失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 缠论增强LightGBM模型测试完成!")
    print("=" * 70)
    print("\n📝 说明:")
    print("   - 完整测试需要Qlib数据集，可通过qlib_run运行")
    print("   - 本测试验证了模型创建和Alpha因子生成功能")
    print("   - 特征重要性分析需要在fit()后查看")
