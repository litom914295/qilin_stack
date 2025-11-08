"""
Phase 1 统一集成Pipeline
整合所有Phase 1改进模块到竞价预测系统

集成模块:
1. 数据质量审计
2. 高频特征可靠性测试  
3. 核心特征生成
4. 因子衰减监控
5. 因子生命周期管理
6. Walk-Forward验证
7. 多分类训练
8. 模型对比
9. 宏观情绪因子
10. 题材扩散因子
11. 流动性波动率因子

作者: Qilin Quant Team
创建: 2025-10-30
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入所有Phase 1模块
from scripts.audit_data_quality import DataQualityAuditor
from scripts.generate_core_features import CoreFeatureGenerator
from scripts.train_baseline_model import BaselineModelTrainer
from monitoring.factor_decay_monitor import FactorDecayMonitor
from factors.factor_lifecycle_manager import FactorLifecycleManager
from scripts.walk_forward_validator import WalkForwardValidator, WalkForwardConfig
from scripts.multiclass_trainer import MulticlassTrainer
from scripts.model_comparison_report import ModelComparisonReport
from features.market_sentiment_factors import MarketSentimentFactors
from features.theme_diffusion_factors import ThemeDiffusionFactors
from features.liquidity_volatility_factors import LiquidityVolatilityFactors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedPhase1Pipeline:
    """Phase 1 统一集成Pipeline"""
    
    def __init__(self, config: Optional[Dict] = None, output_dir: str = "output/unified_pipeline"):
        """
        初始化统一Pipeline
        
        Args:
            config: 配置字典
            output_dir: 输出目录
        """
        self.config = config or self._default_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各模块
        self._init_modules()
        
        logger.info("="*70)
        logger.info("Phase 1 统一Pipeline初始化完成")
        logger.info("="*70)
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'data_quality': {
                'min_coverage': 0.95,
                'max_missing_ratio': 0.05,
                'max_outlier_ratio': 0.02
            },
            'feature_selection': {
                'max_features': 50,
                'min_importance': 0.01,
                'correlation_threshold': 0.8
            },
            'factor_health': {
                'ic_windows': [20, 60, 120],
                'min_ic': 0.02,
                'decay_threshold': 0.5
            },
            'walk_forward': {
                'train_window': 180,
                'test_window': 60,
                'step_size': 30,
                'purge_days': 5
            },
            'multiclass': {
                'n_classes': 3,
                'class_names': ['下跌', '平稳', '上涨'],
                'balance_method': 'class_weight',
                'up_threshold': 0.02,
                'down_threshold': -0.02
            },
            'baseline_model': {
                'model_type': 'lgbm',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05
            }
        }
    
    def _init_modules(self):
        """初始化所有模块"""
        # 1. 数据质量审计
        self.data_auditor = DataQualityAuditor(output_dir=str(self.output_dir / "data_quality"))
        
        # 2. 核心特征生成器
        self.feature_generator = CoreFeatureGenerator(
            max_features=self.config['feature_selection']['max_features'],
            output_dir=str(self.output_dir / "core_features")
        )
        
        # 3. 因子衰减监控
        self.factor_monitor = FactorDecayMonitor(
            ic_windows=self.config['factor_health']['ic_windows'],
            output_dir=str(self.output_dir / "factor_health")
        )
        
        # 4. 因子生命周期管理
        self.lifecycle_manager = FactorLifecycleManager(
            ic_threshold=self.config['factor_health']['min_ic'],
            decay_threshold=self.config['factor_health']['decay_threshold']
        )
        
        # 5. Walk-Forward验证器
        wf_config = WalkForwardConfig(**self.config['walk_forward'])
        self.walk_forward_validator = None  # 延迟初始化(需要model_factory)
        self.wf_config = wf_config
        
        # 6. 多分类训练器
        self.multiclass_trainer = None  # 延迟初始化(需要model)
        
        # 7. 模型对比报告
        self.model_comparer = ModelComparisonReport(output_dir=str(self.output_dir / "model_comparison"))
        
        # 8. 宏观情绪因子
        self.sentiment_calculator = MarketSentimentFactors()
        
        # 9. 题材扩散因子
        self.theme_calculator = ThemeDiffusionFactors()
        
        # 10. 流动性波动率因子
        self.liquidity_calculator = LiquidityVolatilityFactors()
        
        logger.info("所有模块初始化完成")
    
    # ========== Phase 1.1: 数据质量审计与清理 ==========
    
    def run_data_quality_audit(self, data_sources: Dict[str, pd.DataFrame]) -> Dict:
        """
        运行数据质量审计
        
        Args:
            data_sources: 数据源字典 {'source_name': dataframe}
        
        Returns:
            审计结果
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.1: 数据质量审计")
        logger.info("="*70)
        
        results = self.data_auditor.run_full_audit(data_sources)
        
        logger.info(f"✅ 数据质量审计完成")
        logger.info(f"  覆盖率: {results.get('avg_coverage', 0):.2%}")
        logger.info(f"  缺失值比例: {results.get('avg_missing_ratio', 0):.2%}")
        logger.info(f"  异常值比例: {results.get('avg_outlier_ratio', 0):.2%}")
        
        return results
    
    def generate_core_features(self, full_feature_df: pd.DataFrame, 
                              target_col: str = 'target') -> pd.DataFrame:
        """
        生成精简核心特征集
        
        Args:
            full_feature_df: 完整特征数据框
            target_col: 目标列名
        
        Returns:
            精简后的特征数据框
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.1: 生成核心特征集")
        logger.info("="*70)
        
        core_features = self.feature_generator.select_core_features(
            full_feature_df, 
            target_col=target_col
        )
        
        logger.info(f"✅ 特征精简完成: {len(full_feature_df.columns)} → {len(core_features.columns)}")
        
        return core_features
    
    # ========== Phase 1.2: 因子衰减监控 ==========
    
    def monitor_factor_health(self, factor_data: pd.DataFrame, 
                             forward_returns: pd.Series) -> Dict:
        """
        监控因子健康状态
        
        Args:
            factor_data: 因子数据
            forward_returns: 前向收益率
        
        Returns:
            因子健康报告
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.2: 因子衰减监控")
        logger.info("="*70)
        
        # 计算因子IC
        factor_names = list(factor_data.columns)
        health_report = self.factor_monitor.batch_calculate_factor_ic(
            factor_data, 
            forward_returns,
            factor_names=factor_names
        )
        
        # 更新生命周期管理
        for factor_name in factor_names:
            ic_metrics = health_report['factors'].get(factor_name, {})
            ic_mean = ic_metrics.get('ic_mean_60d', 0)
            
            status = self.lifecycle_manager.update_factor(factor_name, ic_mean)
            logger.info(f"  {factor_name}: IC={ic_mean:.4f}, 状态={status}")
        
        # 获取活跃因子
        active_factors = self.lifecycle_manager.get_active_factors()
        logger.info(f"\n✅ 活跃因子数: {len(active_factors)}/{len(factor_names)}")
        
        health_report['active_factors'] = active_factors
        health_report['lifecycle_summary'] = self.lifecycle_manager.get_summary()
        
        return health_report
    
    def get_active_factors(self) -> List[str]:
        """获取当前活跃的因子列表"""
        return self.lifecycle_manager.get_active_factors()
    
    # ========== Phase 1.3: 模型简化与严格验证 ==========
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Dict:
        """
        训练简单基准模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        
        Returns:
            训练结果
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.3: 训练基准模型")
        logger.info("="*70)
        
        trainer = BaselineModelTrainer(
            model_type=self.config['baseline_model']['model_type'],
            output_dir=str(self.output_dir / "baseline_model")
        )
        
        results = trainer.train(X_train, y_train, X_val, y_val)
        
        logger.info(f"✅ 基准模型训练完成")
        logger.info(f"  训练AUC: {results.get('train_auc', 0):.4f}")
        if X_val is not None:
            logger.info(f"  验证AUC: {results.get('val_auc', 0):.4f}")
        
        return results
    
    def run_walk_forward_validation(self, df: pd.DataFrame, 
                                   feature_cols: List[str],
                                   target_col: str,
                                   date_col: str = 'date',
                                   model_factory = None) -> Dict:
        """
        运行Walk-Forward验证
        
        Args:
            df: 完整数据集
            feature_cols: 特征列
            target_col: 目标列
            date_col: 日期列
            model_factory: 模型工厂函数
        
        Returns:
            验证结果
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.3: Walk-Forward验证")
        logger.info("="*70)
        
        if model_factory is None:
            # 使用默认模型
            from lightgbm import LGBMClassifier
            def model_factory():
                return LGBMClassifier(
                    n_estimators=self.config['baseline_model']['n_estimators'],
                    max_depth=self.config['baseline_model']['max_depth'],
                    learning_rate=self.config['baseline_model']['learning_rate'],
                    random_state=42
                )
        
        # 定义评估指标
        from sklearn.metrics import roc_auc_score, accuracy_score
        metrics_funcs = {
            'AUC': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
            'Accuracy': lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int))
        }
        
        # 创建验证器
        validator = WalkForwardValidator(
            config=self.wf_config,
            model_factory=model_factory,
            metrics_funcs=metrics_funcs,
            output_dir=str(self.output_dir / "walk_forward")
        )
        
        # 运行验证
        summary = validator.run_validation(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            date_col=date_col
        )
        
        logger.info(f"✅ Walk-Forward验证完成")
        logger.info(f"  Fold数: {summary['n_folds']}")
        logger.info(f"  平均AUC: {summary['aggregate_metrics'].get('AUC_mean', 0):.4f}")
        logger.info(f"  AUC标准差: {summary['aggregate_metrics'].get('AUC_std', 0):.4f}")
        
        return summary
    
    def train_multiclass_model(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: Optional[np.ndarray] = None,
                              y_val: Optional[np.ndarray] = None,
                              model = None) -> Dict:
        """
        训练多分类模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签(多分类)
            X_val: 验证特征
            y_val: 验证标签
            model: 模型实例(可选)
        
        Returns:
            训练结果
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.3: 多分类训练")
        logger.info("="*70)
        
        if model is None:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=self.config['baseline_model']['n_estimators'],
                max_depth=self.config['baseline_model']['max_depth'],
                learning_rate=self.config['baseline_model']['learning_rate'],
                random_state=42
            )
        
        trainer = MulticlassTrainer(
            model=model,
            n_classes=self.config['multiclass']['n_classes'],
            class_names=self.config['multiclass']['class_names'],
            balance_method=self.config['multiclass']['balance_method'],
            output_dir=str(self.output_dir / "multiclass_model")
        )
        
        trainer.train(X_train, y_train, X_val, y_val)
        
        logger.info(f"✅ 多分类模型训练完成")
        
        return trainer.training_history
    
    def compare_models(self, models_data: List[Dict]) -> Dict:
        """
        对比多个模型
        
        Args:
            models_data: 模型数据列表
                [{'model_name': 'xxx', 'metrics': {...}, 'metadata': {...}}, ...]
        
        Returns:
            对比结果
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.3: 模型对比")
        logger.info("="*70)
        
        for model_data in models_data:
            self.model_comparer.add_model(
                model_name=model_data['model_name'],
                metrics=model_data['metrics'],
                metadata=model_data.get('metadata', {})
            )
        
        # 生成报告
        report = self.model_comparer.generate_report()
        
        # 生成可视化
        self.model_comparer.plot_metrics_comparison(plot_type='bar')
        self.model_comparer.plot_metrics_comparison(plot_type='radar')
        
        logger.info(f"✅ 模型对比完成")
        
        return self.model_comparer.get_summary()
    
    # ========== Phase 1.4: 宏观情绪因子补充 ==========
    
    def calculate_market_factors(self, date: str, 
                                market_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        计算市场宏观因子
        
        Args:
            date: 交易日期
            market_data: 市场数据
        
        Returns:
            所有市场因子
        """
        logger.info("\n" + "="*70)
        logger.info("Phase 1.4: 计算市场宏观因子")
        logger.info("="*70)
        
        all_factors = {}
        
        # 1. 市场情绪因子
        sentiment_factors = self.sentiment_calculator.calculate_all_factors(date, market_data)
        all_factors.update({f'sentiment_{k}': v for k, v in sentiment_factors.items()})
        
        # 2. 题材扩散因子
        theme_factors = self.theme_calculator.calculate_all_factors(date, market_data)
        all_factors.update({f'theme_{k}': v for k, v in theme_factors.items()})
        
        # 3. 流动性波动率因子
        liquidity_factors = self.liquidity_calculator.calculate_all_factors(date, market_data)
        all_factors.update({f'liquidity_{k}': v for k, v in liquidity_factors.items()})
        
        logger.info(f"✅ 市场宏观因子计算完成, 共{len(all_factors)}个因子")
        logger.info(f"  市场情绪评分: {sentiment_factors.get('comprehensive_sentiment_score', 0):.1f}/100")
        logger.info(f"  市场状态: {sentiment_factors.get('market_regime', '未知')}")
        logger.info(f"  流动性健康: {liquidity_factors.get('liquidity_health_score', 0):.1f}/100")
        logger.info(f"  波动率状态: {liquidity_factors.get('volatility_regime', '未知')}")
        
        return all_factors
    
    # ========== 完整Pipeline ==========
    
    def run_full_pipeline(self, data_sources: Dict[str, pd.DataFrame],
                         full_feature_df: pd.DataFrame,
                         target_col: str = 'target',
                         date_col: str = 'date') -> Dict:
        """
        运行完整的Phase 1 Pipeline
        
        Args:
            data_sources: 原始数据源
            full_feature_df: 完整特征数据
            target_col: 目标列
            date_col: 日期列
        
        Returns:
            完整Pipeline结果
        """
        logger.info("\n" + "="*70)
        logger.info("开始运行完整 Phase 1 Pipeline")
        logger.info("="*70)
        
        results = {}
        
        # 1. 数据质量审计
        try:
            results['data_quality'] = self.run_data_quality_audit(data_sources)
        except Exception as e:
            logger.error(f"数据质量审计失败: {e}")
            results['data_quality'] = {'error': str(e)}
        
        # 2. 生成核心特征
        try:
            core_features_df = self.generate_core_features(full_feature_df, target_col)
            results['core_features'] = {
                'n_features': len(core_features_df.columns),
                'feature_names': list(core_features_df.columns)
            }
        except Exception as e:
            logger.error(f"核心特征生成失败: {e}")
            core_features_df = full_feature_df
            results['core_features'] = {'error': str(e)}
        
        # 3. 因子健康监控 (如果有target列)
        if target_col in full_feature_df.columns:
            try:
                factor_cols = [col for col in core_features_df.columns if col not in [target_col, date_col]]
                results['factor_health'] = self.monitor_factor_health(
                    core_features_df[factor_cols],
                    core_features_df[target_col]
                )
            except Exception as e:
                logger.error(f"因子健康监控失败: {e}")
                results['factor_health'] = {'error': str(e)}
        
        # 4. 训练基准模型
        try:
            from sklearn.model_selection import train_test_split
            
            X = core_features_df.drop(columns=[target_col, date_col], errors='ignore')
            y = core_features_df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            results['baseline_model'] = self.train_baseline_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.error(f"基准模型训练失败: {e}")
            results['baseline_model'] = {'error': str(e)}
        
        # 5. Walk-Forward验证
        try:
            feature_cols = [col for col in core_features_df.columns if col not in [target_col, date_col]]
            results['walk_forward'] = self.run_walk_forward_validation(
                core_features_df,
                feature_cols,
                target_col,
                date_col
            )
        except Exception as e:
            logger.error(f"Walk-Forward验证失败: {e}")
            results['walk_forward'] = {'error': str(e)}
        
        # 6. 计算市场宏观因子
        try:
            latest_date = full_feature_df[date_col].max()
            results['market_factors'] = self.calculate_market_factors(str(latest_date))
        except Exception as e:
            logger.error(f"市场因子计算失败: {e}")
            results['market_factors'] = {'error': str(e)}
        
        logger.info("\n" + "="*70)
        logger.info("完整 Phase 1 Pipeline 运行完成")
        logger.info("="*70)
        
        # 保存完整结果
        results_path = self.output_dir / "full_pipeline_results.json"
        import json
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"完整结果已保存: {results_path}")
        
        return results


def example_usage():
    """使用示例"""
    # 创建Pipeline
    pipeline = UnifiedPhase1Pipeline(output_dir="output/example_unified_pipeline")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # 完整特征数据
    full_feature_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    full_feature_df['date'] = dates
    full_feature_df['target'] = np.random.randint(0, 2, n_samples)
    
    # 数据源(模拟)
    data_sources = {
        'Qlib': full_feature_df.iloc[:, :50],
        'AKShare': full_feature_df.iloc[:, 50:]
    }
    
    # 运行完整Pipeline
    results = pipeline.run_full_pipeline(
        data_sources=data_sources,
        full_feature_df=full_feature_df,
        target_col='target',
        date_col='date'
    )
    
    print("\n" + "="*70)
    print("Pipeline执行结果:")
    print("="*70)
    for key, value in results.items():
        if isinstance(value, dict) and 'error' not in value:
            print(f"\n{key}:")
            for k, v in list(value.items())[:5]:  # 只显示前5项
                print(f"  {k}: {v}")


if __name__ == "__main__":
    example_usage()
