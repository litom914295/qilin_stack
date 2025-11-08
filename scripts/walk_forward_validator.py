"""
Walk-Forward验证框架
实现滚动窗口回测,支持多指标评估和参数稳定性分析
Phase 1.3 - 模型简化与严格验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-Forward验证配置"""
    train_window: int = 60  # 训练窗口天数
    test_window: int = 20   # 测试窗口天数
    step_size: int = 20     # 滚动步长天数
    min_train_samples: int = 1000  # 最小训练样本数
    purge_days: int = 5     # 训练/测试之间的隔离期
    embargo_days: int = 2   # 测试集后的禁用期
    

@dataclass
class FoldMetrics:
    """单个fold的评估指标"""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[pd.DataFrame] = None


class WalkForwardValidator:
    """Walk-Forward验证器"""
    
    def __init__(
        self, 
        config: WalkForwardConfig,
        model_factory: Callable,
        metrics_funcs: Dict[str, Callable],
        output_dir: str = "output/walk_forward"
    ):
        """
        初始化验证器
        
        Args:
            config: Walk-Forward配置
            model_factory: 模型工厂函数,返回未训练的模型实例
            metrics_funcs: 评估指标函数字典 {name: func(y_true, y_pred)}
            output_dir: 输出目录
        """
        self.config = config
        self.model_factory = model_factory
        self.metrics_funcs = metrics_funcs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fold_results: List[FoldMetrics] = []
        self.aggregate_metrics: Dict[str, float] = {}
        
    def create_folds(self, df: pd.DataFrame, date_col: str = 'date') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        创建Walk-Forward折叠
        
        Args:
            df: 数据框,必须包含日期列
            date_col: 日期列名
            
        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])
        
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        folds = []
        current_date = min_date + timedelta(days=self.config.train_window)
        
        logger.info(f"创建Walk-Forward折叠: {min_date.date()} 到 {max_date.date()}")
        
        fold_id = 0
        while current_date < max_date:
            # 训练集
            train_start = current_date - timedelta(days=self.config.train_window)
            train_end = current_date
            
            # 测试集 (考虑purge期)
            test_start = current_date + timedelta(days=self.config.purge_days)
            test_end = test_start + timedelta(days=self.config.test_window)
            
            if test_end > max_date:
                break
                
            # 提取数据
            train_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
            test_mask = (df[date_col] >= test_start) & (df[date_col] < test_end)
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            # 检查样本数
            if len(train_df) < self.config.min_train_samples:
                logger.warning(f"Fold {fold_id}: 训练样本不足 ({len(train_df)} < {self.config.min_train_samples})")
                current_date += timedelta(days=self.config.step_size)
                fold_id += 1
                continue
                
            if len(test_df) == 0:
                logger.warning(f"Fold {fold_id}: 测试集为空")
                current_date += timedelta(days=self.config.step_size)
                fold_id += 1
                continue
            
            folds.append((train_df, test_df))
            logger.info(f"Fold {fold_id}: 训练 {train_start.date()}~{train_end.date()} ({len(train_df)}), "
                       f"测试 {test_start.date()}~{test_end.date()} ({len(test_df)})")
            
            # 滚动到下一个窗口
            current_date += timedelta(days=self.config.step_size)
            fold_id += 1
            
        logger.info(f"总共创建 {len(folds)} 个折叠")
        return folds
    
    def run_validation(
        self, 
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        date_col: str = 'date',
        save_predictions: bool = True,
        save_models: bool = False
    ) -> Dict:
        """
        执行Walk-Forward验证
        
        Args:
            df: 完整数据集
            feature_cols: 特征列名列表
            target_col: 目标列名
            date_col: 日期列名
            save_predictions: 是否保存预测结果
            save_models: 是否保存训练的模型
            
        Returns:
            验证结果摘要
        """
        logger.info("="*60)
        logger.info("开始Walk-Forward验证")
        logger.info("="*60)
        
        # 创建折叠
        folds = self.create_folds(df, date_col)
        
        if len(folds) == 0:
            logger.error("无法创建任何有效的折叠!")
            return {}
        
        # 对每个fold进行训练和评估
        self.fold_results = []
        for fold_id, (train_df, test_df) in enumerate(folds):
            logger.info(f"\n--- Fold {fold_id + 1}/{len(folds)} ---")
            
            # 准备数据
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values
            
            # 训练模型
            model = self.model_factory()
            logger.info(f"训练模型: {len(X_train)} 样本, {len(feature_cols)} 特征")
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            metrics = {}
            for metric_name, metric_func in self.metrics_funcs.items():
                try:
                    score = metric_func(y_test, y_pred)
                    metrics[metric_name] = score
                    logger.info(f"  {metric_name}: {score:.4f}")
                except Exception as e:
                    logger.error(f"计算指标 {metric_name} 失败: {e}")
                    metrics[metric_name] = np.nan
            
            # 特征重要性
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            # 保存预测
            predictions_df = None
            if save_predictions:
                predictions_df = test_df[[date_col]].copy()
                predictions_df['y_true'] = y_test
                predictions_df['y_pred'] = y_pred
            
            # 记录结果
            fold_result = FoldMetrics(
                fold_id=fold_id,
                train_start=train_df[date_col].min().strftime('%Y-%m-%d'),
                train_end=train_df[date_col].max().strftime('%Y-%m-%d'),
                test_start=test_df[date_col].min().strftime('%Y-%m-%d'),
                test_end=test_df[date_col].max().strftime('%Y-%m-%d'),
                train_samples=len(train_df),
                test_samples=len(test_df),
                metrics=metrics,
                feature_importance=feature_importance,
                predictions=predictions_df
            )
            self.fold_results.append(fold_result)
            
            # 保存模型
            if save_models:
                model_path = self.output_dir / f"model_fold_{fold_id}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"模型已保存: {model_path}")
        
        # 汇总结果
        self._aggregate_results()
        
        # 保存结果
        self._save_results()
        
        logger.info("\n" + "="*60)
        logger.info("Walk-Forward验证完成")
        logger.info("="*60)
        
        return self.get_summary()
    
    def _aggregate_results(self):
        """汇总所有fold的结果"""
        if not self.fold_results:
            return
        
        # 收集所有指标
        all_metrics = {}
        for fold_result in self.fold_results:
            for metric_name, value in fold_result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # 计算统计量
        self.aggregate_metrics = {}
        for metric_name, values in all_metrics.items():
            values = [v for v in values if not np.isnan(v)]
            if values:
                self.aggregate_metrics[f"{metric_name}_mean"] = np.mean(values)
                self.aggregate_metrics[f"{metric_name}_std"] = np.std(values)
                self.aggregate_metrics[f"{metric_name}_min"] = np.min(values)
                self.aggregate_metrics[f"{metric_name}_max"] = np.max(values)
                self.aggregate_metrics[f"{metric_name}_median"] = np.median(values)
        
        logger.info("\n汇总指标:")
        for key, value in self.aggregate_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    def _save_results(self):
        """保存验证结果"""
        # 保存fold级别结果
        fold_data = []
        for fold_result in self.fold_results:
            fold_dict = {
                'fold_id': fold_result.fold_id,
                'train_start': fold_result.train_start,
                'train_end': fold_result.train_end,
                'test_start': fold_result.test_start,
                'test_end': fold_result.test_end,
                'train_samples': fold_result.train_samples,
                'test_samples': fold_result.test_samples,
                **fold_result.metrics
            }
            fold_data.append(fold_dict)
        
        fold_df = pd.DataFrame(fold_data)
        fold_csv_path = self.output_dir / "fold_results.csv"
        fold_df.to_csv(fold_csv_path, index=False)
        logger.info(f"\nFold结果已保存: {fold_csv_path}")
        
        # 保存汇总指标
        summary_path = self.output_dir / "aggregate_metrics.json"
        with open(summary_path, 'w') as f:
            json.dump(self.aggregate_metrics, indent=2, fp=f)
        logger.info(f"汇总指标已保存: {summary_path}")
        
        # 保存预测结果
        if self.fold_results and self.fold_results[0].predictions is not None:
            all_predictions = []
            for fold_result in self.fold_results:
                if fold_result.predictions is not None:
                    fold_preds = fold_result.predictions.copy()
                    fold_preds['fold_id'] = fold_result.fold_id
                    all_predictions.append(fold_preds)
            
            if all_predictions:
                all_preds_df = pd.concat(all_predictions, ignore_index=True)
                pred_path = self.output_dir / "all_predictions.csv"
                all_preds_df.to_csv(pred_path, index=False)
                logger.info(f"预测结果已保存: {pred_path}")
        
        # 保存特征重要性
        if self.fold_results and self.fold_results[0].feature_importance is not None:
            importance_data = []
            for fold_result in self.fold_results:
                if fold_result.feature_importance:
                    for feature, importance in fold_result.feature_importance.items():
                        importance_data.append({
                            'fold_id': fold_result.fold_id,
                            'feature': feature,
                            'importance': importance
                        })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                importance_path = self.output_dir / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"特征重要性已保存: {importance_path}")
    
    def get_summary(self) -> Dict:
        """获取验证摘要"""
        summary = {
            'config': {
                'train_window': self.config.train_window,
                'test_window': self.config.test_window,
                'step_size': self.config.step_size,
                'purge_days': self.config.purge_days
            },
            'n_folds': len(self.fold_results),
            'aggregate_metrics': self.aggregate_metrics,
            'output_dir': str(self.output_dir)
        }
        return summary
    
    def plot_metrics_over_time(self, metric_names: Optional[List[str]] = None):
        """绘制指标随时间变化图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装,无法绘图")
            return
        
        if not self.fold_results:
            logger.warning("没有fold结果可绘制")
            return
        
        # 默认绘制所有指标
        if metric_names is None:
            metric_names = list(self.fold_results[0].metrics.keys())
        
        # 准备数据
        fold_ids = [f.fold_id for f in self.fold_results]
        test_starts = [f.test_start for f in self.fold_results]
        
        # 创建图形
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4*len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metric_names):
            values = [f.metrics.get(metric_name, np.nan) for f in self.fold_results]
            
            ax.plot(fold_ids, values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Fold ID')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} 随时间变化')
            ax.grid(True, alpha=0.3)
            
            # 添加平均线
            mean_val = np.nanmean(values)
            ax.axhline(mean_val, color='r', linestyle='--', alpha=0.7, label=f'平均: {mean_val:.4f}')
            ax.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / "metrics_over_time.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"指标图已保存: {plot_path}")
        plt.close()


def example_usage():
    """使用示例"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    df = pd.DataFrame({
        'date': dates,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randn(n_samples)
    })
    
    # 配置
    config = WalkForwardConfig(
        train_window=180,
        test_window=60,
        step_size=30,
        purge_days=5
    )
    
    # 模型工厂
    def model_factory():
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 评估指标
    metrics_funcs = {
        'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score,
        'MAE': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    }
    
    # 创建验证器
    validator = WalkForwardValidator(
        config=config,
        model_factory=model_factory,
        metrics_funcs=metrics_funcs,
        output_dir="output/walk_forward_example"
    )
    
    # 运行验证
    feature_cols = ['feature1', 'feature2', 'feature3']
    summary = validator.run_validation(
        df=df,
        feature_cols=feature_cols,
        target_col='target',
        date_col='date',
        save_predictions=True,
        save_models=False
    )
    
    # 绘制指标变化
    validator.plot_metrics_over_time()
    
    print("\n验证摘要:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    example_usage()
