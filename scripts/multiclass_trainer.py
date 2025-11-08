"""
多分类训练增强模块
支持涨/平/跌三分类,包含样本平衡、阈值优化和类别权重调整
Phase 1.3 - 模型简化与严格验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import logging
from pathlib import Path
import json
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MulticlassTrainer:
    """多分类训练器"""
    
    def __init__(
        self,
        model,
        n_classes: int = 3,
        class_names: Optional[List[str]] = None,
        balance_method: str = 'class_weight',  # 'class_weight', 'oversample', 'undersample', 'smote'
        output_dir: str = "output/multiclass_model"
    ):
        """
        初始化多分类训练器
        
        Args:
            model: 分类模型实例
            n_classes: 类别数量
            class_names: 类别名称列表
            balance_method: 样本平衡方法
            output_dir: 输出目录
        """
        self.model = model
        self.n_classes = n_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(n_classes)]
        self.balance_method = balance_method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_weights: Optional[Dict[int, float]] = None
        self.threshold_optimization_results: Optional[Dict] = None
        self.training_history: Dict = {}
        
    @staticmethod
    def create_labels_from_returns(
        returns: np.ndarray,
        up_threshold: float = 0.02,
        down_threshold: float = -0.02
    ) -> np.ndarray:
        """
        从收益率创建三分类标签
        
        Args:
            returns: 收益率数组
            up_threshold: 上涨阈值
            down_threshold: 下跌阈值
            
        Returns:
            标签数组: 0=下跌, 1=平稳, 2=上涨
        """
        labels = np.ones(len(returns), dtype=int)  # 默认为平稳(1)
        labels[returns >= up_threshold] = 2  # 上涨
        labels[returns <= down_threshold] = 0  # 下跌
        return labels
    
    @staticmethod
    def analyze_label_distribution(y: np.ndarray, class_names: Optional[List[str]] = None) -> Dict:
        """
        分析标签分布
        
        Args:
            y: 标签数组
            class_names: 类别名称
            
        Returns:
            分布统计字典
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        distribution = {}
        for label, count in zip(unique, counts):
            class_name = class_names[label] if class_names else f"Class_{label}"
            distribution[class_name] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        logger.info("标签分布:")
        for class_name, stats in distribution.items():
            logger.info(f"  {class_name}: {stats['count']} ({stats['percentage']:.2f}%)")
        
        return distribution
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        计算类别权重
        
        Args:
            y: 标签数组
            
        Returns:
            类别权重字典 {label: weight}
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info("类别权重:")
        for label, weight in class_weights.items():
            class_name = self.class_names[label]
            logger.info(f"  {class_name}: {weight:.4f}")
        
        return class_weights
    
    def balance_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        样本平衡
        
        Args:
            X: 特征矩阵
            y: 标签数组
            method: 平衡方法,None则使用初始化时的方法
            
        Returns:
            平衡后的 (X, y)
        """
        method = method or self.balance_method
        
        if method == 'class_weight':
            # 不改变样本,只计算权重用于模型训练
            self.class_weights = self.compute_class_weights(y)
            return X, y
        
        elif method == 'oversample':
            # 过采样少数类
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            logger.info(f"过采样: {len(X)} -> {len(X_resampled)} 样本")
            return X_resampled, y_resampled
        
        elif method == 'undersample':
            # 欠采样多数类
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            logger.info(f"欠采样: {len(X)} -> {len(X_resampled)} 样本")
            return X_resampled, y_resampled
        
        elif method == 'smote':
            # SMOTE过采样
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"SMOTE: {len(X)} -> {len(X_resampled)} 样本")
            return X_resampled, y_resampled
        
        else:
            logger.warning(f"未知的平衡方法: {method}, 不进行平衡")
            return X, y
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            **kwargs: 传递给模型fit方法的额外参数
        """
        logger.info("="*60)
        logger.info("开始训练多分类模型")
        logger.info("="*60)
        
        # 分析标签分布
        logger.info("\n训练集标签分布:")
        train_dist = self.analyze_label_distribution(y_train, self.class_names)
        
        if X_val is not None and y_val is not None:
            logger.info("\n验证集标签分布:")
            val_dist = self.analyze_label_distribution(y_val, self.class_names)
        
        # 样本平衡
        X_train_balanced, y_train_balanced = self.balance_samples(X_train, y_train)
        
        # 准备训练参数
        fit_params = kwargs.copy()
        
        # 如果使用class_weight方法,传递权重给模型
        if self.balance_method == 'class_weight' and self.class_weights:
            if hasattr(self.model, 'class_weight'):
                self.model.class_weight = self.class_weights
            elif 'class_weight' in fit_params:
                fit_params['class_weight'] = self.class_weights
        
        # 训练模型
        logger.info(f"\n开始训练: {len(X_train_balanced)} 样本, {X_train_balanced.shape[1]} 特征")
        
        if X_val is not None and y_val is not None:
            # 如果模型支持验证集
            if hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
                fit_params['eval_set'] = [(X_val, y_val)]
                
        self.model.fit(X_train_balanced, y_train_balanced, **fit_params)
        logger.info("训练完成")
        
        # 评估训练集
        logger.info("\n训练集评估:")
        train_metrics = self.evaluate(X_train, y_train, prefix="train")
        
        # 评估验证集
        if X_val is not None and y_val is not None:
            logger.info("\n验证集评估:")
            val_metrics = self.evaluate(X_val, y_val, prefix="val")
        else:
            val_metrics = {}
        
        # 保存训练历史
        self.training_history = {
            'train_samples': len(X_train),
            'train_distribution': train_dist,
            'balance_method': self.balance_method,
            'class_weights': self.class_weights,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        # 保存模型
        self.save_model()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("模型不支持predict_proba方法")
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X: 特征矩阵
            y_true: 真实标签
            prefix: 指标名称前缀
            
        Returns:
            评估指标字典
        """
        y_pred = self.predict(X)
        
        # 准确率
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f"  准确率: {accuracy:.4f}")
        
        # 分类报告
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        logger.info("\n分类报告:")
        for class_name in self.class_names:
            if class_name in report:
                logger.info(f"  {class_name}:")
                logger.info(f"    Precision: {report[class_name]['precision']:.4f}")
                logger.info(f"    Recall: {report[class_name]['recall']:.4f}")
                logger.info(f"    F1-score: {report[class_name]['f1-score']:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\n混淆矩阵:\n{cm}")
        
        # 保存混淆矩阵
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{name}" for name in self.class_names],
            columns=[f"Pred_{name}" for name in self.class_names]
        )
        cm_path = self.output_dir / f"{prefix}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        logger.info(f"混淆矩阵已保存: {cm_path}")
        
        # 汇总指标
        metrics = {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_macro_avg_precision": report['macro avg']['precision'],
            f"{prefix}_macro_avg_recall": report['macro avg']['recall'],
            f"{prefix}_macro_avg_f1": report['macro avg']['f1-score'],
            f"{prefix}_weighted_avg_f1": report['weighted avg']['f1-score']
        }
        
        # 保存分类报告
        report_path = self.output_dir / f"{prefix}_classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return metrics
    
    def optimize_thresholds(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        优化分类阈值(仅适用于能输出概率的模型)
        
        Args:
            X: 特征矩阵
            y_true: 真实标签
            metric: 优化目标指标
            
        Returns:
            最优阈值字典
        """
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("模型不支持predict_proba,无法优化阈值")
            return {}
        
        logger.info("开始阈值优化...")
        y_proba = self.predict_proba(X)
        
        # 简单策略: 尝试不同的阈值组合
        # 这里使用网格搜索方式
        best_score = -np.inf
        best_thresholds = None
        
        # 对于三分类,搜索两个阈值点
        threshold_range = np.arange(0.3, 0.7, 0.05)
        
        for t1 in threshold_range:
            for t2 in threshold_range:
                if t2 <= t1:
                    continue
                
                # 应用阈值
                y_pred_custom = np.argmax(y_proba, axis=1)  # 简化版,实际需更复杂逻辑
                
                # 计算指标
                report = classification_report(
                    y_true,
                    y_pred_custom,
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )
                
                if metric == 'f1':
                    score = report['macro avg']['f1-score']
                elif metric == 'precision':
                    score = report['macro avg']['precision']
                elif metric == 'recall':
                    score = report['macro avg']['recall']
                else:
                    score = accuracy_score(y_true, y_pred_custom)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = {'t1': t1, 't2': t2}
        
        logger.info(f"最优阈值: {best_thresholds}, 得分: {best_score:.4f}")
        self.threshold_optimization_results = {
            'best_thresholds': best_thresholds,
            'best_score': best_score,
            'metric': metric
        }
        
        return best_thresholds
    
    def save_model(self, filename: str = "multiclass_model.pkl"):
        """保存模型"""
        model_path = self.output_dir / filename
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        logger.info(f"训练历史已保存: {history_path}")
    
    def load_model(self, filename: str = "multiclass_model.pkl"):
        """加载模型"""
        model_path = self.output_dir / filename
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"模型已加载: {model_path}")


def example_usage():
    """使用示例"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # 生成模拟数据
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        weights=[0.3, 0.5, 0.2],  # 不平衡数据
        random_state=42
    )
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建模型
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 创建训练器
    trainer = MulticlassTrainer(
        model=model,
        n_classes=3,
        class_names=['下跌', '平稳', '上涨'],
        balance_method='class_weight',
        output_dir="output/multiclass_example"
    )
    
    # 训练
    trainer.train(X_train, y_train, X_test, y_test)
    
    # 测试集评估
    logger.info("\n测试集最终评估:")
    test_metrics = trainer.evaluate(X_test, y_test, prefix="test")
    
    print("\n测试集指标:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 阈值优化(可选)
    # trainer.optimize_thresholds(X_test, y_test, metric='f1')


if __name__ == "__main__":
    example_usage()
