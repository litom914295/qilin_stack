"""
麒麟量化系统 - LightGBM模型训练系统
训练模型预测首板→二板成功率,替代加权打分
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("LightGBM未安装,请运行: pip install lightgbm")

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import optuna

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """LightGBM模型训练器"""
    
    def __init__(self, model_dir: str = "models"):
        """
        初始化训练器
        
        Args:
            model_dir: 模型保存目录
        """
        if not HAS_LGB:
            raise ImportError("LightGBM未安装")
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_importance = None
        self.best_params = None
        
        logger.info("LightGBM训练器初始化完成")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        time_split: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            df: 数据集DataFrame
            feature_cols: 特征列(None则自动推断)
            test_size: 测试集比例
            time_split: 是否按时间划分(True=按日期,False=随机)
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # 自动推断特征列
        if feature_cols is None:
            exclude_cols = ['date', 'symbol', 'name', 'label']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        logger.info(f"特征数量: {len(feature_cols)}")
        logger.info(f"特征列: {feature_cols}")
        
        X = df[feature_cols].values
        y = df['label'].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        # 划分数据集
        if time_split and 'date' in df.columns:
            # 按时间划分(更符合实际)
            df_sorted = df.sort_values('date')
            split_idx = int(len(df_sorted) * (1 - test_size))
            
            X_train = df_sorted[feature_cols].iloc[:split_idx].values
            X_test = df_sorted[feature_cols].iloc[split_idx:].values
            y_train = df_sorted['label'].iloc[:split_idx].values
            y_test = df_sorted['label'].iloc[split_idx:].values
            
            logger.info(f"按时间划分: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        else:
            # 随机划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info(f"随机划分: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ) -> lgb.Booster:
        """
        训练LightGBM模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征(可选)
            y_val: 验证标签(可选)
            params: 模型参数(可选)
            feature_names: 特征名称(可选)
            
        Returns:
            训练好的模型
        """
        # 默认参数
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'seed': 42,
                'verbose': -1
            }
        
        logger.info("开始训练LightGBM模型...")
        logger.info(f"参数: {params}")
        
        # 创建数据集
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            feature_name=feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val, 
                label=y_val, 
                feature_name=feature_names,
                reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # 训练
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': feature_names if feature_names else [f'f{i}' for i in range(X_train.shape[1])],
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        logger.info("模型训练完成!")
        logger.info(f"最佳迭代轮数: {self.model.best_iteration}")
        logger.info(f"\nTop 10 重要特征:\n{self.feature_importance.head(10)}")
        
        return self.model
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未训练,请先调用train()")
        
        # 预测
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # 计算指标
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        logger.info("=" * 60)
        logger.info("模型评估结果")
        logger.info("=" * 60)
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        logger.info("=" * 60)
        
        return metrics
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_trials: int = 50
    ) -> Dict:
        """
        超参数优化(使用Optuna)
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            feature_names: 特征名称
            n_trials: 优化轮数
            
        Returns:
            最佳参数
        """
        logger.info(f"开始超参数优化,共{n_trials}轮...")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 0.95),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 0.95),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
                'seed': 42,
                'verbose': -1
            }
            
            # 5折交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
                val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                y_pred = model.predict(X_val)
                auc = roc_auc_score(y_val, y_pred)
                scores.append(auc)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='lgb_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'seed': 42,
            'verbose': -1
        })
        
        logger.info("=" * 60)
        logger.info("超参数优化完成!")
        logger.info(f"最佳AUC: {study.best_value:.4f}")
        logger.info(f"最佳参数:\n{json.dumps(self.best_params, indent=2)}")
        logger.info("=" * 60)
        
        return self.best_params
    
    def save_model(self, filename: Optional[str] = None):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        if filename is None:
            filename = f"lgb_model_{datetime.now():%Y%m%d_%H%M%S}.txt"
        
        model_path = self.model_dir / filename
        self.model.save_model(str(model_path))
        
        # 保存特征重要性和参数
        meta_path = model_path.with_suffix('.meta.json')
        meta = {
            'best_iteration': self.model.best_iteration,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'best_params': self.best_params,
            'timestamp': datetime.now().isoformat()
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存: {model_path}")
        logger.info(f"元数据已保存: {meta_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str):
        """加载模型"""
        self.model = lgb.Booster(model_file=model_path)
        
        # 加载元数据
        meta_path = Path(model_path).with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                self.feature_importance = pd.DataFrame(meta.get('feature_importance', []))
                self.best_params = meta.get('best_params')
        
        logger.info(f"模型已加载: {model_path}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练或未加载")
        
        return self.model.predict(X)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 加载模拟数据集
    from app.data_collector import HistoricalDataCollector
    
    collector = HistoricalDataCollector()
    df = collector.generate_mock_dataset(n_samples=2000, positive_ratio=0.3)
    
    # 训练模型
    trainer = LightGBMTrainer()
    
    # 准备数据
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
        df, 
        time_split=True
    )
    
    # 训练
    model = trainer.train(X_train, y_train, X_test, y_test, feature_names=feature_names)
    
    # 评估
    metrics = trainer.evaluate(X_test, y_test)
    
    # 保存
    model_path = trainer.save_model()
    
    print(f"\n模型训练完成!")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"模型路径: {model_path}")
