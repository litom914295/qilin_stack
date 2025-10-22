"""
涨停板预测系统 - Optuna超参数调优模块
支持多模型自动调参和结果持久化
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

class LimitUpHyperparameterTuner:
    """涨停板预测超参数调优器"""
    
    def __init__(
        self,
        model_type: str = 'lightgbm',
        n_trials: int = 100,
        timeout: int = 3600,
        save_dir: str = './tuning_results'
    ):
        """
        初始化调优器
        
        Args:
            model_type: 模型类型 ('lightgbm', 'xgboost', 'catboost')
            n_trials: 优化试验次数
            timeout: 超时时间（秒）
            save_dir: 结果保存目录
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """根据模型类型建议超参数"""
        
        if self.model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
        
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
        
        elif self.model_type == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            }
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def create_model(self, params: Dict[str, Any]):
        """创建模型实例"""
        
        if self.model_type == 'lightgbm':
            return LGBMClassifier(**params, random_state=42, verbose=-1)
        
        elif self.model_type == 'xgboost':
            return XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        elif self.model_type == 'catboost':
            return CatBoostClassifier(**params, random_state=42, verbose=0)
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """优化目标函数 - 使用时间序列交叉验证"""
        
        # 建议超参数
        params = self.suggest_params(trial)
        
        # 创建模型
        model = self.create_model(params)
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_val)
            
            # 计算F1分数（针对涨停板预测优化）
            score = f1_score(y_val, y_pred, average='weighted')
            scores.append(score)
        
        # 返回平均得分
        mean_score = np.mean(scores)
        return mean_score
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        direction: str = 'maximize'
    ) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            X: 特征数据
            y: 目标变量
            direction: 优化方向 ('maximize' or 'minimize')
            
        Returns:
            最优超参数字典
        """
        
        print(f"\n🚀 开始{self.model_type}模型超参数优化...")
        print(f"试验次数: {self.n_trials}, 超时: {self.timeout}秒")
        
        # 创建Optuna study
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # 执行优化
        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # 保存最优参数
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"\n✅ 优化完成!")
        print(f"最优得分: {self.best_score:.4f}")
        print(f"最优参数: {json.dumps(self.best_params, indent=2, ensure_ascii=False)}")
        
        # 保存结果
        self._save_results()
        
        return self.best_params
    
    def _save_results(self):
        """保存优化结果"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存最优参数
        params_file = self.save_dir / f'{self.model_type}_best_params_{timestamp}.json'
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_type': self.model_type,
                'best_score': float(self.best_score),
                'best_params': self.best_params,
                'n_trials': self.n_trials,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 参数已保存: {params_file}")
        
        # 保存优化历史
        df_trials = self.study.trials_dataframe()
        history_file = self.save_dir / f'{self.model_type}_history_{timestamp}.csv'
        df_trials.to_csv(history_file, index=False, encoding='utf-8-sig')
        
        print(f"📊 历史已保存: {history_file}")
        
        # 生成可视化报告
        self._generate_visualization()
    
    def _generate_visualization(self):
        """生成可视化报告"""
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. 优化历史
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot([t.number for t in self.study.trials], 
                   [t.value for t in self.study.trials])
            ax.set_xlabel('试验次数')
            ax.set_ylabel('F1分数')
            ax.set_title(f'{self.model_type} 优化历史')
            ax.grid(True, alpha=0.3)
            
            history_plot = self.save_dir / f'{self.model_type}_history_{timestamp}.png'
            plt.savefig(history_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 可视化已保存: {history_plot}")
            
        except Exception as e:
            print(f"⚠️ 生成可视化失败: {e}")
    
    def load_best_params(self, params_file: str) -> Dict[str, Any]:
        """加载保存的最优参数"""
        
        with open(params_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.best_params = data['best_params']
        self.best_score = data['best_score']
        
        print(f"✅ 已加载最优参数 (得分: {self.best_score:.4f})")
        return self.best_params


class MultiModelTuner:
    """多模型批量调优器"""
    
    def __init__(
        self,
        models: List[str] = ['lightgbm', 'xgboost', 'catboost'],
        n_trials: int = 100,
        timeout: int = 3600,
        save_dir: str = './tuning_results'
    ):
        """
        初始化多模型调优器
        
        Args:
            models: 模型列表
            n_trials: 每个模型的试验次数
            timeout: 每个模型的超时时间
            save_dir: 结果保存目录
        """
        self.models = models
        self.n_trials = n_trials
        self.timeout = timeout
        self.save_dir = save_dir
        
        self.tuners = {}
        self.results = {}
    
    def optimize_all(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        优化所有模型
        
        Returns:
            所有模型的最优参数字典
        """
        
        print(f"\n{'='*60}")
        print(f"🎯 开始批量超参数优化 - {len(self.models)}个模型")
        print(f"{'='*60}\n")
        
        for model_type in self.models:
            print(f"\n{'='*60}")
            print(f"模型: {model_type.upper()}")
            print(f"{'='*60}")
            
            # 创建调优器
            tuner = LimitUpHyperparameterTuner(
                model_type=model_type,
                n_trials=self.n_trials,
                timeout=self.timeout,
                save_dir=self.save_dir
            )
            
            # 执行优化
            best_params = tuner.optimize(X, y)
            
            # 保存结果
            self.tuners[model_type] = tuner
            self.results[model_type] = {
                'best_params': best_params,
                'best_score': tuner.best_score
            }
        
        # 生成综合报告
        self._generate_summary_report()
        
        return self.results
    
    def _generate_summary_report(self):
        """生成综合报告"""
        
        print(f"\n{'='*60}")
        print("📊 优化结果汇总")
        print(f"{'='*60}\n")
        
        # 创建结果表格
        summary_data = []
        for model_type, result in self.results.items():
            summary_data.append({
                '模型': model_type,
                '最优得分': f"{result['best_score']:.4f}",
                '参数数量': len(result['best_params'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # 保存汇总
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = Path(self.save_dir) / f'tuning_summary_{timestamp}.csv'
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 汇总已保存: {summary_file}")


if __name__ == '__main__':
    # 测试示例
    print("="*60)
    print("涨停板预测系统 - Optuna超参数调优模块")
    print("="*60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 模拟涨停板标签（0=不涨停，1=涨停）
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]))
    
    print(f"\n数据集大小: {X.shape}")
    print(f"涨停板样本占比: {y.mean():.2%}")
    
    # 单模型调优
    print("\n" + "="*60)
    print("1️⃣ 单模型调优示例")
    print("="*60)
    
    tuner = LimitUpHyperparameterTuner(
        model_type='lightgbm',
        n_trials=20,
        timeout=300
    )
    
    best_params = tuner.optimize(X, y)
    
    # 多模型批量调优
    print("\n" + "="*60)
    print("2️⃣ 多模型批量调优示例")
    print("="*60)
    
    multi_tuner = MultiModelTuner(
        models=['lightgbm', 'xgboost'],
        n_trials=20,
        timeout=300
    )
    
    results = multi_tuner.optimize_all(X, y)
    
    print("\n✅ 所有调优任务完成!")
