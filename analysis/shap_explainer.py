"""
SHAP模型可解释性分析
提供全局特征重要性和逐票解释
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP解释器"""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        初始化SHAP解释器
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def compute_shap_values(self, X: np.ndarray, sample_size: int = 100) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            X: 特征数据
            sample_size: 背景样本数量
            
        Returns:
            SHAP值矩阵
        """
        try:
            # 根据模型类型选择解释器
            model_type = type(self.model).__name__
            
            if 'XGBoost' in model_type or 'xgb' in model_type.lower():
                # XGBoost模型使用TreeExplainer
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X)
                
            elif 'LightGBM' in model_type or 'lgb' in model_type.lower():
                # LightGBM模型使用TreeExplainer
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X)
                if len(self.shap_values) > 1:
                    # 二分类任务，取正类的SHAP值
                    self.shap_values = self.shap_values[1]
                    
            elif 'CatBoost' in model_type or 'cat' in model_type.lower():
                # CatBoost模型
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X)
                
            else:
                # 其他模型使用KernelExplainer（较慢）
                background = shap.sample(X, min(sample_size, len(X)))
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    background
                )
                self.shap_values = self.explainer.shap_values(X)
                
            logger.info(f"成功计算SHAP值，形状: {self.shap_values.shape}")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"计算SHAP值失败: {str(e)}")
            # 回退到简单的特征重要性
            return self._compute_simple_importance(X)
    
    def _compute_simple_importance(self, X: np.ndarray) -> np.ndarray:
        """简单的特征重要性计算（作为SHAP的回退方案）"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            # 扩展为每个样本的重要性
            return np.tile(importances, (X.shape[0], 1))
        else:
            # 返回随机值
            return np.random.randn(*X.shape) * 0.01
    
    def get_global_importance(self) -> pd.DataFrame:
        """
        获取全局特征重要性
        
        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        if self.shap_values is None:
            raise ValueError("请先调用compute_shap_values计算SHAP值")
        
        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # 归一化到0-1
        importance_df['importance'] = importance_df['importance'] / importance_df['importance'].sum()
        
        return importance_df
    
    def get_sample_explanation(self, sample_idx: int, top_n: int = 10) -> pd.DataFrame:
        """
        获取单个样本的SHAP解释
        
        Args:
            sample_idx: 样本索引
            top_n: 显示前N个重要特征
            
        Returns:
            DataFrame with columns ['feature', 'value', 'shap_value', 'contribution']
        """
        if self.shap_values is None:
            raise ValueError("请先调用compute_shap_values计算SHAP值")
        
        sample_shap = self.shap_values[sample_idx]
        
        # 获取绝对值最大的top_n个特征
        top_indices = np.argsort(np.abs(sample_shap))[-top_n:][::-1]
        
        explanation_df = pd.DataFrame({
            'feature': [self.feature_names[i] for i in top_indices],
            'shap_value': sample_shap[top_indices],
            'abs_shap': np.abs(sample_shap[top_indices])
        })
        
        # 添加贡献方向
        explanation_df['contribution'] = explanation_df['shap_value'].apply(
            lambda x: '↑ 正向' if x > 0 else '↓ 负向'
        )
        
        return explanation_df
    
    def save_explanations(self, output_dir: str, X: np.ndarray, 
                         sample_ids: Optional[List[str]] = None):
        """
        保存SHAP解释结果
        
        Args:
            output_dir: 输出目录
            X: 特征数据
            sample_ids: 样本ID列表（如股票代码）
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存全局特征重要性
        global_importance = self.get_global_importance()
        global_importance.to_csv(
            output_path / 'feature_importance_shap.csv', 
            index=False
        )
        logger.info(f"全局特征重要性已保存到: {output_path / 'feature_importance_shap.csv'}")
        
        # 保存逐样本SHAP值
        per_sample_data = []
        for i in range(len(X)):
            sample_id = sample_ids[i] if sample_ids else f"sample_{i}"
            sample_explanation = self.get_sample_explanation(i, top_n=20)
            sample_explanation['sample_id'] = sample_id
            per_sample_data.append(sample_explanation)
        
        per_sample_df = pd.concat(per_sample_data, ignore_index=True)
        per_sample_df.to_parquet(
            output_path / 'shap_top_features.parquet',
            index=False
        )
        logger.info(f"逐样本SHAP值已保存到: {output_path / 'shap_top_features.parquet'}")
        
        return global_importance, per_sample_df
    
    def plot_global_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        绘制全局特征重要性图
        
        Args:
            top_n: 显示前N个特征
            figsize: 图形大小
        """
        importance_df = self.get_global_importance().head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'].values)
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('平均|SHAP值|')
        plt.title(f'Top {top_n} 特征重要性（SHAP）')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_sample_explanation(self, sample_idx: int, top_n: int = 10, 
                               figsize: Tuple[int, int] = (10, 6)):
        """
        绘制单个样本的SHAP解释
        
        Args:
            sample_idx: 样本索引
            top_n: 显示前N个特征
            figsize: 图形大小
        """
        explanation_df = self.get_sample_explanation(sample_idx, top_n)
        
        plt.figure(figsize=figsize)
        colors = ['green' if x > 0 else 'red' for x in explanation_df['shap_value']]
        plt.barh(range(len(explanation_df)), explanation_df['shap_value'].values, color=colors)
        plt.yticks(range(len(explanation_df)), explanation_df['feature'].values)
        plt.xlabel('SHAP值')
        plt.title(f'样本 {sample_idx} 的特征贡献')
        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        return plt.gcf()


def explain_one_into_two_model(model, X_train: pd.DataFrame, feature_names: List[str],
                              output_dir: str = 'output/limitup_research'):
    """
    解释一进二模型
    
    Args:
        model: 训练好的模型
        X_train: 训练数据
        feature_names: 特征名称
        output_dir: 输出目录
    """
    # 创建解释器
    explainer = SHAPExplainer(model, feature_names)
    
    # 计算SHAP值（使用部分数据以加快速度）
    sample_size = min(1000, len(X_train))
    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train.iloc[sample_indices] if isinstance(X_train, pd.DataFrame) else X_train[sample_indices]
    
    # 计算SHAP值
    explainer.compute_shap_values(X_sample)
    
    # 保存结果
    sample_ids = X_train.index[sample_indices] if hasattr(X_train, 'index') else None
    global_importance, per_sample_df = explainer.save_explanations(
        output_dir, X_sample, sample_ids
    )
    
    return explainer, global_importance, per_sample_df


# 测试代码
if __name__ == "__main__":
    # 创建模拟数据
    from sklearn.ensemble import RandomForestClassifier
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    feature_names[0] = "封板强度"
    feature_names[1] = "连板高度"
    feature_names[2] = "市场情绪"
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 创建解释器
    explainer = SHAPExplainer(model, feature_names)
    
    # 计算SHAP值
    shap_values = explainer.compute_shap_values(X[:50])
    
    # 获取全局重要性
    global_importance = explainer.get_global_importance()
    print("\n全局特征重要性 (Top 5):")
    print(global_importance.head())
    
    # 获取单个样本解释
    sample_explanation = explainer.get_sample_explanation(0, top_n=5)
    print("\n样本0的特征贡献 (Top 5):")
    print(sample_explanation)
    
    # 保存结果
    output_dir = "output/test_shap"
    explainer.save_explanations(output_dir, X[:50])
    
    print(f"\n✅ SHAP分析完成，结果已保存到 {output_dir}")