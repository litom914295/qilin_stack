"""
SHAP可解释性模块
提供模型预测的SHAP值计算、全局/单样本解释、特征重要性可视化

支持：
- TreeExplainer (适用于树模型: XGBoost, LightGBM, CatBoost)
- 全局特征重要性分析
- 单样本预测解释
- 多种可视化图表 (summary_plot, waterfall, force_plot等)
- 批量样本解释和保存
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 尝试导入SHAP，如果不可用则提供降级方案
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP模型解释器
    用于解释树模型(XGBoost/LightGBM/CatBoost)的预测结果
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        output_dir: str = "./shap_output"
    ):
        """
        初始化SHAP解释器
        
        Args:
            model: 训练好的模型对象
            feature_names: 特征名称列表
            output_dir: 输出目录
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建explainer（使用TreeExplainer适用于树模型）
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception as e:
            warnings.warn(f"Failed to create TreeExplainer: {e}. Trying KernelExplainer...")
            # 降级到KernelExplainer（较慢但更通用）
            self.explainer = None
        
        self.shap_values = None
        self.base_value = None
    
    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            X: 特征数据
            check_additivity: 是否检查加性（调试用，较慢）
        
        Returns:
            SHAP值数组 shape: (n_samples, n_features) 或 (n_samples, n_features, n_classes)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized properly")
        
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        # 计算SHAP值
        self.shap_values = self.explainer.shap_values(
            X_array,
            check_additivity=check_additivity
        )
        
        # 获取base_value
        if hasattr(self.explainer, 'expected_value'):
            self.base_value = self.explainer.expected_value
        
        return self.shap_values
    
    def plot_summary(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        shap_values: Optional[np.ndarray] = None,
        plot_type: str = "dot",
        max_display: int = 20,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        绘制SHAP全局特征重要性汇总图
        
        Args:
            X: 特征数据
            shap_values: SHAP值（如果为None则使用已计算的）
            plot_type: 图表类型 ("dot", "bar", "violin")
            max_display: 最多显示的特征数量
            save_path: 保存路径（如果为None则自动生成）
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        # 处理多分类情况（取第一个类别或平均）
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if save_path is None:
            save_path = self.output_dir / f"shap_summary_{plot_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path)
    
    def plot_waterfall(
        self,
        sample_index: int,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        绘制单样本SHAP瀑布图（显示每个特征对预测的贡献）
        
        Args:
            sample_index: 样本索引
            X: 特征数据
            shap_values: SHAP值
            save_path: 保存路径
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed")
        
        # 处理多分类
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # 创建Explanation对象
        if isinstance(X, pd.DataFrame):
            X_sample = X.iloc[sample_index].values
        else:
            X_sample = X[sample_index] if X is not None else None
        
        explanation = shap.Explanation(
            values=shap_values[sample_index],
            base_values=self.base_value if not isinstance(self.base_value, np.ndarray) else self.base_value[0],
            data=X_sample,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(explanation, show=False)
        
        if save_path is None:
            save_path = self.output_dir / f"shap_waterfall_sample_{sample_index}.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path)
    
    def plot_force(
        self,
        sample_index: int,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        绘制单样本Force plot（交互式HTML）
        
        Args:
            sample_index: 样本索引
            X: 特征数据
            shap_values: SHAP值
            save_path: 保存路径
        
        Returns:
            保存的HTML文件路径
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed")
        
        # 处理多分类
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if isinstance(X, pd.DataFrame):
            X_sample = X.iloc[sample_index].values
        else:
            X_sample = X[sample_index] if X is not None else None
        
        base_value = self.base_value if not isinstance(self.base_value, np.ndarray) else self.base_value[0]
        
        force_plot = shap.force_plot(
            base_value,
            shap_values[sample_index],
            X_sample,
            feature_names=self.feature_names
        )
        
        if save_path is None:
            save_path = self.output_dir / f"shap_force_sample_{sample_index}.html"
        else:
            save_path = Path(save_path)
        
        shap.save_html(str(save_path), force_plot)
        
        return str(save_path)
    
    def get_feature_importance(
        self,
        shap_values: Optional[np.ndarray] = None,
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        获取全局特征重要性排名（基于SHAP值的绝对值均值）
        
        Args:
            shap_values: SHAP值
            top_k: 返回前k个重要特征
        
        Returns:
            特征重要性DataFrame
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed")
        
        # 处理多分类
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # 计算特征重要性（SHAP值绝对值的均值）
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'f{i}' for i in range(len(importance))],
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_k)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
    
    def explain_sample(
        self,
        sample_index: int,
        X: Union[pd.DataFrame, np.ndarray],
        shap_values: Optional[np.ndarray] = None,
        top_k: int = 10
    ) -> Dict:
        """
        解释单个样本的预测
        
        Args:
            sample_index: 样本索引
            X: 特征数据
            shap_values: SHAP值
            top_k: 返回前k个影响最大的特征
        
        Returns:
            解释结果字典
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not computed")
        
        # 处理多分类
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        sample_shap = shap_values[sample_index]
        
        if isinstance(X, pd.DataFrame):
            sample_values = X.iloc[sample_index].values
        else:
            sample_values = X[sample_index]
        
        # 按SHAP值绝对值排序
        abs_shap = np.abs(sample_shap)
        top_indices = np.argsort(abs_shap)[::-1][:top_k]
        
        explanation = {
            'sample_index': sample_index,
            'base_value': float(self.base_value if not isinstance(self.base_value, np.ndarray) else self.base_value[0]),
            'prediction': float(self.base_value + sample_shap.sum()) if not isinstance(self.base_value, np.ndarray) else float(self.base_value[0] + sample_shap.sum()),
            'top_features': []
        }
        
        for idx in top_indices:
            feature_name = self.feature_names[idx] if self.feature_names else f'f{idx}'
            explanation['top_features'].append({
                'feature': feature_name,
                'value': float(sample_values[idx]),
                'shap_value': float(sample_shap[idx]),
                'contribution': 'positive' if sample_shap[idx] > 0 else 'negative'
            })
        
        return explanation
    
    def save_explainer(self, save_path: Optional[str] = None):
        """保存explainer对象到磁盘"""
        if save_path is None:
            save_path = self.output_dir / "shap_explainer.pkl"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'explainer': self.explainer,
                'feature_names': self.feature_names,
                'base_value': self.base_value
            }, f)
        
        return str(save_path)
    
    @classmethod
    def load_explainer(cls, load_path: str, model) -> 'SHAPExplainer':
        """从磁盘加载explainer对象"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        explainer_obj = cls(model, feature_names=data['feature_names'])
        explainer_obj.explainer = data['explainer']
        explainer_obj.base_value = data['base_value']
        
        return explainer_obj


def explain_model_predictions(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    output_dir: str = "./shap_output",
    sample_indices: Optional[List[int]] = None,
    top_k_features: int = 20
) -> Tuple[SHAPExplainer, Dict]:
    """
    一站式模型解释函数
    
    Args:
        model: 训练好的模型
        X: 特征数据
        feature_names: 特征名称
        output_dir: 输出目录
        sample_indices: 需要详细解释的样本索引列表
        top_k_features: 显示前k个重要特征
    
    Returns:
        (explainer对象, 结果字典)
    """
    if not SHAP_AVAILABLE:
        return None, {'error': 'SHAP not available'}
    
    # 创建explainer
    explainer = SHAPExplainer(model, feature_names=feature_names, output_dir=output_dir)
    
    # 计算SHAP值
    print("计算SHAP值...")
    explainer.compute_shap_values(X)
    
    results = {
        'output_dir': output_dir,
        'plots': {},
        'feature_importance': None,
        'sample_explanations': []
    }
    
    # 绘制汇总图
    print("生成汇总图...")
    results['plots']['summary_dot'] = explainer.plot_summary(X, plot_type='dot', max_display=top_k_features)
    results['plots']['summary_bar'] = explainer.plot_summary(X, plot_type='bar', max_display=top_k_features)
    
    # 特征重要性
    print("计算特征重要性...")
    results['feature_importance'] = explainer.get_feature_importance(top_k=top_k_features)
    
    # 样本级解释
    if sample_indices:
        print(f"生成 {len(sample_indices)} 个样本的详细解释...")
        for idx in sample_indices:
            waterfall_path = explainer.plot_waterfall(idx, X)
            force_path = explainer.plot_force(idx, X)
            explanation = explainer.explain_sample(idx, X, top_k=10)
            
            results['sample_explanations'].append({
                'sample_index': idx,
                'waterfall_plot': waterfall_path,
                'force_plot': force_path,
                'explanation': explanation
            })
    
    # 保存explainer
    explainer.save_explainer()
    
    print(f"解释完成！结果已保存到: {output_dir}")
    
    return explainer, results


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试SHAP解释器"""
    
    print("=" * 60)
    print("SHAP可解释性模块测试")
    print("=" * 60)
    
    if not SHAP_AVAILABLE:
        print("SHAP库未安装，请运行: pip install shap")
        exit(1)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 生成标签（模拟非线性关系）
    y = (
        X['feature_0'] * 2 +
        X['feature_1'] ** 2 +
        np.sin(X['feature_2']) * 3 +
        np.random.randn(n_samples) * 0.5
    ) > 0
    y = y.astype(int)
    
    print(f"生成模拟数据: {X.shape}, 正样本比例: {y.mean():.2%}")
    
    # 训练一个简单的模型
    try:
        import lightgbm as lgb
        
        print("\n训练LightGBM模型...")
        model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)
        model.fit(X, y)
        
        print("模型训练完成")
        
        # 使用一站式解释函数
        print("\n" + "=" * 60)
        print("开始模型解释...")
        print("=" * 60)
        
        explainer, results = explain_model_predictions(
            model=model,
            X=X,
            feature_names=X.columns.tolist(),
            output_dir="./test_shap_output",
            sample_indices=[0, 10, 50],  # 解释3个样本
            top_k_features=15
        )
        
        # 打印特征重要性
        print("\n" + "=" * 60)
        print("Top 10 特征重要性:")
        print("=" * 60)
        print(results['feature_importance'].head(10).to_string(index=False))
        
        # 打印样本解释
        print("\n" + "=" * 60)
        print("样本解释示例 (样本0):")
        print("=" * 60)
        sample_exp = results['sample_explanations'][0]['explanation']
        print(f"Base Value: {sample_exp['base_value']:.4f}")
        print(f"Prediction: {sample_exp['prediction']:.4f}")
        print("\nTop 5 影响特征:")
        for feat in sample_exp['top_features'][:5]:
            print(f"  {feat['feature']:15s}: value={feat['value']:8.3f}, "
                  f"SHAP={feat['shap_value']:8.3f} ({feat['contribution']})")
        
        print("\n" + "=" * 60)
        print(f"所有图表已保存到: {results['output_dir']}")
        print("=" * 60)
        
    except ImportError:
        print("LightGBM未安装，无法运行测试。请安装: pip install lightgbm")
