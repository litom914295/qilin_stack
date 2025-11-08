# -*- coding: utf-8 -*-
"""
模型可解释性分析模块 - SHAP分析

功能：
1. 特征重要性分析
2. 单样本预测归因
3. 特征交互分析
4. 可视化生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union
import shap
import logging
from pathlib import Path
import joblib
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class ModelExplainer:
    """模型解释器 - 基于SHAP"""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        初始化模型解释器
        
        Args:
            model: 训练好的模型（支持LightGBM, XGBoost, 随机森林等）
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        # 初始化SHAP解释器
        self._initialize_explainer()
        
    def _initialize_explainer(self):
        """初始化SHAP解释器"""
        try:
            # 尝试使用TreeExplainer（适用于树模型）
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("使用TreeExplainer")
        except:
            try:
                # 如果不是树模型，使用KernelExplainer
                # 需要提供背景数据
                logger.warning("TreeExplainer失败，需要提供背景数据使用KernelExplainer")
            except Exception as e:
                logger.error(f"初始化SHAP解释器失败: {e}")
                
    def explain_predictions(
        self,
        X: pd.DataFrame,
        sample_indices: Optional[List[int]] = None
    ) -> Dict:
        """
        解释模型预测
        
        Args:
            X: 输入特征数据
            sample_indices: 要解释的样本索引（None表示全部）
            
        Returns:
            解释结果字典
        """
        
        if sample_indices is not None:
            X_explain = X.iloc[sample_indices]
        else:
            X_explain = X
            
        # 计算SHAP值
        self.shap_values = self.explainer.shap_values(X_explain)
        
        # 如果是二分类，取正类的SHAP值
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]
            
        # 计算特征重要性
        feature_importance = self._calculate_feature_importance()
        
        # 生成解释结果
        results = {
            'shap_values': self.shap_values,
            'feature_importance': feature_importance,
            'base_value': self.explainer.expected_value,
            'feature_names': self.feature_names
        }
        
        return results
    
    def _calculate_feature_importance(self) -> pd.DataFrame:
        """计算特征重要性"""
        
        # 计算每个特征的平均绝对SHAP值
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # 创建DataFrame
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 计算累积重要性
        df_importance['cumulative_importance'] = df_importance['importance'].cumsum()
        df_importance['importance_pct'] = df_importance['importance'] / df_importance['importance'].sum() * 100
        
        return df_importance
    
    def explain_single_prediction(
        self,
        X_single: pd.DataFrame,
        prediction: float,
        true_label: Optional[float] = None
    ) -> Dict:
        """
        解释单个样本的预测
        
        Args:
            X_single: 单个样本的特征
            prediction: 模型预测值
            true_label: 真实标签（可选）
            
        Returns:
            单样本解释结果
        """
        
        # 计算SHAP值
        shap_values_single = self.explainer.shap_values(X_single)[0]
        
        # 如果是二分类，取正类的SHAP值
        if isinstance(shap_values_single, list) and len(shap_values_single) == 2:
            shap_values_single = shap_values_single[1]
            
        # 创建特征贡献DataFrame
        df_contribution = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_single.values[0],
            'shap_value': shap_values_single,
            'abs_shap_value': np.abs(shap_values_single)
        }).sort_values('abs_shap_value', ascending=False)
        
        # 识别正负贡献
        df_contribution['contribution'] = df_contribution['shap_value'].apply(
            lambda x: '正向' if x > 0 else '负向'
        )
        
        # 计算预测分解
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]  # 二分类取正类
            
        result = {
            'prediction': prediction,
            'true_label': true_label,
            'base_value': base_value,
            'contributions': df_contribution,
            'top_positive': df_contribution[df_contribution['shap_value'] > 0].head(5),
            'top_negative': df_contribution[df_contribution['shap_value'] < 0].head(5)
        }
        
        return result
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个重要特征
            save_path: 保存路径
        """
        
        if self.shap_values is None:
            logger.error("请先调用explain_predictions计算SHAP值")
            return
            
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. SHAP Summary Plot
        plt.sca(axes[0])
        shap.summary_plot(
            self.shap_values[:, :top_n],
            feature_names=self.feature_names[:top_n],
            show=False,
            plot_type="bar"
        )
        axes[0].set_title("特征重要性（SHAP）")
        
        # 2. 累积重要性曲线
        df_importance = self._calculate_feature_importance().head(top_n)
        axes[1].plot(
            range(len(df_importance)),
            df_importance['cumulative_importance'].values,
            'b-',
            marker='o'
        )
        axes[1].set_xlabel("特征排名")
        axes[1].set_ylabel("累积重要性")
        axes[1].set_title("累积特征重要性")
        axes[1].grid(True, alpha=0.3)
        
        # 标记90%重要性线
        cumsum = df_importance['importance'].cumsum() / df_importance['importance'].sum()
        n_90 = (cumsum <= 0.9).sum()
        axes[1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
        axes[1].text(n_90, 0.9, f'90%({n_90}个特征)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"特征重要性图已保存至: {save_path}")
        
        plt.show()
        
    def plot_single_prediction(
        self,
        explanation: Dict,
        save_path: Optional[str] = None
    ):
        """
        绘制单样本预测解释图
        
        Args:
            explanation: explain_single_prediction的返回结果
            save_path: 保存路径
        """
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Waterfall图（显示预测分解）
        df_contrib = explanation['contributions'].head(15)
        
        # 计算累积值
        cumsum = explanation['base_value'] + df_contrib['shap_value'].cumsum()
        
        axes[0].barh(
            range(len(df_contrib)),
            df_contrib['shap_value'].values,
            color=['green' if x > 0 else 'red' for x in df_contrib['shap_value']]
        )
        axes[0].set_yticks(range(len(df_contrib)))
        axes[0].set_yticklabels(
            [f"{feat}: {val:.2f}" for feat, val in 
             zip(df_contrib['feature'], df_contrib['value'])]
        )
        axes[0].set_xlabel("SHAP值（贡献度）")
        axes[0].set_title(
            f"预测分解 - 预测值: {explanation['prediction']:.3f}, "
            f"真实值: {explanation.get('true_label', 'N/A')}"
        )
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 2. 正负贡献对比
        top_pos = explanation['top_positive']
        top_neg = explanation['top_negative']
        
        # 合并正负贡献
        all_features = list(top_pos['feature']) + list(top_neg['feature'])
        all_values = list(top_pos['shap_value']) + list(top_neg['shap_value'])
        all_colors = ['green'] * len(top_pos) + ['red'] * len(top_neg)
        
        axes[1].barh(range(len(all_features)), all_values, color=all_colors)
        axes[1].set_yticks(range(len(all_features)))
        axes[1].set_yticklabels(all_features)
        axes[1].set_xlabel("SHAP值")
        axes[1].set_title("主要正负贡献特征")
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"单样本解释图已保存至: {save_path}")
            
        plt.show()
        
    def plot_feature_interaction(
        self,
        feature1: str,
        feature2: str,
        X: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        绘制特征交互图
        
        Args:
            feature1: 第一个特征名
            feature2: 第二个特征名
            X: 特征数据
            save_path: 保存路径
        """
        
        if self.shap_values is None:
            logger.error("请先调用explain_predictions计算SHAP值")
            return
            
        # 获取特征索引
        idx1 = self.feature_names.index(feature1)
        idx2 = self.feature_names.index(feature2)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制交互图
        shap.dependence_plot(
            idx1,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=idx2,
            ax=ax,
            show=False
        )
        
        ax.set_title(f"特征交互: {feature1} vs {feature2}")
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"特征交互图已保存至: {save_path}")
            
        plt.show()
        
    def generate_report(self, output_path: str) -> Dict:
        """
        生成完整的模型解释报告
        
        Args:
            output_path: 报告输出路径
            
        Returns:
            报告内容字典
        """
        
        if self.shap_values is None:
            logger.error("请先调用explain_predictions计算SHAP值")
            return {}
            
        # 计算各种统计
        feature_importance = self._calculate_feature_importance()
        
        # 找出最重要的特征组合
        top_features = feature_importance.head(10)['feature'].tolist()
        
        # 生成报告
        report = {
            'summary': {
                'total_features': len(self.feature_names),
                'samples_explained': len(self.shap_values),
                'base_value': float(self.explainer.expected_value) if not isinstance(
                    self.explainer.expected_value, list
                ) else float(self.explainer.expected_value[1])
            },
            'top_features': {
                'names': top_features,
                'importance': feature_importance.head(10)['importance'].tolist(),
                'cumulative_importance': feature_importance.head(10)['cumulative_importance'].tolist()
            },
            'feature_stats': {
                'mean_abs_shap': np.abs(self.shap_values).mean(axis=0).tolist(),
                'std_shap': self.shap_values.std(axis=0).tolist(),
                'max_abs_shap': np.abs(self.shap_values).max(axis=0).tolist()
            }
        }
        
        # 保存报告
        output_file = Path(output_path) / "model_explanation_report.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info(f"模型解释报告已生成: {output_file}")
        
        return report


class LimitUpModelExplainer(ModelExplainer):
    """涨停板模型专用解释器"""
    
    def __init__(self, model: Any):
        """
        初始化涨停板模型解释器
        
        Args:
            model: 训练好的涨停预测模型
        """
        # 涨停板策略的特征名称
        feature_names = [
            '封单强度', '打开次数', '涨停时间得分', '连板高度',
            '市场情绪', '龙头地位', '大单流入比例', '题材热度衰减',
            '换手率', '量比', '5日涨幅', '板块涨停数',
            '北向资金', '游资活跃度', '成交额', '流通市值'
        ]
        
        super().__init__(model, feature_names)
        
    def explain_limitup_prediction(
        self,
        stock_code: str,
        features: pd.DataFrame,
        prediction_prob: float
    ) -> Dict:
        """
        解释涨停预测
        
        Args:
            stock_code: 股票代码
            features: 特征数据
            prediction_prob: 预测的涨停概率
            
        Returns:
            涨停预测解释
        """
        
        # 获取基础解释
        base_explanation = self.explain_single_prediction(
            features, prediction_prob
        )
        
        # 添加涨停特定解释
        contrib = base_explanation['contributions']
        
        # 识别关键因素
        key_factors = {
            '封板因素': contrib[contrib['feature'].isin(['封单强度', '打开次数', '涨停时间得分'])]['shap_value'].sum(),
            '市场因素': contrib[contrib['feature'].isin(['市场情绪', '板块涨停数', '游资活跃度'])]['shap_value'].sum(),
            '资金因素': contrib[contrib['feature'].isin(['大单流入比例', '北向资金', '换手率'])]['shap_value'].sum(),
            '题材因素': contrib[contrib['feature'].isin(['题材热度衰减', '龙头地位', '连板高度'])]['shap_value'].sum()
        }
        
        # 生成文字解释
        text_explanation = self._generate_text_explanation(
            stock_code, prediction_prob, key_factors, contrib
        )
        
        result = {
            **base_explanation,
            'stock_code': stock_code,
            'key_factors': key_factors,
            'text_explanation': text_explanation,
            'recommendation': self._generate_recommendation(prediction_prob, key_factors)
        }
        
        return result
        
    def _generate_text_explanation(
        self,
        stock_code: str,
        prediction_prob: float,
        key_factors: Dict,
        contributions: pd.DataFrame
    ) -> str:
        """生成文字解释"""
        
        # 找出最重要的正负因素
        top_positive = contributions[contributions['shap_value'] > 0].head(3)
        top_negative = contributions[contributions['shap_value'] < 0].head(3)
        
        # 找出影响最大的因素类别
        max_factor = max(key_factors, key=lambda k: abs(key_factors[k]))
        
        explanation = f"""
【{stock_code} 涨停概率分析】

预测概率: {prediction_prob:.2%}

主要支撑因素:
"""
        
        for _, row in top_positive.iterrows():
            explanation += f"- {row['feature']}: {row['value']:.2f} (贡献 +{row['shap_value']:.3f})\n"
            
        explanation += "\n主要阻碍因素:\n"
        
        for _, row in top_negative.iterrows():
            explanation += f"- {row['feature']}: {row['value']:.2f} (贡献 {row['shap_value']:.3f})\n"
            
        explanation += f"\n关键影响: {max_factor}因素影响最大 ({key_factors[max_factor]:.3f})"
        
        return explanation
        
    def _generate_recommendation(
        self,
        prediction_prob: float,
        key_factors: Dict
    ) -> str:
        """生成操作建议"""
        
        if prediction_prob > 0.7:
            if key_factors['封板因素'] > 0.2:
                return "强烈推荐：封板强势，建议积极排队"
            else:
                return "推荐：概率较高，但需关注封板情况"
        elif prediction_prob > 0.5:
            if key_factors['市场因素'] > 0.1:
                return "谨慎推荐：市场环境有利，可小仓位尝试"
            else:
                return "观望：概率中等，建议等待更好机会"
        else:
            return "不推荐：涨停概率较低，建议回避"


def test_model_explainer():
    """测试模型解释器"""
    
    # 创建模拟模型和数据
    from sklearn.ensemble import RandomForestClassifier
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100
    n_features = 16
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'特征{i+1}' for i in range(n_features)]
    )
    y = (X.sum(axis=1) > 0).astype(int)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # 创建涨停板解释器
    explainer = LimitUpModelExplainer(model)
    
    # 解释预测
    results = explainer.explain_predictions(X.head(10))
    print("特征重要性:")
    print(results['feature_importance'].head())
    
    # 解释单个预测
    single_result = explainer.explain_limitup_prediction(
        '000001',
        X.head(1),
        model.predict_proba(X.head(1))[0, 1]
    )
    print("\n单样本解释:")
    print(single_result['text_explanation'])
    print(f"\n操作建议: {single_result['recommendation']}")
    
    # 绘制图表
    explainer.plot_feature_importance(top_n=10)
    explainer.plot_single_prediction(single_result)
    
    return explainer


if __name__ == "__main__":
    explainer = test_model_explainer()