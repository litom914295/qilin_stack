"""
Walk-Forward验证框架

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务 1.3
目标：实现严格的样本外测试，避免过拟合

核心功能：
1. 滚动时间窗口回测
2. 严格的时间序列切分
3. 多指标性能评估
4. 稳定性分析（均值、标准差、最小值）
5. 回测结果可视化

作者：Qilin Quant Team
创建：2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class WalkForwardValidator:
    """Walk-Forward验证器"""
    
    def __init__(self, 
                 train_months: int = 12,
                 predict_months: int = 1,
                 step_months: int = 1):
        """
        初始化Walk-Forward验证器
        
        Args:
            train_months: 训练窗口长度（月）
            predict_months: 预测窗口长度（月）
            step_months: 滚动步长（月）
        """
        self.train_months = train_months
        self.predict_months = predict_months
        self.step_months = step_months
        
        # 评估指标历史
        self.metrics_history = []
        
        print(f"🔄 Walk-Forward验证器初始化")
        print(f"  训练窗口: {train_months}个月")
        print(f"  预测窗口: {predict_months}个月")
        print(f"  滚动步长: {step_months}个月")
    
    def generate_time_windows(self, 
                             start_date: str, 
                             end_date: str) -> List[Dict]:
        """
        生成时间窗口
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            List[Dict]: 时间窗口列表，每个窗口包含train_start, train_end, test_start, test_end
        """
        print(f"\n生成时间窗口: {start_date} -> {end_date}")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        windows = []
        current_start = start
        
        while True:
            # 训练集时间范围
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.train_months)
            
            # 测试集时间范围
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.predict_months)
            
            # 如果测试集结束时间超过总结束时间，停止
            if test_end > end:
                break
            
            window = {
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'window_id': len(windows) + 1
            }
            
            windows.append(window)
            
            # 向前滚动
            current_start = current_start + pd.DateOffset(months=self.step_months)
        
        print(f"✅ 生成 {len(windows)} 个时间窗口")
        
        return windows
    
    def validate(self, 
                model_class,
                X: pd.DataFrame,
                y: pd.Series,
                time_col: str = 'date',
                model_params: Dict = None,
                top_k: int = 20) -> pd.DataFrame:
        """
        执行Walk-Forward验证
        
        Args:
            model_class: 模型类（需实现fit和predict方法）
            X: 特征数据
            y: 标签数据
            time_col: 时间列名
            model_params: 模型参数
            top_k: Top K准确率的K值
        
        Returns:
            pd.DataFrame: 每个窗口的性能指标
        """
        print(f"\n开始Walk-Forward验证...")
        
        if model_params is None:
            model_params = {}
        
        # 确保数据按时间排序
        if time_col in X.columns:
            X = X.sort_values(time_col)
            y = y.loc[X.index]
        
        # 生成时间窗口
        start_date = X[time_col].min()
        end_date = X[time_col].max()
        windows = self.generate_time_windows(start_date, end_date)
        
        results = []
        
        for window in windows:
            print(f"\n窗口 {window['window_id']}: "
                  f"训练 {window['train_start']} ~ {window['train_end']}, "
                  f"测试 {window['test_start']} ~ {window['test_end']}")
            
            try:
                # 切分数据
                train_mask = (X[time_col] >= window['train_start']) & (X[time_col] < window['train_end'])
                test_mask = (X[time_col] >= window['test_start']) & (X[time_col] < window['test_end'])
                
                X_train = X[train_mask].drop(columns=[time_col] if time_col in X.columns else [])
                y_train = y[train_mask]
                
                X_test = X[test_mask].drop(columns=[time_col] if time_col in X.columns else [])
                y_test = y[test_mask]
                
                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"  ⚠️ 数据不足，跳过此窗口")
                    continue
                
                print(f"  训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
                
                # 训练模型
                model = model_class(**model_params)
                model.fit(X_train, y_train)
                
                # 预测
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = model.predict(X_test)
                
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # 计算指标
                metrics = self._calculate_metrics(
                    y_test, y_pred, y_pred_proba, top_k
                )
                
                # 记录窗口信息
                metrics.update(window)
                results.append(metrics)
                
                print(f"  AUC: {metrics['auc']:.4f}, "
                      f"P@{top_k}: {metrics[f'precision_at_{top_k}']:.4f}, "
                      f"Hit@{top_k}: {metrics[f'hit_at_{top_k}']:.4f}")
                
            except Exception as e:
                print(f"  ❌ 窗口 {window['window_id']} 验证失败: {e}")
                continue
        
        df_results = pd.DataFrame(results)
        self.metrics_history = results
        
        print(f"\n✅ Walk-Forward验证完成，共 {len(results)} 个窗口")
        
        return df_results
    
    def calculate_stability_metrics(self, df_results: pd.DataFrame) -> Dict:
        """
        计算稳定性指标
        
        Args:
            df_results: 验证结果DataFrame
        
        Returns:
            Dict: 稳定性统计
        """
        print("\n计算稳定性指标...")
        
        metrics_cols = [col for col in df_results.columns 
                       if col not in ['train_start', 'train_end', 'test_start', 'test_end', 'window_id']]
        
        stability = {}
        
        for metric in metrics_cols:
            values = df_results[metric].dropna()
            
            if len(values) == 0:
                continue
            
            stability[f'{metric}_mean'] = float(values.mean())
            stability[f'{metric}_std'] = float(values.std())
            stability[f'{metric}_min'] = float(values.min())
            stability[f'{metric}_max'] = float(values.max())
            stability[f'{metric}_median'] = float(values.median())
            
            # 计算变异系数（CV）
            if values.mean() != 0:
                stability[f'{metric}_cv'] = float(values.std() / abs(values.mean()))
        
        return stability
    
    def plot_metrics_over_time(self, 
                               df_results: pd.DataFrame,
                               save_path: str = None):
        """
        绘制指标时间序列图
        
        Args:
            df_results: 验证结果DataFrame
            save_path: 保存路径
        """
        print("\n绘制指标时间序列图...")
        
        key_metrics = ['auc', 'precision', 'recall', 'f1', 'precision_at_20', 'hit_at_20']
        available_metrics = [m for m in key_metrics if m in df_results.columns]
        
        if not available_metrics:
            print("⚠️ 无可用指标")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        # 使用test_start作为x轴
        x_dates = pd.to_datetime(df_results['test_start'])
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            values = df_results[metric].values
            
            # 绘制折线图
            ax.plot(x_dates, values, marker='o', linewidth=2, markersize=6, label=metric)
            
            # 添加均值线
            mean_val = values.mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', 
                      alpha=0.7, label=f'均值: {mean_val:.4f}')
            
            # 添加标准差区间
            std_val = values.std()
            ax.fill_between(x_dates, mean_val - std_val, mean_val + std_val,
                           alpha=0.2, color='gray', label=f'±1 std')
            
            ax.set_title(f'{metric.upper()} - 时间序列', fontsize=12, fontweight='bold')
            ax.set_xlabel('测试期开始日期', fontsize=10)
            ax.set_ylabel(metric.upper(), fontsize=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  图表已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, 
                       df_results: pd.DataFrame,
                       stability_metrics: Dict,
                       output_path: str = None) -> str:
        """
        生成验证报告
        
        Args:
            df_results: 验证结果DataFrame
            stability_metrics: 稳定性指标
            output_path: 输出路径
        
        Returns:
            str: 报告内容
        """
        print("\n生成Walk-Forward验证报告...")
        
        report_lines = []
        report_lines.append("# Walk-Forward验证报告\n\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**验证窗口数**: {len(df_results)}\n")
        report_lines.append(f"**训练窗口**: {self.train_months}个月\n")
        report_lines.append(f"**预测窗口**: {self.predict_months}个月\n")
        report_lines.append(f"**滚动步长**: {self.step_months}个月\n\n")
        
        # 整体性能概览
        report_lines.append("## 📊 整体性能概览\n\n")
        
        key_metrics = ['auc', 'precision', 'recall', 'f1', 'precision_at_20', 'hit_at_20']
        
        report_lines.append("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 变异系数 |\n")
        report_lines.append("|------|------|--------|--------|--------|----------|\n")
        
        for metric in key_metrics:
            if f'{metric}_mean' in stability_metrics:
                mean_val = stability_metrics[f'{metric}_mean']
                std_val = stability_metrics[f'{metric}_std']
                min_val = stability_metrics[f'{metric}_min']
                max_val = stability_metrics[f'{metric}_max']
                cv_val = stability_metrics.get(f'{metric}_cv', 0)
                
                report_lines.append(
                    f"| {metric.upper()} "
                    f"| {mean_val:.4f} "
                    f"| {std_val:.4f} "
                    f"| {min_val:.4f} "
                    f"| {max_val:.4f} "
                    f"| {cv_val:.4f} |\n"
                )
        
        report_lines.append("\n")
        
        # 稳定性评估
        report_lines.append("## 🎯 稳定性评估\n\n")
        
        auc_cv = stability_metrics.get('auc_cv', 0)
        
        if auc_cv < 0.05:
            stability_level = "优秀（CV < 0.05）"
            emoji = "🌟"
        elif auc_cv < 0.10:
            stability_level = "良好（CV < 0.10）"
            emoji = "✅"
        elif auc_cv < 0.15:
            stability_level = "一般（CV < 0.15）"
            emoji = "⚠️"
        else:
            stability_level = "较差（CV ≥ 0.15）"
            emoji = "❌"
        
        report_lines.append(f"{emoji} **稳定性等级**: {stability_level}\n\n")
        report_lines.append(f"- AUC变异系数: {auc_cv:.4f}\n")
        report_lines.append(f"- AUC标准差: {stability_metrics.get('auc_std', 0):.4f}\n\n")
        
        # 最佳/最差窗口
        report_lines.append("## 🏆 最佳窗口 vs ⚠️ 最差窗口\n\n")
        
        if 'auc' in df_results.columns:
            best_idx = df_results['auc'].idxmax()
            worst_idx = df_results['auc'].idxmin()
            
            best_window = df_results.loc[best_idx]
            worst_window = df_results.loc[worst_idx]
            
            report_lines.append("### 🏆 最佳窗口\n\n")
            report_lines.append(f"- **测试期**: {best_window['test_start']} ~ {best_window['test_end']}\n")
            report_lines.append(f"- **AUC**: {best_window['auc']:.4f}\n")
            report_lines.append(f"- **Precision**: {best_window.get('precision', 0):.4f}\n")
            report_lines.append(f"- **Recall**: {best_window.get('recall', 0):.4f}\n\n")
            
            report_lines.append("### ⚠️ 最差窗口\n\n")
            report_lines.append(f"- **测试期**: {worst_window['test_start']} ~ {worst_window['test_end']}\n")
            report_lines.append(f"- **AUC**: {worst_window['auc']:.4f}\n")
            report_lines.append(f"- **Precision**: {worst_window.get('precision', 0):.4f}\n")
            report_lines.append(f"- **Recall**: {worst_window.get('recall', 0):.4f}\n\n")
        
        # 详细结果表
        report_lines.append("## 📋 详细结果表\n\n")
        report_lines.append("| 窗口ID | 测试期 | AUC | Precision | Recall | F1 | P@20 | Hit@20 |\n")
        report_lines.append("|--------|--------|-----|-----------|--------|----|----- |--------|\n")
        
        for _, row in df_results.iterrows():
            test_period = f"{row['test_start'][:7]} ~ {row['test_end'][:7]}"
            report_lines.append(
                f"| {row['window_id']} "
                f"| {test_period} "
                f"| {row.get('auc', 0):.4f} "
                f"| {row.get('precision', 0):.4f} "
                f"| {row.get('recall', 0):.4f} "
                f"| {row.get('f1', 0):.4f} "
                f"| {row.get('precision_at_20', 0):.4f} "
                f"| {row.get('hit_at_20', 0):.4f} |\n"
            )
        
        report_content = "".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 报告已保存至: {output_path}")
        
        return report_content
    
    # ==================== 内部方法 ====================
    
    def _calculate_metrics(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray,
                          top_k: int = 20) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 基础指标
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc'] = 0.5
        
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Top K指标
        if len(y_pred_proba) >= top_k:
            # 选择预测概率最高的Top K
            top_k_idx = np.argsort(y_pred_proba)[-top_k:]
            
            # Precision@K
            precision_at_k = y_true[top_k_idx].sum() / top_k
            metrics[f'precision_at_{top_k}'] = precision_at_k
            
            # Hit@K (至少命中一个正样本)
            hit_at_k = 1.0 if y_true[top_k_idx].sum() > 0 else 0.0
            metrics[f'hit_at_{top_k}'] = hit_at_k
        else:
            metrics[f'precision_at_{top_k}'] = 0
            metrics[f'hit_at_{top_k}'] = 0
        
        return metrics


def main():
    """主函数 - 示例用法"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # 初始化验证器
    validator = WalkForwardValidator(
        train_months=12,
        predict_months=1,
        step_months=1
    )
    
    # 生成模拟数据
    print("\n生成模拟数据...")
    n_samples = 10000
    n_features = 50
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=10,
        random_state=42
    )
    
    # 添加时间列
    start_date = pd.to_datetime('2020-01-01')
    dates = pd.date_range(start_date, periods=n_samples, freq='D')
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    X_df['date'] = dates
    y_series = pd.Series(y)
    
    # 执行验证
    df_results = validator.validate(
        model_class=RandomForestClassifier,
        X=X_df,
        y=y_series,
        time_col='date',
        model_params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    )
    
    # 计算稳定性
    stability_metrics = validator.calculate_stability_metrics(df_results)
    
    print("\n" + "="*70)
    print("📊 稳定性指标")
    print("="*70)
    for k, v in stability_metrics.items():
        if 'mean' in k or 'std' in k or 'cv' in k:
            print(f"{k}: {v:.4f}")
    
    # 生成报告
    report_path = project_root / 'reports' / 'walk_forward_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validator.generate_report(df_results, stability_metrics, str(report_path))
    
    # 绘制图表
    plot_path = project_root / 'reports' / 'walk_forward_metrics.png'
    validator.plot_metrics_over_time(df_results, str(plot_path))
    
    print("\n✅ Walk-Forward验证演示完成！")


if __name__ == '__main__':
    main()
