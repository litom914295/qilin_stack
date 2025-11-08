"""
数据和特征漂移监测模块
检测特征分布变化、计算PSI (Population Stability Index)、发出漂移预警

功能：
- 计算特征级别的PSI指标
- 检测数据分布漂移
- 生成漂移报告和可视化
- 设置漂移阈值告警
- 支持连续型和离散型特征
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DriftDetector:
    """
    数据漂移检测器
    用于监测特征分布变化和模型输入稳定性
    """
    
    def __init__(
        self,
        baseline_data: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        n_bins: int = 10,
        output_dir: str = "./drift_monitoring"
    ):
        """
        初始化漂移检测器
        
        Args:
            baseline_data: 基线数据（训练集或参考数据）
            feature_cols: 特征列名列表（如果None则使用所有列）
            n_bins: 连续特征分箱数量
            output_dir: 输出目录
        """
        self.baseline_data = baseline_data
        self.feature_cols = feature_cols
        self.n_bins = n_bins
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储基线统计信息
        self.baseline_stats = {}
        self.baseline_bins = {}
        
        if baseline_data is not None:
            self._compute_baseline_stats()
    
    def set_baseline(self, baseline_data: pd.DataFrame):
        """设置基线数据"""
        self.baseline_data = baseline_data
        
        if self.feature_cols is None:
            self.feature_cols = baseline_data.columns.tolist()
        
        self._compute_baseline_stats()
    
    def _compute_baseline_stats(self):
        """计算基线统计信息"""
        if self.baseline_data is None:
            return
        
        if self.feature_cols is None:
            self.feature_cols = self.baseline_data.columns.tolist()
        
        for col in self.feature_cols:
            try:
                # 计算基本统计量
                self.baseline_stats[col] = {
                    'mean': self.baseline_data[col].mean(),
                    'std': self.baseline_data[col].std(),
                    'min': self.baseline_data[col].min(),
                    'max': self.baseline_data[col].max(),
                    'median': self.baseline_data[col].median(),
                    'missing_rate': self.baseline_data[col].isna().mean()
                }
                
                # 为连续特征计算分箱边界
                if self.baseline_data[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                    # 使用分位数分箱，避免极端值影响
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    bins = self.baseline_data[col].quantile(quantiles).values
                    # 去重并排序
                    bins = np.unique(bins)
                    self.baseline_bins[col] = bins
                
            except Exception as e:
                warnings.warn(f"Failed to compute baseline stats for {col}: {e}")
    
    def compute_psi(
        self,
        current_data: pd.DataFrame,
        feature_col: str
    ) -> float:
        """
        计算单个特征的PSI (Population Stability Index)
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        
        PSI < 0.1: 无明显变化
        0.1 <= PSI < 0.25: 中等变化
        PSI >= 0.25: 显著变化
        
        Args:
            current_data: 当前数据
            feature_col: 特征列名
        
        Returns:
            PSI值
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Call set_baseline() first.")
        
        if feature_col not in self.baseline_bins:
            warnings.warn(f"Feature {feature_col} not in baseline. Skipping PSI calculation.")
            return np.nan
        
        # 获取分箱边界
        bins = self.baseline_bins[feature_col]
        
        # 计算基线分布
        baseline_dist, _ = np.histogram(
            self.baseline_data[feature_col].dropna(),
            bins=bins
        )
        baseline_dist = baseline_dist / baseline_dist.sum()
        
        # 计算当前分布
        current_dist, _ = np.histogram(
            current_data[feature_col].dropna(),
            bins=bins
        )
        current_dist = current_dist / current_dist.sum()
        
        # 避免除零和log(0)
        epsilon = 1e-10
        baseline_dist = np.clip(baseline_dist, epsilon, None)
        current_dist = np.clip(current_dist, epsilon, None)
        
        # 计算PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
        
        return psi
    
    def compute_all_psi(
        self,
        current_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算所有特征的PSI
        
        Args:
            current_data: 当前数据
            feature_cols: 特征列名列表（如果None则使用所有特征）
        
        Returns:
            PSI结果DataFrame
        """
        if feature_cols is None:
            feature_cols = self.feature_cols
        
        psi_results = []
        
        for col in feature_cols:
            try:
                psi = self.compute_psi(current_data, col)
                
                # 判断漂移等级
                if psi < 0.1:
                    level = "无漂移"
                elif psi < 0.25:
                    level = "中等漂移"
                else:
                    level = "显著漂移"
                
                psi_results.append({
                    'feature': col,
                    'psi': psi,
                    'drift_level': level
                })
            except Exception as e:
                warnings.warn(f"Failed to compute PSI for {col}: {e}")
        
        return pd.DataFrame(psi_results).sort_values('psi', ascending=False)
    
    def detect_distribution_shift(
        self,
        current_data: pd.DataFrame,
        feature_col: str,
        method: str = 'ks'
    ) -> Dict:
        """
        使用统计检验检测分布偏移
        
        Args:
            current_data: 当前数据
            feature_col: 特征列名
            method: 检验方法 ('ks' for Kolmogorov-Smirnov, 't' for t-test)
        
        Returns:
            检验结果字典
        """
        from scipy import stats
        
        baseline_values = self.baseline_data[feature_col].dropna().values
        current_values = current_data[feature_col].dropna().values
        
        result = {'feature': feature_col, 'method': method}
        
        if method == 'ks':
            # Kolmogorov-Smirnov 检验（适用于连续分布）
            statistic, p_value = stats.ks_2samp(baseline_values, current_values)
            result['statistic'] = statistic
            result['p_value'] = p_value
            result['significant'] = p_value < 0.05
        
        elif method == 't':
            # t检验（检验均值差异）
            statistic, p_value = stats.ttest_ind(baseline_values, current_values)
            result['statistic'] = statistic
            result['p_value'] = p_value
            result['significant'] = p_value < 0.05
        
        return result
    
    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        psi_threshold: float = 0.1,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        生成完整的漂移报告
        
        Args:
            current_data: 当前数据
            feature_cols: 特征列名列表
            psi_threshold: PSI告警阈值
            save_path: 报告保存路径
        
        Returns:
            漂移报告字典
        """
        if feature_cols is None:
            feature_cols = self.feature_cols
        
        # 计算所有PSI
        psi_df = self.compute_all_psi(current_data, feature_cols)
        
        # 识别漂移特征
        drifted_features = psi_df[psi_df['psi'] >= psi_threshold]['feature'].tolist()
        
        # 计算统计差异
        stat_tests = []
        for col in feature_cols[:20]:  # 限制数量避免过慢
            try:
                test_result = self.detect_distribution_shift(current_data, col, method='ks')
                stat_tests.append(test_result)
            except:
                pass
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_features': len(feature_cols),
            'n_drifted': len(drifted_features),
            'drift_rate': len(drifted_features) / len(feature_cols) if len(feature_cols) > 0 else 0,
            'drifted_features': drifted_features,
            'psi_summary': {
                'mean': float(psi_df['psi'].mean()),
                'max': float(psi_df['psi'].max()),
                'min': float(psi_df['psi'].min())
            },
            'top_drifted': psi_df.head(10).to_dict('records'),
            'statistical_tests': stat_tests[:10]  # 只保存前10个
        }
        
        # 保存报告
        if save_path is None:
            save_path = self.output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def plot_feature_distribution(
        self,
        current_data: pd.DataFrame,
        feature_col: str,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        绘制特征分布对比图
        
        Args:
            current_data: 当前数据
            feature_col: 特征列名
            save_path: 保存路径
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图对比
        axes[0].hist(self.baseline_data[feature_col].dropna(), bins=30, alpha=0.6, label='Baseline', density=True)
        axes[0].hist(current_data[feature_col].dropna(), bins=30, alpha=0.6, label='Current', density=True)
        axes[0].set_xlabel(feature_col)
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Distribution Comparison: {feature_col}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 累积分布函数对比
        baseline_sorted = np.sort(self.baseline_data[feature_col].dropna())
        current_sorted = np.sort(current_data[feature_col].dropna())
        
        baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
        current_cdf = np.arange(1, len(current_sorted) + 1) / len(current_sorted)
        
        axes[1].plot(baseline_sorted, baseline_cdf, label='Baseline', linewidth=2)
        axes[1].plot(current_sorted, current_cdf, label='Current', linewidth=2)
        axes[1].set_xlabel(feature_col)
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title(f'CDF Comparison: {feature_col}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # 添加PSI信息
        psi = self.compute_psi(current_data, feature_col)
        fig.suptitle(f'PSI = {psi:.4f}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"drift_dist_{feature_col}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path)
    
    def plot_psi_heatmap(
        self,
        psi_df: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = False,
        top_k: int = 30
    ) -> str:
        """
        绘制PSI热力图
        
        Args:
            psi_df: PSI结果DataFrame
            save_path: 保存路径
            show: 是否显示图表
            top_k: 显示前k个特征
        
        Returns:
            保存的文件路径
        """
        # 选择前k个PSI最高的特征
        top_psi = psi_df.head(top_k).copy()
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_psi) * 0.3)))
        
        # 创建颜色映射
        colors = []
        for psi in top_psi['psi']:
            if psi < 0.1:
                colors.append('green')
            elif psi < 0.25:
                colors.append('orange')
            else:
                colors.append('red')
        
        # 绘制横向条形图
        bars = ax.barh(range(len(top_psi)), top_psi['psi'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_psi)))
        ax.set_yticklabels(top_psi['feature'])
        ax.set_xlabel('PSI Value')
        ax.set_title(f'Top {len(top_psi)} Features by PSI')
        
        # 添加参考线
        ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='PSI=0.1 (threshold)')
        ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='PSI=0.25 (high drift)')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # 在条形上添加数值标签
        for i, (idx, row) in enumerate(top_psi.iterrows()):
            ax.text(row['psi'] + 0.01, i, f"{row['psi']:.3f}", 
                   va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "psi_heatmap.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path)
    
    def monitor_drift_over_time(
        self,
        data_batches: List[Tuple[str, pd.DataFrame]],
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        监测多个时间批次的漂移趋势
        
        Args:
            data_batches: [(时间标签, 数据), ...] 列表
            feature_cols: 特征列名列表
        
        Returns:
            时间序列PSI DataFrame
        """
        if feature_cols is None:
            feature_cols = self.feature_cols[:10]  # 限制特征数量
        
        time_series_psi = []
        
        for time_label, data in data_batches:
            psi_dict = {'timestamp': time_label}
            for col in feature_cols:
                try:
                    psi = self.compute_psi(data, col)
                    psi_dict[col] = psi
                except:
                    psi_dict[col] = np.nan
            
            time_series_psi.append(psi_dict)
        
        return pd.DataFrame(time_series_psi)
    
    def plot_drift_timeline(
        self,
        time_series_psi: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        绘制漂移时间序列图
        
        Args:
            time_series_psi: 时间序列PSI DataFrame
            save_path: 保存路径
            show: 是否显示图表
        
        Returns:
            保存的文件路径
        """
        feature_cols = [col for col in time_series_psi.columns if col != 'timestamp']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in feature_cols:
            ax.plot(time_series_psi['timestamp'], time_series_psi[col], 
                   marker='o', label=col, linewidth=2, markersize=5)
        
        # 添加阈值线
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Low drift threshold')
        ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='High drift threshold')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('PSI')
        ax.set_title('Feature Drift Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "drift_timeline.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path)


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试漂移检测器"""
    
    print("=" * 60)
    print("数据漂移监测模块测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # 基线数据（训练集）
    baseline_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    print(f"生成基线数据: {baseline_data.shape}")
    
    # 创建漂移检测器
    detector = DriftDetector(
        baseline_data=baseline_data,
        n_bins=10,
        output_dir="./test_drift_output"
    )
    
    print("✓ 漂移检测器已初始化")
    
    # 模拟当前数据（有漂移）
    print("\n" + "=" * 60)
    print("生成漂移数据...")
    print("=" * 60)
    
    current_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 对部分特征施加漂移
    current_data['feature_0'] = current_data['feature_0'] + 0.5  # 均值漂移
    current_data['feature_1'] = current_data['feature_1'] * 2    # 方差漂移
    current_data['feature_2'] = current_data['feature_2'] + np.random.randn(n_samples) * 2  # 噪声增大
    
    print("✓ 已生成包含漂移的当前数据")
    
    # 计算PSI
    print("\n" + "=" * 60)
    print("计算PSI...")
    print("=" * 60)
    
    psi_df = detector.compute_all_psi(current_data)
    print("\nTop 10 PSI:")
    print(psi_df.head(10).to_string(index=False))
    
    # 生成漂移报告
    print("\n" + "=" * 60)
    print("生成漂移报告...")
    print("=" * 60)
    
    report = detector.generate_drift_report(
        current_data,
        psi_threshold=0.1
    )
    
    print(f"\n漂移特征数量: {report['n_drifted']} / {report['n_features']}")
    print(f"漂移率: {report['drift_rate']:.2%}")
    print(f"平均PSI: {report['psi_summary']['mean']:.4f}")
    print(f"最大PSI: {report['psi_summary']['max']:.4f}")
    
    # 可视化
    print("\n" + "=" * 60)
    print("生成可视化...")
    print("=" * 60)
    
    # PSI热力图
    heatmap_path = detector.plot_psi_heatmap(psi_df, top_k=15)
    print(f"✓ PSI热力图已保存: {heatmap_path}")
    
    # 单特征分布对比
    dist_path = detector.plot_feature_distribution(current_data, 'feature_0')
    print(f"✓ 特征分布图已保存: {dist_path}")
    
    # 时间序列监测
    print("\n" + "=" * 60)
    print("测试时间序列监测...")
    print("=" * 60)
    
    # 生成多个时间批次
    data_batches = []
    for i in range(5):
        batch_data = pd.DataFrame(
            np.random.randn(500, n_features) + i * 0.1,  # 逐渐漂移
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        data_batches.append((f"T{i}", batch_data))
    
    time_series_psi = detector.monitor_drift_over_time(data_batches, feature_cols=['feature_0', 'feature_1', 'feature_2'])
    print("\n时间序列PSI:")
    print(time_series_psi.to_string(index=False))
    
    # 绘制时间序列
    timeline_path = detector.plot_drift_timeline(time_series_psi)
    print(f"\n✓ 漂移时间序列图已保存: {timeline_path}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"所有结果已保存到: {detector.output_dir}")
    print("=" * 60)
