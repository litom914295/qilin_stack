"""
模型对比报告生成器
对比多个模型的性能指标,生成详细的对比报告和可视化
Phase 1.3 - 模型简化与严格验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparisonReport:
    """模型对比报告生成器"""
    
    def __init__(self, output_dir: str = "output/model_comparison"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_data: List[Dict[str, Any]] = []
        self.comparison_df: Optional[pd.DataFrame] = None
        
    def add_model(
        self,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        predictions: Optional[pd.DataFrame] = None
    ):
        """
        添加模型结果
        
        Args:
            model_name: 模型名称
            metrics: 评估指标字典
            metadata: 元数据(如超参数、训练时间等)
            predictions: 预测结果
        """
        model_data = {
            'model_name': model_name,
            'metrics': metrics,
            'metadata': metadata or {},
            'predictions': predictions,
            'added_at': datetime.now().isoformat()
        }
        self.models_data.append(model_data)
        logger.info(f"添加模型: {model_name}")
    
    def load_from_files(self, model_configs: List[Dict[str, str]]):
        """
        从文件批量加载模型结果
        
        Args:
            model_configs: 模型配置列表
                [{'name': 'Model1', 'metrics_path': 'path/to/metrics.json', ...}, ...]
        """
        for config in model_configs:
            model_name = config['model_name']
            metrics_path = Path(config['metrics_path'])
            
            # 加载指标
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                logger.warning(f"指标文件不存在: {metrics_path}")
                continue
            
            # 加载预测结果(可选)
            predictions = None
            if 'predictions_path' in config:
                pred_path = Path(config['predictions_path'])
                if pred_path.exists():
                    predictions = pd.read_csv(pred_path)
            
            # 加载元数据(可选)
            metadata = config.get('metadata', {})
            
            self.add_model(model_name, metrics, metadata, predictions)
    
    def create_comparison_table(self, metrics_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
        """
        创建对比表格
        
        Args:
            metrics_to_compare: 需要对比的指标列表,None表示全部
            
        Returns:
            对比表格DataFrame
        """
        if not self.models_data:
            logger.warning("没有模型数据可对比")
            return pd.DataFrame()
        
        # 收集所有指标
        all_metrics = set()
        for model_data in self.models_data:
            all_metrics.update(model_data['metrics'].keys())
        
        # 筛选指标
        if metrics_to_compare:
            metrics_to_use = [m for m in metrics_to_compare if m in all_metrics]
        else:
            metrics_to_use = sorted(list(all_metrics))
        
        # 构建对比表
        comparison_data = []
        for model_data in self.models_data:
            row = {'model_name': model_data['model_name']}
            for metric in metrics_to_use:
                row[metric] = model_data['metrics'].get(metric, np.nan)
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # 计算排名
        for metric in metrics_to_use:
            # 假设所有指标都是越大越好,如果需要区分可以添加参数
            self.comparison_df[f'{metric}_rank'] = self.comparison_df[metric].rank(ascending=False, method='min')
        
        return self.comparison_df
    
    def get_best_model(self, metric: str, higher_is_better: bool = True) -> Tuple[str, float]:
        """
        获取在指定指标上表现最好的模型
        
        Args:
            metric: 指标名称
            higher_is_better: True表示指标越大越好
            
        Returns:
            (模型名称, 指标值)
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        if metric not in self.comparison_df.columns:
            logger.warning(f"指标 {metric} 不存在")
            return ("N/A", np.nan)
        
        if higher_is_better:
            best_idx = self.comparison_df[metric].idxmax()
        else:
            best_idx = self.comparison_df[metric].idxmin()
        
        best_model = self.comparison_df.loc[best_idx, 'model_name']
        best_value = self.comparison_df.loc[best_idx, metric]
        
        return (best_model, best_value)
    
    def generate_summary_statistics(self) -> Dict[str, Dict]:
        """
        生成汇总统计信息
        
        Returns:
            汇总统计字典
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        summary = {}
        metric_cols = [col for col in self.comparison_df.columns 
                      if col not in ['model_name'] and not col.endswith('_rank')]
        
        for metric in metric_cols:
            values = self.comparison_df[metric].dropna()
            if len(values) > 0:
                summary[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'range': float(values.max() - values.min()),
                    'best_model': self.get_best_model(metric)[0]
                }
        
        return summary
    
    def generate_report(
        self,
        metrics_to_compare: Optional[List[str]] = None,
        include_metadata: bool = True
    ) -> str:
        """
        生成文本报告
        
        Args:
            metrics_to_compare: 需要对比的指标
            include_metadata: 是否包含元数据
            
        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("模型对比报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"对比模型数量: {len(self.models_data)}")
        report_lines.append("")
        
        # 创建对比表
        comparison_df = self.create_comparison_table(metrics_to_compare)
        
        if comparison_df.empty:
            report_lines.append("没有可对比的数据")
            return "\n".join(report_lines)
        
        # 模型列表
        report_lines.append("对比模型:")
        for i, model_name in enumerate(comparison_df['model_name'], 1):
            report_lines.append(f"  {i}. {model_name}")
        report_lines.append("")
        
        # 指标对比表
        report_lines.append("指标对比:")
        report_lines.append("-" * 80)
        
        # 只显示指标值,不显示排名列
        display_cols = [col for col in comparison_df.columns if not col.endswith('_rank')]
        display_df = comparison_df[display_cols].copy()
        
        # 格式化数值
        for col in display_df.columns:
            if col != 'model_name' and pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        report_lines.append(display_df.to_string(index=False))
        report_lines.append("")
        
        # 最佳模型
        report_lines.append("最佳模型 (按指标):")
        report_lines.append("-" * 80)
        metric_cols = [col for col in comparison_df.columns 
                      if col not in ['model_name'] and not col.endswith('_rank')]
        
        for metric in metric_cols:
            best_model, best_value = self.get_best_model(metric)
            report_lines.append(f"  {metric}: {best_model} ({best_value:.4f})")
        report_lines.append("")
        
        # 汇总统计
        report_lines.append("汇总统计:")
        report_lines.append("-" * 80)
        summary = self.generate_summary_statistics()
        for metric, stats in summary.items():
            report_lines.append(f"\n{metric}:")
            report_lines.append(f"  均值: {stats['mean']:.4f}")
            report_lines.append(f"  标准差: {stats['std']:.4f}")
            report_lines.append(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            report_lines.append(f"  最佳: {stats['best_model']}")
        report_lines.append("")
        
        # 元数据
        if include_metadata:
            report_lines.append("模型元数据:")
            report_lines.append("-" * 80)
            for model_data in self.models_data:
                report_lines.append(f"\n{model_data['model_name']}:")
                if model_data['metadata']:
                    for key, value in model_data['metadata'].items():
                        report_lines.append(f"  {key}: {value}")
                else:
                    report_lines.append("  无元数据")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"报告已保存: {report_path}")
        
        return report_text
    
    def save_comparison_table(self, filename: str = "comparison_table.csv"):
        """保存对比表格"""
        if self.comparison_df is None:
            self.create_comparison_table()
        
        table_path = self.output_dir / filename
        self.comparison_df.to_csv(table_path, index=False)
        logger.info(f"对比表格已保存: {table_path}")
    
    def plot_metrics_comparison(
        self,
        metrics: Optional[List[str]] = None,
        plot_type: str = 'bar'  # 'bar', 'radar', 'heatmap'
    ):
        """
        绘制指标对比图
        
        Args:
            metrics: 要绘制的指标列表
            plot_type: 图表类型
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib或seaborn未安装,无法绘图")
            return
        
        if self.comparison_df is None:
            self.create_comparison_table()
        
        # 选择指标
        if metrics is None:
            metrics = [col for col in self.comparison_df.columns 
                      if col not in ['model_name'] and not col.endswith('_rank')]
        
        if not metrics:
            logger.warning("没有可绘制的指标")
            return
        
        # 准备数据
        plot_df = self.comparison_df[['model_name'] + metrics].set_index('model_name')
        
        if plot_type == 'bar':
            # 条形图
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, metrics):
                plot_df[metric].plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f'{metric} 对比', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric)
                ax.set_xlabel('模型')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = self.output_dir / "metrics_comparison_bar.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"条形图已保存: {plot_path}")
            plt.close()
        
        elif plot_type == 'radar':
            # 雷达图 (需要归一化)
            from math import pi
            
            # 归一化到0-1
            plot_df_norm = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())
            plot_df_norm = plot_df_norm.fillna(0)
            
            num_vars = len(metrics)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for model_name in plot_df_norm.index:
                values = plot_df_norm.loc[model_name].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('模型性能雷达图', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            plt.tight_layout()
            plot_path = self.output_dir / "metrics_comparison_radar.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"雷达图已保存: {plot_path}")
            plt.close()
        
        elif plot_type == 'heatmap':
            # 热力图
            fig, ax = plt.subplots(figsize=(12, len(plot_df)*0.8))
            
            # 归一化以便于比较
            plot_df_norm = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())
            
            sns.heatmap(
                plot_df_norm.T,
                annot=plot_df.T,
                fmt='.4f',
                cmap='RdYlGn',
                center=0.5,
                linewidths=0.5,
                cbar_kws={'label': '归一化得分'},
                ax=ax
            )
            ax.set_title('模型指标热力图', fontsize=16, fontweight='bold')
            ax.set_xlabel('模型')
            ax.set_ylabel('指标')
            
            plt.tight_layout()
            plot_path = self.output_dir / "metrics_comparison_heatmap.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"热力图已保存: {plot_path}")
            plt.close()
    
    def export_to_json(self, filename: str = "comparison_results.json"):
        """导出完整结果到JSON"""
        export_data = {
            'models': [],
            'comparison_table': None,
            'summary_statistics': None,
            'generated_at': datetime.now().isoformat()
        }
        
        # 导出模型数据(不包含预测结果以减小文件大小)
        for model_data in self.models_data:
            export_data['models'].append({
                'model_name': model_data['model_name'],
                'metrics': model_data['metrics'],
                'metadata': model_data['metadata'],
                'added_at': model_data['added_at']
            })
        
        # 导出对比表
        if self.comparison_df is not None:
            export_data['comparison_table'] = self.comparison_df.to_dict(orient='records')
        
        # 导出汇总统计
        export_data['summary_statistics'] = self.generate_summary_statistics()
        
        # 保存
        json_path = self.output_dir / filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON结果已保存: {json_path}")


def example_usage():
    """使用示例"""
    
    # 创建报告生成器
    reporter = ModelComparisonReport(output_dir="output/model_comparison_example")
    
    # 模拟添加多个模型的结果
    models_data = [
        {
            'model_name': 'LightGBM_Baseline',
            'metrics': {
                'accuracy': 0.6521,
                'precision': 0.6234,
                'recall': 0.6521,
                'f1_score': 0.6325,
                'auc': 0.7156
            },
            'metadata': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'training_time_seconds': 45.2
            }
        },
        {
            'model_name': 'RandomForest',
            'metrics': {
                'accuracy': 0.6392,
                'precision': 0.6123,
                'recall': 0.6392,
                'f1_score': 0.6211,
                'auc': 0.6987
            },
            'metadata': {
                'n_estimators': 200,
                'max_depth': 10,
                'training_time_seconds': 120.5
            }
        },
        {
            'model_name': 'XGBoost',
            'metrics': {
                'accuracy': 0.6678,
                'precision': 0.6456,
                'recall': 0.6678,
                'f1_score': 0.6523,
                'auc': 0.7289
            },
            'metadata': {
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.03,
                'training_time_seconds': 67.8
            }
        },
        {
            'model_name': 'LogisticRegression',
            'metrics': {
                'accuracy': 0.5834,
                'precision': 0.5712,
                'recall': 0.5834,
                'f1_score': 0.5656,
                'auc': 0.6234
            },
            'metadata': {
                'C': 1.0,
                'penalty': 'l2',
                'training_time_seconds': 12.3
            }
        }
    ]
    
    # 添加模型
    for model_data in models_data:
        reporter.add_model(
            model_name=model_data['model_name'],
            metrics=model_data['metrics'],
            metadata=model_data['metadata']
        )
    
    # 生成报告
    print("\n生成对比报告...\n")
    report_text = reporter.generate_report()
    print(report_text)
    
    # 保存对比表格
    reporter.save_comparison_table()
    
    # 绘制图表
    print("\n生成可视化图表...")
    reporter.plot_metrics_comparison(plot_type='bar')
    reporter.plot_metrics_comparison(plot_type='radar')
    reporter.plot_metrics_comparison(plot_type='heatmap')
    
    # 导出JSON
    reporter.export_to_json()
    
    print("\n所有报告已生成完成!")


if __name__ == "__main__":
    example_usage()
