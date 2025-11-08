"""
因子衰减监控系统

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务 1.2
目标：实时监控因子有效性，及时发现因子衰减

核心功能：
1. 滚动IC计算（20日/60日/120日窗口）
2. IC统计指标（均值、标准差、IR、胜率）
3. 因子健康度评分
4. IC时间序列可视化
5. 预警机制（因子失效检测）

作者：Qilin Quant Team
创建：2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FactorDecayMonitor:
    """因子衰减监控器"""
    
    def __init__(self, data_path: str = None):
        """
        初始化因子衰减监控器
        
        Args:
            data_path: 历史因子数据路径
        """
        self.data_path = data_path or str(project_root / 'data' / 'factors')
        self.ic_history = {}  # 缓存IC历史数据
        
        # IC阈值配置
        self.thresholds = {
            'excellent': 0.05,    # 优秀因子
            'good': 0.03,         # 良好因子
            'acceptable': 0.02,   # 可接受因子
            'warning': 0.01,      # 警告阈值
            'critical': 0.005     # 危险阈值
        }
        
        # 窗口配置
        self.windows = {
            'short': 20,    # 短期窗口（1个月）
            'medium': 60,   # 中期窗口（3个月）
            'long': 120     # 长期窗口（6个月）
        }
        
        print("📊 因子衰减监控系统初始化完成")
    
    def calculate_rolling_ic(self, 
                            factor_data: pd.DataFrame, 
                            return_data: pd.DataFrame,
                            factor_name: str,
                            windows: List[int] = None) -> pd.DataFrame:
        """
        计算滚动IC
        
        Args:
            factor_data: 因子数据 (日期×股票)
            return_data: 收益率数据 (日期×股票)
            factor_name: 因子名称
            windows: IC计算窗口列表
        
        Returns:
            pd.DataFrame: 滚动IC时间序列
        """
        print(f"  计算因子 {factor_name} 的滚动IC...")
        
        if windows is None:
            windows = [self.windows['short'], self.windows['medium'], self.windows['long']]
        
        # 确保数据对齐
        common_dates = factor_data.index.intersection(return_data.index)
        factor_data = factor_data.loc[common_dates]
        return_data = return_data.loc[common_dates]
        
        ic_results = pd.DataFrame(index=factor_data.index)
        
        # 计算每日IC（截面相关性）
        daily_ic = []
        for date in factor_data.index:
            factor_values = factor_data.loc[date].dropna()
            return_values = return_data.loc[date].dropna()
            
            # 找到共同股票
            common_stocks = factor_values.index.intersection(return_values.index)
            
            if len(common_stocks) > 10:  # 至少需要10只股票
                ic, _ = spearmanr(
                    factor_values[common_stocks], 
                    return_values[common_stocks]
                )
                daily_ic.append(ic)
            else:
                daily_ic.append(np.nan)
        
        ic_results['daily_ic'] = daily_ic
        
        # 计算滚动IC
        for window in windows:
            ic_results[f'ic_{window}d'] = ic_results['daily_ic'].rolling(window).mean()
            ic_results[f'ic_std_{window}d'] = ic_results['daily_ic'].rolling(window).std()
            
            # 计算IR (Information Ratio)
            ic_results[f'ir_{window}d'] = (
                ic_results[f'ic_{window}d'] / ic_results[f'ic_std_{window}d']
            )
        
        # 缓存结果
        self.ic_history[factor_name] = ic_results
        
        return ic_results
    
    def calculate_ic_metrics(self, ic_data: pd.DataFrame, window: int = 60) -> Dict:
        """
        计算IC统计指标
        
        Args:
            ic_data: IC时间序列数据
            window: 统计窗口
        
        Returns:
            Dict: IC统计指标
        """
        ic_col = f'ic_{window}d'
        
        if ic_col not in ic_data.columns:
            ic_col = 'daily_ic'
        
        ic_series = ic_data[ic_col].dropna()
        
        if len(ic_series) == 0:
            return self._get_default_metrics()
        
        metrics = {
            # 基础统计
            'ic_mean': float(ic_series.mean()),
            'ic_std': float(ic_series.std()),
            'ic_median': float(ic_series.median()),
            
            # IR (Information Ratio)
            'ir': float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0,
            
            # IC胜率
            'ic_win_rate': float((ic_series > 0).sum() / len(ic_series)),
            
            # IC稳定性（>0的连续天数）
            'ic_stability': self._calculate_stability(ic_series),
            
            # 趋势（最近N天 vs 历史均值）
            'ic_trend': self._calculate_trend(ic_series),
            
            # 最近IC
            'ic_recent': float(ic_series.iloc[-1]) if len(ic_series) > 0 else 0,
            
            # 历史最大/最小IC
            'ic_max': float(ic_series.max()),
            'ic_min': float(ic_series.min()),
        }
        
        return metrics
    
    def evaluate_factor_health(self, factor_name: str, metrics: Dict) -> Dict:
        """
        评估因子健康度
        
        Args:
            factor_name: 因子名称
            metrics: IC统计指标
        
        Returns:
            Dict: 健康度评估结果
        """
        health = {
            'factor_name': factor_name,
            'health_score': 0,
            'health_level': '未知',
            'status': '未知',
            'warnings': [],
            'recommendations': []
        }
        
        ic_mean = metrics['ic_mean']
        ir = metrics['ir']
        win_rate = metrics['ic_win_rate']
        ic_recent = metrics['ic_recent']
        trend = metrics['ic_trend']
        
        # 1. 计算健康评分（0-100）
        score = 0
        
        # IC均值贡献（40分）
        if ic_mean >= self.thresholds['excellent']:
            score += 40
        elif ic_mean >= self.thresholds['good']:
            score += 30
        elif ic_mean >= self.thresholds['acceptable']:
            score += 20
        elif ic_mean >= self.thresholds['warning']:
            score += 10
        
        # IR贡献（30分）
        if ir >= 1.5:
            score += 30
        elif ir >= 1.0:
            score += 20
        elif ir >= 0.5:
            score += 10
        
        # 胜率贡献（20分）
        if win_rate >= 0.6:
            score += 20
        elif win_rate >= 0.55:
            score += 15
        elif win_rate >= 0.5:
            score += 10
        
        # 趋势贡献（10分）
        if trend == 'improving':
            score += 10
        elif trend == 'stable':
            score += 5
        
        health['health_score'] = score
        
        # 2. 健康等级分类
        if score >= 80:
            health['health_level'] = '优秀'
            health['status'] = '活跃'
        elif score >= 60:
            health['health_level'] = '良好'
            health['status'] = '活跃'
        elif score >= 40:
            health['health_level'] = '一般'
            health['status'] = '观察'
        elif score >= 20:
            health['health_level'] = '较差'
            health['status'] = '警告'
        else:
            health['health_level'] = '危险'
            health['status'] = '休眠'
        
        # 3. 生成警告和建议
        if ic_mean < self.thresholds['warning']:
            health['warnings'].append(f'IC均值过低({ic_mean:.4f})，因子可能失效')
        
        if win_rate < 0.5:
            health['warnings'].append(f'IC胜率低于50%({win_rate:.2%})，预测能力弱')
        
        if ic_recent < self.thresholds['critical']:
            health['warnings'].append(f'最近IC极低({ic_recent:.4f})，建议立即降权或移除')
        
        if trend == 'declining':
            health['warnings'].append('IC呈下降趋势，因子可能正在衰减')
        
        # 生成建议
        if health['status'] == '休眠':
            health['recommendations'].append('建议暂停使用该因子')
        elif health['status'] == '警告':
            health['recommendations'].append('建议降低该因子权重至50%')
            health['recommendations'].append('增加监控频率')
        elif health['status'] == '观察':
            health['recommendations'].append('保持当前权重，持续观察')
        else:
            health['recommendations'].append('因子表现良好，可正常使用')
        
        return health
    
    def batch_monitor_factors(self, 
                              factor_dict: Dict[str, pd.DataFrame],
                              return_data: pd.DataFrame,
                              window: int = 60) -> pd.DataFrame:
        """
        批量监控多个因子
        
        Args:
            factor_dict: 因子字典 {因子名: 因子数据}
            return_data: 收益率数据
            window: 统计窗口
        
        Returns:
            pd.DataFrame: 因子健康度汇总表
        """
        print(f"\n批量监控 {len(factor_dict)} 个因子...")
        
        results = []
        
        for factor_name, factor_data in factor_dict.items():
            try:
                # 计算滚动IC
                ic_data = self.calculate_rolling_ic(
                    factor_data, 
                    return_data, 
                    factor_name
                )
                
                # 计算统计指标
                metrics = self.calculate_ic_metrics(ic_data, window)
                
                # 评估健康度
                health = self.evaluate_factor_health(factor_name, metrics)
                
                # 合并结果
                result = {**health, **metrics}
                results.append(result)
                
            except Exception as e:
                print(f"    ⚠️ 因子 {factor_name} 监控失败: {e}")
                continue
        
        df_results = pd.DataFrame(results)
        
        # 按健康评分排序
        if not df_results.empty:
            df_results = df_results.sort_values('health_score', ascending=False)
        
        return df_results
    
    def plot_ic_timeseries(self, 
                          factor_name: str, 
                          ic_data: pd.DataFrame = None,
                          save_path: str = None) -> None:
        """
        绘制IC时间序列图
        
        Args:
            factor_name: 因子名称
            ic_data: IC数据（如果为None则从缓存读取）
            save_path: 保存路径
        """
        if ic_data is None:
            ic_data = self.ic_history.get(factor_name)
            
            if ic_data is None:
                print(f"⚠️ 未找到因子 {factor_name} 的IC数据")
                return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 子图1：IC时间序列
        ax1 = axes[0]
        
        # 绘制不同窗口的IC
        if 'ic_20d' in ic_data.columns:
            ax1.plot(ic_data.index, ic_data['ic_20d'], 
                    label='IC(20日)', alpha=0.7, linewidth=1.5)
        
        if 'ic_60d' in ic_data.columns:
            ax1.plot(ic_data.index, ic_data['ic_60d'], 
                    label='IC(60日)', alpha=0.8, linewidth=2)
        
        if 'ic_120d' in ic_data.columns:
            ax1.plot(ic_data.index, ic_data['ic_120d'], 
                    label='IC(120日)', alpha=0.9, linewidth=2.5)
        
        # 添加阈值线
        ax1.axhline(y=self.thresholds['excellent'], 
                   color='green', linestyle='--', alpha=0.5, label='优秀阈值')
        ax1.axhline(y=self.thresholds['acceptable'], 
                   color='orange', linestyle='--', alpha=0.5, label='可接受阈值')
        ax1.axhline(y=self.thresholds['warning'], 
                   color='red', linestyle='--', alpha=0.5, label='警告阈值')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax1.set_title(f'因子 {factor_name} - IC时间序列', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日期', fontsize=11)
        ax1.set_ylabel('IC值', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：IR时间序列
        ax2 = axes[1]
        
        if 'ir_60d' in ic_data.columns:
            ir_data = ic_data['ir_60d'].dropna()
            ax2.plot(ir_data.index, ir_data, 
                    label='IR(60日)', color='purple', linewidth=2)
            
            ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='IR=1.0')
            ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='IR=0.5')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax2.set_title(f'因子 {factor_name} - IR时间序列', fontsize=14, fontweight='bold')
            ax2.set_xlabel('日期', fontsize=11)
            ax2.set_ylabel('IR值', fontsize=11)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  图表已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_health_report(self, 
                              health_df: pd.DataFrame, 
                              output_path: str = None) -> str:
        """
        生成因子健康度报告
        
        Args:
            health_df: 因子健康度汇总表
            output_path: 输出路径
        
        Returns:
            str: 报告内容
        """
        print("\n生成因子健康度报告...")
        
        report_lines = []
        report_lines.append("# 因子健康度监控报告\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**监控因子数**: {len(health_df)}\n\n")
        
        # 汇总统计
        report_lines.append("## 📊 整体概况\n")
        
        status_counts = health_df['status'].value_counts()
        report_lines.append(f"- **活跃因子**: {status_counts.get('活跃', 0)}个\n")
        report_lines.append(f"- **观察因子**: {status_counts.get('观察', 0)}个\n")
        report_lines.append(f"- **警告因子**: {status_counts.get('警告', 0)}个\n")
        report_lines.append(f"- **休眠因子**: {status_counts.get('休眠', 0)}个\n\n")
        
        # 优秀因子
        excellent_factors = health_df[health_df['health_level'] == '优秀']
        if not excellent_factors.empty:
            report_lines.append("## 🌟 优秀因子 (健康评分≥80)\n")
            for _, row in excellent_factors.iterrows():
                report_lines.append(f"- **{row['factor_name']}**: "
                                  f"评分{row['health_score']:.1f}, "
                                  f"IC={row['ic_mean']:.4f}, "
                                  f"IR={row['ir']:.2f}, "
                                  f"胜率={row['ic_win_rate']:.2%}\n")
            report_lines.append("\n")
        
        # 警告因子
        warning_factors = health_df[health_df['status'].isin(['警告', '休眠'])]
        if not warning_factors.empty:
            report_lines.append("## ⚠️ 警告因子 (需要关注)\n")
            for _, row in warning_factors.iterrows():
                report_lines.append(f"### {row['factor_name']} ({row['health_level']})\n")
                report_lines.append(f"- **健康评分**: {row['health_score']:.1f}\n")
                report_lines.append(f"- **IC均值**: {row['ic_mean']:.4f}\n")
                report_lines.append(f"- **IC胜率**: {row['ic_win_rate']:.2%}\n")
                
                if row['warnings']:
                    report_lines.append("- **警告信息**:\n")
                    for warning in row['warnings']:
                        report_lines.append(f"  - {warning}\n")
                
                if row['recommendations']:
                    report_lines.append("- **建议措施**:\n")
                    for rec in row['recommendations']:
                        report_lines.append(f"  - {rec}\n")
                
                report_lines.append("\n")
        
        # 详细统计表
        report_lines.append("## 📋 详细统计表\n")
        report_lines.append("| 因子名称 | 健康评分 | 状态 | IC均值 | IR | 胜率 | 趋势 |\n")
        report_lines.append("|---------|---------|------|-------|----|----- |------|\n")
        
        for _, row in health_df.iterrows():
            report_lines.append(
                f"| {row['factor_name']} "
                f"| {row['health_score']:.1f} "
                f"| {row['status']} "
                f"| {row['ic_mean']:.4f} "
                f"| {row['ir']:.2f} "
                f"| {row['ic_win_rate']:.2%} "
                f"| {row['ic_trend']} |\n"
            )
        
        report_content = "".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 报告已保存至: {output_path}")
        
        return report_content
    
    # ==================== 辅助方法 ====================
    
    def _calculate_stability(self, ic_series: pd.Series) -> float:
        """计算IC稳定性（连续正IC的最大天数比例）"""
        if len(ic_series) == 0:
            return 0
        
        positive_runs = []
        current_run = 0
        
        for ic in ic_series:
            if ic > 0:
                current_run += 1
            else:
                if current_run > 0:
                    positive_runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            positive_runs.append(current_run)
        
        if not positive_runs:
            return 0
        
        return max(positive_runs) / len(ic_series)
    
    def _calculate_trend(self, ic_series: pd.Series, recent_days: int = 20) -> str:
        """计算IC趋势"""
        if len(ic_series) < recent_days * 2:
            return 'unknown'
        
        recent_ic = ic_series.iloc[-recent_days:].mean()
        historical_ic = ic_series.iloc[:-recent_days].mean()
        
        if recent_ic > historical_ic * 1.1:
            return 'improving'
        elif recent_ic < historical_ic * 0.9:
            return 'declining'
        else:
            return 'stable'
    
    def _get_default_metrics(self) -> Dict:
        """获取默认指标"""
        return {
            'ic_mean': 0,
            'ic_std': 0,
            'ic_median': 0,
            'ir': 0,
            'ic_win_rate': 0,
            'ic_stability': 0,
            'ic_trend': 'unknown',
            'ic_recent': 0,
            'ic_max': 0,
            'ic_min': 0
        }


def main():
    """主函数 - 示例用法"""
    monitor = FactorDecayMonitor()
    
    # 模拟因子数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    stocks = [f'stock_{i}' for i in range(50)]
    
    # 生成模拟因子和收益率数据
    np.random.seed(42)
    
    factor_dict = {}
    for factor_name in ['momentum', 'value', 'quality']:
        factor_data = pd.DataFrame(
            np.random.randn(len(dates), len(stocks)) * 0.1,
            index=dates,
            columns=stocks
        )
        factor_dict[factor_name] = factor_data
    
    # 收益率数据
    return_data = pd.DataFrame(
        np.random.randn(len(dates), len(stocks)) * 0.02,
        index=dates,
        columns=stocks
    )
    
    # 批量监控
    health_df = monitor.batch_monitor_factors(factor_dict, return_data)
    
    print("\n" + "="*70)
    print("📊 因子健康度监控结果")
    print("="*70)
    print(health_df[['factor_name', 'health_score', 'status', 'ic_mean', 'ir', 'ic_win_rate']])
    
    # 生成报告
    report_path = project_root / 'reports' / 'factor_health_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    monitor.generate_health_report(health_df, str(report_path))
    
    # 绘制第一个因子的IC图
    plot_path = project_root / 'reports' / 'factor_ic_plot.png'
    monitor.plot_ic_timeseries('momentum', save_path=str(plot_path))
    
    print("\n✅ 监控任务完成！")


if __name__ == '__main__':
    main()
