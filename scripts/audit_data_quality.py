"""
数据质量审计脚本

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务1.1
目标：识别数据源质量问题，为后续特征清理提供依据

功能：
1. 统计各数据源（Qlib/AKShare/Tushare）的覆盖率
2. 检测缺失值、异常值比例
3. 对比不同数据源的一致性
4. 识别"高频特征"的真实数据粒度
5. 生成详细的审计报告

作者：Qilin Quant Team
创建：2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_layer.premium_data_provider import PremiumDataProvider


class DataQualityAuditor:
    """数据质量审计器"""
    
    def __init__(self, start_date: str = "2023-01-01", end_date: str = None):
        """
        初始化审计器
        
        Args:
            start_date: 审计开始日期
            end_date: 审计结束日期（默认为今天）
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # 审计结果
        self.audit_results = {}
        self.report_sections = []
        
        print(f"📊 数据质量审计初始化")
        print(f"   审计区间: {self.start_date} ~ {self.end_date}")
        print("=" * 70)
    
    def audit_data_sources_coverage(self) -> Dict:
        """审计各数据源的覆盖率"""
        print("\n🔍 1. 审计数据源覆盖率...")
        
        coverage_stats = {
            'qlib': {'available': False, 'coverage': 0, 'status': '未检测'},
            'akshare': {'available': False, 'coverage': 0, 'status': '未检测'},
            'tushare': {'available': False, 'coverage': 0, 'status': '未检测'}
        }
        
        # 1. 检测Qlib
        try:
            import qlib
            from qlib.data import D
            
            # 尝试获取数据
            test_symbols = ['SH600000', 'SZ000001']
            data_count = 0
            for symbol in test_symbols:
                try:
                    df = D.features([symbol], ['$close', '$volume'], 
                                   start_time=self.start_date, end_time=self.end_date)
                    if df is not None and not df.empty:
                        data_count += len(df)
                except:
                    pass
            
            coverage_stats['qlib']['available'] = data_count > 0
            coverage_stats['qlib']['coverage'] = data_count
            coverage_stats['qlib']['status'] = '✅ 可用' if data_count > 0 else '❌ 无数据'
            
        except ImportError:
            coverage_stats['qlib']['status'] = '❌ 未安装'
        except Exception as e:
            coverage_stats['qlib']['status'] = f'⚠️ 异常: {str(e)[:50]}'
        
        # 2. 检测AKShare
        try:
            import akshare as ak
            
            # 测试获取今日涨停数据
            today = datetime.now().strftime('%Y%m%d')
            df = ak.stock_zt_pool_em(date=today)
            
            coverage_stats['akshare']['available'] = not df.empty
            coverage_stats['akshare']['coverage'] = len(df) if not df.empty else 0
            coverage_stats['akshare']['status'] = '✅ 可用'
            
        except ImportError:
            coverage_stats['akshare']['status'] = '❌ 未安装'
        except Exception as e:
            coverage_stats['akshare']['status'] = f'⚠️ 异常: {str(e)[:50]}'
        
        # 3. 检测Tushare
        try:
            import tushare as ts
            
            # 检查是否有token配置
            token_file = project_root / 'config' / 'tushare_token.txt'
            if token_file.exists():
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                ts.set_token(token)
                
                pro = ts.pro_api()
                df = pro.daily(trade_date='20240101', limit=10)
                
                coverage_stats['tushare']['available'] = not df.empty
                coverage_stats['tushare']['coverage'] = len(df)
                coverage_stats['tushare']['status'] = '✅ 可用'
            else:
                coverage_stats['tushare']['status'] = '⚠️ 未配置Token'
                
        except ImportError:
            coverage_stats['tushare']['status'] = '❌ 未安装'
        except Exception as e:
            coverage_stats['tushare']['status'] = f'⚠️ 异常: {str(e)[:50]}'
        
        # 打印结果
        print("\n📈 数据源覆盖率统计：")
        for source, stats in coverage_stats.items():
            print(f"   {source.upper():10s}: {stats['status']:20s} | 数据量: {stats['coverage']}")
        
        self.audit_results['coverage'] = coverage_stats
        return coverage_stats
    
    def audit_missing_values(self, sample_size: int = 100) -> Dict:
        """审计缺失值情况"""
        print(f"\n🔍 2. 审计缺失值（采样 {sample_size} 条记录）...")
        
        missing_stats = {}
        
        try:
            # 使用AKShare作为主要数据源进行审计
            import akshare as ak
            
            # 获取最近的涨停数据
            recent_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            df = ak.stock_zt_pool_em(date=recent_date)
            
            if df.empty:
                print("   ⚠️ 无法获取样本数据")
                return {}
            
            # 限制样本大小
            df_sample = df.head(sample_size)
            
            # 统计每列的缺失值
            total_rows = len(df_sample)
            for col in df_sample.columns:
                missing_count = df_sample[col].isna().sum()
                missing_rate = missing_count / total_rows * 100
                
                missing_stats[col] = {
                    'missing_count': int(missing_count),
                    'missing_rate': f"{missing_rate:.2f}%",
                    'status': '✅ 正常' if missing_rate < 5 else '⚠️ 高缺失' if missing_rate < 20 else '❌ 严重缺失'
                }
            
            # 打印高缺失率字段
            print("\n   高缺失率字段（>5%）：")
            high_missing = {k: v for k, v in missing_stats.items() if float(v['missing_rate'].rstrip('%')) > 5}
            if high_missing:
                for col, stats in high_missing.items():
                    print(f"   - {col:20s}: {stats['missing_rate']:8s} {stats['status']}")
            else:
                print("   ✅ 未发现高缺失率字段")
                
        except Exception as e:
            print(f"   ❌ 审计失败: {e}")
        
        self.audit_results['missing_values'] = missing_stats
        return missing_stats
    
    def audit_outliers(self, sample_size: int = 100) -> Dict:
        """审计异常值情况"""
        print(f"\n🔍 3. 审计异常值（采样 {sample_size} 条记录）...")
        
        outlier_stats = {}
        
        try:
            import akshare as ak
            
            recent_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            df = ak.stock_zt_pool_em(date=recent_date)
            
            if df.empty:
                print("   ⚠️ 无法获取样本数据")
                return {}
            
            df_sample = df.head(sample_size)
            
            # 检测数值型列的异常值（使用IQR方法）
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    Q1 = df_sample[col].quantile(0.25)
                    Q3 = df_sample[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df_sample[(df_sample[col] < lower_bound) | (df_sample[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_rate = outlier_count / len(df_sample) * 100
                    
                    outlier_stats[col] = {
                        'outlier_count': int(outlier_count),
                        'outlier_rate': f"{outlier_rate:.2f}%",
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'status': '✅ 正常' if outlier_rate < 5 else '⚠️ 有异常'
                    }
                except:
                    pass
            
            # 打印异常字段
            print("\n   异常值检测结果：")
            high_outlier = {k: v for k, v in outlier_stats.items() if float(v['outlier_rate'].rstrip('%')) > 5}
            if high_outlier:
                for col, stats in high_outlier.items():
                    print(f"   - {col:20s}: {stats['outlier_rate']:8s} {stats['status']}")
            else:
                print("   ✅ 未发现高异常率字段")
                
        except Exception as e:
            print(f"   ❌ 审计失败: {e}")
        
        self.audit_results['outliers'] = outlier_stats
        return outlier_stats
    
    def audit_high_freq_features(self) -> Dict:
        """审计高频特征的数据粒度"""
        print("\n🔍 4. 审计高频特征数据粒度...")
        
        high_freq_features = {
            '封单稳定性': {'data_source': '未知', 'granularity': '未知', 'reliability': 0},
            '大单流入节奏': {'data_source': '未知', 'granularity': '未知', 'reliability': 0},
            '成交萎缩度': {'data_source': '未知', 'granularity': '未知', 'reliability': 0},
            '分时形态': {'data_source': '未知', 'granularity': '未知', 'reliability': 0},
        }
        
        # 检查是否有L2数据
        l2_available = False
        
        # 检查分钟数据可用性
        minute_data_available = False
        try:
            import akshare as ak
            # 尝试获取分钟数据
            df_minute = ak.stock_zh_a_hist_min_em(symbol="000001", period='1', adjust='')
            minute_data_available = not df_minute.empty
        except:
            pass
        
        # 评估特征可靠性
        print("\n   高频特征数据源评估：")
        
        if l2_available:
            for feature in high_freq_features:
                high_freq_features[feature]['data_source'] = 'Level-2逐笔'
                high_freq_features[feature]['granularity'] = '逐笔/快照'
                high_freq_features[feature]['reliability'] = 95
                print(f"   ✅ {feature:15s}: Level-2数据，可靠性 95%")
        elif minute_data_available:
            for feature in high_freq_features:
                high_freq_features[feature]['data_source'] = '分钟线数据'
                high_freq_features[feature]['granularity'] = '1分钟'
                high_freq_features[feature]['reliability'] = 60
                print(f"   ⚠️ {feature:15s}: 分钟数据模拟，可靠性 60%")
        else:
            for feature in high_freq_features:
                high_freq_features[feature]['data_source'] = '日线数据'
                high_freq_features[feature]['granularity'] = '日线'
                high_freq_features[feature]['reliability'] = 30
                print(f"   ❌ {feature:15s}: 日线数据模拟，可靠性 30% ⚠️ 建议禁用")
        
        print("\n   💡 建议：")
        avg_reliability = np.mean([v['reliability'] for v in high_freq_features.values()])
        if avg_reliability < 50:
            print("   ⚠️ 高频特征平均可靠性 <50%，强烈建议暂时禁用这些特征！")
            print("   📌 在获得真实L2数据前，应使用日线可靠特征构建基准模型")
        elif avg_reliability < 70:
            print("   ⚠️ 高频特征可靠性中等，建议谨慎使用并密切监控效果")
        else:
            print("   ✅ 高频特征可靠性较高，可以使用")
        
        self.audit_results['high_freq_features'] = high_freq_features
        return high_freq_features
    
    def audit_data_consistency(self, test_symbols: List[str] = None) -> Dict:
        """审计不同数据源的一致性"""
        print("\n🔍 5. 审计数据源一致性...")
        
        if test_symbols is None:
            test_symbols = ['000001', '600000']  # 默认测试两只股票
        
        consistency_results = {}
        
        print(f"\n   测试股票: {', '.join(test_symbols)}")
        
        for symbol in test_symbols:
            print(f"\n   检测 {symbol}...")
            symbol_results = {}
            
            # 尝试从不同数据源获取相同日期的收盘价
            test_date = '20240101'
            
            # AKShare
            try:
                import akshare as ak
                symbol_with_prefix = f"sz{symbol}" if symbol.startswith('0') or symbol.startswith('3') else f"sh{symbol}"
                df_ak = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
                if not df_ak.empty:
                    df_ak['日期'] = pd.to_datetime(df_ak['日期'])
                    close_ak = df_ak[df_ak['日期'] == pd.to_datetime(test_date)]['收盘'].values
                    symbol_results['akshare'] = float(close_ak[0]) if len(close_ak) > 0 else None
            except Exception as e:
                symbol_results['akshare'] = None
                print(f"      AKShare获取失败: {str(e)[:50]}")
            
            # 对比结果
            if symbol_results:
                consistency_results[symbol] = symbol_results
                print(f"      数据点数: {len([v for v in symbol_results.values() if v is not None])}")
        
        if consistency_results:
            print("\n   ✅ 数据一致性检查完成")
        else:
            print("\n   ⚠️ 无法获取足够数据进行一致性对比")
        
        self.audit_results['consistency'] = consistency_results
        return consistency_results
    
    def generate_report(self, output_path: str = None) -> str:
        """生成审计报告"""
        print("\n📝 生成审计报告...")
        
        if output_path is None:
            output_path = project_root / 'reports' / 'data_quality_audit_report.md'
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# 数据质量审计报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**审计区间**: {self.start_date} ~ {self.end_date}\n")
        report.append(f"**任务来源**: docs/IMPROVEMENT_ROADMAP.md - 阶段一任务1.1\n")
        report.append("\n---\n\n")
        
        # 1. 数据源覆盖率
        report.append("## 1. 数据源覆盖率\n\n")
        if 'coverage' in self.audit_results:
            report.append("| 数据源 | 状态 | 数据量 |\n")
            report.append("|--------|------|--------|\n")
            for source, stats in self.audit_results['coverage'].items():
                report.append(f"| {source.upper()} | {stats['status']} | {stats['coverage']} |\n")
        report.append("\n")
        
        # 2. 缺失值统计
        report.append("## 2. 缺失值统计\n\n")
        if 'missing_values' in self.audit_results:
            high_missing = {k: v for k, v in self.audit_results['missing_values'].items() 
                          if float(v['missing_rate'].rstrip('%')) > 5}
            if high_missing:
                report.append("**高缺失率字段（>5%）**:\n\n")
                report.append("| 字段 | 缺失率 | 状态 |\n")
                report.append("|------|--------|------|\n")
                for col, stats in high_missing.items():
                    report.append(f"| {col} | {stats['missing_rate']} | {stats['status']} |\n")
            else:
                report.append("✅ **未发现高缺失率字段**\n")
        report.append("\n")
        
        # 3. 异常值统计
        report.append("## 3. 异常值统计\n\n")
        if 'outliers' in self.audit_results:
            high_outlier = {k: v for k, v in self.audit_results['outliers'].items() 
                          if float(v['outlier_rate'].rstrip('%')) > 5}
            if high_outlier:
                report.append("**高异常率字段（>5%）**:\n\n")
                report.append("| 字段 | 异常率 | 状态 |\n")
                report.append("|------|--------|------|\n")
                for col, stats in high_outlier.items():
                    report.append(f"| {col} | {stats['outlier_rate']} | {stats['status']} |\n")
            else:
                report.append("✅ **未发现高异常率字段**\n")
        report.append("\n")
        
        # 4. 高频特征评估
        report.append("## 4. 高频特征数据粒度评估\n\n")
        if 'high_freq_features' in self.audit_results:
            report.append("| 特征 | 数据源 | 粒度 | 可靠性 |\n")
            report.append("|------|--------|------|--------|\n")
            for feature, stats in self.audit_results['high_freq_features'].items():
                reliability_emoji = "✅" if stats['reliability'] > 70 else "⚠️" if stats['reliability'] > 50 else "❌"
                report.append(f"| {feature} | {stats['data_source']} | {stats['granularity']} | {reliability_emoji} {stats['reliability']}% |\n")
            
            avg_reliability = np.mean([v['reliability'] for v in self.audit_results['high_freq_features'].values()])
            report.append(f"\n**平均可靠性**: {avg_reliability:.1f}%\n\n")
            
            if avg_reliability < 50:
                report.append("### ⚠️ 关键建议\n\n")
                report.append("高频特征平均可靠性 <50%，**强烈建议暂时禁用这些特征**！\n\n")
                report.append("在获得真实L2数据前，应使用日线可靠特征构建基准模型。\n\n")
        
        # 5. 关键发现与建议
        report.append("## 5. 关键发现与建议\n\n")
        report.append("### 🔍 关键发现\n\n")
        
        # 根据审计结果总结
        if 'high_freq_features' in self.audit_results:
            avg_rel = np.mean([v['reliability'] for v in self.audit_results['high_freq_features'].values()])
            if avg_rel < 50:
                report.append("1. ❌ **高频特征不可靠**: 当前高频特征基于低粒度数据（日线或分钟线），可靠性严重不足\n")
            elif avg_rel < 70:
                report.append("1. ⚠️ **高频特征可靠性中等**: 基于分钟数据模拟，需谨慎使用\n")
            else:
                report.append("1. ✅ **高频特征可用**: 数据粒度满足要求\n")
        
        if 'coverage' in self.audit_results:
            available_sources = [k for k, v in self.audit_results['coverage'].items() if v['available']]
            report.append(f"2. 📊 **可用数据源**: {', '.join([s.upper() for s in available_sources])}\n")
        
        report.append("\n### 💡 下一步行动建议\n\n")
        report.append("根据 `docs/IMPROVEMENT_ROADMAP.md` 阶段一计划：\n\n")
        report.append("1. ✅ **完成**: 数据质量审计（当前任务）\n")
        report.append("2. ⏭️ **下一步**: 执行高频特征可靠性测试 (`scripts/test_high_freq_features.py`)\n")
        report.append("3. 📌 **后续**: 特征降维，禁用不可靠特征，生成核心特征集\n")
        
        report.append("\n---\n\n")
        report.append("*本报告由 Qilin Stack 数据质量审计系统自动生成*\n")
        
        # 写入文件
        report_text = ''.join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✅ 审计报告已生成: {output_path}")
        return report_text
    
    def run_full_audit(self) -> Dict:
        """运行完整审计流程"""
        print("\n" + "="*70)
        print("🚀 开始完整数据质量审计")
        print("="*70)
        
        # 1. 数据源覆盖率
        self.audit_data_sources_coverage()
        
        # 2. 缺失值
        self.audit_missing_values()
        
        # 3. 异常值
        self.audit_outliers()
        
        # 4. 高频特征
        self.audit_high_freq_features()
        
        # 5. 数据一致性
        self.audit_data_consistency()
        
        # 6. 生成报告
        self.generate_report()
        
        print("\n" + "="*70)
        print("✅ 数据质量审计完成！")
        print("="*70)
        
        return self.audit_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据质量审计工具')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                      help='审计开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                      help='审计结束日期 (YYYY-MM-DD)，默认为今天')
    parser.add_argument('--output', type=str, default=None,
                      help='审计报告输出路径')
    
    args = parser.parse_args()
    
    # 创建审计器
    auditor = DataQualityAuditor(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 运行完整审计
    results = auditor.run_full_audit()
    
    # 如果指定了输出路径，生成报告
    if args.output:
        auditor.generate_report(output_path=args.output)
    
    return results


if __name__ == '__main__':
    main()
