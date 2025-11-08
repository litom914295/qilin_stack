"""
特征降维脚本 - 生成核心特征集

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务1.3
目标：禁用不可靠特征，生成精简版50核心特征集

降维策略：
1. 强制禁用：可靠性得分<40的特征
2. 条件禁用：可靠性得分40-60，且数据粒度<分钟级
3. 保留：日线可靠特征（价量、技术指标）
4. 保留：封板基础特征（封单强度、涨停时间、开板次数）
5. 保留：历史统计特征（历史竞价表现）

作者：Qilin Quant Team
创建：2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import sys
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CoreFeatureGenerator:
    """核心特征生成器"""
    
    # 预定义的可靠特征类别
    RELIABLE_FEATURE_CATEGORIES = {
        # 日线价量特征（最可靠）
        'daily_price_volume': [
            'close', 'open', 'high', 'low', 'volume',
            'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
            'volume_ratio', 'turnover_rate',
            'amount', 'volume_ma5', 'volume_ma10', 'volume_ma20'
        ],
        
        # 技术指标（可靠）
        'technical_indicators': [
            'ma5', 'ma10', 'ma20', 'ma60',
            'ema5', 'ema10', 'ema20',
            'rsi_6', 'rsi_12', 'rsi_24',
            'macd', 'macd_signal', 'macd_hist',
            'boll_upper', 'boll_middle', 'boll_lower',
            'atr_14', 'atr_20',
            'volatility_20', 'volatility_60'
        ],
        
        # 封板基础特征（基于日线数据，相对可靠）
        'limitup_basic': [
            'is_limit_up',           # 是否涨停
            'limit_up_time',         # 涨停时间（从日线推断）
            'consecutive_days',      # 连板天数
            'first_limit_up_time',   # 首次涨停时间
            'seal_strength_proxy',   # 封单强度代理（成交额/流通市值）
            'open_count_proxy',      # 开板次数代理
        ],
        
        # 历史统计特征（可靠）
        'historical_stats': [
            'past_5d_limit_up_count',    # 过去5天涨停次数
            'past_20d_limit_up_count',   # 过去20天涨停次数
            'past_5d_avg_return',        # 过去5天平均收益
            'past_20d_avg_return',       # 过去20天平均收益
            'past_volatility',           # 历史波动率
        ],
        
        # 市场环境特征（可靠）
        'market_environment': [
            'market_limit_up_count',     # 市场涨停数
            'market_limit_down_count',   # 市场跌停数
            'market_sentiment_score',    # 市场情绪评分
            'index_return',              # 指数收益率
            'index_volatility',          # 指数波动率
        ],
        
        # 板块特征（相对可靠）
        'sector_features': [
            'sector_limit_up_count',     # 板块涨停数
            'sector_avg_return',         # 板块平均收益
            'sector_strength',           # 板块强度
        ],
    }
    
    def __init__(self, max_features: int = 50):
        """
        初始化特征生成器
        
        Args:
            max_features: 最大特征数量
        """
        self.max_features = max_features
        
        # 特征评估结果
        self.feature_scores = {}
        self.selected_features = []
        self.rejected_features = []
        
        print(f"🔧 核心特征生成器初始化")
        print(f"   最大特征数: {self.max_features}")
        print("=" * 70)
    
    def load_test_results(self, test_report_path: str = None) -> pd.DataFrame:
        """加载高频特征测试结果"""
        if test_report_path is None:
            test_report_path = project_root / 'analysis' / 'high_freq_feature_reliability.csv'
        
        print(f"\n📂 加载测试结果: {test_report_path}")
        
        if not Path(test_report_path).exists():
            print(f"   ⚠️ 测试报告不存在，使用默认评分")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(test_report_path, encoding='utf-8-sig')
            print(f"   ✅ 加载成功，共{len(df)}个特征")
            return df
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            return pd.DataFrame()
    
    def evaluate_features(self, test_results: pd.DataFrame) -> Dict:
        """评估所有特征"""
        print("\n🔍 评估特征可靠性...")
        
        feature_evaluation = {}
        
        # 1. 评估高频特征（来自测试报告）
        if not test_results.empty:
            for _, row in test_results.iterrows():
                feature_name = row['特征名称']
                score = row['综合得分']
                reliability = row['可靠性等级']
                
                feature_evaluation[feature_name] = {
                    'score': score,
                    'reliability': reliability,
                    'category': 'high_freq',
                    'action': self._determine_action(score, 'high_freq')
                }
        
        # 2. 评估预定义的可靠特征
        for category, features in self.RELIABLE_FEATURE_CATEGORIES.items():
            for feature in features:
                if feature not in feature_evaluation:
                    # 根据类别给予默认评分
                    default_score = self._get_default_score(category)
                    feature_evaluation[feature] = {
                        'score': default_score,
                        'reliability': self._score_to_reliability(default_score),
                        'category': category,
                        'action': self._determine_action(default_score, category)
                    }
        
        self.feature_scores = feature_evaluation
        
        # 统计
        total = len(feature_evaluation)
        keep = sum(1 for v in feature_evaluation.values() if v['action'] == 'keep')
        reject = sum(1 for v in feature_evaluation.values() if v['action'] == 'reject')
        
        print(f"\n   总特征数: {total}")
        print(f"   保留: {keep}")
        print(f"   拒绝: {reject}")
        
        return feature_evaluation
    
    def _get_default_score(self, category: str) -> float:
        """根据类别获取默认评分"""
        category_scores = {
            'daily_price_volume': 90,    # 日线价量最可靠
            'technical_indicators': 85,  # 技术指标很可靠
            'limitup_basic': 75,         # 封板基础特征较可靠
            'historical_stats': 80,      # 历史统计可靠
            'market_environment': 85,    # 市场环境可靠
            'sector_features': 70,       # 板块特征中等可靠
        }
        return category_scores.get(category, 60)
    
    def _score_to_reliability(self, score: float) -> str:
        """评分转可靠性等级"""
        if score >= 80:
            return '✅ 可靠'
        elif score >= 60:
            return '⚠️ 中等'
        elif score >= 40:
            return '⚠️ 较差'
        else:
            return '❌ 不可靠'
    
    def _determine_action(self, score: float, category: str) -> str:
        """决定特征的处理动作"""
        # 强制禁用：得分<40
        if score < 40:
            return 'reject'
        
        # 条件禁用：得分40-60且是高频特征
        if 40 <= score < 60 and category == 'high_freq':
            return 'reject'
        
        # 保留
        return 'keep'
    
    def select_core_features(self) -> List[str]:
        """选择核心特征"""
        print("\n🎯 选择核心特征...")
        
        # 1. 筛选保留的特征
        kept_features = [
            name for name, info in self.feature_scores.items()
            if info['action'] == 'keep'
        ]
        
        print(f"   初步保留: {len(kept_features)}个特征")
        
        # 2. 如果超过最大数量，按评分排序选择Top N
        if len(kept_features) > self.max_features:
            print(f"   超过最大值{self.max_features}，按评分排序...")
            
            # 按评分排序
            sorted_features = sorted(
                kept_features,
                key=lambda x: self.feature_scores[x]['score'],
                reverse=True
            )
            
            self.selected_features = sorted_features[:self.max_features]
        else:
            self.selected_features = kept_features
        
        # 3. 记录被拒绝的特征
        self.rejected_features = [
            name for name, info in self.feature_scores.items()
            if info['action'] == 'reject' or name not in self.selected_features
        ]
        
        print(f"   ✅ 最终选择: {len(self.selected_features)}个特征")
        print(f"   ❌ 拒绝: {len(self.rejected_features)}个特征")
        
        return self.selected_features
    
    def generate_feature_module(self, output_path: str = None) -> str:
        """生成特征模块代码"""
        print("\n📝 生成特征模块代码...")
        
        if output_path is None:
            output_path = project_root / 'features' / 'core_features_v1.py'
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 生成代码
        code = []
        code.append('"""')
        code.append('核心特征集 v1.0')
        code.append('')
        code.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        code.append('任务来源: docs/IMPROVEMENT_ROADMAP.md - 阶段一任务1.3')
        code.append(f'特征数量: {len(self.selected_features)}')
        code.append('')
        code.append('降维策略:')
        code.append('1. 强制禁用可靠性<40的特征')
        code.append('2. 禁用高频不可靠特征（得分40-60）')
        code.append('3. 保留日线可靠特征')
        code.append('4. 保留封板基础特征')
        code.append('5. 保留历史统计特征')
        code.append('"""')
        code.append('')
        code.append('import pandas as pd')
        code.append('import numpy as np')
        code.append('from typing import Dict, List')
        code.append('')
        code.append('')
        code.append('class CoreFeaturesV1:')
        code.append('    """核心特征集 v1.0"""')
        code.append('    ')
        code.append('    # 核心特征列表')
        code.append('    CORE_FEATURES = [')
        
        # 按类别组织特征
        features_by_category = {}
        for feature in self.selected_features:
            category = self.feature_scores[feature]['category']
            if category not in features_by_category:
                features_by_category[category] = []
            features_by_category[category].append(feature)
        
        for category, features in sorted(features_by_category.items()):
            code.append(f'        # {category}')
            for feature in sorted(features):
                code.append(f"        '{feature}',")
        
        code.append('    ]')
        code.append('    ')
        code.append(f'    # 特征数量: {len(self.selected_features)}')
        code.append('    ')
        code.append('    @classmethod')
        code.append('    def get_features(cls) -> List[str]:')
        code.append('        """获取核心特征列表"""')
        code.append('        return cls.CORE_FEATURES')
        code.append('    ')
        code.append('    @classmethod')
        code.append('    def get_feature_count(cls) -> int:')
        code.append('        """获取特征数量"""')
        code.append('        return len(cls.CORE_FEATURES)')
        code.append('    ')
        code.append('    @classmethod')
        code.append('    def validate_features(cls, df: pd.DataFrame) -> bool:')
        code.append('        """验证数据框是否包含所有核心特征"""')
        code.append('        missing = set(cls.CORE_FEATURES) - set(df.columns)')
        code.append('        if missing:')
        code.append('            print(f"缺失特征: {missing}")')
        code.append('            return False')
        code.append('        return True')
        code.append('')
        code.append('')
        code.append('# 快速访问')
        code.append('CORE_FEATURES = CoreFeaturesV1.CORE_FEATURES')
        code.append('FEATURE_COUNT = CoreFeaturesV1.get_feature_count()')
        code.append('')
        
        # 写入文件
        code_text = '\n'.join(code)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code_text)
        
        print(f"   ✅ 代码已生成: {output_path}")
        
        return code_text
    
    def generate_report(self, output_path: str = None) -> str:
        """生成降维报告"""
        print("\n📄 生成降维报告...")
        
        if output_path is None:
            output_path = project_root / 'reports' / 'feature_reduction_report.md'
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# 特征降维报告\n\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**任务来源**: docs/IMPROVEMENT_ROADMAP.md - 阶段一任务1.3\n")
        report.append(f"**目标**: 生成≤{self.max_features}个核心特征的精简集\n")
        report.append("\n---\n\n")
        
        # 1. 降维概述
        report.append("## 1. 降维概述\n\n")
        total_features = len(self.feature_scores)
        selected_count = len(self.selected_features)
        rejected_count = len(self.rejected_features)
        reduction_rate = (rejected_count / total_features * 100) if total_features > 0 else 0
        
        report.append(f"- **原始特征数**: {total_features}\n")
        report.append(f"- **保留特征数**: {selected_count}\n")
        report.append(f"- **拒绝特征数**: {rejected_count}\n")
        report.append(f"- **降维率**: {reduction_rate:.1f}%\n\n")
        
        # 2. 降维策略
        report.append("## 2. 降维策略\n\n")
        report.append("### 强制禁用规则\n\n")
        report.append("1. 可靠性得分 < 40 的特征\n")
        report.append("2. 高频特征得分 40-60（数据粒度不足）\n\n")
        
        report.append("### 保留规则\n\n")
        report.append("1. ✅ **日线价量特征** (最可靠): close, volume, ret_1d 等\n")
        report.append("2. ✅ **技术指标** (很可靠): MA, MACD, RSI, BOLL 等\n")
        report.append("3. ✅ **封板基础特征** (较可靠): 涨停时间, 连板天数 等\n")
        report.append("4. ✅ **历史统计特征** (可靠): 历史涨停次数, 历史收益 等\n")
        report.append("5. ✅ **市场环境特征** (可靠): 市场涨停数, 指数收益 等\n\n")
        
        # 3. 保留的核心特征
        report.append("## 3. 保留的核心特征\n\n")
        report.append(f"共 **{selected_count}** 个核心特征：\n\n")
        
        # 按类别展示
        features_by_category = {}
        for feature in self.selected_features:
            category = self.feature_scores[feature]['category']
            if category not in features_by_category:
                features_by_category[category] = []
            features_by_category[category].append({
                'name': feature,
                'score': self.feature_scores[feature]['score'],
                'reliability': self.feature_scores[feature]['reliability']
            })
        
        for category, features in sorted(features_by_category.items()):
            count = len(features)
            report.append(f"### {category} ({count}个)\n\n")
            report.append("| 特征名称 | 评分 | 可靠性 |\n")
            report.append("|----------|------|--------|\n")
            for f in sorted(features, key=lambda x: x['score'], reverse=True):
                report.append(f"| {f['name']} | {f['score']:.1f} | {f['reliability']} |\n")
            report.append("\n")
        
        # 4. 被拒绝的特征
        report.append("## 4. 被拒绝的特征\n\n")
        report.append(f"共 **{rejected_count}** 个特征被拒绝：\n\n")
        
        if rejected_count > 0:
            report.append("| 特征名称 | 评分 | 可靠性 | 拒绝原因 |\n")
            report.append("|----------|------|--------|----------|\n")
            
            for feature in sorted(self.rejected_features):
                info = self.feature_scores.get(feature, {})
                score = info.get('score', 0)
                reliability = info.get('reliability', '未知')
                
                # 判断拒绝原因
                if score < 40:
                    reason = '评分<40，严重不可靠'
                elif score < 60 and info.get('category') == 'high_freq':
                    reason = '高频特征数据粒度不足'
                elif feature not in self.selected_features and len(self.selected_features) >= self.max_features:
                    reason = f'超过最大特征数{self.max_features}，评分较低'
                else:
                    reason = '其他原因'
                
                report.append(f"| {feature} | {score:.1f} | {reliability} | {reason} |\n")
        else:
            report.append("✅ 无特征被拒绝\n")
        
        report.append("\n")
        
        # 5. 关键建议
        report.append("## 5. 关键建议\n\n")
        
        avg_score = np.mean([info['score'] for info in self.feature_scores.values()])
        selected_avg_score = np.mean([self.feature_scores[f]['score'] for f in self.selected_features])
        
        report.append(f"### 📊 特征质量统计\n\n")
        report.append(f"- **原始特征平均分**: {avg_score:.1f}\n")
        report.append(f"- **核心特征平均分**: {selected_avg_score:.1f}\n")
        report.append(f"- **质量提升**: +{selected_avg_score - avg_score:.1f}分\n\n")
        
        report.append("### 💡 下一步行动\n\n")
        report.append("根据 `docs/IMPROVEMENT_ROADMAP.md`:\n\n")
        report.append("1. ✅ **完成**: 特征降维（当前任务）\n")
        report.append("2. ⏭️ **下一步**: 使用核心特征集训练基准模型 (`scripts/train_baseline_model.py`)\n")
        report.append("3. 📌 **验证**: 对比使用核心特征前后的模型性能\n\n")
        
        report.append("### ⚠️ 重要提醒\n\n")
        report.append("- 核心特征集已保存到 `features/core_features_v1.py`\n")
        report.append("- 后续训练请使用 `from features.core_features_v1 import CORE_FEATURES`\n")
        report.append("- 禁用的高频特征可在获得更高粒度数据后重新启用\n\n")
        
        report.append("---\n\n")
        report.append("*本报告由 Qilin Stack 特征降维系统自动生成*\n")
        
        # 写入文件
        report_text = ''.join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"   ✅ 报告已生成: {output_path}")
        
        return report_text
    
    def run_full_pipeline(self, test_report_path: str = None) -> Dict:
        """运行完整降维流程"""
        print("\n" + "="*70)
        print("🚀 开始特征降维流程")
        print("="*70)
        
        # 1. 加载测试结果
        test_results = self.load_test_results(test_report_path)
        
        # 2. 评估特征
        self.evaluate_features(test_results)
        
        # 3. 选择核心特征
        self.select_core_features()
        
        # 4. 生成特征模块
        self.generate_feature_module()
        
        # 5. 生成报告
        self.generate_report()
        
        print("\n" + "="*70)
        print("✅ 特征降维完成！")
        print(f"   核心特征数: {len(self.selected_features)}/{self.max_features}")
        print(f"   降维率: {len(self.rejected_features)/len(self.feature_scores)*100:.1f}%")
        print("="*70)
        
        return {
            'selected_features': self.selected_features,
            'rejected_features': self.rejected_features,
            'feature_scores': self.feature_scores
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征降维工具')
    parser.add_argument('--test-report', type=str, default=None,
                      help='高频特征测试报告路径（CSV）')
    parser.add_argument('--max-features', type=int, default=50,
                      help='最大特征数量')
    parser.add_argument('--output', type=str, default=None,
                      help='输出特征模块路径')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = CoreFeatureGenerator(max_features=args.max_features)
    
    # 运行完整流程
    results = generator.run_full_pipeline(test_report_path=args.test_report)
    
    # 如果指定了输出路径，生成到指定位置
    if args.output:
        generator.generate_feature_module(output_path=args.output)
    
    return results


if __name__ == '__main__':
    main()
