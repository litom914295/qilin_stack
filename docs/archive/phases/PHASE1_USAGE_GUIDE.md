# Phase 1 使用指南

> **完整集成的竞价预测系统改进方案使用手册**  
> **版本**: v1.0  
> **更新日期**: 2025-10-30

---

## 📋 目录

1. [快速开始](#快速开始)
2. [核心模块介绍](#核心模块介绍)
3. [完整Pipeline使用](#完整pipeline使用)
4. [独立模块使用](#独立模块使用)
5. [最佳实践](#最佳实践)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 最简单的方式: 使用统一Pipeline

```python
from qlib_enhanced.unified_phase1_pipeline import UnifiedPhase1Pipeline
import pandas as pd

# 1. 创建Pipeline实例
pipeline = UnifiedPhase1Pipeline(output_dir="output/my_pipeline")

# 2. 准备数据
data_sources = {
    'Qlib': qlib_df,      # 你的Qlib数据
    'AKShare': akshare_df  # 你的AKShare数据
}

full_feature_df = pd.read_csv('your_features.csv')  # 包含date, target和特征列

# 3. 运行完整Pipeline
results = pipeline.run_full_pipeline(
    data_sources=data_sources,
    full_feature_df=full_feature_df,
    target_col='target',
    date_col='date'
)

# 4. 查看结果
print(f"数据质量评分: {results['data_quality']['avg_coverage']:.2%}")
print(f"核心特征数: {results['core_features']['n_features']}")
print(f"活跃因子数: {len(results['factor_health']['active_factors'])}")
print(f"基准模型AUC: {results['baseline_model']['val_auc']:.4f}")
print(f"市场情绪评分: {results['market_factors']['sentiment_comprehensive_sentiment_score']:.1f}/100")
```

**就这么简单!** Pipeline会自动完成:
- ✅ 数据质量审计
- ✅ 核心特征筛选  
- ✅ 因子健康监控
- ✅ 基准模型训练
- ✅ Walk-Forward验证
- ✅ 宏观市场因子计算

---

## 🧩 核心模块介绍

### 1. 数据质量审计 (Phase 1.1)

**用途**: 评估数据源质量,发现数据问题

```python
from scripts.audit_data_quality import DataQualityAuditor

auditor = DataQualityAuditor(output_dir="output/audit")
results = auditor.run_full_audit({
    'Qlib': qlib_df,
    'AKShare': akshare_df
})

print(f"覆盖率: {results['avg_coverage']:.2%}")
print(f"缺失值: {results['avg_missing_ratio']:.2%}")
```

### 2. 核心特征生成 (Phase 1.1)

**用途**: 从100+特征中筛选50个核心特征

```python
from scripts.generate_core_features import CoreFeatureGenerator

generator = CoreFeatureGenerator(max_features=50)
core_features_df = generator.select_core_features(
    full_feature_df,
    target_col='target'
)

print(f"精简后特征数: {len(core_features_df.columns)}")
```

### 3. 因子衰减监控 (Phase 1.2)

**用途**: 监控因子IC,识别失效因子

```python
from monitoring.factor_decay_monitor import FactorDecayMonitor

monitor = FactorDecayMonitor(ic_windows=[20, 60, 120])
report = monitor.batch_calculate_factor_ic(
    factor_data=feature_df[factor_cols],
    forward_returns=feature_df['forward_return'],
    factor_names=factor_cols
)

# 查看因子健康状态
for factor, metrics in report['factors'].items():
    print(f"{factor}: IC={metrics['ic_mean_60d']:.4f}, IR={metrics['ic_ir_60d']:.2f}")
```

### 4. 因子生命周期管理 (Phase 1.2)

**用途**: 自动管理因子状态和权重

```python
from factors.factor_lifecycle_manager import FactorLifecycleManager

manager = FactorLifecycleManager(ic_threshold=0.02)

# 更新因子IC
for factor in factors:
    ic = calculate_ic(factor)
    status = manager.update_factor(factor, ic)
    print(f"{factor}: {status}")

# 获取活跃因子
active_factors = manager.get_active_factors()
weights = manager.get_factor_weights(active_factors)
```

### 5. Walk-Forward验证 (Phase 1.3)

**用途**: 严格的滚动回测验证

```python
from scripts.walk_forward_validator import WalkForwardValidator, WalkForwardConfig
from sklearn.metrics import roc_auc_score

# 配置
config = WalkForwardConfig(
    train_window=180,
    test_window=60,
    step_size=30,
    purge_days=5
)

# 模型工厂
def model_factory():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=100, max_depth=6)

# 评估指标
metrics_funcs = {
    'AUC': lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
}

# 创建验证器
validator = WalkForwardValidator(
    config=config,
    model_factory=model_factory,
    metrics_funcs=metrics_funcs
)

# 运行验证
summary = validator.run_validation(
    df=full_df,
    feature_cols=feature_cols,
    target_col='target',
    date_col='date'
)

print(f"平均AUC: {summary['aggregate_metrics']['AUC_mean']:.4f}")
print(f"AUC标准差: {summary['aggregate_metrics']['AUC_std']:.4f}")
```

### 6. 多分类训练 (Phase 1.3)

**用途**: 训练涨/平/跌三分类模型

```python
from scripts.multiclass_trainer import MulticlassTrainer
from lightgbm import LGBMClassifier

# 创建标签(从收益率)
y_multiclass = MulticlassTrainer.create_labels_from_returns(
    returns=df['forward_return'],
    up_threshold=0.02,
    down_threshold=-0.02
)

# 创建训练器
model = LGBMClassifier(n_estimators=100)
trainer = MulticlassTrainer(
    model=model,
    n_classes=3,
    class_names=['下跌', '平稳', '上涨'],
    balance_method='class_weight'
)

# 训练
trainer.train(X_train, y_train, X_val, y_val)

# 评估
metrics = trainer.evaluate(X_test, y_test, prefix='test')
print(f"准确率: {metrics['test_accuracy']:.4f}")
print(f"F1-score: {metrics['test_macro_avg_f1']:.4f}")
```

### 7. 模型对比 (Phase 1.3)

**用途**: 对比多个模型性能

```python
from scripts.model_comparison_report import ModelComparisonReport

reporter = ModelComparisonReport(output_dir="output/comparison")

# 添加模型结果
reporter.add_model(
    model_name='LightGBM',
    metrics={'AUC': 0.72, 'Accuracy': 0.65},
    metadata={'n_estimators': 100, 'max_depth': 6}
)

reporter.add_model(
    model_name='RandomForest',
    metrics={'AUC': 0.68, 'Accuracy': 0.62},
    metadata={'n_estimators': 200, 'max_depth': 10}
)

# 生成报告
report = reporter.generate_report()

# 生成可视化
reporter.plot_metrics_comparison(plot_type='bar')
reporter.plot_metrics_comparison(plot_type='radar')
```

### 8. 宏观市场因子 (Phase 1.4)

**用途**: 计算市场情绪/题材/流动性因子

```python
from features.market_sentiment_factors import MarketSentimentFactors
from features.theme_diffusion_factors import ThemeDiffusionFactors  
from features.liquidity_volatility_factors import LiquidityVolatilityFactors

# 市场情绪
sentiment_calc = MarketSentimentFactors()
sentiment_factors = sentiment_calc.calculate_all_factors(
    date='2025-01-01',
    market_data=market_df
)
print(f"情绪评分: {sentiment_factors['comprehensive_sentiment_score']:.1f}/100")
print(f"市场状态: {sentiment_factors['market_regime']}")

# 题材扩散
theme_calc = ThemeDiffusionFactors()
theme_factors = theme_calc.calculate_all_factors(
    date='2025-01-01',
    market_data=market_df
)
print(f"热门题材: {theme_factors['top_1_theme_name']}")
print(f"龙头数量: {theme_factors['total_leader_count']}")

# 流动性波动率
liquidity_calc = LiquidityVolatilityFactors()
liquidity_factors = liquidity_calc.calculate_all_factors(
    date='2025-01-01',
    market_data=market_df
)
print(f"流动性健康: {liquidity_factors['liquidity_health_score']:.1f}/100")
print(f"波动率状态: {liquidity_factors['volatility_regime']}")
```

---

## 🔄 完整Pipeline使用

### 方式一: 一键运行所有模块

```python
from qlib_enhanced.unified_phase1_pipeline import UnifiedPhase1Pipeline

# 创建Pipeline
pipeline = UnifiedPhase1Pipeline(
    config={
        'data_quality': {
            'min_coverage': 0.95,
            'max_missing_ratio': 0.05
        },
        'feature_selection': {
            'max_features': 50
        },
        'walk_forward': {
            'train_window': 180,
            'test_window': 60
        }
    },
    output_dir="output/my_pipeline"
)

# 运行完整Pipeline
results = pipeline.run_full_pipeline(
    data_sources={'Qlib': qlib_df, 'AKShare': akshare_df},
    full_feature_df=feature_df,
    target_col='target',
    date_col='date'
)

# 结果会保存到: output/my_pipeline/full_pipeline_results.json
```

### 方式二: 分步骤运行

```python
# 1. 数据质量审计
audit_results = pipeline.run_data_quality_audit({'Qlib': qlib_df})

# 2. 生成核心特征
core_features = pipeline.generate_core_features(full_feature_df, 'target')

# 3. 监控因子健康
factor_health = pipeline.monitor_factor_health(
    factor_data=core_features[factor_cols],
    forward_returns=core_features['target']
)

# 4. 训练基准模型
model_results = pipeline.train_baseline_model(X_train, y_train, X_val, y_val)

# 5. Walk-Forward验证
wf_results = pipeline.run_walk_forward_validation(
    df=core_features,
    feature_cols=feature_cols,
    target_col='target'
)

# 6. 计算市场因子
market_factors = pipeline.calculate_market_factors(date='2025-01-01')

# 7. 对比模型
comparison = pipeline.compare_models([
    {'model_name': 'Model1', 'metrics': {'AUC': 0.72}},
    {'model_name': 'Model2', 'metrics': {'AUC': 0.68}}
])
```

---

## 💡 最佳实践

### 1. 数据准备

**必要的列**:
- `date`: 日期(YYYY-MM-DD格式)
- `symbol`: 股票代码(如果是个股数据)
- `target`: 目标变量(收益率或分类标签)
- 其他特征列

**示例数据格式**:
```python
import pandas as pd

df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', ...],
    'symbol': ['000001', '000002', ...],
    'target': [0.05, -0.02, ...],
    'feature_1': [...],
    'feature_2': [...],
    # ... 更多特征
})
```

### 2. 特征筛选建议

- 初始特征数: 50-150个
- 精简后特征: 30-50个
- 相关性阈值: 0.8
- 最小重要性: 0.01

### 3. Walk-Forward配置建议

- **训练窗口**: 120-240天(4-8个月)
- **测试窗口**: 20-60天(1-2个月)
- **步长**: 20-30天
- **Purge期**: 3-7天

### 4. 因子监控建议

- **IC窗口**: [20, 60, 120]天
- **最小IC**: 0.02
- **衰减阈值**: 0.5(相对历史均值)
- **监控频率**: 每日或每周

### 5. 模型训练建议

**基准模型参数**:
```python
{
    'model_type': 'lgbm',
    'n_estimators': 100-200,
    'max_depth': 5-8,
    'learning_rate': 0.03-0.1,
    'min_child_samples': 20-50
}
```

**多分类参数**:
```python
{
    'up_threshold': 0.02,      # 上涨阈值2%
    'down_threshold': -0.02,   # 下跌阈值-2%
    'balance_method': 'class_weight'  # 推荐
}
```

---

## 🛠️ 独立模块使用场景

### 场景1: 只想做数据质量检查

```python
from scripts.audit_data_quality import DataQualityAuditor

auditor = DataQualityAuditor()
results = auditor.run_full_audit({
    'Source1': df1,
    'Source2': df2
})
```

### 场景2: 只想监控因子IC

```python
from monitoring.factor_decay_monitor import FactorDecayMonitor

monitor = FactorDecayMonitor()
report = monitor.batch_calculate_factor_ic(
    factor_data=factors_df,
    forward_returns=returns_series,
    factor_names=['factor1', 'factor2']
)
```

### 场景3: 只想做Walk-Forward验证

```python
from scripts.walk_forward_validator import WalkForwardValidator, WalkForwardConfig

config = WalkForwardConfig(train_window=180, test_window=60)
validator = WalkForwardValidator(config, model_factory, metrics_funcs)
summary = validator.run_validation(df, feature_cols, target_col)
```

### 场景4: 只想计算市场情绪

```python
from features.market_sentiment_factors import MarketSentimentFactors

calc = MarketSentimentFactors()
factors = calc.calculate_all_factors(date='2025-01-01', market_data=df)
```

---

## ❓ 常见问题

### Q1: Pipeline运行很慢怎么办?

**A**: 可以:
1. 减少特征数量
2. 减少Walk-Forward的fold数(增大step_size)
3. 使用更快的模型(如LightGBM而不是深度学习)
4. 并行处理(修改代码添加`n_jobs=-1`)

### Q2: 数据格式不匹配怎么办?

**A**: 确保数据包含必要列:
```python
required_cols = ['date', 'target']
feature_cols = [col for col in df.columns if col not in required_cols]

# 检查
assert 'date' in df.columns, "缺少date列"
assert 'target' in df.columns, "缺少target列"
assert len(feature_cols) > 0, "没有特征列"
```

### Q3: 如何保存和加载训练好的模型?

**A**: 模型会自动保存到output目录:
```python
# 保存位置
baseline_model: output/baseline_model/model.pkl
multiclass_model: output/multiclass_model/multiclass_model.pkl

# 加载
import pickle
with open('output/baseline_model/model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Q4: 如何自定义配置?

**A**: 传入config字典:
```python
custom_config = {
    'feature_selection': {
        'max_features': 30,  # 只要30个特征
        'min_importance': 0.02
    },
    'walk_forward': {
        'train_window': 240,  # 更长的训练窗口
        'test_window': 30
    },
    'baseline_model': {
        'n_estimators': 200,  # 更多树
        'learning_rate': 0.03
    }
}

pipeline = UnifiedPhase1Pipeline(config=custom_config)
```

### Q5: 如何集成到现有系统?

**A**: 两种方式:

**方式1: 作为预处理步骤**
```python
# 在现有系统前添加
pipeline = UnifiedPhase1Pipeline()
core_features = pipeline.generate_core_features(raw_features, 'target')
active_factors = pipeline.get_active_factors()

# 然后用core_features和active_factors继续现有流程
your_existing_system(core_features, active_factors)
```

**方式2: 替换现有模块**
```python
# 用新的验证器替换旧的
from scripts.walk_forward_validator import WalkForwardValidator
validator = WalkForwardValidator(...)
# 集成到你的回测系统
```

### Q6: 输出文件在哪里?

**A**: 所有输出默认保存在`output/`目录:
```
output/
├── unified_pipeline/
│   ├── full_pipeline_results.json  # 完整结果
│   ├── data_quality/               # 数据质量报告
│   ├── core_features/              # 核心特征
│   ├── factor_health/              # 因子健康报告
│   ├── baseline_model/             # 基准模型
│   ├── walk_forward/               # WF验证结果
│   ├── multiclass_model/           # 多分类模型
│   └── model_comparison/           # 模型对比
```

### Q7: 如何解读结果?

**A**: 关键指标解读:

**数据质量**:
- 覆盖率 >95%: 优秀
- 缺失值 <5%: 良好
- 异常值 <2%: 正常

**因子健康**:
- IC >0.05: 强因子
- IC 0.02-0.05: 中等因子
- IC <0.02: 弱因子
- IR >1.0: 稳定因子

**模型性能**:
- AUC >0.70: 优秀
- AUC 0.65-0.70: 良好
- AUC <0.65: 需改进
- AUC标准差 <0.05: 稳定

**市场情绪**:
- 评分 >70: 强势市场
- 评分 50-70: 正常市场
- 评分 <50: 弱势市场

---

## 📞 技术支持

### 遇到问题?

1. **查看日志**: 所有模块都有详细日志输出
2. **查看文档**: `docs/IMPROVEMENT_ROADMAP.md`和`docs/PHASE1_PROGRESS.md`
3. **查看示例**: 每个模块文件末尾都有`example_usage()`函数
4. **调试模式**: 设置`logging.basicConfig(level=logging.DEBUG)`

### 反馈建议

欢迎提供改进建议和bug报告!

---

## 📚 相关文档

- [改进路线图](./IMPROVEMENT_ROADMAP.md) - 完整的三阶段改进计划
- [Phase 1进度](./PHASE1_PROGRESS.md) - 详细的开发进度
- [第一周快速开始](./WEEK1_QUICKSTART.md) - 第一周任务指南

---

**文档版本**: v1.0  
**创建日期**: 2025-10-30  
**适用版本**: Phase 1 Complete

祝使用愉快! 🚀
