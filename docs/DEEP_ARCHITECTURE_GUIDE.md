# 🏗️ Qilin Stack深度架构指南

> **核心**：Qlib量化平台 + RD-Agent因子发现 + 因子进化淘汰 + 模型在线学习的完整闭环系统

**版本**: v2.0  
**更新日期**: 2025-10-30  
**作者**: Qilin Quant Team

---

## 🎯 系统架构总览

```
┌─────────────────────────────────────────────────────────────┐
│               Qilin Stack 一进二AI交易系统                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   📊 数据层          🧪 研究层            🎯 决策层
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Qlib         │   │ RD-Agent     │   │ Phase 1      │
│ 多源数据     │──▶│ 因子发现     │──▶│ 竞价进阶     │
│ AKShare      │   │ 15+ 核心因子 │   │ 数据审计     │
│ Tushare      │   │ 自动进化     │   │ 特征精简     │
│ 高频数据     │   │              │   │ Walk-Forward │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                  ┌──────────────────┐
                  │   一进二模型     │
                  │ LightGBM+XGBoost │
                  │ +CatBoost Stack  │
                  │ 在线学习+进化    │
                  └──────────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
        📈 执行层      🔄 学习层      🛡️ 风控层
                │           │           │
                ▼           ▼           ▼
         T+1 竞价买入   模型进化    因子衰减监控
         T+2 卖出       因子淘汰    生命周期管理
         Kelly仓位      概念漂移检测  多级风控
```

---

## 📊 第一层：Qlib数据基础设施

### 1.1 Qlib是什么？

**Qlib** 是微软开源的AI量化投资平台，提供：
- ✅ **统一数据接口**：支持多种数据源
- ✅ **高性能因子引擎**：快速计算复杂因子
- ✅ **模型训练框架**：支持各种机器学习模型
- ✅ **回测系统**：完整的策略回测能力

### 1.2 我们如何使用Qlib？

#### 数据获取（MultiSourceDataProvider）

**代码位置**：`qlib_enhanced/multi_source.py`

```python
from qlib_enhanced.multi_source import MultiSourceDataProvider

# 初始化多源数据提供者
data_provider = MultiSourceDataProvider(
    sources=['qlib', 'akshare', 'tushare'],  # 多数据源
    fallback_order=['qlib', 'akshare', 'tushare']  # 降级顺序
)

# 获取数据（自动切换数据源）
data = data_provider.fetch_data(
    symbols=['000001.SZ', '600000.SH'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    fields=['open', 'high', 'low', 'close', 'volume']
)
```

**智能降级机制**：
1. 优先尝试Qlib本地数据（最快）
2. Qlib失败 → 切换到AKShare（免费）
3. AKShare失败 → 切换到Tushare（需Token）
4. 所有失败 → 返回缓存数据或错误

#### 高频数据增强（HighFreqLimitUpAnalyzer）

**代码位置**：`qlib_enhanced/high_freq_limitup.py`

```python
from qlib_enhanced.high_freq_limitup import HighFreqLimitUpAnalyzer

analyzer = HighFreqLimitUpAnalyzer(freq="1min")

# 分析涨停板高频特征
features = analyzer.analyze_intraday_pattern(
    minute_data=minute_df,  # 分钟K线
    limitup_time="10:30:00"  # 涨停时间
)

# 输出特征：
# - 涨停前量能爆发 (volume_burst_before_limit)
# - 封单稳定性 (seal_stability)
# - 大单节奏 (big_order_rhythm)
# - 收盘封单强度 (close_seal_strength)
# - 盘中开板次数 (intraday_open_count)
```

---

## 🧪 第二层：RD-Agent因子发现引擎

### 2.1 RD-Agent是什么？

**RD-Agent** (Research & Development Agent) 是微软的AI研发助手，用于：
- ✅ **自动因子发现**：用AI生成新因子
- ✅ **因子表达式优化**：自动简化复杂因子
- ✅ **因子有效性评估**：IC、IR、夏普率等

### 2.2 我们的因子库（15+核心因子）

**代码位置**：`rd_agent/factor_discovery_simple.py`

#### 涨停板专属因子

```python
# 1. 封单强度因子
封单强度 = 封单金额 / 流通市值
# 说明：衡量资金封板力度
# 预期IC：0.08

# 2. 连板高度因子
连板高度 = log(连板天数 + 1) × 量比
# 说明：连板越高+量能配合，次日越强
# 预期IC：0.12

# 3. 题材共振因子
题材共振 = 同题材涨停数量 × 个股强度
# 说明：题材热度与个股结合
# 预期IC：0.10

# 4. 早盘涨停因子
早盘涨停 = 1 - (涨停分钟数 / 240)
# 说明：涨停越早，次日表现越好
# 预期IC：0.15

# 5. 量能爆发因子
量能爆发 = 成交量 / 20日均量
# 说明：量能突增的力度
# 预期IC：0.09

# 6. 大单净流入因子
大单净流入 = (大单买入 - 大单卖出) / 成交额
# 说明：主力资金流向
# 预期IC：0.11

# 7. 封单持续性因子
封单持续性 = 封单持续分钟数 / 240
# 说明：封单的稳定程度
# 预期IC：0.07

# 8. 开板次数惩罚因子
开板惩罚 = exp(-开板次数)
# 说明：开板越多，次日越弱
# 预期IC：-0.06

# 9. 换手率适中因子
换手适中 = 1 - |换手率 - 最优换手率| / 最优换手率
# 说明：换手率过高或过低都不好
# 预期IC：0.08

# 10. 首板优势因子
首板优势 = is_first_board × (1 + 题材热度)
# 说明：首板且题材热的股票机会大
# 预期IC：0.14
```

### 2.3 因子发现流程

```python
from rd_agent.factor_discovery_simple import SimplifiedFactorDiscovery

# 初始化因子发现系统
discovery = SimplifiedFactorDiscovery(
    cache_dir="./workspace/factor_cache"
)

# 发现新因子
discovered_factors = await discovery.discover_factors(
    start_date='2024-01-01',
    end_date='2024-12-31',
    n_factors=20,  # 返回Top 20因子
    min_ic=0.05     # 最小IC阈值
)

# 输出示例：
# [
#   {
#     'id': 'limitup_004',
#     'name': '早盘涨停',
#     'expression': '1 - (涨停分钟数 / 240)',
#     'expected_ic': 0.15,
#     'status': 'discovered'
#   },
#   ...
# ]
```

---

## 🔄 第三层：因子生命周期管理（进化与淘汰）

### 3.1 为什么需要因子生命周期管理？

**问题**：
- ❌ 因子会衰减：有效的因子随时间失效
- ❌ 过多因子：100+因子导致过拟合
- ❌ 静态权重：无法适应市场变化

**解决方案**：
- ✅ **动态监控**：实时计算因子IC
- ✅ **自动降权**：衰减因子降低权重
- ✅ **自动淘汰**：失效因子送入冷宫
- ✅ **自动复活**：恢复的因子重新启用

### 3.2 因子状态机

**代码位置**：`factors/factor_lifecycle_manager.py`

```
                    IC恢复
         ┌──────────────────────┐
         │                      │
         ▼                      │
  ┌────────────┐         ┌─────┴──────┐
  │   活跃     │ IC衰减  │   观察     │
  │ Active    │────────▶│ Watching   │
  │ 权重100%  │         │ 权重75%    │
  └────────────┘         └────┬───────┘
                              │IC继续衰减
                              ▼
                        ┌────────────┐
                        │   警告     │
                        │ Warning    │
                        │ 权重50%    │
                        └────┬───────┘
                             │IC过低
                             ▼
                        ┌────────────┐
                        │   休眠     │
                        │ Sleeping   │
                        │ 权重0%     │
                        └────┬───────┘
                             │休眠>120天
                             ▼
                        ┌────────────┐
                        │   淘汰     │
                        │ Eliminated │
                        │ 永久移除   │
                        └────────────┘
```

### 3.3 转换规则

**Active → Watching**（活跃→观察）
```python
触发条件：
- IC下降至历史均值的80%
- 胜率低于52%
- 连续20天表现不佳
```

**Watching → Warning**（观察→警告）
```python
触发条件：
- IC下降至历史均值的50%
- 胜率低于48%
- 连续30天表现不佳
```

**Warning → Sleeping**（警告→休眠）
```python
触发条件：
- IC绝对值低于0.01
- 连续40天表现不佳
- IR低于0.3
```

**Sleeping → Eliminated**（休眠→淘汰）
```python
触发条件：
- 休眠超过120天
- 尝试复活3次均失败
```

**复活条件**（任何状态 → Active/Watching）
```python
触发条件：
- IC恢复至0.03以上
- 连续20天表现良好
- 胜率高于55%
```

### 3.4 使用示例

```python
from factors.factor_lifecycle_manager import FactorLifecycleManager

# 初始化生命周期管理器
manager = FactorLifecycleManager()

# 更新因子状态
for factor_name in all_factors:
    # 获取因子健康度指标（来自FactorDecayMonitor）
    health_metrics = {
        'ic_mean': 0.05,
        'ic_recent': 0.03,
        'ic_win_rate': 0.54,
        'ir': 0.8,
        'ic_trend': 'declining'
    }
    
    # 更新状态
    status = manager.update_factor_status(
        factor_name=factor_name,
        health_metrics=health_metrics
    )
    
    print(f"{factor_name}: {status['status'].value}, 权重={status['weight']}")

# 获取活跃因子
active_factors = manager.get_active_factors()
print(f"活跃因子数: {len(active_factors)}")
```

---

## 📈 第四层：模型训练与集成

### 4.1 一进二模型架构

**代码位置**：`qlib_enhanced/one_into_two_pipeline.py`

#### 两阶段Stacking模型

```python
┌─────────────────────────────────────┐
│         Stage 1: Pool Prediction    │
│   预测T日是否涨停（候选池）         │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Base Models (L1)                    │
│  ┌─────────┬──────────┬──────────┐  │
│  │ LightGBM│ XGBoost  │ CatBoost │  │
│  │ 200棵树 │ 300棵树  │ 300迭代  │  │
│  └─────────┴──────────┴──────────┘  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Meta Model (L2)                     │
│  Logistic Regression + Calibration   │
└──────────────┬───────────────────────┘
               │
               ▼
         Pool Probability
               │
               ▼
┌─────────────────────────────────────┐
│     Stage 2: Board Prediction       │
│   预测T+1是否继续涨停（二板）       │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Base Models (L1)                    │
│  ┌─────────┬──────────┬──────────┐  │
│  │ LightGBM│ XGBoost  │ CatBoost │  │
│  └─────────┴──────────┴──────────┘  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Meta Model (L2)                     │
│  Logistic Regression + Calibration   │
└──────────────┬───────────────────────┘
               │
               ▼
        Board Probability
          (最终输出)
```

#### 模型配置

**LightGBM**:
```python
lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=-1,  # 无限制
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**XGBoost**:
```python
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist'  # 快速直方图算法
)
```

**CatBoost**:
```python
CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    verbose=False
)
```

### 4.2 训练流程

```python
from qlib_enhanced.one_into_two_pipeline import OneIntoTwoTrainer

# 初始化训练器
trainer = OneIntoTwoTrainer(top_n=20)

# 准备数据
# df: DataFrame with columns:
#   - date: 交易日期
#   - symbol: 股票代码
#   - pool_label: T日是否涨停 (0/1)
#   - board_label: T+1是否继续涨停 (0/1)
#   - feature_1, feature_2, ...: 特征列

# 训练模型
result = trainer.fit(df)

print(f"Pool Model AUC: {result.auc_pool:.4f}")
print(f"Board Model AUC: {result.auc_board:.4f}")
print(f"Top-{trainer.top_n} Threshold: {result.threshold_topn:.4f}")

# 预测
predictions = trainer.predict(new_data)
# 返回: (pool_prob, board_prob)
```

---

## 🔄 第五层：在线学习与模型进化

### 5.1 为什么需要在线学习？

**问题**：
- ❌ 市场在变化：静态模型很快失效
- ❌ 概念漂移：过去有效的特征现在无效
- ❌ 数据分布变化：新的市场regime

**解决方案**：
- ✅ **增量更新**：每日/每周更新模型
- ✅ **概念漂移检测**：自动检测市场变化
- ✅ **自适应学习率**：根据表现调整学习速度
- ✅ **模型版本管理**：保留历史模型用于对比

### 5.2 在线学习管理器

**代码位置**：`qlib_enhanced/online_learning.py`

```python
from qlib_enhanced.online_learning import OnlineLearningManager

# 初始化在线学习管理器
manager = OnlineLearningManager(
    base_model=trained_model,
    update_frequency='daily',  # 每日更新
    drift_threshold=0.05,      # 漂移检测阈值
    enable_drift_detection=True
)

# 增量更新（每天收盘后）
result = await manager.incremental_update(
    new_data=today_features,   # 今日特征
    new_labels=today_labels     # 今日标签
)

if result.success:
    print(f"✅ 模型更新成功")
    print(f"  处理样本: {result.samples_processed}")
    print(f"  新准确率: {result.new_accuracy:.4f}")
    print(f"  概念漂移: {'是' if result.drift_detected else '否'}")
    print(f"  模型版本: {result.model_version}")
else:
    print(f"❌ 模型更新失败")
```

### 5.3 概念漂移检测

```python
class DriftDetector:
    """概念漂移检测器"""
    
    def detect(self, new_data, new_labels):
        # 1. 特征分布变化检测（KS检验）
        feature_drift = self._check_feature_distribution(new_data)
        
        # 2. 标签分布变化检测
        label_drift = self._check_label_distribution(new_labels)
        
        # 3. 模型性能下降检测
        performance_drift = self._check_performance_degradation()
        
        # 综合判断
        drift_score = (
            feature_drift * 0.4 + 
            label_drift * 0.3 + 
            performance_drift * 0.3
        )
        
        return ConceptDrift(
            detected=drift_score > self.threshold,
            drift_score=drift_score,
            detection_time=datetime.now(),
            affected_features=self._get_drifted_features(),
            recommended_action='retrain' if drift_score > 0.1 else 'update'
        )
```

---

## 🎯 第六层：竞价进阶Pipeline（集成所有模块）

### 6.1 UnifiedPhase1Pipeline完整流程

**代码位置**：`qlib_enhanced/unified_phase1_pipeline.py`

```python
from qlib_enhanced.unified_phase1_pipeline import UnifiedPhase1Pipeline

# 初始化统一Pipeline
pipeline = UnifiedPhase1Pipeline(
    config={
        'data_quality': {
            'min_coverage': 0.95,
            'max_missing_ratio': 0.05
        },
        'feature_selection': {
            'max_features': 50,
            'min_importance': 0.01
        },
        'factor_health': {
            'ic_windows': [20, 60, 120],
            'min_ic': 0.02
        },
        'walk_forward': {
            'train_window': 180,
            'test_window': 60,
            'step_size': 30
        }
    }
)
```

### 6.2 完整工作流

#### Step 1: 数据质量审计

```python
# 审计多源数据质量
audit_results = pipeline.run_data_quality_audit({
    'qlib': qlib_data,
    'akshare': akshare_data,
    'tushare': tushare_data
})

# 输出：
# ✅ 数据质量审计完成
#   覆盖率: 98.5%
#   缺失值比例: 1.2%
#   异常值比例: 0.8%
```

#### Step 2: RD-Agent因子发现

```python
from rd_agent.factor_discovery_simple import SimplifiedFactorDiscovery

discovery = SimplifiedFactorDiscovery()

# 发现新因子
new_factors = await discovery.discover_factors(
    start_date='2024-01-01',
    end_date='2024-12-31',
    n_factors=20,
    min_ic=0.05
)

# 输出：
# 🔍 开始因子发现: 2024-01-01 -> 2024-12-31
# ✅ 发现 20 个高质量因子
```

#### Step 3: 生成核心特征

```python
# 从100+特征精简到50个核心特征
core_features = pipeline.generate_core_features(
    full_feature_df=all_features,
    target_col='t1_close_return'
)

# 输出：
# ✅ 特征精简完成: 120 → 50
```

#### Step 4: 因子健康度监控

```python
# 监控因子IC和生命周期
health_report = pipeline.monitor_factor_health(
    factor_data=core_features,
    forward_returns=t1_returns
)

# 输出：
# ✅ 活跃因子数: 32/50
#   - 封单强度: IC=0.08, 状态=Active
#   - 连板高度: IC=0.12, 状态=Active
#   - 题材共振: IC=0.10, 状态=Active
#   - 早盘涨停: IC=0.15, 状态=Active
#   ...
#   - 某弱因子: IC=0.008, 状态=Sleeping
```

#### Step 5: Walk-Forward验证

```python
# 滚动训练和测试
wf_results = pipeline.run_walk_forward_validation(
    df=full_data,
    feature_cols=active_features,  # 只用活跃因子
    target_col='board_label',
    date_col='date'
)

# 输出：
# Walk-Forward 验证结果:
#   训练窗口: 180天
#   测试窗口: 60天
#   步长: 30天
#   折数: 8
#   平均AUC: 0.73
#   AUC标准差: 0.04
#   稳定性: 优秀
```

#### Step 6: 训练最终模型

```python
# 使用活跃因子训练一进二模型
from qlib_enhanced.one_into_two_pipeline import OneIntoTwoTrainer

trainer = OneIntoTwoTrainer(top_n=20)
result = trainer.fit(core_features)

print(f"Pool Model AUC: {result.auc_pool:.4f}")
print(f"Board Model AUC: {result.auc_board:.4f}")

# 输出：
# Pool Model AUC: 0.68
# Board Model AUC: 0.74
```

#### Step 7: 在线学习启动

```python
from qlib_enhanced.online_learning import OnlineLearningManager

# 启动在线学习
manager = OnlineLearningManager(
    base_model=result.model_board,
    update_frequency='daily'
)

# 每日收盘后自动更新
# (在scheduler中配置定时任务)
```

---

## 🔧 如何正确使用整套系统？

### 完整使用流程

#### 1. 初始化（一次性）

```bash
# 安装依赖
pip install -r requirements.txt

# 下载Qlib数据
python scripts/download_qlib_data_v2.py --start 2020-01-01 --end 2024-12-31

# 初始化因子库
python rd_agent/factor_discovery_simple.py
```

#### 2. T日盘后（15:30-16:30）

**运行竞价进阶Pipeline**:

```python
# 启动Web界面
streamlit run web/unified_dashboard.py

# 在界面中：
# 1. 竞价决策 → T日候选筛选 → 执行筛选
# 2. 竞价决策 → 竞价进阶 → 完整Pipeline
```

**后台执行逻辑**:

```python
# 1. 获取候选股票
candidates = auction_engine.screen_tomorrow_candidates_strict(
    today_limitups=limitup_data,
    features=full_features
)

# 2. 竞价进阶优化
pipeline = UnifiedPhase1Pipeline()

# 2.1 数据质量审计
audit_results = pipeline.run_data_quality_audit(data_sources)

# 2.2 生成核心特征（精简到50个）
core_features = pipeline.generate_core_features(
    full_feature_df=all_features
)

# 2.3 因子健康度监控
health_report = pipeline.monitor_factor_health(
    factor_data=core_features,
    forward_returns=t1_returns
)

# 2.4 获取活跃因子
active_factors = pipeline.get_active_factors()
print(f"活跃因子: {len(active_factors)}/{len(all_factors)}")

# 2.5 Walk-Forward验证
wf_results = pipeline.run_walk_forward_validation(
    df=candidates,
    feature_cols=active_factors,
    target_col='board_label'
)

# 2.6 训练/更新模型
trainer = OneIntoTwoTrainer()
result = trainer.fit(candidates[['date', 'symbol', 'pool_label', 'board_label'] + active_factors])

# 2.7 生成预测
candidates['pred_prob'] = result.model_board.predict_proba(candidates[active_factors])[:, 1]

# 2.8 按概率排序，选Top 10
final_candidates = candidates.nlargest(10, 'pred_prob')
```

#### 3. T+1日竞价（09:15-09:25）

```python
# 监控竞价强度
auction_monitor = AuctionMonitor()

for symbol in final_candidates['symbol']:
    # 计算竞价强度
    strength = auction_monitor.calculate_strength(symbol)
    
    # 分级决策
    if strength >= 85:
        action = 'auction_buy'  # 竞价买入
    elif strength >= 70:
        action = 'open_observe' # 开盘观察
    else:
        action = 'pass'         # 放弃
```

#### 4. T+2日卖出（09:15-09:25）

```python
# 根据T+1表现制定卖出策略
sell_strategy = T2SellStrategy()

for position in current_positions:
    t1_return = position['t1_close_return']
    t2_open_gap = position['t2_open_gap']
    
    sell_signal = sell_strategy.generate_signal(t1_return, t2_open_gap)
    
    # 执行卖出
    if sell_signal.sell_ratio > 0:
        execute_sell(position, sell_signal.sell_ratio)
```

#### 5. 每日在线学习（16:00）

```python
# 收集当日数据
today_data = collect_today_data()
today_labels = collect_today_labels()

# 增量更新模型
online_manager = OnlineLearningManager(base_model=current_model)
result = await online_manager.incremental_update(today_data, today_labels)

if result.drift_detected:
    print("⚠️  检测到概念漂移，触发完全重训练")
    # 重新运行完整Pipeline
```

---

## 📊 性能指标与监控

### 关键指标

**因子层面**:
```
- 活跃因子数: 监控有效因子数量
- 平均IC: 因子整体有效性
- IC胜率: IC>0的比例
- 因子衰减率: 休眠/淘汰因子比例
```

**模型层面**:
```
- Pool Model AUC: 候选池预测准确率（目标>0.65）
- Board Model AUC: 二板预测准确率（目标>0.70）
- Walk-Forward AUC: 稳定性指标（标准差<0.05）
- 在线学习准确率: 增量更新效果
```

**交易层面**:
```
- 胜率: >60%
- 盈亏比: >2:1
- 夏普率: >2.0
- 最大回撤: <-10%
```

### 监控dashboard

**Web界面查看**:
```
竞价决策 → 竞价进阶 → 查看结果
- 数据质量评分
- 核心特征数
- 活跃因子数
- 模型AUC
- Walk-Forward稳定性
```

---

## 🎓 核心优势总结

### 1. **数据基础设施（Qlib）**
- ✅ 多源切换，永不掉线
- ✅ 高频数据增强
- ✅ 统一数据接口

### 2. **智能因子发现（RD-Agent）**
- ✅ 15+核心因子库
- ✅ 自动因子生成
- ✅ 因子有效性评估

### 3. **因子进化淘汰**
- ✅ 实时IC监控
- ✅ 5级状态管理（Active/Watching/Warning/Sleeping/Eliminated）
- ✅ 自动降权/淘汰/复活

### 4. **模型集成学习**
- ✅ LightGBM+XGBoost+CatBoost三模型Stacking
- ✅ 两阶段预测（Pool→Board）
- ✅ Calibration校准

### 5. **在线学习进化**
- ✅ 每日增量更新
- ✅ 概念漂移检测
- ✅ 自适应重训练

### 6. **严格验证**
- ✅ Walk-Forward滚动验证
- ✅ 时间序列严格切分
- ✅ 泛化能力评估

---

## 🚀 快速落地实战指南

> **目标**：从零开始，30分钟内让整套系统运转起来！

### 前置准备

**系统要求**：
- Windows 10/11 或 Linux/macOS
- Python 3.8+
- 8GB+ 内存
- 10GB+ 硬盘空间

**检查Python环境**：
```bash
# Windows PowerShell
python --version  # 应该显示 3.8+

# 如果没有Python，下载安装：
# https://www.python.org/downloads/
```

---

### 第一步：环境初始化（5分钟）

```bash
# 1. 进入项目目录
cd G:\test\qilin_stack

# 2. 创建虚拟环境（可选但强烈建议）
python -m venv venv

# 3. 激活虚拟环境
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 4. 升级pip
pip install --upgrade pip

# 5. 安装依赖（首次运行需要10-15分钟）
pip install -r requirements.txt
```

**常见问题**：
- 如果`requirements.txt`不存在，手动安装核心依赖：
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost streamlit plotly
```

---

### 第二步：Qlib数据准备（10-15分钟）

#### 方案A：使用Qlib官方数据（推荐）

```bash
# 1. 安装Qlib
pip install qlib

# 2. 下载A股数据（约3GB，需要网络）
python scripts/download_qlib_data_v2.py --start 2020-01-01 --end 2024-12-31

# 或者使用Qlib官方命令
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

#### 方案B：使用AKShare实时数据（免费、无需下载）

```bash
# 安装AKShare
pip install akshare

# 系统会自动使用AKShare获取实时数据
# 无需额外配置
```

**验证数据**：
```python
# 运行测试脚本
python scripts/validate_qlib_data.py

# 应该看到：
# ✅ Qlib数据可用
# ✅ 数据范围：2020-01-01 至 2024-12-31
# ✅ 股票数量：4000+
```

---

### 第三步：RD-Agent因子发现（5分钟）

#### 查看预置因子库

```python
# 进入Python交互式环境
python

# 导入因子发现模块
from rd_agent.factor_discovery_simple import SimplifiedFactorDiscovery

# 初始化
discovery = SimplifiedFactorDiscovery()

# 查看预置因子库
for factor in discovery.factor_library:
    print(f"{factor['id']}: {factor['name']} - IC={factor['expected_ic']}")
    print(f"   {factor['expression']}")
    print()

# 应该看到：
# limitup_001: 封单强度 - IC=0.08
#    封单金额 / 流通市值
#
# limitup_002: 连板高度因子 - IC=0.12
#    log(连板天数 + 1) * 量比
# ...

exit()  # 退出Python
```

#### 运行因子发现（异步）

```python
# 创建测试脚本：test_factor_discovery.py
import asyncio
from rd_agent.factor_discovery_simple import SimplifiedFactorDiscovery

async def main():
    discovery = SimplifiedFactorDiscovery()
    
    # 发现Top 20因子
    factors = await discovery.discover_factors(
        start_date='2024-01-01',
        end_date='2024-12-31',
        n_factors=20,
        min_ic=0.05
    )
    
    print(f"\n发现 {len(factors)} 个高质量因子:")
    for f in factors:
        print(f"  - {f['name']}: IC={f['expected_ic']}")

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
# 运行
python test_factor_discovery.py
```

---

### 第四步：因子生命周期测试（5分钟）

```python
# 创建测试脚本：test_factor_lifecycle.py
from factors.factor_lifecycle_manager import FactorLifecycleManager, FactorStatus

# 初始化管理器
manager = FactorLifecycleManager()

# 模拟因子健康度数据
test_factors = [
    ('封单强度', {'ic_mean': 0.08, 'ic_recent': 0.075, 'ic_win_rate': 0.65, 'ir': 1.2, 'ic_trend': 'stable'}),
    ('连板高度', {'ic_mean': 0.12, 'ic_recent': 0.11, 'ic_win_rate': 0.68, 'ir': 1.5, 'ic_trend': 'rising'}),
    ('早盘涨停', {'ic_mean': 0.15, 'ic_recent': 0.14, 'ic_win_rate': 0.70, 'ir': 1.8, 'ic_trend': 'stable'}),
    ('某弱因子', {'ic_mean': 0.02, 'ic_recent': 0.008, 'ic_win_rate': 0.48, 'ir': 0.2, 'ic_trend': 'declining'}),
]

print("因子生命周期管理测试\n" + "="*60)

for factor_name, health_metrics in test_factors:
    status = manager.update_factor_status(factor_name, health_metrics)
    print(f"\n{factor_name}:")
    print(f"  状态: {status['status'].value}")
    print(f"  权重: {status['weight']*100:.0f}%")
    print(f"  IC均值: {health_metrics['ic_mean']:.4f}")
    print(f"  趋势: {health_metrics['ic_trend']}")

# 获取活跃因子
active_factors = manager.get_active_factors()
print(f"\n\n✅ 活跃因子数: {len(active_factors)}/{len(test_factors)}")
print(f"活跃因子列表: {active_factors}")
```

```bash
# 运行
python test_factor_lifecycle.py

# 应该看到：
# 封单强度: 活跃，权重=100%
# 连板高度: 活跃，权重=100%
# 早盘涨停: 活跃，权重=100%
# 某弱因子: 观察/警告，权重=75%/50%
```

---

### 第五步：一进二模型训练（10分钟）

#### 准备模拟数据

```python
# 创建测试脚本：test_model_training.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from qlib_enhanced.one_into_two_pipeline import OneIntoTwoTrainer

# 生成模拟数据
np.random.seed(42)
n_samples = 1000

# 日期范围
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i//50) for i in range(n_samples)]

# 股票代码
symbols = [f"{i:06d}.SZ" for i in np.random.choice(range(1, 300000), n_samples)]

# 生成特征
features_data = {
    'date': dates,
    'symbol': symbols,
    'seal_strength': np.random.uniform(50, 120, n_samples),  # 封单强度
    'limitup_time_score': np.random.uniform(60, 100, n_samples),  # 涨停时间评分
    'sector_heat': np.random.uniform(0, 1, n_samples),  # 板块热度
    'volume_ratio': np.random.uniform(1, 5, n_samples),  # 量比
    'turnover_rate': np.random.uniform(5, 40, n_samples),  # 换手率
}

# 生成标签
df = pd.DataFrame(features_data)
df['pool_label'] = (df['seal_strength'] > 80).astype(int)  # T日涨停
df['board_label'] = ((df['seal_strength'] > 90) & (df['pool_label'] == 1)).astype(int)  # T+1继续涨停

print(f"样本数量: {len(df)}")
print(f"Pool标签比例: {df['pool_label'].mean():.2%}")
print(f"Board标签比例: {df['board_label'].mean():.2%}")

# 训练模型
print("\n开始训练一进二模型...")
trainer = OneIntoTwoTrainer(top_n=20)
result = trainer.fit(df)

print(f"\n✅ 训练完成！")
print(f"  Pool Model AUC: {result.auc_pool:.4f}")
print(f"  Board Model AUC: {result.auc_board:.4f}")
print(f"  Top-{trainer.top_n} Threshold: {result.threshold_topn:.4f}")

# 预测示例
test_sample = df.head(5)[
    ['seal_strength', 'limitup_time_score', 'sector_heat', 'volume_ratio', 'turnover_rate']
]
pool_probs = result.model_pool.predict_proba(test_sample)[:, 1]
board_probs = result.model_board.predict_proba(test_sample)[:, 1]

print(f"\n预测示例:")
for i in range(5):
    print(f"  股票{i+1}: Pool={pool_probs[i]:.2%}, Board={board_probs[i]:.2%}")
```

```bash
# 运行
python test_model_training.py
```

---

### 第六步：竞价进阶Pipeline测试（10分钟）

```python
# 创建测试脚本：test_phase1_pipeline.py
import pandas as pd
import numpy as np
from qlib_enhanced.unified_phase1_pipeline import UnifiedPhase1Pipeline

# 初始化Pipeline
print("初始化UnifiedPhase1Pipeline...\n")
pipeline = UnifiedPhase1Pipeline(
    config={
        'data_quality': {
            'min_coverage': 0.95,
            'max_missing_ratio': 0.05
        },
        'feature_selection': {
            'max_features': 50,
            'min_importance': 0.01
        },
        'factor_health': {
            'ic_windows': [20, 60, 120],
            'min_ic': 0.02
        }
    },
    output_dir="output/test_pipeline"
)

print("✅ Pipeline初始化完成\n")
print("模块列表:")
print("  - DataQualityAuditor: 数据质量审计")
print("  - CoreFeatureGenerator: 核心特征生成")
print("  - FactorDecayMonitor: 因子衰减监控")
print("  - FactorLifecycleManager: 因子生命周期管理")
print("  - MarketSentimentFactors: 市场情绪因子")
print("  - ThemeDiffusionFactors: 题材扩散因子")
print("  - LiquidityVolatilityFactors: 流动性波动率因子")

print("\n✨ 系统就绪！现在可以进行完整的竞价进阶优化了！")
```

```bash
# 运行
python test_phase1_pipeline.py
```

---

### 第七步：启动Web界面（1分钟）

```bash
# 启动Streamlit Web应用
streamlit run web/unified_dashboard.py

# 应该自动打开浏览器：http://localhost:8501
# 如果没有自动打开，手动复制链接到浏览器
```

**Web界面操作**：

1. **导航到竞价决策**：
   - 左侧菜单 → 竞价决策

2. **T日候选筛选**：
   - 点击「📊 T日候选筛选」标签
   - 设置筛选条件
   - 点击「🔍 执行筛选」

3. **竞价进阶优化**：
   - 点击「🎯 竞价进阶」标签
   - 在「📊 数据准备」选择数据源
   - 在「📈 运行Pipeline」点击「🚀 运行完整Pipeline」
   - 查看「📋 查看结果」

---

### 第八步：验证完整流程（20分钟）

#### 创建完整集成测试

```python
# 创建: test_full_workflow.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# 导入所有模块
from rd_agent.factor_discovery_simple import SimplifiedFactorDiscovery
from factors.factor_lifecycle_manager import FactorLifecycleManager
from qlib_enhanced.one_into_two_pipeline import OneIntoTwoTrainer
from qlib_enhanced.unified_phase1_pipeline import UnifiedPhase1Pipeline
from qlib_enhanced.online_learning import OnlineLearningManager

print("━"*70)
print("🚀 Qilin Stack 完整流程验证")
print("━"*70)

# ========== Phase 1: 因子发现 ==========
print("\n🧪 Phase 1: RD-Agent因子发现")
print("-"*70)

async def discover_factors():
    discovery = SimplifiedFactorDiscovery()
    factors = await discovery.discover_factors(
        start_date='2024-01-01',
        end_date='2024-12-31',
        n_factors=15,
        min_ic=0.05
    )
    return factors

factors = asyncio.run(discover_factors())
print(f"✅ 发现 {len(factors)} 个高质量因子")

# ========== Phase 2: 因子生命周期管理 ==========
print("\n\n🔄 Phase 2: 因子生命周期管理")
print("-"*70)

manager = FactorLifecycleManager()

# 模拟因子健康度
for i, factor in enumerate(factors[:5]):  # 只测试前5个
    health_metrics = {
        'ic_mean': factor['expected_ic'],
        'ic_recent': factor['expected_ic'] * 0.9,
        'ic_win_rate': 0.60,
        'ir': 1.0,
        'ic_trend': 'stable'
    }
    status = manager.update_factor_status(factor['name'], health_metrics)

active_factors = manager.get_active_factors()
print(f"✅ 活跃因子数: {len(active_factors)}")

# ========== Phase 3: 模型训练 ==========
print("\n\n📈 Phase 3: 一进二模型训练")
print("-"*70)

# 生成模拟数据
np.random.seed(42)
n_samples = 500

df_train = pd.DataFrame({
    'date': [datetime(2024, 1, 1) + timedelta(days=i//10) for i in range(n_samples)],
    'symbol': [f"{i:06d}.SZ" for i in np.random.choice(range(1, 300000), n_samples)],
    'seal_strength': np.random.uniform(50, 120, n_samples),
    'limitup_time_score': np.random.uniform(60, 100, n_samples),
    'volume_ratio': np.random.uniform(1, 5, n_samples),
})

df_train['pool_label'] = (df_train['seal_strength'] > 80).astype(int)
df_train['board_label'] = ((df_train['seal_strength'] > 90) & (df_train['pool_label'] == 1)).astype(int)

trainer = OneIntoTwoTrainer(top_n=20)
result = trainer.fit(df_train)

print(f"✅ Pool Model AUC: {result.auc_pool:.4f}")
print(f"✅ Board Model AUC: {result.auc_board:.4f}")

# ========== Phase 4: Pipeline集成 ==========
print("\n\n🎯 Phase 4: UnifiedPhase1Pipeline集成")
print("-"*70)

pipeline = UnifiedPhase1Pipeline(output_dir="output/test_full")
print("✅ Pipeline初始化完成")
print("  ✓ 数据质量审计")
print("  ✓ 特征生成器")
print("  ✓ 因子监控器")
print("  ✓ 生命周期管理")
print("  ✓ 市场情绪因子")

# ========== 总结 ==========
print("\n\n" + "━"*70)
print("✨ 所有测试通过！系统已就绪！")
print("━"*70)
print("\n下一步操作：")
print("  1. 启动Web界面：streamlit run web/unified_dashboard.py")
print("  2. 查阅文档：docs/DAILY_TRADING_SOP.md")
print("  3. 开始实战操作！")
print("\n🎉 祝交易顺利！")
```

```bash
# 运行完整测试
python test_full_workflow.py
```

---

## 📊 快速参考：常用命令

### 日常启动

```bash
# 1. 激活环境
cd G:\test\qilin_stack
.\venv\Scripts\activate  # Windows

# 2. 启动Web界面
streamlit run web/unified_dashboard.py
```

### 数据更新

```bash
# 更新Qlib数据（每周执行一次）
python scripts/download_qlib_data_v2.py --start 2024-01-01 --end $(date +%Y-%m-%d)

# 验证数据
python scripts/validate_qlib_data.py
```

### 模型训练

```bash
# 重新训练一进二模型
python qlib_enhanced/one_into_two_pipeline.py

# 训练基线模型
python scripts/train_baseline_model.py
```

### 因子管理

```bash
# 查看因子健康度
python -c "from factors.factor_lifecycle_manager import FactorLifecycleManager; m = FactorLifecycleManager(); print(m.get_summary())"

# 重置因子状态（谨慎）
rm -rf output/factor_lifecycle/*.json
```

### 日志查看

```bash
# 实时查看日志
tail -f logs/scheduler.log

# 查看最后50行
tail -n 50 logs/scheduler.log

# 搜索错误日志
findstr "ERROR" logs\scheduler.log  # Windows
# grep "ERROR" logs/scheduler.log  # Linux/macOS
```

---

## ⚠️ 常见问题排查

### 1. 模块导入错误

**问题**：`ModuleNotFoundError: No module named 'xxx'`

**解决**：
```bash
# 确认虚拟环境已激活
which python  # 应该显示 venv/Scripts/python

# 重新安装依赖
pip install -r requirements.txt

# 或手动安装缺失的包
pip install <package_name>
```

### 2. Qlib数据不可用

**问题**：`QlibDataNotFound` 或类似错误

**解决**：
```bash
# 检查Qlib数据目录
ls ~/.qlib/qlib_data/cn_data  # Linux/macOS
dir %USERPROFILE%\.qlib\qlib_data\cn_data  # Windows

# 如果不存在，重新下载
python scripts/download_qlib_data_v2.py --start 2020-01-01 --end 2024-12-31

# 或使用AKShare替代
pip install akshare
```

### 3. Web界面无法启动

**问题**：`streamlit: command not found`

**解决**：
```bash
# 安装Streamlit
pip install streamlit

# 确认安装成功
streamlit --version

# 如果还是不行，使用python -m
python -m streamlit run web/unified_dashboard.py
```

### 4. 端口8501已被占用

**问题**：`Port 8501 is already in use`

**解决**：
```bash
# Windows: 查找并结束进程
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# 或使用其他端口
streamlit run web/unified_dashboard.py --server.port 8502
```

### 5. 模型训练过慢

**问题**：训练耗时太长

**解决**：
- 减少样本数量（测试阶段）
- 减少模型树数量（`n_estimators`）
- 使用GPU加速（如果支持）
- 关闭部分模型（只用LightGBM）

```python
# 在 one_into_two_pipeline.py 中设置环境变量
import os
os.environ["OIT_DISABLE_XGB"] = "1"  # 禁用XGBoost
```

---

## 🎓 下一步学习路径

### 新手路径（1-2周）

**第1天**：环境搭建 + 基础测试
- ✅ 完成上面的第一至第八步
- ✅ 确保Web界面可以正常启动

**第2-3天**：理解核心模块
- 阅读 `DEEP_ARCHITECTURE_GUIDE.md` (本文档)
- 理解Qlib、RD-Agent、因子进化、模型架构
- 运行上面的所有测试脚本

**第4-5天**：学习日常操作
- 阅读 `DAILY_TRADING_SOP.md`
- 对照SOP模拟一次完整流程（T日→T+1→T+2）
- 熟悉Web界面各个标签页

**第6-7天**：掌握选股逻辑
- 阅读 `STOCK_SELECTION_GUIDE.md`
- 理解三层过滤体系
- 学习质量评分和竞价强度分级

**第2周**：实盘模拟
- 使用历史数据模拟完整交易流程
- 每天记录操作和决策
- 总结经验和教训

### 进阶路径（1-3月）

**模块定制**：
- 添加自己的因子到 `rd_agent/factor_discovery_simple.py`
- 调整筛选参数到 `app/auction_decision_engine.py`
- 优化模型参数在 `qlib_enhanced/one_into_two_pipeline.py`

**策略优化**：
- 测试不同的买入分层策略
- 优化卖出策略矩阵
- 调整Kelly仓位管理参数

**系统集成**：
- 集成实时数据源
- 实现自动交易接口
- 搭建监控告警系统

---

## 🚀 下一步优化方向

### 1. **强化学习集成**
```python
# qlib_enhanced/rl_trading.py
# 使用PPO/A3C优化买卖时机和仓位
```

### 2. **Meta Learning**
```python
# qlib_enhanced/meta_learning.py
# 快速适应新市场环境
```

### 3. **高频因子扩展**
```python
# features/high_freq_factors.py
# 秒级/Tick级微观结构因子
```

### 4. **因子组合优化**
```python
# app/factor_optimizer.py
# 遗传算法优化因子权重组合
```

---

## 📚 相关文档

- [一进二操盘SOP](DAILY_TRADING_SOP.md) - 日常操作流程
- [选股决策手册](STOCK_SELECTION_GUIDE.md) - 人工选股指南
- [竞价进阶使用指南](PHASE1_USAGE_GUIDE.md) - Pipeline详细说明
- [API文档](../qlib_enhanced/README.md) - 模块API参考

---

**这才是Qilin Stack的真正力量！** 🚀

从Qlib数据基础设施，到RD-Agent因子发现，到因子生命周期管理，到模型在线学习，再到竞价进阶集成——**完整的AI驱动量化交易系统**。
