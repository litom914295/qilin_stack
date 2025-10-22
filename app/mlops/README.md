# MLOps模块

麒麟量化系统的MLOps（Machine Learning Operations）模块，提供完整的模型生命周期管理功能。

## 功能特性

### 1. 模型注册与版本管理
- ✅ 模型注册表 (Model Registry)
- ✅ 模型版本控制
- ✅ 模型阶段管理 (Staging, Production, Archived)
- ✅ 模型元数据追踪

### 2. 实验追踪
- ✅ 实验管理
- ✅ 参数记录
- ✅ 指标追踪
- ✅ Artifacts管理

### 3. A/B测试框架
- ✅ 多变体测试
- ✅ 流量分配
- ✅ 统计分析
- ✅ 自动选择最优模型

### 4. 在线学习管道
- ✅ 持续训练
- ✅ 自动模型更新
- ✅ 性能监控
- ✅ 数据缓冲区管理

## 快速开始

### 启动MLflow服务

```bash
# 使用Docker Compose
docker-compose up -d mlflow

# 或直接启动
mlflow server --host 0.0.0.0 --port 5000
```

访问MLflow UI: http://localhost:5000

### 基础使用

#### 1. 模型注册

```python
from app.mlops import ModelRegistry
from sklearn.ensemble import RandomForestClassifier

# 初始化注册表
registry = ModelRegistry(tracking_uri="http://localhost:5000")

# 训练模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 注册模型
run_id = registry.register_model(
    model=model,
    model_name="trading_model",
    tags={'version': 'v1.0', 'algorithm': 'rf'},
    input_example=X_train.iloc[:5]
)

# 提升到生产环境
registry.promote_model("trading_model", version=1, stage="Production")

# 加载生产模型
prod_model = registry.get_model("trading_model", stage="Production")
```

#### 2. 实验追踪

```python
from app.mlops import ExperimentTracker

# 初始化追踪器
tracker = ExperimentTracker(tracking_uri="http://localhost:5000")

# 开始实验
tracker.start_experiment(
    experiment_name="hyperparameter_tuning",
    run_name="rf_test_001",
    tags={'optimizer': 'bayesian'}
)

# 记录参数
tracker.log_params({
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.01
})

# 记录指标
tracker.log_metrics({
    'train_accuracy': 0.95,
    'test_accuracy': 0.92,
    'sharpe_ratio': 1.5
})

# 结束实验
tracker.end_experiment()
```

#### 3. A/B测试

```python
from app.mlops import ABTestingFramework, ModelRegistry

# 初始化
registry = ModelRegistry(tracking_uri="http://localhost:5000")
ab_framework = ABTestingFramework(registry)

# 创建测试
test = ab_framework.create_test(
    test_id="model_comparison_001",
    name="Model A vs Model B",
    variants=[
        {
            'name': 'variant_a',
            'model_name': 'model_v1',
            'model_version': 1,
            'traffic_weight': 0.5
        },
        {
            'name': 'variant_b',
            'model_name': 'model_v2',
            'model_version': 2,
            'traffic_weight': 0.5
        }
    ],
    min_sample_size=1000
)

# 启动测试
ab_framework.start_test("model_comparison_001")

# 路由请求并记录结果
variant = ab_framework.route_request()
prediction = model.predict(data)
ab_framework.record_prediction(
    test_id="model_comparison_001",
    variant_name=variant.name,
    prediction=prediction,
    actual=actual_value
)

# 分析结果
analysis = ab_framework.analyze_results("model_comparison_001")
print(f"Winner: {analysis['winner']}")

# 完成测试
ab_framework.complete_test("model_comparison_001")
```

#### 4. 在线学习

```python
from app.mlops import (
    OnlineLearningPipeline,
    ModelRegistry,
    ExperimentTracker,
    ModelEvaluator
)

# 初始化组件
registry = ModelRegistry(tracking_uri="http://localhost:5000")
tracker = ExperimentTracker(tracking_uri="http://localhost:5000")
evaluator = ModelEvaluator()

# 创建管道
pipeline = OnlineLearningPipeline(
    model_registry=registry,
    experiment_tracker=tracker,
    model_evaluator=evaluator,
    model_name="trading_model",
    buffer_size=10000,
    update_interval=3600,  # 1小时
    min_samples_for_update=1000
)

# 启动管道
pipeline.start(initial_model=model, initial_version=1)

# 添加新样本（实时）
pipeline.add_sample(X_new, y_new)

# 批量添加
pipeline.add_batch(X_batch, y_batch)

# 查看缓冲区状态
stats = pipeline.get_buffer_stats()
print(f"Buffer: {stats['size']}/{stats['capacity']}")

# 手动触发更新
update = pipeline.trigger_update(force=True)

# 查看更新历史
history = pipeline.get_update_history()
print(history)

# 停止管道
pipeline.stop()
```

## 运行示例

```bash
# 启动MLflow服务
docker-compose up -d mlflow

# 运行演示程序
python examples/mlops_demo.py
```

## 架构设计

```
┌─────────────────────────────────────────┐
│         MLOps Platform                  │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │   Model      │  │  Experiment  │   │
│  │   Registry   │  │   Tracker    │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │  A/B Testing │  │   Online     │   │
│  │  Framework   │  │   Learning   │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
└─────────────────────────────────────────┘
            │
            ├─────────────────┐
            │                 │
        ┌───▼───┐      ┌──────▼──────┐
        │MLflow │      │   Trading   │
        │Server │      │   System    │
        └───────┘      └─────────────┘
```

## 配置

### 环境变量

```bash
# MLflow Tracking URI
MLFLOW_TRACKING_URI=http://localhost:5000

# 模型存储路径
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

# 后端存储URI
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
```

### Docker Compose配置

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.9.2
  ports:
    - "5000:5000"
  volumes:
    - mlflow_data:/mlflow
  command: >
    mlflow server
    --host 0.0.0.0
    --port 5000
    --backend-store-uri sqlite:///mlflow/mlflow.db
    --default-artifact-root /mlflow/artifacts
```

## 最佳实践

### 1. 模型命名规范

```python
# 使用语义化命名
model_name = "trading_model_rf_v1"
model_name = "risk_predictor_lgbm_v2"

# 使用标签区分
tags = {
    'model_type': 'classifier',
    'algorithm': 'random_forest',
    'market': 'cn_stock',
    'version': 'v1.0'
}
```

### 2. 实验组织

```python
# 按功能组织实验
experiment_name = "hyperparameter_tuning"
experiment_name = "feature_engineering"
experiment_name = "model_comparison"

# 使用有意义的运行名称
run_name = f"rf_n{n_estimators}_d{max_depth}_{timestamp}"
```

### 3. 模型版本管理

```python
# 开发阶段: None (默认)
# 测试阶段: Staging
registry.promote_model(model_name, version, stage="Staging")

# 生产环境: Production
registry.promote_model(model_name, version, stage="Production")

# 废弃版本: Archived
registry.promote_model(model_name, old_version, stage="Archived")
```

### 4. A/B测试策略

```python
# 保守策略: 90/10分流
variants = [
    {'name': 'control', 'traffic_weight': 0.9},
    {'name': 'treatment', 'traffic_weight': 0.1}
]

# 均衡策略: 50/50分流
variants = [
    {'name': 'model_a', 'traffic_weight': 0.5},
    {'name': 'model_b', 'traffic_weight': 0.5}
]

# 设置足够的样本量
min_sample_size = 1000  # 确保统计显著性
```

### 5. 在线学习配置

```python
# 根据数据量调整缓冲区大小
buffer_size = 10000  # 适中的缓冲区

# 设置合理的更新间隔
update_interval = 3600  # 1小时（根据业务需求）

# 设置最小更新样本数
min_samples_for_update = 1000  # 确保训练质量

# 设置性能阈值
performance_threshold = 0.05  # 5%性能下降触发更新
```

## 监控指标

### 关键指标

1. **模型性能指标**
   - 准确率 (Accuracy)
   - 精确率 (Precision)
   - 召回率 (Recall)
   - F1分数
   - 夏普比率 (Sharpe Ratio)
   - 最大回撤 (Max Drawdown)

2. **系统指标**
   - 模型加载时间
   - 预测延迟
   - 内存使用
   - 更新频率

3. **A/B测试指标**
   - 流量分配
   - 样本数量
   - 统计显著性
   - 胜出概率

4. **在线学习指标**
   - 缓冲区利用率
   - 更新触发次数
   - 模型改进幅度
   - 训练时间

## 故障排查

### 常见问题

1. **MLflow连接失败**
```bash
# 检查MLflow服务状态
docker-compose ps mlflow

# 查看日志
docker-compose logs mlflow

# 重启服务
docker-compose restart mlflow
```

2. **模型注册失败**
```python
# 检查模型格式
# 确保使用sklearn格式或支持的框架

# 检查输入示例
# input_example应该是DataFrame格式
```

3. **在线学习不更新**
```python
# 检查缓冲区状态
stats = pipeline.get_buffer_stats()

# 检查更新条件
should_update, reason = pipeline._should_trigger_update()

# 强制更新
pipeline.trigger_update(force=True)
```

## 性能优化

1. **模型序列化**
   - 使用MLflow原生格式
   - 避免大型artifacts
   - 定期清理旧版本

2. **实验追踪**
   - 批量记录指标
   - 使用异步日志
   - 限制artifact大小

3. **在线学习**
   - 调整缓冲区大小
   - 使用增量学习算法
   - 异步训练更新

## 相关文档

- [部署指南](../../docs/DEPLOYMENT_GUIDE.md)
- [技术架构](../../docs/Technical_Architecture_v2.1_Final.md)
- [改进进度](../../docs/IMPROVEMENT_PROGRESS.md)

## 支持

如有问题，请查看：
1. MLflow官方文档: https://mlflow.org/docs/latest/index.html
2. 示例代码: `examples/mlops_demo.py`
3. 单元测试: `tests/unit/test_mlops.py`
