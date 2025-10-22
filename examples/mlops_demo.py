"""
MLOps功能演示
演示模型注册、实验追踪、A/B测试和在线学习功能
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.mlops import (
    ModelRegistry,
    ExperimentTracker,
    ModelEvaluator,
    ABTestingFramework,
    OnlineLearningPipeline

)

def demo_model_registry():
    """演示模型注册功能"""
    print("\n" + "="*60)
    print("演示1: 模型注册与版本管理")
    print("="*60)
    
    # 初始化模型注册表
    registry = ModelRegistry(tracking_uri="http://localhost:5000")
    
    # 创建示例数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 转换为DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_train_s = pd.Series(y_train)
    y_test_s = pd.Series(y_test)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_df, y_train_s)
    
    # 注册模型
    run_id = registry.register_model(
        model=model,
        model_name="demo_trading_model",
        tags={
            'framework': 'sklearn',
            'algorithm': 'random_forest',
            'purpose': 'demo'
        },
        input_example=X_train_df.iloc[:5]
    
    print(f"✓ 模型已注册，run_id: {run_id}")
    
    # 列出所有模型
    models = registry.list_models()
    print(f"✓ 已注册模型数量: {len(models)}")
    
    for model_info in models:
        print(f"  - {model_info['name']}: {len(model_info['versions'])} 个版本")
    
    # 提升到Production
    if models:
        model_name = models[0]['name']
        version = models[0]['versions'][0]['version']
        registry.promote_model(model_name, version, stage="Production")
        print(f"✓ 模型 {model_name} v{version} 已提升到 Production")


def demo_experiment_tracking():
    """演示实验追踪功能"""
    print("\n" + "="*60)
    print("演示2: 实验追踪")
    print("="*60)
    
    # 初始化实验追踪器
    tracker = ExperimentTracker(tracking_uri="http://localhost:5000")
    
    # 创建数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 开始实验
    tracker.start_experiment(
        experiment_name="hyperparameter_tuning",
        run_name="rf_n_estimators_test",
        tags={'optimizer': 'grid_search'}
    
    # 测试不同参数
    for n_estimators in [50, 100, 150]:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        # 记录参数和指标
        tracker.log_params({'n_estimators': n_estimators})
        tracker.log_metrics({'test_accuracy': accuracy})
        
        print(f"✓ n_estimators={n_estimators}, accuracy={accuracy:.4f}")
    
    # 结束实验
    tracker.end_experiment()
    print("✓ 实验已完成")


def demo_ab_testing():
    """演示A/B测试功能"""
    print("\n" + "="*60)
    print("演示3: A/B测试")
    print("="*60)
    
    # 初始化
    registry = ModelRegistry(tracking_uri="http://localhost:5000")
    ab_framework = ABTestingFramework(registry)
    
    # 创建测试数据
    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y)
    
    # 训练两个不同的模型
    model_a = RandomForestClassifier(n_estimators=100, random_state=42)
    model_b = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    model_a.fit(X_df[:800], y_s[:800])
    model_b.fit(X_df[:800], y_s[:800])
    
    # 创建A/B测试
    test = ab_framework.create_test(
        test_id="model_comparison_001",
        name="RandomForest vs GradientBoosting",
        variants=[
            {
                'name': 'variant_a',
                'model_name': 'demo_model_a',
                'model_version': 1,
                'traffic_weight': 0.5,
                'description': 'RandomForest'
            },
            {
                'name': 'variant_b',
                'model_name': 'demo_model_b',
                'model_version': 1,
                'traffic_weight': 0.5,
                'description': 'GradientBoosting'
            }
        ],
        min_sample_size=100
    
    print(f"✓ 已创建A/B测试: {test.test_id}")
    
    # 启动测试
    ab_framework.start_test("model_comparison_001")
    print("✓ 测试已启动")
    
    # 模拟流量分配和预测
    test_data = X_df[800:1200]
    test_labels = y_s[800:1200]
    
    for i in range(len(test_data)):
        # 路由请求
        variant = ab_framework.route_request("model_comparison_001")
        
        # 获取预测
        if variant.name == 'variant_a':
            prediction = model_a.predict([test_data.iloc[i]])[0]
        else:
            prediction = model_b.predict([test_data.iloc[i]])[0]
        
        # 记录结果
        ab_framework.record_prediction(
            test_id="model_comparison_001",
            variant_name=variant.name,
            prediction=prediction,
            actual=test_labels.iloc[i]
    
    print(f"✓ 已处理 {len(test_data)} 个请求")
    
    # 分析结果
    analysis = ab_framework.analyze_results("model_comparison_001")
    
    print("\n测试结果:")
    for variant_stat in analysis['variants']:
        print(f"  {variant_stat['variant']}:")
        print(f"    - 请求数: {variant_stat['requests']}")
        print(f"    - 准确率: {variant_stat['accuracy']:.4f}")
    
    if analysis['winner']:
        print(f"\n✓ 获胜者: {analysis['winner']}")
    
    # 完成测试
    ab_framework.complete_test("model_comparison_001")
    print("✓ 测试已完成")


def demo_online_learning():
    """演示在线学习功能"""
    print("\n" + "="*60)
    print("演示4: 在线学习管道")
    print("="*60)
    
    # 初始化组件
    registry = ModelRegistry(tracking_uri="http://localhost:5000")
    tracker = ExperimentTracker(tracking_uri="http://localhost:5000")
    evaluator = ModelEvaluator()
    
    # 创建在线学习管道
    pipeline = OnlineLearningPipeline(
        model_registry=registry,
        experiment_tracker=tracker,
        model_evaluator=evaluator,
        model_name="online_trading_model",
        buffer_size=1000,
        update_interval=300,  # 5分钟
        min_samples_for_update=100
    
    # 创建初始模型和数据
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y)
    
    initial_model = RandomForestClassifier(n_estimators=50, random_state=42)
    initial_model.fit(X_df[:300], y_s[:300])
    
    # 启动管道
    pipeline.start(initial_model=initial_model, initial_version=1)
    print("✓ 在线学习管道已启动")
    
    # 添加新样本
    for i in range(300, 450):
        pipeline.add_sample(X_df.iloc[i], y_s.iloc[i])
    
    print(f"✓ 已添加 150 个新样本到缓冲区")
    
    # 获取缓冲区统计
    stats = pipeline.get_buffer_stats()
    print(f"  - 缓冲区大小: {stats['size']}/{stats['capacity']}")
    print(f"  - 利用率: {stats['utilization']:.2%}")
    
    # 手动触发更新
    print("\n触发模型更新...")
    update = pipeline.trigger_update(force=True, promote_to_production=False)
    
    if update:
        print(f"✓ 模型已更新到 v{update.model_version}")
        print(f"  - 触发原因: {update.trigger}")
        print(f"  - 训练样本数: {update.samples_count}")
        print(f"  - 更新前准确率: {update.metrics_before.get('accuracy', 'N/A')}")
        print(f"  - 更新后准确率: {update.metrics_after.get('accuracy'):.4f}")
    
    # 查看更新历史
    history = pipeline.get_update_history()
    
    if not history.empty:
        print("\n更新历史:")
        print(history[['model_version', 'trigger', 'samples_count', 'accuracy_after']])
    
    # 停止管道
    pipeline.stop()
    print("\n✓ 在线学习管道已停止")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("MLOps功能演示")
    print("="*60)
    print("\n请确保MLflow服务正在运行:")
    print("  docker-compose up -d mlflow")
    print("  或")
    print("  mlflow server --host 0.0.0.0 --port 5000")
    print("\n按Enter继续...")
    input()
    
    try:
        # 运行各个演示
        demo_model_registry()
        demo_experiment_tracking()
        demo_ab_testing()
        demo_online_learning()
        
        print("\n" + "="*60)
        print("所有演示已完成!")
        print("="*60)
        print("\n访问 http://localhost:5000 查看MLflow UI")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
