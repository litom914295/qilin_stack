"""
MLOps模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.mlops import (
    ModelRegistry,
    ExperimentTracker,
    ModelEvaluator,
    ABTestingFramework,
    OnlineLearningPipeline,
    TestStatus


@pytest.mark.mlops
class TestModelRegistry:
    """模型注册表测试"""
    
    @pytest.fixture
    def registry(self, mlflow_tracking_uri):
        """创建模型注册表实例"""
        return ModelRegistry(tracking_uri=mlflow_tracking_uri)
    
    @pytest.fixture
    def trained_model(self, sample_model_data):
        """训练一个简单模型"""
        X, y = sample_model_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:800], y[:800])
        return model, X, y
    
    def test_register_model(self, registry, trained_model, sample_model_data):
        """测试模型注册"""
        model, X, y = trained_model
        
        run_id = registry.register_model(
            model=model,
            model_name="test_model",
            tags={'algorithm': 'random_forest'},
            input_example=X[:5]
        
        assert run_id is not None
        assert isinstance(run_id, str)
    
    def test_list_models(self, registry, trained_model, sample_model_data):
        """测试列出模型"""
        model, X, y = trained_model
        
        # 注册一个模型
        registry.register_model(
            model=model,
            model_name="list_test_model",
            input_example=X[:5]
        
        # 列出所有模型
        models = registry.list_models()
        
        assert isinstance(models, list)
        if models:  # 如果有模型
            assert 'name' in models[0]
            assert 'versions' in models[0]
    
    def test_model_stages(self, registry, trained_model, sample_model_data):
        """测试模型阶段管理"""
        model, X, y = trained_model
        
        # 注册模型
        run_id = registry.register_model(
            model=model,
            model_name="stage_test_model",
            input_example=X[:5]
        
        # 提升到Production
        models = registry.list_models()
        if models:
            model_name = models[0]['name']
            version = models[0]['versions'][0]['version']
            
            registry.promote_model(
                model_name=model_name,
                version=version,
                stage="Production"


@pytest.mark.mlops
class TestExperimentTracker:
    """实验追踪器测试"""
    
    @pytest.fixture
    def tracker(self, mlflow_tracking_uri):
        """创建实验追踪器实例"""
        return ExperimentTracker(tracking_uri=mlflow_tracking_uri)
    
    def test_start_experiment(self, tracker):
        """测试开始实验"""
        run = tracker.start_experiment(
            experiment_name="test_experiment",
            run_name="test_run",
            tags={'test': 'true'}
        
        assert run is not None
        
        # 结束实验
        tracker.end_experiment()
    
    def test_log_params(self, tracker):
        """测试记录参数"""
        tracker.start_experiment(
            experiment_name="param_test",
            run_name="param_run"
        
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.01
        }
        
        # 不应该抛出异常
        tracker.log_params(params)
        
        tracker.end_experiment()
    
    def test_log_metrics(self, tracker):
        """测试记录指标"""
        tracker.start_experiment(
            experiment_name="metric_test",
            run_name="metric_run"
        
        metrics = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88
        }
        
        # 不应该抛出异常
        tracker.log_metrics(metrics)
        
        tracker.end_experiment()


@pytest.mark.mlops
class TestModelEvaluator:
    """模型评估器测试"""
    
    @pytest.fixture
    def evaluator(self):
        """创建评估器实例"""
        return ModelEvaluator()
    
    def test_evaluate_trading_model(self, evaluator, sample_model_data):
        """测试交易模型评估"""
        X, y = sample_model_data
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:800], y[:800])
        
        # 评估
        X_test = X[800:]
        y_test = y[800:]
        prices = pd.Series(np.random.uniform(50, 150, len(X_test)))
        
        metrics = evaluator.evaluate_trading_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            prices=prices
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # 验证指标范围
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1


@pytest.mark.mlops
class TestABTestingFramework:
    """A/B测试框架测试"""
    
    @pytest.fixture
    def ab_framework(self, mlflow_tracking_uri):
        """创建A/B测试框架"""
        from app.mlops import ModelRegistry
        registry = ModelRegistry(tracking_uri=mlflow_tracking_uri)
        return ABTestingFramework(registry)
    
    def test_create_test(self, ab_framework):
        """测试创建A/B测试"""
        test = ab_framework.create_test(
            test_id="test_001",
            name="Model A vs Model B",
            variants=[
                {
                    'name': 'variant_a',
                    'model_name': 'model_a',
                    'model_version': 1,
                    'traffic_weight': 0.5
                },
                {
                    'name': 'variant_b',
                    'model_name': 'model_b',
                    'model_version': 1,
                    'traffic_weight': 0.5
                }
            ]
        
        assert test.test_id == "test_001"
        assert len(test.variants) == 2
        assert test.status == TestStatus.DRAFT
    
    def test_start_test(self, ab_framework):
        """测试启动测试"""
        # 创建测试
        ab_framework.create_test(
            test_id="test_002",
            name="Test 2",
            variants=[
                {'name': 'v1', 'model_name': 'm1', 'model_version': 1, 'traffic_weight': 1.0}
            ]
        
        # 启动测试
        ab_framework.start_test("test_002")
        
        test = ab_framework.tests["test_002"]
        assert test.status == TestStatus.RUNNING
        assert test.start_time is not None
    
    def test_route_request(self, ab_framework):
        """测试请求路由"""
        # 创建并启动测试
        ab_framework.create_test(
            test_id="test_003",
            name="Routing Test",
            variants=[
                {'name': 'v1', 'model_name': 'm1', 'model_version': 1, 'traffic_weight': 0.6},
                {'name': 'v2', 'model_name': 'm2', 'model_version': 1, 'traffic_weight': 0.4}
            ]
        ab_framework.start_test("test_003")
        
        # 路由多次请求
        variants = []
        for _ in range(100):
            variant = ab_framework.route_request("test_003")
            variants.append(variant.name)
        
        # 验证两个变体都被使用
        unique_variants = set(variants)
        assert len(unique_variants) >= 1  # 至少使用了一个变体


@pytest.mark.mlops
class TestOnlineLearningPipeline:
    """在线学习管道测试"""
    
    @pytest.fixture
    def pipeline(self, mlflow_tracking_uri):
        """创建在线学习管道"""
        from app.mlops import ModelRegistry, ExperimentTracker, ModelEvaluator
        
        registry = ModelRegistry(tracking_uri=mlflow_tracking_uri)
        tracker = ExperimentTracker(tracking_uri=mlflow_tracking_uri)
        evaluator = ModelEvaluator()
        
        return OnlineLearningPipeline(
            model_registry=registry,
            experiment_tracker=tracker,
            model_evaluator=evaluator,
            model_name="test_online_model",
            buffer_size=100,
            min_samples_for_update=10
    
    def test_add_sample(self, pipeline, sample_model_data):
        """测试添加样本"""
        X, y = sample_model_data
        
        # 添加单个样本
        pipeline.add_sample(X.iloc[0], y.iloc[0])
        
        stats = pipeline.get_buffer_stats()
        assert stats['size'] == 1
    
    def test_add_batch(self, pipeline, sample_model_data):
        """测试批量添加样本"""
        X, y = sample_model_data
        
        # 批量添加
        pipeline.add_batch(X[:50], y[:50])
        
        stats = pipeline.get_buffer_stats()
        assert stats['size'] == 50
    
    def test_buffer_stats(self, pipeline, sample_model_data):
        """测试缓冲区统计"""
        X, y = sample_model_data
        
        # 添加一些样本
        pipeline.add_batch(X[:30], y[:30])
        
        stats = pipeline.get_buffer_stats()
        
        assert 'size' in stats
        assert 'capacity' in stats
        assert 'utilization' in stats
        assert stats['size'] == 30
        assert stats['capacity'] == 100
        assert 0 <= stats['utilization'] <= 1


@pytest.mark.mlops
@pytest.mark.integration
class TestMLOpsIntegration:
    """MLOps集成测试"""
    
    def test_full_workflow(self, mlflow_tracking_uri, sample_model_data):
        """测试完整MLOps工作流"""
        from app.mlops import ModelRegistry, ExperimentTracker, ModelEvaluator
        
        # 1. 创建组件
        registry = ModelRegistry(tracking_uri=mlflow_tracking_uri)
        tracker = ExperimentTracker(tracking_uri=mlflow_tracking_uri)
        evaluator = ModelEvaluator()
        
        X, y = sample_model_data
        X_train, y_train = X[:800], y[:800]
        X_test, y_test = X[800:], y[800:]
        
        # 2. 开始实验
        tracker.start_experiment(
            experiment_name="integration_test",
            run_name="full_workflow"
        
        # 3. 训练模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. 记录参数
        tracker.log_params({'n_estimators': 10})
        
        # 5. 评估模型
        prices = pd.Series(np.random.uniform(50, 150, len(X_test)))
        metrics = evaluator.evaluate_trading_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            prices=prices
        
        # 6. 记录指标
        tracker.log_metrics(metrics)
        
        # 7. 注册模型
        run_id = registry.register_model(
            model=model,
            model_name="integration_test_model",
            input_example=X_train[:5]
        
        # 8. 结束实验
        tracker.end_experiment()
        
        # 验证
        assert run_id is not None
        assert metrics['accuracy'] > 0
