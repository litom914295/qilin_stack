"""
MLflow实验跟踪模块
提供模型训练/预测的自动化实验管理、参数跟踪、指标记录、模型版本管理

功能：
- 自动记录训练参数和超参数
- 记录训练和验证指标
- 记录模型性能指标和混淆矩阵
- 保存和版本管理模型
- 记录数据集信息和特征重要性
- 生成实验对比报告
"""

import os
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

# 尝试导入MLflow
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.lightgbm
    import mlflow.xgboost
    import mlflow.catboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Install with: pip install mlflow")


class MLflowTracker:
    """
    MLflow实验跟踪器
    用于记录模型训练和预测的完整实验信息
    """
    
    def __init__(
        self,
        experiment_name: str = "qilin_limitup_ai",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        初始化MLflow跟踪器
        
        Args:
            experiment_name: 实验名称
            tracking_uri: MLflow tracking server URI (默认本地)
            artifact_location: artifact存储位置
            tags: 实验级别的标签
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        
        # 设置tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # 默认使用本地文件系统
            default_uri = "./mlruns"
            mlflow.set_tracking_uri(f"file:///{os.path.abspath(default_uri)}")
        
        # 创建或获取实验
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location,
                    tags=tags or {}
                )
                self.experiment = mlflow.get_experiment(experiment_id)
            self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            warnings.warn(f"Failed to create/get experiment: {e}")
            self.experiment_id = None
        
        self.run_id = None
        self.run_name = None
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """
        开始一个新的MLflow run
        
        Args:
            run_name: run名称
            tags: run级别的标签
            nested: 是否为嵌套run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested
        )
        
        self.run_id = mlflow.active_run().info.run_id
        
        return self.run_id
    
    def end_run(self):
        """结束当前run"""
        mlflow.end_run()
        self.run_id = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        记录参数
        
        Args:
            params: 参数字典
        """
        # MLflow参数值必须是字符串
        for key, value in params.items():
            try:
                if isinstance(value, (dict, list)):
                    mlflow.log_param(key, json.dumps(value))
                else:
                    mlflow.log_param(key, value)
            except Exception as e:
                warnings.warn(f"Failed to log param {key}: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 训练步数（用于绘制曲线）
        """
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                warnings.warn(f"Failed to log metric {key}: {e}")
    
    def log_model(
        self,
        model,
        model_name: str = "model",
        framework: str = "sklearn",
        signature=None,
        input_example=None,
        registered_model_name: Optional[str] = None
    ):
        """
        记录模型
        
        Args:
            model: 模型对象
            model_name: 模型名称（用于artifact路径）
            framework: 模型框架 ("sklearn", "lightgbm", "xgboost", "catboost")
            signature: 模型签名
            input_example: 输入示例
            registered_model_name: 注册模型名称（用于模型注册表）
        """
        try:
            if framework == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif framework == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif framework == "catboost":
                mlflow.catboost.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:  # sklearn or others
                mlflow.sklearn.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
        except Exception as e:
            warnings.warn(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        记录artifact文件
        
        Args:
            local_path: 本地文件路径
            artifact_path: artifact中的路径
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            warnings.warn(f"Failed to log artifact {local_path}: {e}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """
        记录字典为JSON文件
        
        Args:
            dictionary: 字典数据
            filename: 文件名
        """
        try:
            mlflow.log_dict(dictionary, filename)
        except Exception as e:
            warnings.warn(f"Failed to log dict {filename}: {e}")
    
    def log_figure(self, figure, filename: str):
        """
        记录matplotlib图表
        
        Args:
            figure: matplotlib figure对象
            filename: 文件名
        """
        try:
            mlflow.log_figure(figure, filename)
        except Exception as e:
            warnings.warn(f"Failed to log figure {filename}: {e}")
    
    def log_training_session(
        self,
        model,
        params: Dict[str, Any],
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        model_name: str = "limitup_model",
        framework: str = "lightgbm",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        一站式记录完整训练会话
        
        Args:
            model: 训练好的模型
            params: 训练参数
            train_metrics: 训练集指标
            val_metrics: 验证集指标
            feature_importance: 特征重要性
            model_name: 模型名称
            framework: 模型框架
            tags: 标签
        
        Returns:
            run_id
        """
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.start_run(run_name=run_name, tags=tags)
        
        try:
            # 记录参数
            self.log_params(params)
            
            # 记录训练指标
            train_metrics_renamed = {f"train_{k}": v for k, v in train_metrics.items()}
            self.log_metrics(train_metrics_renamed)
            
            # 记录验证指标
            if val_metrics:
                val_metrics_renamed = {f"val_{k}": v for k, v in val_metrics.items()}
                self.log_metrics(val_metrics_renamed)
            
            # 记录特征重要性
            if feature_importance is not None:
                feature_imp_dict = feature_importance.to_dict('records')
                self.log_dict(feature_imp_dict, "feature_importance.json")
            
            # 记录模型
            self.log_model(
                model,
                model_name=model_name,
                framework=framework
            )
            
            mlflow.set_tag("status", "success")
            
        except Exception as e:
            warnings.warn(f"Error during logging: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
        
        finally:
            self.end_run()
        
        return self.run_id
    
    def log_prediction_session(
        self,
        predictions: pd.DataFrame,
        metrics: Dict[str, float],
        model_version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        记录预测会话
        
        Args:
            predictions: 预测结果DataFrame
            metrics: 预测指标
            model_version: 使用的模型版本
            tags: 标签
        
        Returns:
            run_id
        """
        run_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.start_run(run_name=run_name, tags=tags)
        
        try:
            # 记录模型版本
            if model_version:
                mlflow.log_param("model_version", model_version)
            
            # 记录预测统计
            mlflow.log_param("n_predictions", len(predictions))
            
            # 记录指标
            self.log_metrics(metrics)
            
            # 保存预测结果摘要
            pred_summary = {
                'n_predictions': len(predictions),
                'positive_rate': float(predictions['pred_label'].mean()) if 'pred_label' in predictions else None,
                'avg_score': float(predictions['pred_score'].mean()) if 'pred_score' in predictions else None,
                'timestamp': datetime.now().isoformat()
            }
            self.log_dict(pred_summary, "prediction_summary.json")
            
            mlflow.set_tag("status", "success")
            
        except Exception as e:
            warnings.warn(f"Error during prediction logging: {e}")
            mlflow.set_tag("status", "failed")
        
        finally:
            self.end_run()
        
        return self.run_id
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        比较多个runs的指标
        
        Args:
            run_ids: run ID列表
            metrics: 要比较的指标名称列表
        
        Returns:
            比较结果DataFrame
        """
        client = mlflow.tracking.MlflowClient()
        
        runs_data = []
        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                run_info = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000.0),
                    'status': run.info.status
                }
                
                # 添加指标
                if metrics:
                    for metric in metrics:
                        run_info[metric] = run.data.metrics.get(metric, None)
                else:
                    run_info.update(run.data.metrics)
                
                runs_data.append(run_info)
            except Exception as e:
                warnings.warn(f"Failed to get run {run_id}: {e}")
        
        return pd.DataFrame(runs_data)
    
    def get_best_run(
        self,
        metric: str = "val_auc",
        ascending: bool = False
    ) -> Optional[Dict]:
        """
        获取最佳run
        
        Args:
            metric: 排序指标
            ascending: 是否升序
        
        Returns:
            最佳run信息
        """
        client = mlflow.tracking.MlflowClient()
        
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="",
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
            )
            
            if len(runs) == 0:
                return None
            
            best_run = runs.iloc[0]
            
            return {
                'run_id': best_run['run_id'],
                'run_name': best_run.get('tags.mlflow.runName', 'N/A'),
                'metrics': {col.replace('metrics.', ''): best_run[col] 
                           for col in best_run.index if col.startswith('metrics.')},
                'params': {col.replace('params.', ''): best_run[col] 
                          for col in best_run.index if col.startswith('params.')}
            }
        except Exception as e:
            warnings.warn(f"Failed to get best run: {e}")
            return None
    
    def load_model(self, run_id: str, model_name: str = "model"):
        """
        加载指定run的模型
        
        Args:
            run_id: run ID
            model_name: 模型名称
        
        Returns:
            加载的模型对象
        """
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            model = mlflow.pyfunc.load_model(model_uri)
            return model
        except Exception as e:
            warnings.warn(f"Failed to load model from run {run_id}: {e}")
            return None


class AutoMLflowLogger:
    """
    自动MLflow日志记录器（装饰器模式）
    用于自动记录函数调用的参数和结果
    """
    
    def __init__(self, tracker: MLflowTracker):
        self.tracker = tracker
    
    def log_training(self, model_name: str = "model", framework: str = "sklearn"):
        """
        训练函数装饰器
        
        使用示例:
            @logger.log_training(model_name="xgb_model", framework="xgboost")
            def train_model(X, y, params):
                # 训练代码
                return model, metrics
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 提取参数
                params = kwargs.get('params', {})
                
                # 开始run
                self.tracker.start_run(
                    run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                try:
                    # 记录参数
                    self.tracker.log_params(params)
                    
                    # 执行训练
                    result = func(*args, **kwargs)
                    
                    # 假设返回 (model, metrics) 或 (model, train_metrics, val_metrics)
                    if isinstance(result, tuple):
                        model = result[0]
                        metrics = result[1] if len(result) > 1 else {}
                        
                        # 记录指标
                        if isinstance(metrics, dict):
                            self.tracker.log_metrics(metrics)
                        
                        # 记录模型
                        self.tracker.log_model(model, model_name, framework)
                    
                    mlflow.set_tag("status", "success")
                    
                except Exception as e:
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
                    raise
                
                finally:
                    self.tracker.end_run()
                
                return result
            
            return wrapper
        return decorator


# ==================== 工具函数 ====================

def get_mlflow_ui_url(tracking_uri: Optional[str] = None) -> str:
    """
    获取MLflow UI的URL
    
    Args:
        tracking_uri: tracking URI
    
    Returns:
        UI URL
    """
    if tracking_uri is None:
        tracking_uri = mlflow.get_tracking_uri()
    
    if tracking_uri.startswith("file:"):
        return "Run `mlflow ui` in terminal to view experiments"
    else:
        return tracking_uri


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试MLflow跟踪器"""
    
    print("=" * 60)
    print("MLflow实验跟踪模块测试")
    print("=" * 60)
    
    if not MLFLOW_AVAILABLE:
        print("MLflow库未安装，请运行: pip install mlflow")
        exit(1)
    
    # 创建tracker
    tracker = MLflowTracker(
        experiment_name="test_limitup_ai",
        tags={"project": "qilin", "strategy": "one_into_two"}
    )
    
    print(f"实验ID: {tracker.experiment_id}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"UI: {get_mlflow_ui_url()}")
    
    # 模拟训练会话
    print("\n" + "=" * 60)
    print("测试训练会话记录...")
    print("=" * 60)
    
    # 生成模拟数据和模型
    np.random.seed(42)
    from sklearn.ensemble import RandomForestClassifier
    
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 模拟指标
    train_metrics = {
        'accuracy': 0.85,
        'auc': 0.82,
        'precision': 0.80,
        'recall': 0.75
    }
    
    val_metrics = {
        'accuracy': 0.78,
        'auc': 0.76,
        'precision': 0.74,
        'recall': 0.72
    }
    
    # 模拟特征重要性
    feature_importance = pd.DataFrame({
        'feature': [f'f{i}' for i in range(10)],
        'importance': np.random.rand(10)
    }).sort_values('importance', ascending=False)
    
    # 记录训练会话
    run_id = tracker.log_training_session(
        model=model,
        params={
            'n_estimators': 10,
            'max_depth': 5,
            'random_state': 42
        },
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        feature_importance=feature_importance,
        model_name="test_rf_model",
        framework="sklearn",
        tags={"model_type": "random_forest", "test": "true"}
    )
    
    print(f"✓ 训练会话已记录，Run ID: {run_id}")
    
    # 测试预测会话
    print("\n" + "=" * 60)
    print("测试预测会话记录...")
    print("=" * 60)
    
    predictions = pd.DataFrame({
        'symbol': ['000001', '000002', '000003'],
        'pred_score': [0.85, 0.72, 0.68],
        'pred_label': [1, 1, 0]
    })
    
    pred_metrics = {
        'avg_confidence': 0.75,
        'n_positive': 2
    }
    
    pred_run_id = tracker.log_prediction_session(
        predictions=predictions,
        metrics=pred_metrics,
        model_version="v1.0",
        tags={"phase": "test"}
    )
    
    print(f"✓ 预测会话已记录，Run ID: {pred_run_id}")
    
    # 获取最佳run
    print("\n" + "=" * 60)
    print("查询最佳Run...")
    print("=" * 60)
    
    best_run = tracker.get_best_run(metric="val_auc", ascending=False)
    if best_run:
        print(f"最佳Run: {best_run['run_name']}")
        print(f"指标: {best_run['metrics']}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"查看实验: mlflow ui")
    print("=" * 60)
