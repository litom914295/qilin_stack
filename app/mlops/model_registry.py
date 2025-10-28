"""
MLOps模型注册与管理
"""

try:
    import mlflow  # type: ignore
    from mlflow.tracking import MlflowClient  # type: ignore
    from mlflow.models.signature import infer_signature  # type: ignore
except Exception:  # pragma: no cover - allow tests without mlflow installed
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    def infer_signature(*args, **kwargs):  # type: ignore
        return None
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from types import SimpleNamespace
import logging
import pickle
import uuid

logger = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表管理器"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        初始化模型注册表
        
        Args:
            tracking_uri: MLflow tracking server地址
        """
        self.tracking_uri = tracking_uri
        if mlflow is None or MlflowClient is None:
            logger.warning("mlflow is not installed; ModelRegistry will be inactive for tests.")
            self.client = None
        else:
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient(tracking_uri)
            logger.info(f"MLflow client initialized: {tracking_uri}")
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        experiment_name: str = "qilin_trading",
        tags: Optional[Dict[str, str]] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ) -> str:
        """
        注册新模型
        
        Args:
            model: 模型对象
            model_name: 模型名称
            experiment_name: 实验名称
            tags: 模型标签
            signature: 模型签名
            input_example: 输入示例
            
        Returns:
            run_id: MLflow运行ID
        """
        if mlflow is None:
            # 测试环境下无mlflow：返回本地伪run_id
            fake_id = f"local-{uuid.uuid4().hex}"
            logger.warning("mlflow not available; returning fake run_id for tests")
            return fake_id
        # 设置实验
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # 记录标签
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # 自动推断签名（如果未提供）
            if signature is None and input_example is not None:
                signature = infer_signature(input_example, model.predict(input_example))
            
            # 记录模型
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example
            )
            
            run_id = run.info.run_id
            logger.info(f"Model registered: {model_name} (run_id: {run_id})")
            
            return run_id
    
    def get_model(self, model_name: str, version: Optional[int] = None, stage: str = "Production") -> Any:
        """
        获取模型
        
        Args:
            model_name: 模型名称
            version: 模型版本号
            stage: 模型阶段 (None, Staging, Production, Archived)
            
        Returns:
            模型对象
        """
        if mlflow is None:
            raise RuntimeError("mlflow is required for ModelRegistry.get_model; please install mlflow.")
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model: {model_uri}")
        
        return model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有注册的模型"""
        models = []
        
        if self.client is None:
            return []
        for rm in self.client.search_registered_models():
            model_info = {
                'name': rm.name,
                'creation_timestamp': rm.creation_timestamp,
                'last_updated_timestamp': rm.last_updated_timestamp,
                'description': rm.description,
                'versions': []
            }
            
            # 获取版本信息
            for mv in rm.latest_versions:
                version_info = {
                    'version': mv.version,
                    'stage': mv.current_stage,
                    'creation_timestamp': mv.creation_timestamp,
                    'run_id': mv.run_id
                }
                model_info['versions'].append(version_info)
            
            models.append(model_info)
        
        return models
    
    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str = "Production",
        archive_existing: bool = True
    ):
        """
        提升模型到指定阶段
        
        Args:
            model_name: 模型名称
            version: 版本号
            stage: 目标阶段 (Staging, Production)
            archive_existing: 是否归档现有的Production模型
        """
        if self.client is None:
            raise RuntimeError("mlflow is required for ModelRegistry.promote_model; please install mlflow.")
        # 归档现有的Production模型
        if archive_existing and stage == "Production":
            for mv in self.client.get_latest_versions(model_name, stages=["Production"]):
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived"
                )
                logger.info(f"Archived model: {model_name} v{mv.version}")
        
        # 提升新版本
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"Promoted model: {model_name} v{version} to {stage}")
    
    def delete_model_version(self, model_name: str, version: int):
        """删除模型版本"""
        if self.client is None:
            raise RuntimeError("mlflow is required for ModelRegistry.delete_model_version; please install mlflow.")
        self.client.delete_model_version(
            name=model_name,
            version=version
        )
        logger.info(f"Deleted model version: {model_name} v{version}")
    
    def compare_models(
        self,
        model_name: str,
        versions: List[int],
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        比较不同版本的模型
        
        Args:
            model_name: 模型名称
            versions: 版本号列表
            metrics: 要比较的指标列表
            
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        if self.client is None:
            raise RuntimeError("mlflow is required for ModelRegistry.compare_models; please install mlflow.")
        for version in versions:
            # 获取版本信息
            mv = self.client.get_model_version(model_name, version)
            run = self.client.get_run(mv.run_id)
            
            row = {
                'version': version,
                'stage': mv.current_stage,
                'created_at': datetime.fromtimestamp(mv.creation_timestamp / 1000)
            }
            
            # 添加指标
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


class ExperimentTracker:
    """实验追踪器"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        if mlflow is None:
            logger.warning("mlflow is not installed; ExperimentTracker will be inactive for tests.")
        else:
            mlflow.set_tracking_uri(tracking_uri)
        self.current_run = None
    
    def start_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """开始新实验"""
        if mlflow is None:
            # 测试环境降级：创建本地Run占位
            run_id = f"local-{uuid.uuid4().hex}"
            self.current_run = SimpleNamespace(info=SimpleNamespace(run_id=run_id))
            logger.warning("mlflow not available; using local run stub for tests")
            return self.current_run
        mlflow.set_experiment(experiment_name)
        
        self.current_run = mlflow.start_run(run_name=run_name)
        
        # 记录基本信息
        mlflow.set_tag("start_time", datetime.now().isoformat())
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        logger.info(f"Started experiment: {experiment_name}")
        
        return self.current_run
    
    def log_params(self, params: Dict[str, Any]):
        """记录参数"""
        if mlflow is None:
            return
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标"""
        if mlflow is None:
            return
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """记录artifacts"""
        if mlflow is None:
            return
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_figure(self, figure, artifact_file: str):
        """记录matplotlib图表"""
        if mlflow is None:
            return
        mlflow.log_figure(figure, artifact_file)
    
    def end_experiment(self):
        """结束实验"""
        if self.current_run:
            if mlflow is not None:
                mlflow.set_tag("end_time", datetime.now().isoformat())
                mlflow.end_run()
            logger.info(f"Ended experiment: {self.current_run.info.run_id}")
            self.current_run = None
    
    def search_experiments(
        self,
        experiment_name: str,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        搜索实验
        
        Args:
            experiment_name: 实验名称
            filter_string: 过滤条件 (MLflow搜索语法)
            order_by: 排序字段
            max_results: 最大结果数
            
        Returns:
            实验结果DataFrame
        """
        if mlflow is None:
            return pd.DataFrame()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_name}")
            return pd.DataFrame()
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )
        
        return runs


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_trading_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        prices: pd.Series
    ) -> Dict[str, float]:
        """
        评估交易模型
        
        Args:
            model: 模型对象
            X_test: 测试特征
            y_test: 测试标签
            prices: 价格序列
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 分类指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # 交易指标（如果是二分类）
        if len(np.unique(y_pred)) == 2:
            # 计算收益
            returns = prices.pct_change()
            strategy_returns = returns * y_pred
            
            metrics.update({
                'sharpe_ratio': self._calculate_sharpe(strategy_returns),
                'max_drawdown': self._calculate_max_drawdown(strategy_returns),
                'win_rate': (strategy_returns > 0).sum() / len(strategy_returns),
                'total_return': (1 + strategy_returns).prod() - 1
            })
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
