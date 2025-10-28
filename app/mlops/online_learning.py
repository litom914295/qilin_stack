"""
在线学习管道 - 持续模型训练和更新
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class TrainingBatch:
    """训练批次"""
    batch_id: str
    X: pd.DataFrame
    y: pd.Series
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelUpdate:
    """模型更新记录"""
    update_id: str
    model_version: int
    trigger: str  # time, performance, data_drift
    samples_count: int
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class OnlineLearningPipeline:
    """在线学习管道"""
    
    def __init__(
        self,
        model_registry,
        experiment_tracker,
        model_evaluator,
        model_name: str = "trading_model",
        buffer_size: int = 10000,
        update_interval: int = 3600,  # 秒
        min_samples_for_update: int = 1000,
        performance_threshold: float = 0.05  # 性能下降阈值
    ):
        """
        初始化在线学习管道
        
        Args:
            model_registry: 模型注册表
            experiment_tracker: 实验追踪器
            model_evaluator: 模型评估器
            model_name: 模型名称
            buffer_size: 数据缓冲区大小
            update_interval: 更新间隔（秒）
            min_samples_for_update: 最小更新样本数
            performance_threshold: 性能下降阈值
        """
        self.model_registry = model_registry
        self.experiment_tracker = experiment_tracker
        self.model_evaluator = model_evaluator
        self.model_name = model_name
        
        # 配置
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.min_samples_for_update = min_samples_for_update
        self.performance_threshold = performance_threshold
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=buffer_size)
        
        # 当前模型
        self.current_model = None
        self.current_version = 0
        
        # 更新历史
        self.update_history: List[ModelUpdate] = []
        
        # 后台线程
        self.is_running = False
        self.update_thread = None
        
        # 锁
        self.lock = threading.Lock()
    
    def start(self, initial_model: Any = None, initial_version: int = 1):
        """
        启动在线学习管道
        
        Args:
            initial_model: 初始模型
            initial_version: 初始版本号
        """
        with self.lock:
            if self.is_running:
                logger.warning("Pipeline already running")
                return
            
            # 加载或设置初始模型
            if initial_model:
                self.current_model = initial_model
                self.current_version = initial_version
            else:
                try:
                    self.current_model = self.model_registry.get_model(
                        self.model_name,
                        stage="Production"
                    )
                    # 获取版本号
                    models = self.model_registry.list_models()
                    for m in models:
                        if m['name'] == self.model_name:
                            prod_versions = [v for v in m['versions'] if v['stage'] == 'Production']
                            if prod_versions:
                                self.current_version = int(prod_versions[0]['version'])
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise
            
            # 启动后台更新线程
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            logger.info(f"Online learning pipeline started with model v{self.current_version}")
    
    def stop(self):
        """停止在线学习管道"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            if self.update_thread:
                self.update_thread.join(timeout=10)
            
            logger.info("Online learning pipeline stopped")
    
    def add_sample(self, X: pd.Series, y: Any, metadata: Optional[Dict] = None):
        """
        添加新样本到缓冲区
        
        Args:
            X: 特征
            y: 标签
            metadata: 元数据
        """
        sample = {
            'X': X,
            'y': y,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.data_buffer.append(sample)
    
    def add_batch(self, X: pd.DataFrame, y: pd.Series, metadata: Optional[Dict] = None):
        """
        批量添加样本
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            metadata: 元数据
        """
        for i in range(len(X)):
            self.add_sample(X.iloc[i], y.iloc[i], metadata)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        with self.lock:
            buffer_list = list(self.data_buffer)
            
            if not buffer_list:
                return {
                    'size': 0,
                    'capacity': self.buffer_size,
                    'utilization': 0.0
                }
            
            timestamps = [s['timestamp'] for s in buffer_list]
            
            return {
                'size': len(buffer_list),
                'capacity': self.buffer_size,
                'utilization': len(buffer_list) / self.buffer_size,
                'oldest_sample': min(timestamps).isoformat(),
                'newest_sample': max(timestamps).isoformat(),
                'time_span_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600
            }
    
    def _should_trigger_update(self) -> tuple[bool, str]:
        """
        判断是否应该触发模型更新
        
        Returns:
            (should_update, trigger_reason)
        """
        with self.lock:
            buffer_size = len(self.data_buffer)
            
            # 检查样本数量
            if buffer_size < self.min_samples_for_update:
                return False, ""
            
            # 检查时间间隔
            if self.update_history:
                last_update = self.update_history[-1].timestamp
                time_since_update = (datetime.now() - last_update).total_seconds()
                
                if time_since_update >= self.update_interval:
                    return True, "time_based"
            else:
                # 首次更新
                return True, "initial"
            
            # 检查性能下降（如果有足够数据）
            if buffer_size >= 100:
                # 评估当前模型性能
                try:
                    buffer_df = self._buffer_to_dataframe()
                    X_test = buffer_df.drop('y', axis=1)
                    y_test = buffer_df['y']
                    
                    predictions = self.current_model.predict(X_test)
                    
                    # 计算准确率
                    accuracy = (predictions == y_test).mean()
                    
                    # 与历史性能比较
                    if self.update_history:
                        last_accuracy = self.update_history[-1].metrics_after.get('accuracy', 1.0)
                        
                        if last_accuracy - accuracy > self.performance_threshold:
                            return True, "performance_degradation"
                
                except Exception as e:
                    logger.error(f"Error checking performance: {e}")
            
            return False, ""
    
    def _buffer_to_dataframe(self) -> pd.DataFrame:
        """将缓冲区转换为DataFrame"""
        with self.lock:
            buffer_list = list(self.data_buffer)
            
            if not buffer_list:
                return pd.DataFrame()
            
            # 构建DataFrame
            X_list = [s['X'] for s in buffer_list]
            y_list = [s['y'] for s in buffer_list]
            
            df = pd.DataFrame(X_list)
            df['y'] = y_list
            
            return df
    
    def trigger_update(
        self,
        force: bool = False,
        promote_to_production: bool = True
    ) -> Optional[ModelUpdate]:
        """
        手动触发模型更新
        
        Args:
            force: 强制更新，忽略条件检查
            promote_to_production: 是否自动提升到生产环境
            
        Returns:
            ModelUpdate对象或None
        """
        should_update, trigger = self._should_trigger_update()
        
        if not should_update and not force:
            logger.info("Update conditions not met, skipping update")
            return None
        
        if force:
            trigger = "manual"
        
        logger.info(f"Triggering model update (reason: {trigger})")
        
        try:
            # 获取训练数据
            train_df = self._buffer_to_dataframe()
            
            if train_df.empty:
                logger.warning("No data available for training")
                return None
            
            X_train = train_df.drop('y', axis=1)
            y_train = train_df['y']
            
            # 评估当前模型
            metrics_before = {}
            if self.current_model:
                try:
                    predictions = self.current_model.predict(X_train)
                    accuracy = (predictions == y_train).mean()
                    metrics_before = {'accuracy': accuracy}
                except Exception as e:
                    logger.error(f"Error evaluating current model: {e}")
            
            # 训练新模型
            new_model = self._train_model(X_train, y_train, trigger)
            
            # 评估新模型
            predictions = new_model.predict(X_train)
            accuracy = (predictions == y_train).mean()
            metrics_after = {'accuracy': accuracy}
            
            # 决定是否使用新模型
            should_deploy = True
            
            if metrics_before:
                improvement = metrics_after['accuracy'] - metrics_before['accuracy']
                
                if improvement < 0:
                    logger.warning(f"New model performance worse by {-improvement:.4f}")
                    should_deploy = False
                else:
                    logger.info(f"New model performance improved by {improvement:.4f}")
            
            if should_deploy:
                # 注册新模型
                new_version = self.current_version + 1
                
                run_id = self.model_registry.register_model(
                    model=new_model,
                    model_name=self.model_name,
                    tags={
                        'trigger': trigger,
                        'online_learning': 'true',
                        'samples_count': str(len(train_df))
                    }
                )
                
                # 提升到生产环境
                if promote_to_production:
                    self.model_registry.promote_model(
                        model_name=self.model_name,
                        version=new_version,
                        stage="Production"
                    )
                
                # 更新当前模型
                with self.lock:
                    self.current_model = new_model
                    self.current_version = new_version
                
                # 记录更新
                update = ModelUpdate(
                    update_id=f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_version=new_version,
                    trigger=trigger,
                    samples_count=len(train_df),
                    metrics_before=metrics_before,
                    metrics_after=metrics_after
                )
                
                with self.lock:
                    self.update_history.append(update)
                
                logger.info(f"Model updated to v{new_version}")
                
                return update
            else:
                logger.info("New model not deployed due to insufficient improvement")
                return None
        
        except Exception as e:
            logger.error(f"Error during model update: {e}", exc_info=True)
            return None
    
    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        trigger: str
    ) -> Any:
        """
        训练新模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            trigger: 触发原因
            
        Returns:
            训练好的模型
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # 开始实验
        experiment_name = f"{self.model_name}_online_learning"
        
        self.experiment_tracker.start_experiment(
            experiment_name=experiment_name,
            run_name=f"online_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                'trigger': trigger,
                'mode': 'online_learning'
            }
        )
        
        # 记录参数
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'training_samples': len(X_train)
        }
        
        self.experiment_tracker.log_params(params)
        
        # 训练模型
        model = RandomForestClassifier(**{k: v for k, v in params.items() if k != 'training_samples'})
        model.fit(X_train, y_train)
        
        # 记录指标
        train_accuracy = model.score(X_train, y_train)
        
        self.experiment_tracker.log_metrics({
            'train_accuracy': train_accuracy
        })
        
        # 结束实验
        self.experiment_tracker.end_experiment()
        
        return model
    
    def _update_loop(self):
        """后台更新循环"""
        logger.info("Update loop started")
        
        while self.is_running:
            try:
                # 检查是否需要更新
                should_update, trigger = self._should_trigger_update()
                
                if should_update:
                    self.trigger_update()
                
                # 休眠
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}", exc_info=True)
                time.sleep(60)
        
        logger.info("Update loop stopped")
    
    def get_update_history(self) -> pd.DataFrame:
        """获取更新历史"""
        with self.lock:
            if not self.update_history:
                return pd.DataFrame()
            
            records = []
            
            for update in self.update_history:
                record = {
                    'update_id': update.update_id,
                    'model_version': update.model_version,
                    'trigger': update.trigger,
                    'samples_count': update.samples_count,
                    'accuracy_before': update.metrics_before.get('accuracy'),
                    'accuracy_after': update.metrics_after.get('accuracy'),
                    'improvement': update.metrics_after.get('accuracy', 0) - update.metrics_before.get('accuracy', 0),
                    'timestamp': update.timestamp.isoformat()
                }
                
                records.append(record)
            
            return pd.DataFrame(records)
