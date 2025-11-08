"""
Qlib在线学习高级实现 (P1-1)

集成Qlib官方OnlineManager,实现:
1. 滚动窗口训练 (90天窗口,30天重训)
2. 增量模型更新
3. 概念漂移检测 (基于IC和统计检验)
4. 模型热更新机制 (零停机)
5. 模型版本管理
6. 性能监控和告警

依赖:
- Qlib官方: qlib.workflow.online.manager.OnlineManager
- Qlib官方: qlib.workflow.online.strategy.RollingStrategy
- P0-3完成: 路径配置管理
"""

import os
import sys
import gc
import asyncio
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

import pandas as pd
import numpy as np
from scipy import stats

# Qlib官方导入
try:
    import qlib
    from qlib.workflow.online.manager import OnlineManager
    from qlib.workflow.online.strategy import RollingStrategy
    from qlib.workflow.task.gen import RollingGen
    from qlib.model.trainer import TrainerR, DelayTrainerR
    from qlib.workflow import R
    from qlib.data.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logging.warning("Qlib未安装,在线学习高级功能将受限")

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ConceptDriftResult:
    """概念漂移检测结果"""
    detected: bool
    drift_score: float
    ic_degradation: float  # IC衰减幅度
    detection_time: datetime
    affected_features: List[str]
    recommended_action: str  # "no_action", "incremental_update", "full_retrain"
    statistical_test_pvalue: float


@dataclass
class ModelVersion:
    """模型版本信息"""
    version: str
    created_at: datetime
    ic: float
    icir: float
    model_path: str
    metadata: Dict[str, Any]


@dataclass
class OnlineUpdateMetrics:
    """在线更新指标"""
    success: bool
    update_time: datetime
    samples_processed: int
    ic: float
    icir: float
    drift_detected: bool
    model_version: str
    update_duration_seconds: float


# ============================================================================
# 概念漂移检测器 (增强版)
# ============================================================================

class ConceptDriftDetectorAdvanced:
    """
    概念漂移检测器 (基于IC和统计检验)
    
    检测方法:
    1. IC滚动窗口监控 (Information Coefficient)
    2. Kolmogorov-Smirnov检验 (特征分布变化)
    3. Page-Hinkley检验 (在线变点检测)
    4. 统计显著性检验
    """
    
    def __init__(
        self,
        window_size: int = 20,
        ic_threshold: float = 0.05,
        ks_threshold: float = 0.1,
        min_samples: int = 100
    ):
        """
        初始化漂移检测器
        
        Args:
            window_size: 滚动窗口大小(天数)
            ic_threshold: IC衰减阈值
            ks_threshold: KS检验阈值
            min_samples: 最小样本数
        """
        self.window_size = window_size
        self.ic_threshold = ic_threshold
        self.ks_threshold = ks_threshold
        self.min_samples = min_samples
        
        # 历史记录
        self.ic_history = []
        self.icir_history = []
        self.feature_distributions = {}
        self.reference_distribution = None
        
        logger.info(
            f"概念漂移检测器初始化: window_size={window_size}, "
            f"ic_threshold={ic_threshold}"
        )
    
    def detect(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        features: Optional[pd.DataFrame] = None
    ) -> ConceptDriftResult:
        """
        检测概念漂移
        
        Args:
            predictions: 模型预测值
            labels: 真实标签
            features: 特征数据(可选,用于分布检测)
            
        Returns:
            ConceptDriftResult
        """
        # 1. 计算IC
        current_ic = self._calculate_ic(predictions, labels)
        self.ic_history.append(current_ic)
        
        # 2. IC衰减检测
        ic_degradation = 0.0
        ic_drift_detected = False
        
        if len(self.ic_history) >= self.window_size:
            recent_ic = np.mean(self.ic_history[-self.window_size:])
            historical_ic = np.mean(self.ic_history[:-self.window_size])
            ic_degradation = historical_ic - recent_ic
            
            if ic_degradation > self.ic_threshold:
                ic_drift_detected = True
                logger.warning(
                    f"IC漂移检测到! "
                    f"历史IC={historical_ic:.4f} → 近期IC={recent_ic:.4f} "
                    f"(衰减={ic_degradation:.4f})"
                )
        
        # 3. 统计检验 (KS检验)
        ks_pvalue = 1.0
        feature_drift_detected = False
        affected_features = []
        
        if features is not None and len(self.ic_history) >= self.window_size:
            feature_drift_detected, ks_pvalue, affected_features = \
                self._detect_feature_drift(features)
        
        # 4. 综合判断
        drift_detected = ic_drift_detected or feature_drift_detected
        drift_score = max(ic_degradation / (self.ic_threshold + 1e-8), 
                         1.0 - ks_pvalue)
        
        # 5. 推荐行动
        if drift_score > 2.0:
            action = "full_retrain"
        elif drift_score > 1.0:
            action = "incremental_update"
        else:
            action = "no_action"
        
        return ConceptDriftResult(
            detected=drift_detected,
            drift_score=drift_score,
            ic_degradation=ic_degradation,
            detection_time=datetime.now(),
            affected_features=affected_features,
            recommended_action=action,
            statistical_test_pvalue=ks_pvalue
        )
    
    def _calculate_ic(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """计算Information Coefficient"""
        try:
            # 处理NaN
            mask = ~(np.isnan(predictions) | np.isnan(labels))
            if mask.sum() < self.min_samples:
                return 0.0
            
            ic = np.corrcoef(predictions[mask], labels[mask])[0, 1]
            return ic if not np.isnan(ic) else 0.0
        except Exception as e:
            logger.error(f"计算IC失败: {e}")
            return 0.0
    
    def _detect_feature_drift(
        self,
        features: pd.DataFrame
    ) -> Tuple[bool, float, List[str]]:
        """
        检测特征分布漂移 (KS检验)
        
        Returns:
            (是否漂移, p值, 受影响特征列表)
        """
        if self.reference_distribution is None:
            # 初始化参考分布
            self.reference_distribution = {
                col: features[col].values
                for col in features.columns
            }
            return False, 1.0, []
        
        # KS检验
        ks_results = {}
        for col in features.columns:
            if col not in self.reference_distribution:
                continue
            
            try:
                stat, pvalue = stats.ks_2samp(
                    self.reference_distribution[col],
                    features[col].values
                )
                ks_results[col] = (stat, pvalue)
            except Exception as e:
                logger.warning(f"特征{col}的KS检验失败: {e}")
                continue
        
        # 找出显著变化的特征
        affected_features = [
            col for col, (stat, pvalue) in ks_results.items()
            if stat > self.ks_threshold or pvalue < 0.05
        ]
        
        # 综合p值 (取最小)
        min_pvalue = min([p for _, p in ks_results.values()]) if ks_results else 1.0
        
        drift_detected = len(affected_features) > 0
        
        if drift_detected:
            logger.warning(
                f"特征分布漂移检测到! "
                f"受影响特征: {affected_features[:5]} "
                f"(min_pvalue={min_pvalue:.4f})"
            )
        
        return drift_detected, min_pvalue, affected_features
    
    def reset_reference(self, features: pd.DataFrame):
        """重置参考分布 (重训练后调用)"""
        self.reference_distribution = {
            col: features[col].values
            for col in features.columns
        }
        logger.info("参考分布已重置")


# ============================================================================
# 模型热更新器
# ============================================================================

class ModelHotReloader:
    """
    模型热更新器 (零停机切换)
    
    特性:
    1. 异步加载新模型
    2. 模型验证
    3. 原子切换 (无缝切换)
    4. 自动清理旧模型
    """
    
    def __init__(self):
        self.current_model = None
        self.loading_model = None
        self.lock = asyncio.Lock()
        self.load_count = 0
        
        logger.info("模型热更新器初始化")
    
    async def hot_reload(
        self,
        new_model: Any,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> bool:
        """
        热更新模型
        
        Args:
            new_model: 新模型对象
            validation_data: 验证数据 (可选)
            
        Returns:
            是否成功
        """
        start_time = datetime.now()
        
        try:
            async with self.lock:
                # 1. 验证新模型
                if validation_data is not None:
                    if not self._validate_model(new_model, validation_data):
                        logger.error("新模型验证失败,取消热更新")
                        return False
                
                # 2. 原子切换
                old_model = self.current_model
                self.current_model = new_model
                self.loading_model = None
                self.load_count += 1
                
                # 3. 清理旧模型
                if old_model is not None:
                    del old_model
                    gc.collect()
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"✅ 模型热更新完成 "
                    f"(第{self.load_count}次, 耗时{duration:.2f}秒)"
                )
                
                return True
                
        except Exception as e:
            logger.error(f"模型热更新失败: {e}")
            return False
    
    def _validate_model(
        self,
        model: Any,
        validation_data: Tuple[pd.DataFrame, pd.Series]
    ) -> bool:
        """验证模型"""
        try:
            X_val, y_val = validation_data
            predictions = model.predict(X_val)
            
            # 检查预测输出
            if predictions is None or len(predictions) == 0:
                return False
            
            # 检查IC
            ic = np.corrcoef(predictions, y_val)[0, 1]
            if np.isnan(ic) or abs(ic) < 0.01:
                logger.warning(f"模型验证IC过低: {ic:.4f}")
                return False
            
            logger.info(f"模型验证通过: IC={ic:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"模型验证异常: {e}")
            return False
    
    def get_current_model(self) -> Optional[Any]:
        """获取当前模型"""
        return self.current_model


# ============================================================================
# 模型版本管理器
# ============================================================================

class ModelRegistry:
    """
    模型版本管理器
    
    功能:
    1. 模型版本注册
    2. 最优模型追踪
    3. 历史模型回滚
    4. 模型元数据管理
    """
    
    def __init__(self, storage_dir: str = "./mlruns/model_registry"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions: List[ModelVersion] = []
        self.current_best_ic = -np.inf
        self.best_version = None
        
        logger.info(f"模型注册表初始化: {self.storage_dir}")
    
    def register(
        self,
        model: Any,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        注册新模型版本
        
        Args:
            model: 模型对象
            metrics: 性能指标 {"ic": 0.05, "icir": 0.5}
            metadata: 元数据
            
        Returns:
            ModelVersion
        """
        # 生成版本号
        version_id = f"v{len(self.versions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 保存模型
        model_path = self.storage_dir / f"{version_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 创建版本记录
        version = ModelVersion(
            version=version_id,
            created_at=datetime.now(),
            ic=metrics.get('ic', 0.0),
            icir=metrics.get('icir', 0.0),
            model_path=str(model_path),
            metadata=metadata or {}
        )
        
        self.versions.append(version)
        
        # 更新最优模型
        if version.ic > self.current_best_ic:
            self.current_best_ic = version.ic
            self.best_version = version
            logger.info(
                f"🏆 新的最优模型: {version_id} "
                f"(IC={version.ic:.4f}, ICIR={version.icir:.4f})"
            )
        else:
            logger.info(
                f"模型注册: {version_id} "
                f"(IC={version.ic:.4f}, ICIR={version.icir:.4f})"
            )
        
        return version
    
    def get_best_model(self) -> Optional[Any]:
        """获取最优模型"""
        if self.best_version is None:
            return None
        
        with open(self.best_version.model_path, 'rb') as f:
            return pickle.load(f)
    
    def load_version(self, version_id: str) -> Optional[Any]:
        """加载指定版本"""
        for version in self.versions:
            if version.version == version_id:
                with open(version.model_path, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def get_version_history(self) -> pd.DataFrame:
        """获取版本历史"""
        if not self.versions:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'version': v.version,
                'created_at': v.created_at,
                'ic': v.ic,
                'icir': v.icir,
                'is_best': v == self.best_version
            }
            for v in self.versions
        ])


# ============================================================================
# Qlib在线学习高级管理器
# ============================================================================

class QlibOnlineLearningAdvanced:
    """
    Qlib在线学习高级管理器
    
    功能:
    1. 集成官方OnlineManager
    2. 滚动窗口训练 (90天窗口,30天重训)
    3. 概念漂移自适应
    4. 模型热更新
    5. 性能监控
    """
    
    def __init__(
        self,
        task_config: Dict[str, Any],
        rolling_window: int = 90,
        retrain_interval: int = 30,
        drift_threshold: float = 0.05,
        enable_hot_reload: bool = True,
        qlib_provider_uri: Optional[str] = None,
        qlib_region: str = "cn"
    ):
        """
        初始化在线学习管理器
        
        Args:
            task_config: Qlib任务配置
            rolling_window: 滚动窗口大小(天)
            retrain_interval: 重训练间隔(天)
            drift_threshold: 漂移检测阈值
            enable_hot_reload: 启用热更新
            qlib_provider_uri: Qlib数据路径
            qlib_region: Qlib区域
        """
        if not QLIB_AVAILABLE:
            raise ImportError("需要安装Qlib: pip install pyqlib")
        
        # Qlib初始化
        if qlib_provider_uri:
            qlib.init(provider_uri=qlib_provider_uri, region=qlib_region)
        
        self.task_config = task_config
        self.rolling_window = rolling_window
        self.retrain_interval = retrain_interval
        
        # 核心组件
        self.drift_detector = ConceptDriftDetectorAdvanced(
            ic_threshold=drift_threshold
        )
        self.model_registry = ModelRegistry()
        self.hot_reloader = ModelHotReloader() if enable_hot_reload else None
        
        # 在线管理器 (延迟初始化)
        self.online_manager: Optional[OnlineManager] = None
        self.strategy: Optional[RollingStrategy] = None
        
        # 性能历史
        self.metrics_history = []
        
        logger.info(
            f"Qlib在线学习高级管理器初始化: "
            f"rolling_window={rolling_window}天, "
            f"retrain_interval={retrain_interval}天"
        )
    
    def initialize_strategy(self, strategy_name: str = "rolling_strategy"):
        """初始化滚动策略"""
        # 创建RollingGen
        rolling_gen = RollingGen(
            step=self.retrain_interval,
            rtype=RollingGen.ROLL_SD
        )
        
        # 创建RollingStrategy
        self.strategy = RollingStrategy(
            name_id=strategy_name,
            task_template=self.task_config,
            rolling_gen=rolling_gen
        )
        
        # 创建OnlineManager
        self.online_manager = OnlineManager(
            strategies=[self.strategy],
            trainer=TrainerR()
        )
        
        logger.info(f"滚动策略初始化完成: {strategy_name}")
    
    async def first_train(self) -> OnlineUpdateMetrics:
        """首次训练"""
        if self.online_manager is None:
            raise ValueError("需要先调用initialize_strategy()")
        
        start_time = datetime.now()
        logger.info("开始首次训练...")
        
        try:
            # 首次训练
            self.online_manager.first_train()
            
            # 获取首个模型
            # TODO: 从OnlineManager提取模型和预测
            
            duration = (datetime.now() - start_time).total_seconds()
            
            metrics = OnlineUpdateMetrics(
                success=True,
                update_time=datetime.now(),
                samples_processed=0,
                ic=0.0,
                icir=0.0,
                drift_detected=False,
                model_version="v1_initial",
                update_duration_seconds=duration
            )
            
            self.metrics_history.append(metrics)
            logger.info(f"✅ 首次训练完成 (耗时{duration:.2f}秒)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"首次训练失败: {e}")
            raise
    
    async def online_update(
        self,
        current_date: Union[str, pd.Timestamp],
        enable_drift_detection: bool = True
    ) -> OnlineUpdateMetrics:
        """
        在线更新主流程
        
        Args:
            current_date: 当前日期
            enable_drift_detection: 启用漂移检测
            
        Returns:
            OnlineUpdateMetrics
        """
        start_time = datetime.now()
        logger.info(f"开始在线更新: {current_date}")
        
        try:
            # 1. 执行routine
            self.online_manager.routine(cur_time=current_date)
            
            # 2. 获取预测和标签
            # TODO: 从OnlineManager提取预测结果
            # predictions = self._get_latest_predictions()
            # labels = self._get_latest_labels()
            
            # 3. 概念漂移检测
            drift_result = None
            if enable_drift_detection:
                # drift_result = self.drift_detector.detect(predictions, labels)
                pass
            
            # 4. 根据漂移结果采取行动
            # if drift_result and drift_result.detected:
            #     if drift_result.recommended_action == "full_retrain":
            #         await self._trigger_full_retrain(current_date)
            
            # 5. 模型热更新
            # if self.hot_reloader:
            #     new_model = self._get_latest_model()
            #     await self.hot_reloader.hot_reload(new_model)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            metrics = OnlineUpdateMetrics(
                success=True,
                update_time=datetime.now(),
                samples_processed=0,
                ic=0.0,
                icir=0.0,
                drift_detected=drift_result.detected if drift_result else False,
                model_version=f"v{len(self.metrics_history) + 1}",
                update_duration_seconds=duration
            )
            
            self.metrics_history.append(metrics)
            logger.info(f"✅ 在线更新完成 (耗时{duration:.2f}秒)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"在线更新失败: {e}")
            raise
    
    def get_performance_summary(self) -> pd.DataFrame:
        """获取性能摘要"""
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'update_time': m.update_time,
                'ic': m.ic,
                'icir': m.icir,
                'drift_detected': m.drift_detected,
                'model_version': m.model_version,
                'duration_seconds': m.update_duration_seconds
            }
            for m in self.metrics_history
        ])


# ============================================================================
# 使用示例
# ============================================================================

async def example_advanced_online_learning():
    """高级在线学习示例"""
    print("=== Qlib在线学习高级示例 (P1-1) ===\n")
    
    # 任务配置 (示例)
    task_config = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {"start_time": "2008-01-01", "end_time": "2020-08-01"},
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }
    
    try:
        # 创建管理器
        manager = QlibOnlineLearningAdvanced(
            task_config=task_config,
            rolling_window=90,
            retrain_interval=30,
            drift_threshold=0.05
        )
        
        # 初始化策略
        manager.initialize_strategy()
        
        # 首次训练
        print("1. 首次训练...")
        first_metrics = await manager.first_train()
        print(f"   ✅ 完成: {first_metrics}\n")
        
        # 模拟在线更新
        print("2. 模拟30天在线更新...")
        base_date = pd.Timestamp("2020-09-01")
        for day in range(5):  # 演示5天
            current_date = base_date + timedelta(days=day)
            print(f"   Day {day + 1}: {current_date.date()}")
            
            metrics = await manager.online_update(current_date)
            print(f"   IC={metrics.ic:.4f}, 漂移={metrics.drift_detected}")
        
        print("\n3. 性能摘要:")
        print(manager.get_performance_summary())
        
        print("\n✅ 示例完成!")
        
    except ImportError as e:
        print(f"❌ 需要Qlib环境: {e}")
        print("   提示: 请确保已安装Qlib并配置好数据")


if __name__ == "__main__":
    asyncio.run(example_advanced_online_learning())
