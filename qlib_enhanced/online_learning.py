"""
Qlib在线学习模块
实现增量模型更新、概念漂移检测、自适应学习率调整
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
import pickle

logger = logging.getLogger(__name__)


# ============================================================================
# 在线学习管理器
# ============================================================================

@dataclass
class ConceptDrift:
    """概念漂移检测结果"""
    detected: bool
    drift_score: float
    detection_time: datetime
    affected_features: List[str]
    recommended_action: str


@dataclass
class OnlineUpdateResult:
    """在线更新结果"""
    success: bool
    samples_processed: int
    new_accuracy: float
    drift_detected: bool
    update_time: datetime
    model_version: str


class OnlineLearningManager:
    """在线学习管理器"""
    
    def __init__(self, 
                 base_model,
                 update_frequency: str = "daily",
                 drift_threshold: float = 0.05,
                 enable_drift_detection: bool = True):
        """
        初始化在线学习管理器
        
        Args:
            base_model: 基础模型
            update_frequency: 更新频率 (daily, weekly, monthly)
            drift_threshold: 漂移检测阈值
            enable_drift_detection: 是否启用漂移检测
        """
        self.base_model = base_model
        self.update_frequency = update_frequency
        self.drift_threshold = drift_threshold
        self.enable_drift_detection = enable_drift_detection
        
        # 历史性能追踪
        self.performance_history = []
        self.model_versions = []
        
        # 概念漂移检测器
        self.drift_detector = DriftDetector(threshold=drift_threshold)
        
        # 增量学习缓冲区
        self.update_buffer = []
        self.buffer_size = 1000
        
        logger.info(f"在线学习管理器初始化: 更新频率={update_frequency}")
    
    async def incremental_update(self, 
                                 new_data: pd.DataFrame,
                                 new_labels: pd.Series) -> OnlineUpdateResult:
        """
        增量更新模型
        
        Args:
            new_data: 新的特征数据
            new_labels: 新的标签数据
            
        Returns:
            更新结果
        """
        logger.info(f"开始增量更新，新数据量: {len(new_data)}")
        
        try:
            # 1. 检测概念漂移
            drift_result = None
            if self.enable_drift_detection:
                drift_result = self.drift_detector.detect(new_data, new_labels)
                
                if drift_result.detected:
                    logger.warning(f"检测到概念漂移！得分: {drift_result.drift_score:.4f}")
                    # 如果漂移严重，重新训练模型
                    if drift_result.drift_score > self.drift_threshold * 2:
                        return await self._full_retrain(new_data, new_labels)
            
            # 2. 添加到缓冲区
            self.update_buffer.append((new_data, new_labels))
            
            # 3. 如果缓冲区满了，执行批量更新
            if len(self.update_buffer) >= self.buffer_size:
                result = await self._batch_update()
                self.update_buffer = []
                return result
            
            # 4. 返回当前状态
            return OnlineUpdateResult(
                success=True,
                samples_processed=len(new_data),
                new_accuracy=self._estimate_accuracy(new_data, new_labels),
                drift_detected=drift_result.detected if drift_result else False,
                update_time=datetime.now(),
                model_version=self._get_model_version()
            )
            
        except Exception as e:
            logger.error(f"增量更新失败: {e}")
            return OnlineUpdateResult(
                success=False,
                samples_processed=0,
                new_accuracy=0.0,
                drift_detected=False,
                update_time=datetime.now(),
                model_version=self._get_model_version()
            )
    
    async def _batch_update(self) -> OnlineUpdateResult:
        """批量更新模型"""
        logger.info(f"执行批量更新，缓冲区大小: {len(self.update_buffer)}")
        
        # 合并缓冲区数据
        all_data = pd.concat([data for data, _ in self.update_buffer])
        all_labels = pd.concat([labels for _, labels in self.update_buffer])
        
        # 增量训练
        self.base_model.fit_incremental(all_data, all_labels)
        
        # 评估性能
        accuracy = self._estimate_accuracy(all_data, all_labels)
        
        # 保存版本
        version = self._save_model_version()
        
        return OnlineUpdateResult(
            success=True,
            samples_processed=len(all_data),
            new_accuracy=accuracy,
            drift_detected=False,
            update_time=datetime.now(),
            model_version=version
        )
    
    async def _full_retrain(self, 
                           new_data: pd.DataFrame,
                           new_labels: pd.Series) -> OnlineUpdateResult:
        """完全重新训练模型"""
        logger.warning("执行完全重训练...")
        
        # 重新训练
        self.base_model.fit(new_data, new_labels)
        
        # 评估
        accuracy = self._estimate_accuracy(new_data, new_labels)
        
        # 保存版本
        version = self._save_model_version()
        
        return OnlineUpdateResult(
            success=True,
            samples_processed=len(new_data),
            new_accuracy=accuracy,
            drift_detected=True,
            update_time=datetime.now(),
            model_version=version
        )
    
    def _estimate_accuracy(self, data: pd.DataFrame, labels: pd.Series) -> float:
        """估计模型准确率"""
        try:
            predictions = self.base_model.predict(data)
            # 简化：使用相关系数作为准确率
            return np.corrcoef(predictions, labels)[0, 1]
        except:
            return 0.0
    
    def _get_model_version(self) -> str:
        """获取模型版本"""
        return f"v{len(self.model_versions)}"
    
    def _save_model_version(self) -> str:
        """保存模型版本"""
        version = f"v{len(self.model_versions) + 1}"
        self.model_versions.append({
            'version': version,
            'timestamp': datetime.now(),
            'model': pickle.dumps(self.base_model)
        })
        return version
    
    def get_performance_history(self) -> pd.DataFrame:
        """获取性能历史"""
        return pd.DataFrame(self.performance_history)


# ============================================================================
# 概念漂移检测器
# ============================================================================

class DriftDetector:
    """概念漂移检测器"""
    
    def __init__(self, threshold: float = 0.05, window_size: int = 100):
        self.threshold = threshold
        self.window_size = window_size
        self.reference_distribution = None
        self.history = []
    
    def detect(self, new_data: pd.DataFrame, new_labels: pd.Series) -> ConceptDrift:
        """
        检测概念漂移
        
        使用Kolmogorov-Smirnov检验检测分布变化
        """
        # 初始化参考分布
        if self.reference_distribution is None:
            self.reference_distribution = self._compute_distribution(new_data, new_labels)
            return ConceptDrift(
                detected=False,
                drift_score=0.0,
                detection_time=datetime.now(),
                affected_features=[],
                recommended_action="initialize"
            )
        
        # 计算当前分布
        current_distribution = self._compute_distribution(new_data, new_labels)
        
        # 计算KS统计量
        drift_scores = {}
        for col in new_data.columns:
            if col in self.reference_distribution and col in current_distribution:
                ks_stat = self._ks_test(
                    self.reference_distribution[col],
                    current_distribution[col]
                )
                drift_scores[col] = ks_stat
        
        # 综合漂移得分
        avg_drift = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        
        # 判断是否漂移
        detected = avg_drift > self.threshold
        
        # 找出受影响最大的特征
        affected_features = sorted(
            drift_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        affected_features = [f[0] for f in affected_features]
        
        # 推荐行动
        if avg_drift > self.threshold * 2:
            action = "full_retrain"
        elif avg_drift > self.threshold:
            action = "incremental_update"
        else:
            action = "no_action"
        
        return ConceptDrift(
            detected=detected,
            drift_score=avg_drift,
            detection_time=datetime.now(),
            affected_features=affected_features,
            recommended_action=action
        )
    
    def _compute_distribution(self, 
                             data: pd.DataFrame,
                             labels: pd.Series) -> Dict[str, np.ndarray]:
        """计算数据分布"""
        distributions = {}
        for col in data.columns:
            distributions[col] = data[col].values
        return distributions
    
    def _ks_test(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Kolmogorov-Smirnov检验"""
        from scipy import stats
        statistic, _ = stats.ks_2samp(dist1, dist2)
        return statistic


# ============================================================================
# 自适应学习率调整器
# ============================================================================

class AdaptiveLearningRate:
    """自适应学习率调整器"""
    
    def __init__(self, 
                 initial_lr: float = 0.01,
                 min_lr: float = 0.0001,
                 max_lr: float = 0.1,
                 patience: int = 5):
        """
        初始化自适应学习率调整器
        
        Args:
            initial_lr: 初始学习率
            min_lr: 最小学习率
            max_lr: 最大学习率
            patience: 耐心值（性能无提升时等待的轮数）
        """
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        
        self.best_performance = -np.inf
        self.wait_count = 0
        self.performance_history = []
    
    def update(self, current_performance: float) -> float:
        """
        根据性能更新学习率
        
        Args:
            current_performance: 当前性能指标
            
        Returns:
            新的学习率
        """
        self.performance_history.append(current_performance)
        
        # 如果性能提升
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # 如果等待超过patience，降低学习率
        if self.wait_count >= self.patience:
            self.current_lr = max(self.current_lr * 0.5, self.min_lr)
            self.wait_count = 0
            logger.info(f"降低学习率到: {self.current_lr}")
        
        # 如果性能持续提升，尝试提高学习率
        if len(self.performance_history) >= 3:
            recent_trend = np.diff(self.performance_history[-3:])
            if all(recent_trend > 0):
                self.current_lr = min(self.current_lr * 1.1, self.max_lr)
                logger.info(f"提高学习率到: {self.current_lr}")
        
        return self.current_lr
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr


# ============================================================================
# 使用示例
# ============================================================================

async def example_online_learning():
    """在线学习示例"""
    print("=== Qlib在线学习示例 ===\n")
    
    # 模拟基础模型
    class MockModel:
        def fit(self, X, y):
            print("完全训练模型")
        
        def fit_incremental(self, X, y):
            print("增量训练模型")
        
        def predict(self, X):
            return np.random.randn(len(X))
    
    model = MockModel()
    
    # 创建在线学习管理器
    manager = OnlineLearningManager(
        base_model=model,
        update_frequency="daily",
        enable_drift_detection=True
    )
    
    # 模拟新数据
    new_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    new_labels = pd.Series(np.random.randn(100))
    
    # 增量更新
    result = await manager.incremental_update(new_data, new_labels)
    
    print(f"更新成功: {result.success}")
    print(f"处理样本数: {result.samples_processed}")
    print(f"新准确率: {result.new_accuracy:.4f}")
    print(f"检测到漂移: {result.drift_detected}")
    print(f"模型版本: {result.model_version}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_online_learning())
