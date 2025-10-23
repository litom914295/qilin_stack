"""
涨停板预测系统 - GPU加速训练模块
使用类RAPIDS风格API，自动检测GPU可用性，CPU/GPU无缝切换
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import time

# 尝试导入GPU加速库
import logging
logger = logging.getLogger(__name__)
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_AVAILABLE = True
    logger.info("✅ GPU加速库已加载 (cuDF + cuML)")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("⚠️ GPU库不可用，将使用CPU模式")

# CPU后备库
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


class GPUAcceleratedPreprocessor:
    """GPU加速数据预处理器"""
    
    def __init__(self, use_gpu: bool = True):
        """
        初始化预处理器
        
        Args:
            use_gpu: 是否使用GPU加速（如果可用）
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = 'GPU' if self.use_gpu else 'CPU'
        
        logger.info(f"数据处理设备: {self.device}")
    
    def to_device(self, df: pd.DataFrame):
        """将DataFrame转换到目标设备"""
        if self.use_gpu:
            return cudf.from_pandas(df)
        return df
    
    def to_cpu(self, df):
        """将数据转回CPU"""
        if self.use_gpu and isinstance(df, cudf.DataFrame):
            return df.to_pandas()
        return df
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算高级特征（GPU加速）
        
        Args:
            df: 原始数据
            
        Returns:
            特征数据
        """
        logger.info(f"开始特征计算 (设备: {self.device})...")
        start_time = time.time()
        
        # 转换到目标设备
        gdf = self.to_device(df)
        
        # 计算统计特征（向量化操作，GPU自动加速）
        features = []
        
        # 1. 移动平均特征
        for window in [5, 10, 20]:
            col_name = f'ma_{window}'
            if 'close' in gdf.columns:
                gdf[col_name] = gdf['close'].rolling(window=window).mean()
                features.append(col_name)
        
        # 2. 波动率特征
        for window in [5, 10, 20]:
            col_name = f'std_{window}'
            if 'close' in gdf.columns:
                gdf[col_name] = gdf['close'].rolling(window=window).std()
                features.append(col_name)
        
        # 3. 动量特征
        for period in [1, 5, 10]:
            col_name = f'momentum_{period}'
            if 'close' in gdf.columns:
                gdf[col_name] = gdf['close'].pct_change(periods=period)
                features.append(col_name)
        
        # 4. 成交量特征
        if 'volume' in gdf.columns:
            for window in [5, 10]:
                col_name = f'volume_ma_{window}'
                gdf[col_name] = gdf['volume'].rolling(window=window).mean()
                features.append(col_name)
        
        # 转回CPU
        result = self.to_cpu(gdf)
        
        elapsed = time.time() - start_time
        logger.info(f"特征计算完成，耗时: {elapsed:.2f}秒")
        logger.info(f"生成特征数: {len(features)}")
        
        return result


class GPUAcceleratedModel:
    """GPU加速模型训练器"""
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        use_gpu: bool = True,
        **model_params
    ):
        """
        初始化模型
        
        Args:
            model_type: 模型类型 ('xgboost', 'lightgbm', 'random_forest')
            use_gpu: 是否使用GPU
            **model_params: 模型参数
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = 'GPU' if self.use_gpu else 'CPU'
        self.model_params = model_params
        
        self.model = None
        self.training_time = 0
        
        logger.info(f"模型: {model_type}, 设备: {self.device}")
    
    def _create_model(self):
        """创建模型实例"""
        
        if self.model_type == 'xgboost':
            params = {
                'tree_method': 'hist' if not self.use_gpu else 'gpu_hist',
                'device': 'cuda' if self.use_gpu else 'cpu',
                'random_state': 42,
                **self.model_params
            }
            return xgb.XGBClassifier(**params)
        
        elif self.model_type == 'lightgbm':
            params = {
                'device': 'gpu' if self.use_gpu else 'cpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'random_state': 42,
                **self.model_params
            }
            return lgb.LGBMClassifier(**params)
        
        elif self.model_type == 'random_forest':
            if self.use_gpu:
                # cuML RandomForest
                params = {
                    'n_estimators': self.model_params.get('n_estimators', 100),
                    'max_depth': self.model_params.get('max_depth', 10),
                    'random_state': 42
                }
                return cuRF(**params)
            else:
                # Sklearn RandomForest
                params = {
                    'n_estimators': self.model_params.get('n_estimators', 100),
                    'max_depth': self.model_params.get('max_depth', 10),
                    'random_state': 42,
                    'n_jobs': -1
                }
                return RandomForestClassifier(**params)
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        print(f"\n🚀 开始训练 (设备: {self.device})...")
        print(f"训练集大小: {X_train.shape}")
        
        start_time = time.time()
        
        # 创建模型
        self.model = self._create_model()
        
        # 训练
        if X_val is not None and y_val is not None:
            if self.model_type in ['xgboost', 'lightgbm']:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        
        print(f"✅ 训练完成，耗时: {self.training_time:.2f}秒")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        predictions = self.model.predict(X)
        
        # cuML返回cuDF Series，需要转换
        if self.use_gpu and hasattr(predictions, 'to_numpy'):
            predictions = predictions.to_numpy()
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        probas = self.model.predict_proba(X)
        
        # cuML返回cuDF DataFrame，需要转换
        if self.use_gpu and hasattr(probas, 'to_numpy'):
            probas = probas.to_numpy()
        
        return probas


class GPUAcceleratedPipeline:
    """GPU加速完整Pipeline"""
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        use_gpu: bool = True,
        save_dir: str = './gpu_models'
    ):
        """
        初始化Pipeline
        
        Args:
            model_type: 模型类型
            use_gpu: 是否使用GPU
            save_dir: 模型保存目录
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.preprocessor = GPUAcceleratedPreprocessor(use_gpu=use_gpu)
        self.model = None
        
        # 性能统计
        self.stats = {
            'preprocess_time': 0,
            'training_time': 0,
            'inference_time': 0,
            'speedup_ratio': 1.0
        }
        
        print(f"\n{'='*60}")
        print(f"🚀 GPU加速Pipeline初始化")
        print(f"模型: {model_type}")
        print(f"设备: {'GPU' if self.use_gpu else 'CPU'}")
        print(f"{'='*60}\n")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **model_params
    ):
        """训练Pipeline"""
        
        total_start = time.time()
        
        # 1. 数据预处理
        logger.info("数据预处理...")
        preprocess_start = time.time()
        X_train_processed = self.preprocessor.compute_features(X_train)
        if X_val is not None:
            X_val_processed = self.preprocessor.compute_features(X_val)
        else:
            X_val_processed = None
        self.stats['preprocess_time'] = time.time() - preprocess_start
        
        # 2. 模型训练
        logger.info("模型训练...")
        self.model = GPUAcceleratedModel(
            model_type=self.model_type,
            use_gpu=self.use_gpu,
            **model_params
        )
        
        self.model.train(
            X_train_processed,
            y_train,
            X_val_processed if X_val_processed is not None else None,
            y_val
        )
        self.stats['training_time'] = self.model.training_time
        
        # 3. 总结
        total_time = time.time() - total_start
        logger.info("="*60)
        logger.info("Pipeline训练完成!")
        logger.info("="*60)
        logger.info(f"预处理耗时: {self.stats['preprocess_time']:.2f}秒")
        logger.info(f"训练耗时: {self.stats['training_time']:.2f}秒")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info("="*60)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        start_time = time.time()
        
        # 预处理
        X_processed = self.preprocessor.compute_features(X)
        
        # 预测
        predictions = self.model.predict(X_processed)
        
        self.stats['inference_time'] = time.time() - start_time
        
        return predictions
    
    def benchmark(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_runs: int = 3
    ) -> Dict[str, float]:
        """
        性能基准测试（对比CPU vs GPU）
        
        Returns:
            性能统计字典
        """
        logger.info("="*60)
        logger.info(f"性能基准测试 (运行{n_runs}次)")
        logger.info("="*60)
        
        results = {'cpu': [], 'gpu': []}
        
        # CPU测试
        if not self.use_gpu:
            print("⚠️ 当前已在CPU模式，跳过对比测试")
            return {'cpu_time': 0, 'gpu_time': 0, 'speedup': 1.0}
        
        for device_type in ['cpu', 'gpu']:
            use_gpu = (device_type == 'gpu')
            
            logger.info("-"*40)
            logger.info(f"测试设备: {device_type.upper()}")
            logger.info("-"*40)
            
            for run in range(n_runs):
                logger.info(f"运行 {run+1}/{n_runs}...")
                
                # 创建Pipeline
                pipeline = GPUAcceleratedPipeline(
                    model_type=self.model_type,
                    use_gpu=use_gpu,
                    save_dir=str(self.save_dir)
                )
                
                # 训练
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                elapsed = time.time() - start_time
                
                results[device_type].append(elapsed)
                logger.info(f"耗时: {elapsed:.2f}秒")
        
        # 计算统计
        cpu_time = np.mean(results['cpu'])
        gpu_time = np.mean(results['gpu'])
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        
        logger.info("="*60)
        logger.info("基准测试结果汇总")
        logger.info("="*60)
        logger.info(f"CPU平均耗时: {cpu_time:.2f}秒")
        logger.info(f"GPU平均耗时: {gpu_time:.2f}秒")
        logger.info(f"加速比: {speedup:.2f}x")
        logger.info("="*60)
        
        self.stats['speedup_ratio'] = speedup
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        }
    
    def save(self, filename: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        filepath = self.save_dir / filename
        
        # 保存模型（具体实现取决于模型类型）
        if self.model_type in ['xgboost', 'lightgbm']:
            self.model.model.save_model(str(filepath))
        else:
            import joblib
            joblib.dump(self.model.model, filepath)
        
        # 保存元数据
        metadata = {
            'model_type': self.model_type,
            'use_gpu': self.use_gpu,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存: {filepath}")


if __name__ == '__main__':
    from app.core.logging_setup import setup_logging
    setup_logging()
    logger.info("="*60)
    logger.info("涨停板预测系统 - GPU加速训练模块")
    logger.info("="*60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 5000
    n_features = 100
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_train['close'] = np.random.randn(n_samples).cumsum()
    X_train['volume'] = np.random.randint(1000, 10000, n_samples)
    
    y_train = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]))
    
    logger.info(f"数据集大小: {X_train.shape}")
    logger.info(f"涨停板样本占比: {y_train.mean():.2%}")
    
    # 测试GPU Pipeline
    logger.info("="*60)
    logger.info("GPU加速Pipeline测试")
    logger.info("="*60)
    
    pipeline = GPUAcceleratedPipeline(
        model_type='xgboost',
        use_gpu=True
    )
    
    pipeline.fit(
        X_train, y_train,
        n_estimators=100,
        max_depth=6
    )
    
    # 预测测试
    X_test = X_train.head(100)
    predictions = pipeline.predict(X_test)
    logger.info(f"预测结果示例: {predictions[:10]}")
    
    # 如果GPU可用，运行基准测试
    if GPU_AVAILABLE:
        logger.info("="*60)
        logger.info("性能基准测试")
        logger.info("="*60)
        
        benchmark_results = pipeline.benchmark(
            X_train.head(1000),
            y_train.head(1000),
            n_runs=2
        )
    
    logger.info("✅ 所有测试完成!")
