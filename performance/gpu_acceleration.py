"""
GPU加速模块
支持GPU加速回测、模型训练、批量预测
使用CuPy、RAPIDS cuDF、PyTorch等框架
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


# ============================================================================
# GPU后端枚举
# ============================================================================

class GPUBackend(Enum):
    """GPU后端类型"""
    CUPY = "cupy"           # CuPy - NumPy的GPU版本
    RAPIDS = "rapids"       # RAPIDS cuDF - Pandas的GPU版本
    TORCH = "torch"         # PyTorch
    TENSORFLOW = "tensorflow"  # TensorFlow


@dataclass
class GPUInfo:
    """GPU信息"""
    available: bool
    device_count: int
    device_name: str = ""
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0
    cuda_version: str = ""


# ============================================================================
# GPU工具类
# ============================================================================

class GPUUtils:
    """GPU工具函数"""
    
    @staticmethod
    def get_gpu_info() -> GPUInfo:
        """获取GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                memory_free = (torch.cuda.get_device_properties(0).total_memory - 
                              torch.cuda.memory_allocated(0)) / 1024**2
                cuda_version = torch.version.cuda
                
                return GPUInfo(
                    available=True,
                    device_count=device_count,
                    device_name=device_name,
                    memory_total_mb=memory_total,
                    memory_free_mb=memory_free,
                    cuda_version=cuda_version
                )
        except Exception as e:
            logger.warning(f"无法获取GPU信息: {e}")
        
        return GPUInfo(available=False, device_count=0)
    
    @staticmethod
    def check_backend_available(backend: GPUBackend) -> bool:
        """检查GPU后端是否可用"""
        try:
            if backend == GPUBackend.CUPY:
                import cupy as cp
                _ = cp.array([1, 2, 3])
                return True
            elif backend == GPUBackend.RAPIDS:
                import cudf
                _ = cudf.DataFrame({'a': [1, 2, 3]})
                return True
            elif backend == GPUBackend.TORCH:
                import torch
                return torch.cuda.is_available()
            elif backend == GPUBackend.TENSORFLOW:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
        except Exception as e:
            logger.warning(f"{backend.value}后端不可用: {e}")
            return False
        
        return False


# ============================================================================
# GPU加速的数据处理
# ============================================================================

class GPUDataProcessor:
    """GPU加速的数据处理器"""
    
    def __init__(self, backend: GPUBackend = GPUBackend.RAPIDS):
        """
        初始化GPU数据处理器
        
        Args:
            backend: GPU后端
        """
        self.backend = backend
        self.available = GPUUtils.check_backend_available(backend)
        
        if not self.available:
            logger.warning(f"{backend.value}不可用，将回退到CPU")
            self.backend = None
        else:
            logger.info(f"使用GPU后端: {backend.value}")
    
    def calculate_indicators_gpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        GPU加速计算技术指标
        
        Args:
            df: 价格数据（必须包含OHLCV列）
            
        Returns:
            添加了指标的DataFrame
        """
        if self.backend == GPUBackend.RAPIDS:
            return self._calculate_indicators_rapids(df)
        elif self.backend == GPUBackend.CUPY:
            return self._calculate_indicators_cupy(df)
        else:
            return self._calculate_indicators_cpu(df)
    
    def _calculate_indicators_rapids(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用RAPIDS cuDF计算指标"""
        try:
            import cudf
            
            # 转换为GPU DataFrame
            gdf = cudf.from_pandas(df)
            
            # 计算移动平均
            gdf['ma_5'] = gdf['close'].rolling(window=5).mean()
            gdf['ma_10'] = gdf['close'].rolling(window=10).mean()
            gdf['ma_20'] = gdf['close'].rolling(window=20).mean()
            
            # 计算收益率
            gdf['return'] = gdf['close'].pct_change()
            gdf['return_5'] = gdf['close'].pct_change(periods=5)
            gdf['return_10'] = gdf['close'].pct_change(periods=10)
            
            # 计算波动率
            gdf['volatility_20'] = gdf['return'].rolling(window=20).std()
            
            # 计算成交量指标
            gdf['volume_ma_5'] = gdf['volume'].rolling(window=5).mean()
            gdf['volume_ratio'] = gdf['volume'] / gdf['volume_ma_5']
            
            # 转回pandas
            result = gdf.to_pandas()
            logger.info(f"RAPIDS GPU加速: 计算了{len(result)}行数据的指标")
            
            return result
            
        except Exception as e:
            logger.error(f"RAPIDS计算失败: {e}")
            return self._calculate_indicators_cpu(df)
    
    def _calculate_indicators_cupy(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用CuPy计算指标"""
        try:
            import cupy as cp
            
            # 将数据转到GPU
            close_gpu = cp.array(df['close'].values)
            volume_gpu = cp.array(df['volume'].values)
            
            # 计算移动平均（简化版）
            ma_5 = cp.convolve(close_gpu, cp.ones(5)/5, mode='same')
            ma_10 = cp.convolve(close_gpu, cp.ones(10)/10, mode='same')
            ma_20 = cp.convolve(close_gpu, cp.ones(20)/20, mode='same')
            
            # 计算收益率
            returns = cp.diff(close_gpu) / close_gpu[:-1]
            returns = cp.concatenate([cp.array([cp.nan]), returns])
            
            # 转回CPU
            df = df.copy()
            df['ma_5'] = cp.asnumpy(ma_5)
            df['ma_10'] = cp.asnumpy(ma_10)
            df['ma_20'] = cp.asnumpy(ma_20)
            df['return'] = cp.asnumpy(returns)
            
            logger.info(f"CuPy GPU加速: 计算了{len(df)}行数据的指标")
            return df
            
        except Exception as e:
            logger.error(f"CuPy计算失败: {e}")
            return self._calculate_indicators_cpu(df)
    
    def _calculate_indicators_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """CPU回退版本"""
        df = df.copy()
        
        # 移动平均
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # 收益率
        df['return'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(periods=5)
        df['return_10'] = df['close'].pct_change(periods=10)
        
        # 波动率
        df['volatility_20'] = df['return'].rolling(window=20).std()
        
        # 成交量
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        logger.info(f"CPU计算: 计算了{len(df)}行数据的指标")
        return df


# ============================================================================
# GPU加速的回测引擎
# ============================================================================

class GPUBacktestEngine:
    """GPU加速的回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.gpu_available = GPUUtils.check_backend_available(GPUBackend.TORCH)
        
        if self.gpu_available:
            logger.info("GPU回测引擎已启用")
        else:
            logger.warning("GPU不可用，使用CPU回测")
    
    def vectorized_backtest(self, 
                           prices: np.ndarray,
                           signals: np.ndarray,
                           commission: float = 0.001) -> Dict[str, Any]:
        """
        向量化回测（GPU加速）
        
        Args:
            prices: 价格数组 (n_days, n_stocks)
            signals: 信号数组 (n_days, n_stocks) [-1, 0, 1]
            commission: 交易佣金率
            
        Returns:
            回测结果字典
        """
        if self.gpu_available:
            return self._backtest_gpu(prices, signals, commission)
        else:
            return self._backtest_cpu(prices, signals, commission)
    
    def _backtest_gpu(self, prices: np.ndarray, signals: np.ndarray, 
                     commission: float) -> Dict[str, Any]:
        """GPU版本回测"""
        try:
            import torch
            
            device = torch.device('cuda')
            
            # 转到GPU
            prices_gpu = torch.tensor(prices, dtype=torch.float32, device=device)
            signals_gpu = torch.tensor(signals, dtype=torch.float32, device=device)
            
            # 计算收益率
            returns = prices_gpu[1:] / prices_gpu[:-1] - 1.0
            
            # 计算仓位变化
            position_changes = torch.diff(signals_gpu, dim=0)
            
            # 计算交易成本
            trade_costs = torch.abs(position_changes) * commission
            
            # 计算策略收益
            strategy_returns = signals_gpu[:-1] * returns - trade_costs
            
            # 累积收益
            cumulative_returns = torch.cumprod(1 + strategy_returns.sum(dim=1), dim=0)
            
            # 转回CPU
            cumulative_returns_cpu = cumulative_returns.cpu().numpy()
            
            # 计算指标
            total_return = cumulative_returns_cpu[-1] - 1.0
            sharpe_ratio = self._calculate_sharpe_gpu(strategy_returns)
            max_drawdown = self._calculate_max_drawdown_gpu(cumulative_returns)
            
            logger.info(f"GPU回测完成: 收益率={total_return:.2%}, Sharpe={sharpe_ratio:.2f}")
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'cumulative_returns': cumulative_returns_cpu,
                'final_capital': self.initial_capital * (1 + total_return)
            }
            
        except Exception as e:
            logger.error(f"GPU回测失败: {e}")
            return self._backtest_cpu(prices, signals, commission)
    
    def _backtest_cpu(self, prices: np.ndarray, signals: np.ndarray, 
                     commission: float) -> Dict[str, Any]:
        """CPU版本回测"""
        # 简化的CPU实现
        returns = prices[1:] / prices[:-1] - 1.0
        position_changes = np.diff(signals, axis=0)
        trade_costs = np.abs(position_changes) * commission
        strategy_returns = signals[:-1] * returns - trade_costs
        cumulative_returns = np.cumprod(1 + strategy_returns.sum(axis=1))
        
        total_return = cumulative_returns[-1] - 1.0
        sharpe_ratio = np.mean(strategy_returns.sum(axis=1)) / np.std(strategy_returns.sum(axis=1)) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown_cpu(cumulative_returns)
        
        logger.info(f"CPU回测完成: 收益率={total_return:.2%}, Sharpe={sharpe_ratio:.2f}")
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns,
            'final_capital': self.initial_capital * (1 + total_return)
        }
    
    def _calculate_sharpe_gpu(self, returns_gpu) -> float:
        """GPU计算Sharpe比率"""
        import torch
        daily_returns = returns_gpu.sum(dim=1)
        mean_return = torch.mean(daily_returns)
        std_return = torch.std(daily_returns)
        sharpe = (mean_return / std_return * torch.sqrt(torch.tensor(252.0))).item()
        return sharpe
    
    def _calculate_max_drawdown_gpu(self, cumulative_returns_gpu) -> float:
        """GPU计算最大回撤"""
        import torch
        running_max = torch.cummax(cumulative_returns_gpu, dim=0)[0]
        drawdown = (cumulative_returns_gpu - running_max) / running_max
        max_dd = torch.min(drawdown).item()
        return max_dd
    
    def _calculate_max_drawdown_cpu(self, cumulative_returns: np.ndarray) -> float:
        """CPU计算最大回撤"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)


# ============================================================================
# GPU加速的模型训练
# ============================================================================

class GPUModelTrainer:
    """GPU加速的模型训练器"""
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        初始化模型训练器
        
        Args:
            model_type: 模型类型 (lightgbm, xgboost, pytorch)
        """
        self.model_type = model_type
        self.model = None
        self.gpu_available = GPUUtils.get_gpu_info().available
        
        logger.info(f"GPU模型训练器: {model_type}, GPU可用={self.gpu_available}")
    
    def train_gpu(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        GPU加速训练
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练结果
        """
        start_time = time.time()
        
        if self.model_type == "lightgbm":
            result = self._train_lightgbm_gpu(X_train, y_train, X_val, y_val)
        elif self.model_type == "xgboost":
            result = self._train_xgboost_gpu(X_train, y_train, X_val, y_val)
        elif self.model_type == "pytorch":
            result = self._train_pytorch_gpu(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        training_time = time.time() - start_time
        result['training_time'] = training_time
        
        logger.info(f"GPU训练完成: {training_time:.2f}秒")
        return result
    
    def _train_lightgbm_gpu(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """LightGBM GPU训练"""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'device': 'gpu' if self.gpu_available else 'cpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            
            if X_val is not None:
                valid_data = lgb.Dataset(X_val, label=y_val)
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[valid_data],
                    valid_names=['valid']
                )
            else:
                self.model = lgb.train(params, train_data, num_boost_round=100)
            
            return {
                'model': self.model,
                'model_type': 'lightgbm',
                'device': 'gpu' if self.gpu_available else 'cpu'
            }
            
        except Exception as e:
            logger.error(f"LightGBM GPU训练失败: {e}")
            return {}
    
    def _train_xgboost_gpu(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """XGBoost GPU训练"""
        try:
            import xgboost as xgb
            
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist' if self.gpu_available else 'hist',
                'gpu_id': 0,
                'learning_rate': 0.05,
                'max_depth': 6
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            if X_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals = [(dval, 'eval')]
                self.model = xgb.train(params, dtrain, num_boost_round=100, evals=evals)
            else:
                self.model = xgb.train(params, dtrain, num_boost_round=100)
            
            return {
                'model': self.model,
                'model_type': 'xgboost',
                'device': 'gpu' if self.gpu_available else 'cpu'
            }
            
        except Exception as e:
            logger.error(f"XGBoost GPU训练失败: {e}")
            return {}
    
    def _train_pytorch_gpu(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """PyTorch GPU训练"""
        try:
            import torch
            import torch.nn as nn
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 简单的神经网络
            class SimpleNN(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, 1)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    return self.fc3(x)
            
            model = SimpleNN(X_train.shape[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 转换数据
            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
            
            # 训练
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            
            self.model = model
            
            return {
                'model': model,
                'model_type': 'pytorch',
                'device': str(device)
            }
            
        except Exception as e:
            logger.error(f"PyTorch GPU训练失败: {e}")
            return {}
    
    def predict_gpu(self, X: np.ndarray) -> np.ndarray:
        """GPU加速预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        if self.model_type == "pytorch":
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_t = torch.FloatTensor(X).to(device)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_t).cpu().numpy()
            return predictions.flatten()
        
        elif self.model_type in ["lightgbm", "xgboost"]:
            return self.model.predict(X)
        
        return np.array([])


# ============================================================================
# 使用示例
# ============================================================================

def example_gpu_acceleration():
    """GPU加速示例"""
    print("=== GPU加速模块示例 ===\n")
    
    # 1. 检查GPU信息
    print("1. GPU信息")
    gpu_info = GPUUtils.get_gpu_info()
    print(f"  GPU可用: {gpu_info.available}")
    if gpu_info.available:
        print(f"  设备: {gpu_info.device_name}")
        print(f"  显存: {gpu_info.memory_total_mb:.0f}MB")
        print(f"  CUDA版本: {gpu_info.cuda_version}")
    
    # 2. GPU数据处理
    print("\n2. GPU数据处理")
    df = pd.DataFrame({
        'close': np.random.randn(10000).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 10000)
    })
    
    processor = GPUDataProcessor(backend=GPUBackend.RAPIDS)
    df_processed = processor.calculate_indicators_gpu(df)
    print(f"  处理了 {len(df_processed)} 行数据")
    print(f"  新增指标: {[col for col in df_processed.columns if col not in df.columns]}")
    
    # 3. GPU回测
    print("\n3. GPU回测")
    prices = np.random.randn(1000, 50).cumsum(axis=0) + 100
    signals = np.random.choice([-1, 0, 1], size=(1000, 50))
    
    engine = GPUBacktestEngine(initial_capital=1000000)
    results = engine.vectorized_backtest(prices, signals)
    print(f"  总收益: {results['total_return']:.2%}")
    print(f"  Sharpe比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    
    # 4. GPU模型训练
    print("\n4. GPU模型训练")
    X_train = np.random.randn(10000, 20)
    y_train = np.random.randn(10000)
    
    trainer = GPUModelTrainer(model_type="lightgbm")
    train_result = trainer.train_gpu(X_train, y_train)
    print(f"  训练时间: {train_result.get('training_time', 0):.2f}秒")
    print(f"  设备: {train_result.get('device', 'N/A')}")


if __name__ == "__main__":
    example_gpu_acceleration()
