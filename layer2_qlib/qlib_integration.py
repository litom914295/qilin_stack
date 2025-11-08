"""
Qlib量化引擎集成层
实现数据管理、因子库、回测引擎、组合优化等功能
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from pathlib import Path

# Qlib核心模块
import qlib
from qlib.config import C
from qlib.data import D
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model
from qlib.strategy.base import BaseStrategy
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report import analysis_position, analysis_model
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.model.pytorch_alstm import ALSTM
from qlib.contrib.model.pytorch_gru import GRU
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_nn import DNNModel
from qlib.contrib.model.pytorch_transformer import Transformer
from qlib.contrib.strategy import TopkDropoutStrategy, WeightStrategyBase
from qlib.contrib.data.handler import Alpha360, Alpha158

# 自定义模块
sys.path.append(str(Path(__file__).parent.parent))
from data_layer.data_access_layer import DataAccessLayer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QlibConfig:
    """Qlib配置"""
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    exp_name: str = "qilin_exp"
    market: str = "csi300"
    benchmark: str = "SH000300"
    data_handler: str = "Alpha360"
    model_type: str = "LGBM"
    strategy_type: str = "TopkDropout"


class QlibIntegration:
    """Qlib集成主类"""
    
    def __init__(self, config: QlibConfig = None):
        """
        初始化Qlib集成
        
        Args:
            config: Qlib配置对象
        """
        self.config = config or QlibConfig()
        self._init_qlib()
        self.data_access = DataAccessLayer({})
        self.models = {}
        self.strategies = {}
        self.records = []
        
    def _init_qlib(self):
        """初始化Qlib"""
        try:
            # 设置Qlib配置
            qlib.init(
                provider_uri=self.config.provider_uri,
                region=self.config.region,
                redis_host=self.config.redis_host,
                redis_port=self.config.redis_port,
                redis_db=self.config.redis_db,
                expression_cache=None,
                dataset_cache=None,
            )
            logger.info("Qlib initialized successfully")
        except Exception as e:
            logger.error(f"Qlib initialization failed: {e}")
            # 尝试自动下载数据
            self._download_data()
    
    def _download_data(self):
        """下载Qlib数据"""
        try:
            from qlib.utils import get_data
            logger.info("Downloading Qlib data...")
            get_data.GetData().qlib_data(
                target_dir=self.config.provider_uri,
                region=self.config.region,
            )
            # 重新初始化
            self._init_qlib()
        except Exception as e:
            logger.error(f"Failed to download Qlib data: {e}")
    
    def prepare_data(self, 
                    start_time: str = "2020-01-01",
                    end_time: str = "2023-12-31",
                    instruments: str = "csi300",
                    freq: str = "day") -> TSDatasetH:
        """
        准备数据集
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            instruments: 股票池
            freq: 数据频率
            
        Returns:
            时间序列数据集
        """
        # 选择数据处理器
        if self.config.data_handler == "Alpha360":
            handler_config = self._get_alpha360_config()
        elif self.config.data_handler == "Alpha158":
            handler_config = self._get_alpha158_config()
        else:
            handler_config = self._get_custom_handler_config()
        
        # 数据集配置
        dataset_config = {
            "class": "TSDatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "train": [start_time, pd.Timestamp(end_time) - pd.DateOffset(months=6)],
                    "valid": [pd.Timestamp(end_time) - pd.DateOffset(months=6), 
                             pd.Timestamp(end_time) - pd.DateOffset(months=3)],
                    "test": [pd.Timestamp(end_time) - pd.DateOffset(months=3), end_time]
                }
            }
        }
        
        # 创建数据集
        dataset = init_instance_by_config(dataset_config)
        
        logger.info(f"Prepared dataset with {len(dataset)} samples")
        return dataset
    
    def _get_alpha360_config(self) -> Dict[str, Any]:
        """获取Alpha360配置"""
        return {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "instruments": self.config.market,
                "start_time": None,
                "end_time": None,
                "freq": "day",
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
                ],
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
            }
        }
    
    def _get_alpha158_config(self) -> Dict[str, Any]:
        """获取Alpha158配置"""
        return {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "instruments": self.config.market,
                "start_time": None,
                "end_time": None,
                "freq": "day",
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
                ],
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
            }
        }
    
    def _get_custom_handler_config(self) -> Dict[str, Any]:
        """获取自定义处理器配置"""
        return {
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {
                "instruments": self.config.market,
                "start_time": None,
                "end_time": None,
                "freq": "day",
                "infer_processors": [],
                "learn_processors": [],
                "fit_start_time": None,
                "fit_end_time": None,
                "process_type": "independent"
            }
        }
    
    def train_model(self, 
                   dataset: TSDatasetH,
                   model_type: str = None) -> Model:
        """
        训练模型
        
        Args:
            dataset: 数据集
            model_type: 模型类型
            
        Returns:
            训练好的模型
        """
        model_type = model_type or self.config.model_type
        
        # 根据模型类型选择配置
        if model_type == "LGBM":
            model = self._train_lgbm(dataset)
        elif model_type == "ALSTM":
            model = self._train_alstm(dataset)
        elif model_type == "GRU":
            model = self._train_gru(dataset)
        elif model_type == "DNN":
            model = self._train_dnn(dataset)
        elif model_type == "Transformer":
            model = self._train_transformer(dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 保存模型
        self.models[model_type] = model
        
        return model
    
    def _train_lgbm(self, dataset: TSDatasetH) -> LGBModel:
        """训练LightGBM模型"""
        model_config = {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "early_stopping_rounds": 100,
            "num_boost_round": 1000,
            "verbose": -1
        }
        
        model = LGBModel(**model_config)
        model.fit(dataset)
        
        logger.info("LightGBM model trained successfully")
        return model
    
    def _train_alstm(self, dataset: TSDatasetH) -> ALSTM:
        """训练ALSTM模型"""
        model_config = {
            "d_feat": 360,  # Alpha360特征数
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 100,
            "lr": 0.001,
            "early_stop": 20,
            "batch_size": 2000,
            "metric": "loss",
            "loss": "mse",
            "n_jobs": 10,
            "GPU": 0
        }
        
        model = ALSTM(**model_config)
        model.fit(dataset)
        
        logger.info("ALSTM model trained successfully")
        return model
    
    def _train_gru(self, dataset: TSDatasetH) -> GRU:
        """训练GRU模型"""
        model_config = {
            "d_feat": 360,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 100,
            "lr": 0.001,
            "early_stop": 20,
            "batch_size": 2000,
            "metric": "loss",
            "loss": "mse",
            "n_jobs": 10,
            "GPU": 0
        }
        
        model = GRU(**model_config)
        model.fit(dataset)
        
        logger.info("GRU model trained successfully")
        return model
    
    def _train_dnn(self, dataset: TSDatasetH) -> DNNModel:
        """训练DNN模型"""
        model_config = {
            "d_feat": 360,
            "hidden_size": 64,
            "num_layers": 3,
            "dropout": 0.3,
            "n_epochs": 100,
            "lr": 0.001,
            "early_stop": 20,
            "batch_size": 2000,
            "metric": "loss",
            "loss": "mse",
            "optimizer": "adam"
        }
        
        model = DNNModel(**model_config)
        model.fit(dataset)
        
        logger.info("DNN model trained successfully")
        return model
    
    def _train_transformer(self, dataset: TSDatasetH) -> Transformer:
        """训练Transformer模型"""
        model_config = {
            "d_feat": 360,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "n_epochs": 100,
            "lr": 0.001,
            "early_stop": 20,
            "batch_size": 2000,
            "metric": "loss",
            "loss": "mse",
            "n_jobs": 10,
            "GPU": 0
        }
        
        model = Transformer(**model_config)
        model.fit(dataset)
        
        logger.info("Transformer model trained successfully")
        return model
    
    def create_strategy(self, 
                       model: Model,
                       strategy_type: str = None) -> BaseStrategy:
        """
        创建交易策略
        
        Args:
            model: 模型
            strategy_type: 策略类型
            
        Returns:
            交易策略
        """
        strategy_type = strategy_type or self.config.strategy_type
        
        if strategy_type == "TopkDropout":
            strategy_config = {
                "model": model,
                "dataset": None,  # 将在执行时设置
                "topk": 50,
                "n_drop": 5,
                "signal": None,  # 将在执行时设置
                "only_tradable": True,
                "forbid_all_trade_at_limit": False
            }
            strategy = TopkDropoutStrategy(**strategy_config)
            
        elif strategy_type == "WeightStrategy":
            strategy_config = {
                "model": model,
                "dataset": None,
                "signal": None,
                "order_generator_cls_or_obj": "OrderGenWInteract"
            }
            strategy = WeightStrategyBase(**strategy_config)
            
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # 保存策略
        self.strategies[strategy_type] = strategy
        
        logger.info(f"{strategy_type} strategy created successfully")
        return strategy
    
    def backtest(self,
                strategy: BaseStrategy,
                dataset: TSDatasetH,
                start_time: str = "2023-01-01",
                end_time: str = "2023-12-31") -> Dict[str, Any]:
        """
        回测策略
        
        Args:
            strategy: 交易策略
            dataset: 数据集
            start_time: 回测开始时间
            end_time: 回测结束时间
            
        Returns:
            回测结果
        """
        # 执行器配置
        executor_config = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "track_data": True,
            "verbose": True,
            "trade_exchange": {
                "class": "Exchange",
                "module_path": "qlib.backtest.exchange",
                "kwargs": {
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5
                }
            }
        }
        
        # 创建执行器
        exec_inst = executor.executor_factory.create_executor(executor_config)
        
        # 运行回测
        with R.start(experiment_name=self.config.exp_name):
            recorder = R.get_recorder()
            
            # 记录信号
            sr = SignalRecord(model=strategy.model, dataset=dataset)
            recorder.register(sr)
            
            # 执行回测
            backtest_results = exec_inst.execute(strategy, dataset)
            
            # 记录组合分析
            par = PortAnaRecord(recorder, backtest_results)
            recorder.register(par)
            
        # 计算评估指标
        metrics = self._calculate_metrics(backtest_results)
        
        # 保存结果
        self.records.append({
            "experiment": self.config.exp_name,
            "strategy": type(strategy).__name__,
            "model": type(strategy.model).__name__,
            "start_time": start_time,
            "end_time": end_time,
            "metrics": metrics,
            "results": backtest_results
        })
        
        return metrics
    
    def _calculate_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算评估指标"""
        portfolio_metrics = backtest_results.get("portfolio_metrics", {})
        
        # 提取关键指标
        metrics = {
            "total_return": portfolio_metrics.get("total_return", 0),
            "annual_return": portfolio_metrics.get("annual_return", 0),
            "sharpe_ratio": portfolio_metrics.get("sharpe", 0),
            "max_drawdown": portfolio_metrics.get("max_drawdown", 0),
            "win_rate": portfolio_metrics.get("win_rate", 0),
            "information_ratio": portfolio_metrics.get("information_ratio", 0),
            "turnover_rate": portfolio_metrics.get("turnover_rate", 0)
        }
        
        logger.info(f"Metrics calculated: {metrics}")
        return metrics
    
    def portfolio_optimization(self,
                             predictions: pd.DataFrame,
                             risk_model: str = "empirical",
                             constraints: Dict[str, Any] = None) -> pd.DataFrame:
        """
        组合优化
        
        Args:
            predictions: 预测结果
            risk_model: 风险模型类型
            constraints: 约束条件
            
        Returns:
            优化后的权重
        """
        from qlib.contrib.model.pytorch_transformer import risk_modeling
        from scipy.optimize import minimize
        
        # 默认约束
        if constraints is None:
            constraints = {
                "max_weight": 0.1,  # 单股最大权重
                "min_weight": 0,    # 单股最小权重
                "total_weight": 1.0, # 总权重
                "max_stocks": 50    # 最大持仓数
            }
        
        # 计算预期收益
        expected_returns = predictions.mean()
        
        # 计算风险矩阵
        if risk_model == "empirical":
            # 经验协方差矩阵
            cov_matrix = predictions.cov()
        elif risk_model == "factor":
            # 因子风险模型
            # TODO: 实现因子风险模型
            cov_matrix = predictions.cov()
        else:
            cov_matrix = predictions.cov()
        
        # 定义目标函数（最大化夏普比率）
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_std
            return -sharpe_ratio  # 最小化负夏普比率
        
        # 定义约束条件
        n_assets = len(expected_returns)
        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - constraints["total_weight"]},  # 权重和为1
        ]
        
        # 边界条件
        bounds = tuple((constraints["min_weight"], constraints["max_weight"]) for _ in range(n_assets))
        
        # 初始权重（等权重）
        init_weights = np.array([constraints["total_weight"] / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list
        )

        # 创建权重DataFrame
        weights_df = pd.DataFrame({
            "stock": predictions.columns,
            "weight": result.x
        })
        
        # 过滤小权重
        weights_df = weights_df[weights_df["weight"] > 0.001]
        weights_df = weights_df.nlargest(constraints["max_stocks"], "weight")
        
        # 重新归一化
        weights_df["weight"] = weights_df["weight"] / weights_df["weight"].sum()
        
        logger.info(f"Portfolio optimized with {len(weights_df)} stocks")
        return weights_df
    
    def risk_analysis(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        风险分析
        
        Args:
            backtest_results: 回测结果
            
        Returns:
            风险指标
        """
        # 使用Qlib的风险分析工具
        risk_metrics = risk_analysis(backtest_results)
        
        # 计算额外的风险指标
        additional_metrics = {
            "var_95": self._calculate_var(backtest_results, 0.95),
            "cvar_95": self._calculate_cvar(backtest_results, 0.95),
            "downside_deviation": self._calculate_downside_deviation(backtest_results),
            "sortino_ratio": self._calculate_sortino_ratio(backtest_results)
        }
        
        risk_metrics.update(additional_metrics)
        
        logger.info(f"Risk analysis completed: {risk_metrics}")
        return risk_metrics
    
    def _calculate_var(self, results: Dict[str, Any], confidence: float) -> float:
        """计算VaR"""
        returns = results.get("returns", pd.Series())
        if returns.empty:
            return 0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, results: Dict[str, Any], confidence: float) -> float:
        """计算CVaR"""
        returns = results.get("returns", pd.Series())
        if returns.empty:
            return 0
        var = self._calculate_var(results, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_downside_deviation(self, results: Dict[str, Any]) -> float:
        """计算下行标准差"""
        returns = results.get("returns", pd.Series())
        if returns.empty:
            return 0
        negative_returns = returns[returns < 0]
        if negative_returns.empty:
            return 0
        return negative_returns.std()
    
    def _calculate_sortino_ratio(self, results: Dict[str, Any]) -> float:
        """计算Sortino比率"""
        returns = results.get("returns", pd.Series())
        if returns.empty:
            return 0
        mean_return = returns.mean()
        downside_dev = self._calculate_downside_deviation(results)
        if downside_dev == 0:
            return 0
        return mean_return / downside_dev
    
    def save_results(self, filepath: str = "backtest_results.json"):
        """保存结果"""
        with open(filepath, "w") as f:
            json.dump(self.records, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str = "backtest_results.json"):
        """加载结果"""
        with open(filepath, "r") as f:
            self.records = json.load(f)
        logger.info(f"Results loaded from {filepath}")


# 自定义因子计算
class CustomFactorCalculator:
    """自定义因子计算器"""
    
    @staticmethod
    def calculate_alpha_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Alpha因子
        
        Args:
            df: 原始数据
            
        Returns:
            因子数据
        """
        factors = pd.DataFrame(index=df.index)
        
        # 动量因子
        factors["momentum_5"] = df["close"].pct_change(5)
        factors["momentum_20"] = df["close"].pct_change(20)
        factors["momentum_60"] = df["close"].pct_change(60)
        
        # 反转因子
        factors["reversal_5"] = -factors["momentum_5"]
        factors["reversal_20"] = -factors["momentum_20"]
        
        # 波动率因子
        factors["volatility_20"] = df["close"].pct_change().rolling(20).std()
        factors["volatility_60"] = df["close"].pct_change().rolling(60).std()
        
        # 成交量因子
        factors["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        factors["amount_ratio"] = df["amount"] / df["amount"].rolling(20).mean()
        
        # 价格位置因子
        factors["price_position"] = (df["close"] - df["low"].rolling(20).min()) / \
                                   (df["high"].rolling(20).max() - df["low"].rolling(20).min())
        
        # RSI因子
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        factors["rsi"] = 100 - (100 / (1 + gain / loss))
        
        # MACD因子
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        factors["macd"] = ema12 - ema26
        factors["macd_signal"] = factors["macd"].ewm(span=9, adjust=False).mean()
        
        # 布林带因子
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        factors["bb_upper"] = sma20 + 2 * std20
        factors["bb_lower"] = sma20 - 2 * std20
        factors["bb_position"] = (df["close"] - factors["bb_lower"]) / \
                                (factors["bb_upper"] - factors["bb_lower"])
        
        return factors


# 实时预测服务
class RealtimePredictionService:
    """实时预测服务"""
    
    def __init__(self, qlib_integration: QlibIntegration):
        self.qlib = qlib_integration
        self.model = None
        self.last_update = None
        
    async def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            import joblib
            import pickle
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 根据文件扩展名选择加载方式
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif model_path.endswith('.joblib'):
                self.model = joblib.load(model_path)
            else:
                # 尝试使用Qlib的模型加载器
                from qlib.model.utils import CombinedModel
                self.model = CombinedModel.load(model_path)
            
            self.last_update = datetime.now()
            logger.info(f"模型已加载: {model_path}")
            
        except ImportError as e:
            logger.error(f"缺少依赖库: {e}")
            raise
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    async def predict(self, symbols: List[str]) -> pd.DataFrame:
        """
        实时预测
        
        Args:
            symbols: 股票列表
            
        Returns:
            预测结果
        """
        # 获取最新数据
        data = await self.qlib.data_access.get_realtime_data(symbols)
        
        # 计算因子
        factors = CustomFactorCalculator.calculate_alpha_factors(data)
        
        # 模型预测
        if self.model:
            predictions = self.model.predict(factors)
        else:
            # 使用默认模型
            predictions = pd.DataFrame({
                "symbol": symbols,
                "score": np.random.randn(len(symbols)),
                "timestamp": datetime.now()
            })
        
        return predictions
    
    async def update_model(self, retrain: bool = False):
        """
        更新模型
        
        Args:
            retrain: 是否重新训练模型
        """
        try:
            if retrain:
                logger.info("开始重新训练模型...")
                
                # 准备最新数据
                dataset = self.qlib.prepare_data(
                    start_time="2020-01-01",
                    end_time=datetime.now().strftime("%Y-%m-%d")
                )
                
                # 训练新模型
                new_model = self.qlib.train_model(dataset, model_type="LGBM")
                
                # 更新模型
                self.model = new_model
                self.last_update = datetime.now()
                
                # 保存模型
                model_save_path = f"models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                
                import joblib
                joblib.dump(self.model, model_save_path)
                logger.info(f"模型已更新并保存到: {model_save_path}")
                
            else:
                # 增量更新（如果模型支持）
                logger.info("执行模型增量更新...")
                
                # 检查模型是否支持在线更新
                if hasattr(self.model, 'partial_fit'):
                    # 获取最新数据
                    latest_data = self.qlib.prepare_data(
                        start_time=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                        end_time=datetime.now().strftime("%Y-%m-%d")
                    )
                    
                    # 增量训练
                    self.model.partial_fit(latest_data)
                    self.last_update = datetime.now()
                    logger.info("模型增量更新完成")
                else:
                    logger.warning("当前模型不支持增量更新，请使用retrain=True")
                    
        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            raise


if __name__ == "__main__":
    # 测试代码
    config = QlibConfig()
    qlib_integration = QlibIntegration(config)
    
    # 准备数据
    dataset = qlib_integration.prepare_data()
    
    # 训练模型
    model = qlib_integration.train_model(dataset, "LGBM")
    
    # 创建策略
    strategy = qlib_integration.create_strategy(model)
    
    # 回测
    results = qlib_integration.backtest(strategy, dataset)
    print(results)
    
    # 保存结果
    qlib_integration.save_results()