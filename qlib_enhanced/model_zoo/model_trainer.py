"""
Model Zoo训练器 - 负责实际的模型训练流程
Phase 5.1实现
"""

import os
import sys
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

import pandas as pd
import numpy as np

# Qlib imports
try:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.utils import init_instance_by_config
except ImportError:
    logging.warning("Qlib未安装，部分功能将不可用")


class ModelZooTrainer:
    """Model Zoo训练器"""
    
    def __init__(self, 
                 qlib_provider_uri: str = "~/.qlib/qlib_data/cn_data",
                 output_dir: str = "./outputs/model_zoo"):
        """
        初始化训练器
        
        Args:
            qlib_provider_uri: Qlib数据提供者URI
            output_dir: 输出目录
        """
        self.qlib_provider_uri = qlib_provider_uri
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化Qlib
        self._init_qlib()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_qlib(self):
        """初始化Qlib环境"""
        try:
            qlib.init(provider_uri=self.qlib_provider_uri, region="cn")
            self.logger.info(f"Qlib初始化成功: {self.qlib_provider_uri}")
        except Exception as e:
            self.logger.warning(f"Qlib初始化失败: {e}")
    
    def prepare_dataset(self,
                       instruments: str = "csi300",
                       train_start: str = "2015-01-01",
                       train_end: str = "2020-12-31",
                       valid_start: str = "2021-01-01",
                       valid_end: str = "2021-12-31",
                       test_start: str = "2022-01-01",
                       test_end: str = "2022-12-31",
                       features: list = None) -> DatasetH:
        """
        准备数据集
        
        Args:
            instruments: 股票池
            train_start/end: 训练集时间范围
            valid_start/end: 验证集时间范围
            test_start/end: 测试集时间范围
            features: 特征列表
            
        Returns:
            DatasetH对象
        """
        # 默认特征配置
        if features is None:
            features = [
                "($close - Ref($close, 1)) / Ref($close, 1)",  # 收益率
                "($high - $low) / $open",  # 波动率
                "$volume / (Mean($volume, 5) + 1e-5)",  # 成交量比
                "Corr($close, Log($volume + 1), 5)",  # 价量相关性
                "Std(($close - Ref($close, 1)) / Ref($close, 1), 5)",  # 收益率标准差
            ]
        
        # 标签配置（预测未来1日收益）
        label = ["Ref($close, -1) / $close - 1"]
        
        # 数据处理配置
        data_handler_config = {
            "start_time": train_start,
            "end_time": test_end,
            "fit_start_time": train_start,
            "fit_end_time": train_end,
            "instruments": instruments,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
            "label": label,
        }
        
        # 数据集分段
        segments = {
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (test_start, test_end),
        }
        
        # 创建数据集
        dataset = DatasetH(
            handler={
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            segments=segments,
        )
        
        return dataset
    
    def train_model(self,
                   model_name: str,
                   model_config: Dict[str, Any],
                   dataset: DatasetH,
                   save_model: bool = True,
                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model_name: 模型名称
            model_config: 模型配置
            dataset: 数据集
            save_model: 是否保存模型
            progress_callback: 进度回调函数
            
        Returns:
            训练结果字典
        """
        try:
            self.logger.info(f"开始训练模型: {model_name}")
            
            # 回调进度
            if progress_callback:
                progress_callback(0.1, f"正在加载数据...")
            
            # 准备数据（Qlib 标准 DatasetH）
            df_train, df_valid = dataset.prepare(
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            
            if progress_callback:
                progress_callback(0.2, f"数据加载完成，训练集: {len(df_train)}, 验证集: {len(df_valid)}")
            
            # 加载模型实例
            model = self._create_model_instance(model_name, model_config)
            if progress_callback:
                progress_callback(0.3, f"模型实例创建成功: {model_name}")
            
            # 判定是否为 Qlib 原生模型
            try:
                from qlib.model.base import Model as QlibBaseModel  # type: ignore
            except Exception:
                QlibBaseModel = None  # noqa: N806
            is_qlib_model = (
                (QlibBaseModel is not None and isinstance(model, QlibBaseModel))
                or model.__class__.__module__.startswith("qlib.contrib")
            )
            
            # 工具：提取特征与标签
            def _extract_xy(df: pd.DataFrame):
                X = None
                y = None
                if isinstance(df.columns, pd.MultiIndex):
                    if "feature" in df.columns.get_level_values(0):
                        X = df["feature"]
                    if "label" in df.columns.get_level_values(0):
                        y_df = df["label"]
                        if isinstance(y_df, pd.DataFrame):
                            if "LABEL0" in getattr(y_df, 'columns', []):
                                y = y_df["LABEL0"]
                            else:
                                y = y_df.iloc[:, 0]
                        else:
                            y = y_df
                if X is None:
                    # 回退：去除可能的标签列
                    drop_cols = [c for c in df.columns if str(c).upper().startswith("LABEL") or c == "label"]
                    X = df.drop(columns=drop_cols, errors="ignore")
                if y is None:
                    if "LABEL0" in df.columns:
                        y = df["LABEL0"]
                    elif "label" in df.columns:
                        y = df["label"]
                    else:
                        y = df.iloc[:, -1]
                return X, pd.Series(y).astype(float)
            
            # 训练
            self.logger.info("开始训练...")
            if hasattr(model, "fit"):
                if progress_callback:
                    progress_callback(0.4, "正在训练模型...")
                if is_qlib_model:
                    # Qlib 原生模型：直接使用 DatasetH
                    model.fit(dataset)
                else:
                    # 自研模型：使用 (X, y) / (X_valid, y_valid)
                    X_train, y_train = _extract_xy(df_train)
                    X_valid, y_valid = _extract_xy(df_valid)
                    model.fit((X_train, y_train), valid_data=(X_valid, y_valid))
                if progress_callback:
                    progress_callback(0.7, "训练完成，正在评估...")
            
            # 验证集预测
            if is_qlib_model:
                pred_valid = model.predict(dataset, segment="valid")
            else:
                X_valid, _ = _extract_xy(df_valid)
                pred_valid = model.predict(X_valid)
            
            # 计算评估指标（逐日横截面口径，与 Qlib 对齐）
            metrics = self._calculate_metrics_cs(df_valid, pred_valid)
            if progress_callback:
                progress_callback(0.9, f"评估完成: IC={metrics.get('IC', 0):.4f}")
            
            # 保存模型
            model_path = None
            if save_model:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"{model_name}_{timestamp}.pkl"
                model_path = self.output_dir / model_filename
                try:
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                    self.logger.info(f"模型已保存: {model_path}")
                except Exception as _e:
                    self.logger.warning(f"模型持久化失败，已跳过: {_e}")
                    model_path = None
                
                # 保存元数据
                metadata = {
                    "model_name": model_name,
                    "config": model_config,
                    "metrics": metrics,
                    "timestamp": timestamp,
                    "train_samples": int(len(df_train)),
                    "valid_samples": int(len(df_valid)),
                }
                metadata_path = self.output_dir / f"{model_name}_{timestamp}_meta.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            if progress_callback:
                progress_callback(1.0, "训练完成!")
            
            # 返回结果
            result = {
                "success": True,
                "model_name": model_name,
                "model_path": str(model_path) if model_path else None,
                "metrics": metrics,
                "train_samples": int(len(df_train)),
                "valid_samples": int(len(df_valid)),
            }
            self.logger.info(f"模型训练完成: {model_name}, IC={metrics.get('IC', 0):.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}", exc_info=True)
            if progress_callback:
                progress_callback(0, f"训练失败: {str(e)}")
            
            return {
                "success": False,
                "model_name": model_name,
                "error": str(e),
            }
    
    def _create_model_instance(self, model_name: str, model_config: Dict[str, Any]):
        """创建模型实例"""
        # 导入模型注册表
        from .model_registry import MODEL_REGISTRY
        
        model_info = MODEL_REGISTRY.get(model_name)
        if not model_info:
            raise ValueError(f"未知模型: {model_name}")
        
        # 根据不同的模型类型创建实例
        try:
            if model_name == "LightGBM":
                # LightGBM模型 (已有实现)
                model = LGBModel(**model_config)
            
            elif model_name == "XGBoost":
                # XGBoost模型
                from .models.xgboost_model import XGBModel
                model = XGBModel(**model_config)
                self.logger.info(f"已加载XGBoost模型")
            
            elif model_name == "CatBoost":
                # CatBoost模型
                from .models.catboost_model import CatBoostModel
                model = CatBoostModel(**model_config)
                self.logger.info(f"已加载CatBoost模型")
            
            elif model_name == "MLP":
                # MLP模型
                from .models.pytorch_models import MLPModel
                model = MLPModel(**model_config)
                self.logger.info(f"已加载MLP模型")
            
            elif model_name == "LSTM":
                # LSTM模型
                from .models.pytorch_models import LSTMModel
                model = LSTMModel(**model_config)
                self.logger.info(f"已加载LSTM模型")
            
            elif model_name == "GRU":
                # GRU模型
                from .models.pytorch_models import GRUModel
                model = GRUModel(**model_config)
                self.logger.info(f"已加载GRU模型")
            
            elif model_name == "ALSTM":
                # ALSTM模型
                from .models.pytorch_models import ALSTMModel
                model = ALSTMModel(**model_config)
                self.logger.info(f"已加载ALSTM模型")
            
            elif model_name == "Transformer":
                # Transformer模型
                from .models.pytorch_models import TransformerModel
                model = TransformerModel(**model_config)
                self.logger.info(f"已加载Transformer模型")
            
            elif model_name == "TCN":
                # TCN模型
                from .models.pytorch_models import TCNModel
                # 解析num_channels参数
                if 'num_channels' in model_config and isinstance(model_config['num_channels'], str):
                    import ast
                    model_config['num_channels'] = ast.literal_eval(model_config['num_channels'])
                model = TCNModel(**model_config)
                self.logger.info(f"已加载TCN模型")
            
            elif model_name in ["TRA", "HIST"]:
                # TRA和HIST模型 - 暂时使用Transformer代替
                from .models.pytorch_models import TransformerModel
                self.logger.warning(f"{model_name}暂时使用Transformer代替")
                model = TransformerModel(**model_config)
            
            elif model_name == "DoubleEnsemble":
                # DoubleEnsemble集成模型 - 暂时使用LightGBM代替
                self.logger.warning(f"{model_name}暂未完全实现，使用LightGBM代替")
                model = LGBModel(**model_config)
            
            else:
                raise ValueError(f"不支持的模型: {model_name}")
            
            return model
            
        except ImportError as e:
            # 如果依赖包未安装，使用LightGBM作为fallback
            self.logger.warning(f"{model_name}依赖未安装({e})，使用LightGBM代替")
            return LGBModel(**model_config)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标（旧版，基于整体样本，保留作回退）
        """
        # 处理NaN值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {"IC": 0.0, "ICIR": 0.0, "Rank IC": 0.0}
        
        # IC (Pearson)
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Rank IC（Spearman）
        from scipy.stats import spearmanr
        rank_ic, _ = spearmanr(y_true, y_pred)
        
        # 非标准 ICIR（仅回退用途）
        icir = ic / (np.std(y_pred) + 1e-5)
        
        # MSE/MAE
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            "IC": float(ic) if not np.isnan(ic) else 0.0,
            "ICIR": float(icir) if not np.isnan(icir) else 0.0,
            "Rank IC": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            "MSE": float(mse),
            "MAE": float(mae),
        }

    def _calculate_metrics_cs(self, df_valid: pd.DataFrame, pred_valid: pd.Series) -> Dict[str, float]:
        """
        与 Qlib 对齐的评估：按日期横截面计算 IC 序列，并统计 IC/Rank IC/ICIR。
        支持 df_valid 为 MultiIndex 列（'feature'/'label' 分组）。
        """
        # 提取标签序列（与 df_valid 结构兼容）
        y = None
        if isinstance(df_valid.columns, pd.MultiIndex) and "label" in df_valid.columns.get_level_values(0):
            y_df = df_valid["label"]
            if isinstance(y_df, pd.DataFrame):
                if "LABEL0" in getattr(y_df, 'columns', []):
                    y = y_df["LABEL0"]
                else:
                    y = y_df.iloc[:, 0]
            else:
                y = y_df
        else:
            if "LABEL0" in df_valid.columns:
                y = df_valid["LABEL0"]
            elif "label" in df_valid.columns:
                y = df_valid["label"]
            else:
                # 回退：最后一列
                y = df_valid.iloc[:, -1]
        
        # 对齐索引
        aligned = pd.concat([
            pd.Series(pred_valid, name="pred"),
            pd.Series(y, name="label")
        ], axis=1).dropna()
        if aligned.empty:
            return {"IC": 0.0, "ICIR": 0.0, "Rank IC": 0.0, "MSE": 0.0, "MAE": 0.0}
        
        # 分日计算 IC（Pearson）与 Rank IC（Spearman）
        from scipy.stats import spearmanr, pearsonr
        try:
            dates = aligned.index.get_level_values(0)
        except Exception:
            # 若索引不是 MultiIndex，则按整体计算
            dates = pd.to_datetime(aligned.index)
        by_date = aligned.groupby(dates)
        ic_series = by_date.apply(lambda x: pearsonr(x["pred"], x["label"])[0] if len(x) > 1 else np.nan).dropna()
        rank_ic_series = by_date.apply(lambda x: spearmanr(x["pred"], x["label"]).correlation if len(x) > 1 else np.nan).dropna()
        
        ic = float(ic_series.mean()) if len(ic_series) else 0.0
        rank_ic = float(rank_ic_series.mean()) if len(rank_ic_series) else 0.0
        icir = float(ic_series.mean() / (ic_series.std(ddof=1) + 1e-12)) if len(ic_series) > 1 else 0.0
        
        # 整体 MSE/MAE（非分日）
        mse = float(np.mean((aligned["label"].values - aligned["pred"].values) ** 2))
        mae = float(np.mean(np.abs(aligned["label"].values - aligned["pred"].values)))
        
        return {"IC": ic, "Rank IC": rank_ic, "ICIR": icir, "MSE": mse, "MAE": mae}
    
    def compare_models(self, results: list) -> pd.DataFrame:
        """
        比较多个模型的结果
        
        Args:
            results: 训练结果列表
            
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        for result in results:
            if result.get("success"):
                comparison_data.append({
                    "模型": result["model_name"],
                    "IC": result["metrics"].get("IC", 0),
                    "Rank IC": result["metrics"].get("Rank IC", 0),
                    "ICIR": result["metrics"].get("ICIR", 0),
                    "MSE": result["metrics"].get("MSE", 0),
                    "MAE": result["metrics"].get("MAE", 0),
                    "训练样本": result["train_samples"],
                    "验证样本": result["valid_samples"],
                })
        
        df = pd.DataFrame(comparison_data)
        
        # 按IC排序
        if len(df) > 0:
            df = df.sort_values("IC", ascending=False)
        
        return df
