"""
XGBoost模型实现 - 用于Qlib Model Zoo
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost未安装，请运行: pip install xgboost")


class XGBModel:
    """XGBoost模型封装，与Qlib接口兼容"""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 min_child_weight: int = 1,
                 gamma: float = 0.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        初始化XGBoost模型
        
        Args:
            learning_rate: 学习率
            n_estimators: 树的数量
            max_depth: 树的最大深度
            subsample: 样本采样率
            colsample_bytree: 特征采样率
            min_child_weight: 最小子节点权重
            gamma: 分裂最小损失减少
            reg_alpha: L1正则化
            reg_lambda: L2正则化
            random_state: 随机种子
            n_jobs: 并行线程数
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装")
        
        self.params = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
        }
        self.params.update(kwargs)
        
        self.model = None
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, 
            train_data: Union[pd.DataFrame, tuple],
            valid_data: Optional[Union[pd.DataFrame, tuple]] = None,
            **kwargs):
        """
        训练模型
        
        Args:
            train_data: 训练数据 (X, y) 或 DataFrame
            valid_data: 验证数据 (X, y) 或 DataFrame
        """
        # 解析训练数据
        X_train, y_train = self._parse_data(train_data)
        
        # 准备训练参数
        fit_params = {}
        
        # 如果有验证集，添加early stopping
        if valid_data is not None:
            X_valid, y_valid = self._parse_data(valid_data)
            fit_params['eval_set'] = [(X_valid, y_valid)]
            fit_params['early_stopping_rounds'] = 20
            fit_params['verbose'] = False
        
        # 创建并训练模型
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names = list(X_train.columns)
        
        self.logger.info(f"开始训练XGBoost模型，训练样本: {len(X_train)}")
        self.model.fit(X_train, y_train, **fit_params)
        self.logger.info("XGBoost训练完成")
        
        return self
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """
        预测
        
        Args:
            data: 输入数据
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit()")
        
        # 解析数据
        if isinstance(data, pd.DataFrame):
            # 如果是DataFrame，提取特征列
            if 'LABEL0' in data.columns:
                X = data.drop(columns=['LABEL0'])
            else:
                X = data
            index = data.index
        else:
            X = data
            index = None
        
        # 预测
        predictions = self.model.predict(X)
        
        # 返回Series格式
        if index is not None:
            return pd.Series(predictions, index=index)
        else:
            return pd.Series(predictions)
    
    def _parse_data(self, data: Union[pd.DataFrame, tuple]):
        """解析数据格式"""
        if isinstance(data, tuple):
            X, y = data
            return X, y
        elif isinstance(data, pd.DataFrame):
            # 假设最后一列是标签或有LABEL0列
            if 'LABEL0' in data.columns:
                y = data['LABEL0']
                X = data.drop(columns=['LABEL0'])
            else:
                # 默认最后一列是标签
                y = data.iloc[:, -1]
                X = data.iloc[:, :-1]
            return X, y
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False)
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        self.model.save_model(path)
        self.logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(path)
        self.logger.info(f"模型已加载: {path}")
