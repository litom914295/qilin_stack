"""
CatBoost模型实现 - 用于Qlib Model Zoo
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost未安装，请运行: pip install catboost")


class CatBoostModel:
    """CatBoost模型封装，与Qlib接口兼容"""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 iterations: int = 100,
                 depth: int = 6,
                 l2_leaf_reg: float = 3.0,
                 random_state: int = 42,
                 task_type: str = 'CPU',
                 verbose: bool = False,
                 **kwargs):
        """
        初始化CatBoost模型
        
        Args:
            learning_rate: 学习率
            iterations: 迭代次数
            depth: 树的深度
            l2_leaf_reg: L2正则化系数
            random_state: 随机种子
            task_type: 'CPU' 或 'GPU'
            verbose: 是否显示训练日志
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost未安装")
        
        self.params = {
            'learning_rate': learning_rate,
            'iterations': iterations,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
            'random_state': random_state,
            'task_type': task_type,
            'verbose': verbose,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
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
        
        # 创建并训练模型
        self.model = CatBoostRegressor(**self.params)
        self.feature_names = list(X_train.columns)
        
        self.logger.info(f"开始训练CatBoost模型，训练样本: {len(X_train)}")
        self.model.fit(X_train, y_train, **fit_params)
        self.logger.info("CatBoost训练完成")
        
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
            if 'LABEL0' in data.columns:
                y = data['LABEL0']
                X = data.drop(columns=['LABEL0'])
            else:
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
