"""
涨停板集成学习模型 - 多模型Stacking

集成以下模型：
1. XGBoost - 梯度提升树
2. LightGBM - 轻量级梯度提升
3. CatBoost - 类别特征增强
4. GRU - 循环神经网络（Qlib）

使用Stacking策略融合预测结果，提升准确率
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
import logging
import os
import pickle

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 尝试导入各种模型库
MODELS_AVAILABLE = {}

try:
    import xgboost as xgb
    MODELS_AVAILABLE['xgboost'] = True
except ImportError:
    MODELS_AVAILABLE['xgboost'] = False
    logger.warning("XGBoost 未安装，相关基学习器将被跳过")

try:
    import lightgbm as lgb
    MODELS_AVAILABLE['lightgbm'] = True
except ImportError:
    MODELS_AVAILABLE['lightgbm'] = False
    logger.warning("LightGBM 未安装，相关基学习器将被跳过")

try:
    import catboost as cb
    MODELS_AVAILABLE['catboost'] = True
except ImportError:
    MODELS_AVAILABLE['catboost'] = False
    logger.warning("CatBoost 未安装，相关基学习器将被跳过")


class LimitUpEnsembleModel:
    """涨停板集成学习模型"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化集成模型
        
        Parameters:
        -----------
        config : Dict, optional
            配置参数
        """
        self.config = config or {}
        
        # 基础模型
        self.base_models = {}
        self.meta_model = None
        
        # 模型权重（自适应）
        self.model_weights = {}
        
        # 初始化可用模型
        self._init_models()
    
    def _init_models(self):
        """初始化所有可用模型"""
        
        # 1. XGBoost
        if MODELS_AVAILABLE['xgboost']:
            self.base_models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        # 2. LightGBM  
        if MODELS_AVAILABLE['lightgbm']:
            self.base_models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                random_state=42
            )
        
        # 3. CatBoost
        if MODELS_AVAILABLE['catboost']:
            self.base_models['cat'] = cb.CatBoostClassifier(
                iterations=100,
                depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        
        # 如果没有任何模型，使用简单分类器
        if not self.base_models:
            logger.warning("没有可用的 ML 库，使用简单规则分类器(SimpleClassifier)")
            self.base_models['simple'] = SimpleClassifier()
        
        logger.info(f"初始化 {len(self.base_models)} 个基础模型: {list(self.base_models.keys())}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        训练集成模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练标签
        X_val : pd.DataFrame, optional
            验证集特征
        y_val : pd.Series, optional
            验证集标签
        """
        logger.info("训练集成模型...")
        logger.info(f"训练集: {len(X_train)} 样本")
        if X_val is not None:
            logger.info(f"验证集: {len(X_val)} 样本")
        
        # 训练所有基础模型
        base_predictions_train = {}
        base_predictions_val = {}
        
        for name, model in self.base_models.items():
            logger.info(f"开始训练基模型 {name} ...")
            try:
                model.fit(X_train, y_train)
                base_predictions_train[name] = model.predict_proba(X_train)[:, 1]
                if X_val is not None:
                    base_predictions_val[name] = model.predict_proba(X_val)[:, 1]
                    val_pred = model.predict(X_val)
                    val_acc = (val_pred == y_val).mean()
                    logger.info(f"{name} 验证集准确率: {val_acc:.2%}")
                logger.info(f"{name} 训练完成")
            except Exception as e:
                logger.exception(f"基模型 {name} 训练失败: {e}")
                del self.base_models[name]
        
        # 训练meta模型（使用基础模型的预测作为特征）
        if len(self.base_models) > 0:
            logger.info("训练元模型（Stacking 层）...")
            meta_X_train = pd.DataFrame(base_predictions_train)
            self.meta_model = SimpleMetaModel()
            self.meta_model.fit(meta_X_train, y_train)
            if X_val is not None:
                meta_X_val = pd.DataFrame(base_predictions_val)
                meta_pred = self.meta_model.predict(meta_X_val)
                meta_acc = (meta_pred == y_val).mean()
                logger.info(f"元模型验证集准确率: {meta_acc:.2%}")
            logger.info("元模型训练完成")
        logger.info("集成模型训练完成")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        
        # 获取所有基础模型的预测
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                base_predictions[name] = model.predict_proba(X)[:, 1]
            except:
                pass
        
        if not base_predictions:
            # 如果没有基础模型，返回默认值
            return np.column_stack([
                np.ones(len(X)) * 0.5,
                np.ones(len(X)) * 0.5
            ])
        
        # 使用meta模型融合
        if self.meta_model:
            meta_X = pd.DataFrame(base_predictions)
            final_proba = self.meta_model.predict_proba(meta_X)
            return np.column_stack([1 - final_proba, final_proba])
        else:
            # 简单平均
            avg_proba = np.mean(list(base_predictions.values()), axis=0)
            return np.column_stack([1 - avg_proba, avg_proba])
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """评估模型性能"""
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # 计算指标
        accuracy = (y_pred == y).mean()
        
        # 计算F1
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save(self, file_path: str) -> None:
        """保存模型到文件。

        优先使用 joblib（如可用），否则回退到 pickle。
        仅保存必要状态：config、base_models、meta_model。
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
        except Exception:
            pass

        state = {
            'version': 1,
            'config': self.config,
            'base_models': self.base_models,
            'meta_model': self.meta_model,
        }

        # 尝试使用 joblib
        try:
            import joblib  # type: ignore
            joblib.dump(state, file_path)
            logger.info(f"模型已保存: {file_path} (joblib)")
            return
        except Exception as e:
            logger.debug(f"joblib 保存失败，改用 pickle: {e}")

        # 回退到 pickle
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"模型已保存: {file_path} (pickle)")
        except Exception as e:
            logger.exception(f"模型保存失败: {e}")
            raise

    @classmethod
    def load(cls, file_path: str) -> "LimitUpEnsembleModel":
        """从文件加载模型并返回实例。"""
        state = None

        # 尝试使用 joblib 加载
        try:
            import joblib  # type: ignore
            state = joblib.load(file_path)
            logger.info(f"模型已加载: {file_path} (joblib)")
        except Exception as e:
            logger.debug(f"joblib 加载失败，改用 pickle: {e}")
            try:
                with open(file_path, 'rb') as f:
                    state = pickle.load(f)
                logger.info(f"模型已加载: {file_path} (pickle)")
            except Exception as e2:
                logger.exception(f"模型加载失败: {e2}")
                raise

        # 还原实例
        config = state.get('config', {}) if isinstance(state, dict) else {}
        model = cls(config=config)
        if isinstance(state, dict):
            model.base_models = state.get('base_models', {})
            model.meta_model = state.get('meta_model')
        return model


class SimpleClassifier:
    """简单规则分类器（当没有ML库时使用）"""
    
    def __init__(self):
        self.threshold = {}
    
    def fit(self, X, y):
        """学习每个特征的最佳阈值"""
        for col in X.columns:
            # 计算相关系数
            corr = X[col].corr(y)
            # 使用中位数作为阈值
            self.threshold[col] = X[col].median()
    
    def predict_proba(self, X):
        """预测概率"""
        scores = []
        for col in X.columns:
            if col in self.threshold:
                score = (X[col] > self.threshold[col]).astype(float)
                scores.append(score)
        
        if scores:
            avg_score = np.mean(scores, axis=0)
            return np.column_stack([1 - avg_score, avg_score])
        else:
            return np.column_stack([
                np.ones(len(X)) * 0.5,
                np.ones(len(X)) * 0.5
            ])
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class SimpleMetaModel:
    """简单元模型（加权平均）"""
    
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        """学习最佳权重"""
        # 计算每个模型的F1分数作为权重
        weights = []
        for col in X.columns:
            pred = (X[col] > 0.5).astype(int)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            weights.append(f1)
        
        # 归一化权重
        total = sum(weights)
        if total > 0:
            self.weights = {col: w / total for col, w in zip(X.columns, weights)}
        else:
            self.weights = {col: 1 / len(X.columns) for col in X.columns}
    
    def predict_proba(self, X):
        """预测概率"""
        if self.weights is None:
            return np.mean(X.values, axis=1)
        
        weighted_sum = np.zeros(len(X))
        for col, weight in self.weights.items():
            if col in X.columns:
                weighted_sum += X[col].values * weight
        
        return weighted_sum
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)


# ==================== 使用示例 ====================

def main():
    """示例：训练和测试集成模型"""
    logger.info("=" * 80)
    logger.info("涨停板集成学习模型 - 测试")
    logger.info("=" * 80)
    
    # 1. 生成模拟数据
    logger.info("\n📊 生成模拟数据...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    # 生成特征
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 生成目标（基于简单规则）
    y = ((X['feature_0'] > 0) & 
         (X['feature_1'] > 0) & 
         (X['feature_2'] > 0.5)).astype(int)
    
    logger.info(f"   样本数: {n_samples}")
    logger.info(f"   特征数: {n_features}")
    logger.info(f"   正样本率: {y.mean():.1%}")
    
    # 2. 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"   训练集: {len(X_train)} 样本")
    logger.info(f"   测试集: {len(X_test)} 样本")
    
    # 3. 训练集成模型
    model = LimitUpEnsembleModel()
    model.fit(X_train, y_train, X_test, y_test)
    
    # 4. 评估模型
    logger.info("\n" + "=" * 80)
    logger.info("📊 模型评估")
    logger.info("=" * 80)
    
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    logger.info("\n训练集:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\n测试集:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # 5. 预测示例
    logger.info("\n" + "=" * 80)
    logger.info("🎯 预测示例（前10个样本）")
    logger.info("=" * 80)
    
    sample_X = X_test.head(10)
    sample_y = y_test.head(10).values
    predictions = model.predict(sample_X)
    probabilities = model.predict_proba(sample_X)[:, 1]
    
    logger.info("\n样本  真实  预测  概率")
    logger.info("-" * 40)
    for i in range(len(sample_X)):
        logger.info(f"{i+1:4d}  {sample_y[i]:4d}  {predictions[i]:4d}  {probabilities[i]:.2%}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 测试完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    from app.core.logging_setup import setup_logging
    setup_logging()
    # 检查sklearn是否可用
    try:
        from sklearn.model_selection import train_test_split
        main()
    except ImportError:
        logger.warning("⚠️  sklearn未安装，无法运行完整测试")
        logger.info("   请安装: pip install scikit-learn")
        
        # 运行简化版本
        logger.info("\n运行简化测试...")
        model = LimitUpEnsembleModel()
        logger.info(f"\n可用模型: {list(model.base_models.keys())}")
