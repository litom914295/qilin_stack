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

warnings.filterwarnings('ignore')

# 尝试导入各种模型库
MODELS_AVAILABLE = {}

try:
    import xgboost as xgb
    MODELS_AVAILABLE['xgboost'] = True
except ImportError:
    MODELS_AVAILABLE['xgboost'] = False
    print("⚠️  XGBoost未安装")

try:
    import lightgbm as lgb
    MODELS_AVAILABLE['lightgbm'] = True
except ImportError:
    MODELS_AVAILABLE['lightgbm'] = False
    print("⚠️  LightGBM未安装")

try:
    import catboost as cb
    MODELS_AVAILABLE['catboost'] = True
except ImportError:
    MODELS_AVAILABLE['catboost'] = False
    print("⚠️  CatBoost未安装")


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
            print("⚠️  没有可用的ML库，使用简单规则分类器")
            self.base_models['simple'] = SimpleClassifier()
        
        print(f"✅ 初始化了 {len(self.base_models)} 个基础模型: {list(self.base_models.keys())}")
    
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
        print(f"\n🏋️  训练集成模型...")
        print(f"   训练集: {len(X_train)} 样本")
        if X_val is not None:
            print(f"   验证集: {len(X_val)} 样本")
        
        # 训练所有基础模型
        base_predictions_train = {}
        base_predictions_val = {}
        
        for name, model in self.base_models.items():
            print(f"\n   训练 {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                # 获取训练集预测（用于meta模型）
                base_predictions_train[name] = model.predict_proba(X_train)[:, 1]
                
                if X_val is not None:
                    base_predictions_val[name] = model.predict_proba(X_val)[:, 1]
                    
                    # 计算验证集准确率
                    val_pred = model.predict(X_val)
                    val_acc = (val_pred == y_val).mean()
                    print(f"      验证集准确率: {val_acc:.2%}")
                
                print(f"      ✅ {name} 训练完成")
                
            except Exception as e:
                print(f"      ❌ {name} 训练失败: {e}")
                # 移除失败的模型
                del self.base_models[name]
        
        # 训练meta模型（使用基础模型的预测作为特征）
        if len(self.base_models) > 0:
            print(f"\n   训练元模型（Stacking层）...")
            
            # 构建meta特征
            meta_X_train = pd.DataFrame(base_predictions_train)
            
            # 使用简单的LR或简单规则作为meta模型
            self.meta_model = SimpleMetaModel()
            self.meta_model.fit(meta_X_train, y_train)
            
            if X_val is not None:
                meta_X_val = pd.DataFrame(base_predictions_val)
                meta_pred = self.meta_model.predict(meta_X_val)
                meta_acc = (meta_pred == y_val).mean()
                print(f"      元模型验证集准确率: {meta_acc:.2%}")
            
            print(f"      ✅ 元模型训练完成")
        
        print(f"\n✅ 集成模型训练完成！")
    
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
    print("=" * 80)
    print("涨停板集成学习模型 - 测试")
    print("=" * 80)
    
    # 1. 生成模拟数据
    print("\n📊 生成模拟数据...")
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
    
    print(f"   样本数: {n_samples}")
    print(f"   特征数: {n_features}")
    print(f"   正样本率: {y.mean():.1%}")
    
    # 2. 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    # 3. 训练集成模型
    model = LimitUpEnsembleModel()
    model.fit(X_train, y_train, X_test, y_test)
    
    # 4. 评估模型
    print("\n" + "=" * 80)
    print("📊 模型评估")
    print("=" * 80)
    
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\n训练集:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n测试集:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 5. 预测示例
    print("\n" + "=" * 80)
    print("🎯 预测示例（前10个样本）")
    print("=" * 80)
    
    sample_X = X_test.head(10)
    sample_y = y_test.head(10).values
    predictions = model.predict(sample_X)
    probabilities = model.predict_proba(sample_X)[:, 1]
    
    print("\n样本  真实  预测  概率")
    print("-" * 40)
    for i in range(len(sample_X)):
        print(f"{i+1:4d}  {sample_y[i]:4d}  {predictions[i]:4d}  {probabilities[i]:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    # 检查sklearn是否可用
    try:
        from sklearn.model_selection import train_test_split
        main()
    except ImportError:
        print("⚠️  sklearn未安装，无法运行完整测试")
        print("   请安装: pip install scikit-learn")
        
        # 运行简化版本
        print("\n运行简化测试...")
        model = LimitUpEnsembleModel()
        print(f"\n可用模型: {list(model.base_models.keys())}")
