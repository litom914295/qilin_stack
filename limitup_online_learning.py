"""
涨停板预测系统 - 在线学习优化模块
支持增量学习和模型自适应更新
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import pickle
from collections import deque

from sklearn.metrics import f1_score, precision_score, recall_score
import lightgbm as lgb


class OnlineLearningModel:
    """在线学习模型（增量更新）"""
    
    def __init__(
        self,
        window_size: int = 1000,
        update_threshold: float = 0.05,
        min_samples: int = 100,
        save_dir: str = './online_models'
    ):
        """
        初始化在线学习模型
        
        Args:
            window_size: 滑动窗口大小（用于保留最近数据）
            update_threshold: 性能下降阈值（触发重训练）
            min_samples: 最小样本数（触发更新）
            save_dir: 模型保存目录
        """
        self.window_size = window_size
        self.update_threshold = update_threshold
        self.min_samples = min_samples
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 基础模型
        self.model = None
        self.base_score = 0.0
        
        # 增量数据缓存
        self.X_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
        
        # 性能监控
        self.performance_history = []
        self.update_count = 0
        
        print(f"🔄 在线学习模型初始化")
        print(f"   滑动窗口: {window_size}")
        print(f"   更新阈值: {update_threshold}")
        print(f"   最小样本: {min_samples}")
    
    def initial_train(self, X: pd.DataFrame, y: pd.Series, **model_params):
        """初始训练（冷启动）"""
        print(f"\n🚀 开始初始训练...")
        print(f"训练集大小: {X.shape}")
        
        # 创建LightGBM模型（支持增量学习）
        self.model = lgb.LGBMClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=42,
            verbose=-1
        )
        
        # 训练
        self.model.fit(X, y)
        
        # 评估基线性能
        y_pred = self.model.predict(X)
        self.base_score = f1_score(y, y_pred, average='weighted')
        
        print(f"✅ 初始训练完成")
        print(f"   基线F1分数: {self.base_score:.4f}")
        
        # 记录性能
        self._record_performance(self.base_score, 'initial')
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict_proba(X)
    
    def update(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        增量更新模型
        
        Args:
            X_new: 新数据特征
            y_new: 新数据标签
            force: 强制更新（忽略阈值检查）
            
        Returns:
            更新统计信息
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先调用initial_train()")
        
        # 添加到缓冲区
        for i in range(len(X_new)):
            self.X_buffer.append(X_new.iloc[i])
            self.y_buffer.append(y_new.iloc[i])
        
        buffer_size = len(self.X_buffer)
        print(f"\n📊 缓冲区大小: {buffer_size}/{self.window_size}")
        
        # 检查是否需要更新
        should_update = force or buffer_size >= self.min_samples
        
        if not should_update:
            return {
                'updated': False,
                'reason': f'样本不足 ({buffer_size}/{self.min_samples})'
            }
        
        # 评估当前性能
        X_buffer_df = pd.DataFrame(list(self.X_buffer))
        y_buffer_series = pd.Series(list(self.y_buffer))
        
        y_pred = self.model.predict(X_buffer_df)
        current_score = f1_score(y_buffer_series, y_pred, average='weighted')
        
        performance_drop = self.base_score - current_score
        
        print(f"当前F1分数: {current_score:.4f}")
        print(f"性能下降: {performance_drop:.4f}")
        
        # 检查是否需要重训练
        need_retrain = force or performance_drop > self.update_threshold
        
        if not need_retrain:
            return {
                'updated': False,
                'reason': f'性能下降不足 ({performance_drop:.4f} < {self.update_threshold})',
                'current_score': current_score
            }
        
        # 执行增量训练
        print(f"\n🔄 开始增量训练...")
        
        # 使用缓冲区数据重训练
        self.model.fit(
            X_buffer_df,
            y_buffer_series,
            init_model=self.model  # LightGBM支持增量训练
        )
        
        # 重新评估
        y_pred_new = self.model.predict(X_buffer_df)
        new_score = f1_score(y_buffer_series, y_pred_new, average='weighted')
        
        # 更新基线
        self.base_score = new_score
        self.update_count += 1
        
        print(f"✅ 增量训练完成")
        print(f"   新F1分数: {new_score:.4f}")
        print(f"   总更新次数: {self.update_count}")
        
        # 记录性能
        self._record_performance(new_score, 'incremental')
        
        # 自动保存
        self.save(f'online_model_v{self.update_count}.pkl')
        
        return {
            'updated': True,
            'old_score': current_score,
            'new_score': new_score,
            'improvement': new_score - current_score,
            'update_count': self.update_count
        }
    
    def _record_performance(self, score: float, update_type: str):
        """记录性能历史"""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'update_type': update_type,
            'update_count': self.update_count
        })
    
    def get_performance_history(self) -> pd.DataFrame:
        """获取性能历史"""
        return pd.DataFrame(self.performance_history)
    
    def save(self, filename: str):
        """保存模型"""
        filepath = self.save_dir / filename
        
        # 保存模型
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'base_score': self.base_score,
                'update_count': self.update_count,
                'performance_history': self.performance_history
            }, f)
        
        print(f"💾 模型已保存: {filepath}")
    
    def load(self, filename: str):
        """加载模型"""
        filepath = self.save_dir / filename
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.base_score = data['base_score']
        self.update_count = data['update_count']
        self.performance_history = data['performance_history']
        
        print(f"✅ 模型已加载: {filepath}")
        print(f"   F1分数: {self.base_score:.4f}")
        print(f"   更新次数: {self.update_count}")


class AdaptiveLearningPipeline:
    """自适应学习Pipeline（完整工作流）"""
    
    def __init__(
        self,
        window_size: int = 1000,
        update_interval: int = 100,
        update_threshold: float = 0.05,
        save_dir: str = './adaptive_models'
    ):
        """
        初始化自适应学习Pipeline
        
        Args:
            window_size: 滑动窗口大小
            update_interval: 更新间隔（样本数）
            update_threshold: 性能阈值
            save_dir: 保存目录
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.update_threshold = update_threshold
        self.save_dir = save_dir
        
        self.model = OnlineLearningModel(
            window_size=window_size,
            update_threshold=update_threshold,
            min_samples=update_interval,
            save_dir=save_dir
        )
        
        self.samples_since_update = 0
        
        print(f"\n{'='*60}")
        print(f"🎯 自适应学习Pipeline初始化")
        print(f"{'='*60}")
        print(f"滑动窗口: {window_size}")
        print(f"更新间隔: {update_interval}")
        print(f"{'='*60}\n")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """初始训练"""
        self.model.initial_train(X, y)
    
    def predict_and_learn(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        预测并学习（在线模式）
        
        Args:
            X: 输入特征
            y_true: 真实标签（如果可用，用于增量学习）
            
        Returns:
            预测结果
        """
        # 预测
        predictions = self.model.predict(X)
        
        # 如果有真实标签，触发增量学习
        if y_true is not None:
            self.samples_since_update += len(X)
            
            # 检查是否达到更新间隔
            if self.samples_since_update >= self.update_interval:
                print(f"\n🔔 触发增量更新 ({self.samples_since_update}样本)")
                update_result = self.model.update(X, y_true)
                
                if update_result['updated']:
                    self.samples_since_update = 0
                    print(f"✅ 模型已更新")
                else:
                    print(f"⏭️  {update_result['reason']}")
        
        return predictions
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'update_count': self.model.update_count,
            'base_score': self.model.base_score,
            'samples_since_update': self.samples_since_update,
            'buffer_size': len(self.model.X_buffer),
            'performance_history': self.model.performance_history
        }
    
    def plot_performance(self):
        """绘制性能曲线"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            df = self.model.get_performance_history()
            
            if len(df) == 0:
                print("⚠️ 暂无性能历史")
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(range(len(df)), df['score'], marker='o', linewidth=2)
            ax.set_xlabel('更新次数')
            ax.set_ylabel('F1分数')
            ax.set_title('在线学习性能变化')
            ax.grid(True, alpha=0.3)
            
            # 标注更新类型
            for i, row in df.iterrows():
                if row['update_type'] == 'incremental':
                    ax.axvline(i, color='red', linestyle='--', alpha=0.3)
            
            filepath = Path(self.save_dir) / 'performance_history.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 性能曲线已保存: {filepath}")
            
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")


if __name__ == '__main__':
    print("="*60)
    print("涨停板预测系统 - 在线学习优化模块")
    print("="*60)
    
    # 生成模拟数据
    np.random.seed(42)
    
    # 初始训练数据
    n_train = 2000
    X_train = pd.DataFrame(
        np.random.randn(n_train, 50),
        columns=[f'feature_{i}' for i in range(50)]
    )
    y_train = pd.Series(np.random.choice([0, 1], size=n_train, p=[0.7, 0.3]))
    
    print(f"\n训练集大小: {X_train.shape}")
    
    # 创建自适应Pipeline
    pipeline = AdaptiveLearningPipeline(
        window_size=500,
        update_interval=100,
        update_threshold=0.05
    )
    
    # 初始训练
    pipeline.fit(X_train, y_train)
    
    # 模拟在线学习（流式数据）
    print(f"\n{'='*60}")
    print("🌊 模拟流式数据增量学习")
    print(f"{'='*60}\n")
    
    n_streams = 5
    for i in range(n_streams):
        print(f"\n--- 数据流 {i+1}/{n_streams} ---")
        
        # 生成新数据
        X_new = pd.DataFrame(
            np.random.randn(150, 50),
            columns=[f'feature_{i}' for i in range(50)]
        )
        y_new = pd.Series(np.random.choice([0, 1], size=150, p=[0.7, 0.3]))
        
        # 预测并学习
        predictions = pipeline.predict_and_learn(X_new, y_new)
        print(f"预测完成: {len(predictions)}个样本")
    
    # 获取统计信息
    print(f"\n{'='*60}")
    print("📊 最终统计")
    print(f"{'='*60}")
    
    stats = pipeline.get_stats()
    print(f"总更新次数: {stats['update_count']}")
    print(f"当前F1分数: {stats['base_score']:.4f}")
    print(f"缓冲区大小: {stats['buffer_size']}")
    
    # 绘制性能曲线
    pipeline.plot_performance()
    
    print("\n✅ 所有测试完成!")
