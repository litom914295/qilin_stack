"""
P2-8: 元学习与迁移学习模块 (Meta-Learning & Transfer Learning)
实现MAML、模型适配、跨市场迁移等功能
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """元学习配置"""
    inner_lr: float = 0.01          # 内循环学习率
    outer_lr: float = 0.001         # 外循环学习率
    num_inner_steps: int = 5        # 内循环更新步数
    meta_batch_size: int = 4        # 元批次大小
    num_epochs: int = 100           # 训练轮数


class SimpleNet(nn.Module):
    """简单神经网络模型"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class MAMLTrainer:
    """Model-Agnostic Meta-Learning (MAML) 训练器"""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        """
        初始化MAML训练器
        
        Args:
            model: 基础模型
            config: 元学习配置
        """
        self.model = model
        self.config = config
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.outer_lr
        )
        self.loss_fn = nn.MSELoss()
        
        logger.info(f"MAML训练器初始化完成")
    
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """
        内循环更新（任务特定的快速适配）
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
        
        Returns:
            适配后的模型
        """
        # 复制模型
        adapted_model = deepcopy(self.model)
        optimizer = optim.SGD(
            adapted_model.parameters(), 
            lr=self.config.inner_lr
        )
        
        # 在支持集上快速适配
        for _ in range(self.config.num_inner_steps):
            pred = adapted_model(support_x)
            loss = self.loss_fn(pred, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_train_step(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                                  torch.Tensor, torch.Tensor]]) -> float:
        """
        元训练步骤（外循环更新）
        
        Args:
            tasks: 任务列表，每个任务包含(support_x, support_y, query_x, query_y)
        
        Returns:
            元损失
        """
        meta_loss = 0.0
        
        self.meta_optimizer.zero_grad()
        
        for support_x, support_y, query_x, query_y in tasks:
            # 内循环：在支持集上快速适配
            adapted_model = self.inner_loop(support_x, support_y)
            
            # 在查询集上评估
            query_pred = adapted_model(query_x)
            task_loss = self.loss_fn(query_pred, query_y)
            
            meta_loss += task_loss
        
        # 外循环：更新元模型
        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """
        快速适配到新任务
        
        Args:
            support_x: 新任务的支持集输入
            support_y: 新任务的支持集标签
        
        Returns:
            适配后的模型
        """
        return self.inner_loop(support_x, support_y)


class TransferLearner:
    """迁移学习器"""
    
    def __init__(self, source_model: nn.Module, target_task_dim: int):
        """
        初始化迁移学习器
        
        Args:
            source_model: 源域预训练模型
            target_task_dim: 目标任务维度
        """
        self.source_model = source_model
        self.target_model = self._build_transfer_model(target_task_dim)
        
        logger.info("迁移学习器初始化完成")
    
    def _build_transfer_model(self, target_dim: int) -> nn.Module:
        """构建迁移模型（冻结源模型特征层）"""
        # 简化版：复制源模型并添加适配层
        transfer_model = deepcopy(self.source_model)
        
        # 冻结前面的层
        for param in list(transfer_model.parameters())[:-2]:
            param.requires_grad = False
        
        return transfer_model
    
    def fine_tune(self, x: torch.Tensor, y: torch.Tensor, 
                  epochs: int = 10, lr: float = 0.001) -> Dict[str, float]:
        """
        在目标任务上微调
        
        Args:
            x: 目标任务输入
            y: 目标任务标签
            epochs: 训练轮数
            lr: 学习率
        
        Returns:
            训练历史
        """
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.target_model.parameters()),
            lr=lr
        )
        loss_fn = nn.MSELoss()
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            pred = self.target_model(x)
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
        
        return history


class FewShotLearner:
    """少样本学习器"""
    
    def __init__(self, model: nn.Module, metric: str = 'euclidean'):
        """
        初始化少样本学习器
        
        Args:
            model: 特征提取模型
            metric: 距离度量方式
        """
        self.model = model
        self.metric = metric
        self.prototypes = {}
        
        logger.info(f"少样本学习器初始化 (metric={metric})")
    
    def compute_prototypes(self, support_x: torch.Tensor, 
                          support_y: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        计算原型（每个类的特征中心）
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
        
        Returns:
            原型字典
        """
        with torch.no_grad():
            features = self.model(support_x)
        
        prototypes = {}
        for label in torch.unique(support_y):
            mask = (support_y == label).squeeze()
            class_features = features[mask]
            prototypes[label.item()] = class_features.mean(dim=0)
        
        return prototypes
    
    def predict(self, query_x: torch.Tensor, 
                support_x: torch.Tensor, 
                support_y: torch.Tensor) -> torch.Tensor:
        """
        基于原型网络预测
        
        Args:
            query_x: 查询样本
            support_x: 支持集输入
            support_y: 支持集标签
        
        Returns:
            预测结果
        """
        # 计算原型
        prototypes = self.compute_prototypes(support_x, support_y)
        
        # 提取查询特征
        with torch.no_grad():
            query_features = self.model(query_x)
        
        # 计算到每个原型的距离
        distances = {}
        for label, prototype in prototypes.items():
            if self.metric == 'euclidean':
                dist = torch.cdist(query_features, prototype.unsqueeze(0))
            else:
                dist = 1 - torch.cosine_similarity(
                    query_features, 
                    prototype.unsqueeze(0), 
                    dim=-1
                )
            distances[label] = dist
        
        # 返回最近的原型
        predictions = []
        for i in range(len(query_x)):
            min_label = min(distances.keys(), 
                          key=lambda l: distances[l][i])
            predictions.append(min_label)
        
        return torch.tensor(predictions)


def create_sample_tasks(num_tasks: int = 10, 
                       task_size: int = 20) -> List[Tuple]:
    """
    创建示例元学习任务
    
    Args:
        num_tasks: 任务数量
        task_size: 每个任务的样本数
    
    Returns:
        任务列表
    """
    tasks = []
    
    for i in range(num_tasks):
        # 生成任务特定的数据分布
        np.random.seed(i)
        
        # 支持集
        support_x = torch.randn(task_size // 2, 10)
        support_y = torch.randn(task_size // 2, 1)
        
        # 查询集
        query_x = torch.randn(task_size // 2, 10)
        query_y = torch.randn(task_size // 2, 1)
        
        tasks.append((support_x, support_y, query_x, query_y))
    
    return tasks


def main():
    """示例: 元学习与迁移学习"""
    print("=" * 80)
    print("P2-8: 元学习与迁移学习 - 示例")
    print("=" * 80)
    
    # 1. MAML元学习
    print("\n📚 MAML元学习训练...")
    
    config = MetaLearningConfig(
        inner_lr=0.01,
        outer_lr=0.001,
        num_inner_steps=5,
        meta_batch_size=4,
        num_epochs=10
    )
    
    model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=1)
    maml_trainer = MAMLTrainer(model, config)
    
    # 创建元学习任务
    all_tasks = create_sample_tasks(num_tasks=20, task_size=20)
    
    # 元训练
    for epoch in range(5):
        batch_tasks = all_tasks[:config.meta_batch_size]
        meta_loss = maml_trainer.meta_train_step(batch_tasks)
        print(f"  Epoch {epoch+1}/5 - Meta Loss: {meta_loss:.4f}")
    
    print("✅ MAML训练完成")
    
    # 2. 快速适配到新任务
    print("\n🎯 快速适配到新任务...")
    new_task = all_tasks[-1]
    support_x, support_y, query_x, query_y = new_task
    
    adapted_model = maml_trainer.adapt(support_x, support_y)
    
    with torch.no_grad():
        pred = adapted_model(query_x)
        test_loss = nn.MSELoss()(pred, query_y)
    
    print(f"  适配后测试损失: {test_loss:.4f}")
    print("✅ 快速适配完成")
    
    # 3. 迁移学习
    print("\n🔄 迁移学习示例...")
    
    source_model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=1)
    transfer_learner = TransferLearner(source_model, target_task_dim=1)
    
    # 在目标任务上微调
    target_x = torch.randn(50, 10)
    target_y = torch.randn(50, 1)
    
    history = transfer_learner.fine_tune(target_x, target_y, epochs=5)
    print(f"  微调损失: {history['loss'][-1]:.4f}")
    print("✅ 迁移学习完成")
    
    # 4. 少样本学习
    print("\n🎲 少样本学习示例...")
    
    feature_model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=8)
    few_shot = FewShotLearner(feature_model, metric='euclidean')
    
    # 5-way 5-shot 任务
    support_x = torch.randn(25, 10)  # 5类 × 5样本
    support_y = torch.tensor([[i] for i in range(5) for _ in range(5)])
    query_x = torch.randn(10, 10)
    
    predictions = few_shot.predict(query_x, support_x, support_y)
    print(f"  预测结果: {predictions.tolist()}")
    print("✅ 少样本学习完成")
    
    print("\n" + "=" * 80)
    print("✅ 所有元学习功能演示完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
