#!/usr/bin/env python
"""
困难案例挖掘训练器
让AI在错误中成长，持续进化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


class HardCaseMining:
    """困难案例挖掘 - 循环进化训练"""
    
    def __init__(self, base_model=None):
        self.model = base_model
        self.hard_cases = []
        self.training_history = []
        self.iteration_count = 0
    
    def iterative_training(
        self, 
        historical_data: pd.DataFrame, 
        max_iterations: int = 10,
        convergence_threshold: float = 0.85,
        min_hard_cases: int = 50
    ) -> Dict:
        """
        迭代训练流程
        
        Args:
            historical_data: 历史数据
            max_iterations: 最大迭代轮数
            convergence_threshold: 收敛准确率阈值
            min_hard_cases: 最少困难案例数（低于此数即收敛）
        
        Returns:
            训练结果统计
        """
        
        print(f"开始困难案例挖掘训练，最多{max_iterations}轮")
        
        results = {
            'iterations': [],
            'final_accuracy': 0,
            'total_hard_cases': 0,
            'converged': False
        }
        
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"第 {iteration + 1}/{max_iterations} 轮训练")
            print(f"{'='*50}")
            
            # 准备训练数据
            if iteration == 0:
                # 第1轮：全量训练
                train_data = historical_data.copy()
                print("使用全量数据进行初始训练")
            else:
                # 后续轮次：重点训练困难案例
                train_data = self._prepare_hard_case_training_set(
                    historical_data, 
                    iteration
                )
                print(f"使用困难案例增强训练集（{len(train_data)}样本）")
            
            # 模拟训练（实际使用时替换为真实模型训练）
            training_result = self._train_one_iteration(train_data)
            
            # 评估并找出新的困难案例
            predictions = self._predict_all(historical_data)
            new_hard_cases = self._identify_hard_cases(
                historical_data, 
                predictions
            )
            
            print(f"发现 {len(new_hard_cases)} 个新困难案例")
            
            # 累积困难案例
            self.hard_cases.extend(new_hard_cases)
            
            # 计算准确率
            accuracy = self._calculate_accuracy(predictions, historical_data)
            
            # 记录本轮结果
            iteration_result = {
                'iteration': iteration + 1,
                'accuracy': accuracy,
                'new_hard_cases': len(new_hard_cases),
                'total_hard_cases': len(self.hard_cases),
                'training_time': training_result.get('time', 0)
            }
            
            results['iterations'].append(iteration_result)
            
            print(f"✅ 第{iteration + 1}轮完成")
            print(f"   准确率: {accuracy:.2%}")
            print(f"   新困难案例: {len(new_hard_cases)}")
            print(f"   累计困难案例: {len(self.hard_cases)}")
            
            # 收敛判断
            if accuracy >= convergence_threshold and len(new_hard_cases) < min_hard_cases:
                print(f"\n🎉 训练收敛！")
                print(f"   最终准确率: {accuracy:.2%}")
                print(f"   困难案例数: {len(self.hard_cases)}")
                results['converged'] = True
                break
        
        # 汇总结果
        results['final_accuracy'] = accuracy
        results['total_hard_cases'] = len(self.hard_cases)
        results['iteration_count'] = iteration + 1
        
        self.iteration_count = iteration + 1
        self.training_history = results
        
        return results
    
    def _identify_hard_cases(
        self, 
        data: pd.DataFrame, 
        predictions: np.ndarray
    ) -> List[Dict]:
        """识别困难案例"""
        
        hard_cases = []
        
        for i in range(len(data)):
            case_info = {'index': i}
            
            # 获取真实标签和预测
            true_label = data.iloc[i].get('main_label', 0)
            pred_label = predictions[i] if isinstance(predictions[i], (int, float)) else predictions[i].get('label', 0)
            pred_confidence = predictions[i].get('confidence', 0.5) if isinstance(predictions[i], dict) else 0.5
            
            # 类型1: 预测错误的案例
            if true_label != pred_label:
                case_info.update({
                    'type': 'wrong_prediction',
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': pred_confidence,
                    'weight': 3.0  # 预测错误 - 高权重
                })
                hard_cases.append(case_info)
                continue
            
            # 类型2: 低置信度的正确案例（边界案例）
            if pred_confidence < 0.6:
                case_info.update({
                    'type': 'low_confidence',
                    'true_label': true_label,
                    'confidence': pred_confidence,
                    'weight': 2.0  # 低置信度 - 中权重
                })
                hard_cases.append(case_info)
                continue
            
            # 类型3: 反直觉案例
            if self._is_counter_intuitive(data.iloc[i]):
                case_info.update({
                    'type': 'counter_intuitive',
                    'reason': self._get_counter_intuitive_reason(data.iloc[i]),
                    'weight': 3.0  # 反直觉 - 高权重
                })
                hard_cases.append(case_info)
        
        return hard_cases
    
    def _is_counter_intuitive(self, case: pd.Series) -> bool:
        """判断是否为反直觉案例"""
        
        seal_strength = case.get('seal_strength', case.get('封板强度', 0))
        return_1d = case.get('return_1d', 0)
        
        # 反直觉案例示例：
        
        # 1. 强封板但次日下跌
        if seal_strength > 90 and return_1d < 0:
            return True
        
        # 2. 弱封板但次日涨停
        if seal_strength < 60 and return_1d >= 0.095:
            return True
        
        # 3. 高位涨停但持续上涨
        price_position = case.get('price_position', 0.5)
        return_5d = case.get('return_5d', 0)
        if price_position > 0.9 and return_5d > 0.2:
            return True
        
        # 4. 情绪低迷但个股走强
        market_sentiment = case.get('market_sentiment', 'neutral')
        if market_sentiment in ['weak', 'poor'] and return_1d > 0.05:
            return True
        
        return False
    
    def _get_counter_intuitive_reason(self, case: pd.Series) -> str:
        """获取反直觉原因"""
        
        seal_strength = case.get('seal_strength', case.get('封板强度', 0))
        return_1d = case.get('return_1d', 0)
        
        if seal_strength > 90 and return_1d < 0:
            return "强封板但次日下跌（诱多陷阱）"
        
        if seal_strength < 60 and return_1d >= 0.095:
            return "弱封板但次日涨停（隐藏机会）"
        
        return "其他反直觉情况"
    
    def _prepare_hard_case_training_set(
        self, 
        historical_data: pd.DataFrame, 
        iteration: int
    ) -> pd.DataFrame:
        """准备困难案例训练集"""
        
        # 提取困难案例
        hard_case_indices = list(set([case['index'] for case in self.hard_cases]))
        hard_data = historical_data.iloc[hard_case_indices].copy()
        
        # 采样正常案例（保持平衡）
        normal_indices = [i for i in range(len(historical_data)) 
                         if i not in hard_case_indices]
        
        if len(normal_indices) > 0:
            sample_size = min(len(hard_data) * 2, len(normal_indices))
            normal_sample = np.random.choice(
                normal_indices, 
                size=sample_size,
                replace=False
            )
            normal_data = historical_data.iloc[normal_sample].copy()
        else:
            normal_data = pd.DataFrame()
        
        # 合并数据
        if len(normal_data) > 0:
            train_data = pd.concat([hard_data, normal_data], ignore_index=True)
        else:
            train_data = hard_data.copy()
        
        # 设置权重
        train_data['sample_weight'] = 1.0
        
        # 困难案例权重映射
        for case in self.hard_cases:
            idx = case['index']
            if idx in hard_case_indices:
                # 在train_data中找到对应行
                mask = train_data.index.isin([idx])
                if mask.any():
                    train_data.loc[mask, 'sample_weight'] = case.get('weight', 3.0)
        
        return train_data
    
    def _train_one_iteration(self, train_data: pd.DataFrame) -> Dict:
        """训练一轮（模拟）"""
        
        # 实际使用时替换为真实模型训练
        import time
        start_time = time.time()
        
        # 模拟训练过程
        time.sleep(0.1)
        
        training_time = time.time() - start_time
        
        return {
            'time': training_time,
            'samples': len(train_data)
        }
    
    def _predict_all(self, data: pd.DataFrame) -> np.ndarray:
        """预测所有数据（模拟）"""
        
        # 实际使用时替换为真实模型预测
        predictions = []
        
        for i in range(len(data)):
            # 模拟预测结果
            pred = {
                'label': np.random.choice([0, 1, 2, 3]),
                'confidence': np.random.uniform(0.3, 0.95)
            }
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _calculate_accuracy(
        self, 
        predictions: np.ndarray, 
        data: pd.DataFrame
    ) -> float:
        """计算准确率"""
        
        correct = 0
        total = len(predictions)
        
        for i in range(total):
            true_label = data.iloc[i].get('main_label', 0)
            pred_label = predictions[i]['label'] if isinstance(predictions[i], dict) else predictions[i]
            
            if true_label == pred_label:
                correct += 1
        
        return correct / total if total > 0 else 0
    
    def get_hard_cases_summary(self) -> pd.DataFrame:
        """获取困难案例摘要"""
        
        if not self.hard_cases:
            return pd.DataFrame()
        
        # 统计各类型困难案例
        type_counts = {}
        for case in self.hard_cases:
            case_type = case.get('type', 'unknown')
            type_counts[case_type] = type_counts.get(case_type, 0) + 1
        
        summary = pd.DataFrame([
            {'case_type': k, 'count': v, 'percentage': v / len(self.hard_cases)}
            for k, v in type_counts.items()
        ])
        
        return summary
    
    def save(self, save_dir: Path):
        """保存训练结果"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存困难案例
        with open(save_dir / 'hard_cases.json', 'w', encoding='utf-8') as f:
            json.dump(self.hard_cases[:1000], f, ensure_ascii=False, indent=2)  # 只保存前1000个
        
        # 保存训练历史
        with open(save_dir / 'training_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
    
    def load(self, save_dir: Path):
        """加载训练结果"""
        
        save_dir = Path(save_dir)
        
        if (save_dir / 'hard_cases.json').exists():
            with open(save_dir / 'hard_cases.json', 'r', encoding='utf-8') as f:
                self.hard_cases = json.load(f)
        
        if (save_dir / 'training_history.json').exists():
            with open(save_dir / 'training_history.json', 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)


def demo():
    """演示用法"""
    
    # 创建模拟数据
    data = pd.DataFrame({
        'code': ['000001'] * 100,
        'main_label': np.random.choice([0, 1, 2, 3], 100),
        'seal_strength': np.random.uniform(50, 95, 100),
        'return_1d': np.random.normal(0.03, 0.05, 100),
        'return_5d': np.random.normal(0.08, 0.12, 100)
    })
    
    # 创建训练器
    trainer = HardCaseMining()
    
    # 迭代训练
    results = trainer.iterative_training(
        data, 
        max_iterations=5,
        convergence_threshold=0.80
    )
    
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print(f"最终准确率: {results['final_accuracy']:.2%}")
    print(f"总困难案例: {results['total_hard_cases']}")
    print(f"迭代轮数: {results['iteration_count']}")
    print(f"是否收敛: {results['converged']}")


if __name__ == '__main__':
    demo()
