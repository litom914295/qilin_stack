#!/usr/bin/env python
"""
自我对抗训练器 - AI vs AI
让AI生成陷阱案例，训练自己识别伪装
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


class AdversarialTrainer:
    """自我对抗训练 - 提升鲁棒性"""
    
    def __init__(self, base_model=None):
        self.predictor = base_model
        self.adversarial_cases = []
        self.training_history = []
        self.round_count = 0
    
    def adversarial_evolution(
        self,
        historical_data: pd.DataFrame,
        max_rounds: int = 10,
        target_robustness: float = 9.0
    ) -> Dict:
        """
        对抗进化训练
        
        Args:
            historical_data: 历史数据
            max_rounds: 最大对抗轮数
            target_robustness: 目标鲁棒性得分(0-10)
        
        Returns:
            训练结果统计
        """
        
        print(f"开始自我对抗训练，最多{max_rounds}轮")
        
        results = {
            'rounds': [],
            'final_robustness': 0,
            'total_adversarial_cases': 0,
            'success': False
        }
        
        for round_num in range(max_rounds):
            print(f"\n{'='*50}")
            print(f"Round {round_num + 1}/{max_rounds}: 对抗训练")
            print(f"{'='*50}")
            
            # 1. 训练预测模型
            print("训练预测模型...")
            self._train_predictor(historical_data)
            
            # 2. 生成对抗案例
            print("生成对抗案例...")
            new_adversarial_cases = self._generate_adversarial_cases(
                historical_data,
                num_cases=100
            )
            
            print(f"生成 {len(new_adversarial_cases)} 个对抗案例")
            
            # 3. 测试欺骗率
            fooled_rate = self._test_adversarial_cases(new_adversarial_cases)
            print(f"对抗案例欺骗率: {fooled_rate:.1%}")
            
            # 4. 将对抗案例加入训练集
            enhanced_data = pd.concat([
                historical_data,
                new_adversarial_cases
            ], ignore_index=True)
            
            # 5. 重新训练（对抗案例高权重）
            print("重新训练（对抗案例权重5x）...")
            enhanced_data['sample_weight'] = 1.0
            enhanced_data.loc[
                enhanced_data.index >= len(historical_data),
                'sample_weight'
            ] = 5.0
            
            self._train_predictor(enhanced_data)
            
            # 6. 评估鲁棒性
            robustness_score = self._evaluate_robustness(historical_data)
            print(f"鲁棒性得分: {robustness_score:.2f}/10")
            
            # 累积对抗案例
            self.adversarial_cases.extend(new_adversarial_cases.to_dict('records'))
            
            # 记录本轮结果
            round_result = {
                'round': round_num + 1,
                'adversarial_cases': len(new_adversarial_cases),
                'fooled_rate': fooled_rate,
                'robustness_score': robustness_score
            }
            
            results['rounds'].append(round_result)
            
            # 收敛判断
            if robustness_score >= target_robustness:
                print(f"\n🎉 达到目标鲁棒性！")
                results['success'] = True
                break
        
        # 汇总结果
        results['final_robustness'] = robustness_score
        results['total_adversarial_cases'] = len(self.adversarial_cases)
        results['round_count'] = round_num + 1
        
        self.round_count = round_num + 1
        self.training_history = results
        
        return results
    
    def _generate_adversarial_cases(
        self,
        data: pd.DataFrame,
        num_cases: int
    ) -> pd.DataFrame:
        """生成对抗案例"""
        
        adversarial_cases = []
        
        # 类型1: 伪强势（诱多陷阱）
        fake_strong = self._create_fake_strong_cases(data, num_cases // 3)
        adversarial_cases.append(fake_strong)
        
        # 类型2: 隐藏机会
        hidden_gem = self._create_hidden_gem_cases(data, num_cases // 3)
        adversarial_cases.append(hidden_gem)
        
        # 类型3: 情绪陷阱
        emotion_trap = self._create_emotion_trap_cases(data, num_cases // 3)
        adversarial_cases.append(emotion_trap)
        
        result = pd.concat(adversarial_cases, ignore_index=True)
        result['is_adversarial'] = True
        
        return result
    
    def _create_fake_strong_cases(self, data: pd.DataFrame, num_cases: int) -> pd.DataFrame:
        """创建伪强势案例（特征强但结果差）"""
        
        # 从失败案例中采样
        failed_cases = data[data.get('return_1d', 0) < 0].copy()
        
        if len(failed_cases) == 0:
            # 如果没有失败案例，创建合成案例
            failed_cases = data.sample(n=min(num_cases, len(data))).copy()
        
        fake_cases = failed_cases.sample(n=min(num_cases, len(failed_cases))).copy()
        
        # 人为增强特征（制造陷阱）
        fake_cases['seal_strength'] = np.random.uniform(85, 95, len(fake_cases))
        fake_cases['main_inflow'] = np.random.uniform(8000, 15000, len(fake_cases))
        fake_cases['volume_ratio'] = np.random.uniform(2.0, 5.0, len(fake_cases))
        
        # 但实际标签是失败
        fake_cases['main_label'] = 0
        fake_cases['return_1d'] = np.random.uniform(-0.05, -0.01, len(fake_cases))
        fake_cases['adversarial_type'] = 'fake_strong'
        
        return fake_cases
    
    def _create_hidden_gem_cases(self, data: pd.DataFrame, num_cases: int) -> pd.DataFrame:
        """创建隐藏机会案例（特征弱但结果好）"""
        
        # 从成功案例中采样
        success_cases = data[data.get('return_1d', 0) >= 0.095].copy()
        
        if len(success_cases) == 0:
            success_cases = data.sample(n=min(num_cases, len(data))).copy()
        
        hidden_cases = success_cases.sample(n=min(num_cases, len(success_cases))).copy()
        
        # 人为削弱特征（隐藏机会）
        hidden_cases['seal_strength'] = np.random.uniform(50, 70, len(hidden_cases))
        hidden_cases['main_inflow'] = np.random.uniform(-2000, 2000, len(hidden_cases))
        hidden_cases['volume_ratio'] = np.random.uniform(0.8, 1.5, len(hidden_cases))
        
        # 但实际标签是成功（涨停）
        hidden_cases['main_label'] = 3
        hidden_cases['return_1d'] = np.random.uniform(0.095, 0.10, len(hidden_cases))
        hidden_cases['adversarial_type'] = 'hidden_gem'
        
        return hidden_cases
    
    def _create_emotion_trap_cases(self, data: pd.DataFrame, num_cases: int) -> pd.DataFrame:
        """创建情绪陷阱案例（市场情绪与个股相反）"""
        
        cases = data.sample(n=min(num_cases, len(data))).copy()
        
        # 50%: 市场好但个股差
        # 50%: 市场差但个股好
        for i, idx in enumerate(cases.index):
            if i < len(cases) // 2:
                # 市场好但个股差
                cases.loc[idx, 'market_sentiment'] = 'strong'
                cases.loc[idx, 'total_limitup'] = np.random.randint(80, 120)
                cases.loc[idx, 'main_label'] = 0
                cases.loc[idx, 'return_1d'] = np.random.uniform(-0.05, 0)
            else:
                # 市场差但个股好
                cases.loc[idx, 'market_sentiment'] = 'weak'
                cases.loc[idx, 'total_limitup'] = np.random.randint(10, 30)
                cases.loc[idx, 'main_label'] = 3
                cases.loc[idx, 'return_1d'] = np.random.uniform(0.08, 0.10)
        
        cases['adversarial_type'] = 'emotion_trap'
        
        return cases
    
    def _test_adversarial_cases(self, adversarial_cases: pd.DataFrame) -> float:
        """测试对抗案例的欺骗率"""
        
        # 模拟预测
        fooled_count = 0
        
        for idx, case in adversarial_cases.iterrows():
            true_label = case['main_label']
            
            # 模拟预测（实际使用时替换为真实模型预测）
            # 这里简单模拟：根据特征猜测
            if case.get('seal_strength', 0) > 80:
                pred_label = 3  # 预测涨停
            elif case.get('seal_strength', 0) < 60:
                pred_label = 0  # 预测失败
            else:
                pred_label = np.random.choice([0, 1, 2, 3])
            
            # 如果预测错误，说明被欺骗
            if pred_label != true_label:
                fooled_count += 1
        
        return fooled_count / len(adversarial_cases) if len(adversarial_cases) > 0 else 0
    
    def _train_predictor(self, data: pd.DataFrame):
        """训练预测模型（模拟）"""
        
        import time
        time.sleep(0.1)  # 模拟训练时间
    
    def _evaluate_robustness(self, data: pd.DataFrame) -> float:
        """评估模型鲁棒性（0-10分）"""
        
        # 模拟鲁棒性评估
        # 实际使用时：测试模型在各种边界情况下的表现
        
        # 随着训练轮数增加，鲁棒性提升
        base_robustness = 5.0
        improvement = min(self.round_count * 0.4, 4.0)
        
        robustness = base_robustness + improvement + np.random.uniform(-0.3, 0.3)
        
        return min(10.0, max(0.0, robustness))
    
    def get_adversarial_summary(self) -> pd.DataFrame:
        """获取对抗案例摘要"""
        
        if not self.adversarial_cases:
            return pd.DataFrame()
        
        # 统计各类型
        type_counts = {}
        for case in self.adversarial_cases:
            case_type = case.get('adversarial_type', 'unknown')
            type_counts[case_type] = type_counts.get(case_type, 0) + 1
        
        type_names = {
            'fake_strong': '伪强势（诱多陷阱）',
            'hidden_gem': '隐藏机会',
            'emotion_trap': '情绪陷阱'
        }
        
        summary = pd.DataFrame([
            {
                'type': type_names.get(k, k),
                'count': v,
                'percentage': v / len(self.adversarial_cases)
            }
            for k, v in type_counts.items()
        ])
        
        return summary
    
    def save(self, save_dir: Path):
        """保存训练结果"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存对抗案例（限制数量）
        with open(save_dir / 'adversarial_cases.json', 'w', encoding='utf-8') as f:
            json.dump(self.adversarial_cases[:1000], f, ensure_ascii=False, indent=2)
        
        # 保存训练历史
        with open(save_dir / 'adversarial_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
    
    def load(self, save_dir: Path):
        """加载训练结果"""
        
        save_dir = Path(save_dir)
        
        if (save_dir / 'adversarial_cases.json').exists():
            with open(save_dir / 'adversarial_cases.json', 'r', encoding='utf-8') as f:
                self.adversarial_cases = json.load(f)
        
        if (save_dir / 'adversarial_history.json').exists():
            with open(save_dir / 'adversarial_history.json', 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)


def demo():
    """演示用法"""
    
    # 创建模拟数据
    data = pd.DataFrame({
        'code': ['000001'] * 100,
        'main_label': np.random.choice([0, 1, 2, 3], 100),
        'seal_strength': np.random.uniform(50, 95, 100),
        'main_inflow': np.random.uniform(-5000, 15000, 100),
        'return_1d': np.random.normal(0.03, 0.05, 100),
        'market_sentiment': np.random.choice(['strong', 'neutral', 'weak'], 100)
    })
    
    # 创建训练器
    trainer = AdversarialTrainer()
    
    # 对抗训练
    results = trainer.adversarial_evolution(
        data,
        max_rounds=5,
        target_robustness=9.0
    )
    
    print("\n" + "="*50)
    print("对抗训练完成！")
    print("="*50)
    print(f"训练轮数: {results['round_count']}")
    print(f"最终鲁棒性: {results['final_robustness']:.2f}/10")
    print(f"对抗案例总数: {results['total_adversarial_cases']}")
    print(f"达标: {'✅' if results['success'] else '❌'}")


if __name__ == '__main__':
    demo()
