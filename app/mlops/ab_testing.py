"""
A/B测试框架 - 模型对比和流量分配
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """测试状态"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Variant:
    """测试变体"""
    name: str
    model_name: str
    model_version: int
    traffic_weight: float = 0.0
    description: str = ""
    
    # 运行时统计
    requests_count: int = 0
    predictions: List[Any] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ABTest:
    """A/B测试"""
    test_id: str
    name: str
    description: str
    variants: List[Variant]
    
    # 测试配置
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    
    # 测试状态
    status: TestStatus = TestStatus.DRAFT
    winner: Optional[str] = None
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingFramework:
    """A/B测试框架"""
    
    def __init__(self, model_registry):
        """
        初始化A/B测试框架
        
        Args:
            model_registry: 模型注册表实例
        """
        self.model_registry = model_registry
        self.tests: Dict[str, ABTest] = {}
        self.active_test: Optional[ABTest] = None
        
    def create_test(
        self,
        test_id: str,
        name: str,
        variants: List[Dict[str, Any]],
        description: str = "",
        min_sample_size: int = 1000,
        confidence_level: float = 0.95
    ) -> ABTest:
        """
        创建新的A/B测试
        
        Args:
            test_id: 测试ID
            name: 测试名称
            variants: 变体配置列表
            description: 测试描述
            min_sample_size: 最小样本量
            confidence_level: 置信水平
            
        Returns:
            ABTest对象
        """
        # 创建变体对象
        variant_objects = []
        total_weight = 0.0
        
        for v in variants:
            variant = Variant(
                name=v['name'],
                model_name=v['model_name'],
                model_version=v['model_version'],
                traffic_weight=v.get('traffic_weight', 0.5),
                description=v.get('description', '')
            )
            variant_objects.append(variant)
            total_weight += variant.traffic_weight
        
        # 归一化权重
        if total_weight > 0:
            for variant in variant_objects:
                variant.traffic_weight /= total_weight
        
        # 创建测试
        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            variants=variant_objects,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level
        )
        
        self.tests[test_id] = test
        logger.info(f"Created A/B test: {test_id} with {len(variant_objects)} variants")
        
        return test
    
    def start_test(self, test_id: str):
        """开始A/B测试"""
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        if test.status != TestStatus.DRAFT:
            raise ValueError(f"Test {test_id} is not in DRAFT status")
        
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now()
        self.active_test = test
        
        logger.info(f"Started A/B test: {test_id}")
    
    def pause_test(self, test_id: str):
        """暂停测试"""
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        test.status = TestStatus.PAUSED
        
        if self.active_test and self.active_test.test_id == test_id:
            self.active_test = None
        
        logger.info(f"Paused A/B test: {test_id}")
    
    def resume_test(self, test_id: str):
        """恢复测试"""
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        test.status = TestStatus.RUNNING
        self.active_test = test
        
        logger.info(f"Resumed A/B test: {test_id}")
    
    def route_request(self, test_id: Optional[str] = None) -> Variant:
        """
        路由请求到特定变体
        
        Args:
            test_id: 测试ID，如果为None则使用活跃测试
            
        Returns:
            选中的变体
        """
        # 使用指定测试或活跃测试
        test = self.tests.get(test_id) if test_id else self.active_test
        
        if not test or test.status != TestStatus.RUNNING:
            raise ValueError("No active test available")
        
        # 使用权重进行随机分配
        weights = [v.traffic_weight for v in test.variants]
        variant = np.random.choice(test.variants, p=weights)
        
        variant.requests_count += 1
        
        return variant
    
    def record_prediction(
        self,
        test_id: str,
        variant_name: str,
        prediction: Any,
        actual: Optional[Any] = None,
        metadata: Optional[Dict] = None
    ):
        """
        记录预测结果
        
        Args:
            test_id: 测试ID
            variant_name: 变体名称
            prediction: 预测值
            actual: 实际值（可选）
            metadata: 额外元数据
        """
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        # 找到对应变体
        variant = next((v for v in test.variants if v.name == variant_name), None)
        
        if not variant:
            raise ValueError(f"Variant not found: {variant_name}")
        
        # 记录预测
        record = {
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        variant.predictions.append(record)
    
    def calculate_metrics(
        self,
        test_id: str,
        metric_fn: Callable[[List[Any], List[Any]], float]
    ) -> Dict[str, float]:
        """
        计算各变体的指标
        
        Args:
            test_id: 测试ID
            metric_fn: 指标计算函数
            
        Returns:
            各变体的指标字典
        """
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        results = {}
        
        for variant in test.variants:
            if not variant.predictions:
                results[variant.name] = None
                continue
            
            # 提取预测和实际值
            predictions = [p['prediction'] for p in variant.predictions if p.get('actual') is not None]
            actuals = [p['actual'] for p in variant.predictions if p.get('actual') is not None]
            
            if predictions and actuals:
                metric_value = metric_fn(actuals, predictions)
                variant.metrics['primary'] = metric_value
                results[variant.name] = metric_value
            else:
                results[variant.name] = None
        
        return results
    
    def analyze_results(self, test_id: str) -> Dict[str, Any]:
        """
        分析测试结果
        
        Args:
            test_id: 测试ID
            
        Returns:
            分析结果字典
        """
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        # 检查样本量
        sample_sizes = {v.name: len(v.predictions) for v in test.variants}
        min_samples = min(sample_sizes.values()) if sample_sizes else 0
        
        is_significant = min_samples >= test.min_sample_size
        
        # 计算统计指标
        variant_stats = []
        
        for variant in test.variants:
            if not variant.predictions:
                continue
            
            predictions = [p['prediction'] for p in variant.predictions if p.get('actual') is not None]
            actuals = [p['actual'] for p in variant.predictions if p.get('actual') is not None]
            
            if predictions and actuals:
                # 计算准确率（示例）
                accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(predictions)
                
                stats = {
                    'variant': variant.name,
                    'model': f"{variant.model_name}:v{variant.model_version}",
                    'requests': variant.requests_count,
                    'samples': len(predictions),
                    'accuracy': accuracy,
                    'metrics': variant.metrics
                }
                
                variant_stats.append(stats)
        
        # 确定获胜者
        winner = None
        if is_significant and variant_stats:
            winner = max(variant_stats, key=lambda x: x.get('accuracy', 0))['variant']
            test.winner = winner
        
        analysis = {
            'test_id': test_id,
            'status': test.status.value,
            'is_significant': is_significant,
            'min_samples': min_samples,
            'required_samples': test.min_sample_size,
            'winner': winner,
            'variants': variant_stats,
            'start_time': test.start_time.isoformat() if test.start_time else None,
            'duration_hours': (datetime.now() - test.start_time).total_seconds() / 3600 if test.start_time else None
        }
        
        return analysis
    
    def complete_test(self, test_id: str, winner: Optional[str] = None):
        """
        完成测试
        
        Args:
            test_id: 测试ID
            winner: 获胜变体（可选，如果不提供则自动判断）
        """
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        # 分析结果
        analysis = self.analyze_results(test_id)
        
        # 设置获胜者
        if winner:
            test.winner = winner
        else:
            test.winner = analysis.get('winner')
        
        test.status = TestStatus.COMPLETED
        test.end_time = datetime.now()
        
        if self.active_test and self.active_test.test_id == test_id:
            self.active_test = None
        
        logger.info(f"Completed A/B test: {test_id}, winner: {test.winner}")
        
        return analysis
    
    def get_test_report(self, test_id: str) -> str:
        """
        生成测试报告
        
        Args:
            test_id: 测试ID
            
        Returns:
            格式化的报告字符串
        """
        analysis = self.analyze_results(test_id)
        
        report = f"""
A/B Test Report: {test_id}
{'='*60}

Status: {analysis['status']}
Duration: {analysis.get('duration_hours', 0):.2f} hours
Significant: {analysis['is_significant']}
Winner: {analysis.get('winner', 'TBD')}

Variants:
{'-'*60}
"""
        
        for variant in analysis['variants']:
            report += f"""
{variant['variant']} ({variant['model']}):
  - Requests: {variant['requests']:,}
  - Samples: {variant['samples']:,}
  - Accuracy: {variant['accuracy']:.4f}
  - Metrics: {json.dumps(variant['metrics'], indent=4)}
"""
        
        return report
    
    def export_results(self, test_id: str) -> pd.DataFrame:
        """
        导出测试结果为DataFrame
        
        Args:
            test_id: 测试ID
            
        Returns:
            包含所有预测记录的DataFrame
        """
        test = self.tests.get(test_id)
        
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        records = []
        
        for variant in test.variants:
            for pred in variant.predictions:
                record = {
                    'test_id': test_id,
                    'variant': variant.name,
                    'model': variant.model_name,
                    'version': variant.model_version,
                    'prediction': pred['prediction'],
                    'actual': pred.get('actual'),
                    'timestamp': pred['timestamp']
                }
                record.update(pred.get('metadata', {}))
                records.append(record)
        
        return pd.DataFrame(records)
