"""
数据质量监控器 (Data Quality Monitor)
多维度监控数据质量并实时告警

核心监控维度：
1. 完整性：缺失值检测
2. 准确性：异常值检测
3. 一致性：逻辑一致性校验
4. 时效性：数据延迟监控
5. 合规性：数据规范校验
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd


class QualityDimension(Enum):
    """数据质量维度"""
    COMPLETENESS = "完整性"
    ACCURACY = "准确性"
    CONSISTENCY = "一致性"
    TIMELINESS = "时效性"
    VALIDITY = "合规性"


class SeverityLevel(Enum):
    """严重等级"""
    INFO = "信息"
    WARNING = "警告"
    ERROR = "错误"
    CRITICAL = "严重"


@dataclass
class QualityIssue:
    """质量问题"""
    dimension: QualityDimension
    severity: SeverityLevel
    field: str
    description: str
    timestamp: datetime
    value: Any = None
    expected: Any = None
    suggestion: str = ""


@dataclass
class QualityMetrics:
    """质量指标"""
    # 完整性指标
    completeness_rate: float          # 完整率
    missing_fields: List[str]         # 缺失字段
    
    # 准确性指标
    accuracy_rate: float              # 准确率
    outlier_count: int                # 异常值数量
    outlier_fields: List[str]         # 异常字段
    
    # 一致性指标
    consistency_rate: float           # 一致性率
    inconsistency_count: int          # 不一致数量
    inconsistent_fields: List[str]    # 不一致字段
    
    # 时效性指标
    avg_latency_ms: float             # 平均延迟
    max_latency_ms: float             # 最大延迟
    stale_data_count: int             # 过期数据数
    
    # 合规性指标
    validity_rate: float              # 合规率
    invalid_count: int                # 不合规数量
    invalid_fields: List[str]         # 不合规字段
    
    # 汇总
    overall_score: float              # 总体得分(0-100)
    total_issues: int                 # 总问题数
    issues: List[QualityIssue] = field(default_factory=list)


class DataQualityRule:
    """数据质量规则抽象基类"""
    
    def __init__(self, dimension: QualityDimension, severity: SeverityLevel):
        self.dimension = dimension
        self.severity = severity
    
    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查数据质量"""
        raise NotImplementedError


class CompletenessRule(DataQualityRule):
    """完整性规则"""
    
    def __init__(self, required_fields: List[str], 
                 missing_threshold: float = 0.01,
                 severity: SeverityLevel = SeverityLevel.WARNING):
        super().__init__(QualityDimension.COMPLETENESS, severity)
        self.required_fields = required_fields
        self.missing_threshold = missing_threshold  # 允许的缺失率
    
    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查完整性"""
        issues = []
        
        for field in self.required_fields:
            if field not in data.columns:
                issues.append(QualityIssue(
                    dimension=self.dimension,
                    severity=SeverityLevel.CRITICAL,
                    field=field,
                    description=f"字段'{field}'完全缺失",
                    timestamp=datetime.now(),
                    suggestion=f"检查数据源是否正确提供'{field}'字段"
                ))
            else:
                missing_rate = data[field].isna().sum() / len(data)
                if missing_rate > self.missing_threshold:
                    issues.append(QualityIssue(
                        dimension=self.dimension,
                        severity=self.severity,
                        field=field,
                        description=f"字段'{field}'缺失率过高: {missing_rate:.2%}",
                        timestamp=datetime.now(),
                        value=missing_rate,
                        expected=self.missing_threshold,
                        suggestion=f"填充缺失值或移除'{field}'字段"
                    ))
        
        return issues


class AccuracyRule(DataQualityRule):
    """准确性规则（异常值检测）"""
    
    def __init__(self, field: str, 
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None,
                 use_iqr: bool = True,
                 iqr_multiplier: float = 1.5,
                 severity: SeverityLevel = SeverityLevel.WARNING):
        super().__init__(QualityDimension.ACCURACY, severity)
        self.field = field
        self.min_value = min_value
        self.max_value = max_value
        self.use_iqr = use_iqr
        self.iqr_multiplier = iqr_multiplier
    
    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查准确性"""
        issues = []
        
        if self.field not in data.columns:
            return issues
        
        values = data[self.field].dropna()
        
        if len(values) == 0:
            return issues
        
        # 方法1: 硬性边界检查
        if self.min_value is not None:
            outliers = values[values < self.min_value]
            if len(outliers) > 0:
                issues.append(QualityIssue(
                    dimension=self.dimension,
                    severity=self.severity,
                    field=self.field,
                    description=f"检测到{len(outliers)}个低于下限({self.min_value})的异常值",
                    timestamp=datetime.now(),
                    value=outliers.min(),
                    expected=self.min_value,
                    suggestion=f"检查'{self.field}'数据来源是否正确"
                ))
        
        if self.max_value is not None:
            outliers = values[values > self.max_value]
            if len(outliers) > 0:
                issues.append(QualityIssue(
                    dimension=self.dimension,
                    severity=self.severity,
                    field=self.field,
                    description=f"检测到{len(outliers)}个高于上限({self.max_value})的异常值",
                    timestamp=datetime.now(),
                    value=outliers.max(),
                    expected=self.max_value,
                    suggestion=f"检查'{self.field}'数据来源是否正确"
                ))
        
        # 方法2: IQR方法检测离群值
        if self.use_iqr and len(values) >= 10:
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            outlier_rate = len(outliers) / len(values)
            
            if outlier_rate > 0.05:  # 离群值超过5%
                issues.append(QualityIssue(
                    dimension=self.dimension,
                    severity=self.severity,
                    field=self.field,
                    description=f"IQR方法检测到{len(outliers)}个异常值({outlier_rate:.2%})",
                    timestamp=datetime.now(),
                    value=outlier_rate,
                    expected=0.05,
                    suggestion=f"使用Z-score或Winsorize方法处理'{self.field}'的异常值"
                ))
        
        return issues


class ConsistencyRule(DataQualityRule):
    """一致性规则"""
    
    def __init__(self, checks: List[Dict[str, Any]],
                 severity: SeverityLevel = SeverityLevel.ERROR):
        super().__init__(QualityDimension.CONSISTENCY, severity)
        self.checks = checks
    
    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查一致性"""
        issues = []
        
        for check in self.checks:
            check_type = check.get("type")
            
            # 检查类型1: 价格关系
            if check_type == "price_relation":
                fields = check.get("fields", [])
                if all(f in data.columns for f in fields):
                    # 例如：high >= close >= low
                    high_col, close_col, low_col = fields
                    
                    invalid = data[
                        (data[high_col] < data[close_col]) | 
                        (data[close_col] < data[low_col])
                    ]
                    
                    if len(invalid) > 0:
                        issues.append(QualityIssue(
                            dimension=self.dimension,
                            severity=self.severity,
                            field=", ".join(fields),
                            description=f"检测到{len(invalid)}行价格关系不一致(high >= close >= low)",
                            timestamp=datetime.now(),
                            value=len(invalid),
                            suggestion="检查数据源是否存在错误或数据污染"
                        ))
            
            # 检查类型2: 成交量-金额一致性
            elif check_type == "volume_amount":
                volume_field = check.get("volume_field", "volume")
                amount_field = check.get("amount_field", "amount")
                price_field = check.get("price_field", "close")
                
                if all(f in data.columns for f in [volume_field, amount_field, price_field]):
                    # 计算预期金额
                    expected_amount = data[volume_field] * data[price_field]
                    actual_amount = data[amount_field]
                    
                    # 允许2%的误差
                    diff_rate = abs((actual_amount - expected_amount) / expected_amount)
                    invalid = data[diff_rate > 0.02]
                    
                    if len(invalid) > 0:
                        issues.append(QualityIssue(
                            dimension=self.dimension,
                            severity=self.severity,
                            field=f"{volume_field}, {amount_field}, {price_field}",
                            description=f"检测到{len(invalid)}行成交量与金额不一致",
                            timestamp=datetime.now(),
                            value=len(invalid),
                            suggestion="验证成交金额 = 成交量 × 价格是否成立"
                        ))
        
        return issues


class TimelinessRule(DataQualityRule):
    """时效性规则"""
    
    def __init__(self, timestamp_field: str = "timestamp",
                 max_delay_seconds: float = 60,
                 severity: SeverityLevel = SeverityLevel.WARNING):
        super().__init__(QualityDimension.TIMELINESS, severity)
        self.timestamp_field = timestamp_field
        self.max_delay_seconds = max_delay_seconds
    
    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查时效性"""
        issues = []
        
        if self.timestamp_field not in data.columns:
            issues.append(QualityIssue(
                dimension=self.dimension,
                severity=SeverityLevel.ERROR,
                field=self.timestamp_field,
                description=f"缺少时间戳字段'{self.timestamp_field}'",
                timestamp=datetime.now(),
                suggestion=f"确保数据中包含'{self.timestamp_field}'字段"
            ))
            return issues
        
        # 转换为datetime
        timestamps = pd.to_datetime(data[self.timestamp_field], errors='coerce')
        now = pd.Timestamp.now()
        
        # 计算延迟
        delays = (now - timestamps).dt.total_seconds()
        stale_data = delays[delays > self.max_delay_seconds]
        
        if len(stale_data) > 0:
            max_delay = delays.max()
            avg_delay = delays.mean()
            
            severity = self.severity
            if max_delay > self.max_delay_seconds * 10:
                severity = SeverityLevel.CRITICAL
            
            issues.append(QualityIssue(
                dimension=self.dimension,
                severity=severity,
                field=self.timestamp_field,
                description=f"检测到{len(stale_data)}条数据延迟超过{self.max_delay_seconds}秒",
                timestamp=datetime.now(),
                value=max_delay,
                expected=self.max_delay_seconds,
                suggestion=f"平均延迟{avg_delay:.1f}秒，最大延迟{max_delay:.1f}秒，检查数据采集链路"
            ))
        
        return issues


class ValidityRule(DataQualityRule):
    """合规性规则"""
    
    def __init__(self, field: str, valid_values: Optional[List] = None,
                 regex_pattern: Optional[str] = None,
                 severity: SeverityLevel = SeverityLevel.ERROR):
        super().__init__(QualityDimension.VALIDITY, severity)
        self.field = field
        self.valid_values = valid_values
        self.regex_pattern = regex_pattern
    
    def check(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查合规性"""
        issues = []
        
        if self.field not in data.columns:
            return issues
        
        values = data[self.field].dropna()
        
        # 检查方法1: 枚举值校验
        if self.valid_values is not None:
            invalid = values[~values.isin(self.valid_values)]
            if len(invalid) > 0:
                issues.append(QualityIssue(
                    dimension=self.dimension,
                    severity=self.severity,
                    field=self.field,
                    description=f"检测到{len(invalid)}个不合规值",
                    timestamp=datetime.now(),
                    value=invalid.unique().tolist()[:5],
                    expected=self.valid_values,
                    suggestion=f"'{self.field}'的值必须在{self.valid_values}中"
                ))
        
        # 检查方法2: 正则表达式校验
        if self.regex_pattern is not None:
            invalid = values[~values.astype(str).str.match(self.regex_pattern)]
            if len(invalid) > 0:
                issues.append(QualityIssue(
                    dimension=self.dimension,
                    severity=self.severity,
                    field=self.field,
                    description=f"检测到{len(invalid)}个格式不合规值",
                    timestamp=datetime.now(),
                    value=invalid.unique().tolist()[:5],
                    expected=f"匹配正则: {self.regex_pattern}",
                    suggestion=f"'{self.field}'的值必须符合格式要求"
                ))
        
        return issues


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self):
        self.rules: List[DataQualityRule] = []
        self.history: List[QualityMetrics] = []
    
    def add_rule(self, rule: DataQualityRule):
        """添加质量规则"""
        self.rules.append(rule)
        print(f"✅ 添加规则: {rule.dimension.value} - {rule.__class__.__name__}")
    
    def check(self, data: pd.DataFrame) -> QualityMetrics:
        """执行质量检查"""
        all_issues = []
        
        # 执行所有规则
        for rule in self.rules:
            try:
                issues = rule.check(data)
                all_issues.extend(issues)
            except Exception as e:
                print(f"规则执行失败: {rule.__class__.__name__} - {e}")
        
        # 统计各维度指标
        metrics = self._calculate_metrics(data, all_issues)
        
        # 保存历史
        self.history.append(metrics)
        
        return metrics
    
    def _calculate_metrics(self, data: pd.DataFrame, 
                           issues: List[QualityIssue]) -> QualityMetrics:
        """计算质量指标"""
        # 按维度分组问题
        dimension_issues = {dim: [] for dim in QualityDimension}
        for issue in issues:
            dimension_issues[issue.dimension].append(issue)
        
        # 完整性指标
        completeness_issues = dimension_issues[QualityDimension.COMPLETENESS]
        missing_fields = [i.field for i in completeness_issues]
        completeness_rate = 1.0 - len(missing_fields) / max(len(data.columns), 1)
        
        # 准确性指标
        accuracy_issues = dimension_issues[QualityDimension.ACCURACY]
        outlier_fields = [i.field for i in accuracy_issues]
        accuracy_rate = 1.0 - len(outlier_fields) / max(len(data.columns), 1)
        
        # 一致性指标
        consistency_issues = dimension_issues[QualityDimension.CONSISTENCY]
        inconsistent_fields = [i.field for i in consistency_issues]
        consistency_rate = 1.0 - len(consistency_issues) / max(len(data), 1)
        
        # 时效性指标
        timeliness_issues = dimension_issues[QualityDimension.TIMELINESS]
        avg_latency = sum(i.value for i in timeliness_issues if i.value) / max(len(timeliness_issues), 1)
        max_latency = max((i.value for i in timeliness_issues if i.value), default=0)
        stale_data_count = sum(1 for i in timeliness_issues)
        
        # 合规性指标
        validity_issues = dimension_issues[QualityDimension.VALIDITY]
        invalid_fields = [i.field for i in validity_issues]
        validity_rate = 1.0 - len(invalid_fields) / max(len(data.columns), 1)
        
        # 总体得分（加权平均）
        weights = {
            QualityDimension.COMPLETENESS: 0.3,
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.CONSISTENCY: 0.20,
            QualityDimension.TIMELINESS: 0.15,
            QualityDimension.VALIDITY: 0.10
        }
        
        overall_score = (
            completeness_rate * weights[QualityDimension.COMPLETENESS] +
            accuracy_rate * weights[QualityDimension.ACCURACY] +
            consistency_rate * weights[QualityDimension.CONSISTENCY] +
            (1 - min(avg_latency / 100, 1)) * weights[QualityDimension.TIMELINESS] +
            validity_rate * weights[QualityDimension.VALIDITY]
        ) * 100
        
        return QualityMetrics(
            completeness_rate=completeness_rate,
            missing_fields=missing_fields,
            accuracy_rate=accuracy_rate,
            outlier_count=len(accuracy_issues),
            outlier_fields=outlier_fields,
            consistency_rate=consistency_rate,
            inconsistency_count=len(consistency_issues),
            inconsistent_fields=inconsistent_fields,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            stale_data_count=stale_data_count,
            validity_rate=validity_rate,
            invalid_count=len(validity_issues),
            invalid_fields=invalid_fields,
            overall_score=overall_score,
            total_issues=len(issues),
            issues=issues
        )
    
    def print_report(self, metrics: QualityMetrics):
        """打印质量报告"""
        print("\n" + "="*60)
        print("📊 数据质量报告")
        print("="*60)
        
        # 总体评分
        score = metrics.overall_score
        if score >= 90:
            grade = "优秀 ✅"
        elif score >= 80:
            grade = "良好 ⚠️"
        elif score >= 70:
            grade = "合格 ⚠️"
        else:
            grade = "不合格 ❌"
        
        print(f"\n总体得分: {score:.1f}/100 - {grade}")
        print(f"问题总数: {metrics.total_issues}")
        
        # 各维度详情
        print(f"\n📦 完整性: {metrics.completeness_rate:.1%}")
        if metrics.missing_fields:
            print(f"   缺失字段({len(metrics.missing_fields)}): {', '.join(metrics.missing_fields[:5])}")
        
        print(f"\n🎯 准确性: {metrics.accuracy_rate:.1%}")
        if metrics.outlier_fields:
            print(f"   异常字段({len(metrics.outlier_fields)}): {', '.join(metrics.outlier_fields[:5])}")
        
        print(f"\n🔄 一致性: {metrics.consistency_rate:.1%}")
        if metrics.inconsistent_fields:
            print(f"   不一致字段({len(metrics.inconsistent_fields)}): {', '.join(metrics.inconsistent_fields[:5])}")
        
        print(f"\n⏱️  时效性:")
        print(f"   平均延迟: {metrics.avg_latency_ms:.1f}ms")
        print(f"   最大延迟: {metrics.max_latency_ms:.1f}ms")
        print(f"   过期数据: {metrics.stale_data_count}")
        
        print(f"\n✅ 合规性: {metrics.validity_rate:.1%}")
        if metrics.invalid_fields:
            print(f"   不合规字段({len(metrics.invalid_fields)}): {', '.join(metrics.invalid_fields[:5])}")
        
        # 问题详情
        if metrics.issues:
            print(f"\n⚠️  问题详情 (共{len(metrics.issues)}个):")
            
            # 按严重程度分组
            critical = [i for i in metrics.issues if i.severity == SeverityLevel.CRITICAL]
            errors = [i for i in metrics.issues if i.severity == SeverityLevel.ERROR]
            warnings = [i for i in metrics.issues if i.severity == SeverityLevel.WARNING]
            
            if critical:
                print(f"\n❌ 严重问题 ({len(critical)}个):")
                for issue in critical[:3]:
                    print(f"   - [{issue.dimension.value}] {issue.field}: {issue.description}")
                    if issue.suggestion:
                        print(f"     💡 {issue.suggestion}")
            
            if errors:
                print(f"\n⚠️  错误 ({len(errors)}个):")
                for issue in errors[:3]:
                    print(f"   - [{issue.dimension.value}] {issue.field}: {issue.description}")
                    if issue.suggestion:
                        print(f"     💡 {issue.suggestion}")
            
            if warnings:
                print(f"\nℹ️  警告 ({len(warnings)}个):")
                for issue in warnings[:3]:
                    print(f"   - [{issue.dimension.value}] {issue.field}: {issue.description}")
        
        print("\n" + "="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    # 创建监控器
    monitor = DataQualityMonitor()
    
    # 添加规则
    print("🔧 配置质量规则...\n")
    
    # 1. 完整性规则
    monitor.add_rule(CompletenessRule(
        required_fields=["symbol", "open", "high", "low", "close", "volume"],
        missing_threshold=0.01
    ))
    
    # 2. 准确性规则
    monitor.add_rule(AccuracyRule(
        field="close",
        min_value=0.01,
        max_value=10000,
        use_iqr=True
    ))
    
    monitor.add_rule(AccuracyRule(
        field="volume",
        min_value=0,
        use_iqr=True
    ))
    
    # 3. 一致性规则
    monitor.add_rule(ConsistencyRule(checks=[
        {
            "type": "price_relation",
            "fields": ["high", "close", "low"]
        }
    ]))
    
    # 4. 时效性规则
    monitor.add_rule(TimelinessRule(
        timestamp_field="timestamp",
        max_delay_seconds=60
    ))
    
    # 5. 合规性规则
    monitor.add_rule(ValidityRule(
        field="symbol",
        regex_pattern=r"^\d{6}\.(SZ|SH)$"
    ))
    
    # 创建测试数据
    print("\n📦 生成测试数据...\n")
    test_data = pd.DataFrame({
        "symbol": ["000001.SZ", "600000.SH", "INVALID", "000002.SZ"],
        "open": [10.0, 20.0, np.nan, 15.0],
        "high": [10.5, 21.0, 16.0, 15.5],
        "close": [10.3, 20.5, 15.8, 15.2],
        "low": [9.8, 19.5, 15.5, 14.8],
        "volume": [1000000, 2000000, -500, 1500000],  # 异常值: -500
        "timestamp": [
            datetime.now(),
            datetime.now() - timedelta(seconds=30),
            datetime.now() - timedelta(seconds=120),  # 过期数据
            datetime.now() - timedelta(seconds=10)
        ]
    })
    
    # 执行检查
    print("🔍 执行质量检查...\n")
    metrics = monitor.check(test_data)
    
    # 打印报告
    monitor.print_report(metrics)
    
    print("✅ 完成")
