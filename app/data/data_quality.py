"""
数据质量监控系统
实现数据完整性、一致性、时效性、准确性检查
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QualityCheckType(Enum):
    """质量检查类型"""
    COMPLETENESS = "completeness"  # 完整性
    CONSISTENCY = "consistency"    # 一致性
    TIMELINESS = "timeliness"      # 时效性
    ACCURACY = "accuracy"          # 准确性
    VALIDITY = "validity"          # 有效性
    UNIQUENESS = "uniqueness"      # 唯一性


class SeverityLevel(Enum):
    """严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """质量问题"""
    check_type: QualityCheckType
    severity: SeverityLevel
    message: str
    field: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityReport:
    """质量报告"""
    data_source: str
    check_time: datetime
    total_records: int
    passed_checks: int
    failed_checks: int
    quality_score: float
    issues: List[QualityIssue]
    metrics: Dict[str, float] = field(default_factory=dict)


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化质量检查器
        
        Args:
            config: 检查配置
        """
        self.config = config or self._default_config()
        self.issues: List[QualityIssue] = []
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'completeness': {
                'required_fields': ['open', 'high', 'low', 'close', 'volume'],
                'min_completeness_ratio': 0.95
            },
            'timeliness': {
                'max_delay_minutes': 5,
                'check_update_time': True
            },
            'accuracy': {
                'price_range': {'min': 0, 'max': 10000},
                'volume_range': {'min': 0, 'max': 1e12},
                'check_outliers': True,
                'outlier_std_threshold': 3
            },
            'consistency': {
                'check_high_low': True,
                'check_ohlc_order': True,
                'check_duplicate_timestamps': True
            }
        }
    
    def check_all(
        self,
        data: pd.DataFrame,
        data_source: str = "unknown",
        symbol: Optional[str] = None
    ) -> QualityReport:
        """
        执行所有质量检查
        
        Args:
            data: 待检查数据
            data_source: 数据源名称
            symbol: 股票代码
            
        Returns:
            质量报告
        """
        self.issues = []
        
        # 1. 完整性检查
        self._check_completeness(data)
        
        # 2. 一致性检查
        self._check_consistency(data)
        
        # 3. 时效性检查
        self._check_timeliness(data)
        
        # 4. 准确性检查
        self._check_accuracy(data)
        
        # 5. 有效性检查
        self._check_validity(data)
        
        # 生成报告
        return self._generate_report(data, data_source)
    
    def _check_completeness(self, data: pd.DataFrame):
        """检查数据完整性"""
        config = self.config['completeness']
        required_fields = config['required_fields']
        min_ratio = config['min_completeness_ratio']
        
        # 检查必需字段
        missing_fields = [f for f in required_fields if f not in data.columns]
        if missing_fields:
            self.issues.append(QualityIssue(
                check_type=QualityCheckType.COMPLETENESS,
                severity=SeverityLevel.ERROR,
                message=f"缺少必需字段: {missing_fields}",
                details={'missing_fields': missing_fields}
            ))
        
        # 检查空值
        for field in required_fields:
            if field in data.columns:
                null_count = data[field].isnull().sum()
                null_ratio = null_count / len(data)
                
                if null_ratio > (1 - min_ratio):
                    self.issues.append(QualityIssue(
                        check_type=QualityCheckType.COMPLETENESS,
                        severity=SeverityLevel.WARNING,
                        message=f"字段 {field} 空值比例过高: {null_ratio:.2%}",
                        field=field,
                        details={
                            'null_count': int(null_count),
                            'null_ratio': float(null_ratio)
                        }
                    ))
        
        # 检查数据量
        if len(data) == 0:
            self.issues.append(QualityIssue(
                check_type=QualityCheckType.COMPLETENESS,
                severity=SeverityLevel.CRITICAL,
                message="数据为空",
                details={'record_count': 0}
            ))
    
    def _check_consistency(self, data: pd.DataFrame):
        """检查数据一致性"""
        config = self.config['consistency']
        
        if len(data) == 0:
            return
        
        # 检查high >= low
        if config['check_high_low']:
            if 'high' in data.columns and 'low' in data.columns:
                invalid_hl = data[data['high'] < data['low']]
                if len(invalid_hl) > 0:
                    self.issues.append(QualityIssue(
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.ERROR,
                        message=f"发现 {len(invalid_hl)} 条记录high < low",
                        details={
                            'invalid_count': int(len(invalid_hl)),
                            'sample_indices': invalid_hl.index[:5].tolist()
                        }
                    ))
        
        # 检查OHLC顺序
        if config['check_ohlc_order']:
            required = ['open', 'high', 'low', 'close']
            if all(f in data.columns for f in required):
                # high应该是最大值
                invalid_high = data[
                    (data['high'] < data['open']) |
                    (data['high'] < data['low']) |
                    (data['high'] < data['close'])
                ]
                
                # low应该是最小值
                invalid_low = data[
                    (data['low'] > data['open']) |
                    (data['low'] > data['high']) |
                    (data['low'] > data['close'])
                ]
                
                if len(invalid_high) > 0:
                    self.issues.append(QualityIssue(
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.ERROR,
                        message=f"发现 {len(invalid_high)} 条记录high不是最大值",
                        details={'invalid_count': int(len(invalid_high))}
                    ))
                
                if len(invalid_low) > 0:
                    self.issues.append(QualityIssue(
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.ERROR,
                        message=f"发现 {len(invalid_low)} 条记录low不是最小值",
                        details={'invalid_count': int(len(invalid_low))}
                    ))
        
        # 检查重复时间戳
        if config['check_duplicate_timestamps']:
            if isinstance(data.index, pd.DatetimeIndex):
                duplicates = data.index.duplicated()
                dup_count = duplicates.sum()
                
                if dup_count > 0:
                    self.issues.append(QualityIssue(
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.WARNING,
                        message=f"发现 {dup_count} 个重复时间戳",
                        details={'duplicate_count': int(dup_count)}
                    ))
    
    def _check_timeliness(self, data: pd.DataFrame):
        """检查数据时效性"""
        config = self.config['timeliness']
        
        if len(data) == 0:
            return
        
        # 检查最新数据时间
        if isinstance(data.index, pd.DatetimeIndex):
            latest_time = data.index.max()
            current_time = datetime.now()
            delay = (current_time - latest_time).total_seconds() / 60
            
            max_delay = config['max_delay_minutes']
            
            if delay > max_delay:
                self.issues.append(QualityIssue(
                    check_type=QualityCheckType.TIMELINESS,
                    severity=SeverityLevel.WARNING,
                    message=f"数据延迟 {delay:.1f} 分钟",
                    details={
                        'latest_time': latest_time.isoformat(),
                        'delay_minutes': float(delay)
                    }
                ))
        
        # 检查时间序列连续性
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            time_diffs = data.index.to_series().diff()
            median_diff = time_diffs.median()
            
            # 查找异常间隔
            large_gaps = time_diffs[time_diffs > median_diff * 3]
            
            if len(large_gaps) > 0:
                self.issues.append(QualityIssue(
                    check_type=QualityCheckType.TIMELINESS,
                    severity=SeverityLevel.INFO,
                    message=f"发现 {len(large_gaps)} 个时间间隔异常",
                    details={
                        'gap_count': int(len(large_gaps)),
                        'median_interval': str(median_diff)
                    }
                ))
    
    def _check_accuracy(self, data: pd.DataFrame):
        """检查数据准确性"""
        config = self.config['accuracy']
        
        if len(data) == 0:
            return
        
        # 检查价格范围
        price_fields = ['open', 'high', 'low', 'close']
        price_range = config['price_range']
        
        for field in price_fields:
            if field in data.columns:
                out_of_range = data[
                    (data[field] < price_range['min']) |
                    (data[field] > price_range['max'])
                ]
                
                if len(out_of_range) > 0:
                    self.issues.append(QualityIssue(
                        check_type=QualityCheckType.ACCURACY,
                        severity=SeverityLevel.ERROR,
                        message=f"字段 {field} 有 {len(out_of_range)} 个值超出正常范围",
                        field=field,
                        details={
                            'out_of_range_count': int(len(out_of_range)),
                            'min_value': float(data[field].min()),
                            'max_value': float(data[field].max())
                        }
                    ))
        
        # 检查成交量范围
        if 'volume' in data.columns:
            volume_range = config['volume_range']
            out_of_range = data[
                (data['volume'] < volume_range['min']) |
                (data['volume'] > volume_range['max'])
            ]
            
            if len(out_of_range) > 0:
                self.issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.WARNING,
                    message=f"成交量有 {len(out_of_range)} 个值超出正常范围",
                    field='volume',
                    details={'out_of_range_count': int(len(out_of_range))}
                ))
        
        # 检查异常值
        if config['check_outliers']:
            self._check_outliers(data)
    
    def _check_outliers(self, data: pd.DataFrame):
        """检查异常值"""
        threshold = self.config['accuracy']['outlier_std_threshold']
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                
                if std > 0:
                    z_scores = np.abs((data[col] - mean) / std)
                    outliers = data[z_scores > threshold]
                    
                    if len(outliers) > 0:
                        outlier_ratio = len(outliers) / len(data)
                        
                        if outlier_ratio > 0.01:  # 超过1%视为异常
                            self.issues.append(QualityIssue(
                                check_type=QualityCheckType.ACCURACY,
                                severity=SeverityLevel.INFO,
                                message=f"字段 {col} 有 {len(outliers)} 个异常值 (>{threshold}σ)",
                                field=col,
                                details={
                                    'outlier_count': int(len(outliers)),
                                    'outlier_ratio': float(outlier_ratio)
                                }
                            ))
    
    def _check_validity(self, data: pd.DataFrame):
        """检查数据有效性"""
        # 检查负值
        price_fields = ['open', 'high', 'low', 'close', 'volume']
        
        for field in price_fields:
            if field in data.columns:
                negative_count = (data[field] < 0).sum()
                
                if negative_count > 0:
                    self.issues.append(QualityIssue(
                        check_type=QualityCheckType.VALIDITY,
                        severity=SeverityLevel.ERROR,
                        message=f"字段 {field} 有 {negative_count} 个负值",
                        field=field,
                        details={'negative_count': int(negative_count)}
                    ))
        
        # 检查无穷值和NaN
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            inf_count = np.isinf(data[col]).sum()
            
            if inf_count > 0:
                self.issues.append(QualityIssue(
                    check_type=QualityCheckType.VALIDITY,
                    severity=SeverityLevel.ERROR,
                    message=f"字段 {col} 有 {inf_count} 个无穷值",
                    field=col,
                    details={'inf_count': int(inf_count)}
                ))
    
    def _generate_report(
        self,
        data: pd.DataFrame,
        data_source: str
    ) -> QualityReport:
        """生成质量报告"""
        total_checks = len(self.issues) + 100  # 假设总共约100项检查
        failed_checks = len(self.issues)
        passed_checks = total_checks - failed_checks
        
        # 计算质量分数 (0-100)
        severity_weights = {
            SeverityLevel.INFO: 0.1,
            SeverityLevel.WARNING: 0.3,
            SeverityLevel.ERROR: 0.6,
            SeverityLevel.CRITICAL: 1.0
        }
        
        penalty = sum(
            severity_weights[issue.severity]
            for issue in self.issues
        
        quality_score = max(0, 100 - penalty * 2)
        
        # 计算指标
        metrics = self._calculate_metrics(data)
        
        return QualityReport(
            data_source=data_source,
            check_time=datetime.now(),
            total_records=len(data),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            quality_score=quality_score,
            issues=self.issues,
            metrics=metrics
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算质量指标"""
        if len(data) == 0:
            return {}
        
        metrics = {}
        
        # 完整性指标
        required_fields = self.config['completeness']['required_fields']
        present_fields = [f for f in required_fields if f in data.columns]
        metrics['field_completeness'] = len(present_fields) / len(required_fields)
        
        # 数据完整度
        total_cells = len(data) * len(data.columns)
        non_null_cells = data.count().sum()
        metrics['data_completeness'] = non_null_cells / total_cells if total_cells > 0 else 0
        
        # 一致性指标
        if 'high' in data.columns and 'low' in data.columns:
            valid_hl = (data['high'] >= data['low']).sum()
            metrics['high_low_consistency'] = valid_hl / len(data)
        
        # 重复率
        if isinstance(data.index, pd.DatetimeIndex):
            unique_timestamps = data.index.nunique()
            metrics['uniqueness'] = unique_timestamps / len(data)
        
        return metrics


class QualityMonitor:
    """质量监控器"""
    
    def __init__(self, alert_threshold: float = 80.0):
        """
        初始化监控器
        
        Args:
            alert_threshold: 告警阈值（质量分数低于此值触发告警）
        """
        self.alert_threshold = alert_threshold
        self.history: List[QualityReport] = []
        self.checker = DataQualityChecker()
    
    def monitor(
        self,
        data: pd.DataFrame,
        data_source: str,
        symbol: Optional[str] = None
    ) -> Tuple[QualityReport, bool]:
        """
        监控数据质量
        
        Args:
            data: 数据
            data_source: 数据源
            symbol: 股票代码
            
        Returns:
            (质量报告, 是否需要告警)
        """
        # 执行质量检查
        report = self.checker.check_all(data, data_source, symbol)
        
        # 保存历史
        self.history.append(report)
        
        # 判断是否需要告警
        needs_alert = report.quality_score < self.alert_threshold
        
        if needs_alert:
            self._send_alert(report)
        
        return report, needs_alert
    
    def _send_alert(self, report: QualityReport):
        """发送告警"""
        critical_issues = [
            issue for issue in report.issues
            if issue.severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]
        ]
        
        alert_message = f"""
数据质量告警
============
数据源: {report.data_source}
质量分数: {report.quality_score:.1f}
严重问题数: {len(critical_issues)}
总问题数: {len(report.issues)}

关键问题:
"""
        
        for issue in critical_issues[:5]:  # 只显示前5个
            alert_message += f"- [{issue.severity.value}] {issue.message}\n"
        
        logger.warning(alert_message)
    
    def get_quality_trend(self, data_source: Optional[str] = None, window: int = 10) -> pd.DataFrame:
        """
        获取质量趋势
        
        Args:
            data_source: 数据源（可选）
            window: 时间窗口
            
        Returns:
            趋势DataFrame
        """
        reports = self.history
        
        if data_source:
            reports = [r for r in reports if r.data_source == data_source]
        
        if not reports:
            return pd.DataFrame()
        
        # 只取最近的记录
        reports = reports[-window:]
        
        df = pd.DataFrame([
            {
                'time': r.check_time,
                'data_source': r.data_source,
                'quality_score': r.quality_score,
                'issue_count': len(r.issues),
                'total_records': r.total_records
            }
            for r in reports
        ])
        
        return df.set_index('time')
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成监控摘要"""
        if not self.history:
            return {'status': 'no_data'}
        
        recent_reports = self.history[-10:]
        
        avg_score = np.mean([r.quality_score for r in recent_reports])
        min_score = min(r.quality_score for r in recent_reports)
        max_score = max(r.quality_score for r in recent_reports)
        
        total_issues = sum(len(r.issues) for r in recent_reports)
        
        severity_counts = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        
        for report in recent_reports:
            for issue in report.issues:
                severity_counts[issue.severity.value] += 1
        
        return {
            'status': 'ok' if avg_score >= self.alert_threshold else 'degraded',
            'reports_count': len(self.history),
            'recent_avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'total_issues': total_issues,
            'severity_distribution': severity_counts,
            'last_check': self.history[-1].check_time.isoformat()
        }
