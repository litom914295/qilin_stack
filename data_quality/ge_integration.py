"""
Great Expectations集成
实现数据质量门禁和自动降级
"""

import pandas as pd
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from pathlib import Path

try:
    import great_expectations as gx
    from great_expectations.core.batch import RuntimeBatchRequest
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False
    logging.warning("Great Expectations not installed. Install with: pip install great-expectations")

logger = logging.getLogger(__name__)


class DataQualityGate:
    """数据质量门禁"""
    
    def __init__(self, context_root_dir: str = "./gx"):
        """
        初始化数据质量门禁
        
        Args:
            context_root_dir: GX上下文目录
        """
        if not GX_AVAILABLE:
            raise ImportError("Great Expectations is required")
        
        self.context_root_dir = Path(context_root_dir)
        self.context = self._initialize_context()
        self.quality_threshold = 0.8  # 质量分数阈值
        self.degradation_callbacks = []
        
    def _initialize_context(self):
        """初始化GX上下文"""
        if not self.context_root_dir.exists():
            context = gx.get_context(context_root_dir=str(self.context_root_dir))
            logger.info(f"Created new GX context at {self.context_root_dir}")
        else:
            context = gx.get_context(context_root_dir=str(self.context_root_dir))
            logger.info(f"Loaded existing GX context from {self.context_root_dir}")
        
        return context
    
    def create_expectation_suite(
        self,
        suite_name: str,
        data_source: str
    ) -> Dict[str, Any]:
        """
        创建Expectation Suite
        
        Args:
            suite_name: Suite名称
            data_source: 数据源名称（market/capital/news/longhu）
            
        Returns:
            Suite配置
        """
        suite = self.context.add_or_update_expectation_suite(
            expectation_suite_name=suite_name
        
        # 根据数据源类型添加expectations
        if data_source == "market":
            expectations = self._create_market_data_expectations()
        elif data_source == "capital":
            expectations = self._create_capital_flow_expectations()
        elif data_source == "news":
            expectations = self._create_news_expectations()
        elif data_source == "longhu":
            expectations = self._create_longhu_expectations()
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        logger.info(f"Created expectation suite: {suite_name} with {len(expectations)} expectations")
        return {"suite_name": suite_name, "expectations_count": len(expectations)}
    
    def _create_market_data_expectations(self) -> List[Dict]:
        """创建行情数据expectations"""
        expectations = [
            # 必需字段存在性
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "timestamp"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "open"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "high"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "low"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "close"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "volume"}
            },
            
            # 价格合理性（0-10000元）
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "open",
                    "min_value": 0,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "high",
                    "min_value": 0,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "low",
                    "min_value": 0,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "close",
                    "min_value": 0,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            },
            
            # 成交量合理性
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "volume",
                    "min_value": 0,
                    "max_value": 1e12,
                    "mostly": 0.99
                }
            },
            
            # 非空检查
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": "timestamp",
                    "mostly": 0.99
                }
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": "close",
                    "mostly": 0.95
                }
            },
            
            # 价格一致性：high >= low
            {
                "expectation_type": "expect_column_pair_values_A_to_be_greater_than_B",
                "kwargs": {
                    "column_A": "high",
                    "column_B": "low",
                    "or_equal": True,
                    "mostly": 0.99
                }
            }
        ]
        return expectations
    
    def _create_capital_flow_expectations(self) -> List[Dict]:
        """创建资金流expectations"""
        expectations = [
            # 必需字段
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "stock_code"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "net_inflow"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "big_order_inflow"}
            },
            
            # 资金范围检查
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "net_inflow",
                    "min_value": -1e10,
                    "max_value": 1e10,
                    "mostly": 0.95
                }
            },
            
            # 大单占比
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "big_order_ratio",
                    "min_value": 0,
                    "max_value": 1,
                    "mostly": 0.99
                }
            }
        ]
        return expectations
    
    def _create_news_expectations(self) -> List[Dict]:
        """创建新闻数据expectations"""
        expectations = [
            # 必需字段
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "title"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "content"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "publish_time"}
            },
            
            # 非空检查
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": "title",
                    "mostly": 0.99
                }
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": "content",
                    "mostly": 0.95
                }
            },
            
            # 标题长度
            {
                "expectation_type": "expect_column_value_lengths_to_be_between",
                "kwargs": {
                    "column": "title",
                    "min_value": 5,
                    "max_value": 200,
                    "mostly": 0.95
                }
            }
        ]
        return expectations
    
    def _create_longhu_expectations(self) -> List[Dict]:
        """创建龙虎榜expectations"""
        expectations = [
            # 必需字段
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "stock_code"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "seal_amount"}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "seal_time"}
            },
            
            # 封单金额
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "seal_amount",
                    "min_value": 0,
                    "max_value": 1e11,
                    "mostly": 0.99
                }
            },
            
            # 涨停时间格式
            {
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {
                    "column": "seal_time",
                    "regex": r"^\d{2}:\d{2}$",
                    "mostly": 0.95
                }
            }
        ]
        return expectations
    
    def validate_data(
        self,
        data: pd.DataFrame,
        suite_name: str,
        data_source: str
    ) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            data: 待验证数据
            suite_name: Suite名称
            data_source: 数据源名称
            
        Returns:
            验证结果
        """
        # 创建运行时batch
        batch_request = RuntimeBatchRequest(
            datasource_name=f"{data_source}_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name=f"{data_source}_data",
            runtime_parameters={"batch_data": data},
            batch_identifiers={"default_identifier_name": datetime.now().isoformat()}
        
        # 获取expectation suite
        try:
            suite = self.context.get_expectation_suite(suite_name)
        except Exception:
            # Suite不存在，创建一个
            self.create_expectation_suite(suite_name, data_source)
            suite = self.context.get_expectation_suite(suite_name)
        
        # 运行验证
        checkpoint_config = {
            "name": f"{suite_name}_checkpoint",
            "config_version": 1.0,
            "class_name": "SimpleCheckpoint",
            "validations": [
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": suite_name
                }
            ]
        }
        
        try:
            checkpoint = self.context.add_or_update_checkpoint(**checkpoint_config)
            results = checkpoint.run()
            
            # 解析结果
            success = results["success"]
            statistics = results.get("statistics", {})
            
            # 计算质量分数
            total_expectations = statistics.get("evaluated_expectations", 1)
            successful_expectations = statistics.get("successful_expectations", 0)
            quality_score = successful_expectations / total_expectations if total_expectations > 0 else 0
            
            result = {
                "success": success,
                "quality_score": quality_score,
                "total_expectations": total_expectations,
                "successful_expectations": successful_expectations,
                "failed_expectations": total_expectations - successful_expectations,
                "timestamp": datetime.now().isoformat(),
                "data_source": data_source,
                "row_count": len(data)
            }
            
            # 检查是否需要降级
            if quality_score < self.quality_threshold:
                logger.warning(
                    f"Data quality below threshold: {quality_score:.2f} < {self.quality_threshold} "
                    f"for {data_source}"
                self._trigger_degradation(data_source, quality_score, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "success": False,
                "quality_score": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "data_source": data_source
            }
    
    def _trigger_degradation(
        self,
        data_source: str,
        quality_score: float,
        validation_result: Dict
    ):
        """触发降级机制"""
        degradation_event = {
            "event_type": "data_quality_degradation",
            "data_source": data_source,
            "quality_score": quality_score,
            "threshold": self.quality_threshold,
            "timestamp": datetime.now().isoformat(),
            "validation_result": validation_result
        }
        
        logger.critical(f"DEGRADATION TRIGGERED: {json.dumps(degradation_event)}")
        
        # 调用注册的降级回调
        for callback in self.degradation_callbacks:
            try:
                callback(degradation_event)
            except Exception as e:
                logger.error(f"Degradation callback failed: {e}")
    
    def register_degradation_callback(self, callback):
        """注册降级回调函数"""
        self.degradation_callbacks.append(callback)
    
    def batch_validate(
        self,
        data_batches: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        批量验证多个数据源
        
        Args:
            data_batches: {data_source: DataFrame} 字典
            
        Returns:
            验证结果字典
        """
        results = {}
        for data_source, data in data_batches.items():
            suite_name = f"{data_source}_suite"
            result = self.validate_data(data, suite_name, data_source)
            results[data_source] = result
        
        # 计算总体质量分数
        if results:
            avg_quality_score = sum(r.get("quality_score", 0) for r in results.values()) / len(results)
            logger.info(f"Batch validation completed. Average quality score: {avg_quality_score:.2f}")
        
        return results


class DataQualityCI:
    """CI集成"""
    
    @staticmethod
    def run_ci_validation(data_dir: str = "./data") -> int:
        """
        运行CI验证
        
        Args:
            data_dir: 数据目录
            
        Returns:
            退出码（0=成功，1=失败）
        """
        gate = DataQualityGate()
        
        # 加载测试数据
        data_sources = ["market", "capital", "news", "longhu"]
        all_passed = True
        
        for source in data_sources:
            data_file = Path(data_dir) / f"{source}_test.csv"
            if data_file.exists():
                data = pd.read_csv(data_file)
                result = gate.validate_data(
                    data,
                    suite_name=f"{source}_suite",
                    data_source=source
                
                if not result.get("success", False):
                    all_passed = False
                    logger.error(f"CI validation failed for {source}")
                else:
                    logger.info(f"CI validation passed for {source}")
        
        return 0 if all_passed else 1


if __name__ == "__main__":
    # 测试代码
    if GX_AVAILABLE:
        # 创建测试数据
        market_data = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="1min"),
            "open": [10.0 + i * 0.1 for i in range(100)],
            "high": [10.2 + i * 0.1 for i in range(100)],
            "low": [9.8 + i * 0.1 for i in range(100)],
            "close": [10.1 + i * 0.1 for i in range(100)],
            "volume": [1000000 + i * 10000 for i in range(100)]
        })
        
        # 初始化质量门禁
        gate = DataQualityGate()
        
        # 创建并验证
        gate.create_expectation_suite("market_suite", "market")
        result = gate.validate_data(market_data, "market_suite", "market")
        
        print(f"Validation Result: {json.dumps(result, indent=2)}")
    else:
        print("Great Expectations not available. Please install it.")
