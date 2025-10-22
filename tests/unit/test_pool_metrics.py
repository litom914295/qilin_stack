import time

from prometheus_client import CollectorRegistry, generate_latest

from qilin_stack_with_ta.monitoring.pool_metrics import PoolMetricsCollector
from app.pool import DatabaseConfig, DatabaseType, pool_manager


def test_pool_metrics_collect_once_sqlite():
    # 创建一个SQLite内存数据库连接池
    cfg = DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        database=":memory:",
        min_size=1,
        max_size=2,
    )
    pool_manager.create_database_pool("test", cfg)

    # 做一次简单的连接获取/释放以更新统计
    with pool_manager.get_database_connection("test") as conn:
        _ = conn  # no-op

    # 采集一次指标
    registry = CollectorRegistry()
    collector = PoolMetricsCollector(registry=registry, interval_seconds=0.01)
    collector.collect_once()

    output = generate_latest(registry).decode("utf-8")

    # 断言关键指标存在
    assert "qilin_db_pool_current_size" in output
    assert 'pool="test"' in output

    # 清理
    pool_manager.remove_database_pool("test")