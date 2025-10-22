from prometheus_client import CollectorRegistry, generate_latest

from qilin_stack_with_ta.monitoring.slo_metrics import get_slo_metrics, SLOMetrics


def test_slo_metrics_histogram_and_gauge_export():
    registry = CollectorRegistry()
    slo = get_slo_metrics(registry=registry)

    # 观察三次延迟，设置恢复时间
    slo.observe_latency_ms(120)
    slo.observe_latency_ms(35)
    slo.observe_latency_ms(980)
    slo.set_recovery_time(123)

    output = generate_latest(registry).decode('utf-8')

    # 直方图应导出 *_bucket 行
    assert 'qilin_e2e_latency_ms_bucket' in output
    # Gauge 应可见
    assert 'qilin_failover_recovery_seconds' in output