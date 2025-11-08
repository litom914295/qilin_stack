"""
Qlib分析模块
Phase 5.3

提供IC分析、因子分析等功能
"""

from .ic_analysis import (
    load_factor_from_qlib,
    compute_ic_timeseries,
    compute_monthly_ic_heatmap,
    layered_return_analysis,
    ic_statistics,
    run_ic_pipeline,
    ICResult
)

from .ic_visualizer import (
    plot_ic_timeseries,
    plot_monthly_ic_heatmap,
    plot_layered_returns,
    plot_ic_distribution,
    plot_ic_rolling_stats,
    plot_cumulative_ic
)

__all__ = [
    "load_factor_from_qlib",
    "compute_ic_timeseries",
    "compute_monthly_ic_heatmap",
    "layered_return_analysis",
    "ic_statistics",
    "run_ic_pipeline",
    "ICResult",
    "plot_ic_timeseries",
    "plot_monthly_ic_heatmap",
    "plot_layered_returns",
    "plot_ic_distribution",
    "plot_ic_rolling_stats",
    "plot_cumulative_ic",
]
