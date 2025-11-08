"""
Qlib IC分析报告 Web UI
Phase 5.3 实现

功能：
1. 因子表达式输入（Qlib表达式）
2. IC时间序列分析
3. 月度IC热力图
4. 分层收益分析
5. IC统计摘要
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

# 导入IC分析模块
try:
    from qlib_enhanced.analysis import (
        load_factor_from_qlib,
        run_ic_pipeline,
        plot_ic_timeseries,
        plot_monthly_ic_heatmap,
        plot_layered_returns,
        plot_ic_distribution,
        plot_ic_rolling_stats,
        plot_cumulative_ic
    )
    IC_AVAILABLE = True
except ImportError as e:
    st.error(f"IC分析模块导入失败: {e}")
    IC_AVAILABLE = False

# 导入P2-1 Alpha缓存加载器
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.write_chanlun_alphas_to_qlib import load_factor_from_qlib_cache
    ALPHA_CACHE_AVAILABLE = True
except ImportError:
    ALPHA_CACHE_AVAILABLE = False


def render_qlib_ic_analysis_tab():
    """渲染Qlib IC分析报告标签页"""
    st.title("📊 IC分析报告")
    st.markdown("基于Qlib的因子IC分析和可视化")
    
    if not IC_AVAILABLE:
        st.error("❌ IC分析模块未加载，请检查依赖")
        return
    
    # 三个子标签
    tab1, tab2, tab3 = st.tabs([
        "🔬 快速分析",
        "📈 深度分析",
        "📚 使用指南"
    ])
    
    with tab1:
        render_quick_analysis()
    
    with tab2:
        render_deep_analysis()
    
    with tab3:
        render_user_guide()


# ==================== 快速分析 ====================

def render_quick_analysis():
    """渲染快速分析"""
    st.header("🔬 快速IC分析")
    st.markdown("快速评估单个因子的IC表现")
    
    # 左右分栏
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 📝 因子配置")
        
        # 基础配置
        preset = st.selectbox(
            "预设因子",
            ["(无)", "$alpha_confluence", "$alpha_zs_movement", "$alpha_zs_upgrade"],
            index=0,
            help="如果你的因子流水线已包含这些字段，可一键填充"
        )
        if st.button("填充预设", use_container_width=True):
            if preset != "(无)":
                st.session_state["quick_factor"] = preset
        factor_expr = st.text_input(
            "因子表达式（Qlib格式）",
            value=st.session_state.get("quick_factor", "Ref($close, 0) / Ref($close, 1) - 1"),
            help="使用Qlib表达式，如: $alpha_confluence",
            key="quick_factor"
        )
        
        label_expr = st.text_input(
            "标签表达式（预测目标）",
            value="Ref($close, -1) / $close - 1",
            help="通常为未来收益，如: Ref($close, -1) / $close - 1",
            key="quick_label"
        )
        
        st.markdown("---")
        
        # 数据范围
        st.markdown("### 📅 数据范围")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=pd.to_datetime("2018-01-01"),
                key="quick_start"
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=pd.to_datetime("2021-12-31"),
                key="quick_end"
            )
        
        instruments = st.selectbox(
            "股票池",
            ["csi300", "csi500", "all"],
            index=0,
            key="quick_instruments"
        )
        
        st.markdown("---")
        
        # 分析参数
        st.markdown("### ⚙️ 分析参数")
        
        quantiles = st.slider(
            "分层数量",
            min_value=3,
            max_value=10,
            value=5,
            help="将因子值分为N层",
            key="quick_quantiles"
        )
        
        ic_method = st.selectbox(
            "IC计算方法",
            ["spearman", "pearson"],
            index=0,
            help="spearman: 秩相关; pearson: 线性相关",
            key="quick_method"
        )
        
        # 执行按钮
        st.markdown("---")
        if st.button("🚀 开始分析", type="primary", use_container_width=True, key="quick_run"):
            with col_right:
                run_quick_analysis(
                    factor_expr=factor_expr,
                    label_expr=label_expr,
                    instruments=instruments,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    quantiles=quantiles,
                    ic_method=ic_method
                )
    
    with col_right:
        st.markdown("### 📈 分析结果")
        st.info("👈 配置因子参数后，点击\"开始分析\"查看结果")


def run_quick_analysis(factor_expr, label_expr, instruments, start_date, end_date, quantiles, ic_method):
    """执行快速分析"""
    try:
        st.markdown("#### ⏳ 正在加载数据...")
        
        # P2-1: 检测是否为缓存Alpha因子
        is_cached_alpha = factor_expr.startswith('$alpha_') and ALPHA_CACHE_AVAILABLE
        
        # 加载数据
        if is_cached_alpha:
            # 从缓存加载
            try:
                alpha_name = factor_expr.replace('$', '')  # 去掉$前缀
                df = load_factor_from_qlib_cache(
                    alpha_name=alpha_name,
                    instruments=instruments,
                    start=start_date,
                    end=end_date
                )
                st.info(f"ℹ️  从Alpha缓存加载: {alpha_name}")
            except FileNotFoundError:
                st.warning(f"⚠️  Alpha缓存未找到，尝试从Qlib加载...")
                df = load_factor_from_qlib(
                    instruments=instruments,
                    start=start_date,
                    end=end_date,
                    factor_expr=factor_expr,
                    label_expr=label_expr
                )
        else:
            # 标准Qlib表达式加载
            df = load_factor_from_qlib(
                instruments=instruments,
                start=start_date,
                end=end_date,
                factor_expr=factor_expr,
                label_expr=label_expr
            )
        
        st.success(f"✅ 数据加载完成: {len(df)} 条记录")
        
        # 运行IC分析
        st.markdown("#### 📊 正在计算IC...")
        result = run_ic_pipeline(
            df=df,
            factor_col="factor",
            label_col="label",
            quantiles=quantiles,
            method=ic_method
        )
        
        st.success("✅ IC分析完成")
        
        # 显示统计摘要
        st.markdown("#### 📈 统计摘要")
        display_stats_summary(result.stats)
        
        # IC时间序列
        st.markdown("#### 📉 IC时间序列")
        fig_ts = plot_ic_timeseries(result.ic_series)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # 分层收益
        if len(result.layered_returns) > 0:
            st.markdown("#### 📊 分层收益分析")
            fig_layer = plot_layered_returns(result.layered_returns)
            st.plotly_chart(fig_layer, use_container_width=True)
            
            # 显示分层表格
            with st.expander("📋 查看分层详情"):
                st.dataframe(result.layered_returns, use_container_width=True, hide_index=True)
        
        # IC分布
        st.markdown("#### 📊 IC分布")
        fig_dist = plot_ic_distribution(result.ic_series)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 分析失败: {str(e)}")
        with st.expander("🔍 查看详细错误"):
            st.code(traceback.format_exc())


def display_stats_summary(stats):
    """显示统计摘要"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("平均IC", f"{stats['ic_mean']:.4f}")
        st.metric("IC标准差", f"{stats['ic_std']:.4f}")
    
    with col2:
        st.metric("ICIR", f"{stats['icir']:.4f}")
        st.metric("IC正比例", f"{stats['pos_rate']:.2%}")
    
    with col3:
        st.metric("t统计量", f"{stats['t_stat']:.2f}")
        st.metric("5%分位", f"{stats['p05']:.4f}")
    
    with col4:
        # 评级
        ic_mean = stats['ic_mean']
        if ic_mean > 0.05:
            rating = "🟢 优秀"
            color = "green"
        elif ic_mean > 0.03:
            rating = "🟡 良好"
            color = "orange"
        elif ic_mean > 0.01:
            rating = "🟠 一般"
            color = "orange"
        else:
            rating = "🔴 较差"
            color = "red"
        
        st.metric("因子评级", rating)
        st.metric("95%分位", f"{stats['p95']:.4f}")


# ==================== 深度分析 ====================

def render_deep_analysis():
    """渲染深度分析"""
    st.header("📈 深度IC分析")
    st.markdown("多维度IC分析，包含月度热力图、滚动统计、累积IC等")
    
    # 配置区域（折叠）
    with st.expander("⚙️ 配置分析参数", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            preset_deep = st.selectbox(
                "预设因子",
                ["(无)", "$alpha_confluence", "$alpha_zs_movement", "$alpha_zs_upgrade"],
                index=0,
                key="deep_preset"
            )
            if st.button("填充预设(深度)", use_container_width=True):
                if preset_deep != "(无)":
                    st.session_state["deep_factor"] = preset_deep
            factor_expr_deep = st.text_input(
                "因子表达式",
                value=st.session_state.get("deep_factor", "Ref($close, 0) / Ref($close, 5) - 1"),
                key="deep_factor"
            )
            start_date_deep = st.date_input(
                "开始日期",
                value=pd.to_datetime("2018-01-01"),
                key="deep_start"
            )
        
        with col2:
            label_expr_deep = st.text_input(
                "标签表达式",
                value="Ref($close, -1) / $close - 1",
                key="deep_label"
            )
            end_date_deep = st.date_input(
                "结束日期",
                value=pd.to_datetime("2021-12-31"),
                key="deep_end"
            )
        
        with col3:
            instruments_deep = st.selectbox(
                "股票池",
                ["csi300", "csi500", "all"],
                index=0,
                key="deep_instruments"
            )
            rolling_window = st.slider(
                "滚动窗口（天）",
                min_value=20,
                max_value=120,
                value=60,
                step=10,
                key="deep_window"
            )
    
    # 执行按钮
    if st.button("🔬 执行深度分析", type="primary", use_container_width=True, key="deep_run"):
        run_deep_analysis(
            factor_expr=factor_expr_deep,
            label_expr=label_expr_deep,
            instruments=instruments_deep,
            start_date=str(start_date_deep),
            end_date=str(end_date_deep),
            rolling_window=rolling_window
        )


def run_deep_analysis(factor_expr, label_expr, instruments, start_date, end_date, rolling_window):
    """执行深度分析"""
    try:
        with st.spinner("⏳ 正在加载和分析数据..."):
            # 加载数据
            df = load_factor_from_qlib(
                instruments=instruments,
                start=start_date,
                end=end_date,
                factor_expr=factor_expr,
                label_expr=label_expr
            )
            
            # 运行IC分析
            result = run_ic_pipeline(df=df, factor_col="factor", label_col="label", quantiles=5, method="spearman")
        
        st.success("✅ 深度分析完成")
        
        # 1. 统计摘要
        st.markdown("### 📊 统计摘要")
        display_stats_summary(result.stats)
        
        st.markdown("---")
        
        # 2. IC时间序列 + 累积IC
        st.markdown("### 📈 IC时间序列与累积IC")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ts = plot_ic_timeseries(result.ic_series)
            st.plotly_chart(fig_ts, use_container_width=True)
        
        with col2:
            fig_cum = plot_cumulative_ic(result.ic_series)
            st.plotly_chart(fig_cum, use_container_width=True)
        
        st.markdown("---")
        
        # 3. 月度IC热力图
        st.markdown("### 🔥 月度IC热力图")
        if len(result.monthly_ic) > 0:
            fig_heatmap = plot_monthly_ic_heatmap(result.monthly_ic)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with st.expander("📋 查看月度IC数据"):
                st.dataframe(result.monthly_ic, use_container_width=True)
        else:
            st.warning("⚠️ 数据不足以生成月度热力图")
        
        st.markdown("---")
        
        # 4. IC滚动统计
        st.markdown("### 📊 IC滚动统计")
        fig_rolling = plot_ic_rolling_stats(result.ic_series, window=rolling_window)
        st.plotly_chart(fig_rolling, use_container_width=True)
        
        st.markdown("---")
        
        # 5. 分层收益
        st.markdown("### 🎯 分层收益分析")
        if len(result.layered_returns) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_layer = plot_layered_returns(result.layered_returns)
                st.plotly_chart(fig_layer, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 分层详情")
                st.dataframe(result.layered_returns, use_container_width=True, hide_index=True)
                
                # 多空收益
                if "long_short" in result.layered_returns.columns:
                    ls_ret = result.layered_returns["long_short"].iloc[0]
                    st.metric("多空收益", f"{ls_ret:.2%}")
        
        st.markdown("---")
        
        # 6. IC分布
        st.markdown("### 📊 IC分布直方图")
        fig_dist = plot_ic_distribution(result.ic_series)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # 下载数据
        st.markdown("---")
        st.markdown("### 💾 导出数据")
        
        col1, col2 = st.columns(2)
        with col1:
            csv_ic = result.ic_series.to_csv()
            st.download_button(
                label="📥 下载IC时间序列(CSV)",
                data=csv_ic,
                file_name=f"ic_series_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        
        with col2:
            if len(result.layered_returns) > 0:
                csv_layer = result.layered_returns.to_csv(index=False)
                st.download_button(
                    label="📥 下载分层收益(CSV)",
                    data=csv_layer,
                    file_name=f"layered_returns_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
        
    except Exception as e:
        st.error(f"❌ 深度分析失败: {str(e)}")
        with st.expander("🔍 查看详细错误"):
            st.code(traceback.format_exc())


# ==================== 使用指南 ====================

def render_user_guide():
    """渲染使用指南"""
    st.header("📚 IC分析使用指南")
    
    st.markdown("""
    ## 什么是IC？
    
    **IC (Information Coefficient)** 是量化因子分析中最重要的指标，用于衡量**因子对未来收益的预测能力**。
    
    ### 📊 核心概念
    
    - **IC**: 因子值与未来收益的相关系数（Spearman或Pearson）
    - **Rank IC**: 使用Spearman秩相关（更稳健，推荐使用）
    - **ICIR**: IC的信息比率，衡量IC的稳定性 (IC均值 / IC标准差)
    - **分层收益**: 将股票按因子值分为N层，观察各层的平均收益
    
    ---
    
    ## 💡 如何评估IC
    
    ### 1. IC均值（ic_mean）
    
    | IC均值 | 评价 | 说明 |
    |--------|------|------|
    | > 0.05 | 🟢 优秀 | 因子有显著预测能力 |
    | 0.03 - 0.05 | 🟡 良好 | 可以考虑使用 |
    | 0.01 - 0.03 | 🟠 一般 | 需要优化或组合 |
    | < 0.01 | 🔴 较差 | 建议重新设计 |
    
    ### 2. ICIR（IC信息比率）
    
    | ICIR | 评价 | 说明 |
    |------|------|------|
    | > 1.0 | 🟢 优秀 | IC非常稳定 |
    | 0.5 - 1.0 | 🟡 良好 | IC较稳定 |
    | < 0.5 | 🔴 不稳定 | 可能过拟合 |
    
    ### 3. IC正比例（pos_rate）
    
    - **> 60%**: IC大部分时间为正，因子持续有效
    - **50% - 60%**: 因子中性偏正
    - **< 50%**: 因子表现不稳定
    
    ### 4. 分层收益单调性
    
    - **理想情况**: Q1 < Q2 < Q3 < Q4 < Q5 （严格单调递增）
    - **可接受**: 整体趋势递增，允许小幅波动
    - **不理想**: 无明显单调性，因子无效
    
    ---
    
    ## 🔬 快速分析 vs 深度分析
    
    ### 快速分析 🔬
    
    **适用场景**:
    - 快速评估新因子
    - 初步筛选候选因子
    - 日常因子监控
    
    **提供指标**:
    - IC时间序列
    - IC统计摘要
    - 分层收益
    - IC分布
    
    ### 深度分析 📈
    
    **适用场景**:
    - 因子深入研究
    - 论文写作
    - 正式因子评审
    
    **额外提供**:
    - 月度IC热力图（查看季节性）
    - IC滚动统计（查看稳定性变化）
    - 累积IC曲线（查看长期趋势）
    - 数据导出功能
    
    ---
    
    ## 📝 Qlib因子表达式示例
    
    ### 动量类因子
    ```python
    # 5日收益率
    "Ref($close, 0) / Ref($close, 5) - 1"
    
    # 20日收益率
    "Ref($close, 0) / Ref($close, 20) - 1"
    
    # 成交量加权收益
    "($close / Ref($close, 1) - 1) * Log($volume + 1)"
    ```
    
    ### 反转类因子
    ```python
    # 5日反转
    "-(Ref($close, 0) / Ref($close, 5) - 1)"
    
    # 隔夜收益
    "$open / Ref($close, 1) - 1"
    ```
    
    ### 波动率因子
    ```python
    # 20日收益率标准差
    "Std($close / Ref($close, 1) - 1, 20)"
    
    # 振幅
    "($high - $low) / $open"
    ```
    
    ### 量价因子
    ```python
    # 相对成交量
    "$volume / Mean($volume, 20)"
    
    # 价量相关性
    "Corr($close, $volume, 20)"
    ```
    
    ---
    
    ## ⚠️ 常见问题
    
    ### Q1: IC为负数怎么办？
    
    **A**: 负IC表示因子与收益负相关，可以：
    1. 检查标签定义是否正确
    2. 尝试取反因子（加负号）
    3. 检查是否有未来函数（数据泄露）
    
    ### Q2: IC不稳定怎么办？
    
    **A**: ICIR < 0.5表示不稳定，可以：
    1. 增加数据样本量
    2. 调整因子计算窗口
    3. 与其他因子组合
    4. 使用机器学习模型
    
    ### Q3: 分层收益不单调怎么办？
    
    **A**: 可能原因：
    1. 因子无效或过拟合
    2. 样本不足
    3. 标签定义不当
    4. 极端值影响
    
    **解决方案**: 中性化、去极值、分行业分析
    
    ### Q4: 如何提高IC？
    
    **A**: 
    1. **特征工程**: 尝试非线性变换、交叉特征
    2. **因子组合**: 多因子加权平均
    3. **行业中性化**: 去除行业效应
    4. **动态调整**: 根据市场状态切换因子
    
    ---
    
    ## 📚 延伸阅读
    
    - [Qlib官方文档](https://qlib.readthedocs.io/)
    - [因子动物园](https://www.quantopian.com/posts/the-factor-zoo)
    - [Alpha101因子](https://arxiv.org/abs/1601.00991)
    
    ---
    
    ## 💡 最佳实践
    
    1. **先快后深**: 快速分析筛选 → 深度分析精研
    2. **多时期验证**: 测试不同时间段的稳定性
    3. **行业分析**: 分行业查看IC表现
    4. **组合优化**: 单因子IC低不要紧，关键看组合效果
    5. **持续监控**: 定期更新IC，及时发现因子衰减
    """)


# ==================== 主入口 ====================

if __name__ == "__main__":
    render_qlib_ic_analysis_tab()
