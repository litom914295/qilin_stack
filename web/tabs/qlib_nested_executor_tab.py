"""
Qlib 嵌套执行器 (NestedExecutor) Tab
任务 10: 嵌套执行(NestedExecutor)样例与 UI 操作流

功能:
1. 三级嵌套决策 (日级/小时级/分钟级)
2. 订单智能拆分 (TWAP/VWAP/POV)
3. 市场冲击成本模拟
4. 滑点模拟
5. 一键运行嵌套回测
6. 可视化多层级绩效对比
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import logging
import json
from datetime import datetime

# 导入配置中心
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.qlib_config_center import init_qlib, check_qlib_connection

logger = logging.getLogger(__name__)


def render():
    """渲染嵌套执行器标签页"""
    st.title("🔀 Qlib 嵌套执行器 (NestedExecutor)")
    st.markdown("""
    **嵌套执行器**用于多级决策框架,支持:
    - 📊 **三级时间粒度**: 日级策略 → 小时级拆单 → 分钟级执行
    - 💰 **成本模拟**: 市场冲击成本 + 滑点
    - ⚡ **智能拆单**: TWAP / VWAP / POV 策略
    - 📈 **多层级回测**: 对比不同时间粒度的绩效
    
    ---
    """)
    
    # 初始化检查
    connected, info = check_qlib_connection()
    if not connected:
        st.warning("⚠️ Qlib 未初始化")
        if st.button("🚀 初始化 Qlib"):
            with st.spinner("正在初始化 Qlib..."):
                success, msg = init_qlib(mode="auto")
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
                    return
    else:
        st.success(f"✅ Qlib 已连接 | 版本: {info.get('version', '未知')}")
    
    # 主界面
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ 配置", 
        "🚀 运行", 
        "📊 结果分析", 
        "📖 文档"
    ])
    
    with tab1:
        render_config_tab()
    
    with tab2:
        render_run_tab()
    
    with tab3:
        render_results_tab()
    
    with tab4:
        render_docs_tab()


def render_config_tab():
    """渲染配置标签页"""
    st.header("⚙️ 嵌套执行器配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 数据配置")
        
        market = st.selectbox(
            "市场",
            options=["csi300", "csi500", "csi800"],
            index=0,
            help="回测标的池"
        )
        
        benchmark = st.text_input(
            "基准指数",
            value="SH000300",
            help="沪深300指数"
        )
        
        start_date = st.date_input(
            "回测开始日期",
            value=pd.to_datetime("2020-09-20")
        )
        
        end_date = st.date_input(
            "回测结束日期",
            value=pd.to_datetime("2021-05-20")
        )
        
        initial_cash = st.number_input(
            "初始资金 (元)",
            min_value=1000000,
            value=100000000,
            step=10000000,
            help="回测起始资金"
        )
    
    with col2:
        st.subheader("🔀 嵌套层级配置")
        
        level1_freq = st.selectbox(
            "Level 1 频率 (外层)",
            options=["1d", "1day", "day"],
            index=0,
            help="日级策略,组合配置决策"
        )
        
        level2_freq = st.selectbox(
            "Level 2 频率 (中层)",
            options=["30min", "1h", "2h"],
            index=0,
            help="小时级策略,订单生成与拆分"
        )
        
        level3_freq = st.selectbox(
            "Level 3 频率 (内层)",
            options=["1min", "5min", "15min"],
            index=1,
            help="分钟级执行,订单撮合"
        )
        
        st.markdown("---")
        
        st.subheader("📦 订单拆分策略")
        split_strategy = st.selectbox(
            "拆分策略",
            options=["TWAP", "VWAP", "POV"],
            index=0,
            help="TWAP: 时间均匀拆分\nVWAP: 按成交量权重拆分\nPOV: 按参与率拆分"
        )
        
        max_participation = st.slider(
            "最大市场参与率",
            min_value=0.01,
            max_value=0.30,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="单笔订单占日成交量的最大比例"
        )
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("💰 成本模型配置")
        
        permanent_impact = st.slider(
            "永久冲击系数",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Almgren-Chriss 模型永久冲击参数"
        )
        
        temporary_impact = st.slider(
            "临时冲击系数",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="短期价格压力参数"
        )
        
        base_slippage = st.slider(
            "基础滑点 (bps)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="基础滑点 (1 bps = 0.01%)"
        )
    
    with col4:
        st.subheader("💼 策略配置")
        
        topk = st.number_input(
            "持仓数量 (TopK)",
            min_value=1,
            max_value=100,
            value=50,
            help="每日持仓股票数量"
        )
        
        n_drop = st.number_input(
            "换手控制 (N-Drop)",
            min_value=0,
            max_value=20,
            value=5,
            help="每日最多调仓股票数"
        )
        
        open_cost = st.number_input(
            "开仓手续费 (bps)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.1,
            help="买入成本 (1 bps = 0.01%)"
        )
        
        close_cost = st.number_input(
            "平仓手续费 (bps)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.1,
            help="卖出成本 (含印花税)"
        )
    
    st.markdown("---")
    
    # 保存配置按钮
    col_save, col_load = st.columns(2)
    
    with col_save:
        if st.button("💾 保存配置", use_container_width=True):
            config = {
                "market": market,
                "benchmark": benchmark,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_cash": initial_cash,
                "level1_freq": level1_freq,
                "level2_freq": level2_freq,
                "level3_freq": level3_freq,
                "split_strategy": split_strategy,
                "max_participation": max_participation,
                "permanent_impact": permanent_impact,
                "temporary_impact": temporary_impact,
                "base_slippage": base_slippage / 10000,  # 转换为比例
                "topk": topk,
                "n_drop": n_drop,
                "open_cost": open_cost / 10000,
                "close_cost": close_cost / 10000
            }
            
            # 保存到 session_state
            st.session_state['nested_executor_config'] = config
            
            # 保存到文件
            config_path = project_root / "configs" / "nested_executor_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ 配置已保存到: {config_path}")
    
    with col_load:
        config_path = project_root / "configs" / "nested_executor_config.json"
        if config_path.exists():
            if st.button("📂 加载配置", use_container_width=True):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                st.session_state['nested_executor_config'] = config
                st.success("✅ 配置已加载")
                st.rerun()


def render_run_tab():
    """渲染运行标签页"""
    st.header("🚀 运行嵌套回测")
    
    # 检查配置
    if 'nested_executor_config' not in st.session_state:
        st.warning("⚠️ 请先在【配置】页面设置参数")
        return
    
    config = st.session_state['nested_executor_config']
    
    # 显示配置摘要
    with st.expander("📋 配置摘要", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("市场", config['market'])
            st.metric("初始资金", f"{config['initial_cash']/1e8:.1f} 亿")
        with col2:
            st.metric("回测区间", f"{config['start_date']} ~ {config['end_date']}")
            st.metric("嵌套层级", f"{config['level1_freq']} → {config['level2_freq']} → {config['level3_freq']}")
        with col3:
            st.metric("持仓数量", config['topk'])
            st.metric("拆分策略", config['split_strategy'])
    
    st.markdown("---")
    
    # 运行模式选择
    run_mode = st.radio(
        "运行模式",
        options=["仅嵌套回测", "嵌套 + 单层对比", "快速模拟 (测试)"],
        index=0,
        horizontal=True,
        help="仅嵌套回测: 运行三层嵌套\n嵌套 + 单层对比: 同时运行单层日级回测作为对比\n快速模拟: 使用本地模拟器快速测试"
    )
    
    # 运行按钮
    if st.button("🚀 开始运行", type="primary", use_container_width=True):
        st.info("🔧 该功能需要:\n1. Qlib 高频数据 (1min)\n2. 模型训练完成\n3. 大约 10-30 分钟运行时间")
        
        if run_mode == "快速模拟 (测试)":
            with st.spinner("正在运行快速模拟..."):
                run_quick_simulation(config)
        else:
            st.warning("⚠️ 完整嵌套回测需要高频数据,请确保已下载 1min 级别数据")
            st.code("""
# 下载高频数据
python scripts/get_data.py qlib_data --name qlib_data_1min --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

# 或使用 UI 的【数据工具】页面下载
            """, language="bash")


def run_quick_simulation(config):
    """运行快速模拟 (使用本地模拟器)"""
    try:
        from qlib_enhanced.nested_executor_integration import create_production_executor
        
        # 创建执行器
        executor = create_production_executor({
            'impact_model_config': {
                'permanent_impact': config['permanent_impact'],
                'temporary_impact': config['temporary_impact']
            },
            'slippage_model_config': {
                'base_slippage': config['base_slippage']
            },
            'order_splitter_config': {
                'strategy': config['split_strategy'].lower(),
                'max_participation_rate': config['max_participation']
            }
        })
        
        # 模拟 10 笔订单
        st.subheader("📊 模拟订单执行")
        
        progress_bar = st.progress(0)
        results = []
        
        import numpy as np
        for i in range(10):
            order = {
                'symbol': f'00000{i%5 + 1}.SZ',
                'size': np.random.randint(5000, 50000),
                'side': np.random.choice(['buy', 'sell']),
                'price': 10.0 + np.random.randn() * 0.5
            }
            
            market_data = {
                'daily_volume': 5000000,
                'volatility': 0.02 + np.random.rand() * 0.01,
                'current_price': order['price']
            }
            
            result = executor.simulate_order_execution(order, market_data)
            results.append(result)
            
            progress_bar.progress((i + 1) / 10)
        
        # 显示结果
        st.success("✅ 模拟完成!")
        
        # 统计信息
        stats = executor.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总订单数", stats['total_orders'])
        with col2:
            st.metric("总成本", f"{stats['total_cost']:.2f} 元")
        with col3:
            st.metric("平均冲击成本", f"{stats['avg_impact_cost']:.2f} 元")
        with col4:
            st.metric("平均滑点成本", f"{stats['avg_slippage_cost']:.2f} 元")
        
        # 结果表格
        st.subheader("📋 执行详情")
        results_df = pd.DataFrame([
            {
                '股票': r['symbol'],
                '成交量': r['filled_size'],
                '成交价': f"{r['avg_price']:.4f}",
                '基准价': f"{r['benchmark_price']:.4f}",
                '冲击成本': f"{r['impact_cost']:.2f}",
                '滑点成本': f"{r['slippage_cost']:.2f}",
                '总成本': f"{r['total_cost']:.2f}",
                '执行质量': f"{r['execution_quality']:.2%}"
            }
            for r in results
        ])
        st.dataframe(results_df, use_container_width=True)
        
        # 可视化
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['成本分布', '执行质量分布', '订单量分布', '成本占比']
        )
        
        # 成本分布
        fig.add_trace(
            go.Bar(x=[r['symbol'] for r in results],
                   y=[r['total_cost'] for r in results],
                   name='总成本'),
            row=1, col=1
        )
        
        # 执行质量
        fig.add_trace(
            go.Scatter(x=list(range(len(results))),
                      y=[r['execution_quality'] for r in results],
                      mode='lines+markers',
                      name='执行质量'),
            row=1, col=2
        )
        
        # 订单量
        fig.add_trace(
            go.Histogram(x=[r['filled_size'] for r in results],
                        name='订单量'),
            row=2, col=1
        )
        
        # 成本占比
        fig.add_trace(
            go.Pie(labels=['冲击成本', '滑点成本'],
                   values=[stats['total_impact_cost'], stats['total_slippage_cost']]),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 保存到 session_state
        st.session_state['nested_executor_results'] = {
            'results': results,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        st.error(f"❌ 模拟失败: {e}")
        logger.error(f"Quick simulation failed: {e}", exc_info=True)


def render_results_tab():
    """渲染结果分析标签页"""
    st.header("📊 结果分析")
    
    if 'nested_executor_results' not in st.session_state:
        st.info("ℹ️ 暂无结果,请先在【运行】页面执行回测")
        return
    
    results_data = st.session_state['nested_executor_results']
    stats = results_data['stats']
    
    st.success(f"✅ 结果加载成功 | 生成时间: {results_data['timestamp']}")
    
    # 关键指标
    st.subheader("📈 关键指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("订单数量", stats['total_orders'])
    with col2:
        st.metric("总成本", f"{stats['total_cost']:.2f} 元")
    with col3:
        st.metric("平均冲击成本", f"{stats['avg_impact_cost']:.2f} 元")
    with col4:
        st.metric("平均滑点成本", f"{stats['avg_slippage_cost']:.2f} 元")
    with col5:
        st.metric("平均执行质量", f"{stats['avg_execution_quality']:.2%}")
    
    st.markdown("---")
    
    # 详细结果表格
    st.subheader("📋 执行详情")
    results = results_data['results']
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # 下载按钮
    csv = results_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下载 CSV",
        data=csv,
        file_name=f"nested_executor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_docs_tab():
    """渲染文档标签页"""
    st.header("📖 文档")
    
    st.markdown("""
    ## 嵌套执行器原理
    
    ### 1. 三级决策架构
    
    ```
    Level 1 (日级)          Level 2 (小时级)        Level 3 (分钟级)
    ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
    │ 组合配置    │  →→→  │ 订单拆分    │  →→→  │ 订单撮合    │
    │ 策略决策    │        │ TWAP/VWAP   │        │ 市场冲击    │
    │ TopK-Dropout│        │ 风险控制    │        │ 滑点模拟    │
    └─────────────┘        └─────────────┘        └─────────────┘
    ```
    
    ### 2. 市场冲击成本模型 (Almgren-Chriss)
    
    **永久冲击 (Permanent Impact)**:
    ```
    I_perm = γ × (V/ADV) × P × V
    ```
    
    **临时冲击 (Temporary Impact)**:
    ```
    I_temp = η × √(V/ADV) × P × V
    ```
    
    其中:
    - V: 订单量 (股数)
    - ADV: 平均日成交量
    - P: 当前价格
    - γ: 永久冲击系数 (默认 0.1)
    - η: 临时冲击系数 (默认 0.01)
    
    ### 3. 订单拆分策略
    
    #### TWAP (Time Weighted Average Price)
    - 时间均匀拆分
    - 适用于流动性充足、价格平稳的股票
    
    #### VWAP (Volume Weighted Average Price)
    - 按历史成交量权重拆分
    - 适用于跟踪市场节奏
    
    #### POV (Percentage of Volume)
    - 按市场参与率拆分
    - 适用于大单执行
    
    ### 4. 使用场景
    
    #### 场景 1: 一进二涨停策略
    - **Level 1 (日级)**: 筛选涨停开板股票
    - **Level 2 (小时级)**: 监控开板时点,拆分订单
    - **Level 3 (分钟级)**: 开板瞬间快速成交
    
    #### 场景 2: 大单执行
    - **Level 1**: 决定买入量
    - **Level 2**: TWAP 均匀拆单
    - **Level 3**: 控制市场冲击
    
    ### 5. 参考资料
    
    - [Qlib 官方文档 - NestedExecutor](https://qlib.readthedocs.io/en/latest/)
    - [Almgren & Chriss (2000) - Optimal Execution](https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf)
    - 麒麟项目: `qlib_enhanced/nested_executor_integration.py`
    
    ---
    
    ## 常见问题
    
    **Q: 为什么需要高频数据?**  
    A: Level 3 (分钟级) 需要 1min 或 5min 数据来模拟真实执行过程。
    
    **Q: 如何下载高频数据?**  
    A: 使用【数据工具】页面或运行:
    ```bash
    python scripts/get_data.py qlib_data --interval 1min
    ```
    
    **Q: 执行时间多久?**  
    A: 完整回测约 10-30 分钟 (取决于数据量和嵌套层级)。
    
    **Q: 如何优化性能?**  
    A: 
    1. 减少回测区间
    2. 减少持仓数量 (TopK)
    3. 使用 expression_cache 和 dataset_cache
    """)


if __name__ == "__main__":
    render()
