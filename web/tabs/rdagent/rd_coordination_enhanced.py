"""
R&D循环和MLE-Bench增强模块
完整的Research/Development阶段展示和Trace历史查询
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime


def render_rd_coordination_enhanced():
    """研发协同tab - 增强版"""
    st.header("🔬 研发协同")
    st.markdown("""
    **R&D循环管理** ✨增强版
    - 🔬 **Research Agent阶段展示**
    - 🛠️ **Development Agent阶段展示**
    - 📜 **Trace历史查询** ✨新增
    - 🔄 协同进化循环
    """)
    
    # 初始化session state
    if 'rd_loop_running' not in st.session_state:
        st.session_state.rd_loop_running = False
    if 'rd_loop_result' not in st.session_state:
        st.session_state.rd_loop_result = None
    if 'rd_trace_history' not in st.session_state:
        st.session_state.rd_trace_history = []
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R&D循环轮次", "12", "+2")
    with col2:
        st.metric("实验总数", "58", "+7")
    with col3:
        st.metric("成功率", "76%", "+8%")
    
    st.divider()
    
    # R&D流程可视化
    st.subheader("📊 R&D循环流程")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=["Research Agent", "Hypothesis", "Experiment", "Development Agent", "Results", "Feedback"],
            color=["blue", "lightblue", "green", "purple", "orange", "red"]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4],
            target=[1, 2, 3, 4, 0],
            value=[10, 10, 10, 10, 10]
        )
    )])
    fig.update_layout(title="R&D循环数据流", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Research阶段详细展示
    st.subheader("🔬 Research Agent 阶段")
    
    research_tab1, research_tab2, research_tab3 = st.tabs([
        "💡 假设生成",
        "📚 文献检索",
        "🧪 实验设计"
    ])
    
    with research_tab1:
        st.markdown("**当前假设列表**")
        
        # Mock假设数据
        hypotheses = [
            {"id": 1, "hypothesis": "动量因子在短期交易中效果显著", "confidence": 0.85, "status": "测试中"},
            {"id": 2, "hypothesis": "成交量与价格背离预示反转", "confidence": 0.72, "status": "已验证"},
            {"id": 3, "hypothesis": "多因子组合可提升预测准确率", "confidence": 0.68, "status": "设计中"}
        ]
        
        for hyp in hypotheses:
            with st.expander(f"💡 假设 #{hyp['id']}: {hyp['hypothesis']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("置信度", f"{hyp['confidence']:.0%}")
                with col2:
                    st.metric("状态", hyp['status'])
    
    with research_tab2:
        st.markdown("**相关文献**")
        
        papers = [
            {"title": "Momentum Strategies in Stock Trading", "year": 2023, "citations": 145},
            {"title": "Volume-Price Divergence Analysis", "year": 2022, "citations": 89},
            {"title": "Multi-Factor Alpha Models", "year": 2024, "citations": 67}
        ]
        
        for paper in papers:
            with st.container():
                col1, col2, col3 = st.columns([5, 1, 1])
                with col1:
                    st.markdown(f"**{paper['title']}**")
                with col2:
                    st.text(paper['year'])
                with col3:
                    st.text(f"📜 {paper['citations']}")
                st.divider()
    
    with research_tab3:
        st.markdown("**实验设计方案**")
        
        exp_design = {
            "数据集": "CSI 300, 2020-2024",
            "训练集": "2020-2023 (70%)",
            "验证集": "2023-2024 (30%)",
            "评估指标": "IC, IR, Sharpe Ratio",
            "基准模型": "Linear Regression, GBDT"
        }
        
        for key, value in exp_design.items():
            st.text(f"{key}: {value}")
    
    st.divider()
    
    # Development阶段详细展示
    st.subheader("🛠️ Development Agent 阶段")
    
    dev_tab1, dev_tab2, dev_tab3 = st.tabs([
        "💻 代码实现",
        "✅ 测试验证",
        "🚀 部署集成"
    ])
    
    with dev_tab1:
        st.markdown("**代码实现进度**")
        
        impl_progress = [
            {"task": "因子计算模块", "progress": 100, "status": "✅ 完成"},
            {"task": "回测引擎", "progress": 75, "status": "🔄 进行中"},
            {"task": "结果分析", "progress": 30, "status": "🚧 待完成"}
        ]
        
        for task in impl_progress:
            st.text(f"{task['status']} {task['task']}")
            st.progress(task['progress'] / 100)
    
    with dev_tab2:
        st.markdown("**测试结果**")
        
        test_results = pd.DataFrame({
            "测试用例": ["Unit Test", "Integration Test", "Performance Test"],
            "通过": [45, 28, 12],
            "失败": [2, 1, 0],
            "覆盖率": ["95%", "87%", "100%"]
        })
        
        st.dataframe(test_results, use_container_width=True, hide_index=True)
    
    with dev_tab3:
        st.markdown("**部署状态**")
        
        deployment_status = [
            {"env": "Development", "version": "v1.2.3", "status": "✅ 正常"},
            {"env": "Staging", "version": "v1.2.2", "status": "🟡 待更新"},
            {"env": "Production", "version": "v1.2.1", "status": "✅ 正常"}
        ]
        
        for dep in deployment_status:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text(dep['env'])
            with col2:
                st.text(dep['version'])
            with col3:
                st.text(dep['status'])
    
    st.divider()
    
    # Trace历史查询
    st.subheader("📜 Trace 历史查询")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        trace_type = st.selectbox("类型过滤", ["All", "Research", "Development", "Experiment"], key="rdc_trace_type")
    with col_filter2:
        trace_status = st.selectbox("状态过滤", ["All", "Success", "Failed", "Running"], key="rdc_trace_status")
    with col_filter3:
        date_range = st.date_input("日期范围", value=(datetime.now().date(), datetime.now().date()))
    
    if st.button("🔍 查询Trace历史"):
        from .rdagent_api import RDAgentAPI
        import asyncio
        
        api = RDAgentAPI()
        
        with st.spinner("正在查询Trace历史..."):
            result = asyncio.run(api.get_rd_loop_trace(
                trace_type=trace_type if trace_type != "All" else None,
                status=trace_status if trace_status != "All" else None
            ))
        
        if result['success']:
            st.session_state.rd_trace_history = result['traces']
            st.success(f"✅ 查询成功! 找到 {len(result['traces'])} 条记录")
        else:
            st.error(f"❌ 查询失败: {result.get('message', '未知错误')}")
    
    # 显示Trace历史
    if st.session_state.rd_trace_history:
        st.markdown("**Trace记录列表**")
        
        for idx, trace in enumerate(st.session_state.rd_trace_history):
            with st.expander(f"📝 Trace #{trace['id']}: {trace['type']} - {trace['timestamp']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("类型", trace['type'])
                with col2:
                    st.metric("状态", trace['status'])
                with col3:
                    st.metric("耗时", f"{trace.get('duration', 0):.1f}s")
                
                st.markdown("**详情:**")
                st.json(trace.get('details', {}))
    
    st.divider()
    
    # 启动R&D循环
    st.subheader("🚀 启动R&D循环")
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        max_iterations = st.number_input("最大迭代次数", min_value=1, max_value=20, value=5)
    with col_config2:
        auto_deploy = st.checkbox("自动部署", value=False)
    
    if st.button("🔄 启动新一轮R&D循环", type="primary"):
        from .rdagent_api import RDAgentAPI
        api = RDAgentAPI()
        
        st.session_state.rd_loop_running = True
        
        with st.spinner(f"正在运行R&D循环 (最大{max_iterations}迭代)..."):
            result = api.run_rd_loop(
                max_iterations=max_iterations,
                auto_deploy=auto_deploy
            )
        
        st.session_state.rd_loop_running = False
        
        if result['success']:
            st.session_state.rd_loop_result = result
            st.success(f"✅ {result['message']}")
            
            # 显示结果
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("生成假设数", result.get('hypotheses_generated', 0))
            with col_res2:
                st.metric("实验次数", result.get('experiments_run', 0))
            with col_res3:
                st.metric("成功率", f"{result.get('success_rate', 0):.0%}")
        else:
            st.error(f"❌ {result.get('message', '运行失败')}")


def render_mle_bench_enhanced():
    """MLE-Bench tab - 增强版"""
    st.header("📊 MLE-Bench")
    st.markdown("""
    **机器学习工程评估** ✨增强版
    - 🏆 **MLE-Bench基准测试**
    - 🎯 **实际运行评估** ✨新增
    - 📊 性能对比分析
    - 📈 持续改进追踪
    """)
    
    # 初始化session state
    if 'mle_bench_running' not in st.session_state:
        st.session_state.mle_bench_running = False
    if 'mle_bench_result' not in st.session_state:
        st.session_state.mle_bench_result = None
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总得分", "30.22%", "+1.5%")
    with col2:
        st.metric("Low难度", "51.52%", "+6.9%")
    with col3:
        st.metric("Medium难度", "19.3%", "+5.5%")
    with col4:
        st.metric("High难度", "26.67%", "0%")
    
    st.divider()
    
    # 性能对比表
    st.subheader("🏆 与竞争对手对比")
    
    comparison_data = {
        "Agent": ["R&D-Agent (Ours)", "AIDE", "Baseline"],
        "Low (%)": [51.52, 34.3, 28.1],
        "Medium (%)": [19.3, 8.8, 5.2],
        "High (%)": [26.67, 10.0, 6.8],
        "All (%)": [30.22, 16.9, 12.3]
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(
        df.style.highlight_max(subset=["Low (%)", "Medium (%)", "High (%)", "All (%)"], color="lightgreen"),
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # 性能趋势
    st.subheader("📈 性能改进趋势")
    
    dates = pd.date_range(end=datetime.now(), periods=10, freq='W')
    scores = [20.1, 22.3, 24.5, 25.8, 27.2, 28.1, 28.9, 29.5, 30.0, 30.22]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode='lines+markers',
        name='MLE-Bench得分',
        line=dict(color='green', width=3),
        marker=dict(size=10)
    ))
    fig.update_layout(
        title="MLE-Bench得分趋势",
        xaxis_title="日期",
        yaxis_title="得分 (%)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 运行配置
    st.subheader("⚙️ 运行配置")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    with col_config1:
        difficulty = st.selectbox("难度级别", ["All", "Low", "Medium", "High"], key="rdc_mle_difficulty")
    with col_config2:
        task_type = st.selectbox("任务类型", ["All", "Classification", "Regression", "Time Series"], key="rdc_mle_task_type")
    with col_config3:
        timeout = st.number_input("超时时间(分钟)", min_value=5, max_value=120, value=30)
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        max_memory = st.number_input("最大内存(GB)", min_value=4, max_value=64, value=16)
    with col_res2:
        num_workers = st.number_input("并行任务数", min_value=1, max_value=16, value=4)
    
    st.divider()
    
    # 启动评估
    st.subheader("🚀 启动评估")
    
    if st.button("🚀 运行MLE-Bench测试", type="primary"):
        from .rdagent_api import RDAgentAPI
        api = RDAgentAPI()
        
        st.session_state.mle_bench_running = True
        
        config = {
            "difficulty": difficulty,
            "task_type": task_type,
            "timeout": timeout * 60,  # 转换为秒
            "max_memory": max_memory * 1024,  # 转换为MB
            "num_workers": num_workers
        }
        
        # 创建进度容器
        progress_container = st.empty()
        status_container = st.empty()
        
        with st.spinner(f"正在运行MLE-Bench评估 (难度: {difficulty})..."):
            result = api.run_mle_bench(config)
        
        st.session_state.mle_bench_running = False
        
        if result['success']:
            st.session_state.mle_bench_result = result
            st.success(f"✅ {result['message']}")
            
            # 显示结果
            st.subheader("📊 评估结果")
            
            col_result1, col_result2, col_result3, col_result4 = st.columns(4)
            with col_result1:
                st.metric("完成任务", result.get('completed_tasks', 0))
            with col_result2:
                st.metric("总得分", f"{result.get('total_score', 0):.2%}")
            with col_result3:
                st.metric("平均耗时", f"{result.get('avg_time', 0):.1f}s")
            with col_result4:
                st.metric("成功率", f"{result.get('success_rate', 0):.0%}")
            
            # 详细结果表
            if 'task_results' in result:
                st.markdown("**任务详情**")
                task_df = pd.DataFrame(result['task_results'])
                st.dataframe(task_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"❌ {result.get('message', '评估失败')}")
    
    # 显示之前的结果
    if st.session_state.mle_bench_result and not st.session_state.mle_bench_running:
        st.divider()
        st.subheader("📜 历史运行结果")
        
        result = st.session_state.mle_bench_result
        with st.expander("查看详细日志", expanded=False):
            if 'logs' in result:
                st.text_area("运行日志", result['logs'], height=300)
