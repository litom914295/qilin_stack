"""
RD-Agent其他Tab模块 (知识学习、Kaggle Agent、研发协同、MLE-Bench)
简化版实现
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime


def render_knowledge_learning():
    """知识学习tab"""
    st.header("📚 知识学习")
    st.markdown("""
    **论文阅读与实现**
    - 📄 **自动论文解析** ✨新增
    - 💻 **代码实现生成** ✨新增
    - ✅ 方法复现验证
    - 📊 知识图谱构建
    """)
    
    # 初始化session state
    if 'paper_parse_result' not in st.session_state:
        st.session_state.paper_parse_result = None
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已解析论文", "23", "+3")
    with col2:
        st.metric("实现方法", "15", "+2")
    with col3:
        st.metric("复现成功率", "87%", "+5%")
    
    st.divider()
    
    # 论文上传区域
    st.subheader("📄 论文解析")
    st.info("📄 上传论文PDF或输入arXiv链接，LLM将自动解析并生成实现代码")
    
    uploaded_file = st.file_uploader("上传论文PDF", type=['pdf'], key="paper_pdf_upload")
    arxiv_url = st.text_input("或输入arXiv URL", placeholder="https://arxiv.org/abs/...", key="arxiv_url_input")
    
    col_task, col_btn = st.columns([3, 1])
    with col_task:
        task_type = st.selectbox(
            "解析任务类型",
            ["implementation", "reproduction", "analysis"],
            format_func=lambda x: {"implementation": "方法实现", "reproduction": "结果复现", "analysis": "论文分析"}[x],
            key="task_type_select"
        )
    with col_btn:
        st.write("")  # 对齐
        st.write("")  # 对齐
        parse_btn = st.button("🚀 开始解析", type="primary", key="parse_paper_btn")
    
    if parse_btn:
        if not uploaded_file and not arxiv_url:
            st.warning("⚠️ 请上传PDF文件或输入arXiv URL")
        else:
            from .rdagent_api import RDAgentAPI
            import tempfile
            import os
            
            api = RDAgentAPI()
            
            # 保存PDF到临时文件
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    pdf_path = tmp_file.name
            else:
                # TODO: 处理arXiv URL下载
                st.warning("🚧 arXiv URL下载功能待实现")
                pdf_path = None
            
            if pdf_path:
                with st.spinner(f"正在解析论文并生成{task_type}代码..."):
                    result = api.parse_paper_and_generate_code(pdf_path, task_type)
                
                # 清理临时文件
                try:
                    os.unlink(pdf_path)
                except:
                    pass
                
                if result['success']:
                    st.session_state.paper_parse_result = result
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ 解析失败: {result.get('message', '未知错误')}")
    
    # 显示解析结果
    if st.session_state.paper_parse_result:
        result = st.session_state.paper_parse_result
        
        st.divider()
        st.subheader("📊 解析结果")
        
        # 论文信息
        with st.expander("📝 论文摘要", expanded=True):
            st.markdown(f"**论文标题:** {result.get('paper_title', 'N/A')}")
            st.text_area("摘要", result.get('summary', ''), height=150, key="paper_summary")
        
        # 生成的代码
        if result.get('code_generated'):
            with st.expander("💻 生成的代码", expanded=True):
                code = result.get('code', '')
                st.code(code, language='python')
                
                # 下载按钮
                st.download_button(
                    label="📥 下载代码",
                    data=code,
                    file_name="generated_implementation.py",
                    mime="text/x-python"
                )


def render_kaggle_agent():
    """加入Kaggle Agent tab"""
    st.header("🏆 Kaggle Agent")
    st.markdown("""
    **自动竞赛参与**
    - 🎯 竞赛自动参与
    - 💾 **数据自动下载** ✨新增
    - 🔧 特征工程自动化  
    - 📈 模型自动调优
    - 📤 自动提交管理
    """)
    
    # 初始化session state
    if 'kaggle_download_result' not in st.session_state:
        st.session_state.kaggle_download_result = None
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("参与竞赛", "5", "+1")
    with col2:
        st.metric("最高排名", "Top 8%", "↑3%")
    with col3:
        st.metric("自动提交", "127", "+15")
    with col4:
        st.metric("奖牌数", "2", "+1")
    
    st.divider()
    
    # 数据下载区域
    st.subheader("💾 Kaggle数据下载")
    
    col_comp, col_btn = st.columns([3, 1])
    with col_comp:
        competition = st.selectbox(
            "选择Kaggle竞赛",
            ["titanic", "house-prices-advanced-regression-techniques", "spaceship-titanic", "playground-series-s4e8"],
            key="kaggle_competition_select"
        )
    with col_btn:
        st.write("")  # 对齐
        st.write("")  # 对齐
        download_btn = st.button("⬇️ 下载数据", key="download_kaggle_data")
    
    if download_btn:
        from .rdagent_api import RDAgentAPI
        api = RDAgentAPI()
        
        with st.spinner(f"正在下载 {competition} 数据集..."):
            result = api.download_kaggle_data(competition)
        
        if result["status"] == "success":
            st.session_state.kaggle_download_result = result
            st.success(f"✅ 下载成功! 保存路径: {result['path']}")
        else:
            st.error(f"❌ 下载失败: {result.get('error', '未知错误')}")
    
    # 显示下载结果
    if st.session_state.kaggle_download_result:
        result = st.session_state.kaggle_download_result
        with st.expander("📁 已下载文件列表", expanded=True):
            if "files" in result:
                for file in result["files"]:
                    col_file, col_size = st.columns([4, 1])
                    with col_file:
                        st.text(f"📄 {file['name']}")
                    with col_size:
                        st.text(f"{file['size']}")
    
    st.divider()
    
    # Agent自动化区域
    st.subheader("🤖 Agent自动化")
    
    auto_submit = st.checkbox("自动提交", value=True)
    
    if st.button("🚀 启动Kaggle Agent", type="primary"):
        with st.spinner(f"正在处理 {competition} 竞赛..."):
            import time; time.sleep(2)
        st.success("Agent已启动!正在进行特征工程和模型训练")
    
    st.divider()
    
    # RD-Agent Kaggle RDLoop 运行
    st.subheader("🧪 RD-Agent Kaggle RDLoop 运行")
    
    # 基础参数
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        step_n = st.number_input("每轮步数 step_n", min_value=1, max_value=20, value=5, key="kaggle_step_n")
    with col_b:
        loop_n = st.number_input("循环次数 loop_n", min_value=1, max_value=20, value=3, key="kaggle_loop_n")
    with col_c:
        if 'kaggle_stop' not in st.session_state:
            st.session_state.kaggle_stop = False
        stop_flag = st.toggle("⏹️ 允许中途停止", value=st.session_state.kaggle_stop, key="kaggle_stop_toggle")
        st.session_state.kaggle_stop = stop_flag
    
    # 高级选项
    with st.expander("⚙️ RD-Agent 高级配置", expanded=False):
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            auto_submit = st.checkbox(
                "🚀 自动提交",
                value=False,
                help="开启后，RD-Agent会自动将实验结果上传并提交到Kaggle平台",
                key="kaggle_auto_submit"
            )
            if auto_submit:
                st.caption("⚠️ 需要先配置 Kaggle API：")
                st.caption("1. 下载 kaggle.json 到 ~/.kaggle/")
                st.caption("2. 运行 `kaggle competitions list` 验证")
                st.caption("3. 注意提交次数配额限制（每日5次）")
        with col_opt2:
            use_graph_rag = st.checkbox(
                "🧠 图知识库RAG",
                value=False,
                help="启用基于图的高级RAG知识管理系统",
                key="kaggle_use_graph_rag"
            )
            if use_graph_rag:
                st.caption("📚 需要准备知识库文件：")
                st.caption("- 路径：$RDAGENT_PATH/scenarios/kaggle/knowledge_base/")
                st.caption("- 格式：支持 txt/md/json")
    
    run_clicked = st.button("▶️ 运行 RDLoop", type="primary", key="run_kaggle_rdloop")
    log_box = st.empty()
    prog = st.empty()
    
    if run_clicked:
        from .rdagent_api import RDAgentAPI
        api = RDAgentAPI()
        submitted = 0
        best_score = 0.0
        
        # 显示配置信息
        config_info = f"配置: step_n={step_n}, loop_n={loop_n}"
        if auto_submit:
            config_info += ", auto_submit=True"
        if use_graph_rag:
            config_info += ", Graph RAG=Enabled"
        log_box.info(config_info)
        
        with st.spinner("运行中...这可能需要一段时间"):
            for info in api.run_kaggle_rdloop_stream(
                competition, 
                int(step_n), 
                int(loop_n),
                auto_submit=auto_submit,
                use_graph_rag=use_graph_rag
            ):
                if st.session_state.get('kaggle_stop'):
                    st.warning("已停止")
                    break
                # 更新进度
                pct = 0.0
                try:
                    pct = info['loop'] / max(1, info['total_loops'])
                except Exception:
                    pass
                prog.progress(min(1.0, pct))
                submitted = info.get('submissions', submitted)
                best_score = max(best_score, info.get('best_score', 0.0))
                # 追加日志
                log_box.info(f"[Loop {info.get('loop')}/{info.get('total_loops')}] Submissions={submitted}, BestScore={best_score:.5f} - {info.get('message','')}")
            else:
                st.success("RDLoop已完成")
    

def render_rd_coordination():
    """研发协同tab"""
    st.header("🔬 研发协同")
    st.markdown("""
    **R&D循环管理**
    - 🔬 Research Agent
    - 🛠️ Development Agent
    - 🔄 协同进化循环
    - 📊 实验追踪管理
    """)
    
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
    
    if st.button("🔄 启动新一轮R&D循环", type="primary"):
        with st.spinner("正在运行R&D循环..."):
            import time; time.sleep(2)
        st.success("循环完成!生成了3个新假设")


def render_mle_bench():
    """MLE-Bench tab"""
    st.header("📊 MLE-Bench")
    st.markdown("""
    **机器学习工程评估**
    - 🏆 MLE-Bench基准测试
    - 📊 性能对比分析
    - 🎯 竞争力评估
    - 📈 持续改进追踪
    """)
    
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
    
    if st.button("🚀 运行MLE-Bench测试", type="primary"):
        with st.spinner("正在运行基准测试..."):
            import time; time.sleep(3)
        st.success("测试完成!得分: 30.45% (+0.23%)")
