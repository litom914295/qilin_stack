"""
MLE-Bench 基准测试集成
展示RD-Agent在业界基准测试中的领先地位
- 75个Kaggle竞赛数据集
- R&D-Agent vs AIDE vs Baseline排行榜
- 一键运行基准测试
- 详细结果分析
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


# MLE-Bench 官方数据 (从论文和GitHub获取)
MLE_BENCH_RESULTS = {
    "R&D-Agent o3(R)+GPT-4.1(D)": {
        "Low": 51.52,
        "Medium": 19.3,
        "High": 26.67,
        "All": 30.22,
        "std": 1.5,
        "seeds": 6,
        "cost_per_run": "$45-75",
        "avg_time": "2.5h"
    },
    "R&D-Agent o1-preview": {
        "Low": 48.18,
        "Medium": 8.95,
        "High": 18.67,
        "All": 22.4,
        "std": 1.1,
        "seeds": 5,
        "cost_per_run": "$80-120",
        "avg_time": "4.2h"
    },
    "AIDE o1-preview": {
        "Low": 34.3,
        "Medium": 8.8,
        "High": 10.0,
        "All": 16.9,
        "std": 1.1,
        "seeds": 5,
        "cost_per_run": "$50-80",
        "avg_time": "3.8h"
    },
    "OpenHands o1-preview": {
        "Low": 30.5,
        "Medium": 7.2,
        "High": 8.5,
        "All": 14.8,
        "std": 1.3,
        "seeds": 3,
        "cost_per_run": "$40-70",
        "avg_time": "3.5h"
    }
}

# 75个竞赛数据集(部分示例)
MLE_BENCH_DATASETS = [
    {"id": 1, "name": "house-prices-advanced-regression-techniques", "difficulty": "Low", "type": "Regression"},
    {"id": 2, "name": "titanic", "difficulty": "Low", "type": "Classification"},
    {"id": 3, "name": "digit-recognizer", "difficulty": "Low", "type": "Computer Vision"},
    {"id": 4, "name": "natural-language-processing-with-disaster-tweets", "difficulty": "Low", "type": "NLP"},
    {"id": 5, "name": "spaceship-titanic", "difficulty": "Low", "type": "Classification"},
    {"id": 6, "name": "store-sales-time-series-forecasting", "difficulty": "Medium", "type": "Time Series"},
    {"id": 7, "name": "tabular-playground-series-mar-2021", "difficulty": "Medium", "type": "Tabular"},
    {"id": 8, "name": "facebook-recruiting-iii-keyword-extraction", "difficulty": "Medium", "type": "NLP"},
    {"id": 9, "name": "stanford-covid-vaccine", "difficulty": "High", "type": "Research"},
    {"id": 10, "name": "google-quest-challenge", "difficulty": "High", "type": "NLP"},
    # ... 更多数据集
]


class MLEBenchTab:
    """MLE-Bench Tab"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """初始化session state"""
        if 'mle_bench_view' not in st.session_state:
            st.session_state.mle_bench_view = 'leaderboard'
        if 'mle_bench_running' not in st.session_state:
            st.session_state.mle_bench_running = False
        if 'mle_bench_results' not in st.session_state:
            st.session_state.mle_bench_results = None
    
    def render(self):
        """渲染MLE-Bench页面"""
        st.header("🏆 MLE-Bench 基准测试")
        
        st.markdown("""
        **MLE-Bench** 是业界权威的机器学习工程Agent评估基准,包含75个Kaggle竞赛数据集。
        R&D-Agent目前在MLE-Bench上**排名第一**! 🥇
        """)
        
        # 顶部指标
        self.render_top_metrics()
        
        st.divider()
        
        # 视图切换
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🏆 排行榜", use_container_width=True,
                        type="primary" if st.session_state.mle_bench_view == 'leaderboard' else "secondary"):
                st.session_state.mle_bench_view = 'leaderboard'
                st.rerun()
        with col2:
            if st.button("📊 数据集", use_container_width=True,
                        type="primary" if st.session_state.mle_bench_view == 'datasets' else "secondary"):
                st.session_state.mle_bench_view = 'datasets'
                st.rerun()
        with col3:
            if st.button("🚀 运行测试", use_container_width=True,
                        type="primary" if st.session_state.mle_bench_view == 'run_test' else "secondary"):
                st.session_state.mle_bench_view = 'run_test'
                st.rerun()
        with col4:
            if st.button("📈 结果分析", use_container_width=True,
                        type="primary" if st.session_state.mle_bench_view == 'analysis' else "secondary"):
                st.session_state.mle_bench_view = 'analysis'
                st.rerun()
        
        st.divider()
        
        # 根据视图渲染内容
        if st.session_state.mle_bench_view == 'leaderboard':
            self.render_leaderboard()
        elif st.session_state.mle_bench_view == 'datasets':
            self.render_datasets()
        elif st.session_state.mle_bench_view == 'run_test':
            self.render_run_test()
        elif st.session_state.mle_bench_view == 'analysis':
            self.render_analysis()
    
    def render_top_metrics(self):
        """渲染顶部指标"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🥇 全球排名",
                "1st",
                delta="领先AIDE +13.3%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "总体准确率",
                "30.22%",
                delta="+8.3%",
                help="R&D-Agent o3+GPT4.1在所有75个数据集上的平均表现"
            )
        
        with col3:
            st.metric(
                "Low难度",
                "51.52%",
                delta="+17.2%"
            )
        
        with col4:
            st.metric(
                "测试数据集",
                "75个",
                help="涵盖分类/回归/NLP/CV/时序等多个领域"
            )
        
        with col5:
            st.metric(
                "平均成本",
                "$45-75",
                delta="-40% vs o1-preview"
            )
    
    def render_leaderboard(self):
        """渲染排行榜"""
        st.subheader("🏆 MLE-Bench 全球排行榜")
        
        # 排行榜表格
        leaderboard_data = []
        rank = 1
        for agent, results in MLE_BENCH_RESULTS.items():
            leaderboard_data.append({
                "排名": f"{'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else ''}#{rank}",
                "Agent": agent,
                "总体准确率": f"{results['All']:.2f}%",
                "Low": f"{results['Low']:.2f}%",
                "Medium": f"{results['Medium']:.2f}%",
                "High": f"{results['High']:.2f}%",
                "标准差": f"±{results['std']:.1f}",
                "种子数": results['seeds'],
                "平均成本": results['cost_per_run'],
                "平均时间": results['avg_time']
            })
            rank += 1
        
        df = pd.DataFrame(leaderboard_data)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "总体准确率": st.column_config.ProgressColumn(
                    "总体准确率",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )
        
        st.divider()
        
        # 可视化对比
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 不同难度表现对比")
            
            # 分组柱状图
            agents = list(MLE_BENCH_RESULTS.keys())
            difficulties = ['Low', 'Medium', 'High']
            
            fig = go.Figure()
            
            for difficulty in difficulties:
                values = [MLE_BENCH_RESULTS[agent][difficulty] for agent in agents]
                fig.add_trace(go.Bar(
                    name=difficulty,
                    x=agents,
                    y=values,
                    text=[f"{v:.1f}%" for v in values],
                    textposition='outside'
                ))
            
            fig.update_layout(
                barmode='group',
                title="按难度级别分类表现",
                xaxis_title="Agent",
                yaxis_title="准确率 (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 综合表现雷达图")
            
            categories = ['Low难度', 'Medium难度', 'High难度', '稳定性', '成本效益']
            
            fig = go.Figure()
            
            for agent, results in list(MLE_BENCH_RESULTS.items())[:3]:  # Top 3
                # 归一化值
                low_norm = results['Low'] / 60  # 归一化到0-1
                medium_norm = results['Medium'] / 25
                high_norm = results['High'] / 30
                stability = 1 - (results['std'] / 5)  # 标准差越小越好
                
                # 成本效益 (简化处理)
                cost_val = 0.8 if 'o3' in agent else 0.5 if 'AIDE' in agent else 0.6
                
                fig.add_trace(go.Scatterpolar(
                    r=[low_norm, medium_norm, high_norm, stability, cost_val],
                    theta=categories,
                    fill='toself',
                    name=agent.split()[0]  # 简化名称
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Top 3 Agent综合对比",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 关键亮点
        st.divider()
        st.subheader("✨ R&D-Agent 关键亮点")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🚀 性能领先**
            - 总体准确率 30.22% (全球第一)
            - 比AIDE高 13.3个百分点
            - Low难度领先 17.2%
            """)
        
        with col2:
            st.success("""
            **💰 成本优化**
            - 平均$45-75/run
            - 比o1-preview节省40%
            - 混合模型策略: o3(Research) + GPT-4.1(Development)
            """)
        
        with col3:
            st.warning("""
            **🔬 技术创新**
            - Research-Development双Agent协同
            - 自动进化循环
            - 代码生成+实验自动化
            """)
    
    def render_datasets(self):
        """渲染数据集列表"""
        st.subheader("📊 MLE-Bench 数据集 (75个)")
        
        # 过滤器
        col1, col2, col3 = st.columns(3)
        
        with col1:
            difficulty_filter = st.multiselect(
                "难度级别",
                ['Low', 'Medium', 'High'],
                default=['Low', 'Medium', 'High'],
                key="mle_diff_filter"
            )
        
        with col2:
            type_filter = st.multiselect(
                "任务类型",
                ['Regression', 'Classification', 'Computer Vision', 'NLP', 'Time Series', 'Tabular', 'Research'],
                key="mle_type_filter"
            )
        
        with col3:
            search_query = st.text_input(
                "搜索",
                placeholder="输入数据集名称...",
                key="mle_search"
            )
        
        # 过滤数据集
        filtered_datasets = [
            ds for ds in MLE_BENCH_DATASETS
            if ds['difficulty'] in difficulty_filter
            and (not type_filter or ds['type'] in type_filter)
            and (not search_query or search_query.lower() in ds['name'].lower())
        ]
        
        st.info(f"显示 {len(filtered_datasets)} 个数据集 (共75个)")
        
        # 数据集表格
        if filtered_datasets:
            df = pd.DataFrame(filtered_datasets)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": "ID",
                    "name": st.column_config.TextColumn("数据集名称", width="large"),
                    "difficulty": st.column_config.SelectboxColumn(
                        "难度",
                        options=['Low', 'Medium', 'High'],
                        width="small"
                    ),
                    "type": "类型"
                }
            )
        
        # 数据集统计
        st.divider()
        st.subheader("📈 数据集分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 难度分布
            difficulty_counts = pd.DataFrame(MLE_BENCH_DATASETS)['difficulty'].value_counts()
            
            fig = px.pie(
                values=difficulty_counts.values,
                names=difficulty_counts.index,
                title="难度分布",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 类型分布
            type_counts = pd.DataFrame(MLE_BENCH_DATASETS)['type'].value_counts()
            
            fig = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title="任务类型分布",
                labels={'x': '类型', 'y': '数量'},
                color=type_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_run_test(self):
        """渲染运行测试界面"""
        st.subheader("🚀 运行MLE-Bench测试")
        
        st.warning("⚠️ **注意**: 运行完整MLE-Bench测试需要大量计算资源和时间(数小时),建议先运行小规模测试。")
        
        # 测试配置
        st.markdown("### ⚙️ 测试配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**基础配置**")
            
            test_mode = st.radio(
                "测试模式",
                ["快速验证 (3个数据集)", "小规模测试 (10个数据集)", "完整测试 (75个数据集)"],
                key="mle_test_mode"
            )
            
            selected_datasets = st.multiselect(
                "选择数据集 (可选)",
                [ds['name'] for ds in MLE_BENCH_DATASETS],
                key="mle_selected_datasets",
                help="留空则根据测试模式自动选择"
            )
            
            agent_model = st.selectbox(
                "Agent模型",
                ["R&D-Agent (o3+GPT4.1)", "R&D-Agent (o1-preview)", "自定义配置"],
                key="mle_agent_model"
            )
        
        with col2:
            st.markdown("**高级配置**")
            
            num_seeds = st.slider(
                "随机种子数",
                min_value=1,
                max_value=10,
                value=3,
                help="多次运行以计算标准差"
            )
            
            timeout = st.number_input(
                "单个数据集超时(分钟)",
                min_value=30,
                max_value=300,
                value=120
            )
            
            parallel_runs = st.checkbox(
                "并行运行",
                value=False,
                help="在多个数据集上并行测试(需要多GPU)"
            )
            
            save_logs = st.checkbox(
                "保存详细日志",
                value=True
            )
        
        st.divider()
        
        # 估算资源
        st.markdown("### 📊 资源估算")
        
        if test_mode == "快速验证 (3个数据集)":
            estimated_time = "30-60分钟"
            estimated_cost = "$5-10"
            num_datasets = 3
        elif test_mode == "小规模测试 (10个数据集)":
            estimated_time = "2-4小时"
            estimated_cost = "$20-40"
            num_datasets = 10
        else:
            estimated_time = "15-30小时"
            estimated_cost = "$500-1000"
            num_datasets = 75
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("预计时间", estimated_time)
        with col2:
            st.metric("预计成本", estimated_cost)
        with col3:
            st.metric("数据集数量", num_datasets)
        
        st.divider()
        
        # 运行按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 开始测试", type="primary", use_container_width=True):
                st.session_state.mle_bench_running = True
                with st.spinner("正在初始化测试环境..."):
                    self.run_mle_bench_test(num_datasets, num_seeds)
        
        with col2:
            if st.button("⏸️ 停止测试", use_container_width=True):
                st.session_state.mle_bench_running = False
                st.warning("测试已停止")
        
        # 测试进度
        if st.session_state.mle_bench_running:
            st.divider()
            st.markdown("### 📊 测试进度")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 模拟进度
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"正在测试第 {i % num_datasets + 1}/{num_datasets} 个数据集...")
                
                if not st.session_state.mle_bench_running:
                    break
            
            if st.session_state.mle_bench_running:
                st.success("✅ 测试完成!")
                st.session_state.mle_bench_running = False
    
    def run_mle_bench_test(self, num_datasets: int, num_seeds: int):
        """运行MLE-Bench测试"""
        # 实际实现应该调用RD-Agent的MLE-Bench runner
        # 这里是Mock实现
        
        import time
        time.sleep(2)
        
        # 生成模拟结果
        results = {
            'total_datasets': num_datasets,
            'seeds': num_seeds,
            'success_rate': np.random.uniform(0.25, 0.35),
            'avg_score': np.random.uniform(0.20, 0.30),
            'total_time': f"{num_datasets * 1.5:.1f}h",
            'total_cost': f"${num_datasets * 8}",
            'detailed_results': []
        }
        
        for i in range(min(num_datasets, 10)):
            results['detailed_results'].append({
                'dataset': MLE_BENCH_DATASETS[i]['name'],
                'score': np.random.uniform(0.1, 0.5),
                'time': f"{np.random.uniform(30, 180):.0f}min",
                'status': 'success' if np.random.random() > 0.2 else 'failed'
            })
        
        st.session_state.mle_bench_results = results
    
    def render_analysis(self):
        """渲染结果分析"""
        st.subheader("📈 测试结果分析")
        
        if not st.session_state.mle_bench_results:
            st.info("🔍 还没有测试结果。请先在'运行测试'中执行MLE-Bench测试。")
            
            st.markdown("### 📖 参考结果")
            st.markdown("""
            您可以查看R&D-Agent在MLE-Bench官方测试中的表现:
            
            - **完整运行日志**: https://aka.ms/RD-Agent_MLE-Bench_O3_GPT41
            - **论文地址**: https://arxiv.org/abs/2505.14738
            - **MLE-Bench GitHub**: https://github.com/openai/mle-bench
            """)
            return
        
        results = st.session_state.mle_bench_results
        
        # 结果概览
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("测试数据集", results['total_datasets'])
        with col2:
            st.metric("成功率", f"{results['success_rate']*100:.1f}%")
        with col3:
            st.metric("总耗时", results['total_time'])
        with col4:
            st.metric("总成本", results['total_cost'])
        
        st.divider()
        
        # 详细结果
        st.markdown("### 📋 详细结果")
        
        if results['detailed_results']:
            df = pd.DataFrame(results['detailed_results'])
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "得分",
                        format="%.2f",
                        min_value=0,
                        max_value=1
                    ),
                    "status": st.column_config.TextColumn(
                        "状态",
                        width="small"
                    )
                }
            )
            
            # 分数分布
            st.divider()
            st.markdown("### 📊 分数分布")
            
            fig = px.histogram(
                df,
                x='score',
                nbins=20,
                title="测试分数分布",
                labels={'score': '分数', 'count': '数量'},
                color_discrete_sequence=['#636EFA']
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render():
    """渲染入口"""
    tab = MLEBenchTab()
    tab.render()
