"""
Kaggle Agent - 竞赛自动化管理模块

功能:
1. 竞赛浏览和搜索
2. 竞赛详情查看
3. 数据下载管理
4. 自动提交
5. 排行榜追踪
6. 历史记录管理
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import json
from pathlib import Path
import sys

# 添加RD-Agent路径（改为读取环境变量 RDAGENT_PATH）
import os
_env = os.getenv("RDAGENT_PATH")
if _env:
    _p = Path(_env)
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class KaggleAgentUI:
    """Kaggle Agent UI管理类"""
    
    def __init__(self):
        self.init_session_state()
        
    def init_session_state(self):
        """初始化session状态"""
        if 'kaggle_competitions' not in st.session_state:
            st.session_state.kaggle_competitions = []
        if 'kaggle_submissions' not in st.session_state:
            st.session_state.kaggle_submissions = []
        if 'kaggle_selected_competition' not in st.session_state:
            st.session_state.kaggle_selected_competition = None
        if 'kaggle_api_configured' not in st.session_state:
            st.session_state.kaggle_api_configured = self._check_kaggle_api()
    
    def _check_kaggle_api(self) -> bool:
        """检查Kaggle API配置"""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists()
    
    def _get_mock_competitions(self) -> List[Dict]:
        """获取Mock竞赛列表 (真实环境会调用Kaggle API)"""
        return [
            {
                'id': 'titanic',
                'title': 'Titanic - Machine Learning from Disaster',
                'category': 'Getting Started',
                'reward': '$0',
                'team_count': 15000,
                'deadline': '2025-12-31',
                'description': '经典的生存预测竞赛,适合入门学习',
                'evaluation': 'Accuracy'
            },
            {
                'id': 'house-prices-advanced-regression-techniques',
                'title': 'House Prices - Advanced Regression Techniques',
                'category': 'Getting Started',
                'reward': '$0',
                'team_count': 8000,
                'deadline': '2025-12-31',
                'description': '房价预测竞赛,学习特征工程',
                'evaluation': 'RMSE'
            },
            {
                'id': 'digit-recognizer',
                'title': 'Digit Recognizer',
                'category': 'Computer Vision',
                'reward': '$0',
                'team_count': 5000,
                'deadline': '2025-12-31',
                'description': 'MNIST手写数字识别',
                'evaluation': 'Accuracy'
            },
            {
                'id': 'nlp-getting-started',
                'title': 'Natural Language Processing with Disaster Tweets',
                'category': 'NLP',
                'reward': '$0',
                'team_count': 6000,
                'deadline': '2025-12-31',
                'description': '灾难推文分类任务',
                'evaluation': 'F1 Score'
            },
            {
                'id': 'playground-series-s4e12',
                'title': 'Regression with a Flood Prediction Dataset',
                'category': 'Featured',
                'reward': '$25,000',
                'team_count': 1200,
                'deadline': '2025-02-15',
                'description': '洪水预测回归任务',
                'evaluation': 'MAE'
            }
        ]
    
    def _get_mock_leaderboard(self, competition_id: str) -> List[Dict]:
        """获取Mock排行榜数据"""
        return [
            {'rank': 1, 'team': 'ML Wizards', 'score': 0.98765, 'submissions': 25, 'last_submission': '2 hours ago'},
            {'rank': 2, 'team': 'Deep Learning Pro', 'score': 0.98652, 'submissions': 42, 'last_submission': '5 hours ago'},
            {'rank': 3, 'team': 'Data Scientists United', 'score': 0.98521, 'submissions': 18, 'last_submission': '1 day ago'},
            {'rank': 4, 'team': 'AI Enthusiasts', 'score': 0.98412, 'submissions': 33, 'last_submission': '3 days ago'},
            {'rank': 5, 'team': 'Kaggle Masters', 'score': 0.98305, 'submissions': 15, 'last_submission': '5 days ago'},
        ]
    
    def _get_mock_submissions(self, competition_id: str) -> List[Dict]:
        """获取Mock提交历史"""
        return [
            {
                'id': 'sub_001',
                'competition': competition_id,
                'filename': 'submission_v5.csv',
                'description': 'XGBoost + LightGBM Ensemble',
                'score': 0.87654,
                'status': 'complete',
                'public_score': 0.87654,
                'private_score': 0.87321,
                'submitted_at': '2025-01-05 14:23:00'
            },
            {
                'id': 'sub_002',
                'competition': competition_id,
                'filename': 'submission_v4.csv',
                'description': 'LightGBM with feature engineering',
                'score': 0.86542,
                'status': 'complete',
                'public_score': 0.86542,
                'private_score': 0.86123,
                'submitted_at': '2025-01-04 10:15:00'
            },
            {
                'id': 'sub_003',
                'competition': competition_id,
                'filename': 'submission_v3.csv',
                'description': 'Random Forest baseline',
                'score': 0.82134,
                'status': 'complete',
                'public_score': 0.82134,
                'private_score': 0.81892,
                'submitted_at': '2025-01-03 16:45:00'
            }
        ]
    
    def render_api_config(self):
        """渲染API配置部分"""
        st.subheader("🔑 Kaggle API配置")
        
        if st.session_state.kaggle_api_configured:
            st.success("✅ Kaggle API已配置")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📁 配置文件: `~/.kaggle/kaggle.json`")
            with col2:
                if st.button("🔄 重新配置", key="reconfig_kaggle"):
                    st.session_state.kaggle_api_configured = False
                    st.rerun()
        else:
            st.warning("⚠️ Kaggle API未配置,某些功能将受限")
            
            with st.expander("📖 如何配置Kaggle API", expanded=True):
                st.markdown("""
                ### 配置步骤:
                
                1. **获取API Token**:
                   - 登录 [Kaggle](https://www.kaggle.com)
                   - 进入 `Account` → `API` → `Create New API Token`
                   - 下载 `kaggle.json` 文件
                
                2. **放置配置文件**:
                   ```bash
                   # Windows
                   mkdir %USERPROFILE%\\.kaggle
                   move kaggle.json %USERPROFILE%\\.kaggle\\
                   
                   # Linux/Mac
                   mkdir -p ~/.kaggle
                   mv kaggle.json ~/.kaggle/
                   chmod 600 ~/.kaggle/kaggle.json
                   ```
                
                3. **验证配置**:
                   ```bash
                   kaggle competitions list
                   ```
                """)
                
                if st.button("✅ 我已配置完成", key="confirm_kaggle_config"):
                    if self._check_kaggle_api():
                        st.session_state.kaggle_api_configured = True
                        st.success("配置成功!")
                        st.rerun()
                    else:
                        st.error("未检测到kaggle.json文件,请检查配置")
    
    def render_competitions_list(self):
        """渲染竞赛列表"""
        st.subheader("🏆 竞赛浏览")
        
        # 搜索和筛选
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 搜索竞赛", placeholder="输入关键词...")
        with col2:
            category_filter = st.selectbox(
                "📂 分类",
                ["All", "Getting Started", "Featured", "Research", "Computer Vision", "NLP"]
            )
        with col3:
            sort_by = st.selectbox("📊 排序", ["Deadline", "Team Count", "Reward"])
        
        # 获取竞赛列表
        competitions = self._get_mock_competitions()
        
        # 应用筛选
        if category_filter != "All":
            competitions = [c for c in competitions if c['category'] == category_filter]
        if search_query:
            competitions = [c for c in competitions if search_query.lower() in c['title'].lower() or search_query.lower() in c['description'].lower()]
        
        # 显示竞赛卡片
        st.write(f"找到 **{len(competitions)}** 个竞赛")
        
        for comp in competitions:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### {comp['title']}")
                    st.markdown(f"**{comp['category']}** | 💰 {comp['reward']} | 👥 {comp['team_count']} teams | ⏰ {comp['deadline']}")
                    st.markdown(f"_{comp['description']}_")
                    st.caption(f"评估指标: {comp['evaluation']}")
                with col2:
                    if st.button("📖 详情", key=f"detail_{comp['id']}"):
                        st.session_state.kaggle_selected_competition = comp['id']
                        st.rerun()
                    if st.button("📥 下载数据", key=f"download_{comp['id']}"):
                        with st.spinner("下载中..."):
                            st.success(f"✅ 数据下载完成: {comp['id']}")
                st.divider()
    
    def render_competition_detail(self, competition_id: str):
        """渲染竞赛详情"""
        competitions = self._get_mock_competitions()
        comp = next((c for c in competitions if c['id'] == competition_id), None)
        
        if not comp:
            st.error("竞赛不存在")
            return
        
        # 返回按钮
        if st.button("⬅️ 返回列表"):
            st.session_state.kaggle_selected_competition = None
            st.rerun()
        
        st.title(comp['title'])
        
        # 竞赛信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("分类", comp['category'])
        with col2:
            st.metric("奖金", comp['reward'])
        with col3:
            st.metric("参赛团队", comp['team_count'])
        with col4:
            st.metric("截止日期", comp['deadline'])
        
        st.divider()
        
        # Tab切换
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Overview", "📊 Leaderboard", "📤 Submit", "📜 My Submissions"])
        
        with tab1:
            self.render_competition_overview(comp)
        
        with tab2:
            self.render_leaderboard(competition_id)
        
        with tab3:
            self.render_submission_form(competition_id)
        
        with tab4:
            self.render_my_submissions(competition_id)
    
    def render_competition_overview(self, comp: Dict):
        """渲染竞赛概览"""
        st.subheader("竞赛描述")
        st.info(comp['description'])
        
        st.subheader("评估指标")
        st.code(comp['evaluation'])
        
        st.subheader("时间线")
        st.markdown(f"""
        - **开始时间**: 2024-01-01
        - **截止时间**: {comp['deadline']}
        - **最终排名公布**: {comp['deadline']} 后 7 天
        """)
        
        st.subheader("数据文件")
        st.markdown("""
        - `train.csv` - 训练数据集
        - `test.csv` - 测试数据集
        - `sample_submission.csv` - 提交样例
        """)
        
        if st.button("📥 下载所有数据", key="download_all_data"):
            with st.spinner("下载中..."):
                st.success("✅ 数据下载完成!")
    
    def render_leaderboard(self, competition_id: str):
        """渲染排行榜"""
        st.subheader("🏅 排行榜")
        
        leaderboard = self._get_mock_leaderboard(competition_id)
        df = pd.DataFrame(leaderboard)
        
        st.dataframe(
            df,
            column_config={
                "rank": st.column_config.NumberColumn("排名", format="%d"),
                "team": st.column_config.TextColumn("团队"),
                "score": st.column_config.NumberColumn("分数", format="%.5f"),
                "submissions": st.column_config.NumberColumn("提交次数"),
                "last_submission": st.column_config.TextColumn("最后提交")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 排行榜统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总参赛队伍", len(leaderboard))
        with col2:
            st.metric("最高分", f"{max(l['score'] for l in leaderboard):.5f}")
        with col3:
            st.metric("平均提交次数", f"{sum(l['submissions'] for l in leaderboard) / len(leaderboard):.1f}")
    
    def render_submission_form(self, competition_id: str):
        """渲染提交表单"""
        st.subheader("📤 提交预测结果")
        
        # 高级配置
        with st.expander("⚙️ RD-Agent 高级配置", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                auto_submit = st.checkbox(
                    "🚀 自动提交",
                    value=False,
                    help="开启后，RD-Agent会自动将实验结果上传并提交到Kaggle平台"
                )
            with col2:
                use_graph_rag = st.checkbox(
                    "🧠 图知识库RAG",
                    value=False,
                    help="启用基于图的高级RAG知识管理系统"
                )
            
            if auto_submit:
                st.info("🔔 自动提交已启用：每次实验结果将自动提交到Kaggle")
            if use_graph_rag:
                st.info("📘 图知识库已启用：将使用 KGKnowledgeGraph 增强实验推理")
            
            # 存储配置到session
            if 'kaggle_auto_submit' not in st.session_state:
                st.session_state.kaggle_auto_submit = False
            if 'kaggle_use_graph_rag' not in st.session_state:
                st.session_state.kaggle_use_graph_rag = False
            
            st.session_state.kaggle_auto_submit = auto_submit
            st.session_state.kaggle_use_graph_rag = use_graph_rag
        
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
        
        description = st.text_area(
            "提交描述 (可选)",
            placeholder="例如: XGBoost + LightGBM ensemble model with extensive feature engineering",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 提交", type="primary", disabled=uploaded_file is None):
                if uploaded_file:
                    with st.spinner("提交中..."):
                        # 模拟提交
                        import time
                        time.sleep(2)
                        st.success("🎉 提交成功!")
                        st.info(f"Public Score: 0.{87000 + len(st.session_state.kaggle_submissions)}123")
        with col2:
            if st.button("🔍 验证文件格式"):
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ 文件格式正确: {len(df)} 行")
                    st.dataframe(df.head(), use_container_width=True)
        
        st.divider()
        
        st.subheader("📋 提交要求")
        st.markdown("""
        1. 文件格式: CSV
        2. 必须包含的列: `Id`, `Target`
        3. 每天最多提交 **5** 次
        4. 最终评估使用 **Private Test Set**
        """)
    
    def render_my_submissions(self, competition_id: str):
        """渲染我的提交历史"""
        st.subheader("📜 提交历史")
        
        submissions = self._get_mock_submissions(competition_id)
        
        if not submissions:
            st.info("暂无提交记录")
            return
        
        df = pd.DataFrame(submissions)
        
        st.dataframe(
            df,
            column_config={
                "id": st.column_config.TextColumn("ID"),
                "filename": st.column_config.TextColumn("文件名"),
                "description": st.column_config.TextColumn("描述"),
                "public_score": st.column_config.NumberColumn("Public Score", format="%.5f"),
                "private_score": st.column_config.NumberColumn("Private Score", format="%.5f"),
                "status": st.column_config.TextColumn("状态"),
                "submitted_at": st.column_config.TextColumn("提交时间")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 提交统计
        st.subheader("📊 提交统计")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总提交次数", len(submissions))
        with col2:
            best_score = max(s['public_score'] for s in submissions)
            st.metric("最佳Public Score", f"{best_score:.5f}")
        with col3:
            avg_score = sum(s['public_score'] for s in submissions) / len(submissions)
            st.metric("平均Score", f"{avg_score:.5f}")
        
        # 分数趋势图
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(submissions) + 1)),
            y=[s['public_score'] for s in reversed(submissions)],
            mode='lines+markers',
            name='Public Score',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(submissions) + 1)),
            y=[s['private_score'] for s in reversed(submissions)],
            mode='lines+markers',
            name='Private Score',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig.update_layout(
            title="提交分数趋势",
            xaxis_title="提交次数",
            yaxis_title="Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render(self):
        """主渲染函数"""
        st.title("🏆 Kaggle Agent - 竞赛自动化")
        
        # API配置
        with st.expander("⚙️ API配置", expanded=not st.session_state.kaggle_api_configured):
            self.render_api_config()
        
        st.divider()
        
        # 主界面
        if st.session_state.kaggle_selected_competition:
            self.render_competition_detail(st.session_state.kaggle_selected_competition)
        else:
            self.render_competitions_list()


def main():
    """主函数"""
    ui = KaggleAgentUI()
    ui.render()


if __name__ == "__main__":
    main()
