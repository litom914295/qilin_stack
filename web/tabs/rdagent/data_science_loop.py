"""
Data Science RDLoop 集成模块
- 上传/选择数据
- 配置任务
- 运行 RD-Agent DataScience 循环
- 展示结果（指标/特征重要性/日志）
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import tempfile
import os


def _save_uploaded_file(uploaded_file) -> str:
    tmp_dir = Path.cwd() / "workspace" / "data_science"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_path = tmp_dir / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def _render_result(result: Dict[str, Any]):
    st.subheader("📊 结果")
    if not result.get('success'):
        st.error(result.get('message', '运行失败'))
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("任务类型", result.get('task_type', 'N/A'))
    with col2:
        st.metric("指标", result.get('metric', 'auto'))
    with col3:
        st.metric("得分", f"{result.get('score', 0):.5f}")

    st.divider()
    st.subheader("🧩 最佳模型")
    st.info(result.get('best_model', 'N/A'))

    fi = result.get('feature_importance') or []
    if fi:
        import plotly.express as px
        df_fi = pd.DataFrame(fi)
        df_fi = df_fi.sort_values('importance', ascending=False)[:30]
        fig = px.bar(df_fi, x='importance', y='feature', orientation='h', title='特征重要性')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def render():
    st.title("🧪 Data Science RDLoop")
    st.caption("上传数据并运行RD-Agent数据科学循环")

    # 数据源
    st.subheader("📁 数据")
    uploaded = st.file_uploader("上传CSV/Parquet/Excel", type=['csv', 'parquet', 'xlsx'])
    data_path = st.text_input("或输入本地数据路径", value="")

    # 任务配置
    st.subheader("⚙️ 配置")
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox("任务类型", ['classification', 'regression'], index=0)
    with col2:
        metric = st.text_input("评估指标(可选)", value="auto")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        step_n = st.number_input("步数 step_n", min_value=1, max_value=50, value=5, help="每轮运行的步数")
    with col4:
        loop_n = st.number_input("循环次数 loop_n", min_value=1, max_value=20, value=1, help="循环运行的次数")
    with col5:
        timeout = st.number_input("超时(秒)", min_value=60, max_value=7200, value=1800, help="总运行时长限制")

    run = st.button("🚀 运行DataScience RDLoop", type="primary")

    if run:
        # 准备数据路径
        path = None
        if uploaded is not None:
            path = _save_uploaded_file(uploaded)
        elif data_path.strip():
            path = data_path.strip()
        else:
            st.warning("请上传文件或填写数据路径")
            return

        # 调用API
        from .rdagent_api import RDAgentAPI
        api = RDAgentAPI()
        with st.spinner("运行中..."):
            result = api.run_data_science(path, {
                'task_type': task_type,
                'metric': metric,
                'step_n': int(step_n),
                'loop_n': int(loop_n),
                'timeout': int(timeout)
            })
        _render_result(result)


if __name__ == "__main__":
    render()
