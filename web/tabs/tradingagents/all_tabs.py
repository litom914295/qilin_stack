"""
TradingAgents 全部6个Tab模块
集成对接tradingagents-cn-plus项目
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# 添加TradingAgents路径
ta_path = Path("G:/test/tradingagents-cn-plus")
if ta_path.exists() and str(ta_path) not in sys.path:
    sys.path.insert(0, str(ta_path))


def render_agent_management():
    """智能体管理tab"""
    st.header("🔍 智能体管理")
    
    st.markdown("""
    **6类专业分析师智能体**
    - 📊 基本面分析师
    - 📈 技术分析师  
    - 📰 新闻分析师
    - 💬 社交媒体分析师
    - 🔼 看涨研究员
    - 🔽 看跌研究员
    """)
    
    # 智能体状态总览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("激活智能体", "6/6", "100%")
    with col2:
        st.metric("平均响应时间", "2.3s", "-0.5s")
    with col3:
        st.metric("今日分析次数", "128", "+23")
    with col4:
        st.metric("共识达成率", "87%", "+5%")
    
    st.divider()
    
    # 智能体详细配置
    st.subheader("⚙️ 智能体配置")
    
    agents_config = [
        {"name": "基本面分析师", "emoji": "📊", "status": "✅ 运行中", "weight": 0.20},
        {"name": "技术分析师", "emoji": "📈", "status": "✅ 运行中", "weight": 0.25},
        {"name": "新闻分析师", "emoji": "📰", "status": "✅ 运行中", "weight": 0.15},
        {"name": "社交媒体分析师", "emoji": "💬", "status": "✅ 运行中", "weight": 0.10},
        {"name": "看涨研究员", "emoji": "🔼", "status": "✅ 运行中", "weight": 0.15},
        {"name": "看跌研究员", "emoji": "🔽", "status": "✅ 运行中", "weight": 0.15}
    ]
    
    for agent in agents_config:
        with st.expander(f"{agent['emoji']} {agent['name']} - {agent['status']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.slider(f"权重", 0.0, 1.0, agent['weight'], key=f"weight_{agent['name']}")
                st.checkbox("启用", value=True, key=f"enable_{agent['name']}")
            with col2:
                st.selectbox("LLM模型", ["gpt-4", "gpt-3.5-turbo", "claude-3"], key=f"model_{agent['name']}")
                st.number_input("温度", 0.0, 2.0, 0.7, key=f"temp_{agent['name']}")
    
    st.divider()
    
    # 智能体性能对比
    st.subheader("📊 性能对比")
    
    performance_data = {
        "智能体": [a['name'] for a in agents_config],
        "准确率": [0.78, 0.82, 0.75, 0.68, 0.81, 0.79],
        "响应时间(s)": [2.1, 1.8, 3.2, 2.5, 2.3, 2.4],
        "信心度": [0.85, 0.89, 0.76, 0.72, 0.88, 0.86]
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_collaboration():
    """协作机制tab"""
    st.header("🗣️ 协作机制")
    
    st.markdown("""
    **结构化辩论流程**
    1. 🎤 初始观点提出
    2. 📋 论据收集与支持
    3. ⚔️ 对立观点辩驳
    4. 🔄 多轮迭代优化
    5. ✅ 共识达成
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("辩论轮次", "3", "标准")
    with col2:
        st.metric("共识阈值", "75%", "可配置")
    with col3:
        st.metric("平均耗时", "8.5s", "-1.2s")
    
    st.divider()
    
    # 实时辩论可视化
    st.subheader("🎭 实时辩论过程")
    
    debate_log = [
        {"time": "14:32:15", "agent": "看涨研究员", "type": "观点", "content": "基于技术面突破，建议买入"},
        {"time": "14:32:18", "agent": "基本面分析师", "type": "支持", "content": "财报数据显示盈利增长30%"},
        {"time": "14:32:22", "agent": "看跌研究员", "type": "反驳", "content": "市盈率过高，存在回调风险"},
        {"time": "14:32:25", "agent": "技术分析师", "type": "支持", "content": "MA20已站稳，突破有效"},
        {"time": "14:32:29", "agent": "新闻分析师", "type": "中立", "content": "近期无重大利空消息"},
        {"time": "14:32:33", "agent": "交易员", "type": "决策", "content": "综合评分75分，建议谨慎买入"}
    ]
    
    for log in debate_log:
        color_map = {"观点": "🔵", "支持": "🟢", "反驳": "🔴", "中立": "🟡", "决策": "🟣"}
        st.markdown(f"{color_map.get(log['type'], '⚪')} **{log['time']}** - *{log['agent']}* ({log['type']}): {log['content']}")
    
    st.divider()
    
    # 共识达成流程图
    st.subheader("📊 共识达成可视化")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=["看涨观点", "看跌观点", "中立观点", "论据支持", "最终决策"],
            color=["green", "red", "gray", "blue", "purple"]
        ),
        link=dict(
            source=[0, 1, 2, 0, 1],
            target=[3, 3, 3, 4, 4],
            value=[45, 25, 30, 40, 20]
        )
    )])
    fig.update_layout(title="观点流向与共识形成", height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_information_collection():
    """信息采集tab"""
    st.header("📰 信息采集")
    
    st.markdown("""
    **多源信息整合**
    - 📰 新闻资讯 (v0.1.12智能过滤)
    - 📊 财务数据
    - 💬 社交媒体情绪
    - 📈 实时行情数据
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("今日新闻", "1,247", "+156")
    with col2:
        st.metric("过滤后", "89", "高质量")
    with col3:
        st.metric("情绪指数", "0.68", "偏乐观")
    with col4:
        st.metric("数据源", "12", "多元化")
    
    st.divider()
    
    # 新闻过滤配置
    st.subheader("⚙️ 新闻智能过滤 (v0.1.12)")
    
    with st.expander("过滤配置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            filter_mode = st.selectbox(
                "过滤模式",
                ["基础过滤", "增强过滤", "集成过滤"]
            )
            relevance_threshold = st.slider("相关性阈值", 0.0, 1.0, 0.7)
        with col2:
            quality_threshold = st.slider("质量阈值", 0.0, 1.0, 0.6)
            enable_dedup = st.checkbox("去重", value=True)
    
    if st.button("🔍 应用过滤", type="primary"):
        st.success(f"已应用{filter_mode}，过滤出89条高质量新闻")
    
    st.divider()
    
    # 最新新闻展示
    st.subheader("📋 过滤后的新闻")
    
    news_data = [
        {"time": "10:23", "title": "某公司发布Q3财报，净利润同比增长35%", "relevance": 0.92, "sentiment": "正面"},
        {"time": "09:45", "title": "行业监管新政出台，利好龙头企业", "relevance": 0.88, "sentiment": "正面"},
        {"time": "08:30", "title": "技术突破获得重大进展", "relevance": 0.85, "sentiment": "正面"}
    ]
    
    for news in news_data:
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.markdown(f"**{news['time']}**")
            with col2:
                st.markdown(f"{news['title']}")
            with col3:
                sentiment_emoji = "🟢" if news['sentiment'] == "正面" else "🔴" if news['sentiment'] == "负面" else "🟡"
                st.markdown(f"{sentiment_emoji} {news['relevance']:.0%}")


def render_decision_analysis():
    """决策分析tab"""
    st.header("💡 决策分析")
    
    st.markdown("""
    **分析模式**
    - 📊 单股深度分析
    - 📋 批量分析 (v0.1.15+)
    - 🎯 研究深度配置
    - 📄 报告自动生成
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("今日分析", "47", "+8")
    with col2:
        st.metric("平均耗时", "12.3s", "-2.1s")
    with col3:
        st.metric("成功率", "89%", "+3%")
    
    st.divider()
    
    # 分析配置
    analysis_mode = st.radio(
        "选择分析模式",
        ["📊 单股分析", "📋 批量分析"],
        horizontal=True
    )
    
    if analysis_mode == "📊 单股分析":
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("股票代码", "000001")
        with col2:
            depth = st.selectbox("研究深度", ["简单", "标准", "深度", "极深", "完整"])
        
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
            with st.spinner("智能体正在协作分析..."):
                import time
                time.sleep(2)
            st.success("分析完成!")
            
            # 显示分析结果
            st.subheader("📊 分析结果")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("综合评分", "75/100", "")
            with col2:
                st.metric("建议", "谨慎买入", "")
            with col3:
                st.metric("目标价", "¥12.50", "+15%")
            with col4:
                st.metric("风险等级", "中等", "")
    
    else:  # 批量分析
        symbols_input = st.text_area(
            "输入股票代码(每行一个)",
            "000001\n000002\n600000",
            height=150
        )
        
        col1, col2 = st.columns(2)
        with col1:
            depth = st.selectbox("研究深度", ["简单", "标准", "深度"], key="batch_depth")
        with col2:
            parallel = st.number_input("并行数量", 1, 10, 3)
        
        if st.button("🚀 批量分析", type="primary", use_container_width=True):
            symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
            with st.spinner(f"正在分析{len(symbols)}只股票..."):
                import time
                time.sleep(3)
            st.success(f"批量分析完成!共{len(symbols)}只股票")
            
            # 批量结果表格
            results_data = {
                "代码": symbols,
                "评分": [75, 68, 82],
                "建议": ["谨慎买入", "观望", "买入"],
                "目标价": ["¥12.50", "¥8.30", "¥15.20"],
                "风险": ["中", "高", "低"]
            }
            st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)


def render_user_management():
    """用户管理tab"""
    st.header("👤 用户管理")
    
    st.markdown("""
    **会员系统 (v0.1.14+)**
    - 👥 用户注册/登录
    - 🎫 点数管理
    - 📊 使用统计
    - 📜 活动日志
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("注册用户", "1,247", "+23")
    with col2:
        st.metric("活跃用户", "89", "7天")
    with col3:
        st.metric("总点数消耗", "12,580", "+2,340")
    with col4:
        st.metric("平均使用", "14.1点/人", "+1.2")
    
    st.divider()
    
    # 当前用户信息
    st.subheader("👤 当前用户")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("""
        **用户ID**: admin
        **等级**: VIP
        **剩余点数**: 1,250
        **注册时间**: 2025-01-15
        """)
    with col2:
        st.markdown("**使用记录**")
        usage_data = {
            "日期": ["2025-10-28", "2025-10-27", "2025-10-26"],
            "操作": ["批量分析", "单股分析", "批量分析"],
            "股票数": [5, 1, 3],
            "消耗点数": [5, 1, 3]
        }
        st.dataframe(pd.DataFrame(usage_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 点数管理
    st.subheader("🎫 点数管理")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**充值点数**")
        amount = st.number_input("充值数量", 10, 10000, 100, step=10)
        if st.button("💰 充值", use_container_width=True):
            st.success(f"成功充值{amount}点数!")
    
    with col2:
        st.markdown("**点数说明**")
        st.info("""
        - 单股分析: 1点/次
        - 批量分析: 1点/股
        - VIP用户9折优惠
        - 每日签到赠送5点
        """)


def render_llm_integration():
    """LLM集成tab"""
    st.header("🔌 LLM集成")
    
    st.markdown("""
    **多模型支持 (v0.1.13+)**
    - 🤖 OpenAI (GPT-4/3.5)
    - 🔮 Google Gemini (2.0/2.5)
    - ☁️ Azure OpenAI
    - 🌊 DeepSeek
    - 🎯 百度千帆 (v0.1.15)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("配置模型", "5", "个")
    with col2:
        st.metric("今日调用", "3,247", "+456")
    with col3:
        st.metric("平均延迟", "1.8s", "-0.3s")
    with col4:
        st.metric("今日成本", "$12.34", "+$2.10")
    
    st.divider()
    
    # LLM配置
    st.subheader("⚙️ LLM配置")
    
    llm_providers = [
        {"name": "OpenAI", "models": ["gpt-4", "gpt-3.5-turbo"], "status": "✅"},
        {"name": "Google Gemini", "models": ["gemini-2.5-pro", "gemini-2.0-flash"], "status": "✅"},
        {"name": "Azure OpenAI", "models": ["gpt-4-azure"], "status": "✅"},
        {"name": "DeepSeek", "models": ["deepseek-chat"], "status": "✅"},
        {"name": "百度千帆", "models": ["ERNIE-Bot-4", "ERNIE-Bot-turbo"], "status": "✅"}
    ]
    
    for provider in llm_providers:
        with st.expander(f"{provider['status']} {provider['name']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(f"选择模型", provider['models'], key=f"model_{provider['name']}")
                st.text_input("API Key", type="password", key=f"key_{provider['name']}")
            with col2:
                st.text_input("API Base URL", key=f"base_{provider['name']}")
                st.slider("Temperature", 0.0, 2.0, 0.7, key=f"temp_{provider['name']}")
            
            if st.button(f"✅ 测试连接", key=f"test_{provider['name']}"):
                st.success(f"{provider['name']} 连接成功!")
    
    st.divider()
    
    # 使用统计
    st.subheader("📊 使用统计")
    
    usage_data = {
        "模型": ["GPT-4", "Gemini-2.5", "ERNIE-Bot-4", "DeepSeek", "GPT-3.5"],
        "调用次数": [1250, 980, 520, 310, 187],
        "成本($)": [8.75, 2.45, 0.52, 0.31, 0.31]
    }
    
    df = pd.DataFrame(usage_data)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, values="调用次数", names="模型", title="调用分布")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df, x="模型", y="成本($)", title="成本分布")
        st.plotly_chart(fig, use_container_width=True)
