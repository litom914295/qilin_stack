#!/usr/bin/env python
"""
一进二涨停板因子研究Web界面
集成因子发现、优化、回测功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import asyncio
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rd_agent.factor_discovery_simple import SimplifiedFactorDiscovery
from rd_agent.llm_factor_discovery import LLMFactorDiscovery
from app.factor_optimizer import FactorOptimizer


def render_factor_research_tab():
    """渲染因子研究标签页"""
    
    st.title("🧪 一进二涨停板因子研究")
    
    # 添加功能说明和工作流程
    with st.expander("📖 功能说明与使用流程", expanded=False):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### 🎯 系统功能
            
            本系统提供完整的因子研究工作流，专为**A股一进二涨停板**策略设计：
            
            - **📚 因子库**: 15个预定义因子，覆盖封板强度、连板高度、题材共振等维度
            - **🤖 LLM因子生成**: 使用DeepSeek自动生成新因子，成本约¥0.001/因子
            - **⚙️ 因子优化**: 4种权重优化方法（IC加权、等权、最大IC、岭回归）
            - **📊 回测分析**: 五分位回测验证，检查单调性和多空收益
            
            ### ⚠️ 重要提示
            
            - 当前使用**模拟数据**演示功能，IC值非真实市场数据
            - 实盘使用需对接真实数据源（AKShare/Qlib）
            - LLM生成需配置 `.env` 文件中的 `DEEPSEEK_API_KEY`
            """)
        
        with col2:
            st.markdown("""
            ### 🔄 推荐工作流程
            
            ```
            步骤1: 📚 浏览因子库
                   ↓
                了解15个预定义因子
            
            步骤2: 🤖 生成新因子（可选）
                   ↓
                使用LLM探索新思路
            
            步骤3: ⚙️ 优化因子组合
                   ↓
                IC加权 + 去相关筛选
            
            步骤4: 📊 回测验证
                   ↓
                检查单调性和收益
            
            步骤5: 🎯 实盘应用
                   ↓
                使用权重进行选股
            ```
            
            ### 💡 参数建议
            
            - **最小IC**: 0.05-0.08
            - **最大相关**: 0.6-0.8
            - **因子数量**: 5-10个
            - **样本量**: 200+
            """)
    
    # 创建子标签
    sub_tab = st.tabs([
        "📚 因子库",
        "🤖 LLM因子生成", 
        "⚙️ 因子优化",
        "📊 回测分析"
    ])
    
    # 标签1: 因子库
    with sub_tab[0]:
        render_factor_library()
    
    # 标签2: LLM因子生成
    with sub_tab[1]:
        render_llm_generation()
    
    # 标签3: 因子优化
    with sub_tab[2]:
        render_factor_optimization()
    
    # 标签4: 回测分析
    with sub_tab[3]:
        render_backtest_analysis()


def render_factor_library():
    """渲染因子库"""
    st.header("📚 预定义因子库")
    
    # 添加功能说明
    st.info("""
    👉 **功能说明**: 查看15个预定义因子，按类别筛选，了解各因子的IC分布和表达式。  
    🎯 **使用场景**: 了解现有因子体系，为后续优化选择合适的因子。
    """)
    
    # 初始化因子发现系统
    discovery = SimplifiedFactorDiscovery()
    
    # 获取统计信息
    stats = discovery.get_factor_statistics()
    
    # 显示统计卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总因子数", stats['total_factors'])
    col2.metric("因子类别", len(stats['categories']))
    col3.metric("平均IC", f"{stats['avg_ic']:.4f}")
    col4.metric("最大IC", f"{stats['max_ic']:.4f}")
    
    # 按类别展示因子
    st.subheader("因子分类")
    
    category = st.selectbox(
        "选择类别",
        ['全部'] + stats['categories']
    )
    
    if category == '全部':
        factors = discovery.factor_library
    else:
        factors = discovery.get_factors_by_category(category)
    
    # 创建因子表格
    factor_df = pd.DataFrame([
        {
            '因子ID': f['id'],
            '因子名称': f['name'],
            '类别': f['category'],
            '预期IC': f['expected_ic'],
            '表达式': f['expression'],
            '描述': f['description']
        }
        for f in factors
    ])
    
    st.dataframe(factor_df, use_container_width=True, height=400)
    
    # IC分布图
    st.subheader("IC分布")
    fig = px.bar(
        factor_df,
        x='因子名称',
        y='预期IC',
        color='类别',
        title='因子IC分布'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_llm_generation():
    """渲染LLM因子生成"""
    st.header("🤖 LLM驱动因子生成")
    
    # 添加功能说明
    st.info("""
    👉 **功能说明**: 使用DeepSeek大模型自动生成新因子，可指定关注领域和上下文。  
    🎯 **使用场景**: 探索新的因子思路，扩展因子库，成本约¥0.001/因子。  
    ⚠️ **注意**: 需配置 `.env` 中的 `DEEPSEEK_API_KEY`
    """)
    
    # 生成参数
    col1, col2 = st.columns(2)
    
    with col1:
        n_factors = st.slider("生成因子数量", 1, 10, 3)
        focus_areas = st.multiselect(
            "关注领域",
            ["封板强度", "连板动量", "题材共振", "资金行为", "时机选择"],
            default=["封板强度", "连板动量"]
        )
    
    with col2:
        context = st.text_area(
            "额外上下文（可选）",
            placeholder="例如：当前市场题材轮动快，重点关注低位首板...",
            height=100
        )
    
    # 生成按钮
    if st.button("🚀 开始生成因子", type="primary"):
        with st.spinner("正在调用LLM生成因子..."):
            try:
                # 创建发现系统
                discovery = LLMFactorDiscovery()
                
                # 异步调用
                factors = asyncio.run(
                    discovery.discover_new_factors(
                        n_factors=n_factors,
                        focus_areas=focus_areas if focus_areas else None,
                        context=context if context else None
                    )
                )
                
                if factors:
                    st.success(f"✅ 成功生成 {len(factors)} 个因子")
                    
                    # 保存到session state
                    st.session_state['generated_factors'] = factors
                    
                    # 显示生成的因子
                    for i, factor in enumerate(factors, 1):
                        with st.expander(f"因子 {i}: {factor['name']}", expanded=i==1):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**表达式**: {factor['expression']}")
                                st.markdown(f"**投资逻辑**: {factor.get('logic', 'N/A')}")
                                st.markdown(f"**类别**: {factor.get('category', 'N/A')}")
                            
                            with col2:
                                ic = factor.get('expected_ic', 0)
                                st.metric("预期IC", f"{ic:.4f}")
                            
                            st.code(factor.get('code', ''), language='python')
                else:
                    st.error("❌ 生成失败，请检查API配置")
                    
            except Exception as e:
                st.error(f"❌ 错误: {str(e)}")
    
    # 显示历史生成的因子
    if 'generated_factors' in st.session_state:
        st.subheader("📝 本次会话生成的因子")
        factors = st.session_state['generated_factors']
        st.info(f"共 {len(factors)} 个因子")


def render_factor_optimization():
    """渲染因子优化"""
    st.header("⚙️ 因子组合优化")
    
    # 添加功能说明
    st.info("""
    👉 **功能说明**: 从因子库/LLM生成中选择因子，使用4种方法优化权重（IC加权/等权/最大IC/岭回归）。  
    🎯 **使用场景**: 构建最优因子组合，去除相关性高的因子，提升IC。  
    💡 **建议**: 使用IC加权方法 + 最小IC=0.05 + 最大相关=0.7
    """)
    
    # 选择因子来源
    source = st.radio(
        "选择因子来源",
        ["预定义因子库", "LLM生成因子", "自定义上传"],
        horizontal=True
    )
    
    factors = []
    
    if source == "预定义因子库":
        discovery = SimplifiedFactorDiscovery()
        min_ic = st.slider("最小IC阈值", 0.0, 0.2, 0.08, 0.01)
        
        # 获取符合条件的因子
        factors = [f for f in discovery.factor_library if abs(f['expected_ic']) >= min_ic]
        st.success(f"找到 {len(factors)} 个符合条件的因子")
    
    elif source == "LLM生成因子":
        if 'generated_factors' in st.session_state:
            factors = st.session_state['generated_factors']
            st.success(f"使用 {len(factors)} 个LLM生成的因子")
        else:
            st.warning("请先在'LLM因子生成'标签页生成因子")
            return
    
    if not factors:
        st.warning("没有可用的因子")
        return
    
    # 显示因子列表
    st.subheader("📋 待优化因子")
    factor_names = [f['name'] for f in factors]
    selected_names = st.multiselect(
        "选择参与优化的因子",
        factor_names,
        default=factor_names[:min(5, len(factor_names))]
    )
    
    selected_factors = [f for f in factors if f['name'] in selected_names]
    
    # 优化参数
    col1, col2, col3 = st.columns(3)
    
    with col1:
        opt_method = st.selectbox(
            "优化方法",
            ['ic_weighted', 'equal', 'max_ic', 'ridge']
        )
    
    with col2:
        n_select = st.number_input("选择因子数量", 1, len(selected_factors), min(3, len(selected_factors)))
    
    with col3:
        max_corr = st.slider("最大相关性", 0.5, 1.0, 0.7, 0.05)
    
    # 优化按钮
    if st.button("🔧 开始优化", type="primary"):
        with st.spinner("正在优化因子组合..."):
            st.info("📝 注意：这里使用模拟数据演示，实际使用需要连接真实数据源")
            
            # 创建模拟数据
            n_samples = 100
            np.random.seed(42)
            
            factor_matrix = pd.DataFrame({
                f['name']: np.random.randn(n_samples)
                for f in selected_factors
            })
            
            target_returns = sum(
                f.get('expected_ic', 0.1) * factor_matrix[f['name']]
                for f in selected_factors
            ) + np.random.randn(n_samples) * 0.5
            
            # 优化
            optimizer = FactorOptimizer()
            
            # 筛选因子
            best_factors = optimizer.select_best_factors(
                selected_factors,
                factor_matrix,
                target_returns,
                n_select=n_select,
                min_ic=0.01,
                max_corr=max_corr
            )
            
            st.success(f"✅ 选出 {len(best_factors)} 个最优因子")
            
            # 显示结果
            result_df = pd.DataFrame([
                {
                    '因子名称': f['name'],
                    '实际IC': f.get('actual_ic', 0),
                    'Rank IC': f.get('actual_rank_ic', 0),
                    'IR': f.get('ir', 0)
                }
                for f in best_factors
            ])
            
            st.dataframe(result_df, use_container_width=True)
            
            # 优化权重
            weights = optimizer.optimize_factor_weights(
                best_factors,
                factor_matrix[[f['name'] for f in best_factors]],
                target_returns,
                method=opt_method
            )
            
            st.subheader("📊 因子权重")
            weight_df = pd.DataFrame([
                {'因子': name, '权重': weight}
                for name, weight in weights.items()
            ])
            
            fig = px.pie(weight_df, values='权重', names='因子', title='因子权重分布')
            st.plotly_chart(fig, use_container_width=True)
            
            # 保存到session
            st.session_state['optimized_factors'] = best_factors
            st.session_state['factor_weights'] = weights


def render_backtest_analysis():
    """渲染回测分析"""
    st.header("📊 回测分析")
    
    # 添加功能说明
    st.info("""
    👉 **功能说明**: 对优化后的因子组合进行五分位回测，检验因子效果。  
    🎯 **使用场景**: 验证因子组合的区分度，检查单调性（高分组>低分组）和多空收益。  
    💡 **建议**: 单调性通过 + 多空收益>10% = 因子组合可用
    """)
    
    if 'optimized_factors' not in st.session_state:
        st.warning("请先在'因子优化'标签页完成优化")
        return
    
    factors = st.session_state['optimized_factors']
    weights = st.session_state.get('factor_weights', {})
    
    st.success(f"使用 {len(factors)} 个优化后的因子进行回测")
    
    # 回测参数
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("开始日期", datetime(2024, 1, 1))
    
    with col2:
        end_date = st.date_input("结束日期", datetime(2024, 12, 31))
    
    if st.button("🚀 开始回测", type="primary"):
        with st.spinner("正在回测..."):
            # 模拟回测
            st.info("📝 注意：这里使用模拟数据演示")
            
            n_samples = 200
            np.random.seed(42)
            
            factor_matrix = pd.DataFrame({
                f['name']: np.random.randn(n_samples)
                for f in factors
            })
            
            target_returns = sum(
                weights.get(f['name'], 0) * factor_matrix[f['name']]
                for f in factors
            ) + np.random.randn(n_samples) * 0.3
            
            optimizer = FactorOptimizer()
            result = optimizer.backtest_factors(
                factors,
                factor_matrix,
                target_returns,
                weights
            )
            
            # 显示结果
            col1, col2, col3 = st.columns(3)
            col1.metric("多空收益", f"{result['long_short_return']:.2%}")
            col2.metric("单调性", "✅" if result['monotonicity'] else "❌")
            col3.metric("样本数", result['n_samples'])
            
            # 分组收益图
            st.subheader("📈 分组收益")
            group_df = pd.DataFrame([
                {'分组': k, '收益率': v}
                for k, v in result['group_returns'].items()
            ])
            
            fig = px.bar(group_df, x='分组', y='收益率', title='五分位收益对比')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ 回测完成！")


# 如果作为独立脚本运行
if __name__ == "__main__":
    st.set_page_config(page_title="因子研究", layout="wide")
    render_factor_research_tab()
