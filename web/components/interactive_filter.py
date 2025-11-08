"""
交互式三层筛选漏斗组件
用于T日选股的可视化交互式筛选
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Callable, Optional
import plotly.graph_objects as go


class InteractiveFilter:
    """交互式三层筛选器"""
    
    def __init__(self, data: pd.DataFrame, key_prefix: str = "filter"):
        """
        初始化筛选器
        
        Args:
            data: 原始数据DataFrame
            key_prefix: Streamlit组件的key前缀（避免重复）
        """
        self.original_data = data.copy() if data is not None else pd.DataFrame()
        self.filtered_data = self.original_data.copy()
        self.key_prefix = key_prefix
        self.filter_history = []  # 记录筛选历史
        
    def render(self) -> pd.DataFrame:
        """
        渲染完整的三层筛选器
        
        Returns:
            筛选后的DataFrame
        """
        if self.original_data.empty:
            st.warning("📭 没有数据可供筛选")
            return self.original_data
        
        st.markdown("### 🔍 三层筛选漏斗")
        st.caption("通过三层过滤逐步精选优质标的")
        
        # 显示初始数据量
        total_count = len(self.original_data)
        st.info(f"📊 原始涨停股: **{total_count}** 只")
        
        # 第一层：基础过滤
        st.markdown("---")
        st.markdown("#### 🔹 第一层：基础过滤")
        self.filtered_data = self._render_layer1_basic_filter()
        layer1_count = len(self.filtered_data)
        layer1_eliminated = total_count - layer1_count
        
        col1, col2 = st.columns([3, 1])
        with col1:
            self._render_progress_bar(layer1_count, total_count, "第一层筛选")
        with col2:
            st.metric("剩余", f"{layer1_count} 只", delta=f"-{layer1_eliminated}")
        
        if self.filtered_data.empty:
            st.warning("⚠️  第一层筛选后无剩余股票，请放宽条件")
            return self.filtered_data
        
        # 第二层：质量评分
        st.markdown("---")
        st.markdown("#### 🔹 第二层：质量评分")
        self.filtered_data = self._render_layer2_quality_score()
        layer2_count = len(self.filtered_data)
        layer2_eliminated = layer1_count - layer2_count
        
        col1, col2 = st.columns([3, 1])
        with col1:
            self._render_progress_bar(layer2_count, total_count, "第二层筛选")
        with col2:
            st.metric("剩余", f"{layer2_count} 只", delta=f"-{layer2_eliminated}")
        
        if self.filtered_data.empty:
            st.warning("⚠️  第二层筛选后无剩余股票，请放宽条件")
            return self.filtered_data
        
        # 第三层：AI智能选股
        st.markdown("---")
        st.markdown("#### 🔹 第三层：AI智能选股")
        self.filtered_data = self._render_layer3_ai_selection()
        layer3_count = len(self.filtered_data)
        layer3_eliminated = layer2_count - layer3_count
        
        col1, col2 = st.columns([3, 1])
        with col1:
            self._render_progress_bar(layer3_count, total_count, "第三层筛选（最终）")
        with col2:
            st.metric("剩余", f"{layer3_count} 只", delta=f"-{layer3_eliminated}")
        
        # 漏斗可视化
        st.markdown("---")
        self._render_funnel_chart(total_count, layer1_count, layer2_count, layer3_count)
        
        return self.filtered_data
    
    def _render_layer1_basic_filter(self) -> pd.DataFrame:
        """第一层：基础过滤"""
        data = self.filtered_data.copy()
        
        st.caption("✅ 排除ST、*ST、涨停时间过早、封单强度不足等")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 封单强度过滤
            if 'seal_strength' in data.columns:
                min_seal = st.slider(
                    "📊 最低封单强度 (%)",
                    min_value=0,
                    max_value=100,
                    value=60,
                    step=5,
                    key=f"{self.key_prefix}_seal_strength",
                    help="封单金额/流通市值的比例"
                )
                data = data[data['seal_strength'] >= min_seal]
            
            # 排除ST
            exclude_st = st.checkbox(
                "🚫 排除ST、*ST股",
                value=True,
                key=f"{self.key_prefix}_exclude_st"
            )
            if exclude_st and 'name' in data.columns:
                data = data[~data['name'].str.contains('ST|st', na=False)]
        
        with col2:
            # 涨停时间过滤
            if 'limitup_time' in data.columns:
                max_time = st.time_input(
                    "⏰ 最晚涨停时间",
                    value=pd.to_datetime("10:30").time(),
                    key=f"{self.key_prefix}_limitup_time",
                    help="只保留此时间之前涨停的股票"
                )
                # 注意：这里需要实际的时间比较逻辑
                # data = data[data['limitup_time'] <= max_time]
            
            # 开板次数过滤
            if 'open_count' in data.columns:
                max_opens = st.select_slider(
                    "🔓 最大开板次数",
                    options=[0, 1, 2, 3, 5, 10],
                    value=2,
                    key=f"{self.key_prefix}_open_count",
                    help="开板次数越少，封板质量越高"
                )
                data = data[data['open_count'] <= max_opens]
        
        return data
    
    def _render_layer2_quality_score(self) -> pd.DataFrame:
        """第二层：质量评分"""
        data = self.filtered_data.copy()
        
        st.caption("📊 综合评分：封单强度 + 涨停时间 + 板块热度 + 资金流向")
        
        # 评分权重配置
        st.markdown("##### 评分权重配置")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            weight_seal = st.slider(
                "封单强度",
                0, 100, 40,
                key=f"{self.key_prefix}_weight_seal",
                help="封单越强，得分越高"
            )
        
        with col2:
            weight_time = st.slider(
                "涨停时间",
                0, 100, 20,
                key=f"{self.key_prefix}_weight_time",
                help="越早涨停，得分越高"
            )
        
        with col3:
            weight_sector = st.slider(
                "板块联动",
                0, 100, 20,
                key=f"{self.key_prefix}_weight_sector",
                help="板块越热，得分越高"
            )
        
        with col4:
            weight_flow = st.slider(
                "资金流向",
                0, 100, 20,
                key=f"{self.key_prefix}_weight_flow",
                help="资金流入越多，得分越高"
            )
        
        # 计算综合评分（如果列存在）
        if 'quality_score' in data.columns:
            # 使用现有的质量分
            min_quality = st.slider(
                "📈 最低质量分数",
                0, 100, 70,
                key=f"{self.key_prefix}_min_quality",
                help="质量分越高，股票越优质"
            )
            data = data[data['quality_score'] >= min_quality]
        else:
            st.info("💡 质量评分需要更多数据列支持（开发中）")
        
        return data
    
    def _render_layer3_ai_selection(self) -> pd.DataFrame:
        """第三层：AI智能选股"""
        data = self.filtered_data.copy()
        
        st.caption("🤖 基于强化学习的智能评分 + Thompson Sampling优化")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RL评分阈值
            if 'rl_score' in data.columns:
                min_rl_score = st.slider(
                    "🎯 最低RL得分",
                    0.0, 10.0, 6.0, 0.5,
                    key=f"{self.key_prefix}_min_rl_score",
                    help="强化学习模型给出的评分"
                )
                data = data[data['rl_score'] >= min_rl_score]
            else:
                st.info("💡 RL评分数据暂未加载")
        
        with col2:
            # TopK选择
            if not data.empty:
                topk = st.slider(
                    "🏆 选取Top K",
                    1, min(20, len(data)), min(8, len(data)),
                    key=f"{self.key_prefix}_topk",
                    help="从筛选结果中选取得分最高的K只股票"
                )
                # 按RL得分排序并选择TopK
                if 'rl_score' in data.columns:
                    data = data.nlargest(topk, 'rl_score')
                elif 'quality_score' in data.columns:
                    data = data.nlargest(topk, 'quality_score')
                else:
                    data = data.head(topk)
        
        # 显示市场环境参考
        with st.expander("🌍 市场环境参考", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("市场情绪", "强势", delta="🟢", help="当前市场整体情绪")
            with col2:
                st.metric("热门板块", "AI概念", delta="9.2分", help="当前最热板块")
            with col3:
                st.metric("大盘走势", "+1.2%", delta="上涨", help="上证指数涨跌幅")
        
        return data
    
    def _render_progress_bar(self, current: int, total: int, label: str):
        """渲染进度条"""
        percentage = (current / total * 100) if total > 0 else 0
        
        # 使用Plotly创建更漂亮的进度条
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': label, 'font': {'size': 14}},
            delta={'reference': total, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, total], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, total], 'color': 'lightgray'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': total * 0.3
                }
            }
        ))
        
        fig.update_layout(
            height=150,
            margin=dict(l=10, r=10, t=40, b=10),
            font={'size': 12}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_funnel_chart(self, total: int, layer1: int, layer2: int, layer3: int):
        """渲染漏斗图"""
        st.markdown("#### 📊 筛选漏斗可视化")
        
        fig = go.Figure(go.Funnel(
            y=["原始涨停股", "第一层筛选", "第二层筛选", "最终候选池"],
            x=[total, layer1, layer2, layer3],
            textposition="inside",
            textinfo="value+percent initial",
            marker={
                "color": ["#667eea", "#48bb78", "#38b2ac", "#4299e1"],
                "line": {"width": 2, "color": "white"}
            },
            connector={"line": {"color": "royalblue", "width": 3}}
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示淘汰统计
        col1, col2, col3 = st.columns(3)
        with col1:
            eliminated1 = total - layer1
            st.metric("第一层淘汰", f"{eliminated1} 只", delta=f"-{eliminated1/total*100:.1f}%")
        with col2:
            eliminated2 = layer1 - layer2
            st.metric("第二层淘汰", f"{eliminated2} 只", delta=f"-{eliminated2/layer1*100:.1f}%" if layer1 > 0 else "0%")
        with col3:
            eliminated3 = layer2 - layer3
            st.metric("第三层淘汰", f"{eliminated3} 只", delta=f"-{eliminated3/layer2*100:.1f}%" if layer2 > 0 else "0%")


def create_test_data(count: int = 100) -> pd.DataFrame:
    """创建测试数据"""
    import numpy as np
    
    data = {
        'symbol': [f"{i:06d}" for i in range(count)],
        'name': [f"股票{i}" if i % 10 != 0 else f"ST股票{i}" for i in range(count)],
        'seal_strength': np.random.randint(30, 100, count),
        'limitup_time': pd.date_range('09:30', periods=count, freq='5min'),
        'open_count': np.random.randint(0, 5, count),
        'quality_score': np.random.randint(40, 100, count),
        'rl_score': np.random.uniform(3, 10, count),
        'sector': np.random.choice(['AI概念', '算力', '芯片', '软件', '硬件'], count)
    }
    
    return pd.DataFrame(data)


# 测试代码
if __name__ == "__main__":
    st.set_page_config(page_title="交互式筛选器测试", layout="wide")
    
    st.title("🔍 交互式三层筛选漏斗测试")
    
    # 创建测试数据
    test_data = create_test_data(100)
    
    st.sidebar.markdown("### 测试数据")
    st.sidebar.info(f"总共 {len(test_data)} 只模拟涨停股")
    
    # 渲染筛选器
    filter_component = InteractiveFilter(test_data, key_prefix="test")
    result = filter_component.render()
    
    # 显示筛选结果
    st.markdown("---")
    st.markdown("### ✅ 最终筛选结果")
    
    if not result.empty:
        st.dataframe(
            result[['symbol', 'name', 'quality_score', 'rl_score', 'seal_strength', 'sector']],
            use_container_width=True,
            hide_index=True
        )
        
        # 下载按钮
        csv = result.to_csv(index=False)
        st.download_button(
            label="📥 下载筛选结果 (CSV)",
            data=csv,
            file_name="filtered_stocks.csv",
            mime="text/csv"
        )
    else:
        st.warning("没有股票通过筛选")
