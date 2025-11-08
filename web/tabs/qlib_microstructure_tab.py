"""
微观结构可视化UI - Phase 6扩展任务
提供订单簿深度、价差、订单流失衡等微观结构指标的可视化展示
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qlib_enhanced.high_frequency_engine import OrderBook, MicrostructureSignals, Tick


def render_microstructure_tab():
    """渲染微观结构可视化标签页"""
    
    st.header("🔬 微观结构可视化")
    st.markdown("**实时订单簿、价差、订单流等微观结构指标的可视化分析**")
    
    # 创建子标签
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "📊 订单簿深度图",
        "📈 价差分析",
        "⚖️ 订单流失衡",
        "🎯 综合指标"
    ])
    
    with subtab1:
        render_orderbook_depth()
    
    with subtab2:
        render_spread_analysis()
    
    with subtab3:
        render_order_flow()
    
    with subtab4:
        render_综合_signals()


def render_orderbook_depth():
    """渲染订单簿深度图"""
    
    st.subheader("📊 订单簿深度可视化")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### ⚙️ 设置")
        
        # 配置选项
        symbol = st.text_input("交易标的", value="000001.SZ", key="ob_symbol")
        depth = st.slider("订单簿深度", 5, 20, 10, key="ob_depth")
        
        # 生成模拟数据按钮
        if st.button("🔄 生成模拟数据", key="gen_ob_data"):
            st.session_state['ob_data_generated'] = True
            st.success("模拟数据已生成！")
    
    with col1:
        # 检查是否生成了数据
        if st.session_state.get('ob_data_generated', False):
            # 生成模拟订单簿数据
            orderbook = generate_mock_orderbook(symbol, depth)
            
            # 创建订单簿可视化
            fig = create_orderbook_chart(orderbook, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示关键指标
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                mid_price = orderbook.get_mid_price()
                st.metric("中间价", f"¥{mid_price:.2f}")
            
            with col_b:
                spread = orderbook.get_spread()
                spread_bps = (spread / mid_price) * 10000
                st.metric("价差", f"{spread_bps:.2f} bps")
            
            with col_c:
                imbalance = orderbook.get_order_imbalance()
                st.metric("订单不平衡", f"{imbalance:+.2%}", 
                         delta="买盘强" if imbalance > 0 else "卖盘强")
            
            with col_d:
                bid_vol = sum(l.volume for l in orderbook.bids)
                ask_vol = sum(l.volume for l in orderbook.asks)
                total_vol = bid_vol + ask_vol
                st.metric("总挂单量", f"{total_vol:,}")
            
            # 订单簿详细表格
            st.markdown("### 📋 订单簿详情")
            
            col_bid, col_ask = st.columns(2)
            
            with col_bid:
                st.markdown("#### 🟢 买盘 (Bids)")
                bid_df = pd.DataFrame([
                    {
                        '档位': i+1,
                        '价格': l.price,
                        '数量': l.volume,
                        '累计': sum(b.volume for b in orderbook.bids[:i+1])
                    }
                    for i, l in enumerate(orderbook.bids)
                ])
                st.dataframe(bid_df, use_container_width=True, hide_index=True)
            
            with col_ask:
                st.markdown("#### 🔴 卖盘 (Asks)")
                ask_df = pd.DataFrame([
                    {
                        '档位': i+1,
                        '价格': l.price,
                        '数量': l.volume,
                        '累计': sum(a.volume for a in orderbook.asks[:i+1])
                    }
                    for i, l in enumerate(orderbook.asks)
                ])
                st.dataframe(ask_df, use_container_width=True, hide_index=True)
        
        else:
            st.info("👈 请点击左侧「生成模拟数据」按钮开始分析")


def render_spread_analysis():
    """渲染价差分析"""
    
    st.subheader("📈 买卖价差分析")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### ⚙️ 设置")
        
        time_window = st.selectbox(
            "时间窗口",
            ["1分钟", "5分钟", "15分钟", "1小时"],
            key="spread_window"
        )
        
        update_freq = st.slider("更新频率(毫秒)", 100, 1000, 500, key="spread_freq")
        
        if st.button("🔄 生成价差数据", key="gen_spread_data"):
            st.session_state['spread_data_generated'] = True
            st.success("价差数据已生成！")
    
    with col1:
        if st.session_state.get('spread_data_generated', False):
            # 生成模拟价差时间序列
            spread_data = generate_mock_spread_timeseries(time_window)
            
            # 创建价差图表
            fig = create_spread_chart(spread_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # 价差统计
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                avg_spread = spread_data['spread_bps'].mean()
                st.metric("平均价差", f"{avg_spread:.2f} bps")
            
            with col_b:
                min_spread = spread_data['spread_bps'].min()
                st.metric("最小价差", f"{min_spread:.2f} bps")
            
            with col_c:
                max_spread = spread_data['spread_bps'].max()
                st.metric("最大价差", f"{max_spread:.2f} bps")
            
            with col_d:
                std_spread = spread_data['spread_bps'].std()
                st.metric("价差波动率", f"{std_spread:.2f} bps")
            
            # 价差分布直方图
            st.markdown("### 📊 价差分布")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=spread_data['spread_bps'],
                nbinsx=30,
                name='价差分布',
                marker_color='lightblue'
            ))
            
            fig_hist.update_layout(
                title="价差分布直方图",
                xaxis_title="价差 (bps)",
                yaxis_title="频次",
                height=300
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        else:
            st.info("👈 请点击左侧「生成价差数据」按钮开始分析")


def render_order_flow():
    """渲染订单流失衡分析"""
    
    st.subheader("⚖️ 订单流失衡分析")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### ⚙️ 设置")
        
        window_size = st.slider("滑动窗口大小", 50, 500, 100, key="of_window")
        
        if st.button("🔄 生成订单流数据", key="gen_of_data"):
            st.session_state['of_data_generated'] = True
            st.success("订单流数据已生成！")
    
    with col1:
        if st.session_state.get('of_data_generated', False):
            # 生成模拟订单流数据
            order_flow_data = generate_mock_order_flow(window_size)
            
            # 创建订单流图表
            fig = create_order_flow_chart(order_flow_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # 订单流统计
            col_a, col_b, col_c, col_d = st.columns(4)
            
            latest_flow = order_flow_data['net_flow'].iloc[-1]
            total_buy = order_flow_data['buy_volume'].sum()
            total_sell = order_flow_data['sell_volume'].sum()
            
            with col_a:
                st.metric("当前净流入", f"{latest_flow:+,}", 
                         delta="多方占优" if latest_flow > 0 else "空方占优")
            
            with col_b:
                st.metric("总买入量", f"{total_buy:,}")
            
            with col_c:
                st.metric("总卖出量", f"{total_sell:,}")
            
            with col_d:
                imbalance_ratio = (total_buy - total_sell) / (total_buy + total_sell) if (total_buy + total_sell) > 0 else 0
                st.metric("不平衡度", f"{imbalance_ratio:+.2%}")
            
            # 买卖力量对比
            st.markdown("### 🔄 买卖力量对比")
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['买入', '卖出'],
                values=[total_buy, total_sell],
                marker=dict(colors=['#00CC96', '#EF553B']),
                hole=0.4
            )])
            
            fig_pie.update_layout(
                title="买卖量占比",
                height=300
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        else:
            st.info("👈 请点击左侧「生成订单流数据」按钮开始分析")


def render_综合_signals():
    """渲染综合微观结构信号"""
    
    st.subheader("🎯 综合微观结构信号")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### ⚙️ 设置")
        
        interval = st.selectbox(
            "计算间隔",
            ["1秒", "5秒", "10秒", "30秒"],
            key="signals_interval"
        )
        
        if st.button("🔄 计算微观结构信号", key="gen_signals"):
            st.session_state['signals_generated'] = True
            st.success("信号计算完成！")
    
    with col1:
        if st.session_state.get('signals_generated', False):
            # 生成模拟微观结构信号
            signals_data = generate_mock_microstructure_signals()
            
            # 显示关键指标
            col_a, col_b, col_c, col_d = st.columns(4)
            
            latest = signals_data.iloc[-1]
            
            with col_a:
                st.metric("VWAP", f"¥{latest['vwap']:.2f}")
            
            with col_b:
                st.metric("实现波动率", f"{latest['realized_vol']:.4f}")
            
            with col_c:
                st.metric("交易强度", f"{latest['trade_intensity']:.1f} 笔/秒")
            
            with col_d:
                st.metric("净订单流", f"{latest['order_flow']:+,}")
            
            # 创建综合信号图表
            fig = create_综合_signals_chart(signals_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # 信号强度雷达图
            st.markdown("### 📡 信号强度雷达图")
            
            # 归一化信号值用于雷达图
            normalized_signals = {
                'VWAP偏离': abs(latest['vwap'] - signals_data['vwap'].mean()) / signals_data['vwap'].std(),
                '波动率': latest['realized_vol'] / signals_data['realized_vol'].max(),
                '交易强度': latest['trade_intensity'] / signals_data['trade_intensity'].max(),
                '订单流': abs(latest['order_flow']) / abs(signals_data['order_flow']).max(),
            }
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=list(normalized_signals.values()),
                theta=list(normalized_signals.keys()),
                fill='toself',
                marker=dict(color='rgba(99, 110, 250, 0.6)')
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # 信号数据表
            st.markdown("### 📋 实时信号数据")
            st.dataframe(
                signals_data.tail(20).sort_index(ascending=False),
                use_container_width=True
            )
        
        else:
            st.info("👈 请点击左侧「计算微观结构信号」按钮开始分析")


# ===== 辅助函数 =====

def generate_mock_orderbook(symbol: str, depth: int) -> OrderBook:
    """生成模拟订单簿数据"""
    orderbook = OrderBook(symbol, depth)
    
    # 生成中间价
    mid_price = np.random.uniform(10, 50)
    
    # 生成买盘
    bids = []
    for i in range(depth):
        price = mid_price - (i + 1) * 0.01
        volume = int(np.random.exponential(1000) + 100)
        bids.append((price, volume))
    
    # 生成卖盘
    asks = []
    for i in range(depth):
        price = mid_price + (i + 1) * 0.01
        volume = int(np.random.exponential(1000) + 100)
        asks.append((price, volume))
    
    orderbook.update(bids, asks)
    return orderbook


def create_orderbook_chart(orderbook: OrderBook, symbol: str):
    """创建订单簿可视化图表"""
    
    # 准备数据
    bid_prices = [l.price for l in orderbook.bids]
    bid_volumes = [l.volume for l in orderbook.bids]
    bid_cumsum = np.cumsum(bid_volumes)
    
    ask_prices = [l.price for l in orderbook.asks]
    ask_volumes = [l.volume for l in orderbook.asks]
    ask_cumsum = np.cumsum(ask_volumes)
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=("订单簿深度图", "累计订单量"),
        vertical_spacing=0.15
    )
    
    # 订单簿深度 - 买盘
    fig.add_trace(
        go.Bar(
            x=bid_prices,
            y=bid_volumes,
            name='买盘',
            marker_color='green',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 订单簿深度 - 卖盘
    fig.add_trace(
        go.Bar(
            x=ask_prices,
            y=ask_volumes,
            name='卖盘',
            marker_color='red',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 累计订单量 - 买盘
    fig.add_trace(
        go.Scatter(
            x=bid_prices,
            y=bid_cumsum,
            name='累计买盘',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)'
        ),
        row=2, col=1
    )
    
    # 累计订单量 - 卖盘
    fig.add_trace(
        go.Scatter(
            x=ask_prices,
            y=ask_cumsum,
            name='累计卖盘',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        title=f"{symbol} 订单簿深度",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="价格", row=2, col=1)
    fig.update_yaxes(title_text="数量", row=1, col=1)
    fig.update_yaxes(title_text="累计数量", row=2, col=1)
    
    return fig


def generate_mock_spread_timeseries(time_window: str) -> pd.DataFrame:
    """生成模拟价差时间序列"""
    
    # 根据时间窗口确定数据点数
    window_map = {"1分钟": 60, "5分钟": 300, "15分钟": 900, "1小时": 3600}
    n_points = window_map.get(time_window, 300)
    
    # 生成时间序列
    now = datetime.now()
    timestamps = [now - timedelta(seconds=i) for i in range(n_points, 0, -1)]
    
    # 生成价差数据 (基础价差 + 随机波动)
    base_spread_bps = 2.0
    spreads_bps = base_spread_bps + np.random.normal(0, 0.5, n_points)
    spreads_bps = np.maximum(spreads_bps, 0.5)  # 确保非负
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'spread_bps': spreads_bps,
        'mid_price': 30 + np.cumsum(np.random.normal(0, 0.01, n_points))
    })


def create_spread_chart(spread_data: pd.DataFrame):
    """创建价差图表"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("价差变化", "中间价"),
        vertical_spacing=0.1
    )
    
    # 价差
    fig.add_trace(
        go.Scatter(
            x=spread_data['timestamp'],
            y=spread_data['spread_bps'],
            mode='lines',
            name='价差',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ),
        row=1, col=1
    )
    
    # 平均价差线
    avg_spread = spread_data['spread_bps'].mean()
    fig.add_hline(
        y=avg_spread,
        line_dash="dash",
        line_color="red",
        annotation_text=f"平均: {avg_spread:.2f}bps",
        row=1, col=1
    )
    
    # 中间价
    fig.add_trace(
        go.Scatter(
            x=spread_data['timestamp'],
            y=spread_data['mid_price'],
            mode='lines',
            name='中间价',
            line=dict(color='green', width=1.5)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="买卖价差时间序列",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="价差 (bps)", row=1, col=1)
    fig.update_yaxes(title_text="价格 (¥)", row=2, col=1)
    
    return fig


def generate_mock_order_flow(window_size: int) -> pd.DataFrame:
    """生成模拟订单流数据"""
    
    n_points = 200
    timestamps = [datetime.now() - timedelta(seconds=i) for i in range(n_points, 0, -1)]
    
    # 生成买卖量
    buy_volumes = np.random.poisson(500, n_points)
    sell_volumes = np.random.poisson(480, n_points)  # 略小于买入
    
    # 计算净流入
    net_flow = buy_volumes - sell_volumes
    cumulative_flow = np.cumsum(net_flow)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'buy_volume': buy_volumes,
        'sell_volume': sell_volumes,
        'net_flow': net_flow,
        'cumulative_flow': cumulative_flow
    })


def create_order_flow_chart(order_flow_data: pd.DataFrame):
    """创建订单流图表"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("买卖量对比", "净流入", "累计净流入"),
        vertical_spacing=0.08
    )
    
    # 买卖量
    fig.add_trace(
        go.Bar(
            x=order_flow_data['timestamp'],
            y=order_flow_data['buy_volume'],
            name='买入',
            marker_color='green',
            opacity=0.6
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=order_flow_data['timestamp'],
            y=-order_flow_data['sell_volume'],  # 负值显示
            name='卖出',
            marker_color='red',
            opacity=0.6
        ),
        row=1, col=1
    )
    
    # 净流入
    colors = ['green' if x > 0 else 'red' for x in order_flow_data['net_flow']]
    fig.add_trace(
        go.Bar(
            x=order_flow_data['timestamp'],
            y=order_flow_data['net_flow'],
            name='净流入',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 累计净流入
    fig.add_trace(
        go.Scatter(
            x=order_flow_data['timestamp'],
            y=order_flow_data['cumulative_flow'],
            name='累计净流入',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title="订单流分析",
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="时间", row=3, col=1)
    fig.update_yaxes(title_text="数量", row=1, col=1)
    fig.update_yaxes(title_text="净流入", row=2, col=1)
    fig.update_yaxes(title_text="累计", row=3, col=1)
    
    return fig


def generate_mock_microstructure_signals() -> pd.DataFrame:
    """生成模拟微观结构信号"""
    
    n_points = 100
    timestamps = [datetime.now() - timedelta(seconds=i*10) for i in range(n_points, 0, -1)]
    
    base_price = 30.0
    prices = base_price + np.cumsum(np.random.normal(0, 0.02, n_points))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'vwap': prices + np.random.normal(0, 0.01, n_points),
        'realized_vol': np.abs(np.random.normal(0.02, 0.005, n_points)),
        'order_flow': np.random.normal(0, 1000, n_points),
        'trade_intensity': np.random.uniform(5, 20, n_points)
    })


def create_综合_signals_chart(signals_data: pd.DataFrame):
    """创建综合信号图表"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.25, 0.25, 0.2],
        subplot_titles=("VWAP", "实现波动率", "订单流", "交易强度"),
        vertical_spacing=0.06
    )
    
    # VWAP
    fig.add_trace(
        go.Scatter(
            x=signals_data['timestamp'],
            y=signals_data['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 实现波动率
    fig.add_trace(
        go.Scatter(
            x=signals_data['timestamp'],
            y=signals_data['realized_vol'],
            mode='lines',
            name='波动率',
            line=dict(color='orange', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.2)'
        ),
        row=2, col=1
    )
    
    # 订单流
    colors = ['green' if x > 0 else 'red' for x in signals_data['order_flow']]
    fig.add_trace(
        go.Bar(
            x=signals_data['timestamp'],
            y=signals_data['order_flow'],
            name='订单流',
            marker_color=colors,
            opacity=0.6
        ),
        row=3, col=1
    )
    
    # 交易强度
    fig.add_trace(
        go.Scatter(
            x=signals_data['timestamp'],
            y=signals_data['trade_intensity'],
            mode='lines',
            name='交易强度',
            line=dict(color='purple', width=2)
        ),
        row=4, col=1
    )
    
    fig.update_layout(
        title="微观结构综合信号",
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="时间", row=4, col=1)
    fig.update_yaxes(title_text="价格 (¥)", row=1, col=1)
    fig.update_yaxes(title_text="波动率", row=2, col=1)
    fig.update_yaxes(title_text="净流入", row=3, col=1)
    fig.update_yaxes(title_text="笔/秒", row=4, col=1)
    
    return fig


if __name__ == "__main__":
    # 测试代码
    render_microstructure_tab()
