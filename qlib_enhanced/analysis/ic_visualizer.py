"""
IC可视化模块
Phase 5.3 实现

提供plotly图表：
- IC时间序列图
- 月度IC热力图
- 分层收益柱状图
- IC分布直方图
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_ic_timeseries(ic_series: pd.Series, title: str = "IC时间序列") -> go.Figure:
    """绘制IC时间序列图（折线+移动平均+零线）"""
    fig = go.Figure()
    
    # 原始IC序列
    fig.add_trace(go.Scatter(
        x=ic_series.index,
        y=ic_series.values,
        mode='lines',
        name='IC',
        line=dict(color='steelblue', width=1),
        opacity=0.7
    ))
    
    # 20日移动平均
    if len(ic_series) >= 20:
        ma20 = ic_series.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=ma20.index,
            y=ma20.values,
            mode='lines',
            name='IC(MA20)',
            line=dict(color='orange', width=2)
        ))
    
    # 零线
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="IC",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig


def plot_monthly_ic_heatmap(monthly_ic_df: pd.DataFrame, title: str = "月度IC热力图") -> go.Figure:
    """绘制月度IC热力图"""
    # monthly_ic_df: index=Year, columns=1-12
    fig = go.Figure(data=go.Heatmap(
        z=monthly_ic_df.values,
        x=[f"{m}月" for m in monthly_ic_df.columns],
        y=monthly_ic_df.index.astype(str),
        colorscale='RdYlGn',
        zmid=0,
        text=monthly_ic_df.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        colorbar=dict(title="IC")
    ))
    fig.update_layout(
        title=title,
        xaxis_title="月份",
        yaxis_title="年份",
        height=400,
        template='plotly_white'
    )
    return fig


def plot_layered_returns(layered_df: pd.DataFrame, title: str = "分层收益分析") -> go.Figure:
    """绘制分层收益柱状图"""
    # layered_df: columns=['layer','mean_ret','count','long_short']
    fig = go.Figure()
    
    colors = ['red' if x < 0 else 'green' for x in layered_df['mean_ret']]
    
    fig.add_trace(go.Bar(
        x=[f"Q{int(l)}" for l in layered_df['layer']],
        y=layered_df['mean_ret'],
        marker_color=colors,
        text=[f"{r:.2%}" for r in layered_df['mean_ret']],
        textposition='outside',
        name='平均收益率'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="分位数（Q1=低因子值，Q5=高因子值）",
        yaxis_title="平均收益率",
        yaxis=dict(tickformat='.2%'),
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def plot_ic_distribution(ic_series: pd.Series, title: str = "IC分布直方图") -> go.Figure:
    """绘制IC分布直方图+核密度"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ic_series.dropna(),
        nbinsx=50,
        name='IC',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    # 添加统计信息
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    
    fig.add_vline(x=mean_ic, line_dash="dash", line_color="red", 
                  annotation_text=f"均值: {mean_ic:.4f}", 
                  annotation_position="top right")
    
    fig.update_layout(
        title=title,
        xaxis_title="IC",
        yaxis_title="频数",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    return fig


def plot_ic_rolling_stats(ic_series: pd.Series, window: int = 60, title: str = "IC滚动统计") -> go.Figure:
    """绘制IC滚动均值和标准差"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("滚动均值", "滚动标准差"),
        vertical_spacing=0.15,
        shared_xaxes=True
    )
    
    rolling_mean = ic_series.rolling(window).mean()
    rolling_std = ic_series.rolling(window).std()
    
    # 滚动均值
    fig.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        mode='lines',
        name=f'滚动均值({window}天)',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # 滚动标准差
    fig.add_trace(go.Scatter(
        x=rolling_std.index,
        y=rolling_std.values,
        mode='lines',
        name=f'滚动标准差({window}天)',
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,165,0,0.2)'
    ), row=2, col=1)
    
    fig.update_layout(
        title_text=title,
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="IC", row=1, col=1)
    fig.update_yaxes(title_text="标准差", row=2, col=1)
    
    return fig


def plot_cumulative_ic(ic_series: pd.Series, title: str = "累积IC") -> go.Figure:
    """绘制累积IC曲线"""
    cumsum_ic = ic_series.cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumsum_ic.index,
        y=cumsum_ic.values,
        mode='lines',
        name='累积IC',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        fillcolor='rgba(128,0,128,0.1)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="累积IC",
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig
