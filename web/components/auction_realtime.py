"""
竞价实时刷新和强度可视化组件
用于T+1竞价监控的实时数据展示
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import time


class AuctionRealtimeMonitor:
    """竞价实时监控组件"""
    
    def __init__(self, refresh_interval: int = 10, key_prefix: str = "auction"):
        """
        初始化竞价监控
        
        Args:
            refresh_interval: 刷新间隔（秒）
            key_prefix: 组件key前缀
        """
        self.refresh_interval = refresh_interval
        self.key_prefix = key_prefix
        
    def render_with_auto_refresh(self, data_loader: Callable, **kwargs) -> pd.DataFrame:
        """
        渲染带自动刷新的监控面板
        
        Args:
            data_loader: 数据加载函数
            **kwargs: 传递给data_loader的参数
            
        Returns:
            当前的数据DataFrame
        """
        # 创建刷新控制区
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            auto_refresh = st.checkbox(
                "🔄 自动刷新",
                value=False,
                key=f"{self.key_prefix}_auto_refresh",
                help="启用后数据将自动刷新"
            )
        
        with col2:
            if auto_refresh:
                interval = st.slider(
                    "刷新间隔(秒)",
                    5, 60, self.refresh_interval,
                    step=5,
                    key=f"{self.key_prefix}_interval"
                )
                self.refresh_interval = interval
        
        with col3:
            manual_refresh = st.button(
                "🔃 手动刷新",
                key=f"{self.key_prefix}_manual_refresh",
                use_container_width=True
            )
        
        # 显示刷新倒计时
        if auto_refresh:
            self._render_countdown(self.refresh_interval)
        
        # 加载数据
        data = data_loader(**kwargs)
        
        # 显示最后更新时间
        st.caption(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
        
        # 实现自动刷新（使用st.rerun触发）
        if auto_refresh:
            time.sleep(self.refresh_interval)
            st.rerun()
        
        return data
    
    def _render_countdown(self, seconds: int):
        """渲染倒计时"""
        # 使用占位符显示倒计时
        placeholder = st.empty()
        
        # 初始化session state存储倒计时
        if f'{self.key_prefix}_last_refresh' not in st.session_state:
            st.session_state[f'{self.key_prefix}_last_refresh'] = time.time()
        
        elapsed = time.time() - st.session_state[f'{self.key_prefix}_last_refresh']
        remaining = max(0, seconds - int(elapsed))
        
        placeholder.info(f"⏱️ 下次刷新倒计时: {remaining} 秒")
    
    def render_auction_strength_bars(self, data: pd.DataFrame):
        """
        渲染竞价强度条
        
        Args:
            data: 包含竞价数据的DataFrame，需要有auction_strength列
        """
        if data.empty or 'auction_strength' not in data.columns:
            st.warning("无竞价强度数据")
            return
        
        st.markdown("#### 📊 竞价强度实时监控")
        
        for idx, row in data.head(10).iterrows():
            symbol = row.get('symbol', 'N/A')
            name = row.get('name', 'N/A')
            strength = row.get('auction_strength', 0)
            
            # 根据强度确定颜色和等级
            color, level, emoji = self._get_strength_level(strength)
            
            # 渲染强度条
            col1, col2, col3 = st.columns([2, 5, 2])
            
            with col1:
                st.write(f"**{symbol}**")
                st.caption(name)
            
            with col2:
                # 使用progress bar
                progress_val = min(abs(strength) / 10, 1.0)
                if strength >= 0:
                    st.progress(progress_val)
                else:
                    st.progress(0.0)
                st.caption(f"{strength:+.2f}%")
            
            with col3:
                st.markdown(f"{emoji} {level}")
        
    def render_auction_timeline(self, symbol: str, timeline_data: list):
        """
        渲染单个股票的竞价时间线
        
        Args:
            symbol: 股票代码
            timeline_data: 时间线数据列表 [{'time': '9:20', 'strength': 5.2}, ...]
        """
        if not timeline_data:
            st.warning("无时间线数据")
            return
        
        st.markdown(f"#### 🕐 {symbol} 竞价时间线")
        
        times = [d['time'] for d in timeline_data]
        strengths = [d['strength'] for d in timeline_data]
        
        # 创建折线图
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=strengths,
            mode='lines+markers+text',
            name='竞价强度',
            line=dict(color='royalblue', width=3),
            marker=dict(size=10, color='lightblue', line=dict(color='royalblue', width=2)),
            text=[f"{s:+.1f}%" for s in strengths],
            textposition='top center',
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.1)'
        ))
        
        # 添加强弱线
        fig.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="强势线(+5%)")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_hline(y=-5, line_dash="dash", line_color="red", annotation_text="弱势线(-5%)")
        
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="竞价涨幅 (%)",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示AI建议
        latest_strength = strengths[-1] if strengths else 0
        self._render_ai_suggestion(symbol, latest_strength)
    
    def render_strength_distribution(self, data: pd.DataFrame):
        """
        渲染竞价强度分布图
        
        Args:
            data: 包含竞价数据的DataFrame
        """
        if data.empty or 'auction_strength' not in data.columns:
            st.warning("无竞价强度数据")
            return
        
        st.markdown("#### 📊 竞价强度分布")
        
        strengths = data['auction_strength'].values
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=strengths,
            nbinsx=30,
            marker_color='steelblue',
            opacity=0.7,
            name='竞价强度分布'
        ))
        
        # 添加参考线
        fig.add_vline(x=5, line_dash="dash", line_color="green", annotation_text="强势")
        fig.add_vline(x=0, line_dash="solid", line_color="gray")
        fig.add_vline(x=-5, line_dash="dash", line_color="red", annotation_text="弱势")
        
        fig.update_layout(
            xaxis_title="竞价强度 (%)",
            yaxis_title="频数",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            strong = (strengths > 5).sum()
            st.metric("强势股", f"{strong} 只", help="竞价涨幅>5%")
        with col2:
            weak = (strengths < -5).sum()
            st.metric("弱势股", f"{weak} 只", help="竞价跌幅>5%")
        with col3:
            avg = strengths.mean()
            st.metric("平均涨幅", f"{avg:+.2f}%")
        with col4:
            median = pd.Series(strengths).median()
            st.metric("中位数", f"{median:+.2f}%")
    
    def _get_strength_level(self, strength: float) -> tuple:
        """
        根据强度值返回颜色、等级、emoji
        
        Returns:
            (color, level, emoji)
        """
        if strength >= 8:
            return "green", "极强", "🟢💪💪💪"
        elif strength >= 5:
            return "lightgreen", "强势", "🟢💪💪"
        elif strength >= 2:
            return "yellow", "良好", "🟡💪"
        elif strength >= -2:
            return "orange", "观望", "🟡"
        elif strength >= -5:
            return "lightcoral", "走弱", "🔴"
        else:
            return "red", "弱势", "🔴⚠️"
    
    def _render_ai_suggestion(self, symbol: str, strength: float):
        """渲染AI买入建议"""
        if strength >= 8:
            st.success(f"💡 **{symbol}** 竞价极强（{strength:+.2f}%），建议重点关注，优先买入！")
        elif strength >= 5:
            st.info(f"💡 **{symbol}** 竞价表现稳健（{strength:+.2f}%），可考虑买入")
        elif strength >= 0:
            st.warning(f"⚠️  **{symbol}** 竞价涨幅一般（{strength:+.2f}%），建议谨慎观望")
        else:
            st.error(f"❌ **{symbol}** 竞价走弱（{strength:+.2f}%），建议放弃")


def create_test_auction_data(count: int = 10) -> pd.DataFrame:
    """创建测试竞价数据"""
    import numpy as np
    np.random.seed(42)
    
    return pd.DataFrame({
        'symbol': [f"30{i:04d}" for i in range(count)],
        'name': [f"测试股{i}" for i in range(count)],
        'auction_strength': np.random.uniform(-8, 12, count),
        'auction_change': np.random.uniform(-10, 15, count),
        'volume_ratio': np.random.uniform(0.5, 3.0, count)
    })


# 测试代码
if __name__ == "__main__":
    st.set_page_config(page_title="竞价实时监控测试", layout="wide")
    
    st.title("🔥 竞价实时监控测试")
    
    # 创建监控器
    monitor = AuctionRealtimeMonitor(refresh_interval=10, key_prefix="test")
    
    # 数据加载函数
    def load_test_data():
        return create_test_auction_data(15)
    
    # 渲染带自动刷新的监控
    st.markdown("## 📊 实时监控面板")
    data = monitor.render_with_auto_refresh(load_test_data)
    
    st.markdown("---")
    
    # 渲染强度条
    monitor.render_auction_strength_bars(data)
    
    st.markdown("---")
    
    # 渲染强度分布
    monitor.render_strength_distribution(data)
    
    st.markdown("---")
    
    # 渲染时间线（示例）
    test_timeline = [
        {'time': '9:20', 'strength': 3.5},
        {'time': '9:22', 'strength': 6.2},
        {'time': '9:24', 'strength': 8.9},
        {'time': '9:25', 'strength': 9.5}
    ]
    
    if not data.empty:
        test_symbol = data.iloc[0]['symbol']
        monitor.render_auction_timeline(test_symbol, test_timeline)
