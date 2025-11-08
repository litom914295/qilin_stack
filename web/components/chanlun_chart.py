"""缠论交互式图表 - Phase P0-4
Plotly图表,研发效率+50%

功能:
- K线图 + 分型标记
- 笔/线段连线
- 中枢矩形
- 买卖点标注
- MACD副图
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional

class ChanLunChartComponent:
    """缠论交互式图表"""
    
    def __init__(self, width: int = 1400, height: int = 900):
        """初始化
        
        Args:
            width: 图表宽度
            height: 图表高度
        """
        self.width = width
        self.height = height
    
    def render_chanlun_chart(self, df: pd.DataFrame, chan_features: Optional[Dict] = None):
        """绘制完整缠论图表
        
        Args:
            df: DataFrame with [datetime, open, high, low, close, volume]
            chan_features: 缠论特征字典 {
                'fx_mark': Series,      # 分型标记 (1=顶, -1=底)
                'bi_points': List,      # 笔端点列表
                'seg_points': List,     # 线段端点列表
                'zs_list': List,        # 中枢列表
                'buy_points': List,     # 买点列表
                'sell_points': List,    # 卖点列表
            }
        
        Returns:
            plotly Figure对象
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('缠论分析图', 'MACD'),
            row_heights=[0.7, 0.3]
        )
        
        # 1. K线图
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='red',
            decreasing_line_color='green'
        ), row=1, col=1)
        
        if chan_features:
            # 2. 线段连线
            if 'seg_points' in chan_features:
                self._add_seg_lines(fig, chan_features['seg_points'])
            
            # 3. 笔连线
            if 'bi_points' in chan_features:
                self._add_bi_lines(fig, chan_features['bi_points'])
            
            # 4. 中枢矩形
            if 'zs_list' in chan_features:
                self._add_zs_rectangles(fig, chan_features['zs_list'])
            
            # 5. 分型标记
            if 'fx_mark' in chan_features:
                self._add_fractal_marks(fig, df, chan_features['fx_mark'])
            
            # 6. 买卖点标注
            if 'buy_points' in chan_features:
                self._add_buy_sell_points(fig, chan_features['buy_points'], chan_features.get('sell_points', []))
        
        # 7. MACD副图
        self._add_macd_subplot(fig, df, row=2)
        
        # 8. 图表样式
        fig.update_layout(
            title={
                'text': '缠论分析图表',
                'x': 0.5,
                'xanchor': 'center'
            },
            width=self.width,
            height=self.height,
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def _add_seg_lines(self, fig, seg_points):
        """添加线段连线"""
        if not seg_points:
            return
        
        x_vals = [p['datetime'] for p in seg_points]
        y_vals = [p['price'] for p in seg_points]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(color='blue', width=2),
            name='线段'
        ), row=1, col=1)
    
    def _add_bi_lines(self, fig, bi_points):
        """添加笔连线"""
        if not bi_points:
            return
        
        x_vals = [p['datetime'] for p in bi_points]
        y_vals = [p['price'] for p in bi_points]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(color='purple', width=1, dash='dot'),
            name='笔'
        ), row=1, col=1)
    
    def _add_zs_rectangles(self, fig, zs_list):
        """添加中枢矩形"""
        for zs in zs_list:
            fig.add_shape(
                type="rect",
                x0=zs['start_time'],
                x1=zs['end_time'],
                y0=zs['low'],
                y1=zs['high'],
                fillcolor="yellow",
                opacity=0.2,
                line=dict(color="orange", width=1),
                row=1, col=1
            )
    
    def _add_fractal_marks(self, fig, df, fx_mark):
        """添加分型标记"""
        # 顶分型
        top_fx = df[fx_mark == 1]
        if len(top_fx) > 0:
            fig.add_trace(go.Scatter(
                x=top_fx['datetime'],
                y=top_fx['high'] * 1.01,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='顶分型'
            ), row=1, col=1)
        
        # 底分型
        bottom_fx = df[fx_mark == -1]
        if len(bottom_fx) > 0:
            fig.add_trace(go.Scatter(
                x=bottom_fx['datetime'],
                y=bottom_fx['low'] * 0.99,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='底分型'
            ), row=1, col=1)
    
    def _add_buy_sell_points(self, fig, buy_points, sell_points):
        """添加买卖点标注"""
        # 买点
        for bp in buy_points:
            fig.add_annotation(
                x=bp['datetime'],
                y=bp['price'] * 0.97,
                text=f"买{bp.get('type', '?')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor='green',
                arrowsize=1.5,
                arrowwidth=2,
                bgcolor='lightgreen',
                row=1, col=1
            )
        
        # 卖点
        for sp in sell_points:
            fig.add_annotation(
                x=sp['datetime'],
                y=sp['price'] * 1.03,
                text=f"卖{sp.get('type', '?')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                arrowsize=1.5,
                arrowwidth=2,
                bgcolor='lightcoral',
                row=1, col=1
            )
    
    def _add_macd_subplot(self, fig, df, row):
        """添加MACD副图"""
        # 计算MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        
        # MACD柱状图
        colors = ['red' if m >= 0 else 'green' for m in macd]
        fig.add_trace(go.Bar(
            x=df['datetime'],
            y=macd,
            marker_color=colors,
            name='MACD',
            showlegend=False
        ), row=row, col=1)
        
        # DIF线
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=dif,
            mode='lines',
            line=dict(color='blue', width=1),
            name='DIF'
        ), row=row, col=1)
        
        # DEA线
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=dea,
            mode='lines',
            line=dict(color='orange', width=1),
            name='DEA'
        ), row=row, col=1)

if __name__ == '__main__':
    print("✅ P0-4: 缠论图表组件创建完成")
