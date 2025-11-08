"""
核心指标仪表盘组件
展示4个关键指标：候选数、监控数、持仓、盈亏
"""

import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd


class MetricsDashboard:
    """核心指标仪表盘"""
    
    def __init__(self):
        """初始化指标仪表盘"""
        pass
    
    def render(self, metrics: Dict[str, Any]):
        """
        渲染指标仪表盘
        
        Args:
            metrics: 包含各项指标的字典
                - candidate_count: 候选股数量
                - monitor_count: 监控股数量
                - position_count: 持仓数量
                - position_value: 持仓市值
                - total_profit: 总盈亏
                - profit_rate: 盈亏比例
        """
        # 提取指标数据
        candidate_count = metrics.get('candidate_count', 0)
        monitor_count = metrics.get('monitor_count', 0)
        position_count = metrics.get('position_count', 0)
        position_value = metrics.get('position_value', 0.0)
        total_profit = metrics.get('total_profit', 0.0)
        profit_rate = metrics.get('profit_rate', 0.0)
        
        # 获取变化数据（用于显示delta）
        candidate_delta = metrics.get('candidate_delta', None)
        monitor_delta = metrics.get('monitor_delta', None)
        position_delta = metrics.get('position_delta', None)
        profit_delta = metrics.get('profit_delta', None)
        
        # 创建4列布局
        col1, col2, col3, col4 = st.columns(4)
        
        # 1. 候选池数量
        with col1:
            st.metric(
                label="📋 候选池",
                value=f"{candidate_count} 只",
                delta=candidate_delta,
                help="待筛选的涨停股数量，来源于T日涨停板"
            )
        
        # 2. 监控数量
        with col2:
            st.metric(
                label="👁️ 监控中",
                value=f"{monitor_count} 只",
                delta=monitor_delta,
                help="正在监控的股票数量，用于T+1竞价决策"
            )
        
        # 3. 持仓情况
        with col3:
            # 如果有持仓市值，显示市值，否则只显示数量
            if position_value > 0:
                value_str = f"{position_count}只 / {position_value/10000:.2f}万"
            else:
                value_str = f"{position_count} 只"
            
            st.metric(
                label="💼 持仓",
                value=value_str,
                delta=position_delta,
                help="当前持仓股票数量和总市值"
            )
        
        # 4. 盈亏情况
        with col4:
            # 根据盈亏情况显示不同颜色
            if total_profit > 0:
                profit_icon = "🟢"
            elif total_profit < 0:
                profit_icon = "🔴"
            else:
                profit_icon = "⚪"
            
            # 格式化盈亏显示
            if abs(total_profit) >= 10000:
                profit_str = f"{total_profit/10000:.2f}万"
            else:
                profit_str = f"{total_profit:.2f}"
            
            # 添加盈亏比例
            if profit_rate != 0:
                profit_str += f" ({profit_rate:+.2f}%)"
            
            st.metric(
                label=f"{profit_icon} 盈亏",
                value=profit_str,
                delta=profit_delta,
                delta_color="normal" if total_profit >= 0 else "inverse",
                help="当前持仓总盈亏和盈亏比例"
            )
    
    def render_detailed(self, metrics: Dict[str, Any], breakdown: Optional[pd.DataFrame] = None):
        """
        渲染详细的指标仪表盘（包含明细）
        
        Args:
            metrics: 包含各项指标的字典
            breakdown: 指标明细数据（可选）
        """
        # 先渲染核心指标
        self.render(metrics)
        
        # 如果有明细数据，显示展开/收起按钮
        if breakdown is not None and not breakdown.empty:
            st.markdown("---")
            
            with st.expander("📊 查看详细数据", expanded=False):
                # 显示详细表格
                st.dataframe(
                    breakdown,
                    use_container_width=True,
                    hide_index=True
                )
    
    def render_with_charts(self, metrics: Dict[str, Any], history: Optional[pd.DataFrame] = None):
        """
        渲染带图表的指标仪表盘
        
        Args:
            metrics: 当前指标数据
            history: 历史数据（用于绘制趋势图）
        """
        # 先渲染核心指标
        self.render(metrics)
        
        # 如果有历史数据，显示趋势图
        if history is not None and not history.empty:
            st.markdown("---")
            st.markdown("### 📈 趋势分析")
            
            # 创建2列布局显示图表
            col1, col2 = st.columns(2)
            
            with col1:
                # 候选/监控数量趋势
                if 'candidate_count' in history.columns and 'monitor_count' in history.columns:
                    chart_data = history[['date', 'candidate_count', 'monitor_count']].copy()
                    chart_data = chart_data.rename(columns={
                        'candidate_count': '候选数',
                        'monitor_count': '监控数'
                    })
                    st.line_chart(chart_data.set_index('date'))
            
            with col2:
                # 盈亏趋势
                if 'total_profit' in history.columns:
                    chart_data = history[['date', 'total_profit']].copy()
                    chart_data = chart_data.rename(columns={'total_profit': '总盈亏'})
                    st.line_chart(chart_data.set_index('date'))


def render_metrics_dashboard(
    candidate_count: int = 0,
    monitor_count: int = 0,
    position_count: int = 0,
    position_value: float = 0.0,
    total_profit: float = 0.0,
    profit_rate: float = 0.0,
    **kwargs
):
    """
    快速渲染指标仪表盘（便捷函数）
    
    Args:
        candidate_count: 候选股数量
        monitor_count: 监控股数量
        position_count: 持仓数量
        position_value: 持仓市值
        total_profit: 总盈亏
        profit_rate: 盈亏比例
        **kwargs: 其他指标（如delta值等）
    """
    metrics = {
        'candidate_count': candidate_count,
        'monitor_count': monitor_count,
        'position_count': position_count,
        'position_value': position_value,
        'total_profit': total_profit,
        'profit_rate': profit_rate,
        **kwargs
    }
    
    dashboard = MetricsDashboard()
    dashboard.render(metrics)


def create_metrics_from_data(
    limitup_df: Optional[pd.DataFrame] = None,
    candidate_df: Optional[pd.DataFrame] = None,
    monitor_df: Optional[pd.DataFrame] = None,
    position_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    从数据DataFrame自动创建指标字典
    
    Args:
        limitup_df: 涨停股数据
        candidate_df: 候选股数据
        monitor_df: 监控股数据
        position_df: 持仓数据
        
    Returns:
        指标字典
    """
    metrics = {
        'candidate_count': 0,
        'monitor_count': 0,
        'position_count': 0,
        'position_value': 0.0,
        'total_profit': 0.0,
        'profit_rate': 0.0
    }
    
    # 计算候选数量
    if candidate_df is not None and not candidate_df.empty:
        metrics['candidate_count'] = len(candidate_df)
    
    # 计算监控数量
    if monitor_df is not None and not monitor_df.empty:
        metrics['monitor_count'] = len(monitor_df)
    
    # 计算持仓情况
    if position_df is not None and not position_df.empty:
        metrics['position_count'] = len(position_df)
        
        # 如果有市值列，计算总市值
        if 'current_value' in position_df.columns:
            metrics['position_value'] = position_df['current_value'].sum()
        
        # 如果有盈亏列，计算总盈亏
        if 'profit' in position_df.columns:
            metrics['total_profit'] = position_df['profit'].sum()
        
        # 计算盈亏比例
        if 'cost_value' in position_df.columns and 'current_value' in position_df.columns:
            total_cost = position_df['cost_value'].sum()
            total_current = position_df['current_value'].sum()
            if total_cost > 0:
                metrics['profit_rate'] = ((total_current - total_cost) / total_cost) * 100
    
    return metrics


# 用于测试
if __name__ == "__main__":
    st.set_page_config(page_title="指标仪表盘测试", layout="wide")
    
    st.title("核心指标仪表盘测试")
    
    # 测试1: 基础指标
    st.markdown("## 基础指标展示")
    render_metrics_dashboard(
        candidate_count=15,
        monitor_count=8,
        position_count=5,
        position_value=123456.78,
        total_profit=5678.90,
        profit_rate=4.6,
        candidate_delta="+3",
        monitor_delta="-2",
        position_delta="+1",
        profit_delta="+1234.56"
    )
    
    # 测试2: 带明细的指标
    st.markdown("---")
    st.markdown("## 带明细的指标展示")
    
    breakdown_df = pd.DataFrame({
        '股票代码': ['000001', '000002', '000003', '000004', '000005'],
        '股票名称': ['平安银行', '万科A', '国农科技', '国华网安', 'ST星源'],
        '持仓数量': [1000, 2000, 1500, 1200, 800],
        '成本价': [10.5, 8.2, 15.3, 20.1, 5.5],
        '现价': [11.2, 8.0, 16.8, 21.5, 5.2],
        '盈亏': [700, -400, 2250, 1680, -240]
    })
    
    metrics = {
        'candidate_count': 15,
        'monitor_count': 8,
        'position_count': 5,
        'position_value': 123456.78,
        'total_profit': 3990,
        'profit_rate': 3.3
    }
    
    dashboard = MetricsDashboard()
    dashboard.render_detailed(metrics, breakdown_df)
    
    # 测试3: 带趋势图的指标
    st.markdown("---")
    st.markdown("## 带趋势图的指标展示")
    
    history_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'candidate_count': [12, 15, 18, 14, 16, 20, 17, 15, 13, 15],
        'monitor_count': [8, 10, 12, 9, 11, 13, 10, 8, 7, 8],
        'total_profit': [1000, 1500, 2000, 1800, 2500, 3000, 3500, 3800, 3600, 3990]
    })
    
    dashboard.render_with_charts(metrics, history_df)
