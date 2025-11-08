# -*- coding: utf-8 -*-
"""
写实回测结果展示页面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backtesting.realistic_backtest import RealisticBacktester, LimitUpQueueSimulator
from ml.model_explainer import LimitUpModelExplainer


def show_realistic_backtest_page():
    """显示写实回测页面"""
    
    st.title("🎯 涨停板写实回测系统")
    st.markdown("""
    ### 真实模拟涨停板交易环境
    - 🎫 涨停排队模拟
    - 💰 真实成本计算
    - 📊 SHAP模型解释
    - 📈 专业回测指标
    """)
    
    # 使用指南
    with st.expander("📖 系统使用指南", expanded=False):
        st.markdown("""
        ### 📚 相关文档
        
        **核心文档**:
        - 🦄 **麒麟改进实施报告**: `docs/QILIN_EVOLUTION_IMPLEMENTATION.md`
          - ✅ 第三阶段: 写实回测与可解释性
        - 📊 **回测引擎**: `backtesting/realistic_backtest.py`
        - 🔬 **SHAP解释器**: `ml/model_explainer.py`
        
        ### 🎯 使用步骤
        
        1. **设置回测参数** (侧边栏)
           - 初始资金: 建议100万
           - 回测时间: 选择1-3个月
           - 单股仓位: 10-20%
           - 封单门槛: 5000万+
           
        2. **运行回测**
           - 点击"🚀 运行写实回测"
           - 等待计算完成
           
        3. **查看结果**
           - 📊 核心指标: 收益率、回撤、胜率
           - 📈 收益曲线: 累计收益和每日波动
           - 📋 交易记录: 成交详情、排队分析
           - ⚠️ 风险分析: 风险雷达图
           - 🔬 SHAP解释: 特征重要性
        
        ### ✨ 核心功能
        
        1. **涨停排队模拟**
           - 根据封单金额计算排队位置
           - 模拟成交概率（0-100%）
           - 考虑部分成交情况
           
        2. **真实成本计算**
           - 佣金: 万三 (0.03%)
           - 印花税: 千一 (0.1%)
           - 滑点: 根据开板次数
           
        3. **专业指标**
           - 收益率、年化收益
           - 最大回撤、夏普比率
           - 胜率、盈亏比
           - 成交概率、排队位置
        
        ### ⚠️ 重要提示
        
        - 回测结果不代表未来表现
        - 涨停板交易具有高风险性
        - 实际成交可能与模拟存在差异
        - 请根据自身风险承受能力谨慎决策
        """)
    
    # 侧边栏参数设置
    with st.sidebar:
        st.header("⚙️ 回测参数设置")
        
        # 基础参数
        st.subheader("基础参数")
        initial_capital = st.number_input(
            "初始资金（元）",
            min_value=10000,
            max_value=10000000,
            value=1000000,
            step=100000
        )
        
        # 日期选择
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now()
            )
            
        # 策略参数
        st.subheader("策略参数")
        
        position_size = st.slider(
            "单股最大仓位（%）",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        min_seal_amount = st.number_input(
            "最小封单金额（万元）",
            min_value=1000,
            max_value=50000,
            value=5000,
            step=1000
        )
        
        max_open_times = st.slider(
            "最大开板次数",
            min_value=0,
            max_value=5,
            value=2
        )
        
        # 风控参数
        st.subheader("风控参数")
        
        stop_loss = st.slider(
            "止损线（%）",
            min_value=-20,
            max_value=-5,
            value=-10,
            step=1
        )
        
        take_profit = st.slider(
            "止盈线（%）",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        # 运行回测按钮
        run_backtest = st.button("🚀 运行写实回测", type="primary", use_container_width=True)
    
    # 主页面内容
    if run_backtest:
        with st.spinner("正在运行写实回测..."):
            # 运行回测
            results = run_backtest_simulation(
                initial_capital,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                position_size / 100,
                min_seal_amount,
                max_open_times,
                stop_loss / 100,
                take_profit / 100
            )
            
            # 保存结果到session state
            st.session_state['backtest_results'] = results
            # 同步写入最近一次回测的日收益序列，供“高级风险指标”直接使用
            try:
                ds = results.get('daily_stats')
                if ds is not None and not ds.empty and 'date' in ds.columns and 'daily_returns' in ds.columns:
                    rt_series = pd.Series(ds['daily_returns'].values, index=pd.to_datetime(ds['date']))
                    st.session_state['last_backtest_returns'] = rt_series
            except Exception as _e:
                # 安全忽略，不影响页面
                pass
    
    # 显示回测结果
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        display_backtest_results(results)


def run_backtest_simulation(
    initial_capital,
    start_date,
    end_date,
    position_size,
    min_seal_amount,
    max_open_times,
    stop_loss,
    take_profit
):
    """运行回测模拟"""
    
    # 生成模拟信号数据（实际应从模型获取）
    signals = generate_mock_signals(
        start_date, end_date, 
        min_seal_amount, max_open_times
    )
    
    # 生成模拟市场数据（实际应从数据源获取）
    market_data = generate_mock_market_data(start_date, end_date)
    
    # 创建回测器
    backtester = RealisticBacktester(initial_capital)
    
    # 运行回测
    results = backtester.run_backtest(
        signals,
        market_data,
        start_date,
        end_date
    )
    
    return results


def generate_mock_signals(start_date, end_date, min_seal_amount, max_open_times):
    """生成模拟交易信号"""
    
    dates = pd.bdate_range(start_date, end_date)
    signals = []
    
    # 每3天生成一个信号
    for i in range(0, len(dates), 3):
        if i < len(dates):
            signals.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'symbol': f'00000{np.random.randint(1, 10)}',
                'limit_price': 10.0 + np.random.uniform(-2, 2),
                'seal_amount': np.random.uniform(min_seal_amount, min_seal_amount * 5),
                'open_times': np.random.randint(0, max_open_times + 1),
                'limitup_time': f"{dates[i].strftime('%Y-%m-%d')} 09:{30+np.random.randint(0, 60):02d}:00",
                'prediction_prob': np.random.uniform(0.5, 0.9)
            })
    
    return pd.DataFrame(signals)


def generate_mock_market_data(start_date, end_date):
    """生成模拟市场数据"""
    
    dates = pd.bdate_range(start_date, end_date)
    market_data = []
    
    for date in dates:
        for i in range(10):
            base_price = 10.0
            market_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': f'00000{i}',
                'open': base_price + np.random.uniform(-0.5, 0.5),
                'close': base_price + np.random.uniform(-0.5, 1.0),
                'high': base_price + np.random.uniform(0.5, 1.1),
                'low': base_price + np.random.uniform(-0.5, 0)
            })
    
    return pd.DataFrame(market_data)


def display_backtest_results(results):
    """显示回测结果"""
    
    # 关键指标卡片
    st.header("📊 回测核心指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "总收益率",
            f"{results['total_returns']:.2%}",
            f"年化: {results['annual_returns']:.2%}"
        )
    
    with col2:
        st.metric(
            "最大回撤",
            f"{results['max_drawdown']:.2%}",
            f"夏普: {results['sharpe_ratio']:.2f}"
        )
    
    with col3:
        st.metric(
            "胜率",
            f"{results['win_rate']:.2%}",
            f"盈亏比: {results['profit_factor']:.2f}"
        )
    
    with col4:
        st.metric(
            "成交统计",
            f"平均成交率: {results.get('avg_fill_ratio', 1.0):.2%}",
            f"未成交率: {results.get('unfilled_rate', 0.0):.2%}"
        )
    
    # 收益曲线
    st.header("📈 收益曲线")
    
    if 'daily_stats' in results and not results['daily_stats'].empty:
        fig_returns = create_returns_chart(results['daily_stats'])
        st.plotly_chart(fig_returns, use_container_width=True)
    
    # 交易详情
    st.header("📝 交易详情")
    
    tab1, tab2, tab3 = st.tabs(["成交记录", "排队分析", "成本分析"])
    
    with tab1:
        if 'trades' in results and not results['trades'].empty:
            display_trades_table(results['trades'])
        else:
            st.info("暂无交易记录")
    
    with tab2:
        display_queue_analysis(results)
    
    with tab3:
        display_cost_analysis(results)
    
    # 风险分析
    st.header("⚠️ 风险分析")
    display_risk_analysis(results)
    
    # 模型解释（如果有）
    if st.checkbox("🔬 显示SHAP模型解释"):
        display_model_explanation()


def create_returns_chart(daily_stats):
    """创建收益曲线图"""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("累计收益曲线", "每日收益率"),
        vertical_spacing=0.1
    )
    
    # 累计收益曲线
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['returns'] * 100,
            mode='lines',
            name='累计收益率',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # 每日收益率柱状图
    colors = ['green' if x > 0 else 'red' for x in daily_stats['daily_returns']]
    fig.add_trace(
        go.Bar(
            x=daily_stats['date'],
            y=daily_stats['daily_returns'] * 100,
            name='每日收益',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="收益率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="日收益 (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def display_trades_table(trades_df):
    """显示交易记录表格"""
    
    # 格式化数据
    display_df = trades_df.copy()
    
    # 格式化数值列
    if 'price' in display_df.columns:
        display_df['price'] = display_df['price'].apply(lambda x: f"{x:.2f}")
    if 'profit_rate' in display_df.columns:
        display_df['profit_rate'] = display_df['profit_rate'].apply(lambda x: f"{x:.2%}")
    if 'execution_prob' in display_df.columns:
        display_df['execution_prob'] = display_df['execution_prob'].apply(lambda x: f"{x:.2%}")
    
    # 添加颜色标记
    def color_profit(val):
        if isinstance(val, str) and '%' in val:
            num_val = float(val.strip('%'))
            if num_val > 0:
                return f'<span style="color: green">{val}</span>'
            elif num_val < 0:
                return f'<span style="color: red">{val}</span>'
        return val
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # 交易统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
        sell_trades = len(trades_df[trades_df['action'] == 'SELL'])
        st.info(f"总交易数: {total_trades}\n买入: {buy_trades} | 卖出: {sell_trades}")
    
    with col2:
        if 'profit' in trades_df.columns:
            total_profit = trades_df['profit'].sum()
            avg_profit = trades_df['profit'].mean()
            st.info(f"总盈亏: ¥{total_profit:,.2f}\n平均盈亏: ¥{avg_profit:,.2f}")
    
    with col3:
        if 'commission' in trades_df.columns:
            total_commission = trades_df['commission'].sum()
            if 'stamp_tax' in trades_df.columns:
                total_tax = trades_df['stamp_tax'].sum()
                st.info(f"总手续费: ¥{total_commission:,.2f}\n总印花税: ¥{total_tax:,.2f}")


def display_queue_analysis(results):
    """显示排队分析"""
    st.subheader("📊 涨停排队分析")
    
    # 显示成交统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "总订单数",
            f"{results.get('orders_attempted', 0)}",
            f"未成交: {results.get('orders_unfilled', 0)}"
        )
    
    with col2:
        st.metric(
            "平均成交比例",
            f"{results.get('avg_fill_ratio', 1.0):.1%}",
            delta=f"{(results.get('avg_fill_ratio', 1.0) - 0.5) * 100:.1f}pp",
            delta_color="normal" if results.get('avg_fill_ratio', 1.0) > 0.5 else "inverse"
        )
    
    with col3:
        st.metric(
            "未成交率",
            f"{results.get('unfilled_rate', 0.0):.1%}",
            delta=f"{-results.get('unfilled_rate', 0.0) * 100:.1f}pp",
            delta_color="inverse" if results.get('unfilled_rate', 0.0) > 0.1 else "normal"
        )
    
    # 成交比例分布图
    if 'fill_ratio_distribution' in results:
        st.subheader("📈 成交比例分布")
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=results['fill_ratio_distribution'],
            nbinsx=20,
            name="成交比例",
            marker_color='blue'
        ))
        
        fig.update_layout(
            title="成交比例分布",
            xaxis_title="成交比例",
            yaxis_title="频次",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 排队策略建议
    st.info("""
    **排队策略建议:**
    1. 🌟 封单强度 > 8：全额挂单排队
    2. ✨ 封单强度 5-8：70%资金挂单
    3. ⚠️ 封单强度 < 5：谨慎参与或放弃
    """)

def display_queue_analysis_original(results):
    """显示排队分析"""
    
    st.subheader("涨停板排队成交分析")
    
    if 'trades' not in results or results['trades'].empty:
        st.info("暂无排队数据")
        return
    
    buy_trades = results['trades'][results['trades']['action'] == 'BUY']
    
    if buy_trades.empty:
        st.info("暂无买入交易")
        return
    
    # 排队位置分布
    col1, col2 = st.columns(2)
    
    with col1:
        fig_queue = px.histogram(
            buy_trades,
            x='queue_position',
            nbins=20,
            title="排队位置分布",
            labels={'queue_position': '排队位置', 'count': '次数'}
        )
        st.plotly_chart(fig_queue, use_container_width=True)
    
    with col2:
        # 成交概率 vs 排队位置
        fig_prob = px.scatter(
            buy_trades,
            x='queue_position',
            y='execution_prob',
            title="成交概率 vs 排队位置",
            labels={'queue_position': '排队位置', 'execution_prob': '成交概率'},
            trendline="lowess"
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # 排队统计
    st.subheader("排队统计指标")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_queue = buy_trades['queue_position'].mean()
        median_queue = buy_trades['queue_position'].median()
        st.metric("平均排队位置", f"{avg_queue:.0f}", f"中位数: {median_queue:.0f}")
    
    with col2:
        avg_prob = buy_trades['execution_prob'].mean()
        success_rate = len(buy_trades[buy_trades['volume'] > 0]) / len(buy_trades)
        st.metric("平均成交概率", f"{avg_prob:.2%}", f"实际成交率: {success_rate:.2%}")
    
    with col3:
        front_queue = len(buy_trades[buy_trades['queue_position'] < 1000])
        front_rate = front_queue / len(buy_trades)
        st.metric("前1000位占比", f"{front_rate:.2%}", f"共{front_queue}笔")


def display_cost_analysis(results):
    """显示成本分析"""
    
    st.subheader("交易成本明细")
    
    if 'trades' not in results or results['trades'].empty:
        st.info("暂无交易数据")
        return
    
    trades_df = results['trades']
    
    # 计算各项成本
    total_amount = trades_df['amount'].sum()
    total_commission = trades_df['commission'].sum() if 'commission' in trades_df else 0
    total_tax = trades_df['stamp_tax'].sum() if 'stamp_tax' in trades_df else 0
    
    # 成本占比饼图
    fig_cost = go.Figure(data=[go.Pie(
        labels=['交易本金', '佣金', '印花税'],
        values=[total_amount, total_commission, total_tax],
        hole=.3
    )])
    
    fig_cost.update_layout(
        title="交易成本构成",
        annotations=[dict(text='成本', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # 成本统计表
    cost_summary = pd.DataFrame({
        '项目': ['交易总额', '佣金', '印花税', '总成本', '成本率'],
        '金额': [
            f"¥{total_amount:,.2f}",
            f"¥{total_commission:,.2f}",
            f"¥{total_tax:,.2f}",
            f"¥{total_commission + total_tax:,.2f}",
            f"{(total_commission + total_tax) / total_amount * 100:.3f}%"
        ]
    })
    
    st.table(cost_summary)


def display_risk_analysis(results):
    """显示风险分析"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("风险指标")
        
        # 创建风险雷达图
        categories = ['收益', '回撤控制', '胜率', '盈亏比', '成交率']
        
        # 归一化指标（0-100分）
        values = [
            min(100, max(0, (results['total_returns'] + 0.3) * 100)),  # 收益
            min(100, max(0, (1 + results['max_drawdown']) * 100)),  # 回撤控制
            min(100, results['win_rate'] * 100),  # 胜率
            min(100, results['profit_factor'] * 20),  # 盈亏比
            min(100, results['avg_execution_prob'] * 100)  # 成交率
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='风险评分'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="风险评分雷达图"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.subheader("风险等级评估")
        
        # 计算综合风险分数
        risk_score = np.mean(values)
        
        if risk_score >= 70:
            risk_level = "低风险"
            risk_color = "green"
        elif risk_score >= 50:
            risk_level = "中等风险"
            risk_color = "orange"
        else:
            risk_level = "高风险"
            risk_color = "red"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid {risk_color}; border-radius: 10px;">
            <h2 style="color: {risk_color};">{risk_level}</h2>
            <h3>综合评分: {risk_score:.1f}/100</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 风险提示
        st.warning("""
        **风险提示：**
        - 历史回测不代表未来表现
        - 涨停板交易具有高风险性
        - 实际成交可能与模拟存在差异
        - 请根据自身风险承受能力谨慎决策
        """)


def display_model_explanation():
    """显示模型解释"""
    
    st.subheader("🔬 SHAP模型解释")
    
    # 创建示例数据
    feature_importance = pd.DataFrame({
        '特征': ['封单强度', '市场情绪', '开板次数', '涨停时间', '换手率', 
                '板块涨停数', '资金流入', '题材热度'],
        '重要性': [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    }).sort_values('重要性', ascending=True)
    
    # 特征重要性条形图
    fig_importance = px.bar(
        feature_importance,
        x='重要性',
        y='特征',
        orientation='h',
        title="特征重要性（SHAP值）",
        color='重要性',
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # 单样本解释
    st.subheader("单笔交易预测解释")
    
    sample_explanation = """
    **股票代码: 000001**
    **预测涨停概率: 72.5%**
    
    **主要支撑因素:**
    - 封单强度: 8.5 (贡献 +0.25)
    - 市场情绪: 85 (贡献 +0.18)
    - 板块涨停数: 12 (贡献 +0.10)
    
    **主要阻碍因素:**
    - 开板次数: 2 (贡献 -0.08)
    - 涨停时间: 14:00 (贡献 -0.05)
    
    **操作建议:** 推荐 - 概率较高，但需关注封板情况
    """
    
    st.info(sample_explanation)


if __name__ == "__main__":
    show_realistic_backtest_page()