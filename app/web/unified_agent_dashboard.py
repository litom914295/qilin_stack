"""
统一的智能体协作可视化界面
整合TradingAgents原生智能体和Qilin自定义智能体
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

# 导入核心模块
from app.core.tradingagents_native_integration import (
    TradingAgentsNativeIntegration,
    MultiAgentDebateSystem,
    NativeAgentRole
from app.core.advanced_indicators import TechnicalIndicators
from app.core.risk_management import RiskManager

# 页面配置
st.set_page_config(
    page_title="智能交易分析系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #f0f2f6 0%, #e0e5eb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .agent-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    .debate-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 初始化Session State
if 'integration' not in st.session_state:
    st.session_state.integration = TradingAgentsNativeIntegration()
if 'debate_system' not in st.session_state:
    st.session_state.debate_system = MultiAgentDebateSystem(st.session_state.integration)
if 'indicators' not in st.session_state:
    st.session_state.indicators = TechnicalIndicators()
if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = RiskManager()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


def main():
    """主函数"""
    # 标题
    st.markdown('<div class="main-header">🤖 智能交易分析系统</div>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.title("📊 控制面板")
        
        # 股票选择
        st.subheader("股票分析")
        symbol = st.text_input(
            "输入股票代码",
            value="000001",
            help="输入A股股票代码，如000001"
        
        # 分析参数
        st.subheader("分析参数")
        
        analysis_type = st.selectbox(
            "分析类型",
            ["快速分析", "深度分析", "多轮辩论"]
        
        if analysis_type == "多轮辩论":
            debate_rounds = st.slider("辩论轮数", 1, 5, 3)
        else:
            debate_rounds = 1
        
        # 启动分析按钮
        analyze_button = st.button(
            "🚀 启动智能分析",
            type="primary",
            use_container_width=True
        
        # 历史记录
        st.subheader("📜 历史分析")
        if st.session_state.analysis_history:
            for record in st.session_state.analysis_history[-5:]:
                st.text(f"{record['symbol']} - {record['time']}")
    
    # 主要内容区
    if analyze_button:
        run_analysis(symbol, analysis_type, debate_rounds)
    else:
        show_welcome_page()


def show_welcome_page():
    """显示欢迎页面"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>📊 基本面分析师</h3>
            <p>深挖财报数据，计算ROE趋势、现金流健康度，识别隐藏的关联交易风险</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>📈 市场情绪分析师</h3>
            <p>爬取财经新闻、券商研报，实时分析市场情绪，量化看多/看空观点</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="agent-card">
            <h3>💹 技术面分析师</h3>
            <p>自动绘制K线图、计算MACD等12种技术指标，识别支撑位压力位</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="agent-card">
            <h3>🛡️ 风险管控师</h3>
            <p>评估行业政策风险、流动性风险，给出仓位控制建议，模拟黑天鹅事件</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 系统特性
    st.markdown("---")
    st.subheader("🎯 系统特性")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **多轮辩论机制**  
        看涨和看跌的AI进行3-5轮辩论，像真实分析师那样互相驳斥观点
        """)
    
    with col2:
        st.info("""
        **智能协作网络**  
        14个专业智能体协同工作，覆盖基本面、技术面、情绪面全方位分析
        """)
    
    with col3:
        st.info("""
        **实时风险控制**  
        动态评估风险，提供止损止盈建议，智能仓位管理
        """)


def run_analysis(symbol: str, analysis_type: str, debate_rounds: int):
    """运行分析"""
    st.markdown(f"### 正在分析 {symbol}")
    
    # 创建占位符
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 综合分析",
        "💬 智能体辩论",
        "📈 技术指标",
        "🛡️ 风险评估",
        "💡 交易建议"
    ])
    
    try:
        # 运行异步分析
        if analysis_type == "多轮辩论":
            status_text.text("正在进行多轮辩论分析...")
            progress_bar.progress(20)
            
            # 运行辩论
            debate_result = asyncio.run(
                st.session_state.debate_system.conduct_debate(symbol, debate_rounds)
            
            progress_bar.progress(60)
            
            # 获取综合分析
            analysis_result = asyncio.run(
                st.session_state.integration.analyze_stock(symbol)
            
            progress_bar.progress(100)
            
        else:
            status_text.text("正在进行智能体分析...")
            progress_bar.progress(50)
            
            analysis_result = asyncio.run(
                st.session_state.integration.analyze_stock(symbol)
            
            progress_bar.progress(100)
            debate_result = None
        
        # 清除进度条
        progress_bar.empty()
        status_text.empty()
        
        # 保存到历史
        st.session_state.analysis_history.append({
            'symbol': symbol,
            'time': datetime.now().strftime("%H:%M:%S"),
            'result': analysis_result
        })
        
        # 显示结果
        with tab1:
            show_comprehensive_analysis(analysis_result)
        
        with tab2:
            if debate_result:
                show_debate_results(debate_result)
            else:
                st.info("选择'多轮辩论'模式查看辩论过程")
        
        with tab3:
            show_technical_analysis(symbol)
        
        with tab4:
            show_risk_assessment(analysis_result)
        
        with tab5:
            show_trading_recommendation(analysis_result)
        
    except Exception as e:
        st.error(f"分析失败: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def show_comprehensive_analysis(result: dict):
    """显示综合分析结果"""
    st.subheader("📊 智能体分析结果")
    
    # 显示共识
    if result.get("consensus"):
        consensus = result["consensus"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "共识观点",
                consensus.get("type", "未知"),
                f"得分: {consensus.get('score', 0):.2f}"
        
        with col2:
            st.metric(
                "置信度",
                f"{consensus.get('confidence', 0):.1%}",
                "高" if consensus.get('confidence', 0) > 0.7 else "中"
        
        with col3:
            recommendation = result.get("recommendation", {})
            st.metric(
                "建议操作",
                recommendation.get("action", "持有"),
                f"仓位: {recommendation.get('position_size', 0):.0%}"
    
    # 各智能体分析
    st.markdown("---")
    st.subheader("🤖 各智能体分析详情")
    
    agents_analysis = result.get("agents_analysis", {})
    
    for role, analysis in agents_analysis.items():
        with st.expander(f"{role}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 显示分析内容
                if isinstance(analysis, dict):
                    if "analysis" in analysis:
                        st.write(analysis.get("analysis", "无分析结果"))
                    
                    if "metrics" in analysis:
                        st.json(analysis["metrics"])
                    
                    if "signals" in analysis:
                        st.write("**交易信号:**")
                        for signal in analysis.get("signals", []):
                            st.write(f"• {signal}")
                else:
                    st.write(str(analysis))
            
            with col2:
                # 显示得分
                score = analysis.get("score", 0.5) if isinstance(analysis, dict) else 0.5
                
                # 创建仪表盘
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "得分"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.7], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)


def show_debate_results(debate_result: dict):
    """显示辩论结果"""
    st.subheader("💬 智能体辩论过程")
    
    # 显示最终共识
    final_consensus = debate_result.get("final_consensus", {})
    if final_consensus:
        st.markdown(f"""
        <div class="recommendation-box">
            最终建议: {final_consensus.get('recommendation', '未知')} <br>
            得分: {final_consensus.get('final_score', 0):.2f} | 
            置信度: {final_consensus.get('confidence', 0):.1%} | 
            趋势: {final_consensus.get('trend', '未知')}
        </div>
        """, unsafe_allow_html=True)
    
    # 显示辩论轮次
    st.markdown("---")
    for round_data in debate_result.get("rounds", []):
        st.subheader(f"第 {round_data['round']} 轮辩论")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🐂 看多观点:**")
            for arg in round_data.get("bull_arguments", []):
                st.markdown(f"""
                <div class="debate-box" style="border-left-color: #28a745;">
                    {arg}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🐻 看空观点:**")
            for arg in round_data.get("bear_arguments", []):
                st.markdown(f"""
                <div class="debate-box" style="border-left-color: #dc3545;">
                    {arg}
                </div>
                """, unsafe_allow_html=True)
        
        # 显示本轮共识
        consensus = round_data.get("consensus", {})
        if consensus:
            st.info(f"本轮共识: {consensus.get('type', '未知')} (得分: {consensus.get('score', 0):.2f})")


def show_technical_analysis(symbol: str):
    """显示技术分析"""
    st.subheader("📈 技术指标分析")
    
    # 生成模拟数据（实际应该从数据源获取）
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    price_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(100).cumsum() * 2,
        'high': 102 + np.random.randn(100).cumsum() * 2,
        'low': 98 + np.random.randn(100).cumsum() * 2,
        'close': 100 + np.random.randn(100).cumsum() * 2,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    price_data['high'] = price_data[['open', 'close']].max(axis=1) + np.abs(np.random.randn(100))
    price_data['low'] = price_data[['open', 'close']].min(axis=1) - np.abs(np.random.randn(100))
    
    # K线图
    fig = go.Figure(data=[go.Candlestick(
        x=price_data['date'],
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name='K线'
    )])
    
    # 添加均线
    price_data['MA20'] = price_data['close'].rolling(window=20).mean()
    price_data['MA60'] = price_data['close'].rolling(window=60).mean()
    
    fig.add_trace(go.Scatter(
        x=price_data['date'],
        y=price_data['MA20'],
        name='MA20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_data['date'],
        y=price_data['MA60'],
        name='MA60',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=f"{symbol} K线图",
        yaxis_title="价格",
        xaxis_title="日期",
        height=400
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 技术指标
    col1, col2, col3, col4 = st.columns(4)
    
    # 计算指标
    indicators = st.session_state.indicators
    rsi = indicators.rsi(price_data['close'])
    macd_data = indicators.macd(price_data['close'])
    bb_data = indicators.bollinger_bands(price_data['close'])
    
    with col1:
        st.metric("RSI", f"{rsi.iloc[-1]:.2f}")
        st.caption("相对强弱指标")
    
    with col2:
        macd_signal = "金叉" if macd_data['histogram'].iloc[-1] > 0 else "死叉"
        st.metric("MACD", macd_signal)
        st.caption("指数平滑异同移动平均线")
    
    with col3:
        current_price = price_data['close'].iloc[-1]
        bb_position = "上轨" if current_price > bb_data['upper'].iloc[-1] else "下轨"
        st.metric("布林带", bb_position)
        st.caption("布林带位置")
    
    with col4:
        volume_trend = "放量" if price_data['volume'].iloc[-1] > price_data['volume'].mean() else "缩量"
        st.metric("成交量", volume_trend)
        st.caption("成交量趋势")


def show_risk_assessment(result: dict):
    """显示风险评估"""
    st.subheader("🛡️ 风险评估")
    
    # 获取风险分析
    risk_analysis = result.get("agents_analysis", {}).get(NativeAgentRole.RISK, {})
    recommendation = result.get("recommendation", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 风险指标")
        
        # 风险等级
        risk_level = recommendation.get("risk_level", "中")
        color = {"低": "green", "中": "orange", "高": "red"}.get(risk_level, "gray")
        
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            风险等级: {risk_level}
        </div>
        """, unsafe_allow_html=True)
        
        # 风险因子
        st.markdown("**主要风险因子:**")
        for factor in risk_analysis.get("risk_factors", ["暂无数据"]):
            st.write(f"• {factor}")
    
    with col2:
        st.markdown("### 风控建议")
        
        # 止损止盈
        stop_loss = recommendation.get("stop_loss", 0.02)
        take_profit = recommendation.get("take_profit", 0.05)
        
        st.metric("建议止损", f"-{stop_loss:.1%}")
        st.metric("建议止盈", f"+{take_profit:.1%}")
        
        # 仓位建议
        position = recommendation.get("position_size", 0.1)
        st.metric("建议仓位", f"{position:.0%}")


def show_trading_recommendation(result: dict):
    """显示交易建议"""
    st.subheader("💡 交易建议")
    
    recommendation = result.get("recommendation", {})
    consensus = result.get("consensus", {})
    
    # 主要建议
    st.markdown(f"""
    <div class="recommendation-box">
        {recommendation.get('action', '持有')}
    </div>
    """, unsafe_allow_html=True)
    
    # 详细建议
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 操作建议")
        st.write(f"**操作:** {recommendation.get('action', '持有')}")
        st.write(f"**仓位:** {recommendation.get('position_size', 0.1):.0%}")
        st.write(f"**止损:** -{recommendation.get('stop_loss', 0.02):.1%}")
        st.write(f"**止盈:** +{recommendation.get('take_profit', 0.05):.1%}")
    
    with col2:
        st.markdown("### 📈 市场观点")
        st.write(f"**共识:** {consensus.get('type', '中性')}")
        st.write(f"**得分:** {consensus.get('score', 0.5):.2f}")
        st.write(f"**置信度:** {consensus.get('confidence', 0.5):.1%}")
    
    with col3:
        st.markdown("### ⚠️ 风险提示")
        st.write(f"**风险等级:** {recommendation.get('risk_level', '中')}")
        st.warning("投资有风险，决策需谨慎。本系统仅供参考，不构成投资建议。")
    
    # 执行计划
    st.markdown("---")
    st.markdown("### 📝 执行计划")
    
    # 创建执行计划表格
    plan_data = {
        "步骤": ["1. 市场观察", "2. 建仓", "3. 持仓管理", "4. 退出"],
        "条件": [
            "确认市场趋势符合预期",
            f"分批建仓至{recommendation.get('position_size', 0.1):.0%}",
            "动态调整止损位",
            "达到止盈或止损条件"
        ],
        "风控": [
            "观察成交量变化",
            "控制单次买入量",
            "跟踪市场情绪",
            "严格执行止损"
        ]
    }
    
    plan_df = pd.DataFrame(plan_data)
    st.table(plan_df)


if __name__ == "__main__":
    main()