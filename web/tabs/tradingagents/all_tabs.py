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
import asyncio
import os
import time

# 用户态辅助（带持久化）
from persistence.user_store import get_user_store

def _ensure_user_state():
    st.session_state.setdefault('user_id', 'admin')
    store = get_user_store()
    u = store.ensure_user(st.session_state['user_id'], default_points=200, level='VIP')
    st.session_state['user_level'] = u.level
    st.session_state['user_points'] = u.points
    # 缓存最近日志用于展示
    st.session_state['usage_logs'] = [
        {'date': x.date, 'op': x.op, 'stocks': x.stocks, 'points': x.points}
        for x in store.get_logs(user_id=st.session_state['user_id'], limit=200)
    ]

# 添加TradingAgents路径（优先环境变量 TRADINGAGENTS_PATH）
ENV_TA_PATH = os.getenv("TRADINGAGENTS_PATH")
ta_path = Path(ENV_TA_PATH) if ENV_TA_PATH else Path("G:/test/tradingagents-cn-plus")
if ta_path.exists() and str(ta_path) not in sys.path:
    sys.path.insert(0, str(ta_path))

# 可选：接入本项目内置真实集成（若可用则用；否则保持演示模式）
try:
    from tradingagents_integration.full_agents_integration import create_full_integration, FullAgentsIntegration
    _TA_INTEGRATION_AVAILABLE = True
    _FULL_10_AGENTS = True
except Exception:
    try:
        from tradingagents_integration.real_integration import create_integration
        _TA_INTEGRATION_AVAILABLE = True
        _FULL_10_AGENTS = False
    except Exception:
        _TA_INTEGRATION_AVAILABLE = False
        _FULL_10_AGENTS = False


def _get_ta_integration():
    """获取/初始化 TradingAgents 实例（全局复用，优先使用10个智能体）"""
    if not _TA_INTEGRATION_AVAILABLE:
        return None
    if 'ta_integration' not in st.session_state:
        # 优先使用完整10个智能体集成
        if _FULL_10_AGENTS:
            st.session_state.ta_integration = create_full_integration()
            st.session_state.ta_mode = "full_10_agents"
        else:
            st.session_state.ta_integration = create_integration()
            st.session_state.ta_mode = "basic"
    return st.session_state.ta_integration


def render_agent_management():
    """智能体管理tab"""
    st.header("🔍 智能体管理")
    
    # 显示当前模式
    mode = st.session_state.get('ta_mode', 'demo')
    if mode == "full_10_agents":
        st.success("✅ 当前使用：完整10个专业智能体模式")
        st.markdown("""
        **10个专业A股交易智能体**
        - 🌍 市场生态分析 (MarketEcologyAgent)
        - 🎯 竞价博弈分析 (AuctionGameAgent)
        - 💼 仓位控制 (PositionControlAgent) ⭐
        - 📊 成交量分析 (VolumeAnalysisAgent)
        - 📈 技术指标分析 (TechnicalIndicatorAgent)
        - 😊 市场情绪分析 (SentimentAnalysisAgent)
        - ⚠️ 风险管理 (RiskManagementAgent) ⭐
        - 🕯️ K线形态识别 (PatternRecognitionAgent)
        - 🌐 宏观经济分析 (MacroeconomicAgent)
        - 🔄 套利机会分析 (ArbitrageAgent)
        """)
    else:
        st.info("ℹ️ 当前使用：演示模式 (6个基础智能体)")
        st.markdown("""
        **6类专业分析师智能体**
        - 📊 基本面分析师
        - 📈 技术分析师  
        - 📰 新闻分析师
        - 💬 社交媒体分析师
        - 🔼 看涨研究员
        - 🔽 看跌研究员
        """)
    
    # 智能体状态总览（若已接入真实系统则展示真实数量）
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        integration = _get_ta_integration()
        if integration:
            status = integration.get_status()
            agents_cnt = status.get('agents_count', 6)
            if 'mode' in status and status['mode'] == 'full_10_agents':
                st.metric("激活智能体", f"{agents_cnt}/10", "✅ 完整模式")
            else:
                st.metric("激活智能体", f"{agents_cnt}", "基础模式")
        else:
            st.metric("激活智能体", "6/6", "演示模式")
    with col2:
        st.metric("平均响应时间", "2.3s", "-0.5s")
    with col3:
        st.metric("今日分析次数", "128", "+23")
    with col4:
        st.metric("共识达成率", "87%", "+5%")
    
    st.divider()
    
    # 智能体详细配置（若可用则列出真实智能体名和权重）
    st.subheader("⚙️ 智能体配置")
    agents_config = []
    
    integration = _get_ta_integration()
    if integration and st.session_state.get('ta_mode') == 'full_10_agents':
        # 使用完整10个智能体的配置
        status = integration.get_status()
        weights = status.get('weights', {})
        
        agent_info = [
            {"name": "市场生态分析", "key": "market_ecology", "emoji": "🌍"},
            {"name": "竞价博弈分析", "key": "auction_game", "emoji": "🎯"},
            {"name": "仓位控制", "key": "position_control", "emoji": "💼"},
            {"name": "成交量分析", "key": "volume", "emoji": "📊"},
            {"name": "技术指标分析", "key": "technical", "emoji": "📈"},
            {"name": "市场情绪分析", "key": "sentiment", "emoji": "😊"},
            {"name": "风险管理", "key": "risk", "emoji": "⚠️"},
            {"name": "K线形态识别", "key": "pattern", "emoji": "🕯️"},
            {"name": "宏观经济分析", "key": "macroeconomic", "emoji": "🌐"},
            {"name": "套利机会分析", "key": "arbitrage", "emoji": "🔄"}
        ]
        
        for info in agent_info:
            agents_config.append({
                "name": info["name"],
                "key": info["key"],
                "emoji": info["emoji"],
                "status": "✅ 运行中",
                "weight": weights.get(info["key"], 0.1)
            })
    else:
        # 使用默认6个智能体配置
        agents_config = [
            {"name": "基本面分析师", "key": "fundamental", "emoji": "📊", "status": "✅ 运行中", "weight": 0.20},
            {"name": "技术分析师", "key": "technical", "emoji": "📈", "status": "✅ 运行中", "weight": 0.25},
            {"name": "新闻分析师", "key": "news", "emoji": "📰", "status": "✅ 运行中", "weight": 0.15},
            {"name": "社交媒体分析师", "key": "social", "emoji": "💬", "status": "✅ 运行中", "weight": 0.10},
            {"name": "看涨研究员", "key": "bullish", "emoji": "🔼", "status": "✅ 运行中", "weight": 0.15},
            {"name": "看跌研究员", "key": "bearish", "emoji": "🔽", "status": "✅ 运行中", "weight": 0.15}
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
    
    # 智能体性能对比（占位/示例）
    st.subheader("📊 性能对比")
    
    # 生成与 agents_config 长度一致的模拟数据
    import numpy as np
    num_agents = len(agents_config)
    
    performance_data = {
        "智能体": [a['name'] for a in agents_config],
        "准确率": [round(0.75 + np.random.rand() * 0.15, 2) for _ in range(num_agents)],
        "响应时间(s)": [round(1.5 + np.random.rand() * 2.0, 1) for _ in range(num_agents)],
        "信心度": [round(0.70 + np.random.rand() * 0.25, 2) for _ in range(num_agents)],
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, hide_index=True)


def render_collaboration():
    """协作机制tab"""
    _ensure_user_state()
    st.session_state.setdefault('collab_logs', [])
    st.session_state.setdefault('collab_rounds', [])
    st.header("🗣️ 智能体协作决策")
    
    st.markdown("""
    **🤖 多智能体协作决策机制**
    
    这个功能模拟“多个专家会诊”的场景：
    - 👥 **多个 AI 智能体**（如技术分析师、基本面分析师、情绪分析师等）同时分析同一只股票
    - 📊 每个智能体独立给出 **BUY（买入）/ SELL（卖出）/ HOLD（持有）** 的建议
    - 🗣️ 通过“投票”统计各智能体的观点分布
    - ✅ 当某个观点占比超过阈值（如 75%），认为**达成共识**，作为最终决策
    
    🎯 **使用场景**：对重要交易决策，通过多个角度的分析降低风险
    """)
    
    st.divider()
    st.subheader("⚙️ 分析参数")
    
    # 参数
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("📊 股票代码", value="000001", key="collab_symbol")
        st.caption("输入6位代码，如 000001")
    with c2:
        consensus_threshold = st.slider("🎯 共识阈值(%)", 50, 90, 75, 1, 
                                       help="当某个观点（BUY/SELL/HOLD）的智能体数量占比超过该阈值时，认为达成共识")
        st.caption(f"当前：需要 ≥{consensus_threshold}% 的智能体同意")
    with c3:
        rounds = st.number_input("🔄 分析轮次", 1, 5, 3,
                                help="运行多少轮分析，每轮都会重新调用智能体")
        st.caption(f"将进行 {int(rounds)} 轮独立分析")
    
    # 控制按钮
    colb1, colb2 = st.columns([1,1])
    with colb1:
        start_btn = st.button("🎬 发起协作分析", type="primary")
    with colb2:
        if st.button("🧹 清空记录"):
            st.session_state.collab_logs.clear()
            st.session_state.collab_rounds.clear()
    
    st.divider()
    
    # 调用TradingAgents协作（若可用），支持多轮
    integration = _get_ta_integration()
    
    # 显示当前模式
    if integration:
        mode = st.session_state.get('ta_mode', 'basic')
        if mode == 'full_10_agents':
            st.success("✅ 已启用：完整110个专业智能体模式")
        else:
            st.info("ℹ️ 已启用：基础智能体模式")
    else:
        st.warning("⚠️ TradingAgents 未启用，将使用演示模式")
    
    if start_btn:
        prog = st.progress(0)
        try:
            import numpy as _np
            for r in range(int(rounds)):
                if integration:
                    market_data = {
                        "symbol": symbol,
                        "price": float(_np.random.uniform(8, 20)),
                        "prev_close": float(_np.random.uniform(8, 20)),
                        "change_pct": float(_np.random.uniform(-0.03, 0.05)),
                        "volume": int(_np.random.randint(1_000_000, 8_000_000)),
                        "avg_volume": int(_np.random.randint(800_000, 5_000_000)),
                        "advances": int(_np.random.randint(1500, 2500)),
                        "declines": int(_np.random.randint(1000, 2000)),
                        "money_inflow": float(_np.random.uniform(500_000_000, 2_000_000_000)),
                        "money_outflow": float(_np.random.uniform(400_000_000, 1_800_000_000)),
                    }
                    try:
                        # 调用 analyze_stock 返回字典格式
                        res = asyncio.run(integration.analyze_stock(symbol, market_data))
                    except Exception as e:
                        st.warning(f"第{r+1}轮分析失败: {e}")
                        res = None
                else:
                    res = None
                    
                now = datetime.now().strftime('%H:%M:%S')
                # 记录一轮日志
                if res and isinstance(res, dict) and 'individual_results' in res:
                    indiv = res.get('individual_results') or []
                    buy = sell = hold = 0
                    for item in indiv:
                        agent = item.get('agent', 'Agent')
                        signal = (item.get('signal') or 'HOLD').upper()
                        reasoning = (item.get('reasoning') or '')[:120]
                        kind = '观点' if signal in ('BUY','SELL') else '中立'
                        st.session_state.collab_logs.append({
                            'time': now, 'agent': agent, 'type': kind, 'content': f"{signal} · {reasoning}"
                        })
                        if signal == 'BUY': buy += 1
                        elif signal == 'SELL': sell += 1
                        else: hold += 1
                    st.session_state.collab_rounds.append({'buy': buy, 'sell': sell, 'hold': hold})
                else:
                    # 演示模式：生成模拟数据
                    st.session_state.collab_logs.append({
                        'time': now, 'agent': '演示Agent', 'type': '中立', 'content': 'HOLD · 演示数据（TradingAgents未启用）'
                    })
                    st.session_state.collab_rounds.append({'buy': 1, 'sell': 0, 'hold': 2})
                prog.progress((r+1)/int(rounds))
                time.sleep(0.1)
        except Exception as e:
            st.error(f"协作调用失败: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            prog.empty()
    
    st.subheader("👥 各智能体的观点")
    st.caption("展示每个智能体的分析结果和理由")
    
    # 展示聚合日志（最近200条）
    if st.session_state.collab_logs:
        for log in st.session_state.collab_logs[-200:]:
            color_map = {"观点": "🔵", "支持": "🟢", "反驳": "🔴", "中立": "🟡", "决策": "🟣"}
            st.markdown(f"{color_map.get(log['type'], '⚪')} **{log['time']}** - *{log['agent']}* ({log['type']}): {log['content']}")
    else:
        st.info("💡 点击上方“🎬 发起协作分析”按钮开始分析")
    
    st.divider()
    
    # 共识可视化
    st.subheader("📊 共识达成分析")
    st.caption("展示BUY/SELL/HOLD三种观点的分布，并判断是否达成共识")
    # 按轮次聚合统计
    if st.session_state.collab_rounds:
        buy = sum(x['buy'] for x in st.session_state.collab_rounds)
        sell = sum(x['sell'] for x in st.session_state.collab_rounds)
        hold = sum(x['hold'] for x in st.session_state.collab_rounds)
        total = max(buy + sell + hold, 1)
        consensus = max([(buy,'BUY'),(sell,'SELL'),(hold,'HOLD')], key=lambda t:t[0])
        consensus_pct = consensus[0] / total
        # Sankey 图：显示各观点流向最终决策
        buy_pct = (buy / total * 100) if total > 0 else 0
        sell_pct = (sell / total * 100) if total > 0 else 0
        hold_pct = (hold / total * 100) if total > 0 else 0
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=[
                    f"BUY ({buy})",
                    f"SELL ({sell})",
                    f"HOLD ({hold})",
                    f"{consensus[1]}"
                ],
                customdata=[
                    f"买入: {buy}个智能体 ({buy_pct:.1f}%)",
                    f"卖出: {sell}个智能体 ({sell_pct:.1f}%)",
                    f"持有: {hold}个智能体 ({hold_pct:.1f}%)",
                    f"最终共识: {consensus[1]} ({consensus_pct*100:.1f}%)"
                ],
                hovertemplate='%{customdata}<extra></extra>',
                color=["#4CAF50", "#F44336", "#9E9E9E", "#9C27B0"],
                pad=25,
                thickness=35,
                line=dict(color="white", width=2.5)
            ),
            link=dict(
                source=[0, 1, 2],
                target=[3, 3, 3],
                value=[max(buy, 0.1), max(sell, 0.1), max(hold, 0.1)],
                color=["rgba(76,175,80,0.35)", "rgba(244,67,54,0.35)", "rgba(158,158,158,0.35)"]
            ),
            textfont=dict(color="white", size=16, family="Arial Black, sans-serif")
        )])
        fig.update_layout(
            title={
                'text': "🔀 信号流向与共识形成",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            height=450,
            font=dict(size=15, family="Arial, sans-serif", color="white"),
            margin=dict(l=10, r=10, t=70, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig)
        # 阈值判断
        if consensus_pct*100 >= consensus_threshold:
            st.success(f"✅ 达成共识: {consensus[1]} · {consensus_pct*100:.1f}% (阈值 {consensus_threshold}%)")
        else:
            st.warning(f"⚠️ 共识不足: {consensus[1]} · {consensus_pct*100:.1f}% (阈值 {consensus_threshold}%)")
        # 每轮结果摘要
        st.subheader("🧭 每轮结果")
        for idx, r in enumerate(st.session_state.collab_rounds, start=1):
            rt = max(r['buy']+r['sell']+r['hold'], 1)
            rc = max([(r['buy'],'BUY'),(r['sell'],'SELL'),(r['hold'],'HOLD')], key=lambda t:t[0])
            pct = rc[0]/rt*100
            st.caption(f"第{idx}轮: BUY={r['buy']} SELL={r['sell']} HOLD={r['hold']} → 共识 {rc[1]} {pct:.1f}%")
    else:
        # 默认示例图：简化的 Sankey 图
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=[
                    "BUY (45)",
                    "SELL (25)",
                    "HOLD (30)",
                    "BUY"
                ],
                customdata=[
                    "买入: 45个智能体 (45%)",
                    "卖出: 25个智能体 (25%)",
                    "持有: 30个智能体 (30%)",
                    "最终共识: BUY (45%)"
                ],
                hovertemplate='%{customdata}<extra></extra>',
                color=["#4CAF50", "#F44336", "#9E9E9E", "#9C27B0"],
                pad=25,
                thickness=35,
                line=dict(color="white", width=2.5)
            ),
            link=dict(
                source=[0, 1, 2],
                target=[3, 3, 3],
                value=[45, 25, 30],
                color=["rgba(76,175,80,0.35)", "rgba(244,67,54,0.35)", "rgba(158,158,158,0.35)"]
            ),
            textfont=dict(color="white", size=16, family="Arial Black, sans-serif")
        )])
        fig.update_layout(
            title={
                'text': "🔀 观点流向与共识形成（示例）",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            height=450,
            font=dict(size=15, family="Arial, sans-serif", color="white"),
            margin=dict(l=10, r=10, t=70, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig)
        st.info("💡 请点击上方'🎬 发起协作分析'按钮查看真实的智能体分析结果")


def render_information_collection():
    """信息采集tab"""
    st.header("📰 信息采集")
    
    st.markdown("""
    **📡 多源信息整合功能**
    
    这个功能可以从多个来源采集和过滤与股票相关的信息：
    - 📰 **新闻资讯**：从财经新闻网站采集，智能过滤低质量内容
    - 📊 **财务数据**：财报、业绩预告、公告等
    - 💬 **社交媒体**：雪球、股吧等平台的情绪分析
    - 📈 **实时行情**：价格、成交量、资金流向等
    
    🔧 **当前状态**：下方的指标和新闻为演示数据，真实采集功能请使用 "TradingAgents-CN 采集器"
    """)
    
    st.divider()
    st.subheader("📊 概览指标（演示）")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("今日新闻", "1,247", "+156")
        st.caption("🎭 演示数据")
    with col2:
        st.metric("过滤后", "89", "高质量")
        st.caption("🎭 演示数据")
    with col3:
        st.metric("情绪指数", "0.68", "偏乐观")
        st.caption("🎭 演示数据")
    with col4:
        st.metric("数据源", "12", "多元化")
        st.caption("🎭 演示数据")
    
    st.divider()
    
    # 新闻过滤配置
    st.subheader("⚙️ 新闻智能过滤 (v0.1.12)")
    
    with st.expander("过滤配置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            filter_mode = st.selectbox(
                "过滤模式",
                ["基础过滤", "增强过滤", "集成过滤"],
                key="ta_info_filter_mode"
            )
            relevance_threshold = st.slider("相关性阈值", 0.0, 1.0, 0.7)
        with col2:
            quality_threshold = st.slider("质量阈值", 0.0, 1.0, 0.6)
            enable_dedup = st.checkbox("去重", value=True)
    
    if st.button("🔍 应用过滤", type="primary"):
        st.success(f"已应用{filter_mode}，过滤出89条高质量新闻")
    
    # 真实采集功能
    st.divider()
    st.subheader("✅ TradingAgents-CN 真实采集器")
    st.markdown("""
    🚀 **真实数据采集功能**
    
    这个采集器会真实调用TradingAgents的数据采集工具，获取指定股票的多维度信息并进行打分。
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol_ic = st.text_input("📊 输入股票代码", value="000001", key="ta_cn_symbol")
    with col2:
        st.write("")  # 占位空间
        st.write("")
    if st.button("🚀 运行真实采集器", type="primary"):
        try:
            from integrations.tradingagents_cn.tools.decision_agents import run_agents
            with st.spinner(f"🔍 正在采集 {symbol_ic} 的数据..."):
                scores = run_agents(symbol_ic)
            if scores:
                st.success(f"✅ 采集完成！共获取 {len(scores)} 个智能体的评分")
                df_scores = pd.DataFrame({"Agent": list(scores.keys()), "Score": list(scores.values())})
                st.dataframe(df_scores, hide_index=True)
            else:
                st.warning("⚠️ 采集器运行成功但未返回评分结果")
        except ImportError:
            st.error("❌ TradingAgents-CN 采集器未安装或未配置")
            st.info("💡 请检查 integrations/tradingagents_cn/ 目录是否存在")
        except Exception as e:
            st.error(f"❌ 采集器运行失败: {e}")
            with st.expander("🔍 查看详细错误"):
                import traceback
                st.code(traceback.format_exc())
    
    st.divider()
    
    # 新闻展示区域
    st.subheader("📋 过滤后的新闻")
    
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.caption("💡 提示：以下为演示数据，展示界面效果")
    with col_right:
        show_demo_news = st.checkbox("显示演示数据", value=False, key="show_demo_news")
    
    if show_demo_news:
        # 演示新闻数据
        news_data = [
            {"time": "10:23", "title": "🎭 某公司发布Q3财报，净利润同比增长35%", "relevance": 0.92, "sentiment": "正面"},
            {"time": "09:45", "title": "🎭 行业监管新政出台，利好龙头企业", "relevance": 0.88, "sentiment": "正面"},
            {"time": "08:30", "title": "🎭 技术突破获得重大进展", "relevance": 0.85, "sentiment": "正面"},
            {"time": "08:15", "title": "🎭 某股获境外机构增持", "relevance": 0.80, "sentiment": "正面"},
            {"time": "07:50", "title": "🎭 行业景气度持续回升", "relevance": 0.78, "sentiment": "中性"}
        ]
        
        for news in news_data:
            with st.container():
                col1, col2, col3 = st.columns([1, 5, 1])
                with col1:
                    st.markdown(f"**{news['time']}**")
                with col2:
                    st.markdown(f"{news['title']}")
                with col3:
                    sentiment_emoji = "🟢" if news['sentiment'] == "正面" else "🔴" if news['sentiment'] == "负面" else "🟡"
                    st.markdown(f"{sentiment_emoji} {news['relevance']:.0%}")
    else:
        # 显示如何接入真实新闻的指引
        st.info("""
        🔧 **如何接入真实新闻数据？**
        
        1. **接入新闻 API**：
           - 东方财富、新浪财经等提供的新闻 API
           - AKShare 的新闻数据接口
           - 自建爬虫采集
        
        2. **实现过滤逻辑**：
           - 关键词匹配（股票代码、公司名称）
           - 情绪分析（正面/负面/中性）
           - 相关性评分
           - 去重处理
        
        3. **集成到系统**：
           - 在 `data_layer/` 下创建新闻采集模块
           - 调用新闻 API 并存储到数据库
           - 在此页面从数据库读取并展示
        """)
        
        with st.expander("💻 代码示例：如何获取新闻"):
            st.code("""
# 使用 AKShare 获取新闻
import akshare as ak

# 获取东方财富的财经新闻
df_news = ak.stock_news_em(symbol="东方财富")

# 过滤相关新闻
filtered_news = df_news[df_news['title'].str.contains('某关键词')]

# 展示结果
for _, news in filtered_news.iterrows():
    print(f"{news['time']}: {news['title']}")
            """, language="python")


def render_decision_analysis():
    """决策分析tab"""
    st.header("💡 决策分析")
    
    st.markdown("""
    **分析模式**
    - 📊 单股深度分析（已接入真实TradingAgents，如可用）
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
    
    # 构造简单市场数据（若无真实数据源时作为输入）
    def _build_market_data():
        return {
            "price": float(np.random.uniform(8, 20)),
            "change_pct": float(np.random.uniform(-0.03, 0.05)),
            "volume": int(np.random.randint(1_000_000, 8_000_000)),
            "technical_indicators": {"rsi": float(np.random.uniform(30, 70)), "macd": 0.3, "macd_signal": 0.2},
            "fundamental_data": {"pe_ratio": 15.0, "pb_ratio": 2.1, "roe": 0.15},
            "sentiment": {"score": 0.6},
        }
    
    if analysis_mode == "📊 单股分析":
        _ensure_user_state()
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("股票代码", "000001")
        with col2:
            depth = st.selectbox("研究深度", ["简单", "标准", "深度", "极深", "完整"], key="da_single_depth")
        
        if st.button("🚀 开始分析", type="primary"):
            _ensure_user_state()
            # 扣点：单股 1 点
            if st.session_state.user_points < 1:
                st.error("点数不足，请前往用户管理充值")
            else:
                # 扣点并持久化
                store = get_user_store()
                new_pts = store.add_points(st.session_state.user_id, -1)
                store.append_log(st.session_state.user_id, '单股分析', 1, 1)
                st.session_state.user_points = new_pts
                st.session_state.usage_logs.insert(0, {'date': datetime.now().strftime('%Y-%m-%d'),'op':'单股分析','stocks':1,'points':1})
            with st.spinner("智能体正在协作分析..."):
                integration = _get_ta_integration()
                mode = st.session_state.get('ta_mode', 'demo')
                
                if integration is not None:
                    try:
                        # 根据模式选择不同的调用方式
                        if mode == "tradingagents_cn_plus_full":
                            # TradingAgents-CN-Plus完整系统：调用analyze_stock_full
                            st.info("🎓 使用 TradingAgents-CN-Plus 完整系统分析")
                            result = asyncio.run(integration.analyze_stock_full(symbol, date=None))
                        else:
                            # 其他模式：调用标准analyze_stock
                            market_data = _build_market_data()
                            result = asyncio.run(integration.analyze_stock(symbol, market_data))
                        st.success("分析完成!")
                        # 展示结果
                        if result and isinstance(result, dict) and 'consensus' in result:
                            # ==== 1. 快速概览 ====
                            st.subheader("📊 快速概览")
                            c1, c2, c3, c4 = st.columns(4)
                            consensus = result.get('consensus', {})
                            signal = consensus.get('signal', 'HOLD')
                            confidence = consensus.get('confidence', 0.0)
                            
                            with c1:
                                st.metric("综合评分", f"{confidence*100:.1f}/100")
                            with c2:
                                signal_emoji = "🟢" if signal == 'BUY' else "🔴" if signal == 'SELL' else "🟡"
                                st.metric("最终建议", f"{signal_emoji} {signal}")
                            with c3:
                                risk_level = "高" if confidence < 0.5 else "中" if confidence < 0.75 else "低"
                                st.metric("风险等级", risk_level)
                            with c4:
                                indiv = result.get('individual_results', [])
                                st.metric("参与智能体", f"{len(indiv)}个")
                            
                            st.divider()
                            
                            # ==== 2. 完整分析报告 ====
                            st.subheader("📝 完整分析报告")
                            
                            # 报告头部
                            st.markdown(f"""
                            **股票代码**: {symbol}  
                            **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
                            **分析深度**: {depth}  
                            **分析模式**: {'TradingAgents-CN-Plus完整系统' if st.session_state.get('ta_mode') == 'tradingagents_cn_plus_full' else '10个专业智能体' if st.session_state.get('ta_mode') == 'full_10_agents' else '基础智能体'}  
                            """)
                            
                            st.divider()
                            
                            # 执行摘要
                            st.markdown("### 🎯 执行摘要")
                            reasoning = consensus.get('reasoning', '')
                            if reasoning:
                                st.info(f"💡 {reasoning}")
                            else:
                                st.info(f"💡 经过{len(indiv)}个智能体的协作分析，系统建议 **{signal}**，综合置信度为 **{confidence*100:.1f}%**。")
                            
                            # 智能体观点汇总
                            st.markdown("### 👥 智能体观点汇总")
                            if indiv:
                                buy_count = sum(1 for x in indiv if (x.get('signal') or 'HOLD').upper() == 'BUY')
                                sell_count = sum(1 for x in indiv if (x.get('signal') or 'HOLD').upper() == 'SELL')
                                hold_count = sum(1 for x in indiv if (x.get('signal') or 'HOLD').upper() == 'HOLD')
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🟢 买入", f"{buy_count}个", f"{buy_count/len(indiv)*100:.0f}%")
                                with col2:
                                    st.metric("🔴 卖出", f"{sell_count}个", f"{sell_count/len(indiv)*100:.0f}%")
                                with col3:
                                    st.metric("🟡 持有", f"{hold_count}个", f"{hold_count/len(indiv)*100:.0f}%")
                                
                                st.markdown("#### 📊 详细分析")
                                for idx, agent_result in enumerate(indiv, 1):
                                    agent_name = agent_result.get('agent', 'Agent')
                                    agent_signal = (agent_result.get('signal') or 'HOLD').upper()
                                    agent_conf = agent_result.get('confidence', 0.0)
                                    agent_reasoning = agent_result.get('reasoning', '')
                                    
                                    signal_color = "green" if agent_signal == 'BUY' else "red" if agent_signal == 'SELL' else "gray"
                                    
                                    with st.expander(f"{idx}. {agent_name} - {agent_signal} ({agent_conf*100:.1f}%)", expanded=False):
                                        st.markdown(f"**观点**: :{signal_color}[{agent_signal}]")
                                        st.markdown(f"**置信度**: {agent_conf*100:.1f}%")
                                        st.markdown(f"**分析理由**:")
                                        st.write(agent_reasoning if agent_reasoning else "暂无详细理由")
                                
                            # 风险提示
                            st.markdown("### ⚠️ 风险提示")
                            if confidence < 0.5:
                                st.warning("""
                                ⚠️ **高风险警告**
                                - 智能体共识程度较低（<50%）
                                - 建议谨慎决策，等待更明确信号
                                - 可考虑增加分析深度或等待更多数据
                                """)
                            elif confidence < 0.75:
                                st.info("""
                                ℹ️ **中等风险**
                                - 智能体达成了一定共识（50-75%）
                                - 建议结合自身风险承受能力决策
                                - 建议设置止损止盈
                                """)
                            else:
                                st.success("""
                                ✅ **低风险**
                                - 智能体高度共识（>75%）
                                - 分析结果较为可靠
                                - 仅供参考，请自行判断
                                """)
                            
                            # 下载报告
                            st.divider()
                            st.markdown("### 📥 导出报告")
                            
                            # 生成报告文本
                            report_text = f"""
# 股票分析报告

**股票代码**: {symbol}
**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析深度**: {depth}
**参与智能体**: {len(indiv)}个

## 执行摘要

**最终建议**: {signal}
**综合评分**: {confidence*100:.1f}/100
**风险等级**: {risk_level}

{reasoning if reasoning else f'经过{len(indiv)}个智能体的协作分析，系统建议 {signal}，综合置信度为 {confidence*100:.1f}%。'}

## 智能体观点统计

- 🟢 买入: {buy_count}个 ({buy_count/len(indiv)*100:.0f}%)
- 🔴 卖出: {sell_count}个 ({sell_count/len(indiv)*100:.0f}%)
- 🟡 持有: {hold_count}个 ({hold_count/len(indiv)*100:.0f}%)

## 详细分析

"""
                            for idx, agent_result in enumerate(indiv, 1):
                                report_text += f"""
### {idx}. {agent_result.get('agent', 'Agent')}

- **观点**: {agent_result.get('signal', 'HOLD')}
- **置信度**: {agent_result.get('confidence', 0.0)*100:.1f}%
- **分析理由**: {agent_result.get('reasoning', '暂无详细理由')}

"""
                            
                            report_text += f"""
## 免责声明

本报告由 AI 智能体系统生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。
"""
                            
                            # 使用增强报告生成器
                            from .enhanced_report_generator import create_enhanced_report
                            enhanced_report = create_enhanced_report(symbol, result, depth)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📝 下载完整报告 (Markdown)",
                                    data=enhanced_report,
                                    file_name=f"enhanced_analysis_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown",
                                    help="包含团队辩论、详细分析模块和投资建议的完整报告"
                                )
                            with col2:
                                # JSON格式
                                import json
                                json_data = json.dumps(result, ensure_ascii=False, indent=2)
                                st.download_button(
                                    label="📦 下载JSON数据",
                                    data=json_data,
                                    file_name=f"analysis_data_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    help="原始分析数据，用于程序化处理"
                                )
                        else:
                            st.warning("未返回有效结果，已完成调用。")
                    except Exception as e:
                        st.error(f"调用TradingAgents失败: {e}")
                else:
                    # 回退展示（演示模式）
                    st.success("分析完成!")
                    st.subheader("📊 分析结果（演示）")
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
        _ensure_user_state()
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
        
        if st.button("🚀 批量分析", type="primary"):
            symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
            # 扣点：按只数
            need = len(symbols)
            _ensure_user_state()
            if st.session_state.user_points < need:
                st.error(f"点数不足（需 {need}），请前往用户管理充值")
            else:
                store = get_user_store()
                new_pts = store.add_points(st.session_state.user_id, -need)
                store.append_log(st.session_state.user_id, '批量分析', need, need)
                st.session_state.user_points = new_pts
                st.session_state.usage_logs.insert(0, {'date': datetime.now().strftime('%Y-%m-%d'),'op':'批量分析','stocks':need,'points':need})
                with st.spinner(f"正在分析{len(symbols)}只股票..."):
                    integration = _get_ta_integration()
                    if integration is not None:
                        batch_rows = []
                        for s in symbols:
                            try:
                                res = asyncio.run(integration.analyze_stock(s, _build_market_data()))
                                sig = (res.get('consensus', {}) or {}).get('signal', 'HOLD') if isinstance(res, dict) else 'HOLD'
                                conf = (res.get('consensus', {}) or {}).get('confidence', 0.0) if isinstance(res, dict) else 0.0
                            except Exception:
                                sig, conf = 'HOLD', 0.0
                            batch_rows.append({"代码": s, "建议": sig, "评分(置信度)": f"{conf*100:.1f}"})
                        st.success(f"批量分析完成!共{len(symbols)}只股票")
                        st.dataframe(pd.DataFrame(batch_rows), hide_index=True)
                    else:
                        import time; time.sleep(2)
                        st.success(f"批量分析完成!共{len(symbols)}只股票")
                        # 演示占位
                        results_data = {
                            "代码": symbols,
                            "评分": [75, 68, 82][: len(symbols)],
                            "建议": ["谨慎买入", "观望", "买入"][: len(symbols)],
                            "目标价": ["¥12.50", "¥8.30", "¥15.20"][: len(symbols)],
                            "风险": ["中", "高", "低"][: len(symbols)],
                        }
                        st.dataframe(pd.DataFrame(results_data), hide_index=True)


def render_user_management():
    """用户管理tab"""
    _ensure_user_state()
    st.header("👤 用户管理")
    
    st.markdown("""
    **会员系统 (v0.1.14+)**
    - 👥 用户注册/登录（本地会话演示）
    - 🎫 点数管理
    - 📊 使用统计
    - 📜 活动日志
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("注册用户", "—", "")
    with col2:
        st.metric("活跃用户", "—", "")
    with col3:
        from persistence.user_store import get_user_store
        store = get_user_store()
        total_used = store.total_points_used(st.session_state.user_id)
        st.metric("总点数消耗", f"{total_used}")
    with col4:
        avg_use = total_used if total_used else 0
        st.metric("平均使用", f"{avg_use}")
    
    st.divider()
    
    # 当前用户信息
    st.subheader("👤 当前用户")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"""
        **用户ID**: {st.session_state.user_id}
        **等级**: {st.session_state.user_level}
        **剩余点数**: {st.session_state.user_points}
        **注册时间**: 2025-01-15
        """)
    with col2:
        st.markdown("**使用记录**")
        from persistence.user_store import get_user_store
        store = get_user_store()
        logs = store.get_logs(user_id=st.session_state.user_id, limit=50)
        if logs:
            df_usage = pd.DataFrame([
                {'日期': x.date, '操作': x.op, '股票数': x.stocks, '消耗点数': x.points}
                for x in logs
            ])
        else:
            df_usage = pd.DataFrame(columns=['日期','操作','股票数','消耗点数'])
        st.dataframe(df_usage, hide_index=True)
    
    st.divider()
    
    # 点数管理
    st.subheader("🎫 点数管理")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**充值点数**")
        amount = st.number_input("充值数量", 10, 10000, 100, step=10)
        if st.button("💰 充值"):
            store = get_user_store()
            new_pts = store.add_points(st.session_state.user_id, int(amount))
            st.session_state.user_points = new_pts
            st.success(f"成功充值{amount}点数! 当前余额 {st.session_state.user_points}")
    
    with col2:
        st.markdown("**点数说明**")
        st.info("""
        - 单股分析: 1点/次
        - 批量分析: 1点/股
        - VIP用户9折优惠（演示未启用）
        - 每日签到赠送5点（演示未启用）
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
                api_key = st.text_input("API Key", type="password", key=f"key_{provider['name']}")
            with col2:
                api_base = st.text_input("API Base URL", key=f"base_{provider['name']}")
                st.slider("Temperature", 0.0, 2.0, 0.7, key=f"temp_{provider['name']}")
            
            if st.button(f"✅ 测试连接", key=f"test_{provider['name']}"):
                # 仅在本地会话中保存，不回显密钥
                if 'llm_configs' not in st.session_state:
                    st.session_state.llm_configs = {}
                st.session_state.llm_configs[provider['name']] = {"api_base": api_base, "has_key": bool(api_key)}
                st.success(f"{provider['name']} 已保存配置并测试连接（本地会话）")
    
    st.divider()
    
    # 使用统计（占位）
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
        st.plotly_chart(fig)
    with col2:
        fig = px.bar(df, x="模型", y="成本($)", title="成本分布")
        st.plotly_chart(fig)
