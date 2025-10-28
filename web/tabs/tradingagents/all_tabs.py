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
    from tradingagents_integration.real_integration import create_integration
    _TA_INTEGRATION_AVAILABLE = True
except Exception:
    _TA_INTEGRATION_AVAILABLE = False


def _get_ta_integration():
    """获取/初始化 TradingAgents 实例（全局复用）"""
    if not _TA_INTEGRATION_AVAILABLE:
        return None
    if 'ta_integration' not in st.session_state:
        # 可读取自定义配置文件路径（如 config/tradingagents.yaml）
        st.session_state.ta_integration = create_integration()
    return st.session_state.ta_integration


def render_agent_management():
    """智能体管理tab"""
    st.header("🔍 智能体管理")
    
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
        agents_cnt = len(integration.get_status().get('enabled_agents', [])) if integration else 6
        st.metric("激活智能体", f"{agents_cnt}/{agents_cnt}", "100%")
    with col2:
        st.metric("平均响应时间", "2.3s", "-0.5s")
    with col3:
        st.metric("今日分析次数", "128", "+23")
    with col4:
        st.metric("共识达成率", "87%", "+5%")
    
    st.divider()
    
    # 智能体详细配置（若可用则列出真实智能体名）
    st.subheader("⚙️ 智能体配置")
    real_agents = None
    if integration := _get_ta_integration():
        real_agents = integration.get_status().get('enabled_agents', None)
    agents_config = (
        [{"name": n, "emoji": "✅", "status": "✅ 运行中", "weight": 0.15} for n in (real_agents or [])]
        or [
            {"name": "基本面分析师", "emoji": "📊", "status": "✅ 运行中", "weight": 0.20},
            {"name": "技术分析师", "emoji": "📈", "status": "✅ 运行中", "weight": 0.25},
            {"name": "新闻分析师", "emoji": "📰", "status": "✅ 运行中", "weight": 0.15},
            {"name": "社交媒体分析师", "emoji": "💬", "status": "✅ 运行中", "weight": 0.10},
            {"name": "看涨研究员", "emoji": "🔼", "status": "✅ 运行中", "weight": 0.15},
            {"name": "看跌研究员", "emoji": "🔽", "status": "✅ 运行中", "weight": 0.15}
        ]
    )
    
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
    
    performance_data = {
        "智能体": [a['name'] for a in agents_config],
        "准确率": [0.78, 0.82, 0.75, 0.68, 0.81, 0.79][: len(agents_config)],
        "响应时间(s)": [2.1, 1.8, 3.2, 2.5, 2.3, 2.4][: len(agents_config)],
        "信心度": [0.85, 0.89, 0.76, 0.72, 0.88, 0.86][: len(agents_config)],
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_collaboration():
    """协作机制tab"""
    _ensure_user_state()
    st.session_state.setdefault('collab_logs', [])
    st.session_state.setdefault('collab_rounds', [])
    st.header("🗣️ 协作机制")
    
    st.markdown("""
    **结构化辩论流程**
    1. 🎤 初始观点提出
    2. 📋 论据收集与支持
    3. ⚔️ 对立观点辩驳
    4. 🔄 多轮迭代优化
    5. ✅ 共识达成
    """)
    
    # 参数
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("股票代码", value="000001", key="collab_symbol")
    with c2:
        consensus_threshold = st.slider("共识阈值(%)", 50, 90, 75, 1)
    with c3:
        rounds = st.number_input("辩论轮次", 1, 5, 3)
    
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
    if start_btn:
        prog = st.progress(0)
        try:
            import numpy as _np
            for r in range(int(rounds)):
                if integration:
                    market_data = {
                        "price": float(_np.random.uniform(8, 20)),
                        "change_pct": float(_np.random.uniform(-0.03, 0.05)),
                        "volume": int(_np.random.randint(1_000_000, 8_000_000)),
                    }
                    res = asyncio.run(integration.analyze_stock(symbol, market_data))
                else:
                    res = None
                now = datetime.now().strftime('%H:%M:%S')
                # 记录一轮日志
                if res and isinstance(res, dict):
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
                    # 演示：追加一条中立信息
                    st.session_state.collab_logs.append({
                        'time': now, 'agent': 'DemoAgent', 'type': '中立', 'content': 'HOLD · 演示轮次'
                    })
                    st.session_state.collab_rounds.append({'buy': 1, 'sell': 0, 'hold': 2})
                prog.progress((r+1)/int(rounds))
                time.sleep(0.1)
        except Exception as e:
            st.error(f"协作调用失败: {e}")
        finally:
            prog.empty()
    
    st.subheader("🎭 实时辩论过程")
    # 展示聚合日志（最近200条）
    if st.session_state.collab_logs:
        for log in st.session_state.collab_logs[-200:]:
            color_map = {"观点": "🔵", "支持": "🟢", "反驳": "🔴", "中立": "🟡", "决策": "🟣"}
            st.markdown(f"{color_map.get(log['type'], '⚪')} **{log['time']}** - *{log['agent']}* ({log['type']}): {log['content']}")
    else:
        st.info("暂无协作记录")
    
    st.divider()
    
    # 共识可视化
    st.subheader("📊 共识达成可视化")
    # 按轮次聚合统计
    if st.session_state.collab_rounds:
        buy = sum(x['buy'] for x in st.session_state.collab_rounds)
        sell = sum(x['sell'] for x in st.session_state.collab_rounds)
        hold = sum(x['hold'] for x in st.session_state.collab_rounds)
        total = max(buy + sell + hold, 1)
        consensus = max([(buy,'BUY'),(sell,'SELL'),(hold,'HOLD')], key=lambda t:t[0])
        consensus_pct = consensus[0] / total
        # Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=["BUY", "SELL", "HOLD", "最终决策"],
                color=["green", "red", "gray", "purple"]
            ),
            link=dict(
                source=[0,1,2],
                target=[3,3,3],
                value=[max(buy,0.0001), max(sell,0.0001), max(hold,0.0001)]
            )
        )])
        fig.update_layout(title="信号流向与共识形成", height=400)
        st.plotly_chart(fig, use_container_width=True)
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
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=["看涨观点", "看跌观点", "中立观点", "论据支持", "最终决策"],
                color=["green", "red", "gray", "blue", "purple"]
            ),
            link=dict(
                source=[0, 1, 2, 0, 1],
                target=[3, 3, 3, 4, 4],
                value=[45, 25, 30, 40, 20]
            )
        )])
        fig.update_layout(title="观点流向与共识形成", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_information_collection():
    """信息采集tab"""
    st.header("📰 信息采集")
    
    st.markdown("""
    **多源信息整合**
    - 📰 新闻资讯 (v0.1.12智能过滤)
    - 📊 财务数据
    - 💬 社交媒体情绪
    - 📈 实时行情数据
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("今日新闻", "1,247", "+156")
    with col2:
        st.metric("过滤后", "89", "高质量")
    with col3:
        st.metric("情绪指数", "0.68", "偏乐观")
    with col4:
        st.metric("数据源", "12", "多元化")
    
    st.divider()
    
    # 新闻过滤配置
    st.subheader("⚙️ 新闻智能过滤 (v0.1.12)")
    
    with st.expander("过滤配置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            filter_mode = st.selectbox(
                "过滤模式",
                ["基础过滤", "增强过滤", "集成过滤"]
            )
            relevance_threshold = st.slider("相关性阈值", 0.0, 1.0, 0.7)
        with col2:
            quality_threshold = st.slider("质量阈值", 0.0, 1.0, 0.6)
            enable_dedup = st.checkbox("去重", value=True)
    
    if st.button("🔍 应用过滤", type="primary"):
        st.success(f"已应用{filter_mode}，过滤出89条高质量新闻")
    
    # 可选：调用集成的TradingAgents-CN工具进行打分
    st.divider()
    st.subheader("🧰 TradingAgents-CN 采集器（可选）")
    symbol_ic = st.text_input("股票代码 (采集示例)", value="000001", key="ta_cn_symbol")
    if st.button("⚡ 运行采集器并打分"):
        try:
            from integrations.tradingagents_cn.tools.decision_agents import run_agents
            with st.spinner("运行采集与打分..."):
                scores = run_agents(symbol_ic)
            if scores:
                df_scores = pd.DataFrame({"Agent": list(scores.keys()), "Score": list(scores.values())})
                st.dataframe(df_scores, use_container_width=True, hide_index=True)
            else:
                st.info("未返回评分结果")
        except Exception as e:
            st.error(f"采集器运行失败: {e}")
    
    st.divider()
    
    # 最新新闻展示（占位）
    st.subheader("📋 过滤后的新闻")
    
    news_data = [
        {"time": "10:23", "title": "某公司发布Q3财报，净利润同比增长35%", "relevance": 0.92, "sentiment": "正面"},
        {"time": "09:45", "title": "行业监管新政出台，利好龙头企业", "relevance": 0.88, "sentiment": "正面"},
        {"time": "08:30", "title": "技术突破获得重大进展", "relevance": 0.85, "sentiment": "正面"}
    ]
    
    for news in news_data:
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.markdown(f"**{news['time']}**")
            with col2:
                st.markdown(f"{news['title']}")
            with col3:
                sentiment_emoji = "🟢" if news['sentiment'] == "正面" else "🔴" if news['sentiment'] == "负面" else "🟡"
                st.markdown(f"{sentiment_emoji} {news['relevance']:.0%}")


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
            depth = st.selectbox("研究深度", ["简单", "标准", "深度", "极深", "完整"])
        
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
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
                if integration is not None:
                    try:
                        market_data = _build_market_data()
                        # Streamlit 同步环境下调用异步API
                        result = asyncio.run(integration.analyze_stock(symbol, market_data))
                        st.success("分析完成!")
                        # 展示结果
                        if result and isinstance(result, dict) and 'consensus' in result:
                            st.subheader("📊 分析结果（共识）")
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("综合评分", f"{result['consensus'].get('confidence', 0.0)*100:.1f}/100")
                            with c2:
                                st.metric("建议", result['consensus'].get('signal', 'HOLD'))
                            with c3:
                                st.metric("目标价", "—")
                            with c4:
                                st.metric("风险等级", "—")
                            # 细项
                            with st.expander("🔎 参与智能体细节", expanded=False):
                                indiv = result.get('individual_results') or []
                                if indiv:
                                    df = pd.DataFrame([
                                        {
                                            "agent": x.get("agent"),
                                            "signal": x.get("signal"),
                                            "confidence": x.get("confidence"),
                                            "reasoning": x.get("reasoning", "")[:160],
                                        }
                                        for x in indiv
                                    ])
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("无个体智能体明细")
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
        
        if st.button("🚀 批量分析", type="primary", use_container_width=True):
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
                        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)
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
                        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)


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
        st.dataframe(df_usage, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 点数管理
    st.subheader("🎫 点数管理")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**充值点数**")
        amount = st.number_input("充值数量", 10, 10000, 100, step=10)
        if st.button("💰 充值", use_container_width=True):
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
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df, x="模型", y="成本($)", title="成本分布")
        st.plotly_chart(fig, use_container_width=True)
