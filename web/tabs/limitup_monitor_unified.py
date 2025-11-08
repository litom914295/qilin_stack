"""
一进二涨停监控系统 - 统一视图
整合: 阶段识别 + 核心指标 + 业务流程导向的tab
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 导入自定义组件
import sys
sys.path.append(str(Path(__file__).parent.parent / 'components'))
from stage_indicator import StageIndicator, render_stage_indicator
from metrics_dashboard import MetricsDashboard, create_metrics_from_data
from interactive_filter import InteractiveFilter
from auction_realtime import AuctionRealtimeMonitor
from smart_actions import SmartTipSystem, ActionButtons, RiskLevelIndicator
from enhanced_table import EnhancedTable


def render():
    """渲染统一的涨停监控主界面"""
    st.header("🎯 一进二涨停监控")
    st.caption("T日选股 → T+1竞价监控 → T+2卖出决策 · 全流程智能监控")
    
    # ============ 配置侧边栏 ============
    with st.sidebar:
        st.subheader("⚙️ 系统配置")
        reports_dir = st.text_input("Reports目录", value="reports", key="limitup_reports_dir")
        config_dir = st.text_input("Config目录", value="config", key="limitup_config_dir")
        
        # 获取可用日期
        available_dates = get_available_dates(reports_dir)
        if available_dates:
            selected_date = st.selectbox("选择日期", available_dates, key="limitup_unified_selected_date")
        else:
            selected_date = datetime.now().strftime("%Y-%m-%d")
            st.warning("未找到历史报告")
        
        st.divider()
        
        # 自动刷新配置
        st.subheader("🔄 自动刷新")
        auto_refresh = st.checkbox("启用自动刷新", value=False)
        if auto_refresh:
            refresh_interval = st.slider("刷新间隔(秒)", 5, 60, 10)
            st.info(f"每{refresh_interval}秒刷新一次")
            # 使用st.rerun()实现定时刷新（需要配合time.sleep）
    
    # ============ 加载数据 ============
    auction_data = load_auction_report(reports_dir, selected_date)
    rl_data = load_rl_decision(reports_dir, selected_date)
    
    # ============ 阶段识别器 ============
    st.markdown("### 🕐 当前交易阶段")
    indicator = StageIndicator()
    
    # 准备阶段识别所需的数据
    stage_data = {}
    if auction_data:
        candidates = auction_data.get('candidates', [])
        stage_data['candidate_count'] = len(candidates)
        stage_data['limitup_count'] = auction_data.get('total_limitup_count', 0)
        
        # 分析竞价强弱
        if candidates:
            strong_count = sum(1 for c in candidates if c.get('auction_strength', 0) > 5)
            weak_count = sum(1 for c in candidates if c.get('auction_strength', 0) < -5)
            stage_data['strong_count'] = strong_count
            stage_data['weak_count'] = weak_count
    
    if rl_data:
        selected_stocks = rl_data.get('selected_stocks', [])
        stage_data['position_count'] = len(selected_stocks)
        
        # 计算盈亏（如果有的话）
        profit_count = sum(1 for s in selected_stocks if s.get('current_profit', 0) > 0)
        loss_count = sum(1 for s in selected_stocks if s.get('current_profit', 0) < 0)
        stage_data['profit_count'] = profit_count
        stage_data['loss_count'] = loss_count
    
    # 渲染阶段指示器
    render_stage_indicator(stage_data)
    
    # ============ 核心指标仪表盘 ============
    st.markdown("### 📊 核心指标一览")
    
    # 准备指标数据
    metrics = {
        'candidate_count': stage_data.get('candidate_count', 0),
        'monitor_count': stage_data.get('candidate_count', 0),  # 监控数等于候选数
        'position_count': stage_data.get('position_count', 0),
        'position_value': 0.0,  # TODO: 需要从实际持仓数据计算
        'total_profit': 0.0,    # TODO: 需要从实际持仓数据计算
        'profit_rate': 0.0       # TODO: 需要从实际持仓数据计算
    }
    
    dashboard = MetricsDashboard()
    dashboard.render(metrics)
    
    st.divider()
    
    # ============ 业务流程导向的Tabs ============
    # 根据当前阶段，默认选中相应的tab
    stage_name, _, _ = indicator.get_current_stage()
    
    # 映射阶段到tab索引
    stage_to_tab = {
        "T日选股": 0,
        "T+1竞价监控": 1,
        "T+1盘中交易": 1,
        "T+2卖出决策": 2
    }
    default_tab = stage_to_tab.get(stage_name, 0)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 T日选股",
        "🔥 T+1竞价监控",
        "💰 T+2卖出决策",
        "📈 统计分析"
    ])
    
    with tab1:
        render_t_day_selection(reports_dir, selected_date, auction_data, rl_data)
    
    with tab2:
        render_t1_auction_monitor(reports_dir, selected_date, auction_data, rl_data)
    
    with tab3:
        render_t2_sell_decision(reports_dir, selected_date, rl_data)
    
    with tab4:
        render_statistics(reports_dir, config_dir)


# ============ Tab渲染函数 ============

def render_t_day_selection(reports_dir, selected_date, auction_data, rl_data):
    """Tab1: T日选股 - 涨停板筛选和候选池构建"""
    st.subheader("📊 T日涨停板选股")
    st.caption("筛选今日涨停股，构建明日监控池")
    
    if auction_data is None:
        st.warning(f"未找到{selected_date}的数据")
        st.info("💡 运行命令: `python app/daily_workflow.py` 生成选股数据")
        return
    
    candidates = auction_data.get('candidates', [])
    
    if not candidates:
        st.info("今日无涨停候选股")
        return
    
    # ============ 筛选统计 ============
    st.markdown("#### 📋 候选池概况")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("候选股数量", len(candidates), help="今日涨停股筛选结果")
    
    with col2:
        first_board = sum(1 for c in candidates if c.get('is_first_board', False))
        st.metric("首板数量", first_board, help="首次涨停的股票数量")
    
    with col3:
        avg_score = np.mean([c.get('quality_score', 0) for c in candidates]) if candidates else 0
        st.metric("平均质量分", f"{avg_score:.1f}", help="候选股平均质量评分")
    
    with col4:
        high_quality = sum(1 for c in candidates if c.get('quality_score', 0) >= 70)
        st.metric("优质标的", high_quality, help="质量分≥70的候选股")
    
    st.divider()
    
    # ============ 使用 Phase 2 交互式筛选漏斗 ============
    st.markdown("#### 🔍 交互式筛选漏斗")
    
    # 使用交互式筛选器
    filter_component = InteractiveFilter(df_candidates, key_prefix="t_day_filter")
    filtered_result = filter_component.render()
    
    # 更新filtered_df为filter返回的结果
    filtered_df = filtered_result
    
    # 更新三层筛选统计（保留原有显示逻辑）
    st.markdown("#### 📊 筛选结果概览")
    st.caption("第一层：基础过滤 → 第二层：质量评分 → 第三层：AI智能选股")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🔹 第一层：基础过滤")
        total_limitup = auction_data.get('total_limitup_count', len(candidates))
        st.metric("涨停总数", total_limitup)
        st.metric("↓ 基础筛选后", len(candidates))
        st.caption("✅ 排除ST、*ST、涨停时间过早等")
    
    with col2:
        st.markdown("##### 🔹 第二层：质量评分")
        medium_quality = sum(1 for c in candidates if c.get('quality_score', 0) >= 50)
        st.metric("质量分≥50", medium_quality)
        st.metric("质量分≥70", high_quality)
        st.caption("✅ 综合成交量、换手率、板块热度")
    
    with col3:
        st.markdown("##### 🔹 第三层：AI选股")
        if rl_data:
            selected = rl_data.get('selected_stocks', [])
            st.metric("AI最终选中", len(selected))
            if len(candidates) > 0:
                select_rate = len(selected) / len(candidates) * 100
                st.metric("筛选率", f"{select_rate:.1f}%")
        else:
            st.info("尚未运行AI决策")
        st.caption("✅ RL智能评分 + Thompson Sampling")
    
    st.divider()
    
    # ============ 使用 Phase 2 增强表格 ============
    st.markdown("#### 📋 候选股详情")
    
    if not filtered_df.empty:
        # 选择要显示的列
        display_cols = ['symbol', 'name', 'limitup_time', 'quality_score', 
                       'volume_ratio', 'turnover_rate', 'is_first_board', 'sector']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        # 定义颜色规则
        def quality_color(val):
            if val >= 80:
                return 'green'
            elif val >= 60:
                return 'yellow'
            else:
                return 'orange'
        
        color_rules = {}
        if 'quality_score' in filtered_df.columns:
            color_rules['quality_score'] = quality_color
        
        # 使用增强表格
        table = EnhancedTable(key_prefix="t_day_table")
        table_result = table.render(
            filtered_df[available_cols],
            enable_selection=True,
            enable_sort=True,
            enable_filter=True,
            color_rules=color_rules,
            default_sort_column='quality_score',
            default_sort_ascending=False
        )
        
        # 如果有选中的行，显示操作按钮
        if table_result['selected']:
            selected_symbols = table_result['selected_data']['symbol'].tolist() if 'symbol' in table_result['selected_data'].columns else []
            action_buttons = ActionButtons(key_prefix="t_day_actions")
            action_buttons.render_candidate_pool_actions(table_result['selected_data'])
    
    # ============ 质量分分布图 ============
    st.markdown("#### 📊 质量分分布")
    
    if 'quality_score' in df_candidates.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        scores = df_candidates['quality_score'].dropna()
        ax.hist(scores, bins=20, color='#667eea', edgecolor='black', alpha=0.7)
        ax.axvline(50, color='orange', linestyle='--', label='合格线(50)', linewidth=2)
        ax.axvline(70, color='green', linestyle='--', label='优质线(70)', linewidth=2)
        ax.set_xlabel('质量分')
        ax.set_ylabel('频数')
        ax.set_title('候选股质量分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()


def render_t1_auction_monitor(reports_dir, selected_date, auction_data, rl_data):
    """Tab2: T+1竞价监控 - 实时监控竞价表现"""
    st.subheader("🔥 T+1集合竞价监控")
    st.caption("实时监控候选池竞价表现，辅助买入决策")
    
    if auction_data is None or rl_data is None:
        st.warning(f"未找到{selected_date}的完整数据")
        return
    
    candidates = auction_data.get('candidates', [])
    selected_stocks = rl_data.get('selected_stocks', [])
    
    # ============ 竞价监控核心指标 ============
    st.markdown("#### 🎯 竞价核心指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("监控总数", len(selected_stocks), help="AI选中的监控股票")
    
    with col2:
        if candidates:
            avg_strength = np.mean([c.get('auction_strength', 0) for c in candidates])
            st.metric("平均竞价强度", f"{avg_strength:.2f}", help="竞价涨幅均值")
        else:
            st.metric("平均竞价强度", "N/A")
    
    with col3:
        strong = sum(1 for c in candidates if c.get('auction_strength', 0) > 5)
        st.metric("强势股数", strong, help="竞价涨幅>5%的股票")
    
    with col4:
        weak = sum(1 for c in candidates if c.get('auction_strength', 0) < -5)
        st.metric("弱势股数", weak, help="竞价跌幅>5%的股票")
    
    st.divider()
    
    # ============ AI决策结果 ============
    st.markdown("#### 🤖 AI决策结果")
    
    if selected_stocks:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            threshold = rl_data.get('threshold', 0)
            st.metric("RL得分阈值", f"{threshold:.1f}", help="AI决策的分数线")
        
        with col2:
            topk = rl_data.get('topk', 0)
            st.metric("TopK配置", topk, help="选取Top K个候选")
        
        with col3:
            avg_rl_score = np.mean([s.get('rl_score', 0) for s in selected_stocks])
            st.metric("平均RL得分", f"{avg_rl_score:.2f}")
        
        st.divider()
        
        # ============ 选中股票详情 ============
        st.markdown("#### ✅ 选中股票详情")
        
        df_selected = pd.DataFrame(selected_stocks)
        
        if not df_selected.empty:
            display_cols = ['symbol', 'name', 'rl_score', 'auction_strength', 
                           'auction_change', 'quality_score', 'sector']
            available_cols = [col for col in display_cols if col in df_selected.columns]
            
            # 按RL得分排序
            if 'rl_score' in df_selected.columns:
                df_selected = df_selected.sort_values('rl_score', ascending=False)
            
            st.dataframe(
                df_selected[available_cols],
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("今日未选中任何股票")
    
    st.divider()
    
    # ============ 竞价强度分布 ============
    st.markdown("#### 📊 竞价强度分布")
    
    if candidates:
        fig, ax = plt.subplots(figsize=(10, 4))
        strengths = [c.get('auction_strength', 0) for c in candidates]
        
        ax.hist(strengths, bins=30, color='#48bb78', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='gray', linestyle='-', linewidth=1)
        ax.axvline(5, color='green', linestyle='--', label='强势线(+5%)', linewidth=2)
        ax.axvline(-5, color='red', linestyle='--', label='弱势线(-5%)', linewidth=2)
        ax.set_xlabel('竞价强度 (%)')
        ax.set_ylabel('频数')
        ax.set_title('候选股竞价强度分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # ============ RL得分分布 ============
    st.markdown("#### 📊 RL得分分布")
    
    all_scores = rl_data.get('all_scores', [])
    if all_scores:
        fig, ax = plt.subplots(figsize=(10, 4))
        scores = [s.get('rl_score', 0) for s in all_scores]
        threshold = rl_data.get('threshold', 0)
        
        ax.hist(scores, bins=30, color='#805ad5', edgecolor='black', alpha=0.7)
        ax.axvline(threshold, color='red', linestyle='--', label=f'阈值={threshold:.1f}', linewidth=2)
        ax.set_xlabel('RL得分')
        ax.set_ylabel('频数')
        ax.set_title('候选股RL得分分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()


def render_t2_sell_decision(reports_dir, selected_date, rl_data):
    """Tab3: T+2卖出决策 - 持仓管理和卖出策略"""
    st.subheader("💰 T+2卖出决策")
    st.caption("持仓管理、止盈止损、卖出决策")
    
    if rl_data is None:
        st.warning(f"未找到{selected_date}的持仓数据")
        st.info("💡 此功能需要实时持仓数据接入")
        return
    
    # TODO: 这里需要接入实际的持仓数据
    # 当前从rl_data获取的是选中的股票，不是实际持仓
    selected_stocks = rl_data.get('selected_stocks', [])
    
    if not selected_stocks:
        st.info("当前无持仓数据")
        return
    
    # ============ 持仓概况 ============
    st.markdown("#### 💼 持仓概况")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("持仓数量", len(selected_stocks), help="当前持仓股票数")
    
    with col2:
        # TODO: 需要实际的市值数据
        st.metric("总市值", "待接入", help="当前持仓总市值")
    
    with col3:
        # TODO: 需要实际的盈亏数据
        st.metric("总盈亏", "待接入", help="当前持仓总盈亏")
    
    with col4:
        # TODO: 需要实际的盈亏率
        st.metric("盈亏率", "待接入", help="总盈亏/总成本")
    
    st.divider()
    
    # ============ 持仓明细 ============
    st.markdown("#### 📋 持仓明细")
    
    df_positions = pd.DataFrame(selected_stocks)
    
    if not df_positions.empty:
        display_cols = ['symbol', 'name', 'rl_score', 'quality_score', 'sector']
        available_cols = [col for col in display_cols if col in df_positions.columns]
        
        st.dataframe(
            df_positions[available_cols],
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("⚠️  持仓明细需要接入实时行情数据")
    
    st.divider()
    
    # ============ 卖出建议 ============
    st.markdown("#### 💡 卖出建议")
    
    st.info("""
    **T+2卖出策略建议**:
    
    1. **止盈策略**: 
       - 当日高开>5%: 开盘卖出50%，尾盘根据走势决定剩余部分
       - 当日高开3-5%: 冲高卖出，不破均价线持有
       - 当日平开或低开: 观察到10:30，破均价线止损
    
    2. **止损策略**:
       - 跌破均价线: 立即止损
       - 尾盘跳水: 次日开盘无条件卖出
       - 成交量萎缩: 不再持有观望
    
    3. **风控要求**:
       - 单票最大亏损: -5%
       - 整体回撤控制: -10%
       - 连续3日下跌: 清仓观望
    """)
    
    # TODO: 根据实际持仓数据，智能生成个性化的卖出建议


def render_statistics(reports_dir, config_dir):
    """Tab4: 统计分析 - 历史回测和参数优化"""
    st.subheader("📈 统计分析")
    st.caption("历史回测 · 参数优化 · 绩效评估")
    
    # 创建子标签
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📊 回测结果",
        "⚙️ RL参数推荐",
        "🧠 涨停原因分析"
    ])
    
    with sub_tab1:
        render_backtest_results(reports_dir)
    
    with sub_tab2:
        render_rl_recommendations(config_dir)
    
    with sub_tab3:
        render_limitup_explanation(reports_dir)


def render_backtest_results(reports_dir):
    """回测结果子tab"""
    st.markdown("#### 📊 历史回测结果")
    
    backtest_dir = Path(reports_dir) / "backtest"
    
    if not backtest_dir.exists():
        st.warning("未找到回测结果目录")
        st.info("💡 运行命令: `python app/backtest_engine.py`")
        return
    
    # 加载最新的回测指标
    metrics_files = list(backtest_dir.glob("metrics_*.json"))
    
    if not metrics_files:
        st.warning("未找到回测指标文件")
        return
    
    latest_metrics_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    except Exception:
        st.error("加载回测指标失败")
        return
    
    # 显示关键指标
    st.markdown("##### 关键性能指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总收益率", f"{metrics.get('total_return', 0):.2%}")
    
    with col2:
        st.metric("年化收益率", f"{metrics.get('annual_return', 0):.2%}")
    
    with col3:
        st.metric("Sharpe比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
    
    with col4:
        st.metric("最大回撤", f"{metrics.get('max_drawdown', 0):.2%}")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("胜率", f"{metrics.get('win_rate', 0):.2%}")
    
    with col6:
        st.metric("总交易次数", metrics.get('total_trades', 0))
    
    with col7:
        st.metric("平均单笔收益", f"{metrics.get('avg_trade_return', 0):.2%}")
    
    with col8:
        st.metric("波动率", f"{metrics.get('volatility', 0):.2%}")
    
    st.divider()
    
    # 净值曲线
    st.markdown("##### 净值曲线")
    equity_files = list(backtest_dir.glob("equity_curve_*.csv"))
    
    if equity_files:
        latest_equity_file = max(equity_files, key=lambda x: x.stat().st_mtime)
        try:
            df_equity = pd.read_csv(latest_equity_file)
            if 'date' in df_equity.columns and 'equity' in df_equity.columns:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_equity['date'], df_equity['equity'], linewidth=2, color='#2563eb')
                ax.fill_between(df_equity['date'], df_equity['equity'], alpha=0.3, color='#60a5fa')
                ax.set_xlabel('日期')
                ax.set_ylabel('净值')
                ax.set_title('策略净值曲线')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                plt.close()
        except Exception:
            st.error("加载净值曲线失败")


def render_rl_recommendations(config_dir):
    """RL参数推荐子tab"""
    st.markdown("#### ⚙️ Thompson Sampling参数推荐")
    
    weights = load_rl_weights(config_dir)
    
    if weights is None:
        st.warning("未找到RL权重配置文件")
        st.info("文件位置: `config/rl_weights.json`")
        return
    
    bandit_state = weights.get('bandit_state', {})
    best_action = weights.get('best_action', {})
    
    if not bandit_state:
        st.info("暂无Thompson Sampling优化数据")
        return
    
    # 显示推荐参数
    st.markdown("##### 当前最佳推荐")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("推荐min_score", best_action.get('min_score', 'N/A'))
    
    with col2:
        st.metric("推荐topk", best_action.get('topk', 'N/A'))
    
    with col3:
        total_iterations = sum(state.get('n', 0) for state in bandit_state.values())
        st.metric("累计迭代次数", total_iterations)
    
    st.divider()
    
    # Bandit状态详情
    st.markdown("##### Bandit状态 (Beta分布)")
    
    df_bandit = pd.DataFrame([
        {
            'Action': action,
            'Alpha (成功)': state.get('alpha', 1),
            'Beta (失败)': state.get('beta', 1),
            '迭代次数': state.get('n', 0),
            '期望成功率': state.get('alpha', 1) / (state.get('alpha', 1) + state.get('beta', 1))
        }
        for action, state in bandit_state.items()
    ]).sort_values('期望成功率', ascending=False)
    
    st.dataframe(df_bandit, use_container_width=True, hide_index=True)
    
    # 期望成功率可视化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_bandit['Action'], df_bandit['期望成功率'], color='#10b981')
    ax.set_xlabel('Action (min_score_topk)')
    ax.set_ylabel('期望成功率')
    ax.set_title('Thompson Sampling期望成功率')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()


def render_limitup_explanation(reports_dir):
    """涨停原因分析子tab"""
    st.markdown("#### 🧠 涨停原因可解释分析")
    
    # 这里需要从reports中提取涨停原因数据
    # TODO: 实现涨停原因统计逻辑
    
    st.info("💡 涨停原因分析功能开发中...")
    st.caption("将展示: 热门板块、涨停原因分布、个股原因解读等")


# ============ 辅助函数 ============

def get_available_dates(reports_dir):
    """获取可用的报告日期"""
    try:
        reports_path = Path(reports_dir)
        if not reports_path.exists():
            return []
        
        dates = set()
        for file in reports_path.glob("*.json"):
            parts = file.stem.split("_")
            if len(parts) >= 3:
                date_str = parts[2]
                if len(date_str) == 10:  # YYYY-MM-DD
                    dates.add(date_str)
        
        return sorted(list(dates), reverse=True)
    except Exception:
        return []


def load_auction_report(reports_dir, date):
    """加载竞价报告"""
    try:
        reports_path = Path(reports_dir)
        pattern = f"auction_report_{date}_*.json"
        files = list(reports_path.glob(pattern))
        
        if not files:
            return None
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_rl_decision(reports_dir, date):
    """加载RL决策结果"""
    try:
        reports_path = Path(reports_dir)
        pattern = f"rl_decision_{date}_*.json"
        files = list(reports_path.glob(pattern))
        
        if not files:
            return None
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_rl_weights(config_dir):
    """加载RL权重配置"""
    try:
        config_path = Path(config_dir) / "rl_weights.json"
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None
