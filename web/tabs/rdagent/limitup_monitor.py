"""
涨停板选股系统监控模块
整合到unified_dashboard的独立tab
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


def render():
    """渲染涨停板监控主界面"""
    st.header("🎯 涨停板选股系统监控")
    st.caption("实时监控 · AI决策过程 · 涨停原因解释 · RL参数推荐 · 回测结果")
    
    # 配置侧边栏
    with st.sidebar:
        st.subheader("⚙️ 配置")
        reports_dir = st.text_input("Reports目录", value="reports")
        config_dir = st.text_input("Config目录", value="config")
        
        # 获取可用日期
        available_dates = get_available_dates(reports_dir)
        if available_dates:
            selected_date = st.selectbox("选择日期", available_dates)
        else:
            selected_date = datetime.now().strftime("%Y-%m-%d")
            st.warning("未找到历史报告")
    
    # 主标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 今日信号",
        "🤖 AI决策过程",
        "🧠 涨停原因解释",
        "⚙️ RL参数推荐",
        "📊 回测结果"
    ])
    
    with tab1:
        render_today_signals(reports_dir, selected_date)
    
    with tab2:
        render_ai_decision(reports_dir, config_dir, selected_date)
    
    with tab3:
        render_limitup_explanation(reports_dir, selected_date)
    
    with tab4:
        render_rl_recommendations(config_dir)
    
    with tab5:
        render_backtest_results(reports_dir)


def get_available_dates(reports_dir):
    """获取可用的报告日期"""
    try:
        reports_path = Path(reports_dir)
        if not reports_path.exists():
            return []
        
        dates = set()
        for file in reports_path.glob("*.json"):
            # 从文件名提取日期: auction_report_2025-01-15_093000.json
            parts = file.stem.split("_")
            if len(parts) >= 3:
                date_str = parts[2]
                if len(date_str) == 10:  # YYYY-MM-DD
                    dates.add(date_str)
        
        return sorted(list(dates), reverse=True)
    except Exception:
        return []


def render_today_signals(reports_dir, selected_date):
    """Tab1: 今日信号"""
    st.subheader("📋 今日竞价监控信号")
    
    # 加载竞价报告
    auction_data = load_auction_report(reports_dir, selected_date)
    
    if auction_data is None:
        st.warning(f"未找到{selected_date}的竞价报告")
        st.info("请先运行: `python app/daily_workflow.py`")
        return
    
    # 显示统计指标
    candidates = auction_data.get('candidates', [])
    if not candidates:
        st.info("今日无候选股票")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("候选股票数", len(candidates))
    
    with col2:
        avg_strength = np.mean([c.get('auction_strength', 0) for c in candidates])
        st.metric("平均竞价强度", f"{avg_strength:.2f}")
    
    with col3:
        first_board_count = sum(1 for c in candidates if c.get('is_first_board', False))
        st.metric("首板数量", first_board_count)
    
    with col4:
        high_quality_count = sum(1 for c in candidates if c.get('quality_score', 0) > 70)
        st.metric("高质量标的", high_quality_count)
    
    st.divider()
    
    # 候选股票列表
    st.subheader("候选股票详情")
    df_candidates = pd.DataFrame(candidates)
    
    if not df_candidates.empty:
        display_cols = ['symbol', 'name', 'auction_strength', 'auction_change', 
                       'quality_score', 'is_first_board', 'sector']
        available_cols = [col for col in display_cols if col in df_candidates.columns]
        
        st.dataframe(
            df_candidates[available_cols].sort_values(
                'auction_strength', ascending=False
            ).head(20),
            use_container_width=True,
            hide_index=True
        )
    
    # 竞价强度分布图
    st.subheader("竞价强度分布")
    fig, ax = plt.subplots(figsize=(10, 4))
    strengths = [c.get('auction_strength', 0) for c in candidates]
    ax.hist(strengths, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel('竞价强度')
    ax.set_ylabel('频数')
    ax.set_title('候选股票竞价强度分布')
    st.pyplot(fig)
    plt.close()


def render_ai_decision(reports_dir, config_dir, selected_date):
    """Tab2: AI决策过程"""
    st.subheader("🤖 AI决策过程可视化")
    
    # 加载RL决策结果
    rl_data = load_rl_decision(reports_dir, selected_date)
    
    if rl_data is None:
        st.warning(f"未找到{selected_date}的RL决策结果")
        st.info("请先运行: `python app/daily_workflow.py`")
        return
    
    # 显示决策参数
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stocks = rl_data.get('selected_stocks', [])
        st.metric("最终选中股票数", len(selected_stocks))
    
    with col2:
        threshold = rl_data.get('threshold', 0)
        st.metric("RL得分阈值", f"{threshold:.1f}")
    
    with col3:
        topk = rl_data.get('topk', 0)
        st.metric("TopK配置", topk)
    
    st.divider()
    
    # 选中股票详情
    st.subheader("选中股票详情")
    if selected_stocks:
        df_selected = pd.DataFrame(selected_stocks)
        st.dataframe(df_selected, use_container_width=True, hide_index=True)
    else:
        st.info("未选中任何股票")
    
    # RL得分分布
    st.subheader("RL得分分布")
    all_scores = rl_data.get('all_scores', [])
    if all_scores:
        fig, ax = plt.subplots(figsize=(10, 4))
        scores = [s.get('rl_score', 0) for s in all_scores]
        ax.hist(scores, bins=20, color='lightgreen', edgecolor='black')
        ax.axvline(threshold, color='red', linestyle='--', label=f'阈值={threshold:.1f}')
        ax.set_xlabel('RL得分')
        ax.set_ylabel('频数')
        ax.set_title('候选股票RL得分分布')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # 特征权重
    st.subheader("特征权重")
    weights = load_rl_weights(config_dir)
    if weights:
        feature_weights = weights.get('weights', {})
        if feature_weights:
            df_weights = pd.DataFrame([
                {'特征': k, '权重': v} 
                for k, v in feature_weights.items()
            ]).sort_values('权重', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(df_weights['特征'], df_weights['权重'], color='coral')
            ax.set_xlabel('权重')
            ax.set_title('特征重要性权重')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()


def render_limitup_explanation(reports_dir, selected_date):
    """Tab3: 涨停原因解释"""
    st.subheader("🧠 涨停原因可解释分析")
    
    # 加载决策数据
    rl_data = load_rl_decision(reports_dir, selected_date)
    
    if rl_data is None:
        st.warning(f"未找到{selected_date}的决策数据")
        return
    
    selected_stocks = rl_data.get('selected_stocks', [])
    
    if not selected_stocks:
        st.info("今日无选中股票")
        return
    
    # 统计涨停原因
    reason_counts = {}
    for stock in selected_stocks:
        reasons = stock.get('reasons', [])
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    # 显示Top10原因
    st.subheader("涨停原因Top10")
    if reason_counts:
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_reasons = pd.DataFrame(sorted_reasons, columns=['原因', '频次'])
            st.dataframe(df_reasons, use_container_width=True, hide_index=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            reasons, counts = zip(*sorted_reasons)
            ax.barh(reasons, counts, color='steelblue')
            ax.set_xlabel('频次')
            ax.set_title('涨停原因分布')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
    
    st.divider()
    
    # 个股涨停原因
    st.subheader("个股涨停原因详情")
    for stock in selected_stocks[:10]:  # 显示前10只
        with st.expander(f"{stock.get('symbol', 'N/A')} - {stock.get('name', 'N/A')}"):
            reasons = stock.get('reasons', [])
            if reasons:
                for reason in reasons:
                    st.write(f"- {reason}")
            else:
                st.write("无明确原因标签")


def render_rl_recommendations(config_dir):
    """Tab4: RL参数推荐"""
    st.subheader("⚙️ Thompson Sampling参数推荐")
    
    # 加载RL权重
    weights = load_rl_weights(config_dir)
    
    if weights is None:
        st.warning("未找到RL权重配置文件")
        st.info("文件位置: `config/rl_weights.json`")
        return
    
    # Thompson Sampling状态
    bandit_state = weights.get('bandit_state', {})
    best_action = weights.get('best_action', {})
    
    if not bandit_state:
        st.info("暂无Thompson Sampling优化数据")
        return
    
    # 显示推荐参数
    st.subheader("当前最佳推荐")
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
    st.subheader("Bandit状态 (Beta分布)")
    
    if bandit_state:
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
        ax.bar(df_bandit['Action'], df_bandit['期望成功率'], color='mediumseagreen')
        ax.set_xlabel('Action (min_score_topk)')
        ax.set_ylabel('期望成功率')
        ax.set_title('Thompson Sampling期望成功率')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()


def render_backtest_results(reports_dir):
    """Tab5: 回测结果"""
    st.subheader("📊 历史回测结果")
    
    # 查找回测文件
    backtest_dir = Path(reports_dir) / "backtest"
    
    if not backtest_dir.exists():
        st.warning("未找到回测结果目录")
        st.info("请先运行: `python app/backtest_engine.py`")
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
    st.subheader("关键性能指标")
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
        volatility = metrics.get('volatility', 0)
        st.metric("波动率", f"{volatility:.2%}")
    
    st.divider()
    
    # 净值曲线
    st.subheader("净值曲线")
    equity_files = list(backtest_dir.glob("equity_curve_*.csv"))
    
    if equity_files:
        latest_equity_file = max(equity_files, key=lambda x: x.stat().st_mtime)
        try:
            df_equity = pd.read_csv(latest_equity_file)
            if 'date' in df_equity.columns and 'equity' in df_equity.columns:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_equity['date'], df_equity['equity'], linewidth=2, color='darkblue')
                ax.set_xlabel('日期')
                ax.set_ylabel('净值')
                ax.set_title('策略净值曲线')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                plt.close()
        except Exception:
            st.error("加载净值曲线失败")
    
    st.divider()
    
    # 交易记录
    st.subheader("最近交易记录")
    trade_files = list(backtest_dir.glob("trade_log_*.csv"))
    
    if trade_files:
        latest_trade_file = max(trade_files, key=lambda x: x.stat().st_mtime)
        try:
            df_trades = pd.read_csv(latest_trade_file)
            st.dataframe(df_trades.tail(20), use_container_width=True, hide_index=True)
        except Exception:
            st.error("加载交易记录失败")


# ============ 辅助函数 ============

def load_auction_report(reports_dir, date):
    """加载竞价报告"""
    try:
        reports_path = Path(reports_dir)
        pattern = f"auction_report_{date}_*.json"
        files = list(reports_path.glob(pattern))
        
        if not files:
            return None
        
        # 取最新的
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
