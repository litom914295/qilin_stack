# -*- coding: utf-8 -*-
"""
麒麟量化系统 - 涨停板选股 Web Dashboard
实时显示竞价监控、AI决策、涨停原因解释、Thompson Sampling推荐
"""
import os
import glob
import json
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 避免GUI冲突

# 页面配置
st.set_page_config(
    page_title="麒麟涨停板选股系统",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main { padding-top: 2rem !important; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 辅助函数 ====================

def find_latest(pattern):
    """查找最新文件"""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_json_safe(path):
    """安全加载JSON"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载文件失败: {path}\n错误: {e}")
        return None

def load_csv_safe(path):
    """安全加载CSV"""
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"加载文件失败: {path}\n错误: {e}")
        return None

# ==================== 侧边栏配置 ====================

st.sidebar.title("⚙️ 配置")
reports_dir = st.sidebar.text_input("Reports目录", value="reports")
config_dir = st.sidebar.text_input("Config目录", value="config")
selected_date = st.sidebar.date_input(
    "选择日期",
    value=dt.datetime.now(),
    help="查看指定日期的数据"
)
date_str = selected_date.strftime("%Y-%m-%d")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 快速导航")
st.sidebar.markdown("- [今日信号](#tab-signals)")
st.sidebar.markdown("- [AI决策](#tab-ai)")
st.sidebar.markdown("- [涨停原因](#tab-reasons)")
st.sidebar.markdown("- [RL推荐](#tab-rl)")
st.sidebar.markdown("- [回测结果](#tab-backtest)")

# ==================== 主标题 ====================

st.title("🐉 麒麟涨停板选股系统")
st.markdown(f"**当前日期**: {date_str} | **更新时间**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== 标签页 ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 今日信号",
    "🤖 AI决策过程",
    "🧠 涨停原因解释",
    "⚙️ RL参数推荐",
    "📊 回测结果"
])

# ==================== Tab 1: 今日信号 ====================

with tab1:
    st.subheader("📋 今日涨停板候选信号")
    
    col1, col2, col3 = st.columns(3)
    
    # 查找当日竞价报告
    auction_pattern = os.path.join(reports_dir, f"auction_report_{date_str}*.json")
    auction_file = find_latest(auction_pattern)
    
    if auction_file and os.path.exists(auction_file):
        auction_data = load_json_safe(auction_file)
        
        if auction_data:
            col1.metric("候选股票数", len(auction_data.get("stocks", [])))
            
            # 提取统计信息
            stocks = auction_data.get("stocks", [])
            if stocks:
                avg_strength = np.mean([s.get("auction_info", {}).get("strength", 0) for s in stocks])
                col2.metric("平均竞价强度", f"{avg_strength:.1f}")
                
                first_boards = sum(1 for s in stocks if s.get("yesterday_info", {}).get("is_first_board", False))
                col3.metric("首板数量", first_boards)
                
                # 显示股票列表
                st.markdown("### 候选股票详情")
                
                rows = []
                for stock in stocks:
                    yesterday = stock.get("yesterday_info", {})
                    auction = stock.get("auction_info", {})
                    
                    rows.append({
                        "代码": stock.get("symbol", ""),
                        "名称": stock.get("name", ""),
                        "连板天数": yesterday.get("consecutive_days", 0),
                        "封单强度": f"{yesterday.get('seal_ratio', 0):.2%}",
                        "质量分": f"{yesterday.get('quality_score', 0):.1f}",
                        "竞价涨幅": f"{auction.get('final_change', 0):.2%}",
                        "竞价强度": f"{auction.get('strength', 0):.1f}",
                        "是否首板": "✅" if yesterday.get("is_first_board") else "❌",
                        "是否龙头": "⭐" if yesterday.get("is_leader") else "",
                    })
                
                df_stocks = pd.DataFrame(rows)
                st.dataframe(df_stocks, use_container_width=True)
                
                # 竞价强度分布图
                st.markdown("### 竞价强度分布")
                strengths = [s.get("auction_info", {}).get("strength", 0) for s in stocks]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(strengths, bins=20, edgecolor='black', alpha=0.7, color='#1f77b4')
                ax.set_xlabel("竞价强度")
                ax.set_ylabel("数量")
                ax.set_title("竞价强度分布")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            else:
                st.info("当日无候选股票")
    else:
        st.warning(f"未找到 {date_str} 的竞价报告\n\n请运行: `python app/auction_monitor_system.py`")

# ==================== Tab 2: AI决策过程 ====================

with tab2:
    st.subheader("🤖 AI决策过程与权重")
    
    # 加载RL决策结果
    rl_result_pattern = os.path.join(reports_dir, f"rl_decision_{date_str}*.json")
    rl_result_file = find_latest(rl_result_pattern)
    
    if rl_result_file and os.path.exists(rl_result_file):
        rl_data = load_json_safe(rl_result_file)
        
        if rl_data:
            col1, col2, col3 = st.columns(3)
            
            selected = rl_data.get("selected_stocks", [])
            col1.metric("最终选中", len(selected), help="通过RL阈值筛选的股票数")
            
            config = rl_data.get("config", {})
            col2.metric("RL得分阈值", config.get("min_rl_score", 70))
            col3.metric("Top N", config.get("top_n_stocks", 5))
            
            # 显示选中的股票
            if selected:
                st.markdown("### 🎯 最终选中股票")
                
                rows = []
                for i, stock in enumerate(selected, 1):
                    yesterday = stock.get("yesterday_info", {})
                    reasons = stock.get("reasons", [])
                    
                    rows.append({
                        "排名": i,
                        "代码": stock.get("symbol", ""),
                        "名称": stock.get("name", ""),
                        "RL得分": f"{stock.get('rl_score', 0):.2f}",
                        "连板天数": yesterday.get("consecutive_days", 0),
                        "涨停原因": ", ".join(reasons[:3]),
                    })
                
                df_selected = pd.DataFrame(rows)
                st.dataframe(df_selected, use_container_width=True)
                
                # RL得分分布
                st.markdown("### RL得分分布")
                
                all_ranked = rl_data.get("ranked_stocks", [])
                if all_ranked:
                    scores = [s.get("rl_score", 0) for s in all_ranked]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='#ff7f0e')
                    ax.axvline(config.get("min_rl_score", 70), color='red', linestyle='--', label='阈值')
                    ax.set_xlabel("RL得分")
                    ax.set_ylabel("数量")
                    ax.set_title("RL得分分布")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
    else:
        st.warning(f"未找到 {date_str} 的RL决策结果")
    
    # 显示特征权重
    st.markdown("### 📊 特征权重配置")
    
    weights_file = os.path.join(config_dir, "rl_weights.json")
    if os.path.exists(weights_file):
        weights_data = load_json_safe(weights_file)
        
        if weights_data and "feature_weights" in weights_data:
            weights = weights_data["feature_weights"]
            
            # 权重可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(weights.keys())
            values = list(weights.values())
            
            ax.barh(features, values, color='#2ca02c', alpha=0.7)
            ax.set_xlabel("权重")
            ax.set_title("特征权重分布")
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            plt.close()
            
            # 显示权重表格
            with st.expander("查看详细权重"):
                df_weights = pd.DataFrame({
                    "特征": features,
                    "权重": values
                }).sort_values("权重", ascending=False)
                st.dataframe(df_weights, use_container_width=True)
    else:
        st.info("未找到权重配置文件")

# ==================== Tab 3: 涨停原因解释 ====================

with tab3:
    st.subheader("🧠 涨停原因解释")
    
    # 从RL决策结果中提取涨停原因
    rl_result_pattern = os.path.join(reports_dir, f"rl_decision_{date_str}*.json")
    rl_result_file = find_latest(rl_result_pattern)
    
    if rl_result_file and os.path.exists(rl_result_file):
        rl_data = load_json_safe(rl_result_file)
        
        if rl_data:
            all_ranked = rl_data.get("ranked_stocks", [])
            
            if all_ranked:
                # 统计所有原因
                all_reasons = []
                for stock in all_ranked:
                    reasons = stock.get("reasons", [])
                    all_reasons.extend(reasons)
                
                if all_reasons:
                    # 原因频次统计
                    from collections import Counter
                    reason_counts = Counter(all_reasons)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### 涨停原因Top10")
                        
                        top_reasons = reason_counts.most_common(10)
                        reasons_list = [r[0] for r in top_reasons]
                        counts_list = [r[1] for r in top_reasons]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(reasons_list, counts_list, color='#d62728', alpha=0.7)
                        ax.set_xlabel("出现次数")
                        ax.set_title("涨停原因频次统计")
                        ax.grid(True, alpha=0.3, axis='x')
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("### 详细统计")
                        for reason, count in top_reasons:
                            percentage = count / len(all_ranked) * 100
                            st.metric(reason, count, f"{percentage:.1f}%")
                    
                    # 显示每只股票的原因
                    st.markdown("### 个股涨停原因详情")
                    
                    rows = []
                    for stock in all_ranked[:20]:  # 只显示前20只
                        reasons = stock.get("reasons", [])
                        reason_scores = stock.get("reason_scores", [])
                        
                        # 格式化原因
                        reason_str = ""
                        for i, (reason, score_tuple) in enumerate(zip(reasons, reason_scores)):
                            if isinstance(score_tuple, (list, tuple)) and len(score_tuple) >= 2:
                                score = score_tuple[1]
                            else:
                                score = 1.0
                            reason_str += f"{i+1}. {reason} ({score:.2f})\n"
                        
                        rows.append({
                            "代码": stock.get("symbol", ""),
                            "名称": stock.get("name", ""),
                            "RL得分": f"{stock.get('rl_score', 0):.2f}",
                            "涨停原因": reason_str.strip() or "无"
                        })
                    
                    df_reasons = pd.DataFrame(rows)
                    st.dataframe(df_reasons, use_container_width=True, height=400)
                else:
                    st.info("暂无涨停原因数据")
            else:
                st.info("暂无排序股票数据")
    else:
        st.warning(f"未找到 {date_str} 的决策结果")

# ==================== Tab 4: RL参数推荐 ====================

with tab4:
    st.subheader("⚙️ Thompson Sampling RL参数推荐")
    
    weights_file = os.path.join(config_dir, "rl_weights.json")
    
    if os.path.exists(weights_file):
        weights_data = load_json_safe(weights_file)
        
        if weights_data:
            col1, col2, col3 = st.columns(3)
            
            best_action = weights_data.get("best_action", [70.0, 5])
            col1.metric("推荐min_score", f"{best_action[0]:.1f}", help="最低RL得分阈值")
            col2.metric("推荐topk", best_action[1], help="每日选择股票数")
            col3.metric("迭代次数", weights_data.get("iteration", 0), help="累计学习次数")
            
            # Bandit状态
            st.markdown("### Bandit状态（Beta分布参数）")
            
            bandit_state = weights_data.get("bandit_state", {})
            if bandit_state:
                rows = []
                for action_key, params in bandit_state.items():
                    alpha, beta = params
                    # 解析action
                    parts = action_key.split("_")
                    min_score = float(parts[0])
                    topk = int(parts[1])
                    
                    # 计算期望值
                    expected = alpha / (alpha + beta)
                    
                    rows.append({
                        "min_score": min_score,
                        "topk": topk,
                        "alpha": f"{alpha:.2f}",
                        "beta": f"{beta:.2f}",
                        "期望成功率": f"{expected:.2%}",
                        "样本数": int(alpha + beta - 2)
                    })
                
                df_bandit = pd.DataFrame(rows).sort_values("期望成功率", ascending=False)
                st.dataframe(df_bandit, use_container_width=True)
                
                # 可视化期望成功率
                st.markdown("### 期望成功率分布")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 按topk分组
                for topk in sorted(df_bandit["topk"].unique()):
                    subset = df_bandit[df_bandit["topk"] == topk]
                    expected_rates = [float(r.strip('%'))/100 for r in subset["期望成功率"]]
                    min_scores = subset["min_score"].values
                    ax.plot(min_scores, expected_rates, marker='o', label=f'TopK={topk}', linewidth=2)
                
                ax.set_xlabel("Min Score")
                ax.set_ylabel("期望成功率")
                ax.set_title("不同参数组合的期望成功率")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Bandit状态为空，需要积累更多数据")
    else:
        st.warning("未找到RL权重文件，请先运行系统")

# ==================== Tab 5: 回测结果 ====================

with tab5:
    st.subheader("📊 回测结果")
    
    # 查找最新的回测报告
    backtest_pattern = os.path.join(reports_dir, "backtest", "metrics_*.json")
    backtest_file = find_latest(backtest_pattern)
    
    if backtest_file and os.path.exists(backtest_file):
        metrics = load_json_safe(backtest_file)
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("总收益率", f"{metrics.get('total_return', 0):.2%}")
            col2.metric("年化收益率", f"{metrics.get('annualized_return', 0):.2%}")
            col3.metric("Sharpe比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
            col4.metric("最大回撤", f"{metrics.get('max_drawdown', 0):.2%}")
            
            col5, col6, col7, col8 = st.columns(4)
            
            col5.metric("胜率", f"{metrics.get('win_rate', 0):.2%}")
            col6.metric("总交易次数", metrics.get('total_trades', 0))
            col7.metric("平均单笔收益", f"{metrics.get('avg_profit', 0):.2f}")
            col8.metric("平均收益率", f"{metrics.get('avg_profit_rate', 0):.2%}")
            
            # 加载净值曲线
            equity_pattern = os.path.join(reports_dir, "backtest", "equity_curve_*.csv")
            equity_file = find_latest(equity_pattern)
            
            if equity_file and os.path.exists(equity_file):
                equity_df = load_csv_safe(equity_file)
                
                if equity_df is not None and not equity_df.empty:
                    st.markdown("### 净值曲线")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    if "date" in equity_df.columns and "total_value" in equity_df.columns:
                        equity_df["date"] = pd.to_datetime(equity_df["date"])
                        equity_df = equity_df.sort_values("date")
                        
                        # 归一化净值
                        initial_value = equity_df["total_value"].iloc[0]
                        equity_df["净值"] = equity_df["total_value"] / initial_value
                        
                        ax.plot(equity_df["date"], equity_df["净值"], linewidth=2, color='#1f77b4')
                        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                        ax.set_xlabel("日期")
                        ax.set_ylabel("净值（归一化）")
                        ax.set_title("回测净值曲线")
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
            
            # 交易记录
            trades_pattern = os.path.join(reports_dir, "backtest", "trade_log_*.csv")
            trades_file = find_latest(trades_pattern)
            
            if trades_file and os.path.exists(trades_file):
                trades_df = load_csv_safe(trades_file)
                
                if trades_df is not None and not trades_df.empty:
                    st.markdown("### 最近交易记录")
                    st.dataframe(trades_df.tail(20), use_container_width=True)
    else:
        st.warning("未找到回测结果\n\n请运行: `python app/backtest_engine.py`")

# ==================== 页脚 ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>麒麟量化涨停板选股系统 v2.0 | Powered by Streamlit</p>
    <p>集合竞价监控 → AI智能决策 → 涨停原因解释 → Thompson Sampling优化</p>
</div>
""", unsafe_allow_html=True)
