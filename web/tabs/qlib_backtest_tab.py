"""
Qlib原生回测执行器UI集成
实现完整的Qlib回测功能并展示标准报告和指标
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Qlib 导入
try:
    import qlib
    from qlib.backtest import backtest, get_exchange
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import risk_analysis
    from qlib.backtest.exchange import Exchange
    QLIB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qlib导入失败: {e}")
    QLIB_AVAILABLE = False


def _ensure_qlib_initialized():
    """确保Qlib已经初始化"""
    if not QLIB_AVAILABLE:
        return False
    
    try:
        # 检查Qlib是否已初始化
        from qlib.config import C
        if C.get("provider_uri") is None:
            # 尝试使用默认配置初始化
            default_path = Path.home() / ".qlib/qlib_data/cn_data"
            if default_path.exists():
                qlib.init(provider_uri=str(default_path), region=REG_CN)
                return True
            else:
                return False
        return True
    except Exception as e:
        logger.error(f"Qlib初始化检查失败: {e}")
        return False


def render_qlib_backtest_tab():
    """渲染Qlib原生回测页面"""
    st.header("⏪ Qlib原生回测引擎")
    
    if not QLIB_AVAILABLE:
        st.error("❌ Qlib未安装或导入失败")
        st.info("请先安装Qlib: `pip install pyqlib`")
        return
    
    if not _ensure_qlib_initialized():
        st.warning("⚠️ Qlib未初始化或数据路径不存在")
        st.info("请先在'数据管理'页面初始化Qlib并下载数据")
        
        # 提供快速初始化选项
        with st.expander("🔧 快速初始化Qlib"):
            data_path = st.text_input(
                "Qlib数据路径",
                value=str(Path.home() / ".qlib/qlib_data/cn_data"),
                help="Qlib数据存储路径"
            )
            if st.button("初始化Qlib"):
                try:
                    qlib.init(provider_uri=data_path, region=REG_CN)
                    st.success("✅ Qlib初始化成功！")
                    st.rerun()
                except Exception as e:
                    st.error(f"初始化失败: {e}")
        return
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["📋 回测配置", "📊 回测结果", "📈 风险分析"])
    
    with tab1:
        render_backtest_config()
    
    with tab2:
        render_backtest_results()
    
    with tab3:
        render_backtest_risk_analysis()


def render_backtest_config():
    """渲染回测配置界面"""
    st.subheader("📋 回测参数配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**基本参数**")
        
        # 预测信号源
        signal_source = st.selectbox(
            "预测信号源",
            ["从实验加载", "从文件上传", "使用示例数据"],
            help="选择预测信号的来源"
        )
        
        pred_score = None
        
        if signal_source == "从实验加载":
            exp_name = st.text_input("实验名称", value="qlib_models")
            recorder_id = st.text_input("Recorder ID (可选)", value="")
            artifact_name = st.text_input("预测文件名", value="pred.pkl")
            
            if st.button("加载预测结果"):
                try:
                    from qlib.workflow import R
                    if recorder_id:
                        recorder = R.get_recorder(
                            recorder_id=recorder_id,
                            experiment_name=exp_name
                        )
                    else:
                        recorder = R.get_recorder(experiment_name=exp_name)
                    
                    pred_score = recorder.load_object(artifact_name)
                    st.session_state['backtest_pred_score'] = pred_score
                    st.success(f"✅ 加载成功！预测数据shape: {pred_score.shape}")
                except Exception as e:
                    st.error(f"加载失败: {e}")
        
        elif signal_source == "从文件上传":
            uploaded_file = st.file_uploader(
                "上传预测结果文件 (CSV/PKL)",
                type=['csv', 'pkl'],
                help="CSV格式需包含datetime索引和instrument列"
            )
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        pred_score = pd.read_csv(uploaded_file, index_col=0)
                        pred_score.index = pd.to_datetime(pred_score.index)
                    else:
                        pred_score = pd.read_pickle(uploaded_file)
                    
                    st.session_state['backtest_pred_score'] = pred_score
                    st.success(f"✅ 上传成功！预测数据shape: {pred_score.shape}")
                except Exception as e:
                    st.error(f"上传失败: {e}")
        
        else:  # 使用示例数据
            st.info("将使用随机生成的示例预测数据")
            if st.button("生成示例数据"):
                pred_score = _generate_sample_predictions()
                st.session_state['backtest_pred_score'] = pred_score
                st.success(f"✅ 生成成功！预测数据shape: {pred_score.shape}")
        
        # 时间范围
        st.markdown("**回测时间范围**")
        col_start, col_end = st.columns(2)
        with col_start:
            start_time = st.date_input(
                "开始日期",
                value=datetime(2020, 1, 1),
                key="bt_start_time"
            )
        with col_end:
            end_time = st.date_input(
                "结束日期",
                value=datetime(2020, 12, 31),
                key="bt_end_time"
            )
        
        # 股票池
        benchmark = st.selectbox(
            "基准指数",
            ["SH000300", "SH000905", "SH000852", "SZ399006"],
            help="用于对比的基准指数"
        )
    
    with col2:
        st.markdown("**策略参数**")
        
        # 初始资金
        init_cash = st.number_input(
            "初始资金(元)",
            min_value=10000,
            max_value=100000000,
            value=1000000,
            step=100000
        )
        
        # 持仓数量
        topk = st.slider(
            "持仓股票数量",
            min_value=5,
            max_value=100,
            value=30,
            help="每次调仓时持有的股票数量"
        )
        
        # Dropout参数
        n_drop = st.slider(
            "每次卖出数量",
            min_value=0,
            max_value=50,
            value=5,
            help="每次调仓时强制卖出的股票数量"
        )
        
        st.markdown("**交易成本**")
        col_open, col_close = st.columns(2)
        with col_open:
            open_cost = st.number_input(
                "买入手续费(%)",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                format="%.2f"
            ) / 100
        
        with col_close:
            close_cost = st.number_input(
                "卖出手续费(%)",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                format="%.2f"
            ) / 100
        
        # ===== Alpha融合（P2-1）=====
        st.markdown("**Alpha融合(可选)**")
        try:
            from qlib_enhanced.analysis import load_factor_from_qlib as _load_factor
            alpha_enable = st.checkbox("启用 alpha_confluence / alpha_zs_* 融合到预测得分", value=False)
            if alpha_enable:
                colw1, colw2, colw3 = st.columns(3)
                with colw1:
                    w_conf = st.number_input("w_confluence", value=0.30, step=0.05, format="%.2f")
                with colw2:
                    w_move = st.number_input("w_zs_movement", value=0.15, step=0.05, format="%.2f")
                with colw3:
                    w_upgr = st.number_input("w_zs_upgrade", value=0.10, step=0.05, format="%.2f")
                instruments_alpha = st.selectbox("因子数据股票池", ["csi300","csi500","all"], index=0)
                col_alpha1, col_alpha2 = st.columns(2)
                with col_alpha1:
                    if st.button("应用Alpha加权", use_container_width=True):
                        try:
                            pred = st.session_state.get('backtest_pred_score', None)
                            if pred is None:
                                st.warning("请先在左侧加载/生成预测结果")
                        else:
                            s = str(st.session_state.get('bt_start_time', start_time))
                            e = str(st.session_state.get('bt_end_time', end_time))
                            df_c = _load_factor(instruments=instruments_alpha, start=str(start_time), end=str(end_time), factor_expr="$alpha_confluence", label_expr="Ref($close,-1)/$close-1")
                            df_m = _load_factor(instruments=instruments_alpha, start=str(start_time), end=str(end_time), factor_expr="$alpha_zs_movement", label_expr="Ref($close,-1)/$close-1")
                            df_u = _load_factor(instruments=instruments_alpha, start=str(start_time), end=str(end_time), factor_expr="$alpha_zs_upgrade", label_expr="Ref($close,-1)/$close-1")
                            # 统一为长表
                            def _to_long(x, name):
                                if isinstance(x.index, pd.MultiIndex):
                                    xx = x.copy()
                                    xx.columns = [name]
                                    return xx.reset_index().rename(columns={xx.columns[-1]: name})
                                elif 'instrument' in x.columns:
                                    return x.rename(columns={'factor': name})[['datetime','instrument',name]]
                                else:
                                    return x.reset_index().rename(columns={'index':'datetime','factor':name})
                            
                            c_long = _to_long(df_c, 'alpha_confluence')
                            m_long = _to_long(df_m, 'alpha_zs_movement')
                            u_long = _to_long(df_u, 'alpha_zs_upgrade')
                            
                            # 预测得分长表
                            if isinstance(pred.index, pd.MultiIndex):
                                pred_long = pred.stack().reset_index()
                                pred_long.columns = ['datetime','instrument','score']
                            else:
                                try:
                                    pred_long = pred.reset_index().melt(id_vars=['datetime'], var_name='instrument', value_name='score')
                                except Exception:
                                    st.error("预测结果格式不兼容，需(index=datetime, columns=instrument)")
                                    pred_long = None
                            if pred_long is not None:
                                df_merged = pred_long.merge(c_long, on=['datetime','instrument'], how='left') \
                                                   .merge(m_long, on=['datetime','instrument'], how='left') \
                                                   .merge(u_long, on=['datetime','instrument'], how='left')
                                for col in ['alpha_confluence','alpha_zs_movement','alpha_zs_upgrade']:
                                    if col not in df_merged.columns:
                                        df_merged[col] = 0.0
                                df_merged['score_adj'] = df_merged['score'] * (1 + w_conf*df_merged['alpha_confluence'].fillna(0.0)
                                                                                 + w_move*df_merged['alpha_zs_movement'].fillna(0.0)
                                                                                 + w_upgr*df_merged['alpha_zs_upgrade'].fillna(0.0))
                                # 还原到宽表
                                try:
                                    adj = df_merged.pivot(index='datetime', columns='instrument', values='score_adj')
                                    st.session_state['backtest_pred_score'] = adj
                                    # 保存Alpha加权参数
                                    st.session_state['alpha_weighting_applied'] = True
                                    st.session_state['alpha_weighting_params'] = {
                                        'w_confluence': w_conf,
                                        'w_zs_movement': w_move,
                                        'w_zs_upgrade': w_upgr,
                                        'instruments_alpha': instruments_alpha,
                                        'start_time': str(start_time),
                                        'end_time': str(end_time)
                                    }
                                    st.success("✅ 已应用Alpha加权到预测得分")
                                except Exception as e2:
                                    st.error(f"加权还原失败: {e2}")
                        except Exception as ee:
                            st.error(f"融合失败: {ee}")
                with col_alpha2:
                    if st.button("清除加权", use_container_width=True, help="重置为原始预测得分"):
                        st.session_state['alpha_weighting_applied'] = False
                        st.session_state.pop('alpha_weighting_params', None)
                        st.info("✅ 已清除Alpha加权标记")
        except Exception:
            st.caption("Alpha融合可选：需要 qlib_enhanced.analysis.load_factor_from_qlib 支持")
        
        min_cost = st.number_input(
            "最低手续费(元)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )
    
    st.divider()
    
    # 运行回测按钮
    col_run, col_save = st.columns([1, 1])
    
    with col_run:
        if st.button("🚀 运行回测", type="primary", use_container_width=True):
            if 'backtest_pred_score' not in st.session_state:
                st.error("请先加载或生成预测数据！")
            else:
                with st.spinner("正在运行回测..."):
                    try:
                        results = run_qlib_backtest(
                            pred_score=st.session_state['backtest_pred_score'],
                            start_time=start_time.strftime("%Y-%m-%d"),
                            end_time=end_time.strftime("%Y-%m-%d"),
                            benchmark=benchmark,
                            topk=topk,
                            n_drop=n_drop,
                            init_cash=init_cash,
                            open_cost=open_cost,
                            close_cost=close_cost,
                            min_cost=min_cost
                        )
                        
                        st.session_state['backtest_results'] = results
                        st.session_state['last_backtest_returns'] = results.get('daily_returns')
                        st.success("✅ 回测完成！请切换到'回测结果'标签查看")
                        
                    except Exception as e:
                        st.error(f"回测失败: {e}")
                        import traceback
                        with st.expander("🔍 查看详细错误"):
                            st.code(traceback.format_exc())
    
    with col_save:
        if st.button("💾 保存配置", use_container_width=True):
            config = {
                "start_time": start_time.strftime("%Y-%m-%d"),
                "end_time": end_time.strftime("%Y-%m-%d"),
                "benchmark": benchmark,
                "topk": topk,
                "n_drop": n_drop,
                "init_cash": init_cash,
                "open_cost": open_cost,
                "close_cost": close_cost,
                "min_cost": min_cost,
            }
            st.session_state['backtest_config'] = config
            st.success("✅ 配置已保存")


def render_backtest_results():
    """渲染回测结果"""
    st.subheader("📊 回测结果分析")
    
    # P2-Backtest-UI: Alpha加权标注
    if st.session_state.get('alpha_weighting_applied', False):
        st.success("✅ **已使用 Alpha 加权**")
        params = st.session_state.get('alpha_weighting_params', {})
        with st.expander("🔍 Alpha加权参数", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("w_confluence", f"{params.get('w_confluence', 0):.2f}")
            with col2:
                st.metric("w_zs_movement", f"{params.get('w_zs_movement', 0):.2f}")
            with col3:
                st.metric("w_zs_upgrade", f"{params.get('w_zs_upgrade', 0):.2f}")
            with col4:
                st.metric("股票池", params.get('instruments_alpha', 'N/A'))
            st.caption(f"📅 因子时间范围: {params.get('start_time', 'N/A')} ~ {params.get('end_time', 'N/A')}")
            st.caption("ℹ️ 调整公式: score_adj = score × (1 + w_conf×alpha_confluence + w_move×alpha_zs_movement + w_upgr×alpha_zs_upgrade)")
    
    if 'backtest_results' not in st.session_state:
        st.info("请先在'回测配置'标签运行回测")
        return
    
    results = st.session_state['backtest_results']
    
    # 关键指标卡片
    st.markdown("### 📈 关键绩效指标")
    metrics = results.get('metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ann_return = metrics.get('annualized_return', 0)
        st.metric(
            "年化收益率",
            f"{ann_return:.2%}",
            delta=f"{ann_return:.2%}" if ann_return > 0 else None
        )
    
    with col2:
        sharpe = metrics.get('information_ratio', 0)
        st.metric("夏普比率", f"{sharpe:.3f}")
    
    with col3:
        max_dd = metrics.get('max_drawdown', 0)
        st.metric("最大回撤", f"{max_dd:.2%}")
    
    with col4:
        win_rate = metrics.get('win_rate', 0)
        st.metric("胜率", f"{win_rate:.2%}")
    
    # 净值曲线
    st.markdown("### 💰 净值曲线")
    portfolio_value = results.get('portfolio_value')
    if portfolio_value is not None and not portfolio_value.empty:
        fig = go.Figure()
        
        # 策略净值
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            name='策略净值',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # 基准净值
        benchmark_value = results.get('benchmark_value')
        if benchmark_value is not None and not benchmark_value.empty:
            fig.add_trace(go.Scatter(
                x=benchmark_value.index,
                y=benchmark_value.values,
                name='基准净值',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="组合净值走势",
            xaxis_title="日期",
            yaxis_title="净值",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 回撤分析
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 回撤分析")
        drawdown = results.get('drawdown')
        if drawdown is not None and not drawdown.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                fill='tozeroy',
                name='回撤',
                line=dict(color='red')
            ))
            fig_dd.update_layout(
                xaxis_title="日期",
                yaxis_title="回撤 (%)",
                height=300
            )
            st.plotly_chart(fig_dd, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 收益分布")
        daily_returns = results.get('daily_returns')
        if daily_returns is not None and not daily_returns.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=daily_returns.values * 100,
                nbinsx=50,
                name='日收益',
                marker=dict(color='lightblue')
            ))
            fig_hist.update_layout(
                xaxis_title="日收益率 (%)",
                yaxis_title="频数",
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # 详细指标表格
    st.markdown("### 📋 详细指标")
    
    if metrics:
        metrics_df = pd.DataFrame([
            {"指标": "年化收益率", "数值": f"{metrics.get('annualized_return', 0):.2%}"},
            {"指标": "累计收益率", "数值": f"{metrics.get('cumulative_return', 0):.2%}"},
            {"指标": "夏普比率", "数值": f"{metrics.get('information_ratio', 0):.3f}"},
            {"指标": "最大回撤", "数值": f"{metrics.get('max_drawdown', 0):.2%}"},
            {"指标": "波动率", "数值": f"{metrics.get('volatility', 0):.2%}"},
            {"指标": "胜率", "数值": f"{metrics.get('win_rate', 0):.2%}"},
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # 交易记录
    st.markdown("### 📝 交易记录")
    trades = results.get('trades')
    if trades is not None and not trades.empty:
        st.dataframe(
            trades.head(100),
            use_container_width=True,
            height=300
        )
        
        # 下载交易记录
        csv = trades.to_csv(index=True).encode('utf-8-sig')
        st.download_button(
            label="📥 下载完整交易记录",
            data=csv,
            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("无交易记录")


def render_backtest_risk_analysis():
    """渲染风险分析"""
    st.subheader("📈 风险分析")
    
    if 'backtest_results' not in st.session_state:
        st.info("请先在'回测配置'标签运行回测")
        return
    
    results = st.session_state['backtest_results']
    daily_returns = results.get('daily_returns')
    
    if daily_returns is None or daily_returns.empty:
        st.warning("没有可用的收益数据进行风险分析")
        return
    
    # VaR和CVaR分析
    st.markdown("### ⚠️ VaR / CVaR 分析")
    
    confidence_level = st.slider(
        "置信水平",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f"
    )
    
    var_value = daily_returns.quantile(1 - confidence_level)
    cvar_value = daily_returns[daily_returns <= var_value].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"VaR ({confidence_level:.0%})",
            f"{var_value:.2%}",
            help=f"有{confidence_level:.0%}的把握，日损失不超过此值"
        )
    
    with col2:
        st.metric(
            f"CVaR ({confidence_level:.0%})",
            f"{cvar_value:.2%}",
            help="超过VaR时的平均损失"
        )
    
    with col3:
        downside_risk = daily_returns[daily_returns < 0].std()
        st.metric(
            "下行风险",
            f"{downside_risk:.2%}",
            help="负收益的标准差"
        )
    
    # 滚动风险指标
    st.markdown("### 📊 滚动风险指标")
    
    window = st.select_slider(
        "滚动窗口(天)",
        options=[20, 40, 60, 120, 250],
        value=60
    )
    
    rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (
        daily_returns.rolling(window).mean() * 252 /
        (daily_returns.rolling(window).std() * np.sqrt(252))
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values * 100,
        name=f'{window}日滚动波动率',
        yaxis='y',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        name=f'{window}日滚动夏普',
        yaxis='y2',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title=f"{window}日滚动风险指标",
        xaxis=dict(title="日期"),
        yaxis=dict(title="波动率 (%)", side='left'),
        yaxis2=dict(title="夏普比率", side='right', overlaying='y'),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 月度收益热力图
    st.markdown("### 📅 月度收益热力图")
    
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns_pivot = monthly_returns.to_frame('return')
    monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
    monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
    
    pivot_table = monthly_returns_pivot.pivot_table(
        values='return',
        index='year',
        columns='month'
    )
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_table.values * 100,
        x=[f'{i}月' for i in pivot_table.columns],
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot_table.values * 100, 2),
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="收益率(%)")
    ))
    
    fig_heatmap.update_layout(
        title="月度收益率热力图",
        xaxis_title="月份",
        yaxis_title="年份",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)


def run_qlib_backtest(
    pred_score: pd.DataFrame,
    start_time: str,
    end_time: str,
    benchmark: str,
    topk: int,
    n_drop: int,
    init_cash: float,
    open_cost: float,
    close_cost: float,
    min_cost: float
) -> Dict[str, Any]:
    """
    运行Qlib回测
    
    Args:
        pred_score: 预测分数DataFrame
        start_time: 开始时间
        end_time: 结束时间
        benchmark: 基准指数
        topk: 持仓数量
        n_drop: 每次卖出数量
        init_cash: 初始资金
        open_cost: 买入成本
        close_cost: 卖出成本
        min_cost: 最低成本
    
    Returns:
        包含回测结果的字典
    """
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.backtest import backtest
    from qlib.contrib.evaluate import risk_analysis  # ✅ 导入官方 risk_analysis
    
    # 配置策略
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": pred_score,
            "topk": topk,
            "n_drop": n_drop,
        },
    }
    
    # 配置执行器
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    
    # 配置交易所
    exchange_kwargs = {
        "freq": "day",
        "start_time": start_time,
        "end_time": end_time,
        "codes": "all",
        "open_cost": open_cost,
        "close_cost": close_cost,
        "min_cost": min_cost,
    }
    
    # 运行回测
    portfolio_metric, indicator_metric = backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy_config,
        executor=executor_config,
        benchmark=benchmark,
        account=init_cash,
        exchange_kwargs=exchange_kwargs,
    )
    
    # 提取结果
    analysis_freq = 'day'
    portfolio_df = portfolio_metric[analysis_freq][0]
    
    # 计算各项指标
    daily_returns = portfolio_df['return'].dropna()
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # 净值
    portfolio_value = cumulative_returns
    
    # 回撤
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    # ✅ 使用官方 risk_analysis 计算标准风险指标 (修复 P0 问题)
    risk_metrics_df = risk_analysis(daily_returns, freq="day")
    risk_dict = risk_metrics_df["risk"].to_dict()
    
    # 补充额外指标
    win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0
    cumulative_return = cumulative_returns.iloc[-1] - 1
    
    # 整理指标 (使用官方计算结果)
    metrics = {
        'annualized_return': risk_dict.get('annualized_return', 0),
        'cumulative_return': cumulative_return,
        'information_ratio': risk_dict.get('information_ratio', 0),  # 官方名称
        'max_drawdown': risk_dict.get('max_drawdown', 0),
        'volatility': risk_dict.get('std', 0) * np.sqrt(252),  # 年化波动率
        'win_rate': win_rate,
        # 保留官方完整指标供调试
        '_qlib_risk_metrics': risk_dict,
    }
    
    # 获取交易记录
    trades = None
    if 'orders' in portfolio_df.columns:
        trades = portfolio_df[portfolio_df['orders'].notna()][['orders']].copy()
    
    # 基准数据（如果有）
    benchmark_value = None
    try:
        benchmark_returns = D.features(
            [benchmark],
            ['$close/Ref($close, 1)-1'],
            start_time=start_time,
            end_time=end_time
        )
        if not benchmark_returns.empty:
            benchmark_value = (1 + benchmark_returns).cumprod()
    except:
        pass
    
    return {
        'portfolio_value': portfolio_value,
        'benchmark_value': benchmark_value,
        'daily_returns': daily_returns,
        'drawdown': drawdown,
        'metrics': metrics,
        'trades': trades,
        'raw_portfolio': portfolio_df,
    }


def _generate_sample_predictions() -> pd.DataFrame:
    """生成示例预测数据"""
    # 生成日期范围
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    
    # 生成股票代码
    stocks = [f'{str(i).zfill(6)}.SH' for i in range(1, 51)]
    
    # 生成随机预测分数
    np.random.seed(42)
    data = []
    
    for date in dates:
        for stock in stocks:
            score = np.random.randn()  # 标准正态分布
            data.append({
                'datetime': date,
                'instrument': stock,
                'score': score
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['datetime', 'instrument'])
    
    return df['score']


if __name__ == "__main__":
    render_qlib_backtest_tab()
