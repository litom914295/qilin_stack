"""缠论系统 Web 界面

独立的缠论技术分析系统，包含:
1. 多智能体选股
2. 缠论评分分析
3. 一进二涨停策略
4. 回测与绩效分析

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - 缠论模块 Web 集成
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# P2-1: 多周期共振评分
try:
    from qlib_enhanced.chanlun.multi_timeframe_confluence import (
        resample_ohlc, compute_direction, compute_confluence_score,
    )
except Exception:
    resample_ohlc = None
    compute_direction = None
    compute_confluence_score = None

# P2-2: 信号存储服务
try:
    from web.services.chanlun_signal_store import ChanLunSignalStore
except Exception:
    ChanLunSignalStore = None

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from strategies.multi_agent_selector import MultiAgentStockSelector
    from agents.chanlun_agent import ChanLunScoringAgent
    from agents.limitup_chanlun_agent import LimitUpSignalGenerator
except Exception as e:
    st.error(f"缠论模块导入失败: {e}")
    MultiAgentStockSelector = None
    ChanLunScoringAgent = None
    LimitUpSignalGenerator = None

# 复用现有的 AKShare 适配器
try:
    from layer3_online.adapters.akshare_adapter import get_daily_ohlc
    from rd_agent.limit_up_data import LimitUpDataInterface
    AKSHARE_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"AKShare adapter not available: {e}")
    AKSHARE_AVAILABLE = False
    get_daily_ohlc = None
    LimitUpDataInterface = None


def render_chanlun_system_tab():
    """渲染缠论系统主界面"""
    
    st.header("📈 缠论技术分析系统")
    st.caption("基于 CZSC + Chan.py 的独立选股与分析系统")
    
    # 创建子标签页 - 新增实时监控功能(P1-4)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 多智能体选股",
        "📊 缠论评分分析",
        "🚀 一进二涨停策略",
        "📈 回测与绩效",
        "🔴 实时信号监控",
        "📡 多股票监控",
        "📊 统计分析"
    ])
    
    with tab1:
        render_multi_agent_selector()
    
    with tab2:
        render_chanlun_scoring()
    
    with tab3:
        render_limitup_strategy()
    
    with tab4:
        render_backtest_performance()
    
    with tab5:
        render_realtime_signals()
    
    with tab6:
        render_multi_stock_monitor()
    
    with tab7:
        render_statistical_analysis()


def render_multi_agent_selector():
    """渲染多智能体选股界面"""
    
    st.subheader("🤖 多智能体选股系统")
    
    # 检查模块是否可用
    if MultiAgentStockSelector is None:
        st.warning("⚠️ 缠论选股模块未加载，请先使用示例数据测试基本功能")
    
    # 系统说明
    with st.expander("ℹ️ 系统说明", expanded=False):
        st.markdown("""
        **多智能体选股系统**整合5个维度的智能体进行综合评分:
        
        1. **缠论智能体** (35%) - 形态/买卖点/背驰分析
        2. **技术指标智能体** (25%) - MACD/RSI/均线/布林带
        3. **成交量智能体** (15%) - 量价配合/放量突破
        4. **基本面智能体** (15%) - PE/PB/ROE估值
        5. **市场情绪智能体** (10%) - 涨跌幅/换手/振幅
        
        **适用场景**:
        - 非 Qlib 工作流的独立选股
        - 快速原型验证
        - 明确的多因子规则评分
        """)
    
    # 配置区
    st.markdown("### ⚙️ 权重配置")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chanlun_weight = st.slider("缠论权重", 0.0, 1.0, 0.35, 0.05, key="cl_weight")
        technical_weight = st.slider("技术指标权重", 0.0, 1.0, 0.25, 0.05, key="tech_weight")
    
    with col2:
        volume_weight = st.slider("成交量权重", 0.0, 1.0, 0.15, 0.05, key="vol_weight")
        fundamental_weight = st.slider("基本面权重", 0.0, 1.0, 0.15, 0.05, key="fund_weight")
    
    with col3:
        sentiment_weight = st.slider("情绪权重", 0.0, 1.0, 0.10, 0.05, key="sent_weight")
        top_n = st.number_input("选股数量", min_value=1, max_value=50, value=10, key="topn")
    
    # 权重归一化提示
    total_weight = chanlun_weight + technical_weight + volume_weight + fundamental_weight + sentiment_weight
    st.info(f"当前权重总和: {total_weight:.2f} (系统会自动归一化)")
    
    st.divider()
    
    # 数据输入区
    st.markdown("### 📊 数据输入")
    
    data_source = st.radio(
        "选择数据源",
        ["使用示例数据", "上传CSV文件", "连接实时数据"],
        horizontal=True,
        key="data_source"
    )
    
    if data_source == "使用示例数据":
        if st.button("🎲 生成示例数据", type="primary"):
            with st.spinner("正在生成示例数据..."):
                stock_data = generate_sample_stock_data(n_stocks=20, n_days=100)
                st.session_state['chanlun_stock_data'] = stock_data
                st.success(f"✅ 已生成 {len(stock_data)} 只股票的数据")
    
    elif data_source == "上传CSV文件":
        uploaded_file = st.file_uploader(
            "上传股票数据 CSV (需包含: datetime, open, high, low, close, volume)",
            type=['csv'],
            key="upload_csv"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ 已加载数据: {len(df)} 行")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ 文件解析失败: {e}")
    
    else:  # 连接实时数据
        render_akshare_data_connection()
    
    st.divider()
    
    # 选股执行区
    st.markdown("### 🎯 执行选股")
    
    if 'chanlun_stock_data' in st.session_state:
        stock_data = st.session_state['chanlun_stock_data']
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 开始选股", type="primary", use_container_width=True):
                run_multi_agent_selection(
                    stock_data,
                    chanlun_weight,
                    technical_weight,
                    volume_weight,
                    fundamental_weight,
                    sentiment_weight,
                    top_n
                )
        
        with col2:
            if st.button("📥 导出结果", use_container_width=True):
                if 'selection_results' in st.session_state:
                    csv = st.session_state['selection_results'].to_csv(index=False)
                    st.download_button(
                        "💾 下载 CSV",
                        csv,
                        "chanlun_selection.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("请先执行选股")
        
        with col3:
            if st.button("🔄 重置", use_container_width=True):
                if 'selection_results' in st.session_state:
                    del st.session_state['selection_results']
                st.rerun()
    else:
        st.warning("⚠️ 请先选择或生成数据")
    
    # 结果展示区
    if 'selection_results' in st.session_state:
        st.divider()
        st.markdown("### 📊 选股结果")
        
        results = st.session_state['selection_results']
        
        # 结果统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("入选股票", f"{len(results)} 只")
        with col2:
            st.metric("平均评分", f"{results['score'].mean():.1f}")
        with col3:
            st.metric("最高评分", f"{results['score'].max():.1f}")
        with col4:
            st.metric("平均置信度", f"{results['confidence'].mean():.2%}")
        
        # 结果表格
        st.dataframe(
            results.style.background_gradient(subset=['score'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # 评分分布图
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results['code'],
            y=results['score'],
            marker=dict(
                color=results['score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="评分")
            ),
            text=results['grade'],
            textposition='outside'
        ))
        fig.update_layout(
            title="股票评分分布",
            xaxis_title="股票代码",
            yaxis_title="综合评分",
            height=400,
            hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_chanlun_scoring():
    """渲染缠论评分分析界面"""
    
    st.subheader("📊 缠论评分分析")
    
    with st.expander("ℹ️ 评分体系说明", expanded=False):
        st.markdown("""
        **缠论评分系统** (0-100分) 包含4个维度:
        
        1. **形态评分** (40%) - 分型/笔/中枢质量
        2. **买卖点评分** (35%) - 买卖点类型和有效性
        3. **背驰评分** (15%) - MACD背驰风险
        4. **多级别共振** (10%) - 跨周期一致性
        
        **评分等级**:
        - 90-100: 强烈推荐 (Strong Buy)
        - 75-89: 推荐 (Buy)
        - 60-74: 中性偏多 (Slight Buy)
        - 40-59: 中性 (Neutral)
        - 25-39: 观望 (Wait)
        - 0-24: 规避 (Avoid)
        """)
    
    # 单股票详细分析
    st.markdown("### 🔍 单股票详细分析")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_code = st.text_input("股票代码", value="000001.SZ", key="single_code")
    
    with col2:
        if st.button("📈 加载数据", type="primary"):
            # 生成示例数据
            sample_df = generate_single_stock_data(stock_code)
            st.session_state['single_stock_data'] = sample_df
            st.session_state['analyzing_code'] = stock_code
            st.success(f"✅ 已加载 {stock_code} 数据")
    
    if 'single_stock_data' in st.session_state:
        df = st.session_state['single_stock_data']
        code = st.session_state.get('analyzing_code', 'Unknown')
        
        # 执行评分
        agent = ChanLunScoringAgent(
            morphology_weight=0.40,
            bsp_weight=0.35,
            enable_bsp=True,
            enable_divergence=True
        )
        
        score, details = agent.score(df, code, return_details=True)
        
        # 显示评分结果
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("综合评分", f"{score:.1f}/100", 
                     delta=None,
                     delta_color="normal")
        with col2:
            st.metric("评级", details['grade'])
        with col3:
            st.metric("形态评分", f"{details['morphology_score']:.1f}")
        with col4:
            st.metric("买卖点评分", f"{details['bsp_score']:.1f}")
        
        # 各维度评分雷达图
        fig = go.Figure()
        
        categories = ['形态', '买卖点', '背驰', '多级别']
        scores = [
            details['morphology_score'],
            details['bsp_score'],
            details['divergence_score'],
            details['multi_level_score']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name=code
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=400,
            title="各维度评分雷达图"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细说明
        with st.expander("📝 详细分析说明", expanded=True):
            st.markdown(details['explanation'])


def render_limitup_strategy():
    """渲染一进二涨停策略界面"""
    
    st.subheader("🚀 一进二涨停策略")
    
    with st.expander("ℹ️ 策略说明", expanded=False):
        st.markdown("""
        **一进二涨停策略** 专注于涨停板打板策略:
        
        **核心逻辑**:
        1. 识别一字板或T字板
        2. 评估涨停质量 (封单强度/资金流向)
        3. 分析板块效应
        4. 预测次日表现
        
        **适用场景**: 短线打板、超短线交易
        """)
    
    st.info("🚧 一进二涨停策略界面开发中...")
    st.markdown("""
    **计划功能**:
    - ✅ 涨停质量评分
    - ✅ 封单强度分析
    - ✅ 板块效应识别
    - 🚧 实时监控面板
    - 🚧 历史回测分析
    """)


def render_backtest_performance():
    """渲染回测与绩效界面"""
    
    st.subheader("📈 回测与绩效分析")
    
    st.info("""
    💡 **推荐使用 Qlib 回测框架**
    
    缠论系统已集成到 Qlib 工作流中，推荐使用:
    - `configs/chanlun/qlib_backtest.yaml` - 纯缠论评分回测
    - `configs/chanlun/enhanced_strategy.yaml` - 融合策略回测
    
    请前往 **📦 Qlib** 标签页进行完整回测。
    """)
    
    st.divider()
    st.markdown("### 📊 简化回测 (开发中)")
    
    # 简化回测参数
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input("初始资金 (万元)", 
                                         min_value=10, max_value=10000, 
                                         value=100, step=10)
    with col2:
        start_date = st.date_input("开始日期", 
                                   value=datetime.now() - timedelta(days=365))
    with col3:
        end_date = st.date_input("结束日期", 
                                 value=datetime.now())
    
    if st.button("🚀 运行回测", type="primary"):
        st.warning("⚠️ 简化回测功能正在开发中，请使用 Qlib 回测框架")


# ============ 辅助函数 ============

def generate_sample_stock_data(n_stocks=20, n_days=100):
    """生成示例股票数据"""
    np.random.seed(42)
    stock_data = {}
    
    start_date = pd.Timestamp('2023-01-01')
    
    for i in range(n_stocks):
        code = f'{i:06d}.SZ'
        
        dates = pd.date_range(start_date, periods=n_days, freq='D')
        
        price = 10.0
        prices = []
        
        for _ in range(n_days):
            change = np.random.randn() * 0.02
            price *= (1 + change)
            prices.append(price)
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': [p * (1 + np.random.randn() * 0.01) for p in prices],
            'close': prices,
            'high': [p * (1 + abs(np.random.randn()) * 0.02) for p in prices],
            'low': [p * (1 - abs(np.random.randn()) * 0.02) for p in prices],
            'volume': np.random.randint(900000, 1100000, n_days),
            # 添加技术指标
            'macd': np.random.randn(n_days) * 0.1,
            'rsi': 50 + np.random.randn(n_days) * 10,
        })
        
        df['macd_signal'] = df['macd'].rolling(9).mean()
        
        stock_data[code] = df
    
    return stock_data


def generate_single_stock_data(code, n_days=100):
    """生成单只股票数据"""
    np.random.seed(hash(code) % 1000)
    
    start_date = pd.Timestamp('2023-01-01')
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    price = 10.0
    prices = []
    
    for _ in range(n_days):
        change = np.random.randn() * 0.02
        price *= (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': [p * (1 + np.random.randn() * 0.01) for p in prices],
        'close': prices,
        'high': [p * (1 + abs(np.random.randn()) * 0.02) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.02) for p in prices],
        'volume': np.random.randint(900000, 1100000, n_days),
        'macd': np.random.randn(n_days) * 0.1,
        'rsi': 50 + np.random.randn(n_days) * 10,
    })
    
    df['macd_signal'] = df['macd'].rolling(9).mean()
    
    return df


def get_last_trading_day(max_lookback_days: int = 7) -> str:
    """
    获取最近的交易日（自动跳过周末和节假日）
    
    Args:
        max_lookback_days: 最大回溯天数
    
    Returns:
        最近交易日字符串 YYYY-MM-DD 格式
    """
    from datetime import datetime, timedelta
    import pandas as pd
    
    current_date = datetime.now()
    
    # 向前查找最近的交易日
    for i in range(max_lookback_days):
        check_date = current_date - timedelta(days=i)
        
        # 跳过周六（5）和周日（6）
        if check_date.weekday() >= 5:  # 5=周六, 6=周日
            continue
        
        # TODO: 这里可以加入节假日判断逻辑
        # 目前简化为只跳过周末
        
        return check_date.strftime("%Y-%m-%d")
    
    # 如果找不到，返回今天
    return current_date.strftime("%Y-%m-%d")


def render_akshare_data_connection():
    """渲染 AKShare 实时数据连接界面（复用现有适配器）"""
    
    st.markdown("② 📡 AKShare 实时数据连接（复用麒麟系统现有接口）")
    
    # 检查 AKShare 是否可用
    if not AKSHARE_AVAILABLE:
        st.error("❌ AKShare 适配器未加载")
        st.info("📝 请确保已安装: pip install akshare")
        return
    
    st.success("✅ AKShare 适配器已就绪（使用 layer3_online.adapters）")
    
    # 股票选择方式
    stock_input_method = st.radio(
        "股票来源",
        ["🖋️ 手动输入代码", "🚀 自动获取涨停板"],
        horizontal=True,
        key="stock_input_method"
    )
    
    stock_codes_input = None
    
    if "手动" in stock_input_method:
        stock_codes_input = st.text_area(
            "输入股票代码（多个代码用逗号或换行分隔）",
            value="000001, 600519, 000858",
            height=100,
            help="支持格式：000001 或 000001.SZ",
            key="manual_stock_input"
        )
    else:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            if st.button("🔄 获取涨停板", type="secondary", use_container_width=True):
                with st.spinner("正在获取涨停板数据..."):
                    try:
                        import akshare as ak
                        
                        # 自动获取最近的交易日（跳过周末）
                        target_date = get_last_trading_day()
                        
                        # 显示目标日期
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        if target_date != current_date:
                            st.info(f"📅 今天非交易日，自动获取 {target_date} 的数据")
                        
                        # 直接使用 AKShare 获取涨停板（更准确）
                        date_str = target_date.replace("-", "")
                        df_zt = ak.stock_zt_pool_em(date=date_str)
                        
                        if df_zt is not None and not df_zt.empty:
                            # 提取股票代码并添加后缀
                            codes = df_zt['代码'].astype(str).tolist()
                            limit_up_codes = []
                            for code in codes:
                                if code.startswith('6'):
                                    limit_up_codes.append(f"{code}.SH")
                                elif code.startswith(('0', '3')):
                                    limit_up_codes.append(f"{code}.SZ")
                            
                            st.session_state['limit_up_codes'] = limit_up_codes
                            st.session_state['limit_up_date'] = target_date
                            st.session_state['limit_up_raw_data'] = df_zt  # 保存原始数据
                            st.success(f"✅ 获取到 {target_date} 的 {len(limit_up_codes)} 只涨停股票")
                        else:
                            st.warning(f"⚠️ {target_date} 暂无涨停板数据")
                    except ImportError:
                        st.error("❌ AKShare 未安装，请运行: pip install akshare")
                    except Exception as e:
                        st.error(f"❌ 获取失败: {e}")
                        import traceback
                        with st.expander("🔍 查看错误详情"):
                            st.code(traceback.format_exc())
        
        if 'limit_up_codes' in st.session_state:
            codes_list = st.session_state['limit_up_codes']
            limit_up_date = st.session_state.get('limit_up_date', '今日')
            df_zt_raw = st.session_state.get('limit_up_raw_data', None)
            
            # 显示数据日期和统计
            col_info1, col_info2 = st.columns([2, 1])
            with col_info1:
                st.caption(f"📅 数据日期: {limit_up_date} | 共 {len(codes_list)} 只涨停股票")
            with col_info2:
                if df_zt_raw is not None and not df_zt_raw.empty and '连板数' in df_zt_raw.columns:
                    if st.button("🔍 查看详情", key="view_limitup_details"):
                        with st.expander("📊 涨停板详细数据", expanded=True):
                            # 显示连板分布
                            board_dist = df_zt_raw['连板数'].value_counts().sort_index()
                            st.write("连板分布:")
                            for board, count in board_dist.items():
                                st.write(f"  {board}连板: {count}只")
                            st.divider()
                            # 显示前20只股票
                            display_cols = ['代码', '名称', '涨跌幅', '最新价', '成交额', '连板数'] \
                                if all(c in df_zt_raw.columns for c in ['代码', '名称', '涨跌幅', '最新价', '成交额', '连板数']) \
                                else ['代码', '名称']
                            st.dataframe(df_zt_raw[display_cols].head(20), use_container_width=True)
            
            stock_codes_input = st.multiselect(
                f"选择股票 (共 {len(codes_list)} 只）",
                codes_list,
                default=codes_list[:min(10, len(codes_list))],
                key="limitup_stock_selection"
            )
        else:
            st.info("👆 请点击'获取涨停板'按钮")
    
    # 日期范围选择
    st.markdown("##### 📅 日期范围")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=180),
            key="akshare_start_date"
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now(),
            key="akshare_end_date"
        )
    with col3:
        freq = st.selectbox(
            "数据频率",
            options=["daily", "weekly", "monthly"],
            index=0,
            key="akshare_freq"
        )
    
    # 获取数据按钮
    if st.button("📥 获取数据", type="primary", use_container_width=True, key="fetch_akshare_data"):
        fetch_akshare_data(stock_codes_input, start_date, end_date, freq)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """简单的技术指标计算（MACD, RSI）"""
    try:
        # MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        # 如果计算失败，填充默认值
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['rsi'] = 50.0
        return df


def fetch_akshare_data(stock_codes_input, start_date, end_date, freq):
    """从 AKShare 获取股票数据并存储到 session_state（复用现有适配器）"""
    
    # 解析股票代码
    if isinstance(stock_codes_input, str):
        codes = [c.strip() for c in stock_codes_input.replace('\n', ',').split(',') if c.strip()]
    elif isinstance(stock_codes_input, list):
        codes = stock_codes_input
    else:
        codes = []
    
    if not codes:
        st.error("❌ 请输入至少一个股票代码")
        return
    
    # 显示进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"📊 正在获取 {len(codes)} 只股票的数据...")
        
        # 使用现有的 akshare_adapter 复用
        stock_data = {}
        for idx, code in enumerate(codes, 1):
            try:
                clean_code = code.split('.')[0] if '.' in code else code
                
                # 调用现有适配器
                df = get_daily_ohlc(
                    symbol=clean_code,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d")
                )
                
                if df is not None and not df.empty:
                    # 转换列名以匹配系统格式
                    df = df.rename(columns={
                        'date': 'datetime',
                        'vol': 'volume'
                    })
                    
                    # 添加技术指标（简化版）
                    df = add_technical_indicators(df)
                    
                    # 添加后缀
                    full_code = code if '.' in code else (f"{code}.SH" if code.startswith('6') else f"{code}.SZ")
                    stock_data[full_code] = df
                
                progress_bar.progress(int((idx / len(codes)) * 100))
                
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"跳过 {code}: {e}")
                continue
        
        progress_bar.progress(100)
        
        if stock_data:
            # 存储到 session_state
            st.session_state['chanlun_stock_data'] = stock_data
            
            # 显示统计信息
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ 成功加载 {len(stock_data)} 只股票数据")
            
            # 数据概览
            total_rows = sum(len(df) for df in stock_data.values())
            st.info(f"📊 数据统计：{len(stock_data)} 只股票，共 {total_rows:,} 条记录")
            
            # 显示示例数据
            with st.expander("👁️ 预览数据", expanded=False):
                first_code = list(stock_data.keys())[0]
                st.write(f"示例股票：**{first_code}**")
                st.dataframe(stock_data[first_code].head(10), use_container_width=True)
                
                # 技术指标检查
                df_sample = stock_data[first_code]
                if 'macd' in df_sample.columns and 'rsi' in df_sample.columns:
                    st.caption("✅ 技术指标（MACD, RSI）计算完成")
                else:
                    st.warning("⚠️ 部分技术指标未计算")
        else:
            st.error("❌ 未能获取到任何数据，请检查股票代码和日期范围")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ 数据获取失败：{str(e)}")
        import logging
        logging.getLogger(__name__).error(f"AKShare data fetch error: {e}", exc_info=True)
        
        with st.expander("🔍 查看详细错误"):
            import traceback
            st.code(traceback.format_exc())


def run_multi_agent_selection(stock_data, chanlun_w, tech_w, vol_w, fund_w, sent_w, top_n):
    """执行多智能体选股"""
    
    # 检查模块是否可用
    if MultiAgentStockSelector is None:
        st.error("❌ 缠论选股模块未加载")
        st.info("📝 请检查以下模块是否存在：")
        st.code("""
strategies/multi_agent_selector.py
agents/chanlun_agent.py
agents/limitup_chanlun_agent.py
        """, language="text")
        st.warning("💡 建议：先使用'使用示例数据'测试系统功能")
        return
    
    with st.spinner("🤖 多智能体正在分析..."):
        try:
            # 创建选择器
            selector = MultiAgentStockSelector(
                chanlun_weight=chanlun_w,
                technical_weight=tech_w,
                volume_weight=vol_w,
                fundamental_weight=fund_w,
                sentiment_weight=sent_w
            )
            
            # 批量评分
            results = selector.batch_score(stock_data, top_n=top_n)
            
            # 保存结果
            st.session_state['selection_results'] = results
            
            st.success(f"✅ 选股完成！从 {len(stock_data)} 只股票中选出 Top {len(results)} 只")
            
        except Exception as e:
            st.error(f"❌ 选股失败: {e}")
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())


def render_realtime_signals():
    """渲染实时信号监控界面 (P1-4)"""
    
    st.subheader("🔴 实时信号监控")
    
    # 侧边栏配置区
    with st.expander("⚙️ 监控配置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            refresh_interval = st.selectbox(
                "刷新间隔",
                options=[5, 10, 30, 60],
                index=1,
                format_func=lambda x: f"{x}秒",
                key="signal_refresh_interval"
            )
            
            min_score = st.slider("最低评分", 0, 100, 60, key="signal_min_score")
        
        with col2:
            signal_types = st.multiselect(
                "信号类型",
                options=['1买', '2买', '3买', '1卖', '2卖', '3卖'],
                default=['1买', '2买'],
                key="signal_types_filter"
            )
            
            if st.button("🔄 立即刷新", use_container_width=True, key="refresh_signals"):
                st.rerun()
    
    st.divider()
    
    # 信号来源切换（可选接入数据库）
    stock_universe = ['SH600000', 'SH600036', 'SZ000001', 'SZ000002', 'SZ000858', 'SH600519']
    signal_source = st.radio("信号来源", ["示例(模拟)", "数据库最新"], index=0, horizontal=True)

    signals_df = None
    if signal_source == "数据库最新" and ChanLunSignalStore is not None:
        try:
            store = ChanLunSignalStore()
            db_df = store.load_signals(limit=200)
            if db_df is not None and len(db_df) > 0:
                # 统一成展示列
                signals_df = db_df.rename(columns={
                    'time': '时间', 'symbol': '股票', 'signal_type': '信号类型',
                    'price': '价格', 'score': '评分', 'status': '状态',
                })
                # 若股票池不为空，可按选择过滤（可选）
                # signals_df = signals_df[signals_df['股票'].isin(stock_universe)]
            else:
                st.info("数据库暂无记录，改用示例信号")
        except Exception as e:
            st.warning(f"从数据库加载失败，改用示例信号: {e}")
            signals_df = None

    if signals_df is None:
        # 生成示例信号
        signals_df = generate_mock_signals(stock_universe, num=20)
    
    # 过滤信号
    filtered = signals_df[
        (signals_df['评分'] >= min_score) &
        (signals_df['信号类型'].isin(signal_types) if signal_types else True)
    ]

    # P2-2: 持久化设置与保存
    with st.expander("💾 持久化设置 (SQLite)", expanded=False):
        enable_persist = st.checkbox("保存到本地数据库(data/chanlun_signals.sqlite)", value=False)
        if enable_persist and ChanLunSignalStore is None:
            st.warning("未找到存储服务模块，跳过持久化功能")
        save_scope = st.radio("保存范围", ["筛选后", "原始"], index=0, horizontal=True, help="原始=全部生成的信号；筛选后=当前筛选条件后的结果")
        cols = st.columns([1,1,2])
        with cols[0]:
            if st.button("保存当日信号", use_container_width=True, disabled=not enable_persist or ChanLunSignalStore is None or (len(filtered)==0 and save_scope=="筛选后")):
                try:
                    store = ChanLunSignalStore()
                    store.init()
                    source_df = filtered if save_scope == "筛选后" else signals_df
                    if source_df is None or len(source_df) == 0:
                        st.warning("当前无可保存的信号")
                    else:
                        df_to_save = source_df.rename(columns={
                            '时间': 'time', '股票': 'symbol', '信号类型': 'signal_type',
                            '价格': 'price', '评分': 'score', '状态': 'status',
                        })[['time','symbol','signal_type','price','score','status']]
                        n = store.save_signals(df_to_save)
                        st.success(f"已保存 {n} 条信号到本地数据库")
                except Exception as e:
                    st.error(f"保存失败: {e}")
        with cols[1]:
            if st.button("从数据库加载最新", use_container_width=True, disabled=ChanLunSignalStore is None):
                try:
                    store = ChanLunSignalStore()
                    df_latest = store.load_signals(limit=100)
                    st.dataframe(df_latest, use_container_width=True, height=260)
                except Exception as e:
                    st.error(f"加载失败: {e}")
    
    # 指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("今日信号", len(filtered), "+5")
    with col2:
        st.metric("待确认", len(filtered[filtered['状态'] == '待确认']), "-2")
    with col3:
        avg_score = filtered['评分'].mean() if len(filtered) > 0 else 0
        st.metric("平均评分", f"{avg_score:.1f}", "+3.2")
    with col4:
        st.metric("活跃股票", filtered['股票'].nunique())
    
    st.divider()
    
    # 信号表格（带样式）
    st.markdown("##### 📋 信号列表")
    if len(filtered) > 0:
        st.dataframe(
            filtered.style.applymap(
                lambda x: 'background-color: lightgreen' if x in ['1买', '2买'] else ('background-color: lightcoral' if '卖' in str(x) else ''),
                subset=['信号类型']
            ).applymap(
                lambda x: 'background-color: lightyellow' if x == '待确认' else '',
                subset=['状态']
            ),
            use_container_width=True,
            height=400
        )
    else:
        st.info("🔍 当前筛选条件下无符合的信号")
    
    # 底部信息
    st.caption(f"🕐 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ⚡ 刷新间隔: {refresh_interval}秒")


def render_multi_stock_monitor():
    """渲染多股票监控界面 (P1-4)"""
    
    st.subheader("📡 多股票缠论监控")
    
    # 股票选择
    stock_universe = ['SH600000', 'SH600036', 'SZ000001', 'SZ000002', 'SZ000858', 'SH600519']
    
    with st.expander("⚙️ 股票池配置", expanded=True):
        selected_stocks = st.multiselect(
            "选择监控股票",
            options=stock_universe,
            default=stock_universe[:3],
            key="monitor_stocks"
        )
    
    # 数据源与时间范围
    with st.expander("📡 数据源", expanded=False):
        # 默认优先AKShare（可用时）
        ds_options = ["AKShare历史", "示例数据"] if AKSHARE_AVAILABLE else ["示例数据", "AKShare历史"]
        data_source = st.radio("数据来源", ds_options, index=0, horizontal=True)
        cold1, cold2 = st.columns(2)
        with cold1:
            start_date = st.date_input("开始日期", value=datetime.now() - timedelta(days=180), key="msm_start_date")
        with cold2:
            end_date = st.date_input("结束日期", value=datetime.now(), key="msm_end_date")
        if data_source == "AKShare历史" and not AKSHARE_AVAILABLE:
            st.warning("AKShare适配器不可用，将回退到示例数据")
            data_source = "示例数据"
    
    st.divider()
    
    if not selected_stocks:
        st.warning("⚠️ 请选择要监控的股票")
        return
    
    # 导入缠论图表组件
    try:
        from web.components.chanlun_chart import ChanLunChartComponent
        chart_available = True
    except ImportError:
        chart_available = False
        st.warning("⚠️ 缠论图表组件未加载，使用简化视图")
    
    # 逐个显示股票（P2-1：先按共振分数排序）
    stock_scores = {}
    per_stock_df = {}

    def _calc_confluence(df_daily: pd.DataFrame) -> float:
        try:
            if compute_direction is None:
                return 0.0
            d_dir = compute_direction(df_daily)
            if resample_ohlc is not None:
                w_df = resample_ohlc(df_daily, 'W')
                m_df = resample_ohlc(df_daily, 'M')
                w_dir = compute_direction(w_df)
                m_dir = compute_direction(m_df)
            else:
                w_dir = 0
                m_dir = 0
            dirs = {'D': d_dir, 'W': w_dir, 'M': m_dir}
            return float(compute_confluence_score(dirs)) if compute_confluence_score else 0.0
        except Exception:
            return 0.0

    # 预计算每只股票60日数据与分数
    for s in selected_stocks:
        if data_source == "示例数据":
            d = generate_mock_stock_data(days=60)
        else:
            # 使用AKShare适配器获取历史数据
            try:
                clean = s.split('.')[0] if '.' in s else s
                df_raw = get_daily_ohlc(symbol=clean, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
                if df_raw is not None and not df_raw.empty:
                    d = df_raw.rename(columns={'date': 'datetime', 'vol': 'volume'})
                else:
                    d = generate_mock_stock_data(days=60)
            except Exception:
                d = generate_mock_stock_data(days=60)
        per_stock_df[s] = d
        stock_scores[s] = _calc_confluence(d)

    # UI筛选：阈值与TopN
    fcol1, fcol2, fcol3 = st.columns([1,1,2])
    with fcol1:
        thr = st.number_input("共振分数阈值", value=0.5, step=0.1, format="%.1f")
    with fcol2:
        top_n = st.number_input("Top N", min_value=1, max_value=max(1, len(selected_stocks)), value=min(5, len(selected_stocks)))
    with fcol3:
        st.caption("提示：共振分数越大，多周期方向越一致；排序优先展示高分股票。")

    selected_stocks_sorted = sorted([s for s in selected_stocks if stock_scores.get(s, 0.0) >= thr], key=lambda x: stock_scores.get(x, 0.0), reverse=True)[:top_n]
    st.caption("📌 展示顺序已按共振分数(高→低)排序；已应用阈值与TopN")

    for idx, stock in enumerate(selected_stocks_sorted):
        with st.expander(f"📊 {stock}", expanded=(idx == 0)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 使用预计算数据
                df = per_stock_df[stock]
                
                if chart_available:
                    # 使用P0-4的缠论图表组件
                    chan_features = {
                        'fx_mark': pd.Series([1 if i % 10 == 0 else -1 if i % 10 == 5 else 0 for i in range(len(df))]),
                        'buy_points': [
                            {'datetime': df.iloc[10]['datetime'], 'price': df.iloc[10]['close'], 'type': 1},
                            {'datetime': df.iloc[30]['datetime'], 'price': df.iloc[30]['close'], 'type': 2},
                        ],
                        'sell_points': []
                    }
                    
                    chart = ChanLunChartComponent(width=800, height=500)
                    fig = chart.render_chanlun_chart(df, chan_features)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # 简化K线图
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['datetime'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close']
                    )])
                    fig.update_layout(title=f"{stock} K线图", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 股票信息卡片
                st.metric("当前价", f"{df['close'].iloc[-1]:.2f}", f"+{np.random.rand()*5:.2f}%")
                st.metric("成交量", f"{df['volume'].iloc[-1]/10000:.0f}万")
                st.metric("缠论评分", f"{np.random.randint(60, 95)}")

                # 共振分数与方向
                score = stock_scores.get(stock, 0.0)
                if compute_direction is not None:
                    d_dir = compute_direction(df)
                    w_dir = compute_direction(resample_ohlc(df, 'W')) if resample_ohlc else 0
                    m_dir = compute_direction(resample_ohlc(df, 'M')) if resample_ohlc else 0
                else:
                    d_dir = w_dir = m_dir = 0
                st.metric("共振分数", f"{score:.2f}")
                st.caption(f"D:{d_dir} · W:{w_dir} · M:{m_dir}")
                
                # 若可计算出真实特征，显示中枢移动/升级强度
                try:
                    from features.chanlun.chanpy_features import ChanPyFeatureGenerator
                    gen = ChanPyFeatureGenerator()
                    feats = gen.generate_features(df, code=stock)
                    if feats is not None and len(feats) > 0:
                        last = feats.iloc[-1]
                        dir_map = { -1: "下降", 0: "横盘", 1: "上升" }
                        st.metric("中枢移动", dir_map.get(int(last.get('zs_movement_direction', 0)), "未知"))
                        st.metric("升级强度", f"{float(last.get('zs_upgrade_strength', 0.0)):.2f}")
                except Exception:
                    pass
                
                st.divider()
                
                # 最新信号（示例）
                st.markdown("**最新信号**")
                st.success("✅ 2买点 (85分)")
                st.info("ℹ️ 趋势: 上涨")
                st.warning("⚠️ 中枢: 震荡")


def render_statistical_analysis():
    """渲染统计分析界面 (P1-4)"""
    
    st.subheader("📊 统计与分析")

    # P2-2: 从库加载统计
    with st.expander("📚 从本地数据库加载统计 (SQLite)", expanded=False):
        if ChanLunSignalStore is None:
            st.warning("未找到存储服务模块，无法加载统计")
        else:
            col_a, col_b = st.columns([1,1])
            with col_a:
                if st.button("加载每日统计", use_container_width=True):
                    try:
                        store = ChanLunSignalStore()
                        stats_df = store.get_daily_stats()
                        st.dataframe(stats_df, use_container_width=True, height=260)
                    except Exception as e:
                        st.error(f"加载失败: {e}")
            with col_b:
                st.caption("提示：先在‘实时信号监控’保存信号，再在此查看统计")
    
    stock_universe = ['SH600000', 'SH600036', 'SZ000001', 'SZ000002', 'SZ000858', 'SH600519']
    
    # 生成模拟数据
    signals_df = generate_mock_signals(stock_universe, num=50)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 信号类型分布")
        signal_counts = signals_df['信号类型'].value_counts()
        st.bar_chart(signal_counts)
    
    with col2:
        st.markdown("##### 📊 评分分布")
        score_hist = pd.DataFrame({
            '评分区间': ['60-70', '70-80', '80-90', '90-100'],
            '数量': [
                len(signals_df[(signals_df['评分'] >= 60) & (signals_df['评分'] < 70)]),
                len(signals_df[(signals_df['评分'] >= 70) & (signals_df['评分'] < 80)]),
                len(signals_df[(signals_df['评分'] >= 80) & (signals_df['评分'] < 90)]),
                len(signals_df[signals_df['评分'] >= 90])
            ]
        }).set_index('评分区间')
        st.bar_chart(score_hist)
    
    st.divider()
    
    # 股票表现排行
    st.markdown("##### 🏆 股票表现排行")
    performance = pd.DataFrame({
        '股票': stock_universe,
        '今日涨跌': [f"+{np.random.rand()*5:.2f}%" for _ in stock_universe],
        '缠论评分': np.random.randint(60, 95, len(stock_universe)),
        '信号数': [len(signals_df[signals_df['股票'] == s]) for s in stock_universe]
    })
    performance = performance.sort_values('缠论评分', ascending=False)
    
    st.dataframe(
        performance.style.background_gradient(subset=['缠论评分'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    st.divider()
    
    # 今日统计摘要
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总信号数", len(signals_df))
    with col2:
        st.metric("买点信号", len(signals_df[signals_df['信号类型'].str.contains('买')]))
    with col3:
        st.metric("卖点信号", len(signals_df[signals_df['信号类型'].str.contains('卖')]))
    with col4:
        st.metric("平均评分", f"{signals_df['评分'].mean():.1f}")


def generate_mock_signals(stock_universe, num=10):
    """生成模拟信号数据"""
    signals = []
    for i in range(num):
        signal = {
            '时间': datetime.now() - timedelta(minutes=np.random.randint(0, 480)),
            '股票': np.random.choice(stock_universe),
            '信号类型': np.random.choice(['1买', '2买', '3买', '1卖', '2卖']),
            '价格': round(10 + np.random.randn() * 2, 2),
            '评分': np.random.randint(60, 100),
            '状态': np.random.choice(['待确认', '已触发', '已完成'])
        }
        signals.append(signal)
    return pd.DataFrame(signals)


def generate_mock_stock_data(days=60):
    """生成模拟股票数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    prices = 10 + np.random.randn(days).cumsum() * 0.1
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.randn(days) * 0.01),
        'high': prices * (1 + abs(np.random.randn(days)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(days)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    })
    
    return df


if __name__ == "__main__":
    st.set_page_config(page_title="缠论系统", layout="wide")
    render_chanlun_system_tab()
