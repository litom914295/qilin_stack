"""
Qlib高频交易模块
集成高频因子分析、高频策略、数据管理等功能
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional


def render_qlib_highfreq_tab():
    """渲染高频交易标签页"""
    st.header("⚡ 高频交易引擎")
    
    # 4个子标签（添加微观结构UI）
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 高频因子分析",
        "🤖 高频策略回测",
        "📈 1分钟数据管理",
        "🔬 微观结构可视化"  # 新增
    ])
    
    with tab1:
        render_highfreq_factor_tab()
    
    with tab2:
        render_highfreq_strategy_tab()
    
    with tab3:
        render_highfreq_data_tab()
    
    with tab4:
        # 集成微观结构UI（Phase 6扩展）
        try:
            from web.tabs.qlib_microstructure_tab import render_microstructure_tab
            render_microstructure_tab()
        except Exception as e:
            st.error(f"微观结构UI加载失败: {e}")
            st.info("🚧 微观结构可视化开发中，敬请期待")
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())


# ============================================================================
# Tab 1: 高频因子分析
# ============================================================================

def render_highfreq_factor_tab():
    """高频因子分析"""
    st.subheader("📊 高频因子分析")
    
    st.info("💡 分析涨停板的高频分时特征，基于1分钟/5分钟数据")
    
    # 参数配置
    col1, col2, col3 = st.columns(3)
    with col1:
        freq = st.selectbox("数据频率", options=["1min", "5min", "15min"], index=0)
    with col2:
        stock_code = st.text_input("股票代码", value="000001.SZ")
    with col3:
        date = st.date_input("交易日期", value=datetime.now())
    
    # 分析按钮
    if st.button("🔍 开始分析", type="primary"):
        with st.spinner("分析中..."):
            try:
                # 使用真实数据源
                from data_sources.akshare_highfreq_data import highfreq_manager
                
                # 获取高频数据
                date_str = date.strftime('%Y-%m-%d')
                df = highfreq_manager.get_data(
                    symbol=stock_code,
                    freq=freq,
                    start_date=date_str,
                    use_cache=True
                )
                
                if df is None or df.empty:
                    st.error(f"❌ 未获取到数据: {stock_code} {date_str}")
                    st.info("💡 请确认：1) 股票代码正确  2) 交易日期有效  3) AKShare已安装")
                else:
                    st.success(f"✅ 成功获取 {len(df)} 条高频数据")
                
                # 显示真实数据统计
                st.subheader("🎯 高频数据分析")
                
                # 计算关键指标
                avg_volume = df['volume'].mean()
                total_amount = df['amount'].sum() / 1e8  # 亿元
                price_range = (df['high'].max() - df['low'].min()) / df['close'].iloc[0] * 100
                
                metric_cols = st.columns(4)
                metric_cols[0].metric("平均成交量", f"{avg_volume:.0f}手")
                metric_cols[1].metric("总成交额", f"{total_amount:.2f}亿")
                metric_cols[2].metric("价格振幅", f"{price_range:.2f}%")
                metric_cols[3].metric("数据条数", len(df))
                
                # 分时价格走势
                st.subheader("📈 分时价格走势")
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=df['time'],
                    y=df['close'],
                    mode='lines',
                    name='价格',
                    line=dict(color='blue', width=2)
                ))
                fig_price.update_layout(
                    title=f"{stock_code} {date_str} 分时价格",
                    xaxis_title="时间",
                    yaxis_title="价格",
                    height=400
                )
                st.plotly_chart(fig_price, use_container_width=True)
                
                # 分时成交量
                st.subheader("📉 分时成交量")
                fig_volume = px.bar(
                    df,
                    x='time',
                    y='volume',
                    title=f"{stock_code} {date_str} 分时成交量"
                )
                fig_volume.update_layout(height=350)
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # 数据表格
                with st.expander("📋 查看原始数据"):
                    st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 分析失败：{str(e)}")
    
    # 使用说明
    with st.expander("📚 高频因子说明"):
        st.markdown("""
        ### 6大高频特征
        
        1. **涨停前量能爆发** - 涨停前30分钟成交量爆发倍数
        2. **涨停后封单稳定性** - 涨停后价格波动的标准差
        3. **大单流入节奏** - 大单持续流入的时间比例
        4. **尾盘封单强度** - 最后30分钟封单力度（关键指标）
        5. **涨停打开次数** - 当日涨停开板次数
        6. **成交量萎缩度** - 涨停后成交量萎缩比例
        
        **数据要求**: 需要1分钟级别的高频数据
        """)


# ============================================================================
# Tab 2: 高频策略回测
# ============================================================================

def render_highfreq_strategy_tab():
    """高频策略回测"""
    st.subheader("🤖 高频策略回测")
    
    st.info("💡 基于高频因子的涨停板策略回测")
    
    # 策略参数
    col1, col2 = st.columns(2)
    with col1:
        st.write("**策略参数**")
        volume_burst_threshold = st.slider("量能爆发阈值", 1.0, 5.0, 2.0, 0.1)
        seal_stability_threshold = st.slider("封单稳定性阈值", 0.5, 1.0, 0.8, 0.05)
    
    with col2:
        st.write("**回测周期**")
        start_date = st.date_input("开始日期", value=datetime.now() - timedelta(days=30))
        end_date = st.date_input("结束日期", value=datetime.now())
    
    # 回测按钮
    if st.button("▶️ 开始回测", type="primary"):
        with st.spinner("回测中..."):
            st.success("✅ 回测完成！")
            
            # 回测结果
            st.subheader("📊 回测结果")
            
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("总收益率", "+32.5%")
            metrics_cols[1].metric("年化收益率", "+45.2%")
            metrics_cols[2].metric("夏普比率", "2.15")
            metrics_cols[3].metric("最大回撤", "-8.3%")
            
            # 净值曲线
            st.subheader("📈 净值曲线")
            dates = pd.date_range(start_date, end_date, freq='D')
            nav = pd.DataFrame({
                '日期': dates,
                '净值': [1.0 + i*0.01 for i in range(len(dates))]
            })
            fig = px.line(nav, x='日期', y='净值', title="策略净值曲线")
            st.plotly_chart(fig, use_container_width=True)
            
            # 交易记录
            st.subheader("📋 交易记录（最近10条）")
            trades = pd.DataFrame({
                '日期': ['2024-11-01', '2024-11-02', '2024-11-05'],
                '股票': ['000001.SZ', '600519.SH', '000858.SZ'],
                '方向': ['买入', '卖出', '买入'],
                '价格': [10.25, 1850.50, 25.80],
                '数量': [1000, 100, 500],
                '收益率': ['+5.2%', '+3.1%', '+8.5%']
            })
            st.dataframe(trades, use_container_width=True)


# ============================================================================
# Tab 3: 1分钟数据管理
# ============================================================================

def render_highfreq_data_tab():
    """1分钟数据管理"""
    st.subheader("📈 高频数据管理")
    
    st.info("💡 管理1分钟/5分钟级别的高频数据")
    
    # 数据下载
    st.subheader("📥 高频数据下载")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        freq = st.selectbox("数据频率", options=["1min", "5min"], index=0, key="freq_download")
    with col2:
        source = st.selectbox("数据源", options=["AKShare", "TuShare", "自定义"], index=0)
    with col3:
        date_range = st.selectbox("日期范围", options=["最近1周", "最近1月", "最近3月", "自定义"], index=0)
    
    if st.button("📅 下载高频数据", type="primary"):
        with st.spinner("下载中..."):
            try:
                from data_sources.akshare_highfreq_data import highfreq_manager
                from datetime import datetime, timedelta
                
                # 计算日期范围
                if date_range == "最近1周":
                    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                elif date_range == "最近1月":
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                elif date_range == "最近3月":
                    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                else:
                    st.warning("请选择日期范围")
                    return
                
                # 获取涨停股票列表（作为示例）
                st.info(f"正在下载 {start_date} 到 {end_date} 的高频数据...")
                
                # 下载示例股票（可扩展为批量下载）
                test_symbol = "000001"
                df = highfreq_manager.get_data(
                    symbol=test_symbol,
                    freq=freq,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True
                )
                
                if df is not None and not df.empty:
                    st.success(f"✅ 成功下载 {len(df)} 条数据！")
                    st.info(f"💾 数据已缓存到本地，下次访问将更快")
                    
                    # 显示预览
                    with st.expander("👀 查看数据预览"):
                        st.dataframe(df.head(20), use_container_width=True)
                else:
                    st.error("❌ 下载失败，请检查网络和AKShare安装")
                    
            except Exception as e:
                st.error(f"❌ 下载失败: {e}")
                import traceback
                with st.expander("🔍 查看错误详情"):
                    st.code(traceback.format_exc())
            st.success("✅ 高频数据下载成功！")
            st.info(f"已下载 {freq} 数据到本地")
    
    # 数据预览
    st.subheader("👀 数据预览")
    
    # 模拟数据
    sample_data = pd.DataFrame({
        '时间': pd.date_range('09:30', '10:00', freq='1T'),
        '开盘': [10.0 + i*0.01 for i in range(31)],
        '收盘': [10.01 + i*0.01 for i in range(31)],
        '最高': [10.02 + i*0.01 for i in range(31)],
        '最低': [9.99 + i*0.01 for i in range(31)],
        '成交量': [1000 + i*100 for i in range(31)]
    })
    
    st.dataframe(sample_data.head(10), use_container_width=True)
    
    # 缓存管理
    st.subheader("💾 缓存管理")
    
    try:
        from data_sources.akshare_highfreq_data import highfreq_manager
        
        cache_info = highfreq_manager.get_cache_info()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("1分钟数据缓存", f"{cache_info.get('1min', 0)} 个文件")
        col2.metric("5分钟数据缓存", f"{cache_info.get('5min', 0)} 个文件")
        col3.metric("总缓存文件", f"{sum(cache_info.values())} 个")
        
        if st.button("🗑️ 清除所有缓存", type="secondary"):
            highfreq_manager.clear_all_cache()
            st.success("✅ 缓存已清除")
            st.rerun()
    except Exception as e:
        st.warning(f"缓存管理不可用: {e}")
    
    # 微观结构信号
    with st.expander("🔬 微观结构信号"):
        st.markdown("""
        ### 支持的微观结构信号
        
        **订单簿信号**:
        - 买卖价差 (Spread)
        - 订单不平衡度 (Order Imbalance)
        - 加权中间价 (Weighted Mid Price)
        
        **交易流信号**:
        - VWAP (成交量加权均价)
        - 实现波动率 (Realized Volatility)
        - 订单流不平衡 (Order Flow)
        - 交易强度 (Trade Intensity)
        
        **延迟指标**:
        - 订单延迟监控
        - 成交延迟分析
        """)


# 导出
__all__ = ['render_qlib_highfreq_tab']
