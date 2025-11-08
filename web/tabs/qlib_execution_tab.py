"""
Qlib订单执行引擎UI
Phase 5.2实现

功能：
1. 滑点模型配置和模拟
2. 涨停排队模拟
3. 市场冲击分析
4. 执行成本可视化
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback

# 导入执行引擎
try:
    import sys
    sys.path.append(".")
    from qilin_stack.backtest.slippage_model import (
        SlippageEngine, SlippageModel, OrderSide, MarketDepth
    )
    from qilin_stack.backtest.limit_up_queue_simulator import (
        LimitUpQueueSimulator, LimitUpStrength
    )
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    SlippageEngine = None
    LimitUpQueueSimulator = None


def render_qlib_execution_tab():
    """渲染Qlib订单执行引擎标签页"""
    st.title("🚀 订单执行引擎")
    st.markdown("模拟真实交易中的滑点、市场冲击和涨停排队")
    
    # 检查依赖
    if SlippageEngine is None or LimitUpQueueSimulator is None:
        st.error("❌ 执行引擎模块未加载，请检查依赖")
        return
    
    # 三个子标签
    tab1, tab2, tab3 = st.tabs([
        "💸 滑点与市场冲击",
        "📈 涨停排队模拟",
        "📊 执行成本分析"
    ])
    
    with tab1:
        render_slippage_simulator()
    
    with tab2:
        render_limitup_queue_simulator()
    
    with tab3:
        render_execution_cost_analysis()


# ==================== 滑点与市场冲击 ====================

def render_slippage_simulator():
    """渲染滑点模拟器"""
    st.header("💸 滑点与市场冲击模拟器")
    st.markdown("模拟订单执行时的价格偏离和市场影响")
    
    # 左右分栏
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📝 订单配置")
        
        # 基础信息
        symbol = st.text_input("股票代码", value="000001.SZ", key="slip_symbol")
        
        col1, col2 = st.columns(2)
        with col1:
            side = st.selectbox("订单方向", ["买入", "卖出"], key="slip_side")
            target_price = st.number_input("目标价格（元）", value=10.50, step=0.01, format="%.2f", key="slip_price")
        with col2:
            target_shares = st.number_input("目标股数", value=100000, step=1000, key="slip_shares")
            avg_daily_volume = st.number_input("日均成交量", value=3000000, step=100000, key="slip_volume")
        
        st.markdown("---")
        
        # 滑点模型选择
        st.markdown("### ⚙️ 滑点模型配置")
        
        model_type = st.selectbox(
            "滑点模型",
            ["固定滑点", "线性滑点", "平方根滑点", "流动性滑点"],
            index=3,
            key="slip_model"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            fixed_slippage_bps = st.slider("固定滑点（基点）", 1, 50, 5, key="slip_bps")
        with col2:
            impact_coefficient = st.slider("冲击系数", 0.01, 0.5, 0.1, step=0.01, format="%.2f", key="slip_impact")
        
        max_participation_rate = st.slider(
            "最大参与率（%）",
            1.0, 20.0, 5.0, step=0.5,
            help="订单量占日均量的最大比例",
            key="slip_participation"
        ) / 100
        
        st.markdown("---")
        
        # 市场深度配置（仅流动性模型）
        if model_type == "流动性滑点":
            st.markdown("### 📊 市场深度配置")
            
            with st.expander("💡 配置盘口深度（5档）", expanded=False):
                st.markdown("**卖盘（Ask）**")
                ask_prices = []
                ask_volumes = []
                for i in range(5):
                    col_p, col_v = st.columns(2)
                    with col_p:
                        price = st.number_input(
                            f"卖{i+1}价",
                            value=target_price + 0.01 * i,
                            step=0.01,
                            format="%.2f",
                            key=f"ask_price_{i}"
                        )
                        ask_prices.append(price)
                    with col_v:
                        volume = st.number_input(
                            f"卖{i+1}量",
                            value=int(50000 - i * 2000),
                            step=1000,
                            key=f"ask_volume_{i}"
                        )
                        ask_volumes.append(volume)
                
                st.markdown("**买盘（Bid）**")
                bid_prices = []
                bid_volumes = []
                for i in range(5):
                    col_p, col_v = st.columns(2)
                    with col_p:
                        price = st.number_input(
                            f"买{i+1}价",
                            value=target_price - 0.01 * (i + 1),
                            step=0.01,
                            format="%.2f",
                            key=f"bid_price_{i}"
                        )
                        bid_prices.append(price)
                    with col_v:
                        volume = st.number_input(
                            f"买{i+1}量",
                            value=int(48000 - i * 2000),
                            step=1000,
                            key=f"bid_volume_{i}"
                        )
                        bid_volumes.append(volume)
                
                liquidity_score = st.slider("流动性评分", 0, 100, 75, key="slip_liquidity")
        
        # 执行按钮
        st.markdown("---")
        if st.button("🚀 模拟订单执行", type="primary", use_container_width=True, key="slip_execute"):
            with col_right:
                execute_slippage_simulation(
                    symbol=symbol,
                    side=side,
                    target_shares=target_shares,
                    target_price=target_price,
                    avg_daily_volume=avg_daily_volume,
                    model_type=model_type,
                    fixed_slippage_bps=fixed_slippage_bps,
                    impact_coefficient=impact_coefficient,
                    max_participation_rate=max_participation_rate,
                    market_depth_data=(ask_prices, ask_volumes, bid_prices, bid_volumes, liquidity_score) if model_type == "流动性滑点" else None
                )
    
    with col_right:
        st.markdown("### 📈 执行结果")
        st.info("👈 配置订单参数后，点击\"模拟订单执行\"查看结果")


def execute_slippage_simulation(symbol, side, target_shares, target_price, avg_daily_volume,
                                model_type, fixed_slippage_bps, impact_coefficient, max_participation_rate,
                                market_depth_data=None):
    """执行滑点模拟"""
    try:
        # 映射模型类型
        model_map = {
            "固定滑点": SlippageModel.FIXED,
            "线性滑点": SlippageModel.LINEAR,
            "平方根滑点": SlippageModel.SQRT,
            "流动性滑点": SlippageModel.LIQUIDITY_BASED
        }
        
        # 创建引擎
        engine = SlippageEngine(
            model=model_map[model_type],
            fixed_slippage_bps=fixed_slippage_bps,
            impact_coefficient=impact_coefficient,
            max_participation_rate=max_participation_rate
        )
        
        # 准备市场深度（如果需要）
        market_depth = None
        liquidity_score_val = None
        if market_depth_data:
            ask_prices, ask_volumes, bid_prices, bid_volumes, liquidity_score_val = market_depth_data
            market_depth = MarketDepth(
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
                mid_price=(bid_prices[0] + ask_prices[0]) / 2 if bid_prices and ask_prices else target_price,
                spread=ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0.01,
                total_bid_volume=sum(bid_volumes),
                total_ask_volume=sum(ask_volumes),
                liquidity_score=liquidity_score_val
            )
        
        # 执行订单
        order_side = OrderSide.BUY if side == "买入" else OrderSide.SELL
        execution = engine.execute_order(
            symbol=symbol,
            side=order_side,
            target_shares=target_shares,
            target_price=target_price,
            market_depth=market_depth,
            avg_daily_volume=avg_daily_volume,
            liquidity_score=liquidity_score_val
        )
        
        # 显示结果
        st.success("✅ 模拟执行完成")
        
        # 主要指标
        st.markdown("#### 📊 成交结果")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("目标价格", f"¥{execution.target_price:.3f}")
        with col2:
            st.metric("实际成交价", f"¥{execution.avg_execution_price:.3f}")
        with col3:
            st.metric("成交股数", f"{execution.executed_shares:,}")
        with col4:
            st.metric("成交金额", f"¥{execution.total_cost:,.0f}")
        
        # 成本分析
        st.markdown("#### 💰 成本分析")
        cost_analysis = engine.calculate_total_slippage(execution)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("滑点", f"{execution.slippage_pct:.2%}", delta=f"¥{execution.slippage:.4f}/股")
        with col2:
            st.metric("市场冲击", f"¥{execution.market_impact:.4f}/股")
        with col3:
            st.metric("总成本", f"¥{cost_analysis['total_cost']:,.0f}")
        with col4:
            st.metric("成本基点", f"{cost_analysis['cost_bps']:.2f} bps")
        
        # 分笔成交
        if len(execution.fills) > 1:
            st.markdown("#### 📋 分笔成交明细")
            fills_df = pd.DataFrame([
                {"笔数": i+1, "成交股数": shares, "成交价格": f"¥{price:.3f}"}
                for i, (shares, price) in enumerate(execution.fills)
            ])
            st.dataframe(fills_df, use_container_width=True, hide_index=True)
        
        # 执行说明
        st.info(f"💡 {execution.execution_reason}")
        
        # 警告信息
        if execution.warnings:
            st.warning("⚠️ **警告**\n" + "\n".join([f"- {w}" for w in execution.warnings]))
        
        # 参与率分析
        participation_rate = target_shares / avg_daily_volume if avg_daily_volume > 0 else 0
        st.markdown("#### 📈 订单分析")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("参与率", f"{participation_rate:.2%}")
            st.progress(min(participation_rate, 1.0))
        with col2:
            fill_rate = execution.executed_shares / target_shares if target_shares > 0 else 0
            st.metric("成交率", f"{fill_rate:.2%}")
            st.progress(fill_rate)
        
    except Exception as e:
        st.error(f"❌ 执行失败: {str(e)}")
        with st.expander("🔍 查看详细错误"):
            st.code(traceback.format_exc())


# ==================== 涨停排队模拟 ====================

def render_limitup_queue_simulator():
    """渲染涨停排队模拟器"""
    st.header("📈 涨停排队模拟器")
    st.markdown("模拟涨停板封单排队和成交过程")
    
    # 左右分栏
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📝 涨停信息")
        
        # 基础信息
        symbol = st.text_input("股票代码", value="000001.SZ", key="lmt_symbol")
        limit_price = st.number_input("涨停价（元）", value=11.00, step=0.01, format="%.2f", key="lmt_price")
        
        col1, col2 = st.columns(2)
        with col1:
            seal_amount = st.number_input(
                "封单金额（万元）",
                value=5000,
                step=100,
                key="lmt_seal"
            ) * 10000  # 转换为元
        with col2:
            target_shares = st.number_input("目标股数", value=20000, step=1000, key="lmt_shares")
        
        st.markdown("---")
        
        # 时间配置
        st.markdown("### ⏰ 时间配置")
        
        col1, col2 = st.columns(2)
        with col1:
            seal_time = st.time_input("封板时间", value=datetime.strptime("09:35", "%H:%M").time(), key="lmt_seal_time")
        with col2:
            order_time = st.time_input("下单时间", value=datetime.strptime("09:40", "%H:%M").time(), key="lmt_order_time")
        
        open_times = st.slider("开板次数", 0, 5, 0, help="涨停后重新打开的次数", key="lmt_open")
        
        st.markdown("---")
        
        # 成交概率配置
        st.markdown("### ⚙️ 成交概率配置")
        
        with st.expander("💡 自定义各强度成交概率", expanded=False):
            one_word_prob = st.slider("一字板成交概率", 0, 100, 5, key="lmt_one") / 100
            early_seal_prob = st.slider("早盘封板成交概率", 0, 100, 20, key="lmt_early") / 100
            mid_seal_prob = st.slider("盘中封板成交概率", 0, 100, 50, key="lmt_mid") / 100
            late_seal_prob = st.slider("尾盘封板成交概率", 0, 100, 80, key="lmt_late") / 100
            weak_seal_prob = st.slider("弱封成交概率", 0, 100, 95, key="lmt_weak") / 100
        
        # 执行按钮
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📊 评估排队状态", use_container_width=True, key="lmt_evaluate"):
                with col_right:
                    evaluate_limitup_queue(
                        symbol=symbol,
                        limit_price=limit_price,
                        seal_amount=seal_amount,
                        seal_time=seal_time,
                        order_time=order_time,
                        target_shares=target_shares,
                        open_times=open_times,
                        probs=(one_word_prob, early_seal_prob, mid_seal_prob, late_seal_prob, weak_seal_prob)
                    )
        
        with col_btn2:
            if st.button("🎲 模拟成交（10次）", type="primary", use_container_width=True, key="lmt_simulate"):
                with col_right:
                    simulate_limitup_execution(
                        symbol=symbol,
                        limit_price=limit_price,
                        seal_amount=seal_amount,
                        seal_time=seal_time,
                        order_time=order_time,
                        target_shares=target_shares,
                        open_times=open_times,
                        probs=(one_word_prob, early_seal_prob, mid_seal_prob, late_seal_prob, weak_seal_prob),
                        n_simulations=10
                    )
    
    with col_right:
        st.markdown("### 📈 排队分析")
        st.info("👈 配置涨停信息后，点击按钮查看结果")


def evaluate_limitup_queue(symbol, limit_price, seal_amount, seal_time, order_time,
                           target_shares, open_times, probs):
    """评估涨停排队状态"""
    try:
        # 创建模拟器
        one_word_prob, early_seal_prob, mid_seal_prob, late_seal_prob, weak_seal_prob = probs
        simulator = LimitUpQueueSimulator(
            one_word_fill_prob=one_word_prob,
            early_seal_fill_prob=early_seal_prob,
            mid_seal_fill_prob=mid_seal_prob,
            late_seal_fill_prob=late_seal_prob,
            weak_seal_fill_prob=weak_seal_prob
        )
        
        # 构造日期时间
        today = datetime.now().date()
        seal_datetime = datetime.combine(today, seal_time)
        order_datetime = datetime.combine(today, order_time)
        
        # 评估排队状态
        queue_status = simulator.evaluate_queue_status(
            symbol=symbol,
            limit_price=limit_price,
            seal_amount=seal_amount,
            seal_time=seal_datetime,
            current_time=order_datetime,
            target_shares=target_shares,
            open_times=open_times
        )
        
        # 显示结果
        st.success("✅ 排队状态评估完成")
        
        # 涨停强度
        st.markdown("#### 💪 涨停强度")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("强度类型", queue_status.strength.value)
        with col2:
            st.metric("强度评分", f"{queue_status.strength_score:.1f}/100")
            st.progress(queue_status.strength_score / 100)
        
        # 封单信息
        st.markdown("#### 🔒 封单信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("封单金额", f"¥{queue_status.seal_amount:,.0f}")
        with col2:
            st.metric("封单股数", f"{queue_status.seal_shares:,}")
        with col3:
            st.metric("封单笔数", f"{queue_status.seal_orders:,}")
        
        # 排队信息
        st.markdown("#### 👥 排队信息")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("排队位置", f"{queue_status.queue_position:,}股")
            st.metric("排队金额", f"¥{queue_status.queue_ahead_amount:,.0f}")
        with col2:
            st.metric("成交概率", f"{queue_status.fill_probability:.1%}")
            st.progress(queue_status.fill_probability)
            st.metric("预计等待", f"{queue_status.estimated_wait_time:.0f}分钟")
        
        # 预计成交时间
        if queue_status.expected_fill_time:
            st.info(f"⏰ 预计成交时间: **{queue_status.expected_fill_time.strftime('%H:%M')}**")
        else:
            st.warning("⚠️ 成交概率较低，可能无法成交")
        
        # 警告信息
        if queue_status.warnings:
            st.warning("⚠️ **注意事项**\n" + "\n".join([f"- {w}" for w in queue_status.warnings]))
        
        # 可视化排队比例
        st.markdown("#### 📊 排队与封单对比")
        queue_ratio = queue_status.queue_position / queue_status.seal_shares if queue_status.seal_shares > 0 else 0
        
        chart_data = pd.DataFrame({
            "类型": ["封单", "排队"],
            "股数": [queue_status.seal_shares, queue_status.queue_position]
        })
        st.bar_chart(chart_data.set_index("类型"))
        
        st.caption(f"排队/封单比例: {queue_ratio:.2f}x")
        
    except Exception as e:
        st.error(f"❌ 评估失败: {str(e)}")
        with st.expander("🔍 查看详细错误"):
            st.code(traceback.format_exc())


def simulate_limitup_execution(symbol, limit_price, seal_amount, seal_time, order_time,
                               target_shares, open_times, probs, n_simulations=10):
    """模拟涨停成交（多次）"""
    try:
        # 创建模拟器
        one_word_prob, early_seal_prob, mid_seal_prob, late_seal_prob, weak_seal_prob = probs
        simulator = LimitUpQueueSimulator(
            one_word_fill_prob=one_word_prob,
            early_seal_fill_prob=early_seal_prob,
            mid_seal_fill_prob=mid_seal_prob,
            late_seal_fill_prob=late_seal_prob,
            weak_seal_fill_prob=weak_seal_prob
        )
        
        # 构造日期时间
        today = datetime.now().date()
        seal_datetime = datetime.combine(today, seal_time)
        order_datetime = datetime.combine(today, order_time)
        
        # 先评估排队状态
        queue_status = simulator.evaluate_queue_status(
            symbol=symbol,
            limit_price=limit_price,
            seal_amount=seal_amount,
            seal_time=seal_datetime,
            current_time=order_datetime,
            target_shares=target_shares,
            open_times=open_times
        )
        
        # 多次模拟
        st.markdown(f"#### 🎲 模拟成交结果（{n_simulations}次）")
        
        success_count = 0
        total_filled_shares = 0
        fill_times = []
        
        for i in range(n_simulations):
            execution = simulator.simulate_queue_execution(
                symbol=symbol,
                order_time=order_datetime,
                target_shares=target_shares,
                limit_price=limit_price,
                queue_status=queue_status,
                seal_broke=False
            )
            
            if execution.filled:
                success_count += 1
                total_filled_shares += execution.filled_shares
                if execution.fill_time:
                    fill_times.append(execution.fill_time)
        
        # 统计结果
        success_rate = success_count / n_simulations
        avg_filled_shares = total_filled_shares / success_count if success_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("成交次数", f"{success_count}/{n_simulations}")
        with col2:
            st.metric("实际成交率", f"{success_rate:.1%}")
        with col3:
            st.metric("理论成交概率", f"{queue_status.fill_probability:.1%}")
        
        # 成交详情
        if success_count > 0:
            st.markdown("#### ✅ 成交详情")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("平均成交股数", f"{avg_filled_shares:,.0f}")
                fill_ratio = avg_filled_shares / target_shares
                st.metric("平均成交比例", f"{fill_ratio:.1%}")
            with col2:
                if fill_times:
                    avg_fill_time = sum([(ft.hour * 60 + ft.minute) for ft in fill_times]) / len(fill_times)
                    hours = int(avg_fill_time // 60)
                    minutes = int(avg_fill_time % 60)
                    st.metric("平均成交时间", f"{hours:02d}:{minutes:02d}")
        else:
            st.warning("⚠️ 所有模拟均未能成交")
        
        # 可视化
        st.markdown("#### 📊 成交率对比")
        comparison_df = pd.DataFrame({
            "类型": ["理论概率", "实际成交率"],
            "概率": [queue_status.fill_probability, success_rate]
        })
        st.bar_chart(comparison_df.set_index("类型"))
        
        st.caption(f"偏差: {abs(success_rate - queue_status.fill_probability):.1%}")
        
    except Exception as e:
        st.error(f"❌ 模拟失败: {str(e)}")
        with st.expander("🔍 查看详细错误"):
            st.code(traceback.format_exc())


# ==================== 执行成本分析 ====================

def render_execution_cost_analysis():
    """渲染执行成本分析"""
    st.header("📊 执行成本分析")
    st.markdown("对比不同执行策略的成本差异")
    
    st.markdown("### 📝 批量订单配置")
    
    # 订单配置
    col1, col2, col3 = st.columns(3)
    with col1:
        base_price = st.number_input("基准价格（元）", value=10.00, step=0.01, format="%.2f", key="cost_price")
    with col2:
        total_shares = st.number_input("总股数", value=500000, step=10000, key="cost_shares")
    with col3:
        avg_daily_volume = st.number_input("日均量", value=5000000, step=100000, key="cost_volume")
    
    # 策略对比
    st.markdown("### 🔄 执行策略对比")
    
    strategies = []
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**策略1: 激进执行**")
        st.caption("一次性全部下单")
        strategies.append({
            "name": "激进执行",
            "model": SlippageModel.LINEAR,
            "shares": total_shares,
            "color": "🔴"
        })
    
    with col2:
        st.markdown("**策略2: 保守执行**")
        st.caption("分5批下单，每批20%")
        strategies.append({
            "name": "保守执行",
            "model": SlippageModel.SQRT,
            "shares": total_shares // 5,
            "batches": 5,
            "color": "🟢"
        })
    
    if st.button("📊 开始对比分析", type="primary", use_container_width=True, key="cost_compare"):
        compare_execution_strategies(
            base_price=base_price,
            total_shares=total_shares,
            avg_daily_volume=avg_daily_volume,
            strategies=strategies
        )


def compare_execution_strategies(base_price, total_shares, avg_daily_volume, strategies):
    """对比执行策略"""
    try:
        st.markdown("---")
        st.markdown("### 📈 对比结果")
        
        results = []
        
        for strategy in strategies:
            # 创建引擎
            engine = SlippageEngine(
                model=strategy["model"],
                impact_coefficient=0.1
            )
            
            # 执行订单
            if "batches" in strategy:
                # 分批执行
                total_cost = 0
                total_slippage = 0
                for batch in range(strategy["batches"]):
                    execution = engine.execute_order(
                        symbol="TEST",
                        side=OrderSide.BUY,
                        target_shares=strategy["shares"],
                        target_price=base_price,
                        avg_daily_volume=avg_daily_volume
                    )
                    cost_analysis = engine.calculate_total_slippage(execution)
                    total_cost += cost_analysis['total_cost']
                    total_slippage += execution.slippage_pct
                
                avg_slippage_pct = total_slippage / strategy["batches"]
                total_execution_cost = total_cost
            else:
                # 一次性执行
                execution = engine.execute_order(
                    symbol="TEST",
                    side=OrderSide.BUY,
                    target_shares=strategy["shares"],
                    target_price=base_price,
                    avg_daily_volume=avg_daily_volume
                )
                cost_analysis = engine.calculate_total_slippage(execution)
                avg_slippage_pct = execution.slippage_pct
                total_execution_cost = cost_analysis['total_cost']
            
            results.append({
                "策略": f"{strategy['color']} {strategy['name']}",
                "平均滑点": f"{avg_slippage_pct:.3%}",
                "总成本": f"¥{total_execution_cost:,.0f}",
                "成本基点": f"{(total_execution_cost / (base_price * total_shares)) * 10000:.2f} bps"
            })
        
        # 显示对比表
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # 结论
        st.success("✅ 对比分析完成")
        st.info("💡 **建议**: 大额订单建议分批执行，可有效降低市场冲击成本")
        
    except Exception as e:
        st.error(f"❌ 对比失败: {str(e)}")
        with st.expander("🔍 查看详细错误"):
            st.code(traceback.format_exc())


# ==================== 主入口 ====================

if __name__ == "__main__":
    render_qlib_execution_tab()
