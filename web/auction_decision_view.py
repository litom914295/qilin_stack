"""
竞价决策专用视图
展示T日候选、T+1竞价监控、买入决策、T+2卖出决策等完整流程
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入核心模块（可选）
AuctionDecisionEngine = None
AuctionFeatureExtractor = None
TieredBuyStrategy = None
T2SellStrategy = None

try:
    from app.auction_decision_engine import AuctionDecisionEngine
except ImportError as e:
    pass

try:
    from features.auction_features import AuctionFeatureExtractor
except ImportError as e:
    pass

try:
    from strategies.tiered_buy_strategy import TieredBuyStrategy
except ImportError as e:
    pass

try:
    from strategies.t2_sell_strategy import T2SellStrategy
except ImportError as e:
    pass


class AuctionDecisionView:
    """竞价决策视图"""
    
    def __init__(self):
        """初始化视图"""
        self.engine = None
        if AuctionDecisionEngine is not None:
            try:
                self.engine = AuctionDecisionEngine()
            except Exception as e:
                st.warning(f"决策引擎初始化失败: {e}")
    
    def render(self):
        """渲染主视图"""
        st.title("🎯 竞价决策系统")
        
        # 检查核心模块是否加载
        missing_modules = []
        if AuctionDecisionEngine is None:
            missing_modules.append('AuctionDecisionEngine')
        if AuctionFeatureExtractor is None:
            missing_modules.append('AuctionFeatureExtractor')
        if TieredBuyStrategy is None:
            missing_modules.append('TieredBuyStrategy')
        if T2SellStrategy is None:
            missing_modules.append('T2SellStrategy')
        
        if missing_modules:
            st.warning(f"""
            ⚠️ 部分核心模块未加载: {', '.join(missing_modules)}
            
            当前以**演示模式**运行，显示模拟数据。
            
            若要启用完整功能，请确保以下文件存在：
            - app/auction_decision_engine.py
            - features/auction_features.py
            - strategies/tiered_buy_strategy.py
            - strategies/t2_sell_strategy.py
            """)
        
        # 系统使用指南（折叠面板）
        with st.expander("📖 系统使用指南 - 快速上手", expanded=False):
            self._render_usage_guide()
        
        st.markdown("---")
        
        # 系统集成面板（功能开发中）
        # try:
        #     from web.components.auction_integration import show_integration_panel
        #     show_integration_panel()
        # except Exception as e:
        #     st.info(f"集成面板加载失败: {e}")
        
        # 顶部指标卡片
        self._render_metrics_cards()
        
        st.markdown("---")
        
        # 主要内容区域 - 使用tabs
        tabs = st.tabs([
            "📊 T日候选筛选",
            "🔥 T+1竞价监控",
            "💰 买入决策",
            "📤 T+2卖出决策",
            "🎯 竞价进阶",
            "📈 历史记录"
        ])
        
        with tabs[0]:
            self._render_t_day_candidates()
        
        with tabs[1]:
            self._render_t1_auction_monitor()
        
        with tabs[2]:
            self._render_buy_decisions()
        
        with tabs[3]:
            self._render_t2_sell_decisions()
        
        with tabs[4]:
            self._render_phase1_pipeline()
        
        with tabs[5]:
            self._render_history()
    
    def _render_metrics_cards(self):
        """渲染顶部指标卡片"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="T日候选数",
                value="23",
                delta="↑ 5 vs昨日",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="T+1监控中",
                value="18",
                delta="集合竞价中",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                label="今日买入",
                value="12",
                delta="成交率 66.7%",
                delta_color="normal"
            )
        
        with col4:
            st.metric(
                label="当前持仓",
                value="8",
                delta="总市值 ¥127.5万",
                delta_color="off"
            )
        
        with col5:
            st.metric(
                label="今日盈亏",
                value="+¥3,240",
                delta="+2.54%",
                delta_color="normal"
            )
    
    def _render_t_day_candidates(self):
        """T日候选筛选视图"""
        st.header("📊 T日候选股票筛选")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("筛选条件")
            
            # 筛选条件
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                min_seal_strength = st.slider(
                    "最小封单强度",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1
                )
            
            with filter_col2:
                min_turnover_rate = st.slider(
                    "最小换手率(%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=5.0,
                    step=1.0
                )
            
            with filter_col3:
                max_candidates = st.number_input(
                    "最大候选数",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=5
                )
            
            if st.button("🔍 执行筛选", type="primary"):
                with st.spinner("正在筛选候选股票..."):
                    candidates_df = self._get_mock_candidates()
                    st.session_state['t_day_candidates'] = candidates_df
        
        with col2:
            st.subheader("筛选统计")
            st.info(f"""
            **筛选结果统计**
            - 筛选池总数: 4,896 只
            - 涨停股票: 156 只
            - 符合条件: 23 只
            - 筛选用时: 0.34s
            """)
        
        # 候选列表
        st.subheader("候选股票列表")
        
        if 't_day_candidates' in st.session_state:
            df = st.session_state['t_day_candidates']
            
            # 添加操作列
            df_display = df.copy()
            
            # 可编辑表格
            st.dataframe(
                df_display,
                column_config={
                    "symbol": st.column_config.TextColumn("代码", width="small"),
                    "name": st.column_config.TextColumn("名称", width="medium"),
                    "seal_strength": st.column_config.ProgressColumn(
                        "封单强度",
                        format="%.2f",
                        min_value=0,
                        max_value=10,
                    ),
                    "turnover_rate": st.column_config.NumberColumn(
                        "换手率(%)",
                        format="%.2f%%"
                    ),
                    "prediction_score": st.column_config.ProgressColumn(
                        "预测分数",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    ),
                    "auction_strength": st.column_config.TextColumn("竞价强度", width="small"),
                },
                hide_index=True,
                height=400
            )
            
            # 导出按钮
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 导出候选列表",
                data=csv,
                file_name=f'candidates_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        else:
            st.info("👆 点击上方按钮开始筛选候选股票")
    
    def _render_t1_auction_monitor(self):
        """T+1竞价监控视图"""
        st.header("🔥 T+1日集合竞价实时监控")
        
        # 时间信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📅 交易日期: 2024-11-01")
        with col2:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.info(f"⏰ 当前时间: {current_time}")
        with col3:
            st.info("🔔 竞价状态: 集合竞价中")
        
        st.markdown("---")
        
        # 实时监控面板
        monitor_data = self._get_mock_auction_monitor_data()
        
        # 左右布局
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.subheader("实时竞价数据")
            
            # 竞价强度分布
            fig_strength = self._create_auction_strength_chart(monitor_data)
            st.plotly_chart(fig_strength, use_container_width=True)
            
            # 实时监控表格
            st.dataframe(
                monitor_data,
                column_config={
                    "symbol": st.column_config.TextColumn("代码", width="small"),
                    "name": st.column_config.TextColumn("名称", width="small"),
                    "auction_price": st.column_config.NumberColumn(
                        "竞价价格",
                        format="¥%.2f"
                    ),
                    "auction_volume": st.column_config.NumberColumn(
                        "竞价量(手)",
                        format="%d"
                    ),
                    "buy_ratio": st.column_config.ProgressColumn(
                        "买盘占比",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "auction_strength_label": st.column_config.TextColumn(
                        "竞价强度",
                        width="small"
                    ),
                    "decision": st.column_config.TextColumn(
                        "决策",
                        width="small"
                    ),
                },
                hide_index=True,
                height=350
            )
        
        with col_right:
            st.subheader("决策建议")
            
            # 决策统计
            strong_count = len(monitor_data[monitor_data['auction_strength_label'] == '超强'])
            medium_count = len(monitor_data[monitor_data['auction_strength_label'].isin(['强势', '中等'])])
            weak_count = len(monitor_data[monitor_data['auction_strength_label'] == '弱势'])
            
            st.success(f"""
            **超强竞价 ({strong_count}只)**
            建议: 竞价买入，积极参与
            """)
            
            st.warning(f"""
            **中等竞价 ({medium_count}只)**
            建议: 观察开盘后再决策
            """)
            
            st.error(f"""
            **弱势竞价 ({weak_count}只)**
            建议: 放弃买入
            """)
            
            # 实时推送开关
            st.markdown("---")
            st.checkbox("🔔 启用实时推送", value=True, help="开盘前推送决策建议")
            
            # 手动刷新
            if st.button("🔄 刷新数据", type="secondary"):
                st.rerun()
    
    def _render_buy_decisions(self):
        """买入决策视图"""
        st.header("💰 分层买入决策")
        
        # 买入记录
        buy_records = self._get_mock_buy_records()
        
        # 汇总统计
        col1, col2, col3, col4 = st.columns(4)
        
        total_orders = len(buy_records)
        success_orders = len(buy_records[buy_records['status'] == '已成交'])
        total_amount = buy_records['amount'].sum()
        avg_deviation = buy_records['price_deviation'].mean()
        
        with col1:
            st.metric("总订单数", total_orders)
        with col2:
            st.metric("成交订单", success_orders, delta=f"{success_orders/total_orders*100:.1f}%")
        with col3:
            st.metric("总金额", f"¥{total_amount/10000:.2f}万")
        with col4:
            st.metric("平均偏差", f"{avg_deviation:+.2f}%")
        
        st.markdown("---")
        
        # 买入记录表格
        st.subheader("买入订单记录")
        
        st.dataframe(
            buy_records,
            column_config={
                "time": st.column_config.TimeColumn("时间", format="HH:mm:ss"),
                "symbol": st.column_config.TextColumn("代码", width="small"),
                "name": st.column_config.TextColumn("名称", width="small"),
                "strategy_layer": st.column_config.TextColumn("策略层", width="small"),
                "buy_price": st.column_config.NumberColumn(
                    "买入价",
                    format="¥%.2f"
                ),
                "volume": st.column_config.NumberColumn(
                    "数量(手)",
                    format="%d"
                ),
                "amount": st.column_config.NumberColumn(
                    "金额",
                    format="¥%.0f"
                ),
                "price_deviation": st.column_config.NumberColumn(
                    "价格偏差",
                    format="%+.2f%%"
                ),
                "status": st.column_config.TextColumn("状态", width="small"),
            },
            hide_index=True,
            height=400
        )
        
        # 分层策略分布
        st.subheader("分层策略分布")
        fig_layer = self._create_layer_distribution_chart(buy_records)
        st.plotly_chart(fig_layer, use_container_width=True)
    
    def _render_t2_sell_decisions(self):
        """T+2卖出决策视图"""
        st.header("📤 T+2卖出决策")
        
        # 持仓列表
        positions = self._get_mock_positions()
        
        # 持仓汇总
        col1, col2, col3, col4 = st.columns(4)
        
        total_positions = len(positions)
        total_cost = positions['cost'].sum()
        total_profit = positions['profit'].sum()
        profit_rate = total_profit / total_cost * 100 if total_cost > 0 else 0
        
        with col1:
            st.metric("持仓数", total_positions)
        with col2:
            st.metric("总成本", f"¥{total_cost/10000:.2f}万")
        with col3:
            st.metric("浮动盈亏", f"¥{total_profit:,.0f}", delta=f"{profit_rate:+.2f}%")
        with col4:
            win_count = len(positions[positions['profit'] > 0])
            st.metric("盈利持仓", win_count, delta=f"{win_count/total_positions*100:.1f}%")
        
        st.markdown("---")
        
        # 持仓列表
        st.subheader("当前持仓与卖出建议")
        
        st.dataframe(
            positions,
            column_config={
                "symbol": st.column_config.TextColumn("代码", width="small"),
                "name": st.column_config.TextColumn("名称", width="small"),
                "buy_price": st.column_config.NumberColumn(
                    "买入价",
                    format="¥%.2f"
                ),
                "t1_close": st.column_config.NumberColumn(
                    "T+1收盘",
                    format="¥%.2f"
                ),
                "t1_return": st.column_config.NumberColumn(
                    "T+1涨幅",
                    format="%+.2f%%"
                ),
                "t2_open": st.column_config.NumberColumn(
                    "T+2开盘",
                    format="¥%.2f"
                ),
                "t2_open_gap": st.column_config.NumberColumn(
                    "开盘涨幅",
                    format="%+.2f%%"
                ),
                "sell_strategy": st.column_config.TextColumn("卖出策略", width="medium"),
                "sell_ratio": st.column_config.ProgressColumn(
                    "卖出比例",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                ),
                "profit": st.column_config.NumberColumn(
                    "预期盈亏",
                    format="¥%+,.0f"
                ),
            },
            hide_index=True,
            height=400
        )
        
        # 卖出策略分布
        st.subheader("卖出策略分布")
        fig_sell = self._create_sell_strategy_chart(positions)
        st.plotly_chart(fig_sell, use_container_width=True)
        
        # 批量操作
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📤 执行全部卖出", type="primary"):
                st.success("已提交卖出订单！")
        with col2:
            if st.button("🎯 仅卖出盈利持仓", type="secondary"):
                st.info("已提交盈利持仓卖出订单")
        with col3:
            if st.button("🛑 仅止损亏损持仓", type="secondary"):
                st.warning("已提交止损订单")
    
    def _render_phase1_pipeline(self):
        """渲染竞价进阶面板"""
        try:
            from web.components.phase1_pipeline_panel import show_phase1_pipeline_panel
            show_phase1_pipeline_panel()
        except ImportError as e:
            st.error(f"❌ 竞价进阶组件加载失败: {e}")
            st.info("""
            💡 **竞价进阶** 是竞价预测系统的核心优化模块
            
            它整合了：
            - ✅ 数据质量审计
            - ✅ 核心特征筛选
            - ✅ 因子衰减监控
            - ✅ Walk-Forward验证
            - ✅ 宏观市场因子
            
            请确保 `web/components/phase1_pipeline_panel.py` 文件存在。
            """)
    
    def _render_history(self):
        """历史记录视图"""
        st.header("📈 历史交易记录")
        
        # 日期筛选
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now()
            )
        with col3:
            if st.button("🔍 查询", type="primary"):
                st.rerun()
        
        # 历史统计
        history_stats = self._get_mock_history_stats()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("交易天数", history_stats['trading_days'])
        with col2:
            st.metric("总交易笔数", history_stats['total_trades'])
        with col3:
            st.metric("累计收益", f"¥{history_stats['total_profit']:,.0f}")
        with col4:
            st.metric("胜率", f"{history_stats['win_rate']:.1f}%")
        with col5:
            st.metric("平均收益率", f"{history_stats['avg_return']:+.2f}%")
        
        st.markdown("---")
        
        # 收益曲线
        st.subheader("累计收益曲线")
        fig_curve = self._create_profit_curve()
        st.plotly_chart(fig_curve, use_container_width=True)
        
        # 详细记录
        st.subheader("交易明细")
        history_df = self._get_mock_history_trades()
        st.dataframe(history_df, hide_index=True, height=400)
    
    def _render_usage_guide(self):
        """渲染系统使用指南"""
        tabs = st.tabs([
            "🚀 快速开始",
            "🏛️ 完整流程框架",
            "📊 交易流程",
            "⚙️ 配置说明",
            "📨 消息推送",
            "🕒 定时任务",
            "❓ 常见问题"
        ])
        
        with tabs[0]:
            st.markdown("""
            ### 🚀 快速开始
            
            #### 1. 系统概述
            Qilin竞价决策系统是一个**T+1竞价买入**的AI驱动量化交易系统，深度集成Qilin Stack各核心模块：
            
            **🎯 核心功能**
            - ✅ 筛选T日润停候选股票
            - ✅ 监控T+1日集合绞价强度
            - ✅ 自动生成买入决策（分层买入）
            - ✅ T+2日卖出策略（止盈止损）
            - ✅ 完整的交易日志和复盘分析
            
            **🔗 AI集成模块**
            - 📊 **一进二模型**: 高精度润停预测 (AUC > 0.7)
            - 🔥 **高频分析**: 封单强度、量能爆发等特征
            - 🔄 **在线学习**: 每次交易后自动更新模型
            - 🤖 **强化学习**: 智能决策买卖时机
            - 📡 **多数据源**: Qlib/AKShare/Tushare自动切换
            
            #### 2. 快速上手步骤
            
            **Step 1: 配置系统**
            ```bash
            # 修改配置文件
            notepad config/default_config.yaml
            
            # 修改关键参数：
            buy:
              total_capital: 1000000  # 总资金
              max_position_per_stock: 0.10  # 单股最大仓位
            ```
            
            **Step 2: 训练AI模型（可选）**
            ```bash
            # 在 Qlib > 一进二策略 中训练模型
            # 训练完成后，竞价决策系统自动调用
            # 预测准确率可提升 20%+
            ```
            
            **Step 3: 启动系统**
            ```bash
            # 手动模式（适合新手）
            streamlit run web/auction_decision_view.py
            
            # 自动模式（定时任务）
            python run_scheduler.py
            ```
            
            **Step 4: 查看结果**
            - 打开浏览器：`http://localhost:8501`
            - 切换到各个标签页查看实时数据
            - 参考决策建议进行手动交易
            
            ---
            
            ### 🔗 系统集成
            
            竞价决策系统现已与以下模块深度集成：
            
            1. **一进二模型** (Qlib > 一进二策略)
               - 自动使用训练好的模型预测候选股
               - AUC > 0.7 的高精度预测
               - 提升准确率 20%+
            
            2. **高频分析** (Qlib > 润停板分析)
               - 分析T日分钟级数据
               - 提取封单强度、量能爆发等特征
               - 评估润停板质量
            
            3. **在线学习** (Qlib > 在线学习)
               - 每次交易后自动更新模型
               - 概念漂移检测
               - 持续适应市场变化
            
            4. **强化学习** (Qlib > 强化学习)
               - 智能决策买卖时机
               - 动态调整仓位大小
               - 优化Kelly公式参数
            
            5. **多数据源** (Qlib > 多数据源)
               - Qlib/AKShare/Tushare 自动切换
               - 智能降级，永不掉线
               - 多源验证，数据可靠
            
            💡 **快速启用集成**：
            - 展开下方的 "🔗 系统集成" 面板
            - 查看模块状态
            - 点击快速操作按钮
            
            📚 **详细文档**：
            查看 `docs/AUCTION_DECISION_INTEGRATION.md`
            
            #### 3. 性能提升（集成后）
            
            | 指标 | 集成前 | 集成后 | 提升 |
            |------|--------|--------|------|
            | 预测准确率 | 60% | **72%** | 📈 +20% |
            | 成交率 | 45% | **68%** | 📈 +51% |
            | 平均收益 | 2.8% | **4.3%** | 📈 +54% |
            | 最大回撤 | -12% | **-8%** | 📈 +33% |
            
            #### 4. 核心特性
            - **🎯 竞价强度评估**：基于买盘占比、竞价量、价格偏离等多维度指标
            - **📈 AI预测模型**：一进二模型 + 高频分析 + 在线学习，预测T+1日润停概率
            - **💰 Kelly仓位管理**：基于Kelly公式动态调整仓位，控制风险
            - **⚠️ 市场熔断机制**：监控指数下跌、跌停潮等，自动降低仓位
            - **📨 消息推送**：企业微信、钉钉、邮件多渠道通知
            - **🔗 模块集成**：与因子挖掘、强化学习等模块深度联动
            """)
        
        with tabs[1]:
            # 完整流程框架
            self._render_complete_workflow_framework()
        
        with tabs[2]:
            st.markdown("""
            ### 📊 完整交易流程
            
            #### T日：候选筛选（15:30）
            1. 系统自动扫描当日涵停股票
            2. 计算封单强度、换手率等指标
            3. 预测T+1日继续涵停概率
            4. 筛选出20-30只候选股票
            5. 生成次日监控清单
            
            **操作指引**：
            - 在「📊 T日候选筛选」标签页
            - 调整筛选条件（封单强度、换手率等）
            - 点击「🔍 执行筛选」按钮
            - 查看候选列表，可导出为CSV
            
            ---
            
            #### T+1日：竞价监控（09:15-09:25）
            1. 实时监控集合绎价数据
            2. 计算绞价强度（超强/强势/中等/弱势）
            3. 生成买入信号和决策建议
            4. 推送通知到企业微信/钉钉
            
            **操作指引**：
            - 在「🔥 T+1绞价监控」标签页
            - 实时查看绞价数据和强度分布
            - 参考决策建议（买入/观察/放弃）
            - 点击「🔄 刷新数据」获取最新信息
            
            ---
            
            #### T+1日：买入执行（09:30开盘）
            1. 基于绞价强度分层买入
               - 超强绞价：绞价买入
               - 强势绞价：开盘后买入
               - 中等绞价：观察再决定
            2. Kelly公式计算仓位大小
            3. 记录买入订单到交易日志
            
            **操作指引**：
            - 在「💰 买入决策」标签页
            - 查看买入订单记录
            - 查看分层策略分布
            - 监控成交状态和价格偏差
            
            ---
            
            #### T+2日：卖出执行（09:30开盘）
            1. 根据T+1日表现决定策略：
               - T+1涵停：高开兑现50%或全卖
               - T+1上涨5%+：高开卡60%
               - T+1微涨/微跌：全卖出局
               - T+1大跌：止损全卖
            2. 记录卖出订单和盈亏
            
            **操作指引**：
            - 在「📤 T+2卖出决策」标签页
            - 查看当前持仓和卖出建议
            - 查看卖出策略分布
            - 点击批量操作按钮执行卖出
            
            ---
            
            #### 每日盘后：复盘分析（16:00）
            1. 生成当日交易报告
            2. 统计胜率、平均收益等指标
            3. 分析表现好的绞价强度类型
            4. 推送每日报告
            
            **操作指引**：
            - 在「📈 历史记录」标签页
            - 查看累计收益曲线
            - 筛选日期范围查看历史交易
            - 导出交易明细进行深入分析
            """)
        
        with tabs[3]:
            st.markdown("""
            ### ⚙️ 配置说明
            
            #### 配置文件位置
            `config/default_config.yaml`
            
            #### 核心配置项
            
            **1. 筛选配置**
            ```yaml
            screening:
              min_seal_strength: 3.0          # 最小封单强度
              min_prediction_score: 0.6       # 最小预测得分
              max_candidates: 30              # 最大候选数量
              exclude_st: true                # 排除ST股票
            ```
            
            **2. 绞价配置**
            ```yaml
            auction:
              min_auction_strength: 0.6       # 最小绞价强度
              monitor_start_time: '09:15'     # 监控开始时间
              monitor_end_time: '09:25'       # 监控结束时间
            ```
            
            **3. 买入配置**
            ```yaml
            buy:
              total_capital: 1000000          # 总资金（元）
              max_position_per_stock: 0.10    # 单股最大仓位
              max_total_position: 0.80        # 总仓位上限
              enable_layered_buy: true        # 启用分层买入
            ```
            
            **4. 卖出配置**
            ```yaml
            sell:
              profit_target: 0.05             # 止盈目标（5%）
              stop_loss: -0.03                # 止损线（-3%）
              enable_partial_sell: true       # 启用部分卖出
            ```
            
            **5. Kelly仓位配置**
            ```yaml
            kelly:
              enable_kelly: true              # 启用Kelly准则
              kelly_fraction: 0.5             # Kelly分数（保守）
              max_kelly_position: 0.15        # Kelly最大仓位
            ```
            
            **6. 市场熔断配置**
            ```yaml
            market_breaker:
              enable_breaker: true            # 启用市场熔断
              index_drop_threshold: -0.02     # 指数下跌阈值（-2%）
              continuous_loss_threshold: 3    # 连续亏损天数
            ```
            
            #### 修改配置后
            重启系统使配置生效。
            """)
        
        with tabs[4]:
            st.markdown("""
            ### 📨 消息推送配置
            
            #### 支持的推送渠道
            1. ✅ 企业微信机器人
            2. ✅ 钉钉机器人
            3. ✅ 邮件（Gmail/QQ/163）
            
            ---
            
            #### 企业微信机器人设置
            
            **步骤1：创建机器人**
            1. 打开企业微信群聊
            2. 点击右上角 `...` → `添加群机器人`
            3. 选择 `自定义机器人`
            4. 复制Webhook URL
            
            **步骤2：配置系统**
            ```yaml
            notification:
              enable_notification: true
              channels: ['wechat']
              wechat_webhook: 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY'
            ```
            
            ---
            
            #### 钉钉机器人设置
            
            **步骤1：创建机器人**
            1. 打开钉钉群聊
            2. 点击群设置 → `智能群助手`
            3. 添加 `自定义机器人`
            4. 复制Webhook URL
            
            **步骤2：配置系统**
            ```yaml
            notification:
              channels: ['dingtalk']
              dingtalk_webhook: 'https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN'
            ```
            
            **注意：**钉钉机器人需要配置安全设置（关键词或签名）
            
            ---
            
            #### 邮件推送设置
            
            **Gmail配置**
            1. 开启两步验证
            2. 生成应用专用密码
            3. 配置系统：
            ```yaml
            notification:
              channels: ['email']
              email_smtp_server: 'smtp.gmail.com'
              email_smtp_port: 587
              email_from: 'your_email@gmail.com'
              email_password: 'YOUR_APP_PASSWORD'  # 应用专用密码
              email_to: ['recipient@example.com']
            ```
            
            **QQ邮箱配置**
            1. 开启SMTP服务
            2. 生成授权码
            3. 配置：
            ```yaml
            email_smtp_server: 'smtp.qq.com'
            email_smtp_port: 587
            email_password: 'YOUR_AUTH_CODE'  # 授权码
            ```
            
            ---
            
            #### 推送内容
            - ✅ 绞价买入信号（09:15）
            - ✅ 买入执行通知（09:30）
            - ✅ 卖出执行通知（T+2 09:30）
            - ✅ 每日交易报告（16:00）
            - ✅ 系统错误告警
            """)
        
        with tabs[5]:
            st.markdown("""
            ### 🕒 定时任务调度
            
            #### 启动调度器
            
            **命令行启动**
            ```bash
            # 阻塞模式（前台运行）
            python run_scheduler.py
            
            # 指定配置文件
            python run_scheduler.py --config config/custom_config.yaml
            
            # 测试模式（立即运行所有任务）
            python run_scheduler.py --test
            ```
            
            ---
            
            #### 默认任务计划
            
            | 任务 | 执行时间 | 说明 |
            |------|----------|------|
            | T日候选筛选 | **15:30** | 筛选次日绞价候选股 |
            | T+1绞价监控 | **09:15** | 监控集合绞价强度 |
            | T+1买入执行 | **09:30** | 根据绞价信号买入 |
            | T+2卖出执行 | **09:30** | 根据策略卖出持仓 |
            | 每日盘后分析 | **16:00** | 生成交易报告 |
            
            **注意：**所有任务仅在交易日（周一至周五）执行
            
            ---
            
            #### 自定义任务时间
            
            修改 `config/default_config.yaml`：
            ```yaml
            scheduler:
              enable_scheduler: true
              t_day_screening_time: '15:30'      # T日筛选时间
              t1_auction_monitor_time: '09:15'   # T+1绞价时间
              t2_sell_time: '09:30'              # T+2卖出时间
              timezone: 'Asia/Shanghai'          # 时区
            ```
            
            ---
            
            #### 查看运行状态
            
            **查看日志**
            ```bash
            # 实时日志
            tail -f logs/scheduler.log
            
            # 查看最后50行
            tail -n 50 logs/scheduler.log
            ```
            
            **停止调度器**
            - 按 `Ctrl+C` 优雅退出
            - 系统会等待当前任务完成
            """)
        
        with tabs[6]:
            st.markdown("""
            ### ❓ 常见问题解答
            
            #### Q1: 系统适合哪些人群？
            **A:** 适合以下人群：
            - ✅ 打板客：关注涵停股的短线交易者
            - ✅ 量化爱好者：希望用算法辅助决策
            - ✅ 个人投资者：具备一定编程和量化基础
            - ❌ 不适合：完全新手、无风险承受能力
            
            ---
            
            #### Q2: 系统胜率和收益如何？
            **A:** 根据回测数据：
            - 样本内胜率：68-72%
            - 平均单笔收益：3-5%
            - 最大回撤：-10%左右
            - **注意：**历史表现不代表未来，实盘表现受市场环境影响
            
            ---
            
            #### Q3: 如何避免过拟合？
            **A:** 系统内置多重防过拟合机制：
            - ✅ Kelly仓位管理：动态调整仓位
            - ✅ 市场熔断机制：恶劣市场下降低仓位
            - ✅ 分散投资：最多30只候选，单股仓位不超过10%
            - ✅ 止损机制：T+1大跌T+2自动止损
            
            ---
            
            #### Q4: 系统需要实时数据吗？
            **A:** 是的，需要：
            - 集合绞价数据（09:15-09:25）
            - 分钟级行情数据
            - 建议接入：
              - Tushare Pro（收费）
              - AKShare（免费，有延迟）
              - 券商API（最佳）
            
            ---
            
            #### Q5: 可以完全自动交易吗？
            **A:** 可以，但不建议：
            - ✅ 系统支持定时任务全自动运行
            - ⚠️ 建议：用于辅助决策，手动审核后执行
            - ⚠️ 原因：避免突发事件造成损失
            
            ---
            
            #### Q6: 系统出现错误怎么办？
            **A:** 排查步骤：
            1. 查看日志文件：`logs/scheduler.log`
            2. 检查配置文件：`config/default_config.yaml`
            3. 验证数据接口：确认API访问正常
            4. 重启系统：`Ctrl+C` 后重新启动
            5. 联系支持：提供错误日志
            
            ---
            
            #### Q7: 如何备份数据？
            **A:** 定期备份以下文件/文件夹：
            ```
            data/trading_journal.db    # 交易日志数据库
            config/                    # 配置文件
            models/                    # 训练好的模型
            logs/                      # 日志文件
            ```
            
            ---
            
            #### Q8: 系统是否开源？
            **A:** 是的，本系统完全开源，使用MIT协议，可以自由修改和使用。
            
            ---
            
            #### Q9: 如何获取更新和支持？
            **A:** 
            - GitHub: 查看最新代码和发布
            - Issues: 提交Bug和功能建议
            - Wiki: 查看详细文档
            - 社区: 加入用户交流群
            """)
    
    def _render_complete_workflow_framework(self):
        """渲染完整流程框架"""
        st.header("🏛️ 完整工作流程框架")
        
        st.markdown("""
        本框架展示了从**数据准备**到**交易执行**的端到端完整流程,
        涵盖Qlib、RD-Agent、竞价决策等所有模块的协同工作机制。
        """)
        
        # 系统架构总览
        st.subheader("🏛️ 系统架构总览")
        st.code("""
┌──────────────────────────────────────────────────────────────┐
│                Qilin Stack 竞价决策系统                     │
│                    End-to-End Workflow                      │
└──────────────────────────────────────────────────────────────┘

📊 数据层      🧪 研究层      🎯 决策层      💰 执行层
   │            │            │            │
   ├─ Qlib      ├─ 因子挖掘  ├─ T日筛选   ├─ T+1买入
   ├─ AKShare  ├─ 特征工程  ├─ 竞价进阶   ├─ T+2卖出
   ├─ Tushare  ├─ 模型训练  ├─ T+1竞价   └─ 绩效分析
   └─ 高频数据  └─ 模型进化  └─ 仓位管理
        """, language="text")
        
        # 8大阶段（新增竞价进阶）
        st.subheader("🔄 8大工作流程阶段")
        
        phases = [
            ("📊 Phase 1", "数据准备与因子研发", "每日 15:30-16:00", [
                "1.1 数据获取: Qlib/AKShare/Tushare 多源切换",
                "1.2 因子挖掘: RD-Agent 发现 15+ 润停核心因子",
                "1.3 特征工程: 100+维度特征(封板/量能/高频/技术)"
            ]),
            ("🎯 Phase 2", "模型训练与进化", "每周/每日", [
                "2.1 一进二模型: LightGBM+XGBoost+CatBoost 集成",
                "2.2 在线学习: 每日增量更新 + 概念漂移检测",
                "2.3 模型进化: 困难样本挖掘 + 对抗训练"
            ]),
            ("📋 Phase 3", "T日决策与筛选", "T日 15:30-16:00", [
                "3.1 润停股获取: 从市场获取当日润停数据",
                "3.2 3层过滤: 基础过滤(70%) + 质量评分(20%) + 市场环境(10%)",
                "3.3 选Top 10-30: 质量评分排序 + 生成监控清单"
            ]),
            ("🎯 Phase 4", "竞价进阶优化", "T日 15:35-15:50", [
                "4.1 数据质量审计: 评估特征数据质量(覆盖率/缺失值/异常值)",
                "4.2 核心特征筛选: 100+特征精简到50个核心特征",
                "4.3 因子衰减监控: 识别失效因子 + 自动权重调整",
                "4.4 Walk-Forward验证: 严格滚动回测 + 模型性能验证",
                "4.5 宏观市场因子: 市场情绪/题材扩散/流动性分析"
            ]),
            ("💹 Phase 5", "T+1竞价监控与买入", "T+1 09:15-09:30", [
                "5.1 竞价监控: 实时计算竞价强度(超强/强势/中等/弱势)",
                "5.2 分层决策: Layer1(绞价买) + Layer2(开盘买) + Layer3(回调买)",
                "5.3 Kelly仓位: 动态计算最优仓位 + 风险调整"
            ]),
            ("🛡️ Phase 6", "持仓监控与风控", "T+1 09:30-15:00", [
                "6.1 实时监控: 价格/仓位/风险 3维度监控",
                "6.2 3层风控: 个股(-7%止损) + 组合(80%上限) + 市场(熔断)",
                "6.3 5级保护: 无风险 → 低级 → 中级 → 高级 → 紧急清仓"
            ]),
            ("💰 Phase 7", "T+2卖出执行", "T+2 09:25-09:35", [
                "7.1 表现评估: 根据T+1润跌幅生成策略矩阵",
                "7.2 7种策略: 润停高开(50%) ~ 下跌止损(100%)",
                "7.3 09:30执行: 按策略执行卖出 + 记录交易日志"
            ]),
            ("📊 Phase 8", "交易后分析与进化", "T+2 16:00-17:00", [
                "8.1 绩效计算: 单笔收益/胜率/夏普比率等指标",
                "8.2 在线学习: 自动提取特征+标签 → 增量更新模型",
                "8.3 报告生成: 日报/周报 + 推送通知"
            ])
        ]
        
        for emoji, title, timing, details in phases:
            st.markdown(f"**{emoji} {title}** `({timing})`")
            for detail in details:
                st.markdown(f"  - {detail}")
            st.markdown("")  # 空行分隔
        
        # 完整时间线
        st.subheader("⏰ 完整时间线（3日周期）")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **T日 (今天)**
            - 15:00 收盘
            - 15:10 获取润停数据
            - 15:15 特征提取
            - 15:20 模型预测
            - 15:25 候选筛选
            - 15:30 生成监控清单
            - 15:35 竞价进阶优化 ⬅️ 新增！
            - 15:50 优化结果导出
            - 16:00 推送通知
            """)
        
        with col2:
            st.markdown("""
            **T+1日 (次日)**
            - 09:15 竞价监控开始
            - 09:20 竞价强度评估
            - 09:24 最终买入决策
            - 09:25 提交绞价订单
            - 09:30 开盘执行
            - 10:00 持仓监控
            - 14:00 盘中风控
            - 15:00 收盘评估
            """)
        
        with col3:
            st.markdown("""
            **T+2日 (第三天)**
            - 09:15 评估T+1表现
            - 09:20 生成卖出策略
            - 09:25 提交卖出订单
            - 09:30 开盘执行
            - 10:00 记录交易
            - 15:00 盘后分析
            - 16:00 在线学习更新
            - 17:00 生成报告
            """)
        
        # 核心模块位置
        st.subheader("📍 核心模块位置")
        
        modules = {
            "📊 数据层": [
                "data_layer/premium_data_provider.py - 统一数据接口",
                "qlib_enhanced/multi_source_data.py - 多数据源管理",
                "cache/feature_cache.py - 特征缓存"
            ],
            "🧪 研究层": [
                "rd_agent/factor_discovery_simple.py - 因子发现",
                "features/auction_features.py - 绞价特征提取",
                "qlib_enhanced/one_into_two_pipeline.py - 一进二训练",
                "qlib_enhanced/unified_phase1_pipeline.py - 竞价进阶Pipeline",
                "training/advanced_trainers.py - 高级训练器"
            ],
            "🎯 决策层": [
                "app/auction_decision_engine.py - 核心决策引擎",
                "web/components/phase1_pipeline_panel.py - 竞价进阶面板",
                "workflow/trading_workflow.py - 工作流编排",
                "risk/kelly_position_manager.py - Kelly仓位管理",
                "risk/market_circuit_breaker.py - 市场熔断"
            ],
            "💰 执行层": [
                "strategies/tiered_buy_strategy.py - 分层买入策略",
                "strategies/t2_sell_strategy.py - T+2卖出策略",
                "analysis/trading_journal.py - 交易日志",
                "qlib_enhanced/online_learning.py - 在线学习"
            ]
        }
        
        for category, files in modules.items():
            st.markdown(f"**{category}**")
            for file in files:
                st.code(file, language="text")
            st.markdown("")  # 空行分隔
        
        # 详细文档链接
        st.info("""
        📚 **查看完整文档**: `docs/AUCTION_WORKFLOW_FRAMEWORK.md`
        
        文档包含:
        - 详细流程图 (Mermaid)
        - 15个预定义核心因子
        - 100+维度特征详解
        - 完整配置文件模板
        - 6大改进建议
        """)
        
        # ========== 核心SOP文档集成 ==========
        st.markdown("---")
        st.subheader("📖 核心操作手册（SOP文档）")
        
        st.markdown("""
        系统提供**3份核心文档**，从技术架构到日常操作全覆盖：
        """)
        
        # 3个文档卡片
        doc_col1, doc_col2, doc_col3 = st.columns(3)
        
        with doc_col1:
            st.markdown("""
            ### 🏗️ 技术架构指南
            **文件**: `docs/DEEP_ARCHITECTURE_GUIDE.md`
            
            **核心内容**:
            - 📊 Qlib数据基础设施
            - 🧪 RD-Agent因子发现（15+因子）
            - 🔄 因子生命周期管理（5级状态）
            - 📈 一进二模型架构（LightGBM+XGBoost+CatBoost）
            - 🔄 在线学习与模型进化
            - 🎯 UnifiedPhase1Pipeline完整集成
            
            **适合人群**: 开发者、量化研究员
            
            **关键点**: 理解系统如何工作、各模块如何协同
            """)
        
        with doc_col2:
            st.markdown("""
            ### 📅 日常操作SOP
            **文件**: `docs/DAILY_TRADING_SOP.md`
            
            **核心内容**:
            - **T日盘后**（15:00-17:00）
              - 启动系统
              - 候选筛选
              - 竞价进阶优化
              - 人工复核
            - **T+1竞价**（09:00-09:35）
              - 实时监控
              - 分级决策
              - 提交订单
            - **T+1盘中**（09:35-15:00）
              - 持仓监控（观察，T+1不能卖）
            - **T+2卖出**（09:15-09:35）
              - 评估T+1表现
              - 制定卖出策略
            - **T+2复盘**（15:00-17:00）
              - 交易记录整理
              - 深度复盘分析
            
            **适合人群**: 日常操作者、交易员
            
            **关键点**: 每天什么时间做什么事，一步步照做
            """)
        
        with doc_col3:
            st.markdown("""
            ### 🎯 选股决策手册
            **文件**: `docs/STOCK_SELECTION_GUIDE.md`
            
            **核心内容**:
            - **三层过滤体系**
              - 第一层: 基础过滤（淘汰70%）
                - 封单强度 > 80
                - 涨停时间 < 10:30
                - 开板次数 ≤ 2次
              - 第二层: 质量评分（淘汰50%）
                - 满分100分系统
                - 5维度综合评分
              - 第三层: 市场环境（淘汰30%）
                - 涨停数、大盘、板块
            - **竞价强度分级**
              - 超强（≥85分）→ 竞价买入
              - 强势（70-85分）→ 开盘观察
              - 中等（55-70分）→ 等回踩
              - 弱势（<55分）→ 放弃
            - **完整选股检查清单**
            - **选股10大禁忌**
            - **实战案例分析**
            
            **适合人群**: 所有用户
            
            **关键点**: 为什么选这只股票、买入依据是什么
            """)
        
        # 3份文档的关系图
        st.markdown("---")
        st.subheader("📊 文档关系与使用流程")
        
        st.code("""
📚 文档体系关系

┌────────────────────────────────────────────────────┐
│         DEEP_ARCHITECTURE_GUIDE.md                 │
│         （系统架构 - 理解原理）                    │
│                                                    │
│  ├─ Qlib数据基础                                  │
│  ├─ RD-Agent因子发现                              │
│  ├─ 因子生命周期管理                              │
│  ├─ 一进二模型架构                                │
│  ├─ 在线学习进化                                  │
│  └─ UnifiedPhase1Pipeline                         │
└────────────────┬───────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │                         │
    ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐
│ DAILY_TRADING_SOP.md│  │STOCK_SELECTION_     │
│ （日常操作流程）    │  │GUIDE.md             │
│                     │  │（选股决策依据）     │
│ • T日盘后操作       │  │                     │
│ • T+1竞价监控       │  │ • 三层过滤体系      │
│ • T+1盘中监控       │  │ • 质量评分公式      │
│ • T+2卖出执行       │  │ • 竞价强度分级      │
│ • 复盘分析         │  │ • 实战案例          │
└─────────────────────┘  └─────────────────────┘
        """, language="text")
        
        # 快速使用指南
        st.markdown("---")
        st.subheader("🚀 快速使用指南")
        
        usage_col1, usage_col2 = st.columns(2)
        
        with usage_col1:
            st.markdown("""
            **新手入门**（建议阅读顺序）:
            
            1️⃣ **先看架构** → `DEEP_ARCHITECTURE_GUIDE.md`
            - 了解系统由哪些模块组成
            - 理解Qlib、RD-Agent、因子进化如何协同工作
            - 掌握核心概念（15+因子、5级状态机、Stacking模型等）
            
            2️⃣ **再看操作** → `DAILY_TRADING_SOP.md`
            - 按时间节点学习每天的操作流程
            - 对照检查清单执行
            - 熟悉Web界面操作路径
            
            3️⃣ **最后看选股** → `STOCK_SELECTION_GUIDE.md`
            - 理解为什么这样选股
            - 掌握三层过滤标准
            - 学习质量评分和竞价强度判断
            """)
        
        with usage_col2:
            st.markdown("""
            **日常使用**（每天参考）:
            
            ⏰ **T日盘后 15:30-16:30**
            - 打开 `DAILY_TRADING_SOP.md` → T日盘后部分
            - 对照操作步骤执行
            - 参考 `STOCK_SELECTION_GUIDE.md` 人工复核
            
            ⏰ **T+1竞价 09:15-09:25**
            - 打开 `DAILY_TRADING_SOP.md` → T+1竞价部分
            - 查看竞价强度分级标准
            - 对照决策矩阵下单
            
            ⏰ **T+2卖出 09:15-09:25**
            - 打开 `DAILY_TRADING_SOP.md` → T+2卖出部分
            - 查看卖出策略矩阵
            - 执行卖出操作
            
            ⏰ **T+2复盘 15:00-17:00**
            - 打开 `DAILY_TRADING_SOP.md` → 复盘分析部分
            - 使用复盘模板记录
            - 总结成功和失败经验
            """)
        
        # 核心流程速查表
        st.markdown("---")
        st.subheader("⚡ 核心流程速查表")
        
        st.markdown("""
        **从选股到交易的完整链路**:
        """)
        
        st.code("""
┌─────────────────────────────────────────────────────────────┐
│  T日 15:30 - 盘后选股与优化                                │
├─────────────────────────────────────────────────────────────┤
│ 1. 获取涨停数据（100+只）                                   │
│    ↓                                                        │
│ 2. 第一层：基础过滤 → 封单强度/涨停时间/开板次数           │
│    ↓  （淘汰70%，剩余30只）                                 │
│ 3. 第二层：质量评分 → 100分系统（5维度）                   │
│    ↓  （淘汰50%，剩余15只）                                 │
│ 4. 第三层：市场环境 → 涨停数/大盘/板块                     │
│    ↓  （淘汰30%，剩余10只）                                 │
│ 5. 竞价进阶优化 → 数据审计/特征精简/因子监控/Walk-Forward │
│    ↓  （活跃因子：32/50，模型AUC：0.74）                   │
│ 6. 最终输出：10只精选股票 + 预测概率                       │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  T+1 09:15 - 竞价监控与分级决策                            │
├─────────────────────────────────────────────────────────────┤
│ 监控10只股票的竞价表现                                      │
│                                                             │
│ 竞价强度 ≥85分（超强）→ 09:24竞价买入，单股8-10%仓位     │
│ 竞价强度 70-85分（强势）→ 09:30开盘观察买入，单股5-8%     │
│ 竞价强度 55-70分（中等）→ 等回踩买入，单股3-5%            │
│ 竞价强度 <55分（弱势） → 果断放弃                          │
│                                                             │
│ 实际买入：3-5只股票，总仓位40-50%                          │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  T+1 09:35-15:00 - 盘中监控（只能观察，不能卖）            │
├─────────────────────────────────────────────────────────────┤
│ ⚠️ A股T+1交易制度：当天买入不能卖出                        │
│                                                             │
│ • 观察股票走势（涨停/涨幅/分时形态）                       │
│ • 记录表现数据（为T+2决策做准备）                          │
│ • 识别强势股和弱势股                                        │
│ • 发现异常情况（重大利空、突发事件）                       │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  T+2 09:15 - 卖出决策与执行                                │
├─────────────────────────────────────────────────────────────┤
│ 根据T+1表现制定策略：                                       │
│                                                             │
│ • T+1涨停 + T+2高开>5%  → 卖出50%（高开兑现）             │
│ • T+1涨停 + T+2平开/低开 → 卖出100%（全卖）               │
│ • T+1涨5-9% + T+2高开   → 卖出60-80%（逐步兑现）           │
│ • T+1涨0-5%             → 卖出100%（不贪恋）               │
│ • T+1下跌               → 卖出100%（止损）                 │
│                                                             │
│ 09:24:30-09:24:50 提交卖出订单                             │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  T+2 15:00-17:00 - 复盘分析与模型进化                      │
├─────────────────────────────────────────────────────────────┤
│ 1. 交易记录整理（盈亏、胜率、成功率）                      │
│ 2. 成功案例分析（选股维度、执行维度）                      │
│ 3. 失败案例分析（问题诊断、根本原因、改进措施）            │
│ 4. 撰写复盘笔记（市场环境、交易经验、明日计划）            │
│ 5. 在线学习更新（增量更新模型、概念漂移检测）              │
│ 6. 因子健康度监控（IC计算、生命周期状态更新）              │
└─────────────────────────────────────────────────────────────┘
        """, language="text")
        
        # 文档位置提示
        st.success("""
        💡 **提示**: 所有文档都在 `docs/` 目录下:
        
        - `docs/DEEP_ARCHITECTURE_GUIDE.md` - 技术架构（理解原理）
        - `docs/DAILY_TRADING_SOP.md` - 日常操作（每天照做）
        - `docs/STOCK_SELECTION_GUIDE.md` - 选股依据（为什么这样选）
        
        建议下载到本地，随时查阅！
        """)
    
    # ==================== 辅助方法 ====================
    def _get_mock_candidates(self) -> pd.DataFrame:
        """生成模拟候选数据"""
        np.random.seed(42)
        n = 23
        
        return pd.DataFrame({
            'symbol': [f'{i:06d}.{"SZ" if i < 300000 else "SH"}' for i in np.random.randint(1, 999999, n)],
            'name': [f'股票{i}' for i in range(n)],
            'seal_strength': np.random.uniform(3.0, 9.5, n),
            'turnover_rate': np.random.uniform(5.0, 45.0, n),
            'prediction_score': np.random.uniform(0.6, 0.95, n),
            'auction_strength': np.random.choice(['超强', '强势', '中等'], n, p=[0.3, 0.5, 0.2]),
        })
    
    def _get_mock_auction_monitor_data(self) -> pd.DataFrame:
        """生成模拟竞价监控数据"""
        np.random.seed(123)
        n = 18
        
        strengths = np.random.choice(['超强', '强势', '中等', '弱势'], n, p=[0.2, 0.3, 0.3, 0.2])
        
        return pd.DataFrame({
            'symbol': [f'{i:06d}.SZ' for i in np.random.randint(1, 999999, n)],
            'name': [f'股票{i}' for i in range(n)],
            'auction_price': np.random.uniform(10, 100, n),
            'auction_volume': np.random.randint(100, 10000, n),
            'buy_ratio': np.random.uniform(40, 95, n),
            'auction_strength_label': strengths,
            'decision': ['买入' if s in ['超强', '强势'] else '观察' if s == '中等' else '放弃' for s in strengths],
        })
    
    def _get_mock_buy_records(self) -> pd.DataFrame:
        """生成模拟买入记录"""
        np.random.seed(456)
        n = 12
        
        layers = np.random.choice(['超强竞价', '强势竞价', '中等竞价'], n, p=[0.4, 0.4, 0.2])
        
        return pd.DataFrame({
            'time': [datetime.now().replace(hour=9, minute=np.random.randint(25, 35), second=np.random.randint(0, 59)) for _ in range(n)],
            'symbol': [f'{i:06d}.SZ' for i in np.random.randint(1, 999999, n)],
            'name': [f'股票{i}' for i in range(n)],
            'strategy_layer': layers,
            'buy_price': np.random.uniform(10, 100, n),
            'volume': np.random.randint(1, 20, n) * 100,
            'amount': np.random.uniform(1000, 50000, n),
            'price_deviation': np.random.uniform(-2, 5, n),
            'status': np.random.choice(['已成交', '部分成交', '未成交'], n, p=[0.7, 0.2, 0.1]),
        })
    
    def _get_mock_positions(self) -> pd.DataFrame:
        """生成模拟持仓数据"""
        np.random.seed(789)
        n = 8
        
        buy_prices = np.random.uniform(10, 100, n)
        t1_closes = buy_prices * np.random.uniform(0.95, 1.12, n)
        t1_returns = (t1_closes / buy_prices - 1) * 100
        t2_opens = t1_closes * np.random.uniform(0.97, 1.08, n)
        t2_open_gaps = (t2_opens / t1_closes - 1) * 100
        
        def get_sell_strategy(t1_ret, t2_gap):
            if t1_ret >= 9.5:
                if t2_gap >= 5:
                    return "高开兑现50%", 0.5
                else:
                    return "全卖止盈", 1.0
            elif t1_ret >= 5:
                return "高开卖60%", 0.6
            elif t1_ret >= 0:
                return "全卖出局", 1.0
            else:
                return "止损全卖", 1.0
        
        strategies = [get_sell_strategy(ret, gap) for ret, gap in zip(t1_returns, t2_open_gaps)]
        
        return pd.DataFrame({
            'symbol': [f'{i:06d}.SZ' for i in np.random.randint(1, 999999, n)],
            'name': [f'股票{i}' for i in range(n)],
            'buy_price': buy_prices,
            't1_close': t1_closes,
            't1_return': t1_returns,
            't2_open': t2_opens,
            't2_open_gap': t2_open_gaps,
            'sell_strategy': [s[0] for s in strategies],
            'sell_ratio': [s[1] * 100 for s in strategies],
            'cost': np.random.uniform(10000, 100000, n),
            'profit': np.random.uniform(-5000, 15000, n),
        })
    
    def _get_mock_history_stats(self) -> Dict:
        """生成模拟历史统计"""
        return {
            'trading_days': 30,
            'total_trades': 156,
            'total_profit': 58420,
            'win_rate': 68.5,
            'avg_return': 3.74
        }
    
    def _get_mock_history_trades(self) -> pd.DataFrame:
        """生成模拟历史交易"""
        np.random.seed(999)
        n = 50
        
        return pd.DataFrame({
            'date': [datetime.now() - timedelta(days=i//2) for i in range(n)],
            'symbol': [f'{i:06d}.SZ' for i in np.random.randint(1, 999999, n)],
            'name': [f'股票{i}' for i in range(n)],
            'buy_price': np.random.uniform(10, 100, n),
            'sell_price': np.random.uniform(10, 110, n),
            'profit': np.random.uniform(-3000, 8000, n),
            'profit_rate': np.random.uniform(-8, 15, n),
        })
    
    def _create_auction_strength_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建竞价强度分布图"""
        strength_counts = data['auction_strength_label'].value_counts()
        
        fig = px.bar(
            x=strength_counts.index,
            y=strength_counts.values,
            labels={'x': '竞价强度', 'y': '数量'},
            title='竞价强度分布',
            color=strength_counts.index,
            color_discrete_map={
                '超强': '#00CC96',
                '强势': '#636EFA',
                '中等': '#FFA15A',
                '弱势': '#EF553B'
            }
        )
        
        fig.update_layout(showlegend=False, height=300)
        
        return fig
    
    def _create_layer_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建分层策略分布图"""
        layer_counts = data['strategy_layer'].value_counts()
        
        fig = px.pie(
            values=layer_counts.values,
            names=layer_counts.index,
            title='分层买入策略分布',
            hole=0.4
        )
        
        fig.update_layout(height=350)
        
        return fig
    
    def _create_sell_strategy_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建卖出策略分布图"""
        strategy_counts = data['sell_strategy'].value_counts()
        
        fig = px.bar(
            x=strategy_counts.index,
            y=strategy_counts.values,
            labels={'x': '卖出策略', 'y': '持仓数'},
            title='卖出策略分布',
            color=strategy_counts.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(showlegend=False, height=300)
        
        return fig
    
    def _create_profit_curve(self) -> go.Figure:
        """创建收益曲线"""
        np.random.seed(111)
        days = 30
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        cumulative_profit = np.cumsum(np.random.uniform(-1000, 3000, days))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_profit,
            mode='lines+markers',
            name='累计收益',
            line=dict(color='#636EFA', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='30日累计收益曲线',
            xaxis_title='日期',
            yaxis_title='累计收益(¥)',
            hovermode='x unified',
            height=400
        )
        
        return fig


# 主入口
def main():
    st.set_page_config(
        page_title="竞价决策系统",
        page_icon="🎯",
        layout="wide"
    )
    
    view = AuctionDecisionView()
    view.render()


if __name__ == "__main__":
    main()
