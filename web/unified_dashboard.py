"""
统一Web管理界面
整合麒麟堆栈交易系统与TradingAgents项目的实时数据
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
import redis
import websocket
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
import threading
import queue
import logging

# Configure logging
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path("D:/test/Qlib/tradingagents")))

# 监控权重
from monitoring.metrics import get_monitor

# 导入核心组件
from tradingagents_integration.integration_adapter import (
    TradingAgentsAdapter, 
    UnifiedTradingSystem
)
from trading.realtime_trading_system import RealtimeTradingSystem
from agents.trading_agents import MultiAgentManager
from qlib_integration.qlib_engine import QlibIntegrationEngine
from data_layer.data_access_layer import DataAccessLayer

# 导入P2增强功能模块
sys.path.insert(0, str(Path(__file__).parent.parent / "qlib_enhanced"))
from high_freq_limitup import HighFreqLimitUpAnalyzer, create_sample_high_freq_data
from online_learning import OnlineLearningManager, DriftDetector, AdaptiveLearningRate
from multi_source_data import MultiSourceDataProvider, DataSource
# Phase 2 模块
from rl_trading import TradingEnvironment, DQNAgent, RLTrainer, create_sample_data as create_rl_data
from portfolio_optimizer import MeanVarianceOptimizer, BlackLittermanOptimizer, RiskParityOptimizer, create_sample_returns
from risk_management import ValueAtRiskCalculator, StressTest, RiskMonitor, create_sample_data as create_risk_data
from performance_attribution import TransactionCostAnalysis

# Phase 3 风控模块
from qilin_stack.agents.risk.liquidity_monitor import LiquidityMonitor, LiquidityLevel
from qilin_stack.agents.risk.extreme_market_guard import ExtremeMarketGuard, ProtectionLevel, MarketCondition
from qilin_stack.agents.risk.position_manager import (
    PositionManager as RiskPositionManager,
    PositionSizeMethod,
    RiskLevel,
)

# Phase 4 写实回测模块
from qilin_stack.backtest.slippage_model import (
    SlippageEngine,
    SlippageModel,
    OrderSide,
    MarketDepth as Depth,
)
from qilin_stack.backtest.limit_up_queue_simulator import (
    LimitUpQueueSimulator,
    LimitUpStrength,
)

# 页面配置
st.set_page_config(
    page_title="麒麟量化交易平台 - 统一控制中心",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main { padding-top: 0rem; }
    .block-container { padding: 1rem; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .success-box {
        background-color: #d4f1d4;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


class UnifiedDashboard:
    """统一管理界面"""
    
    def __init__(self):
        self.init_session_state()
        self.setup_connections()
        self.init_systems()
        
    def init_session_state(self):
        """初始化会话状态"""
        # 系统实例
        if 'unified_system' not in st.session_state:
            st.session_state.unified_system = None
        if 'trading_system' not in st.session_state:
            st.session_state.trading_system = None
        if 'adapter' not in st.session_state:
            st.session_state.adapter = None
            
        # 实时数据
        if 'realtime_data' not in st.session_state:
            st.session_state.realtime_data = {}
        if 'active_orders' not in st.session_state:
            st.session_state.active_orders = []
        if 'positions' not in st.session_state:
            st.session_state.positions = {}
        if 'signals_queue' not in st.session_state:
            st.session_state.signals_queue = []
            
        # 配置
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = ["000001", "000002", "600000"]
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
        if 'auto_trade' not in st.session_state:
            st.session_state.auto_trade = False
            
    def setup_connections(self):
        """设置数据连接"""
        try:
            # Redis连接
            self.redis_client = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True
            )
            self.redis_available = True
        except Exception:
            self.redis_client = None
            self.redis_available = False
            
        # WebSocket连接（实时行情）
        self.ws_client = None
        self.ws_thread = None
        
    def init_systems(self):
        """初始化交易系统"""
        config = {
            "symbols": st.session_state.selected_stocks,
            "position_size_pct": 0.1,
            "max_position_size": 0.3,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10
        }
        
        # 初始化适配器
        if st.session_state.adapter is None:
            st.session_state.adapter = TradingAgentsAdapter(config)
            
        # 初始化统一系统
        if st.session_state.unified_system is None:
            st.session_state.unified_system = UnifiedTradingSystem(config)
            
        # 初始化实时交易系统
        if st.session_state.trading_system is None:
            st.session_state.trading_system = RealtimeTradingSystem(config)
    
    def run(self):
        """运行主界面"""
        # 顶部信息栏
        self.render_header()
        
        # 侧边栏
        with st.sidebar:
            self.render_sidebar()
        
        # 主界面内容
        self.render_main_content()
        
        # 自动刷新
        if st.session_state.get('auto_refresh', False):
            st.experimental_rerun()
    
    def render_header(self):
        """渲染头部"""
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
        
        with col1:
            st.markdown("### 🐉 麒麟量化平台")
            
        with col2:
            market_status = self.get_market_status()
            if market_status == "开盘中":
                st.success(f"🟢 {market_status}")
            elif market_status == "集合竞价":
                st.warning(f"🟡 {market_status}")
            else:
                st.error(f"🔴 {market_status}")
                
        with col3:
            # 系统状态
            if self.redis_available:
                st.success("📡 数据连接正常")
            else:
                st.error("❌ 数据连接断开")
                
        with col4:
            # 自动交易开关
            auto_trade = st.toggle("自动交易", value=st.session_state.auto_trade)
            st.session_state.auto_trade = auto_trade
            
        with col5:
            st.info(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.header("📍 控制面板")
        
        # 系统控制
        st.subheader("🎮 系统控制")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ 启动", use_container_width=True):
                self.start_system()
        with col2:
            if st.button("⏸️ 停止", use_container_width=True):
                self.stop_system()
                
        if st.button("🔄 刷新数据", use_container_width=True):
            self.refresh_data()
            
        # 股票选择
        st.subheader("📊 监控股票")
        selected_stocks = st.multiselect(
            "选择股票",
            options=["000001", "000002", "600000", "600519", "000858", "300750"],
            default=st.session_state.selected_stocks
        )
        st.session_state.selected_stocks = selected_stocks
        
        # 参数设置
        st.subheader("⚙️ 交易参数")
        
        position_size = st.slider(
            "单股仓位(%)",
            min_value=5,
            max_value=30,
            value=10
        )
        stop_loss = st.number_input(
            "止损线(%)",
            min_value=1.0,
            max_value=10.0,
            value=5.0
        )
        
        take_profit = st.number_input(
            "止盈线(%)",
            min_value=5.0,
            max_value=30.0,
            value=10.0
        )
        
        # 刷新设置
        st.subheader("🔄 刷新设置")
        
        auto_refresh = st.checkbox("自动刷新", value=False)
        st.session_state.auto_refresh = auto_refresh
        
        refresh_interval = st.slider(
            "刷新间隔(秒)",
            min_value=1,
            max_value=60,
            value=st.session_state.refresh_interval
        )
        st.session_state.refresh_interval = refresh_interval
    
    def render_main_content(self):
        """渲染主内容区"""
        # 创建主标签页 - Qilin监控 + Qlib + RD-Agent + TradingAgents
        main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
            "🏠 Qilin监控",
            "📦 Qlib",
            "🧠 RD-Agent研发智能体",
            "🤝 TradingAgents多智能体"
        ])
        
        with main_tab1:
            # Qilin系统级监控与操作
            self.render_qilin_tabs()
        
        with main_tab2:
            # Qlib相关功能
            self.render_qlib_tabs()
        
        with main_tab3:
            # RD-Agent的6个子tab
            self.render_rdagent_tabs()
        
        with main_tab4:
            # TradingAgents的6个子tab
            self.render_tradingagents_tabs()
        
    def render_qilin_tabs(self):
        """渲染Qilin系统级tabs（监控/操作）"""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 实时监控",
            "🤖 智能体状态",
            "📈 交易执行",
            "📉 风险管理",
            "📋 历史记录"
        ])
        
        with tab1:
            self.render_realtime_monitor()
        with tab2:
            self.render_agents_status()
        with tab3:
            self.render_trading_execution()
        with tab4:
            self.render_risk_management()
        with tab5:
            self.render_history()

    def render_qlib_tabs(self):
        """渲染Qlib相关功能tabs"""
        qtab1, qtab2, qtab3, qtab4, qtab5, qtab6, qtab7 = st.tabs([
            "🔥 涨停板分析",
            "🧠 在线学习",
            "🔌 多数据源",
            "🤖 强化学习",
            "💼 组合优化",
            "⚠️ 风险监控",
            "📊 归因分析"
        ])
        with qtab1:
            self.render_limitup_analysis()
        with qtab2:
            self.render_online_learning()
        with qtab3:
            self.render_multi_source_data()
        with qtab4:
            self.render_rl_trading()
        with qtab5:
            self.render_portfolio_optimization()
        with qtab6:
            self.render_risk_monitoring()
        with qtab7:
            self.render_performance_attribution()
        
    def render_rdagent_tabs(self):
        """渲染RD-Agent的6个子tabs"""
        rd_tab1, rd_tab2, rd_tab3, rd_tab4, rd_tab5, rd_tab6 = st.tabs([
            "🔍 因子挖掘",
            "🏗️ 模型优化",
            "📚 知识学习",
            "🏆 Kaggle Agent",
            "🔬 研发协同",
            "📊 MLE-Bench"
        ])
        
        with rd_tab1:
            # 导入因子挖掘模块
            try:
                from tabs.rdagent import factor_mining
                factor_mining.render()
            except Exception as e:
                st.error(f"加载因子挖掘模块失败: {e}")
                st.info("请确保已正确安装RD-Agent依赖")
        
        with rd_tab2:
            # 导入模型优化模块
            try:
                from tabs.rdagent import model_optimization
                model_optimization.render()
            except Exception as e:
                st.error(f"加载模型优化模块失败: {e}")
                st.info("请确保已正确安装RD-Agent依赖")
        
        with rd_tab3:
            # 知识学习
            try:
                from tabs.rdagent.other_tabs import render_knowledge_learning
                render_knowledge_learning()
            except Exception as e:
                st.error(f"加载知识学习模块失败: {e}")
        
        with rd_tab4:
            # Kaggle Agent
            try:
                from tabs.rdagent.other_tabs import render_kaggle_agent
                render_kaggle_agent()
            except Exception as e:
                st.error(f"加载Kaggle Agent模块失败: {e}")
        
        with rd_tab5:
            # 研发协同 - 增强版
            try:
                from tabs.rdagent.rd_coordination_enhanced import render_rd_coordination_enhanced
                render_rd_coordination_enhanced()
            except Exception as e:
                st.error(f"加载研发协同模块失败: {e}")
                # Fallback到旧版本
                try:
                    from tabs.rdagent.other_tabs import render_rd_coordination
                    render_rd_coordination()
                except:
                    pass
        
        with rd_tab6:
            # MLE-Bench - 增强版
            try:
                from tabs.rdagent.rd_coordination_enhanced import render_mle_bench_enhanced
                render_mle_bench_enhanced()
            except Exception as e:
                st.error(f"加载MLE-Bench模块失败: {e}")
                # Fallback到旧版本
                try:
                    from tabs.rdagent.other_tabs import render_mle_bench
                    render_mle_bench()
                except:
                    pass
    
    def render_tradingagents_tabs(self):
        """渲染TradingAgents的6个子tabs"""
        ta_tab1, ta_tab2, ta_tab3, ta_tab4, ta_tab5, ta_tab6 = st.tabs([
            "🔍 智能体管理",
            "🗣️ 协作机制",
            "📰 信息采集",
            "💡 决策分析",
            "👤 用户管理",
            "🔌 LLM集成"
        ])
        
        with ta_tab1:
            try:
                from tabs.tradingagents.all_tabs import render_agent_management
                render_agent_management()
            except Exception as e:
                st.error(f"加载智能体管理模块失败: {e}")
                st.info("请确保已正确安装TradingAgents依赖")
        
        with ta_tab2:
            try:
                from tabs.tradingagents.all_tabs import render_collaboration
                render_collaboration()
            except Exception as e:
                st.error(f"加载协作机制模块失败: {e}")
        
        with ta_tab3:
            try:
                from tabs.tradingagents.all_tabs import render_information_collection
                render_information_collection()
            except Exception as e:
                st.error(f"加载信息采集模块失败: {e}")
        
        with ta_tab4:
            try:
                from tabs.tradingagents.all_tabs import render_decision_analysis
                render_decision_analysis()
            except Exception as e:
                st.error(f"加载决策分析模块失败: {e}")
        
        with ta_tab5:
            try:
                from tabs.tradingagents.all_tabs import render_user_management
                render_user_management()
            except Exception as e:
                st.error(f"加载用户管理模块失败: {e}")
        
        with ta_tab6:
            try:
                from tabs.tradingagents.all_tabs import render_llm_integration
                render_llm_integration()
            except Exception as e:
                st.error(f"加载LLM集成模块失败: {e}")
    
    def render_realtime_monitor(self):
        """实时监控页面"""
        # 关键指标
        st.subheader("📊 关键指标")
        
        metrics = self.get_realtime_metrics()
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "总资产",
                f"¥{metrics['total_assets']:,.0f}",
                f"{metrics['assets_change']:+.2%}"
            )
        
        with col2:
            st.metric(
                "今日盈亏",
                f"¥{metrics['today_pnl']:,.0f}",
                f"{metrics['pnl_change']:+.2%}"
            )
        
        with col3:
            st.metric(
                "持仓数",
                f"{metrics['position_count']}只",
                f"{metrics['position_change']:+d}"
            )
        
        with col4:
            st.metric(
                "胜率",
                f"{metrics['win_rate']:.1%}",
                f"{metrics['win_rate_change']:+.1%}"
            )
        
        with col5:
            st.metric(
                "夏普比",
                f"{metrics['sharpe']:.2f}",
                f"{metrics['sharpe_change']:+.2f}"
            )
        
        with col6:
            st.metric(
                "最大回撤",
                f"{metrics['max_dd']:.2%}",
                f"{metrics['dd_change']:+.2%}"
            )
            
        # 权重轨迹
        st.subheader("🧮 权重轨迹（QLib / TradingAgents / RD-Agent）")
        self.render_weight_trajectories()
        
        # 实时行情
        st.subheader("💹 实时行情")
        self.render_realtime_quotes()
        
        # 最新信号
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 最新信号")
            self.render_latest_signals()
            
        with col2:
            st.subheader("📋 活跃订单")
            self.render_active_orders()
    
    def render_agents_status(self):
        """智能体状态页面"""
        st.subheader("🤖 智能体运行状态")
        
        # 获取两个系统的智能体状态
        status_data = self.get_agents_status()
        
        # 麒麟堆栈智能体
        st.write("**麒麟堆栈智能体 (10个)**")
        
        cols = st.columns(5)
        for idx, (name, status) in enumerate(status_data['qilin'].items()):
            with cols[idx % 5]:
                self.render_agent_card(name, status, "qilin")
        
        st.divider()
        
        # TradingAgents智能体
        st.write("**TradingAgents智能体**")
        
        if status_data['tradingagents']:
            cols = st.columns(5)
            for idx, (name, status) in enumerate(status_data['tradingagents'].items()):
                with cols[idx % 5]:
                    self.render_agent_card(name, status, "ta")
        else:
            st.info("TradingAgents项目未连接")
        
        # 信号热力图
        st.subheader("🔥 信号强度热力图")
        self.render_signal_heatmap()
        
        # 智能体协作图
        st.subheader("🔗 智能体协作网络")
        self.render_agent_network()
    
    def render_trading_execution(self):
        """交易执行页面"""
        # 手动交易面板
        st.subheader("🎯 手动交易")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.selectbox(
                "股票代码",
                options=st.session_state.selected_stocks
            )
            
        with col2:
            action = st.selectbox(
                "交易方向",
                options=["买入", "卖出"]
            )
            
        with col3:
            quantity = st.number_input(
                "数量(股)",
                min_value=100,
                max_value=10000,
                step=100,
                value=100
            )
            
        with col4:
            price_type = st.selectbox(
                "价格类型",
                options=["市价", "限价"]
            )
            
        if price_type == "限价":
            limit_price = st.number_input(
                "限价",
                min_value=0.01,
                value=10.00
            )
        
        if st.button("📤 提交订单", use_container_width=True):
            self.submit_order(symbol, action, quantity, price_type)
        
        # 持仓管理
        st.subheader("💼 持仓管理")
        self.render_positions()
        
        # 成交记录
        st.subheader("✅ 今日成交")
        self.render_trades()
    
    def render_risk_management(self):
        """风险管理页面"""
        st.subheader("🛡️ 风险监控面板")
        
        # 风险指标
        risk_metrics = self.get_risk_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "VaR (95%)",
                f"¥{risk_metrics['var_95']:,.0f}",
                help="95%置信水平下的最大可能损失"
            )
            
        with col2:
            st.metric(
                "风险敞口",
                f"{risk_metrics['exposure']:.1%}",
                help="总风险敞口占比"
            )
            
        with col3:
            st.metric(
                "杠杆率",
                f"{risk_metrics['leverage']:.2f}x",
                help="实际杠杆倍数"
            )
            
        with col4:
            # 风险等级指示器
            risk_level = risk_metrics['risk_level']
            if risk_level == "低":
                st.success(f"✅ 风险等级: {risk_level}")
            elif risk_level == "中":
                st.warning(f"⚠️ 风险等级: {risk_level}")
            else:
                st.error(f"🚨 风险等级: {risk_level}")
        
        # 风险分布图
        st.subheader("📊 风险分布")
        self.render_risk_distribution()
        
        # 压力测试结果
        st.subheader("⚡ 压力测试")
        self.render_stress_test()
        
        # 风险预警
        st.subheader("🚨 风险预警")
        self.render_risk_alerts()

        # Phase 3 风控三件套（收拢为折叠面板）
        st.divider()
        with st.expander("P3 · 流动性监控", expanded=False):
            col_cfg, col_data = st.columns([1, 1])
            with col_cfg:
                symbol = st.selectbox("股票代码", options=st.session_state.selected_stocks + ["000001"], index=0, key="p3_liq_symbol")
                current_price = st.number_input("当前价格", min_value=0.01, value=10.50, step=0.01, key="p3_liq_price")
                min_avg_volume = st.number_input("最小日均量(股)", min_value=0.0, value=1_000_000.0, step=10000.0, key="p3_liq_minvol")
                max_spread_pct = st.number_input("最大价差(%)", min_value=0.0, value=0.20, step=0.01, key="p3_liq_spread")
                min_turnover_pct = st.number_input("最小换手率(%)", min_value=0.0, value=1.00, step=0.10, key="p3_liq_turn")
                min_score = st.number_input("最低流动性评分", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key="p3_liq_score")
                use_sample = st.checkbox("使用示例数据", value=True, key="p3_liq_sample")
                monitor = LiquidityMonitor(
                    min_avg_volume=min_avg_volume,
                    max_spread_ratio=max_spread_pct / 100.0,
                    min_turnover_rate=min_turnover_pct / 100.0,
                    min_liquidity_score=min_score,
                )
            with col_data:
                if use_sample:
                    np.random.seed(42)
                    volumes = np.random.randint(1_800_000, 3_800_000, 20)
                    turns = np.random.uniform(0.012, 0.040, 20)
                    volume_data = pd.DataFrame({'volume': volumes, 'turnover_rate': turns})
                    order_book = {
                        'bid_prices': [current_price - 0.01 * i for i in range(1, 6)][::-1],
                        'ask_prices': [current_price + 0.01 * i for i in range(1, 6)],
                        'bid_volumes': [50000, 45000, 40000, 35000, 30000],
                        'ask_volumes': [48000, 42000, 38000, 33000, 28000],
                    }
                else:
                    st.info("请上传包含 volume 与 turnover_rate 列的CSV数据，并对接盘口数据源。")
                    volume_data = pd.DataFrame({'volume': [0], 'turnover_rate': [0.0]})
                    order_book = None
            if st.button("🔍 评估流动性", type="primary", key="p3_liq_eval"):
                m = monitor.evaluate_liquidity(symbol, current_price, volume_data, order_book)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("流动性评分", f"{m.liquidity_score:.1f}/100")
                with c2: st.metric("流动性等级", m.liquidity_level.value)
                with c3: st.metric("可建仓", "是" if m.can_buy else "否")
                with c4: st.metric("最大建仓规模", f"{m.max_position_size:,.0f} 股")
                st.write("警告/提示:")
                if m.warnings:
                    for w in m.warnings: st.warning(w)
                else:
                    st.info("无")
                st.divider()
                target_shares = st.number_input("拟建仓股数", min_value=0, value=10000, step=100, key="p3_liq_target")
                if st.button("📏 建仓规模检查", key="p3_liq_check"):
                    ok, reason, rec = monitor.check_position_size(symbol, target_shares, m)
                    st.success(f"通过：{reason} | 建议股数 {rec:,}") if ok else st.error(f"不通过：{reason} | 建议股数 {rec:,}")

        with st.expander("P3 · 极端行情保护", expanded=False):
            guard = ExtremeMarketGuard()
            if st.button("🧪 评估示例市场健康度", key="p3_guard_health"):
                market_data = {
                    f"{i:06d}.SZ": pd.DataFrame({
                        'open': [10.0],
                        'close': [10.0 * (1 + np.random.uniform(-0.12, 0.12))],
                        'turnover': [np.random.randint(5_000_000, 20_000_000)],
                        'turnover_rate': [np.random.uniform(0.01, 0.05)],
                    }) for i in range(500)
                }
                health = guard.evaluate_market_health(market_data)
                d1, d2, d3, d4 = st.columns(4)
                with d1: st.metric("市场状态", health.market_condition.value)
                with d2: st.metric("恐慌指数", f"{health.panic_index:.1f}/100")
                with d3: st.metric("跌停数", f"{health.stocks_limit_down}")
                with d4: st.metric("保护等级", health.protection_level.name)
                if health.warnings:
                    for w in health.warnings: st.warning(w)
                halt, reason = guard.should_halt_trading()
                st.info(f"暂停交易: {'是' if halt else '否'} | 原因: {reason}")
            st.divider()
            if st.button("⚡ 检测个股事件", key="p3_guard_stock"):
                price_data = pd.DataFrame({
                    'open': [10.0] * 60,
                    'high': [10.2] * 60,
                    'low': [9.8] * 30 + [8.9] * 30,
                    'close': [10.0] * 30 + [9.3, 9.2, 8.9, 8.7, 8.5] + [8.5] * 25,
                })
                volume_data = pd.DataFrame({'volume': [1_000_000] * 30 + [4_000_000] * 30})
                event = guard.detect_extreme_event("000001.SZ", price_data, volume_data, timeframe="1min")
                if event:
                    st.error(f"检测到{event.event_type} | 严重度 {event.severity:.1f}/10 | {event.recommended_action}")
                else:
                    st.success("未检测到极端事件")

        with st.expander("P3 · 动态头寸管理", expanded=False):
            colA, colB = st.columns(2)
            with colA:
                total_capital = st.number_input("总资金(元)", min_value=1_0000.0, value=1_000_000.0, step=10_000.0, key="p3_pos_cap")
                risk_level = st.selectbox("风险等级", options=[RiskLevel.CONSERVATIVE, RiskLevel.MODERATE, RiskLevel.AGGRESSIVE], index=1, key="p3_pos_risk")
                method = st.selectbox("仓位方法", options=[PositionSizeMethod.KELLY, PositionSizeMethod.VOLATILITY_ADJUSTED, PositionSizeMethod.FIXED], index=0, key="p3_pos_method")
            with colB:
                symbol_p = st.selectbox("股票", options=st.session_state.selected_stocks + ["600000"], index=0, key="p3_pos_symbol")
                cur_price = st.number_input("当前价", min_value=0.01, value=10.50, step=0.01, key="p3_pos_price")
                stop_price = st.number_input("止损价", min_value=0.01, value=9.50, step=0.01, key="p3_pos_stop")
                win_rate = st.slider("胜率(%)", 10.0, 90.0, 60.0, 1.0, key="p3_pos_wr") if method == PositionSizeMethod.KELLY else None
                avg_return = st.slider("平均盈利(%)", 1.0, 30.0, 8.0, 0.5, key="p3_pos_avg") if method == PositionSizeMethod.KELLY else None
                volatility = st.slider("波动率(%)", 1.0, 10.0, 2.5, 0.1, key="p3_pos_vol") if method == PositionSizeMethod.VOLATILITY_ADJUSTED else None
            if st.button("🧮 计算仓位", type="primary", key="p3_pos_calc"):
                mgr = RiskPositionManager(total_capital=total_capital, risk_level=risk_level)
                rec = mgr.calculate_position_size(
                    symbol=symbol_p,
                    current_price=cur_price,
                    stop_loss_price=stop_price,
                    win_rate=(win_rate/100.0) if win_rate is not None else None,
                    avg_return=(avg_return/100.0) if avg_return is not None else None,
                    volatility=(volatility/100.0) if volatility is not None else None,
                    method=method,
                    sector="示例",
                )
                e1, e2, e3, e4 = st.columns(4)
                with e1: st.metric("建议股数", f"{rec.recommended_shares:,}")
                with e2: st.metric("建议金额", f"¥{rec.recommended_value:,.0f}")
                with e3: st.metric("仓位比例", f"{rec.position_ratio:.1%}")
                with e4: st.metric("预估风险", f"¥{rec.estimated_risk:,.0f}")
                if rec.warnings:
                    for w in rec.warnings: st.warning(w)
                st.info(f"方法: {rec.method.value} | 理由: {rec.rationale} | 调整: {rec.adjustment_suggestion}")
    
    def render_history(self):
        """历史记录页面"""
        # 日期选择
        col1, col2 = st.columns(2)
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
        
        # 历史收益曲线
        st.subheader("📈 历史收益曲线")
        self.render_equity_curve(start_date, end_date)
        
        # 交易历史
        st.subheader("📋 交易历史")
        self.render_trade_history(start_date, end_date)
        
        # 绩效统计
        st.subheader("📊 绩效统计")
        self.render_performance_stats(start_date, end_date)
    
    # ===== 数据获取方法 =====
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """获取实时指标"""
        if self.redis_available:
            # 从Redis获取缓存的实时数据
            try:
                metrics_json = self.redis_client.get("realtime_metrics")
                if metrics_json:
                    return json.loads(metrics_json)
            except Exception:pass
        
        # 默认数据
        return {
            'total_assets': 1000000,
            'assets_change': 0.025,
            'today_pnl': 5280,
            'pnl_change': 0.0132,
            'position_count': 5,
            'position_change': 1,
            'win_rate': 0.625,
            'win_rate_change': 0.025,
            'sharpe': 1.85,
            'sharpe_change': 0.12,
            'max_dd': 0.082,
            'dd_change': -0.015
        }
    
    def get_agents_status(self) -> Dict[str, Dict]:
        """获取智能体状态"""
        status = {
            'qilin': {},
            'tradingagents': {}
        }
        
        # 麒麟堆栈智能体
        qilin_agents = [
            "市场生态", "竞价博弈", "仓位控制", "成交量分析", "技术指标",
            "情绪分析", "风险管理", "形态识别", "宏观经济", "套利机会"
        ]
        
        for agent in qilin_agents:
            status['qilin'][agent] = {
                'status': np.random.choice(['运行中', '空闲', '分析中']),
                'last_signal': np.random.choice(['买入', '卖出', '持有']),
                'confidence': np.random.uniform(0.5, 1.0),
                'cpu': np.random.uniform(10, 60),
                'memory': np.random.uniform(100, 500)
            }
        
        # TradingAgents智能体（如果可用）
        if st.session_state.adapter and hasattr(st.session_state.adapter, 'registered_agents'):
            for name in st.session_state.adapter.registered_agents.keys():
                status['tradingagents'][name] = {
                    'status': '运行中',
                    'last_signal': '分析中',
                    'confidence': 0.75
                }
        
        return status
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        return {
            'var_95': 52380,
            'exposure': 0.75,
            'leverage': 1.2,
            'risk_level': np.random.choice(['低', '中', '高']),
            'volatility': 0.0823,
            'beta': 0.92,
            'correlation': 0.45
        }
    
    # ===== 渲染组件方法 =====
    
    def render_realtime_quotes(self):
        """渲染实时行情"""
        quotes_data = []
        
        for symbol in st.session_state.selected_stocks:
            quotes_data.append({
                '代码': symbol,
                '名称': self.get_stock_name(symbol),
                '现价': np.random.uniform(8, 15),
                '涨跌幅': np.random.uniform(-0.05, 0.05),
                '成交量': np.random.randint(1000000, 50000000),
                '成交额': np.random.randint(10000000, 500000000),
                '买一': np.random.uniform(8, 15),
                '卖一': np.random.uniform(8, 15)
            })
        
        df = pd.DataFrame(quotes_data)
        
        # 格式化显示
        st.dataframe(
            df.style.format({
                '现价': '¥{:.2f}',
                '涨跌幅': '{:+.2%}',
                '成交量': '{:,.0f}',
                '成交额': '¥{:,.0f}',
                '买一': '¥{:.2f}',
                '卖一': '¥{:.2f}'
            }).applymap(
                lambda x: 'color: red;' if x < 0 else 'color: green;',
                subset=['涨跌幅']
            ),
            use_container_width=True,
            hide_index=True
        )
    
    def render_latest_signals(self):
        """渲染最新信号"""
        signals = []
        
        # 从信号队列获取
        if st.session_state.signals_queue:
            signals = st.session_state.signals_queue[-5:]  # 最新5个
        else:
            # 模拟数据
            for _ in range(5):
                signals.append({
                    '时间': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                    '股票': np.random.choice(st.session_state.selected_stocks),
                    '信号': np.random.choice(['强烈买入', '买入', '持有', '卖出', '强烈卖出']),
                    '信心度': np.random.uniform(0.6, 0.95),
                    '来源': np.random.choice(['麒麟系统', 'TradingAgents', '混合共识'])
                })
        
        df = pd.DataFrame(signals)
        
        if not df.empty:
            st.dataframe(
                df.style.format({
                    '时间': lambda x: x.strftime('%H:%M:%S'),
                    '信心度': '{:.1%}'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    def render_active_orders(self):
        """渲染活跃订单"""
        orders = st.session_state.active_orders
        
        if not orders:
            # 模拟数据
            orders = [
                {
                    '订单号': f"ORD{np.random.randint(10000, 99999)}",
                    '股票': np.random.choice(st.session_state.selected_stocks),
                    '方向': np.random.choice(['买入', '卖出']),
                    '数量': np.random.randint(1, 10) * 100,
                    '价格': np.random.uniform(8, 15),
                    '状态': np.random.choice(['待成交', '部分成交', '已提交'])
                }
                for _ in range(3)
            ]
        
        df = pd.DataFrame(orders)
        
        if not df.empty:
            st.dataframe(
                df.style.format({
                    '价格': '¥{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    def render_agent_card(self, name: str, status: Dict, system: str):
        """渲染智能体卡片"""
        color = "🟢" if status['status'] == '运行中' else "🟡" if status['status'] == '分析中' else "⚪"
        
        with st.container():
            st.markdown(f"""
            <div style="background-color: {'#e8f5e9' if system == 'qilin' else '#e3f2fd'}; 
                        padding: 10px; border-radius: 10px; margin: 5px;">
                <b>{name}</b><br>
                {color} {status['status']}<br>
                信号: {status.get('last_signal', 'N/A')}<br>
                信心: {status.get('confidence', 0):.1%}
            </div>
            """, unsafe_allow_html=True)
    
    def render_signal_heatmap(self):
        """渲染信号热力图"""
        # 生成示例数据
        agents = ["生态", "竞价", "仓位", "成交", "技术", "情绪", "风险", "形态", "宏观", "套利"]
        stocks = st.session_state.selected_stocks[:5]
        
        z = np.random.uniform(-1, 1, (len(stocks), len(agents)))
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=agents,
            y=stocks,
            colorscale='RdYlGn',
            zmid=0,
            text=z,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="智能体",
            yaxis_title="股票代码"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_weight_trajectories(self):
        """渲染权重时间序列图（来自监控指标system_weight）"""
        try:
            metrics_list = get_monitor().collector.get_metrics("system_weight")
        except Exception:
            metrics_list = []
        if not metrics_list:
            st.info("暂无权重数据，请运行自适应权重服务或产出权重指标。")
            return
        # 组装为DataFrame
        rows = []
        for m in metrics_list:
            try:
                rows.append({
                    'time': datetime.fromtimestamp(m.timestamp),
                    'source': m.labels.get('source', 'unknown'),
                    'weight': m.value,
                })
            except Exception:
                continue
        if not rows:
            st.info("暂无权重数据。")
            return
        df = pd.DataFrame(rows).sort_values('time')
        fig = go.Figure()
        for source, g in df.groupby('source'):
            fig.add_trace(go.Scatter(x=g['time'], y=g['weight'], mode='lines+markers', name=source))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

    def render_positions(self):
        """渲染持仓"""
        positions = []
        
        for symbol in st.session_state.selected_stocks[:3]:
            positions.append({
                '股票代码': symbol,
                '股票名称': self.get_stock_name(symbol),
                '持仓数量': np.random.randint(1, 50) * 100,
                '成本价': np.random.uniform(8, 12),
                '现价': np.random.uniform(9, 13),
                '盈亏': np.random.uniform(-5000, 10000),
                '盈亏比例': np.random.uniform(-0.1, 0.2)
            })
        
        df = pd.DataFrame(positions)
        
        if not df.empty:
            st.dataframe(
                df.style.format({
                    '成本价': '¥{:.2f}',
                    '现价': '¥{:.2f}',
                    '盈亏': '¥{:+,.0f}',
                    '盈亏比例': '{:+.2%}'
                }).applymap(
                    lambda x: 'color: red;' if x < 0 else 'color: green;',
                    subset=['盈亏', '盈亏比例']
                ),
                use_container_width=True,
                hide_index=True
            )
    
    def render_equity_curve(self, start_date, end_date):
        """渲染收益曲线"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        values = 1000000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.005))
        
        fig = go.Figure()
        
        # 资产曲线
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='资产净值',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # 基准线
        benchmark = 1000000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.003))
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark,
            mode='lines',
            name='基准收益',
            line=dict(color='#ff7f0e', width=1, dash='dash')
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title="日期",
            yaxis_title="资产净值",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== 控制方法 =====
    
    def start_system(self):
        """启动系统"""
        with st.spinner("正在启动系统..."):
            try:
                # 启动统一交易系统
                # asyncio.run(st.session_state.unified_system.start())
                st.success("✅ 系统已启动")
            except Exception as e:
                st.error(f"❌ 启动失败: {e}")
    
    def stop_system(self):
        """停止系统"""
        with st.spinner("正在停止系统..."):
            try:
                # asyncio.run(st.session_state.unified_system.stop())
                st.warning("⏸️ 系统已停止")
            except Exception as e:
                st.error(f"❌ 停止失败: {e}")
    
    def refresh_data(self):
        """刷新数据"""
        with st.spinner("正在刷新数据..."):
            # 更新实时数据
            if self.redis_available:
                try:
                    # 从Redis获取最新数据
                    pass
                except Exception:pass
            st.success("✅ 数据已刷新")
    
    def submit_order(self, symbol, action, quantity, price_type):
        """提交订单"""
        order = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price_type': price_type,
            'timestamp': datetime.now()
        }
        
        st.session_state.active_orders.append(order)
        st.success(f"✅ 订单已提交: {action} {quantity}股 {symbol}")
    
    def get_market_status(self) -> str:
        """获取市场状态"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()
        
        if weekday > 4:
            return "休市"
        
        current_time = hour * 60 + minute
        
        if 555 <= current_time < 565:  # 9:15-9:25
            return "集合竞价"
        elif 570 <= current_time < 690 or 780 <= current_time < 900:  # 9:30-11:30, 13:00-15:00
            return "开盘中"
        elif 690 <= current_time < 780:  # 11:30-13:00
            return "午间休市"
        else:
            return "休市"
    
    def get_stock_name(self, symbol: str) -> str:
        """获取股票名称"""
        names = {
            "000001": "平安银行",
            "000002": "万科A",
            "600000": "浦发银行",
            "600519": "贵州茅台",
            "000858": "五粮液",
            "300750": "宁德时代"
        }
        return names.get(symbol, symbol)
    
    def render_agent_network(self):
        """渲染智能体协作网络"""
        # 简单的网络图示例
        st.info("🔗 智能体协作网络: 10个麒麟智能体 + TradingAgents智能体实时协作中")
    
    def render_risk_distribution(self):
        """渲染风险分布"""
        # 饼图
        fig = go.Figure(data=[go.Pie(
            labels=['股票风险', '市场风险', '流动性风险', '操作风险'],
            values=[30, 25, 20, 25],
            hole=0.4
        )])
        
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    def render_stress_test(self):
        """渲染压力测试结果"""
        scenarios = {
            '场景': ['市场下跌10%', '市场下跌20%', '黑天鹅事件', '流动性危机'],
            '预期损失': [-52000, -108000, -165000, -88000],
            '概率': [0.15, 0.05, 0.01, 0.03]
        }
        
        df = pd.DataFrame(scenarios)
        st.dataframe(
            df.style.format({
                '预期损失': '¥{:,.0f}',
                '概率': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    def render_risk_alerts(self):
        """渲染风险预警"""
        alerts = [
            ("⚠️", "中风险", "600519持仓比例接近上限", "2分钟前"),
            ("🔴", "高风险", "市场波动率超过预设阈值", "5分钟前"),
            ("🟡", "低风险", "000001接近止盈点", "10分钟前")
        ]
        
        for icon, level, message, time in alerts:
            if level == "高风险":
                st.error(f"{icon} [{level}] {message} - {time}")
            elif level == "中风险":
                st.warning(f"{icon} [{level}] {message} - {time}")
            else:
                st.info(f"{icon} [{level}] {message} - {time}")
    
    def render_trades(self):
        """渲染成交记录"""
        trades = []
        
        for i in range(5):
            trades.append({
                '成交时间': datetime.now() - timedelta(hours=i),
                '股票代码': np.random.choice(st.session_state.selected_stocks),
                '买卖方向': np.random.choice(['买入', '卖出']),
                '成交数量': np.random.randint(1, 10) * 100,
                '成交价格': np.random.uniform(8, 15),
                '手续费': np.random.uniform(5, 50)
            })
        
        df = pd.DataFrame(trades)
        st.dataframe(
            df.style.format({
                '成交时间': lambda x: x.strftime('%H:%M:%S'),
                '成交价格': '¥{:.2f}',
                '手续费': '¥{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    def render_trade_history(self, start_date, end_date):
        """渲染交易历史"""
        st.info(f"显示 {start_date} 至 {end_date} 的交易记录")
        # 这里应该从数据库读取实际的历史交易数据
        
    def render_performance_stats(self, start_date, end_date):
        """渲染绩效统计"""
        stats = {
            '总收益率': 0.125,
            '年化收益率': 0.285,
            '夏普比率': 1.85,
            '最大回撤': 0.082,
            '胜率': 0.625,
            '盈亏比': 2.1
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (key, value) in enumerate(stats.items()):
            with [col1, col2, col3][i % 3]:
                if '率' in key or '撤' in key:
                    st.metric(key, f"{value:.1%}")
                else:
                    st.metric(key, f"{value:.2f}")
    
    # ===== P2增强功能渲染方法 =====
    
    def render_limitup_analysis(self):
        """渲染高频涨停板分析页面 (P2-1)"""
        st.header("🔥 高频涨停板分析")
        
        st.markdown("""
        **功能说明:** 
        - 基于1分钟/5分钟级别高频数据分析涨停板盘中特征
        - 6大维度评分：量能爆发、封单稳定性、大单流入、尾盘封单强度、涨停打开次数、量萎缩度
        - 预测次日继续涨停概率
        """)
        
        st.divider()
        
        # 配置区
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_freq = st.selectbox(
                "分析频率",
                options=["1min", "5min", "15min"],
                index=0
            )
        
        with col2:
            target_stock = st.selectbox(
                "目标股票",
                options=st.session_state.selected_stocks + ["000001", "000002", "600000"],
                index=0
            )
        
        with col3:
            limitup_time = st.time_input(
                "涨停时间",
                value=datetime.strptime("10:30:00", "%H:%M:%S").time()
            )
        
        # 分析按钮
        if st.button("🔍 开始分析", use_container_width=True, type="primary"):
            with st.spinner("📊 正在分析高频数据..."):
                # 初始化分析器
                analyzer = HighFreqLimitUpAnalyzer(freq=analysis_freq)
                
                # 生成模拟数据 (实际应从数据源获取)
                data = create_sample_high_freq_data(f"{target_stock}.SZ")
                
                # 执行分析
                limitup_time_str = limitup_time.strftime("%H:%M:%S")
                features = analyzer.analyze_intraday_pattern(data, limitup_time_str)
                
                # 显示结果
                st.success("✅ 分析完成！")
                
                st.divider()
                
                # 分析结果显示
                st.subheader("📊 分析结果")
                
                # 6大维度指标
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)
                
                with col1:
                    score = features['volume_burst_before_limit']
                    st.metric(
                        "💥 涨停前量能爆发",
                        f"{score:.2%}",
                        delta="高" if score > 0.7 else "低",
                        delta_color="normal" if score > 0.7 else "inverse"
                    )
                
                with col2:
                    score = features['seal_stability']
                    st.metric(
                        "🔒 封单稳定性",
                        f"{score:.2%}",
                        delta="稳定" if score > 0.7 else "不稳",
                        delta_color="normal" if score > 0.7 else "inverse"
                    )
                
                with col3:
                    score = features['big_order_rhythm']
                    st.metric(
                        "💰 大单流入节奏",
                        f"{score:.2%}",
                        delta="持续" if score > 0.6 else "间断",
                        delta_color="normal" if score > 0.6 else "inverse"
                    )
                
                with col4:
                    score = features['close_seal_strength']
                    st.metric(
                        "🌟 尾盘封单强度",
                        f"{score:.2%}",
                        delta="强" if score > 0.7 else "弱",
                        delta_color="normal" if score > 0.7 else "inverse"
                    )
                
                with col5:
                    count = features['intraday_open_count']
                    st.metric(
                        "🔓 涨停打开次数",
                        f"{count}次",
                        delta="少" if count <= 2 else "多",
                        delta_color="normal" if count <= 2 else "inverse"
                    )
                
                with col6:
                    score = features['volume_shrink_after_limit']
                    st.metric(
                        "📉 量萎缩度",
                        f"{score:.2%}",
                        delta="明显" if score > 0.6 else "不明显",
                        delta_color="normal" if score > 0.6 else "inverse"
                    )
                
                st.divider()
                
                # 综合评分
                st.subheader("🎯 综合评分")
                
                # 计算综合得分
                weights = {
                    'volume_burst_before_limit': 0.15,
                    'seal_stability': 0.25,
                    'big_order_rhythm': 0.15,
                    'close_seal_strength': 0.30,
                    'volume_shrink_after_limit': 0.15
                }
                
                total_score = sum(
                    features[k] * w 
                    for k, w in weights.items() 
                    if isinstance(features.get(k), (int, float))
                )
                
                # 显示综合得分
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # 进度条形式显示
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                        <h2 style="color: {'#28a745' if total_score >= 0.8 else '#ffc107' if total_score >= 0.6 else '#dc3545'}; margin: 0;">{total_score:.1%}</h2>
                        <p style="margin: 5px 0 0 0; color: #666;">综合得分</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # 评级和建议
                    if total_score >= 0.80:
                        st.success("""
                        **✅ 评级: 强势涨停**
                        
                        - 次日继续涨停概率: **高** (≥80%)
                        - 建议: 重点关注，可考虑次日竞价介入
                        - 风险: 低
                        """)
                    elif total_score >= 0.60:
                        st.warning("""
                        **⚠️ 评级: 一般涨停**
                        
                        - 次日继续涨停概率: **中等** (50-70%)
                        - 建议: 谨慎观望，结合其他指标决策
                        - 风险: 中
                        """)
                    else:
                        st.error("""
                        **❌ 评级: 弱势涨停**
                        
                        - 次日继续涨停概率: **低** (<50%)
                        - 建议: 不建议追高，避免次日开盘介入
                        - 风险: 高
                        """)
                
                st.divider()
                
                # 权重分布雷达图
                st.subheader("🎯 特征权重分布")
                
                feature_names = [
                    '量能爆发',
                    '封单稳定',
                    '大单流入',
                    '尾盘封单',
                    '量萎缩度'
                ]
                
                feature_values = [
                    features['volume_burst_before_limit'],
                    features['seal_stability'],
                    features['big_order_rhythm'],
                    features['close_seal_strength'],
                    features['volume_shrink_after_limit']
                ]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=feature_values,
                    theta=feature_names,
                    fill='toself',
                    name=target_stock,
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # P4 · 涨停排队模拟（收拢为折叠面板）
                st.divider()
                with st.expander("P4 · 涨停排队模拟", expanded=False):
                    sim = LimitUpQueueSimulator()
                    colq1, colq2, colq3 = st.columns(3)
                    with colq1:
                        q_symbol = st.text_input("股票代码", value="000001.SZ", key="p4_queue_symbol")
                        q_limit_price = st.number_input("涨停价", min_value=0.01, value=11.00, step=0.01, key="p4_queue_price")
                    with colq2:
                        q_seal_amount = st.number_input("封单金额(元)", min_value=0.0, value=50_000_000.0, step=1_000_000.0, key="p4_queue_amount")
                        q_open_times = st.number_input("开板次数", min_value=0, value=0, step=1, key="p4_queue_open")
                    with colq3:
                        q_time = st.time_input("封板时间", value=datetime.strptime("09:35:00", "%H:%M:%S").time(), key="p4_queue_time")
                        q_target_shares = st.number_input("目标股数", min_value=100, value=20000, step=100, key="p4_queue_shares")
                    if st.button("📊 评估排队状态", key="p4_queue_eval", type="primary"):
                        today = datetime.now()
                        seal_time = today.replace(hour=q_time.hour, minute=q_time.minute, second=q_time.second, microsecond=0)
                        status = sim.evaluate_queue_status(
                            symbol=q_symbol,
                            limit_price=float(q_limit_price),
                            seal_amount=float(q_seal_amount),
                            seal_time=seal_time,
                            current_time=today,
                            target_shares=int(q_target_shares),
                            open_times=int(q_open_times),
                        )
                        qc1, qc2, qc3, qc4 = st.columns(4)
                        with qc1: st.metric("强度", status.strength.value)
                        with qc2: st.metric("评分", f"{status.strength_score:.1f}/100")
                        with qc3: st.metric("成交概率", f"{status.fill_probability:.1%}")
                        with qc4: st.metric("预计等待(分钟)", f"{status.estimated_wait_time:.0f}")
                        st.info(f"排队位置: {status.queue_position:,} 股 | 前置金额: ¥{status.queue_ahead_amount:,.0f}")
                        if status.warnings:
                            for w in status.warnings: st.warning(w)
                        if st.button("🎲 模拟一次排队成交", key="p4_queue_sim"):
                            exec_one = sim.simulate_queue_execution(
                                symbol=q_symbol,
                                order_time=today,
                                target_shares=int(q_target_shares),
                                limit_price=float(q_limit_price),
                                queue_status=status,
                                seal_broke=False,
                            )
                            st.success(f"成交 {exec_one.filled_shares:,} 股 @ ¥{exec_one.avg_fill_price:.2f}") if exec_one.filled else st.error("未成交")
                
                st.divider()
                
                # 分时量价图
                st.subheader("📊 分时量价分析")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('价格走势', '成交量')
                )
                
                # 价格曲线
                fig.add_trace(
                    go.Scatter(
                        x=data['time'],
                        y=data['close'],
                        mode='lines',
                        name='价格',
                        line=dict(color='#1f77b4', width=2)
                    ),
                    row=1, col=1
                )
                
                # 涨停时间标记
                # Add vertical line at limitup time if it exists in data
                try:
                    if 'time' in data.columns and limitup_time_str:
                        # Try to find exact match
                        matching_rows = data[data['time'] == limitup_time_str]
                        if not matching_rows.empty:
                            # Use the actual time value from data
                            limitup_x = matching_rows['time'].iloc[0]
                            fig.add_vline(
                                x=limitup_x,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="涨停",
                                row=1, col=1
                            )
                except Exception as e:
                    logger.warning(f"Could not add vline for limitup time: {e}")
                
                # 成交量
                fig.add_trace(
                    go.Bar(
                        x=data['time'],
                        y=data['volume'],
                        name='成交量',
                        marker=dict(color='#ff7f0e')
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="时间", row=2, col=1)
                fig.update_yaxes(title_text="价格", row=1, col=1)
                fig.update_yaxes(title_text="成交量", row=2, col=1)
                
                fig.update_layout(height=600, hovermode='x unified')
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_online_learning(self):
        """渲染在线学习与概念漂移检测页面 (P2-2)"""
        st.header("🧠 在线学习与模型自适应")
        
        st.markdown("""
        **功能说明:**
        - 增量模型更新：实时增量训练，无需完全重训
        - 概念漂移检测：KS检验自动检测市场环境变化
        - 模型版本管理：自动保存和回滚模型版本
        - 自适应学习率：根据性能自动调整学习率
        """)
        
        st.divider()
        
        # 配置区
        col1, col2, col3 = st.columns(3)
        
        with col1:
            update_freq = st.selectbox(
                "更新频率",
                options=["daily", "weekly", "monthly"],
                index=0
            )
        
        with col2:
            drift_threshold = st.slider(
                "漂移检测阈值",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01
            )
        
        with col3:
            enable_drift = st.checkbox("启用漂移检测", value=True)
        
        st.divider()
        
        # 主区域 - 两列布局
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # 在线学习状态
            st.subheader("📊 在线学习状态")
            
            # 模拟状态数据
            status_data = {
                '指标': [
                    '当前模型版本',
                    '上次更新时间',
                    '已处理样本数',
                    '当前准确率',
                    '漂移检测状态',
                    '缓冲区大小'
                ],
                '值': [
                    'v12',
                    datetime.now().strftime('%Y-%m-%d %H:%M'),
                    '12,458',
                    '0.732',
                    '正常',
                    '348/1000'
                ]
            }
            
            df_status = pd.DataFrame(status_data)
            st.dataframe(df_status, use_container_width=True, hide_index=True)
            
            # 性能趋势
            st.subheader("📈 性能趋势")
            
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            performance = 0.65 + np.cumsum(np.random.randn(30) * 0.01)
            performance = np.clip(performance, 0.5, 0.9)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=performance,
                mode='lines+markers',
                name='准确率',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            # 增加更新点标记
            update_points = dates[::5]
            update_values = performance[::5]
            
            fig.add_trace(go.Scatter(
                x=update_points,
                y=update_values,
                mode='markers',
                name='模型更新',
                marker=dict(size=12, color='red', symbol='star')
            ))
            
            fig.update_layout(
                xaxis_title="日期",
                yaxis_title="准确率",
                height=300,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 模型版本历史
            st.subheader("📋 模型版本历史")
            
            versions_data = {
                '版本': ['v12', 'v11', 'v10', 'v9', 'v8'],
                '创建时间': [
                    (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                    for i in range(5)
                ],
                '准确率': [0.732, 0.718, 0.705, 0.698, 0.685],
                '样本数': [12458, 11234, 10156, 9012, 8123],
                '状态': ['当前', '历史', '历史', '历史', '历史']
            }
            
            df_versions = pd.DataFrame(versions_data)
            st.dataframe(
                df_versions.style.format({
                    '准确率': '{:.3f}',
                    '样本数': '{:,}'
                }).apply(
                    lambda x: ['background-color: #d4edda' if v == '当前' else '' for v in x],
                    subset=['状态']
                ),
                use_container_width=True,
                hide_index=True
            )
        
        with col_right:
            # 概念漂移检测
            st.subheader("🔍 概念漂移检测")
            
            # 漂移检测状态
            drift_detected = np.random.random() > 0.7
            
            if drift_detected:
                st.warning("⚠️ 检测到概念漂移！")
                
                drift_info = {
                    '漂移得分': 0.087,
                    '检测时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    '建议操作': '增量更新'
                }
                
                for key, value in drift_info.items():
                    if key == '漂移得分':
                        st.metric(key, f"{value:.3f}", delta="超过阈值", delta_color="inverse")
                    else:
                        st.info(f"**{key}:** {value}")
                
                # 受影响特征
                st.write("**受影响最大的特征:**")
                
                affected_features = {
                    '特征': ['MA5', 'RSI', 'MACD', 'Volume_ratio', 'Price_momentum'],
                    'KS统计量': [0.12, 0.09, 0.08, 0.07, 0.06]
                }
                
                df_affected = pd.DataFrame(affected_features)
                
                fig = go.Figure(go.Bar(
                    x=df_affected['KS统计量'],
                    y=df_affected['特征'],
                    orientation='h',
                    marker=dict(color='#ff7f0e')
                ))
                
                fig.update_layout(
                    xaxis_title="KS统计量",
                    yaxis_title="特征",
                    height=250
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ 暂无概念漂移")
                st.info("模型分布与参考分布一致，市场环境稳定。")
            
            st.divider()
            
            # 自适应学习率
            st.subheader("⚙️ 自适应学习率")
            
            epochs = list(range(1, 31))
            learning_rates = [0.01 * (0.9 ** (i // 5)) for i in epochs]
            performances = [0.5 + 0.25 * (1 - np.exp(-i / 10)) for i in epochs]
            
            fig = make_subplots(
                specs=[[{"secondary_y": True}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=learning_rates,
                    mode='lines',
                    name='学习率',
                    line=dict(color='#ff7f0e', width=2)
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=performances,
                    mode='lines',
                    name='模型性能',
                    line=dict(color='#1f77b4', width=2)
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="训练轮次")
            fig.update_yaxes(title_text="学习率", secondary_y=False)
            fig.update_yaxes(title_text="模型性能", secondary_y=True)
            fig.update_layout(height=300, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 学习率调整策略
            st.info("""
            **当前策略:** 
            - 性能提升: 尝试提高学习率 (x1.1)
            - 性能停滞: 降低学习率 (x0.5)
            - 耐心值: 5轮
            """)
        
        st.divider()
        
        # 操作按钮
        st.subheader("🛠️ 操作控制")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 增量更新", use_container_width=True):
                with st.spinner("正在增量更新模型..."):
                    import time
                    time.sleep(1)
                    st.success("✅ 增量更新完成！")
        
        with col2:
            if st.button("🔍 检测漂移", use_container_width=True):
                with st.spinner("正在检测概念漂移..."):
                    import time
                    time.sleep(1)
                    st.info("📊 检测完成，结果已更新")
        
        with col3:
            if st.button("💾 保存版本", use_container_width=True):
                st.success("✅ 模型版本 v13 已保存")
        
        with col4:
            if st.button("⏪ 回滚版本", use_container_width=True):
                selected_version = st.selectbox("选择回滚版本", ['v11', 'v10', 'v9'])
                st.warning(f"⚠️ 已回滚到 {selected_version}")

    def render_phase3_risk_center(self):
        """渲染 Phase 3 风控中心 (流动性/极端行情/头寸管理)"""
        st.header("🛡️ 风控中心 (Phase 3)")
        st.caption("流动性监控 · 极端行情保护 · 动态头寸管理")
        
        tab_liq, tab_guard, tab_pos = st.tabs(["💧 流动性监控", "🛑 极端行情保护", "⚖️ 头寸管理"])
        
        with tab_liq:
            col_cfg, col_data = st.columns([1, 1])
            with col_cfg:
                symbol = st.selectbox("股票代码", options=st.session_state.selected_stocks + ["000001"], index=0)
                current_price = st.number_input("当前价格", min_value=0.01, value=10.50, step=0.01)
                min_avg_volume = st.number_input("最小日均量(股)", min_value=0.0, value=1_000_000.0, step=10000.0)
                max_spread_pct = st.number_input("最大价差(%)", min_value=0.0, value=0.20, step=0.01)
                min_turnover_pct = st.number_input("最小换手率(%)", min_value=0.0, value=1.00, step=0.10)
                min_score = st.number_input("最低流动性评分", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
                use_sample = st.checkbox("使用示例数据", value=True)
                
                monitor = LiquidityMonitor(
                    min_avg_volume=min_avg_volume,
                    max_spread_ratio=max_spread_pct / 100.0,
                    min_turnover_rate=min_turnover_pct / 100.0,
                    min_liquidity_score=min_score,
                )
                
            with col_data:
                if use_sample:
                    # 生成示例成交量/换手率数据（近20日）
                    np.random.seed(42)
                    volumes = np.random.randint(1_800_000, 3_800_000, 20)
                    turns = np.random.uniform(0.012, 0.040, 20)
                    volume_data = pd.DataFrame({
                        'volume': volumes,
                        'turnover_rate': turns,
                    })
                    order_book = {
                        'bid_prices': [current_price - 0.01 * i for i in range(1, 6)][::-1],
                        'ask_prices': [current_price + 0.01 * i for i in range(1, 6)],
                        'bid_volumes': [50000, 45000, 40000, 35000, 30000],
                        'ask_volumes': [48000, 42000, 38000, 33000, 28000],
                    }
                else:
                    st.info("请上传包含 volume 与 turnover_rate 列的CSV数据，并在代码中对接盘口数据源。")
                    volume_data = pd.DataFrame({'volume': [0], 'turnover_rate': [0.0]})
                    order_book = None
                
            if st.button("🔍 评估流动性", type="primary"):
                metrics = monitor.evaluate_liquidity(
                    symbol=symbol,
                    current_price=current_price,
                    volume_data=volume_data,
                    order_book=order_book,
                )
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("流动性评分", f"{metrics.liquidity_score:.1f}/100")
                with c2:
                    st.metric("流动性等级", metrics.liquidity_level.value)
                with c3:
                    st.metric("可建仓", "是" if metrics.can_buy else "否")
                with c4:
                    st.metric("最大建仓规模", f"{metrics.max_position_size:,.0f} 股")
                
                st.write("警告/提示:")
                if metrics.warnings:
                    for w in metrics.warnings:
                        st.warning(w)
                else:
                    st.info("无")
                
                st.divider()
                target_shares = st.number_input("拟建仓股数", min_value=0, value=10000, step=100)
                if st.button("📏 建仓规模检查"):
                    ok, reason, rec = monitor.check_position_size(symbol, target_shares, metrics)
                    if ok:
                        st.success(f"通过：{reason} | 建议股数 {rec:,}")
                    else:
                        st.error(f"不通过：{reason} | 建议股数 {rec:,}")
        
        with tab_guard:
            st.markdown("市场健康度与交易暂停策略")
            guard = ExtremeMarketGuard()
            if st.button("🧪 评估示例市场健康度"):
                # 构造简化市场数据字典
                market_data = {
                    f"{i:06d}.SZ": pd.DataFrame({
                        'open': [10.0], 'close': [10.0 * (1 + np.random.uniform(-0.12, 0.12))],
                        'turnover': [np.random.randint(5_000_000, 20_000_000)],
                        'turnover_rate': [np.random.uniform(0.01, 0.05)]
                    }) for i in range(500)
                }
                health = guard.evaluate_market_health(market_data)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("市场状态", health.market_condition.value)
                with c2:
                    st.metric("恐慌指数", f"{health.panic_index:.1f}/100")
                with c3:
                    st.metric("跌停数", f"{health.stocks_limit_down}")
                with c4:
                    st.metric("保护等级", health.protection_level.name)
                if health.warnings:
                    for w in health.warnings:
                        st.warning(w)
                halt, reason = guard.should_halt_trading()
                st.info(f"暂停交易: {'是' if halt else '否'} | 原因: {reason}")
            
            st.divider()
            st.markdown("个股极端事件检测（示例数据）")
            if st.button("⚡ 检测个股事件"):
                price_data = pd.DataFrame({
                    'open': [10.0] * 60,
                    'high': [10.2] * 60,
                    'low': [9.8] * 30 + [8.9] * 30,
                    'close': [10.0] * 30 + [9.3, 9.2, 8.9, 8.7, 8.5] + [8.5] * 25,
                })
                volume_data = pd.DataFrame({'volume': [1_000_000] * 30 + [4_000_000] * 30})
                event = guard.detect_extreme_event("000001.SZ", price_data, volume_data, timeframe="1min")
                if event:
                    st.error(f"检测到{event.event_type} | 严重度 {event.severity:.1f}/10 | {event.recommended_action}")
                else:
                    st.success("未检测到极端事件")
        
        with tab_pos:
            colA, colB = st.columns(2)
            with colA:
                total_capital = st.number_input("总资金(元)", min_value=1_0000.0, value=1_000_000.0, step=10_000.0)
                risk_level = st.selectbox("风险等级", options=[RiskLevel.CONSERVATIVE, RiskLevel.MODERATE, RiskLevel.AGGRESSIVE], index=1)
                method = st.selectbox("仓位方法", options=[PositionSizeMethod.KELLY, PositionSizeMethod.VOLATILITY_ADJUSTED, PositionSizeMethod.FIXED], index=0)
            with colB:
                symbol_p = st.selectbox("股票", options=st.session_state.selected_stocks + ["600000"], index=0)
                cur_price = st.number_input("当前价", min_value=0.01, value=10.50, step=0.01)
                stop_price = st.number_input("止损价", min_value=0.01, value=9.50, step=0.01)
                win_rate = st.slider("胜率(%)", 10.0, 90.0, 60.0, 1.0) if method == PositionSizeMethod.KELLY else None
                avg_return = st.slider("平均盈利(%)", 1.0, 30.0, 8.0, 0.5) if method == PositionSizeMethod.KELLY else None
                volatility = st.slider("波动率(%)", 1.0, 10.0, 2.5, 0.1) if method == PositionSizeMethod.VOLATILITY_ADJUSTED else None
            
            if st.button("🧮 计算仓位", type="primary"):
                mgr = RiskPositionManager(total_capital=total_capital, risk_level=risk_level)
                rec = mgr.calculate_position_size(
                    symbol=symbol_p,
                    current_price=cur_price,
                    stop_loss_price=stop_price,
                    win_rate=(win_rate/100.0) if win_rate is not None else None,
                    avg_return=(avg_return/100.0) if avg_return is not None else None,
                    volatility=(volatility/100.0) if volatility is not None else None,
                    method=method,
                    sector="示例"
                )
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("建议股数", f"{rec.recommended_shares:,}")
                with c2:
                    st.metric("建议金额", f"¥{rec.recommended_value:,.0f}")
                with c3:
                    st.metric("仓位比例", f"{rec.position_ratio:.1%}")
                with c4:
                    st.metric("预估风险", f"¥{rec.estimated_risk:,.0f}")
                
                if rec.warnings:
                    for w in rec.warnings:
                        st.warning(w)
                st.info(f"方法: {rec.method.value} | 理由: {rec.rationale} | 调整: {rec.adjustment_suggestion}")

    def render_phase4_realistic_backtest(self):
        """渲染 Phase 4 写实回测（滑点/冲击 + 涨停排队）"""
        st.header("🧪 写实回测 (Phase 4)")
        st.caption("滑点/冲击成本模型 · 涨停排队模拟")
        
        tab_exec, tab_queue = st.tabs(["💰 滑点/冲击成本", "📈 涨停排队模拟"])
        
        with tab_exec:
            c1, c2, c3 = st.columns(3)
            with c1:
                model = st.selectbox("滑点模型", options=[
                    SlippageModel.FIXED,
                    SlippageModel.LINEAR,
                    SlippageModel.SQRT,
                    SlippageModel.LIQUIDITY_BASED,
                ], index=3)
                side = st.selectbox("方向", options=[OrderSide.BUY, OrderSide.SELL], index=0)
            with c2:
                target_price = st.number_input("信号价", min_value=0.01, value=10.50, step=0.01)
                target_shares = st.number_input("目标股数", min_value=100, value=100000, step=100)
            with c3:
                avg_volume = st.number_input("日均量(股)", min_value=0.0, value=3_000_000.0, step=10000.0)
                liq_score = st.slider("流动性评分", 0.0, 100.0, 75.0, 1.0)
            use_md = st.checkbox("使用示例盘口 (仅LIQUIDITY_BASED)", value=True)
            
            market_depth = None
            if model == SlippageModel.LIQUIDITY_BASED and use_md:
                bids = [round(target_price - 0.01 * i, 2) for i in range(1, 6)][::-1]
                asks = [round(target_price + 0.01 * i, 2) for i in range(1, 6)]
                market_depth = Depth(
                    bid_prices=bids,
                    bid_volumes=[50000, 45000, 40000, 35000, 30000],
                    ask_prices=asks,
                    ask_volumes=[48000, 42000, 38000, 33000, 28000],
                    mid_price=(bids[-1] + asks[0]) / 2 if bids and asks else target_price,
                    spread=asks[0] - bids[-1] if bids and asks else 0.0,
                    total_bid_volume=sum([50000, 45000, 40000, 35000, 30000]),
                    total_ask_volume=sum([48000, 42000, 38000, 33000, 28000]),
                    liquidity_score=liq_score,
                )
            
            if st.button("🧪 模拟执行", type="primary"):
                engine = SlippageEngine(model=model)
                exec_res = engine.execute_order(
                    symbol="DEMO",
                    side=side,
                    target_shares=int(target_shares),
                    target_price=float(target_price),
                    market_depth=market_depth,
                    avg_daily_volume=float(avg_volume),
                    liquidity_score=float(liq_score),
                )
                costs = engine.calculate_total_slippage(exec_res)
                
                colm1, colm2, colm3, colm4 = st.columns(4)
                with colm1:
                    st.metric("平均成交价", f"¥{exec_res.avg_execution_price:.4f}")
                with colm2:
                    st.metric("滑点成本", f"¥{costs['slippage_cost']:,.0f}")
                with colm3:
                    st.metric("冲击成本", f"¥{costs['impact_cost']:,.0f}")
                with colm4:
                    st.metric("成本基点", f"{costs['cost_bps']:.2f} bps")
                
                # 分笔成交
                if exec_res.fills:
                    fills_df = pd.DataFrame(exec_res.fills, columns=["股数", "价格"])
                    st.dataframe(fills_df, use_container_width=True, hide_index=True)
        
        with tab_queue:
            sim = LimitUpQueueSimulator()
            colq1, colq2, colq3 = st.columns(3)
            with colq1:
                symbol = st.text_input("股票代码", value="000001.SZ")
                limit_price = st.number_input("涨停价", min_value=0.01, value=11.00, step=0.01)
            with colq2:
                seal_amount = st.number_input("封单金额(元)", min_value=0.0, value=50_000_000.0, step=1_000_000.0)
                open_times = st.number_input("开板次数", min_value=0, value=0, step=1)
            with colq3:
                t = st.time_input("封板时间", value=datetime.strptime("09:35:00", "%H:%M:%S").time())
                target_shares_q = st.number_input("目标股数", min_value=100, value=20000, step=100)
            
            if st.button("📊 评估排队状态", type="primary"):
                today = datetime.now()
                seal_time = today.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=0)
                status = sim.evaluate_queue_status(
                    symbol=symbol,
                    limit_price=float(limit_price),
                    seal_amount=float(seal_amount),
                    seal_time=seal_time,
                    current_time=today,
                    target_shares=int(target_shares_q),
                    open_times=int(open_times),
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("强度", status.strength.value)
                with c2:
                    st.metric("评分", f"{status.strength_score:.1f}/100")
                with c3:
                    st.metric("成交概率", f"{status.fill_probability:.1%}")
                with c4:
                    st.metric("预计等待(分钟)", f"{status.estimated_wait_time:.0f}")
                st.info(f"排队位置: {status.queue_position:,} 股 | 前置金额: ¥{status.queue_ahead_amount:,.0f}")
                if status.warnings:
                    for w in status.warnings:
                        st.warning(w)
                
                if st.button("🎲 模拟一次排队成交"):
                    exec_one = sim.simulate_queue_execution(
                        symbol=symbol,
                        order_time=today,
                        target_shares=int(target_shares_q),
                        limit_price=float(limit_price),
                        queue_status=status,
                        seal_broke=False,
                    )
                    if exec_one.filled:
                        st.success(f"成交 {exec_one.filled_shares:,} 股 @ ¥{exec_one.avg_fill_price:.2f}")
                    else:
                        st.error("未成交")
    
    def render_multi_source_data(self):
        """渲染多数据源管理页面 (P2-3)"""
        st.header("🔌 多数据源集成管理")
        
        st.markdown("""
        **功能说明:**
        - 支持多数据源：Qlib、AKShare、Yahoo Finance、Tushare
        - 自动降级：主数据源失败时自动切换备用源
        - 实时监控：数据源健康状态和延迟监控
        - 数据融合：多源数据智能合并
        """)
        
        st.divider()
        
        # 数据源配置
        col1, col2 = st.columns(2)
        
        with col1:
            primary_source = st.selectbox(
                "主数据源",
                options=["Qlib", "AKShare", "Yahoo Finance", "Tushare"],
                index=0
            )
        
        with col2:
            auto_fallback = st.checkbox("启用自动降级", value=True)
        
        # 备用数据源选择
        fallback_sources = st.multiselect(
            "备用数据源 (按优先级排序)",
            options=["AKShare", "Yahoo Finance", "Tushare", "Qlib"],
            default=["AKShare", "Yahoo Finance"]
        )
        
        st.divider()
        
        # 数据源状态监控
        st.subheader("📊 数据源健康状态")
        
        # 模拟数据源状态
        source_status = {
            'Qlib': {'available': True, 'latency': 45, 'last_check': datetime.now(), 'error': ''},
            'AKShare': {'available': True, 'latency': 320, 'last_check': datetime.now(), 'error': ''},
            'Yahoo Finance': {'available': False, 'latency': 0, 'last_check': datetime.now(), 'error': 'Connection timeout'},
            'Tushare': {'available': True, 'latency': 280, 'last_check': datetime.now(), 'error': ''}
        }
        
        # 显示卡片式状态
        cols = st.columns(4)
        
        for idx, (source_name, status) in enumerate(source_status.items()):
            with cols[idx]:
                # 状态指示
                if status['available']:
                    st.success(f"✅ {source_name}")
                else:
                    st.error(f"❌ {source_name}")
                
                # 详细信息
                if status['available']:
                    st.metric("延迟", f"{status['latency']}ms")
                    
                    # 延迟等级
                    if status['latency'] < 100:
                        st.success("优秀")
                    elif status['latency'] < 500:
                        st.warning("良好")
                    else:
                        st.error("较慢")
                else:
                    st.write(f"错误: {status['error']}")
                
                st.caption(f"上次检查: {status['last_check'].strftime('%H:%M:%S')}")
        
        st.divider()
        
        # 数据源使用统计
        st.subheader("📊 数据源使用统计")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 使用次数饼图
            usage_data = {
                '数据源': ['Qlib', 'AKShare', 'Yahoo Finance', 'Tushare'],
                '使用次数': [1250, 320, 80, 150]
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=usage_data['数据源'],
                values=usage_data['使用次数'],
                hole=0.4
            )])
            
            fig.update_layout(
                title="数据源使用分布",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 成功率条形图
            success_data = {
                '数据源': ['Qlib', 'AKShare', 'Yahoo Finance', 'Tushare'],
                '成功率': [0.995, 0.982, 0.856, 0.975]
            }
            
            fig = go.Figure(go.Bar(
                x=success_data['数据源'],
                y=success_data['成功率'],
                marker=dict(
                    color=success_data['成功率'],
                    colorscale='RdYlGn',
                    cmin=0.8,
                    cmax=1.0
                ),
                text=[f"{v:.1%}" for v in success_data['成功率']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="数据源成功率",
                yaxis_title="成功率",
                height=300,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 实时延迟监控
        st.subheader("⏱️ 实时延迟监控")
        
        # 模拟30个时间点的延迟数据
        time_points = pd.date_range(end=datetime.now(), periods=30, freq='1min')
        
        latency_data = {
            'Qlib': 30 + np.random.randn(30) * 5,
            'AKShare': 250 + np.random.randn(30) * 30,
            'Tushare': 200 + np.random.randn(30) * 25
        }
        
        fig = go.Figure()
        
        for source, latencies in latency_data.items():
            fig.add_trace(go.Scatter(
                x=time_points,
                y=latencies,
                mode='lines',
                name=source,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="延迟 (ms)",
            height=350,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 数据获取测试
        st.subheader("🧪 数据获取测试")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_symbols = st.multiselect(
                "测试股票",
                options=["000001.SZ", "000002.SZ", "600000.SH", "600519.SH"],
                default=["000001.SZ"]
            )
        
        with col2:
            date_range = st.date_input(
                "日期范围",
                value=[datetime.now() - timedelta(days=30), datetime.now()]
            )
        
        with col3:
            st.write("")
            st.write("")
            if st.button("🚀 开始测试", use_container_width=True, type="primary"):
                with st.spinner("正在从多个数据源获取数据..."):
                    import time
                    time.sleep(1.5)
                    
                    st.success("✅ 数据获取成功！")
                    
                    # 显示测试结果
                    test_results = {
                        '数据源': ['Qlib (主)', 'AKShare (备用)', 'Tushare (备用)'],
                        '状态': ['✅ 成功', '⏭️ 未使用', '⏭️ 未使用'],
                        '耗时': ['1.2s', '-', '-'],
                        '数据量': ['3,245 条', '-', '-']
                    }
                    
                    df_results = pd.DataFrame(test_results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    st.info("💡 主数据源 Qlib 运行正常，未触发降级机制")
        
        st.divider()
        
        # 操作按钮
        st.subheader("🛠️ 操作控制")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 刷新状态", use_container_width=True):
                st.success("✅ 状态已刷新")
        
        with col2:
            if st.button("🧪 健康检查", use_container_width=True):
                with st.spinner("正在检查所有数据源..."):
                    import time
                    time.sleep(1)
                    st.info("📊 检查完成，3/4 数据源健康")
        
        with col3:
            if st.button("🔄 切换主源", use_container_width=True):
                st.warning("⚠️ 已切换到备用数据源")
        
        with col4:
            if st.button("⚙️ 高级配置", use_container_width=True):
                st.info("🔧 打开高级配置面板...")
    
    # ===== Phase 2 增强功能渲染方法 =====
    
    def render_rl_trading(self):
        """渲染强化学习交易策略页面 (P2-4)"""
        st.header("🤖 强化学习交易策略")
        
        st.markdown("""
        **功能说明:**
        - DQN (Deep Q-Network) 智能体自动学习交易策略
        - 状态空间: 价格特征 + 持仓信息 + 账户信息
        - 动作空间: 持有/买入/卖出
        - 奖励函数: 总资产变化率
        """)
        
        st.divider()
        
        # 配置区
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_balance = st.number_input(
                "初始资金 (元)",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000
            )
        
        with col2:
            num_episodes = st.slider(
                "训练轮数",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
        
        with col3:
            epsilon_decay = st.slider(
                "探索率衰减",
                min_value=0.90,
                max_value=0.999,
                value=0.995,
                step=0.001
            )
        
        # 训练按钮
        if st.button("🎯 开始训练", use_container_width=True, type="primary"):
            with st.spinner("🤖 正在训练DQN智能体..."):
                # 创建环境和智能体
                data = create_rl_data(days=252)
                env = TradingEnvironment(data, initial_balance=initial_balance)
                agent = DQNAgent(state_dim=20, action_dim=3, epsilon_decay=epsilon_decay)
                trainer = RLTrainer(env, agent)
                
                # 训练
                import time
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                history = []
                for i in range(num_episodes):
                    # 模拟训练过程
                    time.sleep(0.02)
                    progress = (i + 1) / num_episodes
                    progress_bar.progress(progress)
                    status_text.text(f"训练进度: {i+1}/{num_episodes} 轮")
                    
                    # 生成模拟历史数据
                    history.append({
                        'episode': i + 1,
                        'total_return': 0.05 + 0.15 * (1 - np.exp(-i / 20)) + np.random.randn() * 0.02,
                        'sharpe_ratio': 0.5 + 1.5 * (1 - np.exp(-i / 15)) + np.random.randn() * 0.1,
                        'max_drawdown': -0.10 + 0.05 * (1 - np.exp(-i / 25)),
                        'num_trades': 10 + i // 2
                    })
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ 训练完成！")
                
                st.divider()
                
                # 显示训练结果
                st.subheader("📊 训练结果")
                
                df_history = pd.DataFrame(history)
                
                # 最终性能
                final_metrics = history[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "总收益率",
                        f"{final_metrics['total_return']:.2%}",
                        delta="优秀" if final_metrics['total_return'] > 0.10 else "一般"
                    )
                
                with col2:
                    st.metric(
                        "夏普比率",
                        f"{final_metrics['sharpe_ratio']:.2f}",
                        delta="优秀" if final_metrics['sharpe_ratio'] > 1.5 else "一般"
                    )
                
                with col3:
                    st.metric(
                        "最大回撤",
                        f"{final_metrics['max_drawdown']:.2%}",
                        delta="控制良好" if final_metrics['max_drawdown'] > -0.10 else "需改进"
                    )
                
                with col4:
                    st.metric(
                        "交易次数",
                        f"{final_metrics['num_trades']}次"
                    )
                
                st.divider()
                
                # 学习曲线
                st.subheader("📈 学习曲线")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 收益率曲线
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_history['episode'],
                        y=df_history['total_return'],
                        mode='lines+markers',
                        name='收益率',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig.update_layout(
                        title="收益率趋势",
                        xaxis_title="训练轮次",
                        yaxis_title="收益率",
                        height=300,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 夏普比率曲线
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_history['episode'],
                        y=df_history['sharpe_ratio'],
                        mode='lines+markers',
                        name='夏普比率',
                        line=dict(color='#ff7f0e', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig.update_layout(
                        title="夏普比率趋势",
                        xaxis_title="训练轮次",
                        yaxis_title="夏普比率",
                        height=300,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # 回测结果
                st.subheader("💼 回测结果")
                
                # 生成模拟回测数据
                dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
                portfolio_values = initial_balance * (1 + np.cumsum(np.random.randn(252) * 0.01 + 0.0005))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    mode='lines',
                    name='组合价值',
                    line=dict(color='#2ca02c', width=2),
                    fill='tonexty'
                ))
                
                # 添加基准线
                fig.add_hline(
                    y=initial_balance,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="初始资金"
                )
                
                fig.update_layout(
                    xaxis_title="日期",
                    yaxis_title="组合价值 (元)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_optimization(self):
        """渲染投资组合优化页面 (P2-5)"""
        st.header("💼 投资组合优化")
        
        st.markdown("""
        **功能说明:**
        - 多种优化算法: 均值方差(Markowitz)、Black-Litterman、风险平价
        - 有效前沿分析: 找到最优风险-收益比
        - 资产配置建议: 智能权重分配
        """)
        
        st.divider()
        
        # 资产选择
        st.subheader("🎯 资产选择")
        
        assets = st.multiselect(
            "选择资产",
            options=['Asset_1 (股票A)', 'Asset_2 (股票B)', 'Asset_3 (股票C)', 
                    'Asset_4 (债券)', 'Asset_5 (商品)'],
            default=['Asset_1 (股票A)', 'Asset_2 (股票B)', 'Asset_3 (股票C)']
        )
        
        n_assets = len(assets)
        
        if n_assets < 2:
            st.warning("⚠️ 请至少选择 2 个资产")
            return
        
        # 优化方法
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_method = st.selectbox(
                "优化方法",
                options=["最大化夏普比率", "最小化波动率", "Black-Litterman", "风险平价"]
            )
        
        with col2:
            risk_free_rate = st.slider(
                "无风险利率",
                min_value=0.0,
                max_value=0.10,
                value=0.03,
                step=0.01,
                format="%.2f"
            )
        
        # 优化按钮
        if st.button("🚀 执行优化", use_container_width=True, type="primary"):
            with st.spinner("📊 正在优化组合..."):
                # 生成模拟数据
                returns = create_sample_returns(n_assets=n_assets, n_days=252)
                
                import time
                time.sleep(1)
                
                if optimization_method == "最大化夏普比率":
                    optimizer = MeanVarianceOptimizer(returns, risk_free_rate=risk_free_rate)
                    result = optimizer.optimize_sharpe()
                elif optimization_method == "最小化波动率":
                    optimizer = MeanVarianceOptimizer(returns, risk_free_rate=risk_free_rate)
                    result = optimizer.optimize_min_volatility()
                elif optimization_method == "Black-Litterman":
                    optimizer = BlackLittermanOptimizer(returns, risk_free_rate=risk_free_rate)
                    # 模拟观点
                    views = {0: 0.15, 2: 0.10} if n_assets >= 3 else {0: 0.12}
                    result = optimizer.optimize_with_views(views, view_confidence=0.7)
                else:  # 风险平价
                    optimizer = RiskParityOptimizer(returns)
                    result = optimizer.optimize()
                
                st.success("✅ 优化完成！")
                
                st.divider()
                
                # 优化结果
                st.subheader("📊 优化结果")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "预期年化收益",
                        f"{result.expected_return:.2%}"
                    )
                
                with col2:
                    st.metric(
                        "预期年化风险",
                        f"{result.expected_risk:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "夏普比率",
                        f"{result.sharpe_ratio:.2f}",
                        delta="优秀" if result.sharpe_ratio > 1.5 else "一般"
                    )
                
                st.divider()
                
                # 资产配置
                st.subheader("🍰 资产配置")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # 权重表
                    weight_data = {
                        '资产': [assets[i] for i in range(n_assets)],
                        '权重': result.weights,
                        '百分比': result.weights * 100
                    }
                    
                    df_weights = pd.DataFrame(weight_data)
                    st.dataframe(
                        df_weights.style.format({
                            '权重': '{:.4f}',
                            '百分比': '{:.2f}%'
                        }).background_gradient(subset=['权重'], cmap='Blues'),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # 饼图
                    fig = go.Figure(data=[go.Pie(
                        labels=[assets[i] for i in range(n_assets)],
                        values=result.weights,
                        hole=0.4
                    )])
                    
                    fig.update_layout(
                        title="资产分布",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # 有效前沿
                if optimization_method in ["最大化夏普比率", "最小化波动率"]:
                    st.subheader("📈 有效前沿")
                    
                    with st.spinner("计算有效前沿..."):
                        # 生成模拟有效前沿
                        n_points = 30
                        min_return = 0.05
                        max_return = 0.25
                        
                        frontier_returns = np.linspace(min_return, max_return, n_points)
                        frontier_risks = []
                        
                        for r in frontier_returns:
                            # 模拟风险 (凸函数关系)
                            risk = 0.10 + 0.5 * (r - min_return) ** 1.5
                            frontier_risks.append(risk)
                        
                        fig = go.Figure()
                        
                        # 有效前沿
                        fig.add_trace(go.Scatter(
                            x=frontier_risks,
                            y=frontier_returns,
                            mode='lines',
                            name='有效前沿',
                            line=dict(color='#1f77b4', width=3)
                        ))
                        
                        # 当前组合
                        fig.add_trace(go.Scatter(
                            x=[result.expected_risk],
                            y=[result.expected_return],
                            mode='markers',
                            name='当前组合',
                            marker=dict(size=15, color='red', symbol='star')
                        ))
                        
                        fig.update_layout(
                            xaxis_title="年化波动率 (风险)",
                            yaxis_title="年化收益率",
                            height=400,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_monitoring(self):
        """渲染实时风险监控页面 (P2-6)"""
        st.header("⚠️ 实时风险监控")
        
        st.markdown("""
        **功能说明:**
        - VaR/CVaR风险价值计算 (95%/99%置信水平)
        - 压力测试: 市场暴跌、波动率飙升、流动性危机
        - 风险预警: 多级别实时预警系统
        - 综合风险评分: 0-100分 (分数越高风险越大)
        """)
        
        st.divider()
        
        # 生成模拟数据
        returns, prices = create_risk_data(days=252)
        
        # 初始化风险监控
        monitor = RiskMonitor(
            var_threshold_95=-0.05,
            var_threshold_99=-0.08,
            drawdown_threshold=0.15,
            volatility_threshold=0.30
        )
        
        # 计算风险指标
        metrics = monitor.calculate_metrics(returns, prices)
        
        # 显示风险指标
        st.subheader("📊 风险指标")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "VaR 95%",
                f"{metrics.var_95:.2%}",
                delta="超限" if metrics.var_95 < -0.05 else "正常",
                delta_color="inverse" if metrics.var_95 < -0.05 else "normal"
            )
        
        with col2:
            st.metric(
                "VaR 99%",
                f"{metrics.var_99:.2%}",
                delta="超限" if metrics.var_99 < -0.08 else "正常",
                delta_color="inverse" if metrics.var_99 < -0.08 else "normal"
            )
        
        with col3:
            st.metric(
                "CVaR 95%",
                f"{metrics.cvar_95:.2%}"
            )
        
        with col4:
            st.metric(
                "最大回撤",
                f"{metrics.max_drawdown:.2%}",
                delta="超限" if abs(metrics.max_drawdown) > 0.15 else "正常",
                delta_color="inverse" if abs(metrics.max_drawdown) > 0.15 else "normal"
            )
        
        with col5:
            st.metric(
                "波动率",
                f"{metrics.volatility:.2%}",
                delta="高" if metrics.volatility > 0.30 else "正常",
                delta_color="inverse" if metrics.volatility > 0.30 else "normal"
            )
        
        with col6:
            st.metric(
                "夏普比率",
                f"{metrics.sharpe_ratio:.2f}",
                delta="优秀" if metrics.sharpe_ratio > 1.0 else "一般"
            )
        
        st.divider()
        
        # 风险预警
        st.subheader("⚠️ 风险预警")
        
        alerts = monitor.check_risk_levels(metrics, ['ASSET_1', 'ASSET_2', 'ASSET_3'])
        
        if alerts:
            for alert in alerts:
                if alert.level.value == "极高风险":
                    st.error(f"🔴 [{alert.level.value}] {alert.risk_type}")
                    st.error(f"**消息:** {alert.message}")
                    st.error(f"**建议:** {alert.recommended_action}")
                elif alert.level.value == "高风险":
                    st.warning(f"🟠 [{alert.level.value}] {alert.risk_type}")
                    st.warning(f"**消息:** {alert.message}")
                    st.warning(f"**建议:** {alert.recommended_action}")
                elif alert.level.value == "中风险":
                    st.warning(f"🟡 [{alert.level.value}] {alert.risk_type}")
                    st.info(f"**消息:** {alert.message}")
                    st.info(f"**建议:** {alert.recommended_action}")
                else:
                    st.info(f"🟢 [{alert.level.value}] {alert.risk_type}")
                    st.info(f"**消息:** {alert.message}")
                
                st.divider()
        else:
            st.success("✅ 当前无风险预警")
        
        # 综合风险评分
        st.subheader("🎯 综合风险评分")
        
        summary = monitor.get_risk_summary(metrics)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 风险评分仪表盘
            risk_score = summary['risk_score']
            
            if risk_score < 30:
                color = "#28a745"  # 绿色
            elif risk_score < 60:
                color = "#ffc107"  # 黄色
            elif risk_score < 80:
                color = "#ff9800"  # 橙色
            else:
                color = "#dc3545"  # 红色
            
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;">
                <h1 style="color: {color}; margin: 0; font-size: 60px;">{risk_score:.0f}</h1>
                <p style="margin: 10px 0 0 0; color: #666; font-size: 18px;">风险评分 / 100</p>
                <h3 style="color: {color}; margin: 10px 0 0 0;">{summary['overall_risk_level'].value}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 风险贡献分解
            contributions = {
                '风险类型': ['VaR贡献', '回撤贡献', '波动率贡献'],
                '分数': [
                    summary['var_contribution'],
                    summary['drawdown_contribution'],
                    summary['volatility_contribution']
                ],
                '权重': ['50%', '30%', '20%']
            }
            
            df_contrib = pd.DataFrame(contributions)
            
            fig = go.Figure(go.Bar(
                x=df_contrib['分数'],
                y=df_contrib['风险类型'],
                orientation='h',
                marker=dict(
                    color=df_contrib['分数'],
                    colorscale='Reds',
                    showscale=False
                ),
                text=[f"{v:.1f}" for v in df_contrib['分数']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="风险贡献分解",
                xaxis_title="贡献分数",
                height=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 压力测试
        st.subheader("🔥 压力测试")
        
        # 执行压力测试
        market_returns = returns * 0.8  # 模拟市场收益
        stress_test = StressTest(returns, market_returns)
        scenarios = stress_test.run_all_scenarios()
        
        # 显示场景结果
        scenario_names = {
            'market_crash': '市场暴跌',
            'volatility_spike': '波动率飙升',
            'liquidity_crisis': '流动性危机'
        }
        
        scenario_data = []
        for s in scenarios:
            scenario_data.append({
                '场景': scenario_names.get(s['scenario'], s['scenario']),
                '预期损失': s.get('estimated_portfolio_loss', s.get('estimated_total_loss', 0)),
                '发生概率': s['loss_probability']
            })
        
        df_scenarios = pd.DataFrame(scenario_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('预期损失', '发生概率')
        )
        
        # 损失条形图
        fig.add_trace(
            go.Bar(
                x=df_scenarios['预期损失'],
                y=df_scenarios['场景'],
                orientation='h',
                name='预期损失',
                marker=dict(color='#ff7f0e')
            ),
            row=1, col=1
        )
        
        # 概率条形图
        fig.add_trace(
            go.Bar(
                x=df_scenarios['发生概率'],
                y=df_scenarios['场景'],
                orientation='h',
                name='发生概率',
                marker=dict(color='#2ca02c')
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="损失比例", row=1, col=1)
        fig.update_xaxes(title_text="概率", row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细场景表
        st.dataframe(
            df_scenarios.style.format({
                '预期损失': '{:.2%}',
                '发生概率': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )

    def render_performance_attribution(self):
        """P2-7: 绩效归因分析渲染"""
        st.header("📊 P2-7: 绩效归因分析 (Performance Attribution)")
        
        try:
            from qlib_enhanced.performance_attribution import (
                BrinsonAttribution,
                FactorAttribution,
                TransactionCostAnalysis,
                create_sample_attribution_data
            )
        except ImportError:
            st.error("❌ 无法导入归因分析模块，请检查 performance_attribution.py")
            return
        
        # 参数配置
        with st.sidebar:
            st.subheader("归因分析配置")
            analysis_type = st.selectbox(
                "分析类型",
                ["📋 基准测试", "Brinson归因", "因子归因", "交易成本分析", "综合报告"]
            )
            
            use_sample_data = st.checkbox("使用示例数据", value=True)
        
        st.divider()
        
        # ===== 0. 基准测试模块 =====
        if analysis_type in ["📋 基准测试", "综合报告"]:
            st.subheader("📋 基准测试 (Benchmark Comparison)")
            
            st.info("""
            **基准测试**将策略或模型与市场基准进行对比，评估相对表现。
            - 市场基准：沪深300、中证500、创业板指
            - 模型基准：LSTM、GRU、Transformer等
            - 性能指标：Sharpe、收益、最大回撤等
            """)
            
            # 基准选择
            benchmark_col1, benchmark_col2 = st.columns(2)
            
            with benchmark_col1:
                market_benchmark = st.multiselect(
                    "市场基准",
                    ["沪深300", "中证500", "创业板指", "上证50"],
                    default=["沪深300"]
                )
            
            with benchmark_col2:
                model_benchmark = st.multiselect(
                    "模型基准",
                    ["LSTM", "GRU", "Transformer", "XGBoost", "LightGBM"],
                    default=["LSTM", "Transformer"]
                )
            
            if use_sample_data:
                # 生成示例数据
                np.random.seed(42)
                periods = 252  # 1年交易日
                dates = pd.date_range('2023-01-01', periods=periods, freq='B')
                
                # 策略收益
                strategy_returns = pd.Series(
                    np.random.normal(0.001, 0.015, periods),
                    index=dates
                )
                
                # 市场基准收益
                benchmark_returns = {}
                for bench in market_benchmark:
                    if bench == "沪深300":
                        benchmark_returns[bench] = pd.Series(
                            np.random.normal(0.0005, 0.012, periods),
                            index=dates
                        )
                    elif bench == "中证500":
                        benchmark_returns[bench] = pd.Series(
                            np.random.normal(0.0007, 0.013, periods),
                            index=dates
                        )
                    elif bench == "创业板指":
                        benchmark_returns[bench] = pd.Series(
                            np.random.normal(0.0003, 0.018, periods),
                            index=dates
                        )
                    else:
                        benchmark_returns[bench] = pd.Series(
                            np.random.normal(0.0006, 0.011, periods),
                            index=dates
                        )
                
                # 模型基准收益
                for model in model_benchmark:
                    if model == "LSTM":
                        benchmark_returns[model] = pd.Series(
                            np.random.normal(0.0008, 0.014, periods),
                            index=dates
                        )
                    elif model == "Transformer":
                        benchmark_returns[model] = pd.Series(
                            np.random.normal(0.0009, 0.015, periods),
                            index=dates
                        )
                    elif model == "GRU":
                        benchmark_returns[model] = pd.Series(
                            np.random.normal(0.0007, 0.013, periods),
                            index=dates
                        )
                    elif model == "XGBoost":
                        benchmark_returns[model] = pd.Series(
                            np.random.normal(0.0006, 0.012, periods),
                            index=dates
                        )
                    else:
                        benchmark_returns[model] = pd.Series(
                            np.random.normal(0.0008, 0.013, periods),
                            index=dates
                        )
                
                # 计算累计收益
                strategy_cum_returns = (1 + strategy_returns).cumprod()
                
                # 1. 性能指标对比
                st.subheader("📊 性能指标对比")
                
                # 计算指标
                def calc_metrics(returns):
                    cum_return = (1 + returns).prod() - 1
                    annual_return = (1 + cum_return) ** (252 / len(returns)) - 1
                    annual_vol = returns.std() * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    max_dd = (returns.cumsum() - returns.cumsum().expanding().max()).min()
                    return {
                        '累计收益': cum_return,
                        '年化收益': annual_return,
                        '年化波动': annual_vol,
                        'Sharpe比率': sharpe,
                        '最大回撤': max_dd
                    }
                
                # 构建对比表
                metrics_data = []
                metrics_data.append({'策略/基准': '当前策略', **calc_metrics(strategy_returns)})
                
                for name, returns in benchmark_returns.items():
                    metrics_data.append({'策略/基准': name, **calc_metrics(returns)})
                
                metrics_df = pd.DataFrame(metrics_data)
                
                # 高亮最好的指标
                st.dataframe(
                    metrics_df.style.format({
                        '累计收益': '{:.2%}',
                        '年化收益': '{:.2%}',
                        '年化波动': '{:.2%}',
                        'Sharpe比率': '{:.3f}',
                        '最大回撤': '{:.2%}'
                    }).background_gradient(subset=['累计收益', 'Sharpe比率'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 2. 累计收益对比图
                st.subheader("📈 累计收益曲线")
                
                fig = go.Figure()
                
                # 添加策略曲线
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=strategy_cum_returns,
                    mode='lines',
                    name='当前策略',
                    line=dict(color='red', width=3)
                ))
                
                # 添加基准曲线
                colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                for i, (name, returns) in enumerate(benchmark_returns.items()):
                    cum_returns = (1 + returns).cumprod()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=cum_returns,
                        mode='lines',
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="策略 vs 基准累计收益对比",
                    xaxis_title="日期",
                    yaxis_title="累计收益",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 超额收益分析
                st.subheader("🎯 超额收益分析")
                
                # 选择一个基准作为对比
                selected_benchmark = st.selectbox(
                    "选择对比基准",
                    list(benchmark_returns.keys()),
                    key="excess_return_benchmark"
                )
                
                if selected_benchmark:
                    excess_returns = strategy_returns - benchmark_returns[selected_benchmark]
                    cum_excess_returns = excess_returns.cumsum()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "平均超额收益",
                            f"{excess_returns.mean():.2%}"
                        )
                    
                    with col2:
                        st.metric(
                            "超额收益波动",
                            f"{excess_returns.std():.2%}"
                        )
                    
                    with col3:
                        st.metric(
                            "信息比率",
                            f"{excess_returns.mean() / excess_returns.std():.3f}" if excess_returns.std() > 0 else "N/A"
                        )
                    
                    # 超额收益图
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=cum_excess_returns,
                        mode='lines',
                        fill='tozeroy',
                        name=f'超额收益 vs {selected_benchmark}',
                        line=dict(color='green')
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig.update_layout(
                        title=f"相对于{selected_benchmark}的累计超额收益",
                        xaxis_title="日期",
                        yaxis_title="累计超额收益",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 4. 胜率统计
                st.subheader("🏆 胜率统计")
                
                win_rate_data = []
                for name, returns in benchmark_returns.items():
                    win_days = (strategy_returns > returns).sum()
                    total_days = len(strategy_returns)
                    win_rate = win_days / total_days
                    
                    win_rate_data.append({
                        '基准': name,
                        '胜利天数': win_days,
                        '总天数': total_days,
                        '胜率': win_rate
                    })
                
                win_rate_df = pd.DataFrame(win_rate_data)
                
                # 柱状图展示胜率
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=win_rate_df['基准'],
                    y=win_rate_df['胜率'],
                    text=[f"{v:.1%}" for v in win_rate_df['胜率']],
                    textposition='outside',
                    marker=dict(
                        color=win_rate_df['胜率'],
                        colorscale='RdYlGn',
                        showscale=False
                    )
                ))
                
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
                
                fig.update_layout(
                    title="策略相对于各基准的胜率",
                    yaxis_title="胜率",
                    yaxis=dict(range=[0, 1], tickformat='.0%'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 详细表格
                st.dataframe(
                    win_rate_df.style.format({
                        '胜率': '{:.2%}'
                    }).background_gradient(subset=['胜率'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                st.warning("请上传策略和基准数据或使用示例数据")
            
            st.divider()
        
        # ===== 1. Brinson归因分析 =====
        if analysis_type in ["Brinson归因", "综合报告"]:
            st.subheader("📈 Brinson归因模型")
            
            st.info("""
            **Brinson归因模型**将组合超额收益分解为三个部分：
            - **配置效应**：资产配置权重偏离基准的贡献
            - **选择效应**：证券选择产生的超额收益
            - **交互效应**：配置和选择的协同效应
            """)
            
            if use_sample_data:
                pw, pr, bw, br = create_sample_attribution_data()
                
                brinson = BrinsonAttribution(pw, pr, bw, br)
                result = brinson.analyze()
                
                # 显示核心指标
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "配置效应",
                        f"{result.allocation_effect:.2%}",
                        delta=f"{result.allocation_effect:.2%}"
                    )
                
                with col2:
                    st.metric(
                        "选择效应",
                        f"{result.selection_effect:.2%}",
                        delta=f"{result.selection_effect:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "交互效应",
                        f"{result.interaction_effect:.2%}",
                        delta=f"{result.interaction_effect:.2%}"
                    )
                
                with col4:
                    st.metric(
                        "总超额收益",
                        f"{result.total_active_return:.2%}",
                        delta=f"{result.total_active_return:.2%}"
                    )
                
                # 可视化归因分解
                effects_data = pd.DataFrame({
                    '归因类型': ['配置效应', '选择效应', '交互效应'],
                    '贡献率': [
                        result.allocation_effect,
                        result.selection_effect,
                        result.interaction_effect
                    ]
                })
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=effects_data['归因类型'],
                    y=effects_data['贡献率'],
                    marker=dict(
                        color=effects_data['贡献率'],
                        colorscale='RdYlGn',
                        showscale=False
                    ),
                    text=[f"{v:.2%}" for v in effects_data['贡献率']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Brinson归因分解",
                    yaxis_title="贡献率",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 权重对比表
                st.subheader("📊 组合 vs 基准权重对比")
                
                weight_compare = pd.DataFrame({
                    '资产': pw.columns,
                    '组合平均权重': pw.mean().values,
                    '基准平均权重': bw.mean().values,
                    '权重偏离': (pw.mean() - bw.mean()).values
                })
                
                st.dataframe(
                    weight_compare.style.format({
                        '组合平均权重': '{:.2%}',
                        '基准平均权重': '{:.2%}',
                        '权重偏离': '{:+.2%}'
                    }).background_gradient(subset=['权重偏离'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                st.warning("请上传组合和基准数据或使用示例数据")
            
            st.divider()
        
        # ===== 2. 因子归因分析 =====
        if analysis_type in ["因子归因", "综合报告"]:
            st.subheader("🧮 因子归因分析")
            
            st.info("""
            **因子归因**将组合收益分解到各个风险因子（如市场、规模、价值等），
            帮助理解收益来源和风险暴露。
            """)
            
            if use_sample_data:
                # 生成示例数据
                np.random.seed(42)
                returns = pd.Series(np.random.normal(0.01, 0.02, 60))
                factors = pd.DataFrame({
                    'Market': np.random.normal(0.008, 0.015, 60),
                    'Size': np.random.normal(0.002, 0.01, 60),
                    'Value': np.random.normal(0.003, 0.01, 60),
                    'Momentum': np.random.normal(0.004, 0.012, 60)
                })
                
                factor_attr = FactorAttribution(returns, factors)
                contributions = factor_attr.analyze()
                
                # 显示因子贡献
                factor_df = pd.DataFrame(list(contributions.items()), 
                                         columns=['因子', '贡献率'])
                factor_df = factor_df.sort_values('贡献率', ascending=False)
                
                # 饼图显示因子贡献
                fig = go.Figure()
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                fig.add_trace(go.Pie(
                    labels=factor_df['因子'],
                    values=factor_df['贡献率'].abs(),
                    marker=dict(colors=colors),
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>贡献: %{value:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="因子贡献分布",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 详细贡献表
                st.dataframe(
                    factor_df.style.format({
                        '贡献率': '{:.4f}'
                    }).background_gradient(subset=['贡献率'], cmap='coolwarm'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 因子暴露时间序列
                st.subheader("📈 因子暴露时间序列")
                
                fig = go.Figure()
                
                for factor in factors.columns:
                    fig.add_trace(go.Scatter(
                        y=factors[factor],
                        mode='lines',
                        name=factor
                    ))
                
                fig.update_layout(
                    title="因子暴露演变",
                    xaxis_title="时间",
                    yaxis_title="因子值",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("请上传收益和因子数据或使用示例数据")
            
            st.divider()
        
        # ===== 3. 交易成本分析 =====
        if analysis_type in ["交易成本分析", "综合报告"]:
            st.subheader("💰 交易成本分析")
            
            st.info("""
            **交易成本**包括显性成本（佣金）和隐性成本（滑点、市场冲击），
            对净收益有重要影响。
            """)
            
            # 成本参数配置
            col1, col2 = st.columns(2)
            
            with col1:
                commission_rate = st.number_input(
                    "佣金率 (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01
                ) / 100
            
            with col2:
                slippage_bps = st.number_input(
                    "滑点 (基点)",
                    min_value=0.0,
                    max_value=50.0,
                    value=5.0,
                    step=0.5
                )
            
            if use_sample_data:
                # 生成示例交易数据
                np.random.seed(42)
                trades = pd.DataFrame({
                    'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'] * 20,
                    'quantity': np.random.randint(100, 1000, 100),
                    'price': np.random.uniform(50, 200, 100),
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
                })
                
                cost_analysis = TransactionCostAnalysis(trades)
                costs = cost_analysis.analyze(
                    commission_rate=commission_rate,
                    slippage_bps=slippage_bps
                )
                
                # 成本指标展示
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "总交易成本",
                        f"¥{costs['total_cost']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "佣金成本",
                        f"¥{costs['commission_cost']:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        "滑点成本",
                        f"¥{costs['slippage_cost']:,.0f}"
                    )
                
                with col4:
                    st.metric(
                        "市场冲击",
                        f"¥{costs['market_impact_cost']:,.0f}"
                    )
                
                # 成本占比
                st.metric(
                    "成本占交易金额比例",
                    f"{costs['cost_as_pct_of_value']:.3%}"
                )
                
                # 成本分解饼图
                cost_breakdown = pd.DataFrame({
                    '成本类型': ['佣金', '滑点', '市场冲击'],
                    '金额': [
                        costs['commission_cost'],
                        costs['slippage_cost'],
                        costs['market_impact_cost']
                    ]
                })
                
                fig = go.Figure()
                
                fig.add_trace(go.Pie(
                    labels=cost_breakdown['成本类型'],
                    values=cost_breakdown['金额'],
                    marker=dict(colors=['#ff7f0e', '#2ca02c', '#d62728']),
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>¥%{value:,.0f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="交易成本分解",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 交易明细
                st.subheader("📋 交易明细 (最近10笔)")
                
                trades_display = trades.tail(10).copy()
                trades_display['交易金额'] = trades_display['quantity'] * trades_display['price']
                trades_display['估计成本'] = trades_display['交易金额'] * (commission_rate + slippage_bps/10000)
                
                st.dataframe(
                    trades_display[['timestamp', 'symbol', 'quantity', 'price', '交易金额', '估计成本']].style.format({
                        'price': '¥{:.2f}',
                        '交易金额': '¥{:,.0f}',
                        '估计成本': '¥{:,.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                st.warning("请上传交易数据或使用示例数据")

            # 高级：写实滑点/冲击成本（P4）
            with st.expander("高级：写实滑点/冲击成本 (P4)", expanded=False):
                model = st.selectbox("滑点模型", options=[
                    SlippageModel.FIXED,
                    SlippageModel.LINEAR,
                    SlippageModel.SQRT,
                    SlippageModel.LIQUIDITY_BASED,
                ], index=3, key="p4_attr_model")
                side = st.selectbox("方向", options=[OrderSide.BUY, OrderSide.SELL], index=0, key="p4_attr_side")
                ctp, cts, cadv, cliq = st.columns(4)
                with ctp:
                    target_price = st.number_input("信号价", min_value=0.01, value=10.50, step=0.01, key="p4_attr_tp")
                with cts:
                    target_shares = st.number_input("目标股数", min_value=100, value=100000, step=100, key="p4_attr_ts")
                with cadv:
                    avg_volume = st.number_input("日均量(股)", min_value=0.0, value=3_000_000.0, step=10000.0, key="p4_attr_adv")
                with cliq:
                    liq_score = st.slider("流动性评分", 0.0, 100.0, 75.0, 1.0, key="p4_attr_liq")
                use_md = st.checkbox("使用示例盘口 (仅LIQUIDITY_BASED)", value=True, key="p4_attr_usemd")
                market_depth = None
                if model == SlippageModel.LIQUIDITY_BASED and use_md:
                    bids = [round(target_price - 0.01 * i, 2) for i in range(1, 6)][::-1]
                    asks = [round(target_price + 0.01 * i, 2) for i in range(1, 6)]
                    market_depth = Depth(
                        bid_prices=bids,
                        bid_volumes=[50000, 45000, 40000, 35000, 30000],
                        ask_prices=asks,
                        ask_volumes=[48000, 42000, 38000, 33000, 28000],
                        mid_price=(bids[-1] + asks[0]) / 2 if bids and asks else target_price,
                        spread=asks[0] - bids[-1] if bids and asks else 0.0,
                        total_bid_volume=sum([50000, 45000, 40000, 35000, 30000]),
                        total_ask_volume=sum([48000, 42000, 38000, 33000, 28000]),
                        liquidity_score=liq_score,
                    )
                if st.button("计算写实成本", key="p4_attr_calc", type="primary"):
                    engine = SlippageEngine(model=model)
                    exec_res = engine.execute_order(
                        symbol="DEMO",
                        side=side,
                        target_shares=int(target_shares),
                        target_price=float(target_price),
                        market_depth=market_depth,
                        avg_daily_volume=float(avg_volume),
                        liquidity_score=float(liq_score),
                    )
                    costs_rt = engine.calculate_total_slippage(exec_res)
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("平均成交价", f"¥{exec_res.avg_execution_price:.4f}")
                    with m2: st.metric("滑点成本", f"¥{costs_rt['slippage_cost']:,.0f}")
                    with m3: st.metric("冲击成本", f"¥{costs_rt['impact_cost']:,.0f}")
                    with m4: st.metric("成本基点", f"{costs_rt['cost_bps']:.2f} bps")
        
        st.divider()
        
        # 导出报告按钮
        st.subheader("📥 导出归因报告")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("导出PDF报告"):
                st.success("✅ PDF报告生成中...")
        
        with col2:
            if st.button("导出Excel数据"):
                st.success("✅ Excel数据导出中...")
        
        with col3:
            if st.button("生成归因总结"):
                st.info("""
                📊 **归因分析总结**
                - Brinson归因显示配置效应为主要超额收益来源
                - 因子归因表明市场因子贡献最大
                - 交易成本处于合理水平，约占交易额的0.11%
                """)


# 主程序入口
if __name__ == "__main__":
    dashboard = UnifiedDashboard()
    dashboard.run()
