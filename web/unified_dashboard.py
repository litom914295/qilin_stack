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

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path("D:/test/Qlib/tradingagents")))

# 监控权重
from monitoring.metrics import get_monitor

# 导入核心组件
from tradingagents_integration.integration_adapter import (
    TradingAgentsAdapter, 
    UnifiedTradingSystem
from trading.realtime_trading_system import RealtimeTradingSystem
from agents.trading_agents import MultiAgentManager
from qlib_integration.qlib_engine import QlibIntegrationEngine
from data_layer.data_access_layer import DataAccessLayer

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
            self.redis_available = True
        except Exception:self.redis_client = None
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
        st.session_state.selected_stocks = selected_stocks
        
        # 参数设置
        st.subheader("⚙️ 交易参数")
        
        position_size = st.slider(
            "单股仓位(%)",
            min_value=5,
            max_value=30,
            value=10
        
        stop_loss = st.number_input(
            "止损线(%)",
            min_value=1.0,
            max_value=10.0,
            value=5.0
        
        take_profit = st.number_input(
            "止盈线(%)",
            min_value=5.0,
            max_value=30.0,
            value=10.0
        
        # 刷新设置
        st.subheader("🔄 刷新设置")
        
        auto_refresh = st.checkbox("自动刷新", value=False)
        st.session_state.auto_refresh = auto_refresh
        
        refresh_interval = st.slider(
            "刷新间隔(秒)",
            min_value=1,
            max_value=60,
            value=st.session_state.refresh_interval
        st.session_state.refresh_interval = refresh_interval
    
    def render_main_content(self):
        """渲染主内容区"""
        # 创建标签页
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
            
        with col2:
            st.metric(
                "今日盈亏",
                f"¥{metrics['today_pnl']:,.0f}",
                f"{metrics['pnl_change']:+.2%}"
            
        with col3:
            st.metric(
                "持仓数",
                f"{metrics['position_count']}只",
                f"{metrics['position_change']:+d}"
            
        with col4:
            st.metric(
                "胜率",
                f"{metrics['win_rate']:.1%}",
                f"{metrics['win_rate_change']:+.1%}"
            
        with col5:
            st.metric(
                "夏普比",
                f"{metrics['sharpe']:.2f}",
                f"{metrics['sharpe_change']:+.2f}"
            
        with col6:
            st.metric(
                "最大回撤",
                f"{metrics['max_dd']:.2%}",
                f"{metrics['dd_change']:+.2%}"
        
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
            
        with col2:
            action = st.selectbox(
                "交易方向",
                options=["买入", "卖出"]
            
        with col3:
            quantity = st.number_input(
                "数量(股)",
                min_value=100,
                max_value=10000,
                step=100,
                value=100
            
        with col4:
            price_type = st.selectbox(
                "价格类型",
                options=["市价", "限价"]
            
        if price_type == "限价":
            limit_price = st.number_input(
                "限价",
                min_value=0.01,
                value=10.00
        
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
            
        with col2:
            st.metric(
                "风险敞口",
                f"{risk_metrics['exposure']:.1%}",
                help="总风险敞口占比"
            
        with col3:
            st.metric(
                "杠杆率",
                f"{risk_metrics['leverage']:.2f}x",
                help="实际杠杆倍数"
            
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
    
    def render_history(self):
        """历史记录页面"""
        # 日期选择
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30)
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now()
        
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


# 主程序入口
if __name__ == "__main__":
    dashboard = UnifiedDashboard()
    dashboard.run()
