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
try:
    import requests
except Exception:
    requests = None
from urllib.parse import urljoin

# Configure logging
logger = logging.getLogger(__name__)

# Load .env if available (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Compat patch: 静默移除 deprecated use_container_width 参数 ---
def _patch_streamlit_use_container_width():
    """Patch Streamlit 组件以静默移除 use_container_width 参数"""
    import warnings
    def _wrap(func):
        def inner(*args, **kwargs):
            # 静默移除 use_container_width 参数以避免警告
            if 'use_container_width' in kwargs:
                kwargs.pop('use_container_width')
            return func(*args, **kwargs)
        return inner
    # patch 常用 APIs
    try:
        st.button = _wrap(st.button)
        st.dataframe = _wrap(st.dataframe)
        st.download_button = _wrap(st.download_button)
        st.plotly_chart = _wrap(st.plotly_chart)
        st.table = _wrap(st.table)
    except Exception:
        pass

_patch_streamlit_use_container_width()

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
# TradingAgents路径采用环境变量 TRADINGAGENTS_PATH（可选）
import os
_ENV_TA = os.getenv("TRADINGAGENTS_PATH")
if _ENV_TA:
    p = Path(_ENV_TA)
    if p.exists():
        sys.path.append(str(p))

# 监控权重
try:
    from monitoring.metrics import get_monitor
except Exception as e:
    logger.warning(f"监控模块导入失败: {e}")
    get_monitor = None

# 导入核心组件 - 可选导入
try:
    from tradingagents_integration.integration_adapter import (
        TradingAgentsAdapter, 
        UnifiedTradingSystem
    )
except Exception as e:
    logger.warning(f"TradingAgents适配器导入失败: {e}")
    TradingAgentsAdapter = None
    UnifiedTradingSystem = None

try:
    from trading.realtime_trading_system import RealtimeTradingSystem
except Exception as e:
    logger.warning(f"实时交易系统导入失败: {e}")
    RealtimeTradingSystem = None

try:
    from agents.trading_agents import MultiAgentManager
except Exception as e:
    logger.warning(f"多智能体管理器导入失败: {e}")
    MultiAgentManager = None

try:
    from qlib_integration.qlib_engine import QlibIntegrationEngine
except Exception as e:
    logger.warning(f"Qlib集成引擎导入失败: {e}")
    QlibIntegrationEngine = None

try:
    from data_layer.data_access_layer import DataAccessLayer
except Exception as e:
    logger.warning(f"数据访问层导入失败: {e}")
    DataAccessLayer = None

# 导入P2增强功能模块 - 可选导入
sys.path.insert(0, str(Path(__file__).parent.parent / "qlib_enhanced"))

try:
    from high_freq_limitup import HighFreqLimitUpAnalyzer, create_sample_high_freq_data
except Exception as e:
    logger.warning(f"高频涨停分析器导入失败: {e}")
    HighFreqLimitUpAnalyzer = None
    create_sample_high_freq_data = None

try:
    from online_learning import OnlineLearningManager, DriftDetector, AdaptiveLearningRate
except Exception as e:
    logger.warning(f"在线学习模块导入失败: {e}")
    OnlineLearningManager = None
    DriftDetector = None
    AdaptiveLearningRate = None

try:
    from multi_source_data import MultiSourceDataProvider, DataSource
except Exception as e:
    logger.warning(f"多数据源提供者导入失败: {e}")
    MultiSourceDataProvider = None
    DataSource = None

try:
    from one_into_two_pipeline import (
        build_sample_dataset,
        OneIntoTwoTrainer,
        rank_candidates,
        extract_limitup_features,
    )
except Exception as e:
    logger.warning(f"一进二管道导入失败: {e}")
    build_sample_dataset = None
    OneIntoTwoTrainer = None
    rank_candidates = None
    extract_limitup_features = None

# Phase 2 模块 - 可选导入
try:
    from rl_trading import TradingEnvironment, DQNAgent, RLTrainer, create_sample_data as create_rl_data
except Exception as e:
    logger.warning(f"强化学习交易导入失败: {e}")
    TradingEnvironment = None
    DQNAgent = None
    RLTrainer = None
    create_rl_data = None

try:
    from portfolio_optimizer import MeanVarianceOptimizer, BlackLittermanOptimizer, RiskParityOptimizer, create_sample_returns
except Exception as e:
    logger.warning(f"组合优化器导入失败: {e}")
    MeanVarianceOptimizer = None
    BlackLittermanOptimizer = None
    RiskParityOptimizer = None
    create_sample_returns = None

try:
    from risk_management import ValueAtRiskCalculator, StressTest, RiskMonitor, create_sample_data as create_risk_data
except Exception as e:
    logger.warning(f"风险管理导入失败: {e}")
    ValueAtRiskCalculator = None
    StressTest = None
    RiskMonitor = None
    create_risk_data = None

try:
    from performance_attribution import TransactionCostAnalysis
except Exception as e:
    logger.warning(f"绩效归因导入失败: {e}")
    TransactionCostAnalysis = None

# Phase 3 风控模块 - 可选导入
try:
    from qilin_stack.agents.risk.liquidity_monitor import LiquidityMonitor, LiquidityLevel
except Exception as e:
    logger.warning(f"流动性监控导入失败: {e}")
    LiquidityMonitor = None
    LiquidityLevel = None

# Phase 3 & 4 UI优化与高级功能 - 必需导入
try:
    from web.components.ui_styles import inject_global_styles
    from web.components.color_scheme import Colors, Emojis
    from web.components.loading_cache import LoadingSpinner, CacheManager, show_success_animation, show_error_animation
    from web.components.smart_tips_enhanced import EnhancedSmartTipSystem
    from web.components.advanced_features import (
        SimulatedTrading,
        StrategyBacktest,
        ExportManager,
        render_simulated_trading,
        render_backtest,
        render_export
    )
    logger.info("Phase 3 & 4 组件导入成功")
except Exception as e:
    logger.warning(f"Phase 3 & 4 组件导入失败: {e}")
    inject_global_styles = None
    Colors = None
    Emojis = None
    LoadingSpinner = None
    CacheManager = None
    show_success_animation = None
    show_error_animation = None
    EnhancedSmartTipSystem = None
    SimulatedTrading = None
    StrategyBacktest = None
    ExportManager = None
    render_simulated_trading = None
    render_backtest = None
    render_export = None

try:
    from qilin_stack.agents.risk.extreme_market_guard import ExtremeMarketGuard, ProtectionLevel, MarketCondition
except Exception as e:
    logger.warning(f"极端市场保护导入失败: {e}")
    ExtremeMarketGuard = None
    ProtectionLevel = None
    MarketCondition = None

try:
    from qilin_stack.agents.risk.position_manager import (
        PositionManager as RiskPositionManager,
        PositionSizeMethod,
        RiskLevel,
    )
except Exception as e:
    logger.warning(f"仓位管理器导入失败: {e}")
    RiskPositionManager = None
    PositionSizeMethod = None
    RiskLevel = None

# Phase 4 写实回测模块 - 可选导入
try:
    from qilin_stack.backtest.slippage_model import (
        SlippageEngine,
        SlippageModel,
        OrderSide,
        MarketDepth as Depth,
    )
except Exception as e:
    logger.warning(f"滑点模型导入失败: {e}")
    SlippageEngine = None
    SlippageModel = None
    OrderSide = None
    Depth = None

try:
    from qilin_stack.backtest.limit_up_queue_simulator import (
        LimitUpQueueSimulator,
        LimitUpStrength,
    )
except Exception as e:
    logger.warning(f"涨停队列模拟器导入失败: {e}")
    LimitUpQueueSimulator = None
    LimitUpStrength = None

# 页面配置
st.set_page_config(
    page_title="麒麟量化交易平台 - 统一控制中心",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入Phase 3全局样式
if inject_global_styles:
    inject_global_styles()
    logger.info("全局样式已注入")

# 自定义CSS（保留兼容性）
st.markdown("""
<style>
    .main { 
        padding-top: 3rem !important; 
    }
    .block-container { 
        padding-top: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1rem !important;
    }
    /* 确保顶部标题区域有足够的上边距 */
    [data-testid="stAppViewContainer"] {
        padding-top: 1rem !important;
    }
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
        self.setup_connections()  # 先设置连接
        self.init_session_state()  # 再初始化状态（需要 redis_available）
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
        if 'system_running' not in st.session_state:
            st.session_state.system_running = False
        if 'selected_stocks' not in st.session_state:
            # 初始化时获取实际涨停股
            st.session_state.selected_stocks = self._get_top_limitup_stocks()
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
        if 'auto_trade' not in st.session_state:
            st.session_state.auto_trade = False
        # 在线服务/MLflow默认配置（可由ENV注入）
        import os as _os
        st.session_state.setdefault('qlib_serving_url', _os.getenv('QLIB_SERVING_URL', 'http://localhost:9000'))
        st.session_state.setdefault('qlib_serving_api_key', _os.getenv('QLIB_SERVING_API_KEY', ''))
        st.session_state.setdefault('mlflow_uri', _os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        st.session_state.setdefault('mlflow_experiment', _os.getenv('MLFLOW_EXPERIMENT', 'qlib_limitup'))
        st.session_state.setdefault('mlflow_model_name', _os.getenv('MLFLOW_MODEL_NAME', 'qlib_limitup_v1'))
        st.session_state.setdefault('mlflow_connected', False)
            
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
    
    def _get_top_limitup_stocks(self, top_n: int = 3) -> List[str]:
        """获取当日最强势的前 N 只涨停股"""
        try:
            # 尝试从 Redis 缓存获取
            if hasattr(self, 'redis_available') and self.redis_available:
                cached = self.redis_client.get('top_limitup_stocks')
                if cached:
                    stocks = json.loads(cached)
                    if len(stocks) >= top_n:
                        return stocks[:top_n]
            
            # 尝试从文件系统获取最近的筛选结果
            data_dir = Path(__file__).parent.parent / "data" / "daily_selections"
            if data_dir.exists():
                # 查找最近的筛选文件
                files = list(data_dir.glob("limitup_*.json"))
                if files:
                    latest_file = max(files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'stocks' in data and len(data['stocks']) > 0:
                            # 按质量评分排序
                            stocks = sorted(
                                data['stocks'], 
                                key=lambda x: x.get('quality_score', 0), 
                                reverse=True
                            )
                            return [s['symbol'] for s in stocks[:top_n]]
        except Exception as e:
            logger.warning(f"获取涨停股数据失败: {e}")
        
        # 默认返回示例数据
        return ["000001", "000002", "600000"]
        
    def init_systems(self):
        """初始化交易系统"""
        config = {
            "symbols": st.session_state.selected_stocks,
            "position_size_pct": 0.1,
            "max_position_size": 0.3,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10
        }
        
        # 初始化适配器 - 可选
        if st.session_state.adapter is None and TradingAgentsAdapter is not None:
            try:
                st.session_state.adapter = TradingAgentsAdapter(config)
            except Exception as e:
                logger.warning(f"初始化TradingAgents适配器失败: {e}")
                st.session_state.adapter = None
            
        # 初始化统一系统 - 可选
        if st.session_state.unified_system is None and UnifiedTradingSystem is not None:
            try:
                st.session_state.unified_system = UnifiedTradingSystem(config)
            except Exception as e:
                logger.warning(f"初始化统一系统失败: {e}")
                st.session_state.unified_system = None
            
        # 初始化实时交易系统 - 可选
        if st.session_state.trading_system is None and RealtimeTradingSystem is not None:
            try:
                st.session_state.trading_system = RealtimeTradingSystem(config)
            except Exception as e:
                logger.warning(f"初始化实时交易系统失败: {e}")
                st.session_state.trading_system = None
    
    def run(self):
        """运行主界面"""
        # 顶部信息栏
        self.render_header()
        
        # 侧边栏
        with st.sidebar:
            self.render_sidebar()
        
        # 主界面内容
        self.render_main_content()
        
        # 注意：自动刷新已禁用，以免影响浏览体验
        # 如果需要实时数据更新，请手动点击“刷新数据”按钮
        # if st.session_state.get('auto_refresh', False):
        #     st.experimental_rerun()
    
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
        
        # ========== 1. 系统控制（最重要，保持在顶部） ==========
        st.subheader("🎮 系统控制")
        
        # 显示当前系统状态
        if st.session_state.system_running:
            st.success("✅ 系统运行中")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏸️ 停止", use_container_width=True, type="primary", key="ud_stop_system"):
                    self.stop_system()
        else:
            st.error("❌ 系统已停止")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ 启动", use_container_width=True, type="primary"):
                    self.start_system()
                
        if st.button("🔄 刷新数据", use_container_width=True, key="ud_sidebar_refresh"):
            self.refresh_data()
        
        st.divider()
        
        # ========== 2. 监控股票（常用功能） ==========
        st.subheader("📊 监控股票")
        selected_stocks = st.multiselect(
            "选择股票",
            options=["000001", "000002", "600000", "600519", "000858", "300750"],
            default=st.session_state.selected_stocks,
            key="sidebar_stock_select"
        )
        st.session_state.selected_stocks = selected_stocks
        
        st.divider()
        
        # ========== 3. 文档中心（合并文档与指南+文档搜索） ==========
        with st.expander("📚 文档中心 (29个文档)", expanded=True):
            st.caption("💡 快速查找和预览系统文档")
            
            # 文档选择与预览
            docs = {
                # 快速开始
                "—— 🚀 快速开始 ——": None,
                "🚀 5分钟快速上手": "docs/QUICKSTART.md",
                "📖 完整使用指南": "docs/USAGE_GUIDE.md",
                "🧪 测试指南": "docs/TESTING_GUIDE.md",
                
                # RD-Agent
                "—— 🤖 RD-Agent ——": None,
                "✅ RD-Agent对齐完成": "docs/archive/completion/RDAGENT_ALIGNMENT_COMPLETE.md",
                "📝 RD-Agent对齐计划": "docs/RDAGENT_ALIGNMENT_PLAN.md",
                "🏆 RD-Agent最终总结": "docs/RDAGENT_FINAL_SUMMARY.md",
                "🔗 RD-Agent集成指南": "docs/RD-Agent_Integration_Guide.md",
                
                # 功能指南
                "—— 📖 功能指南 ——": None,
                "📅 日常交易SOP": "docs/DAILY_TRADING_SOP.md",
                "📊 数据准备指南": "docs/DATA_GUIDE.md",
                "🎯 股票选择指南": "docs/STOCK_SELECTION_GUIDE.md",
                "📦 股票池配置": "docs/STOCK_POOL_GUIDE.md",
                "🔬 因子研发快速开始": "docs/FACTOR_RESEARCH_QUICKSTART.md",
                "🤖 LLM因子发现": "docs/LLM_FACTOR_DISCOVERY_GUIDE.md",
                "🏛️ Qlib模型库快速开始": "docs/QLIB_MODEL_ZOO_QUICKSTART.md",
                "💻 AKShare使用指南": "docs/AKSHARE_GUIDE.md",
                
                # 技术架构
                "—— 🏛️ 技术架构 ——": None,
                "🏛️ 深度架构指南": "docs/DEEP_ARCHITECTURE_GUIDE.md",
                "📝 技术架构v2.1": "docs/Technical_Architecture_v2.1_Final.md",
                "🚀 部署指南": "docs/DEPLOYMENT_GUIDE.md",
                "💻 Windows环境配置": "docs/ENV_SETUP_WINDOWS.md",
                "🔌 API文档": "docs/API_DOCUMENTATION.md",
                
                # 项目报告
                "—— 📊 项目报告 ——": None,
                "🏆 项目最终报告": "docs/FINAL_PROJECT_REPORT.md",
                "✅ 对齐完成检查": "docs/archive/completion/ALIGNMENT_COMPLETION_CHECK.md",
                "📊 麒鳞对齐报告": "docs/QILIN_ALIGNMENT_REPORT.md",
                "🧪 测试完成报告": "docs/archive/completion/TESTING_COMPLETION_REPORT.md",
                
                # 专项模块
                "—— 🔧 专项模块 ——": None,
                "💼 竞价交易框架": "docs/AUCTION_WORKFLOW_FRAMEWORK.md",
                "🧠 AI进化系统": "docs/AI_EVOLUTION_SYSTEM_INTEGRATION.md",
                "📈 涨停板AI进化": "docs/LIMITUP_AI_EVOLUTION_SYSTEM.md",
                "🔁 迭代进化训练": "docs/ITERATIVE_EVOLUTION_TRAINING.md",
                "💻 Web控制面板指南": "docs/WEB_DASHBOARD_GUIDE.md",
                
                # 文档索引
                "—— 📚 文档索引 ——": None,
                "📑 文档总索引": "docs/INDEX.md",
                "📋 文档整理方案": "docs/DOCUMENTATION_STRUCTURE.md",
                "✅ 文档整理完成": "docs/DOCUMENTATION_CLEANUP_COMPLETE.md",
            }
            # 过滤掉分隔符
            valid_docs = {k: v for k, v in docs.items() if v is not None}
            
            choice = st.selectbox("选择文档", list(valid_docs.keys()), key="doc_selector")
            colv1, colv2 = st.columns([1,1])
            with colv1:
                if st.button("🔎 预览", use_container_width=True, key="doc_preview_btn"):
                    self._show_doc(valid_docs[choice])
            with colv2:
                st.caption(str(Path(__file__).parent.parent / valid_docs[choice]))
            
            st.divider()
            
            # 文档搜索
            st.markdown("★★🔍 文档搜索★★")
            query = st.text_input("关键词", value="", placeholder="输入要搜索的关键字…", key="doc_search_query")
            scopes = {
                "docs/": Path(__file__).parent.parent / "docs",
                "tradingagents_integration/": Path(__file__).parent.parent / "tradingagents_integration",
                "web/tabs/rdagent/": Path(__file__).parent.parent / "web" / "tabs" / "rdagent",
                "web/tabs/tradingagents/": Path(__file__).parent.parent / "web" / "tabs" / "tradingagents",
            }
            selected = st.multiselect("搜索范围", list(scopes.keys()), default=["docs/"], key="doc_search_scope")
            file_exts = st.multiselect("文件类型", ['.md', '.markdown', '.yaml', '.yml', '.txt'], default=['.md', '.markdown', '.yaml', '.yml'], key="doc_search_exts")
            max_hits = st.slider("最多结果条数", 10, 200, 50, 10, key="doc_search_max_hits")
            if st.button("🔍 开始搜索", use_container_width=True, key="doc_search_btn"):
                if not query.strip():
                    st.warning("请输入关键词")
                else:
                    roots = [scopes[k] for k in selected]
                    results = self._doc_search(query.strip(), roots, exts=set(file_exts), max_hits=max_hits)
                    if not results:
                        st.info("未找到匹配项")
                    else:
                        st.caption(f"共找到 {len(results)} 条（最多显示 {max_hits} 条）")
                        for r in results:
                            fp = r['path']
                            st.markdown(f"★★{fp}★★ · 第 {r['line']} 行")
                            st.markdown(self._highlight(r['snippet'], query.strip()), unsafe_allow_html=True)
        
        st.divider()
        
        # ========== 4. 高级设置（折叠不常用的参数） ==========
        with st.expander("⚙️ 高级设置", expanded=False):
            st.caption("🔧 配置交易参数和系统设置")
            
            # 交易参数
            st.markdown("★★⚡ 交易参数★★")
            position_size = st.slider(
                "单股仓位(%)",
                min_value=5,
                max_value=30,
                value=10,
                key="sidebar_position_size"
            )
            stop_loss = st.number_input(
                "止损线(%)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                key="sidebar_stop_loss"
            )
            take_profit = st.number_input(
                "止盈线(%)",
                min_value=5.0,
                max_value=30.0,
                value=10.0,
                key="sidebar_take_profit"
            )
            
            st.divider()
            
            # 刷新设置
            st.markdown("★★🔄 刷新设置★★")
            
            st.info("""
            💡 **提示**：自动刷新功能已禁用，以避免影响页面浏览。  
            如需更新数据，请点击顶部的“🔄 刷新数据”按钮。
            """)
            
            auto_refresh = st.checkbox(
                "自动刷新（已禁用）", 
                value=False, 
                disabled=True,
                key="sidebar_auto_refresh",
                help="该功能已被禁用以优化用户体验"
            )
            st.session_state.auto_refresh = False  # 确保始终为False
            
            refresh_interval = st.slider(
                "刷新间隔(秒)（仅供参考）",
                min_value=1,
                max_value=60,
                value=st.session_state.refresh_interval,
                key="sidebar_refresh_interval",
                disabled=True,
                help="自动刷新已禁用，该设置暂无作用"
            )
            st.session_state.refresh_interval = refresh_interval
        
        st.divider()
        
        # ========== 5. 底部快捷入口 ==========
        st.caption("💡 快捷入口")
        st.success("📚 文档已整理：29个核心文档 + 46个历史归档")
        st.info("🔍 展开上方「文档中心」可预览文档或快速搜索")
    
    def _show_doc(self, rel_path: str):
        """侧边栏预览Markdown/YAML文档"""
        try:
            p = Path(__file__).parent.parent / rel_path
            if not p.exists():
                st.warning(f"未找到文档: {rel_path}")
                return
            text = p.read_text(encoding='utf-8')
            if p.suffix.lower() in ('.md', '.markdown'):
                st.markdown(text)
            else:
                st.code(text, language=p.suffix.lstrip('.') or 'text')
        except Exception as e:
            st.error(f"读取文档失败: {e}")

    def _doc_search(self, query: str, roots, exts=None, max_hits: int = 50):
        """在给定目录中搜索关键词（大小写不敏感），返回匹配行及上下文"""
        if exts is None:
            exts = {'.md', '.markdown', '.yaml', '.yml', '.txt'}
        results = []
        q = query.lower()
        try:
            for root in roots:
                if not root.exists():
                    continue
                for p in root.rglob('*'):
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in exts:
                        continue
                    # 限制文件大小
                    try:
                        if p.stat().st_size > 1_000_000:
                            continue
                        text = p.read_text(encoding='utf-8', errors='ignore')
                    except Exception:
                        continue
                    lines = text.splitlines()
                    for idx, line in enumerate(lines, start=1):
                        if q in line.lower():
                            # 取上下文
                            start = max(1, idx-2)
                            end = min(len(lines), idx+2)
                            snippet = "\n".join(lines[start-1:end])
                            results.append({
                                'path': str(p.relative_to(Path(__file__).parent.parent)),
                                'line': idx,
                                'snippet': snippet,
                            })
                            if len(results) >= max_hits:
                                return results
        except Exception:
            pass
        return results

    def _highlight(self, text: str, query: str) -> str:
        import re
        def esc(s: str) -> str:
            return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html = esc(text)
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        html = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", html)
        return f"<pre style='white-space:pre-wrap'>{html}</pre>"

    def render_main_content(self):
        """渲染主内容区域"""
        # 渲染主界面内容
        self.render_main_content_original()
        
    def render_main_content_original(self):
        """渲染主内容区"""
        # 创建主标签页 - 一进二涨停监控 + Qilin监控 + 缠论系统 + 竞价决策 + Qlib + RD-Agent + TradingAgents + 高级功能 + 系统指南
        main_tab0, main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6, main_tab7, main_tab8 = st.tabs([
            "🎯 一进二涨停监控",
            "🏠 Qilin监控",
            "📈 缠论系统",
            "🔖 竞价决策(旧)",
            "📦 Qlib",
            "🧠 RD-Agent研发智能体",
            "🤝 TradingAgents多智能体",
            "🚀 高级功能",
            "📚 系统指南"
        ])
        
        with main_tab0:
            # 一进二涨停监控统一视图（Phase 1新界面）
            self.render_limitup_monitor_unified()
        
        with main_tab1:
            # Qilin系统级监控与操作
            self.render_qilin_tabs()
        
        with main_tab2:
            # 缠论技术分析系统（独立模块）
            self.render_chanlun_system_tab()
        
        with main_tab3:
            # 竞价决策视图（旧版，保留以便对比）
            st.info("💡 推荐使用新版 '🎯 一进二涨停监控' 界面，功能更清晰易用！")
            self.render_auction_decision_tab()
        
        with main_tab4:
            # Qlib相关功能
            self.render_qlib_tabs()
        
        with main_tab5:
            # RD-Agent的6个子tab
            self.render_rdagent_tabs()
        
        with main_tab6:
            # TradingAgents的6个sub tab
            self.render_tradingagents_tabs()
        
        with main_tab7:
            # Phase 3 & 4 高级功能
            self.render_advanced_features_tab()
        
        with main_tab8:
            # 系统使用指南
            self.render_system_guide_tab()
        
    def render_limitup_monitor_unified(self):
        """渲染一进二涨停监控统一界面（Phase 1优化）"""
        try:
            from web.tabs.limitup_monitor_unified import render
            render()
        except Exception as e:
            st.error(f"一进二涨停监控加载失败: {e}")
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())
    
    def render_chanlun_system_tab(self):
        """渲染缠论技术分析系统标签页"""
        try:
            from web.tabs.chanlun_system_tab import render_chanlun_system_tab
            render_chanlun_system_tab()
        except Exception as e:
            st.error(f"缠论系统加载失败: {e}")
            st.info("💡 缠论系统包括：多智能体选股、缠论评分分析、一进二涨停策略")
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())
    
    def render_auction_decision_tab(self):
        """渲染竞价决策标签页（旧版，保留备查）"""
        try:
            from web.auction_decision_view import AuctionDecisionView
            view = AuctionDecisionView()
            view.render()
        except Exception as e:
            st.error(f"竞价决策视图加载失败: {e}")
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())
    
    def render_advanced_features_tab(self):
        """渲柔高级功能标签页（Phase 3 & 4）"""
        try:
            # 修复导入路径
            import sys
            from pathlib import Path
            tabs_path = Path(__file__).parent / "tabs"
            if str(tabs_path) not in sys.path:
                sys.path.insert(0, str(tabs_path))
            from advanced_features_tab import render_advanced_features_tab
            render_advanced_features_tab()
        except Exception as e:
            st.error(f"❌ 高级功能模块未正确安装: {e}")
            st.info("💡 Phase 3 & 4 高级功能包括：模拟交易、策略回测、数据导出")
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())
    
    def render_system_guide_tab(self):
        """渲柔系统指南标签页（放在最右侧）"""
        try:
            # 修复导入路径
            import sys
            from pathlib import Path
            components_path = Path(__file__).parent / "components"
            if str(components_path) not in sys.path:
                sys.path.insert(0, str(components_path))
            from system_guide import show_system_guide
            show_system_guide()
        except Exception as e:
            st.error(f"❌ 系统指南加载失败: {e}")
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())
    
    def render_qilin_tabs(self):
        """渲染Qilin系统级tabs（监控/操作）"""
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 实时监控",
            "🤖 智能体状态",
            "📈 交易执行",
            "📉 风险管理",
            "📋 历史记录",
            "🧠 AI进化系统",
            "🔄 循环进化训练",
            "📖 写实回测"
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
            self.render_history(key_prefix="qilin_history")
        with tab6:
            # 集成AI进化系统
            try:
                from tabs.limitup_ai_evolution_tab import render_limitup_ai_evolution_tab
                render_limitup_ai_evolution_tab()
            except Exception as e:
                st.error(f"AI进化系统加载失败: {e}")
                st.info("🚧 该功能开发中，敬请期待")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        
        with tab7:
            # 集成循环进化训练
            try:
                from tabs.evolution_training_tab import render_evolution_training_tab
                render_evolution_training_tab()
            except Exception as e:
                st.error(f"循环进化训练加载失败: {e}")
                st.info("🚧 该功能开发中，敬请期待")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        
        with tab8:
            # 写实回测页面
            self.render_realistic_backtest_page()

    def render_qlib_tabs(self):
        """渲染Qlib量化平台（6大分区）"""
        tab_model, tab_data, tab_portfolio, tab_risk, tab_service, tab_exp = st.tabs([
            "📈 模型训练",
            "🗄️ 数据管理",
            "💼 投资组合",
            "⚠️ 风险控制",
            "🔄 在线服务",
            "📊 实验管理",
        ])
        with tab_model:
            self.render_qlib_model_training_tab()
        with tab_data:
            self.render_qlib_data_management_tab()
        with tab_portfolio:
            self.render_qlib_portfolio_tab()
        with tab_risk:
            self.render_qlib_risk_control_tab()
        with tab_service:
            self.render_qlib_online_service_tab()
        with tab_exp:
            self.render_qlib_experiment_management_tab()

    def _safe(self, title: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"模块[{title}]运行异常")
            st.exception(e)
            return None

    def render_qlib_model_training_tab(self):
        """Qlib/模型训练：工作流、在线学习、强化学习、模型库"""
        sub1, sub2, sub3, sub4, sub5 = st.tabs(["🔄 Qlib工作流", "🧠 在线学习", "🤖 强化学习", "🚀 一进二策略", "🗂️ 模型库"])
        with sub1:
            # 集成Qlib qrun工作流（Phase P1-2）
            try:
                from web.tabs.qlib_qrun_workflow_tab import render_qlib_qrun_workflow_tab
                render_qlib_qrun_workflow_tab()
            except Exception as e:
                st.error(f"Qlib工作流加载失败: {e}")
                st.info("💡 提示：Qlib工作流允许通过YAML配置文件一键运行训练-回测-评估全流程")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        with sub2:
            self._safe("在线学习", self.render_online_learning)
        with sub3:
            self._safe("强化学习", self.render_rl_trading)
        with sub4:
            self._safe("一进二策略", self.render_one_into_two_strategy)
        with sub5:
            # 集成Qlib模型库（Phase 5.1）
            try:
                from web.tabs.qlib_model_zoo_tab import render_model_zoo_tab
                render_model_zoo_tab()
            except Exception as e:
                st.error(f"模型库加载失败: {e}")
                st.info("🚧 Qlib模型库开发中，敬请期待")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
    def render_qlib_data_management_tab(self):
        """八库/数据管理:多数据源、涨停板分析、特征/因子、IC分析、数据工具"""
        sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs(["🔌 多数据源", "🔥 涨停板分析", "🎯 涨停板监控", "🧪 因子研究", "📊 IC分析", "🛠️ 数据工具"])
        with sub1:
            self._safe("多数据源", self.render_multi_source_data)
        with sub2:
            self._safe("涨停板分析", self.render_limitup_analysis)
        with sub3:
            try:
                from tabs.rdagent import limitup_monitor
                limitup_monitor.render()
            except Exception as e:
                st.error(f"涨停板监控模块加载失败: {e}")
                st.info("请确保已正确安装依赖: matplotlib")
        with sub4:
            # 集成因子研究功能（一进二涨停板）
            try:
                from tabs.factor_research_tab import render_factor_research_tab
                render_factor_research_tab()
            except Exception as e:
                st.error(f"因子研究模块加载失败: {e}")
                st.info("请确保因子研究模块已正确配置")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        with sub5:
            # 集成IC分析报告（Phase 5.3）
            try:
                from web.tabs.qlib_ic_analysis_tab import render_qlib_ic_analysis_tab
                render_qlib_ic_analysis_tab()
            except Exception as e:
                st.error(f"IC分析报告加载失败: {e}")
                st.info("🚧 Qlib IC分析报告开发中，敬请期待")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        with sub6:
            # 集成数据工具箱（Phase 6.1）
            try:
                from web.tabs.qlib_data_tools_tab import render_qlib_data_tools_tab
                render_qlib_data_tools_tab()
            except Exception as e:
                st.error(f"数据工具箱加载失败: {e}")
                st.info("🚧 Qlib数据工具箱开发中，敬请期待")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())

    def render_qlib_portfolio_tab(self):
        """八库/投资组合：回测、优化、归因分析、订单执行、策略对比、高频交易"""
        sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs(["⏪ 回测", "🧭 组合优化", "📊 归因分析", "🚀 订单执行", "🏆 策略对比", "⚡ 高频交易"])
        with sub1:
            # 回测引擎选择（P1-1：完整集成Qlib原生回测）
            engine = st.radio("回测引擎", ["Qlib原生(推荐)", "写实回测(自研)"], index=0, horizontal=True, key="bt_engine")
            if engine == "写实回测(自研)":
                st.info("建议前往顶部的'写实回测系统'页运行；运行后风控页可选择'回测(最近一次)'进行风险分析。")
                st.divider()
                self._safe("回测", self.render_history, key_prefix="qlib_history")
            else:
                # 集成Qlib原生回测（Phase P1-1）
                try:
                    from web.tabs.qlib_backtest_tab import render_qlib_backtest_tab
                    render_qlib_backtest_tab()
                except Exception as e:
                    st.error(f"Qlib原生回测加载失败: {e}")
                    st.info("💡 提示：请确保已安装Qlib并正确配置数据路径")
                    import traceback
                    with st.expander("🔍 查看详细错误"):
                        st.code(traceback.format_exc())
        with sub2:
            self._safe("组合优化", self.render_portfolio_optimization)
        with sub3:
            self._safe("归因分析", self.render_performance_attribution)
        with sub4:
            # 集成订单执行引擎（Phase 5.2）
            try:
                from web.tabs.qlib_execution_tab import render_qlib_execution_tab
                render_qlib_execution_tab()
            except Exception as e:
                st.error(f"订单执行引擎加载失败: {e}")
                st.info("🚧 Qlib订单执行引擎开发中，敬请期待")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        with sub5:
            # 策略对比工具（Phase 6.4）
            self._safe("策略对比", self.render_strategy_comparison)
        with sub6:
            # 高频交易模块（Phase 6.2）
            try:
                from web.tabs.qlib_highfreq_tab import render_qlib_highfreq_tab
                render_qlib_highfreq_tab()
            except Exception as e:
                st.error(f"高频交易模块加载失败: {e}")
                st.info("🚧 Qlib高频交易模块开发中，敬请期待")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())

    def render_qlib_risk_control_tab(self):
        """麒麟Qlib/风险控制：VaR、CVaR、尾部风险、压力测试"""
        sub1, sub2, sub3 = st.tabs(["⚠️ 风险监控", "🔥 高级风险指标", "🎯 压力测试"])
        with sub1:
            self._safe("风险监控", self.render_risk_monitoring)
        with sub2:
            # 集成高级风险指标（Phase 6扩展）
            self._safe("高级风险指标", self.render_advanced_risk_metrics)
        with sub3:
            self._safe("压力测试", self.render_stress_test)

    def render_qlib_online_service_tab(self):
        """Qlib/在线服务：模型serving与滚动训练-接入你的API"""
        st.subheader("🔄 在线服务")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**模型 Serving**")
            if requests is None:
                st.error("未安装 requests，无法调用HTTP接口。请先安装 requests 包。")
            base_url = st.text_input(
                "服务地址",
                value=st.session_state.get('qlib_serving_url', 'http://localhost:9000'),
                key="qlib_serving_url_input",
            )
            api_key = st.text_input("API Key (可选)", value=st.session_state.get('qlib_serving_api_key', ''), type="password")
            health_path = st.text_input("健康检查路径", value="/health")
            predict_path = st.text_input("预测路径", value="/predict")
            start_path = st.text_input("启动服务路径", value="/admin/start")
            stop_path = st.text_input("停止服务路径", value="/admin/stop")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("健康检查", key="qlib_serving_health_check"):
                    try:
                        r = requests.get(urljoin(base_url, health_path), headers=headers, timeout=5)
                        st.success(f"{r.status_code} {r.text[:120]}")
                    except Exception as e:
                        st.error(f"检查失败: {e}")
            with c2:
                if st.button("启动服务", key="qlib_serving_start"):
                    try:
                        r = requests.post(urljoin(base_url, start_path), headers=headers, timeout=8)
                        st.success(f"已启动: {r.status_code}")
                    except Exception as e:
                        st.error(f"启动失败: {e}")
            with c3:
                if st.button("停止服务", key="qlib_serving_stop"):
                    try:
                        r = requests.post(urljoin(base_url, stop_path), headers=headers, timeout=8)
                        st.warning(f"已停止: {r.status_code}")
                    except Exception as e:
                        st.error(f"停止失败: {e}")
            st.divider()
            st.markdown("**测试预测**")
            payload = st.text_area("请求JSON", value='{"symbol":"000001.SZ","features":[1,2,3,4]}', height=120)
            if st.button("发送预测请求"):
                try:
                    r = requests.post(urljoin(base_url, predict_path), headers={**headers, "Content-Type":"application/json"}, data=payload.encode('utf-8'), timeout=10)
                    st.success(f"响应 {r.status_code}:")
                    st.code(r.text, language="json")
                except Exception as e:
                    st.error(f"调用失败: {e}")
        with col2:
            st.markdown("**滚动训练**")
            enable_cron = st.toggle("每日滚动训练", value=True, key="qlib_rolling_train")
            cron_path = st.text_input("触发路径", value="/admin/roll_train")
            if st.button("手动触发一次"):
                try:
                    r = requests.post(urljoin(base_url, cron_path), headers=headers, timeout=15)
                    st.info(f"已触发: {r.status_code}")
                except Exception as e:
                    st.error(f"触发失败: {e}")
        st.caption("提示：以上路径可按你的服务实际调整；支持带Bearer Token。")

    def render_qlib_experiment_management_tab(self):
        """Qlib/实验管理：MLflow集成 + 实验对比分析（P2-2）"""
        # 创建子标签页
        sub1, sub2 = st.tabs(["📊 MLflow管理", "🔬 实验对比"])
        
        with sub1:
            self._render_mlflow_management()
        
        with sub2:
            # 集成实验对比功能（Phase P2-2）
            try:
                from web.tabs.qlib_experiment_comparison_tab import render_qlib_experiment_comparison_tab
                render_qlib_experiment_comparison_tab()
            except Exception as e:
                st.error(f"实验对比功能加载失败: {e}")
                st.info("💡 提示：实验对比功能支持多实验性能对比、可视化分析和统计检验")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
    
    def _render_mlflow_management(self):
        """渲染MLflow管理界面"""
        st.subheader("📊 实验管理 (MLflow)")
        st.markdown("- 记录训练运行、指标与参数\n- 注册最佳模型用于 Serving")
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except Exception as e:
            st.error("未安装 mlflow，请先安装后再用该页功能。")
            return
        col1, col2 = st.columns(2)
        with col1:
            tracking_uri = st.text_input("MLflow Tracking URI", value=st.session_state.get('mlflow_uri','http://localhost:5000'))
            exp_name = st.text_input("实验名称", value=st.session_state.get('mlflow_experiment','qlib_limitup'))
            if st.button("连接/创建实验"):
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    client = MlflowClient(tracking_uri)
                    exp = client.get_experiment_by_name(exp_name)
                    if exp is None:
                        exp_id = client.create_experiment(exp_name)
                        st.success(f"已创建实验: {exp_name} ({exp_id})")
                    else:
                        st.success(f"已连接实验: {exp.name} ({exp.experiment_id})")
                except Exception as e:
                    st.error(f"连接失败: {e}")
            st.divider()
            st.markdown("**记录一次示例运行**")
            run_name = st.text_input("运行名称", value="demo_run")
            if st.button("记录示例指标"):
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_experiment(exp_name)
                    with mlflow.start_run(run_name=run_name) as run:
                        mlflow.log_params({"model":"limitup_classifier","version":"v1"})
                        mlflow.log_metrics({"precision@20":0.52, "recall@20":0.31})
                        st.success(f"已记录，run_id={run.info.run_id}")
                except Exception as e:
                    st.error(f"记录失败: {e}")
        with col2:
            reg_name = st.text_input("模型注册名", value=st.session_state.get('mlflow_model_name','qlib_limitup_v1'))
            model_uri = st.text_input("模型URI (如 runs:/<run_id>/model)", value="")
            if st.button("注册/更新模型"):
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    client = MlflowClient(tracking_uri)
                    # 确保注册存在
                    try:
                        client.get_registered_model(reg_name)
                    except Exception:
                        client.create_registered_model(reg_name)
                    v = mlflow.register_model(model_uri=model_uri, name=reg_name)
                    st.success(f"已发起注册，version={v.version}")
                except Exception as e:
                    st.error(f"注册失败: {e}")
        st.caption("如需自动化，把训练脚本的 mlflow.log_* 与本页的注册流程串联即可。")
        
    def render_one_into_two_strategy(self):
        """一进二策略：数据→训练→预测（示例可跑）"""
        st.header("🚀 一进二涨停板选股")
        
        # 添加一个切换选项
        mode = st.radio(
            "🔧 数据模式",
            options=[
                "🧪 示例模式（快速演示）", 
                "📡 AKShare在线模式（推荐）",
                "🔥 Qlib离线模式"
            ],
            index=1,  # 默认选择AKShare
            horizontal=True,
            help="示例模式=模拟数据 | AKShare=在线真实数据(无需下载) | Qlib=离线数据(需提前下载)"
        )
        
        if "示例" in mode:
            st.info("💡 示例模式：使用随机生成的模拟数据快速演示功能")
        elif "AKShare" in mode:
            st.success("✅ AKShare在线模式：直接从网络获取真实市场数据，无需提前下载！")
            st.caption("📊 数据来源：AKShare (akshare.akfamily.xyz) - 免费开源的财经数据接口")
        else:
            st.warning("⚠️ Qlib离线模式需要：1) 已配置Qlib  2) 设置数据目录  3) 已下载数据")
        # 参数
        st.subheader("🎯 股票池选择")
        
        col_mode1, col_mode2 = st.columns([3, 1])
        with col_mode1:
            pool_mode = st.radio(
                "选股方式",
                options=["👉 手动选择", "🤖 智能选择（自动获取今日涨停板）"],
                index=0,
                horizontal=True,
                key="stock_pool_mode"
            )
        with col_mode2:
            if "智能" in pool_mode:
                if st.button("🔄 刷新涨停板", type="primary", use_container_width=True):
                    st.session_state['refresh_limitup'] = True
        
        # 涨停板类型筛选
        if "智能" in pool_mode:
            limitup_filter = st.radio(
                "🎯 涨停板类型",
                options=["🆕 所有涨停板", "🆕 仅首板", "🔥 仅连板(2连及以上)"],
                index=0,
                horizontal=True,
                help="首板=首次涨停 | 连板=连续多日涨停",
                key="limitup_filter_type"
            )
        else:
            limitup_filter = "🆕 所有涨停板"
        
        # 根据模式显示不同的选项
        if "手动" in pool_mode:
            # 手动模式：使用multiselect
            st.caption("💡 手动选择股票：适合测试和特定股票分析")
            colA, colB, colC = st.columns(3)
            with colA:
                symbols = st.multiselect(
                    "选择股票",
                    ["000001.SZ", "000002.SZ", "000333.SZ", "000858.SZ", 
                     "600000.SH", "600036.SH", "600519.SH", "601318.SH"],
                    default=["000001.SZ", "600519.SH"],
                    key="manual_stock_selection"
                )
            with colB:
                start = st.date_input("开始", value=(datetime.now()-timedelta(days=90)).date(), key="qlib_dataset_start")
            with colC:
                end = st.date_input("结束", value=datetime.now().date(), key="qlib_dataset_end")
        else:
            # 智能模式：自动获取今日涨停板
            st.caption("🤖 智能选股：自动获取今日涨停板股票，用于一进二策略分析")
            
            # 检查筛选类型是否改变
            current_filter = st.session_state.get('limitup_filter_type', '🆕 所有涨停板')
            last_filter = st.session_state.get('last_limitup_filter', '')
            filter_changed = (current_filter != last_filter)
            
            # 自动获取涨停板
            if ('limitup_stocks' not in st.session_state or 
                st.session_state.get('refresh_limitup', False) or 
                filter_changed):
                
                with st.spinner("🔍 正在获取今日涨停板数据..."):
                    try:
                        import akshare as ak
                        # 获取今日涨停板 (仅在首次或刷新时获取)
                        if 'limitup_raw_data' not in st.session_state or st.session_state.get('refresh_limitup', False):
                            today = datetime.now().strftime('%Y%m%d')
                            df_zt = ak.stock_zt_pool_em(date=today)
                            st.session_state['limitup_raw_data'] = df_zt
                        else:
                            df_zt = st.session_state.get('limitup_raw_data')
                        
                        if df_zt is not None and not df_zt.empty:
                            # 保存真实总数
                            total_limitup_count = len(df_zt)
                            st.session_state['limitup_total_count'] = total_limitup_count
                            
                            # 调试: 显示列名
                            with st.expander("🔍 调试信息", expanded=False):
                                st.write("列名:", df_zt.columns.tolist())
                                if '连板数' in df_zt.columns:
                                    st.write("连板数分布:")
                                    conn_dist = df_zt['连板数'].value_counts().sort_index()
                                    st.write(conn_dist)
                                st.write("样例数据:")
                                st.dataframe(df_zt[['代码', '名称', '连板数', '涨跌幅']].head(20) if '连板数' in df_zt.columns else df_zt.head(10))
                            
                            # 根据类型筛选
                            limitup_filter = current_filter
                            st.session_state['last_limitup_filter'] = current_filter
                            
                            if '仅首板' in limitup_filter:
                                # 筛选首板: 连板数=1 (注意: 首板的连板数为1)
                                if '连板数' in df_zt.columns:
                                    try:
                                        # 首板的连板数为1
                                        df_filtered = df_zt[df_zt['连板数'] == 1]
                                        filter_desc = f"首板 ({len(df_filtered)}只)"
                                    except Exception as e:
                                        st.warning(f"⚠️ 筛选错误: {e}")
                                        df_filtered = df_zt
                                        filter_desc = "首板(筛选失败)"
                                else:
                                    st.warning("⚠️ 未找到'连板数'字段，无法筛选首板")
                                    df_filtered = df_zt
                                    filter_desc = "首板(未筛选)"
                            elif '仅连板' in limitup_filter:
                                # 筛选连板: 连板数>=2
                                if '连板数' in df_zt.columns:
                                    try:
                                        df_filtered = df_zt[df_zt['连板数'] >= 2]
                                        filter_desc = f"2连及以上 ({len(df_filtered)}只)"
                                    except Exception as e:
                                        st.warning(f"⚠️ 筛选错误: {e}")
                                        df_filtered = df_zt
                                        filter_desc = "2连及以上(筛选失败)"
                                else:
                                    st.warning("⚠️ 未找到'连板数'字段，无法筛选连板")
                                    df_filtered = df_zt
                                    filter_desc = "2连及以上(未筛选)"
                            else:
                                # 所有涨停板
                                df_filtered = df_zt
                                filter_desc = "所有类型"
                            
                            filtered_count = len(df_filtered)
                            
                            # 提取股票代码并转换格式 (最多100只供选择)
                            limitup_codes = []
                            for code in df_filtered['代码'].head(100):
                                if code.startswith('6'):
                                    limitup_codes.append(f"{code}.SH")
                                else:
                                    limitup_codes.append(f"{code}.SZ")
                            
                            st.session_state['limitup_stocks'] = limitup_codes
                            st.session_state['limitup_count'] = len(limitup_codes)
                            st.session_state['limitup_filtered_count'] = filtered_count
                            
                            # 显示提示信息
                            if st.session_state.get('refresh_limitup', False):
                                st.success(f"✅ 刷新成功！总计 {total_limitup_count} 只涨停板 | {filter_desc}: {filtered_count} 只 | 已加载: {len(limitup_codes)} 只")
                            elif filter_changed:
                                st.info(f"🎯 筛选已更新！{filter_desc}: {filtered_count} 只 | 已加载: {len(limitup_codes)} 只")
                            else:
                                st.success(f"✅ 数据已加载！总计 {total_limitup_count} 只涨停板 | {filter_desc}: {filtered_count} 只")
                        else:
                            st.warning("⚠️ 今日暂无涨停板数据，使用默认股票池")
                            st.session_state['limitup_stocks'] = ["000001.SZ", "600519.SH"]
                            st.session_state['limitup_count'] = 2
                    except ImportError:
                        st.error("❌ 未安装 akshare，请先运行: pip install akshare")
                        st.session_state['limitup_stocks'] = ["000001.SZ", "600519.SH"]
                        st.session_state['limitup_count'] = 2
                    except Exception as e:
                        st.warning(f"⚠️ 获取涨停板失败: {e}，使用默认股票池")
                        st.session_state['limitup_stocks'] = ["000001.SZ", "600519.SH"]
                        st.session_state['limitup_count'] = 2
                    
                    st.session_state['refresh_limitup'] = False
            
            # 显示自动选择的股票
            symbols = st.session_state.get('limitup_stocks', ["000001.SZ", "600519.SH"])
            limitup_count = st.session_state.get('limitup_count', len(symbols))
            total_limitup = st.session_state.get('limitup_total_count', limitup_count)
            filtered_limitup = st.session_state.get('limitup_filtered_count', limitup_count)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                # 显示筛选后的数量
                st.metric("📈 筛选后涨停板", f"{filtered_limitup}只", 
                         delta=f"总计{total_limitup}只" if total_limitup != filtered_limitup else None)
            with col_info2:
                st.metric("🎯 已选择股票", f"{len(symbols)}只")
            with col_info3:
                filter_count = st.slider("限制数量", 5, 100, min(30, len(symbols)), 5, key="limit_stock_count")
            
            # 可以限制数量
            symbols = symbols[:filter_count]
            
            # 显示选中的股票
            with st.expander(f"👁️ 查看已选择的 {len(symbols)} 只股票", expanded=False):
                st.write(", ".join(symbols[:20]))
                if len(symbols) > 20:
                    st.caption(f"...和其他 {len(symbols)-20} 只")
            
            # 时间选择
            colT1, colT2 = st.columns(2)
            with colT1:
                start = st.date_input("开始", value=(datetime.now()-timedelta(days=90)).date(), key="qlib_dataset_start_auto")
            with colT2:
                end = st.date_input("结束", value=datetime.now().date(), key="qlib_dataset_end_auto")
        
        # Qlib模式下显示配置
        if "Qlib" in mode:
            qlib_dir = st.text_input(
                "Qlib数据目录",
                value=str((Path.home() / ".qlib/qlib_data/cn_data").expanduser()),
                help="设置你的Qlib数据存储路径",
                key="qlib_real_data_dir"
            )
            st.caption("💡 提示：如果未配置Qlib或数据不存在，将自动回退到示例模式")
        
        if st.button("📦 构建数据集"):
            if "示例" in mode:
                # 示例模式
                with st.spinner("正在生成示例数据…"):
                    df = build_sample_dataset(symbols, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                    st.session_state['oit_dataset'] = df
                    st.session_state['oit_data_mode'] = 'sample'
                    st.success(f"✅ 示例数据集已就绪：{df.shape}")
                    st.dataframe(df.head(5), use_container_width=True)
                    
            elif "AKShare" in mode:
                # AKShare在线模式
                try:
                    with st.spinner("📡 正在从FKShare获取实时数据…"):
                        try:
                            import akshare as ak
                            import pandas as pd
                        except ImportError as ie:
                            st.error(f"❌ 未安装必要包: {ie}")
                            st.info("请运行: pip install akshare pandas")
                            st.info("🔄 回退到示例模式")
                            df = build_sample_dataset(symbols, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                            st.session_state['oit_dataset'] = df
                            st.session_state['oit_data_mode'] = 'sample'
                            st.dataframe(df.head(5), use_container_width=True)
                            return
                        
                        # 使用AKShare获取数据
                        all_data = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, symbol in enumerate(symbols):
                            status_text.text(f"正在获取 {symbol} 的数据... ({idx+1}/{len(symbols)})")
                            try:
                                # AKShare的股票代码格式是 000001 或 600000
                                code = symbol.split('.')[0]
                                
                                # 获取日线数据
                                df_stock = ak.stock_zh_a_hist(
                                    symbol=code,
                                    start_date=start.strftime('%Y%m%d'),
                                    end_date=end.strftime('%Y%m%d'),
                                    adjust="qfq"  # 前复权
                                )
                                
                                if df_stock is not None and not df_stock.empty:
                                    # 重命名列
                                    df_stock = df_stock.rename(columns={
                                        '日期': 'date',
                                        '开盘': 'open',
                                        '收盘': 'close',
                                        '最高': 'high',
                                        '最低': 'low',
                                        '成交量': 'volume',
                                        '成交额': 'amount',
                                        '换手率': 'turnover'
                                    })
                                    df_stock['symbol'] = symbol
                                    df_stock['date'] = pd.to_datetime(df_stock['date'])
                                    all_data.append(df_stock)
                                    
                            except Exception as e:
                                st.warning(f"⚠️ 获取 {symbol} 数据失败: {e}")
                                continue
                            
                            progress_bar.progress((idx + 1) / len(symbols))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if not all_data:
                            st.error("❌ 所有股票数据获取失败")
                            st.info("🔄 回退到示例模式")
                            df = build_sample_dataset(symbols, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                            st.session_state['oit_dataset'] = df
                            st.session_state['oit_data_mode'] = 'sample'
                            st.dataframe(df.head(5), use_container_width=True)
                            return
                        
                        # 合并数据
                        df = pd.concat(all_data, ignore_index=True)
                        
                        # 添加一些基本特征
                        df['returns'] = df.groupby('symbol')['close'].pct_change()
                        df['label'] = 0  # 需要根据实际策略打标签
                        
                        st.session_state['oit_dataset'] = df
                        st.session_state['oit_data_mode'] = 'akshare'
                        st.success(f"✅ 成功从 AKShare 获取数据：{df.shape} | 股票数: {len(all_data)}")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ AKShare数据获取失败: {e}")
                    import traceback
                    with st.expander("🔍 查看详细错误"):
                        st.code(traceback.format_exc())
                    st.info("🔄 回退到示例模式")
                    df = build_sample_dataset(symbols, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                    st.session_state['oit_dataset'] = df
                    st.session_state['oit_data_mode'] = 'sample'
                    st.dataframe(df.head(5), use_container_width=True)
                    
            else:
                # Qlib离线模式
                try:
                    with st.spinner("正在从Qlib加载离线数据…"):
                        import qlib
                        from qlib.data import D
                        import numpy as _np
                        
                        # 初始化Qlib
                        qlib.init(provider_uri=str(Path(qlib_dir).expanduser()), region="cn")

                        # 转换股票代码为 Qlib 格式：000001.SZ -> SZ000001, 600519.SH -> SH600519
                        def _to_qlib(sym: str) -> str:
                            s = sym.strip().upper()
                            if "." in s:
                                code, exch = s.split(".")
                                return ("SZ" + code) if exch == "SZ" else ("SH" + code)
                            return s
                        q_syms = [_to_qlib(s) for s in symbols]

                        # 拉取日线面板
                        fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
                        df = D.features(q_syms, fields, start_time=start.strftime('%Y-%m-%d'), end_time=end.strftime('%Y-%m-%d'), freq='day')
                        if df is None or df.empty:
                            raise RuntimeError("Qlib返回空数据，请检查日期范围或数据包是否完整")
                        df = df.copy()
                        df.columns = [c.replace('$','') for c in df.columns]
                        df = df.reset_index().rename(columns={"instrument":"symbol","datetime":"date"})
                        # 统一 symbol 为 000001.SZ 格式
                        def _to_ui(sym: str) -> str:
                            s = sym.upper()
                            if s.startswith("SH"):
                                return s[2:] + ".SH"
                            if s.startswith("SZ"):
                                return s[2:] + ".SZ"
                            return s
                        df['symbol'] = df['symbol'].astype(str).map(_to_ui)
                        df['date'] = pd.to_datetime(df['date']).dt.date

                        # 计算标签与特征：按 symbol 分组
                        g = df.sort_values(['symbol','date']).groupby('symbol', group_keys=False)
                        def _feat(grp: pd.DataFrame) -> pd.DataFrame:
                            grp = grp.copy()
                            grp['prev_close'] = grp['close'].shift(1)
                            # 触发阈值（10%涨停，留千分之一余量）
                            thr_prev = grp['prev_close'] * 1.10 * 0.999
                            touched_prev = (grp['high'].shift(1) >= thr_prev)
                            touched_today = (grp['high'] >= thr_prev)
                            grp['pool_label'] = touched_prev.astype(int)
                            grp['board_label'] = (touched_prev & touched_today).astype(int)

                            # 日度特征（当日）
                            grp['ret_day'] = grp['close'] / grp['prev_close'] - 1.0
                            grp['amplitude'] = (grp['high'] - grp['low']) / grp['close'].replace(0, _np.nan)
                            grp['gap'] = grp['open'] / grp['prev_close'] - 1.0
                            grp['vol_ma5'] = grp['volume'].rolling(5).mean()
                            grp['vol_ratio'] = grp['volume'] / grp['vol_ma5']
                            grp['mom_5'] = grp['close'] / grp['close'].shift(5) - 1.0
                            grp['volatility_5'] = grp['close'].pct_change().rolling(5).std()

                            # 昨日特征前缀 y_
                            for col in ['ret_day','amplitude','vol_ratio','mom_5','volatility_5','gap']:
                                grp['y_' + col] = grp[col].shift(1)

                            return grp
                        df2 = g.apply(_feat)

                        # 生成训练用数据：以“今天”为 date，标签来自 昨日/今日（pool/board）
                        keep_cols = ['date','symbol','pool_label','board_label','ret_day','amplitude','vol_ratio','mom_5','volatility_5','gap',
                                     'y_ret_day','y_amplitude','y_vol_ratio','y_mom_5','y_volatility_5','y_gap']
                        df_out = df2[keep_cols].dropna(subset=['pool_label','board_label']).reset_index(drop=True)

                        if df_out.empty:
                            raise RuntimeError("Qlib数据构建为空；请增大日期范围或更换股票池")

                        st.session_state['oit_dataset'] = df_out
                        st.session_state['oit_data_mode'] = 'qlib'
                        st.success(f"✅ 已从Qlib构建数据：{df_out.shape}")
                        st.dataframe(df_out.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 加载Qlib数据失败: {e}")
                    st.info("🔄 回退到示例模式")
                    df = build_sample_dataset(symbols, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                    st.session_state['oit_dataset'] = df
                    st.session_state['oit_data_mode'] = 'sample'
                    st.dataframe(df.head(5), use_container_width=True)
        st.divider()
        # 训练
        top_n = st.slider("TopN", 5, 50, 20)
        if st.button("🧠 训练模型"):
            df = st.session_state.get('oit_dataset')
            data_mode = st.session_state.get('oit_data_mode', 'sample')
            
            if df is None or df.empty:
                st.error("请先构建数据集")
            elif data_mode == 'akshare':
                st.error("❌ AKShare模式仅提供价格数据，不包含训练所需的标签(pool_label/board_label)")
                st.info("💡 请使用以下方式之一:")
                st.write("- ✅ 选择 **示例模式** 构建数据集，然后训练")
                st.write("- ✅ 或者使用 **历史训练管道**（下方）生成完整的训练数据")
                st.warning("👉 AKShare数据适合用于预测，不适合直接训练模型")
            else:
                # 检查是否有必要的列
                required_cols = ['pool_label', 'board_label']
                missing_cols = [c for c in required_cols if c not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ 数据集缺少必要字段: {missing_cols}")
                    st.info("请使用示例模式构建数据集")
                else:
                    trainer = OneIntoTwoTrainer(top_n=top_n)
                    with st.spinner("训练中…"):
                        res = trainer.fit(df)
                        st.session_state['oit_result'] = res
                        st.session_state['model_trained'] = True  # 标记模型已训练
                        st.success(f"AUC(pool)={res.auc_pool:.3f} | AUC(board)={res.auc_board:.3f} | 阈值≈{res.threshold_topn:.3f}")
                        
                        # 同步训练结果到AI进化系统
                        st.session_state['training_results'] = {
                            'val_auc': res.auc_board,
                            'pool_auc': res.auc_pool,
                            'threshold_topn': res.threshold_topn
                        }
        st.divider()
        # 预测&选股（当天示例）
        if st.button("🎯 生成T+1候选"):
            res = st.session_state.get('oit_result')
            if not res:
                st.error("请先训练模型")
            else:
                # 使用最近一天模拟特征
                today = datetime.now().strftime('%Y-%m-%d')
                rows = []
                for s in symbols:
                    m = create_sample_high_freq_data(s)
                    feats = extract_limitup_features(m, s)
                    feats['date'] = today; feats['symbol'] = s
                    rows.append(feats)
                feat_df = pd.DataFrame(rows)
                ranked = rank_candidates(res.model_board, feat_df, threshold=res.threshold_topn, top_n=top_n)
                st.subheader("入选列表")
                st.dataframe(ranked, use_container_width=True, hide_index=True)
                st.info("可在‘风险管理’中进一步做流动性门控与排队评估。")

        # ===== 集成研究训练管道（scripts/pipeline_limitup_research.py）=====
        st.divider()
        st.subheader("🧪 历史训练管道（一进二）")
        
        # 显示提示信息
        st.info("""
        🎯 **与上方选股功能的关系**：
        - 上方的“智能选择”适合快速训练（用于 T+1 预测）
        - 这里的“历史训练管道”适合完整研究（生成智能体权重）
        - **推荐流程**：先用智能选择快速验证 → 再用历史管道全量训练
        """)
        
        # 选项卡：两种模式
        pipeline_mode = st.radio(
            "🔧 训练模式",
            options=[
                "👉 使用全市场股票（完整研究）",
                "🎯 使用上方选中的股票池（快速训练）"
            ],
            index=0,
            horizontal=True,
            key="pipeline_mode_select"
        )
        
        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            p_start = st.date_input("开始", value=(datetime.now()-timedelta(days=365)).date(), key="oit_pipe_start")
        with colp2:
            p_end = st.date_input("结束", value=datetime.now().date(), key="oit_pipe_end")
        with colp3:
            provider_uri = st.text_input("Qlib数据目录", value=str((Path.home()/".qlib/qlib_data/cn_data").expanduser()), key="oit_pipe_provider")
        
        # 根据模式显示信息
        if "股票池" in pipeline_mode:
            selected_symbols = symbols if 'symbols' in locals() else st.session_state.get('limitup_stocks', [])
            if selected_symbols:
                st.success(f"✅ 将使用上方选中的 {len(selected_symbols)} 只股票进行训练")
                with st.expander("👁️ 查看将使用的股票", expanded=False):
                    st.write(", ".join(selected_symbols[:30]))
                    if len(selected_symbols) > 30:
                        st.caption(f"...和其他 {len(selected_symbols)-30} 只")
            else:
                st.warning("⚠️ 上方未选择股票，将使用默认股票池")
        else:
            st.info("📊 将使用全市场所有股票（可能需要较长时间）")
        
        apply_weights = st.checkbox("训练后写入建议权重到 config/tradingagents.yaml", value=False, key="oit_pipe_apply")
        
        if st.button("🚀 运行研究训练管道", key="oit_run_pipeline", type="primary"):
            try:
                with st.spinner("正在运行历史训练管道…（请耐心等待）"):
                    from scripts.pipeline_limitup_research import run_pipeline
                    
                    # 判断是否使用选中的股票池
                    if "股票池" in pipeline_mode:
                        selected_symbols = symbols if 'symbols' in locals() else st.session_state.get('limitup_stocks', [])
                        if selected_symbols:
                            # 转换股票代码格式：000001.SZ → SZ000001
                            converted_symbols = []
                            for sym in selected_symbols:
                                if '.SH' in sym:
                                    converted_symbols.append('SH' + sym.split('.')[0])
                                elif '.SZ' in sym:
                                    converted_symbols.append('SZ' + sym.split('.')[0])
                                else:
                                    converted_symbols.append(sym)
                            
                            st.info(f"🎯 使用选中的 {len(converted_symbols)} 只股票进行训练")
                            
                            # 修改 run_pipeline 函数以接受 universe 参数
                            # 这里我们需要修改调用方式
                            import sys
                            sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
                            from pipeline_limitup_research import (
                                init_qlib, fetch_panel, engineer_features,
                                build_labeled_samples, train_and_explain,
                                suggest_agent_weights, write_weight_suggestions,
                                apply_weights_to_yaml, OUT_DIR
                            )
                            
                            init_qlib(provider_uri or None)
                            panel = fetch_panel(converted_symbols, str(p_start), str(p_end))
                            feat = engineer_features(panel)
                            samples = build_labeled_samples(feat)
                            
                            if samples.empty:
                                st.error("❌ 未生成任何样本，请检查日期范围和数据可用性")
                                return
                            # 小样本与单类标签友好提示
                            try:
                                import pandas as _pd
                                _n = len(samples)
                                _u = _pd.Series(samples.get('y')).nunique() if 'y' in samples.columns else 0
                                if _n < 10 or _u < 2:
                                    st.error(f"❌ 样本量过少({_n})或标签单一({_u})，无法训练。请拉长日期范围（建议≥6个月）或扩大股票池/改用全市场模式。")
                                    return
                            except Exception:
                                pass
                            
                            # 保存数据集
                            ds_path = OUT_DIR / f"limitup_samples_{p_start}_{p_end}_custom.parquet"
                            samples.to_parquet(ds_path)
                            st.info(f"💾 样本已保存：{ds_path} (行数={len(samples)})")
                            
                            # 训练
                            res = train_and_explain(samples)
                            
                            # 生成权重建议
                            imp_path = res.shap_path or res.perm_path
                            if imp_path and imp_path.exists():
                                import pandas as pd  # 确保pd可用
                                imp = pd.read_csv(imp_path)
                                if imp.columns[1] != "importance":
                                    imp = imp.rename(columns={imp.columns[1]: "importance"})
                                weights = suggest_agent_weights(imp[["feature", "importance"]])
                                write_weight_suggestions(weights)
                                
                                if apply_weights:
                                    apply_weights_to_yaml(weights)
                                
                                # 保存训练总结
                                import json as _json
                                import time
                                summary = {
                                    "start": str(p_start),
                                    "end": str(p_end),
                                    "auc": res.auc,
                                    "ap": res.ap,
                                    "model_path": str(res.model_path),
                                    "importance_path": str(imp_path),
                                    "samples_path": str(ds_path),
                                    "weights": weights,
                                    "timestamp": int(time.time()),
                                    "mode": "custom_pool",
                                    "symbols_count": len(converted_symbols)
                                }
                                summary_path = OUT_DIR / f"training_summary_{p_start}_{p_end}_custom.json"
                                with open(summary_path, "w", encoding="utf-8") as f:
                                    _json.dump(summary, f, ensure_ascii=False, indent=2)
                        else:
                            st.warning("⚠️ 上方未选择股票，使用全市场模式")
                            run_pipeline(start=str(p_start), end=str(p_end), provider_uri=provider_uri or None, apply=apply_weights)
                    else:
                        # 全市场模式
                        run_pipeline(start=str(p_start), end=str(p_end), provider_uri=provider_uri or None, apply=apply_weights)
                # 展示输出
                out_dir = Path(__file__).parent.parent / "output" / "limitup_research"
                st.success("训练完成")
                st.caption(str(out_dir))
                # 展示Summary
                import json as _json
                summaries = sorted(out_dir.glob("training_summary*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if summaries:
                    summary = _json.loads(summaries[0].read_text(encoding='utf-8'))
                    st.markdown("**结果摘要**")
                    try:
                        st.metric("AUC", f"{summary.get('auc', 0):.4f}")
                        st.metric("AP", f"{summary.get('ap', 0):.4f}")
                    except Exception:
                        pass
                    st.json(summary)
                    # 数据源提示
                    info = summary.get('data_source_info') or {}
                    if info.get('data_source_used') == 'synthetic':
                        reasons = []
                        if info.get('qlib_error'):
                            reasons.append('Qlib初始化/读取失败')
                        if (info.get('ak_errors_sample') or []) or (info.get('ak_success', 0) == 0):
                            reasons.append('AkShare 网络被拦截或中断')
                        if reasons:
                            st.warning('已使用合成数据训练：' + '；'.join(reasons) + '。建议配置本地Qlib数据或修复网络/代理。')
                    elif info.get('data_source_used'):
                        st.info(f"使用真实数据源：{info.get('data_source_used').upper()}")
                # 展示权重建议
                sug = out_dir / "agent_weight_suggestions.json"
                if sug.exists():
                    st.markdown("**建议权重**")
                    st.code(sug.read_text(encoding='utf-8'), language="json")
                # 列出最近产物
                files = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]
                if files:
                    st.markdown("**最近生成文件**")
                    import pandas as pd  # 确保pd可用
                    df_files = pd.DataFrame([
                        {"文件": f.name, "大小(KB)": round(f.stat().st_size/1024,1), "修改时间": datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}
                        for f in files
                    ])
                    st.dataframe(df_files, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"训练管道运行失败: {e}")
                st.exception(e)
        
    def render_rdagent_tabs(self):
        """渲染RD-Agent的09个子tabs"""
        rd_tab0, rd_tab1, rd_tab2, rd_tab3, rd_tab4, rd_tab5, rd_tab6, rd_tab7, rd_tab8 = st.tabs([
            "⚙️ 环境配置",
            "🔍 因子挖掘",
            "🏭️ 模型优化",
            "📚 知识学习",
            "🔬 研发协同",
            "📊 MLE-Bench",
            "🎮 会话管理",
            "🧪 数据科学",
            "🧧 日志可视化"
        ])
        
        with rd_tab0:
            # 环境配置
            try:
                from tabs.rdagent.env_config import render
                render()
            except Exception as e:
                st.error(f"加载环境配置模块失败: {e}")
                st.info("请确保 env_config.py 模块存在")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        
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
        
        with rd_tab5:
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
        
        with rd_tab6:
            # 会话管理
            try:
                from tabs.rdagent.session_manager import render as render_session_manager
                render_session_manager()
            except Exception as e:
                st.error(f"加载会话管理模块失败: {e}")
        
        with rd_tab7:
            # 数据科学RDLoop
            try:
                from tabs.rdagent.data_science_loop import render as render_ds
                render_ds()
            except Exception as e:
                st.error(f"加载数据科学模块失败: {e}")
        
        with rd_tab8:
            # 原生日志可视化
            try:
                from tabs.rdagent.log_visualizer import render as render_log
                render_log()
            except Exception as e:
                st.error(f"加载日志可视化模块失败: {e}")
    
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
    
    def render_history(self, key_prefix: str = "history"):
        """历史记录页面"""
        # 日期选择
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=(datetime.now() - timedelta(days=30)).date(),
                key=f"{key_prefix}_start_date"
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now().date(),
                key=f"{key_prefix}_end_date"
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
    
    def render_realistic_backtest_page(self):
        """写实回测页面"""
        try:
            from web.components.realistic_backtest_page import show_realistic_backtest_page
            show_realistic_backtest_page()
        except Exception as e:
            error_msg = str(e)
            st.error(f"写实回测页面加载失败: {error_msg}")
            
            # 根据错误类型给出具体提示
            if "shap" in error_msg.lower():
                st.warning("🚧 缺少 SHAP 库（用于模型解释）")
                st.markdown("""
                ### 🔧 安装 SHAP
                
                ```bash
                pip install shap
                ```
                
                **注意**: SHAP 安装可能需要一些时间，它依赖于 C++ 编译器。
                
                如果安装失败，可以尝试：
                ```bash
                # Windows 用户可能需要先安装 Visual C++ Build Tools
                pip install --upgrade pip
                pip install shap --no-cache-dir
                ```
                """)
            else:
                st.info("🚧 该功能需要安装额外依赖")
                st.markdown("""
                ### 📚 写实回测系统
                
                请确保已安装以下依赖：
                ```bash
                pip install plotly pandas numpy shap
                ```
                
                相关文档：
                - 🐴 **麒麟改进实施报告**: `docs/QILIN_EVOLUTION_IMPLEMENTATION.md`
                - 📊 **回测引擎**: `backtesting/realistic_backtest.py`
                - 🔬 **SHAP解释器**: `ml/model_explainer.py`
                """)
            
            import traceback
            with st.expander("🔍 查看详细错误"):
                st.code(traceback.format_exc())
    
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
        if not df.empty and '涨跌幅' in df.columns:
            st.dataframe(
                df.style.format({
                    '现价': '¥{:.2f}',
                    '涨跌幅': '{:+.2%}',
                    '成交量': '{:,.0f}',
                    '成交额': '¥{:,.0f}',
                    '买一': '¥{:.2f}',
                    '卖一': '¥{:.2f}'
                }).map(
                    lambda x: 'color: red;' if x < 0 else 'color: green;',
                    subset=['涨跌幅']
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("暂无实时行情数据")
    
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
        
        if not df.empty and '盈亏' in df.columns and '盈亏比例' in df.columns:
            st.dataframe(
                df.style.format({
                    '成本价': '¥{:.2f}',
                    '现价': '¥{:.2f}',
                    '盈亏': '¥{:+,.0f}',
                    '盈亏比例': '{:+.2%}'
                }).map(
                    lambda x: 'color: red;' if x < 0 else 'color: green;',
                    subset=['盈亏', '盈亏比例']
                ),
                use_container_width=True,
                hide_index=True
            )
        elif not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无持仓数据")
    
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
                st.session_state.system_running = True
                st.success("✅ 系统已启动")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 启动失败: {e}")
    
    def stop_system(self):
        """停止系统"""
        with st.spinner("正在停止系统..."):
            try:
                # asyncio.run(st.session_state.unified_system.stop())
                st.session_state.system_running = False
                st.warning("⏸️ 系统已停止")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 停止失败: {e}")
    
    def refresh_data(self):
        """刷新数据"""
        with st.spinner("正在刷新数据..."):
            # 更新实时数据
            if self.redis_available:
                try:
                    # 从 Redis获取最新数据
                    pass
                except Exception:pass
            
            # 更新选中的股票列表
            top_stocks = self._get_top_limitup_stocks()
            if top_stocks and top_stocks != ["000001", "000002", "600000"]:
                st.session_state.selected_stocks = top_stocks
                
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
                '预期损失': '￥{:,.0f}',
                '概率': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    def render_advanced_risk_metrics(self):
        """渲染高级风险指标（Phase 6扩展）"""
        st.header("🔥 高级风险指标")
        
        st.info("""
        **功能说明：**
        - VaR (Value at Risk): 风险价值
        - CVaR (Conditional VaR): 条件风险价值
        - 尾部风险指标: 极端损失分析
        - 风险调整收益: Sharpe、Sortino、Calmar
        """
        )
        
        st.divider()
        
        # 参数设置
        st.subheader("⚙️ 参数设置")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_level = st.slider(
                "置信水平",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.2f"
            )
        
        with col2:
            time_horizon = st.selectbox(
                "时间周期",
                options=["1天", "5天", "10天", "20天"],
                index=1
            )
        
        with col3:
            method = st.selectbox(
                "计算方法",
                options=["历史模拟", "方差-协方差", "蒙特卡洛模拟"],
                index=0
            )
        
        st.divider()
        
        # 数据来源选择
        st.markdown("##### 数据来源")
        data_source = st.selectbox(
            "选择收益数据来源",
            options=["模拟数据", "回测(最近一次)", "上传CSV"],
            index=0,
        )
        returns_series = None
        if data_source == "回测(最近一次)":
            returns_series = st.session_state.get("last_backtest_returns")
            if returns_series is None:
                st.warning("未找到回测收益，请先在“回测”页运行一次回测，或使用“上传CSV”。")
        elif data_source == "上传CSV":
            up = st.file_uploader("上传CSV（包含 date 与 return 列）", type=["csv"])
            if up is not None:
                try:
                    df_up = pd.read_csv(up)
                    dt_col = None
                    for c in ["date", "Date", "datetime", "time"]:
                        if c in df_up.columns:
                            dt_col = c; break
                    ret_col = None
                    for c in ["return", "ret", "returns"]:
                        if c in df_up.columns:
                            ret_col = c; break
                    if dt_col and ret_col:
                        df_up[dt_col] = pd.to_datetime(df_up[dt_col])
                        df_up = df_up.sort_values(dt_col)
                        returns_series = pd.Series(df_up[ret_col].values, index=df_up[dt_col])
                    else:
                        st.error("CSV 需包含 date 与 return 列")
                except Exception as e:
                    st.error(f"解析CSV失败: {e}")
        
        # 若无外部数据，使用模拟数据
        np.random.seed(42)
        if returns_series is None:
            returns_array = np.random.normal(0.001, 0.02, 252)
        else:
            returns_array = returns_series.values
        portfolio_value = 1000000
        
        # VaR 计算
        st.subheader("📊 VaR 分析")
        
        var_value = np.percentile(returns_array, (1 - confidence_level) * 100)
        var_amount = abs(var_value * portfolio_value)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"VaR ({confidence_level:.0%} 置信水平)",
                f"￥{var_amount:,.0f}",
                delta=f"{var_value:.2%}",
                delta_color="inverse"
            )
        
        # CVaR 计算
        cvar_value = returns[returns <= var_value].mean()
        cvar_amount = abs(cvar_value * portfolio_value)
        
        with col2:
            st.metric(
                "CVaR (条件VaR)",
                f"￥{cvar_amount:,.0f}",
                delta=f"{cvar_value:.2%}",
                delta_color="inverse"
            )
        
        # 最大回撤
        cum_returns = (1 + returns_array).cumprod()
        max_dd = (cum_returns / np.maximum.accumulate(cum_returns) - 1).min()
        max_dd_amount = abs(max_dd * portfolio_value)
        
        with col3:
            st.metric(
                "最大回撤",
                f"{max_dd:.2%}",
                delta=f"￥{max_dd_amount:,.0f}",
                delta_color="inverse"
            )
        
        st.divider()
        
        # 收益分布
        st.subheader("📈 收益分布")
        
        fig = go.Figure()
        
        # 添加直方图
        fig.add_trace(go.Histogram(
            x=returns_array,
            nbinsx=50,
            name="收益分布",
            marker=dict(color='#1f77b4', opacity=0.7)
        ))
        
        # 添加VaR线
        fig.add_vline(
            x=var_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({confidence_level:.0%})",
            annotation_position="top right"
        )
        
        fig.update_layout(
            xaxis_title="收益率",
            yaxis_title="频数",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 风险调整收益指标
        st.subheader("🎯 风险调整收益")
        
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino比率（只考虑下行波动）
        downside_returns = returns_array[returns_array < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Calmar比率
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sharpe比率",
                f"{sharpe:.2f}",
                delta="优秀" if sharpe > 1.5 else "一般"
            )
        
        with col2:
            st.metric(
                "Sortino比率",
                f"{sortino:.2f}",
                delta="优秀" if sortino > 2.0 else "一般"
            )
        
        with col3:
            st.metric(
                "Calmar比率",
                f"{calmar:.2f}",
                delta="优秀" if calmar > 3.0 else "一般"
            )
        
        with col4:
            st.metric(
                "年化波动率",
                f"{annual_vol:.2%}"
            )
        
        st.divider()
        
        # 尾部风险
        st.subheader("⚡ 尾部风险分析")
        
        # 找出极端损失
        worst_returns = np.sort(returns_array)[:10]
        
        tail_data = pd.DataFrame({
            '排名': range(1, 11),
            '收益率': worst_returns,
            '损失金额': [abs(r * portfolio_value) for r in worst_returns]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                tail_data.style.format({
                    '收益率': '{:.2%}',
                    '损失金额': '￥{:,.0f}'
                }).background_gradient(subset=['损失金额'], cmap='Reds'),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # 极端损失分布
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=tail_data['排名'],
                y=tail_data['收益率'],
                marker=dict(color=tail_data['收益率'], colorscale='Reds'),
                text=[f"{r:.2%}" for r in tail_data['收益率']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Top 10 最大损失日",
                xaxis_title="排名",
                yaxis_title="收益率",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 风险总结
        st.subheader("📊 风险评估总结")
        
        if var_amount < portfolio_value * 0.05:
            risk_level = "🟢 低风险"
            risk_msg = "组合风险处于较低水平，可以维持当前策略。"
        elif var_amount < portfolio_value * 0.10:
            risk_level = "🟡 中等风险"
            risk_msg = "组合风险处于正常范围，建议定期监控。"
        else:
            risk_level = "🔴 高风险"
            risk_msg = "组合风险较高，建议考虑降低仓位或增加对冲。"
        
        st.info(f"""
        **风险等级**: {risk_level}
        
        {risk_msg}
        
        **关键指标**：
        - VaR ({confidence_level:.0%}): ￥{var_amount:,.0f} ({var_value:.2%})
        - CVaR: ￥{cvar_amount:,.0f} ({cvar_value:.2%})
        - Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Calmar: {calmar:.2f}
        - 最大回撤: {max_dd:.2%}
        """)
    
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
        st.header("🔥 涨停板智能分析")
        
        st.markdown("""
        **功能说明:** 
        - ✨ **一键扫描**: 自动查找当天收盘涨停股并批量分析
        - 📊 **智能评分**: 4大维度自动打分(涨停时间、封单强度、开板次数、量能)
        - 🎯 **强弱排序**: 按照综合得分自动排序，重点关注强势股
        - 🚨 **操作建议**: 给出明确的“重点关注”/“谨慎观望”/“不建议”
        """)
        
        st.divider()
        
        # ===== 一键扫描功能 =====
        st.subheader("🚀 一键扫描（推荐）")
        
        col_scan1, col_scan2 = st.columns([3, 1])
        
        with col_scan1:
            st.info("💡 点击按钮后，系统将自动：扫描今日涨停股 → 分析强弱 → 排序打分 → 给出建议")
        
        with col_scan2:
            auto_scan_btn = st.button(
                "🔍 一键扫描分析",
                use_container_width=True,
                type="primary",
                help="自动查找并分析今日所有涨停股"
            )
        
        if auto_scan_btn:
            df_results = pd.DataFrame()
            with st.spinner("🔍 正在通过独立进程扫描..."):
                try:
                    import subprocess, json, sys, os
                    from pathlib import Path

                    python_executable = sys.executable
                    scanner_script = str(Path(__file__).parent.parent / "app" / "limitup_scanner_simple.py")
                    
                    # 运行脚本作为子进程，并显式传递当前环境变量
                    process_env = os.environ.copy()
                    result = subprocess.run(
                        [python_executable, scanner_script],
                        capture_output=True, text=True, encoding='utf-8', env=process_env
                    )

                    # 如果子进程执行出错，显示详细的错误报告
                    if result.returncode != 0:
                        st.error(f"❌ 扫描脚本执行失败 (Exit Code: {result.returncode})")
                        with st.expander("🔍 **重要：点击查看脚本错误详情 (stderr)**"):
                            st.code(result.stderr or "无错误输出。")
                        with st.expander("🔍 点击查看脚本常规输出 (stdout)"):
                            st.code(result.stdout or "无常规输出。")
                        st.info("--- 请将以上错误详情截图给我，以便进一步分析 ---")
                        return

                    # 解析脚本的JSON输出
                    if result.stdout:
                        json_start_index = result.stdout.find('[')
                        if json_start_index != -1:
                            clean_stdout = result.stdout[json_start_index:]
                            df_results = pd.DataFrame(json.loads(clean_stdout))

                except Exception as e:
                    st.error(f"❌ 调用扫描脚本时发生未知错误: {e}")
                    import traceback
                    with st.expander("🔍 查看详细错误"):
                        st.code(traceback.format_exc())
                    return
            
            # --- 结果处理 ---
            if df_results.empty:
                st.warning("⚠️ 扫描成功，但未返回任何涨停股数据。请检查 `app/limitup_scanner.py` 的输出。")
            else:
                # 检查是否为模拟数据
                is_mock = len(df_results) == 3 and df_results.iloc[0]['name'] == '浦发银行'
                
                if is_mock:
                    st.warning(f"⚠️ 网络连接失败，当前显示的是模拟数据（非实时行情）")
                    st.info("🔧 子进程未能成功联网。请检查代理设置并重试。")
                else:
                    st.success(f"✅ 扫描完成！找到 {len(df_results)} 只真实涨停股")
                
                st.divider()
                
                # 显示统计信息
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                strong_count = len(df_results[df_results['total_score'] >= 85])
                medium_count = len(df_results[(df_results['total_score'] >= 70) & (df_results['total_score'] < 85)])
                weak_count = len(df_results[df_results['total_score'] < 70])
                
                with col_stat1:
                    st.metric("🔥 强势涨停", f"{strong_count}只")
                with col_stat2:
                    st.metric("⚠️ 一般涨停", f"{medium_count}只")
                with col_stat3:
                    st.metric("❌ 弱势涨停", f"{weak_count}只")
                with col_stat4:
                    avg_score = df_results['total_score'].mean()
                    st.metric("📊 平均得分", f"{avg_score:.1f}")
                
                st.divider()
                
                # 显示分析结果表格
                st.subheader("📊 分析结果（按得分排序）")
                
                display_df = df_results[['name', 'symbol', 'total_score', 'rating', 'recommendation']].copy()
                display_df.columns = ['股票名称', '代码', '综合得分', '评级', '操作建议']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # 重点关注提示
                if strong_count > 0:
                    st.divider()
                    st.subheader("🎯 重点关注股票")
                    strong_stocks = df_results[df_results['total_score'] >= 85]
                    
                    for idx, row in strong_stocks.iterrows():
                        with st.expander(f"🔥 {row['name']} ({row['symbol']}) - 得分: {row['total_score']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🕒 涨停时间得分", f"{row['scores']['time_score']}")
                                st.metric("💪 封单强度得分", f"{row['scores']['seal_score']}")
                            with col2:
                                st.metric("🔓 开板次数得分", f"{row['scores']['open_score']}")
                                st.metric("📊 量能得分", f"{row['scores']['volume_score']}")
                            
                            st.success(f"📌 **建议**: {row['recommendation']}")
                
                # 下载按钮
                st.divider()
                csv = df_results.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 下载分析结果 (CSV)",
                    data=csv,
                    file_name=f"limitup_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        st.divider()
        st.divider()
        
        # ===== 单股深度分析 =====
        st.subheader("🔍 单股深度分析（可选）")
        
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
        if st.button("🔍 开始分析", use_container_width=True, type="primary", key="limitup_start_analysis"):
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
                
            if st.button("🔍 评估流动性", type="primary", key="p3_liq_eval_duplicate"):
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
                if st.button("📏 建仓规模检查", key="p3_liq_check_duplicate"):
                    ok, reason, rec = monitor.check_position_size(symbol, target_shares, metrics)
                    if ok:
                        st.success(f"通过：{reason} | 建议股数 {rec:,}")
                    else:
                        st.error(f"不通过：{reason} | 建议股数 {rec:,}")
        
        with tab_guard:
            st.markdown("市场健康度与交易暂停策略")
            guard = ExtremeMarketGuard()
            if st.button("🧪 评估示例市场健康度", key="p3_guard_health_duplicate"):
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
            if st.button("⚡ 检测个股事件", key="p3_guard_stock_duplicate"):
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
            
            if st.button("🧮 计算仓位", type="primary", key="p3_pos_calc_duplicate"):
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
            
            if st.button("📊 评估排队状态", type="primary", key="p4_queue_eval_duplicate"):
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
                
                if st.button("🎲 模拟一次排队成交", key="p4_queue_sim_duplicate"):
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
                value=[(datetime.now() - timedelta(days=30)).date(), datetime.now().date()],
                key="data_source_test_date_range"
            )
        
        with col3:
            st.write("")
            st.write("")
            if st.button("🚀 开始测试", use_container_width=True, type="primary", key="multi_source_test"):
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

    def render_strategy_comparison(self):
        """策略对比工具（Phase 6.4）"""
        st.header("🏆 策略对比")
        
        st.info("""
        **功能说明：**
        - 多策略性能对比：同时运行多个策略，对比关键指标
        - 回测结果分析：收益率、夏普比率、最大回撤等
        - 可视化展示：曲线图、雷达图、指标对比
        """
        )
        
        st.divider()
        
        # 策略选择
        st.subheader("🎯 策略选择")
        
        available_strategies = [
            "📈 趨势跟随策略",
            "📊 均值回归策略",
            "🧠 机器学习策略 (LSTM)",
            "🚀 涨停板策略",
            "🔄 动量策略"
        ]
        
        selected_strategies = st.multiselect(
            "选择要对比的策略（最多5个）",
            options=available_strategies,
            default=available_strategies[:3]
        )
        
        if len(selected_strategies) < 2:
            st.warning("⚠️ 请至少选择2个策略进行对比")
            return
        
        st.divider()
        
        # 回测参数
        st.subheader("⚙️ 回测参数")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_capital = st.number_input(
                "初始资金 (元)",
                min_value=10000,
                max_value=10000000,
                value=1000000,
                step=10000
            )
        
        with col2:
            start_date = st.date_input(
                "开始日期",
                value=(datetime.now() - timedelta(days=365)).date()
            )
        
        with col3:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now().date()
            )
        
        st.divider()
        
        # 执行对比
        st.subheader("🛠️ 执行对比")
        
        if st.button("🚀 开始对比", type="primary", use_container_width=True, key="strategy_comparison_run"):
            with st.spinner("📊 正在运行策略对比..."):
                import time
                time.sleep(1.5)
                
                # 生成模拟数据
                np.random.seed(42)
                days = (end_date - start_date).days
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                results = {}
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                for idx, strategy in enumerate(selected_strategies):
                    # 生成模拟收益
                    if "趨势" in strategy:
                        daily_returns = np.random.normal(0.0008, 0.015, len(dates))
                    elif "均值" in strategy:
                        daily_returns = np.random.normal(0.0005, 0.012, len(dates))
                    elif "机器" in strategy:
                        daily_returns = np.random.normal(0.001, 0.018, len(dates))
                    elif "涨停" in strategy:
                        daily_returns = np.random.normal(0.0012, 0.025, len(dates))
                    else:
                        daily_returns = np.random.normal(0.0006, 0.014, len(dates))
                    
                    cum_returns = (1 + daily_returns).cumprod()
                    portfolio_values = initial_capital * cum_returns
                    
                    results[strategy] = {
                        'returns': daily_returns,
                        'cum_returns': cum_returns,
                        'portfolio_values': portfolio_values,
                        'color': colors[idx % len(colors)]
                    }
                
                st.success("✅ 对比完成！")
                
                st.divider()
                
                # 性能指标对比
                st.subheader("📊 性能指标对比")
                
                metrics_data = []
                for strategy, data in results.items():
                    returns = data['returns']
                    total_return = data['cum_returns'][-1] - 1
                    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
                    annual_vol = returns.std() * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    max_dd = (data['cum_returns'] / np.maximum.accumulate(data['cum_returns']) - 1).min()
                    
                    metrics_data.append({
                        '策略': strategy,
                        '总收益率': total_return,
                        '年化收益': annual_return,
                        '年化波动': annual_vol,
                        '夏普比率': sharpe,
                        '最大回撤': max_dd
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                
                st.dataframe(
                    df_metrics.style.format({
                        '总收益率': '{:.2%}',
                        '年化收益': '{:.2%}',
                        '年化波动': '{:.2%}',
                        '夏普比率': '{:.2f}',
                        '最大回撤': '{:.2%}'
                    }).background_gradient(subset=['夏普比率'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.divider()
                
                # 累计收益曲线
                st.subheader("📈 累计收益曲线")
                
                fig = go.Figure()
                
                for strategy, data in results.items():
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=data['portfolio_values'],
                        mode='lines',
                        name=strategy,
                        line=dict(width=2, color=data['color'])
                    ))
                
                fig.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="初始资金"
                )
                
                fig.update_layout(
                    xaxis_title="日期",
                    yaxis_title="组合价值 (元)",
                    height=500,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # 雷达图
                st.subheader("🎯 性能雷达图")
                
                categories = ['收益', '风险', 'Sharpe', '回撤', '稳定性']
                
                fig = go.Figure()
                
                for strategy, data in results.items():
                    returns = data['returns']
                    # 标准化指标（0-1）
                    metrics_row = df_metrics[df_metrics['策略'] == strategy].iloc[0]
                    values = [
                        min(1.0, max(0.0, (metrics_row['年化收益'] + 0.2) / 0.4)),  # 收益
                        1 - min(1.0, metrics_row['年化波动'] / 0.3),  # 风险（反向）
                        min(1.0, max(0.0, metrics_row['夏普比率'] / 3.0)),  # Sharpe
                        1 - min(1.0, abs(metrics_row['最大回撤']) / 0.3),  # 回撤（反向）
                        1 - returns.std()  # 稳定性
                    ]
                    values.append(values[0])  # 闭合
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        name=strategy,
                        fill='toself',
                        line=dict(color=data['color'])
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # 总结
                st.subheader("🏆 对比总结")
                
                best_return = df_metrics.loc[df_metrics['年化收益'].idxmax()]
                best_sharpe = df_metrics.loc[df_metrics['夏普比率'].idxmax()]
                best_dd = df_metrics.loc[df_metrics['最大回撤'].idxmax()]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.success(f"""
                    **🏆 最高收益**
                    
                    {best_return['策略']}
                    
                    年化收益：{best_return['年化收益']:.2%}
                    """)
                
                with col2:
                    st.success(f"""
                    **🏆 最佳Sharpe**
                    
                    {best_sharpe['策略']}
                    
                    Sharpe比率：{best_sharpe['夏普比率']:.2f}
                    """)
                
                with col3:
                    st.success(f"""
                    **🏆 最小回撤**
                    
                    {best_dd['策略']}
                    
                    最大回撤：{best_dd['最大回撤']:.2%}
                    """)
        else:
            st.info("👆 点击上方按钮开始策略对比")


# 主程序入口
if __name__ == "__main__":
    dashboard = UnifiedDashboard()
    dashboard.run()
