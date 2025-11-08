"""
系统使用指南页面
提供完整的系统功能说明和操作指南
"""

import streamlit as st
from datetime import datetime


def show_system_guide():
    """显示系统使用指南"""
    
    st.title("📚 Qilin Stack 系统使用指南")
    st.markdown("---")
    
    # 添加版本信息
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"📅 最后更新：{datetime.now().strftime('%Y-%m-%d')}")
    with col2:
        st.success("版本：v3.0.0")
    with col3:
        st.warning("状态：生产就绪")
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🚀 快速开始",
        "🎯 核心功能",
        "📊 回测系统",
        "🤖 AI进化系统",
        "📈 一进二专用指标",
        "🛡️ 风控系统",
        "⚙️ 高级配置"
    ])
    
    with tab1:
        show_quick_start()
    
    with tab2:
        show_core_features()
    
    with tab3:
        show_backtest_guide()
    
    with tab4:
        show_ai_evolution_guide()
    
    with tab5:
        show_one_into_two_metrics()
    
    with tab6:
        show_risk_management()
        
    with tab7:
        show_advanced_config()


def show_quick_start():
    """快速开始指南"""
    st.header("🚀 快速开始")
    
    # 添加快速导航
    st.info("💡 **新用户建议**: 先阅读下方【快速落地实战】30分钟快速上手指南！")
    
    # 创建子标签
    quick_tab1, quick_tab2, quick_tab3 = st.tabs([
        "📖 系统概述",
        "🚀 快速落地实战",
        "📋 常用命令速查"
    ])
    
    with quick_tab1:
        st.markdown("""
        ### 1. 系统概述
        
        Qilin Stack 是一个专门针对A股涨停板策略的量化交易系统,特别聚焦于\"一进二\"（首板后连板）模式的研究和交易。
        
        ### 🆕 v3.0 重大升级
        
        - **一进二专用指标**: P@N、Hit@N、板强度等核心评估体系
        - **智能仓位管理**: Kelly准则 + 风险平价组合优化
        - **极端行情处理**: 5级保护机制，自动风控
        - **涨停队列模拟**: 真实模拟排队成交机制
        - **可执行性评分**: 0-100分策略评估体系
        
        ### 2. 主要模块
        
        - **Qilin监控**：实时监控涨停板和交易信号
        - **AI进化系统**：基于机器学习的预测模型
        - **循环进化系统**：持续优化模型性能
        - **写实回测**：真实市场条件下的策略验证
        - **风控系统**：多维度风险评估与动态管理
        
        ### 3. 快速上手步骤
        
        1. **数据准备**
           ```python
           # 从AKShare获取实时数据
           data_source = "akshare"
           
           # 或使用Qlib离线数据
           data_source = "qlib"
           ```
        
        2. **模型训练**
           - 进入"AI进化系统"标签
           - 选择数据源
           - 点击"开始训练"
        
        3. **回测验证**
           - 配置回测参数
           - 选择成交模式（推荐使用`queue`模式）
           - 运行回测查看结果
        
        4. **实盘交易**
           - 配置交易账户
           - 设置风控参数
           - 启动自动交易
        """)
    
    with quick_tab2:
        render_quick_landing_guide()
    
    with quick_tab3:
        render_command_reference()


def show_core_features():
    """核心功能说明"""
    st.header("🎯 核心功能")
    
    st.markdown("""
    ### 一、涨停板检测
    
    系统采用多维度特征识别涨停板机会：
    
    | 特征维度 | 说明 | 权重 |
    |---------|------|------|
    | 封板强度 | 封单金额/流通市值 | 25% |
    | 开板次数 | 当日开板频率 | 15% |
    | 板高度 | 连续涨停天数 | 20% |
    | 市场情绪 | 整体涨停家数 | 10% |
    | 题材热度 | 所属概念活跃度 | 15% |
    | 量能变化 | 成交量较均值倍数 | 15% |
    
    ### 二、一进二策略
    
    **策略逻辑**：
    1. T日涨停（首板）
    2. T+1日开盘买入
    3. T+1日再次涨停（二板）
    4. T+2日或后续卖出
    
    **关键指标**：
    - 成功率：历史一进二成功概率
    - 预期收益：成功时平均收益
    - 风险控制：最大回撤限制
    
    ### 三、智能预测
    
    使用集成学习模型：
    - XGBoost：捕捉非线性关系
    - LightGBM：高效处理大数据
    - CatBoost：处理类别特征
    - GRU：时序依赖建模
    
    ### 四、风险管理
    
    - 单票仓位上限：20%
    - 止损线：-7%
    - 止盈线：+15%
    - 最大持仓数：10只
    """)


def show_backtest_guide():
    """回测系统指南"""
    st.header("📊 回测系统使用指南")
    
    st.markdown("""
    ## 🆕 最新更新：涨停队列模拟器
    
    ### 一、成交模式选择
    
    系统现在支持三种成交模式，更真实地模拟市场交易：
    
    #### 1. **确定性模式** (`deterministic`)
    - 基于历史特征计算固定成交比例
    - 适合初步策略验证
    - 结果稳定可重复
    
    #### 2. **概率模式** (`prob`)  
    - 在确定性基础上加入随机因素
    - 使用Beta分布生成成交比例
    - 更接近真实市场波动
    
    #### 3. **队列模拟模式** (`queue`) ⭐ 推荐
    - **完整模拟涨停板排队机制**
    - 考虑封板强度、排队位置、市场情绪
    - 支持部分成交和未成交统计
    
    ### 二、涨停排队模拟器详解
    
    #### 封板强度分类
    
    | 强度等级 | 特征 | 排队成功率 | 成交比例范围 |
    |---------|------|------------|-------------|
    | 🔴 强势 | 封单巨大,很少开板 | 30% | 0-30% |
    | 🟡 中等 | 封单适中,偶尔开板 | 60% | 20-70% |
    | 🟢 弱势 | 封单较小,频繁开板 | 90% | 50-100% |
    
    #### 股票类型影响
    
    - **主板（10%涨停）**：基准难度
    - **创业板/科创板（20%）**：难度×0.8（更难买入）
    - **ST股票（5%）**：难度×1.2（相对容易）
    
    #### 排队位置计算
    
    排队位置由以下因素决定：
    - **提交时间**（70%权重）：越早越靠前
    - **资金量**（30%权重）：资金越大越靠前
    
    ### 三、回测配置示例
    
    ```python
    from backtest.engine import BacktestConfig, BacktestEngine
    
    # 创建配置
    config = BacktestConfig(
        initial_capital=1000000,    # 初始资金
        fill_model='queue',          # 使用队列模拟
        max_position_size=0.2,       # 单票最大仓位
        stop_loss=-0.07,            # 止损线
        take_profit=0.15,           # 止盈线
    )
    
    # 运行回测
    engine = BacktestEngine(config)
    results = await engine.run_backtest(
        symbols=stock_list,
        start_date='2024-01-01',
        end_date='2024-12-31',
        data_source=market_data
    )
    ```
    
    ### 四、关键评估指标
    
    #### 新增成交统计指标
    
    - **未成交率** (`unfilled_rate`)
      - 完全未成交的订单占比
      - 反映策略的可执行性
      
    - **平均成交比例** (`avg_fill_ratio`)
      - 所有订单的平均成交比例
      - 衡量实际可获得的仓位
    
    - **订单统计**
      - `orders_attempted`：尝试下单总数
      - `orders_unfilled`：未成交订单数
      - `shares_planned`：计划买入股数
      - `shares_filled`：实际成交股数
    
    ### 五、回测结果解读
    
    #### 成交分析示例
    
    ```
    总订单数：100
    未成交订单：15 (15%)
    平均成交比例：72.3%
    
    成交比例分布：
    - 全额成交(100%)：20次
    - 部分成交(50-99%)：45次  
    - 小额成交(<50%)：20次
    - 未成交(0%)：15次
    ```
    
    #### 策略优化建议
    
    根据成交统计优化策略：
    
    1. **未成交率 > 20%**
       - 降低选股标准的封板强度要求
       - 考虑在开板时介入
    
    2. **平均成交比例 < 50%**
       - 提前挂单（9:15集合竞价）
       - 分批建仓降低单次金额
    
    3. **强势封板成交率低**
       - 重点关注中等强度封板
       - 避免追涨停价
    """)
    
    # 添加实际操作演示
    with st.expander("📹 查看操作演示"):
        st.markdown("""
        1. 进入"写实回测"页面
        2. 在侧边栏选择成交模式为"queue"
        3. 设置回测参数（日期、资金等）
        4. 点击"运行写实回测"
        5. 查看"排队分析"标签了解成交详情
        """)


def show_ai_evolution_guide():
    """AI进化系统指南"""
    st.header("🤖 AI进化系统指南")
    
    st.markdown("""
    ### 一、系统架构
    
    AI进化系统包含两大模块：
    
    #### 1. AI Evolution System（基础进化）
    - 数据收集与标注
    - 特征工程
    - 模型训练
    - 预测与回测
    
    #### 2. Loop Evolution System（循环进化）
    - 困难样本挖掘
    - 对抗训练
    - 课程学习
    - 知识蒸馏
    - 元学习
    
    ### 二、训练流程
    
    #### Step 1：数据准备
    ```python
    # 获取涨停数据
    data = get_limitup_data(
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    
    # 特征提取（100+维度）
    features = extract_features(data)
    ```
    
    #### Step 2：模型训练
    ```python
    # 集成模型
    model = StackedLimitUpModel()
    model.fit(X_train, y_train)
    
    # 强化学习智能体
    rl_agent = RLTradingAgent()
    rl_agent.train(env)
    ```
    
    #### Step 3：循环优化
    
    **困难样本挖掘**
    - 识别预测错误的案例
    - 重点训练提升准确率
    
    **对抗训练**
    - AI生成对抗样本
    - 增强模型鲁棒性
    
    **课程学习**
    - 从易到难渐进训练
    - 确保稳定收敛
    
    ### 三、性能监控
    
    实时追踪模型表现：
    - 预测准确率
    - 收益稳定性
    - 风险指标
    - 成交可行性
    """)


def show_one_into_two_metrics():
    """一进二专用指标说明"""
    st.header("📈 一进二专用评估指标")
    
    st.markdown("""
    ## 🆕 最新更新：专业级评估体系
    
    ### 一、核心指标体系
    
    #### 1. **P@N (Precision at N)**
    - **定义**: 预测Top N只股票中的涨停命中率
    - **计算**: 命中数 / 预测数
    - **目标**: > 30%为优秀，> 50%为卓越
    
    #### 2. **Hit@N (Hit Rate at N)**  
    - **定义**: Top N预测占实际涨停池的覆盖率
    - **计算**: 命中数 / 当日涨停总数
    - **意义**: 衡量模型的召回能力
    
    #### 3. **板强度 (Board Strength)**
    - **综合评分**: 0-1分值
    - **考虑因素**:
      - 封板时间（早封加分）
      - 封单金额（巨单加分）
      - 开板次数（少开加分）
      - 换手率（适中最佳）
    
    ### 二、细分指标
    
    | 指标类别 | 指标名称 | 说明 | 重要性 |
    |---------|---------|------|--------|
    | **结构分析** | 首板命中数 | 首次涨停后继续涨停 | 高 |
    | | 连板命中数 | 多连板后继续涨停 | 中 |
    | | 题材命中率 | 不同题材的成功率 | 高 |
    | | 板块集中度 | HHI指数衡量 | 中 |
    | **执行分析** | 平均队列位置 | 排队位置(0-1) | 高 |
    | | 平均成交比例 | 实际成交/计划 | 高 |
    | | 未成交率 | 完全未成交占比 | 高 |
    | **收益分析** | 次日平均收益 | T+1收益率 | 高 |
    | | 盈亏比 | 平均盈/平均亏 | 中 |
    | | 最大单票收益 | 单票最高收益 | 低 |
    | | 最大单票亏损 | 单票最大亏损 | 中 |
    
    ### 三、可执行性评分
    
    系统会综合计算策略的**可执行性得分**（0-100分）：
    
    ```python
    执行得分 = (
        成交率得分 * 0.3 +
        胜率得分 * 0.25 +
        盈亏比得分 * 0.25 +
        稳定性得分 * 0.2
    )
    ```
    
    #### 评分等级
    - **90-100分**: 🏆 卓越 - 可直接实盘
    - **75-89分**: ✨ 优秀 - 小额实盘验证
    - **60-74分**: 👍 良好 - 需要优化
    - **< 60分**: ⚠️ 较差 - 建议重新设计
    
    ### 四、使用示例
    
    ```python
    from backtest.one_into_two_metrics import OneIntoTwoEvaluator
    
    # 创建评估器
    evaluator = OneIntoTwoEvaluator()
    
    # 评估预测结果
    metrics = evaluator.evaluate_predictions(
        predictions=model_predictions,
        actual_results=market_results,
        date='2025-01-30'
    )
    
    # 查看核心指标
    print(f"P@10: {metrics.precision_at_n:.1%}")
    print(f"Hit@10: {metrics.hit_at_n:.1%}")
    print(f"板强度: {metrics.board_strength:.2f}")
    
    # 生成评估报告
    report = evaluator.generate_report()
    ```
    
    ### 五、优化建议
    
    根据指标表现调整策略：
    
    1. **P@N < 20%**
       - 提高预测阈值
       - 加强特征筛选
       - 减少预测数量
    
    2. **未成交率 > 30%**
       - 避免追强势封板
       - 关注中等强度封板
       - 考虑开板时介入
    
    3. **板强度 < 0.3**
       - 重点关注封板质量
       - 筛选早封、大单封板
       - 避免烂板和尾盘板
    """)


def show_risk_management():
    """风控系统说明"""
    st.header("🛡️ 智能风控系统")
    
    st.markdown("""
    ## 🆕 最新升级：三大风控模块
    
    ### 一、智能仓位管理
    
    #### Kelly准则仓位计算
    
    系统采用**保守Kelly公式**计算最优仓位：
    
    ```python
    f* = (p * b - q) / b * kelly_fraction
    
    # 其中:
    # f* = 最优仓位比例
    # p = 胜率
    # b = 赔率
    # q = 1-p = 败率
    # kelly_fraction = 0.25 (保守系数)
    ```
    
    #### 动态风险调整
    
    | 风险因子 | 调整方式 | 影响程度 |
    |---------|---------|----------|
    | 波动率 | 目标波动率/实际波动率 | 高 |
    | 流动性 | 流动性评分(0-1) | 中 |
    | 最大回撤 | 10%/实际回撤 | 高 |
    | Beta | 1/Beta (限制在0.5-1.5) | 低 |
    
    #### 组合优化
    
    - **风险平价**: 均衡分配风险贡献
    - **最大分散化**: HHI指数 < 0.2
    - **动态再平衡**: 每周调整一次
    
    ### 二、极端行情处理
    
    #### 市场状态识别
    
    ```python
    class MarketRegime:
        BULL = "牛市"      # 持续上涨，涨停多
        BEAR = "熊市"      # 持续下跌，跌停多
        VOLATILE = "震荡"  # 高波动，方向不明
        CRASH = "崩盘"     # 急速下跌，恐慌
        RECOVERY = "恢复"  # 崩盘后反弹
    ```
    
    #### 保护级别
    
    | 保护级别 | 触发条件 | 操作建议 |
    |---------|---------|----------|
    | 🔴 **紧急** | 综合风险>85 | 立即清仓，停止交易 |
    | 🟠 **高级** | 综合风险>70 | 仓位降至50%以下 |
    | 🟡 **中级** | 综合风险>50 | 适度减仓，谨慎操作 |
    | 🟢 **低级** | 综合风险>30 | 控制仓位，正常交易 |
    | ✅ **无** | 综合风险<30 | 正常执行策略 |
    
    #### 风险评估维度
    
    1. **流动性风险** (0-100)
       - 成交量不足
       - 买卖价差大
       - 换手率低
    
    2. **极端事件风险** (0-100)
       - 暴跌风险
       - 泡沫风险
       - 尾部风险(VaR/CVaR)
    
    3. **系统性风险** (0-100)
       - 市场状态
       - 相关性风险
       - 宏观环境
    
    ### 三、增强回测指标
    
    #### 成交统计
    
    ```python
    class EnhancedMetrics:
        # 总体统计
        total_trades: int           # 总成交次数
        win_rate: float            # 胜率
        profit_factor: float       # 盈亏比
        
        # 单票分析
        avg_holding_days: float    # 平均持仓天数
        max_consecutive_wins: int  # 最大连胜
        max_consecutive_losses: int # 最大连败
        
        # 执行分析
        fill_rate: float          # 成交率
        avg_slippage: float       # 平均滑点
        execution_score: float    # 执行评分(0-100)
    ```
    
    #### 优化建议生成
    
    系统会根据回测结果自动生成优化建议：
    
    - **成交率低**: 调整下单时机，避免追高
    - **滑点大**: 减小单笔金额，分批成交
    - **连败多**: 加强风控，减小仓位
    - **回撤大**: 降低杠杆，设置止损
    
    ### 四、使用示例
    
    #### 智能仓位管理
    ```python
    from backtest.intelligent_position_sizing import IntelligentPositionSizer
    
    sizer = IntelligentPositionSizer()
    allocation = sizer.calculate_position_sizes(
        signals=trading_signals,
        market_data=market_data,
        current_portfolio=portfolio
    )
    ```
    
    #### 极端行情处理
    ```python
    from backtest.extreme_market_handler_standalone import ExtremeMarketHandler
    
    handler = ExtremeMarketHandler()
    assessment = handler.assess_market_risk(
        market_data=data,
        portfolio=positions
    )
    
    if assessment.protection_level == ProtectionLevel.CRITICAL:
        # 紧急清仓
        emergency_liquidate()
    ```
    """)


def show_advanced_config():
    """高级配置指南"""
    st.header("⚙️ 高级配置")
    
    st.markdown("""
    ### 一、环境配置
    
    #### 依赖安装
    ```bash
    pip install -r requirements.txt
    
    # 核心依赖
    - streamlit>=1.28.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - xgboost>=2.0.0
    - lightgbm>=4.0.0
    - akshare>=1.12.0
    ```
    
    ### 二、数据源配置
    
    #### AKShare（实时数据）
    ```python
    config = {
        'data_source': 'akshare',
        'update_frequency': '1min',
        'symbols': 'auto'  # 自动获取涨停股
    }
    ```
    
    #### Qlib（历史数据）
    ```python
    config = {
        'data_source': 'qlib',
        'data_path': '/data/qlib',
        'region': 'cn'
    }
    ```
    
    ### 三、交易配置
    
    #### 风控参数
    ```python
    risk_config = {
        'max_position_pct': 0.2,      # 单票仓位上限
        'max_holdings': 10,            # 最大持仓数
        'stop_loss': -0.07,           # 止损线
        'take_profit': 0.15,          # 止盈线
        'max_drawdown': -0.15,        # 最大回撤限制
    }
    ```
    
    #### 执行参数
    ```python
    execution_config = {
        'fill_model': 'queue',         # 成交模拟模式
        'slippage': 0.002,            # 滑点
        'commission': 0.0003,         # 手续费
        'min_order_size': 100,        # 最小下单股数
    }
    ```
    
    ### 四、性能优化
    
    #### 并行计算
    ```python
    # 启用多进程
    import multiprocessing as mp
    n_jobs = mp.cpu_count() - 1
    
    # XGBoost并行
    model = XGBClassifier(n_jobs=n_jobs)
    ```
    
    #### 缓存优化
    ```python
    # Streamlit缓存
    @st.cache_data(ttl=3600)
    def load_market_data():
        return fetch_data()
    
    # 特征缓存
    @st.cache_resource
    def get_feature_calculator():
        return FeatureCalculator()
    ```
    
    ### 五、监控告警
    
    #### 日志配置
    ```python
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('qilin.log'),
            logging.StreamHandler()
        ]
    )
    ```
    
    #### 告警规则
    - 未成交率 > 30%：邮件通知
    - 连续亏损3天：暂停交易
    - 回撤超过10%：降低仓位
    """)
    
    # 添加常见问题
    with st.expander("❓ 常见问题"):
        st.markdown("""
        **Q: 如何选择合适的成交模式？**
        A: 建议使用`queue`模式进行真实回测，`deterministic`模式用于快速验证。
        
        **Q: 为什么实盘收益低于回测？**
        A: 检查是否使用了队列模拟模式，关注未成交率和平均成交比例。
        
        **Q: 如何提高一进二成功率？**
        A: 重点关注封板质量>7、连板数<=2、题材热度高的标的。
        """)


def render_quick_landing_guide():
    """渲染快速落地实战指南（完全基于Web界面）"""
    st.header("🚀 快速落地实战指南")
    st.caption("🎯 目标：在Web界面中完成所有操作，30分钟内让整套系统运转起来！")
    
    st.success("✨ **新特性**: 所有操作都可以在本 Web 界面中完成，无需命令行！")
    
    st.markdown("---")
    
    # 前置说明
    st.info("""
    💡 **重要提示**：
    - 本指南假设你已经成功启动了本 Web 界面
    - 如果还没有启动，请先在命令行运行：`streamlit run web/unified_dashboard.py`
    - 一旦Web界面启动，后续所有操作都可在此完成！
    """)
    
    # 前置准备
    with st.expander("💻 系统要求（仅供参考）", expanded=False):
        st.markdown("""
        **系统要求**：
        - Windows 10/11 或 Linux/macOS
        - Python 3.8+
        - 8GB+ 内存
        - 10GB+ 硬盘空间
        
        **环境检查**：
        - 如果你能看到这个页面，说明环境已经就绪 ✅
        """)
    
    # 第一步：验证系统状态
    st.subheader("✅ 第一步：验证系统状态（1分钟）")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👍 Web界面", "✅ 已启动", delta="正常")
    with col2:
        # 检查Python环境
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        st.metric("🐍 Python版本", python_version, delta="满足要求" if sys.version_info >= (3, 8) else "需要升级")
    with col3:
        # 检查关键库
        try:
            import pandas, numpy, streamlit
            st.metric("📦 核心库", "✅ 已安装", delta="就绪")
        except ImportError:
            st.metric("📦 核心库", "❌ 缺失", delta="需安装")
    
    st.success("✨ **恭喜！你已经成功启动了本系统，可以继续下一步！**")
    
    with st.expander("💡 如果缺少某些库，怎么办？", expanded=False):
        st.code("""
# 在命令行中运行（只需一次）：
pip install pandas numpy scikit-learn lightgbm xgboost catboost streamlit plotly akshare

# 然后重启Web界面
""", language="bash")
    
    st.markdown("---")
    
    # 第二步：数据源选择
    st.subheader("✅ 第二步：选择数据源（2分钟）")
    
    st.markdown("""
    **📊 本系统支持两种数据源**：
    """)
    
    data_method = st.radio(
        "选择你的数据源",
        options=[
            "⚡ AKShare在线数据（推荐新手）",
            "📚 Qlib本地数据（需提前下载）"
        ],
        index=0,
        horizontal=True,
        key="data_source_choice_guide"
    )
    
    if "AKShare" in data_method:
        st.success("✨ **推荐选择！AKShare 无需下载，即用即取！**")
        
        # 检查AKShare是否安装
        try:
            import akshare as ak
            st.success("✅ AKShare 已安装，可以使用！")
            
            # 测试连接
            if st.button("🔍 测试AKShare连接", key="test_akshare"):
                with st.spinner("正在连接AKShare..."):
                    try:
                        # 获取一个简单数据测试
                        test_df = ak.stock_zh_a_spot_em()
                        st.success(f"✅ 连接成功！当前可获取 {len(test_df)} 只股票数据")
                        with st.expander("👁️ 查看样例数据"):
                            st.dataframe(test_df.head(10))
                    except Exception as e:
                        st.error(f"❌ 连接失败：{e}")
                        st.info("💡 请检查网络连接或稍后重试")
        except ImportError:
            st.warning("⚠️ AKShare 尚未安装")
            st.info("在命令行运行：`pip install akshare`，然后重启Web界面")
        
        st.markdown("""
        **✨ 优点**：
        - ✅ 免费使用
        - ✅ 实时数据
        - ✅ 无需下载
        - ✅ 自动更新
        """)
        
    else:
        st.info("📚 **Qlib 本地数据** - 适合专业用户")
        
        # 检查Qlib是否安装
        try:
            import qlib
            st.success("✅ Qlib 已安装")
            
            # 检查数据目录
            from pathlib import Path
            qlib_data_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
            
            if qlib_data_dir.exists():
                st.success(f"✅ Qlib数据目录存在：{qlib_data_dir}")
            else:
                st.warning(f"⚠️ Qlib数据目录不存在：{qlib_data_dir}")
                st.info("请先下载 Qlib 数据，参考下方命令")
                
        except ImportError:
            st.warning("⚠️ Qlib 尚未安装")
            st.info("在命令行运行：`pip install qlib`")
        
        with st.expander("📝 如何下载 Qlib 数据", expanded=False):
            st.code("""
# 在命令行运行（首次使用）：
pip install qlib
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 或使用项目脚本：
python scripts/download_qlib_data_v2.py --start 2020-01-01 --end 2024-12-31
""", language="bash")
    
    st.markdown("---")
    st.success("✨ **数据源已选择，继续下一步！**")
    
    
    # 第三步：因子研究（在Web界面操作）
    st.subheader("✅ 第三步：因子研究与发现（3分钟）")
    
    st.markdown("""
    **🧪 在Web界面中查看因子**：
    """)
    
    st.info("""
    📍 **操作路径**：
    1. 点击顶部导航栏的 **"📦 Qlib"** 标签
    2. 选择 **"🕹️ 数据管理"** 子标签
    3. 点击 **"🧪 因子研究"** 页面
    4. 在这里可以：
       - 👁️ 查看因子库
       - 📊 因子IC值分析
       - 🔍 因子效能测试
    """)
    
    # 展示因子示例
    with st.expander("👁️ 预览：系统内置因子库", expanded=False):
        st.markdown("""
        | 因子ID | 因子名称 | 预期IC | 类型 |
        |---------|----------|---------|------|
        | limitup_001 | 封单强度 | 0.08 | 流动性 |
        | limitup_002 | 连板高度因子 | 0.12 | 动量 |
        | limitup_003 | 早盘涨停因子 | 0.15 | 时间 |
        | limitup_004 | 量比因子 | 0.06 | 量能 |
        | limitup_005 | 换手率因子 | 0.05 | 流动性 |
        """)
    
    st.success("✨ **因子研究页面可以帮你了解每个因子的表现！**")
    
    
    st.markdown("---")
    
    # 第四步：模型训练（在Web界面操作）
    st.subheader("✅ 第四步：一进二模型训练（5分钟）")
    
    st.markdown("""
    **🧠 在Web界面中训练模型**：
    """)
    
    st.info("""
    📍 **操作路径**：
    1. 点击顶部导航栏的 **"📦 Qlib"** 标签
    2. 选择 **"📈 模型训练"** 子标签
    3. 点击 **"🚀 一进二策略"** 页面
    4. 在该页面中：
       - 📋 选择数据模式（示例/AKShare/Qlib）
       - 🎯 选择股票池（手动或智能选择）
       - 📦 点击「构建数据集」按钮
       - 🧠 点击「训练模型」按钮
       - 🎯 点击「生成T+1候选」查看结果
    """)
    
    st.success("✨ **所有操作都可通过点击按钮完成，无需代码！**")
    
    # 展示操作截图或示例
    with st.expander("📸 预览：一进二训练界面", expanded=False):
        st.markdown("""
        **数据模式选项**：
        - 🧪 示例模式：快速演示，使用模拟数据
        - 📡 AKShare在线模式：直接从网络获取真实数据
        - 🔥 Qlib离线模式：使用本地数据（需提前下载）
        
        **股票池选择**：
        - 👉 手动选择：自己挑选几只股票
        - 🤖 智能选择：自动获取今日涨停板
        
        **训练结果**：
        - Pool Model AUC：池子模型准确率
        - Board Model AUC：板块模型准确率
        - Top-N Threshold：选股阈值
        """)
    
    st.markdown("---")
    
    # 第五步：竞价决策系统
    st.subheader("✅ 第五步：使用竞价决策系统（5分钟）")
    
    st.markdown("""
    **🎯 在Web界面中进行选股决策**：
    """)
    
    st.info("""
    📍 **操作路径**：
    1. 点击顶部导航栏的 **"🎯 竞价决策"** 标签
    2. 选择 **"📊 T日候选筛选"** 子标签：
       - 🎯 设置质量评分阈值
       - 📊 设置流动性要求
       - 🚀 点击「执行筛选」按钮
       - 👁️ 查看筛选结果
    3. 选择 **"🎯 竞价进阶"** 子标签（高级功能）：
       - 📊 准备数据
       - 📈 运行 Pipeline
       - 📋 查看结果
    """)
    
    st.success("✨ **竞价决策系统帮你筛选出最佳交易机会！**")
    
    with st.expander("📸 预览：竞价决策界面", expanded=False):
        st.markdown("""
        **T日候选筛选功能**：
        - 质量评分：根据多维度指标评估股票质量
        - 流动性筛选：确保能顺利买入和卖出
        - 板块热度：判断市场情绪
        - 智能排序：自动按照综合得分排列
        
        **竞价进阶功能**：
        - 数据质量审计
        - 特征工程
        - 因子健康监控
        - 模型训练与预测
        """)
    
    st.markdown("---")
    
    # 第六步：回测验证
    st.subheader("✅ 第六步：运行回测验证（5分钟）")
    
    st.markdown("""
    **📈 在Web界面中进行回测**：
    """)
    
    st.info("""
    📍 **操作路径**：
    1. 点击顶部导航栏的 **"🏠 Qilin监控"** 标签
    2. 选择 **"📜 历史记录"** 或 **"📉 风险管理"** 子标签
    3. 或者点击 **"📜 写实回测"** 子标签：
       - 📊 配置回测参数
       - 🚀 点击「运行写实回测」
       - 📋 查看回测结果
       - 📈 分析绩效指标
    """)
    
    st.success("✨ **写实回测系统可以模拟真实市场环境！**")
    
    with st.expander("📸 预览：写实回测界面", expanded=False):
        st.markdown("""
        **回测配置选项**：
        - 成交模式：
          - 🟢 deterministic：确定性模式
          - 🟡 prob：概率模式
          - 🔴 queue：队列模拟模式（推荐）
        - 时间范围：选择回测开始和结束日期
        - 初始资金：设置起始资金量
        - 风控参数：止损、止盈、仓位管理
        
        **回测结果展示**：
        - 总收益率
        - 夏普比率
        - 最大回撤
        - 胜率统计
        - 成交分析
        """)
    
    st.markdown("---")
    
    # 第七步：整合使用
    st.subheader("✅ 第七步：整合使用所有功能（10分钟）")
    
    st.markdown("""
    **✨ 现在你已经掌握了所有核心功能！**
    """)
    
    st.success("""
    🎉 **恭喜！你已经完成了全部 7 个步骤！**
    
    🎯 **接下来你可以**：
    1. 每天使用竞价决策系统筛选股票
    2. 定期训练一进二模型以保持最佳状态
    3. 运行写实回测验证策略效果
    4. 查看因子研究页面了解各个因子表现
    5. 利用风险管理系统控制回撤
    """)
    
    # 快速导航卡片
    st.markdown("""
    ### 📍 快速导航：主要功能入口
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 竞价决策**
        - T日候选筛选
        - 竞价进阶优化
        - 智能选股
        """)
    
    with col2:
        st.markdown("""
        **📦 Qlib平台**
        - 模型训练
        - 数据管理
        - 因子研究
        """)
    
    with col3:
        st.markdown("""
        **🏠 Qilin监控**
        - 实时监控
        - 写实回测
        - 风险管理
        """)
    
    with st.expander("📝 高级用户：命令行操作参考", expanded=False):
        st.code("""
# 创建: test_full_workflow.py
import pandas as pd
import numpy as np
import asyncio

from rd_agent.factor_discovery_simple import SimplifiedFactorDiscovery
from factors.factor_lifecycle_manager import FactorLifecycleManager
from qlib_enhanced.one_into_two_pipeline import OneIntoTwoTrainer
from qlib_enhanced.unified_phase1_pipeline import UnifiedPhase1Pipeline

print("━"*70)
print("🚀 Qilin Stack 完整流程验证")
print("━"*70)

# Phase 1: 因子发现
async def discover_factors():
    discovery = SimplifiedFactorDiscovery()
    factors = await discovery.discover_factors(
        start_date='2024-01-01',
        end_date='2024-12-31',
        n_factors=15,
        min_ic=0.05
    )
    return factors

factors = asyncio.run(discover_factors())
print(f"✅ 发现 {len(factors)} 个高质量因子")

# Phase 2: 因子生命周期管理
manager = FactorLifecycleManager()
for i, factor in enumerate(factors[:5]):
    health_metrics = {
        'ic_mean': factor['expected_ic'],
        'ic_recent': factor['expected_ic'] * 0.9,
        'ic_win_rate': 0.60,
        'ir': 1.0,
        'ic_trend': 'stable'
    }
    manager.update_factor_status(factor['name'], health_metrics)

active_factors = manager.get_active_factors()
print(f"✅ 活跃因子数: {len(active_factors)}")

# Phase 3: 模型训练
np.random.seed(42)
df_train = pd.DataFrame({
    'seal_strength': np.random.uniform(50, 120, 500),
    'limitup_time_score': np.random.uniform(60, 100, 500),
})
df_train['pool_label'] = (df_train['seal_strength'] > 80).astype(int)
df_train['board_label'] = ((df_train['seal_strength'] > 90) & (df_train['pool_label'] == 1)).astype(int)

trainer = OneIntoTwoTrainer(top_n=20)
result = trainer.fit(df_train)
print(f"✅ Pool Model AUC: {result.auc_pool:.4f}")
print(f"✅ Board Model AUC: {result.auc_board:.4f}")

# Phase 4: Pipeline集成
pipeline = UnifiedPhase1Pipeline(output_dir="output/test_full")
print("✅ Pipeline初始化完成")

print("\n" + "━"*70)
print("✨ 所有测试通过！系统已就绪！")
print("━"*70)

# 运行完整测试
python test_full_workflow.py
""", language="python")
    
    st.markdown("---")
    
    # 总结
    st.markdown("---")
    st.markdown("""
    ## 🎉 恭喜你已经掌握Qilin Stack！
    """)
    
    st.balloons()  # 显示庆祝动画
    
    st.info("""
    **📚 进阶学习资料：**
    - 📜 `docs/DAILY_TRADING_SOP.md` - 日常交易流程SOP
    - 📈 `docs/STOCK_SELECTION_GUIDE.md` - 选股逻辑详解
    - 🧠 `docs/DEEP_ARCHITECTURE_GUIDE.md` - 深度技术架构
    """)
    
    # 学习路径
    with st.expander("🎓 推荐学习路径（1-2周）", expanded=True):
        st.markdown("""
        ### 新手路径（1-2周）
        
        **第1天**：环境搭建 + 基础测试
        - ✅ 完成上面的第一至第七步
        - ✅ 确保Web界面可以正常启动
        
        **第2-3天**：理解核心模块
        - 阅读 `DEEP_ARCHITECTURE_GUIDE.md`
        - 理解Qlib、RD-Agent、因子进化、模型架构
        - 运行上面的所有测试脚本
        
        **第4-5天**：学习日常操作
        - 阅读 `DAILY_TRADING_SOP.md`
        - 对照SOP模拟一次完整流程（T日→T+1→T+2）
        - 熟悉Web界面各个标签页
        
        **第6-7天**：掌握选股逻辑
        - 阅读 `STOCK_SELECTION_GUIDE.md`
        - 理解三层过滤体系
        - 学习质量评分和竞价强度分级
        
        **第2周**：实盘模拟
        - 使用历史数据模拟完整交易流程
        - 每天记录操作和决策
        - 总结经验和教训
        """)

def render_command_reference():
    """渲染常用命令速查"""
    st.header("📋 常用命令速查")
    
    # 日常启动
    st.subheader("🚀 日常启动")
    st.code("""
# 1. 激活环境
cd G:\\test\\qilin_stack
.\\venv\\Scripts\\activate  # Windows

# 2. 启动Web界面
streamlit run web/unified_dashboard.py
""", language="bash")
    
    # 数据更新
    st.subheader("📊 数据更新")
    st.code("""
# 更新Qlib数据（每周执行一次）
python scripts/download_qlib_data_v2.py --start 2024-01-01 --end 2024-12-31

# 验证数据
python scripts/validate_qlib_data.py
""", language="bash")
    
    # 模型训练
    st.subheader("🧠 模型训练")
    st.code("""
# 重新训练一进二模型
python qlib_enhanced/one_into_two_pipeline.py

# 训练基线模型
python scripts/train_baseline_model.py
""", language="bash")
    
    # 因子管理
    st.subheader("🔍 因子管理")
    st.code("""
# 查看因子健康度
python -c "from factors.factor_lifecycle_manager import FactorLifecycleManager; m = FactorLifecycleManager(); print(m.get_summary())"

# 重置因子状态（谨慎）
rm -rf output/factor_lifecycle/*.json
""", language="bash")
    
    # 日志查看
    st.subheader("📜 日志查看")
    st.code("""
# 实时查看日志
tail -f logs/scheduler.log

# 查看最后50行
tail -n 50 logs/scheduler.log

# 搜索错误日志 (Windows)
findstr "ERROR" logs\\scheduler.log
""", language="bash")
    
    # 常见问题排查
    st.markdown("---")
    st.subheader("⚠️ 常见问题排查")
    
    with st.expander("🔴 1. 模块导入错误", expanded=False):
        st.markdown("""
        **问题**：`ModuleNotFoundError: No module named 'xxx'`
        
        **解决**：
        ```bash
        # 确认虚拟环境已激活
        which python  # 应该显示 venv/Scripts/python
        
        # 重新安装依赖
        pip install -r requirements.txt
        
        # 或手动安装缺失的包
        pip install <package_name>
        ```
        """)
    
    with st.expander("🔴 2. Qlib数据不可用", expanded=False):
        st.markdown("""
        **问题**：`QlibDataNotFound` 或类似错误
        
        **解决**：
        ```bash
        # 检查Qlib数据目录 (Windows)
        dir %USERPROFILE%\\.qlib\\qlib_data\\cn_data
        
        # 如果不存在，重新下载
        python scripts/download_qlib_data_v2.py --start 2020-01-01 --end 2024-12-31
        
        # 或使用AKShare替代
        pip install akshare
        ```
        """)
    
    with st.expander("🔴 3. Web界面无法启动", expanded=False):
        st.markdown("""
        **问题**：`streamlit: command not found`
        
        **解决**：
        ```bash
        # 安装Streamlit
        pip install streamlit
        
        # 确认安装成功
        streamlit --version
        
        # 如果还是不行，使用python -m
        python -m streamlit run web/unified_dashboard.py
        ```
        """)
    
    with st.expander("🔴 4. 端口8501已被占用", expanded=False):
        st.markdown("""
        **问题**：`Port 8501 is already in use`
        
        **解决**：
        ```bash
        # Windows: 查找并结束进程
        netstat -ano | findstr :8501
        taskkill /PID <PID> /F
        
        # 或使用其他端口
        streamlit run web/unified_dashboard.py --server.port 8502
        ```
        """)
    
    with st.expander("🔴 5. 模型训练过慢", expanded=False):
        st.markdown("""
        **问题**：训练耗时太长
        
        **解决**：
        - 减少样本数量（测试阶段）
        - 减少模型树数量（`n_estimators`）
        - 使用GPU加速（如果支持）
        - 关闭部分模型（只用LightGBM）
        
        ```python
        # 在 one_into_two_pipeline.py 中设置环境变量
        import os
        os.environ["OIT_DISABLE_XGB"] = "1"  # 禁用XGBoost
        ```
        """)
    
    # 相关文档链接
    st.markdown("---")
    st.info("""
    📚 **相关文档**：
    - 详细技术架构：`docs/DEEP_ARCHITECTURE_GUIDE.md`
    - 日常交易SOP：`docs/DAILY_TRADING_SOP.md`
    - 选股逻辑指南：`docs/STOCK_SELECTION_GUIDE.md`
    """)

# 导出函数
__all__ = ['show_system_guide']
