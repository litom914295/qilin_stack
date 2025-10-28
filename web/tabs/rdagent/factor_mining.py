"""
RD-Agent 因子挖掘模块
- LLM驱动的因子生成
- 研报因子提取
- 因子进化循环
- 因子性能评估
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any
import sys
from pathlib import Path

# 添加RD-Agent路径（优先环境变量 RDAGENT_PATH）
import os
rdagent_env = os.getenv("RDAGENT_PATH")
rdagent_path = Path(rdagent_env) if rdagent_env else None
if rdagent_path and rdagent_path.exists() and str(rdagent_path) not in sys.path:
    sys.path.insert(0, str(rdagent_path))


class FactorMiningTab:
    """因子挖掘Tab"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """初始化session state"""
        if 'factor_generation_running' not in st.session_state:
            st.session_state.factor_generation_running = False
        if 'generated_factors' not in st.session_state:
            st.session_state.generated_factors = []
        if 'factor_evolution_history' not in st.session_state:
            st.session_state.factor_evolution_history = []
    
    def render(self):
        """渲染因子挖掘页面"""
        st.header("🔍 因子挖掘")
        
        # 顶部指标卡片
        self.render_metrics()
        
        st.divider()
        
        # 主要内容区域
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 LLM因子生成",
            "📄 研报因子提取", 
            "🔄 因子进化循环",
            "📊 因子性能评估"
        ])
        
        with tab1:
            self.render_llm_factor_generation()
        
        with tab2:
            self.render_report_factor_extraction()
        
        with tab3:
            self.render_factor_evolution()
        
        with tab4:
            self.render_factor_evaluation()
    
    def render_metrics(self):
        """渲染顶部指标"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "生成因子数",
                len(st.session_state.generated_factors),
                "+3"
            )
        
        with col2:
            st.metric(
                "有效因子",
                f"{len([f for f in st.session_state.generated_factors if f.get('valid', False)])}",
                "+2"
            )
        
        with col3:
            st.metric(
                "平均IC",
                "0.083",
                "+0.012"
            )
        
        with col4:
            st.metric(
                "最佳因子IC",
                "0.156",
                "+0.023"
            )
        
        with col5:
            st.metric(
                "进化轮次",
                len(st.session_state.factor_evolution_history),
                "+1"
            )
    
    def render_llm_factor_generation(self):
        """LLM驱动因子生成"""
        st.subheader("🤖 LLM驱动因子生成")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            利用大语言模型(LLM)自动生成量化因子:
            - 📝 描述因子想法，LLM生成代码
            - 🔍 自动验证因子有效性
            - 🧬 进化优化因子表达式
            - 📊 自动回测评估
            """)
        
        with col2:
            st.info(f"""
            **当前状态**
            - LLM模型: GPT-4
            - 生成模式: 自动
            - 运行状态: {'🟢 运行中' if st.session_state.factor_generation_running else '🔴 已停止'}
            """)
        
        st.divider()
        
        # 因子生成配置
        with st.expander("⚙️ 因子生成配置", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                factor_type = st.selectbox(
                    "因子类型",
                    ["技术因子", "基本面因子", "量价因子", "情绪因子", "混合因子"]
                )
            
            with col2:
                generation_method = st.selectbox(
                    "生成方法",
                    ["从零生成", "基于模板", "进化改进", "研报启发"]
                )
            
            with col3:
                max_factors = st.number_input(
                    "最大生成数量",
                    min_value=1,
                    max_value=50,
                    value=10
                )
            
            factor_description = st.text_area(
                "因子描述 (可选)",
                placeholder="例如: 生成一个结合成交量和价格动量的因子，用于捕捉短期趋势反转...",
                height=100
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚀 开始生成", use_container_width=True, type="primary"):
                    self.start_factor_generation(factor_type, generation_method, max_factors, factor_description)
            
            with col2:
                if st.button("⏸️ 暂停", use_container_width=True):
                    st.session_state.factor_generation_running = False
                    st.success("已暂停因子生成")
            
            with col3:
                if st.button("🔄 重置", use_container_width=True):
                    st.session_state.generated_factors = []
                    st.session_state.factor_generation_running = False
                    st.rerun()
        
        # 生成进度
        if st.session_state.factor_generation_running:
            st.subheader("📊 生成进度")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 模拟生成过程
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"正在生成因子... {i+1}%")
                if i % 20 == 0:
                    st.session_state.generated_factors.append(self.generate_mock_factor(i))
        
        # 生成的因子列表
        if st.session_state.generated_factors:
            st.subheader("📋 生成的因子")
            self.render_factor_list(st.session_state.generated_factors)
    
    def render_report_factor_extraction(self):
        """研报因子提取"""
        st.subheader("📄 研报因子提取")
        
        st.markdown("""
        从研究报告中自动提取和实现量化因子:
        - 📁 上传PDF研报或输入文本
        - 🔍 LLM解析因子定义
        - 💻 自动生成Python代码
        - ✅ 验证因子可行性
        """)
        
        # 初始化session state
        if 'extracted_factors_from_report' not in st.session_state:
            st.session_state.extracted_factors_from_report = []
        
        st.divider()
        
        # 研报上传
        col1, col2 = st.columns([2, 1])
        
        with col1:
            upload_method = st.radio(
                "选择输入方式",
                ["上传PDF文件", "粘贴文本", "输入URL"],
                horizontal=True
            )
            
            if upload_method == "上传PDF文件":
                uploaded_file = st.file_uploader(
                    "上传研报PDF",
                    type=['pdf'],
                    help="支持券商研报、学术论文等"
                )
                if uploaded_file:
                    st.success(f"已上传: {uploaded_file.name}")
            
            elif upload_method == "粘贴文本":
                report_text = st.text_area(
                    "粘贴研报内容",
                    height=300,
                    placeholder="粘贴研报中关于因子定义的部分..."
                )
            
            else:  # URL
                report_url = st.text_input(
                    "研报URL",
                    placeholder="https://example.com/report.pdf"
                )
        
        with col2:
            st.info("""
            **支持的报告类型**
            - 券商研报
            - 学术论文
            - 因子白皮书
            - 策略说明书
            
            **提取信息**
            - 因子定义
            - 计算公式
            - 数据要求
            - 应用场景
            """)
        
        if st.button("🔍 开始提取", type="primary", use_container_width=True):
            if uploaded_file:
                # 保存上传的文件
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                with st.spinner("正在解析研报..."):
                    self.extract_factors_from_report(pdf_path)
            else:
                st.warning("请先上传PDF文件")
        
        # 提取结果
        st.divider()
        st.subheader("📊 提取结果")
        
        # 使用session state中的提取结果
        if st.session_state.extracted_factors_from_report:
            extracted_factors = st.session_state.extracted_factors_from_report
        else:
            # 默认示例数据
            extracted_factors = [
                {
                    "name": "动量因子_MA20",
                    "description": "20日移动平均线动量",
                    "formulation": "(close - ma(close, 20)) / ma(close, 20)",
                    "variables": ["close", "ma20"],
                    "code": "def factor_ma20(data):\n    ma20 = data['close'].rolling(20).mean()\n    return (data['close'] - ma20) / ma20"
                },
                {
                    "name": "成交量价格背离",
                    "description": "成交量与价格变化的相关性",
                    "formulation": "corr(volume, close, 10)",
                    "variables": ["volume", "close"],
                    "code": "def factor_volume_corr(data):\n    return data['volume'].rolling(10).corr(data['close'])"
                }
            ]
        
        for factor in extracted_factors:
            confidence = factor.get('confidence', 0.85)  # 默认置信度
            with st.expander(f"✅ {factor['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**描述:** {factor.get('description', 'N/A')}")
                    if factor.get('formulation'):
                        st.markdown(f"**公式:** `{factor['formulation']}`")
                    if factor.get('variables'):
                        st.markdown(f"**变量:** {', '.join(factor['variables'])}")
                with col2:
                    if factor.get('code'):
                        st.code(factor['code'], language='python')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ 采纳", key=f"accept_{factor['name']}"):
                        st.success("已添加到因子库")
                with col2:
                    if st.button("✏️ 编辑", key=f"edit_{factor['name']}"):
                        st.info("打开编辑器...")
                with col3:
                    if st.button("❌ 拒绝", key=f"reject_{factor['name']}"):
                        st.warning("已拒绝")
    
    def render_factor_evolution(self):
        """因子进化循环"""
        st.subheader("🔄 因子进化循环")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            通过迭代优化持续改进因子性能:
            - 🧬 遗传算法优化因子参数
            - 🔄 自动生成因子变体
            - 📈 性能驱动的进化方向
            - 🎯 多目标优化(IC/IR/换手率)
            """)
        
        with col2:
            st.metric("当前代数", len(st.session_state.factor_evolution_history), "+1")
            st.metric("种群大小", 20, "0")
            st.metric("最优适应度", "0.876", "+0.023")
        
        st.divider()
        
        # 进化配置
        with st.expander("⚙️ 进化参数配置", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                population_size = st.slider("种群大小", 10, 100, 20)
                mutation_rate = st.slider("变异率", 0.0, 1.0, 0.1)
            
            with col2:
                crossover_rate = st.slider("交叉率", 0.0, 1.0, 0.7)
                max_generations = st.slider("最大代数", 5, 50, 10)
            
            with col3:
                selection_method = st.selectbox(
                    "选择方法",
                    ["锦标赛", "轮盘赌", "精英主义"]
                )
                fitness_function = st.selectbox(
                    "适应度函数",
                    ["IC", "IC_IR", "Sharpe", "综合得分"]
                )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 开始进化", type="primary", use_container_width=True):
                self.start_evolution()
        with col2:
            if st.button("⏸️ 停止", use_container_width=True):
                st.warning("已停止进化")
        
        # 进化历史可视化
        if st.session_state.factor_evolution_history:
            st.subheader("📈 进化历史")
            
            # 模拟进化数据
            generations = list(range(1, len(st.session_state.factor_evolution_history) + 1))
            best_fitness = [0.65 + i * 0.02 + np.random.uniform(-0.01, 0.01) 
                           for i in range(len(generations))]
            avg_fitness = [0.5 + i * 0.015 + np.random.uniform(-0.01, 0.01) 
                          for i in range(len(generations))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=generations, y=best_fitness,
                name='最佳适应度',
                line=dict(color='green', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=generations, y=avg_fitness,
                name='平均适应度',
                line=dict(color='blue', width=2, dash='dash')
            ))
            fig.update_layout(
                title="进化曲线",
                xaxis_title="代数",
                yaxis_title="适应度",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_factor_evaluation(self):
        """因子性能评估"""
        st.subheader("📊 因子性能评估")
        
        st.markdown("""
        全面评估生成因子的性能表现:
        - 📈 IC/IR/Sharpe等核心指标
        - 📊 分层回测和多空收益
        - 🔄 因子衰减分析
        - 🎯 行业/风格中性化
        """)
        
        st.divider()
        
        # 因子选择
        if st.session_state.generated_factors:
            selected_factor = st.selectbox(
                "选择要评估的因子",
                [f.get('name', f'Factor_{i}') for i, f in enumerate(st.session_state.generated_factors)]
            )
            
            # 评估指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("IC均值", "0.083", "+0.012")
                st.metric("IC标准差", "0.156", "-0.008")
            with col2:
                st.metric("IR比率", "0.532", "+0.076")
                st.metric("ICIR", "0.489", "+0.054")
            with col3:
                st.metric("多头年化", "18.3%", "+2.1%")
                st.metric("空头年化", "-12.6%", "-1.8%")
            with col4:
                st.metric("多空年化", "30.9%", "+3.9%")
                st.metric("最大回撤", "-8.2%", "+1.3%")
            
            st.divider()
            
            # IC时序图
            st.subheader("📈 IC时序分析")
            dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
            ic_values = np.random.normal(0.08, 0.15, 250)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=ic_values,
                name='IC值',
                line=dict(color='blue')
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_hline(y=ic_values.mean(), line_dash="dot", line_color="red", 
                         annotation_text=f"均值: {ic_values.mean():.3f}")
            fig.update_layout(
                title="因子IC时序",
                xaxis_title="日期",
                yaxis_title="IC值",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 分层回测
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 分层回测")
                layers = ['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)']
                returns = [5.2, 8.7, 12.3, 16.8, 22.4]
                
                fig = px.bar(x=layers, y=returns, 
                            labels={'x': '分组', 'y': '年化收益率(%)'},
                            title="分层收益")
                fig.update_traces(marker_color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 累计收益")
                dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
                long_ret = np.cumprod(1 + np.random.normal(0.001, 0.02, 250)) - 1
                short_ret = np.cumprod(1 + np.random.normal(-0.0005, 0.015, 250)) - 1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=long_ret * 100, name='多头', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=dates, y=short_ret * 100, name='空头', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=dates, y=(long_ret - short_ret) * 100, 
                                        name='多空', line=dict(color='blue', width=3)))
                fig.update_layout(
                    title="累计收益曲线",
                    xaxis_title="日期",
                    yaxis_title="收益率(%)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("还没有生成的因子，请先在'LLM因子生成'或'研报因子提取'中生成因子")
    
    def start_factor_generation(self, factor_type, method, max_factors, description):
        """开始因子生成：优先调用RD-Agent真实接口，失败则回退到Mock"""
        st.session_state.factor_generation_running = True
        
        with st.spinner(f"正在生成{factor_type}..."):
            try:
                from .rdagent_api import get_rdagent_api
                import asyncio
                api = get_rdagent_api()
                cfg = {
                    'factor_type': factor_type,
                    'method': method,
                    'max_factors': int(max_factors),
                    'description': (description or '').strip(),
                }
                res = asyncio.run(api.run_factor_generation(cfg))
                if res.get('success') and res.get('factors'):
                    st.session_state.generated_factors = res['factors']
                else:
                    # 回退到本地Mock生成
                    st.warning("RD-Agent未可用或返回为空，使用本地模拟生成")
                    st.session_state.generated_factors = [
                        self.generate_mock_factor(i, factor_type) for i in range(min(5, max_factors))
                    ]
            except Exception as e:
                st.error(f"因子生成调用失败: {e}")
                st.warning("使用本地模拟数据代替")
                st.session_state.generated_factors = [
                    self.generate_mock_factor(i, factor_type) for i in range(min(5, max_factors))
                ]
        
        st.session_state.factor_generation_running = False
        st.success(f"成功生成 {len(st.session_state.generated_factors)} 个因子!")
        st.rerun()
    
    def generate_mock_factor(self, idx: int, factor_type: str = "技术因子") -> Dict:
        """生成模拟因子"""
        factor_names = [
            "momentum_ma20", "volume_price_corr", "rsi_divergence",
            "bollinger_width", "macd_signal", "atr_ratio",
            "volume_momentum", "price_acceleration", "liquidity_factor"
        ]
        
        return {
            "name": factor_names[idx % len(factor_names)],
            "type": factor_type,
            "ic": np.random.uniform(0.05, 0.15),
            "ir": np.random.uniform(0.3, 0.8),
            "valid": np.random.random() > 0.3,
            "created_at": datetime.now() - timedelta(minutes=idx),
            "code": f"def factor_{idx}(data):\n    return (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()"
        }
    
    def render_factor_list(self, factors: List[Dict]):
        """渲染因子列表"""
        df_data = []
        for i, factor in enumerate(factors):
            df_data.append({
                "序号": i + 1,
                "因子名称": factor.get('name', f'Factor_{i}'),
                "类型": factor.get('type', '未知'),
                "IC": f"{factor.get('ic', 0):.3f}",
                "IR": f"{factor.get('ir', 0):.3f}",
                "状态": "✅ 有效" if factor.get('valid', False) else "❌ 无效",
                "生成时间": factor.get('created_at', datetime.now()).strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "状态": st.column_config.TextColumn(width="small"),
                "IC": st.column_config.NumberColumn(format="%.3f"),
                "IR": st.column_config.NumberColumn(format="%.3f"),
            }
        )
    
    def extract_factors_from_report(self, pdf_path: str):
        """从研报提取因子"""
        try:
            from .rdagent_api import get_rdagent_api
            import asyncio
            
            api = get_rdagent_api()
            
            # 调用API提取因子
            result = asyncio.run(api.run_factor_from_report(pdf_path))
            
            if result['success']:
                # 保存提取的因子
                st.session_state.extracted_factors_from_report = result.get('factors', [])
                
                # 显示假设
                if result.get('hypothesis'):
                    st.info(f"💡 **研报假设**: {result['hypothesis']}")
                
                st.success(result['message'])
            else:
                st.error(result.get('message', '提取失败'))
                
        except Exception as e:
            st.error(f"提取失败: {e}")
            import time
            time.sleep(1)
            st.warning("使用模拟数据展示")
    
    def start_evolution(self):
        """开始进化"""
        st.session_state.factor_evolution_history.append({
            'generation': len(st.session_state.factor_evolution_history) + 1,
            'best_fitness': 0.65 + len(st.session_state.factor_evolution_history) * 0.02,
            'avg_fitness': 0.5 + len(st.session_state.factor_evolution_history) * 0.015
        })
        st.success(f"完成第 {len(st.session_state.factor_evolution_history)} 代进化!")
        st.rerun()


def render():
    """渲染入口函数"""
    tab = FactorMiningTab()
    tab.render()
