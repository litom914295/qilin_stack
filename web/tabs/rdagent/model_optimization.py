"""
RD-Agent 模型优化模块
- 自动模型搜索
- 架构优化
- 超参数调优
- 模型ensemble
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any

class ModelOptimizationTab:
    """模型优化Tab"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """初始化session state"""
        if 'optimization_running' not in st.session_state:
            st.session_state.optimization_running = False
        if 'optimized_models' not in st.session_state:
            st.session_state.optimized_models = []
        if 'optimization_history' not in st.session_state:
            st.session_state.optimization_history = []
    
    def render(self):
        """渲染模型优化页面"""
        st.header("🏗️ 模型优化")
        
        # 顶部指标
        self.render_metrics()
        
        st.divider()
        
        # 主要内容区域
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 模型架构搜索",
            "⚙️ 超参数调优",
            "🎯 模型Ensemble",
            "📊 性能对比",
            "🔍 模型解释器"
        ])
        
        with tab1:
            self.render_architecture_search()
        
        with tab2:
            self.render_hyperparameter_tuning()
        
        with tab3:
            self.render_model_ensemble()
        
        with tab4:
            self.render_performance_comparison()
        
        with tab5:
            self.render_model_interpreter()
    
    def render_metrics(self):
        """渲染顶部指标"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("优化模型数", len(st.session_state.optimized_models), "+2")
        
        with col2:
            st.metric("最优准确率", "87.3%", "+3.2%")
        
        with col3:
            st.metric("平均Sharpe", "1.85", "+0.23")
        
        with col4:
            st.metric("优化轮次", len(st.session_state.optimization_history), "+1")
        
        with col5:
            st.metric("节约成本", "$42.3", "+$8.5")
    
    def render_architecture_search(self):
        """模型架构搜索"""
        st.subheader("🔍 自动模型架构搜索 (NAS)")
        
        st.markdown("""
        利用神经架构搜索(NAS)自动发现最优模型结构:
        - 🧠 自动设计神经网络架构
        - 🔧 优化层数、宽度、激活函数
        - ⚡ 高效搜索算法(ENAS/DARTS)
        - 📊 性能与效率平衡
        """)
        
        st.divider()
        
        # 搜索配置
        with st.expander("⚙️ 架构搜索配置", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_space = st.selectbox(
                    "搜索空间",
                    ["LSTM变体", "Transformer变体", "CNN-RNN混合", "自定义"]
                )
                search_method = st.selectbox(
                    "搜索方法",
                    ["ENAS", "DARTS", "随机搜索", "贝叶斯优化"]
                )
            
            with col2:
                max_layers = st.slider("最大层数", 2, 20, 10)
                hidden_size_range = st.slider(
                    "隐藏层大小",
                    min_value=32,
                    max_value=512,
                    value=(64, 256)
                )
            
            with col3:
                search_budget = st.number_input(
                    "搜索预算($)",
                    min_value=10,
                    max_value=1000,
                    value=100
                )
                max_trials = st.number_input(
                    "最大试验次数",
                    min_value=10,
                    max_value=500,
                    value=50
                )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 开始搜索", type="primary", use_container_width=True):
                self.start_architecture_search()
        with col2:
            if st.button("⏸️ 停止", use_container_width=True):
                st.session_state.optimization_running = False
        
        # 搜索结果
        if st.session_state.optimized_models:
            st.subheader("📊 发现的模型架构")
            
            models_df = pd.DataFrame([
                {"模型ID": f"Model_{i+1}", 
                 "架构": m.get('architecture', 'Unknown'),
                 "层数": m.get('layers', 0),
                 "参数量": f"{m.get('params', 0)/1e6:.2f}M",
                 "准确率": f"{m.get('accuracy', 0):.2%}",
                 "训练时间": f"{m.get('train_time', 0):.1f}s"}
                for i, m in enumerate(st.session_state.optimized_models[:5])
            ])
            
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            
            # 架构可视化
            st.subheader("🎨 最优架构可视化")
            self.visualize_architecture()
    
    def render_hyperparameter_tuning(self):
        """超参数调优"""
        st.subheader("⚙️ 超参数自动调优")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            使用贝叶斯优化/网格搜索自动调优超参数:
            - 📊 学习率、批大小、正则化系数
            - 🎯 多目标优化(性能+速度+内存)
            - 📈 自适应搜索策略
            - 💾 最优配置保存
            """)
        
        with col2:
            st.info("""
            **支持的算法**
            - Bayesian Optimization
            - Optuna
            - Hyperopt
            - Grid Search
            """)
        
        st.divider()
        
        # 调优配置
        with st.expander("⚙️ 调优参数", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**学习率范围**")
                lr_min = st.number_input("最小值", value=1e-5, format="%.6f", key="lr_min")
                lr_max = st.number_input("最大值", value=1e-2, format="%.4f", key="lr_max")
                
                st.write("**批大小**")
                batch_sizes = st.multiselect(
                    "选择批大小",
                    [16, 32, 64, 128, 256],
                    default=[32, 64]
                )
            
            with col2:
                st.write("**正则化**")
                l2_weight = st.slider("L2权重", 0.0, 0.1, 0.01, 0.001)
                dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
                
                st.write("**优化器**")
                optimizer = st.multiselect(
                    "选择优化器",
                    ["Adam", "AdamW", "SGD", "RMSprop"],
                    default=["Adam", "AdamW"]
                )
            
            tuning_method = st.selectbox(
                "调优方法",
                ["Bayesian Optimization", "Optuna", "Random Search", "Grid Search"]
            )
            
            max_evals = st.slider("最大评估次数", 10, 200, 50)
        
        if st.button("🎯 开始调优", type="primary", use_container_width=True):
            with st.spinner("正在调优超参数..."):
                self.start_hyperparameter_tuning()
        
        # 调优历史
        if st.session_state.optimization_history:
            st.subheader("📈 调优历史")
            
            # 参数重要性
            col1, col2 = st.columns(2)
            
            with col1:
                importance_data = {
                    "参数": ["学习率", "批大小", "Dropout", "L2权重", "隐藏层大小"],
                    "重要性": [0.35, 0.25, 0.18, 0.12, 0.10]
                }
                fig = px.bar(importance_data, x="重要性", y="参数", 
                            orientation='h', title="参数重要性分析")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 优化曲线
                trials = list(range(1, len(st.session_state.optimization_history) + 1))
                best_scores = [0.7 + i * 0.01 + np.random.uniform(-0.005, 0.005) 
                              for i in range(len(trials))]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trials, y=best_scores,
                    mode='lines+markers',
                    name='最优得分',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    title="优化进度",
                    xaxis_title="试验次数",
                    yaxis_title="得分"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_model_ensemble(self):
        """模型集成"""
        st.subheader("🎯 模型Ensemble")
        
        st.markdown("""
        智能组合多个模型以提升性能:
        - 📊 Bagging/Boosting/Stacking
        - 🎲 多样性优化
        - ⚖️ 权重自适应学习
        - 🚀 在线集成更新
        """)
        
        st.divider()
        
        # 模型选择
        with st.expander("🔧 Ensemble配置", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                ensemble_method = st.selectbox(
                    "集成方法",
                    ["Voting", "Stacking", "Blending", "Boosting"]
                )
                
                selected_models = st.multiselect(
                    "选择基学习器",
                    ["LSTM", "GRU", "Transformer", "LightGBM", "XGBoost", "CatBoost"],
                    default=["LSTM", "LightGBM"]
                )
            
            with col2:
                weight_method = st.selectbox(
                    "权重计算",
                    ["均匀权重", "性能加权", "优化权重", "自适应权重"]
                )
                
                diversity_weight = st.slider(
                    "多样性权重",
                    0.0, 1.0, 0.3,
                    help="控制模型多样性的重要程度"
                )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 创建Ensemble", type="primary", use_container_width=True):
                self.create_ensemble()
        with col2:
            if st.button("📊 评估性能", use_container_width=True):
                st.info("正在评估ensemble性能...")
        
        # Ensemble结果
        st.subheader("📊 Ensemble性能")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ensemble准确率", "89.7%", "+5.2%")
        with col2:
            st.metric("相比最优单模型", "+2.4%", "提升")
        with col3:
            st.metric("推理时间", "125ms", "+35ms")
        
        # 模型权重
        st.subheader("⚖️ 模型权重分布")
        
        weights_data = {
            "模型": ["LSTM", "LightGBM", "Transformer", "GRU"],
            "权重": [0.35, 0.28, 0.22, 0.15]
        }
        
        fig = px.pie(weights_data, values="权重", names="模型", 
                    title="Ensemble权重分布")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_comparison(self):
        """性能对比"""
        st.subheader("📊 模型性能对比")
        
        # 性能对比表
        comparison_data = {
            "模型": ["Baseline", "优化后LSTM", "优化后GRU", "Transformer", "Ensemble"],
            "准确率": [0.78, 0.84, 0.83, 0.85, 0.897],
            "Sharpe": [1.23, 1.65, 1.58, 1.72, 1.85],
            "最大回撤": [0.18, 0.14, 0.15, 0.13, 0.11],
            "训练时间(s)": [120, 180, 165, 240, 300],
            "参数量(M)": [2.3, 3.8, 3.5, 5.2, 12.8]
        }
        
        df = pd.DataFrame(comparison_data)
        
        st.dataframe(
            df.style.highlight_max(subset=["准确率", "Sharpe"], color="lightgreen")
                   .highlight_min(subset=["最大回撤", "训练时间(s)"], color="lightgreen"),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # 雷达图对比
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 综合性能雷达图")
            
            categories = ['准确率', 'Sharpe', '稳定性', '速度', '可解释性']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=[0.78, 0.65, 0.70, 0.90, 0.80],
                theta=categories,
                fill='toself',
                name='Baseline'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=[0.897, 0.92, 0.88, 0.65, 0.60],
                theta=categories,
                fill='toself',
                name='Ensemble'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 回测收益对比")
            
            dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
            baseline = np.cumprod(1 + np.random.normal(0.0005, 0.015, 250)) - 1
            optimized = np.cumprod(1 + np.random.normal(0.001, 0.015, 250)) - 1
            ensemble = np.cumprod(1 + np.random.normal(0.0012, 0.012, 250)) - 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=baseline*100, name='Baseline', 
                                    line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=dates, y=optimized*100, name='优化模型',
                                    line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dates, y=ensemble*100, name='Ensemble',
                                    line=dict(color='green', width=3)))
            
            fig.update_layout(
                title="累计收益对比",
                xaxis_title="日期",
                yaxis_title="收益率(%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def start_architecture_search(self):
        """开始架构搜索"""
        with st.spinner("正在搜索最优架构..."):
            import time
            time.sleep(2)
            
            # 生成模拟模型
            for i in range(5):
                model = {
                    'architecture': np.random.choice(['LSTM', 'GRU', 'Transformer']),
                    'layers': np.random.randint(3, 12),
                    'params': np.random.randint(1, 10) * 1e6,
                    'accuracy': np.random.uniform(0.75, 0.90),
                    'train_time': np.random.uniform(50, 200)
                }
                st.session_state.optimized_models.append(model)
        
        st.success(f"搜索完成! 发现 {len(st.session_state.optimized_models)} 个候选架构")
        st.rerun()
    
    def start_hyperparameter_tuning(self):
        """开始超参数调优"""
        import time
        time.sleep(2)
        
        st.session_state.optimization_history.append({
            'trial': len(st.session_state.optimization_history) + 1,
            'score': 0.7 + len(st.session_state.optimization_history) * 0.01
        })
        
        st.success("调优完成!")
        st.rerun()
    
    def create_ensemble(self):
        """创建ensemble"""
        with st.spinner("正在创建ensemble..."):
            import time
            time.sleep(1.5)
        
        st.success("Ensemble创建成功!")
    
    def visualize_architecture(self):
        """可视化模型架构"""
        st.code("""
        最优架构:
        ├── Input Layer (128)
        ├── LSTM Layer (256, dropout=0.2)
        ├── Attention Layer
        ├── LSTM Layer (128, dropout=0.1)
        ├── Dense Layer (64, relu)
        ├── Dropout (0.3)
        └── Output Layer (1, sigmoid)
        
        总参数: 3.2M
        训练参数: 3.2M
        """, language="text")
    
    def render_model_interpreter(self):
        """模型解释器模块"""
        st.subheader("🔍 模型解释器 (Model Interpreter)")
        
        st.info("""
        **模型解释器**帮助理解模型决策过程，提高可解释性。
        - 🎯 SHAP值分析
        - 📊 特征重要性
        - 🔍 决策路径追踪
        - 🧠 注意力可视化
        """)
        
        # 选择解释方法
        interp_method = st.selectbox(
            "解释方法",
            ["SHAP分析", "特征重要性", "LIME局部解释", "注意力热力图"],
            key="interp_method"
        )
        
        st.divider()
        
        # 1. SHAP分析
        if interp_method == "SHAP分析":
            st.subheader("🎯 SHAP (SHapley Additive exPlanations)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sample_idx = st.number_input(
                    "样本索引",
                    min_value=0,
                    max_value=100,
                    value=0
                )
            
            with col2:
                baseline = st.selectbox(
                    "基线选择",
                    ["均值", "中位数", "零值"]
                )
            
            if st.button("🚀 计算SHAP值", key="calc_shap"):
                with st.spinner("正在计算SHAP值..."):
                    import time
                    time.sleep(1)
                
                st.success("✅ SHAP值计算完成!")
                
                # 生成示例数据
                np.random.seed(42)
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD']
                shap_values = np.random.randn(len(features)) * 0.1
                
                # 1.1 SHAP值条形图
                st.subheader("📊 SHAP值分布")
                
                shap_df = pd.DataFrame({
                    '特征': features,
                    'SHAP值': shap_values
                }).sort_values('SHAP值', key=abs, ascending=False)
                
                fig = go.Figure()
                
                colors = ['red' if x < 0 else 'green' for x in shap_df['SHAP值']]
                
                fig.add_trace(go.Bar(
                    x=shap_df['SHAP值'],
                    y=shap_df['特征'],
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{v:.4f}" for v in shap_df['SHAP值']],
                    textposition='outside'
                ))
                
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title=f"样本{sample_idx}的SHAP值分解",
                    xaxis_title="SHAP值 (对预测的贡献)",
                    yaxis_title="特征",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 1.2 SHAP Summary Plot
                st.subheader("📈 SHAP Summary Plot")
                
                # 生成多样本SHAP值
                n_samples = 100
                shap_matrix = np.random.randn(n_samples, len(features)) * 0.1
                feature_values = np.random.randn(n_samples, len(features))
                
                fig = go.Figure()
                
                for i, feature in enumerate(features):
                    fig.add_trace(go.Scatter(
                        x=shap_matrix[:, i],
                        y=[feature] * n_samples,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=feature_values[:, i],
                            colorscale='RdBu',
                            showscale=i==0,
                            colorbar=dict(title="特征值") if i==0 else None
                        ),
                        showlegend=False,
                        hovertemplate=f'{feature}<br>SHAP: %{{x:.4f}}<extra></extra>'
                    ))
                
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title="SHAP Summary Plot - 特征重要性概览",
                    xaxis_title="SHAP值",
                    yaxis_title="特征",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 1.3 SHAP依赖图
                st.subheader("🔗 SHAP依赖图 (Dependence Plot)")
                
                selected_feature = st.selectbox(
                    "选择特征",
                    features,
                    key="shap_dep_feature"
                )
                
                feature_idx = features.index(selected_feature)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=feature_values[:, feature_idx],
                    y=shap_matrix[:, feature_idx],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=feature_values[:, (feature_idx+1)%len(features)],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="交互特征值")
                    ),
                    hovertemplate=f'{selected_feature}: %{{x:.4f}}<br>SHAP: %{{y:.4f}}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"{selected_feature}的SHAP依赖关系",
                    xaxis_title=f"{selected_feature}值",
                    yaxis_title="SHAP值",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 2. 特征重要性
        elif interp_method == "特征重要性":
            st.subheader("📊 特征重要性分析")
            
            importance_type = st.radio(
                "重要性类型",
                ["置换重要性", "增益/信息增益", "梯度重要性"],
                horizontal=True
            )
            
            if st.button("🚀 计算重要性", key="calc_importance"):
                with st.spinner("正在计算特征重要性..."):
                    import time
                    time.sleep(0.8)
                
                st.success("✅ 重要性计算完成!")
                
                # 生成示例数据
                np.random.seed(42)
                features = ['Close', 'Volume', 'MA20', 'RSI', 'MACD', 'BB_width', 'ATR', 'OBV', 'MFI', 'CCI']
                importances = np.random.rand(len(features))
                importances = importances / importances.sum()  # 归一化
                
                importance_df = pd.DataFrame({
                    '特征': features,
                    '重要性': importances,
                    '累计重要性': np.cumsum(importances)
                }).sort_values('重要性', ascending=False)
                
                # 2.1 重要性柱状图
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=importance_df['特征'],
                    y=importance_df['重要性'],
                    marker=dict(
                        color=importance_df['重要性'],
                        colorscale='Blues',
                        showscale=False
                    ),
                    text=[f"{v:.2%}" for v in importance_df['重要性']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="特征重要性排序",
                    xaxis_title="特征",
                    yaxis_title="重要性得分",
                    yaxis=dict(tickformat='.0%'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 2.2 累计重要性曲线
                st.subheader("📈 累计重要性曲线")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(importance_df)+1)),
                    y=importance_df['累计重要性'],
                    mode='lines+markers',
                    line=dict(color='green', width=3),
                    marker=dict(size=10),
                    text=importance_df['特征'],
                    hovertemplate='%{text}<br>累计重要性: %{y:.2%}<extra></extra>'
                ))
                
                fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                            annotation_text="80%阈值")
                
                fig.update_layout(
                    title="累计重要性曲线 (Pareto图)",
                    xaxis_title="特征数量",
                    yaxis_title="累计重要性",
                    yaxis=dict(tickformat='.0%'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 详细表格
                st.dataframe(
                    importance_df.style.format({
                        '重要性': '{:.2%}',
                        '累计重要性': '{:.2%}'
                    }).background_gradient(subset=['重要性'], cmap='Blues'),
                    use_container_width=True,
                    hide_index=True
                )
        
        # 3. LIME局部解释
        elif interp_method == "LIME局部解释":
            st.subheader("🔍 LIME (Local Interpretable Model-agnostic Explanations)")
            
            st.markdown("""
            **LIME**通过局部线性模型近似解释复杂模型的单个预测。
            """)
            
            sample_idx = st.number_input(
                "选择解释样本",
                min_value=0,
                max_value=100,
                value=0,
                key="lime_sample"
            )
            
            if st.button("🚀 生成LIME解释", key="calc_lime"):
                with st.spinner("正在生成LIME解释..."):
                    import time
                    time.sleep(1)
                
                st.success("✅ LIME解释生成完成!")
                
                # 生成示例数据
                np.random.seed(sample_idx)
                features = ['MA5', 'MA10', 'RSI', 'MACD', 'Volume', 'ATR']
                lime_weights = np.random.randn(len(features)) * 0.2
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("预测值", "0.723", "+0.052")
                with col2:
                    st.metric("置信度", "87.3%")
                
                # LIME权重图
                lime_df = pd.DataFrame({
                    '特征': features,
                    '权重': lime_weights
                }).sort_values('权重', key=abs, ascending=False)
                
                fig = go.Figure()
                
                colors = ['red' if x < 0 else 'green' for x in lime_df['权重']]
                
                fig.add_trace(go.Bar(
                    x=lime_df['权重'],
                    y=lime_df['特征'],
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{v:+.3f}" for v in lime_df['权重']],
                    textposition='outside'
                ))
                
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title=f"样本{sample_idx}的LIME局部解释",
                    xaxis_title="特征权重",
                    yaxis_title="特征",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 4. 注意力热力图
        else:
            st.subheader("🧠 注意力机制可视化")
            
            st.markdown("""
            **注意力热力图**展示模型在作出预测时关注的时间步或特征。
            """)
            
            if st.button("🚀 生成注意力热力图", key="calc_attention"):
                with st.spinner("正在计算注意力权重..."):
                    import time
                    time.sleep(0.8)
                
                st.success("✅ 注意力热力图生成完成!")
                
                # 生成注意力矩阵
                seq_len = 20
                n_heads = 4
                attention_weights = np.random.rand(n_heads, seq_len, seq_len)
                # 软化为注意力权重
                attention_weights = np.exp(attention_weights) / np.exp(attention_weights).sum(axis=-1, keepdims=True)
                
                # 选择注意力头
                head_idx = st.slider("选择注意力头", 0, n_heads-1, 0, key="att_head")
                
                fig = go.Figure(data=go.Heatmap(
                    z=attention_weights[head_idx],
                    x=[f't-{i}' for i in range(seq_len-1, -1, -1)],
                    y=[f't-{i}' for i in range(seq_len-1, -1, -1)],
                    colorscale='YlOrRd',
                    colorbar=dict(title="注意力权重")
                ))
                
                fig.update_layout(
                    title=f"注意力头{head_idx+1}的注意力热力图",
                    xaxis_title="键 (Key) 时间步",
                    yaxis_title="查询 (Query) 时间步",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 平均注意力权重
                st.subheader("📊 平均注意力权重")
                
                avg_attention = attention_weights.mean(axis=0).mean(axis=0)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[f't-{i}' for i in range(seq_len-1, -1, -1)],
                    y=avg_attention,
                    marker=dict(
                        color=avg_attention,
                        colorscale='Blues',
                        showscale=False
                    )
                ))
                
                fig.update_layout(
                    title="各时间步的平均注意力",
                    xaxis_title="时间步",
                    yaxis_title="平均注意力权重",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)


def render():
    """渲染入口"""
    tab = ModelOptimizationTab()
    tab.render()
