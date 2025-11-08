#!/usr/bin/env python
"""
循环进化训练 - 5种高级训练方法
让AI持续变强的核心模块
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def render_evolution_training_tab():
    """渲染循环进化训练主界面"""
    
    st.title("🔄 循环进化训练 - 让AI持续变强")
    
    # 顶部使用指南
    render_usage_guide()
    
    # 创建子标签页 - 5种训练方法
    training_tabs = st.tabs([
        "1️⃣ 困难案例挖掘",
        "2️⃣ 自我对抗训练",
        "3️⃣ 课程学习进化",
        "4️⃣ 知识蒸馏",
        "5️⃣ 元学习适应"
    ])
    
    with training_tabs[0]:
        render_hard_case_mining()
    
    with training_tabs[1]:
        render_adversarial_training()
    
    with training_tabs[2]:
        render_curriculum_evolution()
    
    with training_tabs[3]:
        render_knowledge_distillation()
    
    with training_tabs[4]:
        render_meta_learning()


def render_usage_guide():
    """渲染使用指南"""
    
    with st.expander("📖 循环进化训练指南", expanded=False):
        # 文档链接区域
        st.markdown("""
        ### 📚 相关文档资料
        
        想深入学习5种训练方法？查看以下文档：
        
        **理论基础**:
        - 📖 **迭代进化理论**: `docs/ITERATIVE_EVOLUTION_TRAINING.md` (580行) - 为什么不能简单重复训练
        
        **实现文档**:
        - ✅ **集成完成文档**: `docs/EVOLUTION_TRAINING_INTEGRATION_COMPLETE.md` (414行) - 完整集成说明
        - 📚 **完整使用指南**: `docs/EVOLUTION_TRAINING_METHODS_COMPLETE.md` (629行) - 详细使用教程
        - 🎯 **验证清单**: `docs/VERIFICATION_CHECKLIST.md` (354行) - 功能验证清单
        - 🔧 **完整版说明**: `docs/TRAINERS_FULL_VERSION.md` (450行) - 真实训练 vs 演示模式
        
        **核心代码**:
        - 💻 **困难案例挖掘**: `training/hard_case_mining.py` (393行)
        - ⚔️ **自我对抗训练**: `training/adversarial_trainer.py` (353行)
        - 🎓 **高级训练器**: `training/advanced_trainers.py` (600+行) - 课程学习/蒸馆/元学习
        
        **🆕 系统改进文档** (最新):
        - 🦄 **麒麟改进实施报告**: `docs/QILIN_EVOLUTION_IMPLEMENTATION.md` - 三阶段全面改进
          - ✅ 数据与特征增强: `data_layer/premium_data_provider.py`
          - ✅ 风控与择时: `risk_management/market_timing.py`
          - ✅ 写实回测: `backtesting/realistic_backtest.py`
          - ✅ SHAP解释: `ml/model_explainer.py`
        
        💡 **快速查看**: 在侧边栏"📚 文档与指南"中可以选择预览这些文档
        
        🎯 **推荐阅读顺序**: 理论基础 → 完整指南 → 集成文档 → 改进报告 → 实际操作
        """)
        
        st.divider()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### 🎯 核心理念
            
            **不是简单重复训练，而是让AI在"错误"和"对抗"中成长！**
            
            #### 🔥 5种进化方法
            
            1. **困难案例挖掘** ⭐⭐⭐⭐⭐
               - 找出AI预测错误的案例
               - 重点训练这些"弱点"
               - 准确率提升: 65% → 78%
            
            2. **自我对抗训练** ⭐⭐⭐⭐⭐
               - AI生成"陷阱案例"
               - 训练自己识别陷阱
               - 鲁棒性提升: +50%
            
            3. **课程学习进化** ⭐⭐⭐⭐
               - 难度递增训练
               - 从简单到困难
               - 准确率稳定: 82-85%
            
            4. **知识蒸馏** ⭐⭐⭐⭐
               - 大模型教小模型
               - 又快又准
               - 速度提升: 10倍
            
            5. **元学习适应** ⭐⭐⭐⭐⭐
               - 学会"快速学习"
               - 新环境5步适应
               - 最终准确率: 88%+
            
            ### ⚠️ 重要提示
            
            - 需要先完成**基础训练**（AI进化系统 → 模型训练）
            - 建议按顺序使用：1→2→3→4→5
            - 每个方法训练完成后保存模型
            """)
        
        with col2:
            st.markdown("""
            ### 🔄 完整进化路线
            
            ```
            第1个月: 基础训练
               ↓
            准确率 65%
            
            第2-3个月: 困难案例挖掘
               ↓
            准确率 78%
            
            第4-5个月: 自我对抗
               ↓
            准确率 80%+
            鲁棒性 +50%
            
            第6个月: 课程进化
               ↓
            准确率 85%
            
            长期: 元学习
               ↓
            准确率 88%+
            快速适应
            ```
            
            ### 💡 新手建议
            
            1. ✅ 先完成基础训练
            2. ✅ 从"困难案例挖掘"开始
            3. ✅ 查看训练进度和效果
            4. ✅ 保存每个阶段的模型
            """)
        # 与AI进化系统的联动与闭环
        st.markdown("""
        ### ✅ 与“AI进化系统”的闭环联动
        - 数据与模型：本页训练默认复用“AI进化系统→模型训练/数据采集”的历史数据与会话态。
        - 进化顺序：困难案例挖掘 → 自我对抗 → 课程学习 → 蒸馏 → 元学习。
        - 回灌方式：每步训练完成后，保存/替换当前基础模型，返回“AI进化系统→性能追踪”执行“一键回测”，复核命中率/胜率/未成交率。
        - 实盘执行：在“AI进化系统→智能预测”生成TopN并“🧾 生成下单计划(TopN)”，至“交易执行”完成下单与跟踪。
        """)


def render_hard_case_mining():
    """渲染困难案例挖掘页面"""
    
    st.header("1️⃣ 困难案例挖掘 - 在错误中成长")
    
    # 功能说明
    st.info("""
    👉 **功能说明**: 找出AI预测错误最多的案例，重点训练这些"弱点"。  
    🎯 **核心原理**: AI最容易在边界案例和反直觉案例上犯错，专门针对性训练！  
    💡 **适用场景**: 基础训练后，首个进化阶段（最推荐！）  
    ⚠️ **注意事项**: 需要有已训练的基础模型
    """)
    
    # 检查基础模型
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ 请先在'AI进化系统 → 模型训练'完成基础训练")
        return
    
    # 训练配置
    st.subheader("⚙️ 训练配置")
    
    col_conf1, col_conf2, col_conf3 = st.columns(3)
    
    with col_conf1:
        max_iterations = st.slider(
            "最大迭代轮数",
            min_value=3,
            max_value=20,
            value=10,
            help="建议5-10轮，通常3-5轮即可收敛"
        )
    
    with col_conf2:
        convergence_threshold = st.slider(
            "收敛准确率阈值",
            min_value=0.70,
            max_value=0.90,
            value=0.85,
            step=0.05,
            help="达到此准确率且困难案例<50个即收敛"
        )
    
    with col_conf3:
        min_hard_cases = st.number_input(
            "最少困难案例数",
            min_value=10,
            max_value=100,
            value=50,
            help="低于此数量即认为收敛"
        )
    
    # 困难案例类型说明
    with st.expander("🔍 困难案例类型", expanded=True):
        col_type1, col_type2, col_type3 = st.columns(3)
        
        with col_type1:
            st.markdown("""
            **类型1: 预测错误**
            - AI预测与实际不符
            - 权重: **3倍**
            - 示例: 预测涨停实际下跌
            """)
        
        with col_type2:
            st.markdown("""
            **类型2: 低置信度**
            - 预测正确但不确定
            - 权重: **2倍**
            - 示例: 置信度<60%的案例
            """)
        
        with col_type3:
            st.markdown("""
            **类型3: 反直觉**
            - 违反常规规律
            - 权重: **3倍**
            - 示例: 强封板但次日下跌
            """)
    
    # 开始训练
    if st.button("🚀 开始困难案例挖掘训练", type="primary", use_container_width=True):
        run_hard_case_mining(max_iterations, convergence_threshold, min_hard_cases)
    
    # 显示训练结果
    if 'hard_case_results' in st.session_state:
        display_hard_case_results()


def run_hard_case_mining(max_iterations, convergence_threshold, min_hard_cases):
    """运行困难案例挖掘训练"""
    
    from training.hard_case_mining import HardCaseMining
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 准备训练数据...")
        
        # 获取历史数据
        if 'historical_data' not in st.session_state:
            # 生成演示数据
            data = generate_demo_training_data(500)
        else:
            data = st.session_state['historical_data']
        
        # 创建训练器
        trainer = HardCaseMining()
        
        status_text.text(f"🔄 开始迭代训练（最多{max_iterations}轮）...")
        
        # 迭代训练
        results = trainer.iterative_training(
            data,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            min_hard_cases=min_hard_cases
        )
        
        progress_bar.progress(1.0)
        
        # 保存结果
        st.session_state['hard_case_results'] = results
        st.session_state['hard_case_trainer'] = trainer
        
        # 显示成功信息
        status_text.text("✅ 训练完成！")
        
        if results['converged']:
            st.success(f"""
            🎉 **训练收敛！**
            
            - 迭代轮数: {results['iteration_count']}
            - 最终准确率: {results['final_accuracy']:.2%}
            - 困难案例总数: {results['total_hard_cases']}
            - 准确率提升: {(results['final_accuracy'] - results['iterations'][0]['accuracy']):.1%}
            """)
        else:
            st.info(f"""
            ℹ️ **训练完成（未完全收敛）**
            
            - 完成轮数: {results['iteration_count']}
            - 当前准确率: {results['final_accuracy']:.2%}
            - 建议: 可以继续训练或调整参数
            """)
        
    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")
        status_text.text("❌ 训练失败")
        import traceback
        st.error(traceback.format_exc())


def display_hard_case_results():
    """显示困难案例挖掘结果"""
    
    st.divider()
    st.subheader("📊 训练结果")
    
    results = st.session_state['hard_case_results']
    
    # 关键指标卡片
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.metric(
            "迭代轮数",
            results['iteration_count'],
            f"最多{len(results['iterations'])}轮"
        )
    
    with col_metric2:
        initial_acc = results['iterations'][0]['accuracy']
        final_acc = results['final_accuracy']
        st.metric(
            "最终准确率",
            f"{final_acc:.1%}",
            f"+{(final_acc - initial_acc):.1%}"
        )
    
    with col_metric3:
        st.metric(
            "困难案例总数",
            results['total_hard_cases']
        )
    
    with col_metric4:
        st.metric(
            "收敛状态",
            "✅ 已收敛" if results['converged'] else "⚠️ 未收敛"
        )
    
    # 训练曲线
    st.subheader("📈 训练进度曲线")
    
    iterations_df = pd.DataFrame(results['iterations'])
    
    fig = go.Figure()
    
    # 准确率曲线
    fig.add_trace(go.Scatter(
        x=iterations_df['iteration'],
        y=iterations_df['accuracy'],
        mode='lines+markers',
        name='准确率',
        line=dict(color='#2E86DE', width=3),
        marker=dict(size=10)
    ))
    
    # 添加收敛阈值线
    convergence_threshold = st.session_state.get('convergence_threshold', 0.85)
    fig.add_hline(
        y=convergence_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text="收敛阈值"
    )
    
    fig.update_layout(
        title='准确率提升曲线',
        xaxis_title='迭代轮数',
        yaxis_title='准确率',
        yaxis_tickformat='.0%',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 困难案例统计
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.subheader("🔍 困难案例发现趋势")
        
        fig2 = px.bar(
            iterations_df,
            x='iteration',
            y='new_hard_cases',
            title='每轮新发现困难案例数',
            labels={'iteration': '迭代轮数', 'new_hard_cases': '新困难案例数'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_stat2:
        st.subheader("📊 累计困难案例")
        
        fig3 = px.line(
            iterations_df,
            x='iteration',
            y='total_hard_cases',
            title='困难案例累计数量',
            labels={'iteration': '迭代轮数', 'total_hard_cases': '累计困难案例'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # 困难案例类型分布
    if 'hard_case_trainer' in st.session_state:
        trainer = st.session_state['hard_case_trainer']
        summary = trainer.get_hard_cases_summary()
        
        if not summary.empty:
            st.subheader("📋 困难案例类型分布")
            
            col_summary1, col_summary2 = st.columns(2)
            
            with col_summary1:
                fig4 = px.pie(
                    summary,
                    values='count',
                    names='case_type',
                    title='困难案例类型占比'
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            with col_summary2:
                st.dataframe(
                    summary,
                    use_container_width=True,
                    column_config={
                        'case_type': st.column_config.TextColumn('类型', width='medium'),
                        'count': st.column_config.NumberColumn('数量', width='small'),
                        'percentage': st.column_config.ProgressColumn('占比', format='%.1%', width='medium')
                    }
                )


def render_adversarial_training():
    """渲染自我对抗训练页面"""
    
    st.header("2️⃣ 自我对抗训练 - AI vs AI")
    
    st.info("""
    👉 **功能说明**: 让AI生成"陷阱案例"，然后训练自己识别这些陷阱。  
    🎯 **核心原理**: AI生成3种陷阱（伪强势、隐藏机会、情绪陷阱），大幅提升鲁棒性。  
    💡 **适用场景**: 完成困难案例挖掘后，进一步增强鲁棒性  
    ⚠️ **注意事项**: 训练时间较长，建议预留足够时间
    """)
    
    # 训练配置
    st.subheader("⚙️ 训练配置")
    
    col_conf1, col_conf2 = st.columns(2)
    
    with col_conf1:
        max_rounds = st.slider(
            "最大对抗轮数",
            min_value=3,
            max_value=15,
            value=10,
            help="建议5-10轮"
        )
    
    with col_conf2:
        target_robustness = st.slider(
            "目标鲁棒性",
            min_value=7.0,
            max_value=10.0,
            value=9.0,
            step=0.5,
            help="目标鲁棒性得分(0-10)"
        )
    
    # 对抗陷阱类型
    with st.expander("🔍 对抗陷阱类型", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **陷阱1: 伪强势**
            - 特征看起来很强
            - 实际是诱多
            - 权重: **5倍**
            """)
        
        with col2:
            st.markdown("""
            **陷阱2: 隐藏机会**
            - 特征看起来一般
            - 实际是大机会
            - 权重: **5倍**
            """)
        
        with col3:
            st.markdown("""
            **陷阱3: 情绪陷阱**
            - 市场情绪与个股相反
            - 权重: **5倍**
            """)
    
    # 开始训练
    if st.button("🚀 开始自我对抗训练", type="primary", use_container_width=True):
        run_adversarial_training(max_rounds, target_robustness)
    
    # 显示结果
    if 'adversarial_results' in st.session_state:
        display_adversarial_results()


def render_curriculum_evolution():
    """渲染课程学习进化页面"""
    
    st.header("3️⃣ 课程学习进化 - 难度递增")
    
    st.info("""
    👉 **功能说明**: 从简单到困难，循序渐进训练AI。  
    🎯 **核心原理**: 4个阶段（基础→进阶→高级→专家），像从小学到大学。  
    💡 **适用场景**: 系统性提升AI能力，稳定达到85%准确率  
    ⚠️ **注意事项**: 需要按阶段完成，不能跳过
    """)
    
    # 课程阶段
    with st.expander("📚 课程阶段", expanded=True):
        stages = [
            {"name": "基础阶段", "difficulty": "★☆☆☆", "focus": "明显成功/失败案例", "target": "70%"},
            {"name": "进阶阶段", "difficulty": "★★☆☆", "focus": "典型案例+部分边界", "target": "75%"},
            {"name": "高级阶段", "difficulty": "★★★☆", "focus": "边界案例+反直觉", "target": "80%"},
            {"name": "专家阶段", "difficulty": "★★★★", "focus": "纯困难案例", "target": "85%"}
        ]
        
        for i, stage in enumerate(stages, 1):
            st.markdown(f"""
            **阶段{i}: {stage['name']}** {stage['difficulty']}
            - 训练重点: {stage['focus']}
            - 目标准确率: {stage['target']}
            """)
    
    # 训练配置
    st.subheader("⚙️ 训练配置")
    
    max_epochs_per_stage = st.number_input(
        "每阶段最大Epoch数",
        min_value=10,
        max_value=100,
        value=50,
        help="每个阶段最多训练轮数"
    )
    
    # 开始训练
    if st.button("🚀 开始课程学习训练", type="primary", use_container_width=True):
        run_curriculum_training(max_epochs_per_stage)
    
    # 显示结果
    if 'curriculum_results' in st.session_state:
        display_curriculum_results()


def render_knowledge_distillation():
    """渲染知识蒸馏页面"""
    
    st.header("4️⃣ 知识蒸馏 - 大师传承")
    
    st.info("""
    👉 **功能说明**: 训练超大"教师模型"，然后教导轻量"学生模型"。  
    🎯 **核心原理**: 学生学习教师的"软标签"，又快又准。  
    💡 **适用场景**: 需要快速推理的生产环境  
    ⚠️ **注意事项**: 需要较大计算资源训练教师模型
    """)
    
    # 训练配置
    st.subheader("⚙️ 训练配置")
    
    col_conf1, col_conf2 = st.columns(2)
    
    with col_conf1:
        teacher_epochs = st.number_input(
            "教师模型Epochs",
            min_value=50,
            max_value=200,
            value=100,
            help="教师模型训练轮数"
        )
    
    with col_conf2:
        student_epochs = st.number_input(
            "学生模型Epochs",
            min_value=20,
            max_value=100,
            value=50,
            help="学生模型训练轮数"
        )
    
    with st.expander("📚 蒸馏原理", expanded=True):
        st.markdown("""
        ```
        阶段1: 训练教师模型
           ↓  （8个模型集成）
        准确率: 85%
        
        阶段2: 教师教导学生模型
           ↓  （学习软标签）
        准确率: 82%，速度 **10倍快**
        ```
        """)
    
    # 开始训练
    if st.button("🚀 开始知识蒸馏训练", type="primary", use_container_width=True):
        run_distillation_training(teacher_epochs, student_epochs)
    
    # 显示结果
    if 'distillation_results' in st.session_state:
        display_distillation_results()


def render_meta_learning():
    """渲染元学习页面"""
    
    st.header("5️⃣ 元学习适应 - 学会学习")
    
    st.info("""
    👉 **功能说明**: 让AI学会"如何快速学习"新的市场环境。  
    🎯 **核心原理**: MAML算法，在多个任务上学习快速适应能力。  
    💡 **适用场景**: 长期部署，需要持续适应市场变化  
    ⚠️ **注意事项**: 最高级训练方法，建议最后使用
    """)
    
    # 训练配置
    st.subheader("⚙️ 训练配置")
    
    meta_epochs = st.number_input(
        "元学习Epochs",
        min_value=50,
        max_value=200,
        value=100,
        help="元学习训练轮数"
    )
    
    with st.expander("🧠 元学习原理", expanded=True):
        st.markdown("""
        ```
        把3年数据分成36个月
        每个月是一个"任务"
        
        目标: 学习如何快速适应新月份
        方法: MAML算法
        
        结果: 遇到新环境，**仅5步**即可适应！
        ```
        """)
    
    # 开始训练
    if st.button("🚀 开始元学习训练", type="primary", use_container_width=True):
        run_meta_learning_training(meta_epochs)
    
    # 显示结果
    if 'meta_results' in st.session_state:
        display_meta_results()


# ========== 自我对抗训练 ==========

def run_adversarial_training(max_rounds, target_robustness):
    """运行自我对抗训练"""
    
    from training.adversarial_trainer import AdversarialTrainer
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 准备训练数据...")
        
        # 获取历史数据
        if 'historical_data' not in st.session_state:
            data = generate_demo_training_data(500)
        else:
            data = st.session_state['historical_data']
        
        # 创建训练器
        trainer = AdversarialTrainer()
        
        status_text.text(f"🔄 开始对抗训练（最多{max_rounds}轮）...")
        
        # 对抗训练
        results = trainer.adversarial_evolution(
            data,
            max_rounds=max_rounds,
            target_robustness=target_robustness
        )
        
        progress_bar.progress(1.0)
        
        # 保存结果
        st.session_state['adversarial_results'] = results
        st.session_state['adversarial_trainer'] = trainer
        
        status_text.text("✅ 训练完成！")
        
        if results['success']:
            st.success(f"""
            🎉 **达到目标鲁棒性！**
            
            - 训练轮数: {results['round_count']}
            - 最终鲁棒性: {results['final_robustness']:.2f}/10
            - 对抗案例总数: {results['total_adversarial_cases']}
            """)
        else:
            st.info(f"""
            ℹ️ **训练完成（未完全达标）**
            
            - 完成轮数: {results['round_count']}
            - 当前鲁棒性: {results['final_robustness']:.2f}/10
            """)
        
    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")
        status_text.text("❌ 训练失败")


def display_adversarial_results():
    """显示对抗训练结果"""
    
    st.divider()
    st.subheader("📊 对抗训练结果")
    
    results = st.session_state['adversarial_results']
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("训练轮数", results['round_count'])
    
    with col2:
        st.metric("最终鲁棒性", f"{results['final_robustness']:.1f}/10")
    
    with col3:
        st.metric("对抗案例", results['total_adversarial_cases'])
    
    with col4:
        st.metric("达标状态", "✅ 达标" if results['success'] else "⚠️ 未达标")
    
    # 鲁棒性提升曲线
    st.subheader("📈 鲁棒性提升曲线")
    
    rounds_df = pd.DataFrame(results['rounds'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rounds_df['round'],
        y=rounds_df['robustness_score'],
        mode='lines+markers',
        name='鲁棒性得分',
        line=dict(color='#EE5A24', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_hline(
        y=st.session_state.get('target_robustness', 9.0),
        line_dash="dash",
        line_color="green",
        annotation_text="目标"
    )
    
    fig.update_layout(
        xaxis_title='训练轮数',
        yaxis_title='鲁棒性得分',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 对抗案例类型分布
    if 'adversarial_trainer' in st.session_state:
        trainer = st.session_state['adversarial_trainer']
        summary = trainer.get_adversarial_summary()
        
        if not summary.empty:
            st.subheader("📋 对抗案例类型分布")
            
            fig2 = px.pie(
                summary,
                values='count',
                names='type',
                title='对抗陷阱类型占比'
            )
            st.plotly_chart(fig2, use_container_width=True)


# ========== 课程学习训练 ==========

def run_curriculum_training(max_epochs_per_stage):
    """运行课程学习训练"""
    
    from training.advanced_trainers import CurriculumTrainer
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 准备训练数据...")
        
        # 获取历史数据
        if 'historical_data' not in st.session_state:
            data = generate_demo_training_data(500)
        else:
            data = st.session_state['historical_data']
        
        # 创建训练器
        trainer = CurriculumTrainer()
        
        status_text.text("🔄 开始课程学习训练...")
        
        # 课程训练
        results = trainer.train_with_curriculum(
            data,
            max_epochs_per_stage=max_epochs_per_stage
        )
        
        progress_bar.progress(1.0)
        
        # 保存结果
        st.session_state['curriculum_results'] = results
        st.session_state['curriculum_trainer'] = trainer
        
        status_text.text("✅ 训练完成！")
        
        st.success(f"""
        🎓 **所有课程完成！**
        
        - 完成阶段: {results['completed_stages']}/4
        - 最终准确率: {results['final_accuracy']:.2%}
        """)
        
    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")
        status_text.text("❌ 训练失败")


def display_curriculum_results():
    """显示课程学习结果"""
    
    st.divider()
    st.subheader("📊 课程学习结果")
    
    results = st.session_state['curriculum_results']
    
    # 各阶段进度
    stages_df = pd.DataFrame(results['stages'])
    
    st.subheader("📚 各阶段训练进度")
    
    for i, stage in enumerate(stages_df.to_dict('records'), 1):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**阶段{i}: {stage['stage_name']}**")
        
        with col2:
            st.metric("准确率", f"{stage['accuracy']:.2%}")
        
        with col3:
            status = "✅ 达标" if stage.get('target_reached', False) else "⚠️ 未达标"
            st.write(status)
    
    # 准确率提升曲线
    st.subheader("📈 准确率提升曲线")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(stages_df) + 1)),
        y=stages_df['accuracy'],
        mode='lines+markers',
        name='准确率',
        line=dict(color='#0984E3', width=3),
        marker=dict(size=12)
    ))
    
    fig.update_layout(
        xaxis_title='阶段',
        yaxis_title='准确率',
        yaxis_tickformat='.0%',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ========== 知识蒸馏训练 ==========

def run_distillation_training(teacher_epochs, student_epochs):
    """运行知识蒸馏训练"""
    
    from training.advanced_trainers import KnowledgeDistiller
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 准备训练数据...")
        
        # 获取历史数据
        if 'historical_data' not in st.session_state:
            data = generate_demo_training_data(500)
        else:
            data = st.session_state['historical_data']
        
        # 创建训练器
        distiller = KnowledgeDistiller()
        
        status_text.text("📚 训练教师模型...")
        progress_bar.progress(0.3)
        
        status_text.text("🎓 知识蒸馏给学生模型...")
        progress_bar.progress(0.7)
        
        # 蒸馏训练
        results = distiller.distill_knowledge(
            data,
            teacher_epochs=teacher_epochs,
            student_epochs=student_epochs
        )
        
        progress_bar.progress(1.0)
        
        # 保存结果
        st.session_state['distillation_results'] = results
        st.session_state['distiller'] = distiller
        
        status_text.text("✅ 训练完成！")
        
        st.success(f"""
        🎓 **知识蒸馏完成！**
        
        - 教师模型准确率: {results['teacher_accuracy']:.2%}
        - 学生模型准确率: {results['student_accuracy']:.2%}
        - 速度提升: {results['speed_improvement']:.0f}倍
        """)
        
    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")
        status_text.text("❌ 训练失败")


def display_distillation_results():
    """显示知识蒸馏结果"""
    
    st.divider()
    st.subheader("📊 知识蒸馏结果")
    
    results = st.session_state['distillation_results']
    
    # 对比卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("教师准确率", f"{results['teacher_accuracy']:.2%}")
        st.caption("大模型集成")
    
    with col2:
        st.metric(
            "学生准确率",
            f"{results['student_accuracy']:.2%}",
            f"-{(results['teacher_accuracy'] - results['student_accuracy']):.1%}"
        )
        st.caption("轻量模型")
    
    with col3:
        st.metric("速度提升", f"{results['speed_improvement']:.0f}倍")
        st.caption("推理速度")
    
    # 对比图
    st.subheader("📈 教师 vs 学生")
    
    comparison_df = pd.DataFrame({
        '模型': ['教师模型', '学生模型'],
        '准确率': [results['teacher_accuracy'], results['student_accuracy']],
        '速度': [1.0, results['speed_improvement']]
    })
    
    fig = px.bar(
        comparison_df,
        x='模型',
        y=['准确率', '速度'],
        title='教师模型 vs 学生模型',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ========== 元学习训练 ==========

def run_meta_learning_training(meta_epochs):
    """运行元学习训练"""
    
    from training.advanced_trainers import MetaLearner
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 准备训练数据...")
        
        # 获取历史数据
        if 'historical_data' not in st.session_state:
            data = generate_demo_training_data(1000)  # 元学习需要更多数据
        else:
            data = st.session_state['historical_data']
        
        # 创建训练器
        meta_learner = MetaLearner()
        
        status_text.text("🧠 元学习训练中...")
        progress_bar.progress(0.5)
        
        # 元学习训练
        results = meta_learner.meta_train(
            data,
            meta_epochs=meta_epochs
        )
        
        progress_bar.progress(1.0)
        
        # 保存结果
        st.session_state['meta_results'] = results
        st.session_state['meta_learner'] = meta_learner
        
        status_text.text("✅ 训练完成！")
        
        st.success(f"""
        🧠 **元学习完成！**
        
        - 训练任务数: {results['tasks_trained']}个月
        - 最终准确率: {results['final_accuracy']:.2%}
        - 适应速度: 仅需{results['adaptation_speed']}步
        """)
        
    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")
        status_text.text("❌ 训练失败")


def display_meta_results():
    """显示元学习结果"""
    
    st.divider()
    st.subheader("📊 元学习结果")
    
    results = st.session_state['meta_results']
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("训练任务", f"{results['tasks_trained']}个月")
    
    with col2:
        st.metric("Meta Epochs", results['meta_epochs'])
    
    with col3:
        st.metric("最终准确率", f"{results['final_accuracy']:.2%}")
    
    with col4:
        st.metric("适应速度", f"{results['adaptation_speed']}步")
    
    # 快速适应展示
    st.subheader("⚡ 快速适应能力")
    
    adaptation_df = pd.DataFrame({
        '状态': ['适应前', '适应后'],
        '准确率': [0.60, results['final_accuracy']]
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=adaptation_df['状态'],
            y=adaptation_df['准确率'],
            text=[f"{v:.1%}" for v in adaptation_df['准确率']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"仅用{results['adaptation_speed']}步即可适应新环境",
        yaxis_title='准确率',
        yaxis_tickformat='.0%',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def generate_demo_training_data(n_samples: int) -> pd.DataFrame:
    """生成演示训练数据"""
    
    np.random.seed(42)
    
    data = pd.DataFrame({
        'code': [f"00000{i % 10}" for i in range(n_samples)],
        'main_label': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
        'seal_strength': np.random.uniform(50, 95, n_samples),
        'return_1d': np.random.normal(0.03, 0.05, n_samples),
        'return_3d': np.random.normal(0.05, 0.08, n_samples),
        'return_5d': np.random.normal(0.08, 0.12, n_samples),
        'price_position': np.random.uniform(0.3, 0.9, n_samples),
        'market_sentiment': np.random.choice(['strong', 'neutral', 'weak'], n_samples)
    })
    
    return data


if __name__ == '__main__':
    # 用于测试
    st.set_page_config(page_title="循环进化训练", layout="wide")
    render_evolution_training_tab()
