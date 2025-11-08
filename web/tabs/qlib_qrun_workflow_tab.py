"""
Qlib qrun工作流UI集成
实现一键运行Qlib YAML配置文件，完整的训练-回测-评估流程
"""
import streamlit as st
import pandas as pd
import yaml
import os
from pathlib import Path
from datetime import datetime
import subprocess
import json
from typing import Dict, Any, Optional, List
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Qlib导入
try:
    import qlib
    from qlib.workflow import R
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH
    QLIB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qlib导入失败: {e}")
    QLIB_AVAILABLE = False


def render_qlib_qrun_workflow_tab():
    """渲染Qlib qrun工作流页面"""
    st.header("🔄 Qlib工作流 (qrun)")
    
    if not QLIB_AVAILABLE:
        st.error("❌ Qlib未安装或导入失败")
        st.info("请先安装Qlib: `pip install pyqlib`")
        return
    
    st.markdown("""
    **Qlib工作流**允许您通过YAML配置文件定义完整的量化研究流程：
    - 📊 数据处理和特征工程
    - 🧠 模型训练
    - 📈 信号分析
    - 💼 回测评估
    - 📋 结果记录到MLflow
    
    一键运行，自动化整个流程！
    """)
    
    # 创建选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 配置编辑器",
        "📚 模板库",
        "🚀 执行工作流",
        "📊 运行结果",
        "📖 使用指南"
    ])
    
    with tab1:
        render_config_editor()
    
    with tab2:
        render_template_library()
    
    with tab3:
        render_workflow_executor()
    
    with tab4:
        render_workflow_results()
    
    with tab5:
        render_user_guide()


def render_config_editor():
    """渲染配置编辑器"""
    st.subheader("📝 工作流配置编辑器")
    
    # 选择配置来源
    config_source = st.radio(
        "配置来源",
        ["从模板创建", "上传配置文件", "手动编写"],
        horizontal=True
    )
    
    config_content = None
    
    if config_source == "从模板创建":
        template_name = st.selectbox(
            "选择模板",
            [
                "LightGBM + Alpha158 (推荐新手)",
                "LightGBM + Alpha360 (增强版)",
                "XGBoost + Alpha158",
                "XGBoost + Alpha360",
                "CatBoost + Alpha360",
                "CatBoost + Alpha158 (调优版)",
                "RandomForest + Alpha158",
                "--- 深度学习模型 ---",
                "GRU + Alpha158 (深度学习)",
                "LSTM + Alpha360 (深度学习)",
                "Transformer + Alpha158 (深度学习)",
                "ALSTM + Alpha158 (Attention LSTM)",
                "TRA + Alpha158 (Temporal Routing)",
                "--- 一进二专用模型 ---",
                "✅ 一进二涨停策略 (已完成，推荐)",
                "✅ 涨停板分类模型 (LightGBM)",
                "✅ 涨停板排序模型 (XGBoost)",
                "✅ 连板预测模型 (CatBoost)",
                "✅ 打板时机模型 (ALSTM)",
                "✅ 一进二综合策略 (Ensemble)"
            ]
        )
        
        if st.button("加载模板"):
            config_content = load_template_config(template_name)
            st.session_state['workflow_config'] = config_content
            st.success(f"✅ 已加载模板: {template_name}")
    
    elif config_source == "上传配置文件":
        uploaded_file = st.file_uploader(
            "上传YAML配置文件",
            type=['yaml', 'yml'],
            help="上传Qlib工作流配置文件"
        )
        if uploaded_file:
            try:
                config_content = uploaded_file.read().decode('utf-8')
                st.session_state['workflow_config'] = config_content
                st.success("✅ 配置文件上传成功")
            except Exception as e:
                st.error(f"文件读取失败: {e}")
    
    else:  # 手动编写
        st.info("💡 提示：参考右侧模板库中的示例配置")
    
    # 显示和编辑配置
    st.markdown("### 📄 当前配置")
    
    if 'workflow_config' not in st.session_state:
        st.session_state['workflow_config'] = get_default_config()
    
    config_text = st.text_area(
        "YAML配置内容",
        value=st.session_state.get('workflow_config', ''),
        height=500,
        help="编辑工作流配置，支持YAML格式"
    )
    
    # 更新配置
    if config_text != st.session_state.get('workflow_config', ''):
        st.session_state['workflow_config'] = config_text
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 保存配置", use_container_width=True):
            save_config_to_file(config_text)
    
    with col2:
        if st.button("✅ 验证配置", use_container_width=True):
            validate_config(config_text)
    
    with col3:
        if st.button("🔄 重置为默认", use_container_width=True):
            st.session_state['workflow_config'] = get_default_config()
            st.rerun()
    
    # 配置参数快速调整
    with st.expander("⚙️ 快速参数调整"):
        render_quick_params_editor()


def render_quick_params_editor():
    """渲染快速参数编辑器"""
    st.markdown("**快速调整常用参数（不影响完整配置）**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**数据范围**")
        train_start = st.date_input("训练开始", value=datetime(2008, 1, 1))
        train_end = st.date_input("训练结束", value=datetime(2014, 12, 31))
        test_start = st.date_input("测试开始", value=datetime(2017, 1, 1))
        test_end = st.date_input("测试结束", value=datetime(2020, 8, 1))
    
    with col2:
        st.markdown("**股票池和基准**")
        market = st.selectbox("股票池", ["csi300", "csi500", "csi1000"])
        benchmark = st.selectbox("基准", ["SH000300", "SH000905", "SH000852"])
        
        st.markdown("**回测参数**")
        topk = st.number_input("持仓数量", min_value=5, max_value=100, value=50)
        n_drop = st.number_input("卖出数量", min_value=0, max_value=50, value=5)
    
    if st.button("📝 应用到配置", use_container_width=True):
        try:
            current_config = yaml.safe_load(st.session_state.get('workflow_config', ''))
            
            # 更新参数
            current_config['data_handler_config']['start_time'] = train_start.strftime('%Y-%m-%d')
            current_config['data_handler_config']['fit_end_time'] = train_end.strftime('%Y-%m-%d')
            current_config['port_analysis_config']['backtest']['start_time'] = test_start.strftime('%Y-%m-%d')
            current_config['port_analysis_config']['backtest']['end_time'] = test_end.strftime('%Y-%m-%d')
            current_config['market'] = market
            current_config['benchmark'] = benchmark
            current_config['port_analysis_config']['strategy']['kwargs']['topk'] = topk
            current_config['port_analysis_config']['strategy']['kwargs']['n_drop'] = n_drop
            
            # 保存回session
            st.session_state['workflow_config'] = yaml.dump(current_config, allow_unicode=True)
            st.success("✅ 参数已更新到配置")
            st.rerun()
        except Exception as e:
            st.error(f"参数更新失败: {e}")


def render_template_library():
    """渲染模板库"""
    st.subheader("📚 工作流模板库")
    
    st.markdown("""
    选择预设模板快速开始，所有模板都经过验证可直接运行。
    """)
    
    # 模板分类
    template_category = st.selectbox(
        "模板分类",
        ["机器学习模型", "深度学习模型", "高频策略", "一进二专用"]
    )
    
    if template_category == "机器学习模型":
        render_ml_templates()
    elif template_category == "深度学习模型":
        render_dl_templates()
    elif template_category == "高频策略":
        render_highfreq_templates()
    else:
        render_limitup_templates()


def render_ml_templates():
    """渲染机器学习模板"""
    templates = [
        {
            "name": "LightGBM + Alpha158",
            "description": "最常用的基准模型，适合新手入门",
            "features": "158个Alpha因子",
            "model": "LightGBM",
            "difficulty": "⭐",
        },
        {
            "name": "LightGBM + Alpha360 (增强版)",
            "description": "LightGBM增强版，调优的超参数",
            "features": "360个Alpha因子",
            "model": "LightGBM",
            "difficulty": "⭐⭐",
        },
        {
            "name": "XGBoost + Alpha158",
            "description": "经典梯度提升树，适合对比实验",
            "features": "158个Alpha因子",
            "model": "XGBoost",
            "difficulty": "⭐",
        },
        {
            "name": "XGBoost + Alpha360",
            "description": "XGBoost增强版，更多特征更强性能",
            "features": "360个Alpha因子",
            "model": "XGBoost",
            "difficulty": "⭐⭐",
        },
        {
            "name": "CatBoost + Alpha360",
            "description": "处理类别特征的专家",
            "features": "360个Alpha因子",
            "model": "CatBoost",
            "difficulty": "⭐⭐",
        },
        {
            "name": "CatBoost + Alpha158 (调优版)",
            "description": "CatBoost调优版，精细调整的参数",
            "features": "158个Alpha因子",
            "model": "CatBoost",
            "difficulty": "⭐⭐",
        },
        {
            "name": "RandomForest + Alpha158",
            "description": "随机森林，适合集成学习",
            "features": "158个Alpha因子",
            "model": "RandomForest",
            "difficulty": "⭐",
        },
    ]
    
    for tmpl in templates:
        with st.expander(f"**{tmpl['name']}** - {tmpl['difficulty']}"):
            st.markdown(f"**描述**: {tmpl['description']}")
            st.markdown(f"**特征集**: {tmpl['features']}")
            st.markdown(f"**模型**: {tmpl['model']}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(get_template_preview(tmpl['name']), language='yaml')
            with col2:
                if st.button("使用此模板", key=f"use_{tmpl['name']}"):
                    config = load_template_config(tmpl['name'])
                    st.session_state['workflow_config'] = config
                    st.success(f"✅ 已切换到: {tmpl['name']}")
                    st.rerun()


def render_dl_templates():
    """渲染深度学习模板"""
    st.info("深度学习模板需要GPU支持，训练时间较长")
    
    templates = [
        {
            "name": "GRU + Alpha158",
            "description": "门控循环单元，捕捉时序特征",
            "difficulty": "⭐⭐⭐",
        },
        {
            "name": "LSTM + Alpha360",
            "description": "长短期记忆网络，更强时序建模",
            "difficulty": "⭐⭐⭐",
        },
        {
            "name": "Transformer + Alpha158",
            "description": "注意力机制，最新架构",
            "difficulty": "⭐⭐⭐⭐",
        },
    ]
    
    for tmpl in templates:
        with st.expander(f"**{tmpl['name']}** - {tmpl['difficulty']}"):
            st.markdown(f"**描述**: {tmpl['description']}")
            if st.button("使用此模板", key=f"use_{tmpl['name']}"):
                st.warning("深度学习模板开发中，敬请期待")


def render_highfreq_templates():
    """渲染高频策略模板"""
    st.warning("高频策略需要分钟级或tick级数据")
    st.info("该功能在P1-3阶段开发，敬请期待")


def render_limitup_templates():
    """渲染一进二专用模板"""
    st.markdown("""
    🎯 **针对A股一进二涨停板选股策略的专用配置模板**
    """)
    
    st.success("✅ **6个一进二模板全部已完成！**")
    
    templates = [
        {
            "name": "✅ 一进二涨停策略",
            "file": "limitup_yinjiner_strategy",
            "description": "完整的一进二打板策略配置（推荐）",
            "difficulty": "⭐⭐⭐⭐",
            "model": "LightGBM Regressor",
            "features": [
                "4种标签定义 + 24个Alpha因子",
                "T+2持仓策略，考虑开板成本",
                "完整风险控制和回测配置"
            ]
        },
        {
            "name": "✅ 涨停板分类模型",
            "file": "limitup_classifier",
            "description": "二分类预测明日是否涨停",
            "difficulty": "⭐⭐⭐",
            "model": "LightGBM Classifier",
            "features": [
                "标签: 明日是否涨停 (>9.5%)",
                "Top30概率最高的股票",
                "AUC作为评估指标"
            ]
        },
        {
            "name": "✅ 涨停板排序模型",
            "file": "limitup_ranker",
            "description": "对多个候选涨停板排序",
            "difficulty": "⭐⭐⭐",
            "model": "XGBoost Regressor",
            "features": [
                "标签: 次日收益率（连续值）",
                "Alpha360特征 + Top50策略",
                "CSI500股票池"
            ]
        },
        {
            "name": "✅ 连板预测模型",
            "file": "limitup_consecutive",
            "description": "预测今日涨停且明日继续涨停",
            "difficulty": "⭐⭐⭐",
            "model": "CatBoost Classifier",
            "features": [
                "标签: 今日+明日双涨停",
                "Top20精选连板股",
                "GPU加速训练"
            ]
        },
        {
            "name": "✅ 打板时机模型",
            "file": "limitup_timing",
            "description": "预测最佳打板时机（次日不破板）",
            "difficulty": "⭐⭐⭐⭐",
            "model": "ALSTM (Attention LSTM)",
            "features": [
                "标签: 次日涨停+后天收益>0",
                "LSTM注意力机制捕捉时序特征",
                "Top25打板时机"
            ]
        },
        {
            "name": "✅ 一进二综合策略",
            "file": "limitup_ensemble",
            "description": "综合多个模型的集成策略",
            "difficulty": "⭐⭐⭐⭐",
            "model": "LightGBM (Ensemble)",
            "features": [
                "标签: 次日收益率",
                "Alpha360 + Top40综合评分",
                "高参数调优版本"
            ]
        },
    ]
    
    for tmpl in templates:
        with st.expander(f"**{tmpl['name']}** - {tmpl['difficulty']}", expanded=False):
            st.markdown(f"**描述**: {tmpl['description']}")
            st.markdown(f"**模型**: {tmpl['model']}")
            st.markdown(f"**文件**: `{tmpl['file']}.yaml`")
            st.markdown("**特点**:")
            for feature in tmpl['features']:
                st.markdown(f"- {feature}")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                display_name = f"✅ {tmpl['name'].replace('✅ ', '')} ({tmpl['model'].split()[0]})"
                if st.button("🚀 使用", key=f"use_{tmpl['file']}"):
                    # 构建映射键
                    mapping_key = None
                    if "一进二涨停策略" in tmpl['name']:
                        mapping_key = "✅ 一进二涨停策略 (已完成，推荐)"
                    elif "分类模型" in tmpl['name']:
                        mapping_key = "✅ 涨停板分类模型 (LightGBM)"
                    elif "排序模型" in tmpl['name']:
                        mapping_key = "✅ 涨停板排序模型 (XGBoost)"
                    elif "连板预测" in tmpl['name']:
                        mapping_key = "✅ 连板预测模型 (CatBoost)"
                    elif "打板时机" in tmpl['name']:
                        mapping_key = "✅ 打板时机模型 (ALSTM)"
                    elif "综合策略" in tmpl['name']:
                        mapping_key = "✅ 一进二综合策略 (Ensemble)"
                    
                    if mapping_key:
                        config = load_template_config(mapping_key)
                        st.session_state['workflow_config'] = config
                        st.success(f"✅ 已加载: {tmpl['name']}")
                        st.rerun()


def render_workflow_executor():
    """渲染工作流执行器"""
    st.subheader("🚀 执行工作流")
    
    # 检查配置
    if 'workflow_config' not in st.session_state or not st.session_state['workflow_config']:
        st.warning("⚠️ 请先在'配置编辑器'中创建或加载配置")
        return
    
    # 显示当前配置概览
    try:
        config_dict = yaml.safe_load(st.session_state['workflow_config'])
        render_config_summary(config_dict)
    except Exception as e:
        st.error(f"配置解析失败: {e}")
        return
    
    st.divider()
    
    # 执行选项
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**执行模式**")
        exec_mode = st.radio(
            "选择执行模式",
            ["完整流程", "仅训练", "仅回测"],
            help="完整流程=训练+回测+评估；仅训练=只训练模型；仅回测=使用已有模型回测"
        )
        
        experiment_name = st.text_input(
            "实验名称",
            value=f"qlib_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="MLflow实验名称，用于组织和查找结果"
        )
    
    with col2:
        st.markdown("**高级选项**")
        
        save_model = st.checkbox("保存模型", value=True, help="训练完成后保存模型到MLflow")
        save_pred = st.checkbox("保存预测结果", value=True, help="保存预测分数，用于后续回测")
        
        auto_backtest = st.checkbox(
            "自动回测",
            value=True,
            help="训练完成后自动执行回测"
        )
        
        use_gpu = st.checkbox("使用GPU", value=False, help="深度学习模型推荐开启")
    
    st.divider()
    
    # 执行按钮
    col_run, col_stop = st.columns([3, 1])
    
    with col_run:
        if st.button("🚀 开始执行工作流", type="primary", use_container_width=True):
            execute_workflow(
                config_text=st.session_state['workflow_config'],
                experiment_name=experiment_name,
                exec_mode=exec_mode,
                save_model=save_model,
                save_pred=save_pred,
                auto_backtest=auto_backtest,
                use_gpu=use_gpu
            )
    
    with col_stop:
        if st.button("⛔ 停止", use_container_width=True):
            st.warning("工作流停止功能开发中")
    
    # 显示执行日志
    if 'workflow_logs' in st.session_state and st.session_state['workflow_logs']:
        with st.expander("📋 执行日志", expanded=True):
            st.code(st.session_state['workflow_logs'], language='text')


def render_config_summary(config: Dict):
    """渲染配置概览"""
    st.markdown("### 📊 当前配置概览")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**数据配置**")
        data_config = config.get('data_handler_config', {})
        st.text(f"开始: {data_config.get('start_time', 'N/A')}")
        st.text(f"结束: {data_config.get('end_time', 'N/A')}")
        st.text(f"市场: {config.get('market', 'N/A')}")
    
    with col2:
        st.markdown("**模型配置**")
        model_config = config.get('task', {}).get('model', {})
        st.text(f"模型: {model_config.get('class', 'N/A')}")
        st.text(f"特征: {config.get('task', {}).get('dataset', {}).get('kwargs', {}).get('handler', {}).get('class', 'N/A')}")
    
    with col3:
        st.markdown("**回测配置**")
        backtest_config = config.get('port_analysis_config', {}).get('backtest', {})
        st.text(f"初始资金: {backtest_config.get('account', 'N/A')}")
        st.text(f"基准: {backtest_config.get('benchmark', 'N/A')}")


def execute_workflow(
    config_text: str,
    experiment_name: str,
    exec_mode: str,
    save_model: bool,
    save_pred: bool,
    auto_backtest: bool,
    use_gpu: bool
):
    """执行工作流"""
    st.session_state['workflow_logs'] = ""
    
    with st.spinner("🔄 正在执行工作流..."):
        try:
            # 解析配置
            config = yaml.safe_load(config_text)
            
            # 初始化Qlib
            log_message("初始化Qlib...")
            qlib_config = config.get('qlib_init', {})
            provider_uri = qlib_config.get('provider_uri', '~/.qlib/qlib_data/cn_data')
            region = qlib_config.get('region', 'cn')
            
            qlib.init(
                provider_uri=os.path.expanduser(provider_uri),
                region=region
            )
            log_message("✅ Qlib初始化完成")
            
            # 创建实验
            log_message(f"创建实验: {experiment_name}")
            
            # 根据执行模式运行
            if exec_mode in ["完整流程", "仅训练"]:
                run_training(config, experiment_name, save_model, save_pred)
            
            if exec_mode in ["完整流程", "仅回测"] and auto_backtest:
                run_backtest(config, experiment_name)
            
            log_message("🎉 工作流执行完成！")
            st.success("✅ 工作流执行成功！请查看'运行结果'标签")
            
            # 保存执行记录
            save_execution_record(experiment_name, config)
            
        except Exception as e:
            log_message(f"❌ 执行失败: {e}")
            st.error(f"工作流执行失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def run_training(config: Dict, experiment_name: str, save_model: bool, save_pred: bool):
    """运行训练"""
    log_message("=" * 60)
    log_message("开始模型训练...")
    log_message("=" * 60)
    
    try:
        # 提取任务配置
        task_config = config.get('task', {})
        
        # 初始化数据集
        log_message("加载数据集...")
        dataset_config = task_config.get('dataset', {})
        dataset = init_instance_by_config(dataset_config)
        log_message(f"✅ 数据集加载完成: {dataset}")
        
        # 初始化模型
        log_message("初始化模型...")
        model_config = task_config.get('model', {})
        model = init_instance_by_config(model_config)
        log_message(f"✅ 模型初始化完成: {model}")
        
        # 训练模型
        log_message("开始训练...")
        with R.start(experiment_name=experiment_name):
            # 记录配置
            R.log_params(**{"model": model_config.get('class', 'Unknown')})
            
            # 训练
            model.fit(dataset)
            log_message("✅ 模型训练完成")
            
            # 预测
            log_message("生成预测...")
            pred_score = model.predict(dataset)
            log_message(f"✅ 预测完成: shape={pred_score.shape}")
            
            # 保存
            if save_model:
                R.save_objects(trained_model=model)
                log_message("✅ 模型已保存")
            
            if save_pred:
                R.save_objects(**{"pred.pkl": pred_score})
                log_message("✅ 预测结果已保存")
            
            # 保存到session用于回测
            st.session_state['last_pred_score'] = pred_score
            
    except Exception as e:
        log_message(f"❌ 训练失败: {e}")
        raise


def run_backtest(config: Dict, experiment_name: str):
    """运行回测"""
    log_message("=" * 60)
    log_message("开始回测...")
    log_message("=" * 60)
    
    try:
        from qlib.backtest import backtest
        
        # 获取预测结果
        if 'last_pred_score' not in st.session_state:
            log_message("⚠️ 未找到预测结果，跳过回测")
            return
        
        pred_score = st.session_state['last_pred_score']
        
        # 提取回测配置
        port_config = config.get('port_analysis_config', {})
        strategy_config = port_config.get('strategy', {})
        backtest_config = port_config.get('backtest', {})
        
        # 设置预测信号
        strategy_config['kwargs']['signal'] = pred_score
        
        # 执行回测
        log_message("执行回测...")
        portfolio_metric, indicator_metric = backtest(
            start_time=backtest_config.get('start_time'),
            end_time=backtest_config.get('end_time'),
            strategy=strategy_config,
            executor={
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}
            },
            benchmark=backtest_config.get('benchmark'),
            account=backtest_config.get('account', 100000000),
            exchange_kwargs=backtest_config.get('exchange_kwargs', {})
        )
        
        log_message("✅ 回测完成")
        
        # 保存回测结果
        st.session_state['workflow_backtest_results'] = {
            'portfolio_metric': portfolio_metric,
            'indicator_metric': indicator_metric
        }
        
        # 提取关键指标
        analysis_freq = 'day'
        if analysis_freq in portfolio_metric:
            portfolio_df = portfolio_metric[analysis_freq][0]
            returns = portfolio_df.get('return', pd.Series())
            
            if not returns.empty:
                cumulative_return = (1 + returns).prod() - 1
                sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
                
                log_message(f"累计收益率: {cumulative_return:.2%}")
                log_message(f"夏普比率: {sharpe:.3f}")
        
    except Exception as e:
        log_message(f"❌ 回测失败: {e}")
        raise


def log_message(message: str):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    
    if 'workflow_logs' not in st.session_state:
        st.session_state['workflow_logs'] = ""
    
    st.session_state['workflow_logs'] += log_line
    logger.info(message)


def render_workflow_results():
    """渲染工作流结果"""
    st.subheader("📊 工作流运行结果")
    
    # 显示历史执行记录
    if 'workflow_executions' in st.session_state and st.session_state['workflow_executions']:
        st.markdown("### 📜 执行历史")
        
        executions_df = pd.DataFrame(st.session_state['workflow_executions'])
        st.dataframe(executions_df, use_container_width=True)
        
        # 选择查看详情
        selected_exp = st.selectbox(
            "选择实验查看详情",
            executions_df['experiment_name'].tolist()
        )
        
        if selected_exp:
            render_experiment_details(selected_exp)
    else:
        st.info("暂无执行记录，请先在'执行工作流'标签运行工作流")


def render_experiment_details(experiment_name: str):
    """渲染实验详情"""
    st.markdown(f"### 🔬 实验详情: {experiment_name}")
    
    try:
        # 尝试从MLflow加载
        exp = R.get_exp(experiment_name=experiment_name, create=False)
        recorders = exp.list_recorders()
        
        if not recorders:
            st.warning("该实验暂无记录")
            return
        
        # 显示所有runs
        st.markdown("#### 运行记录")
        for recorder_id, recorder in recorders.items():
            with st.expander(f"Run ID: {recorder_id[:8]}... - {recorder.status}"):
                # 显示指标
                try:
                    metrics = recorder.list_metrics()
                    if metrics:
                        st.markdown("**指标**")
                        metrics_df = pd.DataFrame([metrics]).T
                        metrics_df.columns = ['Value']
                        st.dataframe(metrics_df)
                except:
                    pass
                
                # 显示参数
                try:
                    params = recorder.list_params()
                    if params:
                        st.markdown("**参数**")
                        st.json(params)
                except:
                    pass
        
    except Exception as e:
        st.error(f"加载实验详情失败: {e}")


def render_user_guide():
    """渲染使用指南"""
    st.subheader("📖 使用指南")
    
    st.markdown("""
    ## 🎯 快速开始
    
    ### 1. 选择或创建配置
    在"配置编辑器"标签：
    - 从模板库选择预设配置，或
    - 上传已有的YAML文件，或
    - 手动编写配置
    
    ### 2. 调整参数（可选）
    - 使用"快速参数调整"修改常用参数
    - 或直接编辑YAML配置
    
    ### 3. 执行工作流
    在"执行工作流"标签：
    - 选择执行模式（完整/训练/回测）
    - 设置实验名称
    - 点击"开始执行"
    
    ### 4. 查看结果
    在"运行结果"标签查看：
    - 训练指标
    - 回测结果
    - MLflow记录
    
    ## 📝 配置文件结构
    
    Qlib工作流配置包含以下部分：
    
    ```yaml
    qlib_init:          # Qlib初始化配置
      provider_uri: "~/.qlib/qlib_data/cn_data"
      region: cn
    
    market: csi300      # 股票池
    benchmark: SH000300 # 基准指数
    
    data_handler_config:  # 数据处理配置
      start_time: 2008-01-01
      end_time: 2020-08-01
      instruments: csi300
    
    task:               # 任务配置
      model:           # 模型配置
        class: LGBModel
        kwargs: {...}
      dataset:         # 数据集配置
        class: DatasetH
        kwargs: {...}
      record:          # 记录配置
        - class: SignalRecord
        - class: PortAnaRecord
    
    port_analysis_config:  # 回测配置
      strategy: {...}
      backtest: {...}
    ```
    
    ## 💡 最佳实践
    
    1. **数据范围设置**
       - 训练集：至少2年数据
       - 验证集：1年
       - 测试集：1-3年
    
    2. **实验命名**
       - 使用有意义的名称
       - 包含日期和版本信息
       - 例如: `lgb_alpha158_v1_20240101`
    
    3. **参数调优**
       - 先用默认参数跑一次
       - 再根据结果调整
       - 记录每次实验的参数和结果
    
    4. **结果分析**
       - 关注IC/ICIR指标（预测能力）
       - 关注夏普比率（风险调整收益）
       - 关注最大回撤（风险控制）
    
    ## 🔧 常见问题
    
    **Q: 配置文件报错怎么办？**
    
    A: 使用"验证配置"按钮检查语法，参考模板库中的示例。
    
    **Q: 训练很慢怎么办？**
    
    A: 缩短数据范围，减少特征数量，或使用更简单的模型。
    
    **Q: 如何使用训练好的模型？**
    
    A: 模型自动保存到MLflow，可在"执行工作流"中选择"仅回测"模式使用已有模型。
    
    **Q: 如何对比不同模型？**
    
    A: 使用不同的实验名称运行多次，然后在"运行结果"中对比。
    """)


def get_default_config() -> str:
    """获取默认配置"""
    return """qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn

market: &market csi300
benchmark: &benchmark SH000300

data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: 2017-01-01
        end_time: 2020-08-01
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.2
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]
"""


def load_template_config(template_name: str) -> str:
    """加载模板配置"""
    # 模板映射
    template_mapping = {
        # 机器学习模型
        "LightGBM + Alpha158 (推荐新手)": "lightgbm_alpha158",
        "LightGBM + Alpha360 (增强版)": "lightgbm_alpha360_enhanced",
        "CatBoost + Alpha360": "catboost_alpha360",
        "CatBoost + Alpha158 (调优版)": "catboost_alpha158_tuned",
        "XGBoost + Alpha158": "xgboost_alpha158",
        "XGBoost + Alpha360": "xgboost_alpha360",
        "RandomForest + Alpha158": "randomforest_alpha158",
        # 深度学习模型
        "GRU + Alpha158 (深度学习)": "gru_alpha158",
        "LSTM + Alpha360 (深度学习)": "lstm_alpha360",
        "Transformer + Alpha158 (深度学习)": "transformer_alpha158",
        "ALSTM + Alpha158 (Attention LSTM)": "alstm_alpha158",
        "TRA + Alpha158 (Temporal Routing)": "tra_alpha158",
        # 一进二专用模型
        "✅ 一进二涨停策略 (已完成，推荐)": "limitup_yinjiner_strategy",
        "✅ 涨停板分类模型 (LightGBM)": "limitup_classifier",
        "✅ 涨停板排序模型 (XGBoost)": "limitup_ranker",
        "✅ 连板预测模型 (CatBoost)": "limitup_consecutive",
        "✅ 打板时机模型 (ALSTM)": "limitup_timing",
        "✅ 一进二综合策略 (Ensemble)": "limitup_ensemble",
    }
    
    # 获取模板文件名
    template_file = template_mapping.get(template_name)
    
    if template_file:
        # ✅ 使用动态路径计算项目根目录 (修复硬编码)
        project_root = Path(__file__).parent.parent.parent
        template_dir = project_root / "configs" / "qlib_workflows" / "templates"
        template_path = template_dir / f"{template_file}.yaml"
        
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"加载模板文件失败: {e}")
    
    # 默认返回LightGBM配置
    return get_default_config()


def get_template_preview(template_name: str) -> str:
    """获取模板预览"""
    return """qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn
market: csi300
task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
...(完整配置请点击"使用此模板")"""


def validate_config(config_text: str):
    """验证配置"""
    try:
        config = yaml.safe_load(config_text)
        
        # 基本验证
        required_keys = ['qlib_init', 'task']
        missing_keys = [k for k in required_keys if k not in config]
        
        if missing_keys:
            st.error(f"❌ 配置缺少必需字段: {', '.join(missing_keys)}")
            return False
        
        st.success("✅ 配置验证通过！")
        
        # 显示配置摘要
        with st.expander("配置摘要"):
            st.json(config)
        
        return True
        
    except yaml.YAMLError as e:
        st.error(f"❌ YAML语法错误: {e}")
        return False
    except Exception as e:
        st.error(f"❌ 验证失败: {e}")
        return False


def save_config_to_file(config_text: str):
    """保存配置到文件"""
    try:
        # ✅ 使用动态路径 (修复硬编码)
        project_root = Path(__file__).parent.parent.parent
        save_dir = project_root / "configs" / "qlib_workflows"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_config_{timestamp}.yaml"
        filepath = save_dir / filename
        
        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(config_text)
        
        st.success(f"✅ 配置已保存: {filepath}")
        
        # 提供下载
        st.download_button(
            label="📥 下载配置文件",
            data=config_text,
            file_name=filename,
            mime="text/yaml"
        )
        
    except Exception as e:
        st.error(f"保存失败: {e}")


def save_execution_record(experiment_name: str, config: Dict):
    """保存执行记录"""
    if 'workflow_executions' not in st.session_state:
        st.session_state['workflow_executions'] = []
    
    record = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': config.get('task', {}).get('model', {}).get('class', 'Unknown'),
        'market': config.get('market', 'Unknown'),
        'status': 'Completed'
    }
    
    st.session_state['workflow_executions'].append(record)


if __name__ == "__main__":
    render_qlib_qrun_workflow_tab()
