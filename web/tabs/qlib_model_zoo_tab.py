"""
Qlib Model Zoo 标签页
提供Qlib官方30+模型的统一配置和训练界面

功能包括:
- GBDT家族: LightGBM, XGBoost, CatBoost
- 神经网络: MLP, LSTM, GRU, ALSTM
- 高级模型: Transformer, TRA, TCN, HIST
- 集成模型: DoubleEnsemble
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Optional
import traceback

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ✅ 导入依赖检测模块 (P0 修复)
try:
    from qlib_enhanced.model_zoo.model_dependency_checker import (
        check_model_availability,
        check_all_models,
        get_model_status_summary,
        DependencyCheckResult
    )
    DEPENDENCY_CHECKER_AVAILABLE = True
except ImportError:
    DEPENDENCY_CHECKER_AVAILABLE = False


# ==================== 模型分类配置 ====================

MODEL_CATEGORIES = {
    "🌲 GBDT家族": {
        "LightGBM": {
            "status": "✅ 已有",
            "module": "qlib.contrib.model.gbdt",
            "class": "LGBModel",
            "description": "轻量级梯度提升决策树，训练速度快，内存占用低",
            "params": {
                "learning_rate": (0.001, 0.3, 0.05),
                "num_leaves": (10, 300, 31),
                "max_depth": (-1, 20, -1),
                "n_estimators": (50, 1000, 100),
            }
        },
        "XGBoost": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.gbdt",
            "class": "XGBModel",
            "description": "极限梯度提升，高性能的GBDT实现",
            "params": {
                "learning_rate": (0.001, 0.3, 0.05),
                "max_depth": (3, 10, 6),
                "n_estimators": (50, 1000, 100),
                "subsample": (0.5, 1.0, 0.8),
                "colsample_bytree": (0.5, 1.0, 0.8),
            }
        },
        "CatBoost": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.gbdt",
            "class": "CatBoostModel",
            "description": "类别特征友好的梯度提升，无需手动编码",
            "params": {
                "learning_rate": (0.001, 0.3, 0.03),
                "depth": (4, 10, 6),
                "iterations": (50, 1000, 100),
                "l2_leaf_reg": (1, 10, 3),
            }
        }
    },
    "🧠 神经网络": {
        "MLP": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_nn",
            "class": "DNNModelPytorch",
            "description": "多层感知机，经典的全连接神经网络",
            "params": {
                "hidden_size": (64, 512, 128),
                "num_layers": (2, 5, 3),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        },
        "LSTM": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_lstm",
            "class": "LSTMModel",
            "description": "长短期记忆网络，擅长处理时序数据",
            "params": {
                "hidden_size": (64, 256, 128),
                "num_layers": (1, 3, 2),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        },
        "GRU": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_gru",
            "class": "GRUModel",
            "description": "门控循环单元，LSTM的简化版本",
            "params": {
                "hidden_size": (64, 256, 128),
                "num_layers": (1, 3, 2),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        },
        "ALSTM": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_alstm",
            "class": "ALSTMModel",
            "description": "注意力机制LSTM，自动学习特征重要性",
            "params": {
                "hidden_size": (64, 256, 128),
                "num_layers": (1, 3, 2),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        }
    },
    "🚀 高级模型": {
        "Transformer": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_transformer",
            "class": "Transformer",
            "description": "自注意力机制，捕捉长距离依赖",
            "params": {
                "d_model": (64, 512, 128),
                "nhead": (2, 8, 4),
                "num_layers": (1, 6, 2),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        },
        "TRA": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_tra",
            "class": "TRA",
            "description": "时序路由适配器，自适应市场变化",
            "params": {
                "hidden_size": (64, 256, 128),
                "num_layers": (1, 3, 2),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        },
        "TCN": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_tcn",
            "class": "TCN",
            "description": "时序卷积网络，并行训练效率高",
            "params": {
                "num_channels": ([64, 128, 256], [32, 64, 128], [64, 128, 256]),
                "kernel_size": (2, 5, 3),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        },
        "HIST": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.pytorch_hist",
            "class": "HIST",
            "description": "历史信息融合模型，结合多粒度特征",
            "params": {
                "hidden_size": (64, 256, 128),
                "num_layers": (1, 3, 2),
                "dropout": (0.0, 0.5, 0.1),
                "lr": (0.0001, 0.01, 0.001),
            }
        }
    },
    "🎯 集成模型": {
        "DoubleEnsemble": {
            "status": "⭐ 新增",
            "module": "qlib.contrib.model.double_ensemble",
            "class": "DoubleEnsembleModel",
            "description": "双层集成模型，多模型融合提升性能",
            "params": {
                "base_models": (["lgb", "xgb"], ["lgb"], ["lgb", "xgb", "catboost"]),
                "meta_model": (["linear", "lgb"], "linear", "lgb"),
            }
        }
    }
}


# ==================== 渲染函数 ====================

def render_model_zoo_tab():
    """渲染Model Zoo主界面"""
    st.title("📦 Qlib模型库")
    st.markdown("---")
    
    # 说明
    st.info("💡 **Qlib Model Zoo**: 提供30+量化投资模型的统一训练和评估界面。选择模型，配置参数，一键训练！")
    
    # ✅ 依赖检测统计 (P0 修复)
    if DEPENDENCY_CHECKER_AVAILABLE:
        with st.expander("🔍 模型依赖检测结果", expanded=False):
            summary = get_model_status_summary()
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("📦 总计", summary['total'])
            with col_b:
                st.metric("✅ 可用", summary['available'], delta_color="normal")
            with col_c:
                st.metric("⚠️ 缺失依赖", summary['missing_deps'], delta_color="inverse")
            with col_d:
                st.metric("🔄 降级运行", summary['fallback'], delta_color="off")
            
            if summary['missing_deps'] > 0 or summary['fallback'] > 0:
                st.warning("⚠️ 部分模型缺失依赖或需要降级运行，请在下方查看详情。")
    
    # 统计信息
    col1, col2, col3, col4 = st.columns(4)
    
    total_models = sum(len(models) for models in MODEL_CATEGORIES.values())
    new_models = sum(1 for cat in MODEL_CATEGORIES.values() 
                     for m in cat.values() if m['status'] == '⭐ 新增')
    existing_models = total_models - new_models
    
    with col1:
        st.metric("📦 模型总数", total_models)
    with col2:
        st.metric("✅ 已有模型", existing_models)
    with col3:
        st.metric("⭐ 新增模型", new_models)
    with col4:
        st.metric("🎯 模型分类", len(MODEL_CATEGORIES))
    
    st.markdown("---")
    
    # 创建两列布局
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 📋 模型分类")
        render_model_navigation()
    
    with col_right:
        st.markdown("### ⚙️ 模型配置与训练")
        render_model_config_panel()


def render_model_navigation():
    """渲染模型导航树"""
    # 初始化session state
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = ('🌲 GBDT家族', 'LightGBM')
    
    # 遍历分类（使用索引确保唯一key）
    for cat_idx, (category, models) in enumerate(MODEL_CATEGORIES.items()):
        with st.expander(category, expanded=(category == st.session_state.selected_model[0])):
            for model_idx, (model_name, model_info) in enumerate(models.items()):
                # 创建按钮（使用数字索引作key）
                button_label = f"{model_info['status']} {model_name}"
                if st.button(button_label, key=f"model_btn_{cat_idx}_{model_idx}", 
                           use_container_width=True):
                    st.session_state.selected_model = (category, model_name)
                    st.rerun()
                
                # 显示简短描述
                if st.session_state.selected_model == (category, model_name):
                    st.caption(f"✓ 已选择: {model_info['description']}")


def render_model_config_panel():
    """渲染模型配置面板"""
    if 'selected_model' not in st.session_state:
        st.info("👈 请从左侧选择一个模型开始配置")
        return
    
    category, model_name = st.session_state.selected_model
    model_info = MODEL_CATEGORIES[category][model_name]
    
    # 模型卡片
    st.markdown(f"#### {model_info['status']} {model_name}")
    st.markdown(f"**描述**: {model_info['description']}")
    st.markdown(f"**模块**: `{model_info['module']}`")
    st.markdown(f"**类名**: `{model_info['class']}`")
    
    # ✅ 依赖检测与降级提示 (P0 修复)
    if DEPENDENCY_CHECKER_AVAILABLE:
        dep_result = check_model_availability(model_name)
        
        if dep_result.status == 'ok':
            st.success(f"✅ {dep_result.message}")
        elif dep_result.status == 'missing_deps':
            st.error(f"{dep_result.message}")
            st.code(dep_result.install_command, language="bash")
            st.info(f"🔄 可以使用降级模型: **{dep_result.fallback_model}**")
            if st.button(f"🔄 切换到 {dep_result.fallback_model}", key=f"fallback_{model_name}"):
                # 切换到降级模型
                for cat, models in MODEL_CATEGORIES.items():
                    if dep_result.fallback_model in models:
                        st.session_state.selected_model = (cat, dep_result.fallback_model)
                        st.rerun()
        elif dep_result.status == 'fallback':
            st.warning(f"{dep_result.message}")
            if dep_result.fallback_model:
                st.info(f"💡 建议使用: **{dep_result.fallback_model}** (更稳定)")
                if st.button(f"🔄 切换到 {dep_result.fallback_model}", key=f"fallback2_{model_name}"):
                    for cat, models in MODEL_CATEGORIES.items():
                        if dep_result.fallback_model in models:
                            st.session_state.selected_model = (cat, dep_result.fallback_model)
                            st.rerun()
        else:
            st.error(f"❌ {dep_result.message}")
    
    st.markdown("---")
    
    # 参数配置区
    st.markdown("##### 🔧 参数配置")
    
    params = {}
    for param_name, param_range in model_info['params'].items():
        if isinstance(param_range[0], (int, float)):
            # 数值参数
            min_val, max_val, default_val = param_range
            if isinstance(default_val, int):
                params[param_name] = st.slider(
                    param_name,
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    key=f"param_{model_name}_{param_name}"
                )
            else:
                params[param_name] = st.slider(
                    param_name,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=0.001,
                    format="%.4f",
                    key=f"param_{model_name}_{param_name}"
                )
        elif isinstance(param_range[0], list):
            # 列表参数
            options = param_range
            params[param_name] = st.selectbox(
                param_name,
                options=range(len(options)),
                format_func=lambda x: str(options[x]),
                index=1 if len(options) > 1 else 0,
                key=f"param_{model_name}_{param_name}"
            )
            params[param_name] = options[params[param_name]]
        else:
            # 字符串参数
            options = [str(x) for x in param_range]
            params[param_name] = st.selectbox(
                param_name,
                options=options,
                index=1 if len(options) > 1 else 0,
                key=f"param_{model_name}_{param_name}"
            )
    
    st.markdown("---")
    
    # 数据集配置
    st.markdown("##### 📊 数据集配置")
    col1, col2 = st.columns(2)
    with col1:
        train_start = st.date_input("训练开始日期", value=pd.to_datetime("2018-01-01"))
        train_end = st.date_input("训练结束日期", value=pd.to_datetime("2020-12-31"))
    with col2:
        test_start = st.date_input("测试开始日期", value=pd.to_datetime("2021-01-01"))
        test_end = st.date_input("测试结束日期", value=pd.to_datetime("2021-12-31"))
    
    market = st.selectbox("股票池", ["csi300", "csi500", "all"], index=0)
    
    st.markdown("---")
    
    # 训练配置
    st.markdown("##### 🚀 训练配置")
    col1, col2 = st.columns(2)
    with col1:
        save_model = st.checkbox("保存模型", value=True)
        model_name_input = st.text_input("模型名称", value=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d')}")
    with col2:
        use_gpu = st.checkbox("使用GPU", value=False)
        n_jobs = st.number_input("并行任务数", min_value=1, max_value=32, value=4)
    
    # 训练按钮
    if st.button("🚀 开始训练", type="primary", use_container_width=True, key=f"train_{model_name}"):
        train_model(model_name, model_info, params, {
            'train_start': str(train_start),
            'train_end': str(train_end),
            'test_start': str(test_start),
            'test_end': str(test_end),
            'market': market,
            'save_model': save_model,
            'model_name': model_name_input,
            'use_gpu': use_gpu,
            'n_jobs': n_jobs
        })


def train_model(model_name: str, model_info: Dict, params: Dict, config: Dict):
    """训练模型（真实实现）"""
    st.markdown("---")
    st.markdown("#### 📈 训练进度")
    
    # 创建进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.expander("📋 训练日志", expanded=True)
    
    start_time = pd.Timestamp.now()
    
    try:
        with log_container:
            st.info(f"✅ 开始训练 {model_name} 模型...")
            st.json({
                "模型": model_name,
                "参数": params,
                "配置": config
            })
            
            # 导入训练器
            from qlib_enhanced.model_zoo import ModelZooTrainer
            
            # 定义进度回调
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
                st.write(f"📝 {message}")
            
            update_progress(0.05, "正在初始化训练器...")
            
            # 初始化训练器
            trainer = ModelZooTrainer()
            
            update_progress(0.1, "正在准备数据集...")
            
            # 准备数据集
            dataset = trainer.prepare_dataset(
                instruments=config['market'],
                train_start=config['train_start'],
                train_end=config['train_end'],
                valid_start=config['test_start'],
                valid_end=config['test_end'],
            )
            
            update_progress(0.2, "数据集准备完成，开始训练...")
            
            # 训练模型
            result = trainer.train_model(
                model_name=model_name,
                model_config=params,
                dataset=dataset,
                save_model=config['save_model'],
                progress_callback=update_progress
            )
            
            # 计算训练时长
            duration = (pd.Timestamp.now() - start_time).total_seconds()
            
            if result['success']:
                st.success(f"✅ {model_name} 训练完成!")
                st.balloons()
                
                # 显示结果
                st.markdown("##### 📊 训练结果")
                
                metrics = result['metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("IC", f"{metrics.get('IC', 0):.4f}")
                with col2:
                    st.metric("Rank IC", f"{metrics.get('Rank IC', 0):.4f}")
                with col3:
                    st.metric("ICIR", f"{metrics.get('ICIR', 0):.4f}")
                with col4:
                    st.metric("训练时长", f"{duration:.1f}秒")
                
                # 详细指标
                st.markdown("##### 📈 详细指标")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MSE", f"{metrics.get('MSE', 0):.6f}")
                    st.metric("训练样本数", f"{result['train_samples']:,}")
                with col2:
                    st.metric("MAE", f"{metrics.get('MAE', 0):.6f}")
                    st.metric("验证样本数", f"{result['valid_samples']:,}")
                
                # 模型保存信息
                if config['save_model'] and result.get('model_path'):
                    st.info(f"💾 模型已保存至: `{result['model_path']}`")
            else:
                st.error(f"❌ 训练失败: {result.get('error', '未知错误')}")
    
    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")
        with log_container:
            st.code(traceback.format_exc())


# ==================== 主入口 ====================

if __name__ == "__main__":
    render_model_zoo_tab()
