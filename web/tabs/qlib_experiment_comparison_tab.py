"""
Qlib实验对比功能
支持多实验选择、指标对比、可视化分析和统计检验
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)

# Qlib导入
try:
    import qlib
    from qlib.workflow import R
    from qlib.constant import REG_CN
    QLIB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qlib导入失败: {e}")
    QLIB_AVAILABLE = False


def render_qlib_experiment_comparison_tab():
    """渲染Qlib实验对比页面"""
    st.header("🔬 实验对比分析")
    
    if not QLIB_AVAILABLE:
        st.error("❌ Qlib未安装或导入失败")
        st.info("请先安装Qlib: `pip install pyqlib`")
        return
    
    st.markdown("""
    **实验对比分析**工具帮助您：
    - 📊 横向对比多个实验的性能指标
    - 📈 可视化对比训练曲线和回测结果
    - 🔍 参数差异分析和影响评估
    - 📉 统计显著性检验和相关性分析
    - 🏆 智能排名和模型选择建议
    """)
    
    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 实验选择",
        "📊 指标对比",
        "📈 可视化分析",
        "🔬 统计分析"
    ])
    
    with tab1:
        render_experiment_selector()
    
    with tab2:
        render_metrics_comparison()
    
    with tab3:
        render_visualization_comparison()
    
    with tab4:
        render_statistical_analysis()


def render_experiment_selector():
    """渲染实验选择器"""
    st.subheader("📋 选择要对比的实验")
    
    # 获取所有可用实验
    try:
        all_experiments = get_all_experiments()
        
        if not all_experiments:
            st.warning("暂无可用实验，请先在'Qlib工作流'中运行实验")
            return
        
        st.success(f"✅ 找到 {len(all_experiments)} 个实验")
        
        # 显示实验列表
        st.markdown("### 📜 可用实验列表")
        
        # 创建实验表格
        exp_data = []
        for exp_name, exp_info in all_experiments.items():
            exp_data.append({
                "实验名称": exp_name,
                "记录数": exp_info.get('n_recorders', 0),
                "创建时间": exp_info.get('create_time', 'N/A'),
                "状态": exp_info.get('status', 'unknown')
            })
        
        exp_df = pd.DataFrame(exp_data)
        st.dataframe(exp_df, use_container_width=True)
        
        # 多选实验
        st.markdown("### ✅ 选择对比实验")
        
        selected_experiments = st.multiselect(
            "选择2-10个实验进行对比",
            options=list(all_experiments.keys()),
            default=list(all_experiments.keys())[:min(3, len(all_experiments))],
            help="选择要对比的实验（建议2-5个）"
        )
        
        if len(selected_experiments) < 2:
            st.warning("⚠️ 请至少选择2个实验进行对比")
            return
        
        if len(selected_experiments) > 10:
            st.warning("⚠️ 选择实验过多可能影响性能，建议不超过10个")
        
        # 保存选择
        st.session_state['selected_experiments'] = selected_experiments
        
        st.success(f"✅ 已选择 {len(selected_experiments)} 个实验进行对比")
        
        # 加载实验数据按钮
        if st.button("🔄 加载实验数据", type="primary", use_container_width=True):
            with st.spinner("正在加载实验数据..."):
                load_experiment_data(selected_experiments)
        
        # 显示已加载的实验数据摘要
        if 'experiment_data' in st.session_state and st.session_state['experiment_data']:
            st.markdown("### 📊 已加载数据摘要")
            
            for exp_name in selected_experiments:
                if exp_name in st.session_state['experiment_data']:
                    exp_data = st.session_state['experiment_data'][exp_name]
                    
                    with st.expander(f"🔬 {exp_name}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("记录数", len(exp_data.get('recorders', {})))
                        
                        with col2:
                            metrics = exp_data.get('metrics', {})
                            st.metric("指标数", len(metrics) if metrics else 0)
                        
                        with col3:
                            params = exp_data.get('params', {})
                            st.metric("参数数", len(params) if params else 0)
        
    except Exception as e:
        st.error(f"❌ 加载实验列表失败: {e}")
        logger.error(f"加载实验列表失败: {e}", exc_info=True)


def render_metrics_comparison():
    """渲染指标对比表格"""
    st.subheader("📊 性能指标对比")
    
    if 'experiment_data' not in st.session_state or not st.session_state['experiment_data']:
        st.info("请先在'实验选择'标签中加载实验数据")
        return
    
    experiment_data = st.session_state['experiment_data']
    
    # 选择对比维度
    comparison_type = st.radio(
        "对比维度",
        ["预测性能指标", "回测收益指标", "风险指标", "全部指标"],
        horizontal=True
    )
    
    # 构建对比表格
    comparison_df = build_comparison_table(experiment_data, comparison_type)
    
    if comparison_df.empty:
        st.warning("暂无可对比的指标数据")
        return
    
    # 显示对比表格
    st.markdown("### 📋 指标对比表")
    
    # 添加高亮最佳值的功能
    highlight_best = st.checkbox("高亮最佳值", value=True)
    
    if highlight_best:
        # 对每个指标行进行高亮（数值越大越好或越小越好）
        styled_df = style_comparison_table(comparison_df)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(comparison_df, use_container_width=True)
    
    # 下载对比表格
    csv = comparison_df.to_csv(index=True).encode('utf-8-sig')
    st.download_button(
        label="📥 下载对比表格",
        data=csv,
        file_name=f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # 参数差异分析
    st.markdown("### ⚙️ 参数差异分析")
    render_parameter_diff(experiment_data)


def render_visualization_comparison():
    """渲染可视化对比分析"""
    st.subheader("📈 可视化对比分析")
    
    if 'experiment_data' not in st.session_state or not st.session_state['experiment_data']:
        st.info("请先在'实验选择'标签中加载实验数据")
        return
    
    experiment_data = st.session_state['experiment_data']
    
    # 图表类型选择
    chart_type = st.selectbox(
        "选择图表类型",
        [
            "指标雷达图",
            "收益率对比",
            "净值曲线对比",
            "回撤对比",
            "收益分布对比",
            "IC/ICIR对比",
            "参数敏感性分析"
        ]
    )
    
    if chart_type == "指标雷达图":
        render_radar_chart(experiment_data)
    
    elif chart_type == "收益率对比":
        render_returns_comparison(experiment_data)
    
    elif chart_type == "净值曲线对比":
        render_equity_curves_comparison(experiment_data)
    
    elif chart_type == "回撤对比":
        render_drawdown_comparison(experiment_data)
    
    elif chart_type == "收益分布对比":
        render_returns_distribution(experiment_data)
    
    elif chart_type == "IC/ICIR对比":
        render_ic_comparison(experiment_data)
    
    elif chart_type == "参数敏感性分析":
        render_parameter_sensitivity(experiment_data)


def render_statistical_analysis():
    """渲染统计分析"""
    st.subheader("🔬 统计分析")
    
    if 'experiment_data' not in st.session_state or not st.session_state['experiment_data']:
        st.info("请先在'实验选择'标签中加载实验数据")
        return
    
    experiment_data = st.session_state['experiment_data']
    
    # 分析类型选择
    analysis_type = st.selectbox(
        "选择分析类型",
        [
            "统计显著性检验",
            "相关性分析",
            "排名和评分",
            "稳定性分析"
        ]
    )
    
    if analysis_type == "统计显著性检验":
        render_significance_test(experiment_data)
    
    elif analysis_type == "相关性分析":
        render_correlation_analysis(experiment_data)
    
    elif analysis_type == "排名和评分":
        render_ranking_analysis(experiment_data)
    
    elif analysis_type == "稳定性分析":
        render_stability_analysis(experiment_data)


# ============================================================================
# 辅助函数
# ============================================================================

def get_all_experiments() -> Dict[str, Dict[str, Any]]:
    """获取所有可用实验"""
    try:
        experiments = {}
        
        # 方法1: 从session state获取
        if 'workflow_executions' in st.session_state:
            for execution in st.session_state['workflow_executions']:
                exp_name = execution.get('experiment_name')
                if exp_name:
                    experiments[exp_name] = {
                        'create_time': execution.get('timestamp', 'N/A'),
                        'status': execution.get('status', 'unknown'),
                        'n_recorders': 1
                    }
        
        # 方法2: 从MLflow目录扫描
        try:
            mlruns_dir = Path("mlruns")
            if mlruns_dir.exists():
                for exp_dir in mlruns_dir.iterdir():
                    if exp_dir.is_dir() and exp_dir.name.isdigit():
                        # 尝试读取实验元数据
                        meta_file = exp_dir / "meta.yaml"
                        if meta_file.exists():
                            try:
                                import yaml
                                with open(meta_file, 'r') as f:
                                    meta = yaml.safe_load(f)
                                    exp_name = meta.get('name', exp_dir.name)
                                    
                                    if exp_name not in experiments:
                                        # 统计recorder数量
                                        n_recorders = len([d for d in exp_dir.iterdir() 
                                                         if d.is_dir() and len(d.name) == 32])
                                        
                                        experiments[exp_name] = {
                                            'create_time': datetime.fromtimestamp(
                                                exp_dir.stat().st_mtime
                                            ).strftime('%Y-%m-%d %H:%M:%S'),
                                            'status': 'completed',
                                            'n_recorders': n_recorders
                                        }
                            except Exception as e:
                                logger.debug(f"读取实验元数据失败: {e}")
        except Exception as e:
            logger.debug(f"扫描MLflow目录失败: {e}")
        
        return experiments
        
    except Exception as e:
        logger.error(f"获取实验列表失败: {e}", exc_info=True)
        return {}


def load_experiment_data(experiment_names: List[str]):
    """加载实验数据"""
    try:
        experiment_data = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, exp_name in enumerate(experiment_names):
            status_text.text(f"正在加载: {exp_name} ({i+1}/{len(experiment_names)})")
            
            try:
                # 从MLflow加载
                exp = R.get_exp(experiment_name=exp_name, create=False)
                recorders = exp.list_recorders()
                
                if not recorders:
                    logger.warning(f"实验 {exp_name} 无记录")
                    continue
                
                # 获取最新的recorder（假设第一个是最新的）
                recorder = list(recorders.values())[0]
                
                # 提取指标和参数
                metrics = {}
                params = {}
                
                try:
                    metrics = recorder.list_metrics()
                except Exception as e:
                    logger.debug(f"提取指标失败: {e}")
                
                try:
                    params = recorder.list_params()
                except Exception as e:
                    logger.debug(f"提取参数失败: {e}")
                
                experiment_data[exp_name] = {
                    'recorders': recorders,
                    'recorder': recorder,
                    'metrics': metrics,
                    'params': params,
                    'status': recorder.status if hasattr(recorder, 'status') else 'unknown'
                }
                
            except Exception as e:
                logger.error(f"加载实验 {exp_name} 失败: {e}")
                st.warning(f"⚠️ 加载实验 {exp_name} 失败: {e}")
            
            progress_bar.progress((i + 1) / len(experiment_names))
        
        progress_bar.empty()
        status_text.empty()
        
        # 保存到session state
        st.session_state['experiment_data'] = experiment_data
        
        st.success(f"✅ 成功加载 {len(experiment_data)} 个实验的数据")
        
    except Exception as e:
        st.error(f"❌ 加载实验数据失败: {e}")
        logger.error(f"加载实验数据失败: {e}", exc_info=True)


def build_comparison_table(experiment_data: Dict, comparison_type: str) -> pd.DataFrame:
    """构建对比表格"""
    try:
        # 定义指标分类
        prediction_metrics = ['IC', 'ICIR', 'Rank IC', 'Rank ICIR']
        backtest_metrics = ['累计收益率', '年化收益率', '夏普比率', '最大回撤', '胜率']
        risk_metrics = ['波动率', '最大回撤', 'VaR', 'CVaR', '下行波动率']
        
        # 收集所有指标
        all_metrics = set()
        for exp_data in experiment_data.values():
            metrics = exp_data.get('metrics', {})
            all_metrics.update(metrics.keys())
        
        # 根据对比类型筛选指标
        if comparison_type == "预测性能指标":
            selected_metrics = [m for m in all_metrics if any(pm in m for pm in prediction_metrics)]
        elif comparison_type == "回测收益指标":
            selected_metrics = [m for m in all_metrics if any(bm in m for bm in backtest_metrics)]
        elif comparison_type == "风险指标":
            selected_metrics = [m for m in all_metrics if any(rm in m for rm in risk_metrics)]
        else:  # 全部指标
            selected_metrics = list(all_metrics)
        
        if not selected_metrics:
            return pd.DataFrame()
        
        # 构建表格
        table_data = {}
        for exp_name, exp_data in experiment_data.items():
            metrics = exp_data.get('metrics', {})
            table_data[exp_name] = {metric: metrics.get(metric, np.nan) 
                                   for metric in selected_metrics}
        
        df = pd.DataFrame(table_data)
        
        # 转置，使实验名称为列
        df = df.T
        
        # 排序指标列
        df = df[sorted(df.columns)]
        
        return df
        
    except Exception as e:
        logger.error(f"构建对比表格失败: {e}", exc_info=True)
        return pd.DataFrame()


def style_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """样式化对比表格，高亮最佳值"""
    try:
        # 定义指标方向（True=越大越好，False=越小越好）
        metric_directions = {
            'IC': True,
            'ICIR': True,
            'Rank IC': True,
            'Rank ICIR': True,
            '累计收益率': True,
            '年化收益率': True,
            '夏普比率': True,
            '胜率': True,
            '最大回撤': False,
            '波动率': False,
            'VaR': False,
            'CVaR': False,
            '下行波动率': False
        }
        
        def highlight_best(s):
            """高亮最佳值"""
            if s.name not in df.columns:
                return [''] * len(s)
            
            # 判断方向
            is_higher_better = True
            for metric_name, direction in metric_directions.items():
                if metric_name in s.name:
                    is_higher_better = direction
                    break
            
            # 找到最佳值
            if is_higher_better:
                best_idx = s.idxmax()
            else:
                best_idx = s.idxmin()
            
            # 创建样式
            return ['background-color: #90EE90' if idx == best_idx else '' 
                   for idx in s.index]
        
        # 应用样式
        styled = df.style.apply(highlight_best, axis=0)
        
        # 格式化数值
        styled = styled.format("{:.4f}", na_rep="-")
        
        return styled
        
    except Exception as e:
        logger.error(f"样式化表格失败: {e}", exc_info=True)
        return df


def render_parameter_diff(experiment_data: Dict):
    """渲染参数差异分析"""
    try:
        # 收集所有参数
        all_params = {}
        for exp_name, exp_data in experiment_data.items():
            params = exp_data.get('params', {})
            all_params[exp_name] = params
        
        if not all_params:
            st.info("暂无参数数据")
            return
        
        # 找出有差异的参数
        param_keys = set()
        for params in all_params.values():
            param_keys.update(params.keys())
        
        diff_params = {}
        for key in param_keys:
            values = [all_params[exp].get(key, None) for exp in all_params.keys()]
            # 如果不是所有值都相同
            if len(set(str(v) for v in values)) > 1:
                diff_params[key] = {exp: all_params[exp].get(key, 'N/A') 
                                   for exp in all_params.keys()}
        
        if not diff_params:
            st.info("✅ 所有实验的参数完全相同")
            return
        
        st.warning(f"⚠️ 发现 {len(diff_params)} 个参数存在差异")
        
        # 显示差异参数表格
        diff_df = pd.DataFrame(diff_params).T
        st.dataframe(diff_df, use_container_width=True)
        
    except Exception as e:
        logger.error(f"参数差异分析失败: {e}", exc_info=True)
        st.error(f"❌ 参数差异分析失败: {e}")


def render_radar_chart(experiment_data: Dict):
    """渲染雷达图"""
    try:
        st.markdown("### 🎯 关键指标雷达图")
        
        # 选择要展示的指标
        default_metrics = ['IC', 'ICIR', '夏普比率', '年化收益率']
        
        all_metrics = set()
        for exp_data in experiment_data.values():
            all_metrics.update(exp_data.get('metrics', {}).keys())
        
        selected_metrics = st.multiselect(
            "选择指标",
            options=sorted(all_metrics),
            default=[m for m in default_metrics if m in all_metrics]
        )
        
        if not selected_metrics:
            st.warning("请至少选择一个指标")
            return
        
        # 构建雷达图数据
        fig = go.Figure()
        
        for exp_name, exp_data in experiment_data.items():
            metrics = exp_data.get('metrics', {})
            values = [metrics.get(m, 0) for m in selected_metrics]
            
            # 归一化到0-1
            max_vals = [max(abs(experiment_data[e].get('metrics', {}).get(m, 0)) 
                           for e in experiment_data.keys()) for m in selected_metrics]
            normalized_values = [v / max_v if max_v != 0 else 0 
                                for v, max_v in zip(values, max_vals)]
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],  # 闭合
                theta=selected_metrics + [selected_metrics[0]],
                fill='toself',
                name=exp_name
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="关键指标对比（归一化）"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"雷达图渲染失败: {e}", exc_info=True)
        st.error(f"❌ 雷达图渲染失败: {e}")


def render_returns_comparison(experiment_data: Dict):
    """渲染收益率对比柱状图"""
    try:
        st.markdown("### 📊 收益率对比")
        
        # 收集收益率数据
        return_metrics = ['累计收益率', '年化收益率', '最大回撤']
        
        data_dict = {metric: [] for metric in return_metrics}
        exp_names = []
        
        for exp_name, exp_data in experiment_data.items():
            metrics = exp_data.get('metrics', {})
            exp_names.append(exp_name)
            
            for metric in return_metrics:
                value = metrics.get(metric, 0)
                # 如果是百分比格式的字符串，转换为数值
                if isinstance(value, str) and '%' in value:
                    value = float(value.replace('%', '')) / 100
                data_dict[metric].append(value)
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=len(return_metrics),
            subplot_titles=return_metrics
        )
        
        colors = px.colors.qualitative.Plotly
        
        for i, metric in enumerate(return_metrics, 1):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=exp_names,
                    y=data_dict[metric],
                    marker_color=colors[i-1],
                    showlegend=False
                ),
                row=1, col=i
            )
        
        fig.update_layout(height=400, title_text="收益率对比分析")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"收益率对比渲染失败: {e}", exc_info=True)
        st.error(f"❌ 收益率对比渲染失败: {e}")


def render_equity_curves_comparison(experiment_data: Dict):
    """渲染净值曲线对比"""
    st.info("💡 净值曲线对比功能需要完整的回测数据，当前从MLflow获取的数据可能不包含时序数据")
    st.markdown("如需查看净值曲线，请前往'Qlib回测'标签运行完整回测")


def render_drawdown_comparison(experiment_data: Dict):
    """渲染回撤对比"""
    st.info("💡 回撤曲线对比功能需要完整的回测数据，当前从MLflow获取的数据可能不包含时序数据")
    st.markdown("如需查看回撤曲线，请前往'Qlib回测'标签运行完整回测")


def render_returns_distribution(experiment_data: Dict):
    """渲染收益分布对比"""
    st.info("💡 收益分布对比功能需要完整的回测数据，当前从MLflow获取的数据可能不包含时序数据")
    st.markdown("如需查看收益分布，请前往'Qlib回测'标签运行完整回测")


def render_ic_comparison(experiment_data: Dict):
    """渲染IC/ICIR对比"""
    try:
        st.markdown("### 📈 IC/ICIR对比")
        
        # 收集IC和ICIR数据
        ic_data = []
        icir_data = []
        exp_names = []
        
        for exp_name, exp_data in experiment_data.items():
            metrics = exp_data.get('metrics', {})
            exp_names.append(exp_name)
            
            ic_data.append(metrics.get('IC', 0))
            icir_data.append(metrics.get('ICIR', 0))
        
        # 创建对比图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("IC对比", "ICIR对比")
        )
        
        # IC柱状图
        fig.add_trace(
            go.Bar(name='IC', x=exp_names, y=ic_data, marker_color='lightblue'),
            row=1, col=1
        )
        
        # ICIR柱状图
        fig.add_trace(
            go.Bar(name='ICIR', x=exp_names, y=icir_data, marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="IC和ICIR对比"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 添加说明
        st.markdown("""
        **指标说明**:
        - **IC (Information Coefficient)**: 预测值与真实收益的相关系数，越高越好
        - **ICIR (IC/IR)**: IC的标准化版本，考虑了稳定性
        """)
        
    except Exception as e:
        logger.error(f"IC/ICIR对比渲染失败: {e}", exc_info=True)
        st.error(f"❌ IC/ICIR对比渲染失败: {e}")


def render_parameter_sensitivity(experiment_data: Dict):
    """渲染参数敏感性分析"""
    st.markdown("### 🔍 参数敏感性分析")
    st.info("💡 参数敏感性分析需要多组实验系统地改变某个参数，当前功能待增强")


def render_significance_test(experiment_data: Dict):
    """渲染统计显著性检验"""
    try:
        st.markdown("### 📊 统计显著性检验")
        st.markdown("检验不同实验的性能差异是否具有统计显著性")
        
        # 选择要检验的指标
        all_metrics = set()
        for exp_data in experiment_data.values():
            all_metrics.update(exp_data.get('metrics', {}).keys())
        
        test_metric = st.selectbox(
            "选择要检验的指标",
            options=sorted(all_metrics)
        )
        
        if not test_metric:
            return
        
        # 收集数据
        exp_names = list(experiment_data.keys())
        values = [experiment_data[exp].get('metrics', {}).get(test_metric, 0) 
                 for exp in exp_names]
        
        # 显示数据
        st.markdown("#### 📋 数据概览")
        summary_df = pd.DataFrame({
            '实验名称': exp_names,
            test_metric: values
        })
        st.dataframe(summary_df, use_container_width=True)
        
        # 进行t检验（两两比较）
        if len(exp_names) >= 2:
            st.markdown("#### 🔬 两两t检验结果")
            
            # 注意：这里简化了，实际应该用每个实验的多次运行数据
            # 当前只有单个值，所以t检验不适用，改用简单比较
            st.info("💡 当前仅有单次运行结果，无法进行严格的统计检验")
            st.markdown("**简单排名**:")
            
            ranked_df = summary_df.sort_values(test_metric, ascending=False)
            ranked_df['排名'] = range(1, len(ranked_df) + 1)
            st.dataframe(ranked_df[['排名', '实验名称', test_metric]], use_container_width=True)
            
    except Exception as e:
        logger.error(f"统计检验失败: {e}", exc_info=True)
        st.error(f"❌ 统计检验失败: {e}")


def render_correlation_analysis(experiment_data: Dict):
    """渲染相关性分析"""
    try:
        st.markdown("### 🔗 指标相关性分析")
        
        # 构建指标矩阵
        metrics_matrix = {}
        
        for exp_name, exp_data in experiment_data.items():
            metrics = exp_data.get('metrics', {})
            for metric_name, value in metrics.items():
                if metric_name not in metrics_matrix:
                    metrics_matrix[metric_name] = []
                metrics_matrix[metric_name].append(value)
        
        # 转换为DataFrame
        df = pd.DataFrame(metrics_matrix)
        
        # 计算相关矩阵
        corr_matrix = df.corr()
        
        # 绘制热力图
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="相关系数"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        
        fig.update_layout(
            title="指标相关性热力图",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示高相关性对
        st.markdown("#### 🔍 高相关性指标对（|r| > 0.7）")
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        '指标1': corr_matrix.columns[i],
                        '指标2': corr_matrix.columns[j],
                        '相关系数': f"{corr_val:.3f}"
                    })
        
        if high_corr_pairs:
            st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
        else:
            st.info("未发现高度相关的指标对")
        
    except Exception as e:
        logger.error(f"相关性分析失败: {e}", exc_info=True)
        st.error(f"❌ 相关性分析失败: {e}")


def render_ranking_analysis(experiment_data: Dict):
    """渲染排名和评分分析"""
    try:
        st.markdown("### 🏆 实验排名和综合评分")
        
        # 定义评分权重
        st.markdown("#### ⚙️ 设置评分权重")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight_ic = st.slider("IC权重", 0.0, 1.0, 0.3, 0.05)
            weight_icir = st.slider("ICIR权重", 0.0, 1.0, 0.2, 0.05)
            weight_return = st.slider("年化收益率权重", 0.0, 1.0, 0.3, 0.05)
        
        with col2:
            weight_sharpe = st.slider("夏普比率权重", 0.0, 1.0, 0.2, 0.05)
            weight_drawdown = st.slider("最大回撤权重（负向）", 0.0, 1.0, -0.1, 0.05)
        
        # 计算综合得分
        scores = []
        
        for exp_name, exp_data in experiment_data.items():
            metrics = exp_data.get('metrics', {})
            
            score = 0.0
            score += weight_ic * metrics.get('IC', 0)
            score += weight_icir * metrics.get('ICIR', 0)
            score += weight_return * metrics.get('年化收益率', 0)
            score += weight_sharpe * metrics.get('夏普比率', 0)
            score += weight_drawdown * metrics.get('最大回撤', 0)
            
            scores.append({
                '实验名称': exp_name,
                '综合得分': score,
                'IC': metrics.get('IC', 0),
                'ICIR': metrics.get('ICIR', 0),
                '年化收益率': metrics.get('年化收益率', 0),
                '夏普比率': metrics.get('夏普比率', 0),
                '最大回撤': metrics.get('最大回撤', 0)
            })
        
        # 排序
        scores_df = pd.DataFrame(scores).sort_values('综合得分', ascending=False)
        scores_df['排名'] = range(1, len(scores_df) + 1)
        
        # 重新排列列顺序
        scores_df = scores_df[['排名', '实验名称', '综合得分', 'IC', 'ICIR', 
                              '年化收益率', '夏普比率', '最大回撤']]
        
        st.markdown("#### 📊 排名结果")
        st.dataframe(scores_df, use_container_width=True)
        
        # 可视化
        fig = go.Figure(data=[
            go.Bar(
                x=scores_df['实验名称'],
                y=scores_df['综合得分'],
                marker_color=px.colors.sequential.Viridis,
                text=scores_df['综合得分'].round(3),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="综合得分排名",
            xaxis_title="实验名称",
            yaxis_title="综合得分",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 推荐最佳模型
        st.markdown("#### 🥇 推荐模型")
        best_exp = scores_df.iloc[0]
        st.success(f"""
        **最佳实验**: {best_exp['实验名称']}  
        **综合得分**: {best_exp['综合得分']:.3f}  
        **IC**: {best_exp['IC']:.4f} | **ICIR**: {best_exp['ICIR']:.4f}  
        **年化收益率**: {best_exp['年化收益率']:.2%} | **夏普比率**: {best_exp['夏普比率']:.3f}
        """)
        
    except Exception as e:
        logger.error(f"排名分析失败: {e}", exc_info=True)
        st.error(f"❌ 排名分析失败: {e}")


def render_stability_analysis(experiment_data: Dict):
    """渲染稳定性分析"""
    st.markdown("### 📉 稳定性分析")
    st.info("💡 稳定性分析需要每个实验的多次运行数据或时序数据，当前功能待增强")
    st.markdown("""
    **建议**:
    - 对同一模型进行多次训练（不同随机种子）
    - 使用滚动窗口回测评估时间稳定性
    - 对比不同市场环境下的表现
    """)
