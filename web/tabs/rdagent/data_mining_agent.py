"""
Data Mining Agent - 数据质量检测和分析模块

功能:
1. 数据质量检测
2. 缺失值分析
3. 异常值检测
4. 数据分布可视化
5. 特征相关性分析
6. 数据报告生成
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime


class DataMiningAgent:
    """数据挖掘Agent"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """初始化session状态"""
        if 'dm_uploaded_data' not in st.session_state:
            st.session_state.dm_uploaded_data = None
        if 'dm_analysis_results' not in st.session_state:
            st.session_state.dm_analysis_results = {}
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """分析缺失值"""
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / len(df)) * 100
        
        missing_info = []
        for col in df.columns:
            if missing_counts[col] > 0:
                missing_info.append({
                    'column': col,
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_pcts[col]),
                    'dtype': str(df[col].dtype)
                })
        
        return {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': len([c for c in missing_counts if c > 0]),
            'details': sorted(missing_info, key=lambda x: x['percentage'], reverse=True)
        }
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict:
        """检测异常值"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(df)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'min_outlier': float(df[col][outlier_mask].min()),
                        'max_outlier': float(df[col][outlier_mask].max())
                    }
            
            elif method == 'zscore':
                z_scores = np.abs((data - data.mean()) / data.std())
                outlier_mask = z_scores > 3
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(data)) * 100),
                        'method': 'Z-Score > 3'
                    }
        
        return outliers
    
    def analyze_data_distribution(self, df: pd.DataFrame) -> Dict:
        """分析数据分布"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        distributions = {
            'numeric': {},
            'categorical': {}
        }
        
        # 数值型特征
        for col in numeric_cols:
            data = df[col].dropna()
            distributions['numeric'][col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            }
        
        # 类别型特征
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            distributions['categorical'][col] = {
                'unique_count': int(df[col].nunique()),
                'top_values': value_counts.head(10).to_dict(),
                'is_high_cardinality': df[col].nunique() > 50
            }
        
        return distributions
    
    def calculate_correlation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """计算特征相关性"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame(), []
        
        corr_matrix = numeric_df.corr()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # 相关性阈值
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        
        return corr_matrix, high_corr_pairs
    
    def generate_quality_score(self, df: pd.DataFrame, missing_info: Dict, outliers: Dict) -> Dict:
        """生成数据质量评分"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = missing_info['total_missing']
        missing_ratio = missing_cells / total_cells
        
        total_outliers = sum(v['count'] for v in outliers.values())
        outlier_ratio = total_outliers / (df.shape[0] * len(df.select_dtypes(include=[np.number]).columns)) if len(df.select_dtypes(include=[np.number]).columns) > 0 else 0
        
        # 计算质量评分 (0-100)
        completeness_score = (1 - missing_ratio) * 100
        validity_score = (1 - outlier_ratio) * 100
        
        overall_score = (completeness_score * 0.6 + validity_score * 0.4)
        
        return {
            'overall_score': float(overall_score),
            'completeness_score': float(completeness_score),
            'validity_score': float(validity_score),
            'grade': 'A' if overall_score >= 90 else 'B' if overall_score >= 80 else 'C' if overall_score >= 70 else 'D'
        }
    
    def render_data_upload(self):
        """渲染数据上传界面"""
        st.subheader("📁 数据上传")
        
        uploaded_file = st.file_uploader(
            "上传CSV文件进行分析",
            type=['csv'],
            help="支持CSV格式,文件大小不超过200MB"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dm_uploaded_data = df
                
                st.success(f"✅ 文件上传成功: {uploaded_file.name}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总行数", f"{len(df):,}")
                with col2:
                    st.metric("总列数", f"{len(df.columns):,}")
                with col3:
                    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                    st.metric("数值列", numeric_cols)
                with col4:
                    categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
                    st.metric("类别列", categorical_cols)
                
                # 数据预览
                with st.expander("🔍 数据预览", expanded=True):
                    st.dataframe(df.head(20), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 文件读取失败: {str(e)}")
        
        elif st.session_state.dm_uploaded_data is not None:
            df = st.session_state.dm_uploaded_data
            st.info(f"当前数据: {len(df)} 行 × {len(df.columns)} 列")
    
    def render_quality_overview(self):
        """渲染数据质量总览"""
        if st.session_state.dm_uploaded_data is None:
            st.warning("请先上传数据文件")
            return
        
        df = st.session_state.dm_uploaded_data
        
        st.subheader("📊 数据质量总览")
        
        with st.spinner("分析中..."):
            # 分析
            missing_info = self.analyze_missing_values(df)
            outliers = self.detect_outliers(df, method='iqr')
            quality_score = self.generate_quality_score(df, missing_info, outliers)
            
            # 保存结果
            st.session_state.dm_analysis_results = {
                'missing_info': missing_info,
                'outliers': outliers,
                'quality_score': quality_score
            }
        
        # 质量评分卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = quality_score['overall_score']
            st.metric(
                "总体质量评分",
                f"{score:.1f}",
                delta=f"等级: {quality_score['grade']}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "完整性",
                f"{quality_score['completeness_score']:.1f}",
                delta=f"{missing_info['total_missing']:,} 缺失值"
            )
        
        with col3:
            st.metric(
                "有效性",
                f"{quality_score['validity_score']:.1f}",
                delta=f"{sum(v['count'] for v in outliers.values()):,} 异常值"
            )
        
        with col4:
            st.metric(
                "问题列数",
                missing_info['columns_with_missing'] + len(outliers),
                delta=f"共{len(df.columns)}列"
            )
        
        # 质量评分可视化
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score['overall_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "数据质量评分", 'font': {'size': 24}},
            delta={'reference': 80, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccc'},
                    {'range': [50, 70], 'color': '#ffffcc'},
                    {'range': [70, 90], 'color': '#ccffcc'},
                    {'range': [90, 100], 'color': '#99ff99'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_missing_analysis(self):
        """渲染缺失值分析"""
        if 'missing_info' not in st.session_state.dm_analysis_results:
            st.warning("请先进行数据质量分析")
            return
        
        st.subheader("🔍 缺失值分析")
        
        missing_info = st.session_state.dm_analysis_results['missing_info']
        
        if missing_info['total_missing'] == 0:
            st.success("✅ 数据完整,无缺失值!")
            return
        
        st.warning(f"⚠️ 发现 {missing_info['total_missing']:,} 个缺失值,涉及 {missing_info['columns_with_missing']} 列")
        
        # 缺失值详情表
        if missing_info['details']:
            df_missing = pd.DataFrame(missing_info['details'])
            
            st.dataframe(
                df_missing,
                column_config={
                    "column": st.column_config.TextColumn("列名"),
                    "count": st.column_config.NumberColumn("缺失数量", format="%d"),
                    "percentage": st.column_config.ProgressColumn(
                        "缺失比例",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "dtype": st.column_config.TextColumn("数据类型")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # 缺失值可视化
            fig = px.bar(
                df_missing,
                x='column',
                y='percentage',
                title="各列缺失值比例",
                labels={'column': '列名', 'percentage': '缺失比例 (%)'},
                color='percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # 处理建议
            st.subheader("💡 处理建议")
            for item in missing_info['details'][:5]:  # 显示前5个
                if item['percentage'] > 50:
                    st.warning(f"📌 **{item['column']}**: 缺失率{item['percentage']:.1f}%,建议删除该列")
                elif item['percentage'] > 20:
                    st.info(f"📌 **{item['column']}**: 缺失率{item['percentage']:.1f}%,建议谨慎填充或使用模型预测")
                else:
                    st.success(f"📌 **{item['column']}**: 缺失率{item['percentage']:.1f}%,可使用均值/中位数/众数填充")
    
    def render_outlier_analysis(self):
        """渲染异常值分析"""
        if 'outliers' not in st.session_state.dm_analysis_results:
            st.warning("请先进行数据质量分析")
            return
        
        st.subheader("🚨 异常值检测")
        
        outliers = st.session_state.dm_analysis_results['outliers']
        
        if not outliers:
            st.success("✅ 未检测到显著异常值!")
            return
        
        total_outliers = sum(v['count'] for v in outliers.values())
        st.warning(f"⚠️ 检测到 {total_outliers:,} 个异常值,涉及 {len(outliers)} 列")
        
        # 异常值详情
        outlier_data = []
        for col, info in outliers.items():
            outlier_data.append({
                'column': col,
                'count': info['count'],
                'percentage': info['percentage'],
                'lower_bound': info.get('lower_bound', 'N/A'),
                'upper_bound': info.get('upper_bound', 'N/A')
            })
        
        df_outliers = pd.DataFrame(outlier_data)
        
        st.dataframe(
            df_outliers,
            column_config={
                "column": st.column_config.TextColumn("列名"),
                "count": st.column_config.NumberColumn("异常值数量", format="%d"),
                "percentage": st.column_config.ProgressColumn(
                    "异常比例",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                ),
                "lower_bound": st.column_config.NumberColumn("下界", format="%.2f"),
                "upper_bound": st.column_config.NumberColumn("上界", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 异常值可视化 (选择一列)
        selected_col = st.selectbox("选择列查看详情", list(outliers.keys()))
        
        if selected_col:
            df = st.session_state.dm_uploaded_data
            data = df[selected_col].dropna()
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("数据分布", "箱线图")
            )
            
            # 直方图
            fig.add_trace(
                go.Histogram(x=data, name="分布", nbinsx=50),
                row=1, col=1
            )
            
            # 箱线图
            fig.add_trace(
                go.Box(y=data, name="箱线图"),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_distribution_analysis(self):
        """渲染分布分析"""
        if st.session_state.dm_uploaded_data is None:
            st.warning("请先上传数据文件")
            return
        
        st.subheader("📈 数据分布分析")
        
        df = st.session_state.dm_uploaded_data
        
        # 选择分析类型
        analysis_type = st.radio(
            "选择分析类型",
            ["数值型特征", "类别型特征"],
            horizontal=True
        )
        
        if analysis_type == "数值型特征":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.info("数据中没有数值型特征")
                return
            
            selected_col = st.selectbox("选择特征", numeric_cols)
            
            if selected_col:
                data = df[selected_col].dropna()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("均值", f"{data.mean():.2f}")
                with col2:
                    st.metric("中位数", f"{data.median():.2f}")
                with col3:
                    st.metric("标准差", f"{data.std():.2f}")
                with col4:
                    st.metric("偏度", f"{data.skew():.2f}")
                
                # 分布图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("直方图", "Q-Q图", "箱线图", "小提琴图"),
                    specs=[[{"type": "histogram"}, {"type": "scatter"}],
                           [{"type": "box"}, {"type": "violin"}]]
                )
                
                # 直方图
                fig.add_trace(
                    go.Histogram(x=data, name="频次", nbinsx=50),
                    row=1, col=1
                )
                
                # Q-Q图 (简化版)
                sorted_data = np.sort(data)
                theoretical_quantiles = np.linspace(0, 1, len(sorted_data))
                fig.add_trace(
                    go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name="Q-Q"),
                    row=1, col=2
                )
                
                # 箱线图
                fig.add_trace(
                    go.Box(y=data, name="箱线图"),
                    row=2, col=1
                )
                
                # 小提琴图
                fig.add_trace(
                    go.Violin(y=data, name="小提琴图"),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # 类别型特征
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols:
                st.info("数据中没有类别型特征")
                return
            
            selected_col = st.selectbox("选择特征", categorical_cols)
            
            if selected_col:
                value_counts = df[selected_col].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("唯一值数量", df[selected_col].nunique())
                with col2:
                    st.metric("最常见值", value_counts.index[0])
                with col3:
                    st.metric("最常见值占比", f"{(value_counts.iloc[0] / len(df)) * 100:.1f}%")
                
                # 显示前20个类别
                top_values = value_counts.head(20)
                
                fig = px.bar(
                    x=top_values.index,
                    y=top_values.values,
                    title=f"{selected_col} 类别分布 (Top 20)",
                    labels={'x': '类别', 'y': '数量'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_analysis(self):
        """渲染相关性分析"""
        if st.session_state.dm_uploaded_data is None:
            st.warning("请先上传数据文件")
            return
        
        st.subheader("🔗 特征相关性分析")
        
        df = st.session_state.dm_uploaded_data
        corr_matrix, high_corr_pairs = self.calculate_correlation(df)
        
        if corr_matrix.empty:
            st.info("数据中数值型特征少于2个,无法计算相关性")
            return
        
        # 相关性热力图
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="特征相关性热力图"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # 高相关性特征对
        if high_corr_pairs:
            st.subheader("⚠️ 高相关性特征对 (|相关系数| > 0.7)")
            
            df_corr = pd.DataFrame(high_corr_pairs)
            st.dataframe(
                df_corr,
                column_config={
                    "feature1": st.column_config.TextColumn("特征1"),
                    "feature2": st.column_config.TextColumn("特征2"),
                    "correlation": st.column_config.NumberColumn("相关系数", format="%.3f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.info("💡 高相关性特征可能存在多重共线性,建议考虑特征选择")
        else:
            st.success("✅ 未发现高相关性特征对")
    
    def render_report_generation(self):
        """渲染报告生成"""
        if 'missing_info' not in st.session_state.dm_analysis_results:
            st.warning("请先进行数据质量分析")
            return
        
        st.subheader("📄 生成分析报告")
        
        if st.button("📥 导出完整报告 (JSON)", type="primary"):
            report = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_shape': st.session_state.dm_uploaded_data.shape,
                'quality_score': st.session_state.dm_analysis_results['quality_score'],
                'missing_values': st.session_state.dm_analysis_results['missing_info'],
                'outliers': st.session_state.dm_analysis_results['outliers']
            }
            
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            st.download_button(
                label="下载报告",
                data=report_json,
                file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ 报告已生成!")
    
    def render(self):
        """主渲染函数"""
        st.title("🔬 Data Mining Agent - 数据质量分析")
        
        # 数据上传
        self.render_data_upload()
        
        if st.session_state.dm_uploaded_data is not None:
            st.divider()
            
            # Tab切换
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 质量总览",
                "🔍 缺失值",
                "🚨 异常值",
                "📈 分布分析",
                "🔗 相关性",
                "📄 报告"
            ])
            
            with tab1:
                self.render_quality_overview()
            
            with tab2:
                self.render_missing_analysis()
            
            with tab3:
                self.render_outlier_analysis()
            
            with tab4:
                self.render_distribution_analysis()
            
            with tab5:
                self.render_correlation_analysis()
            
            with tab6:
                self.render_report_generation()


def main():
    """主函数"""
    agent = DataMiningAgent()
    agent.render()


if __name__ == "__main__":
    main()
