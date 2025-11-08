"""
增强的数据表格组件
支持颜色标记、排序、筛选、勾选等高级功能
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import numpy as np


class EnhancedTable:
    """增强的数据表格"""
    
    def __init__(self, key_prefix: str = "table"):
        """初始化表格组件"""
        self.key_prefix = key_prefix
        self.selected_rows = []
    
    def render(
        self,
        data: pd.DataFrame,
        columns_config: Optional[Dict[str, Dict]] = None,
        enable_selection: bool = True,
        enable_sort: bool = True,
        enable_filter: bool = True,
        color_rules: Optional[Dict[str, Callable]] = None,
        default_sort_column: Optional[str] = None,
        default_sort_ascending: bool = False
    ) -> Dict[str, Any]:
        """
        渲染增强表格
        
        Args:
            data: DataFrame数据
            columns_config: 列配置 {'列名': {'display': '显示名', 'format': format_func}}
            enable_selection: 是否启用行选择
            enable_sort: 是否启用排序
            enable_filter: 是否启用筛选
            color_rules: 颜色规则 {'列名': lambda val: 'green' if val > 0 else 'red'}
            default_sort_column: 默认排序列
            default_sort_ascending: 默认排序方向
            
        Returns:
            {'data': filtered_data, 'selected': selected_rows}
        """
        if data.empty:
            st.info("📭 暂无数据")
            return {'data': data, 'selected': []}
        
        display_data = data.copy()
        
        # 排序功能
        if enable_sort and not display_data.empty:
            display_data = self._render_sort_controls(display_data, default_sort_column, default_sort_ascending)
        
        # 筛选功能
        if enable_filter and not display_data.empty:
            display_data = self._render_filter_controls(display_data)
        
        # 显示统计信息
        st.caption(f"显示 {len(display_data)} / {len(data)} 条记录")
        
        # 行选择
        selected_indices = []
        if enable_selection and not display_data.empty:
            with st.expander("✅ 批量操作", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    select_all = st.checkbox("全选", key=f"{self.key_prefix}_select_all")
                with col2:
                    if st.button("清空选择", key=f"{self.key_prefix}_clear_selection"):
                        select_all = False
                        st.rerun()
                with col3:
                    st.write(f"已选: {len(selected_indices)} 条")
        
        # 应用颜色规则并渲染表格
        styled_data = self._apply_color_rules(display_data, color_rules)
        
        # 使用streamlit原生dataframe展示
        st.dataframe(
            styled_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # 如果启用选择，提供选择界面
        if enable_selection and not display_data.empty:
            selected_indices = self._render_selection_interface(display_data)
        
        return {
            'data': display_data,
            'selected': selected_indices,
            'selected_data': display_data.iloc[selected_indices] if selected_indices else pd.DataFrame()
        }
    
    def _render_sort_controls(
        self,
        data: pd.DataFrame,
        default_column: Optional[str],
        default_ascending: bool
    ) -> pd.DataFrame:
        """渲染排序控件"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 选择排序列
            sortable_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) or col in ['symbol', 'name']]
            if sortable_columns:
                default_col = default_column if default_column in sortable_columns else sortable_columns[0]
                sort_column = st.selectbox(
                    "排序列",
                    sortable_columns,
                    index=sortable_columns.index(default_col) if default_col in sortable_columns else 0,
                    key=f"{self.key_prefix}_sort_col"
                )
            else:
                return data
        
        with col2:
            # 选择排序方向
            sort_ascending = st.radio(
                "排序",
                ["降序", "升序"],
                index=1 if default_ascending else 0,
                key=f"{self.key_prefix}_sort_dir",
                horizontal=True
            )
        
        # 执行排序
        if sort_column:
            data = data.sort_values(
                by=sort_column,
                ascending=(sort_ascending == "升序")
            ).reset_index(drop=True)
        
        return data
    
    def _render_filter_controls(self, data: pd.DataFrame) -> pd.DataFrame:
        """渲染筛选控件"""
        with st.expander("🔍 高级筛选", expanded=False):
            filters_applied = False
            
            # 数值列筛选
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                st.markdown("##### 数值筛选")
                for col in numeric_columns[:3]:  # 限制显示前3个数值列
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{col}**")
                    with col2:
                        min_val = float(data[col].min())
                        max_val = float(data[col].max())
                        filter_min = st.number_input(
                            f"最小值",
                            value=min_val,
                            key=f"{self.key_prefix}_filter_{col}_min",
                            label_visibility="collapsed"
                        )
                    with col3:
                        filter_max = st.number_input(
                            f"最大值",
                            value=max_val,
                            key=f"{self.key_prefix}_filter_{col}_max",
                            label_visibility="collapsed"
                        )
                    
                    if filter_min != min_val or filter_max != max_val:
                        data = data[(data[col] >= filter_min) & (data[col] <= filter_max)]
                        filters_applied = True
            
            # 文本列筛选
            text_columns = data.select_dtypes(include=['object']).columns.tolist()
            if text_columns and len(text_columns) > 0:
                st.markdown("##### 文本筛选")
                search_col = st.selectbox(
                    "搜索列",
                    text_columns,
                    key=f"{self.key_prefix}_search_col"
                )
                search_text = st.text_input(
                    "搜索内容",
                    key=f"{self.key_prefix}_search_text"
                )
                
                if search_text:
                    data = data[data[search_col].str.contains(search_text, case=False, na=False)]
                    filters_applied = True
            
            if filters_applied:
                st.success(f"✅ 筛选后剩余 {len(data)} 条记录")
        
        return data
    
    def _apply_color_rules(
        self,
        data: pd.DataFrame,
        color_rules: Optional[Dict[str, Callable]]
    ) -> pd.DataFrame:
        """应用颜色规则"""
        if not color_rules:
            return data
        
        # 注意：Streamlit的dataframe不支持直接的单元格颜色
        # 我们可以添加emoji或特殊标记来表示状态
        styled_data = data.copy()
        
        for col, rule_func in color_rules.items():
            if col in styled_data.columns:
                # 在列名前添加状态指示器
                styled_data[f"状态_{col}"] = styled_data[col].apply(
                    lambda val: self._get_status_emoji(rule_func(val))
                )
        
        return styled_data
    
    def _get_status_emoji(self, color: str) -> str:
        """根据颜色返回emoji"""
        color_map = {
            'green': '🟢',
            'yellow': '🟡',
            'orange': '🟠',
            'red': '🔴',
            'gray': '⚪'
        }
        return color_map.get(color, '⚪')
    
    def _render_selection_interface(self, data: pd.DataFrame) -> List[int]:
        """渲染选择界面"""
        st.markdown("---")
        st.markdown("### 📋 选择行进行操作")
        
        # 提供多选框选择行
        if 'symbol' in data.columns and 'name' in data.columns:
            options = [f"{row['symbol']} - {row['name']}" for _, row in data.iterrows()]
        else:
            options = [f"行 {i}" for i in range(len(data))]
        
        selected = st.multiselect(
            "选择要操作的行",
            options=options,
            key=f"{self.key_prefix}_multiselect",
            help="可选择多行进行批量操作"
        )
        
        # 返回选中的索引
        return [options.index(s) for s in selected]


def render_enhanced_table(
    data: pd.DataFrame,
    key_prefix: str = "table",
    **kwargs
) -> Dict[str, Any]:
    """
    快捷函数：渲染增强表格
    
    Args:
        data: DataFrame数据
        key_prefix: 组件key前缀
        **kwargs: 传递给EnhancedTable.render的其他参数
    
    Returns:
        渲染结果字典
    """
    table = EnhancedTable(key_prefix=key_prefix)
    return table.render(data, **kwargs)


# 测试代码
if __name__ == "__main__":
    st.set_page_config(page_title="增强表格测试", layout="wide")
    
    st.title("📊 增强数据表格测试")
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'symbol': [f"00000{i}" for i in range(1, 21)],
        'name': [f"测试股票{i}" for i in range(1, 21)],
        'price': np.random.uniform(10, 100, 20),
        'change': np.random.uniform(-10, 10, 20),
        'volume': np.random.randint(1000, 10000, 20),
        'quality_score': np.random.randint(40, 100, 20)
    })
    
    # 定义颜色规则
    def change_color(val):
        if val > 5:
            return 'green'
        elif val > 0:
            return 'yellow'
        elif val > -5:
            return 'orange'
        else:
            return 'red'
    
    color_rules = {
        'change': change_color
    }
    
    # 渲染表格
    st.markdown("## 基础功能测试")
    result = render_enhanced_table(
        data=test_data,
        key_prefix="test",
        enable_selection=True,
        enable_sort=True,
        enable_filter=True,
        color_rules=color_rules,
        default_sort_column='change',
        default_sort_ascending=False
    )
    
    # 显示选择结果
    if result['selected']:
        st.markdown("### 已选择的行")
        st.dataframe(result['selected_data'], use_container_width=True, hide_index=True)
        
        st.success(f"✅ 已选择 {len(result['selected'])} 行数据")
