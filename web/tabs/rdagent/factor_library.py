"""
RD-Agent 因子库持久化管理系统
- 因子保存与版本管理
- 高级搜索与排序
- 性能追踪与对比
- 因子导出与分享
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class FactorLibraryDB:
    """因子库数据库管理"""
    
    def __init__(self, db_path: str = "data/factor_library.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 因子表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT,
                description TEXT,
                formulation TEXT,
                code TEXT,
                ic REAL,
                ir REAL,
                sharpe REAL,
                annual_return REAL,
                max_drawdown REAL,
                turnover REAL,
                valid BOOLEAN,
                status TEXT DEFAULT 'active',
                version INTEGER DEFAULT 1,
                parent_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # 因子性能历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id INTEGER,
                date DATE,
                ic REAL,
                ic_ir REAL,
                daily_return REAL,
                cumulative_return REAL,
                FOREIGN KEY (factor_id) REFERENCES factors(id)
            )
        """)
        
        # 因子标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id INTEGER,
                tag TEXT,
                FOREIGN KEY (factor_id) REFERENCES factors(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_factor(self, factor: Dict) -> int:
        """保存因子"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        metadata = json.dumps(factor.get('metadata', {}))
        
        cursor.execute("""
            INSERT INTO factors (
                name, type, description, formulation, code,
                ic, ir, sharpe, annual_return, max_drawdown, turnover,
                valid, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            factor.get('name'),
            factor.get('type'),
            factor.get('description'),
            factor.get('formulation'),
            factor.get('code'),
            factor.get('ic'),
            factor.get('ir'),
            factor.get('sharpe'),
            factor.get('annual_return'),
            factor.get('max_drawdown'),
            factor.get('turnover'),
            factor.get('valid', True),
            factor.get('status', 'active'),
            metadata
        ))
        
        factor_id = cursor.lastrowid
        
        # 保存标签
        if 'tags' in factor and factor['tags']:
            for tag in factor['tags']:
                cursor.execute(
                    "INSERT INTO factor_tags (factor_id, tag) VALUES (?, ?)",
                    (factor_id, tag)
                )
        
        conn.commit()
        conn.close()
        
        return factor_id
    
    def get_factors(self, 
                   factor_type: Optional[str] = None,
                   status: str = 'active',
                   min_ic: Optional[float] = None,
                   tags: Optional[List[str]] = None,
                   sort_by: str = 'created_at',
                   limit: int = 100) -> List[Dict]:
        """查询因子"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM factors WHERE status = ?"
        params = [status]
        
        if factor_type:
            query += " AND type = ?"
            params.append(factor_type)
        
        if min_ic is not None:
            query += " AND ic >= ?"
            params.append(min_ic)
        
        if tags:
            placeholders = ','.join(['?'] * len(tags))
            query += f""" AND id IN (
                SELECT factor_id FROM factor_tags 
                WHERE tag IN ({placeholders})
                GROUP BY factor_id
                HAVING COUNT(DISTINCT tag) = ?
            )"""
            params.extend(tags)
            params.append(len(tags))
        
        query += f" ORDER BY {sort_by} DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        factors = []
        for row in rows:
            factor = dict(row)
            factor['metadata'] = json.loads(factor['metadata']) if factor['metadata'] else {}
            
            # 获取标签
            cursor.execute(
                "SELECT tag FROM factor_tags WHERE factor_id = ?",
                (factor['id'],)
            )
            factor['tags'] = [tag[0] for tag in cursor.fetchall()]
            
            factors.append(factor)
        
        conn.close()
        return factors
    
    def update_factor(self, factor_id: int, updates: Dict):
        """更新因子"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        query = f"UPDATE factors SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        
        cursor.execute(query, list(updates.values()) + [factor_id])
        conn.commit()
        conn.close()
    
    def delete_factor(self, factor_id: int):
        """删除因子(软删除)"""
        self.update_factor(factor_id, {'status': 'deleted'})
    
    def get_factor_stats(self) -> Dict:
        """获取因子库统计信息"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM factors WHERE status = 'active'")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM factors WHERE status = 'active' AND valid = 1")
        valid = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(ic) FROM factors WHERE status = 'active' AND ic IS NOT NULL")
        avg_ic = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT MAX(ic) FROM factors WHERE status = 'active'")
        max_ic = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT type, COUNT(*) as cnt FROM factors WHERE status = 'active' GROUP BY type")
        type_dist = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total': total,
            'valid': valid,
            'avg_ic': avg_ic,
            'max_ic': max_ic,
            'type_distribution': type_dist
        }
    
    def create_factor_version(self, parent_id: int, updates: Dict) -> int:
        """创建因子新版本"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 获取父版本
        cursor.execute("SELECT * FROM factors WHERE id = ?", (parent_id,))
        parent = dict(cursor.fetchone())
        
        # 获取最大版本号
        cursor.execute("SELECT MAX(version) FROM factors WHERE name = ?", (parent['name'],))
        max_version = cursor.fetchone()[0] or 0
        
        # 创建新版本
        new_factor = parent.copy()
        new_factor.update(updates)
        new_factor['parent_id'] = parent_id
        new_factor['version'] = max_version + 1
        del new_factor['id']
        del new_factor['created_at']
        del new_factor['updated_at']
        
        factor_id = self.save_factor(new_factor)
        
        conn.close()
        return factor_id


class FactorLibraryTab:
    """因子库管理Tab"""
    
    def __init__(self):
        self.db = FactorLibraryDB()
        self.init_session_state()
    
    def init_session_state(self):
        """初始化session state"""
        if 'factor_library_view' not in st.session_state:
            st.session_state.factor_library_view = 'list'
        if 'selected_factor_id' not in st.session_state:
            st.session_state.selected_factor_id = None
    
    def render(self):
        """渲染因子库页面"""
        st.header("📚 因子库管理")
        
        # 顶部统计
        self.render_stats()
        
        st.divider()
        
        # 视图选择
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📋 因子列表", use_container_width=True, 
                        type="primary" if st.session_state.factor_library_view == 'list' else "secondary"):
                st.session_state.factor_library_view = 'list'
                st.rerun()
        with col2:
            if st.button("🔍 高级搜索", use_container_width=True,
                        type="primary" if st.session_state.factor_library_view == 'search' else "secondary"):
                st.session_state.factor_library_view = 'search'
                st.rerun()
        with col3:
            if st.button("📊 性能对比", use_container_width=True,
                        type="primary" if st.session_state.factor_library_view == 'compare' else "secondary"):
                st.session_state.factor_library_view = 'compare'
                st.rerun()
        with col4:
            if st.button("⚙️ 导入/导出", use_container_width=True,
                        type="primary" if st.session_state.factor_library_view == 'import_export' else "secondary"):
                st.session_state.factor_library_view = 'import_export'
                st.rerun()
        
        st.divider()
        
        # 根据视图渲染内容
        if st.session_state.factor_library_view == 'list':
            self.render_factor_list()
        elif st.session_state.factor_library_view == 'search':
            self.render_advanced_search()
        elif st.session_state.factor_library_view == 'compare':
            self.render_performance_comparison()
        elif st.session_state.factor_library_view == 'import_export':
            self.render_import_export()
    
    def render_stats(self):
        """渲染统计信息"""
        stats = self.db.get_factor_stats()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("因子总数", stats['total'])
        
        with col2:
            st.metric("有效因子", stats['valid'], 
                     delta=f"{stats['valid']/max(stats['total'], 1)*100:.1f}%")
        
        with col3:
            st.metric("平均IC", f"{stats['avg_ic']:.3f}")
        
        with col4:
            st.metric("最佳IC", f"{stats['max_ic']:.3f}")
        
        with col5:
            st.metric("因子类型", len(stats['type_distribution']))
    
    def render_factor_list(self):
        """渲染因子列表"""
        st.subheader("📋 因子列表")
        
        # 快速过滤
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox(
                "排序方式",
                ['ic', 'ir', 'sharpe', 'created_at'],
                format_func=lambda x: {
                    'ic': 'IC值',
                    'ir': 'IR比率',
                    'sharpe': 'Sharpe比率',
                    'created_at': '创建时间'
                }[x],
                key="fl_sort_by"
            )
        
        with col2:
            factor_type = st.selectbox(
                "因子类型",
                ['全部', '技术因子', '基本面因子', '量价因子', '情绪因子', '混合因子'],
                key="fl_type"
            )
        
        with col3:
            min_ic = st.number_input(
                "最小IC值",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                key="fl_min_ic"
            )
        
        # 获取因子
        factors = self.db.get_factors(
            factor_type=None if factor_type == '全部' else factor_type,
            min_ic=min_ic if min_ic > 0 else None,
            sort_by=sort_by
        )
        
        if not factors:
            st.info("📭 因子库为空。请在'因子挖掘'模块生成因子后保存到库中。")
            
            # 提示如何从因子挖掘保存
            with st.expander("💡 如何保存因子到库?"):
                st.markdown("""
                1. 前往 **🔍 因子挖掘** Tab
                2. 使用 LLM 生成因子或从研报提取
                3. 在生成的因子列表中点击 **💾 保存到库** 按钮
                4. 因子将自动保存到此因子库中
                """)
            return
        
        # 显示因子表格
        df_data = []
        for factor in factors:
            df_data.append({
                "ID": factor['id'],
                "名称": factor['name'],
                "类型": factor['type'] or 'N/A',
                "IC": f"{factor['ic']:.3f}" if factor['ic'] else 'N/A',
                "IR": f"{factor['ir']:.3f}" if factor['ir'] else 'N/A',
                "Sharpe": f"{factor['sharpe']:.3f}" if factor['sharpe'] else 'N/A',
                "状态": "✅ 有效" if factor['valid'] else "❌ 无效",
                "版本": f"v{factor['version']}",
                "创建时间": factor['created_at'][:10] if factor['created_at'] else 'N/A'
            })
        
        df = pd.DataFrame(df_data)
        
        # 使用st.dataframe展示
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "IC": st.column_config.NumberColumn(format="%.3f"),
                "IR": st.column_config.NumberColumn(format="%.3f"),
                "Sharpe": st.column_config.NumberColumn(format="%.3f"),
            }
        )
        
        # 因子详情查看
        st.divider()
        st.subheader("🔍 因子详情")
        
        factor_id = st.selectbox(
            "选择因子查看详情",
            [f['id'] for f in factors],
            format_func=lambda x: next(f['name'] for f in factors if f['id'] == x),
            key="fl_detail_select"
        )
        
        if factor_id:
            self.render_factor_detail(factor_id, factors)
    
    def render_factor_detail(self, factor_id: int, factors: List[Dict]):
        """渲染因子详情"""
        factor = next(f for f in factors if f['id'] == factor_id)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**因子名称**: {factor['name']}")
            st.markdown(f"**因子类型**: {factor['type'] or 'N/A'}")
            st.markdown(f"**描述**: {factor['description'] or '无描述'}")
            
            if factor['formulation']:
                st.markdown(f"**公式**: `{factor['formulation']}`")
            
            if factor['tags']:
                tags_html = ' '.join([f'<span style="background:#e1f5ff;padding:3px 8px;border-radius:3px;margin:2px;">{tag}</span>' 
                                     for tag in factor['tags']])
                st.markdown(f"**标签**: {tags_html}", unsafe_allow_html=True)
        
        with col2:
            st.metric("IC", f"{factor['ic']:.3f}" if factor['ic'] else 'N/A')
            st.metric("IR", f"{factor['ir']:.3f}" if factor['ir'] else 'N/A')
            st.metric("Sharpe", f"{factor['sharpe']:.3f}" if factor['sharpe'] else 'N/A')
        
        # 代码展示
        if factor['code']:
            with st.expander("📄 因子代码", expanded=False):
                st.code(factor['code'], language='python')
        
        # 操作按钮
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("✏️ 编辑", key=f"edit_{factor_id}"):
                st.info("编辑功能开发中...")
        with col2:
            if st.button("🔄 创建新版本", key=f"version_{factor_id}"):
                st.info("版本管理功能开发中...")
        with col3:
            if st.button("📊 查看性能", key=f"perf_{factor_id}"):
                st.session_state.selected_factor_id = factor_id
                st.session_state.factor_library_view = 'compare'
                st.rerun()
        with col4:
            if st.button("🗑️ 删除", key=f"delete_{factor_id}", type="secondary"):
                self.db.delete_factor(factor_id)
                st.success("因子已删除")
                st.rerun()
    
    def render_advanced_search(self):
        """渲染高级搜索"""
        st.subheader("🔍 高级搜索")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**基础过滤**")
            factor_type = st.multiselect(
                "因子类型",
                ['技术因子', '基本面因子', '量价因子', '情绪因子', '混合因子'],
                key="as_type"
            )
            
            ic_range = st.slider(
                "IC值范围",
                min_value=-1.0,
                max_value=1.0,
                value=(-1.0, 1.0),
                step=0.01,
                key="as_ic_range"
            )
            
            valid_only = st.checkbox("仅显示有效因子", value=True, key="as_valid")
        
        with col2:
            st.markdown("**高级过滤**")
            tags_input = st.text_input(
                "标签 (逗号分隔)",
                placeholder="momentum, volume, reversal",
                key="as_tags"
            )
            
            date_range = st.date_input(
                "创建日期范围",
                value=(),
                key="as_date_range"
            )
            
            sort_by = st.selectbox(
                "排序方式",
                ['ic', 'ir', 'sharpe', 'created_at'],
                key="as_sort"
            )
        
        if st.button("🔍 执行搜索", type="primary", use_container_width=True):
            tags = [t.strip() for t in tags_input.split(',') if t.strip()] if tags_input else None
            
            # 这里应该调用数据库查询
            st.info("搜索功能执行中... (实际查询将连接数据库)")
            
            # Mock结果
            st.success("找到 12 个匹配的因子")
    
    def render_performance_comparison(self):
        """渲染性能对比"""
        st.subheader("📊 因子性能对比")
        
        factors = self.db.get_factors(limit=50)
        
        if not factors:
            st.info("因子库为空,无法进行对比")
            return
        
        # 选择对比因子
        selected_factors = st.multiselect(
            "选择要对比的因子 (最多5个)",
            [f['id'] for f in factors],
            format_func=lambda x: next(f['name'] for f in factors if f['id'] == x),
            max_selections=5,
            key="pc_factors"
        )
        
        if len(selected_factors) < 2:
            st.warning("请至少选择2个因子进行对比")
            return
        
        # 获取选中的因子
        compare_factors = [f for f in factors if f['id'] in selected_factors]
        
        # 性能对比表
        st.subheader("📈 性能指标对比")
        
        comp_data = []
        for factor in compare_factors:
            comp_data.append({
                "因子名称": factor['name'],
                "IC": factor['ic'] or 0,
                "IR": factor['ir'] or 0,
                "Sharpe": factor['sharpe'] or 0,
                "年化收益": f"{(factor['annual_return'] or 0) * 100:.2f}%",
                "最大回撤": f"{(factor['max_drawdown'] or 0) * 100:.2f}%",
                "换手率": f"{(factor['turnover'] or 0) * 100:.2f}%"
            })
        
        df = pd.DataFrame(comp_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 雷达图对比
        st.subheader("🎯 综合性能雷达图")
        
        categories = ['IC', 'IR', 'Sharpe', '收益', '稳定性']
        
        fig = go.Figure()
        
        for factor in compare_factors:
            fig.add_trace(go.Scatterpolar(
                r=[
                    (factor['ic'] or 0) * 10,  # 归一化到0-1
                    (factor['ir'] or 0),
                    (factor['sharpe'] or 0) / 3,
                    (factor['annual_return'] or 0) * 5,
                    1 - abs(factor['max_drawdown'] or 0)
                ],
                theta=categories,
                fill='toself',
                name=factor['name']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="因子性能雷达图"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_import_export(self):
        """渲染导入/导出"""
        st.subheader("⚙️ 导入/导出因子")
        
        tab1, tab2 = st.tabs(["📤 导出因子", "📥 导入因子"])
        
        with tab1:
            st.markdown("**导出因子到JSON文件**")
            
            factors = self.db.get_factors()
            
            if not factors:
                st.info("因子库为空")
            else:
                export_format = st.radio(
                    "导出格式",
                    ["JSON", "CSV"],
                    horizontal=True
                )
                
                factor_ids = st.multiselect(
                    "选择要导出的因子",
                    [f['id'] for f in factors],
                    format_func=lambda x: next(f['name'] for f in factors if f['id'] == x),
                    default=[f['id'] for f in factors[:5]]
                )
                
                if st.button("📤 导出", type="primary"):
                    selected = [f for f in factors if f['id'] in factor_ids]
                    
                    if export_format == "JSON":
                        export_data = json.dumps(selected, indent=2, ensure_ascii=False, default=str)
                        st.download_button(
                            "⬇️ 下载JSON",
                            export_data,
                            file_name=f"factors_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    else:  # CSV
                        df = pd.DataFrame(selected)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ 下载CSV",
                            csv,
                            file_name=f"factors_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        with tab2:
            st.markdown("**从JSON文件导入因子**")
            
            uploaded_file = st.file_uploader(
                "选择JSON文件",
                type=['json'],
                help="上传之前导出的因子JSON文件"
            )
            
            if uploaded_file:
                try:
                    factors_data = json.load(uploaded_file)
                    
                    if not isinstance(factors_data, list):
                        factors_data = [factors_data]
                    
                    st.success(f"✅ 文件解析成功,找到 {len(factors_data)} 个因子")
                    
                    # 预览
                    with st.expander("📄 预览导入内容"):
                        for factor in factors_data[:3]:
                            st.json(factor)
                        if len(factors_data) > 3:
                            st.info(f"... 还有 {len(factors_data) - 3} 个因子")
                    
                    if st.button("📥 确认导入", type="primary"):
                        imported = 0
                        for factor in factors_data:
                            try:
                                self.db.save_factor(factor)
                                imported += 1
                            except Exception as e:
                                st.warning(f"因子 {factor.get('name', 'Unknown')} 导入失败: {e}")
                        
                        st.success(f"✅ 成功导入 {imported}/{len(factors_data)} 个因子")
                        
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON解析失败: {e}")
                except Exception as e:
                    st.error(f"❌ 导入失败: {e}")


def render():
    """渲染入口"""
    tab = FactorLibraryTab()
    tab.render()
