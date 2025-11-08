"""
Qlib数据工具标签页
提供数据下载、验证、转换、表达式测试、缓存管理等功能
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import sys
import subprocess
from typing import Optional


def render_qlib_data_tools_tab():
    """渲染数据工具标签页"""
    st.header("🛠️ Qlib数据工具箱")
    
    # 5个子标签
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 数据下载",
        "✅ 数据验证", 
        "🔄 格式转换",
        "🧪 表达式测试",
        "💾 缓存管理"
    ])
    
    with tab1:
        render_data_download_tab()
    
    with tab2:
        render_data_validation_tab()
    
    with tab3:
        render_data_conversion_tab()
    
    with tab4:
        render_expression_test_tab()
    
    with tab5:
        render_cache_management_tab()


# ============================================================================
# Tab 1: 数据下载
# ============================================================================

def render_data_download_tab():
    """数据下载UI"""
    st.subheader("📥 Qlib数据下载")
    
    st.info("💡 下载Qlib官方数据到本地，支持中国A股、美股等市场")
    
    # 配置区域
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox(
            "数据区域",
            options=["cn", "us", "all"],
            index=0,
            help="cn=中国A股, us=美国股市, all=全球"
        )
    
    with col2:
        interval = st.selectbox(
            "数据频率",
            options=["1d", "1h", "5min", "1min"],
            index=0,
            help="1d=日线, 1h=小时, 5min/1min=高频"
        )
    
    # 目标目录
    default_dir = str(Path.home() / '.qlib' / 'qlib_data' / f'{region}_data')
    target_dir = st.text_input("目标目录", value=default_dir)
    
    delete_old = st.checkbox("删除旧数据", value=False, help="下载前清空目标目录")
    
    # 下载按钮
    if st.button("🚀 开始下载", type="primary"):
        with st.spinner("正在下载数据..."):
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            
            try:
                # 复用现有的download_qlib_data_v2逻辑
                result = download_qlib_data(
                    region=region,
                    interval=interval,
                    target_dir=target_dir,
                    delete_old=delete_old,
                    progress_callback=lambda p, msg: progress_placeholder.progress(p, text=msg)
                )
                
                if result['success']:
                    st.success(f"✅ 数据下载成功！\n\n{result['message']}")
                else:
                    st.error(f"❌ 下载失败：{result['error']}")
                    
            except Exception as e:
                st.error(f"❌ 下载出错：{str(e)}")
    
    # 使用说明
    with st.expander("📚 使用说明"):
        st.markdown("""
        ### 数据下载说明
        
        **支持的区域**:
        - `cn`: 中国A股数据 (~12-20GB)
        - `us`: 美国股市数据
        - `all`: 全球市场数据
        
        **支持的频率**:
        - `1d`: 日线数据（推荐新手）
        - `1h`, `5min`, `1min`: 高频数据
        
        **注意事项**:
        - 首次下载需要较长时间（10-30分钟）
        - 确保有足够的磁盘空间（至少30GB）
        - 支持断点续传和多方法回退
        """)


def download_qlib_data(region, interval, target_dir, delete_old, progress_callback):
    """下载Qlib数据（复用scripts/download_qlib_data_v2.py逻辑）"""
    try:
        # 方法1: 使用GetData API
        try:
            from qlib.data import GetData
            gd = GetData()
            progress_callback(0.3, "使用GetData API下载...")
            gd.qlib_data(
                target_dir=target_dir,
                region=region,
                interval=interval,
                delete_old=delete_old
            )
            progress_callback(1.0, "下载完成！")
            return {'success': True, 'message': f'数据已保存到: {target_dir}'}
        except Exception as e1:
            progress_callback(0.5, f"方法1失败，尝试方法2... ({str(e1)[:50]})")
            
            # 方法2: 命令行
            cmd = [
                sys.executable, '-m', 'qlib.cli.data',
                'qlib_data',
                '--target_dir', target_dir,
                '--region', region,
                '--interval', interval
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                progress_callback(1.0, "下载完成！")
                return {'success': True, 'message': f'数据已保存到: {target_dir}'}
            else:
                return {'success': False, 'error': result.stderr or result.stdout}
                
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# Tab 2: 数据验证
# ============================================================================

def render_data_validation_tab():
    """数据验证UI"""
    st.subheader("✅ 数据健康检查")
    
    # 数据路径
    data_path = st.text_input(
        "数据路径",
        value=str(Path.home() / '.qlib' / 'qlib_data' / 'cn_data')
    )
    
    market = st.selectbox("股票池", options=["csi300", "csi500", "all"], index=0)
    
    if st.button("🔍 开始验证", type="primary"):
        with st.spinner("验证中..."):
            result = validate_qlib_data_enhanced(data_path, market)
            
            if result['success']:
                # 显示检查结果
                st.success("✅ 数据验证完成！")
                
                # 统计卡片
                cols = st.columns(4)
                cols[0].metric("股票数量", result['stock_count'])
                cols[1].metric("数据完整度", f"{result['completeness']:.1%}")
                cols[2].metric("交易日数", result['trading_days'])
                cols[3].metric("日期范围", f"{result['date_range']['days']}天")
                
                # 详细结果
                st.subheader("📊 详细检查结果")
                
                # 数据完整性可视化
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result['completeness'] * 100,
                    title = {'text': "数据完整度"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 95
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # 异常检测
                if result.get('anomalies'):
                    st.warning(f"⚠️ 发现 {len(result['anomalies'])} 个异常")
                    st.dataframe(pd.DataFrame(result['anomalies']))
                
            else:
                st.error(f"❌ 验证失败：{result['error']}")


def validate_qlib_data_enhanced(data_path, market):
    """增强的数据验证（复用scripts/validate_qlib_data.py并增强）"""
    try:
        import qlib
        from qlib.data import D
        
        # 初始化
        qlib.init(provider_uri=data_path)
        
        # 获取股票列表
        instruments = D.instruments(market=market)
        stock_list = D.list_instruments(instruments=instruments, as_list=True)
        
        # 测试数据
        test_symbols = stock_list[:min(10, len(stock_list))]
        features = D.features(
            test_symbols,
            ['$close', '$volume', '$open', '$high', '$low'],
            start_time='2023-01-01',
            end_time='2024-06-30'
        )
        
        # 统计
        missing = features.isnull().sum().sum()
        total = features.size
        completeness = 1 - (missing / total)
        
        dates = features.index.get_level_values('datetime').unique()
        
        # 异常检测
        anomalies = []
        # TODO: 添加更多异常检测逻辑
        
        return {
            'success': True,
            'stock_count': len(stock_list),
            'completeness': completeness,
            'trading_days': len(dates),
            'date_range': {
                'start': str(dates.min()),
                'end': str(dates.max()),
                'days': len(dates)
            },
            'anomalies': anomalies
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# Tab 3: 格式转换
# ============================================================================

def render_data_conversion_tab():
    """数据格式转换UI"""
    st.subheader("🔄 数据格式转换")
    
    st.info("💡 将CSV/Excel数据转换为Qlib格式")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择文件",
        type=['csv', 'xlsx', 'xls'],
        help="支持CSV和Excel格式"
    )
    
    if uploaded_file:
        st.success(f"✅ 文件已上传: {uploaded_file.name}")
        
        # 预览数据
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, nrows=5)
            else:
                df = pd.read_excel(uploaded_file, nrows=5)
            
            st.write("**数据预览** (前5行):")
            st.dataframe(df)
            
            # 列名映射
            st.subheader("📝 列名映射")
            st.write("请指定各列对应的标准字段:")
            
            col_mapping = {}
            cols = st.columns(4)
            
            for i, (std_name, label) in enumerate([
                ('date', '日期列'),
                ('symbol', '股票代码列'),
                ('close', '收盘价列'),
                ('volume', '成交量列')
            ]):
                with cols[i % 4]:
                    col_mapping[std_name] = st.selectbox(
                        label,
                        options=[''] + df.columns.tolist(),
                        key=f'col_{std_name}'
                    )
            
            # 输出目录
            output_dir = st.text_input(
                "输出目录",
                value="./data/converted_data"
            )
            
            # 转换按钮
            if st.button("🔄 开始转换", type="primary"):
                if not all([col_mapping['date'], col_mapping['symbol'], col_mapping['close']]):
                    st.error("❌ 请至少指定日期、股票代码、收盘价列！")
                else:
                    with st.spinner("转换中..."):
                        try:
                            from qlib_enhanced.data_tools import DataConverter
                            
                            converter = DataConverter()
                            
                            # 读取完整文件
                            uploaded_file.seek(0)
                            if uploaded_file.name.endswith('.csv'):
                                df_full = pd.read_csv(uploaded_file)
                            else:
                                df_full = pd.read_excel(uploaded_file)
                            
                            # 重命名列
                            reverse_mapping = {v: k for k, v in col_mapping.items() if v}
                            df_full = df_full.rename(columns=reverse_mapping)
                            
                            # 保存
                            result_path = converter.save_to_qlib_format(df_full, output_dir)
                            
                            st.success(f"✅ 转换成功！\n\n数据已保存到: {result_path}")
                            
                            # 数据摘要
                            summary = converter.get_data_summary(df_full)
                            st.json(summary)
                            
                        except Exception as e:
                            st.error(f"❌ 转换失败：{str(e)}")
                            
        except Exception as e:
            st.error(f"❌ 文件读取失败：{str(e)}")


# ============================================================================
# Tab 4: 表达式测试
# ============================================================================

def render_expression_test_tab():
    """表达式引擎测试UI"""
    st.subheader("🧪 Qlib表达式测试器")
    
    # 示例表达式
    from qlib_enhanced.data_tools import ExpressionTester
    tester = ExpressionTester()
    examples = tester.get_example_expressions()
    
    # 选择示例
    st.write("**快速选择示例**:")
    example_cat = st.selectbox("表达式类别", options=list(examples.keys()))
    example_expr = st.selectbox("示例表达式", options=examples[example_cat])
    
    # 表达式输入
    expression = st.text_area(
        "Qlib表达式",
        value=example_expr,
        height=100,
        help="输入Qlib格式的因子表达式"
    )
    
    # 语法验证
    if expression:
        is_valid, msg = tester.validate_syntax(expression)
        if is_valid:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")
    
    # 测试参数
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols_input = st.text_input("股票代码", value="000001.SZ, 600519.SH")
    with col2:
        start_date = st.date_input("开始日期", value=datetime.now() - timedelta(days=365))
    with col3:
        end_date = st.date_input("结束日期", value=datetime.now())
    
    # 测试按钮
    if st.button("🧪 测试表达式", type="primary"):
        symbols = [s.strip() for s in symbols_input.split(',')]
        
        with st.spinner("计算中..."):
            result = tester.test_expression(
                expression,
                symbols,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if result.success:
                st.success("✅ 计算成功！")
                
                # 统计摘要
                if result.statistics:
                    st.subheader("📊 统计摘要")
                    cols = st.columns(4)
                    cols[0].metric("数据量", f"{result.statistics['count']:,}")
                    cols[1].metric("均值", f"{result.statistics['mean']:.4f}")
                    cols[2].metric("标准差", f"{result.statistics['std']:.4f}")
                    cols[3].metric("缺失率", f"{result.statistics['missing_rate']:.1%}")
                
                # 数据预览
                if result.data is not None:
                    st.subheader("📈 数据预览")
                    st.dataframe(result.data.head(20))
                    
                    # 分布图
                    fig = px.histogram(
                        result.data.reset_index(),
                        x=expression,
                        title="因子分布",
                        nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"❌ 计算失败：{result.error_message}")


# ============================================================================
# Tab 5: 缓存管理
# ============================================================================

def render_cache_management_tab():
    """缓存管理UI"""
    st.subheader("💾 缓存管理")
    
    try:
        from app.core.cache_manager import get_cache_manager
        cache_mgr = get_cache_manager()
        
        # 统计信息
        cache_dir = cache_mgr.cache_dir
        cache_files = list(cache_dir.glob("*.cache"))
        
        total_size = sum(f.stat().st_size for f in cache_files) / 1024 / 1024  # MB
        memory_items = len(cache_mgr._memory_cache)
        
        # 显示统计
        cols = st.columns(4)
        cols[0].metric("内存缓存", f"{memory_items} 项")
        cols[1].metric("磁盘缓存", f"{len(cache_files)} 项")
        cols[2].metric("磁盘占用", f"{total_size:.2f} MB")
        cols[3].metric("缓存目录", str(cache_dir.name))
        
        # 操作按钮
        st.subheader("🛠️ 缓存操作")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ 清空全部", type="secondary"):
                cache_mgr.clear(memory_only=False)
                st.success("✅ 已清空全部缓存")
                st.rerun()
        
        with col2:
            if st.button("🧹 清空内存", type="secondary"):
                cache_mgr.clear(memory_only=True)
                st.success("✅ 已清空内存缓存")
                st.rerun()
        
        with col3:
            if st.button("⏰ 清理过期", type="secondary"):
                count = cache_mgr.cleanup_expired()
                st.success(f"✅ 已清理 {count} 个过期缓存")
                st.rerun()
        
        # 缓存列表
        if cache_files:
            st.subheader("📋 缓存文件列表")
            cache_data = []
            for f in cache_files[:50]:  # 最多显示50个
                cache_data.append({
                    '文件名': f.name,
                    '大小': f"{f.stat().st_size / 1024:.2f} KB",
                    '修改时间': datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
            st.dataframe(pd.DataFrame(cache_data), use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ 缓存管理器加载失败：{str(e)}")


# 导出
__all__ = ['render_qlib_data_tools_tab']
