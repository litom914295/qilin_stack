"""
全局UI样式管理器
统一管理Streamlit应用的CSS样式
"""

import streamlit as st
from .color_scheme import Colors, ThemeConfig


def inject_global_styles():
    """注入全局CSS样式到Streamlit应用"""
    
    theme = ThemeConfig()
    
    css = f"""
    <style>
        /* ==================== 全局样式 ==================== */
        
        /* 主容器 */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }}
        
        /* 标题样式 */
        h1 {{
            font-size: {theme.font_size_title} !important;
            font-weight: 700 !important;
            color: {Colors.TEXT_DARK} !important;
            margin-bottom: {theme.spacing_large} !important;
        }}
        
        h2 {{
            font-size: {theme.font_size_subtitle} !important;
            font-weight: 600 !important;
            color: {Colors.TEXT_DARK} !important;
            margin-top: {theme.spacing_large} !important;
            margin-bottom: {theme.spacing_medium} !important;
        }}
        
        h3 {{
            font-size: 16px !important;
            font-weight: 600 !important;
            color: {Colors.TEXT_DARK} !important;
            margin-top: {theme.spacing_medium} !important;
            margin-bottom: {theme.spacing_small} !important;
        }}
        
        /* ==================== 指标卡片 ==================== */
        
        .metric-card {{
            background: {Colors.BG_LIGHT};
            border: 1px solid {Colors.BORDER_LIGHT};
            border-radius: {theme.border_radius};
            padding: {theme.spacing_medium};
            box-shadow: {theme.box_shadow};
            transition: all 0.3s ease;
            height: 100%;
        }}
        
        .metric-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        
        .metric-card-title {{
            font-size: {theme.font_size_small};
            color: {Colors.TEXT_MUTED};
            margin-bottom: {theme.spacing_small};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-card-value {{
            font-size: 32px;
            font-weight: 700;
            color: {Colors.TEXT_DARK};
            margin: {theme.spacing_small} 0;
            line-height: 1.2;
        }}
        
        .metric-card-delta {{
            font-size: {theme.font_size_body};
            margin-top: {theme.spacing_small};
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        
        .metric-card-delta.positive {{
            color: {Colors.SUCCESS};
        }}
        
        .metric-card-delta.negative {{
            color: {Colors.DANGER};
        }}
        
        /* ==================== Streamlit原生组件优化 ==================== */
        
        /* 优化metric组件 */
        [data-testid="stMetricValue"] {{
            font-size: 28px !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stMetricDelta"] {{
            font-size: 14px !important;
        }}
        
        /* 优化按钮 */
        .stButton > button {{
            border-radius: {theme.border_radius};
            font-weight: 500;
            transition: all 0.2s ease;
            border: none;
            padding: 0.5rem 1.5rem;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        /* 主要按钮 */
        .stButton > button[kind="primary"] {{
            background-color: {Colors.PRIMARY};
            color: white;
        }}
        
        /* 次要按钮 */
        .stButton > button[kind="secondary"] {{
            background-color: {Colors.BG_LIGHT};
            color: {Colors.TEXT_DARK};
            border: 1px solid {Colors.BORDER_LIGHT};
        }}
        
        /* ==================== 表格样式 ==================== */
        
        /* DataFrame样式 */
        .dataframe {{
            border: 1px solid {Colors.BORDER_LIGHT} !important;
            border-radius: {theme.border_radius} !important;
            overflow: hidden !important;
        }}
        
        .dataframe thead tr th {{
            background-color: {Colors.BG_LIGHT} !important;
            color: {Colors.TEXT_DARK} !important;
            font-weight: 600 !important;
            font-size: {theme.font_size_body} !important;
            padding: 12px !important;
            border-bottom: 2px solid {Colors.BORDER_LIGHT} !important;
        }}
        
        .dataframe tbody tr td {{
            padding: 10px 12px !important;
            font-size: {theme.font_size_body} !important;
            border-bottom: 1px solid {Colors.BORDER_LIGHT} !important;
        }}
        
        .dataframe tbody tr:hover {{
            background-color: {Colors.BG_INFO} !important;
        }}
        
        .dataframe tbody tr:last-child td {{
            border-bottom: none !important;
        }}
        
        /* ==================== 标签页样式 ==================== */
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {Colors.BG_LIGHT};
            padding: 8px;
            border-radius: {theme.border_radius};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            padding: 0 24px;
            background-color: transparent;
            border-radius: 6px;
            color: {Colors.TEXT_MUTED};
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: white;
            color: {Colors.TEXT_DARK};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: white !important;
            color: {Colors.PRIMARY} !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }}
        
        /* ==================== 下拉框样式优化 ==================== */
        
        /* Selectbox - 增加宽度，确保文本完整显示 */
        .stSelectbox > div > div {
            min-width: 250px !important;
        }
        
        [data-baseweb="select"] {
            min-width: 250px !important;
        }
        
        [data-baseweb="select"] > div {
            min-width: 250px !important;
            white-space: nowrap;
        }
        
        /* Multiselect - 增加宽度 */
        .stMultiSelect > div > div {
            min-width: 300px !important;
        }
        
        [data-baseweb="tag"] {
            max-width: none !important;
            white-space: nowrap !important;
        }
        
        /* 下拉菜单选项 - 确保文本不被截断 */
        [role="option"] {
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        
        [data-baseweb="popover"] {
            min-width: 300px !important;
        }
        
        /* ==================== 侧边栏下拉框特殊优化 ==================== */
        
        /* 侧边栏selectbox容器 */
        [data-testid="stSidebar"] .stSelectbox {
            min-width: 100% !important;
        }
        
        [data-testid="stSidebar"] .stSelectbox > div {
            min-width: 100% !important;
        }
        
        /* 侧边栏selectbox下拉控件 */
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            min-width: 280px !important;
        }
        
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            min-width: 280px !important;
        }
        
        /* 侧边栏selectbox弹出菜单 */
        [data-testid="stSidebar"] [data-baseweb="popover"] {
            min-width: 320px !important;
            max-width: 450px !important;
        }
        
        /* 侧边栏selectbox下拉列表 */
        [data-testid="stSidebar"] [role="listbox"] {
            min-width: 320px !important;
            max-width: 450px !important;
        }
        
        /* 侧边栏selectbox选项 - 允许换行显示长文本 */
        [data-testid="stSidebar"] [role="option"] {
            white-space: normal !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
            padding: 8px 12px !important;
            line-height: 1.4 !important;
            min-height: 36px !important;
        }
        
        /* 确保下拉菜单的z-index足够高 */
        [data-testid="stSidebar"] [data-baseweb="popover"] {
            z-index: 9999 !important;
        }
        
        /* ==================== 输入框样式 ==================== */
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            border-radius: {theme.border_radius} !important;
            border: 1px solid {Colors.BORDER_LIGHT} !important;
            padding: 8px 12px !important;
            transition: all 0.2s ease !important;
        }}
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div:focus-within {{
            border-color: {Colors.PRIMARY} !important;
            box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1) !important;
        }}
        
        /* ==================== 滑块样式 ==================== */
        
        .stSlider > div > div > div > div {{
            background-color: {Colors.PRIMARY} !important;
        }}
        
        .stSlider > div > div > div > div > div {{
            background-color: {Colors.PRIMARY} !important;
        }}
        
        /* ==================== 复选框和单选框 ==================== */
        
        .stCheckbox > label {{
            font-size: {theme.font_size_body} !important;
            color: {Colors.TEXT_DARK} !important;
        }}
        
        .stRadio > label {{
            font-size: {theme.font_size_body} !important;
            color: {Colors.TEXT_DARK} !important;
        }}
        
        /* ==================== 扩展器样式 ==================== */
        
        .streamlit-expanderHeader {{
            background-color: {Colors.BG_LIGHT} !important;
            border-radius: {theme.border_radius} !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }}
        
        .streamlit-expanderHeader:hover {{
            background-color: {Colors.BG_INFO} !important;
        }}
        
        /* ==================== 侧边栏样式 ==================== */
        
        [data-testid="stSidebar"] {{
            background-color: {Colors.BG_LIGHT};
        }}
        
        [data-testid="stSidebar"] .stButton > button {{
            width: 100%;
        }}
        
        /* ==================== 自定义组件 ==================== */
        
        /* 状态徽章 */
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: {theme.font_size_small};
            font-weight: 600;
            margin: 2px;
        }}
        
        .status-badge.success {{
            background-color: {Colors.BG_SUCCESS};
            color: {Colors.SUCCESS};
            border: 1px solid {Colors.SUCCESS};
        }}
        
        .status-badge.warning {{
            background-color: {Colors.BG_WARNING};
            color: {Colors.WARNING};
            border: 1px solid {Colors.WARNING};
        }}
        
        .status-badge.danger {{
            background-color: {Colors.BG_DANGER};
            color: {Colors.DANGER};
            border: 1px solid {Colors.DANGER};
        }}
        
        /* 进度条容器 */
        .progress-container {{
            width: 100%;
            background-color: {Colors.BG_LIGHT};
            border-radius: 12px;
            overflow: hidden;
            height: 24px;
            position: relative;
            border: 1px solid {Colors.BORDER_LIGHT};
        }}
        
        .progress-bar {{
            height: 100%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .progress-label {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-size: {theme.font_size_small};
            font-weight: 600;
            color: {Colors.TEXT_DARK};
            z-index: 1;
        }}
        
        /* 警告框 */
        .alert-box {{
            padding: 12px 16px;
            border-radius: 4px;
            margin: 8px 0;
            border-left: 4px solid;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .alert-box.info {{
            background-color: {Colors.BG_INFO};
            border-color: {Colors.PRIMARY};
            color: {Colors.TEXT_DARK};
        }}
        
        .alert-box.success {{
            background-color: {Colors.BG_SUCCESS};
            border-color: {Colors.SUCCESS};
            color: {Colors.TEXT_DARK};
        }}
        
        .alert-box.warning {{
            background-color: {Colors.BG_WARNING};
            border-color: {Colors.WARNING};
            color: {Colors.TEXT_DARK};
        }}
        
        .alert-box.danger {{
            background-color: {Colors.BG_DANGER};
            border-color: {Colors.DANGER};
            color: {Colors.TEXT_DARK};
        }}
        
        /* ==================== 加载动画 ==================== */
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .spinner {{
            border: 3px solid {Colors.BG_LIGHT};
            border-top: 3px solid {Colors.PRIMARY};
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.3s ease;
        }}
        
        /* ==================== 响应式设计 ==================== */
        
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            
            .metric-card-value {{
                font-size: 24px !important;
            }}
            
            h1 {{
                font-size: 20px !important;
            }}
            
            h2 {{
                font-size: 16px !important;
            }}
        }}
        
        /* ==================== 打印样式 ==================== */
        
        @media print {{
            .stButton, .stDownloadButton {{
                display: none !important;
            }}
            
            .metric-card {{
                box-shadow: none !important;
                border: 1px solid {Colors.BORDER_LIGHT} !important;
            }}
        }}
        
        /* ==================== 滚动条美化 ==================== */
        
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {Colors.BG_LIGHT};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {Colors.BORDER_LIGHT};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {Colors.TEXT_MUTED};
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def create_section_header(title: str, emoji: str = "", description: str = ""):
    """
    创建美化的章节标题
    
    Args:
        title: 标题文本
        emoji: 标题图标
        description: 描述文本
    """
    header_html = f"""
    <div class="fade-in" style="margin: 24px 0 16px 0;">
        <h2 style="
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0;
            padding-bottom: 8px;
            border-bottom: 2px solid {Colors.BORDER_LIGHT};
        ">
            {f'<span style="font-size: 24px;">{emoji}</span>' if emoji else ''}
            <span>{title}</span>
        </h2>
        {f'<p style="color: {Colors.TEXT_MUTED}; margin-top: 8px; font-size: 14px;">{description}</p>' if description else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def create_divider(height: str = "1px", color: str = Colors.BORDER_LIGHT):
    """
    创建自定义分隔线
    
    Args:
        height: 分隔线高度
        color: 分隔线颜色
    """
    st.markdown(
        f'<hr style="height: {height}; background-color: {color}; border: none; margin: 24px 0;">',
        unsafe_allow_html=True
    )


def create_info_box(message: str, box_type: str = "info"):
    """
    创建信息提示框
    
    Args:
        message: 提示消息
        box_type: 类型 (info/success/warning/danger)
    """
    from .color_scheme import get_alert_box_html
    st.markdown(get_alert_box_html(message, box_type), unsafe_allow_html=True)


__all__ = [
    'inject_global_styles',
    'create_section_header',
    'create_divider',
    'create_info_box',
]
