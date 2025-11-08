"""
快速测试策略优化闭环UI
========================

运行方式:
    streamlit run test_strategy_loop_ui.py

Author: Qilin Stack Team
Date: 2024-11-08
"""

import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="策略优化闭环测试",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 导入UI组件
try:
    from web.components.strategy_loop_ui import render_strategy_loop_ui
    
    # 渲染界面
    render_strategy_loop_ui()
    
except Exception as e:
    st.error(f"❌ 加载失败: {e}")
    
    st.markdown("""
    ### 🔧 解决方法:
    
    1. 确保已安装依赖:
    ```bash
    pip install streamlit pandas plotly
    ```
    
    2. 确保策略闭环模块已创建:
    - `strategy/strategy_feedback_loop.py`
    - `web/components/strategy_loop_ui.py`
    
    3. 重新运行:
    ```bash
    streamlit run test_strategy_loop_ui.py
    ```
    """)
    
    import traceback
    with st.expander("查看错误详情"):
        st.code(traceback.format_exc())
