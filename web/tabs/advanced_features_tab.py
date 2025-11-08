"""
高级功能标签页
集成Phase 4的模拟交易、策略回测、数据导出功能
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

# 导入Phase 4高级功能
try:
    import sys
    from pathlib import Path
    
    # 确保项目根目录在路径中
    project_root = Path(__file__).parent.parent.parent
    project_root_str = str(project_root.resolve())
    
    # 添加到sys.path的最前面
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # 同时添加web目录
    web_dir = str((project_root / 'web').resolve())
    if web_dir not in sys.path:
        sys.path.insert(0, web_dir)
    
    from components.advanced_features import (
        SimulatedTrading,
        StrategyBacktest,
        ExportManager,
        render_simulated_trading,
        render_backtest,
        render_export
    )
    from components.color_scheme import Colors, Emojis
    from components.ui_styles import create_section_header
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    logger.warning(f"导入高级功能模块失败: {e}")
    logger.debug(f"详细错误:\n{traceback.format_exc()}")
    SimulatedTrading = None
    StrategyBacktest = None
    ExportManager = None
    Colors = None
    Emojis = None
    ADVANCED_FEATURES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# 导入策略优化闭环UI
try:
    from components.strategy_loop_ui import render_strategy_loop_ui
    STRATEGY_LOOP_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"策略优化闭环UI导入失败: {e}")
    STRATEGY_LOOP_AVAILABLE = False


def render_advanced_features_tab():
    """渲染高级功能标签页"""
    
    if not ADVANCED_FEATURES_AVAILABLE:
        st.warning("⚠️ 高级功能模块未完全加载，显示简化版本")
        if 'IMPORT_ERROR' in globals():
            with st.expander("🔍 查看错误详情"):
                st.error(f"导入错误: {IMPORT_ERROR}")
                st.info("💡 请检查以下模块是否正常：")
                st.code("""
web/components/advanced_features.py
web/components/color_scheme.py
web/components/loading_cache.py
web/components/ui_styles.py
                """, language="text")
        render_simplified_advanced_features()
        return
    
    # 页面标题
    st.title("🚀 高级功能")
    st.markdown("---")
    
    # 创建子标签页
    emoji_money = Emojis.MONEY if Emojis else "💰"
    emoji_chart = Emojis.CHART if Emojis else "📈"
    emoji_export = Emojis.EXPORT if Emojis else "📤"
    
    tabs = st.tabs([
        "🔥 策略优化闭环",
        f"{emoji_money} 模拟交易",
        f"{emoji_chart} 策略回测",
        f"{emoji_export} 数据导出"
    ])
    
    # Tab 1: 策略优化闭环
    with tabs[0]:
        if STRATEGY_LOOP_AVAILABLE:
            try:
                render_strategy_loop_ui()
            except Exception as e:
                st.error(f"策略优化闭环加载失败: {str(e)}")
                st.info("💡 策略优化闭环是麒麟系统的核心创新功能，整合了AI因子挖掘、策略构建、回测验证和反馈优化的完整闭环。")
                import traceback
                with st.expander("🔍 查看详细错误"):
                    st.code(traceback.format_exc())
        else:
            st.error("❌ 策略优化闭环模块未安装")
            st.warning("🛠️ **最可能的原因**: pandas/pyarrow 版本冲突")
            
            with st.expander("🔧 快速修复指引", expanded=True):
                st.markdown("""
                ### ✅ 解决方案
                
                在命令行执行以下命令:
                
                ```bash
                # 方法1: 重新安装 (推荐)
                pip uninstall pyarrow pandas -y
                pip install pandas pyarrow
                
                # 方法2: 升级
                pip install --upgrade pandas pyarrow
                
                # 方法3: conda用户
                conda install pandas pyarrow -c conda-forge
                ```
                
                ### 🧪 验证修复
                
                ```bash
                python -c "import pandas as pd; print(f'✅ pandas {pd.__version__} 正常工作')"
                ```
                
                ### 📝 详细文档
                
                查看完整修复指南: `fix_pandas_pyarrow.md`
                """)
            
            st.markdown("""
            ---
            
            ### 🔥 策略优化闭环系统
            
            **核心功能**：整合麒麟系统的AI因子挖掘、策略构建、回测验证和反馈优化，形成完整闭环。
            
            **7阶段闭环流程**：
            1. 🧠 **AI因子挖掘** - RD-Agent智能因子发现
            2. 🏗️ **策略构建** - 组合因子 + 交易规则
            3. 📊 **回测验证** - Qlib历史数据验证
            4. 💼 **模拟交易** - 可选实盘模拟
            5. 📈 **性能评估** - 多维度指标分析
            6. 🔄 **反馈生成** - 智能问题诊断 + 优化建议
            7. 🎯 **目标判定** - 达标终止，未达标继续循环
            
            **典型应用场景**：
            - 寻找A股动量因子 → 年化收益率从12%提升到18%
            - 优化价值投资策略 → 夏普比率从0.8提升到1.5
            - 发现反转信号 → 最大回撤从-25%降低到-15%
            
            **所需依赖**：`web/components/strategy_loop_ui.py`
            """)
    
    # Tab 2: 模拟交易
    with tabs[1]:
        try:
            trading = SimulatedTrading()
            render_simulated_trading(trading)
        except Exception as e:
            st.error(f"模拟交易功能加载失败: {str(e)}")
            st.info("💡 提示: 请检查session_state是否正确初始化")
    
    # Tab 3: 策略回测  
    with tabs[2]:
        try:
            backtest = StrategyBacktest()
            render_backtest(backtest)
        except Exception as e:
            st.error(f"策略回测功能加载失败: {str(e)}")
    
    # Tab 4: 数据导出
    with tabs[3]:
        try:
            st.markdown("### 📤 数据导出")
            
            # 示例数据（实际使用时应从真实数据源获取）
            sample_df = pd.DataFrame({
                'symbol': ['000001', '000002', '000003'],
                'name': ['平安银行', '万科A', '国农科技'],
                'price': [10.5, 8.2, 25.6],
                'change': [2.3, -1.5, 5.8]
            })
            
            sample_stats = {
                'total_count': 3,
                'avg_price': 14.77,
                'positive_count': 2
            }
            
            st.info("💡 当前显示示例数据。在实际使用中，这里会显示您的候选股和统计信息。")
            
            # 显示示例数据
            st.dataframe(sample_df, use_container_width=True)
            
            # 导出功能
            render_export(sample_df, sample_stats)
            
        except Exception as e:
            st.error(f"数据导出功能加载失败: {str(e)}")


def render_simplified_advanced_features():
    """渲染简化版高级功能（当完整模块不可用时）"""
    st.title("🚀 高级功能（简化版）")
    st.markdown("---")
    
    # 显示调试信息
    with st.expander("🔧 调试信息", expanded=False):
        st.markdown("### 模块导入状态")
        import sys
        from pathlib import Path
        
        st.write(f"**Python 版本**: {sys.version}")
        st.write(f"**当前文件**: {__file__}")
        st.write(f"**项目根目录**: {Path(__file__).parent.parent.parent}")
        
        st.markdown("### sys.path 前5条")
        for i, p in enumerate(sys.path[:5], 1):
            st.code(f"{i}. {p}")
        
        st.markdown("### 尝试手动导入")
        try:
            project_root = Path(__file__).parent.parent.parent
            project_root_str = str(project_root.resolve())
            web_dir = str((project_root / 'web').resolve())
            
            st.write(f"**添加路径**: {project_root_str}")
            st.write(f"**添加web目录**: {web_dir}")
            
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
            if web_dir not in sys.path:
                sys.path.insert(0, web_dir)
            
            # 尝试多种导入方式
            st.markdown("#### 方法1: 使用 web.components")
            try:
                from web.components.advanced_features import SimulatedTrading
                st.success("✅ web.components.advanced_features 导入成功！")
            except Exception as e1:
                st.error(f"❌ 失败: {e1}")
                
                st.markdown("#### 方法2: 直接从 components 导入")
                try:
                    from components.advanced_features import SimulatedTrading
                    st.success("✅ components.advanced_features 导入成功！")
                except Exception as e2:
                    st.error(f"❌ 失败: {e2}")
                    
                    st.markdown("#### 详细错误")
                    import traceback
                    st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ 调试失败: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    tabs = st.tabs([
        "💰 模拟交易",
        "📈 策略回测",
        "📤 数据导出"
    ])
    
    with tabs[0]:
        st.markdown("### 💰 模拟交易")
        st.info("💡 模拟交易功能允许您在不使用真实资金的情况下测试策略。")
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("初始资金", value=100000, step=10000, key="sim_capital")
            st.selectbox("交易策略", ["一进二策略", "缠论策略", "自定义"], key="sim_strategy")
        
        with col2:
            st.date_input("开始日期", key="sim_start")
            st.date_input("结束日期", key="sim_end")
        
        if st.button("🚀 开始模拟交易", type="primary"):
            st.success("✅ 模拟交易已启动！")
            st.info("🚧 完整版本将显示实时交易结果和统计数据。")
    
    with tabs[1]:
        st.markdown("### 📈 策略回测")
        st.info("💡 策略回测用于验证历史数据上的策略表现。")
        
        st.markdown("""
        **回测流程**：
        1. 选择回测策略
        2. 设置回测参数（时间范围、初始资金等）
        3. 运行回测
        4. 查看结果分析
        
        **关键指标**：
        - 总收益率
        - 年化收益率
        - 最大回撤
        - 夏普比率
        - 胜率
        """)
        
        if st.button("📈 运行回测", type="primary"):
            st.success("✅ 回测已完成！")
            st.info("🚧 完整版本将显示详细的回测报告和图表。")
    
    with tabs[2]:
        st.markdown("### 📤 数据导出")
        
        # 示例数据
        sample_df = pd.DataFrame({
            'symbol': ['000001', '000002', '000003'],
            'name': ['平安银行', '万科A', '国农科技'],
            'price': [10.5, 8.2, 25.6],
            'change': [2.3, -1.5, 5.8]
        })
        
        st.info("💡 当前显示示例数据。在实际使用中，这里会显示您的候选股和统计信息。")
        st.dataframe(sample_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 CSV 格式"):
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    "⬇️ 下载 CSV",
                    csv,
                    "export.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("📊 Excel 格式"):
                st.info("🚧 Excel 导出功能开发中...")
        
        with col3:
            if st.button("📝 JSON 格式"):
                json_str = sample_df.to_json(orient='records', indent=2)
                st.download_button(
                    "⬇️ 下载 JSON",
                    json_str,
                    "export.json",
                    "application/json"
                )


# 测试入口
if __name__ == "__main__":
    st.set_page_config(
        page_title="高级功能测试",
        page_icon="🚀",
        layout="wide"
    )
    
    render_advanced_features_tab()
