@echo off
REM 麒麟量化交易平台 - 统一Dashboard启动脚本
REM 启动前请确保已安装依赖: pip install streamlit matplotlib pandas numpy

echo ========================================
echo 麒麟量化平台 - 统一控制中心
echo ========================================
echo.

REM 检查streamlit是否已安装
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未检测到 streamlit，正在安装依赖...
    pip install streamlit matplotlib pandas numpy
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败，请手动运行: pip install streamlit matplotlib pandas numpy
        pause
        exit /b 1
    )
    echo [成功] 依赖安装完成！
    echo.
)

echo [启动] 正在启动统一Dashboard...
echo [访问] 浏览器将自动打开 http://localhost:8501
echo [功能] 涨停板监控位于: Qlib → 数据管理 → 🎯涨停板监控
echo [退出] 按 Ctrl+C 停止服务
echo.

REM 启动streamlit
streamlit run web\unified_dashboard.py

pause
