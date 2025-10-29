#!/bin/bash
# 麒麟涨停板选股系统 - Dashboard启动脚本
# 启动前请确保已安装依赖: pip install streamlit matplotlib pandas numpy

echo "========================================"
echo "麒麟涨停板选股系统 Web Dashboard"
echo "========================================"
echo ""

# 检查streamlit是否已安装
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "[错误] 未检测到 streamlit，正在安装依赖..."
    pip3 install streamlit matplotlib pandas numpy
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败，请手动运行: pip3 install streamlit matplotlib pandas numpy"
        exit 1
    fi
    echo "[成功] 依赖安装完成！"
    echo ""
fi

echo "[启动] 正在启动统一Dashboard..."
echo "[访问] 浏览器将自动打开 http://localhost:8501"
echo "[功能] 涨停板监控位于: Qlib → 数据管理 → 🎯涨停板监控"
echo "[退出] 按 Ctrl+C 停止服务"
echo ""

# 启动streamlit
streamlit run web/unified_dashboard.py
