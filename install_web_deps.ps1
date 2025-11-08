# 麒麟堆栈 Web 界面依赖安装脚本
# PowerShell 脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "麒麟堆栈 Web 界面依赖安装" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 核心依赖
Write-Host "1️⃣ 安装核心依赖..." -ForegroundColor Green
$core_deps = @(
    "streamlit",
    "pandas",
    "numpy",
    "plotly"
)

foreach ($dep in $core_deps) {
    Write-Host "   安装 $dep..." -ForegroundColor Yellow
    pip install $dep --quiet
}

Write-Host "✅ 核心依赖安装完成" -ForegroundColor Green
Write-Host ""

# 可选依赖
Write-Host "2️⃣ 安装可选依赖（用于高级功能）..." -ForegroundColor Green
Write-Host "   这可能需要几分钟..." -ForegroundColor Yellow
Write-Host ""

# SHAP - 用于模型解释
Write-Host "   正在安装 SHAP（模型解释库）..." -ForegroundColor Yellow
Write-Host "   注意: SHAP 需要 C++ 编译器，安装可能较慢" -ForegroundColor Gray
try {
    pip install shap --quiet 2>$null
    Write-Host "   ✅ SHAP 安装成功" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ SHAP 安装失败，写实回测功能可能受限" -ForegroundColor Red
    Write-Host "   请手动运行: pip install shap" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 现在可以运行:" -ForegroundColor Cyan
Write-Host "   streamlit run web/unified_dashboard.py" -ForegroundColor Yellow
Write-Host ""
