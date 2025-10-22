# PowerShell测试运行脚本

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "麒麟量化系统测试套件" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

# 检查Python环境
Write-Host "`n检查Python环境..." -ForegroundColor Yellow
python --version
pip list | Select-String "pytest" | Out-Null
if ($LASTEXITCODE -ne 0) {
    pip install pytest pytest-asyncio pytest-cov pytest-xdist
}

# 创建日志目录
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "htmlcov" | Out-Null

# 运行测试
Write-Host "`n========== 1. 单元测试 ==========" -ForegroundColor Yellow
pytest tests/unit/ -v --tb=short -m "not slow"

Write-Host "`n========== 2. 集成测试 ==========" -ForegroundColor Yellow
pytest tests/integration/ -v --tb=short -m integration

Write-Host "`n========== 3. MLOps测试 ==========" -ForegroundColor Yellow
pytest tests/unit/test_mlops.py -v --tb=short -m mlops

Write-Host "`n========== 4. 监控测试 ==========" -ForegroundColor Yellow
pytest tests/unit/test_monitoring.py -v --tb=short -m monitoring

Write-Host "`n========== 5. 覆盖率报告 ==========" -ForegroundColor Yellow
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing --cov-report=xml

# 检查覆盖率
$coverage = coverage report | Select-String "TOTAL" | ForEach-Object { $_.ToString() -match "\s+(\d+)%" | Out-Null; $matches[1] }
Write-Host "`n========== 测试总结 ==========" -ForegroundColor Yellow
Write-Host "代码覆盖率: $coverage%"

if ([int]$coverage -ge 80) {
    Write-Host "✓ 覆盖率达标 (>= 80%)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "✗ 覆盖率不达标 (< 80%)" -ForegroundColor Red
    Write-Host "请查看 htmlcov/index.html 了解详情" -ForegroundColor Yellow
    exit 1
}
