@echo off
REM 麒麟量化系统 - 测试运行脚本 (Windows)
REM 

echo ======================================
echo 麒麟量化系统 - 测试套件
echo ======================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python,请先安装Python
    exit /b 1
)

REM 检查pytest是否安装
python -c "import pytest" >nul 2>&1
if errorlevel 1 (
    echo 正在安装测试依赖...
    pip install pytest pytest-cov pytest-asyncio pytest-mock
)

echo.
echo 选择测试模式:
echo 1. 运行所有测试
echo 2. 只运行单元测试
echo 3. 只运行集成测试
echo 4. 只运行配置测试
echo 5. 只运行Agent测试
echo 6. 运行测试并生成覆盖率报告
echo 7. 运行快速测试(跳过慢速测试)
echo.

set /p choice="请输入选项 (1-7): "

if "%choice%"=="1" (
    echo 运行所有测试...
    pytest tests/ -v
) else if "%choice%"=="2" (
    echo 运行单元测试...
    pytest tests/unit/ -v -m unit
) else if "%choice%"=="3" (
    echo 运行集成测试...
    pytest tests/integration/ -v -m integration
) else if "%choice%"=="4" (
    echo 运行配置测试...
    pytest tests/unit/test_config.py -v
) else if "%choice%"=="5" (
    echo 运行Agent测试...
    pytest tests/ -v -m agents
) else if "%choice%"=="6" (
    echo 运行测试并生成覆盖率报告...
    pytest tests/ -v --cov=app --cov=config --cov-report=html --cov-report=term
    echo.
    echo 覆盖率报告已生成: htmlcov/index.html
) else if "%choice%"=="7" (
    echo 运行快速测试...
    pytest tests/ -v -m "not slow"
) else (
    echo 无效选项,退出
    exit /b 1
)

echo.
echo ======================================
echo 测试完成!
echo ======================================
pause
