@echo off
echo ========================================
echo 麒麟量化系统 - 可选依赖安装
echo ========================================

echo.
echo [1/2] 安装 Optuna (超参数优化)...
pip install "optuna[visualization]"

echo.
echo [2/2] 升级 Kaggle CLI...
pip install --upgrade kaggle

echo.
echo ========================================
echo [TA-Lib 安装说明]
echo ========================================
echo.
echo TA-Lib 需要预编译wheel文件（Windows）：
echo.
echo 步骤1: 访问以下网址
echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo.
echo 步骤2: 下载对应Python版本的whl文件
echo - 当前Python版本: 
python --version
echo - 需要下载: TA_Lib-0.4.28-cp311-cp311-win_amd64.whl (如果是Python 3.11)
echo.
echo 步骤3: 安装下载的whl文件
echo pip install 下载的whl文件路径
echo.
echo 或者使用conda安装:
echo conda install -c conda-forge ta-lib
echo.

echo ========================================
echo [Kaggle 配置说明]
echo ========================================
echo.
echo 步骤1: 访问 https://www.kaggle.com/settings
echo 步骤2: 在API部分点击 "Create New API Token"
echo 步骤3: 下载的kaggle.json文件放到:
echo        %USERPROFILE%\.kaggle\kaggle.json
echo.
echo 快速配置命令:
echo mkdir %USERPROFILE%\.kaggle
echo move Downloads\kaggle.json %USERPROFILE%\.kaggle\
echo.

echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步:
echo 1. 按照上述说明安装 TA-Lib
echo 2. 配置 Kaggle API
echo 3. 运行验证: python scripts\check_dependencies.py
echo 4. 查看详细文档: docs\OPTIONAL_DEPENDENCIES_SETUP.md
echo.
pause
