# 🔧 可选依赖安装配置指南

本文档指导如何安装和配置 TA-Lib、Optuna 和 Kaggle 三个可选依赖。

---

## 1️⃣ TA-Lib 技术指标库

### 📋 功能说明
TA-Lib 提供150+技术指标计算函数，包括：
- 趋势指标：MA、EMA、MACD、ADX
- 动量指标：RSI、STOCH、CCI
- 波动率：ATR、Bollinger Bands
- 成交量：OBV、AD、ADOSC

### 🪟 Windows 安装方法

#### 方法一：使用预编译wheel（推荐）

```bash
# 1. 下载对应Python版本的whl文件
# 访问 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# 下载对应版本，例如：
# - Python 3.11 64位: TA_Lib-0.4.28-cp311-cp311-win_amd64.whl

# 2. 安装下载的whl文件
pip install TA_Lib-0.4.28-cp311-cp311-win_amd64.whl

# 3. 验证安装
python -c "import talib; print(talib.__version__)"
```

#### 方法二：通过conda安装

```bash
# 如果使用Anaconda环境
conda install -c conda-forge ta-lib
```

#### 方法三：从源码编译（高级）

需要预先安装：
- Visual Studio Build Tools
- CMake

```bash
# 下载ta-lib源码
git clone https://github.com/mrjbq7/ta-lib.git
cd ta-lib

# 编译安装
python setup.py install
```

### ✅ 验证安装

```python
import talib
import numpy as np

# 测试RSI计算
prices = np.random.randn(100)
rsi = talib.RSI(prices, timeperiod=14)
print(f"TA-Lib 安装成功! RSI: {rsi[-1]:.2f}")
```

### 💡 在麒麟系统中使用

```python
# 在因子工程中使用TA-Lib
from features.technical_indicators import TechnicalIndicators

# TA-Lib会自动被调用计算技术指标
indicators = TechnicalIndicators(use_talib=True)
features = indicators.calculate_all(df)
```

---

## 2️⃣ Optuna 超参数优化

### 📋 功能说明
Optuna 是自动化超参数优化框架，支持：
- 贝叶斯优化
- 多目标优化
- 剪枝策略
- 可视化优化过程

### 📦 安装

```bash
# 基础安装
pip install optuna

# 完整安装（含可视化）
pip install optuna[visualization]

# 验证安装
python -c "import optuna; print(optuna.__version__)"
```

### 🎯 快速开始

```python
import optuna

# 定义优化目标函数
def objective(trial):
    # LightGBM参数优化示例
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    # 训练模型并返回验证指标
    # model = train_model(params)
    # score = evaluate_model(model)
    # return score
    return 0.85  # 示例

# 创建study并优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 输出最佳参数
print("最佳参数:", study.best_params)
print("最佳分数:", study.best_value)
```

### 💡 在麒麟系统中使用

#### 1. 命令行方式

```bash
# 运行Optuna调优
python scripts/optuna_tuning.py \
    --model lightgbm \
    --n-trials 100 \
    --timeout 3600
```

#### 2. UI界面方式

在 **Qlib > 模型训练 > 参数优化** 标签页：
1. 选择模型类型
2. 设置优化目标（IC、ICIR、收益率等）
3. 配置搜索空间
4. 点击"开始优化"
5. 实时查看优化进度和最佳参数

#### 3. 代码集成

```python
from qlib_enhanced.model_zoo.optuna_tuner import OptunaModelTuner

# 创建调优器
tuner = OptunaModelTuner(
    model_name='lightgbm',
    objective_metric='ic',
    n_trials=100
)

# 执行优化
best_params = tuner.optimize(dataset)
print(f"最佳参数: {best_params}")

# 使用最佳参数训练最终模型
final_model = tuner.train_with_best_params(dataset)
```

### 📊 可视化优化过程

```python
import optuna

# 加载study
study = optuna.load_study(study_name='lightgbm_tuning', storage='sqlite:///optuna.db')

# 生成可视化图表
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)

# 优化历史
fig1 = plot_optimization_history(study)
fig1.show()

# 参数重要性
fig2 = plot_param_importances(study)
fig2.show()

# 参数关系
fig3 = plot_parallel_coordinate(study)
fig3.show()
```

---

## 3️⃣ Kaggle 配置

### 📋 功能说明
Kaggle集成支持：
- 下载Kaggle竞赛数据集
- 提交预测结果
- 查看排行榜
- 访问Kaggle Notebooks

### 🔑 获取API凭证

#### 步骤1: 登录Kaggle

访问 https://www.kaggle.com/ 并登录账户

#### 步骤2: 生成API Token

1. 点击右上角头像
2. 选择 **Settings**
3. 滚动到 **API** 部分
4. 点击 **Create New API Token**
5. 自动下载 `kaggle.json` 文件

### 📁 配置文件放置

#### Windows系统

```bash
# 创建.kaggle目录
mkdir %USERPROFILE%\.kaggle

# 移动kaggle.json到该目录
move Downloads\kaggle.json %USERPROFILE%\.kaggle\

# 检查文件
dir %USERPROFILE%\.kaggle\
```

完整路径示例：
```
C:\Users\Administrator\.kaggle\kaggle.json
```

#### Linux/Mac系统

```bash
# 创建.kaggle目录
mkdir -p ~/.kaggle

# 移动配置文件
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置权限（重要！）
chmod 600 ~/.kaggle/kaggle.json
```

### ✅ 验证配置

```bash
# 测试Kaggle CLI
kaggle competitions list

# 应该看到竞赛列表，表示配置成功
```

### 💡 在麒麟系统中使用

#### 1. 下载数据集

```bash
# 下载竞赛数据
kaggle competitions download -c <competition-name>

# 下载数据集
kaggle datasets download -d <dataset-name>
```

#### 2. UI界面方式

在 **数据管理 > Kaggle数据** 标签页：
1. 输入竞赛/数据集名称
2. 点击"下载"
3. 自动解压到 `data/kaggle/` 目录

#### 3. Python API

```python
from kaggle.api.kaggle_api_extended import KaggleApi

# 初始化API
api = KaggleApi()
api.authenticate()

# 下载数据集
api.competition_download_files('titanic', path='data/kaggle/')

# 提交预测
api.competition_submit('submission.csv', 'My submission', 'titanic')

# 查看排行榜
leaderboard = api.competition_leaderboard_view('titanic')
print(leaderboard)
```

### 🔒 安全提示

**重要**：`kaggle.json` 包含你的API密钥，请：
- ✅ 不要提交到Git仓库
- ✅ 不要分享给他人
- ✅ 定期更换（在Kaggle设置中重新生成）
- ✅ 设置文件权限为仅所有者可读（Linux/Mac: `chmod 600`）

在 `.gitignore` 中添加：
```gitignore
.kaggle/
kaggle.json
```

---

## 📊 安装状态检查

运行依赖检查脚本验证所有安装：

```bash
python scripts/check_dependencies.py
```

期望输出：
```
✅ ta-lib                         v0.4.28
✅ optuna                         v3.x.x
✅ kaggle                         v1.x.x
✅ Kaggle配置文件: C:\Users\Administrator\.kaggle\kaggle.json
```

---

## 🔧 故障排查

### TA-Lib安装失败

**问题**: "error: Microsoft Visual C++ 14.0 or greater is required"

**解决**:
1. 安装 Visual Studio Build Tools
2. 或使用预编译wheel文件（推荐）
3. 或使用conda安装

### Optuna导入错误

**问题**: "No module named 'optuna.visualization'"

**解决**:
```bash
pip install optuna[visualization]
```

### Kaggle认证失败

**问题**: "Could not find kaggle.json"

**解决**:
1. 确认文件位置正确
2. Windows: `%USERPROFILE%\.kaggle\kaggle.json`
3. Linux/Mac: `~/.kaggle/kaggle.json`
4. 检查文件权限（Linux/Mac需要chmod 600）

### Kaggle API限流

**问题**: "Rate limit exceeded"

**解决**:
- Kaggle有API调用频率限制
- 等待1小时后重试
- 或使用Web界面手动下载

---

## 📚 参考资源

### TA-Lib
- 官方文档: https://ta-lib.org/
- Python包文档: https://mrjbq7.github.io/ta-lib/
- 预编译wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### Optuna
- 官方网站: https://optuna.org/
- 文档: https://optuna.readthedocs.io/
- 示例: https://github.com/optuna/optuna-examples

### Kaggle
- API文档: https://github.com/Kaggle/kaggle-api
- 账户设置: https://www.kaggle.com/settings
- 竞赛列表: https://www.kaggle.com/competitions

---

## ✅ 快速安装脚本

将以下内容保存为 `install_optional_deps.bat` (Windows) 或 `install_optional_deps.sh` (Linux/Mac):

### Windows (install_optional_deps.bat)

```batch
@echo off
echo ========================================
echo 安装可选依赖
echo ========================================

echo.
echo [1/3] 安装 Optuna...
pip install optuna[visualization]

echo.
echo [2/3] 安装 Kaggle CLI...
pip install --upgrade kaggle

echo.
echo [3/3] TA-Lib 安装提示
echo 请访问以下网址下载预编译wheel:
echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo 然后运行: pip install 下载的whl文件名

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步:
echo 1. 下载并安装 TA-Lib wheel文件
echo 2. 配置 Kaggle API (访问 https://www.kaggle.com/settings)
echo 3. 运行: python scripts/check_dependencies.py
pause
```

### Linux/Mac (install_optional_deps.sh)

```bash
#!/bin/bash

echo "========================================"
echo "安装可选依赖"
echo "========================================"

echo ""
echo "[1/3] 安装 Optuna..."
pip install optuna[visualization]

echo ""
echo "[2/3] 安装 Kaggle CLI..."
pip install --upgrade kaggle

echo ""
echo "[3/3] 安装 TA-Lib..."
# 尝试通过conda安装
if command -v conda &> /dev/null; then
    echo "检测到conda，尝试通过conda安装..."
    conda install -c conda-forge ta-lib -y
else
    echo "未检测到conda，尝试通过pip安装..."
    pip install TA-Lib
fi

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "1. 配置 Kaggle API (访问 https://www.kaggle.com/settings)"
echo "2. 运行: python scripts/check_dependencies.py"
```

运行脚本：
```bash
# Windows
install_optional_deps.bat

# Linux/Mac
chmod +x install_optional_deps.sh
./install_optional_deps.sh
```

---

**完成后，重新运行依赖检查确认安装成功！** 🎉
