# Web 界面依赖安装指南

## 问题描述

运行 Web 界面时可能遇到的依赖问题：

### 常见错误

#### 1. ModuleNotFoundError: No module named 'shap'
```
Traceback (most recent call last):
  File "web/unified_dashboard.py", line 2106
    from web.components.realistic_backtest_page import show_realistic_backtest_page
  File "web/components/realistic_backtest_page.py", line 21
    from ml.model_explainer import LimitUpModelExplainer
  File "ml/model_explainer.py", line 17
    import shap
ModuleNotFoundError: No module named 'shap'
```

## 快速解决方案

### 方案1: 使用安装脚本（推荐）

```powershell
# 在项目根目录运行
.\install_web_deps.ps1
```

### 方案2: 手动安装

#### 核心依赖（必需）
```bash
pip install streamlit pandas numpy plotly
```

#### 写实回测依赖（可选）
```bash
pip install shap
```

## SHAP 安装详解

### 什么是 SHAP？
SHAP (SHapley Additive exPlanations) 是一个用于解释机器学习模型的库，在写实回测功能中用于：
- 解释模型预测结果
- 展示特征重要性
- 提供可视化分析

### 为什么安装可能失败？

SHAP 依赖于 C++ 编译器，在 Windows 上可能遇到问题：
- 缺少 Visual C++ Build Tools
- 编译环境配置不正确

### 解决方案

#### Windows 用户

**选项A: 安装 Visual C++ Build Tools**（推荐）

1. 下载并安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. 在安装时选择 "C++ build tools"
3. 重启后运行：
   ```bash
   pip install shap
   ```

**选项B: 使用预编译版本**

```bash
# 升级 pip
pip install --upgrade pip

# 使用无缓存安装
pip install shap --no-cache-dir

# 如果还是失败，尝试指定版本
pip install shap==0.41.0
```

**选项C: 使用 conda**（如果使用 Anaconda）

```bash
conda install -c conda-forge shap
```

#### macOS/Linux 用户

通常直接安装即可：
```bash
pip install shap
```

如果失败，确保安装了编译工具：
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum install gcc gcc-c++
```

## 依赖分级说明

### 一级依赖（必需）
这些是运行基础界面的必需依赖：
```bash
streamlit      # Web 框架
pandas         # 数据处理
numpy          # 数值计算
plotly         # 图表可视化
```

### 二级依赖（推荐）
用于增强功能：
```bash
shap           # 模型解释（写实回测）
redis          # 实时数据缓存
websocket      # 实时行情推送
```

### 三级依赖（可选）
用于特定高级功能：
```bash
akshare        # 数据源
pyqlib         # Qlib 集成
scikit-learn   # 机器学习
lightgbm       # 模型训练
xgboost        # 模型训练
```

## 功能降级策略

当某些依赖缺失时，系统会自动降级：

### 缺少 shap
- ❌ 写实回测的 SHAP 解释功能不可用
- ✅ 其他回测功能正常
- ✅ 监控、交易功能正常

界面会显示友好提示：
```
🚧 缺少 SHAP 库（用于模型解释）

安装方法：
pip install shap
```

### 缺少 redis
- ❌ 实时数据缓存不可用
- ✅ 使用模拟数据替代
- ✅ 基础功能正常

### 缺少其他模块
系统会：
1. 捕获导入错误
2. 显示友好提示
3. 提供安装建议
4. 功能优雅降级

## 验证安装

### 检查已安装的包
```bash
pip list | findstr "streamlit pandas numpy plotly shap"
```

### 测试 Web 界面
```bash
streamlit run web/unified_dashboard.py
```

### 检查特定功能

#### 1. 核心界面
- 打开浏览器访问 `http://localhost:8501`
- 检查侧边栏是否正常显示
- 检查主界面标签页是否可点击

#### 2. 写实回测
- 进入 `🏠 Qilin监控` → `📖 写实回测`
- 如果显示错误，按提示安装依赖
- 安装后**重启应用**

## 常见问题

### Q1: 安装 shap 很慢？
A: 是的，SHAP 需要编译 C++ 代码，可能需要 5-10 分钟。

### Q2: 安装 shap 失败怎么办？
A: 
1. 检查是否有 C++ 编译器
2. 尝试使用 conda 安装
3. 或者先不安装，使用其他功能

### Q3: 必须安装所有依赖吗？
A: 不必须。核心依赖（streamlit, pandas, numpy, plotly）是必需的，其他依赖根据需要安装。

### Q4: 安装后还是报错？
A: 
1. 重启应用
2. 清除 Python 缓存：
   ```bash
   Remove-Item -Recurse -Force web\__pycache__
   Remove-Item -Recurse -Force web\pages\__pycache__
   Remove-Item -Recurse -Force web\components\__pycache__
   ```
3. 检查导入路径是否正确

### Q5: 如何跳过某个功能？
A: 系统已经实现了容错机制，缺失的功能会自动降级，不影响其他功能使用。

## 完整依赖列表

### requirements-web.txt
可以创建一个专门的 Web 依赖文件：

```txt
# 核心依赖（必需）
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.21.0
plotly>=5.14.0

# 可选依赖
shap>=0.41.0          # 模型解释
redis>=4.5.0          # 实时缓存
websocket-client      # 实时推送
akshare>=1.8.0        # 数据源
```

安装方式：
```bash
pip install -r requirements-web.txt
```

## 问题排查流程

```
1. 运行应用
   ↓
2. 遇到错误？
   ↓
3. 查看错误信息
   ↓
4. 是 ModuleNotFoundError？
   ↓ 是
5. 安装缺失的模块
   pip install <module_name>
   ↓
6. 清除缓存（可选）
   Remove-Item -Recurse __pycache__
   ↓
7. 重启应用
   ↓
8. 问题解决 ✅
```

## 相关文档

- `requirements.txt` - 完整项目依赖
- `docs/WEB_FIX_SUMMARY.md` - Web 界面修复总结
- `docs/SIDEBAR_NAV_FIX.md` - 侧边栏导航修复
- `install_web_deps.ps1` - 自动安装脚本

## 技术支持

如果遇到无法解决的问题：
1. 查看详细错误信息（点击"🔍 查看详细错误"）
2. 检查 Python 版本（推荐 Python 3.8+）
3. 检查虚拟环境是否激活
4. 尝试在新的虚拟环境中安装

---

**最后更新**: 2025-10-30  
**适用版本**: v3.0+
