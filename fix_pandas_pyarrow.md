# 🔧 修复 pandas/pyarrow 依赖问题

## ❌ 问题现象

启动Dashboard后看到错误:
```
❌ 策略优化闭环模块未安装,请检查依赖
```

或在命令行看到:
```
AttributeError: module 'pyarrow' has no attribute '__version__'
```

## 🎯 问题原因

pandas和pyarrow版本冲突,导致pandas无法正常导入。

## ✅ 解决方案

### 方法1: 重新安装pandas和pyarrow (推荐)

```bash
# 卸载旧版本
pip uninstall pyarrow pandas -y

# 安装新版本
pip install pandas pyarrow

# 或指定版本
pip install pandas==2.1.4 pyarrow==14.0.1
```

### 方法2: 升级现有版本

```bash
pip install --upgrade pandas pyarrow
```

### 方法3: 使用conda (如果用conda环境)

```bash
conda install pandas pyarrow -c conda-forge
```

## 🧪 验证修复

运行以下命令验证pandas能否正常导入:

```bash
python -c "import pandas as pd; print(f'✅ pandas {pd.__version__} 正常工作')"
```

**预期输出**:
```
✅ pandas 2.1.4 正常工作
```

## 🚀 重新启动Dashboard

修复完成后,重新启动Dashboard:

```bash
# Windows
start_dashboard.bat

# Linux/Mac
bash start_dashboard.sh

# 或手动
streamlit run web/unified_dashboard.py
```

然后访问: `http://localhost:8501` → 🚀 高级功能 → 🔥 策略优化闭环

## 📝 详细诊断

如果问题仍然存在,运行详细诊断:

```bash
python -c "
import sys
print('Python版本:', sys.version)
print()

try:
    import pandas as pd
    print('✅ pandas:', pd.__version__)
except Exception as e:
    print('❌ pandas导入失败:', e)

try:
    import pyarrow as pa
    print('✅ pyarrow:', pa.__version__)
except Exception as e:
    print('❌ pyarrow导入失败:', e)

try:
    import streamlit as st
    print('✅ streamlit:', st.__version__)
except Exception as e:
    print('❌ streamlit导入失败:', e)
"
```

## 🆘 其他常见问题

### Q: 我用的是虚拟环境,怎么办?

A: 确保激活了正确的虚拟环境:

```bash
# conda
conda activate qilin

# venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

然后再执行修复命令。

### Q: 修复后还是报错?

A: 尝试完全重建环境:

```bash
# 1. 创建新环境
conda create -n qilin_new python=3.8 -y
conda activate qilin_new

# 2. 安装依赖
cd G:\test\qilin_stack
pip install -r requirements.txt

# 3. 启动Dashboard
streamlit run web/unified_dashboard.py
```

### Q: 我不想修复pandas,能用吗?

A: 可以!策略优化闭环的其他说明文档仍然可以查看,只是无法使用交互式数据上传和示例数据功能。你可以:
- 查看"📖 使用说明"tab了解功能
- 查看文档: `docs/STRATEGY_LOOP_INTEGRATION.md`
- 或直接使用Python API (不需要Web UI)

## 📚 相关资源

- pandas安装文档: https://pandas.pydata.org/docs/getting_started/install.html
- pyarrow安装文档: https://arrow.apache.org/docs/python/install.html
- 麒麟系统文档: `docs/STRATEGY_LOOP_INTEGRATION.md`
