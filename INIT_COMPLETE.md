# 🎉 麒麟量化系统初始化完�?

## �?已完成的初始化步�?

### 1. 环境检�?
- �?Python 3.11.7 (符合要求 3.9-3.11)
- �?G盘可用空�? 224.27 GB
- �?PowerShell 执行策略已配�?

### 2. 虚拟环境
- �?虚拟环境目录: `G:\test\qilin_stack\.qilin`
- �?pip 已升级至 25.3
- �?虚拟环境已激�?

### 3. 依赖安装
已安�?200+ 个包,包括:
- �?**量化框架**: pyqlib (qlib), empyrical
- �?**数据科学**: numpy, pandas, scipy
- �?**机器学习**: scikit-learn, lightgbm, xgboost
- �?**MLOps**: mlflow, mlflow-skinny  
- �?**数据�?*: tushare, akshare, yfinance, ccxt
- �?**Web框架**: fastapi, uvicorn
- �?**数据�?*: pymongo, redis, sqlalchemy, asyncpg
- �?**监控**: prometheus-client, loguru, sentry-sdk
- �?**测试**: pytest 及相关插�?
- �?**开发工�?*: black, flake8, mypy, isort
- �?**Jupyter**: jupyterlab, ipython, ipywidgets

**注意**: 深度学习框架 (torch, tensorflow) 暂未安装,按需后续补充

### 4. 配置文件与目�?
- �?`config.yaml` - 系统配置文件(�?config.example.yaml 复制)
- �?`logs/` - 日志目录
- �?`data/` - 数据目录
- �?`reports/` - 报告目录
- �?`workspace/` - 工作空间目录

### 5. Qlib 数据准备
**状�?*: ⏸️ 待手动下�?

由于 pyqlib 包不包含自动下载功能,您需要手动准备数�?

**选项 A (推荐): 使用 AkShare 在线数据**
- 无需下载本地数据
- 实时获取A股数�?
- 配置已就�?可直接使�?

**选项 B: 下载 Qlib 预处理数�?*
- 详见 `QLIB_DATA_GUIDE.md` 文档
- 数据大小: �?12-20GB
- 目标路径: `C:\Users\Administrator\.qlib\qlib_data\cn_data`

## 🚀 快速开�?

### 1. 激活虚拟环�?
```powershell
.\.qilin\Scripts\Activate.ps1
```

### 2. 验证安装
```powershell
# 检查依�?
python -c "import qlib, pandas, numpy; print('环境正常')"

# 查看项目结构
tree /F /A
```

### 3. 配置数据�?
编辑 `config.yaml`:
```yaml
data:
  # 使用 AkShare 免费数据�?推荐)
  akshare:
    enabled: true
  
  # 或使�?Tushare(需�?token)
  tushare:
    token: "your_tushare_token_here"
```

### 4. 运行示例
```powershell
# 方式1: 使用在线数据快速测�?
python quickstart.py

# 方式2: 运行回测
python main.py --mode backtest --start_date 2024-01-01 --end_date 2024-12-31

# 方式3: 启动 Dashboard
python start_web.py
```

## 📚 重要文档

- **README.md** - 项目总览和使用指�?
- **QLIB_DATA_GUIDE.md** - Qlib 数据准备指南
- **config.example.yaml** - 配置文件示例
- **requirements.txt** - 依赖列表(已修�?

## ⚙️ 可选后续步�?

### 安装深度学习框架
```powershell
# CPU �?PyTorch
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# TensorFlow (CPU)
python -m pip install tensorflow
```

### 固化依赖快照
```powershell
python -m pip freeze > requirements.lock.txt
```

### 配置国内镜像(加速后续安�?
```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🛠�?常用命令

### 虚拟环境管理
```powershell
# 激�?
.\.qilin\Scripts\Activate.ps1

# 退�?
deactivate

# 查看已安装包
pip list

# 更新�?
pip install --upgrade package_name
```

### 代码质量检�?
```powershell
# 格式化代�?
black .

# 代码检�?
flake8 .

# 类型检�?
mypy .

# 排序导入
isort .
```

### 测试
```powershell
# 运行所有测�?
pytest

# 运行特定测试
pytest tests/test_xxx.py

# 生成覆盖率报�?
pytest --cov=. --cov-report=html
```

## 🎯 下一步行�?

1. **阅读文档**: 详细阅读 `README.md` 了解系统功能
2. **配置数据�?*: 根据需要配�?Tushare token 或使�?AkShare
3. **运行示例**: 执行 `python quickstart.py` 体验系统
4. **自定义配�?*: 修改 `config.yaml` 调整系统参数
5. **开始回�?*: 使用历史数据验证"一进二"策略

## 💡 提示

- **数据源选择**: AkShare免费且无需注册,推荐用于学习和测�?
- **Tushare**: 需要积分但数据更全�?适合正式使用
- **本地数据**: 如需高频回测,建议下载 Qlib 本地数据�?
- **GPU加�?*: 如有NVIDIA显卡,可安�?GPU 版本的深度学习框�?

## 📞 获取帮助

遇到问题?
1. 查看日志: `logs/qilin.log`
2. 阅读文档: `docs/` 目录
3. 检查配�? `config.yaml`
4. 验证数据: 运行数据验证脚本

---

**祝您使用愉快!** 🦄�?

**麒麟量化系统 v3.0** - A�?一进二"量化作战平台

