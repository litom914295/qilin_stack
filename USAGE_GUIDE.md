# 🦄 麒麟量化系统使用指南

## 🎯 系统概览

麒麟量化系统提供多种使用方式:
1. **Web 界面** (最推荐) - 可视化仪表板
2. **命令行** - 快速演示和测试
3. **API 服务** - FastAPI REST 接口
4. **Jupyter Notebook** - 交互式研究

---

## 🌐 方式1: Web 界面 (最直观)

### 启动 Streamlit Dashboard

```powershell
# 激活虚拟环境
.\.qilin\Scripts\Activate.ps1

# 方法A: 使用启动脚本(推荐)
python run_dashboard.py

# 方法B: 直接启动
streamlit run app\web\unified_agent_dashboard.py

# 方法C: 统一仪表板
streamlit run web\unified_dashboard.py
```

### 访问地址

启动后在浏览器访问:
- **默认地址**: http://localhost:8501
- **局域网访问**: http://[你的IP]:8501

### 功能特性

✅ **实时监控**
- 实时行情数据
- 持仓状态
- 订单跟踪
- 盈亏分析

✅ **智能分析**
- 多Agent协同决策
- 涨停板预测
- 市场情绪分析
- 资金流向追踪

✅ **可视化图表**
- K线图
- 因子贡献图
- 收益曲线
- 风险分布

✅ **交易管理**
- 策略配置
- 风控参数
- 回测分析
- 实盘模拟

---

## ⚡ 方式2: 命令行快速演示

### quickstart.py - 快速体验

```powershell
# 激活虚拟环境
.\.qilin\Scripts\Activate.ps1

# 运行快速演示
python quickstart.py
```

**功能菜单**:
1. 快速演示 - 分析3只股票,展示决策过程
2. 性能测试 - 测试并行分析50只股票的速度
3. 退出

**示例输出**:
```
股票: 000001
综合得分: 85.60
决策建议: strong_buy
置信度: 82.00%
建议仓位: large
理由: 生态位优势+资金流入+情绪良好
风险等级: medium

各Agent评分:
  - 生态位Agent: 21.5分
  - 资金分析Agent: 19.8分
  - 竞价博弈Agent: 13.5分
  ...
```

### main.py - 完整功能

```powershell
# 回测模式
python main.py --mode backtest --start_date 2024-01-01 --end_date 2024-12-31

# 实盘模式
python main.py --mode live --symbols 000001,600519

# 复盘模式
python main.py --mode replay --date 2024-10-22

# 查看帮助
python main.py --help
```

**参数说明**:
- `--mode`: 运行模式 (backtest/live/replay)
- `--start_date`: 开始日期
- `--end_date`: 结束日期
- `--symbols`: 股票代码(逗号分隔)
- `--config`: 配置文件路径

---

## 🔌 方式3: API 服务

### 启动 FastAPI 服务

```powershell
# 激活虚拟环境
.\.qilin\Scripts\Activate.ps1

# 启动 API 服务
uvicorn api.main:app --reload --port 8000
```

### API 文档

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 常用接口

```bash
# 获取股票分析
curl http://localhost:8000/api/v1/analyze?symbol=000001

# 获取涨停预测
curl http://localhost:8000/api/v1/limitup/predict

# 回测策略
curl -X POST http://localhost:8000/api/v1/backtest \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2024-01-01", "end_date": "2024-12-31"}'
```

---

## 📓 方式4: Jupyter Notebook

### 启动 JupyterLab

```powershell
# 激活虚拟环境
.\.qilin\Scripts\Activate.ps1

# 启动 JupyterLab
jupyter lab
```

访问: http://localhost:8888

### 示例 Notebook

在 `examples/` 目录查看示例:
- `limitup_example.py` - 涨停板分析
- `integration_test_remaining_modules.py` - 系统集成测试
- `predict_limit_up_demo.py` - 预测演示

---

## 🎮 使用场景

### 场景1: 学习和测试

```powershell
# 1. 快速演示功能
python quickstart.py

# 2. 查看 Web 界面
python run_dashboard.py

# 3. 交互式研究
jupyter lab
```

### 场景2: 回测策略

```powershell
# 1. 编辑配置文件
notepad config.yaml

# 2. 运行回测
python main.py --mode backtest --start_date 2024-01-01 --end_date 2024-12-31

# 3. 查看报告
# 报告保存在 reports/ 目录
```

### 场景3: 实盘监控

```powershell
# 1. 启动 Web 界面
python run_dashboard.py

# 2. 在浏览器配置:
#    - 选择股票池
#    - 设置风控参数
#    - 启动实时监控

# 3. 查看实时信号
#    - 涨停预测
#    - 买卖建议
#    - 风险警告
```

### 场景4: API 集成

```powershell
# 1. 启动 API 服务
uvicorn api.main:app --port 8000

# 2. 在你的程序中调用
import requests

response = requests.get("http://localhost:8000/api/v1/analyze?symbol=000001")
result = response.json()
print(result)
```

---

## ⚙️ 配置说明

### 编辑 config.yaml

```yaml
# 数据源配置
data:
  akshare:
    enabled: true  # 使用 AkShare 在线数据
  tushare:
    token: "your_token"  # 如果使用 Tushare
    enabled: false

# 交易配置
trading:
  max_positions: 5  # 最大持仓数
  position_size: 0.2  # 单只股票仓位
  stop_loss: 0.05  # 止损比例
  take_profit: 0.10  # 止盈比例

# 风控配置
risk:
  max_drawdown: 0.15  # 最大回撤
  max_position_size: 0.3  # 单只最大仓位
```

---

## 🛠️ 常用命令速查

```powershell
# 激活环境
.\.qilin\Scripts\Activate.ps1

# Web 界面
python run_dashboard.py

# 快速演示
python quickstart.py

# 完整回测
python main.py --mode backtest --start_date 2024-01-01

# API 服务
uvicorn api.main:app --reload

# Jupyter
jupyter lab

# 运行测试
pytest

# 查看日志
cat logs\qilin.log
```

---

## 📊 Web 界面功能说明

### 主页面布局

```
┌─────────────────────────────────────────────────┐
│  🐉 麒麟量化平台    🟢 开盘中    📡 连接正常  │
├─────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌──────────────────────────┐   │
│ │   侧边栏    │ │      主显示区域          │   │
│ │             │ │                          │   │
│ │ • 股票池    │ │  📈 实时K线              │   │
│ │ • 策略配置  │ │  📊 因子贡献             │   │
│ │ • 风控参数  │ │  💰 持仓状态             │   │
│ │ • 实时监控  │ │  📋 订单列表             │   │
│ │ • 回测分析  │ │  🎯 交易信号             │   │
│ │             │ │                          │   │
│ └─────────────┘ └──────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 核心功能

1. **实时监控** - 查看市场实时变化
2. **智能分析** - AI驱动的决策建议
3. **风险管理** - 实时风险监控和预警
4. **回测系统** - 历史数据策略验证
5. **报表生成** - 一键生成分析报告

---

## 🔥 快速上手 (3分钟)

```powershell
# 1. 激活环境
.\.qilin\Scripts\Activate.ps1

# 2. 启动 Web 界面
python run_dashboard.py

# 3. 浏览器访问
#    http://localhost:8501

# 4. 开始使用!
#    - 左侧选择股票
#    - 点击"开始分析"
#    - 查看实时建议
```

---

## 💡 使用建议

### 初学者

1. 先运行 `python quickstart.py` 了解系统
2. 启动 Web 界面熟悉功能
3. 使用小额资金测试
4. 阅读 README.md 深入了解

### 进阶用户

1. 修改 config.yaml 定制策略
2. 使用 Jupyter 研究因子
3. 通过 API 集成到自己的系统
4. 查看源码理解实现细节

### 专业用户

1. 修改 Agent 权重优化策略
2. 添加自定义因子
3. 集成自己的数据源
4. 部署到服务器实盘运行

---

## ❓ 常见问题

### Q1: Web 界面启动失败?
A: 
```powershell
# 检查 streamlit 是否安装
pip list | findstr streamlit

# 如未安装
pip install streamlit

# 重新启动
python run_dashboard.py
```

### Q2: 数据加载失败?
A: 确保 AkShare 可用,或配置 Tushare token

### Q3: 性能慢?
A: 
- 减少股票池数量
- 调整刷新间隔
- 使用本地 Qlib 数据

### Q4: 如何修改策略?
A: 编辑 `config.yaml`,调整各 Agent 权重和风控参数

---

## 📞 获取帮助

- 查看日志: `logs/qilin.log`
- 查看文档: `README.md`, `docs/` 目录
- 运行测试: `pytest tests/`

---

**祝您使用愉快!** 🦄✨

**麒麟量化系统** - A股"一进二"量化作战平台
