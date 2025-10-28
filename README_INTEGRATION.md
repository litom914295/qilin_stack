# 🦄 麒麟量化统一平台

> 集成三大开源量化项目的统一Web平台

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

## 📖 项目简�?

麒麟量化统一平台将三个世界级的开源量化项目整合到一个统一的Web界面中，提供�?

- 📊 **Qlib**: Microsoft开源的AI量化投资平台
- 🤖 **RD-Agent**: Microsoft开源的自动研发Agent框架  
- 👥 **TradingAgents**: 多智能体交易分析系统（中文增强版�?

### �?核心特�?

#### 1️⃣ Qlib量化平台
- �?股票数据查询与管�?
- �?Alpha158因子自动计算
- �?多种机器学习模型训练
- �?完整的策略回测框�?

#### 2️⃣ RD-Agent自动研发
- �?AI驱动的自动因子生�?
- �?智能模型优化与超参搜�?
- �?自动策略生成与验�?
- �?研究循环自动�?

#### 3️⃣ TradingAgents多智能体
- �?基本�?技术面+新闻面综合分�?
- �?批量股票分析（支持会员积分）
- �?多智能体辩论决策
- �?完整的会员管理系�?

#### 4️⃣ 数据共享桥接
- �?三个项目间的因子共享
- �?模型跨项目复�?
- �?策略配置统一管理
- �?数据格式自动转换

## 🚀 快速开�?

### 前置要求

- Python 3.8+
- 已安装Qlib、RD-Agent、TradingAgents三个项目
- Windows/Linux/MacOS系统

### 安装步骤

#### 1. 克隆或下载项�?

```bash
# 项目已在 G:\test\qilin_stack
cd G:\test\qilin_stack
```

#### 2. 安装依赖

```bash
pip install streamlit pandas numpy
```

#### 3. 配置三个项目路径

编辑 `app/integrations/` 目录下的集成模块，修改项目路径：

```python
# app/integrations/qlib_integration.py
QLIB_PATH = Path(r"G:\test\qlib")

# app/integrations/rdagent_integration.py
RDAGENT_PATH = Path(r"G:\test\RD-Agent")

# app/integrations/tradingagents_integration.py
TRADINGAGENTS_PATH = Path(r"G:\test\tradingagents-cn-plus")
```

#### 4. 启动系统

```bash
python start_web.py
```

或直接运行：

```bash
streamlit run app/web/unified_dashboard.py
```

#### 5. 访问界面

浏览器打开: **http://localhost:8501**

## 📸 界面预览

### 主界�?
![主界面](docs/screenshots/main.png)

### Qlib模块
![Qlib](docs/screenshots/qlib.png)

### RD-Agent模块  
![RD-Agent](docs/screenshots/rdagent.png)

### TradingAgents模块
![TradingAgents](docs/screenshots/tradingagents.png)

## 📚 使用文档

### 完整文档

详细使用指南请查�? [📖 集成指南](docs/INTEGRATION_GUIDE.md)

### 快速示�?

#### 示例1: Qlib数据查询

```python
from app.integrations import qlib_integration

# 查询股票数据
df = qlib_integration.get_stock_data(
    instruments=['000001', '600519'],
    start_time='2024-01-01',
    end_time='2024-12-31'
)
print(df.head())
```

#### 示例2: RD-Agent自动生成因子

```python
from app.integrations import rdagent_integration

# 自动生成10个因�?
factors = rdagent_integration.auto_generate_factors(
    market_data=None,
    num_factors=10,
    iterations=3
)

for factor in factors:
    print(f"{factor['name']}: IC={factor['ic']:.4f}")
```

#### 示例3: TradingAgents批量分析

```python
from app.integrations import tradingagents_integration

# 批量分析股票
results = tradingagents_integration.batch_analyze(
    stock_codes=['000001', '600519', '000858'],
    member_id='member_001',
    analysis_depth=3
)

for result in results:
    print(f"{result['stock_code']}: {result['final_decision']['action']}")
```

#### 示例4: 数据共享

```python
from app.integrations import data_bridge

# 保存因子（来自RD-Agent�?
data_bridge.save_factor(
    factor_name='momentum_5d',
    factor_data={'formula': '(close - close[5]) / close[5]'},
    source='rdagent'
)

# 在Qlib中加载因�?
factor = data_bridge.load_factor('momentum_5d')
print(factor)
```

## 🏗�?项目架构

```
qilin_stack/
├── app/
�?  ├── integrations/              # 集成模块
�?  �?  ├── __init__.py
�?  �?  ├── qlib_integration.py    # Qlib封装
�?  �?  ├── rdagent_integration.py # RD-Agent封装
�?  �?  ├── tradingagents_integration.py  # TradingAgents封装
�?  �?  └── data_bridge.py         # 数据共享桥接
�?  �?
�?  └── web/
�?      ├── enhanced_dashboard.py  # 原有增强界面
�?      └── unified_dashboard.py   # 统一集成界面
�?
├── docs/
�?  ├── INTEGRATION_GUIDE.md       # 集成指南
�?  └── screenshots/               # 界面截图
�?
├── start_web.py      # 启动原有界面
├── start_web.py       # 启动统一界面
└── README_INTEGRATION.md          # 本文�?
```

## 🔧 配置说明

### Qlib数据准备

```bash
# 下载Qlib数据
cd G:\test\qlib
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### RD-Agent配置

需要配置LLM API密钥（如OpenAI、DeepSeek等）�?

```bash
# �?env文件中配�?
OPENAI_API_KEY=your_api_key
# 或其他LLM提供商的密钥
```

### TradingAgents配置

按照原项目文档配置LLM和数据源�?

## 📊 功能对比

| 功能 | Qlib | RD-Agent | TradingAgents | 统一平台 |
|------|------|----------|---------------|----------|
| 数据查询 | �?| �?| �?| �?|
| 因子计算 | �?| �?| �?| �?|
| 自动因子生成 | �?| �?| �?| �?|
| 模型训练 | �?| �?| �?| �?|
| 策略回测 | �?| �?| �?| �?|
| 多智能体分析 | �?| �?| �?| �?|
| 批量分析 | �?| �?| �?| �?|
| 会员管理 | �?| �?| �?| �?|
| 数据共享 | �?| �?| �?| �?|
| 统一界面 | �?| �?| �?| �?|

## 🤝 贡献指南

欢迎提交Issue和Pull Request�?

### 开发环境设�?

```bash
# 安装开发依�?
pip install -r requirements-dev.txt

# 运行测试
pytest tests/
```

## 📝 更新日志

### v1.0.0 (2025-01-10)
- �?首次发布
- 📊 集成Qlib量化平台
- 🤖 集成RD-Agent自动研发
- 👥 集成TradingAgents多智能体
- 🌉 实现数据共享桥接
- 🖥�?统一Web界面

## 🙏 致谢

感谢以下开源项目：

- [Microsoft Qlib](https://github.com/microsoft/qlib) - AI量化投资平台
- [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) - 自动研发Agent
- [TradingAgents-CN-Plus](https://github.com/user/tradingagents-cn-plus) - 多智能体交易分析

## 📄 许可�?

本项目采�?Apache 2.0 许可�?

---

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。使用本系统进行实盘交易的风险由用户自行承担�?

## 📞 联系方式

- 项目地址: `G:\test\qilin_stack`
- 文档更新: 2025-01-10

---

<div align="center">
Made with ❤️ by Qilin Quant Team
</div>

