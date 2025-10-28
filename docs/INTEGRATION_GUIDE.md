# 麒麟量化统一平台 - 集成指南

## 🎯 概述

本项目集成了三个顶级开源量化项目的功能到统一的Web界面：

1. **Qlib** - Microsoft开源的AI量化投资平台
2. **RD-Agent** - Microsoft开源的自动研发Agent框架
3. **TradingAgents-CN-Plus** - 多智能体交易分析系统（中文增强版）

## 📦 项目结构

```
qilin_stack/
├── app/
│   ├── integrations/          # 集成模块
│   │   ├── __init__.py
│   │   ├── qlib_integration.py          # Qlib集成
│   │   ├── rdagent_integration.py       # RD-Agent集成
│   │   └── tradingagents_integration.py # TradingAgents集成
│   └── web/
│       └── unified_dashboard.py  # 统一Web界面
├── run_unified_dashboard.py      # 启动脚本
└── docs/
    └── INTEGRATION_GUIDE.md      # 本文档
```

## 🚀 快速开始

### 1. 环境准备

确保三个项目都已正确安装：

```bash
# Qlib (在 G:\test\qlib)
cd G:\test\qlib
pip install .

# RD-Agent (在 G:\test\RD-Agent)
cd G:\test\RD-Agent
pip install -e .

# TradingAgents (在 G:\test\tradingagents-cn-plus)
cd G:\test\tradingagents-cn-plus
pip install -r requirements.txt
```

### 2. 安装依赖

```bash
cd G:\test\qilin_stack
pip install streamlit pandas
```

### 3. 启动系统

```bash
python run_unified_dashboard.py
```

或直接使用streamlit:

```bash
streamlit run app/web/unified_dashboard.py
```

### 4. 访问界面

浏览器打开: http://localhost:8501

## 📊 功能模块

### 一、Qlib量化平台

#### 1.1 数据查询
- 支持多股票代码查询
- 灵活的日期范围选择
- 多市场支持（csi300、csi500等）

#### 1.2 因子计算
- Alpha158因子自动计算
- 自定义因子表达式
- 因子IC/IR统计

#### 1.3 模型训练
- LightGBM/XGBoost模型
- 神经网络模型
- 自定义模型配置

#### 1.4 策略回测
- 完整回测框架
- 多种回测指标
- 可视化结果展示

### 二、RD-Agent自动研发

#### 2.1 自动因子生成
- AI驱动的因子挖掘
- 多轮迭代优化
- 自动评估因子质量

**使用示例**:
```python
from app.integrations import rdagent_integration

# 生成10个因子，迭代3次
factors = rdagent_integration.auto_generate_factors(
    market_data=None,
    num_factors=10,
    iterations=3
)

for factor in factors:
    print(f"因子: {factor['name']}")
    print(f"公式: {factor['formula']}")
    print(f"IC: {factor['ic']:.4f}")
```

#### 2.2 模型优化
- 自动超参数优化
- 模型架构搜索
- 性能指标追踪

#### 2.3 策略生成
- 基于AI的策略生成
- 风险约束配置
- 策略回测验证

#### 2.4 研究循环
- factor_loop: 因子研究循环
- model_loop: 模型研究循环
- strategy_loop: 策略研究循环

### 三、TradingAgents多智能体

#### 3.1 单股分析
- 基本面分析智能体
- 技术面分析智能体
- 新闻情绪分析智能体
- 综合决策智能体

**使用示例**:
```python
from app.integrations import tradingagents_integration

# 分析股票
result = tradingagents_integration.analyze_stock(
    stock_code='000001',
    analysis_depth=3,
    market='cn'
)

print(f"操作建议: {result['final_decision']['action']}")
print(f"信心度: {result['final_decision']['confidence']}")
```

#### 3.2 批量分析
- 支持批量股票分析
- 会员积分系统
- 并行分析加速

**批量分析示例**:
```python
# 批量分析多只股票
stocks = ['000001', '600519', '000858']
results = tradingagents_integration.batch_analyze(
    stock_codes=stocks,
    member_id='member_001',  # 可选
    analysis_depth=3
)

for result in results:
    print(f"{result['stock_code']}: {result['final_decision']['action']}")
```

#### 3.3 多智能体辩论
- 看涨/看跌智能体辩论
- 多轮辩论机制
- 最终共识达成

#### 3.4 会员管理
- 会员注册与管理
- 积分充值与消费
- 使用历史追踪

**会员管理示例**:
```python
# 添加会员
tradingagents_integration.add_member(
    member_id='member_001',
    name='张三',
    credits=100
)

# 查询会员
member = tradingagents_integration.get_member_info('member_001')
print(f"剩余点数: {member['credits']}")

# 更新点数
tradingagents_integration.update_member_credits('member_001', 50)
```

## 🔧 配置说明

### Qlib配置

```python
# 修改 app/integrations/qlib_integration.py
QLIB_PATH = Path(r"G:\test\qlib")  # Qlib项目路径
data_path = "~/.qlib/qlib_data/cn_data"  # 数据路径
```

### RD-Agent配置

```python
# 修改 app/integrations/rdagent_integration.py
RDAGENT_PATH = Path(r"G:\test\RD-Agent")  # RD-Agent项目路径
workspace = "./rdagent_workspace"  # 工作空间路径
```

### TradingAgents配置

```python
# 修改 app/integrations/tradingagents_integration.py
TRADINGAGENTS_PATH = Path(r"G:\test\tradingagents-cn-plus")  # 项目路径
```

## 🔌 API接口

### Qlib Integration API

```python
from app.integrations import qlib_integration

# 初始化
qlib_integration.initialize()

# 获取股票数据
df = qlib_integration.get_stock_data(
    instruments=['000001'],
    start_time='2024-01-01',
    end_time='2024-12-31'
)

# 计算因子
factors = qlib_integration.calculate_alpha158_factors(
    instruments=['000001'],
    start_time='2024-01-01',
    end_time='2024-12-31'
)

# 运行回测
results = qlib_integration.run_backtest(
    strategy_config={},
    start_time='2024-01-01',
    end_time='2024-12-31'
)
```

### RD-Agent Integration API

```python
from app.integrations import rdagent_integration

# 初始化
rdagent_integration.initialize()

# 生成因子
factors = rdagent_integration.auto_generate_factors(
    market_data=None,
    num_factors=10,
    iterations=3
)

# 优化模型
config = rdagent_integration.optimize_model(
    base_model='LightGBM',
    train_data=None,
    iterations=5
)

# 生成策略
strategy = rdagent_integration.generate_strategy(
    strategy_type='momentum',
    constraints={'max_position': 0.1}
)
```

### TradingAgents Integration API

```python
from app.integrations import tradingagents_integration

# 初始化
tradingagents_integration.initialize()

# 单股分析
result = tradingagents_integration.analyze_stock(
    stock_code='000001',
    analysis_depth=3,
    market='cn'
)

# 批量分析
results = tradingagents_integration.batch_analyze(
    stock_codes=['000001', '600519'],
    member_id='member_001',
    analysis_depth=3
)

# 多智能体辩论
debate = tradingagents_integration.multi_agent_debate(
    stock_code='000001',
    debate_rounds=3
)
```

## 📝 注意事项

1. **数据准备**: 使用Qlib功能前需要先下载数据
2. **模型训练**: 首次训练模型需要较长时间
3. **会员积分**: 批量分析会消耗会员积分
4. **系统资源**: 多智能体分析比较消耗系统资源

## 🐛 故障排除

### 问题1: Qlib模块不可用

**解决方案**:
```bash
cd G:\test\qlib
pip install -e .
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 问题2: RD-Agent导入失败

**解决方案**:
```bash
cd G:\test\RD-Agent
pip install -e .
# 配置LLM API密钥
```

### 问题3: TradingAgents路径错误

**解决方案**:
修改 `app/integrations/tradingagents_integration.py` 中的路径配置

## 🔄 更新日志

### v1.0.0 (2025-01-10)
- ✨ 初始版本发布
- 📊 集成Qlib量化平台
- 🤖 集成RD-Agent自动研发
- 👥 集成TradingAgents多智能体
- 🖥️ 统一Web界面

## 📞 联系方式

- 项目地址: G:\test\qilin_stack
- 文档更新: 2025-01-10

## 📄 许可证

本项目采用 Apache 2.0 许可证
