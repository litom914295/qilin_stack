# Qilin Stack 用户操作手册

**版本**: v1.0  
**更新日期**: 2025-01-XX  
**适用对象**: A股量化交易研究人员、策略开发者

---

## 📖 目录

- [1. 快速开始](#1-快速开始)
- [2. 环境配置](#2-环境配置)
- [3. UI 功能说明](#3-ui-功能说明)
- [4. 一进二策略使用](#4-一进二策略使用)
- [5. 常见问题](#5-常见问题)

---

## 1. 快速开始

### 1.1 依赖安装

#### 一键安装 (推荐)

```bash
# Windows PowerShell
pip install -r requirements.txt
```

#### 手动安装核心依赖

```bash
# 核心依赖
pip install qlib pandas numpy
pip install streamlit plotly

# 机器学习 (可选)
pip install lightgbm xgboost catboost scikit-learn

# 深度学习 (可选)
pip install torch

# MLOps (可选)
pip install mlflow

# RL (可选 - 一进二高级策略需要)
pip install tianshou<=0.4.10 gym
```

### 1.2 数据准备

#### 方式一: 使用 Qlib 官方数据 (推荐)

```bash
# 下载 A股日线数据 (约 1GB)
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

#### 方式二: 使用测试数据 (快速体验)

```bash
# 生成测试数据 (约 50MB, 30只股票 x 2年)
python tests/run_tests.py prepare
```

### 1.3 启动 UI

```bash
# 启动 Streamlit UI
streamlit run app.py

# 或指定端口
streamlit run app.py --server.port 8501
```

访问: http://localhost:8501

---

## 2. 环境配置

### 2.1 Qlib 初始化配置

#### 离线模式 (本地数据)

在 UI 首页或 `config/qlib_config_center.py` 中配置:

```python
from config.qlib_config_center import QlibConfig, QlibInitializer

config = QlibConfig(
    mode="offline",  # 离线模式
    provider_uri="~/.qlib/qlib_data/cn_data",  # 数据目录
    region="cn",  # 区域: cn (A股) / us (美股)
    expression_cache="DiskExpressionCache",  # 表达式缓存
    dataset_cache="DiskDatasetCache",  # 数据集缓存
)

success, message = QlibInitializer.init(config)
```

#### 在线模式 (Qlib-Server)

```python
config = QlibConfig(
    mode="online",  # 在线模式
    server_host="127.0.0.1",  # Qlib-Server 地址
    server_port=9710,  # 端口
    server_token="your_token",  # 鉴权 Token (可选)
    server_timeout=30,  # 超时 (秒)
)
```

#### 自动模式 (优先在线,失败回退离线)

```python
config = QlibConfig(
    mode="auto",  # 自动模式
    provider_uri="~/.qlib/qlib_data/cn_data",  # 离线数据作为回退
    server_host="127.0.0.1",
    server_port=9710,
)
```

### 2.2 缓存配置

#### Expression Cache (表达式缓存)

用于缓存因子计算结果 (如 `$close/Ref($close,1)-1`):

```python
config = QlibConfig(
    expression_cache="DiskExpressionCache",
    expression_provider_kwargs={
        "dir": ".qlib_cache/expression_cache",  # 缓存目录
        "max_workers": 4,  # 并行数
    }
)
```

#### Dataset Cache (数据集缓存)

用于缓存 DatasetH 数据集:

```python
config = QlibConfig(
    dataset_cache="DiskDatasetCache",
    dataset_provider_kwargs={
        "dir": ".qlib_cache/dataset_cache",
        "max_workers": 4,
    }
)
```

#### 清理缓存

```bash
# 删除所有缓存
rm -rf .qlib_cache/

# 或在 UI 的 "数据工具" Tab 中点击 "清理缓存" 按钮
```

---

## 3. UI 功能说明

### 3.1 数据工具 Tab

#### 功能一: 数据下载

1. 选择数据类型: `cn_stock` / `Alpha158` / `Alpha360`
2. 选择日期范围
3. 点击 "下载数据"

#### 功能二: 数据健康检查

- **缺口检测**: 检查交易日是否连续
- **重复检测**: 检查是否有重复数据
- **日历对齐**: 检查数据日期是否与交易日历一致

#### 功能三: 表达式测试

测试因子表达式是否正确:

```python
# 示例表达式
$close / Ref($close, 1) - 1  # 日收益率
Mean($close, 5)  # 5日均价
($close - Mean($close, 20)) / Std($close, 20)  # Z-Score

# 一进二专用表达式
If($close / Ref($close, 1) - 1 > 0.095, 1, 0)  # 涨停标记
If($close/$open - 1 < 0.02, If($close/Ref($close,1)-1>0.095,1,0),0)  # 经典一进二
```

1. 在输入框中输入表达式
2. 选择标的 (如 `000001.SZ`)
3. 选择日期范围
4. 点击 "测试表达式"
5. 查看结果和性能统计

---

### 3.2 qrun 工作流 Tab

#### 完整流程: 训练-预测-回测-评估

**步骤 1: 选择模板**

- 从 `configs/qlib_workflows/templates/` 选择 YAML 配置
- 或点击 "新建模板"

**步骤 2: 配置参数**

```yaml
qlib_init:
  provider_uri: "~/.qlib/qlib_data/cn_data"
  region: cn

market: csi300
benchmark: SH000300

data_handler:
  instruments: csi300
  start_time: 2020-01-01
  end_time: 2023-12-31

model:
  class: LGBModel
  module_path: qlib.contrib.model.gbdt
  kwargs:
    num_boost_round: 100
    early_stopping_rounds: 50

strategy:
  class: TopkDropoutStrategy
  topk: 30
  n_drop: 5
```

**步骤 3: 运行工作流**

1. 点击 "开始训练"
2. 实时查看日志
3. 训练完成后查看指标
4. 查看 MLflow 记录

**步骤 4: 查看结果**

- **训练指标**: IC, Rank IC, Precision@5/10/30
- **回测报告**: 年化收益, 夏普比率, 最大回撤
- **可视化**: 收益曲线, 回撤曲线, 月度热力图

---

### 3.3 回测 Tab

#### 单独回测 (已有预测结果)

**步骤 1: 加载预测**

- 上传 CSV 文件 (格式: `date, instrument, score`)
- 或选择 MLflow run_id

**步骤 2: 配置策略**

```python
strategy = TopkDropoutStrategy(
    topk=30,  # 持仓数
    n_drop=5,  # 每日最多调仓数
    signal=<pred>,  # 预测信号
)
```

**步骤 3: 配置执行器**

```python
executor = SimulatorExecutor(
    time_per_step="day",  # 每日撮合
    generate_portfolio_metrics=True,
)
```

**步骤 4: 运行回测**

```python
from qlib.backtest import backtest
report, positions = backtest(pred, strategy, executor)
```

**步骤 5: 查看报告**

- **风险指标** (来自 `qlib.contrib.evaluate.risk_analysis`):
  - 年化收益率
  - 年化波动率
  - 夏普比率
  - 信息比率
  - 最大回撤
  - Calmar 比率

- **可视化**:
  - 累计收益曲线
  - 回撤曲线
  - 月度收益热力图
  - 分组收益分布

---

### 3.4 模型 Zoo Tab

#### 支持的模型

| 模型类别 | 模型名称 | 依赖 | 状态 |
|---------|---------|------|------|
| **GBDT** | LightGBM | lightgbm | ✅ 可用 |
|  | XGBoost | xgboost | ✅ 可用 |
|  | CatBoost | catboost | ✅ 可用 |
| **神经网络** | MLP | torch | ⚠️ 需安装 |
|  | LSTM | torch | ⚠️ 需安装 |
|  | GRU | torch | ⚠️ 需安装 |
| **Transformer** | Transformer | torch | ⚠️ 需安装 |
|  | TRA | torch | ⚠️ 需安装 |
|  | HIST | torch | ⚠️ 需安装 |
| **图神经网络** | GATs | torch-geometric | ❌ 不支持 |
|  | RSR | torch-geometric | ❌ 不支持 |

#### 依赖检测与安装

UI 会自动检测模型依赖:

- ✅ **绿色**: 已安装,可直接使用
- ⚠️ **黄色**: 未安装,点击 "一键安装" 按钮
- ❌ **红色**: 不支持或依赖冲突

#### 降级策略

当首选模型不可用时,系统会自动降级:

```
TRA/HIST → Transformer → LSTM → LightGBM
```

UI 会显示降级原因和恢复方案。

---

### 3.5 NestedExecutor Tab

#### 三层嵌套决策

**Level 1 (Day)**: 组合优化 - 决定持仓标的和权重  
**Level 2 (Hour/30min)**: 订单生成 - 决定买卖时机和订单分割  
**Level 3 (Minute/5min)**: 订单执行 - 决定具体下单价格和数量  

#### 配置示例

**外层策略 (Level 1)**:

```yaml
outer_strategy:
  class: TopkDropoutStrategy
  topk: 30
  n_drop: 5
```

**内层执行器 (Level 2 + Level 3)**:

```yaml
inner_executor:
  class: NestedExecutor
  time_per_step: "30min"  # Level 2 时间粒度
  inner_executor:
    class: SimulatorExecutor
    time_per_step: "5min"  # Level 3 时间粒度
```

#### 市场冲击模型 (Almgren-Chriss)

```python
# 永久冲击
permanent_cost = gamma * (V/ADV) * P * V

# 临时冲击
temporary_cost = eta * sqrt(V/ADV) * P * V

# 参数:
# V = 成交量
# ADV = 日均成交量
# P = 价格
# gamma, eta = 冲击系数
```

#### 订单分割策略

- **TWAP** (Time-Weighted Average Price): 均匀时间分割
- **VWAP** (Volume-Weighted Average Price): 按成交量分割
- **POV** (Percentage of Volume): 按市场成交量比例下单

---

### 3.6 IC 分析 Tab

#### IC (Information Coefficient) 分析

**功能**:
- 评估因子预测能力
- 计算 IC/IR 时间序列
- 分位数收益分析
- 横截面去极值/标准化

#### 使用步骤

**步骤 1: 准备数据**

CSV 格式 (列: `date`, `instrument`, `factor`, `label`):

```csv
date,instrument,factor,label
2024-01-02,000001.SZ,0.523,0.012
2024-01-02,000002.SZ,-0.231,-0.005
...
```

**步骤 2: 计算 IC**

- 选择 IC 方法: `Pearson` / `Spearman`
- 选择处理 NaN 策略: `drop` / `fill_zero` / `raise`
- 点击 "计算 IC"

**步骤 3: 查看结果**

- **IC 时间序列**: 每日 IC 曲线
- **IC 统计**: 均值、标准差、IR、胜率
- **分位数收益**: Q1-Q5 组收益分布
- **多空收益**: Top 组 - Bottom 组

#### 横截面处理

**去极值 (Winsorize)**:

```python
# 3σ 去极值
mean = factor.mean()
std = factor.std()
upper = mean + 3 * std
lower = mean - 3 * std
factor_winsorized = factor.clip(lower, upper)
```

**标准化 (Z-Score)**:

```python
# 横截面标准化 (每个日期独立)
factor_std = (factor - factor.mean()) / factor.std()
```

**中性化 (Neutralize)**:

```python
# 对市值/行业回归,取残差
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(market_cap, factor)
factor_neutral = factor - model.predict(market_cap)
```

---

## 4. 一进二策略使用

### 4.1 什么是一进二?

**定义**: 涨停板后次日继续上涨的交易机会

**经典一进二模式**:
- 当日低开 (开盘价相比前收盘 < 2%)
- 盘中拉升至涨停 (收盘涨幅 ≥ 9.5%)
- 次日高开高走

### 4.2 一进二策略配置

配置文件: `configs/qlib_workflows/templates/limitup_yinjiner_strategy.yaml`

#### 标签定义

**经典一进二标签**:

```yaml
label:
  classic_yinjiner: |
    If(
      $close / $open - 1 < 0.02,  # 低开 < 2%
      If($close / Ref($close, 1) - 1 > 0.095, 1, 0),  # 收盘涨停
      0
    )
```

**强势一进二标签**:

```yaml
label:
  strong_yinjiner: |
    If(
      $close / $open - 1 >= 0.02,  # 高开 >= 2%
      If($close / Ref($close, 1) - 1 > 0.095, 1, 0),  # 收盘涨停
      0
    )
```

**连板标签**:

```yaml
label:
  continuous_limitup: |
    If(
      And(
        $open / Ref($close, 1) - 1 > 0.095,  # 开盘涨停
        $close / Ref($close, 1) - 1 > 0.095  # 收盘涨停
      ),
      1,
      0
    )
```

#### Alpha 因子 (24个)

**价格因子** (6个):

```yaml
features:
  - name: return_1d
    expression: $close / Ref($close, 1) - 1
  
  - name: return_5d
    expression: $close / Ref($close, 5) - 1
  
  - name: high_low_ratio
    expression: ($high - $low) / $close
```

**涨停因子** (6个):

```yaml
features:
  - name: is_limitup
    expression: If($close / Ref($close, 1) - 1 > 0.095, 1, 0)
  
  - name: limitup_days_3d
    expression: Sum(If($close/Ref($close,1)-1>0.095,1,0), 3)
  
  - name: open_board_flag
    expression: If($high / Ref($close, 1) - 1 > 0.095 And $close / Ref($close, 1) - 1 < 0.095, 1, 0)
```

**成交量因子** (6个):

```yaml
features:
  - name: volume_ratio
    expression: $volume / Mean($volume, 5) - 1
  
  - name: turnover
    expression: $volume / $total_shares
  
  - name: volume_price_corr
    expression: Corr($volume, $close, 20)
```

**技术因子** (3个):

```yaml
features:
  - name: ma_5_20_cross
    expression: Mean($close, 5) / Mean($close, 20) - 1
  
  - name: rsi_6
    expression: RSI($close, 6)
  
  - name: macd
    expression: MACD($close)
```

**强度因子** (3个):

```yaml
features:
  - name: limit_strength
    expression: ($close - $open) / ($high - $low)
  
  - name:封单量
    expression: $bid_volume1 / $volume
```

#### 样本过滤

```yaml
filter:
  # 排除 ST 股票
  - Not(Str$like($name, "ST%"))
  
  # 排除新股 (上市 < 60天)
  - $list_days > 60
  
  # 排除低价股 (< 5元)
  - $close > 5
  
  # 排除流动性差的股票 (日均成交额 < 1000万)
  - Mean($amount, 20) > 10000000
```

#### 回测参数

```yaml
backtest:
  # 开板成本 (涨停开板后买入成本更高)
  open_board_cost: 0.03  # 3%
  
  # 换手约束
  max_turnover: 0.3  # 每日最多换 30% 仓位
  
  # 涨跌停规则
  limit_threshold: 0.095  # 9.5%
  limit_type: "both"  # 涨停和跌停都考虑
```

### 4.3 运行一进二策略

#### 方式一: 通过 UI

1. 打开 "qrun 工作流" Tab
2. 选择模板: `limitup_yinjiner_strategy.yaml`
3. 点击 "开始训练"
4. 查看回测报告:
   - 命中率 (预测涨停的准确率)
   - 平均持有期收益
   - 可交易性 (是否能买进)

#### 方式二: 通过命令行

```bash
# 运行完整流程
qrun configs/qlib_workflows/templates/limitup_yinjiner_strategy.yaml

# 只运行回测
python scripts/run_limitup_backtest.py \
  --config configs/qlib_workflows/templates/limitup_yinjiner_strategy.yaml \
  --start_date 2023-01-01 \
  --end_date 2023-12-31
```

### 4.4 评估指标

#### 命中率 (Hit Rate)

```python
hit_rate = (预测涨停且实际涨停的天数) / (预测涨停的总天数)
```

目标: **> 60%**

#### 平均收益

```python
avg_return = (所有预测涨停标的的次日收益).mean()
```

目标: **> 2%**

#### 可交易性

```python
tradability = (能在涨停价买进的天数) / (预测涨停的总天数)
```

目标: **> 50%** (考虑一字板/开板时机)

---

## 5. 常见问题

### Q1: Qlib 初始化失败

**错误**: `RuntimeError: Qlib is not initialized`

**解决**:

```python
# 检查数据目录是否存在
import os
data_dir = "~/.qlib/qlib_data/cn_data"
print(os.path.exists(os.path.expanduser(data_dir)))

# 重新下载数据
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### Q2: 缓存占用空间过大

**问题**: `.qlib_cache/` 目录占用 10GB+

**解决**:

```bash
# 清理所有缓存
rm -rf .qlib_cache/

# 或只清理表达式缓存
rm -rf .qlib_cache/expression_cache/

# 或在 UI 中点击 "清理缓存" 按钮
```

### Q3: 模型训练很慢

**问题**: LightGBM 训练 1小时+

**优化建议**:

1. **减少样本数**:

```yaml
data_handler:
  start_time: 2022-01-01  # 减少历史数据
```

2. **减少特征数**:

```yaml
features:
  # 只保留重要特征 (从 158 个减少到 20 个)
```

3. **减少迭代次数**:

```yaml
model:
  kwargs:
    num_boost_round: 50  # 从 100 减少到 50
```

4. **使用多核**:

```yaml
model:
  kwargs:
    num_threads: 8  # 使用 8 核
```

### Q4: 回测结果与实盘不符

**可能原因**:

1. **未考虑涨跌停**: 实盘无法买入涨停股票
2. **未考虑滑点**: 实际成交价可能高于理论价格
3. **未考虑手续费**: 佣金 + 印花税
4. **未考虑冲击成本**: 大单对价格的影响

**解决**:

```yaml
backtest:
  slippage: 0.002  # 0.2% 滑点
  commission: 0.0003  # 万三佣金
  min_cost: 5  # 最低 5 元手续费
  limit_threshold: 0.095  # 涨停阈值
  deal_price: "close"  # 撮合价格: close/vwap/twap
```

### Q5: 一进二命中率很低 (<30%)

**可能原因**:

1. **样本不平衡**: 涨停样本太少
2. **特征不足**: 缺少关键因子
3. **过拟合**: 在训练集表现好,测试集差

**优化方向**:

1. **样本平衡**:

```python
# 上采样 (SMOTE)
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
```

2. **增加特征**:

```yaml
# 增加情绪因子、资金流因子
- name: sentiment_score
  expression: ...

- name: money_flow_20d
  expression: ...
```

3. **调整标签**:

```yaml
# 放宽涨停阈值
label:
  expression: If($close / Ref($close, 1) - 1 > 0.08, 1, 0)  # 8% 即标记为 1
```

### Q6: 如何扩展新的数据源?

参考 [开发者文档](developer_guide.md#扩展数据源)

---

## 📞 技术支持

- **文档**: `docs/` 目录
- **示例**: `examples/` 目录
- **测试**: `python tests/run_tests.py -h`
- **Issue**: 在 GitHub 提交 Issue

---

**祝您交易顺利!** 🎯📈
