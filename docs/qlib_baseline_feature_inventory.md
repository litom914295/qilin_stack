# Qlib 基线特性清单与版本对齐文档

**生成日期**: 2025-01-XX  
**Qlib 版本**: v0.9.7-9-gbb7ab1cf (commit: bb7ab1cf, 2025-10-17)  
**对齐目标**: G:\test\qlib 主分支最新 commit  
**麒麟项目路径**: G:\test\qilin_stack  

---

## 1. 版本快照

### 1.1 Qlib 版本信息
- **版本标签**: v0.9.7 (+ 9 commits ahead)
- **当前 Commit**: `bb7ab1cf143b93a1656d13cfdbd40d1e7c15edd3`
- **提交日期**: 2025-10-17 10:50:16 +0530
- **最新变更**: docs: fix spelling mistake: exmaple to example (#2033)
- **版本检测方式**: `setuptools_scm.get_version()` (动态版本管理)

### 1.2 核心依赖 (pyproject.toml)

**基础依赖**:
```toml
pyyaml
numpy
pandas >= 1.1
mlflow
filelock >= 3.16.0
redis
dill
fire
ruamel.yaml >= 0.17.38
python-redis-lock
tqdm
pymongo
loguru
lightgbm
gym
cvxpy
joblib
matplotlib
jupyter
nbconvert
pyarrow
pydantic-settings
setuptools-scm
```

**可选依赖**:
- `[rl]`: tianshou<=0.4.10, torch, numpy<2.0.0
- `[docs]`: scipy<=1.15.3, sphinx, sphinx_rtd_theme, readthedocs_sphinx_ext, snowballstemmer<3.0
- `[test]`: yahooquery, baostock
- `[analysis]`: plotly, statsmodels
- `[lint]`: black, pylint, mypy<1.5.0, flake8, nbqa
- `[package]`: twine, build

---

## 2. Qlib 核心特性清单 (Feature Inventory)

### 2.1 数据层 (Data Layer)

| 特性 | 描述 | 麒麟对齐状态 |
|------|------|-------------|
| **Alpha158 Handler** | 158 个常用技术因子 | ✅ 已对齐 (layer2_qlib) |
| **Alpha360 Handler** | 360 个增强因子 | ✅ 已对齐 |
| **High-Frequency Data** | 1min/5min 高频数据处理 | ⚠️ 部分对齐 (模拟数据为主) |
| **Point-in-Time Database** | PIT 数据库支持 | ❓ 未明确使用 |
| **Expression Engine** | 因子表达式计算引擎 | ✅ 已对齐 |
| **Cache System** | 表达式/数据集/Redis 缓存 | ⚠️ 部分对齐 (Redis未强制要求) |
| **Multi-Frequency Data** | 多频率数据源管理 (day/1min) | ✅ 已支持 (HIGH_FREQ_CONFIG) |

### 2.2 模型层 (Model Zoo)

| 模型类别 | 官方支持模型 | 麒麟对齐状态 |
|---------|-------------|-------------|
| **传统机器学习** | LightGBM, XGBoost, CatBoost, Linear | ✅ 完全对齐 |
| **循环神经网络** | LSTM, GRU, ALSTM, LSTM_TS, GRU_TS | ✅ 完全对齐 |
| **Transformer** | Transformer, Localformer, Transformer_TS | ✅ 已支持 (部分模型作为 TRA/HIST 的 fallback) |
| **TRA (Temporal Routing)** | TRA (Jul 2021 新增) | ⚠️ 降级实现 (UI 未明确标注) |
| **HIST** | HIST (Apr 2022 新增) | ⚠️ 降级实现 (UI 未明确标注) |
| **IGMTF** | IGMTF (Apr 2022 新增) | ✅ 已支持 |
| **DoubleEnsemble** | DoubleEnsemble | ⚠️ 降级实现 (UI 未明确标注) |
| **TCN** | TCN, TCN_TS | ✅ 已支持 |
| **GATS** | GATS, GATS_TS | ✅ 已支持 |
| **Tabnet** | Tabnet | ✅ 已支持 |
| **SFM** | SFM | ✅ 已支持 |
| **TCTS** | TCTS | ✅ 已支持 |
| **AdaRNN** | AdaRNN | ✅ 已支持 |
| **KRNN** | KRNN | ✅ 已支持 |
| **Sandwich** | Sandwich | ✅ 已支持 |
| **TFT** | TFT (Temporal Fusion Transformer) | ❓ 未在麒麟 UI 显式列出 |
| **High-Freq GBDT** | HighFreqGBDTModel | ⚠️ 高频模块待完善 |

### 2.3 工作流层 (Workflow & MLOps)

| 特性 | 描述 | 麒麟对齐状态 |
|------|------|-------------|
| **qrun** | YAML 配置驱动的实验管理 | ✅ 完全对齐 (UI 标签页) |
| **MLflow Integration** | 实验跟踪与模型注册 | ✅ 已集成 |
| **Recorder System** | R.log_*/R.save_objects 等 API | ✅ 已使用 |
| **Experiment Manager** | 实验版本管理 | ✅ 已支持 |
| **Online Serving** | Qlib-Server 在线模式 | ❌ 未集成 |
| **Model Rolling** | 自动模型滚动更新 | ❓ 未明确使用 |
| **Data Rolling** | 滚动数据处理 | ❓ 未明确使用 |

### 2.4 回测与执行 (Backtest & Execution)

| 特性 | 描述 | 麒麟对齐状态 |
|------|------|-------------|
| **SimulatorExecutor** | 基础模拟交易执行器 | ✅ 已使用 |
| **NestedExecutor** | 多级嵌套执行框架 | ❌ 未集成 (官方示例: 1day→30min→5min) |
| **TopkDropoutStrategy** | 核心选股策略 | ✅ 完全对齐 |
| **Signal-based Strategies** | 基于信号的交易策略 | ✅ 已支持 |
| **Rule-based Strategies** | TWAPStrategy, SBBStrategyEMA 等 | ⚠️ 部分支持 (NestedExecutor 未对齐) |
| **Portfolio Analysis** | risk_analysis, indicator_analysis | ⚠️ 手动实现 (未使用官方 `contrib.evaluate.risk_analysis`) |
| **Cost Model** | open_cost, close_cost, min_cost | ✅ 已支持 |
| **Limit Threshold** | 涨跌停处理 | ✅ 已支持 |

### 2.5 强化学习 (Reinforcement Learning)

| 特性 | 描述 | 麒麟对齐状态 |
|------|------|-------------|
| **RL Framework** | Tianshou 集成 (<=0.4.10) | ❌ 未集成 |
| **Order Execution RL** | TWAP/PPO/OPDS 订单执行 | ❌ 未集成 |
| **RL Examples** | examples/rl_order_execution/ | ❌ 未集成 |

### 2.6 高级特性 (Advanced Features)

| 特性 | 描述 | 麒麟对齐状态 |
|------|------|-------------|
| **NestedDecision** | 多层级决策框架 (Oct 2021) | ❌ 未集成 |
| **High-Freq Microstructure** | 订单簿/微观结构信号 | ⚠️ 模拟数据阶段 |
| **Custom Operators** | DayLast, FFillNan, BFillNan 等 | ⚠️ 部分使用 (高频模块) |
| **R&D-Agent-Quant** | LLM 辅助量化研发 (2025 新增) | ❌ 未集成 |
| **BPQP (End-to-End Learning)** | 端到端投资组合学习 (under review) | ❌ 未集成 |

### 2.7 报告与可视化 (Reporting & Visualization)

| 特性 | 描述 | 麒麟对齐状态 |
|------|------|-------------|
| **risk_analysis 函数** | 官方 `contrib.evaluate.risk_analysis` | ⚠️ **未使用** (手动计算指标) |
| **indicator_analysis** | 交易指标统计 | ⚠️ 部分使用 |
| **Portfolio Report** | report_normal/report_long_short | ✅ 已支持 |
| **IC Analysis** | IC/ICIR/Rank IC 计算 | ✅ 已对齐 (方法论一致) |
| **Plotly Graphs** | 官方可视化组件 | ❌ 麒麟使用 Matplotlib/Streamlit |
| **Position Analysis** | analysis_position 报告 | ⚠️ 部分支持 |

---

## 3. 麒麟项目特有增强特性

| 特性 | 描述 |
|------|------|
| **一进二 (Limitup) 策略模板** | limitup_lightgbm/lstm/alstm/transformer 专用配置 |
| **AKShare 数据集成** | 高频/实时数据获取 |
| **Streamlit Web UI** | 全栈可视化界面 (qrun/backtest/model_zoo/ic_analysis 等 8+ 标签页) |
| **增强初始化** | qlib_init.py 统一初始化入口 |
| **Layer2 适配层** | layer2_qlib/qlib_integration.py 封装 |

---

## 4. 对齐差距清单 (Gap Analysis)

### 4.1 高优先级差距 (影响一进二策略核心功能)

| 差距项 | 影响 | 建议优先级 |
|-------|------|-----------|
| **risk_analysis 函数未使用** | 回测指标计算可能与官方不一致 | P0 (立即修复) |
| **模型降级未标注** | 用户不知道 TRA/HIST 实际使用的是 Transformer | P0 (UI 透明化) |
| **NestedExecutor 缺失** | 无法实现多级交易策略 (日内分级执行) | P1 (对一进二 T+0 优化有用) |
| **RL Order Execution 缺失** | 无法使用 RL 优化订单执行 | P1 (智能拆单对大单量有价值) |
| **高频数据链路不完整** | 微观结构信号依赖模拟数据 | P1 (真实 tick 级数据接入) |

### 4.2 中优先级差距 (功能完整性)

| 差距项 | 影响 | 建议优先级 |
|-------|------|-----------|
| **Qlib-Server 在线模式** | 无法部署在线推理服务 | P2 |
| **Model Rolling 自动化** | 模型更新需手动操作 | P2 |
| **完整报告生成** | 缺少官方 Plotly 可视化组件 | P2 |
| **PIT 数据库明确使用** | 财务数据时点正确性未验证 | P2 |

### 4.3 低优先级差距 (研究性特性)

| 差距项 | 影响 | 建议优先级 |
|-------|------|-----------|
| **R&D-Agent-Quant** | LLM 辅助因子挖掘 | P3 |
| **BPQP** | 学术前沿方法 | P3 |
| **TFT 模型** | 额外的 Transformer 变体 | P3 |

---

## 5. 重大差异点 (Breaking Changes & Notes)

### 5.1 核心差异

1. **风险指标计算方式**  
   - **官方**: `from qlib.contrib.evaluate import risk_analysis; risk_analysis(returns, freq="day")`
   - **麒麟**: 手动计算 mean/std/sharpe/max_drawdown 等指标
   - **风险**: 可能导致年化收益率/夏普比率计算不一致 (官方使用累加而非累乘)

2. **模型 Fallback 策略**  
   - **官方**: TRA/HIST/DoubleEnsemble 作为独立模型实现
   - **麒麟**: 部分模型使用 Transformer 作为降级方案
   - **风险**: 性能可能与官方论文结果不符

3. **NestedExecutor 架构**  
   - **官方**: 支持 1day→30min→5min 三级嵌套执行
   - **麒麟**: 仅单层 SimulatorExecutor
   - **风险**: 无法实现日内多次调仓策略

### 5.2 配置与初始化

1. **多初始化模式**  
   - 当前存在 `qlib_init.py`, `qlib_enhanced/__init__.py`, `layer2_qlib` 等多处初始化逻辑
   - 需要统一为配置中心 (支持离线/在线/缓存/版本管理)

2. **硬编码路径**  
   - `web/tabs/qlib_qrun_workflow_tab.py` 存在 `G:/test/qilin_stack` 绝对路径
   - 需要改为配置文件或环境变量

---

## 6. 版本兼容性检查清单

### 6.1 依赖版本对齐

| 依赖 | Qlib 要求 | 麒麟当前版本 | 对齐状态 |
|------|----------|-------------|---------|
| pandas | >= 1.1 | ❓ 待检查 | - |
| tianshou | <= 0.4.10 (rl) | ❌ 未安装 | 待补充 |
| torch | (rl 依赖) | ❓ 待检查 | - |
| numpy | < 2.0.0 (rl) | ❓ 待检查 | - |
| mlflow | - | ✅ 已安装 | ✅ |
| redis | - | ⚠️ 可选 | ⚠️ |

### 6.2 API 变更风险点

- **setuptools_scm 版本管理**: 需要确保 `qlib.__version__` 能正确读取
- **qlib_reset_version 机制**: 支持手动覆盖版本号 (用于 Qlib-Server 兼容性)
- **region 配置**: REG_CN vs REG_US 数据源切换

---

## 7. 下一步行动 (Next Steps)

### 7.1 任务 1 (当前) - 基线对齐 ✅ 完成输出

1. ✅ 提取 Qlib 版本: v0.9.7-9-gbb7ab1cf
2. ✅ 解析依赖清单: pyproject.toml 完整记录
3. ✅ 特性清单梳理: 数据/模型/工作流/回测/RL/高级特性 6 大类
4. ✅ 对齐差距分析: P0/P1/P2/P3 优先级划分
5. ✅ 生成基线文档: 本文件

### 7.2 任务 2 - 麒麟代码映射与静态扫描

- 扫描 `web/tabs/*.py` 所有 UI 标签页
- 扫描 `qlib_enhanced/`, `layer2_qlib/`, `app/integrations/`
- 生成代码使用矩阵 (哪些官方 API 被调用,哪些未使用)

### 7.3 任务 3 - 统一初始化与配置中心

- 合并多处初始化逻辑
- 设计配置文件结构 (离线/在线/缓存/版本)
- 移除硬编码路径

---

## 附录 A: Qlib 官方 Examples 结构

```
examples/
├── benchmarks/           # LightGBM, TFT, TRA 等基准测试
├── data_demo/            # 数据缓存与内存优化示例
├── highfreq/             # 高频数据处理 (1min)
├── model_interpreter/    # 模型可解释性
├── model_rolling/        # 模型滚动更新
├── nested_decision_execution/  # NestedExecutor 嵌套执行 ⚠️ 关键示例
├── online_srv/           # Qlib-Server 在线服务
├── orderbook_data/       # 订单簿数据处理
├── portfolio/            # 投资组合优化 (Enhanced Indexing)
├── rolling_process_data/ # 滚动数据处理
├── run_all_model.py      # 批量模型测试
└── workflow_by_code.py   # 代码化工作流示例
```

---

## 附录 B: Qlib README "What's NEW" 重要特性时间线

- **2025**: R&D-Agent-Quant 集成
- **Under Review**: BPQP 端到端学习
- **Nov 2022**: RL Learning Framework (多个 PR)
- **Apr 2022**: HIST & IGMTF 模型
- **Mar 2022**: Point-in-Time 数据库
- **Oct 2021**: Nested Decision Framework ⚠️ 与一进二多级策略相关
- **Jul 2021**: TRA & Transformer & Localformer
- **May 2021**: Online Serving & Auto Model Rolling
- **Jan-Feb 2021**: High-Frequency Data (1min)

---

**文档维护**: 本文档应随 Qlib 主分支更新而定期同步  
**责任人**: AI Agent / 项目负责人  
**更新频率**: 每次 Qlib 版本升级后 (建议每季度检查一次)
