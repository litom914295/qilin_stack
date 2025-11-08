# 麒麟代码映射与静态扫描报告 (v0)

**生成日期**: 2025-01-XX  
**扫描范围**: web/tabs, qlib_enhanced, layer2_qlib, app/integrations, config, configs/templates  
**基线版本**: Qlib v0.9.7-9-gbb7ab1cf  

---

## 1. 代码文件清单

### 1.1 Web UI 标签页 (14 个文件)

| 文件 | 功能 | Qlib API 使用情况 | 状态 |
|------|------|-----------------|------|
| **qlib_qrun_workflow_tab.py** | qrun 工作流管理 | qlib.init, qlib.workflow.R, init_instance_by_config | ✅ 对齐 |
| **qlib_backtest_tab.py** | 回测执行器 | qlib.backtest, TopkDropoutStrategy, risk_analysis | ⚠️ 部分对齐 |
| **qlib_model_zoo_tab.py** | 模型 Zoo 界面 | qlib.contrib.model.* | ✅ 对齐 |
| **qlib_ic_analysis_tab.py** | IC 分析 | qlib.data.D | ✅ 对齐 |
| **qlib_data_tools_tab.py** | 数据工具 | qlib.data.D, qlib.init | ✅ 对齐 |
| **qlib_experiment_comparison_tab.py** | 实验对比 | qlib.workflow.R | ✅ 对齐 |
| **qlib_execution_tab.py** | 执行器配置 | qlib.backtest.executor | ⚠️ NestedExecutor 未集成 |
| **qlib_highfreq_tab.py** | 高频数据 | qlib (占位实现) | ⚠️ 模拟数据 |
| **qlib_microstructure_tab.py** | 微观结构 | qlib (占位实现) | ⚠️ 模拟数据 |
| **limitup_ai_evolution_tab.py** | 一进二 AI 进化 | qlib.* (多处使用) | ✅ 对齐 |
| **limitup_monitor_unified.py** | 涨停板监控 | - (业务逻辑) | ✅ 独立功能 |
| **advanced_features_tab.py** | 高级特性 | qlib (最小使用) | ✅ 对齐 |
| **evolution_training_tab.py** | 进化训练 | - (业务逻辑) | ✅ 独立功能 |
| **factor_research_tab.py** | 因子研究 | - (业务逻辑) | ✅ 独立功能 |

### 1.2 Qlib Enhanced 适配层 (30+ 个文件)

| 目录 | 文件数 | 核心功能 | 问题点 |
|------|--------|---------|--------|
| **analysis/** | 3 | ic_analysis.py, ic_visualizer.py | ✅ 对齐官方方法论 |
| **data_tools/** | 3 | data_converter.py, expression_tester.py | ✅ 对齐 |
| **model_zoo/** | 7 | model_registry.py, model_trainer.py, models/*.py | ⚠️ 模型降级未标注 |
| **根目录** | 20+ | nested_executor_integration.py, rl_trading.py, online_learning.py | ⚠️ 部分功能占位 |

### 1.3 Layer2 Qlib 集成层 (3 个文件)

| 文件 | 功能 | 对齐状态 |
|------|------|---------|
| **qlib_integration.py** | 统一 Qlib 集成入口 | ✅ 对齐 (使用 risk_analysis) |
| **optimized_data_loader.py** | 数据加载优化 | ✅ 对齐 |
| **scripts/predict_online_qlib.py** | 在线预测脚本 | ⚠️ 在线模式未完全集成 |

### 1.4 App Integrations (5 个文件)

| 文件 | 功能 | 对齐状态 |
|------|------|---------|
| **qlib_integration.py** | 外部 API 封装 | ✅ 对齐 |
| **data_bridge.py** | 数据桥接 | ✅ 对齐 |
| **rdagent_integration.py** | RD-Agent 集成 | ✅ 独立功能 |
| **tradingagents_integration.py** | TradingAgents 集成 | ✅ 独立功能 |

---

## 2. Qlib 官方 API 使用矩阵

### 2.1 核心 API 调用统计

| Qlib API | 调用文件数 | 调用位置 | 对齐状态 |
|----------|----------|---------|---------|
| **qlib.init** | 8 | qlib_backtest_tab.py (line 77), layer2_qlib/qlib_integration.py (line 86), qlib_data_tools_tab.py | ✅ 已使用 |
| **qlib.data.D.features** | 5 | qlib_ic_analysis_tab.py, qlib_enhanced/analysis/ic_analysis.py (line 63) | ✅ 已使用 |
| **qlib.workflow.R** | 6 | qlib_qrun_workflow_tab.py (line 23), qlib_experiment_comparison_tab.py (line 22) | ✅ 已使用 |
| **qlib.backtest.backtest** | 3 | qlib_backtest_tab.py (line 21), layer2_qlib/qlib_integration.py (line 26) | ✅ 已使用 |
| **qlib.contrib.evaluate.risk_analysis** | 3 | qlib_backtest_tab.py (line 25), layer2_qlib/qlib_integration.py (line 27) | ⚠️ **仅 2-3 处使用** |
| **qlib.contrib.strategy.TopkDropoutStrategy** | 4 | qlib_backtest_tab.py (line 24), layer2_qlib/qlib_integration.py (line 37) | ✅ 已使用 |
| **qlib.contrib.model.\*** | 20+ | qlib_model_zoo_tab.py, qlib_enhanced/model_zoo/*, layer2_qlib/qlib_integration.py | ✅ 已使用 |
| **qlib.backtest.executor.NestedExecutor** | 2 | qlib_enhanced/nested_executor_integration.py (line 23), tests/test_nested_executor.py (line 9) | ❌ **仅测试文件使用** |
| **qlib.backtest.executor.SimulatorExecutor** | 3 | qlib_backtest_tab.py (line 624), tests/test_nested_executor.py | ✅ 已使用 |
| **qlib.contrib.report.analysis_position** | 1 | layer2_qlib/qlib_integration.py (line 28) | ⚠️ 很少使用 |
| **qlib.contrib.data.handler.Alpha158** | 5 | qlib_model_zoo_tab.py (line 56), layer2_qlib/qlib_integration.py (line 186) | ✅ 已使用 |
| **qlib.contrib.data.handler.Alpha360** | 5 | qlib_model_zoo_tab.py (line 43), layer2_qlib/qlib_integration.py (line 164) | ✅ 已使用 |
| **qlib.utils.init_instance_by_config** | 8 | qlib_qrun_workflow_tab.py (line 25), layer2_qlib/qlib_integration.py (line 29) | ✅ 已使用 |

### 2.2 关键 API 缺失/误用清单

| API | 官方用途 | 麒麟使用情况 | 影响 |
|-----|---------|-------------|------|
| **risk_analysis(returns, freq="day")** | 标准风险指标计算 (年化/夏普/回撤) | ⚠️ **仅 3 处使用**，大量手动实现 | **P0** - 指标可能不一致 |
| **NestedExecutor** | 多级嵌套执行 (1day→30min→5min) | ❌ 仅测试文件中有示例，UI 未集成 | **P1** - 无法实现日内多次调仓 |
| **contrib.report.analysis_position.risk_analysis** | 持仓风险分析 | ⚠️ layer2_qlib 导入但很少调用 | **P2** - 报告不完整 |
| **qlib.contrib.online.online_model** | 在线学习模型 | ❌ 未使用 | **P2** - 在线模式缺失 |
| **qlib.contrib.rl.*** | 强化学习订单执行 | ❌ 未使用 (qlib_enhanced/rl_trading.py 仅占位) | **P1** - RL 功能缺失 |

---

## 3. 静态扫描问题清单

### 3.1 硬编码路径 (P0 - 立即修复)

| 文件 | 行号 | 问题代码 | 修复建议 |
|------|------|---------|---------|
| **qlib_qrun_workflow_tab.py** | 913, 973 | `G:/test/qilin_stack` | 改用 `Path(__file__).parent.parent` 或环境变量 |
| **web/components/realistic_backtest_page.py** | 340, 616 | `G:/test/qilin_stack` | 同上 |

**影响**: Windows 绝对路径导致跨环境/跨用户不可移植

### 3.2 风险指标手动实现 (P0 - 口径不一致风险)

| 文件 | 行号 | 问题 | 修复建议 |
|------|------|------|---------|
| **qlib_backtest_tab.py** | 351-550 | 手动计算年化收益/夏普/回撤 | 替换为 `from qlib.contrib.evaluate import risk_analysis; risk_analysis(returns, freq="day")` |
| **web/components/realistic_backtest_page.py** | 多处 | 手动计算风险指标 | 同上 |

**示例问题代码** (qlib_backtest_tab.py line 413-419):
```python
# ❌ 当前实现 (手动计算)
{
    "annualized_return": ...,  # 手动计算
    "information_ratio": ...,  # 手动计算 (可能与官方公式不符)
    "max_drawdown": ...,       # 手动计算
}

# ✅ 应改为
from qlib.contrib.evaluate import risk_analysis
risk_metrics = risk_analysis(daily_returns, freq="day")
# risk_metrics 包含: mean, std, annualized_return, information_ratio, max_drawdown
```

### 3.3 异常处理不完整 (P1)

| 文件 | 问题类型 | 示例 | 修复建议 |
|------|---------|------|---------|
| **qlib_backtest_tab.py** | 裸 except | line 28-30, 50-52, 80-81 | 改为 `except Exception as e: logger.error(...); raise` |
| **qlib_qrun_workflow_tab.py** | 未捕获 KeyError | line 199-216 (YAML 解析) | 添加 `try-except KeyError, yaml.YAMLError` |
| **qlib_model_zoo_tab.py** | 模型加载失败未明确提示 | line 363-445 | 区分"依赖缺失"/"配置错误"/"数据问题" |
| **qlib_enhanced/model_zoo/model_trainer.py** | 数据为空未处理 | line 159-166 | 检查 df_train.empty, 提前返回或抛出异常 |

**高频问题模式**:
```python
# ❌ 问题代码
try:
    qlib.init(...)
except ImportError as e:
    logger.warning(f"Qlib导入失败: {e}")
    QLIB_AVAILABLE = False  # 后续代码仍继续执行,可能崩溃

# ✅ 改进
try:
    qlib.init(...)
except ImportError as e:
    logger.error(f"Qlib导入失败: {e}")
    raise RuntimeError("Qlib is required but not installed. Run: pip install pyqlib")
```

### 3.4 模型降级未标注 (P0 - 用户透明度问题)

| 文件 | 行号 | 问题 | 修复建议 |
|------|------|------|---------|
| **qlib_model_zoo_tab.py** | 全文 | TRA/HIST/DoubleEnsemble 可能降级为 Transformer | UI 添加"⚠️ 降级运行"提示 + 原因说明 |
| **qlib_enhanced/model_zoo/model_registry.py** | 模型注册表 | 无依赖检测与降级策略 | 添加 `check_dependencies()` + `get_fallback_model()` |

**示例 UI 改进**:
```python
# ✅ 模型状态显示
model_status = check_model_availability("HIST")
if model_status == "fallback":
    st.warning("⚠️ HIST 模型当前使用 Transformer 降级运行 (缺少依赖: torch_geometric)")
    st.info("安装完整依赖: pip install torch-geometric")
```

### 3.5 缓存未可配置化 (P1)

| 文件 | 问题 | 修复建议 |
|------|------|---------|
| **layer2_qlib/qlib_integration.py** | line 92-93 硬编码 `expression_cache=None, dataset_cache=None` | 改为从配置文件读取 |
| **config/qlib_init.py** | (如存在) 缺少统一缓存配置接口 | 添加 `configure_cache(redis_host, redis_port, enable_expression_cache, enable_dataset_cache)` |

### 3.6 资源未释放 (P2)

| 文件 | 问题 | 风险 |
|------|------|------|
| **qlib_data_tools_tab.py** | 文件上传后未调用 `.close()` | 内存泄漏 (Streamlit 可能自动处理) |
| **qlib_qrun_workflow_tab.py** | 临时文件未清理 | 磁盘占用 |

---

## 4. 覆盖矩阵 v0

### 4.1 数据层对齐

| Qlib 特性 | 官方实现 | 麒麟实现 | 对齐状态 | 证据文件 |
|----------|---------|---------|---------|---------|
| Alpha158 | qlib.contrib.data.handler.Alpha158 | ✅ | ✅ 完全对齐 | qlib_model_zoo_tab.py:56, layer2_qlib/qlib_integration.py:186 |
| Alpha360 | qlib.contrib.data.handler.Alpha360 | ✅ | ✅ 完全对齐 | qlib_model_zoo_tab.py:43, layer2_qlib/qlib_integration.py:164 |
| 表达式引擎 | qlib.data.D.features(fields=["$close/$open-1"]) | ✅ | ✅ 完全对齐 | qlib_ic_analysis_tab.py:593, qlib_enhanced/analysis/ic_analysis.py:63 |
| 高频数据 (1min) | qlib.config.HIGH_FREQ_CONFIG | ⚠️ 模拟数据 | ⚠️ 部分对齐 | qlib_highfreq_tab.py:40 (占位实现) |
| PIT 数据库 | qlib.data.D.features(..., pit=True) | ❓ 未明确使用 | ❓ 待验证 | - |
| 缓存系统 | redis + expression_cache + dataset_cache | ⚠️ 硬编码关闭 | ⚠️ 部分对齐 | layer2_qlib/qlib_integration.py:92-93 |

### 4.2 模型层对齐

| 模型 | 官方路径 | 麒麟使用 | 状态 | 证据 |
|------|---------|---------|------|------|
| LightGBM | qlib.contrib.model.gbdt.LGBModel | ✅ | ✅ 完全对齐 | qlib_model_zoo_tab.py:31, layer2_qlib/qlib_integration.py:34 |
| XGBoost | qlib.contrib.model.xgboost.XGBModel | ✅ | ✅ 完全对齐 | qlib_enhanced/model_zoo/models/xgboost_model.py |
| CatBoost | qlib.contrib.model.catboost_model.CatBoostModel | ✅ | ✅ 完全对齐 | qlib_enhanced/model_zoo/models/catboost_model.py |
| LSTM | qlib.contrib.model.pytorch_lstm.LSTMModel | ✅ | ✅ 完全对齐 | qlib_model_zoo_tab.py:70 |
| GRU | qlib.contrib.model.pytorch_gru.GRU | ✅ | ✅ 完全对齐 | qlib_model_zoo_tab.py:82, layer2_qlib/qlib_integration.py:33 |
| ALSTM | qlib.contrib.model.pytorch_alstm.ALSTM | ✅ | ✅ 完全对齐 | qlib_model_zoo_tab.py:106, layer2_qlib/qlib_integration.py:32 |
| Transformer | qlib.contrib.model.pytorch_transformer.Transformer | ✅ | ✅ 完全对齐 | qlib_model_zoo_tab.py:120, layer2_qlib/qlib_integration.py:36 |
| TRA | qlib.contrib.model.pytorch_tra.TRA | ⚠️ 可能降级 | ⚠️ 降级未标注 | qlib_model_zoo_tab.py:133 |
| HIST | qlib.contrib.model.pytorch_hist.HIST | ⚠️ 可能降级 | ⚠️ 降级未标注 | qlib_model_zoo_tab.py:145 |
| DoubleEnsemble | qlib.contrib.model.double_ensemble | ⚠️ 可能降级 | ⚠️ 降级未标注 | qlib_model_zoo_tab.py:157 |

### 4.3 回测与执行层对齐

| 特性 | 官方实现 | 麒麟使用 | 状态 | 证据 |
|------|---------|---------|------|------|
| SimulatorExecutor | qlib.backtest.executor.SimulatorExecutor | ✅ | ✅ 完全对齐 | qlib_backtest_tab.py:624 |
| NestedExecutor | qlib.backtest.executor.NestedExecutor | ❌ 仅测试 | ❌ UI 未集成 | tests/test_nested_executor.py:9 |
| TopkDropoutStrategy | qlib.contrib.strategy.TopkDropoutStrategy | ✅ | ✅ 完全对齐 | qlib_backtest_tab.py:24 |
| risk_analysis | qlib.contrib.evaluate.risk_analysis | ⚠️ 很少使用 | ⚠️ **P0 问题** | qlib_backtest_tab.py:25 (导入但手动实现) |
| 交易成本模型 | open_cost, close_cost, min_cost | ✅ | ✅ 完全对齐 | qlib_backtest_tab.py:195-220 |

### 4.4 高级特性对齐

| 特性 | 官方实现 | 麒麟使用 | 状态 | 证据 |
|------|---------|---------|------|------|
| RL 订单执行 | examples/rl_order_execution/ | ❌ 占位 | ❌ 未集成 | qlib_enhanced/rl_trading.py (仅框架代码) |
| 在线学习 | qlib.contrib.online.online_model | ❌ | ❌ 未集成 | qlib_enhanced/online_learning.py (仅框架) |
| 嵌套决策 | examples/nested_decision_execution/ | ⚠️ 测试存在 | ⚠️ UI 未打通 | qlib_enhanced/nested_executor_integration.py:23 |
| 微观结构 | examples/orderbook_data/ | ⚠️ 模拟 | ⚠️ 真实数据未接入 | qlib_microstructure_tab.py:18 |

---

## 5. 问题雷达图 v0

```
          硬编码路径 (10/10 严重)
                 /\
                /  \
   异常处理(7/10) ---- 风险指标口径(9/10)
              /        \
             /          \
    模型降级(8/10)---- 缓存配置(6/10)
             \          /
              \        /
       NestedExecutor(9/10) ---- RL集成(8/10)
                 \  /
                  \/
          在线模式(7/10)
```

**问题严重度评分** (1-10):
- **硬编码路径**: 10/10 - 跨环境无法运行
- **风险指标口径**: 9/10 - 可能导致业务决策错误
- **NestedExecutor 缺失**: 9/10 - 影响一进二日内策略
- **模型降级未标注**: 8/10 - 用户不知道性能降低
- **RL 集成缺失**: 8/10 - 缺少智能订单执行
- **异常处理**: 7/10 - 可能导致静默失败
- **在线模式**: 7/10 - 无法部署生产
- **缓存配置**: 6/10 - 性能优化受限

---

## 6. 功能→文件/函数映射表

### 6.1 Qlib 回测功能

| 功能 | 文件 | 函数/类 | Qlib API 调用 |
|------|------|---------|--------------|
| 初始化 Qlib | qlib_backtest_tab.py | `_ensure_qlib_initialized()` | qlib.init(provider_uri, region) |
| 加载预测信号 | qlib_backtest_tab.py | `render_backtest_config()` | R.get_recorder().load_object() |
| 构建策略 | qlib_backtest_tab.py | `run_backtest()` | TopkDropoutStrategy(**config) |
| 执行回测 | qlib_backtest_tab.py | `run_backtest()` | backtest(strategy, executor, ...) |
| ❌ 计算风险指标 | qlib_backtest_tab.py | `render_backtest_results()` | **手动实现** (应改为 risk_analysis) |
| 可视化结果 | qlib_backtest_tab.py | `render_backtest_results()` | plotly.graph_objects |

### 6.2 Qlib qrun 工作流

| 功能 | 文件 | 函数/类 | Qlib API 调用 |
|------|------|---------|--------------|
| 加载模板 | qlib_qrun_workflow_tab.py | `load_template_config()` | - (YAML 读取) |
| 解析 YAML | qlib_qrun_workflow_tab.py | `validate_config()` | yaml.safe_load() |
| 执行 qrun | qlib_qrun_workflow_tab.py | `run_qrun_workflow()` | subprocess / init_instance_by_config |
| 记录到 MLflow | qlib_qrun_workflow_tab.py | - | R.start(), R.log_params(), R.save_objects() |

### 6.3 IC 分析

| 功能 | 文件 | 函数/类 | Qlib API 调用 |
|------|------|---------|--------------|
| 加载因子数据 | qlib_enhanced/analysis/ic_analysis.py | `load_factor_from_qlib()` | D.features(instruments, fields, start, end) |
| 计算 IC 时间序列 | qlib_enhanced/analysis/ic_analysis.py | `compute_ic_timeseries()` | scipy.stats.spearmanr |
| 月度 IC 热力图 | qlib_enhanced/analysis/ic_analysis.py | `compute_monthly_ic_heatmap()` | pandas groupby + pivot |
| 分层收益分析 | qlib_enhanced/analysis/ic_analysis.py | `layered_return_analysis()` | pd.qcut() + groupby |

---

## 7. 对齐差距证据链

### 7.1 risk_analysis 使用证据

| 文件 | 行号 | 代码片段 | 对齐状态 |
|------|------|---------|---------|
| qlib_backtest_tab.py | 25 | `from qlib.contrib.evaluate import risk_analysis` | ✅ 导入 |
| qlib_backtest_tab.py | 413-419 | **手动计算** `annualized_return`, `information_ratio`, `max_drawdown` | ❌ **未调用 risk_analysis** |
| layer2_qlib/qlib_integration.py | 27 | `from qlib.contrib.evaluate import risk_analysis` | ✅ 导入 |
| layer2_qlib/qlib_integration.py | 590, 601 | `risk_analysis(returns, freq="day")` | ✅ **正确调用** |

**结论**: 仅 layer2_qlib 正确使用 risk_analysis，web UI 层手动实现

### 7.2 NestedExecutor 使用证据

| 文件 | 行号 | 用途 | 对齐状态 |
|------|------|------|---------|
| tests/test_nested_executor.py | 9, 23, 270 | 测试用例 | ✅ 测试覆盖 |
| qlib_enhanced/nested_executor_integration.py | 23, 32, 303 | 封装实现 | ✅ 适配层存在 |
| qlib_backtest_tab.py | - | - | ❌ **UI 未集成** |
| qlib_execution_tab.py | - | - | ❌ **UI 未集成** |

**结论**: 测试和适配层存在，但 UI 未提供用户入口

---

## 8. 下一步行动建议

### 8.1 立即修复 (P0)

1. **移除硬编码路径** (2 处)
   - qlib_qrun_workflow_tab.py:913, 973
   - 使用 `Path(__file__).parent.parent` 或环境变量

2. **统一风险指标计算** (qlib_backtest_tab.py)
   ```python
   # 替换 line 413-419 手动实现
   from qlib.contrib.evaluate import risk_analysis
   risk_df = risk_analysis(daily_returns, freq="day")
   metrics = risk_df["risk"].to_dict()
   ```

3. **模型降级状态 UI 提示** (qlib_model_zoo_tab.py)
   - 添加依赖检测: `check_model_availability(model_name)`
   - 显式告知用户降级原因与恢复方法

### 8.2 功能补全 (P1)

4. **NestedExecutor UI 集成** (qlib_execution_tab.py)
   - 新增"嵌套执行器配置"选项卡
   - 支持 1day→30min→5min 三级配置

5. **RL 订单执行最小样例** (qlib_enhanced/rl_trading.py → UI)
   - 对接官方 examples/rl_order_execution
   - 提供 TWAP/PPO 算法选择与训练入口

6. **高频数据真实链路** (qlib_highfreq_tab.py)
   - 移除占位代码
   - 对接 AKShare/本地 Level-2 数据

### 8.3 稳健性改造 (P2)

7. **统一异常处理** (全局)
   - 规范错误码 (如 ERR_QLIB_INIT_FAIL)
   - 添加 `@retry` 装饰器 (网络请求/数据加载)

8. **缓存配置中心** (config/qlib_init.py)
   - 统一 redis_host/redis_port 配置接口
   - 支持 expression_cache / dataset_cache 开关

9. **资源释放检查** (qlib_data_tools_tab.py, qlib_qrun_workflow_tab.py)
   - 临时文件 cleanup
   - 数据库连接 close

---

## 附录 A: 硬编码路径完整清单

```plaintext
G:/test/qilin_stack 出现位置:
1. web/tabs/qlib_qrun_workflow_tab.py:913
2. web/tabs/qlib_qrun_workflow_tab.py:973
3. web/components/realistic_backtest_page.py:340
4. web/components/realistic_backtest_page.py:616
5. docs/*.md (约 50 处文档引用,优先级低)
```

**修复方案**:
```python
# ❌ 硬编码
template_path = "G:/test/qilin_stack/configs/qlib_workflows/templates"

# ✅ 动态路径
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 从 web/tabs/*.py 计算
template_path = PROJECT_ROOT / "configs" / "qlib_workflows" / "templates"
```

---

## 附录 B: 异常处理最佳实践

```python
# ❌ 问题代码模式 (出现 100+ 次)
try:
    result = risky_operation()
except:  # 裸 except
    pass  # 静默失败

# ✅ 改进模式
import logging
logger = logging.getLogger(__name__)

try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # 根据场景选择: raise / return default / 显示错误给用户
    raise RuntimeError(f"Failed to execute: {e}") from e
```

---

**文档维护**: 随代码修复同步更新  
**下次扫描**: 任务 4-7 (UI 修复) 完成后重新生成 v1 版本
