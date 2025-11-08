# 涨停监控模块功能对比报告

**生成时间**: 2025-11-07T10:43:31.832614

## 📊 执行摘要

### 总体覆盖率

| 项目 | 覆盖率 | 说明 |
|------|--------|------|
| 标签页 | 100.0% | limitup_monitor.py 覆盖 limitup_dashboard.py 的标签页功能 |
| 函数 | 0.0% | 核心函数的实现覆盖率 |

### 关键发现

- **limitup_dashboard.py**: 506 行代码, 3 个函数
- **limitup_monitor.py**: 492 行代码, 10 个函数

---

## 🏷️ 标签页对比

### 共同标签页 (5个)

- ✅ ⚙️ RL参数推荐
- ✅ 📋 今日信号
- ✅ 🤖 AI决策过程
- ✅ 📊 回测结果
- ✅ 🧠 涨停原因解释

### limitup_dashboard.py 独有标签页 (0个)

- *无*

### limitup_monitor.py 独有标签页 (0个)

- *无*

---

## 🔧 函数对比

### 共同函数

- *无共同函数*

### limitup_dashboard.py 独有函数 (3个)

- ⚠️ `find_latest(pattern)` - 查找最新文件
- ⚠️ `load_csv_safe(path)` - 安全加载CSV
- ⚠️ `load_json_safe(path)` - 安全加载JSON

### limitup_monitor.py 独有函数 (10个)

- ➕ `get_available_dates(reports_dir)` - 获取可用的报告日期
- ➕ `load_auction_report(reports_dir, date)` - 加载竞价报告
- ➕ `load_rl_decision(reports_dir, date)` - 加载RL决策结果
- ➕ `load_rl_weights(config_dir)` - 加载RL权重配置
- ➕ `render()` - 渲染涨停板监控主界面
- ➕ `render_ai_decision(reports_dir, config_dir, selected_date)` - Tab2: AI决策过程
- ➕ `render_backtest_results(reports_dir)` - Tab5: 回测结果
- ➕ `render_limitup_explanation(reports_dir, selected_date)` - Tab3: 涨停原因解释
- ➕ `render_rl_recommendations(config_dir)` - Tab4: RL参数推荐
- ➕ `render_today_signals(reports_dir, selected_date)` - Tab1: 今日信号

---

## 📊 Streamlit组件使用对比

| 组件 | limitup_dashboard.py | limitup_monitor.py | 差异 |
|------|---------------------|-------------------|------|
| `st.caption()` | 0 | 1 | +1 |
| `st.columns()` | 6 | 6 | +0 |
| `st.dataframe()` | 6 | 5 | -1 |
| `st.divider()` | 0 | 7 | +7 |
| `st.error()` | 2 | 3 | +1 |
| `st.expander()` | 1 | 1 | +0 |
| `st.header()` | 0 | 1 | +1 |
| `st.info()` | 5 | 8 | +3 |
| `st.markdown()` | 16 | 0 | -16 |
| `st.metric()` | 1 | 18 | +17 |
| `st.pyplot()` | 6 | 6 | +0 |
| `st.selectbox()` | 0 | 1 | +1 |
| `st.set_page_config()` | 1 | 0 | -1 |
| `st.subheader()` | 5 | 18 | +13 |
| `st.tabs()` | 1 | 1 | +0 |
| `st.text_input()` | 0 | 2 | +2 |
| `st.title()` | 1 | 0 | -1 |
| `st.warning()` | 5 | 7 | +2 |
| `st.write()` | 0 | 2 | +2 |

---

## 📈 代码指标对比

| 指标 | limitup_dashboard.py | limitup_monitor.py |
|------|---------------------|-------------------|
| total_lines | 506 | 492 |
| total_functions | 3 | 10 |
| total_classes | 0 | 0 |
| render_functions | 0 | 6 |

---

## 🎯 结论与建议

### 功能覆盖情况


✅ **功能基本一致** (覆盖率 ≥ 90%)

limitup_monitor.py 已经实现了 limitup_dashboard.py 的绝大部分功能，可以安全地替代使用。

**建议：**
1. 确认 limitup_monitor.py 已正确集成到 unified_dashboard.py
2. 将 limitup_dashboard.py 标记为已归档或删除
3. 更新相关文档，统一使用 unified_dashboard.py 作为主入口


### 数据源对比

#### limitup_dashboard.py 使用的数据源：

- `- [回测结果](#tab-backtest)`
- `reports`
- `rl_weights.json`
- `backtest`
- `metrics_*.json`
- `auction_report_`
- `*.json`
- `未找到回测结果

请运行: `python app/backtest_engine.py``
- `equity_curve_*.csv`
- `trade_log_*.csv`

#### limitup_monitor.py 使用的数据源：

- `backtest`
- `*.json`
- `文件位置: `config/rl_weights.json``
- `请先运行: `python app/backtest_engine.py``
- `metrics_*.json`
- `equity_curve_*.csv`
- `trade_log_*.csv`
- `auction_report_`
- `_*.json`
- `rl_weights.json`
- `reports`

---

## 📝 附录

### 详细函数列表

#### limitup_dashboard.py 函数列表

- **find_latest**`(pattern)` (第55行, 5行代码)
  - 查找最新文件
- **load_json_safe**`(path)` (第62行, 7行代码)
  - 安全加载JSON
- **load_csv_safe**`(path)` (第71行, 6行代码)
  - 安全加载CSV

#### limitup_monitor.py 函数列表

- **render**`()` (第17行, 41行代码)
  - 渲染涨停板监控主界面
- **get_available_dates**`(reports_dir)` (第61行, 18行代码)
  - 获取可用的报告日期
- **render_today_signals**`(reports_dir, selected_date)` (第82行, 63行代码)
  - Tab1: 今日信号
- **render_ai_decision**`(reports_dir, config_dir, selected_date)` (第148行, 71行代码)
  - Tab2: AI决策过程
- **render_limitup_explanation**`(reports_dir, selected_date)` (第222行, 56行代码)
  - Tab3: 涨停原因解释
- **render_rl_recommendations**`(config_dir)` (第281行, 61行代码)
  - Tab4: RL参数推荐
- **render_backtest_results**`(reports_dir)` (第345行, 94行代码)
  - Tab5: 回测结果
- **load_auction_report**`(reports_dir, date)` (第444行, 16行代码)
  - 加载竞价报告
- **load_rl_decision**`(reports_dir, date)` (第463行, 15行代码)
  - 加载RL决策结果
- **load_rl_weights**`(config_dir)` (第481行, 11行代码)
  - 加载RL权重配置

---

*本报告由自动化脚本生成*
