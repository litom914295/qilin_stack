# 麒麟涨停板选股系统 - Web Dashboard使用指南

## 📊 Dashboard功能

全新的涨停板选股系统Web Dashboard，提供5大核心功能：

1. **📋 今日信号** - 实时显示竞价监控状态
2. **🤖 AI决策过程** - 可视化AI决策过程和特征权重
3. **🧠 涨停原因解释** - 查看涨停原因分析和统计
4. **⚙️ RL参数推荐** - Thompson Sampling阈值优化推荐
5. **📊 回测结果** - 查看历史交易记录和收益曲线

---

## 🚀 快速启动

### 方式1: 统一Dashboard (推荐)

涨停板监控已整合到统一Dashboard中:

```bash
streamlit run web/unified_dashboard.py
```

然后访问: **Qlib → 数据管理 → 🎯涨停板监控**

### ~方式2: 独立Dashboard~ (已归档)

⚠️ **独立Dashboard已于2025-11-07归档**

所有功能已完全集成到统一Dashboard中，请使用方式1。

如需使用归档版本，请参考 `archive/web/README.md`

### 访问地址

启动后会自动打开浏览器，访问地址通常为：
```
http://localhost:8501
```

---

## 📁 数据文件要求

Dashboard需要以下数据文件：

### 1. 竞价报告 (必需)
- **位置**: `reports/auction_report_YYYY-MM-DD_HHMMSS.json`
- **生成**: 运行 `python app/auction_monitor_system.py`
- **内容**: 集合竞价监控结果

### 2. RL决策结果 (必需)
- **位置**: `reports/rl_decision_YYYY-MM-DD_HHMMSS.json`
- **生成**: 运行 `python app/daily_workflow.py`
- **内容**: AI决策排序结果和选中股票

### 3. RL权重文件 (必需)
- **位置**: `config/rl_weights.json`
- **生成**: 自动生成或手动创建
- **内容**: 特征权重和Bandit状态

### 4. 回测结果 (可选)
- **位置**: `reports/backtest/metrics_*.json`
- **生成**: 运行 `python app/backtest_engine.py`
- **内容**: 回测性能指标

---

## 🔄 完整工作流

### 步骤1: 运行每日工作流

```bash
python app/daily_workflow.py
```

这会生成：
- `reports/auction_report_YYYY-MM-DD_HHMMSS.json`
- `reports/rl_decision_YYYY-MM-DD_HHMMSS.json`

### 步骤2: (可选) 运行回测

```bash
python app/backtest_engine.py
```

这会生成：
- `reports/backtest/metrics_*.json`
- `reports/backtest/equity_curve_*.csv`
- `reports/backtest/trade_log_*.csv`
```

### 步骤3: 启动Dashboard

```bash
streamlit run web/unified_dashboard.py
```

然后导航到：**Qlib → 数据管理 → 🎯涨停板监控**

---

## 📊 Dashboard功能详解

### Tab 1: 今日信号

**显示内容**:
- 候选股票数量
- 平均竞价强度
- 首板数量统计
- 候选股票详情列表
- 竞价强度分布图

**数据来源**: `reports/auction_report_*.json`

**使用场景**: 
- 查看今日涨停板候选池
- 分析竞价强度分布
- 筛选高质量标的

---

### Tab 2: AI决策过程

**显示内容**:
- 最终选中股票数
- RL得分阈值
- TopK配置
- 选中股票详情
- RL得分分布图
- 特征权重可视化

**数据来源**: 
- `reports/rl_decision_*.json`
- `config/rl_weights.json`

**使用场景**:
- 理解AI决策逻辑
- 查看特征权重
- 分析选股标准

**特征权重说明**:
```
consecutive_days: 连板天数权重
seal_ratio: 封单强度权重
quality_score: 质量分权重
is_leader: 龙头地位权重
auction_change: 竞价涨幅权重
auction_strength: 竞价强度权重
vwap_slope: VWAP斜率权重
max_drawdown: 最大回撤权重
afternoon_strength: 午后强度权重
sector_heat: 板块热度权重
sector_count: 板块涨停数权重
is_first_board: 首板标识权重
```

---

### Tab 3: 涨停原因解释

**显示内容**:
- 涨停原因Top10统计
- 原因频次柱状图
- 个股涨停原因详情

**数据来源**: `reports/rl_decision_*.json`

**涨停原因维度**:
1. **强竞价** - VWAP斜率 >= 0.03
2. **上午抗回撤** - 最大回撤 >= -2%
3. **午后延续性** - 午后强度 >= 1%
4. **题材热度高** - 板块热度 >= 0.7
5. **量能放大** - 买卖比×竞价强度 >= 0.4
6. **封板迅速** - 连板天数 >= 1
7. **封单强度高** - 封单比 >= 8%
8. **龙头地位** - 是否板块龙头

**使用场景**:
- 分析市场热点
- 理解涨停逻辑
- 优化选股策略

---

### Tab 4: RL参数推荐

**显示内容**:
- Thompson Sampling推荐阈值
- 推荐min_score
- 推荐topk
- 累计迭代次数
- Bandit状态(Beta分布)
- 期望成功率曲线

**数据来源**: `config/rl_weights.json`

**使用场景**:
- 查看智能推荐的阈值
- 理解参数优化过程
- 调整选股参数

**参数说明**:
- **min_score**: 最低RL得分阈值 (60/70/80)
- **topk**: 每日选择股票数 (3/5/10)
- **alpha**: Beta分布成功参数
- **beta**: Beta分布失败参数
- **期望成功率**: alpha / (alpha + beta)

---

### Tab 5: 回测结果

**显示内容**:
- 总收益率
- 年化收益率
- Sharpe比率
- 最大回撤
- 胜率
- 总交易次数
- 平均单笔收益
- 净值曲线
- 最近交易记录

**数据来源**: 
- `reports/backtest/metrics_*.json`
- `reports/backtest/equity_curve_*.csv`
- `reports/backtest/trade_log_*.csv`

**使用场景**:
- 评估策略表现
- 分析收益曲线
- 查看交易记录

---

## ⚙️ Dashboard配置

### 侧边栏配置项

1. **Reports目录** - 数据文件存放目录 (默认: `reports`)
2. **Config目录** - 配置文件目录 (默认: `config`)
3. **选择日期** - 查看指定日期的数据

### 自定义配置

如果数据文件在其他位置，可以修改侧边栏的路径配置。

---

## 🔧 故障排查

### 问题1: Dashboard启动失败

**错误**: `ModuleNotFoundError: No module named 'streamlit'`

**解决**: 
```bash
pip install streamlit matplotlib pandas numpy
```

### 问题2: 找不到数据文件

**错误**: Dashboard显示"未找到XXX的竞价报告"

**解决**:
1. 确认已运行 `python app/daily_workflow.py`
2. 检查 `reports/` 目录是否存在JSON文件
3. 确认日期选择正确

### 问题3: 显示空白或异常

**解决**:
1. 刷新浏览器 (Ctrl + R)
2. 清除Streamlit缓存:
   ```bash
   streamlit cache clear
   ```
3. 检查JSON文件格式是否正确

### 问题4: 图表不显示

**错误**: matplotlib相关错误

**解决**:
```bash
pip install matplotlib --upgrade
```

---

## 📈 使用示例

### 示例1: 查看今日选股结果

1. 运行每日工作流:
   ```bash
   python app/daily_workflow.py
   ```

2. 启动Dashboard:
   ```bash
   streamlit run web/limitup_dashboard.py
   ```

3. 在Dashboard中查看:
   - Tab1: 查看所有候选股票
   - Tab2: 查看AI最终选中的股票
   - Tab3: 分析涨停原因

### 示例2: 分析历史表现

1. 运行回测:
   ```bash
   python app/backtest_engine.py
   ```

2. 在Dashboard的Tab5中查看:
   - 净值曲线
   - 性能指标
   - 交易记录

### 示例3: 优化选股参数

1. 在Tab4查看Thompson Sampling推荐
2. 记录推荐的min_score和topk
3. 修改 `config/rl_weights.json` 中的best_action
4. 重新运行工作流测试效果

---

## 🎨 界面特点

- ✅ **响应式布局** - 支持不同屏幕尺寸
- ✅ **实时更新** - 自动加载最新数据
- ✅ **交互式图表** - matplotlib可视化
- ✅ **清晰的指标展示** - Metric卡片布局
- ✅ **详细的表格** - 完整数据展示

---

## 📚 相关文档

- [补丁整合完成报告](PATCH_INTEGRATION_COMPLETE.md)
- [补丁核对报告](PATCH_VERIFICATION_REPORT.md)
- [AKShare数据使用指南](AKSHARE_DATA_USAGE_GUIDE.md)

---

## 🎉 功能亮点

1. **涨停原因可解释** - 8大维度清晰解释
2. **Thompson Sampling优化** - 智能推荐最佳阈值
3. **实时数据可视化** - 直观的图表展示
4. **完整决策流程** - 从竞价到决策全透明
5. **性能指标齐全** - Sharpe/回撤/胜率一应俱全

---

**Web Dashboard已完成！🎉**

启动命令: `streamlit run web/limitup_dashboard.py`
