# 📊 Dashboard整合说明

## ✅ 整合完成

涨停板选股系统的Web Dashboard已成功整合到原有的统一Dashboard (`unified_dashboard.py`) 中，**原有功能完整保留**。

---

## 🔧 整合方式

### 新增文件

1. **`web/tabs/rdagent/limitup_monitor.py`** - 涨停板监控模块
   - 独立的功能模块
   - 可在统一Dashboard中调用
   - 支持5大核心功能

2. **`web/limitup_dashboard.py`** - 独立Dashboard (保留)
   - 可单独启动
   - 提供完整的涨停板监控功能
   - 适合只需要涨停板监控的场景

### 修改文件

1. **`web/unified_dashboard.py`**
   - 在 `render_qlib_data_management_tab()` 方法中新增了第4个tab
   - 原有的3个tab (多数据源、涨停板分析、因子/特征) **完整保留**
   - 新增的tab调用 `limitup_monitor.render()`

---

## 🎯 访问路径

### 方式1: 统一Dashboard (推荐)

启动统一Dashboard:
```bash
streamlit run web/unified_dashboard.py
```

然后在Web界面中访问:
```
🏠 Qilin监控 → 📦 Qlib → 🗄️ 数据管理 → 🎯 涨停板监控
```

**优点**:
- 所有功能集成在一个界面
- 可以方便地切换不同功能
- 与原有系统无缝整合

### 方式2: 独立Dashboard

启动独立Dashboard:
```bash
streamlit run web/limitup_dashboard.py
```

**优点**:
- 启动更快
- 界面更简洁
- 专注于涨停板监控

---

## 📂 文件结构

```
qilin_stack/
├── web/
│   ├── unified_dashboard.py           # 统一Dashboard (已修改✓)
│   ├── limitup_dashboard.py           # 独立Dashboard (新增✓)
│   └── tabs/
│       └── rdagent/
│           ├── limitup_monitor.py     # 涨停板监控模块 (新增✓)
│           ├── factor_mining.py       # 因子挖掘 (原有✓)
│           ├── model_optimization.py  # 模型优化 (原有✓)
│           └── ...其他模块...
├── start_dashboard.bat                # Windows启动脚本 (已修改✓)
├── start_dashboard.sh                 # Linux/Mac启动脚本 (已修改✓)
├── WEB_DASHBOARD_GUIDE.md             # 使用指南 (已更新✓)
├── README_DASHBOARD.md                # 快速入门 (已更新✓)
└── DASHBOARD_INTEGRATION_NOTES.md    # 本文档 (新增✓)
```

---

## 🚀 启动脚本

### Windows

双击运行:
```
start_dashboard.bat
```

会启动统一Dashboard，涨停板监控位于:
```
Qlib → 数据管理 → 🎯涨停板监控
```

### Linux/Mac

```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

---

## 🔍 功能对比

| 功能 | 统一Dashboard | 独立Dashboard |
|------|--------------|--------------|
| 涨停板监控 (5大Tab) | ✅ | ✅ |
| Qilin实时监控 | ✅ | ❌ |
| RD-Agent研发智能体 | ✅ | ❌ |
| TradingAgents多智能体 | ✅ | ❌ |
| Qlib模型训练 | ✅ | ❌ |
| 多数据源管理 | ✅ | ❌ |
| 投资组合优化 | ✅ | ❌ |
| 风控中心 | ✅ | ❌ |
| 写实回测 | ✅ | ❌ |

---

## 📋 涨停板监控功能

无论使用哪种方式，涨停板监控都提供以下5大功能:

### Tab1: 📋 今日信号
- 候选股票数量统计
- 平均竞价强度
- 首板数量
- 候选股票详情列表
- 竞价强度分布图

### Tab2: 🤖 AI决策过程
- 最终选中股票数
- RL得分阈值
- TopK配置
- 选中股票详情
- RL得分分布图
- 特征权重可视化

### Tab3: 🧠 涨停原因解释
- 涨停原因Top10统计
- 原因频次柱状图
- 个股涨停原因详情
- 8大维度分析

### Tab4: ⚙️ RL参数推荐
- Thompson Sampling推荐阈值
- 推荐min_score和topk
- 累计迭代次数
- Bandit状态 (Beta分布)
- 期望成功率曲线

### Tab5: 📊 回测结果
- 关键性能指标
- 净值曲线
- 最近交易记录
- Sharpe/回撤/胜率

---

## 🔐 原有功能保护

**重要**: 所有原有的Dashboard功能都**完整保留**，包括:

### Qilin监控 (原有✓)
- 📊 实时监控
- 🤖 智能体状态
- 📈 交易执行
- 📉 风险管理
- 📋 历史记录

### Qlib (原有✓)
- 📈 模型训练 (在线学习、强化学习、一进二策略)
- 🗄️ 数据管理 (多数据源、涨停板分析、**🎯涨停板监控【新增】**、因子/特征)
- 💼 投资组合 (回测、优化、归因分析)
- ⚠️ 风险控制 (流动性监控、极端行情保护、头寸管理)
- 🔄 在线服务 (模型部署)
- 📊 实验管理 (MLflow)

### RD-Agent研发智能体 (原有✓)
- 所有原有功能完整保留

### TradingAgents多智能体 (原有✓)
- 所有原有功能完整保留

---

## 💡 使用建议

1. **日常使用**: 使用统一Dashboard (`unified_dashboard.py`)
   - 功能最全面
   - 可以方便切换不同模块

2. **专注监控**: 使用独立Dashboard (`limitup_dashboard.py`)
   - 启动快速
   - 界面简洁
   - 专注涨停板监控

3. **开发调试**: 使用独立Dashboard
   - 不影响原有系统
   - 方便测试新功能

---

## 📝 数据文件要求

两种Dashboard都需要以下数据文件:

```
reports/
├── auction_report_YYYY-MM-DD_HHMMSS.json    # 竞价报告
├── rl_decision_YYYY-MM-DD_HHMMSS.json       # RL决策结果
└── backtest/
    ├── metrics_*.json                        # 回测指标
    ├── equity_curve_*.csv                    # 净值曲线
    └── trade_log_*.csv                       # 交易记录

config/
└── rl_weights.json                           # RL权重配置
```

生成数据:
```bash
python app/daily_workflow.py      # 生成竞价报告和RL决策
python app/backtest_engine.py     # 生成回测结果
```

---

## 🎯 总结

✅ **整合完成，原有功能完整保留**  
✅ **新增功能作为独立模块整合到统一Dashboard**  
✅ **保留独立Dashboard，提供灵活选择**  
✅ **启动脚本已更新，指向统一Dashboard**  
✅ **文档已更新，说明两种使用方式**

**推荐使用统一Dashboard以获得最佳体验！**

启动命令: `start_dashboard.bat` (Windows) 或 `./start_dashboard.sh` (Linux/Mac)
