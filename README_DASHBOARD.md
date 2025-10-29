# 🎯 麒麟涨停板选股系统 - Web Dashboard

## 📋 简介

专为**涨停板选股系统**设计的可视化Web Dashboard，提供实时监控、AI决策可视化、涨停原因解释、强化学习参数推荐和回测结果展示。

---

## 🚀 快速启动

### 方式1: 统一Dashboard (推荐)

**Windows:**
```
start_dashboard.bat
```

**Linux/Mac:**
```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

**手动启动:**
```bash
streamlit run web/unified_dashboard.py
```

然后在Web界面中访问: **Qlib → 数据管理 → 🎯涨停板监控**

### 方式2: 独立Dashboard

如果只想使用涨停板监控功能:
```bash
streamlit run web/limitup_dashboard.py
```

启动后浏览器自动打开 `http://localhost:8501`

---

## 📊 Dashboard功能概览

| Tab | 功能 | 说明 |
|-----|------|------|
| 📋 今日信号 | 竞价监控状态 | 实时候选股票池、竞价强度分布 |
| 🤖 AI决策过程 | 决策可视化 | RL得分、特征权重、最终选股 |
| 🧠 涨停原因解释 | 原因分析 | 8大维度涨停逻辑解释 |
| ⚙️ RL参数推荐 | Thompson Sampling | 智能阈值优化推荐 |
| 📊 回测结果 | 历史回测 | 净值曲线、交易记录、性能指标 |

---

## 📁 数据要求

Dashboard需要以下数据文件 (由系统自动生成):

```
qilin_stack/
├── reports/
│   ├── auction_report_YYYY-MM-DD_HHMMSS.json    # 竞价报告
│   ├── rl_decision_YYYY-MM-DD_HHMMSS.json       # RL决策结果
│   └── backtest/
│       ├── metrics_*.json                        # 回测指标
│       ├── equity_curve_*.csv                    # 净值曲线
│       └── trade_log_*.csv                       # 交易记录
└── config/
    └── rl_weights.json                           # RL权重配置
```

---

## 🔄 完整使用流程

### 1️⃣ 生成数据

运行每日工作流:
```bash
python app/daily_workflow.py
```

这将生成竞价报告和RL决策结果。

### 2️⃣ (可选) 运行回测

```bash
python app/backtest_engine.py
```

生成回测性能数据。

### 3️⃣ 启动Dashboard

```bash
start_dashboard.bat  # Windows
./start_dashboard.sh # Linux/Mac
```

或手动启动:
```bash
streamlit run web/limitup_dashboard.py
```

### 4️⃣ 查看结果

浏览器访问 http://localhost:8501

---

## 🎨 Dashboard特性

✅ **实时数据** - 自动加载最新报告  
✅ **交互可视化** - matplotlib图表展示  
✅ **可解释AI** - 清晰的决策逻辑  
✅ **Thompson Sampling** - 智能参数优化  
✅ **完整回测** - Sharpe/回撤/胜率  

---

## 🔧 依赖安装

```bash
pip install streamlit matplotlib pandas numpy
```

## ✅ 验证整合

运行验证脚本检查Dashboard是否正确整合:

```bash
python verify_dashboard_integration.py
```

该脚本会检查:
- ✅ 关键文件是否存在
- ✅ 模块是否可导入
- ✅ unified_dashboard是否正确整合
- ✅ 依赖包是否安装

---

## 📚 详细文档

- [完整使用指南](WEB_DASHBOARD_GUIDE.md) - 详细功能说明和故障排查
- [补丁整合报告](PATCH_INTEGRATION_COMPLETE.md) - 系统整合历史
- [AKShare使用指南](AKSHARE_DATA_USAGE_GUIDE.md) - 数据源说明

---

## 🎉 核心亮点

### 1. 涨停原因可解释 (8大维度)

- 强竞价 (VWAP斜率 >= 0.03)
- 上午抗回撤 (最大回撤 >= -2%)
- 午后延续性 (午后强度 >= 1%)
- 题材热度高 (板块热度 >= 0.7)
- 量能放大 (买卖比×竞价强度 >= 0.4)
- 封板迅速 (连板天数 >= 1)
- 封单强度高 (封单比 >= 8%)
- 龙头地位 (是否板块龙头)

### 2. Thompson Sampling强化学习

- 智能推荐最佳min_score阈值 (60/70/80)
- 自适应topk选股数量 (3/5/10)
- Beta分布期望成功率可视化
- 累计迭代优化历史

### 3. 完整决策流程透明

```
竞价监控 → 特征提取 → RL评分 → 权重解释 → 最终选股
```

### 4. 性能指标齐全

- 总收益率、年化收益率
- Sharpe比率、最大回撤
- 胜率、平均单笔收益
- 完整交易记录

---

## 💡 使用建议

1. **每日启动**: 运行工作流后立即启动Dashboard查看结果
2. **参数优化**: 参考Tab4的Thompson Sampling推荐调整阈值
3. **原因分析**: 通过Tab3理解市场热点和涨停逻辑
4. **回测验证**: 定期运行回测评估策略表现

---

## 🛠️ 故障排查

### Dashboard无法启动

```bash
pip install streamlit matplotlib pandas numpy --upgrade
```

### 找不到数据文件

1. 检查是否运行了 `python app/daily_workflow.py`
2. 确认 `reports/` 目录存在JSON文件
3. 在Dashboard侧边栏调整日期选择

### 图表不显示

```bash
pip install matplotlib --upgrade
streamlit cache clear
```

---

## 📞 技术支持

- 详细文档: [WEB_DASHBOARD_GUIDE.md](WEB_DASHBOARD_GUIDE.md)
- 系统架构: [PATCH_INTEGRATION_COMPLETE.md](PATCH_INTEGRATION_COMPLETE.md)

---

**启动命令**: `start_dashboard.bat` (Windows) 或 `./start_dashboard.sh` (Linux/Mac)

**访问地址**: http://localhost:8501

**停止服务**: 按 `Ctrl+C`
