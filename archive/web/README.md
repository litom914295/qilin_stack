# Web 归档文件说明

## 📦 归档时间

2025-11-07

## 📋 归档内容

### limitup_dashboard.py

**归档原因：** 功能已完全集成到统一控制面板中

**原功能：**
- 独立的涨停板选股系统 Web Dashboard
- 提供5个核心功能标签页：
  1. 📋 今日信号
  2. 🤖 AI决策过程
  3. 🧠 涨停原因解释
  4. ⚙️ RL参数推荐
  5. 📊 回测结果

**替代方案：**

该功能已完全集成到统一控制面板中，请使用以下方式访问：

```bash
# 启动统一控制面板
streamlit run web/unified_dashboard.py
```

然后在界面中导航到：
```
🏠 Qilin监控 → 📦 Qlib → 🗄️ 数据管理 → 🎯 涨停板监控
```

**集成模块位置：**
- `web/tabs/rdagent/limitup_monitor.py`

**验证报告：**
- `docs/LIMITUP_MODULES_COMPARISON_REPORT.md`

---

## 🔧 如需使用归档文件

如果您需要临时使用独立面板，可以：

```bash
# 复制回原位置
cp archive/web/limitup_dashboard.py web/

# 启动独立面板
streamlit run web/limitup_dashboard.py
```

**注意：** 归档文件不再维护，建议使用统一控制面板的集成版本。

---

## 📊 功能对比

根据自动化对比分析（详见 `docs/LIMITUP_MODULES_COMPARISON_REPORT.md`）：

| 项目 | 覆盖率 | 说明 |
|------|--------|------|
| 标签页功能 | 100% | 所有5个标签页功能完全实现 |
| 代码质量 | 更优 | 集成版本采用模块化设计 |
| UI组件 | 更丰富 | 更多指标展示和视觉分隔 |

---

*此归档由系统自动生成 - 2025-11-07*
