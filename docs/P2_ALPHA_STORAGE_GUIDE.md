# P2-1 缠论Alpha因子存储与使用指南

## 📌 概述

本文档描述如何将P2-1的三个缠论Alpha派生因子持久化到Qlib存储，并在IC分析和回测Tab中无缝使用。

### 三个Alpha因子

1. **alpha_zs_movement** - 中枢移动强度
   - 公式: `zs_movement_direction × zs_movement_confidence`
   - 含义: 中枢向上/下移动的方向和置信度乘积
   - 取值范围: [-1, 1]

2. **alpha_zs_upgrade** - 中枢升级强度
   - 公式: `zs_upgrade_flag × zs_upgrade_strength`
   - 含义: 是否发生中枢升级及升级强度
   - 取值范围: [0, 1]

3. **alpha_confluence** - 多周期共振强度
   - 公式: `tanh(confluence_score)`
   - 含义: 日线/周线/月线方向共振程度（tanh归一化）
   - 取值范围: [-1, 1]

---

## 🔧 第一步：生成并写入Alpha因子

### 使用脚本

运行 `scripts/write_chanlun_alphas_to_qlib.py` 脚本将Alpha因子写入本地缓存：

```bash
# 默认参数（csi300, 2020-01-01 ~ 2023-12-31）
python scripts/write_chanlun_alphas_to_qlib.py

# 自定义股票池和时间范围
python scripts/write_chanlun_alphas_to_qlib.py \
    --instruments csi500 \
    --start 2019-01-01 \
    --end 2024-12-31

# 指定Qlib数据路径
python scripts/write_chanlun_alphas_to_qlib.py \
    --provider-uri G:/qlib_data/cn_data

# 同时输出CSV用于验证
python scripts/write_chanlun_alphas_to_qlib.py \
    --output-csv analysis/alpha_debug.csv
```

### 验证写入结果

```bash
# 仅验证模式（不重新生成）
python scripts/write_chanlun_alphas_to_qlib.py --verify
```

验证输出示例：
```
🔍 验证Alpha因子...

📊 验证结果:

   alpha_zs_movement:
      status: ✅ 正常
      shape: (30000,)
      null_ratio: 2.35%
      mean: 0.0124
      std: 0.3421
      min: -0.9821
      max: 0.9654

   alpha_zs_upgrade:
      status: ✅ 正常
      shape: (30000,)
      null_ratio: 1.87%
      mean: 0.0532
      std: 0.1234
      min: 0.0000
      max: 0.9876

   alpha_confluence:
      status: ✅ 正常
      shape: (30000,)
      null_ratio: 0.00%
      mean: 0.0021
      std: 0.4523
      min: -0.9912
      max: 0.9901
```

### 存储位置

Alpha因子缓存存储在：
```
G:/test/qilin_stack/data/qlib_alpha_cache/
├── alpha_zs_movement_csi300_2020-01-01_2023-12-31.pkl
├── alpha_zs_upgrade_csi300_2020-01-01_2023-12-31.pkl
├── alpha_confluence_csi300_2020-01-01_2023-12-31.pkl
└── _meta_csi300_2020-01-01_2023-12-31.json
```

---

## 📊 第二步：在IC分析Tab中使用

### 快速分析

1. 打开Web界面 → "📊 IC分析" → "🔬 快速分析"
2. 在"预设因子"下拉框中选择：
   - `$alpha_confluence`
   - `$alpha_zs_movement`
   - `$alpha_zs_upgrade`
3. 点击"填充预设"按钮
4. 点击"🚀 开始分析"

系统会自动：
- 检测到这是缓存Alpha因子
- 从本地pickle缓存加载（无需重新计算）
- 显示"ℹ️ 从Alpha缓存加载: alpha_confluence"提示
- 进行IC分析并可视化

### 深度分析

在"📈 深度分析"标签中：
1. 同样可以在"预设因子"中选择三个Alpha之一
2. 点击"填充预设"
3. 调整更多参数（滚动窗口、分位数等）
4. 点击"🚀 开始深度分析"

### 自定义因子表达式

如果需要组合Alpha：
```
# 线性组合
$alpha_confluence * 0.5 + $alpha_zs_movement * 0.3 + $alpha_zs_upgrade * 0.2

# 逻辑组合
If($alpha_confluence > 0.3, $alpha_zs_movement, 0)
```

**注意**：组合表达式将通过Qlib表达式引擎计算（需Alpha已在Qlib存储中）

---

## 🔄 第三步：在回测Tab中使用Alpha加权

### 启用Alpha融合

1. 打开Web界面 → "📦 Qlib → 🧪 回测" 
2. 展开"🎯 Alpha融合(可选)"面板
3. 启用"启用Alpha加权"开关
4. 设置各Alpha权重：
   - w_confluence: 多周期共振权重（默认0.3）
   - w_zs_movement: 中枢移动权重（默认0.2）
   - w_zs_upgrade: 中枢升级权重（默认0.1）

### Alpha加权机制

回测系统会：
1. 从Alpha缓存加载三个因子
2. 对每个股票计算调整后的预测分数：
   ```python
   score_adj = score * (1 + w_confluence * alpha_confluence 
                            + w_movement * alpha_zs_movement 
                            + w_upgrade * alpha_zs_upgrade)
   ```
3. 使用调整后的分数进行TopK/Dropout选股
4. 在回测结果中标注"✅ 已使用 Alpha 加权"

### 回测结果展示

回测完成后，结果页面会显示：

```
✅ 已使用 Alpha 加权
   参数:
   - w_confluence: 0.30
   - w_zs_movement: 0.20
   - w_zs_upgrade: 0.10
   - instruments_alpha: csi300

📊 回测指标:
   年化收益率: 18.5%
   夏普比率: 1.92
   最大回撤: -12.3%
   ...
```

---

## 🛠️ 高级用法

### Python API使用

在自定义脚本中加载Alpha：

```python
from scripts.write_chanlun_alphas_to_qlib import load_factor_from_qlib_cache

# 加载单个Alpha
df = load_factor_from_qlib_cache(
    alpha_name='alpha_confluence',
    instruments='csi300',
    start='2020-01-01',
    end='2023-12-31'
)

# df结构: MultiIndex[instrument, datetime] + columns['factor', 'label']
print(df.head())
```

### 重新生成Alpha（增量更新）

如果需要更新更长时间范围的Alpha：

```bash
# 延长到2024年
python scripts/write_chanlun_alphas_to_qlib.py \
    --instruments csi300 \
    --start 2020-01-01 \
    --end 2024-12-31
```

系统会覆盖旧缓存文件，保存新的Alpha数据。

### 多股票池并存

可以为不同股票池生成独立的Alpha缓存：

```bash
# csi300
python scripts/write_chanlun_alphas_to_qlib.py --instruments csi300

# csi500
python scripts/write_chanlun_alphas_to_qlib.py --instruments csi500

# 存储在不同文件:
# alpha_confluence_csi300_2020-01-01_2023-12-31.pkl
# alpha_confluence_csi500_2020-01-01_2023-12-31.pkl
```

---

## ⚠️ 注意事项

### 1. 缓存一致性

- Alpha缓存与原始OHLCV数据关联，如果Qlib数据更新，需要重新生成Alpha
- 验证命令可用于检查缓存是否有效

### 2. 缺失值处理

- 生成Alpha时，某些股票可能因数据不足无法计算中枢移动/共振
- 系统会自动填充0或NaN，IC分析会自动过滤
- 验证报告中会显示`null_ratio`指标

### 3. 计算性能

- 首次生成Alpha（300只股票，3年数据）约需5-10分钟
- 后续加载Alpha缓存仅需数秒
- 建议在夜间定时任务中更新Alpha缓存

### 4. 存储空间

- 每个Alpha因子（300股×3年）约占用10-20 MB
- 三个Alpha总计约30-60 MB
- 多股票池会倍增存储需求

---

## 🔍 故障排查

### 问题1：Alpha缓存未找到

**现象**：IC分析显示"⚠️ Alpha缓存未找到，尝试从Qlib加载..."

**原因**：
- 未运行 `write_chanlun_alphas_to_qlib.py`
- 股票池/时间范围参数不匹配

**解决**：
```bash
# 先验证缓存状态
python scripts/write_chanlun_alphas_to_qlib.py --verify

# 重新生成对应参数的Alpha
python scripts/write_chanlun_alphas_to_qlib.py \
    --instruments csi300 \
    --start 2020-01-01 \
    --end 2023-12-31
```

### 问题2：Alpha全为0或NaN

**现象**：验证报告显示mean=0或null_ratio=100%

**原因**：
- 基础缠论特征生成失败（缺少Chan.py/ZSAnalyzer依赖）
- OHLCV数据不足（股票数据缺失）

**解决**：
1. 检查日志文件 `logs/chanlun_alpha_write.log`
2. 验证Chan.py和ZSAnalyzer正常工作：
   ```python
   from features.chanlun.chanpy_features import ChanPyFeatureGenerator
   gen = ChanPyFeatureGenerator()
   # 测试能否正常导入
   ```
3. 检查Qlib数据完整性：
   ```python
   from qlib.data import D
   df = D.features(['SH600000'], ['$close'], '2020-01-01', '2023-12-31')
   print(df)
   ```

### 问题3：回测中Alpha加权无效

**现象**：启用Alpha加权后回测结果无变化

**原因**：
- Alpha权重设置为0
- instruments_alpha与回测instruments不匹配
- Alpha缓存时间范围与回测时间范围不重叠

**解决**：
1. 检查Alpha权重是否 > 0
2. 确保Alpha缓存涵盖回测时间范围
3. 查看回测结果页"Alpha加权参数"确认实际使用值

---

## 📚 相关文档

- [P2-1 实施计划](P2-1_Implementation_Plan.md)
- [缠论Alpha因子定义](../qlib_enhanced/chanlun/chanlun_alpha.py)
- [中枢分析器文档](../chanpy/ZS/ZSAnalyzer.py)
- [多周期共振引擎](../qlib_enhanced/chanlun/multi_timeframe_confluence.py)

---

## ✅ 总结

通过 `write_chanlun_alphas_to_qlib.py` 脚本，您可以：

1. ✅ 一次性计算并持久化三个P2-1 Alpha因子
2. ✅ 在IC分析Tab中一键加载和评估
3. ✅ 在回测Tab中灵活应用Alpha加权策略
4. ✅ 避免重复计算，提高分析和回测效率

**下一步建议**：
- 设置定时任务每日更新Alpha缓存
- 在多个股票池上验证Alpha有效性
- 结合IC分析结果调整Alpha权重配置

**作者**: Warp AI Assistant  
**日期**: 2025-01  
**版本**: 1.0
