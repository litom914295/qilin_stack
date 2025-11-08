# 阶段一第一周快速启动指南

> **基于**: `docs/IMPROVEMENT_ROADMAP.md` 阶段一任务  
> **目标**: 挤压水分，建立可靠基线  
> **预计时间**: 7天

---

## 📋 本周任务概览

| 任务 | 天数 | 状态 | 脚本 |
|------|------|------|------|
| 1.1 数据质量审计 | Day 1-2 | ✅ 已完成 | `scripts/audit_data_quality.py` |
| 1.2 高频特征可靠性测试 | Day 3-4 | ✅ 已完成 | `scripts/test_high_freq_features.py` |
| 1.3 特征降维 | Day 5 | ⏳ 待执行 | `scripts/generate_core_features.py` |
| 1.4 基准模型 | Day 6-7 | ⏳ 待执行 | `scripts/train_baseline_model.py` |

---

## 🚀 Day 1-2: 数据质量审计

### 目标
识别数据源质量问题，为后续特征清理提供依据

### 执行命令

```powershell
# 基础审计（使用默认参数）
python scripts/audit_data_quality.py

# 指定审计区间
python scripts/audit_data_quality.py --start-date 2023-01-01 --end-date 2024-12-31

# 自定义输出路径
python scripts/audit_data_quality.py --output reports/my_audit_report.md
```

### 输出文件
- `reports/data_quality_audit_report.md` - 审计报告（Markdown格式）
- 控制台实时输出审计进度

### 预期结果
1. ✅ 数据源覆盖率统计（Qlib/AKShare/Tushare）
2. ✅ 缺失值与异常值分析
3. ✅ 高频特征数据粒度评估
4. ⚠️ **关键发现**: 如果高频特征可靠性<50%，将建议禁用

### 验收标准
- [ ] 审计报告生成成功
- [ ] 识别出所有数据源的可用状态
- [ ] 明确高频特征的真实数据粒度

---

## 🧪 Day 3-4: 高频特征可靠性测试

### 目标
测试每个高频特征的计算逻辑和数据质量，标记不可靠特征

### 执行命令

```powershell
# 完整测试（默认50个样本）
python scripts/test_high_freq_features.py

# 指定测试日期和样本数量
python scripts/test_high_freq_features.py --test-date 2024-12-01 --sample-size 100

# 只测试特定特征
python scripts/test_high_freq_features.py --features 封单稳定性,大单流入节奏,成交萎缩度

# 自定义输出
python scripts/test_high_freq_features.py --output analysis/my_feature_test.csv
```

### 输出文件
- `analysis/high_freq_feature_reliability.csv` - 特征测试结果（CSV格式）
- `reports/high_freq_feature_test_report.md` - 详细测试报告（Markdown格式）

### 测试维度
1. **数据粒度**: L2逐笔 vs 分钟线 vs 日线
2. **计算逻辑**: 是否使用正确的数据源
3. **数值稳定性**: NaN/Inf/极端值检测
4. **时序一致性**: 未来信息泄露检测

### 评分规则
| 综合得分 | 可靠性等级 | 建议 |
|----------|-----------|------|
| 80-100 | ✅ 可靠 | 可以使用 |
| 60-79 | ⚠️ 中等 | 谨慎使用，需监控 |
| 40-59 | ⚠️ 较差 | 建议暂时禁用 |
| 0-39 | ❌ 不可靠 | 强烈建议禁用 |

### 验收标准
- [ ] 生成CSV和Markdown两份报告
- [ ] 每个高频特征都有综合评分
- [ ] 明确列出不可靠特征清单（得分<60）

---

## 📉 Day 5: 特征降维（待实施）

### 目标
禁用不可靠特征，生成精简版50核心特征集

### 执行命令（预计）

```powershell
# 基于前两天的测试结果，自动生成核心特征集
python scripts/generate_core_features.py \
  --audit-report reports/data_quality_audit_report.md \
  --test-report analysis/high_freq_feature_reliability.csv \
  --max-features 50 \
  --output features/core_features_v1.py
```

### 输出文件
- `features/core_features_v1.py` - 精简版核心特征集（50个特征）
- `reports/feature_reduction_report.md` - 降维报告

### 降维策略
1. **强制禁用**: 可靠性得分<40的特征
2. **条件禁用**: 可靠性得分40-60，且数据粒度<分钟级
3. **保留**: 日线可靠特征（价量、技术指标）
4. **保留**: 封板基础特征（封单强度、涨停时间、开板次数）
5. **保留**: 历史统计特征（历史竞价表现）

### 验收标准
- [ ] 生成的特征集≤50个
- [ ] 所有保留特征的可靠性得分≥60
- [ ] 提供每个被禁用特征的原因说明

---

## 🎯 Day 6-7: 建立简单基准模型（待实施）

### 目标
使用单一LightGBM和50核心特征训练基准模型

### 执行命令（预计）

```powershell
# 训练基准模型
python scripts/train_baseline_model.py \
  --features features/core_features_v1.py \
  --model lgbm \
  --params conservative \
  --output models/baseline_lgbm_v1.pkl

# 评估基准性能
python scripts/evaluate_baseline.py \
  --model models/baseline_lgbm_v1.pkl \
  --output reports/baseline_performance.md
```

### 模型配置
- **算法**: LightGBM（单一模型，无集成）
- **超参数**: 保守设置
  - `max_depth`: 5
  - `num_leaves`: 31
  - `learning_rate`: 0.05
  - `n_estimators`: 100
- **数据划分**: 60%训练 / 20%验证 / 20%测试（严格时间切分）

### 输出文件
- `models/baseline_lgbm_v1.pkl` - 基准模型文件
- `reports/baseline_performance.md` - 性能评估报告
- `analysis/baseline_feature_importance.csv` - 特征重要性

### 验收标准
- [ ] 样本外AUC > 0.68（即使下降也要确保稳定）
- [ ] AUC标准差 < 0.05（稳定性检查）
- [ ] 生成SHAP特征解释
- [ ] 记录训练时间和推理时间

---

## 📊 本周验收汇总

### 必达指标
1. ✅ **数据质量**: 完成数据源审计，明确可用数据源
2. ✅ **特征清理**: 识别并标记不可靠特征
3. ⏳ **特征集**: 生成≤50个核心特征的精简集
4. ⏳ **基准模型**: 样本外AUC > 0.68

### 关键交付物
- [ ] `reports/data_quality_audit_report.md`
- [ ] `analysis/high_freq_feature_reliability.csv`
- [ ] `features/core_features_v1.py`
- [ ] `models/baseline_lgbm_v1.pkl`
- [ ] `reports/baseline_performance.md`

---

## 💡 使用建议

### 1. 按顺序执行
**严格遵循 Day 1 → Day 7 的顺序**，因为：
- Day 3-4的测试依赖Day 1-2的审计结果
- Day 5的降维依赖Day 3-4的测试结果
- Day 6-7的模型训练依赖Day 5的特征集

### 2. 审查每步输出
每完成一个任务，务必：
1. 打开生成的报告，仔细审查
2. 检查是否有意外发现或异常
3. 根据报告的建议调整下一步计划

### 3. 记录实验日志
建议创建 `logs/week1_experiment_log.md`，记录：
- 每天的执行时间
- 遇到的问题和解决方案
- 关键发现和洞察
- 对改进计划的调整建议

### 4. 数据备份
在开始前，备份以下数据：
```powershell
# 备份现有特征定义
cp features/one_into_two_feature_builder.py features/one_into_two_feature_builder_backup.py

# 备份现有模型配置
cp config/default_config.yaml config/default_config_backup.yaml
```

---

## 🚨 常见问题

### Q1: 数据质量审计报告显示"无法获取样本数据"？
**A**: 可能是网络问题或API限制。尝试：
1. 检查网络连接
2. 更换测试日期（避开周末和节假日）
3. 减小sample_size参数

### Q2: 高频特征测试结果显示全部不可靠？
**A**: 这是**预期结果**！如果你没有L2数据或分钟数据，高频特征确实不可靠。这正是本周任务的目的——**挤压水分，识别真相**。

### Q3: 如果没有Qlib数据怎么办？
**A**: 没关系！系统会自动降级使用AKShare。数据质量审计会明确告诉你当前可用的数据源。

### Q4: 基准模型AUC低于0.68怎么办？
**A**: 
1. 检查是否真的使用了精简后的可靠特征
2. 检查数据质量（缺失值、异常值）
3. **不要急于调参**！如果使用可靠特征后AUC仍低，说明当前数据/特征对一进二预测能力有限，这是真实反馈
4. 记录下来，在阶段二改进竞价策略时重点突破

---

## 📚 相关文档

- **总体路线图**: `docs/IMPROVEMENT_ROADMAP.md`
- **工作流框架**: `docs/AUCTION_WORKFLOW_FRAMEWORK.md`
- **系统架构**: `docs/Technical_Architecture_v2.1_Final.md`

---

## 🎯 下周预告

完成本周任务后，**阶段一第二周**将聚焦：
1. 因子衰减监控系统（滚动IC/IR计算）
2. Walk-Forward严格验证框架
3. 多分类标签优化（5档位）
4. 宏观情绪因子补充

---

**祝你本周改进顺利！记住：先简后繁，先验证后复杂化，先挤水分后加功能。** 🚀

*最后更新：2025-10-30*
