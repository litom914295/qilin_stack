# 任务 7 完成报告: 模型 Zoo 降级状态 UI 提示

**完成日期**: 2025-01-XX  
**优先级**: P0 (用户透明度问题)  
**关联任务**: 任务 3 (模型 Zoo 与降级策略)  

---

## 1. 实现内容

### 1.1 核心功能

#### ✅ 模型依赖检测模块 (`model_dependency_checker.py`)

**文件路径**: `qlib_enhanced/model_zoo/model_dependency_checker.py` (238 行新文件)

**核心功能**:
1. **依赖检测**: 检查模型所需的 Python 包 (torch, xgboost, catboost 等)
2. **导入测试**: 验证模型类是否可成功导入
3. **降级策略**: 为不可用模型提供fallback建议
4. **状态分类**: 'ok', 'missing_deps', 'fallback', 'error' 四种状态

**支持的模型** (14 个):
- GBDT 家族: LightGBM, XGBoost, CatBoost
- 神经网络: MLP, LSTM, GRU, ALSTM
- 高级模型: Transformer, TRA, TCN, HIST
- 集成模型: DoubleEnsemble

**降级策略映射**:
```python
TRA → Transformer  # TRA 需要额外图结构依赖
HIST → Transformer  # HIST 需要额外依赖
DoubleEnsemble → LightGBM
XGBoost/CatBoost → LightGBM
神经网络模型 (缺torch) → LightGBM
```

#### ✅ UI 集成 (`qlib_model_zoo_tab.py`)

**修改位置**:
1. **导入依赖检测模块** (line 24-34)
2. **全局统计展示** (line 205-220): 展示可用/缺失依赖/降级运行模型数量
3. **单模型状态提示** (line 290-316): 针对每个模型显示详细状态

**UI 展示效果**:

```
✅ 可用状态 (status='ok'):
  ✅ 可用

⚠️ 降级状态 (status='fallback'):
  ⚠️ 模型不可用 (可能降级为 Transformer)
  💡 建议使用: Transformer (更稳定)
  [🔄 切换到 Transformer] 按钮

❌ 缺失依赖 (status='missing_deps'):
  ❌ 缺少依赖: torch
  `pip install torch`
  🔄 可以使用降级模型: LightGBM
  [🔄 切换到 LightGBM] 按钮
```

---

## 2. 关键设计决策

### 2.1 依赖检测策略

**两级检测**:
1. **包级检测** (`importlib.import_module`): 检查依赖包是否安装
2. **类级检测** (`exec(import_statement)`): 检查模型类是否可导入

**为什么需要两级**:
- torch 已安装,但 pytorch_tra.TRA 仍可能因特殊依赖(如 torch_geometric)导入失败
- 避免误报"可用"但实际训练时崩溃

### 2.2 降级策略设计

**分层降级**:
```
高级模型 → 基础深度学习模型 → 传统机器学习
TRA/HIST → Transformer → LSTM → LightGBM
```

**降级原则**:
1. **性能优先**: Transformer > LSTM > LightGBM
2. **稳定性优先**: LightGBM 无额外依赖,最稳定
3. **任务相关**: 时序模型优先降级到时序模型

### 2.3 UI 交互设计

**用户友好性**:
- ✅ 绿色表示完全可用
- ⚠️ 黄色警告表示降级运行
- ❌ 红色错误表示缺失依赖
- 🔄 一键切换按钮,无需手动查找fallback模型

**信息完整性**:
- 显示具体缺失的包名
- 提供 pip install 命令
- 说明降级原因 (如"需要额外图结构依赖")

---

## 3. 代码示例

### 3.1 依赖检测 API

```python
from qlib_enhanced.model_zoo.model_dependency_checker import check_model_availability

# 检测单个模型
result = check_model_availability("TRA")

if result.available:
    print(f"✅ {result.message}")
else:
    print(f"❌ {result.message}")
    if result.status == 'missing_deps':
        print(f"安装命令: {result.install_command}")
    if result.fallback_model:
        print(f"降级方案: {result.fallback_model}")

# 输出示例:
# ⚠️ 模型不可用 (可能降级为 Transformer)
# 降级方案: Transformer
```

### 3.2 批量检测

```python
from qlib_enhanced.model_zoo.model_dependency_checker import (
    check_all_models,
    get_model_status_summary
)

# 检测所有模型
results = check_all_models()
for model_name, result in results.items():
    print(f"{model_name}: {result.status}")

# 统计摘要
summary = get_model_status_summary()
print(f"总计: {summary['total']}")
print(f"可用: {summary['available']}")
print(f"降级: {summary['fallback']}")
print(f"缺失依赖: {summary['missing_deps']}")
```

---

## 4. 测试验证

### 4.1 本地测试

```powershell
# 1. 运行依赖检测脚本
cd qlib_enhanced/model_zoo
python model_dependency_checker.py

# 预期输出:
# === Qlib 模型依赖检测 ===
# LightGBM       | ✅ 可用
# XGBoost        | ✅ 可用 (或 ❌ 缺少依赖: xgboost)
# LSTM           | ✅ 可用 (或 ❌ 缺少依赖: torch)
# TRA            | ⚠️ 模型不可用 (可能降级为 Transformer)
# ...
```

### 4.2 UI 测试

**测试场景 1**: torch 未安装
```
预期: LSTM/GRU/Transformer 等显示 ❌ 缺少依赖: torch
      提供 pip install torch 命令
      显示降级选项: LightGBM
```

**测试场景 2**: torch 已安装,但 TRA 不可用
```
预期: TRA 显示 ⚠️ 模型不可用 (可能降级为 Transformer)
      提供切换按钮
```

**测试场景 3**: 全部依赖已安装
```
预期: 所有模型显示 ✅ 可用
      依赖检测统计: 可用=14, 降级=0, 缺失依赖=0
```

---

## 5. 影响分析

### 5.1 解决的问题 (P0)

| 问题 | 修复前 | 修复后 |
|------|-------|-------|
| **用户不知道模型降级** | TRA 静默使用 Transformer,性能降低 | ⚠️ 明确提示"可能降级为 Transformer" |
| **依赖缺失无提示** | 训练时崩溃,错误信息难懂 | ❌ 提前检测,显示 pip install 命令 |
| **手动查找 fallback** | 用户需要自己试错哪个模型可用 | 🔄 一键切换到推荐模型 |

### 5.2 用户体验提升

**修复前**:
```
用户: 我要训练 TRA 模型
系统: [开始训练...] ImportError: No module named 'torch_geometric'
用户: 😕 什么情况? 要装什么?
```

**修复后**:
```
用户: 我要训练 TRA 模型
系统: ⚠️ 模型不可用 (可能降级为 Transformer)
      💡 建议使用: Transformer (更稳定)
      [🔄 切换到 Transformer]
用户: 👍 好的,那就用 Transformer
```

---

## 6. 局限性与未来优化

### 6.1 当前局限

1. **GPU 检测不完整**
   - 仅检测 torch 包,未检测 CUDA 可用性
   - 未来: 添加 `torch.cuda.is_available()` 检测

2. **版本兼容性未检查**
   - 未验证 torch 版本是否满足 Qlib 要求 (如 torch>=1.8)
   - 未来: 添加版本号解析 (`importlib.metadata.version`)

3. **特殊依赖未覆盖**
   - TRA/HIST 的额外依赖 (torch_geometric, torch_scatter) 未精确检测
   - 未来: 扩展 MODEL_DEPENDENCIES 配置

### 6.2 性能优化

**当前实现**: 每次渲染 UI 都执行依赖检测 (约 14 次 import 测试)

**优化方案**:
```python
# 缓存检测结果 (5 分钟有效)
@st.cache_data(ttl=300)
def check_all_models_cached():
    return check_all_models()
```

---

## 7. 相关文档更新

### 7.1 需要更新的文档

- [ ] **用户指南**: 添加"模型依赖检测"章节
- [ ] **开发者指南**: 说明如何添加新模型的依赖配置
- [ ] **FAQ**: 常见依赖安装问题 (torch, xgboost, catboost)

### 7.2 示例文档片段

````markdown
## 模型依赖检测

麒麟项目会自动检测模型所需依赖是否满足:

### 依赖状态说明

- ✅ **可用**: 所有依赖已安装,模型可直接训练
- ⚠️ **降级运行**: 模型不可用,但提供了替代方案
- ❌ **缺失依赖**: 需要安装额外的 Python 包

### 常见依赖安装

```bash
# 安装深度学习模型依赖 (LSTM/GRU/Transformer)
pip install torch

# 安装 XGBoost
pip install xgboost

# 安装 CatBoost
pip install catboost
```
````

---

## 8. 与其他任务的关联

| 任务 | 关系 | 说明 |
|------|------|------|
| **任务 14** (适配层稳健性) | 依赖 | model_trainer.py 应使用依赖检测结果过滤可训练模型 |
| **任务 15** (自动化测试) | 依赖 | 需添加依赖检测模块的单元测试 |
| **任务 16** (文档) | 依赖 | 用户指南需说明依赖检测功能 |

---

## 9. 验收标准

- [x] 创建 `model_dependency_checker.py` 模块
- [x] 实现 `check_model_availability()` 函数
- [x] UI 集成依赖检测统计面板
- [x] UI 显示单模型状态 (ok/fallback/missing_deps)
- [x] 降级模型一键切换按钮
- [ ] 添加单元测试 (待任务 15)
- [ ] 更新用户文档 (待任务 16)

---

## 10. 后续行动

### 10.1 立即行动 (本次 PR)

- [x] 创建依赖检测模块
- [x] UI 集成基础功能
- [ ] 本地测试验证

### 10.2 短期改进 (后续 PR)

- [ ] 添加 GPU 可用性检测
- [ ] 添加版本兼容性检查
- [ ] 缓存检测结果 (性能优化)
- [ ] 扩展 TRA/HIST 特殊依赖检测

### 10.3 长期优化

- [ ] 自动依赖安装引导 (一键安装按钮)
- [ ] 依赖冲突检测 (torch 与其他包版本冲突)
- [ ] 云端依赖数据库 (定期更新推荐版本)

---

**完成时间**: 2025-01-XX  
**验证人**: AI Agent  
**审核状态**: ✅ 代码完成,待功能测试
