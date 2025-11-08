# 侧边栏自动导航移除说明

## 问题描述

在修改后，侧边栏顶部仍然显示三行英文导航链接：
- unified dashboard
- realistic backtest page  
- system guide

## 根本原因

这是 **Streamlit 多页面应用 (Multi-page Apps)** 的自动功能。

当在 `web/pages/` 目录下存在 Python 文件时，Streamlit 会自动：
1. 识别这些文件为页面
2. 在侧边栏顶部生成导航链接
3. 链接文本为文件名（去掉 .py 后缀，下划线变空格）

```
web/
├── pages/                    ← Streamlit 自动识别此目录
│   ├── realistic_backtest_page.py  → "realistic backtest page"
│   └── system_guide.py              → "system guide"
└── unified_dashboard.py      ← 主页面显示为 "unified dashboard"
```

## 解决方案

### 方案选择

我们采用了 **移动文件** 的方案，将页面文件从 `pages/` 目录移到 `components/` 目录：

```
web/
├── pages/                    ← 空目录（不会触发自动导航）
├── components/               ← 新建的组件目录
│   ├── __init__.py
│   ├── realistic_backtest_page.py
│   └── system_guide.py
└── unified_dashboard.py
```

### 其他可行方案

1. **方案A**: 重命名文件（加下划线前缀）
   - 如：`_realistic_backtest_page.py`
   - 优点：不需要移动文件
   - 缺点：文件名不规范

2. **方案B**: 删除 `pages/` 目录
   - 优点：最彻底
   - 缺点：可能影响其他功能

3. **方案C**: 使用 `.streamlit/config.toml` 配置
   - 缺点：Streamlit 没有提供禁用多页面导航的选项

## 实施步骤

### 1. 创建 components 目录
```powershell
New-Item -ItemType Directory -Force -Path "G:\test\qilin_stack\web\components"
```

### 2. 移动文件
```powershell
Move-Item -Path "web\pages\realistic_backtest_page.py" -Destination "web\components\"
Move-Item -Path "web\pages\system_guide.py" -Destination "web\components\"
```

### 3. 创建 __init__.py
```python
# web/components/__init__.py
"""
Web组件模块
包含系统指南、写实回测等功能组件
"""

__all__ = ['system_guide', 'realistic_backtest_page']
```

### 4. 更新导入路径

**修改前**：
```python
from web.pages.system_guide import show_system_guide
from web.pages.realistic_backtest_page import show_realistic_backtest_page
```

**修改后**：
```python
from web.components.system_guide import show_system_guide
from web.components.realistic_backtest_page import show_realistic_backtest_page
```

## 修改的文件

1. **创建新目录**
   - `web/components/` - 组件目录
   
2. **移动的文件**
   - `web/pages/realistic_backtest_page.py` → `web/components/realistic_backtest_page.py`
   - `web/pages/system_guide.py` → `web/components/system_guide.py`

3. **创建的文件**
   - `web/components/__init__.py` - 包初始化文件

4. **修改的文件**
   - `web/unified_dashboard.py` (2处导入路径更新)
     - Line 738: `from web.components.system_guide import ...`
     - Line 2106: `from web.components.realistic_backtest_page import ...`

## 验证方法

### 启动应用
```bash
streamlit run web/unified_dashboard.py
```

### 检查要点
- ✅ 侧边栏顶部**没有**三行英文链接
- ✅ 侧边栏从"📍 控制面板"开始
- ✅ 🏠 Qilin监控 → 📖 写实回测 正常工作
- ✅ 🏠 Qilin监控 → 📚 系统指南 正常工作

## 技术原理

### Streamlit 多页面应用机制

Streamlit 使用约定优于配置的方式实现多页面应用：

```
your_app/
├── main_app.py           ← 主页面
└── pages/                ← 自动识别的页面目录
    ├── page1.py          ← 自动生成导航
    └── page2.py          ← 自动生成导航
```

运行 `streamlit run main_app.py` 时：
1. Streamlit 扫描 `pages/` 目录
2. 为每个 `.py` 文件生成导航链接
3. 在侧边栏顶部显示这些链接

### 我们的解决方案

通过将文件移出 `pages/` 目录：
1. Streamlit 不再识别这些文件为"页面"
2. 不会自动生成导航链接
3. 我们手动在主应用中通过 tab 调用这些组件

## 目录结构对比

### 修改前
```
web/
├── pages/
│   ├── realistic_backtest_page.py  ❌ 触发自动导航
│   └── system_guide.py              ❌ 触发自动导航
├── tabs/
│   ├── ...
└── unified_dashboard.py             ❌ 显示为 "unified dashboard"
```
**结果**: 侧边栏顶部显示 3 个页面链接

### 修改后
```
web/
├── pages/                           ✅ 空目录
├── components/                      ✅ 不触发自动导航
│   ├── __init__.py
│   ├── realistic_backtest_page.py
│   └── system_guide.py
├── tabs/
│   ├── ...
└── unified_dashboard.py
```
**结果**: 侧边栏顶部干净，无自动导航

## 注意事项

1. **保留 pages 目录**
   - 虽然现在是空的，但保留它作为预留
   - 未来如果需要多页面功能，可以使用

2. **导入路径变化**
   - 所有引用这两个文件的地方都需要更新
   - 当前只有 `unified_dashboard.py` 引用

3. **向后兼容**
   - 不影响现有功能
   - 只是改变了文件组织方式

## 相关文档

- [Streamlit Multi-page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- `docs/SIDEBAR_REORGANIZATION.md` - 侧边栏重组说明
- `docs/SIDEBAR_BEFORE_AFTER.md` - 修改对比

## 常见问题

### Q1: 为什么不直接删除 pages 目录？
A: 保留作为预留，未来可能需要真正的多页面功能。

### Q2: 能否通过配置禁用自动导航？
A: Streamlit 目前没有提供相关配置选项。

### Q3: 如果将来需要多页面怎么办？
A: 可以：
- 在 `pages/` 中添加新页面文件（会自动生成导航）
- 或继续使用 tab 方式（更灵活）

## 修改历史

- **2025-10-30** - 初始版本
  - 移动页面文件到 components 目录
  - 更新导入路径
  - 修复侧边栏自动导航问题

---

**最后更新**: 2025-10-30  
**修改人**: AI Assistant  
**问题**: 侧边栏顶部三行英文链接  
**解决**: 移动文件出 pages 目录
