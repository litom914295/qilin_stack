# 📊 Qilin Stack 数据接入指南

本指南说明如何配置和使用多个数据源。

---

## 🎯 支持的数据源

| 数据源 | 覆盖范围 | 是否免费 | 延迟 |
|--------|---------|---------|------|
| **Qlib** | A股 | ✅ | 1天 |
| **AKShare** | A股/基金/期货 | ✅ | 实时-15分钟 |
| **Tushare** | A股/港股/美股 | ⚠️ 需token | 实时-1天 |
| **Yahoo Finance** | 全球股票 | ✅ | 15-20分钟 |

---

## 📖 快速配置

### 1. Qlib数据 (推荐)

```bash
# 下载A股数据
python scripts/get_data.py qlib cn_data
```

### 2. AKShare (无需配置)

```python
from qlib_enhanced.unified_data_interface import UnifiedDataInterface

interface = UnifiedDataInterface()
data = interface.fetch_data(
    symbols=['000001.SZ'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    provider='akshare'
)
```

### 3. Tushare (需要Token)

```bash
# 设置环境变量
export TUSHARE_TOKEN=your_token_here

# 或在代码中设置
import tushare as ts
ts.set_token('your_token_here')
```

注册地址: https://tushare.pro/register

### 4. Yahoo Finance

```python
data = interface.fetch_data(
    symbols=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    provider='yahoo'
)
```

---

## 🔧 统一数据接口

```python
from qlib_enhanced.unified_data_interface import UnifiedDataInterface

# 创建接口 (自动降级)
interface = UnifiedDataInterface()

# 获取数据 (自动尝试多个数据源)
data = interface.fetch_with_fallback(
    symbols=['000001.SZ', '600000.SH'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 优先级: Qlib → AKShare → Tushare → Yahoo
```

---

## 📝 更多信息

详见完整文档:
- `AKSHARE_DATA_USAGE_GUIDE.md`
- `DOWNLOAD_QLIB_DATA.md`
- `QLIB_DATA_GUIDE.md`
