# Qlib 本地数据下载指南

## 🎯 推荐方案

由于 pyqlib 0.9.7 不包含自动下载功能,以下是几种获取 Qlib 数据的实用方法:

---

## 方案 A: 使用 Qlib 官方数据包 (推荐用于离线回测)

### 步骤1: 从 Gitee 镜像下载(国内推荐)

Qlib 在 Gitee 有镜像,下载速度较快:

```powershell
# 1. 访问 Gitee 镜像
# https://gitee.com/mirrors/qlib

# 2. 或使用命令行下载(需要 git)
git clone https://gitee.com/mirrors/qlib.git --depth=1
cd qlib/scripts/data_collector/cn_data
```

### 步骤2: 从百度网盘下载

根据 Qlib 官方文档,数据也可从百度网盘获取:
- 链接: 参见 Qlib GitHub README
- 提取码: 参见官方文档

### 步骤3: 解压到目标目录

```powershell
# 目标目录
$target = "$HOME\.qlib\qlib_data\cn_data"

# 创建目录
New-Item -ItemType Directory -Path $target -Force

# 解压数据包
# 如果是 .zip 文件
Expand-Archive -Path qlib_cn_data.zip -DestinationPath $target

# 如果是 .tar.gz 文件(需要7-Zip或其他工具)
# 7z x qlib_cn_data.tar.gz -o$target
```

### 步骤4: 验证数据

```powershell
.\.qilin\Scripts\Activate.ps1

python -c "
import qlib
from qlib.data import D

# 初始化
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

# 测试
cal = D.calendar(start_time='2024-01-01', end_time='2024-01-10')
print(f'✓ 获取到 {len(cal)} 个交易日')

instruments = D.instruments(market='all')
print(f'✓ 获取到 {len(instruments)} 只股票')
print('✅ 数据验证通过!')
"
```

---

## 方案 B: 使用 AkShare 在线数据 (推荐用于实时交易)

**优点**: 
- ✅ 无需下载,节省磁盘空间
- ✅ 实时更新,数据最新
- ✅ 免费,无需注册

**缺点**:
- ⚠️ 需要网络连接
- ⚠️ 首次加载较慢

### 配置方法

编辑 `config.yaml`:

```yaml
# 数据源配置
data:
  # 使用 AkShare (推荐)
  akshare:
    enabled: true
  
  # 禁用本地 Qlib 数据
  storage:
    use_local: false
```

### 直接使用

```powershell
# 激活环境
.\.qilin\Scripts\Activate.ps1

# 运行系统(会自动使用 AkShare)
python quickstart.py
```

---

## 方案 C: 使用 Tushare 数据 (适合专业用户)

**优点**:
- ✅ 数据质量高
- ✅ 数据全面
- ✅ 支持分钟级数据

**缺点**:
- ⚠️ 需要注册并获取积分
- ⚠️ 高级功能需要VIP

### 步骤

1. **注册 Tushare**
   - 访问: https://tushare.pro
   - 注册并获取 token

2. **配置 token**

编辑 `config.yaml`:

```yaml
data:
  tushare:
    token: "your_tushare_token_here"
    enabled: true
```

3. **使用**

```python
from data_layer.data_access_layer import DataAccessLayer

dal = DataAccessLayer({"tushare_token": "your_token"})
df = await dal.get_daily_data(symbols=['000001.SZ'], start_date='2024-01-01')
```

---

## 方案 D: 自己收集数据 (适合定制需求)

如果您有特殊需求,可以自己收集数据:

```python
import akshare as ak
import pandas as pd
from pathlib import Path

# 获取股票列表
stock_list = ak.stock_zh_a_spot_em()

# 下载数据
for stock_code in stock_list['代码'][:10]:  # 示例:前10只
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date="20200101",
            end_date="20241231"
        )
        
        # 保存
        output_dir = Path.home() / ".qilin" / "csv_data" / "cn_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{stock_code}.csv")
        print(f"✓ {stock_code}")
    except Exception as e:
        print(f"✗ {stock_code}: {e}")
```

然后使用 Qlib 工具转换格式:

```powershell
python -m qlib.data.storage `
    --csv_path "$HOME\.qlib\csv_data\cn_data" `
    --qlib_dir "$HOME\.qlib\qlib_data\cn_data" `
    --include_fields open,close,high,low,volume,factor
```

---

## 💡 我的建议

根据您的使用场景:

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 学习测试 | **方案 B (AkShare)** | 简单,无需下载 |
| 离线回测 | **方案 A (官方数据包)** | 速度快,数据完整 |
| 实盘交易 | **方案 B + C (AkShare/Tushare)** | 数据实时 |
| 研究开发 | **方案 A + B** | 灵活切换 |

---

## ❓ 常见问题

### Q1: 为什么自动下载失败?
A: pyqlib 0.9.7 版本不包含数据下载功能,需要手动下载或使用在线数据源

### Q2: 官方数据包在哪里下载?
A: 
- GitHub: https://github.com/microsoft/qlib (可能较慢)
- Gitee 镜像: https://gitee.com/mirrors/qlib (国内推荐)
- 百度网盘: 参见 Qlib 官方文档

### Q3: 数据多久更新一次?
A: 
- 官方数据包: 不定期更新
- AkShare: 实时更新
- Tushare: 每日更新

### Q4: 可以混合使用多个数据源吗?
A: 可以!麒麟系统支持配置多个数据源,会自动选择最合适的

---

## 🔗 相关链接

- Qlib GitHub: https://github.com/microsoft/qlib
- Qlib Gitee 镜像: https://gitee.com/mirrors/qlib
- AkShare 文档: https://akshare.akfamily.xyz/
- Tushare 官网: https://tushare.pro
- 麒麟系统文档: `README.md`

---

## 📞 需要帮助?

如果遇到问题:
1. 查看 `INIT_COMPLETE.md` 获取完整初始化指南
2. 查看 `QLIB_DATA_GUIDE.md` 了解数据配置
3. 查看系统日志: `logs/qilin.log`

**建议**: 对于大多数用户,直接使用 AkShare (方案 B) 是最简单的选择!
