# Qlib 数据准备指南

## 数据目录
**默认路径**: `C:\Users\Administrator\.qlib\qlib_data\cn_data`

## 下载方法

### 方法一: 使用 Qlib 官方脚本(推荐)

```bash
# 1. 激活虚拟环境
.\.qilin\Scripts\Activate.ps1

# 2. 使用 Python 脚本下载
python scripts/get_data.py dump_all --csv_path ~/.qlib/csv_data/cn_data --qlib_dir ~/.qlib/qlib_data/cn_data --include_fields open,close,high,low,volume,factor
```

### 方法二: 使用在线数据源

Qlib 支持从多个数据源获取数据:

1. **YahooFinance** (国际股票)
2. **Tushare** (中国A股,需要token)  
3. **AkShare** (中国A股,免费)

### 方法三: 下载预处理数据包

访问 Qlib 数据仓库:
- **GitHub**: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/cn_data
- **百度网盘**: 参见 Qlib 官方文档获取分享链接

下载后解压到: `C:\Users\Administrator\.qlib\qlib_data\cn_data`

## 数据验证

下载完成后,运行以下命令验证:

```python
import qlib
from qlib.data import D

# 初始化 Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

# 获取交易日历
cal = D.calendar(start_time='2024-01-01', end_time='2024-12-31')
print(f"获取到 {len(cal)} 个交易日")

# 获取股票列表
instruments = D.instruments(market='all')
print(f"获取到 {len(instruments)} 只股票")
```

## 使用麒麟系统内置数据接口

麒麟系统已集成多个数据源,可以不依赖 Qlib 数据:

```python
# 使用 AkShare 获取实时数据
from data_layer.data_access_layer import DataAccessLayer

dal = DataAccessLayer({})
df = await dal.get_daily_data(symbols=['000001.SZ'], start_date='2024-01-01')
```

## 常见问题

**Q: 数据下载很慢怎么办?**  
A: 可以使用国内镜像或百度网盘下载预处理数据包

**Q: 不下载数据能用吗?**  
A: 可以!麒麟系统支持 AkShare、Tushare 等在线数据源,无需本地数据

**Q: 如何更新数据?**  
A: 使用相同的下载命令,或使用 `python scripts/get_data.py update`

## 快速开始(跳过 Qlib 数据)

如果您想快速开始,可以直接使用在线数据源:

```python
# 修改 config.yaml
data:
  sources:
    - akshare  # 使用 AkShare 在线数据
    - tushare  # 或使用 Tushare (需配置 token)
```

然后直接运行系统:
```bash
python main.py --mode backtest --start_date 2024-01-01 --end_date 2024-12-31
```
