# AKShare 数据源集成指南

## 📡 简介

AKShare 是一个优秀的免费开源财经数据接口库，提供股票、期货、基金、债券等金融数据。

**优势**：
- ✅ 完全免费，无需注册
- ✅ 在线获取，无需提前下载数据
- ✅ 数据更新及时
- ✅ 覆盖A股、港股、美股等多个市场

## 🚀 快速开始

### 1. 安装 AKShare

```bash
pip install akshare
```

或者升级到最新版本：

```bash
pip install akshare --upgrade
```

### 2. 在Web界面中使用

1. 启动Web界面：
   ```bash
   python -m streamlit run web/unified_dashboard.py
   ```

2. 在"Qlib → 一进二策略"标签页中：
   - 选择 **"📡 AKShare在线模式（推荐）"**
   - 选择股票池（如 000001.SZ, 600519.SH）
   - 设置时间范围
   - 点击 **"📦 构建数据集"**

3. 系统会自动从网络获取真实市场数据！

## 📊 支持的数据类型

### A股数据
- 日线行情（前复权、后复权、不复权）
- 分钟级数据
- 实时行情
- 涨跌停板数据
- 龙虎榜数据

### 基本用法示例

```python
import akshare as ak

# 获取股票历史数据
df = ak.stock_zh_a_hist(
    symbol="000001",           # 股票代码（不含后缀）
    start_date="20240101",     # 开始日期
    end_date="20241029",       # 结束日期
    adjust="qfq"               # 前复权
)

# 获取实时行情
df_realtime = ak.stock_zh_a_spot_em()

# 获取涨停板数据
df_limit_up = ak.stock_zt_pool_em(date="20241029")
```

## 🔧 配置说明

### 网络配置

如果遇到网络问题，可以设置代理：

```python
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
```

### 数据格式

AKShare返回的DataFrame格式：

| 列名 | 说明 |
|-----|------|
| 日期 | 交易日期 |
| 开盘 | 开盘价 |
| 收盘 | 收盘价 |
| 最高 | 最高价 |
| 最低 | 最低价 |
| 成交量 | 成交量 |
| 成交额 | 成交金额 |
| 振幅 | 振幅 |
| 涨跌幅 | 涨跌幅 |
| 涨跌额 | 涨跌额 |
| 换手率 | 换手率 |

## 🎯 在"一进二策略"中的应用

### 数据流程

1. **数据获取**：从AKShare获取历史数据
2. **特征工程**：计算涨停相关特征
3. **模型训练**：训练Stacking模型
4. **选股预测**：生成T+1候选股票

### 涨停板数据获取

```python
import akshare as ak

# 获取指定日期的涨停板数据
df_zt = ak.stock_zt_pool_em(date="20241029")

# 包含的信息：
# - 涨停时间
# - 封单金额
# - 开板次数
# - 连板数
# - 涨停原因
```

## ⚠️ 常见问题

### 1. 导入失败

**问题**：`ModuleNotFoundError: No module named 'akshare'`

**解决**：
```bash
pip install akshare
```

### 2. 网络超时

**问题**：获取数据时网络超时

**解决**：
- 检查网络连接
- 设置HTTP代理（如果使用代理）
- 增加超时时间

### 3. 股票代码格式

**问题**：股票代码格式不正确

**说明**：
- AKShare使用纯数字代码：`000001`, `600000`
- 系统会自动转换：`000001.SZ` → `000001`

### 4. 数据缺失

**问题**：某些股票数据获取失败

**原因**：
- 股票停牌
- 股票代码不存在
- 数据源暂时不可用

**处理**：系统会自动跳过失败的股票，继续获取其他数据

## 📚 更多资源

- **官方文档**：https://akshare.akfamily.xyz/
- **GitHub仓库**：https://github.com/akfamily/akshare
- **在线教程**：https://akshare.akfamily.xyz/tutorial.html

## 🆚 与Qlib的对比

| 特性 | AKShare | Qlib |
|-----|---------|------|
| 安装 | 简单 | 复杂 |
| 数据获取 | 在线实时 | 离线预下载 |
| 网络依赖 | 需要 | 不需要 |
| 数据更新 | 自动 | 手动 |
| 速度 | 较慢（网络） | 快（本地） |
| 适用场景 | 快速开发、测试 | 生产环境 |

## 💡 建议

1. **开发测试阶段**：使用 AKShare（快速便捷）
2. **生产环境**：使用 Qlib（稳定高效）
3. **混合使用**：开发用AKShare，部署用Qlib

## 🔄 从AKShare迁移到Qlib

当你的系统稳定后，可以无缝切换到Qlib：

1. 下载Qlib数据：
   ```bash
   python -m qlib.run.get_data qlib_data --target_dir ./qlib_data --region cn
   ```

2. 在Web界面切换到 **"🔥 Qlib离线模式"**

3. 指定Qlib数据目录

4. 数据格式和特征保持一致，无需修改代码！
