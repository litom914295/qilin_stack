# 每日涨停股筛选数据

## 功能说明

这个目录存储每日涨停股筛选结果，Dashboard 会自动读取最新的筛选数据，在左侧导航栏显示当日最强势的前 3 只股票。

## 使用方法

### 1. 手动运行筛选脚本

```bash
# 运行筛选脚本，生成当日数据
python scripts/daily_limitup_selector.py
```

脚本会：
- 获取当日涨停股数据（目前使用模拟数据，实际应接入 AKShare/Tushare）
- 使用 `EnhancedLimitUpSelector` 计算质量评分和置信度
- 筛选出优质股票
- 保存结果到 `data/daily_selections/limitup_YYYYMMDD.json`
- 显示前 3 只最强势股票

### 2. 设置定时任务（推荐）

**Windows 任务计划程序：**

每日 15:30（收盘后）自动运行：

```powershell
# 创建任务
$action = New-ScheduledTaskAction -Execute "python" -Argument "G:\test\qilin_stack\scripts\daily_limitup_selector.py" -WorkingDirectory "G:\test\qilin_stack"
$trigger = New-ScheduledTaskTrigger -Daily -At 15:30
Register-ScheduledTask -TaskName "DailyLimitUpSelector" -Action $action -Trigger $trigger
```

**Linux/Mac crontab：**

```bash
# 编辑 crontab
crontab -e

# 添加任务（每日 15:30 运行）
30 15 * * 1-5 cd /path/to/qilin_stack && python scripts/daily_limitup_selector.py
```

### 3. Dashboard 自动加载

启动 Dashboard 后：
- 初始化时自动读取最新的筛选数据
- 点击"刷新数据"按钮可更新股票列表
- 如果没有筛选数据，会使用默认股票代码

## 数据格式

```json
{
  "date": "2025-11-07",
  "total_limitup": 6,
  "qualified_count": 5,
  "stocks": [
    {
      "symbol": "000858",
      "name": "股票000858",
      "quality_score": 82.0,
      "confidence": 0.73,
      "limit_up_time": "09:50:00",
      "open_times": 0,
      "seal_ratio": 0.27,
      "consecutive_days": 2,
      "is_first_board": true,
      "sector": "科技板块",
      "themes": ["人工智能", "芯片"],
      "sector_limit_count": 5
    }
  ]
}
```

## 接入实际数据源

修改 `scripts/daily_limitup_selector.py` 中的 `get_limitup_data_today()` 函数：

### 使用 AKShare

```python
import akshare as ak

def get_limitup_data_today():
    # 获取涨停股池
    df = ak.stock_zt_pool_em(date=datetime.now().strftime('%Y%m%d'))
    
    stocks = []
    for _, row in df.iterrows():
        stocks.append(LimitUpStock(
            symbol=row['代码'],
            name=row['名称'],
            limit_up_time=row['首次涨停时间'],
            open_times=row['涨停统计']['开板次数'],
            # ... 映射其他字段
        ))
    return stocks
```

### 使用 Tushare

```python
import tushare as ts

def get_limitup_data_today():
    pro = ts.pro_api('YOUR_TOKEN')
    df = pro.limit_list(
        trade_date=datetime.now().strftime('%Y%m%d'),
        limit_type='U'  # U=涨停
    )
    # ... 转换为 LimitUpStock 对象
```

## 评分标准

质量评分（0-100分）基于：
- **涨停时间**（25分）：越早越好
- **封单强度**（25分）：封单比例越高越好
- **开板次数**（20分）：0次最佳
- **换手率**（10分）：适中为好
- **板块热度**（10分）：同板块涨停数越多越好
- **连板天数**（10分）：连板有溢价

置信度（0-1）基于：
- 收盘接近最高价（40%）
- VWAP 斜率（30%）
- 板块热度（30%）

## 故障排查

**问题：Dashboard 显示默认股票**
- 检查 `data/daily_selections/` 目录是否有数据文件
- 运行筛选脚本生成数据
- 点击"刷新数据"按钮

**问题：筛选脚本报错**
- 确保已安装依赖：`pip install -r requirements.txt`
- 检查数据源连接（AKShare/Tushare）
- 查看日志输出定位具体错误
