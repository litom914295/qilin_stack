# 定时任务调度系统

## 概述

基于APScheduler的交易自动化调度系统，实现每日交易流程的自动化执行。

## 功能特性

- ✅ **每日定时任务**：T日筛选、T+1竞价监控、买入执行、T+2卖出执行、盘后分析
- ✅ **灵活调度**：支持Cron、Interval、Date三种触发器
- ✅ **双模式运行**：阻塞模式和后台模式
- ✅ **任务管理**：启动、停止、暂停、恢复、移除任务
- ✅ **执行监控**：任务执行历史记录和事件监听
- ✅ **配置驱动**：完全基于配置文件的时间设置

## 核心任务

### 1. T日候选筛选（15:30）
- 在T日收盘后执行
- 筛选次日竞价候选股票
- 基于封单强度和预测得分

### 2. T+1竞价监控（09:15）
- 集合竞价时段监控
- 生成买入信号
- 评估竞价强度

### 3. T+1买入执行（09:30）
- 开盘后立即执行
- 基于竞价信号买入
- Kelly仓位分配

### 4. T+2卖出执行（09:30）
- T+2日开盘执行
- 基于止盈止损策略
- 支持部分卖出

### 5. 每日盘后分析（16:00）
- 收盘后数据分析
- 生成交易日志
- 复盘报告生成

## 快速开始

### 基本使用

```python
from scheduler.task_scheduler import TradingScheduler

# 创建调度器
scheduler = TradingScheduler(mode='background')

# 添加每日任务
scheduler.add_daily_tasks()

# 启动调度器
scheduler.start()

# 查看任务列表
scheduler.print_jobs()
```

### 命令行启动

```bash
# 启动调度器（阻塞模式）
python run_scheduler.py

# 指定配置文件
python run_scheduler.py --config config/custom_config.yaml

# 后台模式
python run_scheduler.py --mode background

# 测试模式（立即运行所有任务）
python run_scheduler.py --test
```

## 配置说明

在 `config/default_config.yaml` 中配置调度时间：

```yaml
scheduler:
  enable_scheduler: true
  t_day_screening_time: '15:30'      # T日筛选时间
  t1_auction_monitor_time: '09:15'   # T+1竞价监控时间
  t2_sell_time: '09:30'              # T+2卖出时间
  timezone: 'Asia/Shanghai'          # 时区
```

## 高级用法

### 添加自定义任务

```python
# Cron任务（每天10:00执行）
scheduler.add_custom_task(
    func=my_function,
    trigger='cron',
    task_id='my_task',
    task_name='自定义任务',
    hour=10,
    minute=0,
    day_of_week='mon-fri'
)

# Interval任务（每30分钟执行一次）
scheduler.add_custom_task(
    func=my_function,
    trigger='interval',
    task_id='interval_task',
    task_name='间隔任务',
    minutes=30
)

# Date任务（指定时间执行一次）
from datetime import datetime
scheduler.add_custom_task(
    func=my_function,
    trigger='date',
    task_id='date_task',
    task_name='一次性任务',
    run_date=datetime(2024, 12, 1, 10, 0)
)
```

### 任务控制

```python
# 暂停任务
scheduler.pause_job('t_day_screening')

# 恢复任务
scheduler.resume_job('t_day_screening')

# 移除任务
scheduler.remove_job('t_day_screening')

# 立即运行任务
scheduler.run_job_now('t_day_screening')

# 查看执行历史
history = scheduler.get_execution_history(limit=10)
for record in history:
    print(f"{record['time']}: {record['job_id']} - {record['status']}")
```

### 事件监听

调度器自动监听任务执行事件：
- 任务执行成功 → 记录到执行历史
- 任务执行失败 → 记录错误信息

## 运行模式

### 阻塞模式（Blocking）
- 主线程被调度器占用
- 适合独立运行的调度服务
- 简单可靠

```python
scheduler = TradingScheduler(mode='blocking')
scheduler.add_daily_tasks()
scheduler.start()  # 阻塞在这里
```

### 后台模式（Background）
- 调度器在后台线程运行
- 主线程可以做其他事情
- 适合集成到其他应用

```python
scheduler = TradingScheduler(mode='background')
scheduler.add_daily_tasks()
scheduler.start()

# 主线程继续运行
while True:
    time.sleep(1)
```

## 日志

调度器日志输出到：
- 控制台（INFO级别）
- `logs/scheduler.log`（文件）

日志包含：
- 任务添加信息
- 任务执行状态
- 错误和异常
- 系统事件

## 注意事项

1. **交易日判断**：所有任务默认只在周一至周五执行（`day_of_week='mon-fri'`）
2. **时区设置**：确保配置文件中时区设置正确（默认 `Asia/Shanghai`）
3. **依赖模块**：确保工作流和配置管理器正确初始化
4. **错误处理**：任务失败不会影响其他任务的执行
5. **优雅退出**：使用 Ctrl+C 停止调度器会等待当前任务完成

## 依赖

```
apscheduler>=3.10.0
pytz>=2023.3
```

安装依赖：
```bash
pip install apscheduler pytz
```

## 示例

完整示例见 `scheduler/task_scheduler.py` 的 `__main__` 部分。

## 故障排查

### 任务不执行
- 检查系统时间和时区
- 确认任务触发时间配置正确
- 查看日志文件中的错误信息

### 工作流未初始化
- 检查依赖模块是否正确导入
- 确认配置文件路径正确
- 验证配置参数有效性

### 调度器无法启动
- 检查端口占用（如果使用远程调度）
- 确认APScheduler版本兼容
- 查看完整异常堆栈

## 相关文档

- [工作流文档](../workflow/README.md)
- [配置管理文档](../config/README.md)
- [APScheduler官方文档](https://apscheduler.readthedocs.io/)
