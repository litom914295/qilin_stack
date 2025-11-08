# 多渠道消息推送系统

## 概述

支持企业微信、钉钉、邮件等多种渠道的统一消息推送系统。

## 功能特性

- ✅ **企业微信机器人**：Markdown格式，支持表情和样式
- ✅ **钉钉机器人**：Markdown格式，支持表情和样式
- ✅ **邮件推送**：HTML邮件，美观的样式和配色
- ✅ **消息级别**：INFO、WARNING、ERROR、SUCCESS
- ✅ **便捷方法**：竞价信号、买入通知、卖出通知、每日报告、错误告警
- ✅ **推送历史**：记录所有推送请求和结果

## 快速开始

### 基本使用

```python
from notification.notifier import Notifier, NotificationLevel

# 创建推送器
notifier = Notifier()

# 发送消息
notifier.send(
    title="测试消息",
    content="这是一条测试消息",
    level=NotificationLevel.INFO
)
```

### 配置

在 `config/default_config.yaml` 中配置推送渠道：

```yaml
notification:
  enable_notification: true
  channels: ['wechat', 'dingtalk', 'email']  # 启用的推送渠道
  
  # 企业微信配置
  wechat_webhook: 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY'
  
  # 钉钉配置
  dingtalk_webhook: 'https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN'
  
  # 邮件配置
  email_smtp_server: 'smtp.gmail.com'
  email_smtp_port: 587
  email_from: 'your_email@gmail.com'
  email_password: 'your_password'  # 应用密码或授权码
  email_to: ['recipient1@example.com', 'recipient2@example.com']
```

## 推送渠道设置

### 企业微信机器人

1. 在企业微信群聊中添加机器人
2. 获取Webhook URL
3. 配置到 `wechat_webhook`
4. 支持Markdown格式

### 钉钉机器人

1. 在钉钉群聊中添加自定义机器人
2. 获取Webhook URL（带access_token）
3. 配置到 `dingtalk_webhook`
4. 支持Markdown格式

### 邮件配置

常用SMTP服务器配置：

**Gmail**
```yaml
email_smtp_server: 'smtp.gmail.com'
email_smtp_port: 587
```

**QQ邮箱**
```yaml
email_smtp_server: 'smtp.qq.com'
email_smtp_port: 587
```

**163邮箱**
```yaml
email_smtp_server: 'smtp.163.com'
email_smtp_port: 465
```

**注意**：Gmail需要使用应用专用密码，QQ/163邮箱需要开启SMTP服务并使用授权码。

## 便捷方法

### 1. 竞价信号推送

```python
signals = [
    {
        'symbol': '000001.SZ',
        'name': '平安银行',
        'auction_strength': 0.85,
        'auction_price': 12.50
    },
    # ...
]

notifier.send_auction_signal(signals)
```

**推送效果**：
```
✅ 竞价买入信号

信号数量: 2

1. **000001.SZ** 平安银行
   - 竞价强度: 85.00%
   - 竞价价格: 12.50

2. **600519.SH** 贵州茅台
   - 竞价强度: 92.00%
   - 竞价价格: 1680.00
```

### 2. 买入通知

```python
orders = [
    {
        'symbol': '000001.SZ',
        'price': 12.50,
        'volume': 1000,
        'amount': 12500
    },
    # ...
]

notifier.send_buy_notification(orders)
```

### 3. 卖出通知

```python
orders = [
    {
        'symbol': '000001.SZ',
        'sell_price': 13.00,
        'profit': 500,
        'profit_rate': 0.04
    },
    # ...
]

notifier.send_sell_notification(orders)
```

### 4. 每日报告

```python
report = {
    'date': '2024-11-01',
    'candidates': 23,
    'buy_orders': 12,
    'sell_orders': 8,
    'profit': 3240.50,
    'profit_rate': 0.0254
}

notifier.send_daily_report(report)
```

### 5. 错误告警

```python
notifier.send_error_alert("数据加载失败: 文件不存在")
```

## 消息级别

系统支持4个消息级别，不同级别有不同的表情和颜色：

| 级别 | 表情 | 颜色 | 用途 |
|------|------|------|------|
| INFO | ℹ️ | 蓝色 | 一般信息 |
| SUCCESS | ✅ | 绿色 | 成功操作 |
| WARNING | ⚠️ | 橙色 | 警告信息 |
| ERROR | ❌ | 红色 | 错误告警 |

使用示例：

```python
from notification.notifier import NotificationLevel

# INFO级别
notifier.send("信息", "普通信息", level=NotificationLevel.INFO)

# SUCCESS级别
notifier.send("成功", "操作成功", level=NotificationLevel.SUCCESS)

# WARNING级别
notifier.send("警告", "注意风险", level=NotificationLevel.WARNING)

# ERROR级别
notifier.send("错误", "系统异常", level=NotificationLevel.ERROR)
```

## 高级用法

### 指定推送渠道

```python
# 只推送到企业微信
notifier.send(
    title="测试消息",
    content="只发送到企业微信",
    channels=['wechat']
)

# 推送到企业微信和邮件
notifier.send(
    title="测试消息",
    content="发送到多个渠道",
    channels=['wechat', 'email']
)
```

### 查看推送历史

```python
# 获取最近10条推送记录
history = notifier.get_history(limit=10)

for record in history:
    print(f"时间: {record['timestamp']}")
    print(f"标题: {record['title']}")
    print(f"级别: {record['level']}")
    print(f"渠道: {record['channels']}")
    print(f"结果: {record['results']}")
```

### 单例模式

```python
from notification.notifier import get_notifier

# 获取全局推送器实例
notifier = get_notifier()

# 之后所有地方都使用同一个实例
notifier2 = get_notifier()
assert notifier is notifier2  # True
```

### 与工作流集成

在工作流中自动推送：

```python
from workflow.trading_workflow import TradingWorkflow
from notification.notifier import get_notifier

workflow = TradingWorkflow()
notifier = get_notifier()

# T+1竞价监控后推送信号
result = workflow.stage_t1_auction_monitor(date)
if result['status'] == 'completed':
    signals = result['data']['buy_signals']
    notifier.send_auction_signal(signals)

# T+1买入执行后推送通知
result = workflow.stage_t1_buy_execution(date)
if result['status'] == 'completed':
    orders = result['data']['buy_orders']
    notifier.send_buy_notification(orders)
```

## 依赖

```
requests>=2.28.0
```

安装依赖：
```bash
pip install requests
```

## 故障排查

### 企业微信推送失败

- 检查Webhook URL是否正确
- 确认机器人未被删除或禁用
- 检查消息格式是否符合规范
- 查看返回的错误码和错误消息

### 钉钉推送失败

- 检查access_token是否正确
- 确认机器人安全设置（关键词、IP白名单、签名）
- 检查消息频率限制
- 查看返回的错误码

### 邮件推送失败

- 检查SMTP服务器和端口
- 确认已开启SMTP服务
- 使用应用专用密码或授权码（不是登录密码）
- 检查是否被防火墙拦截
- 查看异常堆栈信息

### 推送未启用

```python
# 检查配置
notifier.enabled  # True/False

# 强制启用
notifier.enabled = True
```

### 未配置推送渠道

```python
# 检查配置的渠道
notifier.channels  # []

# 临时指定渠道
notifier.send(
    title="测试",
    content="内容",
    channels=['wechat']
)
```

## 最佳实践

1. **安全性**
   - 不要将Webhook URL和密码提交到代码仓库
   - 使用环境变量或配置文件管理敏感信息
   - 定期更换密钥和密码

2. **推送频率**
   - 避免频繁推送，影响用户体验
   - 重要信号才推送，过滤噪音
   - 合并相关消息，避免刷屏

3. **消息内容**
   - 标题简洁明了
   - 内容结构清晰
   - 包含关键信息
   - 适当使用表情和格式

4. **错误处理**
   - 推送失败不应影响主流程
   - 记录推送历史便于追溯
   - 监控推送成功率

## 示例

完整示例见 `notification/notifier.py` 的 `__main__` 部分。

## 相关文档

- [工作流文档](../workflow/README.md)
- [调度器文档](../scheduler/README.md)
- [企业微信机器人文档](https://developer.work.weixin.qq.com/document/path/91770)
- [钉钉机器人文档](https://open.dingtalk.com/document/robots/custom-robot-access)
