# 环境变量与启动指南（Windows PowerShell）

本项目关键组件均可通过环境变量进行配置，避免在代码中硬编码路径与密钥。

## 1) 必备与可选环境变量

- 路径类
  - TRADINGAGENTS_PATH：TradingAgents 项目路径（例：G:\test\tradingagents-cn-plus）
  - RDAGENT_PATH：RD-Agent 项目路径（例：G:\test\RD-Agent）
- Qlib Serving
  - QLIB_SERVING_URL：服务地址（默认 http://localhost:9000）
  - QLIB_SERVING_API_KEY：访问密钥（可选）
- MLflow
  - MLFLOW_TRACKING_URI（默认 http://localhost:5000）
  - MLFLOW_EXPERIMENT（默认 qilin_limitup）
  - MLFLOW_MODEL_NAME（默认 qilin_limitup_v1）
- 数据库（可二选一）
  - PostgreSQL：DB_HOST、DB_PORT、DB_NAME、DB_USER、DB_PASSWORD
  - SQLite：SQLITE_PATH（缺省为工作目录下 qilin_stack.db）

## 2) PowerShell 设置方式

- 当前会话生效（窗口关闭失效）

```powershell
$env:TRADINGAGENTS_PATH = "G:\test\tradingagents-cn-plus"
$env:RDAGENT_PATH       = "G:\test\RD-Agent"
$env:QLIB_SERVING_URL   = "http://localhost:9000"
$env:MLFLOW_TRACKING_URI= "http://localhost:5000"
# 可选：Postgres；如未配置将自动回退SQLite
$env:DB_HOST = "127.0.0.1"; $env:DB_PORT = "5432"
$env:DB_NAME = "qilin_stack"; $env:DB_USER = "admin"; $env:DB_PASSWORD = "{{DB_PASSWORD}}"
```

- 永久生效（需重启终端）

```powershell
setx TRADINGAGENTS_PATH "G:\test\tradingagents-cn-plus"
setx RDAGENT_PATH       "G:\test\RD-Agent"
setx QLIB_SERVING_URL   "http://localhost:9000"
setx MLFLOW_TRACKING_URI "http://localhost:5000"
# 如需 Postgres：
setx DB_HOST "127.0.0.1"
setx DB_PORT "5432"
setx DB_NAME "qilin_stack"
setx DB_USER "admin"
setx DB_PASSWORD "<请填入数据库密码>"
```

> 提示：请勿在终端回显密钥内容。将密钥写入环境变量后直接使用，避免 `echo` 输出。

## 3) 启动界面

- 安装依赖（示例）

```powershell
pip install -r requirements.txt
```

- 运行 Streamlit 界面

```powershell
streamlit run .\web\unified_dashboard.py
```

启动后，“在线服务/MLflow/TradingAgents/RD-Agent”等配置将以环境变量为默认值，可在界面内覆盖。
