---
description: Warp Code 开发配置与工作流（Qilin 量化栈：Python + FastAPI + asyncio + Qlib）
globs: **/*
alwaysApply: true
---

# Qilin Stack 开发 Profile（Warp Code）

- 语言与运行时
  - Python >= 3.8，优先使用类型注解与 dataclass/TypedDict，保持类型友好
  - 异步优先（asyncio/aiohttp/uvicorn），IO 密集场景尽量并发
  - 配置统一用 pydantic-settings + YAML/ENV，严禁在代码中硬编码密钥

- 主要依赖（参考 requirements.txt）
  - 量化/数值：numpy, pandas, scipy, qlib, ta-lib, empyrical, pyfolio
  - ML/MLOps：scikit-learn, lightgbm, xgboost, torch, tensorflow, mlflow
  - Web/异步：fastapi, uvicorn, aiohttp, aiofiles
  - 数据与消息：sqlalchemy, asyncpg, pymongo, redis, clickhouse-driver, aiokafka, celery
  - 监控与日志：prometheus-client, python-json-logger, loguru, sentry-sdk
  - 质量工具：pytest(+plugins), black, flake8, isort, mypy

- 目录约定（只列关键）
  - app/core：交易/风控/回测/执行等核心模块
  - app/agents：多智能体（集成决策、市场生态、风控等）与上下文建模
  - app/web/unified_agent_dashboard.py：Streamlit 统一分析面板
  - tests：pytest 测试与集成脚本（pytest.ini 设定覆盖率≥80%，并行、标记、报告）
  - scripts、deploy、k8s、docker-compose.yml：运维与部署
  - .taskmaster、.mcp.json：任务编排与 MCP 集成

- 代码风格与约束
  - 函数式优先，小而清晰的模块边界；长函数拆分，复用公共工具
  - 数据处理尽量向量化（pandas/numba），避免 Python 级别大循环
  - 异步边界清晰：CPU 密集放线程/进程池；IO 密集用 await 并发
  - 错误处理：显式异常类型 + 结构化日志；对外 API 返回标准错误模型
  - 日志：统一 logger（logging/loguru），关键路径埋点（时延、吞吐、错误）
  - 配置：所有密钥/端点走环境变量或配置文件；禁止泄露到客户端或日志

- 常用命令（Warp 内一键执行）
  - 安装依赖（本地）：
    - Windows: `python -m venv .venv && .venv\\Scripts\\python -m pip install -U pip && .venv\\Scripts\\pip install -r requirements.txt`
    - Unix: `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt`
  - 运行快速演示：`python quickstart.py`
  - 启动主系统（模拟/纸面/实盘）：`python main.py --mode simulation|paper|live`
  - 启动统一仪表板：`python -m streamlit run app/web/unified_agent_dashboard.py`
  - 运行测试（pytest.ini 已配置并行与覆盖率）：`pytest`
  - 质量检查：
    - 格式化：`black . && isort .`
    - 静态检查：`flake8 .`
    - 类型检查：`mypy app`

- 测试与质量门槛（pytest.ini 已设定）
  - 覆盖率门槛 80%（--cov-fail-under=80）
  - 并行执行（-n auto），最慢用例统计与严格标记
  - 推荐为新增功能补齐：单元 + 集成 + 性能标记用例

- Web/API 规范
  - FastAPI 路由：输入输出模型均用 Pydantic v2；参数校验与错误响应一致化
  - Uvicorn 启动参数：`--workers` 按 CPU 与 IO 模型权衡；生产接入反向代理/网关

- 数据与性能
  - 优先批量/向量化处理；必要时引入缓存（内存/Redis），控制 TTL 与一致性校验
  - 大型回测/训练任务标注为离线任务，Celery/Kafka 编排，监控指标上报 Prometheus

- 监控与可观测
  - Prometheus 指标在关键模块打点（交易数、活跃仓位、系统健康等），Grafana 看板
  - 日志按模块/请求/任务关联 trace_id，便于归因

- 安全
  - 密钥统一放 .env 或外部 Secret 管理；禁入仓、禁打印
  - 交易/下单相关模块引入签名与审计；对外接口限流与鉴权

- 任务编排（Task Master + MCP）
  - 建议在 Warp 中直接使用 MCP 工具：列任务（get_tasks）、next、expand、set-status 等
  - 大型需求建议先写 PRD（.taskmaster/docs），用 `parse_prd` 生成任务，再 `analyze/expand`

- 提交与变更
  - 小步提交，附带任务/子任务 ID 与变更要点；与规则文档同步更新

附：路径速查
- 快速演示：`quickstart.py`
- 主入口：`main.py`
- 仪表板：`app/web/unified_agent_dashboard.py`
- 测试脚本：`run_tests.py`
- 任务系统：`.taskmaster/`，MCP：`.mcp.json`
