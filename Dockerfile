# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 基础构建依赖（尽量精简）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 仅安装容器运行所需的精简依赖（重度依赖后续按需加入）
COPY requirements.docker.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 复制项目
COPY . .

EXPOSE 9090

# 默认以模拟模式启动（有Prometheus指标暴露）
CMD ["python", "main.py", "--mode", "simulation", "--log-level", "INFO"]