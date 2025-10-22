FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 基础构建依赖（用于编译numpy/pandas等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 预装Python依赖
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# 复制项目源码
COPY . /app

# Streamlit默认端口
EXPOSE 8501

# 默认启动Dashboard（Compose中会覆盖为其它命令）
CMD ["python","-m","streamlit","run","web/unified_dashboard.py","--server.port=8501","--server.address=0.0.0.0"]
