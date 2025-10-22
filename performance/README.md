# 性能优化模块

## 📋 概述

本模块提供**并发优化**和**多级缓存**功能，显著提升Qilin Stack决策引擎性能。

## 🚀 核心功能

### 1. 并发优化 (`concurrency.py`)

- ✅ 线程池执行器（8个worker）
- ✅ 进程池执行器（4个worker）
- ✅ 异步任务并行执行
- ✅ 批量并行处理
- ✅ 装饰器支持

### 2. 多级缓存 (`cache.py`)

- ✅ L1内存缓存（LRU淘汰）
- ✅ L2 Redis缓存（可选）
- ✅ 自动缓存回填
- ✅ 装饰器支持
- ✅ 缓存失效管理

## 📦 文件结构

```
performance/
├── __init__.py           # 模块初始化
├── concurrency.py        # 并发优化（105行）
├── cache.py              # 多级缓存（197行）
├── benchmark.py          # 性能基准测试（217行）
├── demo.py               # 演示脚本（196行）
└── README.md             # 本文档
```

## 🔧 使用方法

### 并发优化

```python
from performance.concurrency import get_optimizer

optimizer = get_optimizer()

# 并行执行多个异步任务
tasks = [task1(), task2(), task3()]
results = await optimizer.gather_parallel(*tasks)

# 在线程池运行同步函数
result = await optimizer.run_in_thread(sync_function, args)

# 批量并行处理
results = await optimizer.batch_process(items, func, batch_size=10)

# 清理资源
optimizer.cleanup()
```

### 多级缓存

```python
from performance.cache import get_cache, cached

# 基本操作
cache = get_cache()
cache.set('key', value, ttl=600)
value = cache.get('key')

# 装饰器（推荐）
@cached(ttl=600, key_prefix="market_data")
async def get_market_data(symbol: str, date: str):
    # 昂贵的计算或IO操作
    return data
```

### 集成到决策引擎

决策引擎已自动集成性能优化：

```python
from decision_engine.core import get_decision_engine

# 创建引擎（默认启用优化）
engine = get_decision_engine()

# 并行生成决策
decisions = await engine.make_decisions(symbols, date)
```

## 📊 性能基准测试

### 运行测试

```bash
# 快速测试（3轮）
python performance/benchmark.py quick

# 完整测试（10轮）
python performance/benchmark.py full

# 压力测试（100只股票）
python performance/benchmark.py stress
```

### 预期结果

| 模式 | 10只股票 | 加速比 |
|------|---------|--------|
| 串行 | ~1.5秒  | 1.0x   |
| 并行 | ~0.5秒  | 3.0x   |

**性能提升**:
- ⚡ 加速比: **2.5-3.0x**
- ⏱️ 时间节省: **65-70%**
- 📊 吞吐量: **提升200-300%**

## 🎬 演示

运行性能演示：

```bash
python performance/demo.py
```

演示包括：
1. 并发优化效果
2. 缓存优化效果
3. 组合优化效果

## ⚙️ 配置

### 并发配置

```python
from performance.concurrency import ConcurrencyOptimizer

optimizer = ConcurrencyOptimizer(
    max_workers=8  # 线程池大小
)
```

### 缓存配置

```python
from performance.cache import MultiLevelCache

cache = MultiLevelCache(
    use_redis=True  # 启用Redis（需安装redis-py）
)
```

## 🔍 技术细节

### 并发优化原理

1. **AsyncIO并行**: 利用`asyncio.gather`并行执行三个信号生成器
2. **线程池**: 处理同步IO操作（文件、数据库）
3. **进程池**: 处理CPU密集型计算（可选）

### 缓存策略

1. **L1缓存（内存）**:
   - LRU淘汰策略
   - TTL: 5分钟
   - 容量: 1000项

2. **L2缓存（Redis）**:
   - TTL: 1小时
   - 自动降级到内存
   - 支持分布式

### 缓存Key生成

```python
cache_key = md5(f"{prefix}:{args}:{kwargs}")
```

## 📈 优化建议

### 1. 调整并发数

根据CPU核心数调整：

```python
import os
max_workers = os.cpu_count()  # 或 os.cpu_count() * 2
```

### 2. 优化缓存TTL

根据数据更新频率：

- 实时数据: 60-300秒
- 日内数据: 600-1800秒
- 历史数据: 3600-86400秒

### 3. 启用Redis

生产环境建议启用Redis：

```bash
# 安装Redis
pip install redis

# 启动Redis
docker run -d -p 6379:6379 redis:latest

# 配置
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

## 🐛 故障排查

### 问题1: 性能优化未启用

**症状**: 看到 "⚠️ 性能优化未启用"

**解决**:
```bash
# 确保performance模块在正确位置
ls qilin_stack_with_ta/performance/
```

### 问题2: Redis连接失败

**症状**: "⚠️ Redis未安装"

**解决**:
```bash
pip install redis
# 或使用内存缓存（默认）
```

### 问题3: 加速比低于预期

**原因**:
- IO延迟不足（需要真实网络/数据库调用）
- 任务数太少
- CPU资源不足

## 📚 相关文档

- [决策引擎文档](../docs/ARCHITECTURE.md)
- [快速开始](../docs/QUICKSTART.md)
- [配置指南](../docs/CONFIGURATION.md)

## 🎯 最佳实践

1. ✅ **总是使用装饰器**: 简洁且易维护
2. ✅ **合理设置TTL**: 根据数据特性调整
3. ✅ **监控缓存命中率**: 通过监控系统跟踪
4. ✅ **避免缓存过大对象**: 影响内存和序列化性能
5. ✅ **定期清理**: 归档旧数据，避免缓存污染

## 📞 支持

如有问题，请参考：
- [GitHub Issues](https://github.com/your-repo/issues)
- [文档](../docs/)
- [示例代码](demo.py)

---

**版本**: 1.0  
**更新**: 2025-10-21  
**状态**: 生产就绪
