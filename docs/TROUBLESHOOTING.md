# RD-Agent 集成故障排查指南

**版本**: 1.0  
**更新时间**: 2024-11-08  
**适用范围**: RD-Agent 兼容层

---

## 📋 目录

1. [常见错误和解决方案](#1-常见错误和解决方案)
2. [日志分析指南](#2-日志分析指南)
3. [性能调优](#3-性能调优)
4. [配置最佳实践](#4-配置最佳实践)
5. [诊断清单](#5-诊断清单)

---

## 1. 常见错误和解决方案

### 1.1 官方组件初始化失败

**症状**:
```
OfficialIntegrationError: Failed to initialize official RD-Agent manager
```

**可能原因**:
- 缺少必要配置 (llm_api_key)
- 环境变量未设置
- LLM provider 配置错误

**解决方案**:

```python
import os

# 方案1: 检查配置完整性
config = {
    'llm_model': 'gpt-4',
    'llm_api_key': os.getenv('OPENAI_API_KEY'),  # 确保设置
    'llm_provider': 'openai',
    'max_iterations': 10
}

# 验证必要配置
assert config['llm_api_key'], "❌ 需要设置 OPENAI_API_KEY 环境变量"

# 方案2: 使用环境变量
# Linux/Mac:
# export OPENAI_API_KEY="sk-xxx"

# Windows:
# $env:OPENAI_API_KEY="sk-xxx"

# 方案3: 从配置文件加载
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
```

---

### 1.2 Factor Loop 超时

**症状**:
```
TimeoutError: Factor loop execution timeout
```
或长时间无响应 (>5分钟)

**可能原因**:
- `max_iterations` 设置过大
- LLM API 响应慢/网络问题
- Qlib 数据加载缓慢

**解决方案**:

```python
# 1. 减小迭代次数
config['max_iterations'] = 5  # 推荐 3-10, 不要超过20

# 2. 增加超时时间
config['timeout'] = 300  # 5分钟

# 3. 使用更快的模型
config['llm_model'] = 'gpt-3.5-turbo'  # 而不是 gpt-4

# 4. 检查网络连接
import requests
try:
    requests.get('https://api.openai.com', timeout=5)
    print("✅ 网络连接正常")
except:
    print("❌ 网络连接失败,请检查代理设置")

# 5. 使用本地模型 (推荐生产环境)
config['llm_base_url'] = 'http://localhost:8000/v1'  # vllm
config['llm_provider'] = 'openai'  # 兼容 API
```

---

### 1.3 FileStorage 记录失败

**症状**:
```
Can't pickle <class 'Mock'>: it's not the same object as unittest.mock.Mock
```

**可能原因**:
- 对象包含不可序列化的数据 (Mock, Lambda, Thread)
- 磁盘空间不足
- 权限问题

**解决方案**:

```python
from rd_agent.logging_integration import QilinRDAgentLogger

logger = QilinRDAgentLogger('./logs')

# 方案1: 只记录可序列化的数据
try:
    logger.log_experiment(exp, tag='factor')
except Exception as e:
    logger.warning(f"⚠️ 日志记录失败: {e}")
    # 继续执行,不中断主流程

# 方案2: 清理对象
import copy
clean_exp = copy.deepcopy(exp)
# 移除不可序列化的属性
if hasattr(clean_exp, '_mock_data'):
    delattr(clean_exp, '_mock_data')

logger.log_experiment(clean_exp)

# 方案3: 使用 JSON 格式 (牺牲完整性换取可靠性)
logger.log_metrics({
    'hypothesis': str(exp.hypothesis),
    'ic': exp.result['IC'],
    'timestamp': datetime.now().isoformat()
}, tag='factor.metrics')
```

---

### 1.4 数据加载失败

**症状**:
```
DataNotFoundError: Cannot load factors from workspace
```

**可能原因**:
- 工作空间路径错误
- 没有运行过实验
- 数据文件损坏

**解决方案**:

```python
from rd_agent.compat_wrapper import RDAgentWrapper, DataNotFoundError

agent = RDAgentWrapper(config)

# 方案1: 使用多级兜底
try:
    factors = agent.load_factors_with_fallback(
        workspace_path='./logs/rdagent',
        n_factors=10
    )
    print(f"✅ 加载了 {len(factors)} 个因子")
except DataNotFoundError as e:
    print(f"❌ {e}")
    # 查看诊断信息
    
# 方案2: 检查工作空间
from pathlib import Path
workspace = Path('./logs/rdagent')
if not workspace.exists():
    print("❌ 工作空间不存在,正在创建...")
    workspace.mkdir(parents=True)

# 检查是否有数据
pkl_files = list(workspace.glob('**/*.pkl'))
json_files = list(workspace.glob('**/*.json'))
print(f"找到 {len(pkl_files)} 个 pkl 文件")
print(f"找到 {len(json_files)} 个 json 文件")

# 方案3: 先运行一次实验
if not pkl_files:
    print("🔄 正在运行首次实验...")
    result = await agent.research_pipeline(
        "测试实验",
        pd.DataFrame({'close': [100, 101, 102]}),
        max_iterations=2
    )
```

---

### 1.5 代码沙盒执行失败

**症状**:
```
Code validation failed: Unsafe import: os
```

**可能原因**:
- 代码包含危险操作
- 安全级别设置过严

**解决方案**:

```python
from rd_agent.code_sandbox import CodeSandbox, SecurityLevel

# 方案1: 调整安全级别 (谨慎!)
sandbox = CodeSandbox(
    security_level=SecurityLevel.MODERATE,  # 而不是 STRICT
    timeout=10
)

# 方案2: 添加允许的模块
result = sandbox.execute(
    code="import custom_lib",
    context={},
    allowed_modules=['custom_lib']
)

# 方案3: 预先导入需要的模块到 context
import numpy as np
import pandas as pd

result = sandbox.execute(
    code="result = np.mean([1,2,3])",
    context={'np': np, 'pd': pd}
)
```

---

## 2. 日志分析指南

### 2.1 设置日志级别

```python
import logging

# 开发环境 - 详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# 生产环境 - 正常日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# 只看错误
logging.basicConfig(level=logging.ERROR)

# 输出到文件
logging.basicConfig(
    level=logging.INFO,
    filename='rdagent.log',
    format='%(asctime)s [%(levelname)s] %(message)s'
)
```

### 2.2 关键日志标记

| 标记 | 含义 | 说明 |
|------|------|------|
| ✅ | 成功 | 操作成功完成 |
| ⚠️ | 警告 | 不影响主流程,但需关注 |
| ❌ | 错误 | 操作失败,需要处理 |
| 🔄 | 处理中 | 正在执行耗时操作 |
| 📊 | 统计 | 性能/统计信息 |
| 🔍 | 调试 | 调试信息 (DEBUG) |

### 2.3 日志分析示例

**正常运行日志**:
```
2024-11-08 10:00:00 [INFO] RDAgentWrapper initialized successfully
2024-11-08 10:00:01 [INFO] Starting research pipeline: A股动量因子研究
2024-11-08 10:00:02 [INFO] 🔄 Running FactorRDLoop for 10 iterations...
2024-11-08 10:02:30 [INFO] ✅ FileStorage logging enabled at ./logs
2024-11-08 10:05:00 [INFO] ✅ Logged experiments to FileStorage
2024-11-08 10:05:01 [INFO] Research pipeline completed. Found 5 factors.
```

**异常日志**:
```
2024-11-08 10:00:00 [ERROR] ❌ Failed to initialize: No API key provided
2024-11-08 10:00:01 [WARNING] ⚠️ FileStorage logging unavailable: ImportError
2024-11-08 10:00:02 [ERROR] ❌ Research pipeline failed: Connection timeout
```

---

## 3. 性能调优

### 3.1 提升速度

```python
import asyncio

# 1. 并行执行多个任务
tasks = [
    agent.discover_factors(data, n_factors=5),
    agent.discover_factors(data, n_factors=5),
    agent.discover_factors(data, n_factors=5)
]
results = await asyncio.gather(*tasks)
# 3x 速度提升

# 2. 减少迭代次数
config['max_iterations'] = 5  # 而不是 20
# 4x 速度提升

# 3. 使用缓存
factors = agent.load_factors_with_fallback()  # 从缓存加载
# 即时返回

# 4. 限制数据量
df = df.tail(10000)  # 只用最近1万条
# 2-3x 速度提升

# 5. 使用更快的模型
config['llm_model'] = 'gpt-3.5-turbo'
# 2x 速度,1/10 成本
```

### 3.2 内存优化

```python
import gc
import pandas as pd

# 1. 及时释放大对象
large_df = pd.read_csv('huge_data.csv')
result = process(large_df)
del large_df  # 立即释放
gc.collect()

# 2. 使用数据类型优化
df['volume'] = df['volume'].astype('int32')  # 而不是 int64
df['close'] = df['close'].astype('float32')  # 而不是 float64
# 内存减半

# 3. 分块处理
for chunk in pd.read_csv('data.csv', chunksize=10000):
    process_chunk(chunk)
# 恒定内存使用

# 4. 清理日志文件
from rd_agent.logging_integration import QilinRDAgentLogger
logger = QilinRDAgentLogger('./logs')
logger.clear_logs(tag='old_experiments')  # 清理旧数据
```

### 3.3 性能监控

```python
import time
import psutil

def monitor_performance(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        # CPU/内存基线
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        # CPU/内存使用
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        print(f"📊 性能统计:")
        print(f"   - 耗时: {elapsed:.2f}s")
        print(f"   - CPU: {cpu_after:.1f}%")
        print(f"   - 内存: {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")
        
        return result
    return wrapper

@monitor_performance
async def run_research():
    return await agent.research_pipeline("test", data, max_iterations=5)
```

---

## 4. 配置最佳实践

### 4.1 开发环境配置

```python
# config_dev.yaml
dev_config = {
    # 使用更快更便宜的模型
    'llm_model': 'gpt-3.5-turbo',
    'llm_api_key': os.getenv('OPENAI_API_KEY'),
    'llm_provider': 'openai',
    
    # 较少的迭代
    'max_iterations': 3,
    
    # 较高的温度 (更多样性)
    'llm_temperature': 0.7,
    
    # 本地工作空间
    'workspace_path': './dev_logs',
    'qlib_data_path': './dev_data',
    
    # 启用详细日志
    'log_level': 'DEBUG'
}
```

### 4.2 生产环境配置

```python
# config_prod.yaml
prod_config = {
    # 使用最好的模型
    'llm_model': 'gpt-4-turbo',
    'llm_api_key': os.getenv('OPENAI_API_KEY'),
    'llm_provider': 'openai',
    
    # 标准迭代次数
    'max_iterations': 10,
    
    # 较低的温度 (更确定性)
    'llm_temperature': 0.5,
    
    # 生产路径
    'workspace_path': '/var/logs/rdagent',
    'qlib_data_path': '/data/qlib',
    
    # 正常日志级别
    'log_level': 'INFO',
    
    # 启用所有功能
    'enable_filestorage': True,
    'enable_caching': True
}
```

### 4.3 本地模型配置 (推荐)

```python
# 使用 vllm 部署本地模型
local_config = {
    'llm_model': 'Qwen/Qwen-14B-Chat',
    'llm_provider': 'openai',  # vllm 兼容 OpenAI API
    'llm_base_url': 'http://localhost:8000/v1',
    'llm_api_key': 'EMPTY',  # 本地不需要
    
    'max_iterations': 10,
    'llm_temperature': 0.6
}

# 优势:
# - 无网络延迟
# - 无成本
# - 数据隐私
# - 可定制
```

---

## 5. 诊断清单

### 5.1 启动前检查

**环境检查**:
```bash
# 1. 检查 Python 版本
python --version  # 应该 >= 3.8

# 2. 检查依赖
pip list | grep -E "rdagent|pandas|numpy"

# 3. 检查环境变量
echo $OPENAI_API_KEY  # Linux/Mac
echo $env:OPENAI_API_KEY  # Windows
```

**配置检查**:
- [ ] 环境变量已设置 (OPENAI_API_KEY)
- [ ] Qlib 数据已初始化
- [ ] 工作空间路径存在且可写
- [ ] 所有依赖包已安装

```python
# 自动化检查脚本
import os
from pathlib import Path

def pre_flight_check():
    """启动前检查"""
    checks = []
    
    # 1. API Key
    if os.getenv('OPENAI_API_KEY'):
        checks.append("✅ API Key 已设置")
    else:
        checks.append("❌ API Key 未设置")
    
    # 2. 工作空间
    workspace = Path('./logs/rdagent')
    if workspace.exists() and workspace.is_dir():
        checks.append("✅ 工作空间存在")
    else:
        checks.append("⚠️ 工作空间不存在,将自动创建")
        workspace.mkdir(parents=True, exist_ok=True)
    
    # 3. 依赖包
    try:
        import pandas
        import numpy
        checks.append("✅ 依赖包已安装")
    except ImportError as e:
        checks.append(f"❌ 缺少依赖: {e}")
    
    # 输出结果
    for check in checks:
        print(check)
    
    return all('✅' in c for c in checks)

if __name__ == '__main__':
    if pre_flight_check():
        print("\n🚀 所有检查通过,可以启动!")
    else:
        print("\n⚠️ 存在问题,请先解决")
```

### 5.2 运行时检查

**性能监控**:
- [ ] CPU 使用率 < 80%
- [ ] 内存使用 < 8GB
- [ ] 磁盘空间 > 10GB
- [ ] 网络连接正常

```python
import psutil

def runtime_check():
    """运行时检查"""
    # CPU
    cpu = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu}% {'✅' if cpu < 80 else '⚠️'}")
    
    # 内存
    mem = psutil.virtual_memory()
    mem_gb = mem.used / 1024 / 1024 / 1024
    print(f"内存: {mem_gb:.1f}GB / {mem.total/1024/1024/1024:.1f}GB "
          f"({'✅' if mem_gb < 8 else '⚠️'})")
    
    # 磁盘
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / 1024 / 1024 / 1024
    print(f"磁盘: {disk_gb:.1f}GB 可用 "
          f"({'✅' if disk_gb > 10 else '⚠️'})")
```

### 5.3 结果验证

**输出检查**:
- [ ] 因子数量 > 0
- [ ] 性能指标合理 (IC > 0.02, IR > 0.5)
- [ ] FileStorage 有记录
- [ ] 无异常错误

```python
def validate_results(results):
    """验证研究结果"""
    issues = []
    
    # 1. 因子数量
    if len(results['factors']) == 0:
        issues.append("❌ 未发现任何因子")
    else:
        print(f"✅ 发现 {len(results['factors'])} 个因子")
    
    # 2. 性能指标
    for factor in results['factors']:
        ic = factor.performance.get('ic', 0)
        if ic < 0.02:
            issues.append(f"⚠️ 因子 {factor.name} IC过低: {ic:.4f}")
        else:
            print(f"✅ 因子 {factor.name} IC正常: {ic:.4f}")
    
    # 3. 最佳方案
    if results.get('best_solution'):
        print("✅ 找到最佳方案")
    else:
        issues.append("⚠️ 未找到最佳方案")
    
    return len(issues) == 0, issues
```

---

## 6. 常见问题 FAQ

### Q1: 如何加速因子发现?
**A**: 
1. 减少 `max_iterations` (3-5即可)
2. 使用 `gpt-3.5-turbo` 而非 `gpt-4`
3. 并行执行多个任务
4. 使用本地模型 (vllm)

### Q2: FileStorage 日志在哪里?
**A**: 默认在 `workspace_path` 目录下:
```
./logs/rdagent/
├── experiments/
│   ├── exp_001.pkl
│   └── exp_002.pkl
└── metrics/
    └── summary_001.json
```

### Q3: 如何从历史恢复?
**A**: 使用多级兜底:
```python
factors = agent.load_factors_with_fallback(
    workspace_path='./logs/rdagent',
    n_factors=10
)
```

### Q4: Windows 上超时不生效?
**A**: Phase 3.1 将添加 Windows 超时支持。当前请:
1. 使用 Linux/Mac (推荐)
2. 手动监控并终止长时间运行的进程
3. 减小 `max_iterations`

### Q5: 如何调试代码沙盒问题?
**A**: 启用详细日志:
```python
import logging
logging.getLogger('rd_agent.code_sandbox').setLevel(logging.DEBUG)
```

---

## 7. 获取帮助

### 日志收集

出现问题时,请收集以下信息:

```python
# collect_debug_info.py
import sys
import platform
import os

print("=== 系统信息 ===")
print(f"Python: {sys.version}")
print(f"平台: {platform.platform()}")
print(f"工作目录: {os.getcwd()}")

print("\n=== 环境变量 ===")
print(f"OPENAI_API_KEY: {'已设置' if os.getenv('OPENAI_API_KEY') else '未设置'}")

print("\n=== 依赖版本 ===")
import pandas
import numpy
print(f"pandas: {pandas.__version__}")
print(f"numpy: {numpy.__version__}")

print("\n=== 配置信息 ===")
print(f"Workspace: {config.get('workspace_path')}")
print(f"Model: {config.get('llm_model')}")
print(f"Max iterations: {config.get('max_iterations')}")
```

### 联系支持

- 📧 Email: support@example.com
- 💬 Issues: https://github.com/example/rdagent/issues
- 📖 文档: https://rdagent.readthedocs.io

---

**文档版本**: 1.0  
**最后更新**: 2024-11-08  
**维护者**: AI Agent Team
