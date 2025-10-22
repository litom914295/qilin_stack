# 测试文档

麒麟量化系统的测试套件，提供全面的单元测试、集成测试和性能测试。

## 📁 测试结构

```
tests/
├── conftest.py           # Pytest配置和通用fixtures
├── unit/                 # 单元测试
│   ├── test_agents.py
│   ├── test_mlops.py
│   └── test_monitoring.py
├── integration/          # 集成测试
├── performance/          # 性能测试
└── README.md
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行所有测试

```bash
# Linux/Mac
./run_tests.sh

# Windows
.\run_tests.ps1

# 或使用pytest直接运行
pytest
```

### 运行特定测试

```bash
# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/ -m integration

# MLOps测试
pytest tests/unit/test_mlops.py -m mlops

# 监控测试
pytest tests/unit/test_monitoring.py -m monitoring
```

## 🏷️ 测试标记

使用pytest标记来分类和选择测试：

```bash
# 单元测试
pytest -m unit

# 集成测试
pytest -m integration

# 性能测试
pytest -m performance

# 慢速测试
pytest -m slow

# MLOps测试
pytest -m mlops

# 排除慢速测试
pytest -m "not slow"
```

## 📊 覆盖率报告

### 生成覆盖率报告

```bash
# HTML报告
pytest --cov=app --cov-report=html

# 终端报告
pytest --cov=app --cov-report=term-missing

# XML报告(CI/CD)
pytest --cov=app --cov-report=xml
```

### 查看报告

```bash
# 打开HTML报告
open htmlcov/index.html  # Mac
start htmlcov/index.html # Windows
```

## 🎯 测试覆盖率目标

| 模块 | 目标覆盖率 | 当前覆盖率 |
|------|-----------|-----------|
| app/mlops | 90% | - |
| app/monitoring | 85% | - |
| app/core | 80% | - |
| app/agents | 80% | - |
| **总体** | **80%+** | - |

## 📝 编写测试

### 基础测试结构

```python
import pytest

@pytest.mark.unit
class TestMyModule:
    \"\"\"模块测试\"\"\"
    
    @pytest.fixture
    def sample_data(self):
        \"\"\"准备测试数据\"\"\"
        return {'key': 'value'}
    
    def test_basic_function(self, sample_data):
        \"\"\"测试基础功能\"\"\"
        result = my_function(sample_data)
        assert result is not None
```

### 异步测试

```python
@pytest.mark.asyncio
async def test_async_function():
    \"\"\"测试异步函数\"\"\"
    result = await async_function()
    assert result['status'] == 'success'
```

### 使用Fixtures

```python
def test_with_fixture(sample_ohlcv_data, sample_symbols):
    \"\"\"使用fixtures的测试\"\"\"
    assert len(sample_ohlcv_data) > 0
    assert len(sample_symbols) == 6
```

## 🔧 可用的Fixtures

### 数据Fixtures

- `sample_ohlcv_data`: OHLCV格式的市场数据
- `sample_tick_data`: Tick级别的数据
- `sample_symbols`: 股票代码列表
- `sample_model_data`: 模型训练数据
- `sample_portfolio`: 投资组合数据
- `sample_trades`: 成交记录

### 配置Fixtures

- `mock_config`: 模拟系统配置
- `temp_directory`: 临时目录
- `mlflow_tracking_uri`: MLflow URI
- `redis_client`: Mock Redis客户端

### 测试工具Fixtures

- `event_loop`: 异步事件循环
- `benchmark_data`: 性能测试数据
- `docker_services`: Docker服务检查

## 🔍 调试测试

### 显示print输出

```bash
pytest -s
```

### 详细输出

```bash
pytest -v
```

### 显示本地变量

```bash
pytest --showlocals
```

### 仅运行失败的测试

```bash
pytest --lf
```

### 进入调试器

```bash
pytest --pdb
```

## 📈 性能测试

### 运行性能测试

```bash
pytest tests/performance/ -m performance
```

### 查看最慢的测试

```bash
pytest --durations=10
```

## 🐛 常见问题

### 1. 导入错误

**问题**: `ModuleNotFoundError`

**解决**: 确保项目根目录在Python路径中

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### 2. 异步测试失败

**问题**: `RuntimeError: Event loop is closed`

**解决**: 使用`pytest-asyncio`并添加标记

```python
@pytest.mark.asyncio
async def test_async():
    ...
```

### 3. MLflow测试失败

**问题**: `ConnectionError`

**解决**: 使用本地file URI

```python
tracking_uri = f"file://{temp_directory}/mlruns"
```

### 4. 覆盖率不准确

**问题**: 某些文件未包含在覆盖率中

**解决**: 检查`.coveragerc`配置

```ini
[coverage:run]
source = app
omit = */tests/*
```

## 🔐 CI/CD集成

测试已集成到CI/CD流水线：

```yaml
# .github/workflows/ci-cd.yml
- name: Run tests
  run: |
    pytest --cov=app --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## 📚 参考资料

- [Pytest文档](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)

## 💡 最佳实践

1. **测试命名**: 使用描述性的测试名称
2. **测试隔离**: 每个测试应该独立运行
3. **使用Fixtures**: 重用测试数据和设置
4. **标记测试**: 使用pytest标记分类测试
5. **代码覆盖**: 目标80%+覆盖率
6. **测试文档**: 为复杂测试添加文档字符串
7. **持续测试**: 在CI/CD中自动运行测试

## 🆘 获取帮助

如有问题：

1. 查看测试日志: `logs/pytest.log`
2. 查看覆盖率报告: `htmlcov/index.html`
3. 运行单个测试调试: `pytest tests/path/to/test.py::test_name -v`
4. 查看fixture定义: `pytest --fixtures`
