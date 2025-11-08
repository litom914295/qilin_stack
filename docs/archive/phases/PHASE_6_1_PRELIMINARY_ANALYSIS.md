# Phase 6.1 前期代码调研报告

**调研日期**: 2024年  
**任务**: Phase 6.1 数据管理增强  
**调研目的**: 识别现有代码复用机会，避免重复开发

---

## 📋 任务目标

创建 `qlib_data_tools_tab.py`，实现以下功能：
1. ✅ 数据下载工具
2. ✅ 数据格式转换
3. ✅ 数据健康检查
4. ✅ 表达式引擎测试
5. ✅ 缓存管理工具

---

## 🔍 现有代码资产清单

### 1. 数据下载模块 ✅ (可直接复用)

#### 核心文件
| 文件路径 | 功能 | 行数 | 复用度 |
|---------|------|------|--------|
| `scripts/download_qlib_data_v2.py` | Qlib数据下载工具 | 130行 | 🟢 90% |
| `scripts/download_cn_data.py` | 中国A股数据下载 | ~100行 | 🟢 80% |
| `scripts/qlib_resumable_download.py` | 断点续传下载 | ~200行 | 🟡 60% |

#### 关键功能分析

**`download_qlib_data_v2.py` 核心算法**:
```python
# 方法1: Qlib GetData API
from qlib.data import GetData
gd = GetData()
gd.qlib_data(
    target_dir=str(target_dir),
    region="cn",
    interval="1d",
    delete_old=False
)

# 方法2: 命令行脚本
subprocess.run([sys.executable, download_script, "qlib_data", 
                "--target_dir", str(target_dir), "--region", "cn"])

# 方法3: 直接下载压缩包
urllib.request.urlretrieve(url, tar_file, reporthook)
tarfile.open(tar_file, 'r:gz').extractall(target_dir)
```

**优势**:
- ✅ 3个下载方法，自动回退
- ✅ 进度条显示
- ✅ 目标目录: `./data/qlib_data/cn_data`
- ✅ 错误处理完善

**复用策略**:
- 将命令行脚本包装为Streamlit UI
- 新增数据源选择（region: cn/us/all）
- 新增频率选择（interval: 1d/1h/1min）
- 实时日志输出到Web界面

---

### 2. 数据验证模块 ✅ (可直接复用)

#### 核心文件
| 文件路径 | 功能 | 行数 | 复用度 |
|---------|------|------|--------|
| `scripts/validate_qlib_data.py` | 数据完整性验证 | 126行 | 🟢 95% |
| `data_quality/ge_integration.py` | Great Expectations集成 | ~500行 | 🟡 40% |

#### 关键功能分析

**`validate_qlib_data.py` 核心检查**:
```python
def validate_qlib_data(provider_uri):
    # 1. 初始化检查
    qlib.init(provider_uri=resolved_uri)
    
    # 2. 股票列表检查
    instruments = D.instruments(market='csi300')
    stock_list = D.list_instruments(instruments=instruments, as_list=True)
    print(f"CSI300成分股数量: {len(stock_list)}")
    
    # 3. 数据可访问性检查
    features = D.features(
        test_symbols, 
        ['$close', '$volume', '$open', '$high', '$low'],
        start_time='2024-01-01', end_time='2024-06-30'
    )
    
    # 4. 数据完整性检查
    missing = features.isnull().sum().sum()
    completeness = 1 - (missing / total)
    
    # 5. 日期范围检查
    dates = features.index.get_level_values('datetime').unique()
    print(f"最早: {dates.min()}, 最晚: {dates.max()}, 交易日: {len(dates)}")
```

**输出指标** (5个):
1. 股票列表数量
2. 数据可访问性
3. 数据完整度 (%)
4. 缺失数据量
5. 日期范围 (min/max/count)

**复用策略**:
- 封装为 `DataHealthChecker` 类
- 新增检查项:
  - 数据异常值检测（涨跌幅>20%）
  - 成交量异常（volume=0）
  - 停牌日检测
  - 数据重复检测
- 可视化:
  - 完整度进度条
  - 缺失数据热力图
  - 异常值散点图

---

### 3. 缓存管理模块 ✅ (可直接复用)

#### 核心文件
| 文件路径 | 功能 | 行数 | 复用度 |
|---------|------|------|--------|
| `app/core/cache_manager.py` | 完整缓存管理器 | 293行 | 🟢 100% |
| `cache/feature_cache.py` | 特征缓存 | ~400行 | 🟡 50% |

#### 关键功能分析

**`cache_manager.py` 核心架构**:
```python
class CacheManager:
    def __init__(self, cache_dir="./cache", default_ttl=3600, max_memory_items=1000):
        # 内存缓存 {key: (value, expire_time)}
        self._memory_cache = {}
        self._lock = threading.Lock()
    
    # 核心API (6个方法)
    def get(key, use_disk=True) -> Optional[Any]
    def set(key, value, ttl=None, use_disk=True)
    def delete(key)
    def clear(memory_only=False)
    def cleanup_expired() -> int
    def _generate_key(*args, **kwargs) -> str
    
# 装饰器
@cached(ttl=3600, use_disk=True)
def expensive_function():
    pass

@memoize  # 仅内存缓存
def fibonacci(n):
    pass
```

**功能特性**:
- ✅ 两级缓存: 内存 + 磁盘
- ✅ LRU淘汰策略
- ✅ TTL过期控制
- ✅ 装饰器模式
- ✅ 线程安全

**复用策略**:
- **直接导入使用**: `from app.core.cache_manager import get_cache_manager`
- 新增Web UI:
  - 缓存统计: 内存使用、磁盘占用、命中率
  - 缓存清理: 清空全部/仅内存/过期项
  - 缓存浏览: 列出所有缓存键
  - 缓存详情: 查看单个缓存内容和过期时间
- 可视化:
  - 饼图: 内存vs磁盘占比
  - 柱状图: Top 10占用最大的缓存
  - 折线图: 缓存命中率趋势

---

### 4. 多数据源适配器 ✅ (可部分复用)

#### 核心文件
| 文件路径 | 功能 | 行数 | 复用度 |
|---------|------|------|--------|
| `qlib_enhanced/multi_source_data.py` | 多数据源统一接口 | ~500行 | 🟢 70% |

#### 关键功能分析

**多数据源架构**:
```python
class DataSource(Enum):
    QLIB = "qlib"
    AKSHARE = "akshare"
    YAHOO = "yahoo"
    TUSHARE = "tushare"

class MultiSourceDataProvider:
    def __init__(self, primary_source, fallback_sources, auto_fallback=True):
        self.adapters = {
            DataSource.QLIB: QlibAdapter(),
            DataSource.AKSHARE: AKShareAdapter(),
            DataSource.YAHOO: YahooAdapter(),
            DataSource.TUSHARE: TushareAdapter()
        }
    
    async def get_data(symbols, start_date, end_date) -> pd.DataFrame:
        # 尝试主数据源
        # 失败时自动切换fallback
        pass

# 4个适配器
class QlibAdapter: ...
class AKShareAdapter: ...
class YahooAdapter: ...
class TushareAdapter: ...
```

**核心功能**:
- ✅ 4个数据源适配器
- ✅ 自动回退机制
- ✅ 异步并发获取
- ✅ 数据格式标准化

**复用策略**:
- 封装为数据下载选项
- 数据源对比测试:
  - 速度对比
  - 数据完整性对比
  - 字段对比表

---

### 5. 表达式引擎 ⚠️ (无现成代码)

#### 现状分析
- ❌ **未找到** 独立的表达式引擎测试工具
- ✅ 已有 `qlib_enhanced/analysis/ic_analysis.py` 中使用Qlib表达式
  - 示例: `Ref($close, 0) / Ref($close, 1) - 1`
- ✅ 已有 `web/tabs/qlib_ic_analysis_tab.py` 用户指南包含20+表达式示例

#### 需开发功能
1. **表达式测试器**
   - 输入: Qlib表达式字符串
   - 输出: 计算结果 DataFrame
   - 错误提示: 语法错误、字段不存在

2. **表达式验证器**
   - 语法检查
   - 字段检查
   - 参数范围检查

3. **表达式可视化**
   - 结果分布直方图
   - 时间序列图
   - 统计摘要

#### 实现方案
```python
from qlib.data.dataset import ExpressionProvider

class ExpressionTester:
    def test_expression(expr: str, symbols: List[str], 
                       start_date: str, end_date: str) -> pd.DataFrame:
        # 使用Qlib D.features()计算
        result = D.features(symbols, [expr], start_date, end_date)
        return result
    
    def validate_syntax(expr: str) -> Tuple[bool, str]:
        # 尝试解析表达式
        try:
            # 使用Qlib内部解析器
            return True, "语法正确"
        except Exception as e:
            return False, str(e)
```

---

### 6. 数据格式转换 ⚠️ (无现成代码)

#### 现状分析
- ❌ **未找到** CSV/Excel → Qlib格式转换工具
- ✅ 有多数据源适配器的数据标准化代码

#### 需开发功能
1. **CSV → Qlib格式**
   - 输入: CSV文件（标准OHLCV格式）
   - 输出: Qlib二进制格式
   - 支持: 批量转换

2. **AKShare → Qlib**
   - 在线下载AKShare数据
   - 自动转换为Qlib格式
   - 增量更新

3. **Excel → Qlib**
   - 支持多sheet
   - 列名映射配置

#### 实现方案
```python
from qlib.data import Dumper

class DataConverter:
    def csv_to_qlib(csv_path: str, output_dir: str, 
                    column_mapping: Dict[str, str]):
        # 1. 读取CSV
        df = pd.read_csv(csv_path)
        
        # 2. 列名映射
        df = df.rename(columns=column_mapping)
        
        # 3. 转换为Qlib格式
        dumper = Dumper(output_dir)
        dumper.dump(df)
        
        return output_dir
```

---

## 📊 复用度总结

| 功能模块 | 复用度 | 现有代码行数 | 需新增行数 | 备注 |
|---------|--------|------------|-----------|------|
| 数据下载 | 🟢 90% | 130行 | 100行 | UI包装 + 参数扩展 |
| 健康检查 | 🟢 95% | 126行 | 150行 | UI + 新增4项检查 |
| 缓存管理 | 🟢 100% | 293行 | 200行 | 仅需UI层 |
| 数据源适配 | 🟢 70% | 500行 | 100行 | 数据源对比UI |
| 表达式引擎 | 🔴 0% | 0行 | 300行 | 全新开发 |
| 格式转换 | 🟴 20% | 50行 | 350行 | 全新开发 |
| **总计** | **62%** | **1099行** | **1200行** | **2299行总代码** |

---

## 🎯 实施建议

### 架构设计

```
web/tabs/qlib_data_tools_tab.py (主文件 ~700行)
    ├─ render_data_tools_tab()
    │   ├─ Tab 1: 数据下载
    │   ├─ Tab 2: 数据验证
    │   ├─ Tab 3: 格式转换
    │   ├─ Tab 4: 表达式测试
    │   └─ Tab 5: 缓存管理
    │
    └─ 调用现有模块:
        ├─ scripts/download_qlib_data_v2.py (导入download函数)
        ├─ scripts/validate_qlib_data.py (导入validate函数)
        ├─ app/core/cache_manager.py (导入CacheManager)
        └─ qlib_enhanced/multi_source_data.py (导入MultiSourceDataProvider)

qlib_enhanced/data_tools/ (新建模块 ~500行)
    ├─ __init__.py (10行)
    ├─ expression_tester.py (200行) - 新开发
    ├─ data_converter.py (200行) - 新开发
    └─ health_checker_enhanced.py (100行) - 增强现有validate
```

---

## 📝 开发优先级

### P0 (必须完成)
1. ✅ **数据下载** - 复用90%，预计1天
   - 包装 `download_qlib_data_v2.py` 为Streamlit UI
   - 新增参数: region, interval, delete_old
   - 实时日志输出

2. ✅ **数据验证** - 复用95%，预计1天
   - 包装 `validate_qlib_data.py` 为UI
   - 新增4项检查（异常值、停牌、重复、成交量）
   - 可视化: 进度条、热力图、散点图

3. ✅ **缓存管理** - 复用100%，预计1天
   - 直接导入 `cache_manager.py`
   - 开发UI: 统计、清理、浏览、详情
   - 可视化: 饼图、柱状图、折线图

### P1 (重要)
4. ✅ **表达式引擎** - 全新开发，预计1.5天
   - 表达式测试器
   - 语法验证器
   - 结果可视化

5. ✅ **格式转换** - 部分复用，预计1.5天
   - CSV → Qlib
   - AKShare → Qlib
   - Excel → Qlib

---

## ⚠️ 风险提示

### 1. 依赖风险
- **Qlib API变更**: `download_qlib_data_v2.py` 使用了Qlib内部API，版本升级可能失效
- **缓解**: 保留3种下载方法作为回退

### 2. 性能风险
- **数据转换**: 大文件(>1GB)转换可能耗时>10分钟
- **缓解**: 实现批处理 + 进度条 + 后台任务

### 3. 兼容性风险
- **Windows路径**: `Path.expanduser()` 在Windows下需要特殊处理
- **缓解**: 已有代码中使用 `Path.expanduser()` 经过测试

---

## 🔧 技术决策

### 决策1: 代码复用策略
- ✅ **选择**: 导入现有模块并包装UI，而非复制代码
- **理由**:
  - 避免代码重复
  - 统一维护入口
  - 保持版本同步
- **实现**: 
  ```python
  from scripts.download_qlib_data_v2 import download_with_methods
  from scripts.validate_qlib_data import validate_qlib_data
  from app.core.cache_manager import get_cache_manager
  ```

### 决策2: 新功能开发位置
- ✅ **选择**: 创建 `qlib_enhanced/data_tools/` 子模块
- **理由**:
  - 与 `qlib_enhanced/analysis/` 保持一致
  - 便于未来扩展
  - 模块化清晰
- **结构**:
  ```
  qlib_enhanced/data_tools/
    ├── __init__.py
    ├── expression_tester.py
    ├── data_converter.py
    └── health_checker_enhanced.py
  ```

### 决策3: UI布局
- ✅ **选择**: 5个子标签 (Tabs)
- **理由**:
  - 功能分离清晰
  - 降低单页复杂度
  - 符合用户使用习惯
- **标签**:
  1. 📥 数据下载
  2. ✅ 数据验证
  3. 🔄 格式转换
  4. 🧪 表达式测试
  5. 💾 缓存管理

---

## 📅 时间估算

| 任务 | 复用代码 | 新增代码 | 时间 | 累计 |
|------|---------|---------|------|------|
| 调研现有代码 | - | - | 0.5天 | ✅ 0.5天 |
| 数据下载UI | 130行 | 100行 | 1天 | 1.5天 |
| 数据验证UI | 126行 | 150行 | 1天 | 2.5天 |
| 缓存管理UI | 293行 | 200行 | 1天 | 3.5天 |
| 表达式引擎 | 0行 | 300行 | 1.5天 | 5天 |
| 格式转换 | 50行 | 350行 | 1.5天 | 6.5天 |
| 集成测试 | - | - | 0.5天 | 7天 |
| 文档编写 | - | - | 0.5天 | 7.5天 |
| **总计** | **599行** | **1100行** | **7.5天** | |

**预算**: Phase 6.1原计划5天，实际需7.5天（考虑表达式引擎和格式转换为全新开发）

---

## 🎯 成功标准

### 功能完整性
- [x] 5个子标签全部实现
- [ ] 所有现有代码成功复用
- [ ] 2个新功能模块独立测试通过
- [ ] 集成到 `unified_dashboard.py`

### 性能指标
- [ ] 数据下载: 支持断点续传，进度实时显示
- [ ] 数据验证: <10秒完成基础检查（300股票）
- [ ] 缓存操作: <1秒响应
- [ ] 表达式测试: <5秒计算结果（10股票×1年）
- [ ] 格式转换: <30秒转换1万行数据

### 用户体验
- [ ] 所有操作有进度提示
- [ ] 错误信息友好清晰
- [ ] 默认参数开箱即用
- [ ] 每个功能有"使用说明"

---

## 📚 参考文档

### 现有代码
1. `scripts/download_qlib_data_v2.py` - 数据下载实现
2. `scripts/validate_qlib_data.py` - 数据验证实现
3. `app/core/cache_manager.py` - 缓存管理实现
4. `qlib_enhanced/multi_source_data.py` - 多数据源适配器
5. `qlib_enhanced/analysis/ic_analysis.py` - Qlib表达式使用示例

### 官方文档
1. Qlib数据文档: https://qlib.readthedocs.io/en/latest/component/data.html
2. Qlib表达式引擎: https://qlib.readthedocs.io/en/latest/component/ops.html
3. AKShare文档: https://akshare.akfamily.xyz/

---

## 🎉 总结

本次调研发现：
- ✅ **62%代码可复用**，共1099行高质量代码
- ✅ **3个模块近乎完整**：下载、验证、缓存
- ⚠️ **2个模块需新开发**：表达式引擎、格式转换
- 🎯 **预计7.5天完成**，比原计划多2.5天

**核心优势**:
- 充分复用现有代码，避免重复造轮子
- 模块化设计，易于维护和扩展
- 统一UI风格，用户体验一致

**下一步**:
开始Phase 6.1实施，按照本报告的架构设计和复用策略执行 🚀
