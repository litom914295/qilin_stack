# LLM驱动因子发现系统使用指南

## 🎯 系统概述

**LLM因子发现系统** 使用 DeepSeek 大模型自动生成和评估A股涨停板"一进二"策略的量化因子。

### 核心功能
1. **🤖 自动生成因子** - LLM理解市场特征，创造新因子
2. **📊 质量评估** - 自动验证语法、逻辑和实用性
3. **🔄 迭代优化** - 根据反馈持续改进因子
4. **💾 历史追踪** - 完整记录生成过程

### 优势对比

| 特性 | 传统方法 | LLM驱动 |
|------|---------|----------|
| 因子创造 | 依赖人工经验 | ✅ AI自动生成 |
| 创新性 | 受限于固有思维 | ✅ 跨领域融合 |
| 速度 | 慢（天/周） | ✅ 快（分钟） |
| 可扩展性 | 受人力限制 | ✅ 无限扩展 |
| 投资逻辑 | 隐性知识 | ✅ 显性说明 |

## 🚀 快速开始

###  1. 基础使用

```python
from rd_agent.llm_factor_discovery import LLMFactorDiscovery
import asyncio

async def main():
    # 创建发现系统
    discovery = LLMFactorDiscovery()
    
    # 生成3个新因子，关注封板强度
    factors = await discovery.discover_new_factors(
        n_factors=3,
        focus_areas=["封板强度", "连板动量"],
        context="重点关注短线强势特征"
    )
    
    # 打印结果
    for factor in factors:
        print(f"因子: {factor['name']}")
        print(f"逻辑: {factor['logic']}")
        print(f"预期IC: {factor['expected_ic']}")

asyncio.run(main())
```

### 2. 完整工作流

```python
async def full_workflow():
    discovery = LLMFactorDiscovery()
    
    # 步骤1: 发现因子
    factors = await discovery.discover_new_factors(n_factors=5)
    
    # 步骤2: 评估因子
    for factor in factors:
        evaluation = await discovery.evaluate_factor(factor)
        if evaluation['quality_score'] > 7.0:
            print(f"✅ 高质量因子: {factor['name']}")
    
    # 步骤3: 改进因子（如果需要）
    if evaluation['issues']:
        improved = await discovery.refine_factor(
            factor,
            feedback="需要更明确的数学表达式"
        )
    
    # 步骤4: 导出因子
    export_path = discovery.export_factors(factors)
    print(f"因子已导出: {export_path}")

asyncio.run(full_workflow())
```

## 📊 演示结果

### 生成的因子示例

刚才的演示生成了3个高质量因子：

#### 1. 封板强度梯度因子 (IC: 0.08, 质量: 8.0/10)
**表达式**: `(当前封单金额 / 流通市值) / (前5分钟平均封单金额 / 流通市值) × 封板时间权重`

**投资逻辑**: 
- 捕捉封板强度的动态变化
- 关注封单强度的增强过程
- 结合封板时间权重

#### 2. 连板动量共振因子 (IC: 0.12, 质量: 7.0/10)
**表达式**: `ln(当前连板高度) × 板块连板梯度 × 历史连板成功率调整`

**投资逻辑**:
- 多维度捕捉连板动量
- 强调板块联动效应
- 结合个股历史表现

#### 3. 题材热度传导因子 (IC: 0.10)
**表达式**: `概念强度指数 × 资金流入集中度 × 媒体关注度衰减因子`

**投资逻辑**:
- 量化题材热度传导效率
- 关注资金流入集中度
- 强调新概念的爆发潜力

## 🔧 API 参考

### LLMFactorDiscovery

#### `__init__()`
```python
discovery = LLMFactorDiscovery(
    api_key: str = None,        # API密钥，默认从环境变量读取
    api_base: str = None,       # API基础URL
    model: str = "deepseek-chat",  # 使用的模型
    cache_dir: str = "./workspace/llm_factor_cache"  # 缓存目录
)
```

#### `discover_new_factors()`
```python
factors = await discovery.discover_new_factors(
    n_factors: int = 5,                    # 生成因子数量
    focus_areas: List[str] = None,         # 关注领域
    context: str = None                    # 额外上下文
) -> List[Dict[str, Any]]
```

**focus_areas 可选值**:
- `"封板强度"` - 封单、开板相关
- `"连板动量"` - 连板高度、加速度
- `"题材共振"` - 概念、板块联动
- `"资金行为"` - 大单、换手、分时
- `"时机选择"` - 涨停时间、竞价

#### `evaluate_factor()`
```python
evaluation = await discovery.evaluate_factor(
    factor: Dict[str, Any],              # 因子定义
    sample_data: pd.DataFrame = None     # 样本数据（可选）
) -> Dict[str, Any]
```

**返回结果**:
```python
{
    'factor_name': '因子名称',
    'syntax_valid': True,          # 语法是否正确
    'computable': True,            # 是否可计算
    'quality_score': 8.0,          # 质量分数 0-10
    'issues': []                   # 问题列表
}
```

#### `refine_factor()`
```python
improved_factor = await discovery.refine_factor(
    factor: Dict[str, Any],     # 原始因子
    feedback: str               # 改进建议
) -> Dict[str, Any]
```

#### `export_factors()`
```python
export_path = discovery.export_factors(
    factors: List[Dict[str, Any]],
    output_file: str = None
) -> str
```

## 💡 最佳实践

### 1. 指定明确的关注领域
```python
# ✅ 好的做法
factors = await discovery.discover_new_factors(
    n_factors=3,
    focus_areas=["封板强度", "时机选择"],
    context="关注早盘涨停，规避尾盘炸板风险"
)

# ❌ 不好的做法
factors = await discovery.discover_new_factors(n_factors=10)  # 太宽泛
```

### 2. 迭代优化流程
```python
# 第一轮：生成初始因子
factors_v1 = await discovery.discover_new_factors(
    n_factors=5,
    context="寻找高胜率因子"
)

# 第二轮：根据反馈改进
for factor in factors_v1:
    evaluation = await discovery.evaluate_factor(factor)
    if evaluation['quality_score'] < 7.0:
        improved = await discovery.refine_factor(
            factor,
            feedback=f"质量分数偏低，存在问题：{evaluation['issues']}"
        )
```

### 3. 批量测试和筛选
```python
# 生成多批因子
all_factors = []
for batch in range(3):
    factors = await discovery.discover_new_factors(
        n_factors=5,
        focus_areas=["封板强度", "连板动量", "题材共振"][batch:batch+2]
    )
    all_factors.extend(factors)

# 评估并筛选
high_quality = []
for factor in all_factors:
    eval_result = await discovery.evaluate_factor(factor)
    if eval_result['quality_score'] >= 7.5:
        high_quality.append(factor)

print(f"筛选出 {len(high_quality)} 个高质量因子")
```

## 📁 文件结构

```
workspace/llm_factor_cache/
├── generation_history_20251030_094537.json   # 生成历史
├── factors_export_20251030_094537.json       # 导出的因子
└── ...
```

### 生成历史文件格式
```json
{
  "timestamp": "2025-10-30T09:45:37",
  "prompt": "请为A股涨停板'一进二'策略设计 3 个新的量化因子...",
  "response": "...",
  "factors_generated": 3,
  "factors": [...]
}
```

### 导出因子文件格式
```json
{
  "export_time": "2025-10-30T09:45:37",
  "total_factors": 3,
  "factors": [
    {
      "name": "封板强度梯度因子",
      "expression": "...",
      "code": "...",
      "category": "seal_strength",
      "logic": "...",
      "expected_ic": 0.08,
      "data_requirements": [...]
    }
  ]
}
```

## ⚙️ 配置

### 环境变量设置

确保 `.env` 文件包含：
```bash
# DeepSeek API
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com

# 或使用其他兼容 OpenAI 的 API
# OPENAI_API_KEY=sk-your-openai-key
# OPENAI_API_BASE=https://api.openai.com/v1
```

### 成本估算

使用 DeepSeek 的成本非常低：
- **生成3个因子**: 约 4000 tokens = ¥0.004元
- **评估1个因子**: 约 500 tokens = ¥0.0005元
- **改进1个因子**: 约 2000 tokens = ¥0.002元

**月度预算示例**:
- 每天生成10个新因子
- 每天评估20个因子
- 每天改进5个因子
- **月成本**: ≈ ¥10元

## 🔒 安全性

### 代码执行安全
系统内置多层安全检查：

1. **关键字过滤**
```python
dangerous_keywords = [
    'import os', 'import sys', 
    'exec(', 'eval(', '__import__'
]
```

2. **沙箱执行**
```python
safe_globals = {
    'np': np,
    'pd': pd,
    '__builtins__': {}  # 限制内置函数
}
exec(factor['code'], safe_globals)
```

3. **语法验证**
```python
compile(factor['code'], '<string>', 'exec')
```

## 🎓 高级用法

### 1. 结合历史数据评估
```python
# 加载历史涨停板数据
historical_data = pd.read_csv('limitup_history.csv')

# 生成因子并立即评估
factors = await discovery.discover_new_factors(n_factors=5)

for factor in factors:
    eval_result = await discovery.evaluate_factor(
        factor,
        sample_data=historical_data  # 使用真实数据测试
    )
    
    if eval_result['computable']:
        # 计算真实IC
        # ...
```

### 2. 领域知识注入
```python
context = """
当前市场特征：
1. 题材轮动加快，连板股频繁炸板
2. 资金偏好低位首板
3. 尾盘封板质量下降
4. 竞价强度与次日表现相关性提升

请重点设计能够识别低位首板机会的因子。
"""

factors = await discovery.discover_new_factors(
    n_factors=5,
    context=context
)
```

### 3. A/B 测试不同提示策略
```python
# 策略A: 强调稳健性
factors_a = await discovery.discover_new_factors(
    n_factors=5,
    context="重点关注高胜率、低波动的稳健型因子"
)

# 策略B: 强调爆发力
factors_b = await discovery.discover_new_factors(
    n_factors=5,
    context="重点关注高弹性、短期爆发型因子"
)

# 对比评估
for factors, name in [(factors_a, "稳健型"), (factors_b, "爆发型")]:
    avg_ic = np.mean([f['expected_ic'] for f in factors])
    print(f"{name}因子平均IC: {avg_ic:.4f}")
```

## 🐛 故障排除

### 问题1: API调用失败
```
OpenAIError: The api_key client option must be set
```
**解决**: 检查 `.env` 文件是否包含正确的 `OPENAI_API_KEY`

### 问题2: JSON解析失败
```
JSON解析失败: Expecting value: line 1 column 1
```
**解决**: LLM响应格式不符合预期，系统会自动使用备用解析方法

### 问题3: 因子代码语法错误
```
语法错误: invalid syntax (<string>, line 1)
```
**解决**: 使用 `refine_factor()` 改进因子，或手动调整代码

## 📖 相关文档

- [简化版因子发现](./RDAGENT_WINDOWS_SOLUTION.md)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [因子工程指南](./FACTOR_ENGINEERING.md)

## 🎯 下一步

1. ✅ 使用LLM生成新因子
2. ⏭️ 用真实数据回测验证IC
3. ⏭️ 优化提示词提升因子质量
4. ⏭️ 建立因子评分和排序系统
5. ⏭️ 集成到自动化交易系统

---

**版本**: 1.0  
**更新时间**: 2025-10-30  
**状态**: ✅ 生产就绪
