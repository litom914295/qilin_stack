# RD-Agent兼容层架构设计

**任务**: P0-1.3  
**设计时间**: 2025-11-07  
**目标**: 设计兼容层保持现有API,内部调用官方RD-Agent

---

## 1. 设计目标

### 1.1 核心原则
- ✅ **API兼容**: 保持现有`RDAgent`类的API不变
- ✅ **透明迁移**: 上层代码无需修改
- ✅ **渐进式**: 支持新旧实现共存
- ✅ **可测试**: 每个组件独立可测试

### 1.2 非目标
- ❌ **不追求100%行为一致**: 允许内部实现差异(如缓存机制)
- ❌ **不保留所有自研功能**: 舍弃与官方冲突的自研扩展
- ❌ **不支持旧版本回退**: 迁移后不再维护自研代码

---

## 2. 架构设计

### 2.1 总体架构图

```
┌────────────────────────────────────────────────────────────────┐
│                    上层应用代码 (不变)                           │
│  - strategies/选股策略.py                                       │
│  - backtest/回测引擎.py                                         │
│  - research/研究脚本.py                                         │
└──────────────────────────┬─────────────────────────────────────┘
                           │ 调用
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              兼容层 (rd_agent/compat_wrapper.py)                │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │         RDAgentWrapper                           │          │
│  │  - __init__(config: Dict)                        │          │
│  │  - research_pipeline(...)  → Dict                │          │
│  │  - discover_factors(...)   → List[FactorDef]     │          │
│  │  - optimize_strategy(...)  → StrategyTemplate    │          │
│  └───────────────────┬──────────────────────────────┘          │
│                      │                                          │
│        ┌─────────────┴─────────────┐                           │
│        ▼                           ▼                           │
│  配置转换器                    结果转换器                       │
│  _ConfigAdapter               _ResultAdapter                   │
│  - dict → Settings            - Trace → Dict                   │
│  - dict → .env                - Exp → FactorDef                │
└────────────────────────┬───────────────────────────────────────┘
                         │ 调用
                         ▼
┌────────────────────────────────────────────────────────────────┐
│         官方集成层 (rd_agent/official_integration.py)           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │      OfficialRDAgentManager                      │          │
│  │  - get_factor_loop()    → FactorRDLoop           │          │
│  │  - get_model_loop()     → ModelRDLoop            │          │
│  │  - configure_llm()      (集成P0-2)               │          │
│  │  - configure_paths()    (集成P0-3)               │          │
│  └───────────────────┬──────────────────────────────┘          │
│                      │                                          │
└──────────────────────┼──────────────────────────────────────────┘
                       │ 调用
                       ▼
┌────────────────────────────────────────────────────────────────┐
│              官方RD-Agent (G:\test\RD-Agent\rdagent)            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ FactorRDLoop │  │ ModelRDLoop  │  │ RDLoop核心   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ HypothesisGen│  │ Developer    │  │ Feedback     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ Trace        │  │ KnowledgeBase│                           │
│  └──────────────┘  └──────────────┘                           │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 组件职责

| 组件 | 文件 | 职责 |
|-----|------|-----|
| **RDAgentWrapper** | `compat_wrapper.py` | 保持原有API,内部调用官方组件 |
| **OfficialRDAgentManager** | `official_integration.py` | 管理官方组件的初始化和配置 |
| **_ConfigAdapter** | `compat_wrapper.py` | Dict配置 → Pydantic Settings |
| **_ResultAdapter** | `compat_wrapper.py` | Trace/Experiment → Dict/FactorDef |

---

## 3. 配置转换设计

### 3.1 自研配置格式 (输入)

```python
# 当前自研代码使用的配置
config = {
    # LLM配置
    "llm_model": "gpt-4-turbo",
    "llm_api_key": "sk-xxx",
    "llm_base_url": "https://api.openai.com/v1",
    "llm_temperature": 0.7,
    
    # 路径配置
    "storage_path": "./knowledge_base",
    "qlib_data_path": "~/.qlib/qlib_data/cn_data",
    "tradingagents_path": "G:/test/tradingagents-cn-plus",
    
    # 研究配置
    "max_iterations": 10,
    "n_factors": 10,
    "backtest_start": "2020-01-01",
    "backtest_end": "2023-12-31",
    
    # 执行配置
    "use_docker": False,
    "parallel_workers": 4,
    "cache_enabled": True,
}
```

### 3.2 官方配置格式 (输出)

```python
# 官方RD-Agent使用的配置 (Pydantic Settings)
from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting

official_setting = FactorBasePropSetting(
    # 场景配置
    scen="rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario",
    
    # 组件配置
    hypothesis_gen="rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen",
    hypothesis2experiment="rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment",
    coder="rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER",
    runner="rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner",
    summarizer="rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback",
    
    # 执行配置
    evolving_n=10,  # 对应 max_iterations
)

# 环境变量配置 (.env)
# 官方RD-Agent通过环境变量读取LLM和路径配置
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4-turbo
LLM_TEMPERATURE=0.7

QLIB_DATA_PATH=~/.qlib/qlib_data/cn_data
RDAGENT_PATH=G:/test/RD-Agent
```

### 3.3 配置转换逻辑

```python
class _ConfigAdapter:
    """配置适配器: Dict → Pydantic Settings + .env"""
    
    @staticmethod
    def to_official_setting(config: Dict[str, Any]) -> FactorBasePropSetting:
        """转换为官方FactorBasePropSetting"""
        return FactorBasePropSetting(
            # 使用默认场景
            scen="rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario",
            
            # 使用默认组件 (通常不需要修改)
            hypothesis_gen="rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen",
            coder="rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER",
            runner="rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner",
            summarizer="rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback",
            
            # 映射执行配置
            evolving_n=config.get("max_iterations", 10),
        )
    
    @staticmethod
    def to_env_vars(config: Dict[str, Any]) -> Dict[str, str]:
        """转换为环境变量 (官方通过.env读取)"""
        env_vars = {}
        
        # LLM配置
        if "llm_model" in config:
            env_vars["LLM_MODEL"] = config["llm_model"]
        if "llm_api_key" in config:
            env_vars["OPENAI_API_KEY"] = config["llm_api_key"]
        if "llm_base_url" in config:
            env_vars["OPENAI_BASE_URL"] = config["llm_base_url"]
        if "llm_temperature" in config:
            env_vars["LLM_TEMPERATURE"] = str(config["llm_temperature"])
        
        # 路径配置
        if "qlib_data_path" in config:
            env_vars["QLIB_DATA_PATH"] = config["qlib_data_path"]
        
        return env_vars
    
    @staticmethod
    def apply_env_vars(env_vars: Dict[str, str]):
        """应用环境变量到os.environ"""
        import os
        for key, value in env_vars.items():
            os.environ[key] = value
```

---

## 4. 结果转换设计

### 4.1 官方结果格式 (输入)

```python
# 官方RD-Agent的结果格式
loop.trace = Trace(
    hist=[
        (Experiment1, Feedback1),  # 第1轮
        (Experiment2, Feedback2),  # 第2轮
        ...
    ],
    knowledge_base=KnowledgeBase(...)
)

# 获取SOTA实验
hypothesis, experiment = loop.trace.get_sota_hypothesis_and_experiment()
experiment.result = pd.DataFrame({
    "IC": 0.045,
    "1day.excess_return_with_cost.annualized_return": 0.18,
    "1day.excess_return_with_cost.max_drawdown": -0.12,
    ...
})
```

### 4.2 自研结果格式 (输出)

```python
# 当前自研代码期望的结果格式
results = {
    "topic": "A股动量因子研究",
    "hypotheses": [ResearchHypothesis(...)],
    "factors": [FactorDefinition(...)],
    "strategies": [StrategyTemplate(...)],
    "models": [],
    "best_solution": {
        "type": "factor",
        "solution": FactorDefinition(...),
        "performance": {"ic": 0.045, "ir": 1.2, ...}
    }
}
```

### 4.3 结果转换逻辑

```python
class _ResultAdapter:
    """结果适配器: Trace/Experiment → Dict/FactorDef"""
    
    @staticmethod
    def trace_to_results_dict(trace: Trace, topic: str) -> Dict[str, Any]:
        """Trace → research_pipeline返回的Dict"""
        results = {
            "topic": topic,
            "hypotheses": [],
            "factors": [],
            "strategies": [],
            "models": [],
            "best_solution": None
        }
        
        # 转换所有历史实验
        for exp, feedback in trace.hist:
            # 转换为ResearchHypothesis
            hypo = ResearchHypothesis(
                id=f"hypo_{id(exp)}",
                title=exp.hypothesis.hypothesis[:50],  # 截取前50字符
                description=exp.hypothesis.hypothesis,
                category="factor",
                confidence=0.8 if feedback.decision else 0.3,
                created_at=datetime.now(),
                status="validated" if feedback.decision else "rejected",
                results={"decision": feedback.decision}
            )
            results["hypotheses"].append(hypo)
            
            # 如果实验成功,转换为Factor
            if feedback.decision and exp.result is not None:
                factor = _ResultAdapter.exp_to_factor(exp)
                results["factors"].append(factor)
        
        # 选择最佳解决方案
        if results["factors"]:
            results["best_solution"] = {
                "type": "factor",
                "solution": results["factors"][-1],  # SOTA是最后一个
                "performance": results["factors"][-1].performance
            }
        
        return results
    
    @staticmethod
    def exp_to_factor(exp: Experiment) -> FactorDefinition:
        """Experiment → FactorDefinition"""
        # 提取因子代码
        factor_code = ""
        if exp.sub_workspace_list:
            factor_code = exp.sub_workspace_list[0].file_dict.get("factor.py", "")
        
        # 提取性能指标
        performance = {}
        if exp.result is not None:
            result_df = exp.result
            if "IC" in result_df.index:
                performance["ic"] = float(result_df.loc["IC", "0"])
            if "1day.excess_return_with_cost.annualized_return" in result_df.index:
                performance["annual_return"] = float(
                    result_df.loc["1day.excess_return_with_cost.annualized_return", "0"]
                )
        
        return FactorDefinition(
            name=f"factor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            expression=factor_code,
            description=exp.hypothesis.hypothesis,
            category="auto_generated",
            parameters={},
            performance=performance
        )
    
    @staticmethod
    def experiments_to_factors(trace: Trace, n_factors: int = 10) -> List[FactorDefinition]:
        """提取前N个有效因子"""
        factors = []
        for exp, feedback in trace.hist:
            if feedback.decision and exp.result is not None:
                factor = _ResultAdapter.exp_to_factor(exp)
                factors.append(factor)
                if len(factors) >= n_factors:
                    break
        return factors
```

---

## 5. API映射设计

### 5.1 `research_pipeline()` 映射

#### 5.1.1 自研API (保持不变)
```python
async def research_pipeline(
    self,
    research_topic: str,
    data: pd.DataFrame,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """完整的研究流程"""
    pass
```

#### 5.1.2 官方API调用
```python
# 官方RD-Agent执行
await self._factor_loop.run(
    step_n=None,
    loop_n=max_iterations,
    all_duration=None
)
```

#### 5.1.3 映射逻辑
```python
async def research_pipeline(
    self,
    research_topic: str,
    data: pd.DataFrame,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """兼容层实现"""
    
    # 1. 获取官方FactorLoop
    factor_loop = self._official_manager.get_factor_loop()
    
    # 2. 调用官方run方法
    await factor_loop.run(loop_n=max_iterations)
    
    # 3. 转换结果格式
    results = _ResultAdapter.trace_to_results_dict(
        factor_loop.trace,
        topic=research_topic
    )
    
    return results
```

### 5.2 `discover_factors()` 映射

#### 5.2.1 自研API (保持不变)
```python
async def discover_factors(
    self,
    data: pd.DataFrame,
    target: str = "returns",
    n_factors: int = 10
) -> List[FactorDefinition]:
    """自动发现因子"""
    pass
```

#### 5.2.2 映射逻辑
```python
async def discover_factors(
    self,
    data: pd.DataFrame,
    target: str = "returns",
    n_factors: int = 10
) -> List[FactorDefinition]:
    """兼容层实现"""
    
    # 1. 获取官方FactorLoop
    factor_loop = self._official_manager.get_factor_loop()
    
    # 2. 运行1-2轮发现新因子
    await factor_loop.run(loop_n=2)
    
    # 3. 提取因子
    factors = _ResultAdapter.experiments_to_factors(
        factor_loop.trace,
        n_factors=n_factors
    )
    
    return factors
```

### 5.3 `optimize_strategy()` 映射

#### 5.3.1 自研API (保持不变)
```python
async def optimize_strategy(
    self,
    strategy: StrategyTemplate,
    data: pd.DataFrame,
    n_trials: int = 100
) -> StrategyTemplate:
    """优化策略参数"""
    pass
```

#### 5.3.2 映射逻辑
```python
async def optimize_strategy(
    self,
    strategy: StrategyTemplate,
    data: pd.DataFrame,
    n_trials: int = 100
) -> StrategyTemplate:
    """兼容层实现"""
    
    # ModelLoop的映射逻辑
    # (暂时保留自研实现,因为ModelLoop主要用于模型优化而非策略优化)
    
    # 选项1: 保留自研的Optuna优化逻辑
    return await self._legacy_optimize_strategy(strategy, data, n_trials)
    
    # 选项2: 等待官方支持策略优化后再迁移
    # TODO: 研究官方是否有策略优化的场景
```

---

## 6. 数据流设计

### 6.1 初始化流程

```
用户创建RDAgent
    │
    ├─→ RDAgentWrapper.__init__(config)
    │       │
    │       ├─→ _ConfigAdapter.to_official_setting(config)
    │       │       └─→ 返回 FactorBasePropSetting
    │       │
    │       ├─→ _ConfigAdapter.to_env_vars(config)
    │       │       └─→ 设置环境变量 (LLM_MODEL, OPENAI_API_KEY, ...)
    │       │
    │       └─→ OfficialRDAgentManager.init(official_setting)
    │               ├─→ 配置LLM (使用P0-2的ProductionLLMManager)
    │               ├─→ 配置路径 (使用P0-3的PathConfig)
    │               └─→ 创建 FactorRDLoop, ModelRDLoop (懒加载)
    │
    └─→ 返回 RDAgentWrapper实例
```

### 6.2 研究流程

```
用户调用 research_pipeline()
    │
    ├─→ RDAgentWrapper.research_pipeline(topic, data, max_iterations)
    │       │
    │       ├─→ 获取 FactorRDLoop
    │       │       └─→ OfficialRDAgentManager.get_factor_loop()
    │       │               └─→ FactorRDLoop(FACTOR_PROP_SETTING)
    │       │
    │       ├─→ 运行官方循环
    │       │       └─→ await factor_loop.run(loop_n=max_iterations)
    │       │               ├─→ 提议阶段: HypothesisGen.gen()
    │       │               ├─→ 编码阶段: QlibFactorCoSTEER.develop()
    │       │               ├─→ 执行阶段: QlibFactorRunner.develop()
    │       │               └─→ 反馈阶段: QlibFactorExperiment2Feedback.generate_feedback()
    │       │
    │       └─→ 转换结果
    │               └─→ _ResultAdapter.trace_to_results_dict(trace, topic)
    │                       ├─→ 遍历 trace.hist
    │                       ├─→ 转换为 ResearchHypothesis
    │                       ├─→ 转换为 FactorDefinition
    │                       └─→ 选择 best_solution
    │
    └─→ 返回 Dict结果
```

---

## 7. 错误处理设计

### 7.1 配置错误

```python
class ConfigValidationError(Exception):
    """配置验证错误"""
    pass

class _ConfigAdapter:
    @staticmethod
    def validate_config(config: Dict[str, Any]):
        """验证配置完整性"""
        required_keys = ["llm_api_key", "qlib_data_path"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ConfigValidationError(f"Missing required config keys: {missing}")
```

### 7.2 官方组件错误

```python
class OfficialIntegrationError(Exception):
    """官方组件集成错误"""
    pass

class OfficialRDAgentManager:
    def get_factor_loop(self) -> FactorRDLoop:
        try:
            if self._factor_loop is None:
                self._factor_loop = FactorRDLoop(self._factor_setting)
            return self._factor_loop
        except Exception as e:
            raise OfficialIntegrationError(
                f"Failed to create FactorRDLoop: {e}"
            ) from e
```

### 7.3 结果转换错误

```python
class ResultConversionError(Exception):
    """结果转换错误"""
    pass

class _ResultAdapter:
    @staticmethod
    def exp_to_factor(exp: Experiment) -> FactorDefinition:
        try:
            # 转换逻辑
            ...
        except Exception as e:
            raise ResultConversionError(
                f"Failed to convert Experiment to FactorDefinition: {e}"
            ) from e
```

---

## 8. 测试策略

### 8.1 单元测试

```python
# tests/test_config_adapter.py
def test_dict_to_official_setting():
    config = {"max_iterations": 20}
    setting = _ConfigAdapter.to_official_setting(config)
    assert setting.evolving_n == 20

# tests/test_result_adapter.py
def test_trace_to_results_dict():
    mock_trace = create_mock_trace()
    results = _ResultAdapter.trace_to_results_dict(mock_trace, "test")
    assert "factors" in results
    assert "best_solution" in results
```

### 8.2 集成测试

```python
# tests/test_compat_wrapper.py
@pytest.mark.asyncio
async def test_research_pipeline():
    config = load_test_config()
    wrapper = RDAgentWrapper(config)
    
    results = await wrapper.research_pipeline(
        research_topic="测试因子研究",
        data=load_test_data(),
        max_iterations=1
    )
    
    assert isinstance(results, dict)
    assert "factors" in results
```

---

## 9. 实施计划

### 9.1 第一阶段: 基础框架 (P0-1.4)
- [ ] 创建 `official_integration.py`
- [ ] 实现 `OfficialRDAgentManager`
- [ ] 配置LLM集成 (P0-2)
- [ ] 配置路径管理 (P0-3)

### 9.2 第二阶段: 兼容层 (P0-1.5)
- [ ] 创建 `compat_wrapper.py`
- [ ] 实现 `RDAgentWrapper`
- [ ] 实现 `_ConfigAdapter`
- [ ] 实现 `_ResultAdapter`

### 9.3 第三阶段: 测试 (P0-1.7)
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能对比

---

## 10. 设计决策记录

### 10.1 为什么使用环境变量而非直接传参?
**决策**: 使用环境变量配置LLM和路径  
**原因**: 官方RD-Agent通过环境变量读取配置,保持一致性  
**权衡**: 需要在运行时设置`os.environ`,增加了复杂度

### 10.2 为什么不完全兼容旧行为?
**决策**: 允许内部行为差异(如缓存机制)  
**原因**: 官方实现更先进,强制兼容会限制功能  
**权衡**: 可能影响极少数依赖特定行为的代码

### 10.3 为什么暂时保留`optimize_strategy`自研实现?
**决策**: 策略优化暂时不迁移  
**原因**: 官方ModelLoop主要用于模型优化,策略优化需要研究  
**权衡**: 存在新旧实现混用的情况

---

**设计完成时间**: 2025-11-07  
**下一步**: P0-1.4 实现官方集成模块
