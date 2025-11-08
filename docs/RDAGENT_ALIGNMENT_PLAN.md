# 🎯 RD-Agent 全功能对齐计划与里程碑

**制定日期**: 2025-11-07  
**目标**: 实现RD-Agent所有功能通过Web UI完整访问  
**当前覆盖率**: 75% → 目标95%+

---

## 📋 执行概览

| 阶段 | 任务 | 工作量 | 优先级 | 状态 |
|------|------|--------|--------|------|
| Phase 1 | 环境配置UI | 4h | P0 | ✅ 完成 |
| Phase 2 | 健康检查增强 | 2h | P0 | 🟡 进行中 |
| Phase 3 | 主界面集成 | 2h | P0 | 📝 待开始 |
| Phase 4 | 会话管理 | 6h | P1 | 📝 待开始 |
| Phase 5 | Kaggle真实运行 | 8h | P1 | 📝 待开始 |
| Phase 6 | DataScience集成 | 6h | P1 | 📝 待开始 |
| Phase 7 | 日志可视化 | 8h | P2 | 📝 待开始 |

**总计**: 36小时 (~5个工作日)

---

## 🎯 Phase 1: 环境配置UI (已完成 ✅)

### 目标
提供RD-Agent环境检测与配置管理界面

### 已实现功能
- ✅ `.env` 文件编辑器
- ✅ RDAGENT_PATH 配置
- ✅ DS_LOCAL_DATA_PATH 配置
- ✅ DS_IF_USING_MLE_DATA 开关
- ✅ DS_CODER_COSTEER_ENV_TYPE 选择
- ✅ 环境摘要显示(RD-Agent/Kaggle/Docker状态)
- ✅ 诊断建议

### 文件
- `web/tabs/rdagent/env_config.py` (164行)

---

## 🎯 Phase 2: RDAgentAPI健康检查增强 (进行中 🟡)

### 目标
增强API层健康检查能力,支持环境配置UI

### 实现内容

#### 2.1 health_check() 方法
```python
def health_check(self) -> Dict[str, Any]:
    """全面环境健康检查
    
    Returns:
        {
            'success': bool,
            'rdagent_importable': bool,
            'rdagent_version': str,
            'kaggle_api_configured': bool,
            'kaggle_cli': bool,
            'docker': bool,
            'env_type': 'docker' | 'conda',
            'details': {...}
        }
    """
```

#### 2.2 检查项
1. **RD-Agent导入**
   - 检测rdagent包可用性
   - 获取版本信息
   - 检查核心模块导入

2. **Kaggle API**
   - 检测 `~/.kaggle/kaggle.json`
   - 验证API密钥有效性
   - 检查kaggle CLI可用

3. **Docker环境**
   - 检测Docker守护进程
   - 验证Docker命令可用
   - 获取Docker版本

4. **运行环境**
   - 读取DS_CODER_COSTEER_ENV_TYPE
   - 验证conda/docker环境一致性

### 技术实现
```python
# rdagent_api.py 新增方法
def health_check(self) -> Dict[str, Any]:
    result = {
        'success': True,
        'rdagent_importable': self.rdagent_available,
        'details': {}
    }
    
    # 1. RD-Agent检查
    if self.rdagent_available:
        try:
            import rdagent
            result['rdagent_version'] = getattr(rdagent, '__version__', 'unknown')
        except Exception as e:
            result['details']['rdagent_error'] = str(e)
    
    # 2. Kaggle API检查
    result['kaggle_api_configured'] = self._check_kaggle_api()
    
    # 3. Kaggle CLI检查
    result['kaggle_cli'] = self._check_kaggle_cli()
    
    # 4. Docker检查
    result['docker'] = self._check_docker()
    
    # 5. 环境类型
    result['env_type'] = os.getenv('DS_CODER_COSTEER_ENV_TYPE', 'docker')
    
    return result
```

### 验收标准
- [x] health_check() 方法实现
- [x] 所有检查项工作正常
- [x] env_config.py 正确调用
- [x] 诊断信息准确

---

## 🎯 Phase 3: RD-Agent主界面集成 (待开始 📝)

### 目标
将环境配置标签页集成到主RD-Agent界面

### 实现内容

#### 3.1 修改 unified_dashboard.py
```python
def render_rdagent_tabs(self):
    """渲染RD-Agent的7个子tab"""
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "⚙️ 环境配置",      # 新增
        "🔬 因子挖掘",
        "🎯 模型优化",
        "🏆 Kaggle竞赛",
        "📚 因子库",
        "🔧 其他功能",
        "🎖️ MLE-Bench"
    ])
    
    with tab1:
        from web.tabs.rdagent.env_config import render
        render()
    
    # ... 其他标签页
```

#### 3.2 标签页顺序优化
将环境配置放在第一位,确保用户首先配置环境

### 验收标准
- [ ] 环境配置标签页显示正常
- [ ] 所有功能按钮工作
- [ ] 健康检查实时更新
- [ ] UI布局美观

---

## 🎯 Phase 4: 会话管理UI骨架 (待开始 📝)

### 目标
提供RD-Agent运行会话的管理界面

### 功能设计

#### 4.1 会话列表
- 显示所有历史会话
- 会话类型(Factor/Model/Kaggle/DataScience)
- 会话状态(Running/Completed/Failed)
- 创建时间、持续时间

#### 4.2 会话控制
- 启动新会话
- 暂停/恢复会话
- 停止会话
- 删除会话

#### 4.3 会话详情
- 实时进度条
- 当前步骤信息
- 已完成步骤列表
- 资源使用情况

#### 4.4 日志查看
- 实时日志流
- 日志过滤(级别/关键词)
- 日志下载

### 文件结构
```
web/tabs/rdagent/
├── session_manager.py     # 会话管理主模块 (新建)
├── session_viewer.py      # 会话详情查看器 (新建)
└── session_storage.py     # 会话持久化存储 (新建)
```

### 技术实现
```python
# session_manager.py
class SessionManager:
    def __init__(self):
        self.sessions_dir = Path("workspace/sessions")
        self.active_sessions = {}
    
    def list_sessions(self, session_type=None, status=None):
        """列出所有会话"""
        pass
    
    def get_session(self, session_id):
        """获取会话详情"""
        pass
    
    def start_session(self, session_type, config):
        """启动新会话"""
        pass
    
    def stop_session(self, session_id):
        """停止会话"""
        pass
    
    def get_session_logs(self, session_id, tail=100):
        """获取会话日志"""
        pass
```

### 验收标准
- [ ] 会话列表显示正常
- [ ] 会话控制按钮工作
- [ ] 会话详情页完整
- [ ] 日志实时更新

---

## 🎯 Phase 5: KaggleRDLoop真实运行 (待开始 📝)

### 目标
实现Kaggle竞赛的完整RD-Agent自动化工作流

### 功能设计

#### 5.1 竞赛选择
- 列出热门Kaggle竞赛
- 竞赛信息展示
- 数据集下载

#### 5.2 运行配置
- step_n: 单次迭代步数
- loop_n: 总循环次数
- max_workers: 并行数
- timeout: 超时设置

#### 5.3 实时监控
- 当前步骤进度
- 已生成方案数
- 最佳得分
- 提交历史

#### 5.4 结果展示
- 提交结果列表
- 排行榜位置
- 性能对比图表

### 技术实现
```python
# kaggle_agent.py 增强版
async def run_kaggle_rdloop(competition: str, config: Dict):
    """运行Kaggle RD Loop
    
    Args:
        competition: 竞赛名称
        config: {
            'step_n': 5,
            'loop_n': 3,
            'max_workers': 4,
            'timeout': 3600
        }
    """
    from rdagent.scenarios.kaggle.kaggle_crawler import KaggleRDLoop
    from rdagent.scenarios.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
    
    # 更新配置
    KAGGLE_IMPLEMENT_SETTING.competition = competition
    
    # 创建loop
    loop = KaggleRDLoop(KAGGLE_IMPLEMENT_SETTING)
    
    # 运行循环
    for i in range(config['loop_n']):
        await loop.run(step_n=config['step_n'])
        
        # 发送进度更新
        yield {
            'progress': (i+1) / config['loop_n'],
            'current_loop': i+1,
            'solutions_generated': len(loop.trace.hist)
        }
```

### 验收标准
- [ ] 竞赛列表正常加载
- [ ] 可配置运行参数
- [ ] 实时进度显示
- [ ] 结果正确展示
- [ ] 错误处理完善

---

## 🎯 Phase 6: DataScienceRDLoop集成 (待开始 📝)

### 目标
集成通用数据科学场景的RD-Agent循环

### 功能设计

#### 6.1 数据上传
- 支持CSV/Excel/JSON
- 数据预览
- 基础统计信息

#### 6.2 任务定义
- 任务类型(分类/回归/聚类)
- 目标列选择
- 评估指标

#### 6.3 自动建模
- 特征工程
- 模型选择
- 超参调优
- 交叉验证

#### 6.4 结果分析
- 模型性能报告
- 特征重要性
- 预测结果下载

### 技术实现
```python
# data_science_loop.py
from rdagent.scenarios.data_science.loop import DataScienceRDLoop

class DataScienceAgent:
    def __init__(self):
        self.loop = None
    
    async def run(self, data_path, task_config):
        """运行数据科学循环
        
        Args:
            data_path: 数据文件路径
            task_config: {
                'task_type': 'classification',
                'target_col': 'label',
                'metric': 'accuracy',
                'step_n': 5
            }
        """
        # 初始化loop
        self.loop = DataScienceRDLoop(
            data_path=data_path,
            task_type=task_config['task_type']
        )
        
        # 运行
        await self.loop.run(step_n=task_config['step_n'])
        
        return self._extract_results()
```

### 验收标准
- [ ] 数据上传正常
- [ ] 任务配置完整
- [ ] 建模流程正常
- [ ] 结果展示清晰

---

## 🎯 Phase 7: 原生日志可视化嵌入 (待开始 📝)

### 目标
将RD-Agent原生trace日志进行可视化展示

### 功能设计

#### 7.1 时间轴视图
- 显示完整RD流程时间线
- Research/Development/Experiment阶段标注
- 关键节点高亮

#### 7.2 步骤详情
- 每个步骤的输入/输出
- 代码生成结果
- 测试运行结果
- 错误信息

#### 7.3 交互功能
- 点击节点查看详情
- 过滤特定阶段
- 搜索关键词
- 导出报告

### 技术实现
```python
# log_visualizer.py
import json
from pathlib import Path
from datetime import datetime

class RDAgentLogVisualizer:
    def __init__(self, trace_file: Path):
        self.trace_data = self._load_trace(trace_file)
    
    def _load_trace(self, file_path):
        """加载trace.json文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def render_timeline(self):
        """渲染时间轴视图"""
        import plotly.graph_objects as go
        
        # 解析trace数据
        events = self._parse_events()
        
        # 创建时间轴图表
        fig = go.Figure()
        
        for event in events:
            fig.add_trace(go.Scatter(
                x=[event['timestamp']],
                y=[event['stage']],
                mode='markers+text',
                text=event['description'],
                marker=dict(size=10)
            ))
        
        return fig
    
    def render_step_detail(self, step_id):
        """渲染步骤详情"""
        step = self._get_step(step_id)
        
        st.subheader(f"步骤: {step['name']}")
        st.write(f"**类型**: {step['type']}")
        st.write(f"**状态**: {step['status']}")
        st.write(f"**耗时**: {step['duration']}s")
        
        # 输入
        with st.expander("输入"):
            st.json(step['input'])
        
        # 输出
        with st.expander("输出"):
            st.code(step['output'], language='python')
        
        # 日志
        with st.expander("日志"):
            st.text(step['logs'])
```

### UI布局
```
┌─────────────────────────────────────┐
│       🕐 RD-Agent 执行时间轴         │
├─────────────────────────────────────┤
│ Research ━━●━━━━━━━━━━━━━━━━━━━━━━ │
│          ↓                          │
│ Development ━━━━━●━━━━━━━━━━━━━━━  │
│                  ↓                  │
│ Experiment ━━━━━━━━━━●━━━━━━━━━━   │
│                      ↓              │
│ Evaluation ━━━━━━━━━━━━━━●━━━━━━  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│       📋 步骤详情                    │
├─────────────────────────────────────┤
│ 步骤ID: step_001                    │
│ 类型: Research - Hypothesis         │
│ 状态: ✅ Success                     │
│ 耗时: 45.2s                         │
│                                     │
│ [输入] [输出] [代码] [日志] [指标]   │
└─────────────────────────────────────┘
```

### 验收标准
- [ ] 时间轴正确显示
- [ ] 步骤详情完整
- [ ] 交互流畅
- [ ] 性能良好(大日志文件)

---

## 📊 整体里程碑

### Week 1 (Day 1-3)
- [x] Phase 1: 环境配置UI
- [x] Phase 2: 健康检查增强
- [ ] Phase 3: 主界面集成

### Week 1 (Day 4-5)
- [ ] Phase 4: 会话管理UI

### Week 2 (Day 1-3)
- [ ] Phase 5: Kaggle真实运行

### Week 2 (Day 4-5)
- [ ] Phase 6: DataScience集成
- [ ] Phase 7: 日志可视化

---

## 🎯 成功指标

### 功能完整性
- ✅ 环境配置: 100%
- 🟡 健康检查: 80%
- ⬜ 会话管理: 0%
- ⬜ Kaggle运行: 0%
- ⬜ DataScience: 0%
- ⬜ 日志可视化: 0%

### 用户体验
- 界面响应速度 < 1s
- 实时日志延迟 < 2s
- 错误提示清晰友好
- 文档完整易懂

### 代码质量
- 测试覆盖率 > 80%
- 代码复杂度 < 10
- 文档字符串完整
- 类型注解完整

---

## 🔗 相关资源

### 文档
- RD-Agent官方文档: https://rdagent.readthedocs.io/
- 麒麟项目文档: `docs/RDAGENT_FINAL_SUMMARY.md`

### 代码
- 环境配置: `web/tabs/rdagent/env_config.py`
- API层: `web/tabs/rdagent/rdagent_api.py`
- 主界面: `web/unified_dashboard.py`

### 工具
- Streamlit文档: https://docs.streamlit.io/
- Plotly文档: https://plotly.com/python/

---

**更新日期**: 2025-11-07  
**下次审查**: Week 1 结束  
**项目负责人**: AI Agent
