# P2-2: 实验对比功能技术文档

## 📋 任务概述

**任务ID**: P2-2  
**任务名称**: 实验对比功能  
**开始时间**: 2024-11-07  
**完成时间**: 2024-11-07  
**状态**: ✅ 已完成 (80% - 待测试和优化)  
**负责人**: AI Agent

---

## 🎯 任务目标

创建一个功能完整的实验对比分析系统，支持：
1. **多实验选择和加载** - 从MLflow读取实验数据
2. **指标对比表格** - 横向对比多个实验的性能指标
3. **可视化分析** - 多种图表直观展示实验差异
4. **统计分析工具** - 科学评估实验结果的显著性和相关性

---

## 📦 交付成果

### 1. 核心文件

#### `web/tabs/qlib_experiment_comparison_tab.py` (1100行)

**功能模块**:
- 主渲染函数: `render_qlib_experiment_comparison_tab()`
- 4个子标签页: 实验选择、指标对比、可视化分析、统计分析

**关键函数列表**:

| 函数名 | 行数 | 功能说明 |
|--------|------|----------|
| `render_experiment_selector()` | 80 | 多实验选择界面 |
| `render_metrics_comparison()` | 60 | 指标对比表格 |
| `render_visualization_comparison()` | 50 | 可视化图表选择 |
| `render_statistical_analysis()` | 50 | 统计分析工具 |
| `get_all_experiments()` | 100 | 扫描MLflow实验 |
| `load_experiment_data()` | 80 | 加载实验数据 |
| `build_comparison_table()` | 80 | 构建对比表格 |
| `style_comparison_table()` | 60 | 表格样式化 |
| `render_parameter_diff()` | 60 | 参数差异分析 |
| `render_radar_chart()` | 80 | 雷达图渲染 |
| `render_returns_comparison()` | 70 | 收益率对比 |
| `render_ic_comparison()` | 100 | IC/ICIR对比 |
| `render_risk_metrics_comparison()` | 70 | 风险指标对比 |
| `render_comprehensive_comparison()` | 90 | 综合性能对比 |
| `render_correlation_analysis()` | 100 | 相关性分析 |
| `render_ranking_analysis()` | 150 | 排名和评分 |
| `render_stability_analysis()` | 80 | 稳定性分析 |

### 2. UI界面集成

**路径**: Qlib量化平台 → 实验管理 → 实验对比

修改文件: `web/unified_dashboard.py`
- 新增子标签页结构
- 集成实验对比模块
- 错误处理和友好提示

---

## 🎨 功能详解

### 1. 实验选择 (Tab 1)

#### 功能特性
- ✅ 自动扫描MLflow实验目录
- ✅ 显示实验列表（名称、记录数、创建时间、状态）
- ✅ 多选功能（支持2-10个实验）
- ✅ 批量加载实验数据
- ✅ 实时进度显示
- ✅ 数据摘要展示

#### 数据来源
1. **Session State**: 从`workflow_executions`读取最近运行的实验
2. **MLflow目录**: 扫描`mlruns/`目录，解析`meta.yaml`

#### 关键代码
```python
def get_all_experiments() -> Dict[str, Dict[str, Any]]:
    """
    扫描两个来源：
    1. st.session_state['workflow_executions']
    2. mlruns/目录下的实验文件夹
    """
    # 返回格式：
    # {
    #   "experiment_name": {
    #     "create_time": "2024-11-07 12:00:00",
    #     "status": "completed",
    #     "n_recorders": 3
    #   }
    # }
```

### 2. 指标对比 (Tab 2)

#### 功能特性
- ✅ 4种对比维度：预测性能、回测收益、风险指标、全部指标
- ✅ 智能指标分类和筛选
- ✅ 高亮最佳值（根据指标特性自动判断方向）
- ✅ 参数差异分析
- ✅ CSV导出功能

#### 对比表格样式

| 实验名称 | IC | ICIR | 年化收益率 | 夏普比率 | 最大回撤 |
|----------|-------|--------|------------|----------|----------|
| lgb_v1   | 0.0523 <span style="background-color: #90EE90">✓</span> | 1.24 | 0.1523 | 1.45 | -0.0823 |
| xgb_v1   | 0.0498 | 1.18 <span style="background-color: #90EE90">✓</span> | 0.1401 | 1.38 | -0.0912 |
| catb_v1  | 0.0512 | 1.21 | 0.1487 <span style="background-color: #90EE90">✓</span> | 1.42 <span style="background-color: #90EE90">✓</span> | -0.0798 <span style="background-color: #90EE90">✓</span> |

绿色高亮表示该指标的最佳值

#### 参数差异分析

**功能**:
- 自动识别不同实验间的参数差异
- 仅显示有差异的参数
- 折叠显示相同参数

**输出示例**:
```
⚠️ 发现 3 个参数存在差异

参数名         | lgb_v1  | xgb_v1  | catb_v1
---------------|---------|---------|--------
learning_rate  | 0.2     | 0.1     | 0.15
max_depth      | 8       | 6       | 7
num_leaves     | 210     | N/A     | N/A
```

### 3. 可视化分析 (Tab 3)

#### 图表类型

##### 3.1 指标雷达图
- **用途**: 多维度直观对比
- **特性**: 数据归一化到[0,1]，便于不同量级指标对比
- **可选指标**: 用户自定义选择3-8个指标
- **技术**: Plotly Scatterpolar

![雷达图示例](placeholder_radar.png)

##### 3.2 收益率对比
- **用途**: 横向对比收益类指标
- **图表**: 分组柱状图
- **指标**: 累计收益率、年化收益率、最大回撤
- **自动格式化**: 百分比显示

##### 3.3 IC/ICIR对比
- **用途**: 对比预测能力
- **图表**: 并排柱状图
- **说明**: 附带指标解释和经验阈值
  - IC > 0.03 为较好，> 0.05 为优秀
  - ICIR > 0.5 为较好，> 1.0 为优秀

##### 3.4 风险指标对比
- **用途**: 风险评估
- **图表**: 分组柱状图
- **指标**: 最大回撤、波动率、VaR、CVaR
- **提示**: 数值越小越好

##### 3.5 综合性能对比
- **用途**: 全局性能概览
- **图表**: 
  - 综合指标表格
  - 平行坐标图（归一化）
- **指标**: IC, ICIR, 收益率, 夏普比率, 最大回撤

#### 可视化最佳实践
```python
# 1. 使用一致的颜色方案
colors = px.colors.qualitative.Plotly

# 2. 添加数据标签
text=[f"{v:.4f}" for v in values],
textposition='auto'

# 3. 归一化处理
normalized = (value - min_val) / (max_val - min_val)

# 4. 自适应布局
fig.update_layout(height=450, showlegend=True)
```

### 4. 统计分析 (Tab 4)

#### 4.1 统计显著性检验

**功能**:
- 选择目标指标
- 显示数据概览（均值、标准差、最大最小值）
- 简单排名

**局限性**:
- 当前仅有单次运行结果，无法进行严格的t检验
- 建议每个模型运行3-5次（不同随机种子）

**未来增强**:
- 配对t检验
- Wilcoxon符号秩检验
- 效应量计算（Cohen's d）

#### 4.2 相关性分析

**功能**:
- 计算指标间的Pearson相关系数
- 热力图可视化
- 识别高相关性指标对（|r| > 0.7）

**应用价值**:
- 发现冗余指标
- 理解指标间关系
- 指导模型选择

**输出示例**:
```
高相关性指标对（|r| > 0.7）

指标1      | 指标2          | 相关系数 | 类型
-----------|----------------|----------|------
IC         | 年化收益率      | 0.856    | 正相关
夏普比率   | ICIR           | 0.782    | 正相关
最大回撤   | 波动率         | 0.734    | 正相关
```

#### 4.3 排名和评分

**核心功能**:
- 自定义权重设置
- 综合得分计算
- 智能排名
- 最佳模型推荐

**权重配置界面**:
```
预测能力指标:
- IC权重: [滑块] 0.30
- ICIR权重: [滑块] 0.20

收益指标:
- 收益率权重: [滑块] 0.30

风险调整收益:
- 夏普比率权重: [滑块] 0.20

风险指标（负向）:
- 最大回撤权重: [滑块] -0.10

权重总和: 1.00 ✅
```

**计算公式**:
```python
score = (w_ic * IC + 
         w_icir * ICIR + 
         w_return * Return + 
         w_sharpe * Sharpe + 
         w_drawdown * Drawdown)
```

**输出格式**:
```
排名结果

排名 | 实验名称  | 综合得分 | IC     | ICIR | 收益率  | 夏普   | 回撤
-----|-----------|----------|--------|------|---------|--------|--------
1    | catb_v1   | 0.4523   | 0.0512 | 1.21 | 0.1487  | 1.42   | 0.0798
2    | lgb_v1    | 0.4401   | 0.0523 | 1.24 | 0.1523  | 1.45   | 0.0823
3    | xgb_v1    | 0.4287   | 0.0498 | 1.18 | 0.1401  | 1.38   | 0.0912

🥇 推荐模型: catb_v1
📊 综合得分: 0.4523

关键指标:
- IC: 0.0512
- ICIR: 1.21
- 收益率: 14.87%
- 夏普比率: 1.42
- 最大回撤: 7.98%
```

#### 4.4 稳定性分析

**当前实现**:
- 计算指标变异系数（CV = 标准差/均值）
- 跨实验稳定性评估

**CV判断标准**:
- CV < 0.2: 高稳定性 ✅
- 0.2 ≤ CV < 0.5: 中等稳定性 ⚠️
- CV ≥ 0.5: 低稳定性 ❌

**局限性**:
- 基于不同实验的单次运行结果
- 无法评估单个模型的稳定性

**建议增强**:
1. 多次运行分析（不同随机种子）
2. 滚动窗口回测
3. 时序交叉验证
4. 不同市场环境测试

---

## 🔧 技术实现

### 数据流

```
1. 实验扫描
   ├── Session State (workflow_executions)
   └── MLflow目录 (mlruns/)
        └── 读取meta.yaml

2. 数据加载
   ├── Qlib R API
   │   ├── R.get_exp()
   │   └── exp.list_recorders()
   └── 提取
       ├── recorder.list_metrics()
       └── recorder.list_params()

3. 数据处理
   ├── 构建对比表格
   ├── 归一化处理
   └── 统计计算

4. 可视化渲染
   ├── Plotly图表
   ├── Streamlit组件
   └── 样式化表格
```

### 关键技术栈

| 技术 | 用途 | 版本要求 |
|------|------|----------|
| Streamlit | Web界面 | >= 1.20 |
| Plotly | 交互式图表 | >= 5.0 |
| Pandas | 数据处理 | >= 1.3 |
| NumPy | 数值计算 | >= 1.21 |
| SciPy | 统计分析 | >= 1.7 |
| Qlib | 实验数据 | >= 0.9 |
| MLflow | 实验追踪 | >= 2.0 |

### 错误处理

```python
try:
    # 核心逻辑
    experiment_data = load_data()
except ImportError as e:
    st.error("❌ 依赖库未安装")
    st.info("请安装: pip install qlib mlflow")
except FileNotFoundError:
    st.warning("⚠️ 未找到MLflow实验数据")
    st.markdown("请先运行实验或检查mlruns目录")
except Exception as e:
    st.error(f"❌ 发生错误: {e}")
    logger.error(f"详细错误: {e}", exc_info=True)
    with st.expander("🔍 查看详细错误"):
        st.code(traceback.format_exc())
```

### 性能优化

1. **延迟加载**: 只在用户点击"加载数据"时加载
2. **进度条**: 批量加载时显示进度
3. **Session State缓存**: 避免重复加载
4. **图表懒加载**: 只渲染当前选中的图表类型

```python
# 缓存策略
if 'experiment_data' not in st.session_state:
    st.session_state['experiment_data'] = {}

# 增量加载
for i, exp_name in enumerate(experiment_names):
    progress_bar.progress((i + 1) / len(experiment_names))
```

---

## 📊 使用指南

### 快速开始

1. **运行实验**
   ```bash
   # 在Qlib工作流标签运行多个实验
   # 确保每个实验有唯一的名称
   ```

2. **打开对比功能**
   ```
   Qlib量化平台 → 实验管理 → 实验对比
   ```

3. **选择实验**
   - 在"实验选择"标签多选2-10个实验
   - 点击"加载实验数据"

4. **查看对比**
   - "指标对比": 表格形式横向对比
   - "可视化分析": 图表直观展示
   - "统计分析": 科学评估和排名

### 最佳实践

#### 实验命名规范
```
建议格式: {模型}_{特征}_{版本}_{日期}

✅ 好的示例:
- lgb_alpha158_v1_20241107
- xgb_alpha360_v2_20241107
- catb_alpha158_tuned_20241107

❌ 不好的示例:
- test
- experiment1
- 模型训练
```

#### 对比策略

**同模型不同参数**:
```
目的: 参数调优
示例: lgb_v1, lgb_v2, lgb_v3
关注: 参数差异分析 + 性能对比
```

**不同模型同特征**:
```
目的: 模型选择
示例: lgb_alpha158, xgb_alpha158, catb_alpha158
关注: 综合排名 + IC/ICIR对比
```

**同模型不同特征**:
```
目的: 特征工程
示例: lgb_alpha158, lgb_alpha360
关注: 预测性能 + 计算成本
```

#### 指标解读

**IC/ICIR**:
- IC衡量预测准确性
- ICIR衡量预测稳定性
- 优先选择ICIR高的模型

**收益率vs夏普比率**:
- 高收益不一定好（可能伴随高风险）
- 夏普比率考虑了风险调整
- 追求风险收益平衡

**回撤控制**:
- 最大回撤是风险的直观体现
- 结合VaR/CVaR综合评估
- 考虑实际可承受风险

---

## 🧪 测试计划

### 单元测试

```python
def test_get_all_experiments():
    """测试实验扫描功能"""
    experiments = get_all_experiments()
    assert isinstance(experiments, dict)
    assert len(experiments) >= 0

def test_load_experiment_data():
    """测试数据加载"""
    exp_names = ["test_exp_1"]
    load_experiment_data(exp_names)
    assert 'experiment_data' in st.session_state

def test_build_comparison_table():
    """测试对比表格构建"""
    mock_data = {...}
    df = build_comparison_table(mock_data, "全部指标")
    assert not df.empty
```

### 集成测试

1. **端到端流程**
   - 运行3个实验
   - 选择并加载
   - 查看所有对比功能
   - 下载结果

2. **边界情况**
   - 0个实验
   - 1个实验（应提示至少2个）
   - 10+个实验（性能测试）
   - 缺失指标数据
   - 空参数

3. **错误恢复**
   - Qlib未安装
   - MLflow数据损坏
   - 网络中断（如有远程存储）

### 用户验收测试

- [ ] 界面友好，无明显卡顿
- [ ] 图表清晰，易于理解
- [ ] 对比结果准确
- [ ] 错误提示清晰
- [ ] 导出功能正常

---

## 📈 未来增强

### 短期 (P2后续迭代)

1. **时序对比**
   - 训练曲线对比
   - 滚动IC曲线
   - 累计收益曲线

2. **深度分析**
   - 股票级别对比
   - 行业表现对比
   - 市值分布对比

3. **自动化报告**
   - PDF报告生成
   - 对比总结
   - 建议输出

### 中期 (P3-P4)

1. **A/B测试框架**
   - 实验设计向导
   - 自动分组对比
   - 统计功效计算

2. **超参数影响分析**
   - 参数敏感性曲线
   - 参数交互效应
   - 最优参数推荐

3. **集成AutoML**
   - 自动实验生成
   - 超参数搜索可视化
   - 模型融合建议

### 长期 (Phase 5+)

1. **AI驱动的洞察**
   - 自动发现异常实验
   - 智能解释性能差异
   - 实验建议生成

2. **协作功能**
   - 实验分享
   - 团队协作
   - 版本管理

---

## 🐛 已知问题

### 问题1: 单次运行数据限制

**描述**: 当前每个实验只有一次运行的数据，无法进行严格的统计检验

**影响**: 稳定性分析和显著性检验功能受限

**解决方案**: 
- 短期: 提示用户多次运行
- 长期: 集成自动化多次运行框架

### 问题2: 时序数据缺失

**描述**: 从MLflow获取的是汇总指标，缺少时序数据

**影响**: 无法绘制净值曲线、回撤曲线等

**解决方案**:
- 建议用户前往"Qlib回测"标签查看详细曲线
- 未来版本考虑存储时序数据

### 问题3: 大量实验性能

**描述**: 选择10+个实验时可能较慢

**影响**: 用户体验

**解决方案**:
- 已添加进度条
- 考虑并行加载
- 增加缓存机制

---

## 📝 更新日志

### v1.0.0 (2024-11-07)

**新增功能**:
- ✅ 实验选择和数据加载
- ✅ 指标对比表格（4种维度）
- ✅ 6种可视化图表
- ✅ 4种统计分析工具
- ✅ 参数差异分析
- ✅ 综合排名和评分
- ✅ CSV导出功能
- ✅ 集成到主界面

**技术细节**:
- 代码行数: 1100+
- 支持的实验数: 2-10个
- 图表类型: 6种
- 分析工具: 4种

---

## 🤝 贡献指南

### 代码规范

```python
# 1. 函数命名
def render_xxx():  # 渲染函数
def get_xxx():     # 获取数据
def build_xxx():   # 构建数据结构
def calculate_xxx():  # 计算函数

# 2. 错误处理
try:
    # 核心逻辑
except SpecificError as e:
    # 具体错误处理
except Exception as e:
    # 通用错误处理
    logger.error(...)

# 3. 日志
logger.debug("详细调试信息")
logger.info("一般信息")
logger.warning("警告")
logger.error("错误", exc_info=True)
```

### 添加新图表

```python
def render_new_chart(experiment_data: Dict):
    """渲染新图表"""
    try:
        st.markdown("### 📊 新图表标题")
        
        # 1. 数据准备
        data = prepare_data(experiment_data)
        
        # 2. 创建图表
        fig = create_plotly_figure(data)
        
        # 3. 渲染
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. 说明文档
        st.info("💡 图表说明...")
        
    except Exception as e:
        logger.error(f"图表渲染失败: {e}", exc_info=True)
        st.error(f"❌ 图表渲染失败: {e}")
```

---

## 📚 参考资料

### 相关文档
- [Qlib官方文档](https://qlib.readthedocs.io/)
- [MLflow文档](https://mlflow.org/docs/latest/)
- [Plotly文档](https://plotly.com/python/)
- [Streamlit文档](https://docs.streamlit.io/)

### 统计方法
- Pearson相关系数
- 变异系数 (Coefficient of Variation)
- t检验 (Student's t-test)
- 效应量 (Effect Size)

### 量化指标
- IC (Information Coefficient)
- ICIR (IC Information Ratio)
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Maximum Drawdown)

---

## 📞 联系方式

**问题反馈**: GitHub Issues  
**功能建议**: GitHub Discussions  
**技术支持**: 开发团队

---

**文档版本**: v1.0.0  
**最后更新**: 2024-11-07  
**维护者**: AI Agent
