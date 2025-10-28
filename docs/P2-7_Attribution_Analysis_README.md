# P2-7 绩效归因分析系统 (Performance Attribution)

## 📋 概述

绩效归因分析系统用于解析投资组合的超额收益来源，帮助投资者理解：
- 配置决策vs选择决策的贡献
- 各风险因子对收益的影响
- 交易成本对净收益的侵蚀

## 🏗️ 系统架构

```
performance_attribution.py
├── BrinsonAttribution        # Brinson归因模型
├── FactorAttribution         # 因子归因分析
└── TransactionCostAnalysis   # 交易成本分析
```

## 🚀 快速开始

### 1. Brinson归因分析

```python
from performance_attribution import BrinsonAttribution

# 准备数据
portfolio_weights = pd.DataFrame(...)  # 组合权重
portfolio_returns = pd.DataFrame(...)  # 组合收益
benchmark_weights = pd.DataFrame(...)  # 基准权重
benchmark_returns = pd.DataFrame(...)  # 基准收益

# 执行归因
brinson = BrinsonAttribution(
    portfolio_weights, portfolio_returns,
    benchmark_weights, benchmark_returns
)
result = brinson.analyze()

print(f"配置效应: {result.allocation_effect:.2%}")
print(f"选择效应: {result.selection_effect:.2%}")
print(f"交互效应: {result.interaction_effect:.2%}")
print(f"总超额收益: {result.total_active_return:.2%}")
```

**输出示例:**
```
配置效应: 1.39%
选择效应: -5.46%
交互效应: -1.39%
总超额收益: -5.46%
```

### 2. 因子归因分析

```python
from performance_attribution import FactorAttribution

# 准备数据
returns = pd.Series(...)       # 组合收益率
factors = pd.DataFrame({       # 因子暴露
    'Market': [...],
    'Size': [...],
    'Value': [...]
})

# 执行因子归因
factor_attr = FactorAttribution(returns, factors)
contributions = factor_attr.analyze()

for factor, contrib in contributions.items():
    print(f"{factor}: {contrib:.4f}")
```

**输出示例:**
```
Market: -0.0014
Size: -0.0006
Value: 0.0001
Residual: 0.0078
```

### 3. 交易成本分析

```python
from performance_attribution import TransactionCostAnalysis

# 准备交易数据
trades = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', ...],
    'quantity': [100, 200, ...],
    'price': [150.5, 2800.0, ...],
    'timestamp': [...]
})

# 分析成本
cost_analysis = TransactionCostAnalysis(trades)
costs = cost_analysis.analyze(
    commission_rate=0.001,  # 0.1%佣金
    slippage_bps=5.0        # 5基点滑点
)

print(f"总交易成本: ¥{costs['total_cost']:,.2f}")
print(f"佣金成本: ¥{costs['commission_cost']:,.2f}")
print(f"滑点成本: ¥{costs['slippage_cost']:,.2f}")
print(f"市场冲击: ¥{costs['market_impact_cost']:,.2f}")
print(f"成本占比: {costs['cost_as_pct_of_value']:.3%}")
```

**输出示例:**
```
总交易成本: ¥1,546.50
佣金成本: ¥966.56
滑点成本: ¥483.28
市场冲击: ¥96.66
成本占比: 0.160%
```

## 📊 Web界面使用

### 启动仪表板

```bash
streamlit run web/unified_dashboard.py
```

### 使用步骤

1. **选择分析类型**
   - 在侧边栏选择: Brinson归因、因子归因、交易成本分析或综合报告

2. **配置参数**
   - 使用示例数据 或 上传自己的数据
   - 设置佣金率、滑点等参数

3. **查看结果**
   - 关键指标面板
   - 交互式可视化图表
   - 详细数据表格

4. **导出报告**
   - PDF报告
   - Excel数据
   - 归因总结

## 📈 核心功能详解

### Brinson归因模型

将组合超额收益分解为三个部分：

1. **配置效应** (Allocation Effect)
   ```
   Σ(Wp - Wb) × Rb
   ```
   - 衡量资产配置权重偏离基准的贡献
   - 正值表示配置决策增加收益

2. **选择效应** (Selection Effect)
   ```
   Σ Wb × (Rp - Rb)
   ```
   - 衡量证券选择产生的超额收益
   - 正值表示选股能力优秀

3. **交互效应** (Interaction Effect)
   ```
   Σ(Wp - Wb) × (Rp - Rb)
   ```
   - 配置和选择的协同效应
   - 可正可负

### 因子归因分析

使用回归分析将收益分解到各风险因子：

```python
Return = β1×Market + β2×Size + β3×Value + ... + Residual
```

- **Market**: 市场因子（贝塔）
- **Size**: 规模因子（小盘vs大盘）
- **Value**: 价值因子（账面市值比）
- **Momentum**: 动量因子（过去收益）
- **Residual**: 特异性收益（选股能力）

### 交易成本组成

1. **佣金成本** (Commission)
   - 直接支付给券商的费用
   - 通常为交易金额的0.1%-0.3%

2. **滑点成本** (Slippage)
   - 实际成交价与预期价格的差异
   - 通常为2-10基点

3. **市场冲击** (Market Impact)
   - 大额订单对市场价格的影响
   - 与订单规模和流动性相关

## 🧪 测试与验证

运行完整测试套件：

```bash
python tests/test_attribution_integration.py
```

测试包括：
- ✅ Brinson归因一致性检验
- ✅ 因子归因回归验证
- ✅ 交易成本计算正确性
- ✅ 完整工作流集成测试

## 📊 实际应用场景

### 场景1: 主动管理基金绩效分析

**问题**: 基金跑赢基准2.3%，超额收益来自哪里？

**分析步骤**:
```python
# 1. Brinson归因
brinson_result = brinson.analyze()
# 结果: 配置效应+1.5%, 选择效应+0.8%

# 2. 因子归因
factor_contrib = factor_attr.analyze()
# 结果: 价值因子贡献+0.9%, 特异性收益+1.4%

# 3. 扣除交易成本
net_excess = 2.3% - 0.15%  # 交易成本
# 净超额收益: 2.15%
```

**结论**: 
- 资产配置决策贡献更大(1.5% > 0.8%)
- 价值风格暴露带来正贡献
- 良好的选股能力(特异性收益1.4%)

### 场景2: 量化策略优化

**问题**: 策略年化收益15%，但交易过于频繁

**分析步骤**:
```python
cost_analysis = TransactionCostAnalysis(trades)
annual_cost = cost_analysis.analyze()

# 结果: 年化交易成本2.5%
# 优化后: 降低换手率, 成本降至1.2%
# 净收益提升: 15% - 1.2% = 13.8% (vs 原12.5%)
```

**优化建议**:
- 增加持仓周期
- 优化订单执行算法
- 考虑流动性约束

### 场景3: 多策略组合管理

**问题**: 评估各子策略对总收益的贡献

**分析方法**:
```python
# 对每个子策略执行因子归因
for strategy in ['value', 'momentum', 'mean_reversion']:
    contrib = analyze_factor_attribution(strategy)
    
# 汇总各策略贡献
total_attribution = aggregate_contributions()
```

## 📚 技术参考

### Brinson模型文献
- Brinson, G.P., Hood, L.R., and Beebower, G.L. (1986). "Determinants of Portfolio Performance"
- Brinson, G.P., and Fachler, N. (1985). "Measuring Non-US Equity Portfolio Performance"

### 因子模型
- Fama, E.F., and French, K.R. (1993). "Common Risk Factors in Stock Returns"
- Carhart, M.M. (1997). "On Persistence in Mutual Fund Performance"

### 交易成本
- Perold, A.F. (1988). "The Implementation Shortfall"
- Almgren, R., and Chriss, N. (2001). "Optimal Execution of Portfolio Transactions"

## 🛠️ 配置与调优

### 推荐参数

**佣金率**: 
- A股: 0.025% (万2.5)
- 美股: 0.005-0.01%

**滑点估计**:
- 高流动性股票: 2-5基点
- 中等流动性: 5-10基点
- 低流动性: 10-20基点

**因子选择**:
- 必选: Market (市场)
- 可选: Size, Value, Momentum, Quality

## 🔧 扩展开发

### 自定义归因模型

```python
from performance_attribution import BrinsonAttribution

class EnhancedBrinsonAttribution(BrinsonAttribution):
    def analyze_sector_level(self):
        """行业层面归因"""
        # 自定义实现
        pass
    
    def analyze_timing_effect(self):
        """择时效应分析"""
        # 自定义实现
        pass
```

### 添加新因子

```python
# 在FactorAttribution中添加
factors = pd.DataFrame({
    'Market': [...],
    'Size': [...],
    'Custom_Factor': [...]  # 自定义因子
})
```

## ⚠️ 注意事项

1. **数据质量**: 确保权重和收益数据准确对齐
2. **频率匹配**: 组合和基准数据频率应一致
3. **货币一致**: 所有金额单位需统一
4. **生存偏差**: 注意剔除退市股票的影响
5. **成本估计**: 交易成本参数应定期校准

## 📞 支持与反馈

- 问题反馈: GitHub Issues
- 技术文档: `/docs`
- 示例代码: `/examples`
- 测试用例: `/tests`

---

**版本**: v1.0.0  
**更新日期**: 2024年  
**维护者**: QiLin Quant Team
