# TradingAgents-CN-Plus 完整集成指南

## 📋 概述

本适配器实现了对 TradingAgents-CN-Plus 项目的完整集成，支持调用真实的多智能体协作系统进行深度股票分析。

## 🎯 功能特性

- ✅ **完整智能体系统**: 调用原项目的10+个专业智能体
- ✅ **深度分析报告**: 包含团队辩论、详细分析模块、投资建议
- ✅ **多维度分析**: 技术、基本面、情绪、新闻、风险等全方位分析
- ✅ **专业投资建议**: 仓位管理、止损止盈、时机选择等实战建议

## 🚀 快速开始

### 1. 安装依赖

运行自动安装脚本:

```bash
python scripts/install_tradingagents_deps.py
```

或手动安装核心依赖:

```bash
pip install langgraph langchain-anthropic langchain-openai langchain-google-genai akshare yfinance pandas openai google-generativeai streamlit plotly
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件:

```env
# LLM Provider 选择 (google/openai/anthropic)
LLM_PROVIDER=google

# Google Gemini API
GOOGLE_API_KEY=your_google_api_key_here

# 或者 OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# 或者 Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 模型配置
DEEP_THINK_LLM=gemini-2.0-flash
QUICK_THINK_LLM=gemini-2.0-flash
```

### 3. 验证安装

测试适配器是否正常工作:

```python
from tradingagents_integration.tradingagents_cn_plus_adapter import create_tradingagents_cn_plus_adapter

# 创建适配器
adapter = create_tradingagents_cn_plus_adapter()

# 查看状态
status = adapter.get_status()
print(status)

# 如果 status['available'] == True，说明安装成功
```

### 4. 运行分析

在 Streamlit 应用中:

1. 启动应用: `streamlit run web/main.py`
2. 进入 "TradingAgents" → "决策分析" tab
3. 输入股票代码 (如: 000001)
4. 选择分析深度 "完整"
5. 点击 "🚀 开始分析"

## 📊 智能体系统架构

### 核心智能体

1. **市场技术分析 (MarketAnalyst)**
   - 技术指标分析
   - 趋势判断
   - 支撑阻力位识别

2. **基本面分析 (FundamentalsAnalyst)**
   - 财务数据分析
   - 估值评估
   - 盈利能力分析

3. **新闻事件分析 (NewsAnalyst)**
   - 新闻情绪分析
   - 事件影响评估
   - 舆情监控

4. **社交媒体情绪 (SentimentAnalyst)**
   - 社交媒体情绪分析
   - 投资者情绪指标
   - 市场热度评估

5. **多头研究员 (BullAnalyst)**
   - 看涨论据分析
   - 上涨潜力评估

6. **空头研究员 (BearAnalyst)**
   - 看跌论据分析
   - 下跌风险评估

7. **研究经理 (ResearchManager)**
   - 综合多空观点
   - 形成一致性决策

8. **交易团队 (TraderTeam)**
   - 交易策略制定
   - 执行计划设计

9. **风险管理团队 (RiskTeam)**
   - 风险识别与评估
   - 风险控制建议

10. **投资组合经理 (PortfolioManager)**
    - 最终决策
    - 组合管理建议

## 🔧 配置说明

### 基础配置

```python
config = {
    "llm_provider": "google",  # google/openai/anthropic
    "deep_think_llm": "gemini-2.0-flash",
    "quick_think_llm": "gemini-2.0-flash",
    "max_debate_rounds": 2,
    "online_tools": True,
}

adapter = create_tradingagents_cn_plus_adapter(
    tradingagents_path="G:/test/tradingagents-cn-plus",
    config=config
)
```

### 高级配置

- `max_debate_rounds`: 辩论轮次 (1-5)，越多越深入但耗时越长
- `online_tools`: 是否启用在线数据工具
- `selected_analysts`: 选择参与的分析师 (默认: market, fundamentals, news, social)

## 📝 使用示例

### 异步调用

```python
import asyncio

async def analyze():
    adapter = create_tradingagents_cn_plus_adapter()
    
    # 分析单只股票
    result = await adapter.analyze_stock_full(
        symbol="000001",
        date="2025-01-20"
    )
    
    # 查看结果
    print(f"最终建议: {result['consensus']['signal']}")
    print(f"置信度: {result['consensus']['confidence']}")
    
    # 查看详细分析
    for agent in result['individual_results']:
        print(f"{agent['agent']}: {agent['signal']} ({agent['confidence']*100:.1f}%)")

# 运行
asyncio.run(analyze())
```

## ❓ 常见问题

### Q1: 提示 "No module named 'langgraph'"

**A**: 运行依赖安装脚本:
```bash
python scripts/install_tradingagents_deps.py
```

### Q2: 提示 "API key not configured"

**A**: 配置 `.env` 文件中的 API 密钥:
```env
GOOGLE_API_KEY=your_key_here
```

### Q3: 分析速度慢

**A**: 调整配置:
- 减少 `max_debate_rounds` (如设为1)
- 使用更快的模型 (如 gemini-2.0-flash)
- 减少参与的分析师数量

### Q4: 内存占用大

**A**: 
- 使用较小的模型
- 减少并发分析数量
- 定期重启服务

### Q5: TradingAgents-CN-Plus 项目路径不存在

**A**: 确保已克隆项目:
```bash
git clone https://github.com/your-repo/tradingagents-cn-plus.git G:/test/tradingagents-cn-plus
```

## 🔗 相关链接

- [TradingAgents-CN-Plus 项目](https://github.com/your-repo/tradingagents-cn-plus)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [AKShare 文档](https://akshare.akfamily.xyz/)

## 📄 许可证

本适配器遵循 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

*最后更新: 2025-01-20*
