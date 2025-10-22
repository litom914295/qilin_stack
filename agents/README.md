# Agents 目录说明

## 📁 目录结构

- `trading_agents.py.deprecated` - 旧版本Agent实现(已废弃)

## ⚠️ 重要说明

**请使用 `app/agents/trading_agents_impl.py` 作为Agent的标准实现!**

旧版本 `trading_agents.py` 已重命名为 `.deprecated`,不再维护。

### 为什么废弃?

1. **功能重复**: 与 `app/agents/trading_agents_impl.py` 功能完全重复
2. **代码质量**: 新版本实现更完整,逻辑更清晰
3. **维护成本**: 保留两份实现会增加维护难度

### 新版本的优势

- ✅ 完整的10个专业Agent实现
- ✅ 清晰的评分逻辑和决策规则
- ✅ 异步并行分析,性能更好
- ✅ 完善的错误处理和日志
- ✅ 详细的文档注释

## 📚 相关文档

- [新版Agent实现](../app/agents/trading_agents_impl.py)
- [Agent设计文档](../docs/Technical_Architecture_v2.1_Final.md)
- [代码审查报告](../docs/CODE_REVIEW_REPORT.md)
