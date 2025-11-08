#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RD-Agent 完整功能集成验证脚本

验证所有集成改动是否正确实施:
1. DataScience Loop参数支持
2. 日志可视化FileStorage支持
3. Kaggle高级配置
4. 会话存储线程安全
5. 环境默认值
6. 文档链接
"""

import sys
from pathlib import Path
import re

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_status(msg, success=True):
    """打印带颜色的状态消息"""
    color = Colors.GREEN if success else Colors.RED
    symbol = '✅' if success else '❌'
    print(f"{color}{symbol} {msg}{Colors.RESET}")

def print_section(title):
    """打印章节标题"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.RESET}\n")

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if filepath.exists():
        print_status(f"{description}: {filepath.name}", True)
        return True
    else:
        print_status(f"{description}: 文件不存在 - {filepath}", False)
        return False

def check_code_pattern(filepath, pattern, description):
    """检查代码中是否包含特定模式"""
    if not filepath.exists():
        print_status(f"{description}: 文件不存在", False)
        return False
    
    try:
        content = filepath.read_text(encoding='utf-8')
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print_status(f"{description}", True)
            return True
        else:
            print_status(f"{description}: 未找到预期代码", False)
            return False
    except Exception as e:
        print_status(f"{description}: 读取失败 - {e}", False)
        return False

def main():
    """主验证函数"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("  🔍 RD-Agent 完整功能集成验证")
    print(f"{'='*60}{Colors.RESET}")
    
    base_dir = Path(__file__).parent
    results = []
    
    # ============================================================
    # 1. DataScience Loop参数验证
    # ============================================================
    print_section("1. DataScience Loop参数支持验证")
    
    ds_loop_file = base_dir / "web/tabs/rdagent/data_science_loop.py"
    rdagent_api_file = base_dir / "web/tabs/rdagent/rdagent_api.py"
    
    # 检查UI层loop_n参数
    results.append(check_code_pattern(
        ds_loop_file,
        r'loop_n\s*=\s*st\.number_input.*循环次数',
        "UI层包含loop_n参数输入"
    ))
    
    # 检查UI层timeout参数
    results.append(check_code_pattern(
        ds_loop_file,
        r'timeout\s*=\s*st\.number_input.*超时',
        "UI层包含timeout参数输入"
    ))
    
    # 检查API层参数透传
    results.append(check_code_pattern(
        rdagent_api_file,
        r"await loop\.run\(step_n=step_n,\s*loop_n=loop_n,\s*all_duration=timeout\)",
        "API层正确透传loop_n和timeout参数"
    ))
    
    # ============================================================
    # 2. 日志可视化FileStorage支持验证
    # ============================================================
    print_section("2. 日志可视化FileStorage支持验证")
    
    log_viz_file = base_dir / "web/tabs/rdagent/log_visualizer.py"
    
    # 检查FileStorage导入
    results.append(check_code_pattern(
        log_viz_file,
        r'import pickle',
        "导入pickle模块用于FileStorage"
    ))
    
    # 检查FileStorage函数
    results.append(check_code_pattern(
        log_viz_file,
        r'def _load_traces_from_filestorage',
        "包含_load_traces_from_filestorage函数"
    ))
    
    # 检查日志源选择
    results.append(check_code_pattern(
        log_viz_file,
        r'log_source\s*=\s*st\.radio.*日志源类型.*FileStorage',
        "UI提供日志源类型选择"
    ))
    
    # 检查POSSIBLE_LOG_DIRS
    results.append(check_code_pattern(
        log_viz_file,
        r'POSSIBLE_LOG_DIRS\s*=',
        "定义了POSSIBLE_LOG_DIRS"
    ))
    
    # ============================================================
    # 3. Kaggle高级配置验证
    # ============================================================
    print_section("3. Kaggle高级配置支持验证")
    
    kaggle_agent_file = base_dir / "web/tabs/rdagent/kaggle_agent.py"
    
    # 检查auto_submit开关
    results.append(check_code_pattern(
        kaggle_agent_file,
        r'auto_submit\s*=\s*st\.checkbox.*自动提交',
        "Kaggle UI包含auto_submit开关"
    ))
    
    # 检查图RAG开关
    results.append(check_code_pattern(
        kaggle_agent_file,
        r'use_graph_rag\s*=\s*st\.checkbox.*图知识库RAG',
        "Kaggle UI包含图知识库RAG开关"
    ))
    
    # 检查API层配置应用
    results.append(check_code_pattern(
        rdagent_api_file,
        r'KAGGLE_IMPLEMENT_SETTING\.auto_submit\s*=',
        "API层应用auto_submit配置"
    ))
    
    results.append(check_code_pattern(
        rdagent_api_file,
        r'KAGGLE_IMPLEMENT_SETTING\.knowledge_base\s*=.*KGKnowledgeGraph',
        "API层配置知识库RAG"
    ))
    
    # ============================================================
    # 4. 会话存储线程安全验证
    # ============================================================
    print_section("4. 会话存储线程安全验证")
    
    session_mgr_file = base_dir / "web/tabs/rdagent/session_manager.py"
    
    # 检查线程锁
    results.append(check_code_pattern(
        session_mgr_file,
        r'self\._lock\s*=\s*threading\.Lock\(\)',
        "SessionStorage包含主线程锁"
    ))
    
    results.append(check_code_pattern(
        session_mgr_file,
        r'self\._log_locks\s*=\s*\{\}',
        "SessionStorage包含日志锁字典"
    ))
    
    # 检查加锁使用
    results.append(check_code_pattern(
        session_mgr_file,
        r'with self\._lock:.*load_sessions',
        "load_sessions使用锁保护"
    ))
    
    results.append(check_code_pattern(
        session_mgr_file,
        r'with self\._log_locks\[session_id\]:',
        "日志操作使用独立锁"
    ))
    
    # ============================================================
    # 5. 环境默认值验证
    # ============================================================
    print_section("5. 环境默认值验证")
    
    env_config_file = base_dir / "web/tabs/rdagent/env_config.py"
    
    # 检查conda默认值
    results.append(check_code_pattern(
        env_config_file,
        r"env_vals\.get\('DS_CODER_COSTEER_ENV_TYPE',\s*'conda'\)",
        "env_config默认使用conda"
    ))
    
    # 检查帮助提示
    results.append(check_code_pattern(
        env_config_file,
        r'help=.*Windows.*conda',
        "包含Windows使用conda的帮助提示"
    ))
    
    # 检查API health_check默认值
    results.append(check_code_pattern(
        rdagent_api_file,
        r"result\['env_type'\]\s*=\s*os\.getenv\(.*'conda'\)",
        "health_check默认使用conda"
    ))
    
    # ============================================================
    # 6. 文档链接验证
    # ============================================================
    print_section("6. 文档链接验证")
    
    dashboard_file = base_dir / "web/unified_dashboard.py"
    
    # 检查归档文档链接
    results.append(check_code_pattern(
        dashboard_file,
        r'docs/archive/completion/RDAGENT_ALIGNMENT_COMPLETE\.md',
        "RD-Agent对齐完成文档链接正确"
    ))
    
    results.append(check_code_pattern(
        dashboard_file,
        r'docs/archive/completion/ALIGNMENT_COMPLETION_CHECK\.md',
        "对齐完成检查文档链接正确"
    ))
    
    results.append(check_code_pattern(
        dashboard_file,
        r'docs/archive/completion/TESTING_COMPLETION_REPORT\.md',
        "测试完成报告文档链接正确"
    ))
    
    # 验证文档实际存在
    results.append(check_file_exists(
        base_dir / "docs/archive/completion/RDAGENT_ALIGNMENT_COMPLETE.md",
        "归档文档实际存在"
    ))
    
    # ============================================================
    # 7. 新增报告文档验证
    # ============================================================
    print_section("7. 新增文档验证")
    
    report_file = base_dir / "docs/RDAGENT_COMPLETE_INTEGRATION_REPORT.md"
    results.append(check_file_exists(
        report_file,
        "完整集成报告已创建"
    ))
    
    # ============================================================
    # 总结
    # ============================================================
    print_section("验证总结")
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"总计: {total} 项检查")
    print(f"通过: {Colors.GREEN}{passed}{Colors.RESET} 项")
    print(f"失败: {Colors.RED}{failed}{Colors.RESET} 项")
    print(f"成功率: {passed/total*100:.1f}%\n")
    
    if failed == 0:
        print(f"{Colors.GREEN}{'='*60}")
        print("  🎉 所有检查通过! RD-Agent集成验证成功!")
        print(f"{'='*60}{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{'='*60}")
        print(f"  ⚠️  有 {failed} 项检查未通过,请检查相关文件")
        print(f"{'='*60}{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
