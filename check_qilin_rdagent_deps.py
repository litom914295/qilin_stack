#!/usr/bin/env python
"""
检查本项目 (qilin_stack) 的 RD-Agent 依赖状态
"""

import subprocess
import sys
from pathlib import Path


def check_rdagent_package():
    """检查 rdagent 包是否安装"""
    print("=" * 60)
    print("🔍 检查 rdagent 包安装状态")
    print("=" * 60)
    
    try:
        import rdagent
        print(f"✅ rdagent 包已安装")
        print(f"   版本: {getattr(rdagent, '__version__', '未知')}")
        print(f"   路径: {rdagent.__file__}")
        return True
    except ImportError:
        print(f"❌ rdagent 包未安装")
        return False


def check_rdagent_path():
    """检查 RD-Agent 项目路径"""
    print("\n" + "=" * 60)
    print("📂 检查 RD-Agent 项目路径")
    print("=" * 60)
    
    # 从配置读取路径
    rdagent_paths = [
        Path("G:/test/RD-Agent"),
        Path("D:/test/Qlib/RD-Agent"),
    ]
    
    found_paths = []
    for path in rdagent_paths:
        if path.exists():
            print(f"✅ 找到 RD-Agent 项目: {path}")
            found_paths.append(path)
        else:
            print(f"❌ 路径不存在: {path}")
    
    return found_paths


def check_rdagent_imports():
    """检查 rdagent 相关的导入"""
    print("\n" + "=" * 60)
    print("🔬 测试 rdagent 模块导入")
    print("=" * 60)
    
    # 本项目需要的 rdagent 模块
    required_modules = [
        "rdagent.scenarios.qlib.experiment.factor_experiment",
        "rdagent.scenarios.qlib.experiment.model_experiment",
        "rdagent.app.qlib_rd_loop.factor",
        "rdagent.app.qlib_rd_loop.model",
        "rdagent.components.workflow.rd_loop",
        "rdagent.core.exception",
        "rdagent.log",
    ]
    
    missing_modules = []
    
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name} - {e}")
            missing_modules.append(module_name)
    
    return missing_modules


def check_project_integration():
    """检查项目集成文件"""
    print("\n" + "=" * 60)
    print("📋 检查项目集成文件")
    print("=" * 60)
    
    integration_files = [
        "rd_agent/full_integration.py",
        "rd_agent/limitup_integration.py",
        "rd_agent/real_integration.py",
        "rd_agent/config.py",
        "app/integration/rdagent_adapter.py",
        "app/integrations/rdagent_integration.py",
    ]
    
    project_root = Path("G:/test/qilin_stack")
    
    existing_files = []
    for file_path in integration_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
            existing_files.append(full_path)
        else:
            print(f"  ❌ {file_path} (不存在)")
    
    return existing_files


def get_install_instructions():
    """返回安装说明"""
    print("\n" + "=" * 60)
    print("💡 解决方案")
    print("=" * 60)
    
    print("\n方式1: 从 RD-Agent 源码安装 (推荐)")
    print("-" * 60)
    print("cd G:\\test\\RD-Agent")
    print("pip install -e .")
    print("# 或安装完整依赖:")
    print("pip install -e .[torch]")
    
    print("\n方式2: 从 PyPI 安装")
    print("-" * 60)
    print("pip install rdagent")
    
    print("\n方式3: 手动添加到 Python 路径 (临时)")
    print("-" * 60)
    print("在代码中添加:")
    print("import sys")
    print("sys.path.insert(0, 'G:/test/RD-Agent')")
    
    print("\n⚠️  注意:")
    print("- 本项目 (qilin_stack) 的 rd_agent/ 模块需要 RD-Agent 官方包作为依赖")
    print("- 推荐使用方式1从源码安装，这样可以获得最新功能")
    print("- 安装后需要配置环境变量 (LLM API keys 等)")


def main():
    print("\n" + "🎯" * 30)
    print("  本项目 (qilin_stack) 的 RD-Agent 依赖检查")
    print("🎯" * 30 + "\n")
    
    # 1. 检查 rdagent 包
    rdagent_installed = check_rdagent_package()
    
    # 2. 检查 RD-Agent 项目路径
    rdagent_paths = check_rdagent_path()
    
    # 3. 检查模块导入
    if rdagent_installed:
        missing_modules = check_rdagent_imports()
    else:
        missing_modules = ["所有模块 (rdagent 未安装)"]
    
    # 4. 检查项目集成文件
    integration_files = check_project_integration()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 检查总结")
    print("=" * 60)
    
    print(f"\n✅ RD-Agent 包已安装: {'是' if rdagent_installed else '否'}")
    print(f"✅ RD-Agent 项目路径: {len(rdagent_paths)} 个")
    print(f"✅ 项目集成文件: {len(integration_files)} 个")
    
    if rdagent_installed and not missing_modules:
        print(f"\n🎉 所有依赖检查通过！")
        print("\n✅ 本项目的 rd_agent 模块可以正常使用")
    else:
        print(f"\n⚠️  缺少依赖:")
        if not rdagent_installed:
            print("  - rdagent 包未安装")
        if missing_modules:
            print(f"  - {len(missing_modules)} 个模块无法导入")
        
        print(f"\n❌ 本项目的 rd_agent 模块无法正常使用")
        
        # 显示安装说明
        get_install_instructions()


if __name__ == '__main__':
    main()
