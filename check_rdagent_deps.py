#!/usr/bin/env python
"""
检查 RD-Agent 项目的依赖是否已正确安装
Check if RD-Agent project dependencies are correctly installed
"""

import subprocess
import sys
from pathlib import Path


def parse_requirements(req_file: Path) -> list[str]:
    """解析 requirements 文件"""
    if not req_file.exists():
        return []
    
    requirements = []
    with open(req_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            # 移除版本号，只保留包名
            pkg = line.split('>=')[0].split('==')[0].split('[')[0].strip()
            if pkg:
                requirements.append(pkg)
    
    return requirements


def get_installed_packages() -> set[str]:
    """获取已安装的包列表"""
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
        capture_output=True,
        text=True
    )
    
    installed = set()
    for line in result.stdout.split('\n'):
        if '==' in line:
            pkg = line.split('==')[0].strip()
            installed.add(pkg.lower())
    
    return installed


def normalize_package_name(pkg: str) -> str:
    """规范化包名（处理 - 和 _ 的区别）"""
    return pkg.lower().replace('_', '-')


def main():
    # RD-Agent 项目路径
    rdagent_path = Path('G:/test/RD-Agent')
    
    if not rdagent_path.exists():
        print(f"❌ RD-Agent 项目路径不存在: {rdagent_path}")
        return
    
    print(f"🔍 检查 RD-Agent 依赖安装情况...")
    print(f"📂 项目路径: {rdagent_path}\n")
    
    # 获取已安装的包
    installed = get_installed_packages()
    print(f"✅ 当前环境已安装 {len(installed)} 个包\n")
    
    # 检查主要依赖文件
    req_files = [
        ('核心依赖', rdagent_path / 'requirements.txt'),
        ('Torch依赖', rdagent_path / 'requirements' / 'torch.txt'),
        ('包管理依赖', rdagent_path / 'requirements' / 'package.txt'),
    ]
    
    all_missing = []
    
    for name, req_file in req_files:
        print(f"\n{'=' * 60}")
        print(f"📋 检查 {name}: {req_file.name}")
        print('=' * 60)
        
        requirements = parse_requirements(req_file)
        
        if not requirements:
            print(f"⚠️  文件不存在或为空")
            continue
        
        print(f"📦 需要检查 {len(requirements)} 个包:")
        
        missing = []
        installed_count = 0
        
        for pkg in requirements:
            normalized = normalize_package_name(pkg)
            # 检查多种可能的包名形式
            pkg_variants = [
                normalized,
                normalized.replace('-', '_'),
                pkg.lower(),
                pkg.lower().replace('_', '-'),
                pkg.lower().replace('-', '_'),
            ]
            
            is_installed = any(variant in installed for variant in pkg_variants)
            
            if is_installed:
                print(f"  ✅ {pkg}")
                installed_count += 1
            else:
                print(f"  ❌ {pkg} - 未安装")
                missing.append(pkg)
        
        print(f"\n📊 统计: {installed_count}/{len(requirements)} 已安装")
        
        if missing:
            all_missing.extend(missing)
            print(f"\n⚠️  缺失 {len(missing)} 个包:")
            for pkg in missing:
                print(f"     - {pkg}")
    
    # 总结
    print(f"\n\n{'=' * 60}")
    print("📊 总体检查结果")
    print('=' * 60)
    
    if all_missing:
        print(f"\n❌ 发现 {len(all_missing)} 个缺失的依赖包:\n")
        for pkg in sorted(set(all_missing)):
            print(f"  - {pkg}")
        
        print(f"\n💡 安装命令:")
        print(f"  cd {rdagent_path}")
        print(f"  pip install -r requirements.txt")
        print(f"  pip install -r requirements/torch.txt  # 如需 PyTorch 支持")
        print(f"\n或使用开发模式安装:")
        print(f"  cd {rdagent_path}")
        print(f"  pip install -e .")
        print(f"  pip install -e .[torch]  # 如需 PyTorch 支持")
    else:
        print(f"\n✅ 所有 RD-Agent 依赖已正确安装！")
    
    # 检查 rdagent 命令是否可用
    print(f"\n{'=' * 60}")
    print("🔧 检查 rdagent 命令")
    print('=' * 60)
    
    try:
        result = subprocess.run(
            ['rdagent', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ rdagent 命令可用")
        else:
            print("❌ rdagent 命令执行失败")
    except FileNotFoundError:
        print("❌ rdagent 命令未找到 - 需要安装 RD-Agent 包")
        print(f"\n💡 安装命令:")
        print(f"  cd {rdagent_path}")
        print(f"  pip install -e .")
    except Exception as e:
        print(f"⚠️  检查 rdagent 命令时出错: {e}")


if __name__ == '__main__':
    main()
