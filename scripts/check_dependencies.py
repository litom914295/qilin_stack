#!/usr/bin/env python
"""
依赖检查脚本 - 在虚拟环境中运行
检查所有必需和可选依赖的安装状态
"""
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

def check_package(package_name: str) -> Tuple[bool, str]:
    """
    检查包是否已安装
    
    Returns:
        (is_installed, version)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # 解析版本号
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    return True, version
            return True, "unknown"
        else:
            return False, ""
    except Exception as e:
        return False, str(e)


def check_dependencies():
    """检查所有依赖"""
    
    print("=" * 80)
    print("麒麟量化系统 - 依赖检查报告")
    print("=" * 80)
    print()
    
    # 定义依赖分类
    dependencies = {
        "🔴 核心依赖（必需）": [
            "streamlit",
            "pandas",
            "numpy",
            "plotly",
            "pyyaml",
            "pyqlib",  # Qlib
        ],
        "🟡 量化功能（推荐）": [
            "akshare",
            "tushare",
            "ta-lib",  # 技术指标（可能需要预编译）
            "scikit-learn",
            "lightgbm",
            "xgboost",
            "catboost",
        ],
        "🟢 深度学习（可选）": [
            "torch",
            "tensorflow",
        ],
        "🔵 高级功能（可选）": [
            "mlflow",
            "optuna",
            "rdagent",
            "kaggle",
        ],
        "⚪ 其他工具（可选）": [
            "matplotlib",
            "seaborn",
            "scipy",
            "requests",
        ]
    }
    
    # 统计结果
    stats = {
        "total": 0,
        "installed": 0,
        "missing": 0
    }
    
    missing_packages = []
    
    # 遍历检查
    for category, packages in dependencies.items():
        print(f"\n{category}")
        print("-" * 80)
        
        for package in packages:
            stats["total"] += 1
            is_installed, version = check_package(package)
            
            if is_installed:
                stats["installed"] += 1
                status = f"✅ {package:30s} v{version}"
                print(status)
            else:
                stats["missing"] += 1
                status = f"❌ {package:30s} 未安装"
                print(status)
                missing_packages.append(package)
    
    # 打印统计
    print("\n" + "=" * 80)
    print("检查统计")
    print("=" * 80)
    print(f"总计: {stats['total']}")
    print(f"已安装: {stats['installed']} ({stats['installed']/stats['total']*100:.1f}%)")
    print(f"未安装: {stats['missing']} ({stats['missing']/stats['total']*100:.1f}%)")
    
    # 安装建议
    if missing_packages:
        print("\n" + "=" * 80)
        print("安装建议")
        print("=" * 80)
        
        # 分类缺失的包
        core_missing = [p for p in missing_packages if p in dependencies["🔴 核心依赖（必需）"]]
        recommended_missing = [p for p in missing_packages if p in dependencies["🟡 量化功能（推荐）"]]
        optional_missing = [p for p in missing_packages if p not in core_missing and p not in recommended_missing]
        
        if core_missing:
            print("\n⚠️  核心依赖缺失（必须安装）:")
            print(f"pip install {' '.join(core_missing)}")
        
        if recommended_missing:
            print("\n💡 推荐安装（提升功能）:")
            print(f"pip install {' '.join(recommended_missing)}")
        
        if optional_missing:
            print("\n📦 可选安装（按需）:")
            print(f"pip install {' '.join(optional_missing)}")
    
    else:
        print("\n🎉 所有依赖都已安装！")
    
    # 特殊检查
    print("\n" + "=" * 80)
    print("特殊配置检查")
    print("=" * 80)
    
    # 检查Kaggle配置
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_config.exists():
        print(f"✅ Kaggle配置文件: {kaggle_config}")
    else:
        print(f"❌ Kaggle配置文件未找到: {kaggle_config}")
        print("   提示: 从 https://www.kaggle.com/settings 下载 kaggle.json")
    
    # 检查Qlib数据
    qlib_data_paths = [
        Path.home() / ".qlib" / "qlib_data" / "cn_data",
        Path("G:/test/qlib/qlib_data/cn_data"),
    ]
    
    qlib_data_found = False
    for path in qlib_data_paths:
        if path.exists():
            print(f"✅ Qlib数据目录: {path}")
            qlib_data_found = True
            break
    
    if not qlib_data_found:
        print("❌ Qlib数据目录未找到")
        print("   提示: 运行 python download_qlib_data.py 下载数据")
    
    # 检查虚拟环境
    print("\n" + "=" * 80)
    print("环境信息")
    print("=" * 80)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print(f"✅ 运行在虚拟环境: {sys.prefix}")
    else:
        print("⚠️  未检测到虚拟环境")
        print("   建议: 创建并激活虚拟环境")
        print("   python -m venv venv")
        print("   venv\\Scripts\\activate  # Windows")
        print("   source venv/bin/activate  # Linux/Mac")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)
    
    return stats["missing"] == 0


if __name__ == "__main__":
    try:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n检查已取消")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
