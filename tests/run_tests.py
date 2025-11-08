"""
测试运行脚本 (Test Runner)
Task 15: 自动化测试与口径校验

功能:
- 运行单元测试/集成测试/E2E测试
- 生成测试报告和覆盖率报告
- 基准对齐校验
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_unit_tests(verbose=False):
    """运行单元测试"""
    print("=" * 60)
    print("运行单元测试...")
    print("=" * 60)
    
    cmd = [
        "pytest",
        "tests/unit/",
        "-v" if verbose else "",
        "-m", "unit",
        "--tb=short",
    ]
    
    cmd = [c for c in cmd if c]  # 移除空字符串
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_integration_tests(verbose=False):
    """运行集成测试"""
    print("\n" + "=" * 60)
    print("运行集成测试...")
    print("=" * 60)
    
    cmd = [
        "pytest",
        "tests/integration/",
        "-v" if verbose else "",
        "-m", "integration",
        "--tb=short",
    ]
    
    cmd = [c for c in cmd if c]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_e2e_tests(verbose=False):
    """运行 E2E 测试"""
    print("\n" + "=" * 60)
    print("运行 E2E 测试...")
    print("=" * 60)
    
    cmd = [
        "pytest",
        "tests/integration/",
        "-v" if verbose else "",
        "-m", "e2e",
        "--tb=short",
    ]
    
    cmd = [c for c in cmd if c]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_all_tests(verbose=False):
    """运行所有测试"""
    print("=" * 60)
    print("运行所有测试...")
    print("=" * 60)
    
    cmd = [
        "pytest",
        "tests/",
        "-v" if verbose else "",
        "--tb=short",
    ]
    
    cmd = [c for c in cmd if c]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_with_coverage(verbose=False):
    """运行测试并生成覆盖率报告"""
    print("=" * 60)
    print("运行测试 + 覆盖率分析...")
    print("=" * 60)
    
    cmd = [
        "pytest",
        "tests/",
        "-v" if verbose else "",
        "--cov=qlib_enhanced",
        "--cov=config",
        "--cov=web",
        "--cov-report=html",
        "--cov-report=term",
        "--tb=short",
    ]
    
    cmd = [c for c in cmd if c]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print("\n覆盖率报告已生成: htmlcov/index.html")
    
    return result.returncode == 0


def prepare_test_data():
    """准备测试数据"""
    print("=" * 60)
    print("准备测试数据...")
    print("=" * 60)
    
    test_data_script = Path(__file__).parent / "data" / "prepare_test_data.py"
    
    if not test_data_script.exists():
        print(f"警告: 测试数据生成脚本不存在: {test_data_script}")
        return False
    
    result = subprocess.run(
        [sys.executable, str(test_data_script)],
        cwd=Path(__file__).parent.parent
    )
    
    return result.returncode == 0


def run_baseline_alignment():
    """运行基准对齐测试"""
    print("\n" + "=" * 60)
    print("运行基准对齐测试...")
    print("=" * 60)
    
    cmd = [
        "pytest",
        "tests/integration/test_qlib_baseline_alignment.py",
        "-v",
        "-m", "integration",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    # 检查报告
    report_path = Path(__file__).parent / "reports" / "baseline_alignment_report.json"
    if report_path.exists():
        import json
        with open(report_path) as f:
            report = json.load(f)
        
        print("\n" + "=" * 60)
        print("基准对齐结果:")
        print(f"  状态: {report['overall_status']}")
        print(f"  Qlib 版本: {report['qlib_version']}")
        print(f"  误差阈值: {report['threshold'] * 100}%")
        print("=" * 60)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Qilin 测试运行器")
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["unit", "integration", "e2e", "all", "coverage", "baseline", "prepare"],
        help="测试类型 (默认: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["PYTEST_SEED"] = "42"  # 固定随机种子
    os.environ["QILIN_TEST_MODE"] = "1"
    
    # 运行测试
    success = False
    
    if args.test_type == "prepare":
        success = prepare_test_data()
    elif args.test_type == "unit":
        success = run_unit_tests(args.verbose)
    elif args.test_type == "integration":
        success = run_integration_tests(args.verbose)
    elif args.test_type == "e2e":
        success = run_e2e_tests(args.verbose)
    elif args.test_type == "coverage":
        success = run_with_coverage(args.verbose)
    elif args.test_type == "baseline":
        success = run_baseline_alignment()
    elif args.test_type == "all":
        success = run_all_tests(args.verbose)
    
    # 返回状态
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
