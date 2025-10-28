"""
启动统一的麒麟量化平台Dashboard
集成Qlib、RD-Agent、TradingAgents三大开源项目
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("=" * 70)
    print("🦄 启动麒麟量化统一平台 - 集成三大开源项目")
    print("=" * 70)
    
    # 获取项目路径
    project_root = Path(__file__).parent
    dashboard_path = project_root / "app" / "web" / "unified_dashboard.py"
    
    # 检查文件是否存在
    if not dashboard_path.exists():
        print(f"❌ 错误: 找不到文件 {dashboard_path}")
        return 1
    
    print(f"\n📁 项目路径: {project_root}")
    print(f"📊 Dashboard路径: {dashboard_path}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # 构建命令
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    
    print("\n🚀 正在启动统一Dashboard...")
    print(f"命令: {' '.join(cmd)}")
    print("\n" + "=" * 70)
    print("✅ 系统启动成功！")
    print("🌐 请在浏览器中访问: http://localhost:8501")
    print("\n🎯 集成功能模块:")
    print("  📊 Qlib量化平台")
    print("     - 股票数据查询")
    print("     - Alpha158因子计算")
    print("     - 模型训练")
    print("     - 策略回测")
    print("\n  🤖 RD-Agent自动研发")
    print("     - 自动因子生成")
    print("     - 模型自动优化")
    print("     - 策略自动生成")
    print("     - 研究循环")
    print("\n  👥 TradingAgents多智能体")
    print("     - 单股智能分析")
    print("     - 批量股票分析")
    print("     - 多智能体辩论")
    print("     - 会员管理系统")
    print("\n📌 按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    try:
        # 运行Streamlit
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\n👋 系统已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 启动失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
