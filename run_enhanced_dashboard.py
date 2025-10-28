"""
启动麒麟量化系统增强版Dashboard
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("=" * 60)
    print("🦄 启动麒麟量化增强版Dashboard")
    print("=" * 60)
    
    # 获取项目路径
    project_root = Path(__file__).parent
    dashboard_path = project_root / "app" / "web" / "enhanced_dashboard.py"
    
    # 检查文件是否存在
    if not dashboard_path.exists():
        print(f"❌ 错误: 找不到文件 {dashboard_path}")
        return 1
    
    print(f"📁 项目路径: {project_root}")
    print(f"📊 Dashboard路径: {dashboard_path}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # 构建命令
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    
    print("\n正在启动增强版 Streamlit Dashboard...")
    print(f"命令: {' '.join(cmd)}")
    print("\n" + "=" * 60)
    print("✅ 系统启动成功！")
    print("🌐 请在浏览器中访问: http://localhost:8501")
    print("\n🎯 增强版功能包括:")
    print("  - 📈 一进二涨停板选股")
    print("  - 🌡️ 市场风格动态切换")
    print("  - 📊 策略回测系统")
    print("  - 👁️ 实时市场监控")
    print("  - 🤖 多智能体协作")
    print("  - ⚙️ 系统配置管理")
    print("\n📌 按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")
    
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
