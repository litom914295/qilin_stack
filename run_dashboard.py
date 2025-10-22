"""
一键启动智能交易分析系统
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("="*60)
    print("🚀 启动智能交易分析系统")
    print("="*60)
    
    # 获取项目路径
    project_root = Path(__file__).parent
    dashboard_path = project_root / "app" / "web" / "unified_agent_dashboard.py"
    
    # 检查文件是否存在
    if not dashboard_path.exists():
        print(f"❌ 错误: 找不到文件 {dashboard_path}")
        return 1
    
    print(f"📁 项目路径: {project_root}")
    print(f"📊 仪表板路径: {dashboard_path}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # 构建命令
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    
    print("\n正在启动 Streamlit 应用...")
    print(f"命令: {' '.join(cmd)}")
    print("\n" + "="*60)
    print("✅ 系统启动成功！")
    print("🌐 请在浏览器中访问: http://localhost:8501")
    print("📌 按 Ctrl+C 停止服务")
    print("="*60 + "\n")
    
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
