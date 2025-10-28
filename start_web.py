#!/usr/bin/env python
"""
麒麟量化系统 - Web 界面启动脚本
一键启动 Streamlit 交互式仪表板
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函数"""
    print("=" * 70)
    print("🦄 麒麟量化系统 - Web 交互式仪表板")
    print("=" * 70)
    
    # 获取项目路径
    project_root = Path(__file__).parent
    dashboard_path = project_root / "web" / "unified_dashboard.py"
    
    # 检查文件是否存在
    if not dashboard_path.exists():
        print(f"❌ 错误: 找不到文件 {dashboard_path}")
        print(f"📁 项目路径: {project_root}")
        return 1
    
    print(f"📁 项目路径: {project_root}")
    print(f"📊 仪表板: {dashboard_path}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # 构建命令
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    
    print("\n正在启动 Streamlit Web 界面...")
    print(f"💡 命令: {' '.join(cmd)}")
    print("\n" + "=" * 70)
    print("✅ 系统启动成功！")
    print("🌐 请在浏览器中访问: http://localhost:8501")
    print("\n🎯 主要功能:")
    print("  - 📈 一进二涨停板智能选股")
    print("  - 🌡️ 市场风格动态识别")
    print("  - 📊 策略回测与绩效分析")
    print("  - 🤖 多 Agent 协作决策")
    print("  - ⚙️ 实时参数调整")
    print("  - 📉 可视化图表分析")
    print("\n📌 按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    try:
        # 运行 Streamlit
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
