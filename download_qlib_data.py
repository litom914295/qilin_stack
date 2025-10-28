#!/usr/bin/env python
"""
Qlib数据下载脚本
下载中国A股市场的日线数据
"""
import os
from pathlib import Path

def download_qlib_data():
    """下载Qlib中国A股数据"""
    target_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    
    print(f"准备下载 Qlib 中国A股数据到: {target_dir}")
    print("数据规模约 12-20GB,请耐心等待...")
    print("-" * 60)
    
    try:
        # 方法1: 使用qlib.contrib
        try:
            from qlib.contrib.data.handler import GetData
            print("使用 qlib.contrib.data 下载数据...")
            GetData().qlib_data(target_dir=str(target_dir), region="cn")
            print("数据下载完成!")
            return True
        except (ImportError, AttributeError) as e:
            print(f"方法1失败: {e}")
        
        # 方法2: 使用命令行工具
        try:
            import subprocess
            print("\n尝试使用命令行工具下载...")
            cmd = f"python -m qlib.run.get_data qlib_data --target_dir {target_dir} --region cn"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("数据下载完成!")
                return True
            else:
                print(f"命令行下载失败: {result.stderr}")
        except Exception as e:
            print(f"方法2失败: {e}")
        
        # 方法3: 手动指引
        print("\n" + "=" * 60)
        print("自动下载失败,请按以下步骤手动下载:")
        print("=" * 60)
        print(f"1. 目标目录: {target_dir}")
        print(f"2. 创建目录(如不存在): mkdir -p {target_dir}")
        print("3. 访问 Qlib 数据源获取数据:")
        print("   - GitHub: https://github.com/microsoft/qlib/tree/main/scripts/data_collector")
        print("   - 百度云盘: 参见 Qlib 官方文档")
        print("\n或者尝试以下Python代码:")
        print("-" * 60)
        print("""
import qlib
from qlib.data import D
from qlib.config import REG_CN

# 初始化qlib(首次会尝试下载数据)
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)
        """)
        print("=" * 60)
        
        # 检查是否已有数据
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"\n注意: 目录 {target_dir} 已存在且包含文件")
            print("可能数据已经下载,请验证数据完整性")
            return True
        
        return False
        
    except Exception as e:
        print(f"下载过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_qlib_data()
    if success:
        print("\n✅ 数据准备就绪!")
    else:
        print("\n⚠️  请参考上述提示手动准备数据")
