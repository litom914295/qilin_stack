#!/usr/bin/env python
"""
Qlib 中国A股数据下载脚本
使用官方推荐的方法下载数据
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil

def download_qlib_data():
    """下载 Qlib 中国 A 股数据"""
    
    # 目标目录
    target_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    
    print("=" * 80)
    print("麒麟量化系统 - Qlib 数据下载工具")
    print("=" * 80)
    print(f"目标目录: {target_dir}")
    print(f"数据规模: 约 12-20 GB")
    print(f"预计时间: 根据网络速度,可能需要 30分钟 ~ 2小时")
    print("=" * 80)
    print()
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 目标目录已创建: {target_dir}")
    
    # 方法1: 使用 qlib 自带的下载脚本(推荐)
    print("\n方法1: 尝试使用 Qlib 官方下载脚本...")
    print("-" * 80)
    
    try:
        # 检查 qlib scripts 目录
        import qlib
        qlib_path = Path(qlib.__file__).parent
        scripts_dir = qlib_path / "scripts"
        
        if scripts_dir.exists():
            print(f"找到 Qlib scripts 目录: {scripts_dir}")
            
            # 查找数据下载脚本
            get_data_script = scripts_dir / "get_data.py"
            if get_data_script.exists():
                print(f"✓ 找到下载脚本: {get_data_script}")
                print("\n开始下载数据...")
                print("提示: 如果下载速度慢,可以按 Ctrl+C 中断,尝试方法2")
                print("-" * 80)
                
                # 执行下载命令
                cmd = [
                    sys.executable,
                    str(get_data_script),
                    "qlib_data",
                    "--target_dir", str(target_dir),
                    "--region", "cn"
                ]
                
                print(f"执行命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=str(scripts_dir))
                
                if result.returncode == 0:
                    print("\n✅ 数据下载完成!")
                    verify_data(target_dir)
                    return True
                else:
                    print(f"\n⚠️ 下载失败,返回码: {result.returncode}")
            else:
                print(f"⚠️ 未找到 get_data.py 脚本")
        else:
            print(f"⚠️ 未找到 scripts 目录")
            
    except Exception as e:
        print(f"⚠️ 方法1 失败: {e}")
    
    # 方法2: 使用 wget 或 curl 下载预处理包
    print("\n\n方法2: 从官方镜像下载预处理数据包")
    print("-" * 80)
    print("由于自动下载失败,建议手动下载预处理数据包:")
    print()
    print("步骤:")
    print("1. 访问 Qlib 数据仓库:")
    print("   https://github.com/microsoft/qlib/tree/main/scripts/data_collector/cn_data")
    print()
    print("2. 下载对应的数据包:")
    print("   - cn_data.zip (约 12-20 GB)")
    print("   - 或使用百度网盘(参见 Qlib 文档)")
    print()
    print("3. 解压到目标目录:")
    print(f"   {target_dir}")
    print()
    print("4. 验证数据:")
    print("   python scripts/verify_qlib_data.py")
    print()
    
    # 方法3: 使用命令行工具
    print("\n方法3: 使用命令行手动下载")
    print("-" * 80)
    print("您也可以使用以下命令手动下载:")
    print()
    print("# Windows PowerShell:")
    print(f'python -m qlib.run.get_data qlib_data --target_dir "{target_dir}" --region cn')
    print()
    print("# 或使用 Python 代码:")
    print("""
import qlib
from qlib.data.data import GetData

# 下载数据
GetData().qlib_data(
    target_dir='~/.qlib/qlib_data/cn_data',
    region='cn',
    delete_old=False
)
""")
    
    return False


def verify_data(target_dir):
    """验证下载的数据"""
    print("\n" + "=" * 80)
    print("验证数据...")
    print("=" * 80)
    
    try:
        # 检查目录是否包含文件
        if not target_dir.exists():
            print(f"❌ 目录不存在: {target_dir}")
            return False
        
        files = list(target_dir.rglob("*"))
        if not files:
            print(f"❌ 目录为空: {target_dir}")
            return False
        
        print(f"✓ 找到 {len(files)} 个文件")
        
        # 检查必要的文件结构
        required_dirs = ['features', 'calendars', 'instruments']
        for dir_name in required_dirs:
            dir_path = target_dir / dir_name
            if dir_path.exists():
                print(f"✓ {dir_name}/ 存在")
            else:
                print(f"⚠️ {dir_name}/ 不存在 (可能数据不完整)")
        
        # 尝试初始化 qlib
        print("\n测试 Qlib 初始化...")
        import qlib
        from qlib.data import D
        
        try:
            qlib.init(provider_uri=str(target_dir), region='cn')
            print("✓ Qlib 初始化成功")
            
            # 获取交易日历
            cal = D.calendar(start_time='2024-01-01', end_time='2024-01-10')
            print(f"✓ 获取到交易日历: {len(cal)} 个交易日")
            
            # 获取股票列表
            instruments = D.instruments(market='all')
            print(f"✓ 获取到股票列表: {len(instruments)} 只股票")
            
            print("\n✅ 数据验证通过!")
            return True
            
        except Exception as e:
            print(f"⚠️ Qlib 初始化失败: {e}")
            print("数据可能不完整或格式不正确")
            return False
            
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        return False


def main():
    """主函数"""
    try:
        success = download_qlib_data()
        
        if not success:
            print("\n" + "=" * 80)
            print("自动下载失败,但不用担心!")
            print("=" * 80)
            print("\n麒麟系统支持多种数据源,您可以:")
            print("1. 使用 AkShare 在线数据 (推荐,无需下载)")
            print("2. 使用 Tushare 数据 (需要 token)")
            print("3. 手动下载 Qlib 数据包后解压")
            print()
            print("查看详细说明: cat QLIB_DATA_GUIDE.md")
            
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 下载已取消")
        print("您可以稍后重新运行此脚本继续下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
