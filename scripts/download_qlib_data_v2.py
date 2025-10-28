"""
使用Qlib官方API下载数据
"""
import sys
import os
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent
target_dir = project_root / "data" / "qlib_data" / "cn_data"

print("=" * 70)
print("Qlib数据下载工具 v2")
print("=" * 70)
print(f"\n目标目录: {target_dir}")
print("开始下载中国A股数据...")
print("预计大小: ~12-20GB")
print("预计时间: 10-30分钟\n")

try:
    # 方法1: 使用qlib.data.GetData
    try:
        from qlib.data import GetData
        
        gd = GetData()
        print("使用 GetData API 下载...")
        
        gd.qlib_data(
            target_dir=str(target_dir),
            region="cn",
            interval="1d",
            delete_old=False
        )
        
        print("\n✅ 数据下载成功!")
        print(f"数据位置: {target_dir}")
        
    except Exception as e1:
        print(f"方法1失败: {e1}")
        print("\n尝试方法2...")
        
        # 方法2: 直接调用download
        try:
            import subprocess
            
            # 尝试使用scripts目录下的下载脚本
            qlib_path = Path(sys.executable).parent.parent / "Scripts"
            download_script = qlib_path / "get_data.py"
            
            if download_script.exists():
                cmd = [
                    sys.executable,
                    str(download_script),
                    "qlib_data",
                    "--target_dir", str(target_dir),
                    "--region", "cn"
                ]
                
                print(f"执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)
                
                print("\n✅ 数据下载成功!")
                
            else:
                raise FileNotFoundError("下载脚本不存在")
                
        except Exception as e2:
            print(f"方法2失败: {e2}")
            print("\n尝试方法3...")
            
            # 方法3: 使用wget下载
            try:
                import urllib.request
                import tarfile
                
                # 备用下载地址
                urls = [
                    "https://github.com/microsoft/qlib/releases/download/data/qlib_bin.tar.gz",
                    "http://fintech.msra.cn/stock_data/downloads/latest/qlib_bin.tar.gz"
                ]
                
                tar_file = target_dir.parent / "qlib_bin.tar.gz"
                
                for url in urls:
                    try:
                        print(f"尝试从 {url} 下载...")
                        
                        def reporthook(count, block_size, total_size):
                            percent = int(count * block_size * 100 / total_size)
                            sys.stdout.write(f"\r下载进度: {percent}%")
                            sys.stdout.flush()
                        
                        urllib.request.urlretrieve(url, tar_file, reporthook)
                        print("\n下载完成!")
                        
                        # 解压
                        print("正在解压...")
                        with tarfile.open(tar_file, 'r:gz') as tar:
                            tar.extractall(target_dir)
                        
                        # 删除压缩文件
                        tar_file.unlink()
                        
                        print("\n✅ 数据安装成功!")
                        print(f"数据位置: {target_dir}")
                        break
                        
                    except Exception as e:
                        print(f"\n从 {url} 下载失败: {e}")
                        continue
                
            except Exception as e3:
                print(f"方法3失败: {e3}")
                
                print("\n" + "=" * 70)
                print("❌ 所有自动下载方法都失败了")
                print("=" * 70)
                print("\n手动下载步骤:")
                print("1. 访问: https://github.com/microsoft/qlib")
                print("2. 或搜索'qlib data download'找国内镜像")
                print(f"3. 下载数据后解压到: {target_dir}")
                print("\n或者直接使用 AKShare 数据源（推荐）")
                
except KeyboardInterrupt:
    print("\n\n用户中断下载")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
