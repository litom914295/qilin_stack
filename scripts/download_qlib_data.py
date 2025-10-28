"""
下载Qlib数据到项目目录
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from qlib.data import D
    from qlib.config import REG_CN
    import qlib
    
    print("=" * 70)
    print("Qlib数据下载工具")
    print("=" * 70)
    
    # 目标目录
    target_dir = project_root / "data" / "qlib_data" / "cn_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n目标目录: {target_dir}")
    print("开始下载数据...\n")
    
    # 使用qlib的数据下载功能
    import subprocess
    
    # 尝试使用qlib命令行工具
    cmd = [
        sys.executable,
        "-m", "qlib.data.download",
        "--target_dir", str(target_dir),
        "--region", "cn"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 数据下载成功!")
        print(f"数据位置: {target_dir}")
    else:
        print("使用备用下载方法...")
        # 备用方法：直接从URL下载
        import requests
        import zipfile
        import io
        
        url = "http://fintech.msra.cn/stock_data/downloads/latest/qlib_bin.tar.gz"
        print(f"从 {url} 下载...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"文件大小: {total_size / (1024**3):.2f} GB")
        print("正在下载（这可能需要10-30分钟）...")
        
        # 下载并解压
        with open(target_dir.parent / "qlib_bin.tar.gz", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("下载完成，正在解压...")
        import tarfile
        with tarfile.open(target_dir.parent / "qlib_bin.tar.gz") as tar:
            tar.extractall(target_dir)
        
        print("✅ 数据安装完成!")
        
except ImportError as e:
    print(f"错误: {e}")
    print("\n请先安装qlib: pip install qlib")
except Exception as e:
    print(f"下载失败: {e}")
    print("\n请尝试手动下载:")
    print("1. 访问: https://github.com/microsoft/qlib/tree/main/scripts")
    print("2. 下载数据文件")
    print(f"3. 解压到: {project_root / 'data' / 'qlib_data' / 'cn_data'}")
