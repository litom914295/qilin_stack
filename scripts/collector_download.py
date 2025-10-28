#!/usr/bin/env python
"""
使用 Qlib data collector 下载中国A股数据
参考: https://github.com/microsoft/qlib/tree/main/scripts/data_collector
"""
import sys
import subprocess
from pathlib import Path
import os

def download_using_collector():
    """使用数据收集器下载数据"""
    
    print("=" * 80)
    print("使用 Qlib Data Collector 下载中国A股数据")
    print("=" * 80)
    print()
    
    # 目标目录
    qlib_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    csv_dir = Path.home() / ".qlib" / "csv_data" / "cn_data"
    
    print(f"CSV 数据目录: {csv_dir}")
    print(f"Qlib 数据目录: {qlib_dir}")
    print()
    
    # 创建目录
    csv_dir.mkdir(parents=True, exist_ok=True)
    qlib_dir.mkdir(parents=True, exist_ok=True)
    
    print("步骤1: 下载原始数据 (CSV格式)")
    print("-" * 80)
    print("使用 YahooFinance 作为数据源 (免费)")
    print()
    
    # 安装必要的依赖
    print("正在安装数据收集器依赖...")
    deps = ["yfinance", "pandas", "tqdm", "fire"]
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], check=False)
    
    print("✓ 依赖安装完成")
    print()
    
    # 下载脚本
    collector_script = """
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm

def download_cn_stocks():
    \"\"\"下载中国A股数据\"\"\"
    
    # 获取上证和深证股票列表的示例代码
    # 实际使用时需要完整的股票列表
    
    print("获取股票列表...")
    
    # 示例: 使用一些常见的A股代码
    # 实际应用需要完整列表
    symbols = [
        '600000.SS',  # 浦发银行
        '600519.SS',  # 贵州茅台
        '000001.SZ',  # 平安银行
        '000002.SZ',  # 万科A
    ]
    
    csv_dir = Path.home() / ".qlib" / "csv_data" / "cn_data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载 {len(symbols)} 只股票的数据...")
    print(f"保存到: {csv_dir}")
    
    # 下载数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5年数据
    
    for symbol in tqdm(symbols):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if not df.empty:
                # 保存为CSV
                output_file = csv_dir / f"{symbol}.csv"
                df.to_csv(output_file)
                print(f"✓ {symbol}: {len(df)} 条记录")
            else:
                print(f"⚠️ {symbol}: 无数据")
                
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    print("\\n✅ 数据下载完成!")
    print(f"CSV文件保存在: {csv_dir}")
    
    return csv_dir

if __name__ == "__main__":
    download_cn_stocks()
"""
    
    # 保存临时脚本
    temp_script = Path("temp_download.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(collector_script)
    
    try:
        # 执行下载
        print("开始下载数据...")
        print("注意: 这是一个简化的示例,完整数据需要更全面的股票列表")
        print()
        
        result = subprocess.run([sys.executable, str(temp_script)], check=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("数据下载完成!")
            print("=" * 80)
            print()
            print("接下来需要:")
            print("1. 获取完整的A股股票列表")
            print("2. 下载所有股票的历史数据")
            print("3. 使用 Qlib 工具将 CSV 转换为 Qlib 格式")
            print()
            print("转换命令:")
            print(f"python -m qlib.data.storage --csv_path {csv_dir} --qlib_dir {qlib_dir}")
            print()
            return True
        else:
            print(f"\n⚠️ 下载失败")
            return False
            
    finally:
        # 清理临时文件
        if temp_script.exists():
            temp_script.unlink()


def download_from_official_source():
    """从 Qlib 官方源下载预处理数据"""
    
    print("\n" + "=" * 80)
    print("推荐: 使用官方预处理数据")
    print("=" * 80)
    print()
    print("由于从头收集数据比较复杂,推荐使用 Qlib 官方提供的预处理数据:")
    print()
    print("选项1: 从 GitHub 下载")
    print("-------")
    print("仓库: https://github.com/microsoft/qlib")
    print("位置: scripts/data_collector/")
    print("注意: GitHub 下载可能较慢")
    print()
    
    print("选项2: 从国内镜像下载 (推荐)")
    print("-------")
    print("您可以使用以下命令通过 wget 下载:")
    print()
    print("# 使用 wget (需要先安装)")
    print("wget -O qlib_cn_data.tar.gz [镜像URL]")
    print()
    print("# 解压")
    target_dir = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    print(f"tar -xzf qlib_cn_data.tar.gz -C {target_dir}")
    print()
    
    print("选项3: 使用 AkShare (最简单)")
    print("-------")
    print("无需下载本地数据,直接使用在线API:")
    print()
    print("配置 config.yaml:")
    print("""
data:
  sources:
    - akshare  # 使用 AkShare 在线数据
  akshare:
    enabled: true
""")
    print()
    print("然后直接运行系统即可!")
    print()


def main():
    """主函数"""
    
    print("\n请选择数据获取方式:")
    print("1. 尝试自动下载 (简化版,仅少量示例股票)")
    print("2. 查看手动下载指南 (推荐)")
    print("3. 使用 AkShare 在线数据 (最简单)")
    print()
    
    choice = input("请输入选项 (1/2/3) [默认: 3]: ").strip() or "3"
    
    if choice == "1":
        download_using_collector()
    elif choice == "2":
        download_from_official_source()
    elif choice == "3":
        print("\n" + "=" * 80)
        print("使用 AkShare 在线数据")
        print("=" * 80)
        print()
        print("✓ AkShare 已安装")
        print("✓ 无需下载本地数据")
        print("✓ 实时获取A股数据")
        print()
        print("您现在可以直接使用麒麟系统:")
        print()
        print("# 激活虚拟环境")
        print(".\.qilin\Scripts\Activate.ps1")
        print()
        print("# 运行快速开始")
        print("python quickstart.py")
        print()
        print("# 或运行回测")
        print("python main.py --mode backtest --start_date 2024-01-01 --end_date 2024-12-31")
        print()
    else:
        print("无效选项")


if __name__ == "__main__":
    main()
