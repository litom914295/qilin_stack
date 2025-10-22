"""
Qlib数据验证脚本
"""
import sys
from pathlib import Path

try:
    import qlib
    from qlib.data import D
    print("✅ Qlib导入成功")
except ImportError:
    print("❌ Qlib未安装，请运行: pip install pyqlib")
    sys.exit(1)


def validate_qlib_data(provider_uri='~/.qlib/qlib_data/cn_data'):
    """验证Qlib数据"""
    print(f"\n{'='*60}")
    print("Qlib数据验证")
    print(f"{'='*60}\n")
    
    try:
        # 初始化Qlib
        print(f"1️⃣ 初始化Qlib: {provider_uri}")
        qlib.init(provider_uri=provider_uri)
        print("   ✅ 初始化成功")
        
        # 测试获取股票列表
        print("\n2️⃣ 获取股票列表...")
        instruments = D.instruments(market='csi300')
        print(f"   ✅ CSI300成分股数量: {len(instruments)}")
        print(f"   示例股票: {list(instruments[:5])}")
        
        # 测试获取特征数据
        print("\n3️⃣ 获取特征数据...")
        test_symbols = list(instruments[:3])
        features = D.features(
            test_symbols, 
            ['$close', '$volume', '$open', '$high', '$low'],
            start_time='2024-01-01',
            end_time='2024-06-30'
        )
        print(f"   ✅ 数据形状: {features.shape}")
        print(f"   数据预览:\n{features.head()}")
        
        # 检查数据完整性
        print("\n4️⃣ 检查数据完整性...")
        missing = features.isnull().sum().sum()
        total = features.size
        completeness = 1 - (missing / total)
        print(f"   总数据点: {total}")
        print(f"   缺失数据: {missing}")
        print(f"   完整度: {completeness:.2%}")
        
        if completeness > 0.95:
            print("   ✅ 数据质量良好")
        else:
            print("   ⚠️ 数据缺失较多，可能需要重新下载")
        
        # 测试日期范围
        print("\n5️⃣ 检查日期范围...")
        dates = features.index.get_level_values('datetime').unique()
        print(f"   最早日期: {dates.min()}")
        print(f"   最晚日期: {dates.max()}")
        print(f"   交易日数量: {len(dates)}")
        
        print(f"\n{'='*60}")
        print("✅ Qlib数据验证通过！")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        print("\n建议:")
        print("1. 下载Qlib数据:")
        print("   python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
        print("\n2. 检查数据路径是否正确")
        print("\n3. 确保有足够的磁盘空间")
        return False


def download_qlib_data():
    """下载Qlib数据"""
    print("\n开始下载Qlib数据...")
    print("这可能需要几分钟时间，请耐心等待...\n")
    
    import subprocess
    cmd = [
        sys.executable, '-m', 'qlib.run.get_data',
        'qlib_data',
        '--target_dir', str(Path.home() / '.qlib/qlib_data/cn_data'),
        '--region', 'cn'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 数据下载成功！")
            return True
        else:
            print(f"❌ 下载失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Qlib数据验证工具')
    parser.add_argument('--download', action='store_true', help='下载Qlib数据')
    parser.add_argument('--path', default='~/.qlib/qlib_data/cn_data', help='数据路径')
    
    args = parser.parse_args()
    
    if args.download:
        if download_qlib_data():
            validate_qlib_data(args.path)
    else:
        validate_qlib_data(args.path)
