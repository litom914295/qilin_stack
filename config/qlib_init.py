"""
Qlib初始化配置
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# Qlib数据目录（项目内）
QLIB_DATA_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"

def init_qlib():
    """初始化Qlib"""
    try:
        import qlib
        
        # 检查数据目录是否存在
        if not QLIB_DATA_DIR.exists():
            print(f"⚠️  Qlib数据目录不存在: {QLIB_DATA_DIR}")
            print("请使用AKShare作为数据源，或手动下载Qlib数据")
            return False
        
        # 初始化Qlib
        qlib.init(
            provider_uri=str(QLIB_DATA_DIR),
            region="cn"
        )
        
        print(f"✅ Qlib已初始化")
        print(f"   数据目录: {QLIB_DATA_DIR}")
        return True
        
    except ImportError:
        print("⚠️  Qlib未安装")
        return False
    except Exception as e:
        print(f"⚠️  Qlib初始化失败: {e}")
        return False

if __name__ == "__main__":
    init_qlib()
