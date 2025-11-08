"""
Qlib 统一配置中心 (Qlib Configuration Center)
任务 3: 统一初始化与配置中心

功能:
- 统一 qlib.init() 调用入口
- 离线/在线模式切换与自动回退
- 缓存策略配置 (expression/dataset/redis)
- 版本检测与兼容性校验
- Windows 路径兼容
- 配置文件/环境变量/命令行参数三级优先级
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QlibMode(Enum):
    """Qlib 运行模式"""
    OFFLINE = "offline"  # 离线模式 (本地数据)
    ONLINE = "online"    # 在线模式 (Qlib-Server)
    AUTO = "auto"        # 自动模式 (优先在线,失败回退离线)


@dataclass
class QlibConfig:
    """Qlib 配置类"""
    
    # ============================================================================
    # 基础配置
    # ============================================================================
    mode: QlibMode = QlibMode.OFFLINE
    region: str = "cn"  # cn, us
    
    # ============================================================================
    # 数据路径配置 (离线模式)
    # ============================================================================
    provider_uri: Optional[str] = None  # 数据目录路径
    provider_uri_map: Optional[Dict[str, str]] = None  # 多频率数据 (如 {'day': path1, '1min': path2})
    mount_path: Optional[str] = None  # 挂载路径
    
    # ============================================================================
    # 在线模式配置 (Qlib-Server)
    # ============================================================================
    server_host: str = "127.0.0.1"
    server_port: int = 9710
    server_timeout: int = 30  # 秒
    server_token: Optional[str] = None  # 鉴权 token
    
    # ============================================================================
    # 缓存配置
    # ============================================================================
    # Expression Cache (计算缓存)
    expression_cache: Optional[str] = None  # None 或 'DiskExpressionCache' 或自定义类
    expression_provider_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "dir": ".qlib_cache/expression_cache",
        "max_workers": 1
    })
    
    # Dataset Cache (数据集缓存)
    dataset_cache: Optional[str] = None  # None 或 'DiskDatasetCache' 或自定义类
    dataset_provider_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "dir": ".qlib_cache/dataset_cache",
        "max_workers": 1
    })
    
    # Redis Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    redis_password: Optional[str] = None
    redis_enabled: bool = False  # 默认关闭 Redis
    
    # ============================================================================
    # 高级配置
    # ============================================================================
    kernels: int = 1  # 并行处理器数量
    logging_level: str = "INFO"
    auto_mount: bool = True
    flask_server: bool = False
    flask_port: int = 9710
    
    # 版本兼容性
    qlib_reset_version: Optional[str] = None  # 手动覆盖 Qlib 版本 (用于兼容旧版 Qlib-Server)
    
    def to_qlib_init_kwargs(self) -> Dict[str, Any]:
        """
        转换为 qlib.init() 的参数字典
        
        Returns:
            qlib.init() 参数字典
        """
        kwargs = {
            "region": self.region,
            "logging_level": self.logging_level,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "kernels": self.kernels,
            "auto_mount": self.auto_mount,
            "flask_server": self.flask_server,
            "flask_port": self.flask_port,
        }
        
        # Redis 密码 (可选)
        if self.redis_password:
            kwargs["redis_password"] = self.redis_password
        
        # 版本覆盖 (可选)
        if self.qlib_reset_version:
            kwargs["qlib_reset_version"] = self.qlib_reset_version
        
        # 离线模式: provider_uri
        if self.mode == QlibMode.OFFLINE:
            if self.provider_uri_map:
                kwargs["provider_uri"] = self.provider_uri_map
            elif self.provider_uri:
                kwargs["provider_uri"] = self.provider_uri
            else:
                raise ValueError("离线模式必须提供 provider_uri 或 provider_uri_map")
        
        # 在线模式: Qlib-Server
        elif self.mode == QlibMode.ONLINE:
            kwargs["provider_uri"] = f"http://{self.server_host}:{self.server_port}"
            if self.server_token:
                kwargs["provider_kwargs"] = {"token": self.server_token}
        
        # 缓存配置
        if self.expression_cache:
            kwargs["expression_cache"] = self.expression_cache
            if self.expression_provider_kwargs:
                kwargs["expression_provider_kwargs"] = self.expression_provider_kwargs
        
        if self.dataset_cache:
            kwargs["dataset_cache"] = self.dataset_cache
            if self.dataset_provider_kwargs:
                kwargs["dataset_provider_kwargs"] = self.dataset_provider_kwargs
        
        return kwargs


class QlibInitializer:
    """Qlib 统一初始化管理器"""
    
    _instance = None
    _initialized = False
    _config: Optional[QlibConfig] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def init(cls, config: Optional[QlibConfig] = None, **kwargs) -> Tuple[bool, str]:
        """
        统一初始化 Qlib
        
        Args:
            config: QlibConfig 配置对象
            **kwargs: 直接传递给 qlib.init() 的参数 (会覆盖 config 中的设置)
        
        Returns:
            (success: bool, message: str)
        """
        # 避免重复初始化
        if cls._initialized:
            logger.warning("Qlib 已经初始化过,跳过重复初始化")
            return True, "Qlib 已初始化 (跳过重复)"
        
        # 使用默认配置
        if config is None:
            config = cls._get_default_config()
        
        cls._config = config
        
        # 尝试初始化
        try:
            import qlib
            
            # 准备参数
            init_kwargs = config.to_qlib_init_kwargs()
            init_kwargs.update(kwargs)  # 命令行参数优先级最高
            
            # 根据模式初始化
            if config.mode == QlibMode.AUTO:
                success, message = cls._init_with_fallback(qlib, init_kwargs, config)
            else:
                success, message = cls._init_direct(qlib, init_kwargs, config)
            
            if success:
                cls._initialized = True
                cls._log_init_success(qlib, config)
            
            return success, message
        
        except ImportError:
            error_msg = "❌ Qlib 未安装,请运行: pip install pyqlib"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"❌ Qlib 初始化失败: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    @classmethod
    def _init_direct(cls, qlib, init_kwargs: Dict, config: QlibConfig) -> Tuple[bool, str]:
        """直接初始化 (无回退)"""
        try:
            qlib.init(**init_kwargs)
            
            mode_name = "离线" if config.mode == QlibMode.OFFLINE else "在线"
            return True, f"✅ Qlib {mode_name}模式初始化成功"
        
        except Exception as e:
            return False, f"❌ 初始化失败: {e}"
    
    @classmethod
    def _init_with_fallback(cls, qlib, init_kwargs: Dict, config: QlibConfig) -> Tuple[bool, str]:
        """
        自动模式: 先尝试在线,失败则回退到离线
        """
        # 1. 尝试在线模式
        logger.info("尝试在线模式 (Qlib-Server)...")
        online_kwargs = init_kwargs.copy()
        online_kwargs["provider_uri"] = f"http://{config.server_host}:{config.server_port}"
        
        try:
            import requests
            # 健康检查
            response = requests.get(
                f"http://{config.server_host}:{config.server_port}/health",
                timeout=config.server_timeout
            )
            if response.status_code == 200:
                qlib.init(**online_kwargs)
                logger.info("✅ 在线模式连接成功")
                return True, "✅ Qlib 在线模式初始化成功"
        except Exception as e:
            logger.warning(f"在线模式失败: {e}, 回退到离线模式...")
        
        # 2. 回退到离线模式
        logger.info("回退到离线模式...")
        offline_kwargs = init_kwargs.copy()
        
        # 使用默认离线路径
        if not config.provider_uri and not config.provider_uri_map:
            default_path = cls._get_default_data_path()
            if default_path.exists():
                offline_kwargs["provider_uri"] = str(default_path)
            else:
                return False, f"❌ 离线数据目录不存在: {default_path}"
        elif config.provider_uri:
            offline_kwargs["provider_uri"] = config.provider_uri
        else:
            offline_kwargs["provider_uri"] = config.provider_uri_map
        
        try:
            qlib.init(**offline_kwargs)
            logger.info("✅ 离线模式初始化成功 (从在线回退)")
            return True, "✅ Qlib 离线模式初始化成功 (从在线回退)"
        except Exception as e:
            return False, f"❌ 离线模式也失败: {e}"
    
    @classmethod
    def _get_default_config(cls) -> QlibConfig:
        """
        获取默认配置 (环境变量 > 默认值)
        """
        # 从环境变量读取
        mode_str = os.getenv("QLIB_MODE", "offline")
        mode = QlibMode(mode_str)
        
        provider_uri = os.getenv("QLIB_PROVIDER_URI")
        if not provider_uri:
            provider_uri = str(cls._get_default_data_path())
        
        return QlibConfig(
            mode=mode,
            region=os.getenv("QLIB_REGION", "cn"),
            provider_uri=provider_uri,
            server_host=os.getenv("QLIB_SERVER_HOST", "127.0.0.1"),
            server_port=int(os.getenv("QLIB_SERVER_PORT", "9710")),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_enabled=os.getenv("QLIB_REDIS_ENABLED", "False").lower() == "true",
            expression_cache=os.getenv("QLIB_EXPRESSION_CACHE"),  # None 或 'DiskExpressionCache'
            dataset_cache=os.getenv("QLIB_DATASET_CACHE"),
        )
    
    @classmethod
    def _get_default_data_path(cls) -> Path:
        """
        获取默认数据路径 (跨平台兼容)
        """
        # 优先级: 环境变量 > 项目内 > 用户主目录
        if "QLIB_PROVIDER_URI" in os.environ:
            return Path(os.environ["QLIB_PROVIDER_URI"])
        
        # 项目根目录
        project_root = Path(__file__).parent.parent
        project_data = project_root / "data" / "qlib_data" / "cn_data"
        if project_data.exists():
            return project_data
        
        # 用户主目录
        home = Path.home()
        return home / ".qlib" / "qlib_data" / "cn_data"
    
    @classmethod
    def _log_init_success(cls, qlib, config: QlibConfig):
        """记录初始化成功信息"""
        try:
            version = getattr(qlib, "__version__", "未知")
            logger.info("=" * 60)
            logger.info("✅ Qlib 初始化成功")
            logger.info(f"   版本: {version}")
            logger.info(f"   模式: {config.mode.value}")
            logger.info(f"   区域: {config.region}")
            
            if config.mode == QlibMode.OFFLINE:
                logger.info(f"   数据路径: {config.provider_uri or config.provider_uri_map}")
            else:
                logger.info(f"   服务地址: {config.server_host}:{config.server_port}")
            
            logger.info(f"   Expression Cache: {config.expression_cache or '未启用'}")
            logger.info(f"   Dataset Cache: {config.dataset_cache or '未启用'}")
            logger.info(f"   Redis: {'已启用' if config.redis_enabled else '未启用'}")
            logger.info("=" * 60)
        except Exception as e:
            logger.debug(f"日志输出失败: {e}")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """检查 Qlib 是否已初始化"""
        return cls._initialized
    
    @classmethod
    def get_config(cls) -> Optional[QlibConfig]:
        """获取当前配置"""
        return cls._config
    
    @classmethod
    def reset(cls):
        """重置初始化状态 (用于测试)"""
        cls._initialized = False
        cls._config = None


# ============================================================================
# 便捷函数
# ============================================================================

def init_qlib(
    mode: str = "offline",
    provider_uri: Optional[str] = None,
    server_host: str = "127.0.0.1",
    server_port: int = 9710,
    redis_enabled: bool = False,
    **kwargs
) -> Tuple[bool, str]:
    """
    便捷初始化函数
    
    Args:
        mode: 模式 ('offline', 'online', 'auto')
        provider_uri: 数据路径 (离线模式)
        server_host: 服务器地址 (在线模式)
        server_port: 服务器端口 (在线模式)
        redis_enabled: 是否启用 Redis
        **kwargs: 其他 qlib.init() 参数
    
    Returns:
        (success: bool, message: str)
    
    Example:
        >>> success, msg = init_qlib(mode="offline", provider_uri="~/.qlib/qlib_data/cn_data")
        >>> if success:
        >>>     print(msg)
    """
    config = QlibConfig(
        mode=QlibMode(mode),
        provider_uri=provider_uri,
        server_host=server_host,
        server_port=server_port,
        redis_enabled=redis_enabled,
    )
    
    return QlibInitializer.init(config, **kwargs)


def check_qlib_connection() -> Tuple[bool, Dict[str, Any]]:
    """
    检查 Qlib 连接状态
    
    Returns:
        (connected: bool, info: dict)
    """
    if not QlibInitializer.is_initialized():
        return False, {"error": "Qlib 未初始化"}
    
    try:
        import qlib
        from qlib.config import C
        
        info = {
            "initialized": True,
            "version": getattr(qlib, "__version__", "未知"),
            "provider_uri": C.get("provider_uri"),
            "region": C.get("region"),
            "redis_host": C.get("redis_host"),
            "redis_port": C.get("redis_port"),
            "expression_cache": C.get("expression_cache"),
            "dataset_cache": C.get("dataset_cache"),
        }
        
        return True, info
    except Exception as e:
        return False, {"error": str(e)}


# ============================================================================
# 测试与示例
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Qlib 统一配置中心测试 ===\n")
    
    # 测试 1: 离线模式
    print("【测试 1】离线模式")
    success, msg = init_qlib(mode="offline")
    print(f"结果: {msg}\n")
    
    # 测试 2: 检查连接
    print("【测试 2】检查连接状态")
    connected, info = check_qlib_connection()
    if connected:
        print("✅ 连接成功")
        for key, value in info.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ 连接失败: {info.get('error')}")
