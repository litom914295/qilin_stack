"""
统一路径配置管理
解决硬编码路径问题,提供环境变量支持和自动发现机制

问题: 项目中存在20+处硬编码路径,如:
- D:/test/Qlib/tradingagents
- D:/test/Qlib/RD-Agent  
- G:/data/qilin_data

解决方案: 统一配置管理,支持:
1. 环境变量配置 (优先级最高)
2. 自动路径发现 (搜索常见位置)
3. 默认相对路径 (项目内部)

使用示例:
    from config.paths import PathConfig
    
    # 检查配置有效性
    status = PathConfig.validate()
    print(f"TradingAgents可用: {status['tradingagents_available']}")
    
    # 获取路径
    ta_path = PathConfig.get_tradingagents_path()
    if ta_path:
        import sys
        sys.path.insert(0, str(ta_path))
"""

from pathlib import Path
import os
import sys
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class PathConfig:
    """统一路径配置管理器"""
    
    # ============================================================================
    # 基础路径
    # ============================================================================
    
    BASE_DIR = Path(__file__).parent.parent.absolute()
    """项目根目录 (qilin_stack/)"""
    
    CONFIG_DIR = BASE_DIR / "config"
    """配置文件目录"""
    
    # ============================================================================
    # 外部项目依赖路径 (动态获取)
    # ============================================================================
    
    @staticmethod
    def get_tradingagents_path() -> Optional[Path]:
        """
        获取TradingAgents项目路径
        
        查找顺序:
        1. 环境变量 TRADINGAGENTS_PATH
        2. 相对路径 (../tradingagents-cn-plus)
        3. 常见安装位置
        
        Returns:
            Path对象,如果未找到返回None
        """
        # 1. 环境变量 (最高优先级)
        env_path = os.getenv("TRADINGAGENTS_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.info(f"✅ TradingAgents路径(环境变量): {path}")
                return path.absolute()
            else:
                logger.warning(f"⚠️ 环境变量TRADINGAGENTS_PATH指向的路径不存在: {env_path}")
        
        # 2. 自动发现 (搜索常见位置)
        common_paths = [
            # 相对路径
            PathConfig.BASE_DIR.parent / "tradingagents-cn-plus",
            PathConfig.BASE_DIR.parent / "TradingAgents-CN",
            PathConfig.BASE_DIR.parent / "tradingagents",
            
            # 绝对路径 (Windows)
            Path("G:/test/tradingagents-cn-plus"),
            Path("D:/test/tradingagents-cn-plus"),
            Path("C:/Projects/tradingagents-cn-plus"),
            
            # 绝对路径 (Linux/Mac)
            Path("/opt/tradingagents"),
            Path("/usr/local/tradingagents"),
            Path.home() / "tradingagents-cn-plus",
        ]
        
        for candidate in common_paths:
            if candidate.exists() and (candidate / "tradingagents").exists():
                logger.info(f"✅ TradingAgents路径(自动发现): {candidate}")
                return candidate.absolute()
        
        logger.warning("❌ 未找到TradingAgents项目路径,请设置TRADINGAGENTS_PATH环境变量")
        return None
    
    @staticmethod
    def get_rdagent_path() -> Optional[Path]:
        """
        获取RD-Agent项目路径
        
        查找顺序:
        1. 环境变量 RDAGENT_PATH
        2. 相对路径 (../RD-Agent)
        3. 常见安装位置
        
        Returns:
            Path对象,如果未找到返回None
        """
        # 1. 环境变量
        env_path = os.getenv("RDAGENT_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.info(f"✅ RD-Agent路径(环境变量): {path}")
                return path.absolute()
            else:
                logger.warning(f"⚠️ 环境变量RDAGENT_PATH指向的路径不存在: {env_path}")
        
        # 2. 自动发现
        common_paths = [
            # 相对路径
            PathConfig.BASE_DIR.parent / "RD-Agent",
            PathConfig.BASE_DIR.parent / "rdagent",
            
            # 绝对路径 (Windows)
            Path("G:/test/RD-Agent"),
            Path("D:/test/RD-Agent"),
            Path("C:/Projects/RD-Agent"),
            
            # 绝对路径 (Linux/Mac)
            Path("/opt/rdagent"),
            Path("/usr/local/rdagent"),
            Path.home() / "RD-Agent",
        ]
        
        for candidate in common_paths:
            if candidate.exists() and (candidate / "rdagent").exists():
                logger.info(f"✅ RD-Agent路径(自动发现): {candidate}")
                return candidate.absolute()
        
        logger.warning("❌ 未找到RD-Agent项目路径,请设置RDAGENT_PATH环境变量")
        return None
    
    @staticmethod
    def get_qlib_path() -> Optional[Path]:
        """
        获取Qlib项目路径 (如果需要访问源码)
        
        Returns:
            Path对象,如果未找到返回None
        """
        env_path = os.getenv("QLIB_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path.absolute()
        
        common_paths = [
            PathConfig.BASE_DIR.parent / "qlib",
            Path("G:/test/qlib"),
            Path("D:/test/qlib"),
        ]
        
        for candidate in common_paths:
            if candidate.exists():
                return candidate.absolute()
        
        return None
    
    @staticmethod
    def get_qlib_data_path() -> Path:
        """
        获取Qlib数据路径
        
        查找顺序:
        1. 环境变量 QLIB_DATA_PATH
        2. 默认路径 ~/.qlib/qlib_data/cn_data
        
        Returns:
            Path对象
        """
        env_path = os.getenv("QLIB_DATA_PATH")
        if env_path:
            return Path(env_path).absolute()
        
        # 默认Qlib数据路径
        return Path.home() / ".qlib" / "qlib_data" / "cn_data"
    
    # ============================================================================
    # 项目内部路径
    # ============================================================================
    
    @staticmethod
    def _get_env_path(env_var: str, default: Path) -> Path:
        """从环境变量获取路径,如果未设置则使用默认值"""
        env_value = os.getenv(env_var)
        if env_value:
            return Path(env_value).absolute()
        return default.absolute()
    
    # 数据目录
    DATA_DIR = _get_env_path.__func__("QILIN_DATA_DIR", BASE_DIR / "data")
    """数据存储目录"""
    
    # 模型目录
    MODELS_DIR = _get_env_path.__func__("QILIN_MODELS_DIR", BASE_DIR / "models")
    """训练模型存储目录"""
    
    # 日志目录
    LOGS_DIR = _get_env_path.__func__("QILIN_LOGS_DIR", BASE_DIR / "logs")
    """日志文件目录"""
    
    # 缓存目录
    CACHE_DIR = _get_env_path.__func__("QILIN_CACHE_DIR", BASE_DIR / ".cache")
    """缓存数据目录"""
    
    # 临时目录
    TEMP_DIR = _get_env_path.__func__("QILIN_TEMP_DIR", BASE_DIR / "temp")
    """临时文件目录"""
    
    # 输出目录
    OUTPUT_DIR = _get_env_path.__func__("QILIN_OUTPUT_DIR", BASE_DIR / "output")
    """输出结果目录"""
    
    # 报告目录
    REPORTS_DIR = _get_env_path.__func__("QILIN_REPORTS_DIR", BASE_DIR / "reports")
    """报告文件目录"""
    
    # 检查点目录
    CHECKPOINTS_DIR = _get_env_path.__func__("QILIN_CHECKPOINTS_DIR", BASE_DIR / "checkpoints")
    """模型检查点目录"""
    
    # ============================================================================
    # 工具方法
    # ============================================================================
    
    @classmethod
    def ensure_dirs(cls) -> Dict[str, bool]:
        """
        确保所有必要的目录存在
        
        Returns:
            字典,记录每个目录的创建状态
        """
        dirs_to_create = {
            "data": cls.DATA_DIR,
            "models": cls.MODELS_DIR,
            "logs": cls.LOGS_DIR,
            "cache": cls.CACHE_DIR,
            "temp": cls.TEMP_DIR,
            "output": cls.OUTPUT_DIR,
            "reports": cls.REPORTS_DIR,
            "checkpoints": cls.CHECKPOINTS_DIR,
        }
        
        results = {}
        for name, dir_path in dirs_to_create.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                results[name] = True
                logger.debug(f"✅ 目录已创建/存在: {dir_path}")
            except Exception as e:
                results[name] = False
                logger.error(f"❌ 创建目录失败 {name}: {e}")
        
        return results
    
    @classmethod
    def validate(cls, verbose: bool = True) -> Dict[str, bool]:
        """
        验证所有路径配置
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            验证结果字典
        """
        results = {
            "tradingagents_available": cls.get_tradingagents_path() is not None,
            "rdagent_available": cls.get_rdagent_path() is not None,
            "qlib_path_available": cls.get_qlib_path() is not None,
            "qlib_data_exists": cls.get_qlib_data_path().exists(),
            "base_dir_exists": cls.BASE_DIR.exists(),
        }
        
        # 检查内部目录
        dir_status = cls.ensure_dirs()
        results["dirs_created"] = all(dir_status.values())
        
        if verbose:
            print("\n" + "=" * 60)
            print("🔍 路径配置验证结果")
            print("=" * 60)
            
            print("\n📦 外部项目:")
            print(f"  TradingAgents: {'✅' if results['tradingagents_available'] else '❌'}")
            if results['tradingagents_available']:
                print(f"    路径: {cls.get_tradingagents_path()}")
            
            print(f"  RD-Agent:      {'✅' if results['rdagent_available'] else '❌'}")
            if results['rdagent_available']:
                print(f"    路径: {cls.get_rdagent_path()}")
            
            print(f"  Qlib源码:      {'✅' if results['qlib_path_available'] else '⚪'} (可选)")
            print(f"  Qlib数据:      {'✅' if results['qlib_data_exists'] else '❌'}")
            print(f"    路径: {cls.get_qlib_data_path()}")
            
            print("\n📁 内部目录:")
            for name, created in dir_status.items():
                print(f"  {name:12s}: {'✅' if created else '❌'}")
            
            print("\n" + "=" * 60)
            
            # 统计
            total = len(results)
            passed = sum(1 for v in results.values() if v)
            print(f"总体: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
            print("=" * 60 + "\n")
        
        return results
    
    @classmethod
    def get_all_paths(cls) -> Dict[str, Optional[Path]]:
        """
        获取所有配置的路径
        
        Returns:
            路径字典
        """
        return {
            # 外部项目
            "tradingagents": cls.get_tradingagents_path(),
            "rdagent": cls.get_rdagent_path(),
            "qlib": cls.get_qlib_path(),
            "qlib_data": cls.get_qlib_data_path(),
            
            # 项目根目录
            "base": cls.BASE_DIR,
            "config": cls.CONFIG_DIR,
            
            # 内部目录
            "data": cls.DATA_DIR,
            "models": cls.MODELS_DIR,
            "logs": cls.LOGS_DIR,
            "cache": cls.CACHE_DIR,
            "temp": cls.TEMP_DIR,
            "output": cls.OUTPUT_DIR,
            "reports": cls.REPORTS_DIR,
            "checkpoints": cls.CHECKPOINTS_DIR,
        }
    
    @classmethod
    def add_external_to_path(cls) -> Dict[str, bool]:
        """
        将外部项目路径添加到sys.path
        
        Returns:
            添加结果字典
        """
        results = {}
        
        # TradingAgents
        ta_path = cls.get_tradingagents_path()
        if ta_path and str(ta_path) not in sys.path:
            sys.path.insert(0, str(ta_path))
            results["tradingagents"] = True
            logger.info(f"✅ TradingAgents已添加到sys.path: {ta_path}")
        else:
            results["tradingagents"] = False
        
        # RD-Agent
        rd_path = cls.get_rdagent_path()
        if rd_path and str(rd_path) not in sys.path:
            sys.path.insert(0, str(rd_path))
            results["rdagent"] = True
            logger.info(f"✅ RD-Agent已添加到sys.path: {rd_path}")
        else:
            results["rdagent"] = False
        
        return results


# ============================================================================
# 便捷函数
# ============================================================================

def init_paths(verbose: bool = True) -> bool:
    """
    初始化路径配置 (推荐在程序启动时调用)
    
    Args:
        verbose: 是否打印详细信息
        
    Returns:
        是否初始化成功
    """
    # 确保目录存在
    PathConfig.ensure_dirs()
    
    # 验证配置
    results = PathConfig.validate(verbose=verbose)
    
    # 添加外部路径到sys.path
    PathConfig.add_external_to_path()
    
    # 判断是否成功 (至少TradingAgents或RD-Agent之一可用)
    success = results["tradingagents_available"] or results["rdagent_available"]
    
    return success


def get_env_template() -> str:
    """
    生成.env配置文件模板
    
    Returns:
        .env文件内容
    """
    return """# Qilin Stack 路径配置
# 复制此文件为 .env 并根据实际情况修改

# ============================================================================
# 外部项目路径 (必须配置)
# ============================================================================

# TradingAgents项目路径
TRADINGAGENTS_PATH=G:/test/tradingagents-cn-plus

# RD-Agent项目路径  
RDAGENT_PATH=G:/test/RD-Agent

# Qlib源码路径 (可选,仅当需要访问源码时)
# QLIB_PATH=G:/test/qlib

# Qlib数据路径
QLIB_DATA_PATH=~/.qlib/qlib_data/cn_data

# ============================================================================
# 项目内部路径 (可选,默认在项目目录下)
# ============================================================================

# 数据目录
# QILIN_DATA_DIR=./data

# 模型目录
# QILIN_MODELS_DIR=./models

# 日志目录
# QILIN_LOGS_DIR=./logs

# 缓存目录
# QILIN_CACHE_DIR=./.cache

# 临时文件目录
# QILIN_TEMP_DIR=./temp

# 输出目录
# QILIN_OUTPUT_DIR=./output

# 报告目录
# QILIN_REPORTS_DIR=./reports

# 检查点目录
# QILIN_CHECKPOINTS_DIR=./checkpoints

# ============================================================================
# LLM配置 (用于TradingAgents和RD-Agent)
# ============================================================================

# LLM提供商 (openai/anthropic/azure)
LLM_PROVIDER=openai

# OpenAI配置
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4-turbo

# Anthropic配置 (如果使用Claude)
# ANTHROPIC_API_KEY=your-key-here
# LLM_MODEL=claude-3-opus-20240229

# Azure OpenAI配置 (如果使用Azure)
# AZURE_API_KEY=your-key-here
# AZURE_API_BASE=https://your-resource.openai.azure.com
# AZURE_API_VERSION=2024-02-15-preview
"""


# ============================================================================
# 命令行工具
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qilin路径配置管理工具")
    parser.add_argument(
        "--action",
        choices=["validate", "init", "template", "show"],
        default="validate",
        help="操作类型"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    if args.action == "validate":
        # 验证配置
        results = PathConfig.validate(verbose=True)
        sys.exit(0 if all(results.values()) else 1)
    
    elif args.action == "init":
        # 初始化路径
        success = init_paths(verbose=True)
        if success:
            print("\n✅ 路径初始化成功!")
            sys.exit(0)
        else:
            print("\n❌ 路径初始化失败,请检查配置")
            sys.exit(1)
    
    elif args.action == "template":
        # 生成.env模板
        template = get_env_template()
        output_path = PathConfig.BASE_DIR / ".env.template"
        output_path.write_text(template, encoding="utf-8")
        print(f"✅ .env模板已生成: {output_path}")
        print("\n请复制为 .env 并修改配置:")
        print(f"  copy {output_path} .env")
    
    elif args.action == "show":
        # 显示所有路径
        print("\n" + "=" * 60)
        print("📂 当前路径配置")
        print("=" * 60 + "\n")
        
        paths = PathConfig.get_all_paths()
        for name, path in paths.items():
            if path:
                exists = "✅" if path.exists() else "❌"
                print(f"{name:15s}: {exists} {path}")
            else:
                print(f"{name:15s}: ⚪ (未配置)")
