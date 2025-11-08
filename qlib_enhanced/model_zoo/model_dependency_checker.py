"""
模型依赖检测模块
检测 Qlib 模型所需依赖是否满足,并提供降级策略建议
"""
import importlib
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DependencyCheckResult:
    """依赖检测结果"""
    available: bool  # 是否可用
    status: str  # 'ok', 'missing_deps', 'fallback', 'error'
    missing_packages: List[str]  # 缺失的包
    fallback_model: Optional[str]  # 降级模型
    install_command: Optional[str]  # 安装命令
    message: str  # 详细信息


# 模型依赖映射
MODEL_DEPENDENCIES = {
    # GBDT 家族 (无额外依赖)
    "LightGBM": {
        "packages": ["lightgbm"],
        "import_test": "from qlib.contrib.model.gbdt import LGBModel",
        "fallback": None,
    },
    "XGBoost": {
        "packages": ["xgboost"],
        "import_test": "from qlib.contrib.model.xgboost import XGBModel",
        "fallback": "LightGBM",
    },
    "CatBoost": {
        "packages": ["catboost"],
        "import_test": "from qlib.contrib.model.catboost_model import CatBoostModel",
        "fallback": "LightGBM",
    },
    
    # 神经网络 (需要 torch)
    "MLP": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_nn import DNNModelPytorch",
        "fallback": "LightGBM",
    },
    "LSTM": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_lstm import LSTMModel",
        "fallback": "LightGBM",
    },
    "GRU": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_gru import GRUModel",
        "fallback": "LSTM",
    },
    "ALSTM": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_alstm import ALSTMModel",
        "fallback": "LSTM",
    },
    
    # 高级模型
    "Transformer": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_transformer import Transformer",
        "fallback": "LSTM",
    },
    "TRA": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_tra import TRA",
        "fallback": "Transformer",  # TRA 降级为 Transformer
        "note": "⚠️ TRA 需要额外的图结构依赖,可能降级为 Transformer"
    },
    "TCN": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_tcn import TCN",
        "fallback": "LSTM",
    },
    "HIST": {
        "packages": ["torch"],
        "import_test": "from qlib.contrib.model.pytorch_hist import HIST",
        "fallback": "Transformer",  # HIST 降级为 Transformer
        "note": "⚠️ HIST 需要额外的依赖,可能降级为 Transformer"
    },
    
    # 集成模型
    "DoubleEnsemble": {
        "packages": ["lightgbm"],
        "import_test": "from qlib.contrib.model.double_ensemble import DoubleEnsembleModel",
        "fallback": "LightGBM",
        "note": "⚠️ DoubleEnsemble 依赖基础模型,可能降级为单模型"
    },
}


def check_package(package_name: str) -> bool:
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def check_model_import(import_statement: str) -> bool:
    """检查模型类是否可导入"""
    try:
        exec(import_statement)
        return True
    except Exception as e:
        logger.debug(f"模型导入失败: {import_statement}, 错误: {e}")
        return False


def check_model_availability(model_name: str) -> DependencyCheckResult:
    """
    检查模型是否可用
    
    Args:
        model_name: 模型名称 (如 'TRA', 'HIST')
    
    Returns:
        DependencyCheckResult: 检测结果
    """
    if model_name not in MODEL_DEPENDENCIES:
        return DependencyCheckResult(
            available=False,
            status='error',
            missing_packages=[],
            fallback_model=None,
            install_command=None,
            message=f"未知模型: {model_name}"
        )
    
    config = MODEL_DEPENDENCIES[model_name]
    packages = config["packages"]
    import_test = config["import_test"]
    fallback = config.get("fallback")
    note = config.get("note", "")
    
    # 检查依赖包
    missing_packages = [pkg for pkg in packages if not check_package(pkg)]
    
    if missing_packages:
        # 缺少依赖包
        install_cmd = f"pip install {' '.join(missing_packages)}"
        return DependencyCheckResult(
            available=False,
            status='missing_deps',
            missing_packages=missing_packages,
            fallback_model=fallback,
            install_command=install_cmd,
            message=f"❌ 缺少依赖: {', '.join(missing_packages)}"
        )
    
    # 检查模型类是否可导入
    if not check_model_import(import_test):
        # 依赖包存在但模型不可用 (可能是版本不兼容或特殊依赖缺失)
        return DependencyCheckResult(
            available=False,
            status='fallback',
            missing_packages=[],
            fallback_model=fallback,
            install_command=None,
            message=f"⚠️ 模型不可用 (可能降级为 {fallback})" + (f" {note}" if note else "")
        )
    
    # 模型完全可用
    return DependencyCheckResult(
        available=True,
        status='ok',
        missing_packages=[],
        fallback_model=None,
        install_command=None,
        message=f"✅ 可用" + (f" {note}" if note else "")
    )


def check_all_models() -> Dict[str, DependencyCheckResult]:
    """检查所有模型的可用性"""
    results = {}
    for model_name in MODEL_DEPENDENCIES.keys():
        results[model_name] = check_model_availability(model_name)
    return results


def get_available_models() -> List[str]:
    """获取所有可用模型列表"""
    results = check_all_models()
    return [name for name, result in results.items() if result.available]


def get_model_status_summary() -> Dict[str, int]:
    """获取模型状态统计"""
    results = check_all_models()
    return {
        'total': len(results),
        'available': sum(1 for r in results.values() if r.available),
        'missing_deps': sum(1 for r in results.values() if r.status == 'missing_deps'),
        'fallback': sum(1 for r in results.values() if r.status == 'fallback'),
        'error': sum(1 for r in results.values() if r.status == 'error'),
    }


# 降级策略映射 (反向查询)
def get_models_using_fallback(fallback_model: str) -> List[str]:
    """获取使用某个降级模型的原始模型列表"""
    models = []
    for model_name, config in MODEL_DEPENDENCIES.items():
        if config.get("fallback") == fallback_model:
            result = check_model_availability(model_name)
            if result.status == 'fallback':
                models.append(model_name)
    return models


if __name__ == "__main__":
    # 测试用例
    print("=== Qlib 模型依赖检测 ===\n")
    
    results = check_all_models()
    for model_name, result in results.items():
        print(f"{model_name:15} | {result.message}")
        if result.status == 'missing_deps':
            print(f"               | 安装命令: {result.install_command}")
        if result.fallback_model:
            print(f"               | 降级方案: {result.fallback_model}")
        print()
    
    print("=== 统计摘要 ===")
    summary = get_model_status_summary()
    print(f"总计: {summary['total']}")
    print(f"可用: {summary['available']}")
    print(f"缺少依赖: {summary['missing_deps']}")
    print(f"降级运行: {summary['fallback']}")
