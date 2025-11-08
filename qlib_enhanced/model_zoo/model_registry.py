"""
模型注册表 - 包含所有Qlib Model Zoo支持的模型配置
Phase 5.1: 支持12个模型（包含1个已有模型LightGBM）
"""

from typing import Dict, Any, List


class ModelInfo:
    """模型信息类"""
    
    def __init__(self, 
                 name: str,
                 category: str,
                 status: str,
                 module: str,
                 class_name: str,
                 description: str,
                 params: Dict[str, Any],
                 requires_gpu: bool = False,
                 paper_url: str = None):
        self.name = name
        self.category = category
        self.status = status
        self.module = module
        self.class_name = class_name
        self.description = description
        self.params = params
        self.requires_gpu = requires_gpu
        self.paper_url = paper_url
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'category': self.category,
            'status': self.status,
            'module': self.module,
            'class': self.class_name,
            'description': self.description,
            'params': self.params,
            'requires_gpu': self.requires_gpu,
            'paper_url': self.paper_url
        }


# 模型注册表
MODEL_REGISTRY: Dict[str, ModelInfo] = {}


def register_model(model_info: ModelInfo):
    """注册模型到注册表"""
    MODEL_REGISTRY[model_info.name] = model_info
    return model_info


# ========== GBDT类模型 ==========

register_model(ModelInfo(
    name="LightGBM",
    category="GBDT",
    status="existing",
    module="qlib.contrib.model.gbdt",
    class_name="LGBModel",
    description="微软开源的梯度提升决策树，速度快、内存占用低，适合大规模数据",
    params={
        'learning_rate': {'type': 'numeric', 'default': 0.1, 'min': 0.001, 'max': 0.3},
        'n_estimators': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 500},
        'max_depth': {'type': 'numeric', 'default': 6, 'min': 3, 'max': 15},
        'num_leaves': {'type': 'numeric', 'default': 31, 'min': 10, 'max': 100},
        'min_child_samples': {'type': 'numeric', 'default': 20, 'min': 5, 'max': 100},
    },
    requires_gpu=False,
    paper_url="https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree"
))

register_model(ModelInfo(
    name="XGBoost",
    category="GBDT",
    status="new",
    module="qlib.contrib.model.xgboost",
    class_name="XGBModel",
    description="经典的梯度提升树模型，广泛应用于Kaggle竞赛和工业界",
    params={
        'learning_rate': {'type': 'numeric', 'default': 0.1, 'min': 0.001, 'max': 0.3},
        'n_estimators': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 500},
        'max_depth': {'type': 'numeric', 'default': 6, 'min': 3, 'max': 15},
        'subsample': {'type': 'numeric', 'default': 0.8, 'min': 0.5, 'max': 1.0},
        'colsample_bytree': {'type': 'numeric', 'default': 0.8, 'min': 0.5, 'max': 1.0},
    },
    requires_gpu=False,
    paper_url="https://arxiv.org/abs/1603.02754"
))

register_model(ModelInfo(
    name="CatBoost",
    category="GBDT",
    status="new",
    module="qlib.contrib.model.catboost_model",
    class_name="CatBoostModel",
    description="Yandex开发的GBDT，自动处理类别特征，防止过拟合能力强",
    params={
        'learning_rate': {'type': 'numeric', 'default': 0.1, 'min': 0.001, 'max': 0.3},
        'iterations': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 500},
        'depth': {'type': 'numeric', 'default': 6, 'min': 3, 'max': 15},
        'l2_leaf_reg': {'type': 'numeric', 'default': 3.0, 'min': 1.0, 'max': 10.0},
    },
    requires_gpu=False,
    paper_url="https://arxiv.org/abs/1706.09516"
))

# ========== 神经网络类模型 ==========

register_model(ModelInfo(
    name="MLP",
    category="Neural Networks",
    status="new",
    module="qlib.contrib.model.pytorch_nn",
    class_name="DNNModelPytorch",
    description="多层感知机，简单的前馈神经网络，适合快速验证",
    params={
        'lr': {'type': 'numeric', 'default': 0.001, 'min': 0.0001, 'max': 0.01},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 512, 'min': 128, 'max': 2048},
        'hidden_size': {'type': 'numeric', 'default': 128, 'min': 64, 'max': 512},
        'dropout': {'type': 'numeric', 'default': 0.3, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://en.wikipedia.org/wiki/Multilayer_perceptron"
))

register_model(ModelInfo(
    name="LSTM",
    category="Neural Networks",
    status="new",
    module="qlib.contrib.model.pytorch_lstm",
    class_name="LSTMModel",
    description="长短期记忆网络，捕捉时间序列的长期依赖关系",
    params={
        'lr': {'type': 'numeric', 'default': 0.001, 'min': 0.0001, 'max': 0.01},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 512, 'min': 128, 'max': 2048},
        'hidden_size': {'type': 'numeric', 'default': 64, 'min': 32, 'max': 256},
        'num_layers': {'type': 'numeric', 'default': 2, 'min': 1, 'max': 4},
        'dropout': {'type': 'numeric', 'default': 0.3, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://www.bioinf.jku.at/publications/older/2604.pdf"
))

register_model(ModelInfo(
    name="GRU",
    category="Neural Networks",
    status="new",
    module="qlib.contrib.model.pytorch_gru",
    class_name="GRUModel",
    description="门控循环单元，LSTM的简化版本，训练速度更快",
    params={
        'lr': {'type': 'numeric', 'default': 0.001, 'min': 0.0001, 'max': 0.01},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 512, 'min': 128, 'max': 2048},
        'hidden_size': {'type': 'numeric', 'default': 64, 'min': 32, 'max': 256},
        'num_layers': {'type': 'numeric', 'default': 2, 'min': 1, 'max': 4},
        'dropout': {'type': 'numeric', 'default': 0.3, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://arxiv.org/abs/1406.1078"
))

register_model(ModelInfo(
    name="ALSTM",
    category="Neural Networks",
    status="new",
    module="qlib.contrib.model.pytorch_alstm",
    class_name="ALSTMModel",
    description="注意力LSTM，增加注意力机制，关注重要的时间步",
    params={
        'lr': {'type': 'numeric', 'default': 0.001, 'min': 0.0001, 'max': 0.01},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 512, 'min': 128, 'max': 2048},
        'hidden_size': {'type': 'numeric', 'default': 64, 'min': 32, 'max': 256},
        'num_layers': {'type': 'numeric', 'default': 2, 'min': 1, 'max': 4},
        'dropout': {'type': 'numeric', 'default': 0.3, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://arxiv.org/abs/1901.07891"
))

# ========== 高级模型 ==========

register_model(ModelInfo(
    name="Transformer",
    category="Advanced",
    status="new",
    module="qlib.contrib.model.pytorch_transformer",
    class_name="TransformerModel",
    description="Transformer架构，利用自注意力机制建模时序关系",
    params={
        'lr': {'type': 'numeric', 'default': 0.0001, 'min': 0.00001, 'max': 0.001},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 256, 'min': 128, 'max': 1024},
        'd_model': {'type': 'numeric', 'default': 64, 'min': 32, 'max': 256},
        'nhead': {'type': 'list', 'default': 4, 'options': [2, 4, 8, 16]},
        'num_layers': {'type': 'numeric', 'default': 2, 'min': 1, 'max': 6},
        'dropout': {'type': 'numeric', 'default': 0.3, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://arxiv.org/abs/1706.03762"
))

register_model(ModelInfo(
    name="TRA",
    category="Advanced",
    status="new",
    module="qlib.contrib.model.pytorch_tra",
    class_name="TRA",
    description="Temporal Routing Adaptor，动态路由的时序注意力模型",
    params={
        'lr': {'type': 'numeric', 'default': 0.0001, 'min': 0.00001, 'max': 0.001},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 256, 'min': 128, 'max': 1024},
        'd_model': {'type': 'numeric', 'default': 64, 'min': 32, 'max': 256},
        'num_head': {'type': 'list', 'default': 4, 'options': [2, 4, 8]},
        'dropout': {'type': 'numeric', 'default': 0.3, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://arxiv.org/abs/2106.12950"
))

register_model(ModelInfo(
    name="TCN",
    category="Advanced",
    status="new",
    module="qlib.contrib.model.pytorch_tcn",
    class_name="TCN",
    description="时间卷积网络，使用因果卷积处理时序数据",
    params={
        'lr': {'type': 'numeric', 'default': 0.001, 'min': 0.0001, 'max': 0.01},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 512, 'min': 128, 'max': 2048},
        'num_channels': {'type': 'list', 'default': '[32,32,32]', 'options': ['[16,16]', '[32,32,32]', '[64,64,64,64]']},
        'kernel_size': {'type': 'numeric', 'default': 3, 'min': 2, 'max': 7},
        'dropout': {'type': 'numeric', 'default': 0.2, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://arxiv.org/abs/1803.01271"
))

register_model(ModelInfo(
    name="HIST",
    category="Advanced",
    status="new",
    module="qlib.contrib.model.pytorch_hist",
    class_name="HIST",
    description="层次化的时序Transformer，分层建模不同时间尺度",
    params={
        'lr': {'type': 'numeric', 'default': 0.0001, 'min': 0.00001, 'max': 0.001},
        'n_epochs': {'type': 'numeric', 'default': 100, 'min': 50, 'max': 300},
        'batch_size': {'type': 'numeric', 'default': 256, 'min': 128, 'max': 1024},
        'd_model': {'type': 'numeric', 'default': 64, 'min': 32, 'max': 256},
        'num_layers': {'type': 'numeric', 'default': 2, 'min': 1, 'max': 4},
        'dropout': {'type': 'numeric', 'default': 0.3, 'min': 0.0, 'max': 0.5},
    },
    requires_gpu=True,
    paper_url="https://arxiv.org/abs/2110.13716"
))

# ========== 集成学习 ==========

register_model(ModelInfo(
    name="DoubleEnsemble",
    category="Ensemble",
    status="new",
    module="qlib.contrib.model.double_ensemble",
    class_name="DEnsembleModel",
    description="双层集成模型，结合多个基础模型提高预测稳定性",
    params={
        'base_model': {'type': 'list', 'default': 'LightGBM', 'options': ['LightGBM', 'XGBoost', 'CatBoost']},
        'num_models': {'type': 'numeric', 'default': 5, 'min': 3, 'max': 10},
        'enable_sr': {'type': 'list', 'default': True, 'options': [True, False]},
        'enable_fs': {'type': 'list', 'default': True, 'options': [True, False]},
    },
    requires_gpu=False,
    paper_url="https://arxiv.org/abs/2012.06679"
))


def get_models_by_category(category: str) -> List[ModelInfo]:
    """获取指定类别的所有模型"""
    return [m for m in MODEL_REGISTRY.values() if m.category == category]


def get_all_categories() -> List[str]:
    """获取所有模型类别"""
    categories = set(m.category for m in MODEL_REGISTRY.values())
    return sorted(categories)


def get_model_info(model_name: str) -> ModelInfo:
    """获取模型信息"""
    return MODEL_REGISTRY.get(model_name)
