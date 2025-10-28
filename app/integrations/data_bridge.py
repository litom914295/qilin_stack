"""
数据共享桥接模块
实现三个项目之间的数据、因子、模型共享
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class DataBridge:
    """数据共享桥接类"""
    
    def __init__(self, cache_dir: str = "./shared_cache"):
        """
        初始化数据桥接
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.factors_dir = self.cache_dir / "factors"
        self.models_dir = self.cache_dir / "models"
        self.strategies_dir = self.cache_dir / "strategies"
        self.data_dir = self.cache_dir / "data"
        
        for d in [self.factors_dir, self.models_dir, self.strategies_dir, self.data_dir]:
            d.mkdir(exist_ok=True)
    
    # ========== 因子共享 ==========
    
    def save_factor(self, factor_name: str, factor_data: Dict[str, Any], source: str):
        """
        保存因子数据
        
        Args:
            factor_name: 因子名称
            factor_data: 因子数据（包含formula、description等）
            source: 来源（qlib、rdagent、tradingagents）
        """
        try:
            factor_info = {
                'name': factor_name,
                'source': source,
                'data': factor_data,
                'metadata': {
                    'created_at': pd.Timestamp.now().isoformat(),
                }
            }
            
            factor_file = self.factors_dir / f"{factor_name}.json"
            with open(factor_file, 'w', encoding='utf-8') as f:
                json.dump(factor_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存因子: {factor_name} (来源: {source})")
            return True
        except Exception as e:
            logger.error(f"保存因子失败: {e}")
            return False
    
    def load_factor(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """
        加载因子数据
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子数据字典
        """
        try:
            factor_file = self.factors_dir / f"{factor_name}.json"
            if not factor_file.exists():
                return None
            
            with open(factor_file, 'r', encoding='utf-8') as f:
                factor_info = json.load(f)
            
            return factor_info
        except Exception as e:
            logger.error(f"加载因子失败: {e}")
            return None
    
    def list_factors(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出所有因子
        
        Args:
            source: 过滤来源
            
        Returns:
            因子列表
        """
        factors = []
        
        for factor_file in self.factors_dir.glob("*.json"):
            try:
                with open(factor_file, 'r', encoding='utf-8') as f:
                    factor_info = json.load(f)
                
                if source is None or factor_info.get('source') == source:
                    factors.append(factor_info)
            except Exception as e:
                logger.error(f"读取因子文件失败 {factor_file}: {e}")
        
        return factors
    
    # ========== 模型共享 ==========
    
    def save_model(self, model_name: str, model_obj: Any, metadata: Dict[str, Any], source: str):
        """
        保存模型
        
        Args:
            model_name: 模型名称
            model_obj: 模型对象
            metadata: 元数据（性能指标等）
            source: 来源
        """
        try:
            # 保存模型对象
            model_file = self.models_dir / f"{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_obj, f)
            
            # 保存元数据
            meta_file = self.models_dir / f"{model_name}.json"
            model_info = {
                'name': model_name,
                'source': source,
                'metadata': metadata,
                'created_at': pd.Timestamp.now().isoformat(),
            }
            
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存模型: {model_name} (来源: {source})")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[tuple]:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            (模型对象, 元数据)
        """
        try:
            model_file = self.models_dir / f"{model_name}.pkl"
            meta_file = self.models_dir / f"{model_name}.json"
            
            if not model_file.exists() or not meta_file.exists():
                return None
            
            # 加载模型对象
            with open(model_file, 'rb') as f:
                model_obj = pickle.load(f)
            
            # 加载元数据
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return model_obj, metadata
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None
    
    def list_models(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有模型"""
        models = []
        
        for meta_file in self.models_dir.glob("*.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                if source is None or model_info.get('source') == source:
                    models.append(model_info)
            except Exception as e:
                logger.error(f"读取模型元数据失败 {meta_file}: {e}")
        
        return models
    
    # ========== 策略共享 ==========
    
    def save_strategy(self, strategy_name: str, strategy_config: Dict[str, Any], source: str):
        """
        保存策略配置
        
        Args:
            strategy_name: 策略名称
            strategy_config: 策略配置
            source: 来源
        """
        try:
            strategy_info = {
                'name': strategy_name,
                'source': source,
                'config': strategy_config,
                'created_at': pd.Timestamp.now().isoformat(),
            }
            
            strategy_file = self.strategies_dir / f"{strategy_name}.json"
            with open(strategy_file, 'w', encoding='utf-8') as f:
                json.dump(strategy_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存策略: {strategy_name} (来源: {source})")
            return True
        except Exception as e:
            logger.error(f"保存策略失败: {e}")
            return False
    
    def load_strategy(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """加载策略配置"""
        try:
            strategy_file = self.strategies_dir / f"{strategy_name}.json"
            if not strategy_file.exists():
                return None
            
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy_info = json.load(f)
            
            return strategy_info
        except Exception as e:
            logger.error(f"加载策略失败: {e}")
            return None
    
    def list_strategies(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有策略"""
        strategies = []
        
        for strategy_file in self.strategies_dir.glob("*.json"):
            try:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    strategy_info = json.load(f)
                
                if source is None or strategy_info.get('source') == source:
                    strategies.append(strategy_info)
            except Exception as e:
                logger.error(f"读取策略文件失败 {strategy_file}: {e}")
        
        return strategies
    
    # ========== 数据共享 ==========
    
    def save_data(self, data_name: str, df: pd.DataFrame, source: str):
        """
        保存数据
        
        Args:
            data_name: 数据名称
            df: DataFrame数据
            source: 来源
        """
        try:
            # 保存数据
            data_file = self.data_dir / f"{data_name}.parquet"
            df.to_parquet(data_file)
            
            # 保存元数据
            meta_file = self.data_dir / f"{data_name}.json"
            meta_info = {
                'name': data_name,
                'source': source,
                'shape': df.shape,
                'columns': list(df.columns),
                'created_at': pd.Timestamp.now().isoformat(),
            }
            
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存数据: {data_name} (来源: {source})")
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return False
    
    def load_data(self, data_name: str) -> Optional[pd.DataFrame]:
        """加载数据"""
        try:
            data_file = self.data_dir / f"{data_name}.parquet"
            if not data_file.exists():
                return None
            
            df = pd.read_parquet(data_file)
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return None
    
    def list_data(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有数据"""
        data_list = []
        
        for meta_file in self.data_dir.glob("*.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_info = json.load(f)
                
                if source is None or meta_info.get('source') == source:
                    data_list.append(meta_info)
            except Exception as e:
                logger.error(f"读取数据元信息失败 {meta_file}: {e}")
        
        return data_list
    
    # ========== 跨项目转换 ==========
    
    def convert_qlib_to_rdagent(self, qlib_factor: Dict) -> Dict:
        """将Qlib因子转换为RD-Agent格式"""
        return {
            'name': qlib_factor.get('name'),
            'formula': qlib_factor.get('expression'),
            'description': qlib_factor.get('description', ''),
            'source': 'qlib_converted'
        }
    
    def convert_rdagent_to_tradingagents(self, rdagent_strategy: Dict) -> Dict:
        """将RD-Agent策略转换为TradingAgents格式"""
        return {
            'strategy_name': rdagent_strategy.get('name'),
            'rules': rdagent_strategy.get('rules', []),
            'risk_params': rdagent_strategy.get('risk_management', {}),
            'source': 'rdagent_converted'
        }


# 全局实例
data_bridge = DataBridge()
