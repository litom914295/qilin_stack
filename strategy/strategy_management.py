"""
策略管理系统
策略创建、版本管理、参数优化、策略库管理
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
import asyncio
import logging
from pathlib import Path
import pickle
import yaml
from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import TimeSeriesSplit
import git

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"      # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    MOMENTUM = "momentum"                     # 动量策略
    ARBITRAGE = "arbitrage"                  # 套利策略
    MARKET_MAKING = "market_making"          # 做市策略
    STATISTICAL_ARB = "statistical_arb"      # 统计套利
    MACHINE_LEARNING = "machine_learning"    # 机器学习
    MULTI_FACTOR = "multi_factor"            # 多因子
    HYBRID = "hybrid"                        # 混合策略


class StrategyStatus(Enum):
    """策略状态"""
    DRAFT = "draft"              # 草稿
    TESTING = "testing"          # 测试中
    PAPER_TRADING = "paper"      # 模拟交易
    LIVE = "live"                # 实盘运行
    PAUSED = "paused"            # 暂停
    DEPRECATED = "deprecated"     # 废弃


@dataclass
class StrategyParameter:
    """策略参数"""
    name: str
    value: Any
    type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    is_optimizable: bool = True


@dataclass
class StrategyMetadata:
    """策略元数据"""
    id: str
    name: str
    type: StrategyType
    status: StrategyStatus
    version: str
    author: str
    created_at: datetime
    updated_at: datetime
    description: str
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, metadata: StrategyMetadata, parameters: Dict[str, StrategyParameter]):
        self.metadata = metadata
        self.parameters = parameters
        self.logger = logging.getLogger(f"{__name__}.{metadata.name}")
        
    @abstractmethod
    async def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """验证参数"""
        pass
    
    def get_parameter(self, name: str) -> Any:
        """获取参数值"""
        if name in self.parameters:
            return self.parameters[name].value
        return None
    
    def set_parameter(self, name: str, value: Any):
        """设置参数值"""
        if name in self.parameters:
            self.parameters[name].value = value
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            'metadata': self.metadata.to_dict(),
            'parameters': {
                name: {
                    'value': param.value,
                    'type': param.type,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'description': param.description
                }
                for name, param in self.parameters.items()
            }
        }


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self):
        metadata = StrategyMetadata(
            id=self._generate_id(),
            name="TrendFollowing",
            type=StrategyType.TREND_FOLLOWING,
            status=StrategyStatus.TESTING,
            version="1.0.0",
            author="System",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="基于移动平均线的趋势跟踪策略",
            tags=["MA", "Trend", "Classic"]
        
        parameters = {
            'fast_ma': StrategyParameter(
                name='fast_ma',
                value=5,
                type='int',
                min_value=2,
                max_value=50,
                description='快速移动平均线周期'
            ),
            'slow_ma': StrategyParameter(
                name='slow_ma',
                value=20,
                type='int',
                min_value=10,
                max_value=200,
                description='慢速移动平均线周期'
            ),
            'stop_loss': StrategyParameter(
                name='stop_loss',
                value=0.05,
                type='float',
                min_value=0.01,
                max_value=0.2,
                description='止损比例'
            ),
            'take_profit': StrategyParameter(
                name='take_profit',
                value=0.15,
                type='float',
                min_value=0.05,
                max_value=0.5,
                description='止盈比例'
        }
        
        super().__init__(metadata, parameters)
    
    def _generate_id(self) -> str:
        """生成策略ID"""
        return hashlib.md5(
            f"TrendFollowing_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        if len(market_data) < self.get_parameter('slow_ma'):
            return {'signal': 'HOLD', 'confidence': 0}
        
        # 计算移动平均线
        fast_ma = market_data['close'].rolling(
            window=self.get_parameter('fast_ma')
        ).mean()
        slow_ma = market_data['close'].rolling(
            window=self.get_parameter('slow_ma')
        ).mean()
        
        # 生成信号
        if fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2]:
            return {
                'signal': 'BUY',
                'confidence': 0.8,
                'reason': 'Golden Cross'
            }
        elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and fast_ma.iloc[-2] >= slow_ma.iloc[-2]:
            return {
                'signal': 'SELL',
                'confidence': 0.8,
                'reason': 'Death Cross'
            }
        else:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'reason': 'No clear signal'
            }
    
    def validate_parameters(self) -> bool:
        """验证参数"""
        fast_ma = self.get_parameter('fast_ma')
        slow_ma = self.get_parameter('slow_ma')
        
        if fast_ma >= slow_ma:
            self.logger.error("Fast MA period must be less than Slow MA period")
            return False
        
        return True


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self):
        metadata = StrategyMetadata(
            id=self._generate_id(),
            name="MeanReversion",
            type=StrategyType.MEAN_REVERSION,
            status=StrategyStatus.TESTING,
            version="1.0.0",
            author="System",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="基于布林带的均值回归策略",
            tags=["Bollinger", "MeanReversion", "Volatility"]
        
        parameters = {
            'lookback': StrategyParameter(
                name='lookback',
                value=20,
                type='int',
                min_value=10,
                max_value=100,
                description='回看周期'
            ),
            'num_std': StrategyParameter(
                name='num_std',
                value=2.0,
                type='float',
                min_value=1.0,
                max_value=3.0,
                description='标准差倍数'
            ),
            'entry_threshold': StrategyParameter(
                name='entry_threshold',
                value=0.95,
                type='float',
                min_value=0.8,
                max_value=1.0,
                description='入场阈值'
            ),
            'exit_threshold': StrategyParameter(
                name='exit_threshold',
                value=0.5,
                type='float',
                min_value=0.3,
                max_value=0.7,
                description='出场阈值'
        }
        
        super().__init__(metadata, parameters)
    
    def _generate_id(self) -> str:
        """生成策略ID"""
        return hashlib.md5(
            f"MeanReversion_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        lookback = self.get_parameter('lookback')
        
        if len(market_data) < lookback:
            return {'signal': 'HOLD', 'confidence': 0}
        
        # 计算布林带
        close_prices = market_data['close']
        ma = close_prices.rolling(window=lookback).mean()
        std = close_prices.rolling(window=lookback).std()
        
        upper_band = ma + self.get_parameter('num_std') * std
        lower_band = ma - self.get_parameter('num_std') * std
        
        current_price = close_prices.iloc[-1]
        
        # 计算位置
        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        # 生成信号
        if bb_position < (1 - self.get_parameter('entry_threshold')):
            return {
                'signal': 'BUY',
                'confidence': 0.7,
                'reason': 'Price near lower band'
            }
        elif bb_position > self.get_parameter('entry_threshold'):
            return {
                'signal': 'SELL',
                'confidence': 0.7,
                'reason': 'Price near upper band'
            }
        else:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'reason': 'Price within bands'
            }
    
    def validate_parameters(self) -> bool:
        """验证参数"""
        return True


class StrategyManager:
    """策略管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_history: Dict[str, List[Dict]] = {}
        self.storage_path = Path(self.config.get('storage_path', './strategies'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化版本控制
        self._init_version_control()
        
        # 加载已有策略
        self._load_strategies()
    
    def _init_version_control(self):
        """初始化版本控制"""
        try:
            self.repo = git.Repo(self.storage_path)
        except Exception:self.repo = git.Repo.init(self.storage_path)
        self.repo.index.commit("Initial commit")
    
    def _load_strategies(self):
        """加载策略"""
        strategy_files = self.storage_path.glob("*.json")
        
        for file in strategy_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    strategy = self._deserialize_strategy(data)
                    if strategy:
                        self.strategies[strategy.metadata.id] = strategy
                        logger.info(f"Loaded strategy: {strategy.metadata.name}")
            except Exception as e:
                logger.error(f"Failed to load strategy from {file}: {e}")
    
    def _deserialize_strategy(self, data: Dict) -> Optional[BaseStrategy]:
        """反序列化策略"""
        try:
            strategy_type = StrategyType(data['metadata']['type'])
            
            if strategy_type == StrategyType.TREND_FOLLOWING:
                strategy = TrendFollowingStrategy()
            elif strategy_type == StrategyType.MEAN_REVERSION:
                strategy = MeanReversionStrategy()
            else:
                return None
            
            # 恢复参数
            for name, param_data in data['parameters'].items():
                if name in strategy.parameters:
                    strategy.parameters[name].value = param_data['value']
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to deserialize strategy: {e}")
            return None
    
    def create_strategy(self, 
                       name: str,
                       type: StrategyType,
                       parameters: Dict[str, Any]) -> str:
        """创建策略"""
        # 根据类型创建策略实例
        if type == StrategyType.TREND_FOLLOWING:
            strategy = TrendFollowingStrategy()
        elif type == StrategyType.MEAN_REVERSION:
            strategy = MeanReversionStrategy()
        else:
            raise ValueError(f"Unsupported strategy type: {type}")
        
        # 设置参数
        for param_name, param_value in parameters.items():
            strategy.set_parameter(param_name, param_value)
        
        # 验证参数
        if not strategy.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        # 保存策略
        self.strategies[strategy.metadata.id] = strategy
        self._save_strategy(strategy)
        
        logger.info(f"Created strategy: {strategy.metadata.name} (ID: {strategy.metadata.id})")
        return strategy.metadata.id
    
    def _save_strategy(self, strategy: BaseStrategy):
        """保存策略"""
        file_path = self.storage_path / f"{strategy.metadata.id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(strategy.to_dict(), f, indent=2, default=str)
        
        # 提交到版本控制
        self.repo.index.add([str(file_path)])
        self.repo.index.commit(f"Save strategy {strategy.metadata.name} v{strategy.metadata.version}")
    
    def update_strategy(self, 
                       strategy_id: str,
                       parameters: Optional[Dict[str, Any]] = None,
                       status: Optional[StrategyStatus] = None) -> bool:
        """更新策略"""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        strategy = self.strategies[strategy_id]
        
        # 保存历史版本
        self._save_history(strategy)
        
        # 更新参数
        if parameters:
            for param_name, param_value in parameters.items():
                strategy.set_parameter(param_name, param_value)
        
        # 更新状态
        if status:
            strategy.metadata.status = status
        
        # 更新版本号
        version_parts = strategy.metadata.version.split('.')
        version_parts[2] = str(int(version_parts[2]) + 1)
        strategy.metadata.version = '.'.join(version_parts)
        
        strategy.metadata.updated_at = datetime.now()
        
        # 保存更新
        self._save_strategy(strategy)
        
        logger.info(f"Updated strategy {strategy_id} to version {strategy.metadata.version}")
        return True
    
    def _save_history(self, strategy: BaseStrategy):
        """保存历史版本"""
        if strategy.metadata.id not in self.strategy_history:
            self.strategy_history[strategy.metadata.id] = []
        
        self.strategy_history[strategy.metadata.id].append({
            'version': strategy.metadata.version,
            'timestamp': datetime.now(),
            'snapshot': strategy.to_dict()
        })
    
    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """获取策略"""
        return self.strategies.get(strategy_id)
    
    def list_strategies(self, 
                       type: Optional[StrategyType] = None,
                       status: Optional[StrategyStatus] = None) -> List[BaseStrategy]:
        """列出策略"""
        strategies = list(self.strategies.values())
        
        if type:
            strategies = [s for s in strategies if s.metadata.type == type]
        
        if status:
            strategies = [s for s in strategies if s.metadata.status == status]
        
        return strategies
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """删除策略（标记为废弃）"""
        if strategy_id not in self.strategies:
            return False
        
        strategy = self.strategies[strategy_id]
        strategy.metadata.status = StrategyStatus.DEPRECATED
        self._save_strategy(strategy)
        
        logger.info(f"Deprecated strategy {strategy_id}")
        return True
    
    def clone_strategy(self, strategy_id: str, new_name: str) -> str:
        """克隆策略"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        original = self.strategies[strategy_id]
        
        # 创建新策略
        if isinstance(original, TrendFollowingStrategy):
            new_strategy = TrendFollowingStrategy()
        elif isinstance(original, MeanReversionStrategy):
            new_strategy = MeanReversionStrategy()
        else:
            raise ValueError("Unknown strategy type")
        
        # 复制参数
        for param_name in original.parameters:
            new_strategy.set_parameter(
                param_name,
                original.get_parameter(param_name)
        
        # 更新元数据
        new_strategy.metadata.name = new_name
        new_strategy.metadata.created_at = datetime.now()
        new_strategy.metadata.updated_at = datetime.now()
        
        # 保存新策略
        self.strategies[new_strategy.metadata.id] = new_strategy
        self._save_strategy(new_strategy)
        
        logger.info(f"Cloned strategy {strategy_id} as {new_name}")
        return new_strategy.metadata.id


class StrategyOptimizer:
    """策略参数优化器"""
    
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.logger = logging.getLogger(f"{__name__}.Optimizer")
    
    def optimize(self,
                market_data: pd.DataFrame,
                objective: str = 'sharpe',
                n_trials: int = 100,
                n_jobs: int = -1) -> Dict[str, Any]:
        """优化策略参数"""
        
        def objective_function(trial):
            # 采样参数
            params = {}
            for name, param in self.strategy.parameters.items():
                if not param.is_optimizable:
                    continue
                
                if param.type == 'int':
                    value = trial.suggest_int(
                        name,
                        param.min_value,
                        param.max_value
                elif param.type == 'float':
                    value = trial.suggest_float(
                        name,
                        param.min_value,
                        param.max_value
                else:
                    continue
                
                params[name] = value
            
            # 设置参数
            for name, value in params.items():
                self.strategy.set_parameter(name, value)
            
            # 验证参数
            if not self.strategy.validate_parameters():
                return -1000
            
            # 回测
            performance = self._backtest(market_data)
            
            # 返回目标值
            if objective == 'sharpe':
                return performance.get('sharpe_ratio', 0)
            elif objective == 'returns':
                return performance.get('total_return', 0)
            elif objective == 'calmar':
                return performance.get('calmar_ratio', 0)
            else:
                return performance.get('sharpe_ratio', 0)
        
        # 创建优化研究
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_function, n_trials=n_trials, n_jobs=n_jobs)
        
        # 获取最优参数
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Optimization complete. Best {objective}: {best_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'optimization_history': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params
                }
                for t in study.trials
            ]
        }
    
    def _backtest(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """回测策略"""
        # 简化的回测逻辑
        signals = []
        
        # 生成信号
        for i in range(50, len(market_data)):
            data_slice = market_data.iloc[:i+1]
            signal = asyncio.run(self.strategy.generate_signal(data_slice))
            signals.append(signal)
        
        # 计算绩效
        returns = self._calculate_returns(signals, market_data[50:])
        
        # 计算指标
        if len(returns) == 0:
            return {'sharpe_ratio': 0, 'total_return': 0}
        
        total_return = np.prod(1 + returns) - 1
        
        if np.std(returns) == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        max_drawdown = self._calculate_max_drawdown(returns)
        
        calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': len([s for s in signals if s['signal'] != 'HOLD'])
        }
    
    def _calculate_returns(self, 
                          signals: List[Dict],
                          market_data: pd.DataFrame) -> np.ndarray:
        """计算收益率"""
        returns = []
        position = 0  # 0: 空仓, 1: 多头, -1: 空头
        
        for i, signal in enumerate(signals):
            if i >= len(market_data) - 1:
                break
            
            price_change = (market_data['close'].iloc[i+1] - market_data['close'].iloc[i]) / market_data['close'].iloc[i]
            
            if signal['signal'] == 'BUY' and position <= 0:
                position = 1
            elif signal['signal'] == 'SELL' and position >= 0:
                position = -1
            
            if position != 0:
                returns.append(position * price_change)
            else:
                returns.append(0)
        
        return np.array(returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0


class StrategyValidator:
    """策略验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Validator")
    
    def validate(self, 
                strategy: BaseStrategy,
                market_data: pd.DataFrame,
                validation_type: str = 'walk_forward') -> Dict[str, Any]:
        """验证策略"""
        
        if validation_type == 'walk_forward':
            return self._walk_forward_validation(strategy, market_data)
        elif validation_type == 'cross_validation':
            return self._cross_validation(strategy, market_data)
        else:
            return self._simple_validation(strategy, market_data)
    
    def _walk_forward_validation(self,
                                strategy: BaseStrategy,
                                market_data: pd.DataFrame) -> Dict[str, Any]:
        """前向验证"""
        window_size = 252  # 1年
        step_size = 63     # 3个月
        
        results = []
        
        for i in range(window_size, len(market_data), step_size):
            train_data = market_data.iloc[i-window_size:i]
            test_data = market_data.iloc[i:min(i+step_size, len(market_data))]
            
            if len(test_data) < 10:
                break
            
            # 在训练集上优化
            optimizer = StrategyOptimizer(strategy)
            opt_result = optimizer.optimize(train_data, n_trials=20)
            
            # 在测试集上验证
            for name, value in opt_result['best_params'].items():
                strategy.set_parameter(name, value)
            
            performance = optimizer._backtest(test_data)
            
            results.append({
                'period': f"{i-window_size}:{i}",
                'performance': performance
            })
        
        # 汇总结果
        avg_sharpe = np.mean([r['performance']['sharpe_ratio'] for r in results])
        avg_return = np.mean([r['performance']['total_return'] for r in results])
        
        return {
            'method': 'walk_forward',
            'num_windows': len(results),
            'avg_sharpe': avg_sharpe,
            'avg_return': avg_return,
            'details': results
        }
    
    def _cross_validation(self,
                         strategy: BaseStrategy,
                         market_data: pd.DataFrame) -> Dict[str, Any]:
        """交叉验证"""
        tscv = TimeSeriesSplit(n_splits=5)
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(market_data)):
            train_data = market_data.iloc[train_idx]
            test_data = market_data.iloc[test_idx]
            
            # 优化
            optimizer = StrategyOptimizer(strategy)
            opt_result = optimizer.optimize(train_data, n_trials=20)
            
            # 验证
            for name, value in opt_result['best_params'].items():
                strategy.set_parameter(name, value)
            
            performance = optimizer._backtest(test_data)
            
            results.append({
                'fold': fold,
                'performance': performance
            })
        
        # 汇总
        avg_sharpe = np.mean([r['performance']['sharpe_ratio'] for r in results])
        std_sharpe = np.std([r['performance']['sharpe_ratio'] for r in results])
        
        return {
            'method': 'cross_validation',
            'num_folds': len(results),
            'avg_sharpe': avg_sharpe,
            'std_sharpe': std_sharpe,
            'details': results
        }
    
    def _simple_validation(self,
                         strategy: BaseStrategy,
                         market_data: pd.DataFrame) -> Dict[str, Any]:
        """简单验证"""
        # 8:2 划分
        split_point = int(len(market_data) * 0.8)
        train_data = market_data.iloc[:split_point]
        test_data = market_data.iloc[split_point:]
        
        # 优化
        optimizer = StrategyOptimizer(strategy)
        opt_result = optimizer.optimize(train_data, n_trials=50)
        
        # 设置最优参数
        for name, value in opt_result['best_params'].items():
            strategy.set_parameter(name, value)
        
        # 验证
        train_performance = optimizer._backtest(train_data)
        test_performance = optimizer._backtest(test_data)
        
        return {
            'method': 'simple',
            'train_performance': train_performance,
            'test_performance': test_performance,
            'overfit_ratio': train_performance['sharpe_ratio'] / max(test_performance['sharpe_ratio'], 0.01)
        }


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test():
        # 创建策略管理器
        manager = StrategyManager()
        
        # 创建趋势策略
        strategy_id = manager.create_strategy(
            name="MA_Crossover",
            type=StrategyType.TREND_FOLLOWING,
            parameters={
                'fast_ma': 10,
                'slow_ma': 30,
                'stop_loss': 0.03,
                'take_profit': 0.1
            }
        
        print(f"Created strategy: {strategy_id}")
        
        # 获取策略
        strategy = manager.get_strategy(strategy_id)
        
        if strategy:
            # 生成模拟数据
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            prices = 100 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.01))
            market_data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            })
            
            # 生成信号
            signal = await strategy.generate_signal(market_data)
            print(f"Signal: {signal}")
            
            # 优化策略
            optimizer = StrategyOptimizer(strategy)
            opt_result = optimizer.optimize(market_data, n_trials=10)
            print(f"Optimization result: {opt_result['best_params']}")
            
            # 验证策略
            validator = StrategyValidator()
            validation = validator.validate(strategy, market_data, 'simple')
            print(f"Validation result: {validation}")
    
    # 运行测试
    # asyncio.run(test())
