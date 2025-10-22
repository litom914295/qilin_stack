"""
分布式计算系统
使用Dask实现分布式计算，并行分析多只股票
支持本地多进程和分布式集群两种模式
"""

import logging
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


# ============================================================================
# 分布式模式枚举
# ============================================================================

class DistributedMode(Enum):
    """分布式模式"""
    LOCAL = "local"           # 本地多进程
    CLUSTER = "cluster"       # 分布式集群
    THREADS = "threads"       # 多线程


@dataclass
class ClusterConfig:
    """集群配置"""
    mode: DistributedMode
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit: str = "4GB"
    scheduler_address: Optional[str] = None


# ============================================================================
# Dask分布式管理器
# ============================================================================

class DaskDistributedManager:
    """Dask分布式计算管理器"""
    
    def __init__(self, config: ClusterConfig):
        """
        初始化分布式管理器
        
        Args:
            config: 集群配置
        """
        self.config = config
        self.client = None
        self.cluster = None
        
        self._setup_cluster()
        logger.info(f"Dask集群初始化: {config.mode.value}, {config.n_workers}个Worker")
    
    def _setup_cluster(self):
        """设置集群"""
        try:
            import dask
            from dask.distributed import Client, LocalCluster
            
            if self.config.mode == DistributedMode.LOCAL:
                # 本地多进程集群
                self.cluster = LocalCluster(
                    n_workers=self.config.n_workers,
                    threads_per_worker=self.config.threads_per_worker,
                    memory_limit=self.config.memory_limit
                )
                self.client = Client(self.cluster)
                
            elif self.config.mode == DistributedMode.CLUSTER:
                # 连接到已有集群
                if not self.config.scheduler_address:
                    raise ValueError("集群模式需要提供scheduler_address")
                self.client = Client(self.config.scheduler_address)
                
            elif self.config.mode == DistributedMode.THREADS:
                # 多线程模式
                dask.config.set(scheduler='threads', num_workers=self.config.n_workers)
                
            logger.info(f"Dask集群设置完成")
            
        except ImportError:
            logger.error("Dask未安装，请运行: pip install dask distributed")
            raise
        except Exception as e:
            logger.error(f"集群设置失败: {e}")
            raise
    
    def get_dashboard_link(self) -> Optional[str]:
        """获取Dashboard链接"""
        if self.client:
            return self.client.dashboard_link
        return None
    
    def shutdown(self):
        """关闭集群"""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        logger.info("Dask集群已关闭")


# ============================================================================
# 分布式股票分析
# ============================================================================

class DistributedStockAnalyzer:
    """分布式股票分析器"""
    
    def __init__(self, manager: DaskDistributedManager):
        """
        初始化分析器
        
        Args:
            manager: Dask管理器
        """
        self.manager = manager
        self.client = manager.client
    
    def analyze_stocks_parallel(self, 
                                symbols: List[str],
                                data_dict: Dict[str, pd.DataFrame],
                                analysis_func: Callable) -> Dict[str, Any]:
        """
        并行分析多只股票
        
        Args:
            symbols: 股票代码列表
            data_dict: 股票数据字典 {symbol: df}
            analysis_func: 分析函数
            
        Returns:
            分析结果字典
        """
        try:
            import dask
            from dask import delayed
            
            # 创建延迟任务
            tasks = []
            for symbol in symbols:
                if symbol in data_dict:
                    task = delayed(analysis_func)(symbol, data_dict[symbol])
                    tasks.append((symbol, task))
            
            # 并行执行
            logger.info(f"开始并行分析 {len(tasks)} 只股票")
            start_time = time.time()
            
            results = {}
            if self.client:
                # 使用Client执行
                futures = {symbol: self.client.submit(analysis_func, symbol, data_dict[symbol])
                          for symbol in symbols if symbol in data_dict}
                results = {symbol: future.result() for symbol, future in futures.items()}
            else:
                # 使用Dask默认调度器
                computed = dask.compute(*[task for _, task in tasks])
                results = {symbol: result for (symbol, _), result in zip(tasks, computed)}
            
            elapsed = time.time() - start_time
            logger.info(f"并行分析完成: {elapsed:.2f}秒, 平均{elapsed/len(tasks):.2f}秒/股")
            
            return results
            
        except Exception as e:
            logger.error(f"并行分析失败: {e}")
            return {}
    
    def backtest_parallel(self,
                         strategies: List[Dict[str, Any]],
                         data: pd.DataFrame,
                         backtest_func: Callable) -> List[Dict[str, Any]]:
        """
        并行回测多个策略
        
        Args:
            strategies: 策略配置列表
            data: 数据
            backtest_func: 回测函数
            
        Returns:
            回测结果列表
        """
        try:
            import dask
            from dask import delayed
            
            # 创建延迟任务
            tasks = [delayed(backtest_func)(strategy, data) for strategy in strategies]
            
            logger.info(f"开始并行回测 {len(strategies)} 个策略")
            start_time = time.time()
            
            # 执行
            results = dask.compute(*tasks)
            
            elapsed = time.time() - start_time
            logger.info(f"并行回测完成: {elapsed:.2f}秒")
            
            return list(results)
            
        except Exception as e:
            logger.error(f"并行回测失败: {e}")
            return []
    
    def optimize_parameters_parallel(self,
                                    param_grid: Dict[str, List[Any]],
                                    optimization_func: Callable) -> List[Dict[str, Any]]:
        """
        并行参数优化
        
        Args:
            param_grid: 参数网格
            optimization_func: 优化函数
            
        Returns:
            优化结果列表
        """
        try:
            import dask
            from dask import delayed
            import itertools
            
            # 生成参数组合
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
            
            logger.info(f"参数组合总数: {len(param_combinations)}")
            
            # 创建延迟任务
            tasks = [delayed(optimization_func)(params) for params in param_combinations]
            
            start_time = time.time()
            results = dask.compute(*tasks)
            elapsed = time.time() - start_time
            
            logger.info(f"参数优化完成: {elapsed:.2f}秒, 测试了{len(param_combinations)}组参数")
            
            return list(results)
            
        except Exception as e:
            logger.error(f"参数优化失败: {e}")
            return []


# ============================================================================
# 分布式数据处理
# ============================================================================

class DistributedDataProcessor:
    """分布式数据处理器"""
    
    def __init__(self, manager: DaskDistributedManager):
        """
        初始化处理器
        
        Args:
            manager: Dask管理器
        """
        self.manager = manager
        self.client = manager.client
    
    def process_large_dataset(self, 
                             file_path: str,
                             processing_func: Callable) -> pd.DataFrame:
        """
        分布式处理大数据集
        
        Args:
            file_path: 文件路径（支持通配符）
            processing_func: 处理函数
            
        Returns:
            处理后的DataFrame
        """
        try:
            import dask.dataframe as dd
            
            # 读取数据
            logger.info(f"读取数据: {file_path}")
            ddf = dd.read_csv(file_path)
            
            # 应用处理函数
            logger.info("应用处理函数")
            ddf_processed = ddf.map_partitions(processing_func)
            
            # 计算结果
            logger.info("计算结果")
            result = ddf_processed.compute()
            
            logger.info(f"处理完成: {len(result)} 行")
            return result
            
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            return pd.DataFrame()
    
    def calculate_indicators_distributed(self,
                                        df: pd.DataFrame,
                                        indicator_funcs: Dict[str, Callable]) -> pd.DataFrame:
        """
        分布式计算多个技术指标
        
        Args:
            df: 数据DataFrame
            indicator_funcs: 指标函数字典 {name: func}
            
        Returns:
            添加了指标的DataFrame
        """
        try:
            import dask.dataframe as dd
            from dask import delayed
            
            # 转换为Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=self.manager.config.n_workers)
            
            # 应用每个指标函数
            for name, func in indicator_funcs.items():
                logger.info(f"计算指标: {name}")
                ddf[name] = ddf.map_partitions(func, meta=(name, 'float64'))
            
            # 计算
            result = ddf.compute()
            
            logger.info(f"分布式指标计算完成: {len(indicator_funcs)} 个指标")
            return result
            
        except Exception as e:
            logger.error(f"指标计算失败: {e}")
            return df
    
    def aggregate_multi_symbol_data(self,
                                   data_sources: List[str],
                                   aggregation_func: Callable) -> pd.DataFrame:
        """
        聚合多只股票数据
        
        Args:
            data_sources: 数据源列表（文件路径或URL）
            aggregation_func: 聚合函数
            
        Returns:
            聚合后的DataFrame
        """
        try:
            import dask.dataframe as dd
            
            # 读取所有数据源
            ddf_list = [dd.read_csv(source) for source in data_sources]
            
            # 合并
            ddf_combined = dd.concat(ddf_list)
            
            # 应用聚合函数
            result = aggregation_func(ddf_combined).compute()
            
            logger.info(f"聚合完成: {len(data_sources)} 个数据源")
            return result
            
        except Exception as e:
            logger.error(f"数据聚合失败: {e}")
            return pd.DataFrame()


# ============================================================================
# 分布式因子计算
# ============================================================================

class DistributedFactorCalculator:
    """分布式因子计算器"""
    
    def __init__(self, manager: DaskDistributedManager):
        """
        初始化因子计算器
        
        Args:
            manager: Dask管理器
        """
        self.manager = manager
        self.client = manager.client
    
    def calculate_factors_batch(self,
                                symbols: List[str],
                                data_dict: Dict[str, pd.DataFrame],
                                factor_definitions: Dict[str, str]) -> pd.DataFrame:
        """
        批量计算因子
        
        Args:
            symbols: 股票代码列表
            data_dict: 数据字典
            factor_definitions: 因子定义字典 {factor_name: expression}
            
        Returns:
            因子值DataFrame
        """
        try:
            from dask import delayed
            import dask
            
            def calc_single_stock(symbol, df, factors):
                """单只股票因子计算"""
                result = {'symbol': symbol}
                for factor_name, expr in factors.items():
                    try:
                        # 简化的因子计算（实际应使用Qlib表达式引擎）
                        result[factor_name] = eval(expr, {'df': df, 'np': np})
                    except Exception as e:
                        result[factor_name] = np.nan
                return result
            
            # 创建任务
            tasks = [delayed(calc_single_stock)(symbol, data_dict[symbol], factor_definitions)
                    for symbol in symbols if symbol in data_dict]
            
            # 并行计算
            logger.info(f"开始计算 {len(tasks)} 只股票的 {len(factor_definitions)} 个因子")
            start_time = time.time()
            
            results = dask.compute(*tasks)
            
            elapsed = time.time() - start_time
            logger.info(f"因子计算完成: {elapsed:.2f}秒")
            
            # 转换为DataFrame
            factors_df = pd.DataFrame(list(results))
            return factors_df
            
        except Exception as e:
            logger.error(f"因子计算失败: {e}")
            return pd.DataFrame()
    
    def rolling_factor_calculation(self,
                                  data: pd.DataFrame,
                                  window: int,
                                  factor_func: Callable) -> pd.DataFrame:
        """
        滚动因子计算
        
        Args:
            data: 数据
            window: 滚动窗口
            factor_func: 因子计算函数
            
        Returns:
            因子值DataFrame
        """
        try:
            import dask.dataframe as dd
            
            ddf = dd.from_pandas(data, npartitions=self.manager.config.n_workers)
            
            # 滚动计算
            result = ddf.rolling(window).apply(factor_func, raw=False).compute()
            
            logger.info(f"滚动因子计算完成: 窗口={window}")
            return result
            
        except Exception as e:
            logger.error(f"滚动计算失败: {e}")
            return pd.DataFrame()


# ============================================================================
# 使用示例
# ============================================================================

def example_distributed_computing():
    """分布式计算示例"""
    print("=== 分布式计算系统示例 ===\n")
    
    # 1. 初始化集群
    print("1. 初始化Dask集群")
    config = ClusterConfig(
        mode=DistributedMode.LOCAL,
        n_workers=4,
        threads_per_worker=2,
        memory_limit="2GB"
    )
    
    manager = DaskDistributedManager(config)
    dashboard = manager.get_dashboard_link()
    if dashboard:
        print(f"  Dashboard: {dashboard}")
    
    # 2. 并行股票分析
    print("\n2. 并行股票分析")
    symbols = [f"SH60{i:04d}" for i in range(10)]
    data_dict = {
        symbol: pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }) for symbol in symbols
    }
    
    def analyze_single_stock(symbol, df):
        """单股分析函数"""
        return {
            'symbol': symbol,
            'mean_price': df['close'].mean(),
            'volatility': df['close'].std(),
            'total_volume': df['volume'].sum()
        }
    
    analyzer = DistributedStockAnalyzer(manager)
    results = analyzer.analyze_stocks_parallel(symbols, data_dict, analyze_single_stock)
    print(f"  分析了 {len(results)} 只股票")
    
    # 3. 并行回测
    print("\n3. 并行策略回测")
    strategies = [
        {'name': f'Strategy_{i}', 'param1': i*0.1, 'param2': i*2}
        for i in range(5)
    ]
    
    def backtest_strategy(strategy, data):
        """回测函数"""
        # 简化的回测逻辑
        return {
            'strategy': strategy['name'],
            'return': np.random.random(),
            'sharpe': np.random.random() * 2
        }
    
    df_test = pd.DataFrame({'close': np.random.randn(252).cumsum() + 100})
    backtest_results = analyzer.backtest_parallel(strategies, df_test, backtest_strategy)
    print(f"  回测了 {len(backtest_results)} 个策略")
    
    # 4. 参数优化
    print("\n4. 并行参数优化")
    param_grid = {
        'window': [5, 10, 20],
        'threshold': [0.01, 0.02, 0.03]
    }
    
    def optimize_params(params):
        """优化函数"""
        # 模拟优化
        score = np.random.random()
        return {'params': params, 'score': score}
    
    opt_results = analyzer.optimize_parameters_parallel(param_grid, optimize_params)
    print(f"  测试了 {len(opt_results)} 组参数")
    best_result = max(opt_results, key=lambda x: x['score'])
    print(f"  最佳参数: {best_result['params']}, 得分: {best_result['score']:.4f}")
    
    # 5. 关闭集群
    print("\n5. 关闭集群")
    manager.shutdown()
    print("  集群已关闭")


if __name__ == "__main__":
    example_distributed_computing()
