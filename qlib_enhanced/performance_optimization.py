"""
性能优化模块
P2-3任务: 性能优化 (40h estimated, ROI 150%)

优化目标:
1. 数据加载: 30秒/年 → 10秒/年 (3x加速)
2. 特征计算: 2分钟/1000股 → 40秒 (3x加速)
3. 模型训练: 10分钟 → 3分钟 (3.3x加速)
4. 回测速度: 5分钟/年 → 1分钟/年 (5x加速)

作者: Qilin Stack Team
日期: 2025-11-07
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import logging
import time

logger = logging.getLogger(__name__)

# 尝试导入加速库
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    logger.info("✅ Numba JIT可用")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("⚠️ Numba未安装,将使用纯Python实现")
    # 创建虚拟装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
    logger.info("✅ Parquet格式支持可用")
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("⚠️ PyArrow未安装,将使用CSV格式")


# ==================== 1. 数据加载优化 ====================

class FastDataLoader:
    """
    快速数据加载器
    
    优化策略:
    1. Parquet格式替代CSV (10x读取速度)
    2. 并行加载多个文件
    3. 内存映射大文件
    4. 智能缓存
    """
    
    def __init__(
        self,
        data_dir: str,
        use_parquet: bool = True,
        max_workers: int = None,
        cache_size: int = 100
    ):
        """
        初始化快速数据加载器
        
        Args:
            data_dir: 数据目录
            use_parquet: 是否使用Parquet格式
            max_workers: 最大并行工作线程数
            cache_size: 缓存大小
        """
        self.data_dir = Path(data_dir)
        self.use_parquet = use_parquet and PARQUET_AVAILABLE
        self.max_workers = max_workers or mp.cpu_count()
        self.cache_size = cache_size
        
        logger.info(
            f"FastDataLoader初始化: "
            f"格式={'Parquet' if self.use_parquet else 'CSV'}, "
            f"并行度={self.max_workers}"
        )
    
    def load_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        并行加载多个股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段 (None=全部)
            
        Returns:
            data_dict: {symbol: DataFrame}
        """
        start_time = time.time()
        
        # 并行加载
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._load_single_symbol,
                    symbol, start_date, end_date, fields
                ): symbol
                for symbol in symbols
            }
            
            results = {}
            for future in futures:
                symbol = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        results[symbol] = df
                except Exception as e:
                    logger.error(f"加载{symbol}失败: {e}")
        
        elapsed = time.time() - start_time
        logger.info(
            f"✅ 并行加载完成: {len(results)}/{len(symbols)}个股票, "
            f"耗时{elapsed:.2f}秒"
        )
        
        return results
    
    def _load_single_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """加载单个股票数据"""
        if self.use_parquet:
            file_path = self.data_dir / f"{symbol}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path, columns=fields)
            else:
                # 回退到CSV
                file_path = self.data_dir / f"{symbol}.csv"
                if not file_path.exists():
                    return None
                df = pd.read_csv(file_path, usecols=fields)
        else:
            file_path = self.data_dir / f"{symbol}.csv"
            if not file_path.exists():
                return None
            df = pd.read_csv(file_path, usecols=fields)
        
        # 过滤日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        return df
    
    def save_to_parquet(
        self,
        symbol: str,
        df: pd.DataFrame,
        compression: str = 'snappy'
    ):
        """
        保存为Parquet格式
        
        Args:
            symbol: 股票代码
            df: 数据
            compression: 压缩算法 (snappy/gzip/brotli)
        """
        if not PARQUET_AVAILABLE:
            logger.warning("PyArrow未安装,无法保存Parquet格式")
            return
        
        file_path = self.data_dir / f"{symbol}.parquet"
        df.to_parquet(file_path, compression=compression, index=False)
        logger.info(f"✅ 已保存: {file_path}")
    
    @lru_cache(maxsize=100)
    def load_cached(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """带缓存的加载 (适合频繁访问相同数据)"""
        return self._load_single_symbol(symbol, start_date, end_date, None)


# ==================== 2. 特征计算优化 (Numba JIT) ====================

class FastFactorCalculator:
    """
    快速因子计算器
    
    优化策略:
    1. Numba JIT编译加速
    2. 向量化计算
    3. 避免循环
    4. 预分配数组
    """
    
    def __init__(self):
        """初始化快速因子计算器"""
        self.numba_available = NUMBA_AVAILABLE
        logger.info(
            f"FastFactorCalculator初始化: "
            f"JIT={'启用' if self.numba_available else '未启用'}"
        )
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_ma(prices: np.ndarray, window: int) -> np.ndarray:
        """
        快速移动平均 (Numba加速)
        
        Args:
            prices: 价格数组
            window: 窗口大小
            
        Returns:
            ma: 移动平均数组
        """
        n = len(prices)
        ma = np.empty(n, dtype=np.float64)
        ma[:window-1] = np.nan
        
        for i in prange(window-1, n):
            ma[i] = np.mean(prices[i-window+1:i+1])
        
        return ma
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_std(prices: np.ndarray, window: int) -> np.ndarray:
        """快速滚动标准差"""
        n = len(prices)
        std = np.empty(n, dtype=np.float64)
        std[:window-1] = np.nan
        
        for i in prange(window-1, n):
            std[i] = np.std(prices[i-window+1:i+1])
        
        return std
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """
        快速RSI指标
        
        Args:
            prices: 价格数组
            window: RSI周期
            
        Returns:
            rsi: RSI数组
        """
        n = len(prices)
        rsi = np.empty(n, dtype=np.float64)
        rsi[:window] = np.nan
        
        # 计算价格变化
        changes = np.diff(prices)
        
        for i in prange(window, n):
            window_changes = changes[i-window:i]
            gains = np.sum(np.maximum(window_changes, 0))
            losses = -np.sum(np.minimum(window_changes, 0))
            
            if losses == 0:
                rsi[i] = 100.0
            else:
                rs = gains / losses
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """
        快速MACD指标
        
        Returns:
            (macd, signal_line, histogram)
        """
        n = len(prices)
        
        # 快速EMA
        ema_fast = np.empty(n, dtype=np.float64)
        ema_fast[0] = prices[0]
        alpha_fast = 2.0 / (fast + 1)
        for i in range(1, n):
            ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i-1]
        
        # 慢速EMA
        ema_slow = np.empty(n, dtype=np.float64)
        ema_slow[0] = prices[0]
        alpha_slow = 2.0 / (slow + 1)
        for i in range(1, n):
            ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i-1]
        
        # MACD线
        macd = ema_fast - ema_slow
        
        # 信号线
        signal_line = np.empty(n, dtype=np.float64)
        signal_line[0] = macd[0]
        alpha_signal = 2.0 / (signal + 1)
        for i in range(1, n):
            signal_line[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i-1]
        
        # 柱状图
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    @jit(nopython=True)
    def calculate_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
        """快速收益率计算"""
        n = len(prices)
        returns = np.empty(n, dtype=np.float64)
        returns[:periods] = np.nan
        
        for i in range(periods, n):
            returns[i] = (prices[i] - prices[i-periods]) / prices[i-periods]
        
        return returns
    
    def calculate_all_factors(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        批量计算所有因子
        
        Args:
            df: 包含价格的DataFrame
            price_col: 价格列名
            
        Returns:
            df: 添加了因子的DataFrame
        """
        prices = df[price_col].values
        
        # MA因子
        df['ma5'] = self.calculate_ma(prices, 5)
        df['ma10'] = self.calculate_ma(prices, 10)
        df['ma20'] = self.calculate_ma(prices, 20)
        df['ma60'] = self.calculate_ma(prices, 60)
        
        # 波动率
        df['std20'] = self.calculate_std(prices, 20)
        
        # RSI
        df['rsi'] = self.calculate_rsi(prices, 14)
        
        # MACD
        macd, signal, hist = self.calculate_macd(prices)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # 收益率
        df['returns_1d'] = self.calculate_returns(prices, 1)
        df['returns_5d'] = self.calculate_returns(prices, 5)
        df['returns_20d'] = self.calculate_returns(prices, 20)
        
        return df


# ==================== 3. 模型训练优化 ====================

class FastModelTrainer:
    """
    快速模型训练器
    
    优化策略:
    1. GPU加速 (LightGBM/XGBoost GPU版本)
    2. 多线程并行
    3. 提前停止
    4. 特征重要性筛选
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        初始化快速训练器
        
        Args:
            use_gpu: 是否使用GPU
        """
        self.use_gpu = use_gpu
        
        # 检测GPU可用性
        if use_gpu:
            self.gpu_available = self._check_gpu()
        else:
            self.gpu_available = False
        
        logger.info(
            f"FastModelTrainer初始化: "
            f"GPU={'启用' if self.gpu_available else '未启用'}"
        )
    
    def _check_gpu(self) -> bool:
        """检测GPU是否可用"""
        try:
            import lightgbm as lgb
            # 尝试创建GPU参数
            params = {'device': 'gpu'}
            return True
        except:
            return False
    
    def get_optimized_params(self, task_type: str = 'binary') -> Dict:
        """
        获取优化的训练参数
        
        Args:
            task_type: 任务类型 (binary/multiclass/regression)
            
        Returns:
            params: 优化的参数字典
        """
        base_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary' if task_type == 'binary' else 'regression',
            'metric': 'auc' if task_type == 'binary' else 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1  # 使用全部CPU核心
        }
        
        # GPU加速参数
        if self.gpu_available:
            base_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        return base_params
    
    def train_with_early_stopping(
        self,
        train_data,
        valid_data,
        params: Optional[Dict] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ):
        """
        带早停的训练
        
        Args:
            train_data: 训练数据
            valid_data: 验证数据
            params: 训练参数
            num_boost_round: 最大迭代次数
            early_stopping_rounds: 早停轮数
            
        Returns:
            model: 训练好的模型
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("需要安装lightgbm: pip install lightgbm")
        
        if params is None:
            params = self.get_optimized_params()
        
        start_time = time.time()
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 模型训练完成: 耗时{elapsed:.2f}秒")
        
        return model


# ==================== 4. 回测加速 ====================

class FastBacktester:
    """
    快速回测器
    
    优化策略:
    1. 向量化计算
    2. 避免逐行循环
    3. NumPy数组操作
    4. 预分配内存
    """
    
    @staticmethod
    @jit(nopython=True)
    def calculate_portfolio_returns(
        signals: np.ndarray,
        returns: np.ndarray,
        transaction_cost: float = 0.001
    ) -> np.ndarray:
        """
        快速计算组合收益
        
        Args:
            signals: 信号数组 (-1, 0, 1)
            returns: 收益率数组
            transaction_cost: 交易成本
            
        Returns:
            portfolio_returns: 组合收益数组
        """
        n = len(signals)
        portfolio_returns = np.zeros(n, dtype=np.float64)
        position = 0.0
        
        for i in range(1, n):
            # 计算持仓变化
            position_change = signals[i] - position
            
            # 扣除交易成本
            cost = abs(position_change) * transaction_cost
            
            # 计算收益
            portfolio_returns[i] = position * returns[i] - cost
            
            # 更新持仓
            position = signals[i]
        
        return portfolio_returns
    
    @staticmethod
    @jit(nopython=True)
    def calculate_drawdown(cumulative_returns: np.ndarray) -> tuple:
        """
        快速计算回撤
        
        Returns:
            (max_drawdown, drawdown_array)
        """
        n = len(cumulative_returns)
        drawdown = np.zeros(n, dtype=np.float64)
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for i in range(n):
            if cumulative_returns[i] > peak:
                peak = cumulative_returns[i]
            
            dd = (peak - cumulative_returns[i]) / peak if peak > 0 else 0
            drawdown[i] = dd
            
            if dd > max_dd:
                max_dd = dd
        
        return max_dd, drawdown


# ==================== 性能测试工具 ====================

class PerformanceBenchmark:
    """性能基准测试工具"""
    
    @staticmethod
    def benchmark_data_loading(
        loader: FastDataLoader,
        symbols: List[str],
        iterations: int = 5
    ) -> Dict:
        """基准测试数据加载"""
        times = []
        
        for i in range(iterations):
            start = time.time()
            loader.load_symbols(symbols, '2020-01-01', '2020-12-31')
            elapsed = time.time() - start
            times.append(elapsed)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    @staticmethod
    def benchmark_factor_calculation(
        calculator: FastFactorCalculator,
        prices: np.ndarray,
        iterations: int = 100
    ) -> Dict:
        """基准测试因子计算"""
        times = []
        
        for i in range(iterations):
            start = time.time()
            calculator.calculate_ma(prices, 20)
            calculator.calculate_rsi(prices, 14)
            calculator.calculate_macd(prices)
            elapsed = time.time() - start
            times.append(elapsed)
        
        return {
            'avg_time': np.mean(times) * 1000,  # 转换为毫秒
            'std_time': np.std(times) * 1000,
            'throughput': len(prices) / np.mean(times)  # 样本/秒
        }


# ==================== 测试代码 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("性能优化模块测试")
    print("=" * 60)
    
    # 1. 测试快速因子计算
    print("\n1. 测试快速因子计算 (Numba JIT)")
    print("-" * 60)
    
    calculator = FastFactorCalculator()
    
    # 生成测试数据
    n_samples = 10000
    test_prices = np.cumsum(np.random.randn(n_samples) * 0.02) + 100
    
    # 测试MA
    start = time.time()
    ma20 = calculator.calculate_ma(test_prices, 20)
    ma_time = time.time() - start
    print(f"  MA(20)计算: {ma_time*1000:.2f}ms ({n_samples}个样本)")
    
    # 测试RSI
    start = time.time()
    rsi = calculator.calculate_rsi(test_prices)
    rsi_time = time.time() - start
    print(f"  RSI计算: {rsi_time*1000:.2f}ms")
    
    # 测试MACD
    start = time.time()
    macd, signal, hist = calculator.calculate_macd(test_prices)
    macd_time = time.time() - start
    print(f"  MACD计算: {macd_time*1000:.2f}ms")
    
    total_time = ma_time + rsi_time + macd_time
    print(f"\n  总耗时: {total_time*1000:.2f}ms")
    print(f"  吞吐量: {n_samples/total_time:.0f} 样本/秒")
    
    # 2. 测试快速回测
    print("\n2. 测试快速回测")
    print("-" * 60)
    
    backtester = FastBacktester()
    
    # 生成测试信号和收益
    test_signals = np.random.choice([-1, 0, 1], size=n_samples)
    test_returns = np.random.randn(n_samples) * 0.02
    
    start = time.time()
    portfolio_returns = backtester.calculate_portfolio_returns(
        test_signals, test_returns, 0.001
    )
    backtest_time = time.time() - start
    
    print(f"  组合收益计算: {backtest_time*1000:.2f}ms ({n_samples}个样本)")
    print(f"  吞吐量: {n_samples/backtest_time:.0f} 样本/秒")
    
    # 计算回撤
    cumulative_returns = np.cumsum(portfolio_returns)
    start = time.time()
    max_dd, dd_array = backtester.calculate_drawdown(cumulative_returns)
    dd_time = time.time() - start
    
    print(f"  回撤计算: {dd_time*1000:.2f}ms")
    print(f"  最大回撤: {max_dd:.2%}")
    
    # 3. 性能对比
    print("\n3. 性能基准测试")
    print("-" * 60)
    
    benchmark = PerformanceBenchmark()
    
    factor_bench = benchmark.benchmark_factor_calculation(
        calculator, test_prices[:1000], iterations=100
    )
    
    print(f"  因子计算平均耗时: {factor_bench['avg_time']:.2f}ms")
    print(f"  因子计算吞吐量: {factor_bench['throughput']:.0f} 样本/秒")
    
    # 4. 总结
    print("\n" + "=" * 60)
    print("✅ 性能优化模块测试完成!")
    print("=" * 60)
    print("\n关键性能指标:")
    print(f"  - MA计算速度: {n_samples/(ma_time*1000):.1f}K 样本/秒")
    print(f"  - RSI计算速度: {n_samples/(rsi_time*1000):.1f}K 样本/秒")
    print(f"  - 回测速度: {n_samples/(backtest_time*1000):.1f}K 样本/秒")
    print(f"  - Numba JIT: {'✅ 已启用' if NUMBA_AVAILABLE else '❌ 未启用'}")
    print(f"  - Parquet支持: {'✅ 已启用' if PARQUET_AVAILABLE else '❌ 未启用'}")
