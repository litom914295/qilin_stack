"""
麒麟量化系统 - 高级技术指标库
包含市场微观结构、高频交易、机器学习等高级指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class MicrostructureIndicators:
    """市场微观结构指标"""
    
    @staticmethod
    def calculate_vpin(df: pd.DataFrame, volume_bucket_size: int = 50) -> pd.Series:
        """
        计算VPIN (Volume-Synchronized Probability of Informed Trading)
        
        Args:
            df: 包含price和volume的数据
            volume_bucket_size: 成交量桶大小
            
        Returns:
            VPIN值序列
        """
        # 计算价格变化
        df['price_change'] = df['close'].diff()
        
        # 使用Lee-Ready算法分类买卖单
        df['trade_sign'] = np.where(df['price_change'] > 0, 1, -1)
        df['buy_volume'] = np.where(df['trade_sign'] > 0, df['volume'], 0)
        df['sell_volume'] = np.where(df['trade_sign'] < 0, df['volume'], 0)
        
        # 创建成交量桶
        df['volume_bucket'] = (df['volume'].cumsum() // volume_bucket_size).astype(int)
        
        # 计算每个桶的买卖不平衡
        bucket_imbalance = df.groupby('volume_bucket').agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'volume': 'sum'
        })
        
        bucket_imbalance['abs_imbalance'] = abs(
            bucket_imbalance['buy_volume'] - bucket_imbalance['sell_volume']
        )
        
        # 计算VPIN
        vpin = bucket_imbalance['abs_imbalance'] / bucket_imbalance['volume']
        
        # 映射回原始索引
        df['vpin'] = df['volume_bucket'].map(vpin)
        
        return df['vpin'].fillna(method='ffill')
    
    @staticmethod
    def calculate_kyle_lambda(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        计算Kyle's Lambda (价格冲击系数)
        
        Args:
            df: 包含价格和成交量的数据
            window: 回归窗口
            
        Returns:
            Kyle's Lambda序列
        """
        # 计算价格变化和净买入量
        price_change = df['close'].pct_change()
        net_volume = df['volume'] * np.sign(df['close'].diff())
        
        # 滚动回归计算Lambda
        lambda_values = []
        
        for i in range(window, len(df)):
            y = price_change[i-window:i].values
            x = net_volume[i-window:i].values
            
            # 去除NaN值
            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < window // 2:
                lambda_values.append(np.nan)
                continue
            
            y_clean = y[mask]
            x_clean = x[mask].reshape(-1, 1)
            
            if len(x_clean) > 0 and x_clean.std() > 0:
                # 简单线性回归
                coef = np.linalg.lstsq(x_clean, y_clean, rcond=None)[0]
                lambda_values.append(abs(coef[0]))
            else:
                lambda_values.append(np.nan)
        
        # 补充前面的值
        lambda_series = pd.Series(
            [np.nan] * window + lambda_values,
            index=df.index
        )
        
        return lambda_series.fillna(method='ffill')
    
    @staticmethod
    def calculate_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        计算Amihud非流动性指标
        
        Args:
            df: 包含价格和成交额的数据
            window: 计算窗口
            
        Returns:
            Amihud非流动性指标
        """
        # 计算日收益率绝对值除以成交额
        returns = df['close'].pct_change().abs()
        dollar_volume = df['amount']  # 成交额
        
        # 避免除零
        dollar_volume = dollar_volume.replace(0, np.nan)
        
        # 计算每日的price impact
        daily_illiquidity = returns / dollar_volume * 1e6  # 放大系数
        
        # 滚动平均
        amihud = daily_illiquidity.rolling(window=window).mean()
        
        return amihud
    
    @staticmethod
    def calculate_roll_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        计算Roll有效买卖价差
        
        Args:
            df: 价格数据
            window: 计算窗口
            
        Returns:
            Roll价差估计
        """
        # 计算价格变化
        price_changes = df['close'].diff()
        
        # 滚动计算自协方差
        roll_spread = []
        
        for i in range(window, len(df)):
            changes = price_changes[i-window:i]
            
            # 计算自协方差
            cov = np.cov(changes[:-1], changes[1:])[0, 1]
            
            # Roll模型：spread = 2 * sqrt(-cov) if cov < 0
            if cov < 0:
                spread = 2 * np.sqrt(-cov)
            else:
                spread = 0
            
            roll_spread.append(spread)
        
        # 转换为Series
        roll_series = pd.Series(
            [np.nan] * window + roll_spread,
            index=df.index
        )
        
        return roll_series


class HighFrequencyIndicators:
    """高频交易指标"""
    
    @staticmethod
    def calculate_realized_volatility(
        df: pd.DataFrame, 
        freq: str = '5min',
        daily_periods: int = 48
    ) -> pd.Series:
        """
        计算已实现波动率
        
        Args:
            df: 高频价格数据
            freq: 数据频率
            daily_periods: 每日周期数
            
        Returns:
            已实现波动率
        """
        # 计算收益率
        returns = df['close'].pct_change()
        
        # 计算平方收益率的滚动和
        squared_returns = returns ** 2
        
        # 计算已实现波动率
        rv = np.sqrt(squared_returns.rolling(window=daily_periods).sum())
        
        return rv * np.sqrt(252)  # 年化
    
    @staticmethod
    def calculate_jump_detection(
        df: pd.DataFrame,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        检测价格跳跃
        
        Args:
            df: 价格数据
            threshold: 跳跃阈值（标准差倍数）
            
        Returns:
            包含跳跃标记的DataFrame
        """
        # 计算收益率
        returns = df['close'].pct_change()
        
        # 计算滚动标准差
        rolling_std = returns.rolling(window=20).std()
        
        # 标准化收益率
        standardized_returns = returns / rolling_std
        
        # 检测跳跃
        jumps = pd.DataFrame(index=df.index)
        jumps['is_jump'] = abs(standardized_returns) > threshold
        jumps['jump_size'] = np.where(jumps['is_jump'], returns, 0)
        jumps['jump_direction'] = np.sign(jumps['jump_size'])
        
        return jumps
    
    @staticmethod
    def calculate_order_flow_imbalance(
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        计算订单流不平衡
        
        Args:
            df: 包含买卖成交量的数据
            window: 计算窗口
            
        Returns:
            订单流不平衡指标
        """
        # 估算买卖成交量
        df['price_change'] = df['close'].diff()
        df['buy_volume'] = np.where(df['price_change'] > 0, df['volume'], 0)
        df['sell_volume'] = np.where(df['price_change'] < 0, df['volume'], 0)
        
        # 计算累积不平衡
        buy_cumsum = df['buy_volume'].rolling(window=window).sum()
        sell_cumsum = df['sell_volume'].rolling(window=window).sum()
        
        # 计算不平衡比率
        total_volume = buy_cumsum + sell_cumsum
        ofi = (buy_cumsum - sell_cumsum) / total_volume.replace(0, np.nan)
        
        return ofi.fillna(0)


class MachineLearningIndicators:
    """机器学习衍生指标"""
    
    @staticmethod
    def calculate_price_pattern_similarity(
        df: pd.DataFrame,
        lookback: int = 20,
        pattern_length: int = 10
    ) -> pd.Series:
        """
        计算价格形态相似度
        
        Args:
            df: 价格数据
            lookback: 历史回看期
            pattern_length: 形态长度
            
        Returns:
            相似度得分
        """
        prices = df['close'].values
        similarity_scores = []
        
        for i in range(lookback + pattern_length, len(prices)):
            # 当前形态
            current_pattern = prices[i-pattern_length:i]
            current_pattern = (current_pattern - current_pattern.mean()) / current_pattern.std()
            
            # 历史形态
            best_similarity = 0
            for j in range(i-lookback-pattern_length, i-pattern_length):
                hist_pattern = prices[j:j+pattern_length]
                hist_pattern = (hist_pattern - hist_pattern.mean()) / hist_pattern.std()
                
                # 计算相关性
                if not np.isnan(hist_pattern).any():
                    correlation = np.corrcoef(current_pattern, hist_pattern)[0, 1]
                    best_similarity = max(best_similarity, correlation)
            
            similarity_scores.append(best_similarity)
        
        # 转换为Series
        result = pd.Series(
            [np.nan] * (lookback + pattern_length) + similarity_scores,
            index=df.index
        )
        return result
    
    @staticmethod
    def calculate_regime_probability(
        df: pd.DataFrame,
        n_regimes: int = 3
    ) -> pd.DataFrame:
        """
        计算市场状态概率（简化版HMM）
        
        Args:
            df: 价格和成交量数据
            n_regimes: 状态数量
            
        Returns:
            各状态的概率
        """
        # 计算特征
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        volume_ratio = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # 组合特征
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'volume_ratio': volume_ratio
        }).dropna()
        
        # 使用K-means进行简单聚类
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 聚类
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        # 计算到各聚类中心的距离
        distances = kmeans.transform(features_scaled)
        
        # 转换为概率（使用softmax）
        probabilities = np.exp(-distances)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # 创建结果DataFrame
        result = pd.DataFrame(
            probabilities,
            columns=[f'regime_{i}_prob' for i in range(n_regimes)],
            index=features.index
        )
        
        # 补充缺失值
        result = result.reindex(df.index).fillna(method='ffill')
        
        return result
    
    @staticmethod
    def calculate_feature_importance_score(
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算特征重要性得分
        
        Args:
            df: 数据
            target_col: 目标列
            feature_cols: 特征列
            
        Returns:
            特征重要性字典
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        if feature_cols is None:
            feature_cols = ['volume', 'high', 'low', 'open']
        
        # 准备数据
        features = df[feature_cols].copy()
        target = df[target_col].shift(-1)  # 预测下一期
        
        # 去除NaN
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]
        
        if len(features) < 100:
            return {col: 0 for col in feature_cols}
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 训练随机森林
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf.fit(features_scaled, target)
        
        # 获取特征重要性
        importance_dict = {
            col: float(imp) 
            for col, imp in zip(feature_cols, rf.feature_importances_)
        }
        
        return importance_dict


class SentimentIndicators:
    """情绪指标"""
    
    @staticmethod
    def calculate_put_call_ratio(options_data: pd.DataFrame) -> pd.Series:
        """
        计算认沽认购比率
        
        Args:
            options_data: 期权数据
            
        Returns:
            PCR比率
        """
        if 'put_volume' not in options_data.columns:
            # 模拟数据
            return pd.Series(np.random.uniform(0.8, 1.2, len(options_data)), 
                           index=options_data.index)
        
        pcr = options_data['put_volume'] / options_data['call_volume'].replace(0, np.nan)
        return pcr.fillna(1.0)
    
    @staticmethod
    def calculate_market_sentiment_index(
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        计算市场情绪综合指数
        
        Args:
            df: 市场数据
            window: 计算窗口
            
        Returns:
            情绪指数（0-100）
        """
        sentiment_factors = []
        
        # 1. 价格动量
        momentum = df['close'].pct_change(window)
        momentum_score = (momentum + 0.1) / 0.2 * 100  # 标准化到0-100
        sentiment_factors.append(momentum_score)
        
        # 2. 成交量比率
        volume_ratio = df['volume'] / df['volume'].rolling(window).mean()
        volume_score = np.clip(volume_ratio * 50, 0, 100)
        sentiment_factors.append(volume_score)
        
        # 3. 波动率（反向）
        returns = df['close'].pct_change()
        volatility = returns.rolling(window).std()
        vol_score = 100 - np.clip(volatility * 1000, 0, 100)
        sentiment_factors.append(vol_score)
        
        # 4. 涨跌比率
        up_days = (df['close'].diff() > 0).rolling(window).sum()
        total_days = window
        advance_decline = up_days / total_days * 100
        sentiment_factors.append(advance_decline)
        
        # 综合计算
        sentiment_df = pd.DataFrame(sentiment_factors).T
        sentiment_index = sentiment_df.mean(axis=1)
        
        return np.clip(sentiment_index, 0, 100)


class NetworkIndicators:
    """网络分析指标"""
    
    @staticmethod
    def calculate_correlation_network_centrality(
        returns_df: pd.DataFrame,
        window: int = 60
    ) -> pd.Series:
        """
        计算相关性网络中心度
        
        Args:
            returns_df: 多只股票的收益率数据
            window: 相关性计算窗口
            
        Returns:
            网络中心度
        """
        if returns_df.shape[1] < 2:
            return pd.Series(0.5, index=returns_df.index)
        
        centrality_scores = []
        
        for i in range(window, len(returns_df)):
            # 计算相关性矩阵
            corr_matrix = returns_df.iloc[i-window:i].corr()
            
            # 计算每个节点的度中心性（相关性之和）
            centrality = corr_matrix.abs().sum(axis=1) / (len(corr_matrix) - 1)
            centrality_scores.append(centrality.mean())
        
        result = pd.Series(
            [np.nan] * window + centrality_scores,
            index=returns_df.index
        )
        
        return result
    
    @staticmethod
    def calculate_spillover_index(
        returns_df: pd.DataFrame,
        lag: int = 5
    ) -> pd.DataFrame:
        """
        计算溢出效应指数
        
        Args:
            returns_df: 多资产收益率
            lag: VAR模型滞后阶数
            
        Returns:
            溢出指数
        """
        from statsmodels.tsa.api import VAR
        
        if returns_df.shape[1] < 2 or len(returns_df) < 100:
            return pd.DataFrame(0, index=returns_df.index, 
                              columns=['spillover_index'])
        
        # 清理数据
        clean_data = returns_df.dropna()
        
        if len(clean_data) < lag * 2:
            return pd.DataFrame(0, index=returns_df.index,
                              columns=['spillover_index'])
        
        try:
            # 拟合VAR模型
            model = VAR(clean_data)
            results = model.fit(lag)
            
            # 获取预测误差方差分解
            fevd = results.fevd(10)
            
            # 计算溢出效应
            spillover_matrix = fevd.decomp[-1]
            
            # 总溢出指数
            n = spillover_matrix.shape[0]
            total_spillover = (spillover_matrix.sum() - np.diag(spillover_matrix).sum()) / n
            
            # 创建结果
            result = pd.DataFrame(
                total_spillover,
                index=returns_df.index,
                columns=['spillover_index']
            )
        
        except Exception:
            result = pd.DataFrame(0, index=returns_df.index,
                                columns=['spillover_index'])
        
        return result


class CompositeIndicators:
    """复合指标"""
    
    @staticmethod
    def calculate_market_quality_index(df: pd.DataFrame) -> pd.Series:
        """
        计算市场质量综合指数
        
        Args:
            df: 市场数据
            
        Returns:
            市场质量指数
        """
        micro = MicrostructureIndicators()
        
        # 计算各子指标
        spread = micro.calculate_roll_spread(df, window=20)
        illiquidity = micro.calculate_amihud_illiquidity(df, window=20)
        
        # 标准化
        spread_norm = (spread - spread.mean()) / spread.std()
        illiq_norm = (illiquidity - illiquidity.mean()) / illiquidity.std()
        
        # 反向指标（越低越好）
        quality_index = 100 - (spread_norm + illiq_norm) * 10
        
        return np.clip(quality_index, 0, 100)
    
    @staticmethod
    def calculate_trading_signal_strength(
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.Series:
        """
        计算综合交易信号强度
        
        Args:
            df: 包含多个指标的数据
            indicators: 要使用的指标列表
            
        Returns:
            信号强度（-100到100）
        """
        if indicators is None:
            # 使用所有可用的指标
            indicators = [col for col in df.columns if col not in 
                        ['open', 'high', 'low', 'close', 'volume', 'amount']]
        
        if not indicators:
            return pd.Series(0, index=df.index)
        
        # 标准化各指标
        signals = []
        for ind in indicators:
            if ind in df.columns:
                # Z-score标准化
                z_score = (df[ind] - df[ind].mean()) / df[ind].std()
                # 转换到-100到100
                signal = np.tanh(z_score) * 100
                signals.append(signal)
        
        if not signals:
            return pd.Series(0, index=df.index)
        
        # 计算平均信号强度
        signal_df = pd.DataFrame(signals).T
        signal_strength = signal_df.mean(axis=1)
        
        return signal_strength


def calculate_all_advanced_indicators(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    计算所有高级指标
    
    Args:
        df: 原始市场数据
        config: 配置参数
        
    Returns:
        包含所有指标的DataFrame
    """
    result = df.copy()
    config = config or {}
    
    # 微观结构指标
    micro = MicrostructureIndicators()
    result['vpin'] = micro.calculate_vpin(df)
    result['kyle_lambda'] = micro.calculate_kyle_lambda(df)
    result['amihud_illiq'] = micro.calculate_amihud_illiquidity(df)
    result['roll_spread'] = micro.calculate_roll_spread(df)
    
    # 高频指标
    hf = HighFrequencyIndicators()
    result['realized_vol'] = hf.calculate_realized_volatility(df)
    jump_df = hf.calculate_jump_detection(df)
    result['is_jump'] = jump_df['is_jump']
    result['jump_size'] = jump_df['jump_size']
    result['order_flow_imb'] = hf.calculate_order_flow_imbalance(df)
    
    # 机器学习指标
    ml = MachineLearningIndicators()
    result['pattern_similarity'] = ml.calculate_price_pattern_similarity(df)
    regime_probs = ml.calculate_regime_probability(df)
    for col in regime_probs.columns:
        result[col] = regime_probs[col]
    
    # 情绪指标
    sent = SentimentIndicators()
    result['market_sentiment'] = sent.calculate_market_sentiment_index(df)
    
    # 复合指标
    comp = CompositeIndicators()
    result['market_quality'] = comp.calculate_market_quality_index(df)
    result['signal_strength'] = comp.calculate_trading_signal_strength(result)
    
    return result


# 导出便捷函数
def get_indicator(
    df: pd.DataFrame,
    indicator_name: str,
    **kwargs
) -> pd.Series:
    """
    获取单个指标
    
    Args:
        df: 数据
        indicator_name: 指标名称
        **kwargs: 指标参数
        
    Returns:
        指标序列
    """
    indicator_map = {
        'vpin': MicrostructureIndicators.calculate_vpin,
        'kyle_lambda': MicrostructureIndicators.calculate_kyle_lambda,
        'amihud': MicrostructureIndicators.calculate_amihud_illiquidity,
        'roll_spread': MicrostructureIndicators.calculate_roll_spread,
        'realized_vol': HighFrequencyIndicators.calculate_realized_volatility,
        'order_flow_imb': HighFrequencyIndicators.calculate_order_flow_imbalance,
        'pattern_sim': MachineLearningIndicators.calculate_price_pattern_similarity,
        'market_sentiment': SentimentIndicators.calculate_market_sentiment_index,
        'market_quality': CompositeIndicators.calculate_market_quality_index,
        'signal_strength': CompositeIndicators.calculate_trading_signal_strength
    }
    
    if indicator_name not in indicator_map:
        raise ValueError(f"Unknown indicator: {indicator_name}")
    
    return indicator_map[indicator_name](df, **kwargs)


if __name__ == "__main__":
    # 测试代码
    import yfinance as yf
    
    # 获取测试数据
    # ticker = yf.Ticker("AAPL")
    # df = ticker.history(period="6mo")
    
    # 使用模拟数据测试
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    df = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'amount': np.random.randint(10000000, 100000000, len(dates))
    }, index=dates)
    
    # 计算所有指标
    result = calculate_all_advanced_indicators(df)
    
    print("高级指标计算完成")
    print(f"总指标数: {len(result.columns)}")
    print("\n主要指标统计:")
    print(result[['vpin', 'kyle_lambda', 'market_sentiment', 'signal_strength']].describe())
