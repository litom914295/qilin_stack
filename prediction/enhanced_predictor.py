"""
增强预测引擎
集成市场择时门控、流动性风险过滤、漂移监测等高级功能

功能：
- 基础模型预测
- 市场择时门控
- 流动性和风险过滤
- 数据漂移监测
- MLflow实验跟踪
- 特征缓存加速
"""

import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 导入风控模块
try:
    from risk.market_timing_gate import MarketTimingGate
    from risk.liquidity_risk_filter import LiquidityRiskFilter
except ImportError:
    warnings.warn("Risk modules not available")
    MarketTimingGate = None
    LiquidityRiskFilter = None

# 导入监控模块
try:
    from monitoring.drift_detector import DriftDetector
except ImportError:
    warnings.warn("Drift detector not available")
    DriftDetector = None

# 导入MLflow跟踪
try:
    from training.mlflow_tracker import MLflowTracker
except ImportError:
    warnings.warn("MLflow tracker not available")
    MLflowTracker = None

# 导入缓存
try:
    from cache.feature_cache import FeatureCache
except ImportError:
    warnings.warn("Feature cache not available")
    FeatureCache = None


class EnhancedPredictor:
    """
    增强预测引擎
    集成风控、监控、缓存等高级功能的预测系统
    """
    
    def __init__(
        self,
        model,
        enable_market_timing: bool = True,
        enable_liquidity_filter: bool = True,
        enable_drift_monitor: bool = True,
        enable_mlflow: bool = False,
        enable_cache: bool = True,
        config: Optional[Dict] = None
    ):
        """
        初始化增强预测引擎
        
        Args:
            model: 基础预测模型
            enable_market_timing: 是否启用市场择时
            enable_liquidity_filter: 是否启用流动性过滤
            enable_drift_monitor: 是否启用漂移监测
            enable_mlflow: 是否启用MLflow跟踪
            enable_cache: 是否启用特征缓存
            config: 配置字典
        """
        self.model = model
        self.config = config or {}
        
        # 初始化市场择时门控
        self.market_gate = None
        if enable_market_timing and MarketTimingGate:
            self.market_gate = MarketTimingGate(
                enable_timing=True,
                risk_threshold=self.config.get('risk_threshold', 0.5),
                sentiment_window=self.config.get('sentiment_window', 20)
            )
        
        # 初始化流动性过滤器
        self.liquidity_filter = None
        if enable_liquidity_filter and LiquidityRiskFilter:
            self.liquidity_filter = LiquidityRiskFilter(
                min_volume=self.config.get('min_volume', 1e8),
                min_turnover=self.config.get('min_turnover', 0.02),
                max_volatility=self.config.get('max_volatility', 0.15),
                min_price=self.config.get('min_price', 2.0),
                filter_st=True,
                filter_suspended=True
            )
        
        # 初始化漂移监测器
        self.drift_detector = None
        if enable_drift_monitor and DriftDetector:
            self.drift_detector = DriftDetector(
                n_bins=10,
                output_dir=self.config.get('drift_output_dir', './drift_monitoring')
            )
        
        # 初始化MLflow跟踪器
        self.mlflow_tracker = None
        if enable_mlflow and MLflowTracker:
            self.mlflow_tracker = MLflowTracker(
                experiment_name=self.config.get('mlflow_experiment', 'limitup_prediction'),
                tracking_uri=self.config.get('mlflow_uri', './mlruns')
            )
        
        # 初始化特征缓存
        self.feature_cache = None
        if enable_cache and FeatureCache:
            self.feature_cache = FeatureCache(
                cache_dir=self.config.get('cache_dir', './feature_cache'),
                ttl_hours=self.config.get('cache_ttl', 24)
            )
        
        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'market_gate_blocked': 0,
            'liquidity_filtered': 0,
            'drift_warnings': 0,
            'cache_hits': 0
        }
    
    def check_market_condition(
        self,
        market_data: pd.DataFrame
    ) -> Tuple[bool, str, Dict]:
        """
        检查市场条件
        
        Args:
            market_data: 市场数据
        
        Returns:
            (是否允许交易, 原因, 市场信号)
        """
        if self.market_gate is None:
            return True, "市场择时未启用", {}
        
        # 生成择时信号
        timing_signal = self.market_gate.generate_timing_signal(market_data)
        
        # 判断是否允许交易
        should_trade, reason = self.market_gate.should_trade(market_data)
        
        if not should_trade:
            self.stats['market_gate_blocked'] += 1
        
        return should_trade, reason, timing_signal
    
    def filter_candidates(
        self,
        candidates: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        过滤候选股票
        
        Args:
            candidates: 候选股票DataFrame
        
        Returns:
            (过滤后的候选股票, 过滤统计信息)
        """
        if self.liquidity_filter is None:
            return candidates, {'total': len(candidates), 'passed': len(candidates)}
        
        # 应用流动性过滤
        filter_stats = self.liquidity_filter.get_filter_stats(candidates)
        passed_candidates = self.liquidity_filter.get_passed_stocks(candidates)
        
        filtered_count = len(candidates) - len(passed_candidates)
        self.stats['liquidity_filtered'] += filtered_count
        
        return passed_candidates, filter_stats
    
    def check_data_drift(
        self,
        current_features: pd.DataFrame,
        baseline_features: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        检查数据漂移
        
        Args:
            current_features: 当前特征数据
            baseline_features: 基线特征数据
        
        Returns:
            漂移检测结果
        """
        if self.drift_detector is None:
            return {'drift_detected': False, 'message': '漂移监测未启用'}
        
        # 如果有基线数据，设置基线
        if baseline_features is not None:
            self.drift_detector.set_baseline(baseline_features)
        
        # 如果没有基线，返回
        if self.drift_detector.baseline_data is None:
            return {'drift_detected': False, 'message': '基线数据未设置'}
        
        # 计算PSI
        psi_df = self.drift_detector.compute_all_psi(current_features)
        
        # 检查是否有显著漂移
        significant_drift = psi_df[psi_df['psi'] >= 0.25]
        
        if len(significant_drift) > 0:
            self.stats['drift_warnings'] += 1
            return {
                'drift_detected': True,
                'n_drifted_features': len(significant_drift),
                'top_drifted': significant_drift.head(5).to_dict('records'),
                'message': f'检测到{len(significant_drift)}个特征显著漂移'
            }
        
        return {
            'drift_detected': False,
            'avg_psi': float(psi_df['psi'].mean()),
            'message': '未检测到显著漂移'
        }
    
    def predict(
        self,
        features: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        candidate_info: Optional[pd.DataFrame] = None,
        baseline_features: Optional[pd.DataFrame] = None,
        force_predict: bool = False
    ) -> Dict:
        """
        执行增强预测
        
        Args:
            features: 特征数据
            market_data: 市场数据（用于择时）
            candidate_info: 候选股票信息（用于过滤）
            baseline_features: 基线特征（用于漂移检测）
            force_predict: 是否强制预测（忽略门控）
        
        Returns:
            预测结果字典
        """
        self.stats['total_predictions'] += 1
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'predictions': None,
            'market_condition': None,
            'filter_stats': None,
            'drift_check': None,
            'status': 'success',
            'message': ''
        }
        
        # 1. 检查市场条件
        if market_data is not None and not force_predict:
            should_trade, reason, timing_signal = self.check_market_condition(market_data)
            result['market_condition'] = timing_signal
            
            if not should_trade:
                result['status'] = 'blocked'
                result['message'] = f'市场门控阻止交易: {reason}'
                return result
        
        # 2. 模型预测
        try:
            # 尝试从缓存获取预测（如果特征未变）
            pred_proba = self.model.predict_proba(features)
            
            # 处理单列或双列预测
            if pred_proba.ndim == 1:
                scores = pred_proba
            else:
                scores = pred_proba[:, 1] if pred_proba.shape[1] == 2 else pred_proba[:, 0]
            
            # 构建预测结果
            predictions = pd.DataFrame({
                'pred_score': scores,
                'pred_label': (scores >= self.config.get('threshold', 0.5)).astype(int)
            })
            
            # 添加索引信息
            if hasattr(features, 'index'):
                predictions.index = features.index
        
        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'预测失败: {str(e)}'
            return result
        
        # 3. 流动性和风险过滤
        if candidate_info is not None and self.liquidity_filter is not None:
            # 合并预测结果和候选信息
            merged = candidate_info.copy()
            merged['pred_score'] = predictions['pred_score'].values
            merged['pred_label'] = predictions['pred_label'].values
            
            # 过滤
            passed_candidates, filter_stats = self.filter_candidates(merged)
            result['filter_stats'] = filter_stats
            
            # 更新预测结果为过滤后的
            if len(passed_candidates) > 0:
                predictions = passed_candidates[['pred_score', 'pred_label']]
            else:
                result['status'] = 'filtered'
                result['message'] = '所有候选股票均被流动性过滤器过滤'
                return result
        
        # 4. 数据漂移检测
        if baseline_features is not None:
            drift_result = self.check_data_drift(features, baseline_features)
            result['drift_check'] = drift_result
            
            if drift_result.get('drift_detected'):
                result['message'] += f" 警告: {drift_result['message']}"
        
        # 5. 应用仓位调整因子（如果有市场择时）
        if self.market_gate is not None and market_data is not None:
            position_factor = self.market_gate.get_position_size_factor(market_data)
            predictions['position_factor'] = position_factor
        
        result['predictions'] = predictions
        result['n_predictions'] = len(predictions)
        
        # 6. MLflow记录
        if self.mlflow_tracker is not None:
            try:
                metrics = {
                    'n_predictions': len(predictions),
                    'avg_score': float(predictions['pred_score'].mean()),
                    'positive_rate': float(predictions['pred_label'].mean())
                }
                
                if result['market_condition']:
                    metrics['market_sentiment'] = result['market_condition']['sentiment']['overall_score']
                    metrics['market_risk'] = result['market_condition']['risk']['risk_score']
                
                self.mlflow_tracker.log_prediction_session(
                    predictions=predictions,
                    metrics=metrics,
                    tags={'enhanced_predictor': 'true'}
                )
            except:
                pass  # MLflow记录失败不影响预测
        
        return result
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加命中率
        if self.feature_cache:
            cache_stats = self.feature_cache.get_cache_stats()
            stats['cache_hit_rate'] = cache_stats.get('hit_rate', 0)
        
        # 计算过滤率
        if stats['total_predictions'] > 0:
            stats['market_block_rate'] = stats['market_gate_blocked'] / stats['total_predictions']
            stats['liquidity_filter_rate'] = stats['liquidity_filtered'] / max(stats['total_predictions'], 1)
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_predictions': 0,
            'market_gate_blocked': 0,
            'liquidity_filtered': 0,
            'drift_warnings': 0,
            'cache_hits': 0
        }


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试增强预测引擎"""
    
    print("=" * 60)
    print("增强预测引擎测试")
    print("=" * 60)
    
    # 创建模拟模型
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    
    # 训练简单模型
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 2, 1000)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("✓ 模拟模型已训练")
    
    # 创建增强预测引擎
    config = {
        'risk_threshold': 0.5,
        'min_volume': 5e7,
        'min_turnover': 0.01,
        'threshold': 0.5
    }
    
    predictor = EnhancedPredictor(
        model=model,
        enable_market_timing=True,
        enable_liquidity_filter=True,
        enable_drift_monitor=True,
        enable_mlflow=False,
        enable_cache=False,
        config=config
    )
    
    print("✓ 增强预测引擎已创建")
    
    # 准备测试数据
    print("\n" + "=" * 60)
    print("准备测试数据...")
    print("=" * 60)
    
    # 特征数据
    test_features = pd.DataFrame(
        np.random.randn(50, 50),
        columns=[f'feature_{i}' for i in range(50)]
    )
    
    # 市场数据
    from risk.market_timing_gate import create_mock_market_data
    market_data = create_mock_market_data(n_days=60)
    
    # 候选股票信息
    candidate_info = pd.DataFrame({
        'symbol': [f'{i:06d}' for i in range(50)],
        'close': np.random.lognormal(3, 0.5, 50),
        'volume': np.random.lognormal(20, 1, 50),
        'amount': np.random.lognormal(20, 1, 50),
        'turnover': np.random.lognormal(-3, 0.5, 50).clip(0.01, 0.5),
        'volatility': np.random.lognormal(-2.5, 0.3, 50).clip(0.01, 0.2),
        'market_cap': np.random.lognormal(22, 1, 50),
        'is_st': np.random.choice([True, False], 50, p=[0.05, 0.95]),
        'is_suspended': np.random.choice([True, False], 50, p=[0.02, 0.98])
    })
    
    print(f"特征数据: {test_features.shape}")
    print(f"市场数据: {len(market_data)} 天")
    print(f"候选股票: {len(candidate_info)} 只")
    
    # 执行预测
    print("\n" + "=" * 60)
    print("执行增强预测...")
    print("=" * 60)
    
    result = predictor.predict(
        features=test_features,
        market_data=market_data,
        candidate_info=candidate_info
    )
    
    print(f"\n预测状态: {result['status']}")
    print(f"消息: {result['message']}")
    
    if result['market_condition']:
        mc = result['market_condition']
        print(f"\n市场条件:")
        print(f"  信号: {mc['signal']}")
        print(f"  门控状态: {mc['gate_status']}")
        print(f"  情绪得分: {mc['sentiment']['overall_score']:.2f}")
        print(f"  风险得分: {mc['risk']['risk_score']:.2f}")
    
    if result['filter_stats']:
        fs = result['filter_stats']
        print(f"\n过滤统计:")
        print(f"  总数: {fs['total']}")
        print(f"  通过: {fs['passed']}")
        print(f"  通过率: {fs['pass_rate']:.1%}")
    
    if result['predictions'] is not None:
        preds = result['predictions']
        print(f"\n预测结果:")
        print(f"  预测数量: {len(preds)}")
        print(f"  平均得分: {preds['pred_score'].mean():.3f}")
        print(f"  正样本率: {preds['pred_label'].mean():.1%}")
        
        if 'position_factor' in preds.columns:
            print(f"  建议仓位: {preds['position_factor'].iloc[0]:.1%}")
        
        print(f"\n前10个预测:")
        print(preds.head(10).to_string())
    
    # 统计信息
    print("\n" + "=" * 60)
    print("预测引擎统计:")
    print("=" * 60)
    
    stats = predictor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试多次预测
    print("\n" + "=" * 60)
    print("测试多次预测...")
    print("=" * 60)
    
    for i in range(3):
        test_features_i = pd.DataFrame(
            np.random.randn(30, 50),
            columns=[f'feature_{i}' for i in range(50)]
        )
        
        result_i = predictor.predict(
            features=test_features_i,
            market_data=market_data,
            candidate_info=candidate_info.sample(30)
        )
        
        print(f"\n第{i+1}次预测: 状态={result_i['status']}, "
              f"预测数={result_i.get('n_predictions', 0)}")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("最终统计:")
    print("=" * 60)
    
    final_stats = predictor.get_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
