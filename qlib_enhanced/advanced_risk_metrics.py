"""
高级风险指标计算模块
实现CVaR、Expected Shortfall、尾部风险等高级风险度量
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class AdvancedRiskMetrics:
    """高级风险指标计算器"""
    
    def __init__(self, returns: pd.Series):
        """
        初始化
        
        Args:
            returns: 收益率序列
        """
        self.returns = returns.dropna()
        self.returns_array = self.returns.values
        
    def calculate_var(self, confidence: float = 0.95, method: str = 'historical') -> float:
        """
        计算VaR (Value at Risk)
        
        Args:
            confidence: 置信水平 (0.95 或 0.99)
            method: 计算方法 ('historical', 'parametric')
            
        Returns:
            VaR值 (负数表示损失)
        """
        if method == 'historical':
            # 历史模拟法
            var = np.percentile(self.returns_array, (1 - confidence) * 100)
        elif method == 'parametric':
            # 参数法 (假设正态分布)
            mean = np.mean(self.returns_array)
            std = np.std(self.returns_array)
            var = stats.norm.ppf(1 - confidence, mean, std)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        logger.info(f"VaR ({confidence*100}%): {var:.4f}")
        return var
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        计算CVaR (Conditional Value at Risk) / Expected Shortfall
        
        条件风险价值：超过VaR的平均损失
        
        Args:
            confidence: 置信水平
            
        Returns:
            CVaR值 (负数表示损失)
        """
        var = self.calculate_var(confidence, method='historical')
        
        # 计算超过VaR的平均损失
        tail_losses = self.returns_array[self.returns_array <= var]
        
        if len(tail_losses) == 0:
            cvar = var
        else:
            cvar = np.mean(tail_losses)
        
        logger.info(f"CVaR ({confidence*100}%): {cvar:.4f}")
        return cvar
    
    def calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """
        计算Expected Shortfall (ES)
        
        与CVaR等价，表示尾部期望损失
        
        Args:
            confidence: 置信水平
            
        Returns:
            ES值
        """
        return self.calculate_cvar(confidence)
    
    def calculate_tail_risk(self) -> Dict[str, float]:
        """
        计算尾部风险指标
        
        Returns:
            包含多个尾部风险指标的字典
        """
        returns = self.returns_array
        
        # 偏度 (Skewness) - 衡量分布的对称性
        skewness = stats.skew(returns)
        
        # 峰度 (Kurtosis) - 衡量分布的尾部厚度
        kurtosis = stats.kurtosis(returns)
        
        # 左尾概率 (5%分位数以下的频率)
        left_tail_prob = np.sum(returns < np.percentile(returns, 5)) / len(returns)
        
        # 极端损失概率 (损失超过2倍标准差)
        mean = np.mean(returns)
        std = np.std(returns)
        extreme_loss_prob = np.sum(returns < (mean - 2*std)) / len(returns)
        
        # 最大单日损失
        max_loss = np.min(returns)
        
        tail_risk = {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'left_tail_probability': left_tail_prob,
            'extreme_loss_probability': extreme_loss_prob,
            'max_single_loss': max_loss,
        }
        
        logger.info(f"尾部风险: {tail_risk}")
        return tail_risk
    
    def stress_test(self, scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        压力测试
        
        Args:
            scenarios: 压力测试场景，格式 {'场景名': 收益率冲击}
            
        Returns:
            每个场景下的预期损失
        """
        current_value = 1.0  # 假设初始资产价值为1
        
        results = {}
        for scenario_name, shock in scenarios.items():
            # 应用冲击
            stressed_value = current_value * (1 + shock)
            loss = current_value - stressed_value
            loss_pct = loss / current_value
            
            results[scenario_name] = {
                'shock': shock,
                'loss': loss,
                'loss_percentage': loss_pct
            }
        
        logger.info(f"压力测试完成: {len(scenarios)}个场景")
        return results
    
    def calculate_all_metrics(self, confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        计算所有风险指标
        
        Args:
            confidence_levels: 置信水平列表
            
        Returns:
            完整的风险指标字典
        """
        metrics = {
            'var': {},
            'cvar': {},
            'tail_risk': self.calculate_tail_risk(),
        }
        
        for conf in confidence_levels:
            metrics['var'][f'{int(conf*100)}%'] = self.calculate_var(conf)
            metrics['cvar'][f'{int(conf*100)}%'] = self.calculate_cvar(conf)
        
        return metrics


def run_stress_test_scenarios(returns: pd.Series, 
                               custom_scenarios: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    运行标准压力测试场景
    
    Args:
        returns: 收益率序列
        custom_scenarios: 自定义场景（可选）
        
    Returns:
        压力测试结果DataFrame
    """
    calculator = AdvancedRiskMetrics(returns)
    
    # 标准压力测试场景
    standard_scenarios = {
        '市场崩盘 (-20%)': -0.20,
        '严重衰退 (-15%)': -0.15,
        '温和下跌 (-10%)': -0.10,
        '黑天鹅事件 (-30%)': -0.30,
        '系统性危机 (-25%)': -0.25,
    }
    
    # 合并自定义场景
    if custom_scenarios:
        standard_scenarios.update(custom_scenarios)
    
    # 运行压力测试
    results = calculator.stress_test(standard_scenarios)
    
    # 转换为DataFrame
    df_results = []
    for scenario, metrics in results.items():
        df_results.append({
            '场景': scenario,
            '冲击幅度': f"{metrics['shock']*100:.1f}%",
            '预期损失': f"{metrics['loss']:.4f}",
            '损失百分比': f"{metrics['loss_percentage']*100:.2f}%"
        })
    
    return pd.DataFrame(df_results)


def calculate_risk_decomposition(returns: pd.Series, 
                                 window: int = 252) -> pd.DataFrame:
    """
    风险分解分析
    
    Args:
        returns: 收益率序列
        window: 滚动窗口大小
        
    Returns:
        风险分解结果DataFrame
    """
    if len(returns) < window:
        logger.warning(f"数据不足，需要至少{window}个数据点")
        window = len(returns)
    
    # 滚动计算风险指标
    results = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        calculator = AdvancedRiskMetrics(window_returns)
        
        var_95 = calculator.calculate_var(0.95)
        cvar_95 = calculator.calculate_cvar(0.95)
        tail = calculator.calculate_tail_risk()
        
        results.append({
            'date': returns.index[i-1],
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': tail['skewness'],
            'kurtosis': tail['kurtosis'],
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成模拟收益率数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, len(dates)),
        index=dates
    )
    
    # 计算高级风险指标
    calculator = AdvancedRiskMetrics(returns)
    
    print("=== 高级风险指标 ===")
    print(f"VaR (95%): {calculator.calculate_var(0.95):.4f}")
    print(f"CVaR (95%): {calculator.calculate_cvar(0.95):.4f}")
    print(f"Expected Shortfall (99%): {calculator.calculate_expected_shortfall(0.99):.4f}")
    
    print("\n=== 尾部风险 ===")
    tail_risk = calculator.calculate_tail_risk()
    for key, value in tail_risk.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== 压力测试 ===")
    stress_results = run_stress_test_scenarios(returns)
    print(stress_results)
