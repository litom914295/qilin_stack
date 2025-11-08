"""
市场择时门控系统
基于市场情绪指标判断交易时机，控制交易开关

功能：
- 计算多维度市场情绪指标
- 生成择时信号
- 风险等级评估
- 交易开关控制
- 支持多种择时策略
"""

import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class MarketTimingGate:
    """
    市场择时门控器
    根据市场情绪和技术指标判断是否适合交易
    """
    
    def __init__(
        self,
        enable_timing: bool = True,
        risk_threshold: float = 0.5,
        sentiment_window: int = 20
    ):
        """
        初始化择时门控器
        
        Args:
            enable_timing: 是否启用择时功能
            risk_threshold: 风险阈值（0-1，越高越保守）
            sentiment_window: 情绪计算窗口期（交易日）
        """
        self.enable_timing = enable_timing
        self.risk_threshold = risk_threshold
        self.sentiment_window = sentiment_window
        
        self.market_state = "unknown"
        self.gate_status = "open"  # "open", "closed", "restricted"
        self.last_update = None
    
    def compute_market_sentiment(
        self,
        market_data: pd.DataFrame
    ) -> Dict:
        """
        计算市场情绪指标
        
        Args:
            market_data: 市场数据，需包含列：
                - date: 日期
                - limitup_count: 涨停数量
                - limitdown_count: 跌停数量
                - index_return: 指数收益率
                - volume_ratio: 成交量比率
                - advance_decline: 上涨/下跌比率
        
        Returns:
            情绪指标字典
        """
        sentiment = {}
        
        # 1. 涨停/跌停比率
        limitup_rate = market_data['limitup_count'].rolling(self.sentiment_window).mean()
        limitdown_rate = market_data['limitdown_count'].rolling(self.sentiment_window).mean()
        
        sentiment['limitup_avg'] = float(limitup_rate.iloc[-1]) if len(limitup_rate) > 0 else 0
        sentiment['limitdown_avg'] = float(limitdown_rate.iloc[-1]) if len(limitdown_rate) > 0 else 0
        sentiment['limit_ratio'] = sentiment['limitup_avg'] / max(sentiment['limitdown_avg'], 1)
        
        # 2. 指数动量
        if 'index_return' in market_data.columns:
            index_momentum = market_data['index_return'].rolling(self.sentiment_window).sum()
            sentiment['index_momentum'] = float(index_momentum.iloc[-1]) if len(index_momentum) > 0 else 0
        else:
            sentiment['index_momentum'] = 0
        
        # 3. 市场活跃度（成交量）
        if 'volume_ratio' in market_data.columns:
            volume_score = market_data['volume_ratio'].rolling(self.sentiment_window).mean()
            sentiment['volume_activity'] = float(volume_score.iloc[-1]) if len(volume_score) > 0 else 1.0
        else:
            sentiment['volume_activity'] = 1.0
        
        # 4. 涨跌家数比
        if 'advance_decline' in market_data.columns:
            ad_ratio = market_data['advance_decline'].rolling(self.sentiment_window).mean()
            sentiment['advance_decline_ratio'] = float(ad_ratio.iloc[-1]) if len(ad_ratio) > 0 else 1.0
        else:
            sentiment['advance_decline_ratio'] = 1.0
        
        # 5. 综合情绪得分 (0-1)
        # 归一化各项指标并加权
        sentiment_score = 0.0
        sentiment_score += min(sentiment['limit_ratio'] / 5.0, 1.0) * 0.3  # 涨停比率权重30%
        sentiment_score += (1.0 if sentiment['index_momentum'] > 0 else 0.0) * 0.3  # 指数动量权重30%
        sentiment_score += min(sentiment['volume_activity'], 1.0) * 0.2  # 成交量权重20%
        sentiment_score += min(sentiment['advance_decline_ratio'], 1.0) * 0.2  # 涨跌比权重20%
        
        sentiment['overall_score'] = sentiment_score
        sentiment['timestamp'] = datetime.now().isoformat()
        
        return sentiment
    
    def assess_market_risk(
        self,
        sentiment: Dict,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        评估市场风险等级
        
        Args:
            sentiment: 市场情绪指标
            volatility: 市场波动率（可选）
        
        Returns:
            风险评估结果
        """
        risk_score = 0.0
        
        # 1. 情绪风险（情绪过低为高风险）
        if sentiment['overall_score'] < 0.3:
            risk_score += 0.4
        elif sentiment['overall_score'] < 0.5:
            risk_score += 0.2
        
        # 2. 涨停数量风险（过少或过多都有风险）
        if sentiment['limitup_avg'] < 10:
            risk_score += 0.3  # 市场冷清
        elif sentiment['limitup_avg'] > 150:
            risk_score += 0.2  # 市场过热
        
        # 3. 跌停风险
        if sentiment['limitdown_avg'] > 50:
            risk_score += 0.3  # 市场恐慌
        
        # 4. 波动率风险
        if volatility is not None:
            if volatility > 0.03:  # 日波动率超过3%
                risk_score += 0.3
        
        # 归一化到0-1
        risk_score = min(risk_score, 1.0)
        
        # 确定风险等级
        if risk_score < 0.3:
            risk_level = "low"
            risk_desc = "低风险"
        elif risk_score < 0.6:
            risk_level = "medium"
            risk_desc = "中等风险"
        else:
            risk_level = "high"
            risk_desc = "高风险"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_desc': risk_desc,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_timing_signal(
        self,
        market_data: pd.DataFrame,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        生成择时信号
        
        Args:
            market_data: 市场数据
            volatility: 市场波动率
        
        Returns:
            择时信号字典
        """
        if not self.enable_timing:
            return {
                'signal': 'neutral',
                'gate_status': 'open',
                'reason': '择时功能已禁用'
            }
        
        # 计算市场情绪
        sentiment = self.compute_market_sentiment(market_data)
        
        # 评估风险
        risk_assessment = self.assess_market_risk(sentiment, volatility)
        
        # 生成信号
        signal = 'neutral'
        gate_status = 'open'
        reason = ''
        
        # 判断逻辑
        if risk_assessment['risk_score'] > self.risk_threshold:
            if risk_assessment['risk_score'] > 0.7:
                signal = 'avoid'
                gate_status = 'closed'
                reason = f"高风险市场环境（风险分数={risk_assessment['risk_score']:.2f}）"
            else:
                signal = 'caution'
                gate_status = 'restricted'
                reason = f"中高风险市场环境（风险分数={risk_assessment['risk_score']:.2f}）"
        
        elif sentiment['overall_score'] > 0.6:
            signal = 'bullish'
            gate_status = 'open'
            reason = f"市场情绪积极（情绪分数={sentiment['overall_score']:.2f}）"
        
        elif sentiment['overall_score'] > 0.4:
            signal = 'neutral'
            gate_status = 'open'
            reason = f"市场情绪中性（情绪分数={sentiment['overall_score']:.2f}）"
        
        else:
            signal = 'caution'
            gate_status = 'restricted'
            reason = f"市场情绪低迷（情绪分数={sentiment['overall_score']:.2f}）"
        
        # 更新状态
        self.market_state = signal
        self.gate_status = gate_status
        self.last_update = datetime.now()
        
        return {
            'signal': signal,
            'gate_status': gate_status,
            'reason': reason,
            'sentiment': sentiment,
            'risk': risk_assessment,
            'timestamp': datetime.now().isoformat()
        }
    
    def should_trade(
        self,
        market_data: pd.DataFrame,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        判断是否应该交易
        
        Args:
            market_data: 市场数据
            force: 是否强制交易（忽略门控）
        
        Returns:
            (是否交易, 原因)
        """
        if force or not self.enable_timing:
            return True, "强制交易或择时功能已禁用"
        
        # 生成择时信号
        timing_signal = self.generate_timing_signal(market_data)
        
        gate_status = timing_signal['gate_status']
        
        if gate_status == 'closed':
            return False, timing_signal['reason']
        elif gate_status == 'restricted':
            return True, f"受限交易: {timing_signal['reason']}"
        else:  # open
            return True, timing_signal['reason']
    
    def get_position_size_factor(
        self,
        market_data: pd.DataFrame
    ) -> float:
        """
        根据市场状态调整仓位因子
        
        Args:
            market_data: 市场数据
        
        Returns:
            仓位调整因子 (0-1)
        """
        timing_signal = self.generate_timing_signal(market_data)
        
        sentiment_score = timing_signal['sentiment']['overall_score']
        risk_score = timing_signal['risk']['risk_score']
        
        # 根据情绪和风险计算仓位因子
        if timing_signal['gate_status'] == 'closed':
            return 0.0  # 完全停止交易
        
        elif timing_signal['gate_status'] == 'restricted':
            # 降低仓位
            base_factor = 0.5
            factor = base_factor * sentiment_score * (1 - risk_score)
            return max(0.2, min(factor, 0.6))  # 限制在20%-60%
        
        else:  # open
            # 正常或增加仓位
            factor = 0.5 + (sentiment_score - 0.5) * 0.5  # 基础50%，根据情绪调整
            factor = factor * (1 - risk_score * 0.3)  # 风险折扣
            return max(0.3, min(factor, 1.0))  # 限制在30%-100%
    
    def get_status_report(self) -> Dict:
        """获取当前状态报告"""
        return {
            'enable_timing': self.enable_timing,
            'market_state': self.market_state,
            'gate_status': self.gate_status,
            'risk_threshold': self.risk_threshold,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


def create_mock_market_data(n_days: int = 60) -> pd.DataFrame:
    """
    创建模拟市场数据（用于测试）
    
    Args:
        n_days: 天数
    
    Returns:
        市场数据DataFrame
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # 模拟牛熊市场切换
    market_phase = np.sin(np.arange(n_days) / 20.0) * 0.5 + 0.5  # 0-1之间波动
    
    data = {
        'date': dates,
        'limitup_count': (50 + market_phase * 100 + np.random.randn(n_days) * 20).clip(0, None).astype(int),
        'limitdown_count': (20 + (1 - market_phase) * 50 + np.random.randn(n_days) * 10).clip(0, None).astype(int),
        'index_return': market_phase * 0.02 + np.random.randn(n_days) * 0.015,
        'volume_ratio': 0.8 + market_phase * 0.4 + np.random.randn(n_days) * 0.2,
        'advance_decline': 0.5 + market_phase * 0.5 + np.random.randn(n_days) * 0.2
    }
    
    return pd.DataFrame(data)


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试市场择时门控系统"""
    
    print("=" * 60)
    print("市场择时门控系统测试")
    print("=" * 60)
    
    # 创建模拟市场数据
    market_data = create_mock_market_data(n_days=60)
    print(f"生成模拟市场数据: {len(market_data)} 天")
    print(f"涨停数量范围: {market_data['limitup_count'].min()} - {market_data['limitup_count'].max()}")
    print(f"指数收益率范围: {market_data['index_return'].min():.2%} - {market_data['index_return'].max():.2%}")
    
    # 创建择时门控器
    gate = MarketTimingGate(
        enable_timing=True,
        risk_threshold=0.5,
        sentiment_window=20
    )
    
    print("\n✓ 择时门控器已创建")
    
    # 计算市场情绪
    print("\n" + "=" * 60)
    print("计算市场情绪...")
    print("=" * 60)
    
    sentiment = gate.compute_market_sentiment(market_data)
    print(f"\n涨停平均数: {sentiment['limitup_avg']:.1f}")
    print(f"跌停平均数: {sentiment['limitdown_avg']:.1f}")
    print(f"涨跌停比率: {sentiment['limit_ratio']:.2f}")
    print(f"指数动量: {sentiment['index_momentum']:.2%}")
    print(f"成交量活跃度: {sentiment['volume_activity']:.2f}")
    print(f"涨跌家数比: {sentiment['advance_decline_ratio']:.2f}")
    print(f"综合情绪得分: {sentiment['overall_score']:.2f}")
    
    # 评估风险
    print("\n" + "=" * 60)
    print("评估市场风险...")
    print("=" * 60)
    
    risk = gate.assess_market_risk(sentiment, volatility=0.02)
    print(f"\n风险得分: {risk['risk_score']:.2f}")
    print(f"风险等级: {risk['risk_level']} ({risk['risk_desc']})")
    
    # 生成择时信号
    print("\n" + "=" * 60)
    print("生成择时信号...")
    print("=" * 60)
    
    timing_signal = gate.generate_timing_signal(market_data, volatility=0.02)
    print(f"\n交易信号: {timing_signal['signal']}")
    print(f"门控状态: {timing_signal['gate_status']}")
    print(f"原因: {timing_signal['reason']}")
    
    # 判断是否应该交易
    print("\n" + "=" * 60)
    print("判断交易决策...")
    print("=" * 60)
    
    should_trade, reason = gate.should_trade(market_data)
    print(f"\n是否交易: {'是' if should_trade else '否'}")
    print(f"原因: {reason}")
    
    # 获取仓位调整因子
    position_factor = gate.get_position_size_factor(market_data)
    print(f"建议仓位比例: {position_factor:.1%}")
    
    # 模拟多天择时
    print("\n" + "=" * 60)
    print("模拟历史择时信号...")
    print("=" * 60)
    
    signals_history = []
    for i in range(40, len(market_data)):
        window_data = market_data.iloc[:i+1]
        signal_result = gate.generate_timing_signal(window_data)
        signals_history.append({
            'date': window_data['date'].iloc[-1],
            'signal': signal_result['signal'],
            'gate': signal_result['gate_status'],
            'sentiment': signal_result['sentiment']['overall_score'],
            'risk': signal_result['risk']['risk_score']
        })
    
    signals_df = pd.DataFrame(signals_history)
    print(f"\n生成 {len(signals_df)} 天的择时信号")
    print("\n最近10天信号:")
    print(signals_df.tail(10).to_string(index=False))
    
    # 统计信号分布
    print("\n" + "=" * 60)
    print("信号统计:")
    print("=" * 60)
    signal_counts = signals_df['signal'].value_counts()
    gate_counts = signals_df['gate'].value_counts()
    
    print("\n交易信号分布:")
    for signal, count in signal_counts.items():
        print(f"  {signal}: {count} 天 ({count/len(signals_df)*100:.1f}%)")
    
    print("\n门控状态分布:")
    for gate_status, count in gate_counts.items():
        print(f"  {gate_status}: {count} 天 ({count/len(signals_df)*100:.1f}%)")
    
    # 状态报告
    print("\n" + "=" * 60)
    print("当前状态报告:")
    print("=" * 60)
    status = gate.get_status_report()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
