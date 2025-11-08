"""
流动性与波动率因子系统

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务1.4
目标:评估市场流动性和波动率状态,捕捉市场风险信号

核心维度:
1. 流动性指标: 成交额、换手率、买卖价差、市场深度
2. 波动率指标: 历史波动率、隐含波动率、波动率偏度
3. 流动性风险: 流动性枯竭、流动性冲击
4. 市场微观结构: 订单流、价格影响、信息不对称

作者: Qilin Quant Team
创建: 2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class LiquidityVolatilityFactors:
    """流动性与波动率因子计算器"""
    
    def __init__(self):
        """初始化流动性与波动率因子计算器"""
        self.cache = {}
        print("💧 流动性与波动率因子计算器初始化")
    
    def calculate_all_factors(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算所有流动性与波动率因子
        
        Args:
            date: 交易日期
            market_data: 市场数据
        
        Returns:
            Dict: 包含所有流动性与波动率因子的字典
        """
        print(f"\n计算 {date} 流动性与波动率因子...")
        
        factors = {}
        
        # 1. 流动性指标
        factors.update(self.calculate_liquidity_metrics(date, market_data))
        
        # 2. 波动率指标
        factors.update(self.calculate_volatility_metrics(date, market_data))
        
        # 3. 流动性风险指标
        factors.update(self.calculate_liquidity_risk(date, market_data))
        
        # 4. 市场微观结构指标
        factors.update(self.calculate_microstructure_metrics(date, market_data))
        
        # 5. 综合评分
        factors['liquidity_health_score'] = self._calculate_liquidity_health(factors)
        factors['volatility_regime'] = self._classify_volatility_regime(factors)
        
        print(f"✅ 共计算 {len(factors)} 个流动性与波动率因子")
        
        return factors
    
    def calculate_liquidity_metrics(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算流动性指标
        
        成交额、换手率、市场深度等
        """
        print("  计算流动性指标...")
        
        factors = {}
        
        try:
            if market_data is not None:
                # 1. 市场成交额指标
                if 'amount' in market_data.columns:
                    total_amount = market_data['amount'].sum() / 1e8  # 转为亿
                    factors['market_total_amount'] = float(total_amount)
                    
                    # 相对5日均值
                    if 'amount_ma5' in market_data.columns:
                        amount_ratio = market_data['amount'] / market_data['amount_ma5']
                        factors['amount_vs_ma5'] = float(amount_ratio.mean())
                    else:
                        factors['amount_vs_ma5'] = 1.0
                    
                    # 相对20日均值
                    if 'amount_ma20' in market_data.columns:
                        amount_ratio_20 = market_data['amount'] / market_data['amount_ma20']
                        factors['amount_vs_ma20'] = float(amount_ratio_20.mean())
                    else:
                        factors['amount_vs_ma20'] = 1.0
                else:
                    factors['market_total_amount'] = 0
                    factors['amount_vs_ma5'] = 1.0
                    factors['amount_vs_ma20'] = 1.0
                
                # 2. 换手率指标
                if 'turnover_rate' in market_data.columns:
                    factors['market_avg_turnover'] = float(market_data['turnover_rate'].mean())
                    factors['market_median_turnover'] = float(market_data['turnover_rate'].median())
                    
                    # 高换手股票占比 (>10%)
                    high_turnover_ratio = (market_data['turnover_rate'] > 10).sum() / len(market_data)
                    factors['high_turnover_ratio'] = float(high_turnover_ratio)
                    
                    # 极低换手股票占比 (<1%)
                    low_turnover_ratio = (market_data['turnover_rate'] < 1).sum() / len(market_data)
                    factors['low_turnover_ratio'] = float(low_turnover_ratio)
                    
                    # 换手率分布标准差
                    factors['turnover_std'] = float(market_data['turnover_rate'].std())
                else:
                    factors['market_avg_turnover'] = 0
                    factors['market_median_turnover'] = 0
                    factors['high_turnover_ratio'] = 0
                    factors['low_turnover_ratio'] = 0
                    factors['turnover_std'] = 0
                
                # 3. 流动性分层
                if 'amount' in market_data.columns:
                    # 大盘股成交额 (市值>500亿)
                    if 'market_cap' in market_data.columns:
                        large_cap_amount = market_data[market_data['market_cap'] > 500e8]['amount'].sum() / 1e8
                        mid_cap_amount = market_data[(market_data['market_cap'] >= 100e8) & 
                                                    (market_data['market_cap'] <= 500e8)]['amount'].sum() / 1e8
                        small_cap_amount = market_data[market_data['market_cap'] < 100e8]['amount'].sum() / 1e8
                        
                        factors['large_cap_amount'] = float(large_cap_amount)
                        factors['mid_cap_amount'] = float(mid_cap_amount)
                        factors['small_cap_amount'] = float(small_cap_amount)
                        
                        # 大盘股成交占比
                        total = large_cap_amount + mid_cap_amount + small_cap_amount
                        factors['large_cap_amount_ratio'] = large_cap_amount / total if total > 0 else 0
                    else:
                        factors['large_cap_amount'] = 0
                        factors['mid_cap_amount'] = 0
                        factors['small_cap_amount'] = 0
                        factors['large_cap_amount_ratio'] = 0
                
                # 4. 流动性集中度
                if 'amount' in market_data.columns:
                    # Top 10股票成交额占比
                    top10_amount = market_data.nlargest(10, 'amount')['amount'].sum()
                    total_amount = market_data['amount'].sum()
                    factors['top10_amount_concentration'] = top10_amount / total_amount if total_amount > 0 else 0
                    
                    # Top 50股票成交额占比
                    top50_amount = market_data.nlargest(50, 'amount')['amount'].sum()
                    factors['top50_amount_concentration'] = top50_amount / total_amount if total_amount > 0 else 0
                else:
                    factors['top10_amount_concentration'] = 0
                    factors['top50_amount_concentration'] = 0
                
                # 5. 市场广度 (有成交的股票数)
                if 'volume' in market_data.columns:
                    active_stocks = (market_data['volume'] > 0).sum()
                    factors['active_stock_count'] = int(active_stocks)
                    factors['active_stock_ratio'] = active_stocks / len(market_data)
                else:
                    factors['active_stock_count'] = 0
                    factors['active_stock_ratio'] = 0
                
            else:
                self._fill_default_liquidity_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 流动性指标计算失败: {e}")
            self._fill_default_liquidity_factors(factors)
        
        return factors
    
    def calculate_volatility_metrics(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算波动率指标
        
        历史波动率、ATR、波动率分布等
        """
        print("  计算波动率指标...")
        
        factors = {}
        
        try:
            if market_data is not None and 'return' in market_data.columns:
                returns = market_data['return']
                
                # 1. 当日波动率 (收益率标准差)
                factors['daily_volatility'] = float(returns.std())
                
                # 2. 市场平均波动率 (个股波动率的均值)
                if 'volatility_20d' in market_data.columns:
                    factors['avg_stock_volatility_20d'] = float(market_data['volatility_20d'].mean())
                else:
                    factors['avg_stock_volatility_20d'] = 0
                
                # 3. 波动率分布
                if 'volatility_20d' in market_data.columns:
                    volatility_20d = market_data['volatility_20d']
                    
                    # 高波动股票占比 (波动率>30%)
                    high_vol_ratio = (volatility_20d > 30).sum() / len(market_data)
                    factors['high_volatility_ratio'] = float(high_vol_ratio)
                    
                    # 低波动股票占比 (波动率<10%)
                    low_vol_ratio = (volatility_20d < 10).sum() / len(market_data)
                    factors['low_volatility_ratio'] = float(low_vol_ratio)
                    
                    # 波动率标准差 (衡量波动率的离散度)
                    factors['volatility_dispersion'] = float(volatility_20d.std())
                else:
                    factors['high_volatility_ratio'] = 0
                    factors['low_volatility_ratio'] = 0
                    factors['volatility_dispersion'] = 0
                
                # 4. 涨跌幅分布
                # 正收益股票占比
                positive_return_ratio = (returns > 0).sum() / len(returns)
                factors['positive_return_ratio'] = float(positive_return_ratio)
                
                # 大涨大跌股票数 (|return|>5%)
                large_move_count = (np.abs(returns) > 5).sum()
                factors['large_move_count'] = int(large_move_count)
                factors['large_move_ratio'] = large_move_count / len(returns)
                
                # 收益率偏度 (衡量分布的不对称性)
                factors['return_skewness'] = float(returns.skew())
                
                # 收益率峰度 (衡量分布的尾部厚度)
                factors['return_kurtosis'] = float(returns.kurtosis())
                
                # 5. ATR (平均真实波幅)
                if all(col in market_data.columns for col in ['high', 'low', 'close', 'pre_close']):
                    # TR = max(high-low, abs(high-pre_close), abs(low-pre_close))
                    tr = np.maximum(
                        market_data['high'] - market_data['low'],
                        np.maximum(
                            np.abs(market_data['high'] - market_data['pre_close']),
                            np.abs(market_data['low'] - market_data['pre_close'])
                        )
                    )
                    factors['market_avg_atr'] = float(tr.mean())
                    
                    # ATR相对价格的比例
                    atr_pct = tr / market_data['close'] * 100
                    factors['market_avg_atr_pct'] = float(atr_pct.mean())
                else:
                    factors['market_avg_atr'] = 0
                    factors['market_avg_atr_pct'] = 0
                
                # 6. 波动率趋势 (当前波动率 vs 历史均值)
                if 'volatility_60d' in market_data.columns and 'volatility_20d' in market_data.columns:
                    vol_ratio = market_data['volatility_20d'] / market_data['volatility_60d']
                    factors['volatility_trend'] = float(vol_ratio.mean())
                    
                    # 波动率上升股票占比
                    vol_rising_ratio = (vol_ratio > 1.2).sum() / len(market_data)
                    factors['volatility_rising_ratio'] = float(vol_rising_ratio)
                else:
                    factors['volatility_trend'] = 1.0
                    factors['volatility_rising_ratio'] = 0
                
            else:
                self._fill_default_volatility_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 波动率指标计算失败: {e}")
            self._fill_default_volatility_factors(factors)
        
        return factors
    
    def calculate_liquidity_risk(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算流动性风险指标
        
        流动性枯竭、流动性冲击等异常信号
        """
        print("  计算流动性风险指标...")
        
        factors = {}
        
        try:
            if market_data is not None:
                # 1. 流动性枯竭信号
                # 缩量 + 低换手
                if 'amount' in market_data.columns and 'turnover_rate' in market_data.columns:
                    # 成交额低于20日均值的80%
                    if 'amount_ma20' in market_data.columns:
                        low_amount = (market_data['amount'] < market_data['amount_ma20'] * 0.8).sum()
                        factors['low_amount_stock_count'] = int(low_amount)
                        factors['low_amount_stock_ratio'] = low_amount / len(market_data)
                    else:
                        factors['low_amount_stock_count'] = 0
                        factors['low_amount_stock_ratio'] = 0
                    
                    # 换手率<0.5%的股票
                    ultra_low_turnover = (market_data['turnover_rate'] < 0.5).sum()
                    factors['ultra_low_turnover_count'] = int(ultra_low_turnover)
                    factors['ultra_low_turnover_ratio'] = ultra_low_turnover / len(market_data)
                    
                    # 流动性枯竭综合指标
                    liquidity_drought_score = (factors['low_amount_stock_ratio'] + 
                                              factors['ultra_low_turnover_ratio']) / 2
                    factors['liquidity_drought_score'] = float(liquidity_drought_score)
                    
                    # 风险等级
                    if liquidity_drought_score > 0.3:
                        factors['liquidity_risk_level'] = '高风险'
                    elif liquidity_drought_score > 0.15:
                        factors['liquidity_risk_level'] = '中等风险'
                    else:
                        factors['liquidity_risk_level'] = '低风险'
                else:
                    factors['low_amount_stock_count'] = 0
                    factors['low_amount_stock_ratio'] = 0
                    factors['ultra_low_turnover_count'] = 0
                    factors['ultra_low_turnover_ratio'] = 0
                    factors['liquidity_drought_score'] = 0
                    factors['liquidity_risk_level'] = '未知'
                
                # 2. 流动性冲击 (异常放量)
                if 'volume' in market_data.columns and 'volume_ma5' in market_data.columns:
                    # 成交量>5日均量的3倍
                    volume_surge = (market_data['volume'] > market_data['volume_ma5'] * 3).sum()
                    factors['volume_surge_count'] = int(volume_surge)
                    factors['volume_surge_ratio'] = volume_surge / len(market_data)
                else:
                    factors['volume_surge_count'] = 0
                    factors['volume_surge_ratio'] = 0
                
                # 3. 价格冲击 (大幅波动)
                if 'return' in market_data.columns:
                    # 单日涨跌幅>7%
                    price_shock = (np.abs(market_data['return']) > 7).sum()
                    factors['price_shock_count'] = int(price_shock)
                    factors['price_shock_ratio'] = price_shock / len(market_data)
                else:
                    factors['price_shock_count'] = 0
                    factors['price_shock_ratio'] = 0
                
                # 4. Amihud非流动性指标
                # Amihud = |return| / amount (价格变化/成交额,越大越不流动)
                if 'return' in market_data.columns and 'amount' in market_data.columns:
                    amihud = np.abs(market_data['return']) / (market_data['amount'] / 1e8 + 1e-6)
                    factors['market_avg_amihud'] = float(amihud.mean())
                    
                    # 高Amihud股票占比 (流动性差)
                    high_amihud_ratio = (amihud > amihud.quantile(0.75)).sum() / len(market_data)
                    factors['high_amihud_ratio'] = float(high_amihud_ratio)
                else:
                    factors['market_avg_amihud'] = 0
                    factors['high_amihud_ratio'] = 0
                
                # 5. 流动性分层风险
                if 'market_cap' in market_data.columns and 'turnover_rate' in market_data.columns:
                    # 小盘股平均换手率
                    small_cap_turnover = market_data[market_data['market_cap'] < 100e8]['turnover_rate'].mean()
                    # 大盘股平均换手率
                    large_cap_turnover = market_data[market_data['market_cap'] > 500e8]['turnover_rate'].mean()
                    
                    factors['small_cap_avg_turnover'] = float(small_cap_turnover) if not np.isnan(small_cap_turnover) else 0
                    factors['large_cap_avg_turnover'] = float(large_cap_turnover) if not np.isnan(large_cap_turnover) else 0
                    
                    # 流动性分层度 (小盘/大盘换手率比值)
                    if large_cap_turnover > 0:
                        factors['liquidity_stratification'] = small_cap_turnover / large_cap_turnover
                    else:
                        factors['liquidity_stratification'] = 0
                else:
                    factors['small_cap_avg_turnover'] = 0
                    factors['large_cap_avg_turnover'] = 0
                    factors['liquidity_stratification'] = 0
            
            else:
                self._fill_default_liquidity_risk_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 流动性风险指标计算失败: {e}")
            self._fill_default_liquidity_risk_factors(factors)
        
        return factors
    
    def calculate_microstructure_metrics(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算市场微观结构指标
        
        买卖价差、订单不平衡、价格影响等
        """
        print("  计算微观结构指标...")
        
        factors = {}
        
        try:
            if market_data is not None:
                # 1. 买卖价差 (Bid-Ask Spread)
                # 简化计算: 用日内高低价差作为代理
                if 'high' in market_data.columns and 'low' in market_data.columns and 'close' in market_data.columns:
                    spread_pct = (market_data['high'] - market_data['low']) / market_data['close'] * 100
                    factors['avg_spread_pct'] = float(spread_pct.mean())
                    factors['median_spread_pct'] = float(spread_pct.median())
                    
                    # 宽价差股票占比 (>5%)
                    wide_spread_ratio = (spread_pct > 5).sum() / len(market_data)
                    factors['wide_spread_ratio'] = float(wide_spread_ratio)
                else:
                    factors['avg_spread_pct'] = 0
                    factors['median_spread_pct'] = 0
                    factors['wide_spread_ratio'] = 0
                
                # 2. 价格效率 (收盘价相对日内均价的偏离)
                if all(col in market_data.columns for col in ['high', 'low', 'close']):
                    vwap_proxy = (market_data['high'] + market_data['low']) / 2
                    price_efficiency = np.abs(market_data['close'] - vwap_proxy) / vwap_proxy * 100
                    factors['avg_price_efficiency'] = float(price_efficiency.mean())
                else:
                    factors['avg_price_efficiency'] = 0
                
                # 3. 订单不平衡 (简化版)
                # 用涨跌分布作为代理
                if 'return' in market_data.columns:
                    rise_count = (market_data['return'] > 0).sum()
                    fall_count = (market_data['return'] < 0).sum()
                    
                    if fall_count > 0:
                        order_imbalance = (rise_count - fall_count) / (rise_count + fall_count)
                    else:
                        order_imbalance = 1.0
                    
                    factors['order_imbalance'] = float(order_imbalance)
                else:
                    factors['order_imbalance'] = 0
                
                # 4. 市场深度指标
                # 用成交额和波动率的比值作为深度的代理
                if 'amount' in market_data.columns and 'return' in market_data.columns:
                    # 深度 = 成交额 / 价格变动
                    depth_proxy = market_data['amount'] / (np.abs(market_data['return']) + 0.01)
                    factors['avg_market_depth'] = float(depth_proxy.mean())
                else:
                    factors['avg_market_depth'] = 0
                
                # 5. 信息不对称指标
                # 用换手率和波动率的比值
                if 'turnover_rate' in market_data.columns and 'return' in market_data.columns:
                    volatility = np.abs(market_data['return'])
                    info_asymmetry = volatility / (market_data['turnover_rate'] + 0.1)
                    factors['avg_info_asymmetry'] = float(info_asymmetry.mean())
                else:
                    factors['avg_info_asymmetry'] = 0
                
                # 6. 价格影响 (Price Impact)
                # Kyle's Lambda: 价格变化 / 成交量
                if 'return' in market_data.columns and 'volume' in market_data.columns:
                    price_impact = np.abs(market_data['return']) / (market_data['volume'] / 1e6 + 1)
                    factors['avg_price_impact'] = float(price_impact.mean())
                else:
                    factors['avg_price_impact'] = 0
                
            else:
                self._fill_default_microstructure_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 微观结构指标计算失败: {e}")
            self._fill_default_microstructure_factors(factors)
        
        return factors
    
    def _calculate_liquidity_health(self, factors: Dict) -> float:
        """
        计算流动性健康评分 (0-100)
        
        整合多个维度评估市场流动性状态
        """
        score = 50.0  # 基准分
        
        try:
            # 1. 成交额得分 (25分)
            amount_score = 0
            amount_vs_ma20 = factors.get('amount_vs_ma20', 1.0)
            if amount_vs_ma20 > 1.5:
                amount_score += 20  # 放量
            elif amount_vs_ma20 > 1.2:
                amount_score += 15
            elif amount_vs_ma20 > 1.0:
                amount_score += 10
            elif amount_vs_ma20 > 0.8:
                amount_score += 5
            else:
                amount_score -= 10  # 缩量严重
            
            # 活跃股票比例
            active_ratio = factors.get('active_stock_ratio', 0)
            if active_ratio > 0.95:
                amount_score += 5
            
            score += amount_score
            
            # 2. 换手率得分 (20分)
            turnover_score = 0
            avg_turnover = factors.get('market_avg_turnover', 0)
            if avg_turnover > 3:
                turnover_score += 15  # 换手活跃
            elif avg_turnover > 2:
                turnover_score += 10
            elif avg_turnover > 1:
                turnover_score += 5
            elif avg_turnover < 0.5:
                turnover_score -= 10  # 换手极低
            
            # 低换手股票占比
            low_turnover_ratio = factors.get('low_turnover_ratio', 0)
            if low_turnover_ratio < 0.1:
                turnover_score += 5
            elif low_turnover_ratio > 0.3:
                turnover_score -= 10
            
            score += turnover_score
            
            # 3. 流动性风险得分 (20分)
            risk_score = 0
            liquidity_risk = factors.get('liquidity_risk_level', '低风险')
            if liquidity_risk == '低风险':
                risk_score += 15
            elif liquidity_risk == '中等风险':
                risk_score += 5
            elif liquidity_risk == '高风险':
                risk_score -= 15
            
            drought_score = factors.get('liquidity_drought_score', 0)
            if drought_score < 0.1:
                risk_score += 5
            elif drought_score > 0.3:
                risk_score -= 10
            
            score += risk_score
            
            # 4. 波动率得分 (15分)
            volatility_score = 0
            daily_vol = factors.get('daily_volatility', 2)
            if daily_vol > 5:
                volatility_score -= 10  # 波动过大
            elif daily_vol > 3:
                volatility_score += 5  # 适度波动
            elif daily_vol < 1:
                volatility_score -= 5  # 波动过小
            else:
                volatility_score += 10
            
            # 波动率趋势
            vol_trend = factors.get('volatility_trend', 1.0)
            if vol_trend > 1.5:
                volatility_score -= 5  # 波动率急升
            
            score += volatility_score
            
            # 5. 市场微观结构得分 (10分)
            micro_score = 0
            spread = factors.get('avg_spread_pct', 3)
            if spread < 2:
                micro_score += 5  # 价差小,流动性好
            elif spread > 5:
                micro_score -= 5  # 价差大
            
            # 订单不平衡
            imbalance = abs(factors.get('order_imbalance', 0))
            if imbalance < 0.3:
                micro_score += 5  # 买卖平衡
            elif imbalance > 0.7:
                micro_score -= 5  # 严重失衡
            
            score += micro_score
            
            # 6. 集中度得分 (10分)
            concentration_score = 0
            top10_concentration = factors.get('top10_amount_concentration', 0.3)
            if top10_concentration < 0.2:
                concentration_score += 10  # 分散良好
            elif top10_concentration < 0.3:
                concentration_score += 5
            elif top10_concentration > 0.5:
                concentration_score -= 10  # 过度集中
            
            score += concentration_score
            
        except Exception as e:
            print(f"    ⚠️ 流动性健康评分计算失败: {e}")
        
        # 限制在0-100范围
        score = max(0, min(100, score))
        
        return float(score)
    
    def _classify_volatility_regime(self, factors: Dict) -> str:
        """
        波动率状态分类
        
        根据波动率水平和趋势,将市场分为不同状态
        """
        daily_vol = factors.get('daily_volatility', 2)
        vol_trend = factors.get('volatility_trend', 1.0)
        high_vol_ratio = factors.get('high_volatility_ratio', 0)
        
        if daily_vol > 5 and vol_trend > 1.3:
            return '极度波动'
        elif daily_vol > 3 and high_vol_ratio > 0.3:
            return '高波动'
        elif daily_vol > 2:
            return '中等波动'
        elif daily_vol < 1 and vol_trend < 0.8:
            return '低波动'
        else:
            return '正常波动'
    
    # ==================== 辅助方法 ====================
    
    def _fill_default_liquidity_factors(self, factors: Dict):
        """填充流动性指标默认值"""
        factors.update({
            'market_total_amount': 0,
            'amount_vs_ma5': 1.0,
            'amount_vs_ma20': 1.0,
            'market_avg_turnover': 0,
            'market_median_turnover': 0,
            'high_turnover_ratio': 0,
            'low_turnover_ratio': 0,
            'turnover_std': 0,
            'large_cap_amount': 0,
            'mid_cap_amount': 0,
            'small_cap_amount': 0,
            'large_cap_amount_ratio': 0,
            'top10_amount_concentration': 0,
            'top50_amount_concentration': 0,
            'active_stock_count': 0,
            'active_stock_ratio': 0
        })
    
    def _fill_default_volatility_factors(self, factors: Dict):
        """填充波动率指标默认值"""
        factors.update({
            'daily_volatility': 0,
            'avg_stock_volatility_20d': 0,
            'high_volatility_ratio': 0,
            'low_volatility_ratio': 0,
            'volatility_dispersion': 0,
            'positive_return_ratio': 0.5,
            'large_move_count': 0,
            'large_move_ratio': 0,
            'return_skewness': 0,
            'return_kurtosis': 0,
            'market_avg_atr': 0,
            'market_avg_atr_pct': 0,
            'volatility_trend': 1.0,
            'volatility_rising_ratio': 0
        })
    
    def _fill_default_liquidity_risk_factors(self, factors: Dict):
        """填充流动性风险指标默认值"""
        factors.update({
            'low_amount_stock_count': 0,
            'low_amount_stock_ratio': 0,
            'ultra_low_turnover_count': 0,
            'ultra_low_turnover_ratio': 0,
            'liquidity_drought_score': 0,
            'liquidity_risk_level': '未知',
            'volume_surge_count': 0,
            'volume_surge_ratio': 0,
            'price_shock_count': 0,
            'price_shock_ratio': 0,
            'market_avg_amihud': 0,
            'high_amihud_ratio': 0,
            'small_cap_avg_turnover': 0,
            'large_cap_avg_turnover': 0,
            'liquidity_stratification': 0
        })
    
    def _fill_default_microstructure_factors(self, factors: Dict):
        """填充微观结构指标默认值"""
        factors.update({
            'avg_spread_pct': 0,
            'median_spread_pct': 0,
            'wide_spread_ratio': 0,
            'avg_price_efficiency': 0,
            'order_imbalance': 0,
            'avg_market_depth': 0,
            'avg_info_asymmetry': 0,
            'avg_price_impact': 0
        })


def main():
    """主函数 - 示例用法"""
    calculator = LiquidityVolatilityFactors()
    
    # 计算今日流动性与波动率
    today = datetime.now().strftime('%Y-%m-%d')
    factors = calculator.calculate_all_factors(today)
    
    print("\n" + "="*70)
    print("💧 流动性与波动率因子计算结果")
    print("="*70)
    
    print("\n【流动性指标】")
    print(f"  市场总成交额: {factors['market_total_amount']:.2f}亿")
    print(f"  成交额 vs MA20: {factors['amount_vs_ma20']:.2f}倍")
    print(f"  平均换手率: {factors['market_avg_turnover']:.2f}%")
    print(f"  高换手股占比: {factors['high_turnover_ratio']:.2%}")
    print(f"  Top10成交集中度: {factors['top10_amount_concentration']:.2%}")
    
    print("\n【波动率指标】")
    print(f"  当日波动率: {factors['daily_volatility']:.2f}%")
    print(f"  平均波动率(20日): {factors['avg_stock_volatility_20d']:.2f}%")
    print(f"  高波动股占比: {factors['high_volatility_ratio']:.2%}")
    print(f"  波动率趋势: {factors['volatility_trend']:.2f}")
    print(f"  波动率状态: {factors['volatility_regime']}")
    
    print("\n[流动性风险]")
    print(f"流动性枯竭评分: {factors['liquidity_drought_score']:.4f}")
    print(f"流动性风险等级: {factors['liquidity_risk_level']}")
    print(f"异常放量股数: {factors['volume_surge_count']}")
    print(f"  价格冲击股数: {factors['price_shock_count']}")
    print(f"  Amihud非流动性: {factors['market_avg_amihud']:.2f}")
    
    print("\n【微观结构】")
    print(f"  平均价差: {factors['avg_spread_pct']:.2f}%")
    print(f"  订单不平衡: {factors['order_imbalance']:.2f}")
    print(f"  市场深度: {factors['avg_market_depth']:.2f}")
    print(f"  价格影响: {factors['avg_price_impact']:.2f}")
    
    print("\n【综合评估】")
    print(f"  流动性健康评分: {factors['liquidity_health_score']:.1f}/100")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
