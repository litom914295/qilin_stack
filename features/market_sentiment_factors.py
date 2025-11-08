"""
宏观市场情绪因子系统

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务1.8
目标：构建多维度市场情绪评估体系

核心维度：
1. 涨跌停结构：涨停数、跌停数、连板梯队
2. 市场资金流向：北向、南向、大单、散户
3. 指数表现：主要指数走势、波动率
4. 成交量能：市场活跃度、换手率
5. 情绪指标：新高新低、涨跌家数比

作者：Qilin Quant Team
创建：2025-10-30
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


class MarketSentimentFactors:
    """市场情绪因子计算器"""
    
    def __init__(self):
        """初始化市场情绪因子计算器"""
        self.factor_cache = {}
        print("📊 市场情绪因子计算器初始化")
    
    def calculate_all_factors(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算所有市场情绪因子
        
        Args:
            date: 交易日期
            market_data: 市场数据（可选，如果没有则尝试获取）
        
        Returns:
            Dict: 包含所有情绪因子的字典
        """
        print(f"\n计算 {date} 市场情绪因子...")
        
        factors = {}
        
        # 1. 涨跌停结构因子
        factors.update(self.calculate_limitup_structure(date, market_data))
        
        # 2. 市场资金流向因子
        factors.update(self.calculate_capital_flow(date, market_data))
        
        # 3. 指数表现因子
        factors.update(self.calculate_index_performance(date))
        
        # 4. 市场活跃度因子
        factors.update(self.calculate_market_activity(date, market_data))
        
        # 5. 情绪指标因子
        factors.update(self.calculate_sentiment_indicators(date, market_data))
        
        # 6. 综合情绪评分
        factors['comprehensive_sentiment_score'] = self._calculate_comprehensive_score(factors)
        
        # 7. 市场状态分类
        factors['market_regime'] = self._classify_market_regime(factors)
        
        print(f"✅ 共计算 {len(factors)} 个情绪因子")
        
        return factors
    
    def calculate_limitup_structure(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算涨跌停结构因子
        
        涨跌停结构是A股市场情绪的最直接体现
        """
        print("  计算涨跌停结构因子...")
        
        factors = {}
        
        try:
            # 获取涨跌停数据
            limitup_data = self._get_limitup_data(date, market_data)
            
            if limitup_data is not None and not limitup_data.empty:
                # 1. 基础涨跌停数量
                factors['limit_up_count'] = int(limitup_data['is_limit_up'].sum())
                factors['limit_down_count'] = int(limitup_data['is_limit_down'].sum()) if 'is_limit_down' in limitup_data.columns else 0
                
                # 2. 涨停占比（相对全市场）
                total_stocks = len(limitup_data)
                factors['limit_up_ratio'] = factors['limit_up_count'] / total_stocks if total_stocks > 0 else 0
                
                # 3. 连板梯队结构
                if 'consecutive_days' in limitup_data.columns:
                    consecutive_counts = limitup_data[limitup_data['is_limit_up'] == 1]['consecutive_days'].value_counts()
                    factors['first_board_count'] = int(consecutive_counts.get(1, 0))  # 首板数
                    factors['second_board_count'] = int(consecutive_counts.get(2, 0))  # 二板数
                    factors['third_board_plus_count'] = int(consecutive_counts[consecutive_counts.index >= 3].sum())  # 三板及以上
                else:
                    factors['first_board_count'] = factors['limit_up_count']
                    factors['second_board_count'] = 0
                    factors['third_board_plus_count'] = 0
                
                # 4. 连板高度（最高连板数）
                if 'consecutive_days' in limitup_data.columns:
                    factors['max_consecutive_boards'] = int(limitup_data['consecutive_days'].max())
                else:
                    factors['max_consecutive_boards'] = 1
                
                # 5. 涨停质量评分（平均封单强度）
                if 'seal_strength' in limitup_data.columns:
                    limitup_stocks = limitup_data[limitup_data['is_limit_up'] == 1]
                    factors['avg_seal_strength'] = float(limitup_stocks['seal_strength'].mean()) if len(limitup_stocks) > 0 else 0
                else:
                    factors['avg_seal_strength'] = 0
                
                # 6. 涨停早晚（平均涨停时间，越早越强）
                if 'limitup_time' in limitup_data.columns:
                    limitup_stocks = limitup_data[limitup_data['is_limit_up'] == 1]
                    # 假设涨停时间格式为 "HH:MM" 或分钟数
                    factors['avg_limitup_time_minutes'] = 0  # 需要实际解析时间
                else:
                    factors['avg_limitup_time_minutes'] = 0
                
                # 7. 炸板率（开板次数>0的比例）
                if 'open_count' in limitup_data.columns:
                    limitup_stocks = limitup_data[limitup_data['is_limit_up'] == 1]
                    broken_count = (limitup_stocks['open_count'] > 0).sum()
                    factors['broken_board_ratio'] = broken_count / len(limitup_stocks) if len(limitup_stocks) > 0 else 0
                else:
                    factors['broken_board_ratio'] = 0
                
            else:
                # 无数据时填充默认值
                factors.update({
                    'limit_up_count': 0,
                    'limit_down_count': 0,
                    'limit_up_ratio': 0,
                    'first_board_count': 0,
                    'second_board_count': 0,
                    'third_board_plus_count': 0,
                    'max_consecutive_boards': 0,
                    'avg_seal_strength': 0,
                    'avg_limitup_time_minutes': 0,
                    'broken_board_ratio': 0
                })
        
        except Exception as e:
            print(f"    ⚠️ 涨跌停结构因子计算失败: {e}")
            factors.update({
                'limit_up_count': 0,
                'limit_down_count': 0,
                'limit_up_ratio': 0,
                'first_board_count': 0,
                'second_board_count': 0,
                'third_board_plus_count': 0,
                'max_consecutive_boards': 0,
                'avg_seal_strength': 0,
                'avg_limitup_time_minutes': 0,
                'broken_board_ratio': 0
            })
        
        return factors
    
    def calculate_capital_flow(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算市场资金流向因子
        
        资金是市场的血液，资金流向决定短期走势
        """
        print("  计算资金流向因子...")
        
        factors = {}
        
        try:
            # 1. 北向资金（陆股通）
            northbound = self._get_northbound_flow(date)
            factors['northbound_net_flow'] = northbound.get('net_flow', 0)  # 亿元
            factors['northbound_net_flow_3d'] = northbound.get('net_flow_3d', 0)  # 3日累计
            factors['northbound_net_flow_5d'] = northbound.get('net_flow_5d', 0)  # 5日累计
            
            # 2. 南向资金（港股通）
            southbound = self._get_southbound_flow(date)
            factors['southbound_net_flow'] = southbound.get('net_flow', 0)
            
            # 3. 主力资金流向（大单）
            if market_data is not None and 'big_order_net' in market_data.columns:
                factors['main_net_inflow'] = float(market_data['big_order_net'].sum())
                factors['main_net_inflow_ratio'] = float(market_data['big_order_net'].mean())
            else:
                factors['main_net_inflow'] = 0
                factors['main_net_inflow_ratio'] = 0
            
            # 4. 散户资金（小单）
            if market_data is not None and 'small_order_net' in market_data.columns:
                factors['retail_net_inflow'] = float(market_data['small_order_net'].sum())
            else:
                factors['retail_net_inflow'] = 0
            
            # 5. 资金流向一致性（主力与散户方向是否一致）
            if factors['main_net_inflow'] * factors['retail_net_inflow'] > 0:
                factors['capital_flow_consistency'] = 1  # 一致
            else:
                factors['capital_flow_consistency'] = 0  # 分歧
            
            # 6. 杠杆资金（融资融券）
            margin_data = self._get_margin_data(date)
            factors['margin_balance'] = margin_data.get('balance', 0)  # 融资余额（亿）
            factors['margin_balance_change'] = margin_data.get('balance_change', 0)  # 余额变化
            
        except Exception as e:
            print(f"    ⚠️ 资金流向因子计算失败: {e}")
            factors.update({
                'northbound_net_flow': 0,
                'northbound_net_flow_3d': 0,
                'northbound_net_flow_5d': 0,
                'southbound_net_flow': 0,
                'main_net_inflow': 0,
                'main_net_inflow_ratio': 0,
                'retail_net_inflow': 0,
                'capital_flow_consistency': 0,
                'margin_balance': 0,
                'margin_balance_change': 0
            })
        
        return factors
    
    def calculate_index_performance(self, date: str) -> Dict:
        """
        计算指数表现因子
        
        指数是市场的晴雨表
        """
        print("  计算指数表现因子...")
        
        factors = {}
        
        try:
            # 主要指数列表
            indices = {
                'sh000001': '上证指数',
                'sz399001': '深证成指',
                'sz399006': '创业板指',
                'sh000688': '科创50',
                'sh000300': '沪深300',
                'sh000905': '中证500',
                'sh000852': '中证1000'
            }
            
            for code, name in indices.items():
                index_data = self._get_index_data(code, date)
                
                if index_data:
                    factors[f'{name}_return'] = index_data.get('return', 0)
                    factors[f'{name}_volume_ratio'] = index_data.get('volume_ratio', 1.0)
                else:
                    factors[f'{name}_return'] = 0
                    factors[f'{name}_volume_ratio'] = 1.0
            
            # 综合指数强度
            returns = [v for k, v in factors.items() if k.endswith('_return')]
            factors['avg_index_return'] = np.mean(returns) if returns else 0
            
            # 指数分化度（标准差）
            factors['index_divergence'] = np.std(returns) if returns else 0
            
            # 指数波动率（20日）
            factors['index_volatility_20d'] = self._calculate_index_volatility(date)
            
        except Exception as e:
            print(f"    ⚠️ 指数表现因子计算失败: {e}")
            factors.update({
                '上证指数_return': 0,
                '深证成指_return': 0,
                '创业板指_return': 0,
                '科创50_return': 0,
                '沪深300_return': 0,
                '中证500_return': 0,
                '中证1000_return': 0,
                'avg_index_return': 0,
                'index_divergence': 0,
                'index_volatility_20d': 0
            })
        
        return factors
    
    def calculate_market_activity(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算市场活跃度因子
        
        成交量能是市场热度的直接体现
        """
        print("  计算市场活跃度因子...")
        
        factors = {}
        
        try:
            if market_data is not None:
                # 1. 市场总成交额
                if 'amount' in market_data.columns:
                    total_amount = market_data['amount'].sum() / 1e8  # 转换为亿元
                    factors['market_total_amount'] = float(total_amount)
                else:
                    factors['market_total_amount'] = 0
                
                # 2. 市场平均换手率
                if 'turnover_rate' in market_data.columns:
                    factors['market_avg_turnover'] = float(market_data['turnover_rate'].mean())
                else:
                    factors['market_avg_turnover'] = 0
                
                # 3. 高换手股票数（换手率>10%）
                if 'turnover_rate' in market_data.columns:
                    factors['high_turnover_count'] = int((market_data['turnover_rate'] > 10).sum())
                else:
                    factors['high_turnover_count'] = 0
                
                # 4. 市场量比（今日成交量/5日均量）
                if 'volume' in market_data.columns and 'volume_ma5' in market_data.columns:
                    volume_ratio = market_data['volume'] / market_data['volume_ma5']
                    factors['market_volume_ratio'] = float(volume_ratio.mean())
                else:
                    factors['market_volume_ratio'] = 1.0
                
                # 5. 放量股票占比（量比>1.5）
                if 'volume_ratio' in market_data.columns:
                    high_volume_count = (market_data['volume_ratio'] > 1.5).sum()
                    factors['high_volume_ratio'] = high_volume_count / len(market_data)
                else:
                    factors['high_volume_ratio'] = 0
                
            else:
                factors.update({
                    'market_total_amount': 0,
                    'market_avg_turnover': 0,
                    'high_turnover_count': 0,
                    'market_volume_ratio': 1.0,
                    'high_volume_ratio': 0
                })
            
            # 6. 相对历史成交额（今日/20日均）
            factors['amount_vs_ma20'] = self._get_amount_vs_ma(date, 20)
            
        except Exception as e:
            print(f"    ⚠️ 市场活跃度因子计算失败: {e}")
            factors.update({
                'market_total_amount': 0,
                'market_avg_turnover': 0,
                'high_turnover_count': 0,
                'market_volume_ratio': 1.0,
                'high_volume_ratio': 0,
                'amount_vs_ma20': 1.0
            })
        
        return factors
    
    def calculate_sentiment_indicators(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算情绪指标因子
        
        涨跌家数、新高新低等经典情绪指标
        """
        print("  计算情绪指标因子...")
        
        factors = {}
        
        try:
            if market_data is not None:
                # 1. 涨跌家数
                if 'return' in market_data.columns:
                    rise_count = (market_data['return'] > 0).sum()
                    fall_count = (market_data['return'] < 0).sum()
                    flat_count = (market_data['return'] == 0).sum()
                    
                    factors['rise_count'] = int(rise_count)
                    factors['fall_count'] = int(fall_count)
                    factors['rise_fall_ratio'] = rise_count / fall_count if fall_count > 0 else 10
                else:
                    factors['rise_count'] = 0
                    factors['fall_count'] = 0
                    factors['rise_fall_ratio'] = 1.0
                
                # 2. 涨幅分布
                if 'return' in market_data.columns:
                    returns = market_data['return']
                    factors['return_median'] = float(returns.median())
                    factors['return_mean'] = float(returns.mean())
                    factors['return_std'] = float(returns.std())
                    
                    # 大涨大跌股票数（涨跌幅>5%）
                    factors['big_rise_count'] = int((returns > 5).sum())
                    factors['big_fall_count'] = int((returns < -5).sum())
                else:
                    factors['return_median'] = 0
                    factors['return_mean'] = 0
                    factors['return_std'] = 0
                    factors['big_rise_count'] = 0
                    factors['big_fall_count'] = 0
                
                # 3. 新高新低（创60日新高/新低的股票数）
                if 'close' in market_data.columns and 'high_60d' in market_data.columns:
                    factors['new_high_60d_count'] = int((market_data['close'] >= market_data['high_60d']).sum())
                    factors['new_low_60d_count'] = int((market_data['close'] <= market_data['low_60d']).sum()) if 'low_60d' in market_data.columns else 0
                else:
                    factors['new_high_60d_count'] = 0
                    factors['new_low_60d_count'] = 0
                
                # 4. 强势股占比（涨幅>3%）
                if 'return' in market_data.columns:
                    strong_count = (market_data['return'] > 3).sum()
                    factors['strong_stock_ratio'] = strong_count / len(market_data)
                else:
                    factors['strong_stock_ratio'] = 0
                
                # 5. 均线多头排列股票数（MA5>MA10>MA20）
                if all(col in market_data.columns for col in ['ma5', 'ma10', 'ma20']):
                    bullish_count = ((market_data['ma5'] > market_data['ma10']) & 
                                   (market_data['ma10'] > market_data['ma20'])).sum()
                    factors['bullish_ma_count'] = int(bullish_count)
                    factors['bullish_ma_ratio'] = bullish_count / len(market_data)
                else:
                    factors['bullish_ma_count'] = 0
                    factors['bullish_ma_ratio'] = 0
                
            else:
                factors.update({
                    'rise_count': 0,
                    'fall_count': 0,
                    'rise_fall_ratio': 1.0,
                    'return_median': 0,
                    'return_mean': 0,
                    'return_std': 0,
                    'big_rise_count': 0,
                    'big_fall_count': 0,
                    'new_high_60d_count': 0,
                    'new_low_60d_count': 0,
                    'strong_stock_ratio': 0,
                    'bullish_ma_count': 0,
                    'bullish_ma_ratio': 0
                })
        
        except Exception as e:
            print(f"    ⚠️ 情绪指标因子计算失败: {e}")
            factors.update({
                'rise_count': 0,
                'fall_count': 0,
                'rise_fall_ratio': 1.0,
                'return_median': 0,
                'return_mean': 0,
                'return_std': 0,
                'big_rise_count': 0,
                'big_fall_count': 0,
                'new_high_60d_count': 0,
                'new_low_60d_count': 0,
                'strong_stock_ratio': 0,
                'bullish_ma_count': 0,
                'bullish_ma_ratio': 0
            })
        
        return factors
    
    def _calculate_comprehensive_score(self, factors: Dict) -> float:
        """
        计算综合情绪评分（0-100）
        
        整合所有维度的情绪因子，给出一个综合评分
        """
        score = 50.0  # 中性基准
        
        try:
            # 1. 涨停结构得分（30分）
            limitup_score = 0
            if factors.get('limit_up_count', 0) >= 100:
                limitup_score += 15  # 涨停数>100，极度活跃
            elif factors.get('limit_up_count', 0) >= 50:
                limitup_score += 10
            elif factors.get('limit_up_count', 0) >= 30:
                limitup_score += 5
            
            if factors.get('third_board_plus_count', 0) >= 5:
                limitup_score += 10  # 有5个以上高度板，情绪高
            elif factors.get('third_board_plus_count', 0) >= 3:
                limitup_score += 5
            
            if factors.get('avg_seal_strength', 0) > 5:
                limitup_score += 5  # 封单强度高
            
            score += limitup_score
            
            # 2. 资金流向得分（25分）
            capital_score = 0
            if factors.get('northbound_net_flow', 0) > 50:
                capital_score += 10  # 北向大幅流入
            elif factors.get('northbound_net_flow', 0) > 20:
                capital_score += 5
            elif factors.get('northbound_net_flow', 0) < -50:
                capital_score -= 10  # 北向大幅流出
            
            if factors.get('main_net_inflow', 0) > 0:
                capital_score += 8  # 主力资金净流入
            else:
                capital_score -= 8
            
            if factors.get('capital_flow_consistency', 0) == 1:
                capital_score += 7  # 主力与散户一致
            
            score += capital_score
            
            # 3. 指数表现得分（15分）
            index_score = 0
            avg_return = factors.get('avg_index_return', 0)
            if avg_return > 2:
                index_score += 10  # 指数大涨
            elif avg_return > 1:
                index_score += 5
            elif avg_return < -2:
                index_score -= 10  # 指数大跌
            elif avg_return < -1:
                index_score -= 5
            
            if factors.get('index_divergence', 0) < 0.5:
                index_score += 5  # 指数分化小，整体性强
            
            score += index_score
            
            # 4. 市场活跃度得分（15分）
            activity_score = 0
            if factors.get('market_volume_ratio', 1) > 1.5:
                activity_score += 8  # 放量
            elif factors.get('market_volume_ratio', 1) < 0.8:
                activity_score -= 8  # 缩量
            
            if factors.get('high_turnover_count', 0) > 500:
                activity_score += 7  # 高换手股票多
            elif factors.get('high_turnover_count', 0) > 300:
                activity_score += 4
            
            score += activity_score
            
            # 5. 涨跌家数得分（15分）
            sentiment_score = 0
            rise_fall_ratio = factors.get('rise_fall_ratio', 1)
            if rise_fall_ratio > 3:
                sentiment_score += 10  # 大面积上涨
            elif rise_fall_ratio > 1.5:
                sentiment_score += 5
            elif rise_fall_ratio < 0.5:
                sentiment_score -= 10  # 大面积下跌
            elif rise_fall_ratio < 0.7:
                sentiment_score -= 5
            
            if factors.get('strong_stock_ratio', 0) > 0.3:
                sentiment_score += 5  # 强势股多
            
            score += sentiment_score
            
        except Exception as e:
            print(f"    ⚠️ 综合评分计算失败: {e}")
        
        # 限制在0-100范围
        score = max(0, min(100, score))
        
        return float(score)
    
    def _classify_market_regime(self, factors: Dict) -> str:
        """
        市场状态分类
        
        根据综合评分和关键指标，将市场分为5种状态
        """
        score = factors.get('comprehensive_sentiment_score', 50)
        limit_up_count = factors.get('limit_up_count', 0)
        third_board_plus = factors.get('third_board_plus_count', 0)
        
        if score >= 80 and limit_up_count > 100 and third_board_plus > 10:
            return '牛市狂热'  # 极度活跃
        elif score >= 65 and limit_up_count > 50:
            return '温和上涨'  # 健康上涨
        elif score >= 35 and limit_up_count >= 30:
            return '震荡整理'  # 震荡市
        elif score >= 20 and limit_up_count < 30:
            return '调整恐慌'  # 调整期
        else:
            return '熊市极寒'  # 极度低迷
    
    # ==================== 辅助方法 ====================
    
    def _get_limitup_data(self, date: str, market_data: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """获取涨跌停数据"""
        if market_data is not None:
            return market_data
        
        # 尝试使用AKShare获取
        try:
            import akshare as ak
            date_str = date.replace('-', '')
            df = ak.stock_zt_pool_em(date=date_str)
            
            if not df.empty:
                # 添加is_limit_up列
                df['is_limit_up'] = 1
                return df
        except:
            pass
        
        return None
    
    def _get_northbound_flow(self, date: str) -> Dict:
        """获取北向资金流向"""
        try:
            import akshare as ak
            # 获取历史北向资金数据
            df = ak.stock_hsgt_hist_em()
            
            if not df.empty:
                date_data = df[df['日期'] == date]
                if not date_data.empty:
                    net_flow = date_data['当日资金流入'].iloc[0] / 1e8  # 转为亿
                    
                    # 计算3日和5日累计
                    recent_5d = df.head(5)
                    net_flow_3d = recent_5d.head(3)['当日资金流入'].sum() / 1e8
                    net_flow_5d = recent_5d['当日资金流入'].sum() / 1e8
                    
                    return {
                        'net_flow': net_flow,
                        'net_flow_3d': net_flow_3d,
                        'net_flow_5d': net_flow_5d
                    }
        except:
            pass
        
        return {'net_flow': 0, 'net_flow_3d': 0, 'net_flow_5d': 0}
    
    def _get_southbound_flow(self, date: str) -> Dict:
        """获取南向资金流向"""
        # 南向资金数据获取较困难，返回默认值
        return {'net_flow': 0}
    
    def _get_margin_data(self, date: str) -> Dict:
        """获取融资融券数据"""
        try:
            import akshare as ak
            df = ak.stock_margin_detail_em()
            
            if not df.empty:
                # 返回最新的融资余额
                latest = df.iloc[0]
                balance = latest.get('融资余额', 0) / 1e8
                return {
                    'balance': balance,
                    'balance_change': 0  # 变化量需要对比前一天
                }
        except:
            pass
        
        return {'balance': 0, 'balance_change': 0}
    
    def _get_index_data(self, index_code: str, date: str) -> Optional[Dict]:
        """获取指数数据"""
        try:
            import akshare as ak
            # 获取指数日线数据
            df = ak.stock_zh_index_daily(symbol=index_code)
            
            if not df.empty:
                date_data = df[df['date'] == date]
                if not date_data.empty:
                    row = date_data.iloc[0]
                    return_pct = (row['close'] - row['open']) / row['open'] * 100
                    
                    # 计算量比
                    recent_5d = df.head(5)
                    avg_volume_5d = recent_5d['volume'].mean()
                    volume_ratio = row['volume'] / avg_volume_5d if avg_volume_5d > 0 else 1.0
                    
                    return {
                        'return': return_pct,
                        'volume_ratio': volume_ratio
                    }
        except:
            pass
        
        return None
    
    def _calculate_index_volatility(self, date: str, window: int = 20) -> float:
        """计算指数波动率"""
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol='sh000001')  # 上证指数
            
            if not df.empty and len(df) >= window:
                recent = df.head(window)
                returns = recent['close'].pct_change()
                volatility = returns.std() * np.sqrt(252)  # 年化波动率
                return float(volatility * 100)  # 转为百分比
        except:
            pass
        
        return 0.0
    
    def _get_amount_vs_ma(self, date: str, window: int = 20) -> float:
        """计算成交额相对均值"""
        # 简化实现，返回默认值
        return 1.0


def main():
    """主函数 - 示例用法"""
    calculator = MarketSentimentFactors()
    
    # 计算今日市场情绪
    today = datetime.now().strftime('%Y-%m-%d')
    factors = calculator.calculate_all_factors(today)
    
    print("\n" + "="*70)
    print("📊 市场情绪因子计算结果")
    print("="*70)
    
    # 分类展示
    print("\n【涨跌停结构】")
    print(f"  涨停数: {factors['limit_up_count']}")
    print(f"  跌停数: {factors['limit_down_count']}")
    print(f"  首板/二板/三板+: {factors['first_board_count']}/{factors['second_board_count']}/{factors['third_board_plus_count']}")
    print(f"  最高连板: {factors['max_consecutive_boards']}")
    
    print("\n【资金流向】")
    print(f"  北向资金: {factors['northbound_net_flow']:.2f}亿")
    print(f"  主力净流入: {factors['main_net_inflow']:.2f}亿")
    print(f"  融资余额: {factors['margin_balance']:.2f}亿")
    
    print("\n【指数表现】")
    print(f"  平均涨幅: {factors['avg_index_return']:.2f}%")
    print(f"  指数分化度: {factors['index_divergence']:.4f}")
    
    print("\n【市场活跃度】")
    print(f"  总成交额: {factors['market_total_amount']:.2f}亿")
    print(f"  平均换手率: {factors['market_avg_turnover']:.2f}%")
    
    print("\n【情绪指标】")
    print(f"  涨跌家数比: {factors['rise_fall_ratio']:.2f}")
    print(f"  强势股占比: {factors['strong_stock_ratio']:.2%}")
    
    print("\n【综合评估】")
    print(f"  情绪评分: {factors['comprehensive_sentiment_score']:.1f}/100")
    print(f"  市场状态: {factors['market_regime']}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
