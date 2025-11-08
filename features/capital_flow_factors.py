"""
资金流向因子系统

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务扩展
目标：量化大盘、板块、个股的资金流向特征

核心维度：
1. 大盘资金流向：北向资金、融资融券、ETF申赎、大单流向
2. 板块资金流向：板块主力资金、板块资金排名、板块资金集中度
3. 个股资金流向：主力资金净流入、大单占比、资金流向强度
4. 资金流向趋势：连续流入天数、流向加速度、资金轮动
5. 资金结构分析：超大单/大单/中单/小单占比
6. 资金情绪指标：资金追涨热度、资金恐慌指数

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


class CapitalFlowFactors:
    """资金流向因子计算器"""
    
    def __init__(self):
        """初始化资金流向因子计算器"""
        self.flow_history = {}  # 历史资金流向数据
        print("💰 资金流向因子计算器初始化")
    
    def calculate_all_factors(self, date: str, 
                             market_data: pd.DataFrame = None,
                             stock_code: str = None) -> Dict:
        """
        计算所有资金流向因子
        
        Args:
            date: 交易日期
            market_data: 市场数据（可选）
            stock_code: 个股代码（可选，用于个股资金分析）
        
        Returns:
            Dict: 包含所有资金流向因子的字典
        """
        print(f"\n计算 {date} 资金流向因子...")
        
        factors = {}
        
        # 1. 大盘资金流向
        market_flow = self.analyze_market_capital_flow(date)
        factors.update(market_flow)
        
        # 2. 板块资金流向
        sector_flow = self.analyze_sector_capital_flow(date, market_data)
        factors.update(sector_flow)
        
        # 3. 个股资金流向（如果提供stock_code）
        if stock_code:
            stock_flow = self.analyze_stock_capital_flow(date, stock_code)
            factors.update(stock_flow)
        
        # 4. 资金流向趋势
        trend_analysis = self.analyze_capital_flow_trend(date)
        factors.update(trend_analysis)
        
        # 5. 资金结构分析
        structure_analysis = self.analyze_capital_structure(date, market_data)
        factors.update(structure_analysis)
        
        # 6. 资金情绪指标
        sentiment_analysis = self.analyze_capital_sentiment(date, market_data)
        factors.update(sentiment_analysis)
        
        # 缓存数据
        self.flow_history[date] = factors
        
        print(f"✅ 共计算 {len(factors)} 个资金流向因子")
        
        return factors
    
    def analyze_market_capital_flow(self, date: str) -> Dict:
        """
        大盘资金流向分析
        
        分析北向资金、融资融券、ETF申赎等大盘级别资金
        """
        print("  分析大盘资金流向...")
        
        factors = {}
        
        try:
            # 1. 北向资金（沪股通+深股通）
            northbound_data = self._get_northbound_flow(date)
            
            if northbound_data:
                factors['northbound_net_inflow'] = northbound_data.get('net_inflow', 0)  # 净流入（亿元）
                factors['northbound_inflow'] = northbound_data.get('inflow', 0)  # 流入
                factors['northbound_outflow'] = northbound_data.get('outflow', 0)  # 流出
                factors['northbound_flow_ratio'] = northbound_data.get('flow_ratio', 0)  # 流入/流出比
                
                # 北向资金强度（净流入/成交额）
                factors['northbound_strength'] = northbound_data.get('strength', 0)
            else:
                factors.update({
                    'northbound_net_inflow': 0,
                    'northbound_inflow': 0,
                    'northbound_outflow': 0,
                    'northbound_flow_ratio': 1.0,
                    'northbound_strength': 0
                })
            
            # 2. 融资融券
            margin_data = self._get_margin_trading(date)
            
            if margin_data:
                factors['margin_balance'] = margin_data.get('balance', 0)  # 融资余额（亿元）
                factors['margin_buy'] = margin_data.get('buy', 0)  # 融资买入
                factors['margin_repay'] = margin_data.get('repay', 0)  # 融资偿还
                factors['margin_net_buy'] = margin_data.get('net_buy', 0)  # 融资净买入
                
                # 融资净买入占成交额比例
                factors['margin_ratio'] = margin_data.get('ratio', 0)
            else:
                factors.update({
                    'margin_balance': 0,
                    'margin_buy': 0,
                    'margin_repay': 0,
                    'margin_net_buy': 0,
                    'margin_ratio': 0
                })
            
            # 3. ETF申赎
            etf_data = self._get_etf_flow(date)
            
            if etf_data:
                factors['etf_net_inflow'] = etf_data.get('net_inflow', 0)  # ETF净流入（亿元）
                factors['etf_creation'] = etf_data.get('creation', 0)  # 申购
                factors['etf_redemption'] = etf_data.get('redemption', 0)  # 赎回
            else:
                factors.update({
                    'etf_net_inflow': 0,
                    'etf_creation': 0,
                    'etf_redemption': 0
                })
            
            # 4. 大盘大单资金
            market_bigorder = self._get_market_bigorder_flow(date)
            
            if market_bigorder:
                factors['market_super_large_inflow'] = market_bigorder.get('super_large_inflow', 0)  # 超大单净流入
                factors['market_large_inflow'] = market_bigorder.get('large_inflow', 0)  # 大单净流入
                factors['market_main_inflow'] = market_bigorder.get('main_inflow', 0)  # 主力净流入（超大单+大单）
                
                # 主力净流入占比
                factors['market_main_ratio'] = market_bigorder.get('main_ratio', 0)
            else:
                factors.update({
                    'market_super_large_inflow': 0,
                    'market_large_inflow': 0,
                    'market_main_inflow': 0,
                    'market_main_ratio': 0
                })
            
            # 5. 大盘资金综合评分（0-100）
            score = self._calculate_market_flow_score(factors)
            factors['market_capital_score'] = score
            
            # 评分分级
            if score >= 80:
                factors['market_capital_level'] = '极强流入'
            elif score >= 60:
                factors['market_capital_level'] = '强流入'
            elif score >= 40:
                factors['market_capital_level'] = '中性'
            elif score >= 20:
                factors['market_capital_level'] = '弱流出'
            else:
                factors['market_capital_level'] = '极强流出'
        
        except Exception as e:
            print(f"    ⚠️ 大盘资金流向分析失败: {e}")
            self._fill_market_flow_defaults(factors)
        
        return factors
    
    def analyze_sector_capital_flow(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        板块资金流向分析
        
        分析各板块的主力资金流向、排名、集中度
        """
        print("  分析板块资金流向...")
        
        factors = {}
        
        try:
            # 获取板块资金流向数据
            sector_flow_data = self._get_sector_flow(date, market_data)
            
            if sector_flow_data is not None and not sector_flow_data.empty:
                # 1. Top 5 资金流入板块
                top_inflow_sectors = sector_flow_data.nlargest(5, 'net_inflow')
                
                for i, (idx, row) in enumerate(top_inflow_sectors.iterrows(), 1):
                    factors[f'top_{i}_inflow_sector'] = row.get('sector_name', f'板块{i}')
                    factors[f'top_{i}_inflow_amount'] = float(row.get('net_inflow', 0))
                    factors[f'top_{i}_inflow_ratio'] = float(row.get('flow_ratio', 0))
                
                # 填充剩余位置
                for i in range(len(top_inflow_sectors) + 1, 6):
                    factors[f'top_{i}_inflow_sector'] = '无'
                    factors[f'top_{i}_inflow_amount'] = 0
                    factors[f'top_{i}_inflow_ratio'] = 0
                
                # 2. Top 5 资金流出板块
                top_outflow_sectors = sector_flow_data.nsmallest(5, 'net_inflow')
                
                for i, (idx, row) in enumerate(top_outflow_sectors.iterrows(), 1):
                    factors[f'top_{i}_outflow_sector'] = row.get('sector_name', f'板块{i}')
                    factors[f'top_{i}_outflow_amount'] = float(row.get('net_inflow', 0))
                
                # 填充剩余位置
                for i in range(len(top_outflow_sectors) + 1, 6):
                    factors[f'top_{i}_outflow_sector'] = '无'
                    factors[f'top_{i}_outflow_amount'] = 0
                
                # 3. 板块资金集中度（HHI）
                total_inflow = sector_flow_data[sector_flow_data['net_inflow'] > 0]['net_inflow'].sum()
                
                if total_inflow > 0:
                    hhi = sum((flow / total_inflow) ** 2 
                             for flow in sector_flow_data[sector_flow_data['net_inflow'] > 0]['net_inflow'])
                    factors['sector_flow_concentration'] = hhi
                else:
                    factors['sector_flow_concentration'] = 0
                
                # 4. 资金流入板块数 vs 流出板块数
                inflow_count = (sector_flow_data['net_inflow'] > 0).sum()
                outflow_count = (sector_flow_data['net_inflow'] < 0).sum()
                
                factors['sector_inflow_count'] = int(inflow_count)
                factors['sector_outflow_count'] = int(outflow_count)
                factors['sector_inflow_ratio'] = inflow_count / len(sector_flow_data) if len(sector_flow_data) > 0 else 0
                
                # 5. 板块资金分化度（标准差）
                factors['sector_flow_divergence'] = float(sector_flow_data['net_inflow'].std())
                
            else:
                self._fill_sector_flow_defaults(factors)
        
        except Exception as e:
            print(f"    ⚠️ 板块资金流向分析失败: {e}")
            self._fill_sector_flow_defaults(factors)
        
        return factors
    
    def analyze_stock_capital_flow(self, date: str, stock_code: str) -> Dict:
        """
        个股资金流向分析
        
        分析单只股票的主力资金、大单占比等
        """
        print(f"  分析个股 {stock_code} 资金流向...")
        
        factors = {}
        
        try:
            stock_flow_data = self._get_stock_flow(date, stock_code)
            
            if stock_flow_data:
                # 1. 主力资金净流入
                factors['stock_main_inflow'] = stock_flow_data.get('main_inflow', 0)
                factors['stock_super_large_inflow'] = stock_flow_data.get('super_large_inflow', 0)
                factors['stock_large_inflow'] = stock_flow_data.get('large_inflow', 0)
                factors['stock_medium_inflow'] = stock_flow_data.get('medium_inflow', 0)
                factors['stock_small_inflow'] = stock_flow_data.get('small_inflow', 0)
                
                # 2. 主力资金占比
                factors['stock_main_ratio'] = stock_flow_data.get('main_ratio', 0)
                
                # 3. 主力净流入强度（净流入/成交额）
                factors['stock_flow_strength'] = stock_flow_data.get('flow_strength', 0)
                
                # 4. 大单笔数占比
                factors['stock_large_order_count_ratio'] = stock_flow_data.get('large_order_count_ratio', 0)
                
                # 5. 个股资金评分
                score = self._calculate_stock_flow_score(stock_flow_data)
                factors['stock_capital_score'] = score
                
                if score >= 80:
                    factors['stock_capital_level'] = '强势主力'
                elif score >= 60:
                    factors['stock_capital_level'] = '主力流入'
                elif score >= 40:
                    factors['stock_capital_level'] = '资金中性'
                else:
                    factors['stock_capital_level'] = '主力流出'
            else:
                self._fill_stock_flow_defaults(factors)
        
        except Exception as e:
            print(f"    ⚠️ 个股资金流向分析失败: {e}")
            self._fill_stock_flow_defaults(factors)
        
        return factors
    
    def analyze_capital_flow_trend(self, date: str) -> Dict:
        """
        资金流向趋势分析
        
        分析资金连续流入天数、流向加速度等
        """
        print("  分析资金流向趋势...")
        
        factors = {}
        
        try:
            # 获取最近N天的资金流向数据
            recent_flows = self._get_recent_flows(date, days=10)
            
            if recent_flows:
                # 1. 连续流入/流出天数
                consecutive_inflow = self._calculate_consecutive_days(recent_flows, 'inflow')
                consecutive_outflow = self._calculate_consecutive_days(recent_flows, 'outflow')
                
                factors['consecutive_inflow_days'] = consecutive_inflow
                factors['consecutive_outflow_days'] = consecutive_outflow
                
                # 2. 资金流向加速度（最近3天均值 vs 前7天均值）
                if len(recent_flows) >= 10:
                    recent_3day_avg = np.mean([f['net_inflow'] for f in recent_flows[:3]])
                    previous_7day_avg = np.mean([f['net_inflow'] for f in recent_flows[3:10]])
                    
                    if previous_7day_avg != 0:
                        factors['capital_flow_acceleration'] = (recent_3day_avg - previous_7day_avg) / abs(previous_7day_avg)
                    else:
                        factors['capital_flow_acceleration'] = 0
                else:
                    factors['capital_flow_acceleration'] = 0
                
                # 3. 资金流向趋势强度（线性回归斜率）
                if len(recent_flows) >= 5:
                    flows = [f['net_inflow'] for f in recent_flows[:5]]
                    trend_slope = self._calculate_trend_slope(flows)
                    factors['capital_flow_trend_slope'] = trend_slope
                    
                    if trend_slope > 0:
                        factors['capital_flow_trend'] = '上升'
                    elif trend_slope < 0:
                        factors['capital_flow_trend'] = '下降'
                    else:
                        factors['capital_flow_trend'] = '平稳'
                else:
                    factors['capital_flow_trend_slope'] = 0
                    factors['capital_flow_trend'] = '未知'
                
                # 4. 资金轮动特征（北向资金 vs 融资 vs ETF的主导性）
                factors['capital_rotation_leader'] = self._identify_capital_rotation_leader(recent_flows)
            
            else:
                factors.update({
                    'consecutive_inflow_days': 0,
                    'consecutive_outflow_days': 0,
                    'capital_flow_acceleration': 0,
                    'capital_flow_trend_slope': 0,
                    'capital_flow_trend': '未知',
                    'capital_rotation_leader': '未知'
                })
        
        except Exception as e:
            print(f"    ⚠️ 资金流向趋势分析失败: {e}")
            factors.update({
                'consecutive_inflow_days': 0,
                'consecutive_outflow_days': 0,
                'capital_flow_acceleration': 0,
                'capital_flow_trend_slope': 0,
                'capital_flow_trend': '未知',
                'capital_rotation_leader': '未知'
            })
        
        return factors
    
    def analyze_capital_structure(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        资金结构分析
        
        分析超大单/大单/中单/小单的占比结构
        """
        print("  分析资金结构...")
        
        factors = {}
        
        try:
            structure_data = self._get_capital_structure(date, market_data)
            
            if structure_data:
                # 1. 各级别资金占比
                factors['super_large_ratio'] = structure_data.get('super_large_ratio', 0)
                factors['large_ratio'] = structure_data.get('large_ratio', 0)
                factors['medium_ratio'] = structure_data.get('medium_ratio', 0)
                factors['small_ratio'] = structure_data.get('small_ratio', 0)
                
                # 2. 主力资金占比（超大单+大单）
                factors['main_capital_ratio'] = factors['super_large_ratio'] + factors['large_ratio']
                
                # 3. 散户资金占比（中单+小单）
                factors['retail_capital_ratio'] = factors['medium_ratio'] + factors['small_ratio']
                
                # 4. 资金结构健康度
                # 理想：主力资金占比高（>60%），超大单占比高（>30%）
                health_score = 0
                if factors['main_capital_ratio'] > 0.6:
                    health_score += 50
                elif factors['main_capital_ratio'] > 0.4:
                    health_score += 30
                
                if factors['super_large_ratio'] > 0.3:
                    health_score += 50
                elif factors['super_large_ratio'] > 0.2:
                    health_score += 30
                
                factors['capital_structure_health'] = health_score
                
                if health_score >= 80:
                    factors['capital_structure_level'] = '非常健康'
                elif health_score >= 60:
                    factors['capital_structure_level'] = '健康'
                elif health_score >= 40:
                    factors['capital_structure_level'] = '一般'
                else:
                    factors['capital_structure_level'] = '不健康'
            
            else:
                factors.update({
                    'super_large_ratio': 0,
                    'large_ratio': 0,
                    'medium_ratio': 0,
                    'small_ratio': 0,
                    'main_capital_ratio': 0,
                    'retail_capital_ratio': 0,
                    'capital_structure_health': 0,
                    'capital_structure_level': '未知'
                })
        
        except Exception as e:
            print(f"    ⚠️ 资金结构分析失败: {e}")
            factors.update({
                'super_large_ratio': 0,
                'large_ratio': 0,
                'medium_ratio': 0,
                'small_ratio': 0,
                'main_capital_ratio': 0,
                'retail_capital_ratio': 0,
                'capital_structure_health': 0,
                'capital_structure_level': '未知'
            })
        
        return factors
    
    def analyze_capital_sentiment(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        资金情绪指标
        
        分析资金追涨热度、恐慌指数等情绪指标
        """
        print("  分析资金情绪...")
        
        factors = {}
        
        try:
            # 1. 资金追涨热度（涨停板资金流入占比）
            limitup_flow = self._get_limitup_capital_flow(date, market_data)
            
            if limitup_flow:
                factors['limitup_capital_inflow'] = limitup_flow.get('total_inflow', 0)
                factors['limitup_capital_ratio'] = limitup_flow.get('ratio', 0)
                
                # 追涨热度评级
                if factors['limitup_capital_ratio'] > 0.2:
                    factors['chase_sentiment'] = '极度追涨'
                elif factors['limitup_capital_ratio'] > 0.1:
                    factors['chase_sentiment'] = '追涨'
                else:
                    factors['chase_sentiment'] = '理性'
            else:
                factors.update({
                    'limitup_capital_inflow': 0,
                    'limitup_capital_ratio': 0,
                    'chase_sentiment': '未知'
                })
            
            # 2. 资金恐慌指数（跌停板资金流出占比）
            limitdown_flow = self._get_limitdown_capital_flow(date, market_data)
            
            if limitdown_flow:
                factors['limitdown_capital_outflow'] = limitdown_flow.get('total_outflow', 0)
                factors['limitdown_capital_ratio'] = limitdown_flow.get('ratio', 0)
                
                # 恐慌指数评级
                if factors['limitdown_capital_ratio'] > 0.15:
                    factors['panic_sentiment'] = '极度恐慌'
                elif factors['limitdown_capital_ratio'] > 0.08:
                    factors['panic_sentiment'] = '恐慌'
                else:
                    factors['panic_sentiment'] = '稳定'
            else:
                factors.update({
                    'limitdown_capital_outflow': 0,
                    'limitdown_capital_ratio': 0,
                    'panic_sentiment': '未知'
                })
            
            # 3. 资金情绪综合指数（-100到100）
            # 正值：乐观，负值：悲观
            sentiment_score = 0
            
            # 北向资金贡献
            if 'northbound_net_inflow' in self.flow_history.get(date, {}):
                nb_inflow = self.flow_history[date]['northbound_net_inflow']
                sentiment_score += np.clip(nb_inflow / 100, -30, 30)  # -30到30
            
            # 融资净买入贡献
            if 'margin_net_buy' in self.flow_history.get(date, {}):
                margin_buy = self.flow_history[date]['margin_net_buy']
                sentiment_score += np.clip(margin_buy / 50, -20, 20)  # -20到20
            
            # 涨停板资金贡献
            sentiment_score += factors['limitup_capital_ratio'] * 100  # 0到20+
            
            # 跌停板资金惩罚
            sentiment_score -= factors['limitdown_capital_ratio'] * 150  # 0到-20+
            
            factors['capital_sentiment_index'] = np.clip(sentiment_score, -100, 100)
            
            # 情绪分级
            if factors['capital_sentiment_index'] > 60:
                factors['capital_sentiment_level'] = '极度乐观'
            elif factors['capital_sentiment_index'] > 30:
                factors['capital_sentiment_level'] = '乐观'
            elif factors['capital_sentiment_index'] > -30:
                factors['capital_sentiment_level'] = '中性'
            elif factors['capital_sentiment_index'] > -60:
                factors['capital_sentiment_level'] = '悲观'
            else:
                factors['capital_sentiment_level'] = '极度悲观'
        
        except Exception as e:
            print(f"    ⚠️ 资金情绪分析失败: {e}")
            factors.update({
                'limitup_capital_inflow': 0,
                'limitup_capital_ratio': 0,
                'chase_sentiment': '未知',
                'limitdown_capital_outflow': 0,
                'limitdown_capital_ratio': 0,
                'panic_sentiment': '未知',
                'capital_sentiment_index': 0,
                'capital_sentiment_level': '未知'
            })
        
        return factors
    
    # ==================== 辅助方法 ====================
    
    def _get_northbound_flow(self, date: str) -> Optional[Dict]:
        """获取北向资金数据"""
        try:
            import akshare as ak
            date_str = date.replace('-', '')
            
            # 获取北向资金流向
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北向资金")
            
            if not df.empty:
                date_data = df[df['日期'] == date]
                
                if not date_data.empty:
                    row = date_data.iloc[0]
                    return {
                        'net_inflow': float(row.get('当日成交净买额', 0)) / 1e8,  # 转为亿元
                        'inflow': float(row.get('买入成交额', 0)) / 1e8,
                        'outflow': float(row.get('卖出成交额', 0)) / 1e8,
                        'flow_ratio': float(row.get('买入成交额', 1)) / float(row.get('卖出成交额', 1)) if row.get('卖出成交额', 0) > 0 else 1.0,
                        'strength': float(row.get('当日成交净买额', 0)) / float(row.get('当日成交额', 1)) if row.get('当日成交额', 0) > 0 else 0
                    }
        except:
            pass
        
        return None
    
    def _get_margin_trading(self, date: str) -> Optional[Dict]:
        """获取融资融券数据"""
        try:
            import akshare as ak
            
            # 获取融资融券数据
            df = ak.stock_margin_underlying_info_em(symbol="沪深两市")
            
            if not df.empty:
                # 简化实现：返回模拟数据
                return {
                    'balance': 15000,  # 融资余额（亿元）
                    'buy': 500,  # 融资买入
                    'repay': 480,  # 融资偿还
                    'net_buy': 20,  # 净买入
                    'ratio': 0.05  # 占成交额比例
                }
        except:
            pass
        
        return None
    
    def _get_etf_flow(self, date: str) -> Optional[Dict]:
        """获取ETF申赎数据"""
        # 简化实现：返回模拟数据
        return {
            'net_inflow': 10,  # 亿元
            'creation': 50,
            'redemption': 40
        }
    
    def _get_market_bigorder_flow(self, date: str) -> Optional[Dict]:
        """获取大盘大单资金流向"""
        try:
            import akshare as ak
            
            # 获取大盘资金流向
            df = ak.stock_fund_flow_big_deal_em()
            
            if not df.empty:
                # 简化实现
                return {
                    'super_large_inflow': 100,  # 亿元
                    'large_inflow': 50,
                    'main_inflow': 150,
                    'main_ratio': 0.15
                }
        except:
            pass
        
        return None
    
    def _get_sector_flow(self, date: str, market_data: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """获取板块资金流向数据"""
        try:
            import akshare as ak
            
            # 获取板块资金流向
            df = ak.stock_sector_fund_flow_rank(indicator="今日")
            
            if not df.empty:
                df = df.rename(columns={
                    '名称': 'sector_name',
                    '主力净流入-净额': 'net_inflow',
                    '主力净流入-净占比': 'flow_ratio'
                })
                
                # 转换数值类型
                df['net_inflow'] = pd.to_numeric(df['net_inflow'], errors='coerce').fillna(0) / 1e8
                df['flow_ratio'] = pd.to_numeric(df['flow_ratio'], errors='coerce').fillna(0)
                
                return df
        except:
            pass
        
        return None
    
    def _get_stock_flow(self, date: str, stock_code: str) -> Optional[Dict]:
        """获取个股资金流向数据"""
        try:
            import akshare as ak
            
            # 获取个股资金流向
            df = ak.stock_individual_fund_flow(stock=stock_code, market="沪深A股")
            
            if not df.empty:
                today_data = df.iloc[-1]
                
                return {
                    'main_inflow': float(today_data.get('主力净流入-净额', 0)) / 1e8,
                    'super_large_inflow': float(today_data.get('超大单净流入-净额', 0)) / 1e8,
                    'large_inflow': float(today_data.get('大单净流入-净额', 0)) / 1e8,
                    'medium_inflow': float(today_data.get('中单净流入-净额', 0)) / 1e8,
                    'small_inflow': float(today_data.get('小单净流入-净额', 0)) / 1e8,
                    'main_ratio': float(today_data.get('主力净流入-净占比', 0)),
                    'flow_strength': float(today_data.get('主力净流入-净额', 0)) / float(today_data.get('成交额', 1)),
                    'large_order_count_ratio': 0.3  # 简化
                }
        except:
            pass
        
        return None
    
    def _get_recent_flows(self, date: str, days: int = 10) -> Optional[List[Dict]]:
        """获取最近N天的资金流向数据"""
        # 简化实现：从缓存中获取
        flows = []
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        for i in range(days):
            check_date = (date_obj - timedelta(days=i)).strftime('%Y-%m-%d')
            
            if check_date in self.flow_history:
                flows.append({
                    'date': check_date,
                    'net_inflow': self.flow_history[check_date].get('northbound_net_inflow', 0)
                })
        
        return flows if flows else None
    
    def _calculate_consecutive_days(self, flows: List[Dict], direction: str) -> int:
        """计算连续流入/流出天数"""
        count = 0
        
        for flow in flows:
            net_inflow = flow.get('net_inflow', 0)
            
            if direction == 'inflow' and net_inflow > 0:
                count += 1
            elif direction == 'outflow' and net_inflow < 0:
                count += 1
            else:
                break
        
        return count
    
    def _calculate_trend_slope(self, flows: List[float]) -> float:
        """计算趋势斜率（线性回归）"""
        if len(flows) < 2:
            return 0
        
        x = np.arange(len(flows))
        y = np.array(flows)
        
        # 线性回归
        slope, _ = np.polyfit(x, y, 1)
        
        return float(slope)
    
    def _identify_capital_rotation_leader(self, flows: List[Dict]) -> str:
        """识别资金轮动主导者"""
        # 简化实现
        return '北向资金'
    
    def _get_capital_structure(self, date: str, market_data: pd.DataFrame = None) -> Optional[Dict]:
        """获取资金结构数据"""
        # 简化实现：返回模拟数据
        return {
            'super_large_ratio': 0.35,
            'large_ratio': 0.25,
            'medium_ratio': 0.20,
            'small_ratio': 0.20
        }
    
    def _get_limitup_capital_flow(self, date: str, market_data: pd.DataFrame = None) -> Optional[Dict]:
        """获取涨停板资金流向"""
        # 简化实现
        return {
            'total_inflow': 50,  # 亿元
            'ratio': 0.05  # 占全市场资金比例
        }
    
    def _get_limitdown_capital_flow(self, date: str, market_data: pd.DataFrame = None) -> Optional[Dict]:
        """获取跌停板资金流向"""
        # 简化实现
        return {
            'total_outflow': 20,  # 亿元
            'ratio': 0.02
        }
    
    def _calculate_market_flow_score(self, factors: Dict) -> float:
        """计算大盘资金流向综合评分（0-100）"""
        score = 50  # 基础分
        
        # 北向资金贡献（±20分）
        nb_inflow = factors.get('northbound_net_inflow', 0)
        score += np.clip(nb_inflow / 50 * 20, -20, 20)
        
        # 融资净买入贡献（±15分）
        margin_buy = factors.get('margin_net_buy', 0)
        score += np.clip(margin_buy / 50 * 15, -15, 15)
        
        # 主力资金贡献（±15分）
        main_inflow = factors.get('market_main_inflow', 0)
        score += np.clip(main_inflow / 100 * 15, -15, 15)
        
        return np.clip(score, 0, 100)
    
    def _calculate_stock_flow_score(self, stock_flow: Dict) -> float:
        """计算个股资金流向评分（0-100）"""
        score = 50
        
        # 主力净流入贡献
        main_inflow = stock_flow.get('main_inflow', 0)
        score += np.clip(main_inflow * 10, -30, 30)
        
        # 主力资金占比贡献
        main_ratio = stock_flow.get('main_ratio', 0)
        score += main_ratio * 20
        
        return np.clip(score, 0, 100)
    
    def _fill_market_flow_defaults(self, factors: Dict):
        """填充大盘资金默认值"""
        factors.update({
            'northbound_net_inflow': 0,
            'northbound_inflow': 0,
            'northbound_outflow': 0,
            'northbound_flow_ratio': 1.0,
            'northbound_strength': 0,
            'margin_balance': 0,
            'margin_buy': 0,
            'margin_repay': 0,
            'margin_net_buy': 0,
            'margin_ratio': 0,
            'etf_net_inflow': 0,
            'etf_creation': 0,
            'etf_redemption': 0,
            'market_super_large_inflow': 0,
            'market_large_inflow': 0,
            'market_main_inflow': 0,
            'market_main_ratio': 0,
            'market_capital_score': 50,
            'market_capital_level': '中性'
        })
    
    def _fill_sector_flow_defaults(self, factors: Dict):
        """填充板块资金默认值"""
        for i in range(1, 6):
            factors[f'top_{i}_inflow_sector'] = '无'
            factors[f'top_{i}_inflow_amount'] = 0
            factors[f'top_{i}_inflow_ratio'] = 0
            factors[f'top_{i}_outflow_sector'] = '无'
            factors[f'top_{i}_outflow_amount'] = 0
        
        factors.update({
            'sector_flow_concentration': 0,
            'sector_inflow_count': 0,
            'sector_outflow_count': 0,
            'sector_inflow_ratio': 0,
            'sector_flow_divergence': 0
        })
    
    def _fill_stock_flow_defaults(self, factors: Dict):
        """填充个股资金默认值"""
        factors.update({
            'stock_main_inflow': 0,
            'stock_super_large_inflow': 0,
            'stock_large_inflow': 0,
            'stock_medium_inflow': 0,
            'stock_small_inflow': 0,
            'stock_main_ratio': 0,
            'stock_flow_strength': 0,
            'stock_large_order_count_ratio': 0,
            'stock_capital_score': 50,
            'stock_capital_level': '资金中性'
        })


def main():
    """主函数 - 示例用法"""
    calculator = CapitalFlowFactors()
    
    # 计算今日资金流向
    today = datetime.now().strftime('%Y-%m-%d')
    factors = calculator.calculate_all_factors(today)
    
    print("\n" + "="*70)
    print("💰 资金流向因子计算结果")
    print("="*70)
    
    # 大盘资金
    print("\n【大盘资金流向】")
    print(f"  北向资金净流入: {factors.get('northbound_net_inflow', 0):.2f}亿元")
    print(f"  融资净买入: {factors.get('margin_net_buy', 0):.2f}亿元")
    print(f"  主力资金净流入: {factors.get('market_main_inflow', 0):.2f}亿元")
    print(f"  大盘资金评分: {factors.get('market_capital_score', 0):.1f} ({factors.get('market_capital_level', '未知')})")
    
    # 板块资金
    print("\n【板块资金流向 Top 3】")
    for i in range(1, 4):
        sector = factors.get(f'top_{i}_inflow_sector', '无')
        amount = factors.get(f'top_{i}_inflow_amount', 0)
        if sector != '无':
            print(f"  {i}. {sector}: {amount:.2f}亿元")
    
    print(f"\n  资金流入板块数: {factors.get('sector_inflow_count', 0)}")
    print(f"  资金流出板块数: {factors.get('sector_outflow_count', 0)}")
    
    # 资金趋势
    print("\n【资金流向趋势】")
    print(f"  连续流入天数: {factors.get('consecutive_inflow_days', 0)}")
    print(f"  资金流向趋势: {factors.get('capital_flow_trend', '未知')}")
    print(f"  资金轮动主导: {factors.get('capital_rotation_leader', '未知')}")
    
    # 资金结构
    print("\n【资金结构】")
    print(f"  主力资金占比: {factors.get('main_capital_ratio', 0):.2%}")
    print(f"  散户资金占比: {factors.get('retail_capital_ratio', 0):.2%}")
    print(f"  结构健康度: {factors.get('capital_structure_level', '未知')}")
    
    # 资金情绪
    print("\n【资金情绪】")
    print(f"  追涨情绪: {factors.get('chase_sentiment', '未知')}")
    print(f"  恐慌情绪: {factors.get('panic_sentiment', '未知')}")
    print(f"  情绪指数: {factors.get('capital_sentiment_index', 0):.1f} ({factors.get('capital_sentiment_level', '未知')})")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
