"""
竞价决策引擎 - 完整工作流编排
处理从T日筛选到T+2卖出的完整流程
适配A股T+1交易制度
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class AuctionDecisionEngine:
    """
    竞价决策引擎 - 核心模块
    
    完整流程：
    T日盘后 → 严格筛选候选股
    T+1竞价 → 实时监控决策
    T+1开盘 → 买入执行
    T+1盘中 → 持仓监控（只能观察，不能卖）
    T+2开盘 → 卖出执行
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化决策引擎
        
        Parameters:
        -----------
        config: Dict
            配置参数字典
        """
        self.config = config or self._default_config()
        self.positions = {}  # 持仓记录
        self.trade_history = []  # 交易历史
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            # 筛选阈值
            'min_seal_strength': 80,  # 最低封单强度
            'min_limitup_time': '10:30:00',  # 最晚涨停时间
            'max_open_count': 2,  # 最多开板次数
            'min_quality_score': 85,  # 最低质量评分
            
            # 竞价强度分级
            'auction_levels': {
                'super_strong': 85,  # 超强 >85分
                'strong': 70,        # 强势 70-85分
                'medium': 55,        # 中等 55-70分
                'weak': 0            # 弱势 <55分
            },
            
            # 买入策略
            'buy_strategies': {
                'super_strong': {'timing': 'auction_end', 'price_adj': 0.005, 'position': 0.10},
                'strong': {'timing': 'open_observe', 'price_adj': -0.005, 'position': 0.08},
                'medium': {'timing': 'wait_pullback', 'price_adj': -0.03, 'position': 0.05}
            },
            
            # 风控参数
            'max_position_per_stock': 0.10,  # 单票最大10%
            'max_total_position': 0.50,      # 总仓位最大50%
            'min_market_limitup_count': 30,  # 市场最少涨停数
            
            # T+2卖出规则
            'sell_rules': {
                't1_limitup_t2_high_open': 0.50,  # T+1涨停且T+2高开>5%，卖50%
                't1_big_gain_t2_continue': 0.60,   # T+1涨5-9%且T+2高开，卖60%
                't1_small_gain': 1.00,              # T+1涨0-3%，全卖
                't1_loss': 1.00                     # T+1亏损，全卖止损
            }
        }
    
    # ========== T日盘后筛选 ==========
    
    def screen_tomorrow_candidates_strict(self, 
                                         today_limitups: pd.DataFrame,
                                         features: pd.DataFrame) -> pd.DataFrame:
        """
        T日盘后严格筛选候选股
        
        因为T+1无法止损，所以筛选必须极其严格！
        
        Parameters:
        -----------
        today_limitups: T日涨停股票数据
        features: 特征数据
        
        Returns:
        --------
        筛选后的候选股（5-10只）
        """
        print(f"\n{'='*60}")
        print(f"T日盘后严格筛选 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"初始涨停池: {len(today_limitups)} 只")
        
        # 合并特征
        data = today_limitups.merge(features, on=['date', 'symbol'], how='left')
        
        # ========== 第一层：基础过滤（淘汰70%）==========
        print("\n第一层：基础过滤")
        
        # 封单强度过滤
        if 'seal_strength' in data.columns:
            before = len(data)
            data = data[data['seal_strength'] > self.config['min_seal_strength']]
            print(f"  封单强度>{self.config['min_seal_strength']}: {before} → {len(data)} (-{before-len(data)})")
        
        # 涨停时间过滤
        if 'limitup_time' in data.columns:
            before = len(data)
            data = data[data['limitup_time'] < self.config['min_limitup_time']]
            print(f"  涨停时间<{self.config['min_limitup_time']}: {before} → {len(data)} (-{before-len(data)})")
        
        # 开板次数过滤
        if 'open_count' in data.columns:
            before = len(data)
            data = data[data['open_count'] <= self.config['max_open_count']]
            print(f"  开板次数≤{self.config['max_open_count']}: {before} → {len(data)} (-{before-len(data)})")
        
        # 过滤ST、退市股
        before = len(data)
        data = data[~data['symbol'].str.contains('ST|退', case=False, na=False)]
        print(f"  排除ST/退市: {before} → {len(data)} (-{before-len(data)})")
        
        # ========== 第二层：质量评分（淘汰50%）==========
        print("\n第二层：质量评分")
        
        # 计算综合质量分（如果没有现成评分）
        if 'quality_score' not in data.columns:
            data['quality_score'] = self._calculate_quality_score(data)
        
        before = len(data)
        data = data[data['quality_score'] >= self.config['min_quality_score']]
        print(f"  质量评分≥{self.config['min_quality_score']}: {before} → {len(data)} (-{before-len(data)})")
        
        # ========== 第三层：市场环境（淘汰30%）==========
        print("\n第三层：市场环境")
        
        # 市场涨停数检查
        if 'market_limitup_count' in data.columns:
            market_limitup = data['market_limitup_count'].iloc[0] if len(data) > 0 else 0
            print(f"  市场涨停数: {market_limitup}")
            
            if market_limitup < self.config['min_market_limitup_count']:
                print(f"  ⚠️  市场涨停数不足{self.config['min_market_limitup_count']}，建议观望")
                return pd.DataFrame()  # 返回空，不交易
        
        # 板块分散度检查
        if 'sector' in data.columns and len(data) > 0:
            sector_dist = data['sector'].value_counts()
            max_sector_ratio = sector_dist.max() / len(data)
            print(f"  最大板块占比: {max_sector_ratio:.1%}")
            
            if max_sector_ratio > 0.5:  # 超过50%集中在一个板块
                print(f"  ⚠️  板块过于集中，适当分散")
        
        # ========== 最终排序选Top N ==========
        print("\n最终排序")
        
        # 按质量评分排序
        data = data.sort_values('quality_score', ascending=False)
        
        # 选取Top 10（或更少）
        top_n = min(10, len(data))
        final_candidates = data.head(top_n).copy()
        
        print(f"\n✅ 最终选出: {len(final_candidates)} 只")
        print(f"{'='*60}\n")
        
        if len(final_candidates) > 0:
            print("候选股列表:")
            for idx, row in final_candidates.iterrows():
                print(f"  {row['symbol']}: 质量评分={row['quality_score']:.1f}")
        
        return final_candidates
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """计算综合质量评分"""
        score = pd.Series(0.0, index=data.index)
        
        # 封单强度（40分）
        if 'seal_strength' in data.columns:
            score += (data['seal_strength'] / 100) * 40
        
        # 涨停时间（20分，越早越好）
        if 'limitup_time_score' in data.columns:
            score += (data['limitup_time_score'] / 100) * 20
        
        # 板块联动（20分）
        if 'sector_strength' in data.columns:
            score += (data['sector_strength'] / 100) * 20
        
        # 资金性质（20分）
        if 'fund_quality' in data.columns:
            score += (data['fund_quality'] / 100) * 20
        
        return score.fillna(50)  # 默认50分
    
    # ========== T+1竞价监控 ==========
    
    def auction_final_check(self,
                           candidates: pd.DataFrame,
                           auction_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        T+1竞价最后确认
        
        这是最后的"反悔窗口"，竞价表现不佳的要果断放弃
        
        Parameters:
        -----------
        candidates: T日筛选的候选股
        auction_metrics: T+1竞价实时指标
        
        Returns:
        --------
        确认买入的股票列表
        """
        print(f"\n{'='*60}")
        print(f"T+1竞价最后确认 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # 合并竞价指标
        data = candidates.merge(auction_metrics, on='symbol', how='left')
        
        # 计算竞价强度评分
        data['auction_score'] = self._calculate_auction_score(data)
        
        # 分级
        data['auction_level'] = pd.cut(
            data['auction_score'],
            bins=[0, 55, 70, 85, 100],
            labels=['weak', 'medium', 'strong', 'super_strong']
        )
        
        # 统计各级别数量
        level_counts = data['auction_level'].value_counts()
        print("\n竞价强度分布:")
        for level in ['super_strong', 'strong', 'medium', 'weak']:
            count = level_counts.get(level, 0)
            print(f"  {level}: {count} 只")
        
        # 只买"超强"和"强势"的（因为T+1不能卖）
        final_buy = data[data['auction_level'].isin(['super_strong', 'strong'])].copy()
        
        # 放弃的股票
        abandoned = data[~data['symbol'].isin(final_buy['symbol'])]
        
        print(f"\n✅ 确认买入: {len(final_buy)} 只")
        print(f"❌ 放弃买入: {len(abandoned)} 只（竞价表现不佳）")
        
        if len(final_buy) > 0:
            print("\n确认买入列表:")
            for idx, row in final_buy.iterrows():
                print(f"  {row['symbol']}: 竞价强度={row['auction_score']:.1f} ({row['auction_level']})")
        
        print(f"{'='*60}\n")
        
        return final_buy
    
    def _calculate_auction_score(self, data: pd.DataFrame) -> pd.Series:
        """计算竞价强度评分"""
        score = pd.Series(50.0, index=data.index)  # 基准50分
        
        # 竞价涨幅（40分）
        if 'auction_gap' in data.columns:
            # >5%: 40分, 3-5%: 30分, 1-3%: 20分, <1%: 10分
            score += data['auction_gap'].apply(lambda x: 
                40 if x > 0.05 else 30 if x > 0.03 else 20 if x > 0.01 else 10
            )
        
        # 买卖单比（30分）
        if 'buy_sell_ratio' in data.columns:
            # >2: 30分, 1.5-2: 20分, 1-1.5: 10分
            score += data['buy_sell_ratio'].apply(lambda x:
                30 if x > 2 else 20 if x > 1.5 else 10 if x > 1 else 0
            )
        
        # 大单占比（20分）
        if 'big_order_ratio' in data.columns:
            score += (data['big_order_ratio'] * 20)
        
        # 价格稳定性（10分）
        if 'price_stability' in data.columns:
            score += (data['price_stability'] * 10)
        
        return score.clip(0, 100)
    
    # ========== T+1买入执行 ==========
    
    def execute_buy_on_t1(self,
                         final_candidates: pd.DataFrame,
                         current_cash: float = 100000) -> List[Dict]:
        """
        T+1日9:30买入执行
        
        Parameters:
        -----------
        final_candidates: 最终确认的买入列表
        current_cash: 当前可用资金
        
        Returns:
        --------
        买入订单列表
        """
        print(f"\n{'='*60}")
        print(f"T+1买入执行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"可用资金: ¥{current_cash:,.2f}")
        
        orders = []
        
        for idx, stock in final_candidates.iterrows():
            level = stock['auction_level']
            strategy = self.config['buy_strategies'].get(level, {})
            
            if not strategy:
                continue
            
            # 计算买入价格
            auction_price = stock.get('auction_price', stock.get('close', 0))
            buy_price = auction_price * (1 + strategy['price_adj'])
            
            # 计算仓位
            position_ratio = strategy['position']
            position_value = current_cash * position_ratio
            volume = int(position_value / buy_price / 100) * 100  # 整百股
            
            if volume < 100:
                print(f"  ❌ {stock['symbol']}: 资金不足，跳过")
                continue
            
            order = {
                'symbol': stock['symbol'],
                'buy_price': buy_price,
                'volume': volume,
                'cost': buy_price * volume,
                'auction_level': level,
                'buy_time': datetime.now(),
                't_day_date': stock.get('date'),
                'expected_t1_return': stock.get('t1_close_return', 0),  # 预期收益
            }
            
            orders.append(order)
            
            # 记录持仓
            self.positions[stock['symbol']] = order
            
            print(f"  ✅ {stock['symbol']}: {volume}股 @ ¥{buy_price:.2f} = ¥{order['cost']:,.2f}")
        
        # 记录交易历史
        self.trade_history.extend(orders)
        
        total_cost = sum(o['cost'] for o in orders)
        print(f"\n买入汇总:")
        print(f"  买入股票数: {len(orders)}")
        print(f"  总成本: ¥{total_cost:,.2f}")
        print(f"  剩余资金: ¥{current_cash - total_cost:,.2f}")
        print(f"  仓位占比: {total_cost/current_cash:.1%}")
        print(f"{'='*60}\n")
        
        return orders
    
    # ========== T+1持仓监控（只能观察）==========
    
    def monitor_t1_position(self, current_prices: Dict[str, float]) -> Dict:
        """
        T+1日持仓监控（只能观察，不能卖出）
        
        Parameters:
        -----------
        current_prices: {symbol: current_price}
        
        Returns:
        --------
        持仓状态报告
        """
        report = {
            'positions': [],
            'total_cost': 0,
            'total_market_value': 0,
            'total_profit': 0,
            'profit_rate': 0
        }
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['buy_price'])
            
            market_value = current_price * position['volume']
            profit = market_value - position['cost']
            profit_rate = profit / position['cost']
            
            position_report = {
                'symbol': symbol,
                'buy_price': position['buy_price'],
                'current_price': current_price,
                'volume': position['volume'],
                'cost': position['cost'],
                'market_value': market_value,
                'profit': profit,
                'profit_rate': profit_rate,
                'status': self._get_position_status(profit_rate)
            }
            
            report['positions'].append(position_report)
            report['total_cost'] += position['cost']
            report['total_market_value'] += market_value
        
        report['total_profit'] = report['total_market_value'] - report['total_cost']
        report['profit_rate'] = report['total_profit'] / report['total_cost'] if report['total_cost'] > 0 else 0
        
        return report
    
    def _get_position_status(self, profit_rate: float) -> str:
        """根据盈亏判断状态"""
        if profit_rate >= 0.095:
            return "✅ 接近涨停（T+2高开卖）"
        elif profit_rate >= 0.05:
            return "✅ 大涨（T+2择机卖）"
        elif profit_rate >= 0.02:
            return "🟡 小涨（观察T+2开盘）"
        elif profit_rate >= -0.03:
            return "⚠️  盘整（T+2见机行事）"
        else:
            return "❌ 亏损（无法止损，等T+2）"
    
    # ========== T+2卖出执行 ==========
    
    def execute_sell_on_t2(self,
                          t1_close_prices: Dict[str, float],
                          t2_open_prices: Dict[str, float]) -> List[Dict]:
        """
        T+2日卖出策略
        
        Parameters:
        -----------
        t1_close_prices: T+1收盘价
        t2_open_prices: T+2开盘价
        
        Returns:
        --------
        卖出订单列表
        """
        print(f"\n{'='*60}")
        print(f"T+2卖出执行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        sell_orders = []
        
        for symbol, position in list(self.positions.items()):
            t1_close = t1_close_prices.get(symbol, position['buy_price'])
            t2_open = t2_open_prices.get(symbol, t1_close)
            
            # 计算T+1表现
            t1_return = (t1_close / position['buy_price'] - 1)
            t2_open_gap = (t2_open / t1_close - 1)
            
            # 决策卖出比例
            sell_ratio = self._decide_sell_ratio(t1_return, t2_open_gap)
            
            if sell_ratio > 0:
                sell_volume = int(position['volume'] * sell_ratio / 100) * 100
                sell_price = t2_open  # 简化：按开盘价卖出
                
                if sell_volume >= 100:
                    order = {
                        'symbol': symbol,
                        'sell_price': sell_price,
                        'volume': sell_volume,
                        'revenue': sell_price * sell_volume,
                        'profit': (sell_price - position['buy_price']) * sell_volume,
                        'profit_rate': sell_price / position['buy_price'] - 1,
                        'sell_time': datetime.now(),
                        'hold_days': 2,
                    }
                    
                    sell_orders.append(order)
                    
                    print(f"  {symbol}: 卖出{sell_volume}股 @ ¥{sell_price:.2f}, "
                          f"盈亏={order['profit']:+,.2f} ({order['profit_rate']:+.2%})")
                    
                    # 更新或清除持仓
                    if sell_ratio >= 1.0:
                        del self.positions[symbol]
                    else:
                        self.positions[symbol]['volume'] -= sell_volume
        
        total_profit = sum(o['profit'] for o in sell_orders)
        print(f"\n卖出汇总:")
        print(f"  卖出笔数: {len(sell_orders)}")
        print(f"  总盈亏: ¥{total_profit:+,.2f}")
        print(f"{'='*60}\n")
        
        return sell_orders
    
    def _decide_sell_ratio(self, t1_return: float, t2_open_gap: float) -> float:
        """决定卖出比例"""
        rules = self.config['sell_rules']
        
        if t1_return >= 0.095:  # T+1涨停了
            if t2_open_gap >= 0.05:  # T+2高开>5%
                return rules['t1_limitup_t2_high_open']
            else:
                return 1.0  # 高开不及预期，全卖
        
        elif t1_return >= 0.05:  # T+1涨5-9%
            if t2_open_gap >= 0.02:  # T+2继续高开
                return rules['t1_big_gain_t2_continue']
            else:
                return 1.0  # 全卖保利润
        
        elif t1_return >= 0:  # T+1微涨或平
            return rules['t1_small_gain']
        
        else:  # T+1亏损
            return rules['t1_loss']
