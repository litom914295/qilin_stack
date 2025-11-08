"""
T+2卖出策略模块
根据T+1表现和T+2开盘情况制定智能卖出策略
适配A股T+1交易制度
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class SellSignal:
    """卖出信号"""
    symbol: str
    name: str
    sell_ratio: float  # 卖出比例 0-1.0
    recommended_price: float
    sell_timing: str  # 'open_immediately', 'wait_high', 'stop_loss'
    t1_performance: str  # 'limit_up', 'big_gain', 'small_gain', 'loss'
    t1_return: float
    t2_open_gap: float
    expected_profit: float
    confidence: float
    reason: str


class T2SellStrategy:
    """
    T+2卖出策略
    
    核心理念：
    T+1表现决定T+2卖出策略
    - T+1涨停：T+2高开>5%卖50%，否则全卖
    - T+1涨5-9%：T+2高开卖60%，否则全卖保利润
    - T+1涨0-5%：T+2全卖（不贪恋）
    - T+1亏损：T+2开盘止损全卖
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化T+2卖出策略
        
        Parameters:
        -----------
        config: Dict
            策略配置参数
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            # T+1表现分类阈值
            't1_performance_levels': {
                'limit_up': 0.095,      # ≥9.5%视为涨停
                'big_gain': 0.05,       # 5-9%大涨
                'small_gain': 0.02,     # 2-5%小涨
                'tiny_gain': 0,         # 0-2%微涨
                'loss': -999            # <0亏损
            },
            
            # T+2开盘分类阈值
            't2_open_levels': {
                'high_open_strong': 0.05,   # ≥5%强势高开
                'high_open': 0.02,          # 2-5%高开
                'flat_open': -0.01,         # -1%~2%平开
                'low_open': -999            # <-1%低开
            },
            
            # 卖出策略矩阵
            'sell_strategies': {
                # T+1涨停的情况
                ('limit_up', 'high_open_strong'): {
                    'ratio': 0.50,
                    'timing': 'open_immediately',
                    'confidence': 0.90,
                    'reason': 'T+1涨停+T+2强势高开>5%，先兑现50%利润'
                },
                ('limit_up', 'high_open'): {
                    'ratio': 0.30,
                    'timing': 'wait_high',
                    'confidence': 0.80,
                    'reason': 'T+1涨停+T+2高开2-5%，减仓30%观察'
                },
                ('limit_up', 'flat_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.70,
                    'reason': 'T+1涨停但T+2平开，高开不及预期全卖'
                },
                ('limit_up', 'low_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.60,
                    'reason': 'T+1涨停但T+2低开，兑现利润全卖'
                },
                
                # T+1大涨5-9%的情况
                ('big_gain', 'high_open_strong'): {
                    'ratio': 0.60,
                    'timing': 'open_immediately',
                    'confidence': 0.85,
                    'reason': 'T+1大涨+T+2高开>5%，减仓60%锁定利润'
                },
                ('big_gain', 'high_open'): {
                    'ratio': 0.60,
                    'timing': 'wait_high',
                    'confidence': 0.75,
                    'reason': 'T+1大涨+T+2高开，逢高减仓60%'
                },
                ('big_gain', 'flat_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.65,
                    'reason': 'T+1大涨但T+2平开，全卖保住利润'
                },
                ('big_gain', 'low_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.55,
                    'reason': 'T+1大涨但T+2低开，全卖避免利润回吐'
                },
                
                # T+1小涨2-5%的情况
                ('small_gain', 'high_open_strong'): {
                    'ratio': 0.80,
                    'timing': 'open_immediately',
                    'confidence': 0.70,
                    'reason': 'T+1小涨+T+2强势高开，大部分兑现'
                },
                ('small_gain', 'high_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.65,
                    'reason': 'T+1小涨+T+2高开，全卖（涨幅有限）'
                },
                ('small_gain', 'flat_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.60,
                    'reason': 'T+1小涨+T+2平开，全卖出局'
                },
                ('small_gain', 'low_open'): {
                    'ratio': 1.00,
                    'timing': 'stop_loss',
                    'confidence': 0.50,
                    'reason': 'T+1小涨但T+2低开，止损出局'
                },
                
                # T+1微涨0-2%的情况
                ('tiny_gain', 'high_open_strong'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.65,
                    'reason': 'T+1微涨+T+2强势高开，全卖离场'
                },
                ('tiny_gain', 'high_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.60,
                    'reason': 'T+1微涨+T+2高开，全卖（走势不佳）'
                },
                ('tiny_gain', 'flat_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.55,
                    'reason': 'T+1微涨+T+2平开，果断出局'
                },
                ('tiny_gain', 'low_open'): {
                    'ratio': 1.00,
                    'timing': 'stop_loss',
                    'confidence': 0.45,
                    'reason': 'T+1微涨+T+2低开，止损离场'
                },
                
                # T+1亏损的情况（一律止损）
                ('loss', 'high_open_strong'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.60,
                    'reason': 'T+1亏损，T+2高开减少损失全卖'
                },
                ('loss', 'high_open'): {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.55,
                    'reason': 'T+1亏损，T+2高开止损'
                },
                ('loss', 'flat_open'): {
                    'ratio': 1.00,
                    'timing': 'stop_loss',
                    'confidence': 0.50,
                    'reason': 'T+1亏损，T+2平开止损'
                },
                ('loss', 'low_open'): {
                    'ratio': 1.00,
                    'timing': 'stop_loss',
                    'confidence': 0.45,
                    'reason': 'T+1亏损+T+2低开，果断止损'
                }
            },
            
            # 价格策略
            'price_strategy': {
                'open_immediately': 0,      # 开盘价
                'wait_high': 0.01,          # 等待冲高1%
                'stop_loss': -0.005         # 止损价（开盘价-0.5%）
            }
        }
    
    def generate_sell_signals(self,
                             positions: pd.DataFrame,
                             t1_close_prices: Dict[str, float],
                             t2_open_prices: Dict[str, float]) -> List[SellSignal]:
        """
        生成T+2卖出信号
        
        Parameters:
        -----------
        positions: DataFrame
            持仓数据，包含：symbol, name, buy_price, volume, cost
        t1_close_prices: Dict
            T+1收盘价 {symbol: price}
        t2_open_prices: Dict
            T+2开盘价 {symbol: price}
            
        Returns:
        --------
        List[SellSignal]: 卖出信号列表
        """
        if positions.empty:
            return []
        
        signals = []
        
        print(f"\n{'='*60}")
        print(f"T+2卖出信号生成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"持仓数量: {len(positions)} 只\n")
        
        for idx, pos in positions.iterrows():
            symbol = pos['symbol']
            buy_price = pos.get('buy_price', pos.get('cost', 0))
            
            if buy_price <= 0:
                print(f"❌ {symbol}: 买入价格无效，跳过")
                continue
            
            # 获取T+1收盘价和T+2开盘价
            t1_close = t1_close_prices.get(symbol, buy_price)
            t2_open = t2_open_prices.get(symbol, t1_close)
            
            # 计算T+1表现
            t1_return = (t1_close / buy_price - 1)
            t1_performance = self._classify_t1_performance(t1_return)
            
            # 计算T+2开盘涨幅
            t2_open_gap = (t2_open / t1_close - 1)
            t2_open_level = self._classify_t2_open(t2_open_gap)
            
            # 查找对应策略
            strategy_key = (t1_performance, t2_open_level)
            strategy = self.config['sell_strategies'].get(strategy_key)
            
            if not strategy:
                # 默认策略：全部卖出
                strategy = {
                    'ratio': 1.00,
                    'timing': 'open_immediately',
                    'confidence': 0.50,
                    'reason': '未匹配到具体策略，默认全卖'
                }
            
            # 计算卖出价格
            price_adj = self.config['price_strategy'].get(strategy['timing'], 0)
            recommended_price = t2_open * (1 + price_adj)
            
            # 计算预期收益
            expected_profit = (recommended_price / buy_price - 1) * strategy['ratio']
            
            # 生成卖出信号
            signal = SellSignal(
                symbol=symbol,
                name=pos.get('name', symbol),
                sell_ratio=strategy['ratio'],
                recommended_price=recommended_price,
                sell_timing=strategy['timing'],
                t1_performance=t1_performance,
                t1_return=t1_return,
                t2_open_gap=t2_open_gap,
                expected_profit=expected_profit,
                confidence=strategy['confidence'],
                reason=strategy['reason']
            )
            
            signals.append(signal)
            
            # 打印信号
            timing_desc = {
                'open_immediately': '开盘立即卖出',
                'wait_high': '等待冲高卖出',
                'stop_loss': '开盘止损卖出'
            }.get(strategy['timing'], strategy['timing'])
            
            print(f"  📤 {symbol} ({pos.get('name', '')})")
            print(f"     买入价: ¥{buy_price:.2f}")
            print(f"     T+1收盘: ¥{t1_close:.2f} ({t1_return:+.2%}) - {t1_performance}")
            print(f"     T+2开盘: ¥{t2_open:.2f} ({t2_open_gap:+.2%}) - {t2_open_level}")
            print(f"     卖出策略: {timing_desc}")
            print(f"     卖出比例: {strategy['ratio']:.0%}")
            print(f"     卖出价: ¥{recommended_price:.2f}")
            print(f"     预期收益: {expected_profit:+.2%}")
            print(f"     置信度: {strategy['confidence']:.0%}")
            print(f"     理由: {strategy['reason']}")
            print()
        
        print(f"生成卖出信号: {len(signals)} 个")
        print(f"{'='*60}\n")
        
        return signals
    
    def _classify_t1_performance(self, t1_return: float) -> str:
        """分类T+1表现"""
        levels = self.config['t1_performance_levels']
        
        if t1_return >= levels['limit_up']:
            return 'limit_up'
        elif t1_return >= levels['big_gain']:
            return 'big_gain'
        elif t1_return >= levels['small_gain']:
            return 'small_gain'
        elif t1_return >= levels['tiny_gain']:
            return 'tiny_gain'
        else:
            return 'loss'
    
    def _classify_t2_open(self, t2_open_gap: float) -> str:
        """分类T+2开盘"""
        levels = self.config['t2_open_levels']
        
        if t2_open_gap >= levels['high_open_strong']:
            return 'high_open_strong'
        elif t2_open_gap >= levels['high_open']:
            return 'high_open'
        elif t2_open_gap >= levels['flat_open']:
            return 'flat_open'
        else:
            return 'low_open'
    
    def execute_sell_orders(self,
                           signals: List[SellSignal],
                           positions: pd.DataFrame) -> List[Dict]:
        """
        执行卖出订单
        
        Parameters:
        -----------
        signals: List[SellSignal]
            卖出信号列表
        positions: DataFrame
            当前持仓
            
        Returns:
        --------
        List[Dict]: 卖出订单列表
        """
        orders = []
        
        print(f"\n{'='*60}")
        print(f"批量卖出执行")
        print(f"{'='*60}")
        
        for signal in signals:
            # 找到对应持仓
            pos = positions[positions['symbol'] == signal.symbol]
            if pos.empty:
                print(f"⚠️  {signal.symbol}: 未找到持仓，跳过")
                continue
            
            pos = pos.iloc[0]
            volume = pos['volume']
            buy_price = pos.get('buy_price', pos.get('cost', 0))
            
            # 计算卖出数量
            sell_volume = int(volume * signal.sell_ratio / 100) * 100  # 整百股
            
            if sell_volume < 100:
                print(f"⚠️  {signal.symbol}: 卖出数量不足100股，跳过")
                continue
            
            # 计算收益
            revenue = signal.recommended_price * sell_volume
            profit = (signal.recommended_price - buy_price) * sell_volume
            profit_rate = signal.recommended_price / buy_price - 1
            
            order = {
                'symbol': signal.symbol,
                'name': signal.name,
                'sell_price': signal.recommended_price,
                'volume': sell_volume,
                'revenue': revenue,
                'profit': profit,
                'profit_rate': profit_rate,
                'sell_ratio': signal.sell_ratio,
                'timing': signal.sell_timing,
                't1_performance': signal.t1_performance,
                't1_return': signal.t1_return,
                't2_open_gap': signal.t2_open_gap,
                'sell_time': datetime.now(),
                'reason': signal.reason
            }
            
            orders.append(order)
            
            print(f"✅ 卖出订单: {signal.symbol}")
            print(f"   价格: ¥{signal.recommended_price:.2f}")
            print(f"   数量: {sell_volume} 股")
            print(f"   金额: ¥{revenue:,.2f}")
            print(f"   盈亏: ¥{profit:+,.2f} ({profit_rate:+.2%})")
            print(f"   比例: {signal.sell_ratio:.0%}")
        
        total_profit = sum(o['profit'] for o in orders)
        total_revenue = sum(o['revenue'] for o in orders)
        
        print(f"\n执行汇总:")
        print(f"  卖出笔数: {len(orders)}")
        print(f"  总收入: ¥{total_revenue:,.2f}")
        print(f"  总盈亏: ¥{total_profit:+,.2f}")
        print(f"{'='*60}\n")
        
        return orders


# 使用示例
if __name__ == "__main__":
    # 模拟持仓
    positions = pd.DataFrame({
        'symbol': ['000001.SZ', '600519.SH', '300750.SZ', '688036.SH'],
        'name': ['平安银行', '贵州茅台', '宁德时代', '传音控股'],
        'buy_price': [11.5, 1850, 245, 88],
        'volume': [1000, 100, 400, 500],
        'cost': [11500, 185000, 98000, 44000]
    })
    
    # 模拟T+1收盘价（不同表现）
    t1_close_prices = {
        '000001.SZ': 12.7,   # +10.4% 涨停
        '600519.SH': 1965,   # +6.2% 大涨
        '300750.SZ': 250,    # +2.0% 小涨
        '688036.SH': 86      # -2.3% 亏损
    }
    
    # 模拟T+2开盘价（不同开盘情况）
    t2_open_prices = {
        '000001.SZ': 13.4,   # +5.5% 强势高开
        '600519.SH': 2005,   # +2.0% 高开
        '300750.SZ': 249,    # -0.4% 平开
        '688036.SH': 84      # -2.3% 低开
    }
    
    # 创建策略
    strategy = T2SellStrategy()
    
    # 生成卖出信号
    signals = strategy.generate_sell_signals(
        positions,
        t1_close_prices,
        t2_open_prices
    )
    
    # 执行卖出
    orders = strategy.execute_sell_orders(signals, positions)
    
    print(f"\n✅ 完成！共卖出 {len(orders)} 只股票")
    print(f"总盈亏: ¥{sum(o['profit'] for o in orders):+,.2f}")
