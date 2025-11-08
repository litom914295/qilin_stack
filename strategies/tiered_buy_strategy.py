"""
分层买入策略模块
根据竞价强度分级制定不同的买入策略
适配A股T+1交易制度
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class BuySignal:
    """买入信号"""
    symbol: str
    name: str
    auction_level: str  # 'super_strong', 'strong', 'medium', 'weak'
    auction_score: float
    recommended_timing: str  # 'auction_end', 'open_observe', 'wait_pullback', 'pass'
    recommended_price: float
    price_adjustment: float  # 相对竞价价的调整
    recommended_position: float  # 建议仓位比例
    confidence: float  # 置信度
    reason: str  # 买入理由
    risk_level: str  # 'low', 'medium', 'high'


class TieredBuyStrategy:
    """
    分层买入策略
    
    核心理念：
    - 超强股：竞价价+0.5%立即买（因为确定性高）
    - 强势股：竞价价-0.5%等回踩买（兼顾成本和确定性）
    - 中等股：竞价价-3%等大回踩买（成本优先，但要承担可能买不到）
    - 弱势股：放弃（因为T+1不能止损）
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化分层买入策略
        
        Parameters:
        -----------
        config: Dict
            策略配置参数
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            # 竞价强度分级阈值
            'auction_levels': {
                'super_strong': 85,  # ≥85分
                'strong': 70,        # 70-85分
                'medium': 55,        # 55-70分
                'weak': 0            # <55分
            },
            
            # 分层买入策略
            'strategies': {
                'super_strong': {
                    'timing': 'auction_end',      # 竞价结束立即买
                    'price_adj': 0.005,           # +0.5%
                    'position': 0.10,             # 单票10%仓位
                    'confidence': 0.85,           # 85%置信度
                    'risk': 'medium',             # 中等风险（虽然强势但仍有T+1风险）
                    'description': '超强股，竞价后立即抢筹'
                },
                'strong': {
                    'timing': 'open_observe',     # 开盘观察30秒-1分钟
                    'price_adj': -0.005,          # -0.5%
                    'position': 0.08,             # 单票8%仓位
                    'confidence': 0.75,           # 75%置信度
                    'risk': 'medium',
                    'description': '强势股，开盘后等小回踩买入'
                },
                'medium': {
                    'timing': 'wait_pullback',    # 等待回踩2-3%
                    'price_adj': -0.03,           # -3%
                    'position': 0.05,             # 单票5%仓位
                    'confidence': 0.60,           # 60%置信度
                    'risk': 'high',
                    'description': '中等股，等大回踩买入（可能买不到）'
                },
                'weak': {
                    'timing': 'pass',             # 放弃
                    'price_adj': 0,
                    'position': 0,
                    'confidence': 0.40,
                    'risk': 'very_high',
                    'description': '弱势股，放弃（T+1无法止损）'
                }
            },
            
            # 价格限制
            'price_limits': {
                'max_premium': 0.02,    # 最多比竞价价高2%
                'min_discount': -0.05,  # 最多比竞价价低5%
            },
            
            # 仓位控制
            'position_limits': {
                'max_single_position': 0.10,  # 单票最大10%
                'max_total_position': 0.50,   # 总仓位最大50%
            }
        }
    
    def generate_buy_signals(self, 
                            candidates: pd.DataFrame,
                            current_cash: float = 100000,
                            existing_positions: Optional[Dict] = None) -> List[BuySignal]:
        """
        生成买入信号
        
        Parameters:
        -----------
        candidates: DataFrame
            候选股票，必须包含：symbol, name, auction_score, auction_price
        current_cash: float
            当前可用资金
        existing_positions: Dict
            现有持仓 {symbol: position_value}
            
        Returns:
        --------
        List[BuySignal]: 买入信号列表
        """
        if candidates.empty:
            return []
        
        signals = []
        existing_positions = existing_positions or {}
        
        # 计算已占用仓位
        total_position_used = sum(existing_positions.values()) / current_cash if current_cash > 0 else 0
        available_position = self.config['position_limits']['max_total_position'] - total_position_used
        
        if available_position <= 0:
            print(f"⚠️  总仓位已达上限 {total_position_used:.1%}，无法新增持仓")
            return []
        
        print(f"\n{'='*60}")
        print(f"分层买入信号生成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"可用资金: ¥{current_cash:,.2f}")
        print(f"已用仓位: {total_position_used:.1%}")
        print(f"可用仓位: {available_position:.1%}")
        print(f"候选股票: {len(candidates)} 只\n")
        
        for idx, row in candidates.iterrows():
            # 确定竞价强度等级
            auction_score = row.get('auction_score', 50)
            auction_level = self._classify_auction_level(auction_score)
            
            # 获取对应策略
            strategy = self.config['strategies'].get(auction_level, {})
            
            if not strategy or strategy['timing'] == 'pass':
                print(f"  ❌ {row['symbol']}: 竞价强度{auction_score:.1f}分（{auction_level}）- 放弃")
                continue
            
            # 跳过已持仓股票
            if row['symbol'] in existing_positions:
                print(f"  ⏭️  {row['symbol']}: 已持仓，跳过")
                continue
            
            # 计算买入价格
            auction_price = row.get('auction_price', row.get('close', 0))
            if auction_price <= 0:
                print(f"  ❌ {row['symbol']}: 竞价价格无效")
                continue
            
            price_adj = strategy['price_adj']
            recommended_price = auction_price * (1 + price_adj)
            
            # 价格限制检查
            price_limits = self.config['price_limits']
            if price_adj > price_limits['max_premium']:
                print(f"  ⚠️  {row['symbol']}: 价格溢价{price_adj:.1%}超过上限")
                price_adj = price_limits['max_premium']
                recommended_price = auction_price * (1 + price_adj)
            
            # 计算建议仓位
            base_position = strategy['position']
            recommended_position = min(
                base_position,
                self.config['position_limits']['max_single_position'],
                available_position
            )
            
            if recommended_position < 0.02:  # 少于2%不值得买
                print(f"  ⚠️  {row['symbol']}: 可用仓位不足2%，跳过")
                continue
            
            # 生成买入信号
            signal = BuySignal(
                symbol=row['symbol'],
                name=row.get('name', row['symbol']),
                auction_level=auction_level,
                auction_score=auction_score,
                recommended_timing=strategy['timing'],
                recommended_price=recommended_price,
                price_adjustment=price_adj,
                recommended_position=recommended_position,
                confidence=strategy['confidence'],
                reason=strategy['description'],
                risk_level=strategy['risk']
            )
            
            signals.append(signal)
            
            # 更新可用仓位
            available_position -= recommended_position
            
            # 打印信号
            timing_desc = {
                'auction_end': '9:25竞价结束',
                'open_observe': '9:30开盘观察',
                'wait_pullback': '等待回踩'
            }.get(strategy['timing'], strategy['timing'])
            
            print(f"  ✅ {row['symbol']} ({row.get('name', '')})")
            print(f"     竞价强度: {auction_score:.1f}分（{auction_level}）")
            print(f"     买入时机: {timing_desc}")
            print(f"     买入价格: ¥{recommended_price:.2f} (竞价价{price_adj:+.1%})")
            print(f"     建议仓位: {recommended_position:.1%}")
            print(f"     置信度: {strategy['confidence']:.0%}")
            print(f"     风险等级: {strategy['risk']}")
        
        print(f"\n生成买入信号: {len(signals)} 个")
        print(f"{'='*60}\n")
        
        return signals
    
    def _classify_auction_level(self, auction_score: float) -> str:
        """根据竞价评分分类"""
        levels = self.config['auction_levels']
        
        if auction_score >= levels['super_strong']:
            return 'super_strong'
        elif auction_score >= levels['strong']:
            return 'strong'
        elif auction_score >= levels['medium']:
            return 'medium'
        else:
            return 'weak'
    
    def prioritize_signals(self, 
                          signals: List[BuySignal],
                          max_stocks: int = 5) -> List[BuySignal]:
        """
        信号优先级排序（资金有限时选最优）
        
        Parameters:
        -----------
        signals: List[BuySignal]
            买入信号列表
        max_stocks: int
            最多买入股票数
            
        Returns:
        --------
        List[BuySignal]: 排序后的信号列表（前max_stocks个）
        """
        if len(signals) <= max_stocks:
            return signals
        
        # 排序规则：
        # 1. 优先超强和强势
        # 2. 然后按竞价评分
        # 3. 最后按置信度
        
        level_priority = {
            'super_strong': 4,
            'strong': 3,
            'medium': 2,
            'weak': 1
        }
        
        sorted_signals = sorted(
            signals,
            key=lambda s: (
                level_priority.get(s.auction_level, 0),
                s.auction_score,
                s.confidence
            ),
            reverse=True
        )
        
        return sorted_signals[:max_stocks]
    
    def execute_buy_order(self,
                         signal: BuySignal,
                         current_cash: float) -> Optional[Dict]:
        """
        执行买入订单（模拟）
        
        Parameters:
        -----------
        signal: BuySignal
            买入信号
        current_cash: float
            当前可用资金
            
        Returns:
        --------
        Dict: 订单信息，None表示无法执行
        """
        # 计算买入金额和股数
        position_value = current_cash * signal.recommended_position
        volume = int(position_value / signal.recommended_price / 100) * 100  # 整百股
        
        if volume < 100:
            print(f"❌ {signal.symbol}: 资金不足100股，无法买入")
            return None
        
        actual_cost = signal.recommended_price * volume
        
        order = {
            'symbol': signal.symbol,
            'name': signal.name,
            'buy_price': signal.recommended_price,
            'volume': volume,
            'cost': actual_cost,
            'position_ratio': actual_cost / current_cash,
            'auction_level': signal.auction_level,
            'auction_score': signal.auction_score,
            'timing': signal.recommended_timing,
            'confidence': signal.confidence,
            'risk_level': signal.risk_level,
            'buy_time': datetime.now(),
            'reason': signal.reason
        }
        
        print(f"✅ 买入订单: {signal.symbol}")
        print(f"   价格: ¥{signal.recommended_price:.2f}")
        print(f"   数量: {volume} 股")
        print(f"   金额: ¥{actual_cost:,.2f}")
        print(f"   仓位: {order['position_ratio']:.1%}")
        
        return order
    
    def batch_execute(self,
                     signals: List[BuySignal],
                     current_cash: float) -> List[Dict]:
        """
        批量执行买入订单
        
        Parameters:
        -----------
        signals: List[BuySignal]
            买入信号列表
        current_cash: float
            当前可用资金
            
        Returns:
        --------
        List[Dict]: 成功执行的订单列表
        """
        orders = []
        remaining_cash = current_cash
        
        print(f"\n{'='*60}")
        print(f"批量买入执行")
        print(f"{'='*60}")
        
        for signal in signals:
            order = self.execute_buy_order(signal, remaining_cash)
            if order:
                orders.append(order)
                remaining_cash -= order['cost']
        
        total_cost = sum(o['cost'] for o in orders)
        print(f"\n执行汇总:")
        print(f"  成功买入: {len(orders)} 只")
        print(f"  总成本: ¥{total_cost:,.2f}")
        print(f"  剩余资金: ¥{remaining_cash:,.2f}")
        print(f"  总仓位: {total_cost/current_cash:.1%}")
        print(f"{'='*60}\n")
        
        return orders


# 使用示例
if __name__ == "__main__":
    # 模拟候选股票
    candidates = pd.DataFrame({
        'symbol': ['000001.SZ', '600519.SH', '300750.SZ', '688036.SH'],
        'name': ['平安银行', '贵州茅台', '宁德时代', '传音控股'],
        'auction_score': [92, 78, 68, 52],
        'auction_price': [11.5, 1850, 245, 88],
        'close': [11.2, 1820, 240, 86]
    })
    
    # 创建策略
    strategy = TieredBuyStrategy()
    
    # 生成信号
    signals = strategy.generate_buy_signals(
        candidates,
        current_cash=100000
    )
    
    # 优先级排序
    prioritized = strategy.prioritize_signals(signals, max_stocks=3)
    
    # 批量执行
    orders = strategy.batch_execute(prioritized, current_cash=100000)
    
    print(f"\n✅ 完成！共买入 {len(orders)} 只股票")
