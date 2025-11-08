"""
Kelly准则仓位管理优化器
根据胜率和赔率动态调整单只股票仓位
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class PositionSize:
    """仓位配置"""
    symbol: str
    name: str
    kelly_fraction: float  # Kelly准则计算的最优仓位比例
    adjusted_fraction: float  # 调整后的实际仓位比例
    recommended_amount: float  # 推荐金额
    win_rate: float  # 胜率
    win_loss_ratio: float  # 盈亏比
    confidence_level: float  # 置信度
    reason: str  # 调整原因


class KellyPositionManager:
    """
    Kelly准则仓位管理器
    
    核心公式：
    f* = (p * b - q) / b
    
    其中：
    f* = Kelly最优仓位比例
    p = 胜率
    q = 1 - p (败率)
    b = 盈亏比 (平均盈利/平均亏损)
    
    调整策略：
    1. 保守调整：实际仓位 = Kelly仓位 * 调整系数(0.25-0.5)
    2. 上限控制：单只股票不超过总资金的10%
    3. 下限控制：低于1%的不建议配置
    4. 市场环境调整：根据市场环境动态调整
    """
    
    def __init__(self, 
                 total_capital: float = 1000000,
                 kelly_fraction: float = 0.25,
                 max_position: float = 0.10,
                 min_position: float = 0.01):
        """
        初始化仓位管理器
        
        Parameters:
        -----------
        total_capital: float
            总资金
        kelly_fraction: float
            Kelly准则调整系数（保守策略，通常0.25-0.5）
        max_position: float
            单只股票最大仓位比例
        min_position: float
            单只股票最小仓位比例
        """
        self.total_capital = total_capital
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position
    
    def calculate_kelly_position(self,
                                 win_rate: float,
                                 avg_win: float,
                                 avg_loss: float) -> float:
        """
        计算Kelly最优仓位
        
        Parameters:
        -----------
        win_rate: float
            胜率 (0-1)
        avg_win: float
            平均盈利金额
        avg_loss: float
            平均亏损金额（正数）
            
        Returns:
        --------
        float: Kelly最优仓位比例
        """
        if avg_loss == 0:
            return 0
        
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss  # 盈亏比
        
        # Kelly公式
        kelly = (p * b - q) / b
        
        # 确保在合理范围内
        kelly = max(0, min(kelly, 1))
        
        return kelly
    
    def calculate_positions(self,
                           candidates: pd.DataFrame,
                           historical_performance: Optional[Dict[str, Dict]] = None) -> List[PositionSize]:
        """
        批量计算候选股票的仓位配置
        
        Parameters:
        -----------
        candidates: DataFrame
            候选股票列表，包含symbol, name等信息
        historical_performance: Dict
            历史表现数据 {symbol: {'win_rate': 0.65, 'avg_win': 500, 'avg_loss': 300, ...}}
            
        Returns:
        --------
        List[PositionSize]: 仓位配置列表
        """
        positions = []
        
        print(f"\n{'='*80}")
        print(f"Kelly准则仓位管理 - 计算开始")
        print(f"{'='*80}")
        print(f"总资金: ¥{self.total_capital:,.0f}")
        print(f"Kelly调整系数: {self.kelly_fraction}")
        print(f"单只最大仓位: {self.max_position * 100:.1f}%")
        print(f"单只最小仓位: {self.min_position * 100:.1f}%")
        print(f"候选股票数: {len(candidates)}")
        print(f"{'='*80}\n")
        
        for idx, row in candidates.iterrows():
            symbol = row['symbol']
            name = row.get('name', symbol)
            
            # 获取历史表现
            if historical_performance and symbol in historical_performance:
                perf = historical_performance[symbol]
                win_rate = perf.get('win_rate', 0.6)
                avg_win = perf.get('avg_win', 500)
                avg_loss = perf.get('avg_loss', 300)
                confidence = perf.get('confidence', 0.7)
            else:
                # 使用默认值或从候选数据中提取
                win_rate = row.get('prediction_score', 0.65)
                avg_win = 500
                avg_loss = 300
                confidence = 0.7
            
            # 计算Kelly仓位
            kelly_raw = self.calculate_kelly_position(win_rate, avg_win, avg_loss)
            
            # 应用Kelly调整系数（保守策略）
            kelly_adjusted = kelly_raw * self.kelly_fraction
            
            # 应用限制条件
            final_fraction, reason = self._apply_constraints(
                kelly_adjusted,
                win_rate,
                avg_win / avg_loss if avg_loss > 0 else 0,
                confidence
            )
            
            # 计算推荐金额
            recommended_amount = self.total_capital * final_fraction
            
            position = PositionSize(
                symbol=symbol,
                name=name,
                kelly_fraction=kelly_raw,
                adjusted_fraction=final_fraction,
                recommended_amount=recommended_amount,
                win_rate=win_rate,
                win_loss_ratio=avg_win / avg_loss if avg_loss > 0 else 0,
                confidence_level=confidence,
                reason=reason
            )
            
            positions.append(position)
        
        # 归一化处理（确保总仓位不超过100%）
        positions = self._normalize_positions(positions)
        
        # 打印结果
        self._print_positions(positions)
        
        return positions
    
    def _apply_constraints(self,
                          kelly_fraction: float,
                          win_rate: float,
                          win_loss_ratio: float,
                          confidence: float) -> Tuple[float, str]:
        """
        应用约束条件
        
        Parameters:
        -----------
        kelly_fraction: float
            Kelly调整后仓位
        win_rate: float
            胜率
        win_loss_ratio: float
            盈亏比
        confidence: float
            置信度
            
        Returns:
        --------
        Tuple[float, str]: (最终仓位, 调整原因)
        """
        reasons = []
        final_fraction = kelly_fraction
        
        # 1. 上限控制
        if final_fraction > self.max_position:
            final_fraction = self.max_position
            reasons.append(f"触及上限{self.max_position*100:.1f}%")
        
        # 2. 下限控制
        if final_fraction < self.min_position:
            if final_fraction > 0:
                reasons.append(f"低于下限，不建议配置")
            final_fraction = 0
        
        # 3. 胜率调整
        if win_rate < 0.5:
            final_fraction *= 0.5
            reasons.append("胜率偏低，减半")
        elif win_rate > 0.7:
            final_fraction *= 1.2
            reasons.append("高胜率，+20%")
        
        # 4. 盈亏比调整
        if win_loss_ratio < 1.5:
            final_fraction *= 0.8
            reasons.append("盈亏比偏低，-20%")
        elif win_loss_ratio > 3:
            final_fraction *= 1.2
            reasons.append("高盈亏比，+20%")
        
        # 5. 置信度调整
        if confidence < 0.6:
            final_fraction *= 0.7
            reasons.append("低置信度，-30%")
        elif confidence > 0.8:
            final_fraction *= 1.1
            reasons.append("高置信度，+10%")
        
        # 6. 再次检查上下限
        final_fraction = max(0, min(final_fraction, self.max_position))
        
        # 7. 低于下限则设为0
        if final_fraction < self.min_position:
            final_fraction = 0
            reasons.append("最终低于下限")
        
        reason = "; ".join(reasons) if reasons else "标准配置"
        
        return final_fraction, reason
    
    def _normalize_positions(self, positions: List[PositionSize]) -> List[PositionSize]:
        """
        归一化仓位配置，确保总仓位不超过100%
        
        Parameters:
        -----------
        positions: List[PositionSize]
            原始仓位配置
            
        Returns:
        --------
        List[PositionSize]: 归一化后的仓位配置
        """
        # 计算总仓位
        total_fraction = sum(p.adjusted_fraction for p in positions)
        
        # 如果总仓位超过100%，按比例缩减
        if total_fraction > 1.0:
            scale_factor = 0.95 / total_fraction  # 保守一点，只用95%
            
            normalized_positions = []
            for pos in positions:
                new_fraction = pos.adjusted_fraction * scale_factor
                new_amount = self.total_capital * new_fraction
                
                normalized_pos = PositionSize(
                    symbol=pos.symbol,
                    name=pos.name,
                    kelly_fraction=pos.kelly_fraction,
                    adjusted_fraction=new_fraction,
                    recommended_amount=new_amount,
                    win_rate=pos.win_rate,
                    win_loss_ratio=pos.win_loss_ratio,
                    confidence_level=pos.confidence_level,
                    reason=pos.reason + "; 归一化调整"
                )
                normalized_positions.append(normalized_pos)
            
            return normalized_positions
        
        return positions
    
    def _print_positions(self, positions: List[PositionSize]):
        """打印仓位配置"""
        print(f"\n{'='*120}")
        print(f"仓位配置结果")
        print(f"{'='*120}")
        
        # 表头
        print(f"{'股票':<15} {'胜率':>8} {'盈亏比':>8} {'置信度':>8} {'Kelly原始':>10} {'调整后':>10} {'推荐金额':>15} {'调整原因':<30}")
        print(f"{'-'*120}")
        
        # 数据行
        valid_positions = [p for p in positions if p.adjusted_fraction > 0]
        for pos in sorted(valid_positions, key=lambda x: x.adjusted_fraction, reverse=True):
            print(f"{pos.symbol:<15} "
                  f"{pos.win_rate*100:>7.1f}% "
                  f"{pos.win_loss_ratio:>8.2f} "
                  f"{pos.confidence_level*100:>7.1f}% "
                  f"{pos.kelly_fraction*100:>9.2f}% "
                  f"{pos.adjusted_fraction*100:>9.2f}% "
                  f"¥{pos.recommended_amount:>13,.0f} "
                  f"{pos.reason:<30}")
        
        print(f"{'-'*120}")
        
        # 汇总统计
        total_fraction = sum(p.adjusted_fraction for p in positions)
        total_amount = sum(p.recommended_amount for p in positions)
        used_count = len(valid_positions)
        unused_count = len(positions) - used_count
        
        print(f"\n汇总统计:")
        print(f"  配置股票数: {used_count}/{len(positions)}")
        print(f"  未配置股票数: {unused_count}")
        print(f"  总仓位比例: {total_fraction*100:.2f}%")
        print(f"  总配置金额: ¥{total_amount:,.0f}")
        print(f"  剩余可用资金: ¥{self.total_capital - total_amount:,.0f}")
        print(f"  资金使用率: {total_amount/self.total_capital*100:.2f}%")
        print(f"{'='*120}\n")
    
    def export_positions(self, positions: List[PositionSize], output_path: str):
        """导出仓位配置到CSV"""
        df = pd.DataFrame([{
            '股票代码': p.symbol,
            '股票名称': p.name,
            '胜率(%)': f"{p.win_rate*100:.2f}",
            '盈亏比': f"{p.win_loss_ratio:.2f}",
            '置信度(%)': f"{p.confidence_level*100:.2f}",
            'Kelly原始仓位(%)': f"{p.kelly_fraction*100:.2f}",
            '调整后仓位(%)': f"{p.adjusted_fraction*100:.2f}",
            '推荐金额(¥)': f"{p.recommended_amount:.0f}",
            '调整原因': p.reason
        } for p in positions if p.adjusted_fraction > 0])
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"仓位配置已导出到: {output_path}")
    
    def update_performance(self,
                          symbol: str,
                          is_win: bool,
                          profit_loss: float,
                          historical_performance: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        更新股票的历史表现
        
        Parameters:
        -----------
        symbol: str
            股票代码
        is_win: bool
            是否盈利
        profit_loss: float
            盈亏金额
        historical_performance: Dict
            历史表现字典
            
        Returns:
        --------
        Dict: 更新后的历史表现
        """
        if symbol not in historical_performance:
            historical_performance[symbol] = {
                'total_trades': 0,
                'win_trades': 0,
                'total_win': 0,
                'total_loss': 0,
                'win_rate': 0.5,
                'avg_win': 0,
                'avg_loss': 0,
                'confidence': 0.5
            }
        
        perf = historical_performance[symbol]
        
        # 更新交易统计
        perf['total_trades'] += 1
        if is_win:
            perf['win_trades'] += 1
            perf['total_win'] += profit_loss
        else:
            perf['total_loss'] += abs(profit_loss)
        
        # 重新计算指标
        perf['win_rate'] = perf['win_trades'] / perf['total_trades']
        perf['avg_win'] = perf['total_win'] / perf['win_trades'] if perf['win_trades'] > 0 else 0
        loss_trades = perf['total_trades'] - perf['win_trades']
        perf['avg_loss'] = perf['total_loss'] / loss_trades if loss_trades > 0 else 0
        
        # 更新置信度（交易次数越多，置信度越高）
        perf['confidence'] = min(0.5 + perf['total_trades'] * 0.02, 0.95)
        
        return historical_performance


# 使用示例
if __name__ == "__main__":
    # 创建仓位管理器
    manager = KellyPositionManager(
        total_capital=1000000,
        kelly_fraction=0.25,
        max_position=0.10,
        min_position=0.01
    )
    
    # 模拟候选股票
    candidates = pd.DataFrame({
        'symbol': [f'{i:06d}.SZ' for i in range(20)],
        'name': [f'股票{i}' for i in range(20)],
        'prediction_score': np.random.uniform(0.5, 0.9, 20)
    })
    
    # 模拟历史表现
    historical_performance = {}
    for i in range(20):
        symbol = f'{i:06d}.SZ'
        historical_performance[symbol] = {
            'win_rate': np.random.uniform(0.45, 0.75),
            'avg_win': np.random.uniform(300, 800),
            'avg_loss': np.random.uniform(200, 500),
            'confidence': np.random.uniform(0.6, 0.9)
        }
    
    # 计算仓位配置
    positions = manager.calculate_positions(
        candidates=candidates,
        historical_performance=historical_performance
    )
    
    # 导出结果
    manager.export_positions(positions, 'kelly_positions.csv')
    
    print("\n✅ Kelly准则仓位管理计算完成！")
