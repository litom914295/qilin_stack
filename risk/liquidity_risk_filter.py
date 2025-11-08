"""
流动性和风险过滤器
对候选股票进行多维度流动性和风险筛选

功能：
- 成交量和换手率过滤
- 波动率过滤
- 价格和市值过滤
- ST股票过滤
- 异常波动过滤
- 综合风险评分
"""

import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd


class LiquidityRiskFilter:
    """
    流动性和风险过滤器
    用于筛选符合流动性和风险要求的股票
    """
    
    def __init__(
        self,
        min_volume: float = 1e8,  # 最小成交额（元）
        min_turnover: float = 0.02,  # 最小换手率（2%）
        max_volatility: float = 0.15,  # 最大日波动率（15%）
        min_price: float = 2.0,  # 最小股价
        max_price: float = 300.0,  # 最大股价
        min_market_cap: float = 10e8,  # 最小市值（10亿）
        max_market_cap: Optional[float] = None,  # 最大市值
        filter_st: bool = True,  # 是否过滤ST股票
        filter_suspended: bool = True  # 是否过滤停牌股票
    ):
        """
        初始化过滤器
        
        Args:
            min_volume: 最小成交额
            min_turnover: 最小换手率
            max_volatility: 最大波动率
            min_price: 最小股价
            max_price: 最大股价
            min_market_cap: 最小市值
            max_market_cap: 最大市值
            filter_st: 是否过滤ST股票
            filter_suspended: 是否过滤停牌股票
        """
        self.min_volume = min_volume
        self.min_turnover = min_turnover
        self.max_volatility = max_volatility
        self.min_price = min_price
        self.max_price = max_price
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.filter_st = filter_st
        self.filter_suspended = filter_suspended
    
    def check_volume(self, stock_data: pd.Series) -> Tuple[bool, str]:
        """
        检查成交量
        
        Args:
            stock_data: 股票数据（需包含volume或amount字段）
        
        Returns:
            (是否通过, 原因)
        """
        if 'amount' in stock_data:
            volume = stock_data['amount']
        elif 'volume' in stock_data and 'close' in stock_data:
            volume = stock_data['volume'] * stock_data['close']
        else:
            return False, "缺少成交量数据"
        
        if volume < self.min_volume:
            return False, f"成交额过低 ({volume/1e8:.2f}亿 < {self.min_volume/1e8:.2f}亿)"
        
        return True, "成交量充足"
    
    def check_turnover(self, stock_data: pd.Series) -> Tuple[bool, str]:
        """
        检查换手率
        
        Args:
            stock_data: 股票数据（需包含turnover字段）
        
        Returns:
            (是否通过, 原因)
        """
        if 'turnover' not in stock_data:
            return True, "换手率数据缺失（跳过检查）"
        
        turnover = stock_data['turnover']
        
        if turnover < self.min_turnover:
            return False, f"换手率过低 ({turnover:.2%} < {self.min_turnover:.2%})"
        
        return True, "换手率充足"
    
    def check_volatility(self, stock_data: pd.Series) -> Tuple[bool, str]:
        """
        检查波动率
        
        Args:
            stock_data: 股票数据（需包含volatility或high/low字段）
        
        Returns:
            (是否通过, 原因)
        """
        if 'volatility' in stock_data:
            volatility = stock_data['volatility']
        elif 'high' in stock_data and 'low' in stock_data and 'close' in stock_data:
            # 使用当日振幅作为波动率近似
            volatility = (stock_data['high'] - stock_data['low']) / stock_data['close']
        else:
            return True, "波动率数据缺失（跳过检查）"
        
        if volatility > self.max_volatility:
            return False, f"波动率过高 ({volatility:.2%} > {self.max_volatility:.2%})"
        
        return True, "波动率正常"
    
    def check_price(self, stock_data: pd.Series) -> Tuple[bool, str]:
        """
        检查价格
        
        Args:
            stock_data: 股票数据（需包含close字段）
        
        Returns:
            (是否通过, 原因)
        """
        if 'close' not in stock_data:
            return False, "缺少价格数据"
        
        price = stock_data['close']
        
        if price < self.min_price:
            return False, f"股价过低 ({price:.2f} < {self.min_price:.2f})"
        
        if price > self.max_price:
            return False, f"股价过高 ({price:.2f} > {self.max_price:.2f})"
        
        return True, "价格适中"
    
    def check_market_cap(self, stock_data: pd.Series) -> Tuple[bool, str]:
        """
        检查市值
        
        Args:
            stock_data: 股票数据（需包含market_cap字段）
        
        Returns:
            (是否通过, 原因)
        """
        if 'market_cap' not in stock_data:
            return True, "市值数据缺失（跳过检查）"
        
        market_cap = stock_data['market_cap']
        
        if market_cap < self.min_market_cap:
            return False, f"市值过小 ({market_cap/1e8:.2f}亿 < {self.min_market_cap/1e8:.2f}亿)"
        
        if self.max_market_cap and market_cap > self.max_market_cap:
            return False, f"市值过大 ({market_cap/1e8:.2f}亿 > {self.max_market_cap/1e8:.2f}亿)"
        
        return True, "市值适中"
    
    def check_st_status(self, stock_data: pd.Series) -> Tuple[bool, str]:
        """
        检查ST状态
        
        Args:
            stock_data: 股票数据（需包含is_st或symbol字段）
        
        Returns:
            (是否通过, 原因)
        """
        if not self.filter_st:
            return True, "ST过滤已禁用"
        
        # 检查is_st字段
        if 'is_st' in stock_data:
            if stock_data['is_st']:
                return False, "ST股票"
        
        # 检查symbol中是否包含ST标识
        if 'symbol' in stock_data:
            symbol = str(stock_data['symbol'])
            if 'ST' in symbol or '*ST' in symbol:
                return False, "ST股票"
        
        # 检查name中是否包含ST标识
        if 'name' in stock_data:
            name = str(stock_data['name'])
            if 'ST' in name or '*ST' in name or 'S' == name[0]:
                return False, "ST股票"
        
        return True, "非ST股票"
    
    def check_suspended(self, stock_data: pd.Series) -> Tuple[bool, str]:
        """
        检查停牌状态
        
        Args:
            stock_data: 股票数据（需包含is_suspended或volume字段）
        
        Returns:
            (是否通过, 原因)
        """
        if not self.filter_suspended:
            return True, "停牌过滤已禁用"
        
        # 检查is_suspended字段
        if 'is_suspended' in stock_data:
            if stock_data['is_suspended']:
                return False, "停牌中"
        
        # 检查成交量是否为0（可能停牌）
        if 'volume' in stock_data:
            if stock_data['volume'] == 0:
                return False, "成交量为0（可能停牌）"
        
        return True, "正常交易"
    
    def compute_risk_score(self, stock_data: pd.Series) -> float:
        """
        计算综合风险得分
        
        Args:
            stock_data: 股票数据
        
        Returns:
            风险得分 (0-1, 越高风险越大)
        """
        risk_score = 0.0
        
        # 1. 流动性风险（成交量、换手率）
        if 'amount' in stock_data:
            volume_score = min(stock_data['amount'] / self.min_volume, 1.0)
            risk_score += (1 - volume_score) * 0.3  # 流动性不足风险
        
        if 'turnover' in stock_data:
            turnover_score = min(stock_data['turnover'] / self.min_turnover, 1.0)
            risk_score += (1 - turnover_score) * 0.2
        
        # 2. 波动性风险
        if 'volatility' in stock_data:
            vol_ratio = stock_data['volatility'] / self.max_volatility
            risk_score += min(vol_ratio, 1.0) * 0.3  # 波动率过高风险
        
        # 3. 价格风险（过低或过高）
        if 'close' in stock_data:
            price = stock_data['close']
            if price < self.min_price * 2:  # 低价股
                risk_score += 0.1
            elif price > self.max_price * 0.5:  # 高价股
                risk_score += 0.05
        
        # 4. ST风险
        if self.filter_st:
            is_st, _ = self.check_st_status(stock_data)
            if not is_st:
                risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def filter_stock(self, stock_data: pd.Series) -> Dict:
        """
        对单只股票进行全面过滤
        
        Args:
            stock_data: 股票数据
        
        Returns:
            过滤结果字典
        """
        checks = {
            'volume': self.check_volume(stock_data),
            'turnover': self.check_turnover(stock_data),
            'volatility': self.check_volatility(stock_data),
            'price': self.check_price(stock_data),
            'market_cap': self.check_market_cap(stock_data),
            'st_status': self.check_st_status(stock_data),
            'suspended': self.check_suspended(stock_data)
        }
        
        # 判断是否通过所有检查
        passed = all(result[0] for result in checks.values())
        
        # 收集失败原因
        failed_reasons = [reason for result, reason in checks.values() if not result]
        
        # 计算风险得分
        risk_score = self.compute_risk_score(stock_data)
        
        return {
            'symbol': stock_data.get('symbol', 'unknown'),
            'passed': passed,
            'risk_score': risk_score,
            'checks': {name: {'passed': result[0], 'reason': result[1]} 
                      for name, result in checks.items()},
            'failed_reasons': failed_reasons,
            'timestamp': datetime.now().isoformat()
        }
    
    def filter_stocks(self, stocks_data: pd.DataFrame) -> pd.DataFrame:
        """
        批量过滤股票
        
        Args:
            stocks_data: 股票数据DataFrame
        
        Returns:
            过滤结果DataFrame
        """
        results = []
        
        for idx, row in stocks_data.iterrows():
            filter_result = self.filter_stock(row)
            results.append({
                'symbol': filter_result['symbol'],
                'passed': filter_result['passed'],
                'risk_score': filter_result['risk_score'],
                'failed_reasons': '; '.join(filter_result['failed_reasons']) if filter_result['failed_reasons'] else 'OK'
            })
        
        return pd.DataFrame(results)
    
    def get_passed_stocks(self, stocks_data: pd.DataFrame) -> pd.DataFrame:
        """
        获取通过过滤的股票
        
        Args:
            stocks_data: 股票数据DataFrame
        
        Returns:
            通过过滤的股票DataFrame
        """
        filter_results = self.filter_stocks(stocks_data)
        passed_symbols = filter_results[filter_results['passed']]['symbol'].tolist()
        
        return stocks_data[stocks_data['symbol'].isin(passed_symbols)]
    
    def get_filter_stats(self, stocks_data: pd.DataFrame) -> Dict:
        """
        获取过滤统计信息
        
        Args:
            stocks_data: 股票数据DataFrame
        
        Returns:
            统计信息字典
        """
        filter_results = self.filter_stocks(stocks_data)
        
        total = len(filter_results)
        passed = filter_results['passed'].sum()
        failed = total - passed
        
        # 统计各类失败原因
        reason_counts = {}
        for reasons_str in filter_results[~filter_results['passed']]['failed_reasons']:
            for reason in reasons_str.split('; '):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            'total': total,
            'passed': int(passed),
            'failed': int(failed),
            'pass_rate': passed / total if total > 0 else 0,
            'avg_risk_score': float(filter_results['risk_score'].mean()),
            'failure_reasons': reason_counts,
            'timestamp': datetime.now().isoformat()
        }


def create_mock_stocks_data(n_stocks: int = 100) -> pd.DataFrame:
    """
    创建模拟股票数据（用于测试）
    
    Args:
        n_stocks: 股票数量
    
    Returns:
        股票数据DataFrame
    """
    np.random.seed(42)
    
    symbols = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    
    data = {
        'symbol': symbols,
        'name': [f"Stock_{i}" for i in range(n_stocks)],
        'close': np.random.lognormal(2.5, 0.8, n_stocks).clip(2, 200),
        'volume': np.random.lognormal(18, 1.5, n_stocks),
        'amount': np.random.lognormal(18, 1.5, n_stocks),
        'turnover': np.random.lognormal(-3.5, 0.5, n_stocks).clip(0.001, 0.5),
        'volatility': np.random.lognormal(-2.5, 0.5, n_stocks).clip(0.01, 0.3),
        'market_cap': np.random.lognormal(22, 1.0, n_stocks),
        'is_st': np.random.choice([True, False], n_stocks, p=[0.05, 0.95]),
        'is_suspended': np.random.choice([True, False], n_stocks, p=[0.02, 0.98])
    }
    
    return pd.DataFrame(data)


# ==================== 测试和示例代码 ====================

if __name__ == "__main__":
    """测试流动性和风险过滤器"""
    
    print("=" * 60)
    print("流动性和风险过滤器测试")
    print("=" * 60)
    
    # 创建模拟股票数据
    stocks_data = create_mock_stocks_data(n_stocks=100)
    print(f"生成模拟股票数据: {len(stocks_data)} 只")
    print(f"价格范围: {stocks_data['close'].min():.2f} - {stocks_data['close'].max():.2f}")
    print(f"换手率范围: {stocks_data['turnover'].min():.2%} - {stocks_data['turnover'].max():.2%}")
    print(f"ST股票数量: {stocks_data['is_st'].sum()}")
    
    # 创建过滤器
    filter_engine = LiquidityRiskFilter(
        min_volume=1e8,
        min_turnover=0.02,
        max_volatility=0.15,
        min_price=5.0,
        max_price=200.0,
        min_market_cap=10e8,
        filter_st=True,
        filter_suspended=True
    )
    
    print("\n✓ 过滤器已创建")
    
    # 测试单只股票过滤
    print("\n" + "=" * 60)
    print("测试单只股票过滤...")
    print("=" * 60)
    
    test_stock = stocks_data.iloc[0]
    filter_result = filter_engine.filter_stock(test_stock)
    
    print(f"\n股票代码: {filter_result['symbol']}")
    print(f"是否通过: {filter_result['passed']}")
    print(f"风险得分: {filter_result['risk_score']:.2f}")
    print("\n检查详情:")
    for check_name, check_result in filter_result['checks'].items():
        status = "✓" if check_result['passed'] else "✗"
        print(f"  {status} {check_name}: {check_result['reason']}")
    
    # 批量过滤
    print("\n" + "=" * 60)
    print("批量过滤股票...")
    print("=" * 60)
    
    filter_results = filter_engine.filter_stocks(stocks_data)
    print(f"\n过滤完成，共 {len(filter_results)} 只股票")
    print("\n通过的前10只股票:")
    print(filter_results[filter_results['passed']].head(10).to_string(index=False))
    
    print("\n失败的前10只股票:")
    print(filter_results[~filter_results['passed']].head(10).to_string(index=False))
    
    # 获取过滤统计
    print("\n" + "=" * 60)
    print("过滤统计信息:")
    print("=" * 60)
    
    stats = filter_engine.get_filter_stats(stocks_data)
    print(f"\n总数: {stats['total']}")
    print(f"通过: {stats['passed']}")
    print(f"失败: {stats['failed']}")
    print(f"通过率: {stats['pass_rate']:.1%}")
    print(f"平均风险得分: {stats['avg_risk_score']:.2f}")
    
    print("\n失败原因分布:")
    for reason, count in sorted(stats['failure_reasons'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} 次")
    
    # 获取通过的股票
    print("\n" + "=" * 60)
    print("获取通过过滤的股票...")
    print("=" * 60)
    
    passed_stocks = filter_engine.get_passed_stocks(stocks_data)
    print(f"\n通过过滤的股票数量: {len(passed_stocks)}")
    print("\n样本:")
    print(passed_stocks[['symbol', 'name', 'close', 'turnover', 'amount']].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
