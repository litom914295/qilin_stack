#!/usr/bin/env python
"""
每日涨停股筛选脚本
自动获取当日涨停股并筛选出最强势的前 3-10 只
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from app.enhanced_limitup_selector import EnhancedLimitUpSelector, LimitUpStock
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_limitup_data_today():
    """
    获取当日涨停股数据
    
    实际使用时应该从：
    1. AKShare: ak.stock_zt_pool_em()
    2. Tushare: pro.limit_list()
    3. 本地数据库
    
    这里返回模拟数据用于测试
    """
    # TODO: 实际接入数据源
    # import akshare as ak
    # df = ak.stock_zt_pool_em(date=datetime.now().strftime('%Y%m%d'))
    
    # 模拟数据
    stocks = []
    for i, code in enumerate(['000001', '000002', '600000', '600519', '000858', '300750']):
        stocks.append(LimitUpStock(
            symbol=code,
            name=f'股票{code}',
            date=datetime.now().strftime('%Y-%m-%d'),
            limit_up_time=f'09:{30+i*5}:00',
            open_times=np.random.randint(0, 3),
            seal_ratio=np.random.uniform(0.03, 0.3),
            is_one_word=np.random.choice([True, False]),
            consecutive_days=np.random.randint(1, 4),
            is_first_board=np.random.choice([True, False]),
            prev_limit_up=False,
            sector='科技板块',
            themes=['人工智能', '芯片'],
            sector_limit_count=np.random.randint(3, 12),
            is_sector_leader=i < 2,
            prev_close=10.0,
            open=10.5,
            high=11.0,
            low=10.3,
            close=11.0,
            limit_price=11.0,
            volume=1000000,
            amount=11000000,
            turnover_rate=np.random.uniform(0.05, 0.25),
            volume_ratio=np.random.uniform(1.5, 5.0),
            vwap_slope_morning=np.random.uniform(0.001, 0.01),
            max_drawdown_morning=np.random.uniform(0.01, 0.05),
            afternoon_strength=np.random.uniform(0.6, 1.0),
            quality_score=0.0,  # 待计算
            confidence=0.0  # 待计算
        ))
    
    return stocks


def main():
    """主函数"""
    logger.info("开始执行每日涨停股筛选...")
    
    # 1. 获取当日涨停股数据
    limitup_stocks = get_limitup_data_today()
    logger.info(f"获取到 {len(limitup_stocks)} 只涨停股")
    
    # 2. 创建筛选器并计算评分
    selector = EnhancedLimitUpSelector()
    
    for stock in limitup_stocks:
        stock.quality_score = selector.calculate_quality_score(stock)
        stock.confidence = selector.calculate_confidence(stock)
    
    # 3. 筛选优质股票
    qualified = selector.select_qualified_stocks(
        limitup_stocks,
        min_quality_score=60.0,
        min_confidence=0.4,
        max_open_times=2,
        prefer_first_board=True,
        check_market_timing=False  # 测试时关闭市场择时
    )
    
    logger.info(f"筛选出 {len(qualified)} 只优质股票")
    
    # 4. 保存结果
    output_dir = Path(__file__).parent.parent / "data" / "daily_selections"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"limitup_{datetime.now().strftime('%Y%m%d')}.json"
    
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_limitup': len(limitup_stocks),
        'qualified_count': len(qualified),
        'stocks': [
            {
                'symbol': s.symbol,
                'name': s.name,
                'quality_score': float(s.quality_score),
                'confidence': float(s.confidence),
                'limit_up_time': s.limit_up_time,
                'open_times': int(s.open_times),
                'seal_ratio': float(s.seal_ratio),
                'consecutive_days': int(s.consecutive_days),
                'is_first_board': bool(s.is_first_board),
                'sector': s.sector,
                'themes': s.themes,
                'sector_limit_count': int(s.sector_limit_count)
            }
            for s in qualified[:10]  # 保存前10只
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {output_file}")
    
    # 5. 输出前3只股票
    top_3 = qualified[:3]
    print("\n" + "="*60)
    print("今日最强势前3只涨停股:")
    print("="*60)
    for i, stock in enumerate(top_3, 1):
        print(f"\n{i}. {stock.symbol} - {stock.name}")
        print(f"   质量评分: {stock.quality_score:.1f}")
        print(f"   置信度: {stock.confidence:.2%}")
        print(f"   涨停时间: {stock.limit_up_time}")
        print(f"   开板次数: {stock.open_times}")
        print(f"   封单比例: {stock.seal_ratio:.2%}")
        print(f"   连板天数: {stock.consecutive_days}")
    print("="*60 + "\n")
    
    return qualified


if __name__ == '__main__':
    main()
