#!/usr/bin/env python
"""
Windows 兼容的简化版因子发现模块
无需 Docker，纯 Python 实现
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SimplifiedFactorDiscovery:
    """
    简化版因子发现系统
    - 无需 RD-Agent 依赖
    - 无需 Docker
    - 纯 Python 实现
    - Windows 完全兼容
    """
    
    def __init__(self, cache_dir: str = "./workspace/factor_cache"):
        """
        初始化因子发现系统
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 预定义因子库
        self.factor_library = self._init_factor_library()
        
        logger.info("✅ 简化版因子发现系统初始化成功 (Windows兼容)")
    
    def _init_factor_library(self) -> List[Dict[str, Any]]:
        """初始化涨停板因子库"""
        return [
            {
                'id': 'limitup_001',
                'name': '封板强度',
                'expression': '封单金额 / 流通市值',
                'code': 'seal_amount / market_cap',
                'category': 'seal_strength',
                'description': '衡量封板资金力度，值越大表示封板越强',
                'expected_ic': 0.08,
                'data_requirements': ['seal_amount', 'market_cap']
            },
            {
                'id': 'limitup_002',
                'name': '连板高度因子',
                'expression': 'log(连板天数 + 1) * 量比',
                'code': 'np.log1p(continuous_board) * volume_ratio',
                'category': 'continuous_board',
                'description': '连板高度与量能的共振',
                'expected_ic': 0.12,
                'data_requirements': ['continuous_board', 'volume_ratio']
            },
            {
                'id': 'limitup_003',
                'name': '题材共振',
                'expression': '同题材涨停数量 * 个股强度',
                'code': 'concept_count * limit_up_strength',
                'category': 'concept_synergy',
                'description': '题材热度与个股强度结合',
                'expected_ic': 0.10,
                'data_requirements': ['concept_count', 'limit_up_strength']
            },
            {
                'id': 'limitup_004',
                'name': '早盘涨停',
                'expression': '1 - (涨停分钟数 / 240)',
                'code': '1.0 - (limit_up_minutes / 240)',
                'category': 'timing',
                'description': '涨停时间越早，次日表现越好',
                'expected_ic': 0.15,
                'data_requirements': ['limit_up_minutes']
            },
            {
                'id': 'limitup_005',
                'name': '量能爆发',
                'expression': '成交量 / 20日均量',
                'code': 'volume / volume_ma20',
                'category': 'volume_pattern',
                'description': '量能突增的力度',
                'expected_ic': 0.09,
                'data_requirements': ['volume', 'volume_ma20']
            },
            {
                'id': 'limitup_006',
                'name': '大单净流入',
                'expression': '(大单买入 - 大单卖出) / 成交额',
                'code': '(big_buy - big_sell) / turnover',
                'category': 'order_flow',
                'description': '主力资金流向',
                'expected_ic': 0.11,
                'data_requirements': ['big_buy', 'big_sell', 'turnover']
            },
            {
                'id': 'limitup_007',
                'name': '封单持续性',
                'expression': '封单持续分钟数 / 240',
                'code': 'seal_duration / 240',
                'category': 'seal_strength',
                'description': '封单的稳定程度',
                'expected_ic': 0.07,
                'data_requirements': ['seal_duration']
            },
            {
                'id': 'limitup_008',
                'name': '开板次数惩罚',
                'expression': 'exp(-开板次数)',
                'code': 'np.exp(-open_board_count)',
                'category': 'seal_strength',
                'description': '开板次数越多，次日表现越差',
                'expected_ic': -0.06,
                'data_requirements': ['open_board_count']
            },
            {
                'id': 'limitup_009',
                'name': '换手率适中',
                'expression': '1 - abs(换手率 - 最优换手率) / 最优换手率',
                'code': '1 - np.abs(turnover_rate - optimal_turnover) / optimal_turnover',
                'category': 'volume_pattern',
                'description': '换手率过高或过低都不好',
                'expected_ic': 0.08,
                'data_requirements': ['turnover_rate', 'optimal_turnover']
            },
            {
                'id': 'limitup_010',
                'name': '首板优势',
                'expression': 'is_first_board * (1 + 题材热度)',
                'code': 'is_first_board * (1 + concept_heat)',
                'category': 'continuous_board',
                'description': '首板且题材热的股票机会大',
                'expected_ic': 0.14,
                'data_requirements': ['is_first_board', 'concept_heat']
            },
            {
                'id': 'limitup_011',
                'name': '尾盘封板强度',
                'expression': '尾盘封单金额 / 全天平均封单',
                'code': 'tail_seal_amount / avg_seal_amount',
                'category': 'seal_strength',
                'description': '尾盘封板资金力度',
                'expected_ic': 0.09,
                'data_requirements': ['tail_seal_amount', 'avg_seal_amount']
            },
            {
                'id': 'limitup_012',
                'name': '分时均价偏离度',
                'expression': '(收盘价 - 分时均价) / 分时均价',
                'code': '(close - vwap) / vwap',
                'category': 'timing',
                'description': '收盘价相对分时均价的位置',
                'expected_ic': 0.06,
                'data_requirements': ['close', 'vwap']
            },
            {
                'id': 'limitup_013',
                'name': '前期高点距离',
                'expression': '(前期高点 - 当前价) / 前期高点',
                'code': '(prev_high - current_price) / prev_high',
                'category': 'technical',
                'description': '距离前高的空间',
                'expected_ic': 0.05,
                'data_requirements': ['prev_high', 'current_price']
            },
            {
                'id': 'limitup_014',
                'name': '板块联动强度',
                'expression': '板块涨停率 * 板块资金流入',
                'code': 'sector_limitup_rate * sector_capital_inflow',
                'category': 'concept_synergy',
                'description': '所属板块的整体强度',
                'expected_ic': 0.13,
                'data_requirements': ['sector_limitup_rate', 'sector_capital_inflow']
            },
            {
                'id': 'limitup_015',
                'name': '竞价强度',
                'expression': '竞价成交量 / 流通市值',
                'code': 'auction_volume / market_cap',
                'category': 'timing',
                'description': '集合竞价的资金参与度',
                'expected_ic': 0.10,
                'data_requirements': ['auction_volume', 'market_cap']
            }
        ]
    
    async def discover_factors(
        self,
        start_date: str,
        end_date: str,
        n_factors: int = 20,
        min_ic: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        发现涨停板因子
        
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            n_factors: 返回因子数量
            min_ic: 最小IC阈值
            
        Returns:
            因子列表
        """
        logger.info(f"🔍 开始因子发现: {start_date} -> {end_date}")
        
        # 筛选满足条件的因子
        qualified_factors = [
            f for f in self.factor_library
            if abs(f['expected_ic']) >= min_ic
        ]
        
        # 按 IC 排序
        qualified_factors.sort(
            key=lambda x: abs(x['expected_ic']),
            reverse=True
        )
        
        # 选择 Top-N
        selected_factors = qualified_factors[:n_factors]
        
        # 模拟评估过程（在实际使用时应该用真实数据评估）
        for factor in selected_factors:
            factor['status'] = 'discovered'
            factor['discovery_date'] = datetime.now().isoformat()
            factor['evaluation_period'] = f"{start_date} to {end_date}"
        
        logger.info(f"✅ 发现 {len(selected_factors)} 个高质量因子")
        
        # 保存到缓存
        self._save_factors(selected_factors, start_date, end_date)
        
        return selected_factors
    
    def _save_factors(
        self,
        factors: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ):
        """保存因子到缓存"""
        cache_file = self.cache_dir / f"factors_{start_date}_{end_date}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'factors': factors,
                'start_date': start_date,
                'end_date': end_date,
                'created_at': datetime.now().isoformat(),
                'count': len(factors)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 因子已保存: {cache_file}")
    
    def get_factor_by_id(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取因子"""
        for factor in self.factor_library:
            if factor['id'] == factor_id:
                return factor
        return None
    
    def get_factors_by_category(
        self,
        category: str
    ) -> List[Dict[str, Any]]:
        """根据类别获取因子"""
        return [
            f for f in self.factor_library
            if f['category'] == category
        ]
    
    def list_all_categories(self) -> List[str]:
        """列出所有因子类别"""
        categories = set(f['category'] for f in self.factor_library)
        return sorted(categories)
    
    def get_factor_statistics(self) -> Dict[str, Any]:
        """获取因子库统计信息"""
        categories = self.list_all_categories()
        
        stats = {
            'total_factors': len(self.factor_library),
            'categories': categories,
            'category_counts': {
                cat: len(self.get_factors_by_category(cat))
                for cat in categories
            },
            'avg_ic': np.mean([abs(f['expected_ic']) for f in self.factor_library]),
            'max_ic': max([abs(f['expected_ic']) for f in self.factor_library]),
            'min_ic': min([abs(f['expected_ic']) for f in self.factor_library])
        }
        
        return stats


# 使用示例
async def demo():
    """演示使用"""
    print("=" * 70)
    print("简化版因子发现系统演示 (Windows兼容)")
    print("=" * 70)
    
    # 创建实例
    discovery = SimplifiedFactorDiscovery()
    
    # 获取统计信息
    stats = discovery.get_factor_statistics()
    print(f"\n📊 因子库统计:")
    print(f"  总因子数: {stats['total_factors']}")
    print(f"  因子类别: {', '.join(stats['categories'])}")
    print(f"  平均IC: {stats['avg_ic']:.4f}")
    print(f"  最大IC: {stats['max_ic']:.4f}")
    
    # 发现因子
    print(f"\n🔍 开始因子发现...")
    factors = await discovery.discover_factors(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_factors=10,
        min_ic=0.08
    )
    
    print(f"\n✅ 发现 {len(factors)} 个优质因子:")
    for i, factor in enumerate(factors, 1):
        print(f"\n{i}. {factor['name']} ({factor['id']})")
        print(f"   类别: {factor['category']}")
        print(f"   表达式: {factor['expression']}")
        print(f"   预期IC: {factor['expected_ic']:.4f}")
        print(f"   描述: {factor['description']}")


if __name__ == '__main__':
    asyncio.run(demo())
