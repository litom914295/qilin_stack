#!/usr/bin/env python
"""
测试 Dashboard 股票加载功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 模拟 Dashboard 的加载逻辑
import json
from typing import List

def get_top_limitup_stocks(top_n: int = 3) -> List[str]:
    """获取当日最强势的前 N 只涨停股"""
    try:
        # 从文件系统获取最近的筛选结果
        data_dir = Path(__file__).parent.parent / "data" / "daily_selections"
        if data_dir.exists():
            # 查找最近的筛选文件
            files = list(data_dir.glob("limitup_*.json"))
            if files:
                latest_file = max(files, key=lambda p: p.stat().st_mtime)
                print(f"✓ 找到数据文件: {latest_file.name}")
                
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'stocks' in data and len(data['stocks']) > 0:
                        # 按质量评分排序
                        stocks = sorted(
                            data['stocks'], 
                            key=lambda x: x.get('quality_score', 0), 
                            reverse=True
                        )
                        
                        print(f"\n✓ 筛选日期: {data.get('date')}")
                        print(f"✓ 总涨停数: {data.get('total_limitup')}")
                        print(f"✓ 合格数量: {data.get('qualified_count')}")
                        print(f"\n前 {top_n} 只最强势股票:")
                        print("="*60)
                        
                        result = []
                        for i, stock in enumerate(stocks[:top_n], 1):
                            symbol = stock['symbol']
                            result.append(symbol)
                            print(f"{i}. {symbol} - {stock.get('name', 'N/A')}")
                            print(f"   质量评分: {stock.get('quality_score', 0):.1f}")
                            print(f"   置信度: {stock.get('confidence', 0):.2%}")
                            print(f"   涨停时间: {stock.get('limit_up_time', 'N/A')}")
                            print(f"   开板次数: {stock.get('open_times', 0)}")
                        
                        print("="*60)
                        return result
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 默认返回示例数据
    print("\n⚠ 使用默认股票代码")
    return ["000001", "000002", "600000"]


if __name__ == '__main__':
    print("测试 Dashboard 涨停股加载功能\n")
    stocks = get_top_limitup_stocks(3)
    print(f"\n✓ 最终返回: {stocks}")
    print("\n测试完成！Dashboard 将显示这些股票。")
