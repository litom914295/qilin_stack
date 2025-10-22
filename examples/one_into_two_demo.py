import asyncio
from datetime import datetime
from typing import List

from decision_engine.core import get_decision_engine
from rd_agent.limit_up_data import LimitUpDataInterface


async def main(target_date: str = None, top_k: int = 10):
    date = target_date or datetime.now().strftime('%Y-%m-%d')

    # 1) 找出“首板”候选（昨日涨停）
    data_if = LimitUpDataInterface(data_source='qlib')
    limit_ups = data_if.get_limit_up_stocks(date)
    symbols: List[str] = [rec.symbol for rec in limit_ups]

    if not symbols:
        print(f"[{date}] 未发现涨停股票，无法进行一进二评估")
        return

    # 2) 提取特征并排序，选TopK
    feats = data_if.get_limit_up_features(symbols, date)
    if not feats.empty:
        symbols = list(feats.sort_values(['limit_up_strength','seal_quality','volume_surge'], ascending=False).head(top_k).index)

    print(f"[{date}] 一进二候选（Top-{top_k}）: {symbols}\n")

    # 3) 通过决策引擎融合 Qlib + TradingAgents + RD-Agent 三方信号
    engine = get_decision_engine()
    decisions = await engine.make_decisions(symbols, date)

    # 4) 打印结果
    for d in decisions:
        print(f"股票: {d.symbol}")
        print(f"  最终信号: {d.final_signal.value}")
        print(f"  置信度: {d.confidence:.2%}")
        print(f"  强度: {d.strength:.3f}")
        print(f"  说明: {d.reasoning[:160]}")
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='一进二候选评估（融合三方信号）')
    parser.add_argument('--date', help='交易日 YYYY-MM-DD，默认=今天', default=None)
    parser.add_argument('--topk', help='入选股票数量', type=int, default=10)
    args = parser.parse_args()

    asyncio.run(main(args.date, args.topk))
