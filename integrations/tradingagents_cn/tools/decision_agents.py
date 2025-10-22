
from layer3_online.agents import (ZTQualityAgent, LeaderAgent, LHBSeatAgent, NewsAgent, ChipAgent, ChanAgent, ElliottAgent, FibAgent, MarketMoodAgent, RiskGuardAgent, AgentResult)
from layer3_online.adapters.selector import get_ohlc
import pandas as pd

AGENTS = [ZTQualityAgent(), LeaderAgent(), LHBSeatAgent(), NewsAgent(), ChipAgent(), ChanAgent(), ElliottAgent(), FibAgent(), MarketMoodAgent(), RiskGuardAgent()]

def run_agents(symbol: str) -> dict:
    ohlc = get_ohlc(symbol, start="2024-01-01")
    ctx = {'ohlc': ohlc, 'news_titles': [], 'lhb_netbuy': 0.0, 'last_price': float(ohlc['close'].iloc[-1]) if len(ohlc)>0 else 10.0}
    out = {}
    for ag in AGENTS:
        r: AgentResult = ag.run(symbol, ctx)
        out[ag.name] = r.score
    return out
