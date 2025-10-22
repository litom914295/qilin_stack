
import yaml
from layer3_online.fuse import fuse, DEFAULT_WEIGHTS

def fused_score(agent_scores: dict, qlib_score: float, weights_path='integrations/tradingagents_cn/weights.yaml') -> float:
    try:
        w = yaml.safe_load(open(weights_path,'r',encoding='utf-8')).get('weights', DEFAULT_WEIGHTS)
    except Exception:
        w = DEFAULT_WEIGHTS
    base = fuse(agent_scores, w)
    return float(0.7*base + 0.3*(qlib_score if qlib_score is not None else 0.5))
