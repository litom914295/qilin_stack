
import os
from datetime import datetime
from layer3_online.fuse import fuse, DEFAULT_WEIGHTS

def _marginal_contrib(scores: dict, qlib_score: float, weights: dict) -> dict:
    base = 0.7 * fuse(scores, weights) + 0.3 * (qlib_score if qlib_score is not None else 0.5)
    contrib = {}
    for k in scores.keys():
        tmp = scores.copy(); tmp[k] = 0.5
        alt = 0.7 * fuse(tmp, weights) + 0.3 * (qlib_score if qlib_score is not None else 0.5)
        contrib[k] = base - alt
    return contrib

def write_markdown_report(symbol, scores, qlib_score, final_score, out_dir, weights=None):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{symbol}.md")
    weights = weights or DEFAULT_WEIGHTS
    lines = [f"# {symbol} — 一进二盘前报告", "", f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", f"- 融合得分：**{final_score:.3f}**；模型分：{(qlib_score if qlib_score is not None else 0.5):.3f}", "", "## 分项打分"]
    for k,v in scores.items(): lines.append(f"- **{k}**: {v:.3f}")
    lines.append(""); lines.append("## Agent 贡献度归因（边际贡献）")
    for k,v in sorted(_marginal_contrib(scores, qlib_score, weights).items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {k}: {v:+.4f}")
    open(p,"w",encoding="utf-8").write("\n".join(lines))
    return p
