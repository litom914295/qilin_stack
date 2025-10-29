# -*- coding: utf-8 -*-
"""
麒麟量化系统 - 涨停原因解释器
解释每日涨停股票的原因，帮助理解模型预测逻辑
输出: reports/explanations/explainer_YYYY-MM-DD.csv 和汇总报告
"""
import os
import argparse
import json
import warnings
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 8大涨停原因规则
REASON_RULES = [
    ("强竞价", lambda r: r.get("vwap_slope", 0) >= 0.03),
    ("上午抗回撤", lambda r: r.get("max_drawdown", 0) >= -0.02),
    ("午后延续性", lambda r: r.get("afternoon_strength", 0) >= 0.01),
    ("题材热度高", lambda r: r.get("sector_heat", 0) >= 0.7),
    ("量能放大", lambda r: r.get("volume_ratio", 1.0) >= 1.8),
    ("封板迅速", lambda r: r.get("limit_up_hour", 14.0) < 11.0),
    ("封单强度高", lambda r: r.get("seal_ratio", 0) >= 0.08),
    ("龙头地位", lambda r: r.get("is_leader", 0) >= 0.5),
]


class LimitUpExplainer:
    """涨停原因解释器"""
    
    def __init__(self, output_dir: str = "reports/explanations"):
        """
        初始化解释器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.motifs_file = self.output_dir / "motifs.json"
        self.motifs = self._load_motifs()
        
        logger.info("涨停原因解释器初始化完成")
    
    def _load_motifs(self) -> Dict:
        """加载历史原因组合统计"""
        if self.motifs_file.exists():
            try:
                with open(self.motifs_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载motifs失败: {e}")
        return {"motifs": {}, "total_count": 0, "success_count": 0}
    
    def _save_motifs(self):
        """保存原因组合统计"""
        try:
            with open(self.motifs_file, "w", encoding="utf-8") as f:
                json.dump(self.motifs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存motifs失败: {e}")
    
    def score_reasons(self, stock_features: Dict) -> List[Tuple[str, float]]:
        """
        评估股票的涨停原因得分
        
        Args:
            stock_features: 股票特征字典(16维特征)
            
        Returns:
            [(原因名称, 得分)] 排序列表
        """
        scores = []
        
        for name, rule_fn in REASON_RULES:
            try:
                match = rule_fn(stock_features)
                score = 1.0 if match else 0.0
            except Exception as e:
                logger.debug(f"评估原因'{name}'失败: {e}")
                score = 0.0
            
            scores.append((name, score))
        
        # 加权排序(关键因素权重更高)
        weights = {
            "强竞价": 1.2,
            "上午抗回撤": 1.0,
            "午后延续性": 1.0,
            "题材热度高": 1.2,
            "量能放大": 1.0,
            "封板迅速": 1.1,
            "封单强度高": 1.0,
            "龙头地位": 1.1,
        }
        
        weighted_scores = [
            (name, score * weights.get(name, 1.0))
            for name, score in scores
        ]
        
        return sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    
    def explain_stock(self, symbol: str, features: Dict, actual_limit_up: bool = False) -> Dict:
        """
        解释单只股票的涨停原因
        
        Args:
            symbol: 股票代码
            features: 特征字典
            actual_limit_up: 是否实际涨停
            
        Returns:
            解释结果字典
        """
        reason_scores = self.score_reasons(features)
        
        # 取前5个原因
        top_reasons = [name for name, score in reason_scores if score > 0][:5]
        top_scores = [float(score) for name, score in reason_scores if score > 0][:5]
        
        result = {
            "symbol": symbol,
            "actual_limit_up": actual_limit_up,
            "reasons": top_reasons,
            "reason_scores": top_scores,
            "features": features
        }
        
        return result
    
    def explain_date(
        self,
        date: str,
        scored_df: pd.DataFrame,
        actual_limitup_symbols: List[str] = None
    ) -> pd.DataFrame:
        """
        解释某日的涨停候选
        
        Args:
            date: 日期 YYYY-MM-DD
            scored_df: 打分后的候选DataFrame(包含16维特征)
            actual_limitup_symbols: 实际涨停的股票代码列表
            
        Returns:
            解释结果DataFrame
        """
        logger.info(f"解释 {date} 的涨停原因...")
        
        if scored_df.empty:
            logger.warning(f"{date} 无候选数据")
            return pd.DataFrame()
        
        # 过滤当日数据
        day_df = scored_df[scored_df["date"] == date].copy()
        
        if day_df.empty:
            return pd.DataFrame()
        
        actual_limitup_symbols = actual_limitup_symbols or []
        
        # 逐只股票解释
        results = []
        for _, row in day_df.iterrows():
            symbol = row["symbol"]
            
            # 提取特征
            features = {
                "consecutive_days": row.get("consecutive_days", 1),
                "seal_ratio": row.get("seal_ratio", 0),
                "quality_score": row.get("quality_score", 0),
                "is_leader": row.get("is_leader", 0),
                "turnover_rate": row.get("turnover_rate", 0),
                "volume_ratio": row.get("volume_ratio", 1.0),
                "open_times": row.get("open_times", 0),
                "limit_up_hour": row.get("limit_up_hour", 14.0),
                "is_one_word": row.get("is_one_word", 0),
                "vwap_slope": row.get("vwap_slope", 0),
                "max_drawdown": row.get("max_drawdown", 0),
                "afternoon_strength": row.get("afternoon_strength", 0),
                "sector_heat": row.get("sector_heat", 0.5),
                "sector_count": row.get("sector_count", 0),
                "is_first_board": row.get("is_first_board", 1),
                "score": row.get("score", 0),
            }
            
            actual_limit_up = (symbol in actual_limitup_symbols)
            
            result = self.explain_stock(symbol, features, actual_limit_up)
            
            results.append({
                "date": date,
                "symbol": symbol,
                "actual_limit_up": actual_limit_up,
                "score": features["score"],
                "reasons": json.dumps(result["reasons"], ensure_ascii=False),
                "reason_scores": json.dumps(result["reason_scores"], ensure_ascii=False),
                **{k: v for k, v in features.items() if k != "score"}
            })
        
        result_df = pd.DataFrame(results)
        
        # 更新motifs统计
        self._update_motifs(result_df)
        
        # 保存
        out_path = self.output_dir / f"explainer_{date}.csv"
        result_df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"解释结果已保存: {out_path}")
        
        return result_df
    
    def _update_motifs(self, result_df: pd.DataFrame):
        """更新原因组合统计"""
        for _, row in result_df.iterrows():
            try:
                reasons = json.loads(row["reasons"]) if isinstance(row["reasons"], str) else []
                key = "|".join(reasons[:3]) if reasons else "<NONE>"
                
                if key not in self.motifs["motifs"]:
                    self.motifs["motifs"][key] = {"count": 0, "success_count": 0}
                
                self.motifs["motifs"][key]["count"] += 1
                
                if row.get("actual_limit_up", False):
                    self.motifs["motifs"][key]["success_count"] += 1
                
                self.motifs["total_count"] += 1
                if row.get("actual_limit_up", False):
                    self.motifs["success_count"] += 1
            except Exception as e:
                logger.debug(f"更新motif失败: {e}")
                continue
        
        self._save_motifs()
    
    def generate_summary(self, start_date: str, end_date: str) -> str:
        """
        生成区间汇总报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Markdown格式汇总报告
        """
        md_lines = [
            "# 涨停原因解释汇总报告",
            f"",
            f"**时间区间**: {start_date} ~ {end_date}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 整体统计",
            f"",
            f"- 总候选数: {self.motifs.get('total_count', 0)}",
            f"- 实际涨停数: {self.motifs.get('success_count', 0)}",
            f"- 成功率: {self.motifs.get('success_count', 0) / max(self.motifs.get('total_count', 1), 1):.2%}",
            f"",
            f"## 高频原因组合(Top 10)",
            f""
        ]
        
        # 按出现频率排序
        motifs = self.motifs.get("motifs", {})
        sorted_motifs = sorted(
            motifs.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        for i, (key, stats) in enumerate(sorted_motifs[:10], 1):
            success_rate = stats["success_count"] / max(stats["count"], 1)
            md_lines.append(
                f"{i}. **{key}**  "
                f"出现次数: {stats['count']}, "
                f"成功率: {success_rate:.1%}"
            )
        
        md_lines.append("")
        md_lines.append("## 单一原因统计")
        md_lines.append("")
        
        # 统计单一原因
        reason_stats = {}
        for motif_key, stats in motifs.items():
            reasons = motif_key.split("|")
            for reason in reasons:
                if reason not in reason_stats:
                    reason_stats[reason] = {"count": 0, "success_count": 0}
                reason_stats[reason]["count"] += stats["count"]
                reason_stats[reason]["success_count"] += stats["success_count"]
        
        sorted_reasons = sorted(
            reason_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        for reason, stats in sorted_reasons:
            success_rate = stats["success_count"] / max(stats["count"], 1)
            md_lines.append(
                f"- **{reason}**: 出现{stats['count']}次, "
                f"成功率{success_rate:.1%}"
            )
        
        return "\n".join(md_lines)


def main():
    ap = argparse.ArgumentParser(description="涨停原因解释器")
    ap.add_argument("--date", help="单日解释 YYYY-MM-DD")
    ap.add_argument("--start", help="区间开始 YYYY-MM-DD")
    ap.add_argument("--end", help="区间结束 YYYY-MM-DD")
    ap.add_argument("--scored_csv", required=True, help="打分结果CSV路径")
    ap.add_argument("--actual_limitup_csv", help="实际涨停股票CSV路径(可选)")
    args = ap.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    
    if not args.date and not (args.start and args.end):
        raise SystemExit("请提供 --date 或 (--start 和 --end)")
    
    # 加载打分结果
    scored_df = pd.read_csv(args.scored_csv, parse_dates=["date"])
    scored_df["date"] = scored_df["date"].dt.strftime("%Y-%m-%d")
    
    # 加载实际涨停(可选)
    actual_limitup_symbols = []
    if args.actual_limitup_csv and os.path.exists(args.actual_limitup_csv):
        actual_df = pd.read_csv(args.actual_limitup_csv)
        actual_limitup_symbols = actual_df["symbol"].tolist()
    
    explainer = LimitUpExplainer()
    
    if args.date:
        # 单日解释
        result_df = explainer.explain_date(args.date, scored_df, actual_limitup_symbols)
        print(f"\n[OK] {args.date} 解释完成: {len(result_df)} 只股票")
    else:
        # 区间解释
        dates = pd.date_range(args.start, args.end, freq="D")
        
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            try:
                result_df = explainer.explain_date(date_str, scored_df, actual_limitup_symbols)
                if not result_df.empty:
                    print(f"[OK] {date_str}: {len(result_df)} 只股票")
            except Exception as e:
                logger.error(f"{date_str} 解释失败: {e}")
                continue
        
        # 生成汇总报告
        summary = explainer.generate_summary(args.start, args.end)
        summary_path = explainer.output_dir / f"explainer_{args.start}_{args.end}.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"\n[OK] 汇总报告已保存: {summary_path}")


if __name__ == "__main__":
    main()
