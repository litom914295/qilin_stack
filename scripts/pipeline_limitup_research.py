#!/usr/bin/env python
"""
Pipeline: A-share limit-up (一进二) research workflow
- Build historical limit-up attribution dataset
- Train classifier (T+1 连板/继续强势 标签)
- Explain model (SHAP / permutation)
- Map feature importances back to agent weights and optionally write suggestions

Usage (PowerShell):
  .\.qilin\Scripts\Activate.ps1
  python scripts\pipeline_limitup_research.py --start 2024-01-01 --end 2024-12-31 --provider-uri "G:/test/qlib/qlib_data/cn_data" --apply-weights

Notes:
- Prefers Qlib day-level data; falls back to AkShare for limited features if Qlib not available
- Outputs under output/limitup_research/
"""
from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Optional libs (used if present)
try:
    import qlib  # type: ignore
    from qlib.data import D  # type: ignore
    HAS_QLIB = True
except Exception:
    HAS_QLIB = False

try:
    import akshare as ak  # type: ignore
    HAS_AK = True
except Exception:
    HAS_AK = False

# ML stack
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib

# Paths
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "limitup_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CFG_FILE = ROOT / "config" / "tradingagents.yaml"

# -----------------------------
# Helpers
# -----------------------------

def init_qlib(provider_uri: Optional[str]) -> None:
    if not HAS_QLIB:
        print("[WARN] Qlib not available; will fallback to AkShare where possible")
        return
    uri = provider_uri or os.getenv("QLIB_PROVIDER_URI") or "G:/test/qlib/qlib_data/cn_data"
    try:
        qlib.init(provider_uri=uri, region="cn")
        print(f"[INFO] Qlib initialized at {uri}")
    except Exception as e:
        print(f"[WARN] Qlib init failed: {e}")


def list_instruments(limit: Optional[int] = None) -> List[str]:
    codes: List[str] = []
    if HAS_QLIB:
        try:
            inst = D.list_instruments(market="all")  # type: ignore
            codes = list(inst)
        except Exception:
            codes = []
    if not codes and HAS_AK:
        try:
            # disable proxies to avoid interception
            import os as _os
            for _k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
                _os.environ.pop(_k, None)
            _os.environ["NO_PROXY"] = "*"
            df = ak.stock_zh_a_spot_em()
            # map to Qlib-style codes: SH/SZ prefix uppercase
            raw_codes = df["代码"].astype(str).tolist() if isinstance(df, pd.DataFrame) and "代码" in df.columns else []
            codes = [
                ("SH" + c) if str(c).startswith("6") else ("SZ" + c) if str(c).startswith(("0", "3")) else None
                for c in raw_codes
            ]
            codes = [c for c in codes if c]
        except Exception:
            codes = []
    # Final fallback: small static universe to ensure pipeline can run
    if not codes:
        codes = ["SZ000001", "SH600519", "SZ000858", "SH600000"]
    if limit and len(codes) > limit:
        codes = codes[:limit]
    print(f"[INFO] Universe size: {len(codes)}")
    return codes


def fetch_panel(universe: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily panel with minimal fields.
    Returns MultiIndex: [date, symbol] with columns [open, high, low, close, volume, amount, turnover?]
    """
    def _to_ak_code(sym: str) -> str:
        if not isinstance(sym, str):
            return ""
        s = sym.strip()
        # SH600519 / SZ000001
        if len(s) >= 8 and (s[:2].upper() in ("SH", "SZ")):
            return s[2:]
        # 600519.SH / 000001.SZ
        if "." in s and len(s.split(".")) == 2:
            return s.split(".")[0]
        # sh600519 / sz000001
        if len(s) >= 8 and (s[:2].lower() in ("sh", "sz")):
            return s[2:]
        # pure 6-digit
        if s.isdigit() and len(s) == 6:
            return s
        return s

    frames = []
    qlib_error = None
    ak_errors = []
    ak_success = 0

    if HAS_QLIB:
        fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$turnover"]
        try:
            df = D.features(universe, fields, start_time=start, end_time=end, freq="day")  # type: ignore
            # df index: DatetimeIndex with instrument level; columns as fields
            # Normalize columns
            df.columns = [c.replace("$", "") for c in df.columns]
            # Ensure level names
            df.index.names = ["date", "symbol"] if len(df.index.names) == 2 else df.index.names
            frames.append(df)
        except Exception as e:
            qlib_error = str(e)
            print(f"[WARN] Qlib fetch failed: {e}")
    # Fallback: AkShare (slow for large universe)
    if not frames and HAS_AK:
        # disable proxies to avoid interception
        import os as _os
        for _k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
            _os.environ.pop(_k, None)
        _os.environ["NO_PROXY"] = "*"
        rows = []
        # cap to 300 for speed when using AkShare
        universe_ak = universe[:300] if len(universe) > 300 else universe
        for i, sym in enumerate(universe_ak):
            code = _to_ak_code(sym)
            if not code:
                continue
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq")
                if not isinstance(df, pd.DataFrame) or df.empty or "日期" not in df.columns:
                    ak_errors.append(f"{sym}: empty or bad columns")
                    continue
                df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
                df["symbol"] = sym
                df["date"] = pd.to_datetime(df["date"]).dt.date
                rows.append(df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]])
                ak_success += 1
            except Exception as e:
                if len(ak_errors) < 5:
                    ak_errors.append(f"{sym}: {e}")
                continue
            if (i + 1) % 100 == 0:
                print(f"  [AK] fetched {i+1}/{len(universe_ak)}")
        if rows:
            df2 = pd.concat(rows, ignore_index=True)
            df2 = df2.set_index(["date", "symbol"]).sort_index()
            frames.append(df2)

    if not frames:
        msg = (
            "No data source available for panel | "
            f"HAS_QLIB={HAS_QLIB}, HAS_AK={HAS_AK}, "
            f"qlib_error={qlib_error}, ak_success={ak_success}, "
            f"ak_errors_sample={ak_errors[:3]}"
        )
        raise RuntimeError(msg)
    panel = frames[0]
    # basic cleaning
    panel = panel.sort_index()
    return panel


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    # panel indexed by [date, symbol]
    df = panel.copy()
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    # group by symbol for rolling
    def _by_sym(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["ret_1"] = g["close"].pct_change(1)
        g["ret_5"] = g["close"].pct_change(5)
        g["vol_ratio_5"] = g["volume"] / (g["volume"].rolling(5).mean())
        g["vol_ratio_20"] = g["volume"] / (g["volume"].rolling(20).mean())
        g["ma5"] = g["close"].rolling(5).mean()
        g["ma10"] = g["close"].rolling(10).mean()
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma5_cross_up"] = ((g["ma5"] > g["ma10"]) & (g["ma10"] > g["ma20"])) * 1.0
        g["amp"] = (g["high"] - g["low"]) / g["low"].replace(0, np.nan)
        g["gap_open"] = (g["open"] / g["close"].shift(1) - 1.0)
        # limit-up detection (approx: >= 9.5%)
        g["limit_up"] = (g["ret_1"] >= 0.095).astype(int)
        # consecutive limit-up count (rolling)
        g["cont_limit"] = g["limit_up"] * (1 + g["limit_up"].shift(1).fillna(0))
        # proximity to high (strong close)
        g["close_to_high"] = (g["high"] - g["close"]) / g["high"].replace(0, np.nan)
        # turnover if present
        if "turnover" in g.columns:
            g["turnover"] = pd.to_numeric(g["turnover"], errors="coerce")
        else:
            g["turnover"] = np.nan
        return g
    df = df.groupby(level=1, group_keys=False).apply(_by_sym)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_1", "ret_5", "vol_ratio_5", "ma5", "ma10"]).fillna(0)
    return df


def build_labeled_samples(feat: pd.DataFrame) -> pd.DataFrame:
    # Label: next day limit-up (T+1) given Today was limit-up (T)
    df = feat.copy()
    def _label(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["next_limit_up"] = g["limit_up"].shift(-1)
        return g
    df = df.groupby(level=1, group_keys=False).apply(_label)
    # Only keep rows where today was limit-up (T)
    df = df[df["limit_up"] == 1]
    # Drop last day per symbol without next label
    df = df.dropna(subset=["next_limit_up"])
    df["y"] = (df["next_limit_up"] > 0).astype(int)
    return df


@dataclass
class TrainResult:
    model_path: Path
    features: List[str]
    auc: float
    ap: float
    shap_path: Optional[Path]
    perm_path: Optional[Path]


def train_and_explain(samples: pd.DataFrame) -> TrainResult:
    # Train-test split by time
    samples = samples.reset_index().rename(columns={"level_0": "date", "level_1": "symbol"}) if "level_0" in samples.columns else samples.reset_index()
    samples["date"] = pd.to_datetime(samples["date"])  # ensure datetime
    samples = samples.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Features (exclude labels and raw OHLC to avoid leakage except allowed ones)
    drop_cols = {"date", "symbol", "y", "next_limit_up", "limit_up"}
    X_cols = [c for c in samples.columns if c not in drop_cols]
    X = samples[X_cols].values
    y = samples["y"].values

    # Time-based split: last 20% as test
    cut = int(len(samples) * 0.8)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    if HAS_LGB:
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "binary",
            "metric": ["auc"],
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=500)
        y_pred = model.predict(X_test)
    else:
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, y_pred))
    ap = float(average_precision_score(y_test, y_pred))
    print(f"[INFO] Test AUC={auc:.4f} AP={ap:.4f}")

    # Save model
    model_path = OUT_DIR / f"limitup_model_{int(time.time())}.pkl"
    joblib.dump({"model": model, "features": X_cols}, model_path)

    # Explain
    shap_path = None
    perm_path = None
    try:
        import shap  # type: ignore
        if HAS_LGB and hasattr(model, "predict"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test)[1] if isinstance(explainer.shap_values(X_test), list) else explainer.shap_values(X_test)
            shap_vals = np.abs(sv).mean(axis=0)
            imp = pd.DataFrame({"feature": X_cols, "mean_abs_shap": shap_vals}).sort_values("mean_abs_shap", ascending=False)
            shap_path = OUT_DIR / "feature_importance_shap.csv"
            imp.to_csv(shap_path, index=False)
            print(f"[INFO] SHAP importance saved: {shap_path}")
    except Exception as e:
        print(f"[WARN] SHAP not available or failed: {e}")

    # Permutation importance (model-agnostic)
    try:
        res = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        imp2 = pd.DataFrame({"feature": X_cols, "perm_importance": res.importances_mean}).sort_values("perm_importance", ascending=False)
        perm_path = OUT_DIR / "feature_importance_permutation.csv"
        imp2.to_csv(perm_path, index=False)
        print(f"[INFO] Permutation importance saved: {perm_path}")
    except Exception as e:
        print(f"[WARN] Permutation importance failed: {e}")

    return TrainResult(model_path=model_path, features=X_cols, auc=auc, ap=ap, shap_path=shap_path, perm_path=perm_path)


AGENT_FEATURE_MAP = {
    # feature keyword -> agent key
    "close_to_high": "seal_quality",
    "vol_ratio": "volume_surge",
    "cont_limit": "board_continuity",
    "ret_5": "qlib_momentum",
    "ma5_cross_up": "technical_analyst",
    "amp": "pattern",  # 波动/形态 proxy
    "turnover": "risk",  # 流动性/交易拥挤 proxy
    "gap_open": "market_analyst",
}


def suggest_agent_weights(importance: pd.DataFrame) -> Dict[str, float]:
    # Normalize by agent buckets
    buckets: Dict[str, float] = {}
    for _, row in importance.iterrows():
        feat = str(row["feature"]).lower()
        val = float(row.iloc[1])  # second column is importance
        matched = False
        for key, agent in AGENT_FEATURE_MAP.items():
            if key in feat:
                buckets[agent] = buckets.get(agent, 0.0) + max(val, 0.0)
                matched = True
                break
        if not matched:
            buckets["technical_analyst"] = buckets.get("technical_analyst", 0.0) + max(val, 0.0)
    total = sum(buckets.values()) or 1.0
    weights = {k: v / total for k, v in buckets.items()}
    return weights


def write_weight_suggestions(weights: Dict[str, float]) -> Path:
    out = OUT_DIR / "agent_weight_suggestions.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"agent_weights": weights, "ts": int(time.time())}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Weight suggestions saved: {out}")
    return out


def apply_weights_to_yaml(weights: Dict[str, float]) -> Optional[Path]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("[WARN] pyyaml not installed; skip applying to YAML")
        return None
    if not CFG_FILE.exists():
        print(f"[WARN] Config YAML not found: {CFG_FILE}")
        return None
    try:
        cfg = yaml.safe_load(CFG_FILE.read_text(encoding="utf-8")) or {}
        cfg.setdefault("tradingagents", {})
        cfg["tradingagents"]["agent_weights"] = weights
        CFG_FILE.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"[INFO] Applied agent weights into {CFG_FILE}")
        return CFG_FILE
    except Exception as e:
        print(f"[WARN] Failed to write YAML: {e}")
        return None


def run_pipeline(start: str, end: str, provider_uri: Optional[str], apply: bool) -> None:
    init_qlib(provider_uri)
    universe = list_instruments()
    # For quick demo, cap universe size (tune/remove for full run)
    if len(universe) > 2000:
        universe = universe[:2000]
    panel = fetch_panel(universe, start, end)
    feat = engineer_features(panel)
    samples = build_labeled_samples(feat)
    if samples.empty:
        print("[WARN] No samples generated. Check date range and data availability.")
        return
    # Persist dataset
    ds_path = OUT_DIR / f"limitup_samples_{start}_{end}.parquet"
    samples.to_parquet(ds_path)
    print(f"[INFO] Samples saved: {ds_path} (rows={len(samples)})")

    # Train + explain
    res = train_and_explain(samples)
    # Choose best importance file
    imp_path = res.shap_path or res.perm_path
    if not imp_path or not imp_path.exists():
        print("[WARN] No importance file produced; skip weight suggestion")
        return
    imp = pd.read_csv(imp_path)
    # Normalize second column name
    if imp.columns[1] != "importance":
        imp = imp.rename(columns={imp.columns[1]: "importance"})
    weights = suggest_agent_weights(imp[["feature", "importance"]])
    write_weight_suggestions(weights)

    # Write training summary for UI consumption
    try:
        summary = {
            "start": start,
            "end": end,
            "auc": res.auc,
            "ap": res.ap,
            "model_path": str(res.model_path),
            "importance_path": str(imp_path),
            "samples_path": str(ds_path),
            "weights": weights,
            "timestamp": int(time.time()),
        }
        summary_path = OUT_DIR / f"training_summary_{start}_{end}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Training summary saved: {summary_path}")
    except Exception as e:
        print(f"[WARN] Failed to save training summary: {e}")

    if apply:
        apply_weights_to_yaml(weights)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--provider-uri", default=None)
    p.add_argument("--apply-weights", action="store_true")
    args = p.parse_args()
    run_pipeline(args.start, args.end, args.provider_uri, args.apply_weights)

if __name__ == "__main__":
    main()
