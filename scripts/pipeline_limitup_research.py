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

# Import premium data provider and advanced factors
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_layer.premium_data_provider import PremiumDataProvider
from factors.limitup_advanced_factors import LimitUpAdvancedFactors

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
            # optionally disable proxies (default: respect proxies)
            import os as _os
            if _os.getenv("AKSHARE_DISABLE_PROXY", "0").lower() in ("1", "true", "yes"):
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


def generate_offline_panel(universe: List[str], start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Generate an offline synthetic daily OHLCV panel when online data is unavailable.
    Index: [date, symbol]; Columns: open, high, low, close, volume, amount, turnover
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="B")
    if len(dates) == 0:
        return pd.DataFrame()
    frames = []
    for i, sym in enumerate(universe):
        base = float(rng.uniform(5, 80))
        ret = rng.normal(loc=0.0005, scale=0.02, size=len(dates))
        close = base * np.exp(np.cumsum(ret))
        noise = rng.normal(0, 0.003, size=len(dates))
        open_ = close * (1 + noise)
        high = np.maximum(open_, close) * (1 + np.abs(noise) * 3 + 0.002)
        low = np.minimum(open_, close) * (1 - np.abs(noise) * 3 - 0.002)
        low = np.clip(low, a_min=0.01, a_max=None)
        volume = rng.lognormal(mean=12, sigma=0.7, size=len(dates))  # ~e12 scale
        amount = close * volume / 100  # 粗略金额
        turnover = rng.uniform(0.1, 3.0, size=len(dates))
        df = pd.DataFrame(
            {
                "open": open_.astype(float),
                "high": high.astype(float),
                "low": low.astype(float),
                "close": close.astype(float),
                "volume": volume.astype(float),
                "amount": amount.astype(float),
                "turnover": turnover.astype(float),
                "symbol": sym,
                "date": dates.date,
            }
        )
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.set_index(["date", "symbol"]).sort_index()
    return panel


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
        # optionally disable proxies (default: respect proxies)
        import os as _os
        if _os.getenv("AKSHARE_DISABLE_PROXY", "0").lower() in ("1", "true", "yes"):
            for _k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
                _os.environ.pop(_k, None)
            _os.environ["NO_PROXY"] = "*"
        rows = []
        # cap to 300 for speed when using AkShare
        universe_ak = universe[:300] if len(universe) > 300 else universe
        # throttle to avoid anti-scraping
        import time as _time
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
            if (i + 1) % 50 == 0:
                _time.sleep(0.5)
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
        print(f"[WARN] {msg}")
        print("[WARN] Falling back to offline synthetic panel (no internet / qlib data).")
        panel = generate_offline_panel(universe, start, end)
        if panel.empty:
            raise RuntimeError(msg)
        print(f"[INFO] Offline synthetic panel generated: {panel.shape}")
        return panel
    panel = frames[0]
    # basic cleaning
    panel = panel.sort_index()
    return panel


def engineer_features(panel: pd.DataFrame, premium_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    特征工程 - 使用高级因子库
    
    Args:
        panel: 基础日线数据
        premium_data: 高级数据（流通市值、封单等）
    
    Returns:
        包含高级因子的DataFrame
    """
    # panel indexed by [date, symbol]
    df = panel.copy()
    
    # 准备数据格式，符合LimitUpAdvancedFactors的要求
    df_reset = df.reset_index()
    df_reset.columns = ['date', 'symbol'] + list(df.columns)
    
    # 添加必要的涨停相关字段
    df_reset['is_limitup'] = (df_reset['close'].pct_change(1) >= 0.095).astype(int)
    
    # 如果有高级数据，合并进来
    if premium_data is not None and not premium_data.empty:
        # 重置索引以便合并
        premium_reset = premium_data.reset_index()
        premium_reset['date'] = pd.to_datetime(df_reset['date'].iloc[0]).date()  # 使用相同日期
        
        # 合并高级数据
        df_reset = df_reset.merge(
            premium_reset[['symbol', 'circulating_market_cap', 'seal_amount', 'sector', 'themes', 
                          'turnover_rate', 'limit_up_time', 'open_times', 'consecutive_days']],
            on='symbol',
            how='left',
            suffixes=('', '_premium')
        )
        
        # 填充缺失值
        df_reset['float_mv'] = df_reset.get('circulating_market_cap', 100) * 1e8  # 转换回元
        df_reset['buy_amount'] = df_reset.get('seal_amount', 0) * 1e4  # 转换回元
        df_reset['limitup_time'] = df_reset.get('limit_up_time', '14:30:00')
        df_reset['industry'] = df_reset.get('sector', '未分类')
        df_reset['theme'] = df_reset.get('themes', '').apply(lambda x: x[0] if isinstance(x, list) and x else '未知')
    else:
        # 生成模拟的高级数据
        df_reset['float_mv'] = np.random.uniform(1e9, 1e11, len(df_reset))
        df_reset['buy_amount'] = np.random.uniform(1e6, 1e8, len(df_reset))
        df_reset['limitup_time'] = '14:30:00'
        df_reset['industry'] = np.random.choice(['科技', '医药', '消费', '金融'], len(df_reset))
        df_reset['theme'] = np.random.choice(['AI', '新能源', '半导体', '医疗', '消费'], len(df_reset))
        df_reset['open_count'] = np.random.randint(0, 3, len(df_reset))
    
    # 添加更多必要字段
    df_reset['sell_amount'] = df_reset.get('sell_amount', np.random.uniform(1e5, 1e7, len(df_reset)))
    df_reset['big_buy_volume'] = df_reset.get('big_buy_volume', df_reset['volume'] * 0.3)
    df_reset['total_buy_volume'] = df_reset.get('total_buy_volume', df_reset['volume'] * 0.5)
    df_reset['turnover'] = df_reset.get('turnover_rate', np.random.uniform(5, 30, len(df_reset)))
    
    # 使用高级因子计算器
    calculator = LimitUpAdvancedFactors()
    df_with_factors = calculator.calculate_all_factors(df_reset)
    
    # 确保包含 limit_up 列（用于标签生成）
    if 'limit_up' not in df_with_factors.columns and 'is_limitup' in df_with_factors.columns:
        df_with_factors['limit_up'] = df_with_factors['is_limitup']
    elif 'limit_up' not in df_with_factors.columns:
        # 如果两者都不存在，计算涨停标志
        if 'close' in df_with_factors.columns:
            df_with_factors_sorted = df_with_factors.sort_values(['symbol', 'date'])
            df_with_factors['limit_up'] = (
                df_with_factors_sorted.groupby('symbol')['close']
                .pct_change()
                .fillna(0)
                .apply(lambda x: 1 if x >= 0.095 else 0)
                .values
            )
        else:
            # 最终兜底：使用 is_limitup 或设为0
            df_with_factors['limit_up'] = df_with_factors.get('is_limitup', 0)
    
    # 设置索引回原始格式
    df_with_factors = df_with_factors.set_index(['date', 'symbol'])
    
    return df_with_factors


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

    # 基本校验：样本量与标签
    n_rows = len(samples)
    if n_rows < 10:
        raise ValueError(f"样本量过少({n_rows})，无法训练。请扩大日期范围或增加股票池（建议≥10条样本且包含两类标签）。")
    if samples["y"].nunique() < 2:
        raise ValueError("标签只有单一类别，无法训练。请扩大日期范围或调整筛选条件以覆盖正负样本。")

    # Features (exclude labels and raw OHLC to avoid leakage except allowed ones)
    drop_cols = {"date", "symbol", "y", "next_limit_up", "limit_up"}
    X_cols = [c for c in samples.columns if c not in drop_cols]
    X = samples[X_cols].values
    y = samples["y"].values

    # Time-based split: last 20% as test，且保证训练/测试集均非空
    cut = int(n_rows * 0.8)
    cut = max(1, min(n_rows - 1, cut))
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
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=300)
        y_pred = model.predict(X_test) if len(X_test) else model.predict(X_train)
    else:
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1] if len(X_test) else model.predict_proba(X_train)[:, 1]

    # 安全评估：若测试集单一类别或为空，则返回NaN并跳过严格评估
    try:
        if len(y_test) == 0 or pd.Series(y_test).nunique() < 2:
            auc, ap = float("nan"), float("nan")
        else:
            auc = float(roc_auc_score(y_test, y_pred))
            ap = float(average_precision_score(y_test, y_pred))
    except Exception:
        auc, ap = float("nan"), float("nan")
    print(f"[INFO] Test AUC={auc if not math.isnan(auc) else float('nan'):.4f} AP={ap if not math.isnan(ap) else float('nan'):.4f}")

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
    
    # Track data source status for final report
    data_source_info = {
        "qlib_available": HAS_QLIB,
        "akshare_available": HAS_AK,
        "data_source_used": "unknown",
        "is_synthetic": False,
        "network_issues": []
    }
    
    try:
        panel = fetch_panel(universe, start, end)
        # Check if we got synthetic data (would be indicated in the panel generation)
        if hasattr(panel, 'index') and len(panel) > 0:
            # If all symbols have exactly same date range with perfect regularity, likely synthetic
            date_counts = panel.groupby('symbol').size()
            if date_counts.nunique() == 1 and date_counts.iloc[0] > 50:  # Heuristic for synthetic
                data_source_info["is_synthetic"] = True
                data_source_info["data_source_used"] = "synthetic"
            else:
                data_source_info["data_source_used"] = "qlib" if HAS_QLIB else "akshare"
    except Exception as e:
        if "Connection aborted" in str(e) or "Remote end closed" in str(e):
            data_source_info["network_issues"].append("AkShare网络连接被拦截或中断")
        panel = fetch_panel(universe, start, end)  # This will use fallback
        data_source_info["is_synthetic"] = True
        data_source_info["data_source_used"] = "synthetic"
    
    # 获取高级数据（如果可用）
    premium_data = None
    try:
        # 使用最后一个交易日作为示例获取高级数据
        last_date = str(panel.index.get_level_values('date').max())
        provider = PremiumDataProvider(use_cache=True)
        premium_data = provider.get_daily_advanced_metrics(last_date)
        
        # 同时获取市场情绪
        sentiment = provider.get_market_sentiment(last_date)
        print(f"[INFO] Market sentiment score: {sentiment['sentiment_score']:.1f}")
        print(f"[INFO] Limit up count: {sentiment['limit_up_count']}")
    except Exception as e:
        print(f"[WARN] Failed to get premium data: {e}")
    
    feat = engineer_features(panel, premium_data)
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
            "data_source_info": data_source_info
        }
        summary_path = OUT_DIR / f"training_summary_{start}_{end}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Training summary saved: {summary_path}")
    except Exception as e:
        print(f"[WARN] Failed to save training summary: {e}")
    
    # Print data source explanation
    print("\n" + "="*60)
    print("📊 数据源使用说明")
    print("="*60)
    if data_source_info["is_synthetic"]:
        print("⚠️  使用了合成数据进行训练")
        print("📝 原因分析:")
        if not HAS_QLIB:
            print("   • Qlib未安装或初始化失败")
        elif not HAS_AK:
            print("   • AkShare未安装")
        elif data_source_info["network_issues"]:
            for issue in data_source_info["network_issues"]:
                print(f"   • {issue}")
            print("\n💡 AkShare网络拦截常见原因:")
            print("   1. 企业网络防火墙限制外部API访问")
            print("   2. 代理服务器阻止金融数据请求")
            print("   3. ISP对频繁数据访问的限制")
            print("   4. 目标服务器临时不可用")
            print("\n🔧 解决建议:")
            print("   • 使用Qlib本地数据（推荐）")
            print("   • 配置网络代理或VPN")
            print("   • 联系网络管理员开放相关域名")
        else:
            print("   • 实时数据源暂不可用，已自动回退")
    else:
        print(f"✅ 使用真实数据源: {data_source_info['data_source_used'].upper()}")
    print("="*60)

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
