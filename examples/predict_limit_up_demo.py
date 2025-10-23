# -*- coding: utf-8 -*-
"""
examples/predict_limit_up_demo.py

English alias of the beginner-friendly demo to predict next-day limit-up candidates.
Safe to run out-of-the-box using simulated data and probabilities.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_setup import setup_logging
setup_logging()

SEED = 42
TOPK = 10
STRONG_FOCUS_TH = 0.80
FOCUS_TH = 0.70

stock_list = [
    "000001.SZ", "000002.SZ", "000333.SZ", "000651.SZ", "000768.SZ",
    "002415.SZ", "300750.SZ", "600036.SH", "600519.SH", "601318.SH",
]

np.random.seed(SEED)
N_FEATURES = 50
X_demo = pd.DataFrame(
    np.random.randn(len(stock_list), N_FEATURES),
    index=stock_list,
    columns=[f"feature_{i}" for i in range(N_FEATURES)],
)

# Simulated probabilities for a quick run
proba_demo = np.random.uniform(0.60, 0.95, size=len(stock_list))

results = (
    pd.DataFrame({
        "stock": stock_list,
        "limit_up_prob": proba_demo,
    })
    .sort_values("limit_up_prob", ascending=False)
    .head(TOPK)
    .reset_index(drop=True)
)

labels, advices = [], []
for p in results["limit_up_prob"].values:
    if p >= STRONG_FOCUS_TH:
        labels.append("✅ Likely Limit-Up")
        advices.append("Strong focus")
    elif p >= FOCUS_TH:
        labels.append("✅ Some Chance")
        advices.append("Watch")
    else:
        labels.append("❌ Low Chance")
        advices.append("Wait")

results["label"], results["advice"] = labels, advices

logger.info("=" * 60)
logger.info("📊 Next-day Limit-Up Prediction (Demo)")
logger.info("=" * 60)
for i, row in results.iterrows():
    stock = row["stock"]
    prob = row["limit_up_prob"]
    label = row["label"]
    advice = row["advice"]
    logger.info(f"{i+1:02d}. {stock}")
    logger.info(f"Prediction: {label}")
    logger.info(f"Probability: {prob:.1%}")
    logger.info(f"Advice: {advice}")

logger.info("=" * 60)
logger.info("✅ Demo complete.")
logger.info("=" * 60)
