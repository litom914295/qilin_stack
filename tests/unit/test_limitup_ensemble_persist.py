import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.limitup_ensemble import LimitUpEnsembleModel


@pytest.mark.parametrize("n_samples,n_features", [(40, 6), (25, 4)])
def test_limitup_ensemble_save_and_load_roundtrip(n_samples: int, n_features: int) -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # 构造可学习的简单目标: f0>0 且 f1>0.5 更可能为1
    y = ((X.iloc[:, 0] > 0) & (X.iloc[:, 1] > 0.5)).astype(int)

    model = LimitUpEnsembleModel()
    model.fit(X, y)

    # 基本预测形状
    proba = model.predict_proba(X)
    preds = model.predict(X)
    assert proba.shape == (n_samples, 2)
    assert preds.shape == (n_samples,)

    # 评估返回必要指标
    metrics = model.evaluate(X, y)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in metrics

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pkl"
        model.save(str(path))
        assert path.exists()

        loaded = LimitUpEnsembleModel.load(str(path))
        # 载入后预测形状一致
        proba2 = loaded.predict_proba(X)
        preds2 = loaded.predict(X)
        assert proba2.shape == proba.shape
        assert preds2.shape == preds.shape
