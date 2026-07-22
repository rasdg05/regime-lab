"""The flow experiment must build causal features and leak-free labels. The
sklearn-dependent AUC path is smoke-tested only when sklearn is present."""
import numpy as np
import pytest

from regimelab.data import Series, synthetic_regimes
from regimelab.flow_experiment import (
    flow_features, regime_transition_labels, run_flow_experiment,
)


def _synth_flow_series(n=4000, seed=0):
    """Synthetic regimes + a fabricated taker-buy volume so flow features exist."""
    base = synthetic_regimes(n_blocks=max(2, n // 300), block_len=300, seed=seed)
    rng = np.random.default_rng(seed)
    vol = np.abs(rng.normal(100, 20, len(base))) + 1.0
    # taker-buy loosely tracks the sign of returns (so imbalance isn't degenerate)
    r = np.diff(base.close, prepend=base.close[0])
    frac = np.clip(0.5 + 0.3 * np.sign(r) + rng.normal(0, 0.1, len(base)), 0.05, 0.95)
    return Series(ts=base.ts, close=base.close, volume=vol, taker_buy=vol * frac)


def test_flow_imbalance_in_range():
    s = _synth_flow_series(1500)
    imb = s.flow_imbalance()
    finite = np.isfinite(imb)
    assert (imb[finite] >= -1.0).all() and (imb[finite] <= 1.0).all()


def test_flow_features_causal():
    s = _synth_flow_series(1800)
    X_full, names = flow_features(s)
    k = 1000
    s_pre = Series(ts=s.ts[:k], close=s.close[:k], volume=s.volume[:k],
                   taker_buy=s.taker_buy[:k])
    X_pre, _ = flow_features(s_pre)
    m = np.isfinite(X_full[:k]) & np.isfinite(X_pre)
    assert np.allclose(X_full[:k][m], X_pre[m])


def test_transition_labels_binary_and_last_bar_zero():
    s = _synth_flow_series(2000)
    y = regime_transition_labels(s.close, window=64, horizon=12)
    assert set(np.unique(y)).issubset({0, 1})
    # the final bar has an empty forward window -> cannot be a transition-ahead
    assert y[-1] == 0
    # and the label is forward-looking: some 1s should exist given real switches
    assert y.sum() > 0


def test_run_flow_experiment_smoke():
    pytest.importorskip("sklearn")
    s = _synth_flow_series(5000, seed=3)
    res = run_flow_experiment(s, window=64, horizon=12, n_splits=4,
                              embargo=48, min_train=1200)
    for k in ("auc_price_only", "auc_price_plus_flow", "auc_flow_only"):
        assert 0.0 <= res[k] <= 1.0
    assert 0.0 <= res["transition_rate"] <= 1.0
