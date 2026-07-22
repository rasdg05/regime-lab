"""Bonus experiment: does order flow predict *regime transitions*?

The main repo asks whether a regime signal helps you trade. This asks a
different, sharper question the FQ framework cares about: **can taker order-flow
imbalance tell you a regime change is coming before it shows up in price?**

Setup
-----
* Regime is the irreversibility detector's label (0 reversible / 1 irreversible),
  fit causally.
* A **transition** is any bar where that label differs from the previous bar's.
* The target: does a transition occur in the next ``horizon`` bars? (binary)
* The signal: order-flow features (current imbalance, its short/long rolling
  means, and its volatility) — all causal.
* The control: the same model on price-only features (returns + realized vol),
  so "flow helps" means flow beats price at *anticipating* the switch.

Evaluated with the same purged walk-forward + AUC as everything else. Reported
straight — anticipating regime changes is hard, and the honest number is the
deliverable.
"""
from __future__ import annotations

import numpy as np

from .detectors.irreversibility import IrreversibilityDetector
from .evaluation.walkforward import purged_walk_forward_splits

__all__ = ["flow_features", "regime_transition_labels", "run_flow_experiment"]


def _auc(y_true, score) -> float:
    y_true = np.asarray(y_true).astype(int)
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) /
                 (len(pos) * len(neg)))


def _cross_val_auc(X, y, folds, random_state=0) -> float:
    """Pooled OOS AUC of a gradient-boosted classifier over purged folds.
    sklearn is imported lazily so the core package stays numpy-only."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    yt, ys = [], []
    for f in folds:
        if len(np.unique(y[f.train])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                             max_depth=4, l2_regularization=1.0,
                                             random_state=random_state)
        clf.fit(X[f.train], y[f.train])
        yt.append(y[f.test])
        ys.append(clf.predict_proba(X[f.test])[:, 1])
    if not yt:
        return float("nan")
    return _auc(np.concatenate(yt), np.concatenate(ys))


def _roll_mean(x, n):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.nan_to_num(x))
    cnt = np.cumsum(np.isfinite(x).astype(float))
    for t in range(n, len(x) + 1):
        s = c[t - 1] - (c[t - n - 1] if t - n - 1 >= 0 else 0.0)
        k = cnt[t - 1] - (cnt[t - n - 1] if t - n - 1 >= 0 else 0.0)
        out[t - 1] = s / k if k > 0 else np.nan
    return out


def _roll_std(x, n):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    for t in range(n, len(x) + 1):
        w = x[t - n:t]
        w = w[np.isfinite(w)]
        if len(w) > 1:
            out[t - 1] = np.std(w, ddof=1)
    return out


def flow_features(series) -> tuple[np.ndarray, list[str]]:
    """Causal order-flow features from a Series carrying volume + taker_buy."""
    imb = series.flow_imbalance()
    feats = {
        "flow_imb": imb,
        "flow_imb_ma6": _roll_mean(imb, 6),
        "flow_imb_ma24": _roll_mean(imb, 24),
        "flow_imb_vol24": _roll_std(imb, 24),
        "flow_cvd48": _roll_mean(imb, 48) * 48,      # rolling signed-flow accumulation
    }
    names = list(feats)
    X = np.column_stack([feats[k] for k in names])
    return X, names


def _price_control_features(close) -> tuple[np.ndarray, list[str]]:
    close = np.asarray(close, float)
    r = np.zeros_like(close)
    r[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
    def rstd(n):
        out = np.full_like(close, np.nan)
        for t in range(n, len(close) + 1):
            out[t - 1] = np.std(r[t - n:t], ddof=1)
        return out
    X = np.column_stack([r, rstd(6), rstd(24)])
    return X, ["ret_1", "vol_6", "vol_24"]


def regime_transition_labels(close, *, window=64, horizon=12) -> np.ndarray:
    """1 if a regime (irreversibility label) transition occurs within the next
    ``horizon`` bars, else 0. Causal: uses the detector's causal labels."""
    det = IrreversibilityDetector(window=window).fit(close)
    lab = det.label(close)
    n = len(close)
    switch = np.zeros(n, dtype=int)
    prev = None
    for t in range(n):
        if lab[t] < 0:
            continue
        if prev is not None and lab[t] != prev:
            switch[t] = 1
        prev = lab[t]
    y = np.zeros(n, dtype=int)
    for t in range(n):
        j0, j1 = t + 1, min(t + horizon + 1, n)
        if j0 < n and switch[j0:j1].any():
            y[t] = 1
    return y


def run_flow_experiment(series, *, window=64, horizon=12, n_splits=5,
                        embargo=48, min_train=2000, random_state=0) -> dict:
    close = series.close
    Xf, fnames = flow_features(series)
    Xc, cnames = _price_control_features(close)
    y = regime_transition_labels(close, window=window, horizon=horizon)

    n = len(close)
    valid = np.isfinite(Xf).all(axis=1) & np.isfinite(Xc).all(axis=1)
    valid[-horizon:] = False
    idx = np.where(valid)[0]
    Xf, Xc, y = Xf[idx], Xc[idx], y[idx]

    folds = purged_walk_forward_splits(len(y), n_splits=n_splits, embargo=embargo,
                                       min_train=min_train)
    if not folds:
        raise ValueError("not enough rows for the requested CV config")

    auc_ctrl = _cross_val_auc(Xc, y, folds, random_state)
    auc_flow = _cross_val_auc(np.hstack([Xc, Xf]), y, folds, random_state)
    auc_flow_only = _cross_val_auc(Xf, y, folds, random_state)
    return {
        "n_rows": len(y),
        "transition_rate": float(y.mean()),
        "auc_price_only": auc_ctrl,
        "auc_price_plus_flow": auc_flow,
        "auc_flow_only": auc_flow_only,
        "auc_uplift": auc_flow - auc_ctrl,
        "n_folds": len(folds),
    }
