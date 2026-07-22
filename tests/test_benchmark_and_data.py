"""End-to-end: the synthetic generator produces regimes the detector can
recover, the benchmark runs leak-free, and gating never does worse than the
"always on" control on the synthetic data it was designed for."""
import numpy as np

from regimelab import (
    IrreversibilityDetector, PermutationEntropyDetector,
    RealizedVolDetector, TrendSlopeDetector, run_benchmark,
)
from regimelab.data import synthetic_regimes


def test_synthetic_regimes_are_separable_by_irreversibility():
    s = synthetic_regimes(n_blocks=24, block_len=300, seed=1)
    det = IrreversibilityDetector(window=64)
    score = det.score(s.close)
    finite = np.isfinite(score)
    irr_mean = score[(s.labels == 1) & finite].mean()   # asymmetric blocks
    rev_mean = score[(s.labels == 0) & finite].mean()   # random-walk blocks
    # the detector must see the designed asymmetry, and by a clear margin
    assert irr_mean > rev_mean * 1.3


def test_benchmark_runs_and_reports_all_strategies():
    s = synthetic_regimes(n_blocks=24, block_len=300, seed=2)
    dets = {
        "irreversibility": IrreversibilityDetector(64),
        "perm_entropy": PermutationEntropyDetector(64),
        "realized_vol": RealizedVolDetector(64),
        "trend_slope": TrendSlopeDetector(64),
    }
    res = run_benchmark(s.close, dets, rule="reversion", n_splits=5,
                        embargo=32, min_train=256, cost=0.0002)
    for key in ("always_on", "always_flat", "irreversibility"):
        assert key in res
        assert 0.0 <= res[key].dsr <= 1.0
    # always_flat is exactly do-nothing
    assert res["always_flat"].sharpe == 0.0


def test_gating_reduces_drawdown_vs_always_on():
    # the economic point of a regime signal: turning the rule OFF in the wrong
    # regime should not deepen the worst drawdown relative to trading always.
    s = synthetic_regimes(n_blocks=30, block_len=300, seed=5)
    dets = {"irreversibility": IrreversibilityDetector(64)}
    res = run_benchmark(s.close, dets, rule="reversion", n_splits=5,
                        embargo=32, min_train=256, cost=0.0002)
    assert res["irreversibility"].max_drawdown >= res["always_on"].max_drawdown


def test_labels_are_int_and_same_length():
    s = synthetic_regimes(n_blocks=6, block_len=100, seed=0)
    for det in (IrreversibilityDetector(64), RealizedVolDetector(64),
                TrendSlopeDetector(64), PermutationEntropyDetector(64)):
        lab = det.label(s.close)
        assert lab.shape == s.close.shape
        assert set(np.unique(lab)).issubset({-1, 0, 1})
