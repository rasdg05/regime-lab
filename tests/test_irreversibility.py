"""The HVG time-irreversibility statistic must behave the way the theory says:
symmetric processes score ~0, temporally asymmetric ones score high. These are
the properties a reader should be able to trust before believing any downstream
result — so we pin them."""
import numpy as np

from regimelab.detectors.irreversibility import (
    IrreversibilityDetector, hvg_degrees, irreversibility,
)


def test_monotonic_ramp_is_reversible():
    # a clean linear trend is perfectly time-symmetric under the HVG -> ~0.
    y = np.linspace(0.0, 10.0, 256)
    assert irreversibility(y) < 1e-6


def test_sawtooth_is_strongly_irreversible():
    # slow up, fast down: the textbook irreversible signal -> large value.
    v = 0.0
    saw = []
    for i in range(256):
        v += 1.0 if i % 20 < 15 else -3.0
        saw.append(v)
    assert irreversibility(np.asarray(saw, float)) > 1.0


def test_symmetric_noise_below_asymmetric():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0, 1, 512))
    saw = np.asarray([(-3.0 if i % 20 >= 15 else 1.0) for i in range(512)]).cumsum()
    assert irreversibility(walk) < irreversibility(saw)


def test_short_or_constant_series_return_zero():
    assert irreversibility(np.ones(50)) == 0.0
    assert irreversibility(np.array([1.0, 2.0, 3.0])) == 0.0


def test_hvg_degrees_shape_and_nonneg():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 64)
    indeg, outdeg = hvg_degrees(y)
    assert indeg.shape == outdeg.shape == y.shape
    assert (indeg >= 0).all() and (outdeg >= 0).all()
    # every visibility edge is counted once as an out (from i) and once as an in (to j)
    assert indeg.sum() == outdeg.sum()


def test_detector_labels_are_causal():
    # score[t] must not change when future bars are appended (no look-ahead).
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(0, 1, 400))
    det = IrreversibilityDetector(window=64)
    s_full = det.score(y)
    s_prefix = det.score(y[:300])
    finite = np.isfinite(s_prefix)
    assert np.allclose(s_full[:300][finite], s_prefix[finite])


def test_fit_sets_median_threshold_and_balances():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 1, 1000))
    det = IrreversibilityDetector(window=64).fit(y)
    assert det.threshold is not None
    lab = det.label(y)
    active = lab[lab >= 0]
    # median split -> roughly balanced classes
    frac = active.mean()
    assert 0.3 < frac < 0.7
