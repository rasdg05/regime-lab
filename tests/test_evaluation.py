"""The evaluation harness is the real product — if it leaks or fails to deflate,
every downstream number is a lie. These tests pin the two properties that
matter: no train/test leakage, and the DSR actually penalizes multiple testing."""
import numpy as np

from regimelab.evaluation.deflated_sharpe import (
    deflated_sharpe_ratio, expected_max_sharpe, probability_backtest_overfitting,
    sharpe_ratio,
)
from regimelab.evaluation.walkforward import (
    assert_no_leakage, purged_walk_forward_splits,
)


def test_walkforward_is_ordered_and_embargoed():
    folds = purged_walk_forward_splits(2000, n_splits=5, embargo=32, min_train=256)
    assert folds
    for f in folds:
        # train strictly before test, with the embargo gap
        assert f.train.max() < f.test.min() - 32 + 1
    assert_no_leakage(folds, embargo=32)          # must not raise


def test_assert_no_leakage_catches_overlap():
    from regimelab.evaluation.walkforward import Fold
    bad = [Fold(train=np.arange(0, 100), test=np.arange(90, 120))]
    try:
        assert_no_leakage(bad, embargo=0)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_expected_max_sharpe_grows_with_trials():
    # the multiple-testing benchmark must increase as you try more configs
    a = expected_max_sharpe(2, 1.0)
    b = expected_max_sharpe(50, 1.0)
    c = expected_max_sharpe(1000, 1.0)
    assert a < b < c


def test_dsr_penalizes_more_trials():
    rng = np.random.default_rng(0)
    # a genuinely decent return stream
    r = rng.normal(0.05, 1.0, 2000)
    few = deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    many = deflated_sharpe_ratio(r, n_trials=500)["dsr"]
    assert few >= many          # more trials -> harder to clear
    assert 0.0 <= many <= 1.0 and 0.0 <= few <= 1.0


def test_dsr_rejects_pure_noise():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 1.0, 3000)          # zero-mean: no edge
    dsr = deflated_sharpe_ratio(r, n_trials=100)["dsr"]
    assert dsr < 0.5


def test_pbo_high_for_noise_configs():
    # N pure-noise configs: the in-sample winner should not persist OOS -> high PBO
    rng = np.random.default_rng(2)
    M = rng.normal(0.0, 1.0, size=(1200, 12))
    pbo = probability_backtest_overfitting(M, n_splits=8)
    assert pbo > 0.4


def test_sharpe_zero_variance():
    assert sharpe_ratio(np.ones(100)) == 0.0
